//! Definicion del sistema solar y de la nave.
//!
//! Cada cuerpo celeste es un "modelo" con su propio shader, su propia matriz
//! de modelo (orbita + inclinacion + rotacion + escala) y sus parametros.
//! Todos los movimientos dependen del tiempo transcurrido, no de los frames.

use nalgebra_glm::{Mat4, Vec3};
use nalgebra_glm as glm;

use crate::math::approach;
use crate::renderer::FragmentShader;
use crate::shaders;

#[derive(Clone, Copy, PartialEq)]
pub enum Kind {
    Star,
    Planet,
    /// Agujero negro: al cruzarlo se viaja a la siguiente galaxia
    Portal,
}

/// Forma de la malla. Las que no son esfera se generan deformando una
/// icosfera (o con su propio generador, el toro) al cargar la galaxia.
#[derive(Clone, Copy, PartialEq)]
pub enum Shape {
    Sphere,
    /// cubo redondeado (superelipsoide)
    Cube,
    /// dos esferas unidas por la cintura
    Peanut,
    /// picos de ruido "ridged"; (altura, semilla)
    Spiky(f32, u32),
    Torus,
}

impl Shape {
    /// Que tanto sobresale la forma respecto al radio (para colisiones).
    pub fn extent(&self) -> f32 {
        match self {
            Shape::Sphere => 1.0,
            Shape::Cube => 1.1,
            Shape::Peanut => 1.6,
            Shape::Spiky(h, _) => 1.0 + h,
            Shape::Torus => 1.45,
        }
    }
}

#[derive(Clone)]
pub struct Body {
    pub name: &'static str,
    pub kind: Kind,
    pub shape: Shape,
    pub radius: f32,
    pub orbit_radius: f32,
    /// radianes por segundo alrededor del padre
    pub orbit_speed: f32,
    pub orbit_phase: f32,
    /// radianes por segundo sobre su propio eje
    pub spin_speed: f32,
    /// inclinacion del eje (radianes)
    pub tilt: f32,
    pub parent: Option<usize>,
    pub shader: FragmentShader,
    pub params: [f32; 4],
    pub has_ring: bool,
    pub warp_key: u8,
    pub label_color: u32,
}

impl Body {
    /// Posicion en el mundo. Los padres deben estar antes en la lista.
    pub fn position(&self, time: f32, positions: &[Vec3]) -> Vec3 {
        let center = self.parent.map(|p| positions[p]).unwrap_or_default();
        let a = self.orbit_phase + self.orbit_speed * time;
        center + Vec3::new(a.cos(), 0.0, a.sin()) * self.orbit_radius
    }

    /// Matriz de modelo: traslacion * inclinacion * rotacion propia * escala.
    pub fn model_matrix(&self, position: &Vec3, time: f32) -> Mat4 {
        glm::translation(position)
            * glm::rotation(self.tilt, &Vec3::z())
            * glm::rotation(self.spin_speed * time, &Vec3::y())
            * glm::scaling(&Vec3::repeat(self.radius))
    }

    /// El anillo comparte posicion, inclinacion y escala, pero no gira.
    pub fn ring_matrix(&self, position: &Vec3) -> Mat4 {
        glm::translation(position)
            * glm::rotation(self.tilt, &Vec3::z())
            * glm::scaling(&Vec3::repeat(self.radius))
    }

    pub fn is_sun(&self) -> bool {
        self.kind == Kind::Star
    }

    pub fn is_portal(&self) -> bool {
        self.kind == Kind::Portal
    }

    /// Radio que la nave no puede atravesar.
    pub fn collision_radius(&self) -> f32 {
        self.radius * self.shape.extent()
    }
}

/// Valores por defecto para declarar cuerpos con `..Body::base()`.
impl Body {
    pub fn base() -> Body {
        Body {
            name: "",
            kind: Kind::Planet,
            shape: Shape::Sphere,
            radius: 1.0,
            orbit_radius: 0.0,
            orbit_speed: 0.0,
            orbit_phase: 0.0,
            spin_speed: 0.0,
            tilt: 0.0,
            parent: None,
            shader: shaders::sun_shader,
            params: [1.0, 1.0, 0.0, 0.0],
            has_ring: false,
            warp_key: 0,
            label_color: 0xffffff,
        }
    }
}

#[derive(Default, Clone, Copy)]
pub struct ShipInput {
    pub forward: bool,
    pub back: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
    pub boost: bool,
}

/// Nave con aceleracion, frenado y giro progresivos. Todo se integra con dt
/// y factores exponenciales, asi responde igual a 30 o a 144 FPS.
pub struct Ship {
    pub position: Vec3,
    pub velocity: Vec3,
    pub yaw: f32,
    yaw_rate: f32,
    pub bank: f32,
    pub pitch: f32,
    pub thrust: f32,
    pub boost: f32,
}

pub const SHIP_SCALE: f32 = 0.4;
const THRUST: f32 = 26.0;
const BOOST_MULT: f32 = 4.0;
const BRAKE: f32 = 45.0;
const VERTICAL: f32 = 20.0;
const DRAG: f32 = 0.6;
const GRIP: f32 = 2.5;
const TURN_MAX: f32 = 1.7;
const TURN_RESPONSE: f32 = 5.0;

impl Ship {
    pub fn new(position: Vec3, yaw: f32) -> Self {
        Ship { position, velocity: Vec3::zeros(), yaw, yaw_rate: 0.0, bank: 0.0, pitch: 0.0, thrust: 0.0, boost: 0.0 }
    }

    /// La nave apunta a -Z en su espacio local; con yaw gira sobre Y.
    pub fn forward(&self) -> Vec3 {
        Vec3::new(-self.yaw.sin(), 0.0, -self.yaw.cos())
    }

    pub fn speed(&self) -> f32 {
        self.velocity.norm()
    }

    pub fn update(&mut self, dt: f32, input: &ShipInput) {
        let turn = (input.left as i32 - input.right as i32) as f32;
        self.yaw_rate += (turn * TURN_MAX - self.yaw_rate) * approach(TURN_RESPONSE, dt);
        self.yaw += self.yaw_rate * dt;

        let fwd = self.forward();
        let boost = if input.boost && input.forward { BOOST_MULT } else { 1.0 };
        let mut accel = Vec3::zeros();
        if input.forward {
            accel += fwd * THRUST * boost;
        }
        if input.back {
            if self.velocity.dot(&fwd) > 1.0 {
                accel -= self.velocity.normalize() * BRAKE; // frenar
            } else {
                accel -= fwd * THRUST * 0.4; // reversa lenta
            }
        }
        accel.y += (input.up as i32 - input.down as i32) as f32 * VERTICAL;
        self.velocity += accel * dt;

        // "Agarre": la velocidad horizontal se alinea poco a poco con la nariz
        let horizontal = Vec3::new(self.velocity.x, 0.0, self.velocity.z);
        let side = horizontal - fwd * horizontal.dot(&fwd);
        self.velocity -= side * approach(GRIP, dt);

        self.velocity *= (-DRAG * dt).exp();
        self.velocity.y *= (-1.5 * dt).exp();
        self.position += self.velocity * dt;

        // Solo visual: alabeo al girar, cabeceo al subir/bajar, brillo de motores
        let k = approach(6.0, dt);
        self.bank += (self.yaw_rate * 0.4 - self.bank) * k;
        self.pitch += ((input.up as i32 - input.down as i32) as f32 * 0.25 - self.pitch) * k;
        self.thrust += (input.forward as i32 as f32 - self.thrust) * k;
        self.boost += ((boost > 1.0) as i32 as f32 - self.boost) * k;
    }

    pub fn model_matrix(&self) -> Mat4 {
        glm::translation(&self.position)
            * glm::rotation(self.yaw, &Vec3::y())
            * glm::rotation(self.bank, &Vec3::z())
            * glm::rotation(self.pitch, &Vec3::x())
            * glm::scaling(&Vec3::repeat(SHIP_SCALE))
    }
}
