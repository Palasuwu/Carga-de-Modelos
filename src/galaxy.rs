//! Las cuatro galaxias. Cada una tiene su cielo (paleta del shader de
//! nebulosa + nubes), una estrella, sus planetas y un agujero negro que sirve
//! de portal hacia la siguiente (la cuarta regresa a la primera).
//!
//! `Galaxy::build` hace todo el trabajo pesado (hornear el cielo y generar
//! las mallas deformadas). Se llama en un hilo aparte mientras se ve la
//! animacion del tunel, asi la "carga" no congela el juego.

use nalgebra_glm::Vec3;

use crate::exotic;
use crate::mesh::{Mesh, deformed_sphere, torus};
use crate::noise::voronoi;
use crate::scene::{Body, Kind, Shape};
use crate::shaders;
use crate::sky::{Cloud, SkyBox, SkyStyle};

pub const GALAXY_COUNT: usize = 4;
pub const MAX_LOD: usize = 5;

pub struct Galaxy {
    pub index: usize,
    pub name: &'static str,
    pub bodies: Vec<Body>,
    pub sky: SkyBox,
    /// Mallas propias de los cuerpos que no son esferas: un nivel de detalle
    /// por indice 0..=MAX_LOD (None = usa las icosferas compartidas).
    pub shape_meshes: Vec<Option<Vec<Mesh>>>,
}

pub fn name(index: usize) -> &'static str {
    ["VIA SOLAR", "NEBULOSA VIOLETA", "JARDIN ESMERALDA", "ABISMO CARMESI"][index % GALAXY_COUNT]
}

impl Galaxy {
    /// `threads` = hilos para hornear el cielo (1 cuando se carga en segundo plano).
    pub fn build(index: usize, time: f32, threads: usize) -> Galaxy {
        let bodies = bodies(index);
        let shape_meshes = bodies.iter().map(|b| shape_lods(b.shape)).collect();
        Galaxy { index, name: name(index), sky: SkyBox::new(style(index), time, threads), bodies, shape_meshes }
    }

    pub fn next_index(&self) -> usize {
        (self.index + 1) % GALAXY_COUNT
    }
}

fn rgbf(r: f32, g: f32, b: f32) -> Vec3 {
    Vec3::new(r, g, b)
}

/// Cielo de cada galaxia.
pub fn style(index: usize) -> SkyStyle {
    let cloud = |dir: Vec3, color: Vec3, size: f32, seed: u32| Cloud { dir: dir.normalize(), color, size, seed };
    match index % GALAXY_COUNT {
        0 => SkyStyle {
            // Los colores originales del shader
            colours: [rgbf(0.04, 0.08, 0.14), rgbf(0.08, 0.14, 0.22), rgbf(0.01, 0.01, 0.02)],
            clouds: vec![],
            tint: rgbf(0.3, 0.6, 1.0),
        },
        1 => SkyStyle {
            colours: [rgbf(0.08, 0.03, 0.14), rgbf(0.17, 0.06, 0.25), rgbf(0.02, 0.0, 0.04)],
            clouds: vec![
                cloud(Vec3::new(0.5, 0.3, -0.8), rgbf(0.55, 0.12, 0.45), 0.7, 3),
                cloud(Vec3::new(-0.7, -0.2, -0.3), rgbf(0.15, 0.22, 0.65), 0.55, 5),
                cloud(Vec3::new(0.2, 0.7, 0.6), rgbf(0.4, 0.1, 0.6), 0.5, 9),
            ],
            tint: rgbf(0.8, 0.3, 1.0),
        },
        2 => SkyStyle {
            colours: [rgbf(0.02, 0.10, 0.07), rgbf(0.05, 0.20, 0.13), rgbf(0.0, 0.02, 0.01)],
            clouds: vec![
                cloud(Vec3::new(-0.4, 0.2, -0.9), rgbf(0.15, 0.55, 0.35), 0.75, 11),
                cloud(Vec3::new(0.8, -0.1, -0.2), rgbf(0.55, 0.48, 0.12), 0.5, 13),
                cloud(Vec3::new(0.0, -0.6, 0.8), rgbf(0.1, 0.45, 0.5), 0.6, 17),
            ],
            tint: rgbf(0.3, 1.0, 0.6),
        },
        _ => SkyStyle {
            colours: [rgbf(0.14, 0.03, 0.03), rgbf(0.24, 0.07, 0.05), rgbf(0.03, 0.0, 0.01)],
            clouds: vec![
                cloud(Vec3::new(0.3, 0.25, -0.9), rgbf(0.7, 0.22, 0.06), 0.8, 19),
                cloud(Vec3::new(-0.8, 0.4, 0.1), rgbf(0.5, 0.04, 0.12), 0.6, 23),
                cloud(Vec3::new(0.5, -0.5, 0.6), rgbf(0.6, 0.35, 0.1), 0.45, 29),
            ],
            tint: rgbf(1.0, 0.35, 0.2),
        },
    }
}

/// El portal de cada galaxia: un agujero negro al frente a la izquierda del
/// punto de inicio. params = [horizonte, disco interno, disco externo, inclinacion].
fn portal(label_color: u32) -> Body {
    Body {
        name: "PORTAL",
        kind: Kind::Portal,
        radius: 38.0,
        orbit_radius: 430.0,
        orbit_phase: 3.9,
        shader: exotic::black_hole_shader,
        params: [0.28, 0.4, 0.95, 0.3],
        warp_key: 9,
        label_color,
        ..Body::base()
    }
}

/// Planetas de cada galaxia. El primero siempre es la estrella.
/// Valores faciles de cambiar en vivo: radio, orbita, velocidades y params.
fn bodies(index: usize) -> Vec<Body> {
    let base = Body::base;
    match index % GALAXY_COUNT {
        0 => vec![
            Body { name: "SOL", kind: Kind::Star, radius: 30.0, spin_speed: 0.03, warp_key: 0, label_color: 0xffc040, ..base() },
            Body {
                name: "MAGMA", radius: 6.5, orbit_radius: 80.0, orbit_speed: 0.060, orbit_phase: 0.6,
                spin_speed: 0.25, tilt: 0.1, shader: shaders::magma_shader, params: [1.6, 0.0, 0.0, 0.0],
                warp_key: 1, label_color: 0xff6030, ..base()
            },
            Body {
                name: "VERDALIA", radius: 10.0, orbit_radius: 140.0, orbit_speed: 0.040, orbit_phase: 2.2,
                spin_speed: 0.12, tilt: 0.4, shader: shaders::verdalia_shader, warp_key: 2,
                label_color: 0x60c0ff, ..base()
            },
            Body {
                name: "LUNA", radius: 2.6, orbit_radius: 22.0, orbit_speed: 0.35, spin_speed: 0.35,
                parent: Some(2), shader: shaders::moon_shader, warp_key: 6, label_color: 0xc0c0c0, ..base()
            },
            Body {
                name: "DUNA", radius: 8.0, orbit_radius: 205.0, orbit_speed: 0.030, orbit_phase: 4.0,
                spin_speed: 0.10, tilt: 0.2, shader: shaders::duna_shader, warp_key: 3,
                label_color: 0xf0a060, ..base()
            },
            Body {
                name: "BANDUS", radius: 18.0, orbit_radius: 295.0, orbit_speed: 0.020, orbit_phase: 5.3,
                spin_speed: 0.18, tilt: 0.35, shader: shaders::bandus_shader, has_ring: true,
                warp_key: 4, label_color: 0xe0c090, ..base()
            },
            Body {
                name: "GLACIA", radius: 7.5, orbit_radius: 385.0, orbit_speed: 0.015, orbit_phase: 1.2,
                spin_speed: 0.09, tilt: 0.25, shader: shaders::glacia_shader, warp_key: 5,
                label_color: 0xa0e8ff, ..base()
            },
            portal(0xb080ff),
        ],
        1 => vec![
            Body {
                name: "ESTRELLA AZUL", kind: Kind::Star, radius: 24.0, spin_speed: 0.05,
                params: [1.3, 2.0, 0.0, 0.0], warp_key: 0, label_color: 0x90b8ff, ..base()
            },
            Body {
                name: "CUBIX", shape: Shape::Cube, radius: 9.0, orbit_radius: 90.0, orbit_speed: 0.05,
                orbit_phase: 0.8, spin_speed: 0.3, tilt: 0.4, shader: exotic::cubix_shader,
                warp_key: 1, label_color: 0x40e8ff, ..base()
            },
            Body {
                name: "ESPEJO", radius: 8.0, orbit_radius: 150.0, orbit_speed: 0.035, orbit_phase: 2.6,
                spin_speed: 0.1, shader: exotic::mirror_shader, warp_key: 2, label_color: 0xe0e0ff, ..base()
            },
            Body {
                name: "TOROIDE", shape: Shape::Torus, radius: 12.0, orbit_radius: 220.0, orbit_speed: 0.025,
                orbit_phase: 3.1, spin_speed: 0.2, tilt: 1.15, shader: exotic::torus_candy_shader,
                warp_key: 3, label_color: 0xff80c0, ..base()
            },
            Body {
                name: "CRISTAL", shape: Shape::Spiky(0.35, 7), radius: 9.0, orbit_radius: 300.0,
                orbit_speed: 0.018, orbit_phase: 5.6, spin_speed: 0.12, tilt: 0.2,
                shader: exotic::crystal_shader, params: [1.0, 0.35, 0.0, 0.0], warp_key: 4,
                label_color: 0xc090ff, ..base()
            },
            portal(0x60ffb0),
        ],
        2 => vec![
            Body {
                name: "ESTRELLA VERDE", kind: Kind::Star, radius: 26.0, spin_speed: 0.04,
                params: [0.8, 3.0, 0.0, 0.0], warp_key: 0, label_color: 0x90ffb0, ..base()
            },
            Body {
                name: "CACAHUATE", shape: Shape::Peanut, radius: 7.0, orbit_radius: 95.0, orbit_speed: 0.05,
                orbit_phase: 1.0, spin_speed: 0.2, tilt: 0.3, shader: exotic::jungle_shader,
                warp_key: 1, label_color: 0x80e060, ..base()
            },
            Body {
                name: "OJO", radius: 9.0, orbit_radius: 160.0, orbit_speed: 0.035, orbit_phase: 2.4,
                spin_speed: 0.1, shader: exotic::eye_shader, warp_key: 2, label_color: 0xf0f0e0, ..base()
            },
            Body {
                name: "BURBUJA", radius: 10.0, orbit_radius: 230.0, orbit_speed: 0.025, orbit_phase: 4.1,
                spin_speed: 0.15, shader: exotic::bubble_shader, warp_key: 3, label_color: 0xffb0ff, ..base()
            },
            Body {
                name: "CORAL", radius: 11.0, orbit_radius: 310.0, orbit_speed: 0.018, orbit_phase: 5.5,
                spin_speed: 0.08, tilt: 0.3, shader: exotic::coral_shader, warp_key: 4,
                label_color: 0x50ffe0, ..base()
            },
            portal(0xff7050),
        ],
        _ => vec![
            Body {
                name: "GIGANTE ROJA", kind: Kind::Star, radius: 42.0, spin_speed: 0.02,
                params: [0.6, 0.0, 0.0, 0.0], warp_key: 0, label_color: 0xff6040, ..base()
            },
            Body {
                name: "VACIO", radius: 10.0, orbit_radius: 115.0, orbit_speed: 0.04, orbit_phase: 0.9,
                spin_speed: 0.15, shader: exotic::void_shader, warp_key: 1, label_color: 0xff7090, ..base()
            },
            Body {
                name: "TABLERO", radius: 9.0, orbit_radius: 180.0, orbit_speed: 0.03, orbit_phase: 2.5,
                spin_speed: 0.12, tilt: 0.3, shader: exotic::checker_shader, warp_key: 2,
                label_color: 0xf0e0c0, ..base()
            },
            Body {
                name: "ERIZO", shape: Shape::Spiky(0.45, 13), radius: 8.0, orbit_radius: 250.0,
                orbit_speed: 0.022, orbit_phase: 4.0, spin_speed: 0.18, shader: exotic::urchin_shader,
                params: [1.0, 0.45, 0.0, 0.0], warp_key: 3, label_color: 0xffa040, ..base()
            },
            Body {
                name: "HALO", shape: Shape::Torus, radius: 14.0, orbit_radius: 330.0, orbit_speed: 0.016,
                orbit_phase: 5.4, spin_speed: 0.1, tilt: 1.1, shader: exotic::halo_shader, warp_key: 4,
                label_color: 0xffc060, ..base()
            },
            portal(0x60b0ff),
        ],
    }
}

/// Genera los niveles de detalle de una forma que no es esfera.
fn shape_lods(shape: Shape) -> Option<Vec<Mesh>> {
    let levels = 0..=MAX_LOD as u32;
    match shape {
        Shape::Sphere => None,
        // Superelipsoide |x|^8 + |y|^8 + |z|^8 = 1: un cubo de esquinas redondas
        Shape::Cube => Some(levels.map(|l| deformed_sphere(l, |p| {
            let k = (p.x.abs().powi(8) + p.y.abs().powi(8) + p.z.abs().powi(8)).powf(1.0 / 8.0);
            p / k
        })).collect()),
        // Se estira en X y se adelgaza la cintura
        Shape::Peanut => Some(levels.map(|l| deformed_sphere(l, |p| {
            let waist = 1.0 - 0.38 * (-p.x * p.x * 5.0).exp();
            Vec3::new(p.x * 1.55, p.y * waist, p.z * waist)
        })).collect()),
        // Un cono por cada celda de Voronoi: picos repartidos en la esfera
        Shape::Spiky(height, seed) => Some(levels.map(|l| deformed_sphere(l, |p| {
            let (f1, _, _) = voronoi(p * 2.6, seed);
            let cone = (1.0 - f1 * 2.2).max(0.0).powf(1.3);
            p * (1.0 + height * cone)
        })).collect()),
        Shape::Torus => Some(
            [(12, 6), (16, 8), (28, 12), (48, 20), (90, 36), (150, 56)]
                .iter()
                .map(|&(nu, nv)| torus(1.0, 0.42, nu, nv))
                .collect(),
        ),
    }
}

/// Punto de llegada al entrar a una galaxia: frente a la estrella, un poco
/// arriba del plano, mirando hacia ella.
pub fn arrival() -> (Vec3, f32) {
    (Vec3::new(0.0, 22.0, 260.0), 0.0)
}
