//! Logica del juego: actualiza la simulacion, controla la camara en tercera
//! persona, el warp entre planetas, el mapa del sistema, y arma cada frame
//! pasando cada modelo por el pipeline con su matriz y su shader.

use std::thread::{self, JoinHandle};

use nalgebra_glm::{Mat4, Vec3, Vec4};
use nalgebra_glm as glm;

use crate::framebuffer::{Framebuffer, pack_color, unpack_color};
use crate::hud;
use crate::line::draw_line;
use crate::math::{angle_diff, approach, ease_in_cubic, ease_out_cubic, lerp3, look_at, perspective, smoothstep, viewport};
use crate::galaxy::{self, Galaxy, MAX_LOD};
use crate::mesh::{Mesh, icosphere, ring};
use crate::renderer::{CullMode, RenderStats, Renderer, Uniforms};
use crate::scene::{Ship, ShipInput};
use crate::shaders;
use crate::sky::hyperspace;
use crate::visit::{self, Scene};

pub const WIDTH: usize = 960;
pub const HEIGHT: usize = 540;

const BASE_FOV: f32 = 60.0;
const WARP_FOV_KICK: f32 = 35.0;
/// El turbo abre un poco el campo de vision para sentir mas velocidad.
const BOOST_FOV_KICK: f32 = 12.0;
const NEAR: f32 = 0.1;
const FAR: f32 = 10_000.0;
/// Radio de la esfera del cielo: contiene a todo el sistema solar.
const SKY_RADIUS: f32 = 4000.0;
const WORLD_LIMIT: f32 = 1400.0;
/// Camara en tercera persona: distancia constante detras y arriba de la nave.
const CAM_DISTANCE: f32 = 7.0;
const CAM_HEIGHT: f32 = 2.2;
const MAP_HEIGHT: f32 = 820.0;
const MAP_TRANSITION: f32 = 0.9;

// Viaje entre galaxias por el portal
const JUMP_ENTER: f32 = 1.0;
/// Duracion minima del tunel; si la galaxia nueva tarda mas, el tunel sigue.
const JUMP_TUNNEL_MIN: f32 = 1.6;
const JUMP_EXIT: f32 = 1.3;
const JUMP_FOV_KICK: f32 = 40.0;
/// Atraccion del agujero negro: aceleracion = PORTAL_PULL * R^2 / d^2
const PORTAL_PULL: f32 = 22.0;
/// Segundos que se muestra el nombre de la galaxia al llegar.
const BANNER_TIME: f32 = 4.0;

// Visita a la atmosfera de un planeta (burbuja + click)
/// Turbo y entrada al tunel antes de la animacion.
const VISIT_ENTER: f32 = 1.2;
/// Fundido del tunel a la animacion, y de la animacion al aire del planeta.
const VISIT_FADE: f32 = 0.7;
/// Duracion minima del despegue; si el mundo tarda mas en cargar, sigue.
const VISIT_LEAVE_MIN: f32 = 2.4;
/// Fundido del despegue al mundo ya cargado.
const VISIT_EXIT: f32 = 0.9;
const VISIT_FOV_KICK: f32 = 30.0;
const BUBBLE_W: f32 = 184.0;
const BUBBLE_H: f32 = 44.0;

const CLOSE_TIME: f32 = 0.55;
const HOLD_TIME: f32 = 0.2;
const OPEN_TIME: f32 = 0.6;

pub enum Action {
    Warp(u8),
    ToggleMap,
    ToggleHelp,
    ToggleStats,
    ToggleLod,
    ToggleCulling,
    /// Entrar a ver la atmosfera del planeta cercano (igual que el click en la burbuja)
    Visit,
}

#[derive(Default)]
pub struct Input {
    pub ship: ShipInput,
    pub actions: Vec<Action>,
    pub mouse: Option<(f32, f32)>,
    pub click: bool,
}

#[derive(Clone, Copy)]
enum WarpTarget {
    Body(usize),
    Point(Vec3),
}

#[derive(PartialEq)]
enum Phase {
    Closing,
    Hold,
    Opening,
}

struct Transition {
    phase: Phase,
    t: f32,
    target: WarpTarget,
    label: String,
}

#[derive(Clone, Copy, PartialEq)]
enum JumpPhase {
    /// La nave cae al agujero negro y el tunel aparece encima de la escena
    Enter,
    /// Solo tunel: aqui se termina de cargar la galaxia nueva (en otro hilo)
    Tunnel,
    /// Ya en la galaxia nueva, el tunel se desvanece
    Exit,
}

/// Viaje por el portal. La galaxia de destino se construye en un hilo aparte
/// (`loader`) mientras se ve el tunel, asi la animacion nunca se detiene.
struct Jump {
    phase: JumpPhase,
    t: f32,
    loader: Option<JoinHandle<Galaxy>>,
    center: Vec3,
    from_tint: Vec3,
    to_tint: Vec3,
}

impl Jump {
    /// Cuanto cubre el tunel la escena (0..1).
    fn alpha(&self) -> f32 {
        match self.phase {
            JumpPhase::Enter => smoothstep(0.0, JUMP_ENTER, self.t),
            JumpPhase::Tunnel => 1.0,
            JumpPhase::Exit => 1.0 - smoothstep(0.0, JUMP_EXIT, self.t),
        }
    }

    /// El color del tunel pasa del de la galaxia de origen al de destino.
    fn tint(&self) -> Vec3 {
        match self.phase {
            JumpPhase::Enter => self.from_tint,
            JumpPhase::Tunnel => lerp3(&self.from_tint, &self.to_tint, smoothstep(0.0, JUMP_TUNNEL_MIN, self.t)),
            JumpPhase::Exit => self.to_tint,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum VisitPhase {
    /// Turbo: el cielo se estira y aparece el tunel
    Enter,
    /// Animacion de la atmosfera a pantalla completa (el mundo esta descargado)
    Show,
    /// La nave sale del planeta: es la pantalla de carga mientras el mundo
    /// se vuelve a cargar en otro hilo
    Leave,
    /// El mundo ya esta cargado y aparece detras de la nave
    Exit,
}

/// Visita a un planeta. Dos disparadores marcan la carga:
///  1. al terminar `Enter` (pantalla cubierta por el tunel) se descarga el
///     mundo, asi toda la maquina se dedica a la animacion;
///  2. al terminar `Show` se lanza la recarga en otro hilo y se ve el despegue
///     hasta que termina.
///
/// Como cada cambio ocurre con la pantalla cubierta, nunca se ve una carga.
#[derive(Clone, Copy)]
struct Visit {
    phase: VisitPhase,
    t: f32,
    body: usize,
    scene: Scene,
    /// Cuanto duro el despegue (para continuar su animacion en el fundido)
    leave_time: f32,
}

impl Visit {
    /// Cuanto cubre el tunel la imagen (0..1).
    fn tunnel(&self) -> f32 {
        match self.phase {
            VisitPhase::Enter => smoothstep(VISIT_ENTER * 0.35, VISIT_ENTER, self.t),
            VisitPhase::Show => 1.0 - smoothstep(0.0, VISIT_FADE, self.t),
            VisitPhase::Leave | VisitPhase::Exit => 0.0,
        }
    }

    /// El mundo 3D no se dibuja (ni existe) durante la animacion y el despegue.
    fn world_hidden(&self) -> bool {
        matches!(self.phase, VisitPhase::Show | VisitPhase::Leave)
    }

    /// Cuanto se abre el campo de vision. En el despegue se abre poco para
    /// que la nave se vea de cerca.
    fn fov_kick(&self) -> f32 {
        match self.phase {
            VisitPhase::Enter | VisitPhase::Show => self.turbo(),
            VisitPhase::Leave => 0.3,
            VisitPhase::Exit => 0.3 * (1.0 - smoothstep(0.0, VISIT_EXIT, self.t)),
        }
    }

    /// Intensidad del efecto de turbo sobre el cielo.
    fn turbo(&self) -> f32 {
        match self.phase {
            VisitPhase::Enter => smoothstep(0.0, VISIT_ENTER * 0.5, self.t),
            VisitPhase::Show | VisitPhase::Leave => 1.0,
            VisitPhase::Exit => 1.0 - smoothstep(0.0, VISIT_EXIT, self.t),
        }
    }
}

/// Oscurece un color 0RGB (para desvanecer textos).
fn dim_color(c: u32, k: f32) -> u32 {
    let k = k.clamp(0.0, 1.0);
    let f = |sh: u32| ((((c >> sh) & 0xff) as f32 * k) as u32) << sh;
    f(16) | f(8) | f(0)
}

/// Sale un poco de mas y regresa: da el "pop" de la burbuja.
fn ease_out_back(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0) - 1.0;
    1.0 + 2.7 * t * t * t + 1.7 * t * t
}

pub struct App {
    pub time: f32,
    galaxy: Galaxy,
    positions: Vec<Vec3>,
    prev_positions: Vec<Vec3>,
    ship: Ship,
    cam_yaw: f32,
    map_open: bool,
    map_blend: f32,
    transition: Option<Transition>,
    ship_mesh: Mesh,
    spheres: Vec<Mesh>,
    ring_mesh: Mesh,
    jump: Option<Jump>,
    visit: Option<Visit>,
    /// Recarga del mundo en segundo plano al salir de un planeta
    visit_loader: Option<JoinHandle<Galaxy>>,
    /// Planeta que muestra la burbuja y su animacion de aparicion (0..1)
    bubble_body: Option<usize>,
    bubble_k: f32,
    bubble_hover: bool,
    /// Buffer de media resolucion para las animaciones de atmosfera
    low: Framebuffer,
    banner: f32,
    renderer: Renderer,
    show_help: bool,
    show_stats: bool,
    lod_enabled: bool,
    local_body: Option<usize>,
    hover: Option<usize>,
    fps: f32,
    // Camara del ultimo frame (para proyectar marcadores y el clic del mapa)
    view: Mat4,
    view_proj: Mat4,
    fov: f32,
    stats: RenderStats,
}

impl App {
    pub fn new(ship_mesh: Mesh) -> Self {
        let threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let galaxy = Galaxy::build(0, 0.0, threads);
        let mut app = App {
            time: 0.0,
            positions: vec![Vec3::zeros(); galaxy.bodies.len()],
            prev_positions: vec![Vec3::zeros(); galaxy.bodies.len()],
            galaxy,
            ship: Ship::new(Vec3::new(0.0, 22.0, 230.0), 0.0),
            cam_yaw: 0.0,
            map_open: false,
            map_blend: 0.0,
            transition: None,
            ship_mesh,
            // Niveles de detalle: 20, 80, 320, 1280, 5120 y 20480 triangulos
            spheres: (0..=MAX_LOD as u32).map(icosphere).collect(),
            ring_mesh: ring(1.35, 2.3, 160),
            jump: None,
            visit: None,
            visit_loader: None,
            bubble_body: None,
            bubble_k: 0.0,
            bubble_hover: false,
            low: Framebuffer::new(visit::LOW_W, visit::LOW_H),
            banner: BANNER_TIME,
            renderer: Renderer::new(WIDTH, HEIGHT),
            show_help: false,
            show_stats: false,
            lod_enabled: true,
            local_body: None,
            hover: None,
            fps: 60.0,
            view: Mat4::identity(),
            view_proj: Mat4::identity(),
            fov: BASE_FOV,
            stats: RenderStats::default(),
        };
        app.update_positions();
        app.prev_positions = app.positions.clone();
        app
    }

    fn update_positions(&mut self) {
        for i in 0..self.galaxy.bodies.len() {
            self.positions[i] = self.galaxy.bodies[i].position(self.time, &self.positions);
        }
    }

    // ------------------------------------------------------------ update

    pub fn update(&mut self, dt: f32, input: &Input) {
        let dt = dt.clamp(0.0, 0.05);
        self.time += dt;
        if dt > 0.0 {
            self.fps += (1.0 / dt - self.fps) * approach(3.0, dt);
        }
        self.prev_positions.clone_from(&self.positions);
        self.update_positions();

        for action in &input.actions {
            match action {
                Action::Warp(key) => {
                    if self.jump.is_some() || self.visit.is_some() {
                        continue;
                    }
                    if let Some(i) = self.galaxy.bodies.iter().position(|b| b.warp_key == *key) {
                        self.start_warp(WarpTarget::Body(i));
                    }
                }
                Action::Visit => {
                    if let Some(i) = self.visitable().filter(|_| self.free_flight()) {
                        self.start_visit(i);
                    }
                }
                Action::ToggleMap => {
                    if self.transition.is_none() && self.jump.is_none() && self.visit.is_none() {
                        self.map_open = !self.map_open;
                    }
                }
                Action::ToggleHelp => self.show_help = !self.show_help,
                Action::ToggleStats => self.show_stats = !self.show_stats,
                Action::ToggleLod => self.lod_enabled = !self.lod_enabled,
                Action::ToggleCulling => self.renderer.frustum_culling = !self.renderer.frustum_culling,
            }
        }

        self.update_transition(dt);
        self.update_jump(dt);
        self.update_visit(dt, input);
        self.banner = (self.banner - dt).max(0.0);

        let goal = if self.map_open { 1.0 } else { 0.0 };
        let step = dt / MAP_TRANSITION;
        self.map_blend = if self.map_blend < goal { (self.map_blend + step).min(goal) } else { (self.map_blend - step).max(goal) };

        // La nave solo obedece en vuelo normal; en el mapa o en un warp flota
        let controls = self.free_flight();
        let ship_input = if controls { input.ship } else { ShipInput::default() };
        let falling = matches!(self.jump.as_ref().map(|j| j.phase), Some(JumpPhase::Enter | JumpPhase::Tunnel));
        // Al entrar y al despegar de un planeta tambien
        let burning = matches!(self.visit.map(|v| v.phase), Some(VisitPhase::Enter | VisitPhase::Leave));
        if falling || burning {
            // Durante la caida los motores van a todo
            self.ship.thrust = 1.0;
            self.ship.boost = 1.0;
        } else {
            self.ship.update(dt, &ship_input);
        }

        if self.jump.is_none() {
            self.apply_portal_gravity(dt);
            self.apply_local_frame();
            self.apply_collisions();
            if self.visit.is_none() {
                self.check_portal();
            }
        }
        self.update_bubble(dt, input, controls);
        self.cam_yaw += angle_diff(self.ship.yaw, self.cam_yaw) * approach(4.0, dt);

        self.update_map_input(input);
        if !self.visit.is_some_and(|v| v.world_hidden()) {
            self.galaxy.sky.update(self.time);
        }
    }

    /// Vuelo normal: sin mapa, sin warp, sin viaje y sin visita en curso.
    fn free_flight(&self) -> bool {
        self.transition.is_none() && self.jump.is_none() && self.visit.is_none() && !self.map_open && self.map_blend < 0.01
    }

    // ------------------------------------------------------------ visitas

    /// Cuerpo con animacion de atmosfera que esta lo bastante cerca de la nave.
    fn visitable(&self) -> Option<usize> {
        let mut best = None;
        let mut best_d = f32::MAX;
        for (i, b) in self.galaxy.bodies.iter().enumerate() {
            if visit::scene_for(b.name).is_none() {
                continue;
            }
            let surface = (self.ship.position - self.positions[i]).norm() - b.collision_radius();
            let range = if b.is_sun() { 70.0 } else { b.radius * 2.5 + 20.0 };
            if surface < range && surface < best_d {
                best_d = surface;
                best = Some(i);
            }
        }
        best
    }

    fn start_visit(&mut self, body: usize) {
        let Some(scene) = visit::scene_for(self.galaxy.bodies[body].name) else { return };
        self.visit = Some(Visit { phase: VisitPhase::Enter, t: 0.0, body, scene, leave_time: 0.0 });
        self.ship.velocity = Vec3::zeros();
        self.bubble_body = None;
        self.bubble_k = 0.0;
        self.bubble_hover = false;
    }

    /// Para pruebas sin ventana: visita el cuerpo mas cercano sin importar la distancia.
    pub fn debug_visit(&mut self) {
        let nearest = (0..self.galaxy.bodies.len())
            .filter(|&i| visit::scene_for(self.galaxy.bodies[i].name).is_some())
            .min_by(|&a, &b| {
                let d = |i: usize| (self.ship.position - self.positions[i]).norm() - self.galaxy.bodies[i].radius;
                d(a).total_cmp(&d(b))
            });
        if let Some(i) = nearest {
            self.start_visit(i);
        }
    }

    fn update_visit(&mut self, dt: f32, input: &Input) {
        let Some(v) = &mut self.visit else { return };
        v.t += dt;
        match v.phase {
            VisitPhase::Enter => {
                if v.t >= VISIT_ENTER {
                    // Disparador 1: el tunel ya cubre todo, se descarga el mundo
                    self.galaxy.unload();
                    v.phase = VisitPhase::Show;
                    v.t = 0.0;
                }
            }
            VisitPhase::Show => {
                let dur = v.scene.duration();
                // Un click salta al final de la animacion
                if input.click && v.t > VISIT_FADE && v.t < dur - VISIT_FADE {
                    v.t = dur - VISIT_FADE;
                }
                if v.t >= dur {
                    // Disparador 2: se recarga el mundo en otro hilo (con 1
                    // hilo, para no quitarle nucleos al render del despegue)
                    let index = self.galaxy.index;
                    self.visit_loader = Some(thread::spawn(move || Galaxy::build(index, 0.0, 1)));
                    v.phase = VisitPhase::Leave;
                    v.t = 0.0;
                }
            }
            VisitPhase::Leave => {
                let ready = self.visit_loader.as_ref().is_some_and(|l| l.is_finished());
                if v.t >= VISIT_LEAVE_MIN && ready {
                    let mut galaxy = self.visit_loader.take().unwrap().join().expect("no se pudo recargar la galaxia");
                    galaxy.sky.rebase(self.time);
                    self.galaxy = galaxy;
                    v.leave_time = v.t;
                    v.phase = VisitPhase::Exit;
                    v.t = 0.0;
                }
            }
            VisitPhase::Exit => {
                if v.t >= VISIT_EXIT {
                    self.visit = None;
                }
            }
        }
    }

    /// La burbuja aparece al acercarse a un planeta visitable y se puede
    /// pulsar con el mouse.
    fn update_bubble(&mut self, dt: f32, input: &Input, controls: bool) {
        let candidate = if controls { self.visitable() } else { None };
        match candidate {
            Some(i) => {
                if self.bubble_body != Some(i) {
                    self.bubble_body = Some(i);
                    self.bubble_k = 0.0;
                }
                self.bubble_k = (self.bubble_k + dt / 0.45).min(1.0);
            }
            None => {
                self.bubble_k = (self.bubble_k - dt / 0.2).max(0.0);
                if self.bubble_k == 0.0 {
                    self.bubble_body = None;
                }
            }
        }
        self.bubble_hover = false;
        let (Some(i), Some((cx, cy, _, _)), Some((mx, my))) = (candidate, self.bubble_rect(), input.mouse) else { return };
        self.bubble_hover = (mx - cx).abs() < BUBBLE_W * 0.5 && (my - cy).abs() < BUBBLE_H * 0.5;
        if self.bubble_hover && input.click && self.bubble_k > 0.5 {
            self.start_visit(i);
        }
    }

    /// Centro de la burbuja en pantalla y punto del planeta al que apunta.
    fn bubble_rect(&self) -> Option<(f32, f32, f32, f32)> {
        let i = self.bubble_body?;
        let b = &self.galaxy.bodies[i];
        let p = self.positions[i];
        let (sx, _, _) = self.project(&p)?;
        if !(-150.0..WIDTH as f32 + 150.0).contains(&sx) {
            return None;
        }
        // Encima del planeta, segun el "arriba" de la camara
        let up = Vec3::new(self.view[(1, 0)], self.view[(1, 1)], self.view[(1, 2)]);
        let (ax, ay, _) = self.project(&(p + up * b.collision_radius() * 1.05))?;
        let bob = (self.time * 3.0).sin() * 2.5;
        let cx = ax.clamp(BUBBLE_W * 0.5 + 8.0, WIDTH as f32 - BUBBLE_W * 0.5 - 8.0);
        let cy = (ay - 16.0 - BUBBLE_H * 0.5 + bob).clamp(BUBBLE_H * 0.5 + 60.0, HEIGHT as f32 - BUBBLE_H - 40.0);
        Some((cx, cy, ax, ay))
    }

    /// Modo de control local: cerca de un planeta, la nave se mueve junto con
    /// el (se le suma el desplazamiento orbital del planeta en este frame).
    /// Asi se puede rodear un planeta sin que se aleje por su orbita.
    fn apply_local_frame(&mut self) {
        self.local_body = None;
        let mut best = f32::MAX;
        for (i, b) in self.galaxy.bodies.iter().enumerate() {
            if b.is_sun() || b.is_portal() {
                continue;
            }
            let surface = (self.ship.position - self.positions[i]).norm() - b.radius;
            if surface < b.radius * 3.0 + 20.0 && surface < best {
                best = surface;
                self.local_body = Some(i);
            }
        }
        // Durante una visita la nave se queda junto al planeta visitado
        if let Some(v) = &self.visit {
            self.local_body = Some(v.body);
        }
        if let Some(i) = self.local_body {
            self.ship.position += self.positions[i] - self.prev_positions[i];
        }
    }

    /// Colision basica: la nave no puede atravesar ningun cuerpo.
    fn apply_collisions(&mut self) {
        for (i, b) in self.galaxy.bodies.iter().enumerate() {
            if b.is_portal() {
                continue; // el portal se cruza, no se choca
            }
            let min = b.collision_radius() + if b.is_sun() { 6.0 } else { 1.6 };
            let d = self.ship.position - self.positions[i];
            let dist = d.norm();
            if dist < min && dist > 1e-4 {
                let n = d / dist;
                self.ship.position = self.positions[i] + n * min;
                let vn = self.ship.velocity.dot(&n);
                if vn < 0.0 {
                    self.ship.velocity -= n * vn;
                }
            }
        }
        let r = self.ship.position.norm();
        if r > WORLD_LIMIT {
            let n = self.ship.position / r;
            self.ship.position = n * WORLD_LIMIT;
            let vn = self.ship.velocity.dot(&n);
            if vn > 0.0 {
                self.ship.velocity -= n * vn;
            }
        }
    }

    fn start_warp(&mut self, target: WarpTarget) {
        if self.transition.is_some() {
            return;
        }
        let label = match target {
            WarpTarget::Body(i) => self.galaxy.bodies[i].name.to_string(),
            WarpTarget::Point(_) => "COORDENADAS".to_string(),
        };
        self.transition = Some(Transition { phase: Phase::Closing, t: 0.0, target, label });
    }

    /// Animacion de "pantalla de carga": el iris se cierra sobre la nave, en
    /// negro se teletransporta, y el iris se vuelve a abrir en el destino.
    fn update_transition(&mut self, dt: f32) {
        let Some(tr) = &mut self.transition else { return };
        tr.t += dt;
        let mut arrive = None;
        match tr.phase {
            Phase::Closing if tr.t >= CLOSE_TIME => {
                arrive = Some(tr.target);
                tr.phase = Phase::Hold;
                tr.t = 0.0;
            }
            Phase::Hold if tr.t >= HOLD_TIME => {
                tr.phase = Phase::Opening;
                tr.t = 0.0;
            }
            Phase::Opening if tr.t >= OPEN_TIME => self.transition = None,
            _ => {}
        }
        if let Some(target) = arrive {
            self.arrive(target);
        }
    }

    fn arrive(&mut self, target: WarpTarget) {
        let (position, look_at_point) = match target {
            WarpTarget::Body(i) => {
                let b = &self.galaxy.bodies[i];
                let p = self.positions[i];
                if b.is_sun() {
                    (Vec3::new(0.0, 20.0, b.radius * 4.5), p)
                } else {
                    // Llegamos del lado iluminado: entre el sol y el planeta
                    let to_sun = Vec3::new(-p.x, 0.0, -p.z).normalize();
                    let reach = if b.is_portal() {
                        b.radius * 2.6
                    } else {
                        b.collision_radius() * if b.has_ring { 3.4 } else { 3.0 } + 8.0
                    };
                    (p + to_sun * reach + Vec3::new(0.0, b.radius * 0.5, 0.0), p)
                }
            }
            WarpTarget::Point(q) => (q + Vec3::new(0.0, 4.0, 0.0), q + self.ship.forward()),
        };
        let dir = look_at_point - position;
        self.ship.position = position;
        self.ship.velocity = Vec3::zeros();
        self.ship.yaw = (-dir.x).atan2(-dir.z);
        self.cam_yaw = self.ship.yaw;
        self.map_open = false;
        self.map_blend = 0.0;
        self.update_positions();
        self.prev_positions.clone_from(&self.positions);
    }

    // ------------------------------------------------------------ portal

    /// Atraccion gravitatoria del agujero negro: modifica la velocidad de la
    /// nave con una aceleracion que crece con el inverso del cuadrado de la
    /// distancia (limitada para que no sea violenta).
    fn apply_portal_gravity(&mut self, dt: f32) {
        for (i, b) in self.galaxy.bodies.iter().enumerate() {
            if !b.is_portal() {
                continue;
            }
            let d = self.positions[i] - self.ship.position;
            let dist = d.norm();
            if dist < b.radius * 6.0 && dist > 1e-3 {
                let accel = (PORTAL_PULL * b.radius * b.radius / (dist * dist)).min(60.0);
                self.ship.velocity += d / dist * accel * dt;
            }
        }
    }

    /// Si la nave cruza el portal, empieza el viaje a la siguiente galaxia.
    fn check_portal(&mut self) {
        if self.transition.is_some() {
            return;
        }
        let entered = self.galaxy.bodies.iter().enumerate().find(|(i, b)| {
            b.is_portal() && (self.ship.position - self.positions[*i]).norm() < b.radius * 0.8
        });
        if let Some((i, _)) = entered {
            let center = self.positions[i];
            self.start_jump(center);
        }
    }

    /// Lanza la carga de la siguiente galaxia en otro hilo (con 1 hilo para
    /// hornear el cielo, asi no le quita nucleos al render) y arranca la
    /// animacion del tunel.
    pub fn start_jump(&mut self, center: Vec3) {
        let target = self.galaxy.next_index();
        let loader = thread::spawn(move || Galaxy::build(target, 0.0, 1));
        self.jump = Some(Jump {
            phase: JumpPhase::Enter,
            t: 0.0,
            loader: Some(loader),
            center,
            from_tint: self.galaxy.sky.tint(),
            to_tint: galaxy::style(target).tint,
        });
        self.map_open = false;
    }

    fn update_jump(&mut self, dt: f32) {
        let Some(jump) = &mut self.jump else { return };
        jump.t += dt;
        let mut arrived = None;
        match jump.phase {
            JumpPhase::Enter => {
                // La nave es tragada hacia el centro
                self.ship.position += (jump.center - self.ship.position) * approach(1.8, dt);
                if jump.t >= JUMP_ENTER {
                    jump.phase = JumpPhase::Tunnel;
                    jump.t = 0.0;
                }
            }
            JumpPhase::Tunnel => {
                let ready = jump.loader.as_ref().is_some_and(|l| l.is_finished());
                if jump.t >= JUMP_TUNNEL_MIN && ready {
                    arrived = jump.loader.take().map(|l| l.join().expect("no se pudo cargar la galaxia"));
                    jump.phase = JumpPhase::Exit;
                    jump.t = 0.0;
                }
            }
            JumpPhase::Exit => {
                if jump.t >= JUMP_EXIT {
                    self.jump = None;
                }
            }
        }
        if let Some(galaxy) = arrived {
            self.enter_galaxy(galaxy);
        }
    }

    /// Cambia a la galaxia nueva (con la pantalla cubierta por el tunel).
    fn enter_galaxy(&mut self, mut galaxy: Galaxy) {
        galaxy.sky.rebase(self.time);
        self.galaxy = galaxy;
        self.positions = vec![Vec3::zeros(); self.galaxy.bodies.len()];
        self.update_positions();
        self.prev_positions.clone_from(&self.positions);

        // Salimos del hiperespacio con inercia, mirando hacia la estrella
        let (position, yaw) = galaxy::arrival();
        self.ship.position = position;
        self.ship.yaw = yaw;
        self.ship.velocity = self.ship.forward() * 70.0;
        self.cam_yaw = yaw;
        self.local_body = None;
        self.hover = None;
        self.map_open = false;
        self.map_blend = 0.0;
        self.banner = BANNER_TIME;
    }

    pub fn ship_position(&self) -> Vec3 {
        self.ship.position
    }

    /// Para pruebas sin ventana: cambia de galaxia al instante.
    pub fn set_galaxy(&mut self, index: usize) {
        let threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        self.enter_galaxy(Galaxy::build(index, self.time, threads));
        self.ship.velocity = Vec3::zeros();
    }

    // ------------------------------------------------------------ mapa

    fn project(&self, p: &Vec3) -> Option<(f32, f32, f32)> {
        let c = self.view_proj * Vec4::new(p.x, p.y, p.z, 1.0);
        if c.w <= 0.01 {
            return None;
        }
        let sx = (c.x / c.w + 1.0) * 0.5 * WIDTH as f32;
        let sy = (1.0 - c.y / c.w) * 0.5 * HEIGHT as f32;
        Some((sx, sy, c.w))
    }

    fn pixels_per_unit(&self, w: f32) -> f32 {
        (HEIGHT as f32 * 0.5) / ((self.fov.to_radians() * 0.5).tan() * w)
    }

    fn update_map_input(&mut self, input: &Input) {
        self.hover = None;
        if !self.map_open || self.map_blend < 0.98 || self.transition.is_some() || self.jump.is_some() {
            return;
        }
        let Some((mx, my)) = input.mouse else { return };

        let mut best = f32::MAX;
        for (i, b) in self.galaxy.bodies.iter().enumerate() {
            if let Some((sx, sy, w)) = self.project(&self.positions[i]) {
                let r = (b.radius * self.pixels_per_unit(w) + 8.0).max(16.0);
                let d = ((sx - mx).powi(2) + (sy - my).powi(2)).sqrt();
                if d < r && d < best {
                    best = d;
                    self.hover = Some(i);
                }
            }
        }
        if !input.click {
            return;
        }
        if let Some(i) = self.hover {
            self.start_warp(WarpTarget::Body(i));
        } else if let Some(p) = self.mouse_to_plane(mx, my) {
            self.start_warp(WarpTarget::Point(p));
        }
    }

    /// Rayo desde la camara por el pixel del mouse, intersectado con y = 0.
    fn mouse_to_plane(&self, mx: f32, my: f32) -> Option<Vec3> {
        let inv = self.view_proj.try_inverse()?;
        let nx = mx / WIDTH as f32 * 2.0 - 1.0;
        let ny = 1.0 - my / HEIGHT as f32 * 2.0;
        let unproject = |z: f32| {
            let p = inv * Vec4::new(nx, ny, z, 1.0);
            p.xyz() / p.w
        };
        let (a, b) = (unproject(-1.0), unproject(1.0));
        let dir = b - a;
        if dir.y.abs() < 1e-6 {
            return None;
        }
        let t = -a.y / dir.y;
        (t > 0.0).then(|| a + dir * t)
    }

    // ------------------------------------------------------------ camara

    /// Tercera persona: el objetivo es el centro de la nave y la camara se
    /// coloca a distancia constante detras de ella (siguiendo su yaw con un
    /// pequeno retraso). Si quedara dentro de un planeta, se empuja afuera.
    fn third_person(&self) -> (Vec3, Vec3) {
        let target = self.ship.position;
        let offset = Vec3::new(self.cam_yaw.sin() * CAM_DISTANCE, CAM_HEIGHT, self.cam_yaw.cos() * CAM_DISTANCE);
        let mut eye = target + offset;
        for (i, b) in self.galaxy.bodies.iter().enumerate() {
            if b.is_portal() {
                continue;
            }
            let d = eye - self.positions[i];
            let min = b.collision_radius() + 1.0;
            if d.norm() < min {
                eye = self.positions[i] + d.normalize() * min;
            }
        }
        (eye, target)
    }

    fn camera(&self) -> (Vec3, Vec3, Vec3, f32) {
        let (tp_eye, tp_target) = self.third_person();
        let s = smoothstep(0.0, 1.0, self.map_blend);
        let eye = lerp3(&tp_eye, &Vec3::new(0.0, MAP_HEIGHT, 0.0), s);
        let target = lerp3(&tp_target, &Vec3::zeros(), s);
        let up = lerp3(&Vec3::y(), &Vec3::new(0.0, 0.0, -1.0), s).normalize();

        let kick = match &self.transition {
            Some(tr) => match tr.phase {
                Phase::Closing => ease_in_cubic(tr.t / CLOSE_TIME),
                Phase::Hold => 1.0,
                Phase::Opening => 1.0 - ease_out_cubic((tr.t / OPEN_TIME).min(1.0)),
            },
            None => 0.0,
        };
        let jump = self.jump.as_ref().map(|j| j.alpha()).unwrap_or(0.0);
        let visit = self.visit.map(|v| v.fov_kick()).unwrap_or(0.0);
        let fov = BASE_FOV + WARP_FOV_KICK * kick + BOOST_FOV_KICK * self.ship.boost + JUMP_FOV_KICK * jump + VISIT_FOV_KICK * visit;
        (eye, target, up, fov)
    }

    fn lod_for(&self, radius: f32, distance: f32) -> usize {
        if !self.lod_enabled {
            return MAX_LOD;
        }
        let px = radius * self.pixels_per_unit(distance.max(0.01));
        match px {
            p if p > 260.0 => 5,
            p if p > 110.0 => 4,
            p if p > 40.0 => 3,
            p if p > 12.0 => 2,
            _ => 1,
        }
    }

    // ------------------------------------------------------------ render

    pub fn render(&mut self, fb: &mut Framebuffer) {
        let (eye, target, up, fov) = self.camera();
        let aspect = WIDTH as f32 / HEIGHT as f32;
        let view = look_at(&eye, &target, &up);
        let projection = perspective(fov.to_radians(), aspect, NEAR, FAR);
        self.view = view;
        self.view_proj = projection * view;
        self.fov = fov;

        // Planetas ordenados de cerca a lejos: la prueba de profundidad
        // descarta antes los pixeles tapados y se ahorran fragment shaders.
        let mut order: Vec<usize> = (0..self.galaxy.bodies.len()).collect();
        let dist = |i: usize| (self.positions[i] - eye).norm() - self.galaxy.bodies[i].radius;
        order.sort_by(|&a, &b| dist(a).total_cmp(&dist(b)));
        let lods: Vec<usize> = (0..self.galaxy.bodies.len())
            .map(|i| self.lod_for(self.galaxy.bodies[i].radius, (self.positions[i] - eye).norm()))
            .collect();

        fb.clear(0);
        let base = Uniforms {
            model: Mat4::identity(),
            normal_matrix: glm::Mat3::identity(),
            view,
            projection,
            viewport: viewport(WIDTH as f32, HEIGHT as f32),
            time: self.time,
            camera_pos: eye,
            light_pos: self.positions[0],
            params: [0.0; 4],
            sky: Some(&self.galaxy.sky),
        };
        // Tunel (viaje entre galaxias o visita a un planeta) y efecto de turbo
        let (tunnel_alpha, tunnel_tint, turbo) = match (&self.jump, &self.visit) {
            (Some(j), _) => (j.alpha(), j.tint(), j.alpha()),
            (None, Some(v)) => {
                let label = unpack_color(self.galaxy.bodies[v.body].label_color);
                (v.tunnel(), lerp3(&self.galaxy.sky.tint(), &label, 0.6), v.turbo())
            }
            (None, None) => (0.0, Vec3::zeros(), 0.0),
        };

        // Animacion de la atmosfera: reemplaza por completo a la escena 3D.
        // Al final se funde al color del aire, donde empieza el despegue.
        if let Some(v) = self.visit.filter(|v| v.phase == VisitPhase::Show) {
            visit::render(v.scene, v.t, fb, &mut self.low);
            self.hyperspace_overlay(fb, eye, target, up, fov, tunnel_alpha, tunnel_tint);
            let dur = v.scene.duration();
            let out = smoothstep(dur - VISIT_FADE, dur, v.t);
            if out > 0.0 {
                hud::fill_rect(fb, 0, 0, WIDTH as i32, HEIGHT as i32, visit::air_color(v.scene) * 0.85, out);
            }
            self.stats = RenderStats::default();
            self.draw_visit_hud(fb, &v);
            return;
        }
        // Despegue: (escena, tiempo, opacidad). En `Leave` es todo el fondo;
        // en `Exit` se desvanece sobre el mundo recien cargado.
        let leave = self.visit.and_then(|v| match v.phase {
            VisitPhase::Leave => Some((v.scene, v.t, 1.0)),
            VisitPhase::Exit => Some((v.scene, v.leave_time + v.t, 1.0 - smoothstep(0.0, VISIT_EXIT, v.t))),
            _ => None,
        });
        let world_hidden = self.visit.is_some_and(|v| v.world_hidden());

        // Pase 1: el mundo. Si el tunel cubre toda la pantalla no hace falta.
        let mut stats = RenderStats::default();
        if tunnel_alpha < 0.999 && !world_hidden {
            let mut frame = self.renderer.begin();

            // Estrella, planetas y portal: para cada uno se cambia la matriz
            // de modelo, la malla (segun su forma y nivel de detalle) y el shader activo
            for &i in &order {
                let b = &self.galaxy.bodies[i];
                let p = self.positions[i];
                let uniforms = base.with_model(b.model_matrix(&p, self.time)).with_params(b.params);
                if b.is_portal() {
                    // Desde afuera basta con las caras frontales. Desde adentro de
                    // la lente todas las caras quedan de espaldas: sin descarte se
                    // dibuja exactamente una capa.
                    let inside = (eye - p).norm() < b.radius * 1.02;
                    let cull = if inside { CullMode::None } else { CullMode::Back };
                    let lod = (lods[i] + 1).min(MAX_LOD);
                    frame.draw(&self.spheres[lod], uniforms, b.shader, cull);
                    continue;
                }
                let mesh = match &self.galaxy.shape_meshes[i] {
                    Some(levels) => &levels[lods[i]],
                    None => &self.spheres[lods[i]],
                };
                frame.draw(mesh, uniforms, b.shader, CullMode::Back);
                if b.has_ring {
                    let ring_uniforms = base.with_model(b.ring_matrix(&p)).with_params([b.radius, 0.0, 0.0, 0.0]);
                    frame.draw(&self.ring_mesh, ring_uniforms, shaders::ring_shader, CullMode::None);
                }
            }

            // Cielo al final: solo se sombrean los pixeles que quedaron vacios
            let sky_model = glm::scaling(&Vec3::repeat(SKY_RADIUS));
            let look = (target - eye).normalize();
            let sky_params = [self.ship.boost.max(turbo), look.x, look.y, look.z];
            frame.draw(&self.spheres[3], base.with_model(sky_model).with_params(sky_params), shaders::sky_shader, CullMode::None);
            stats = frame.finish(fb);
            self.sun_glow(fb);
        }

        // Tunel hiperespacial encima de la escena
        self.hyperspace_overlay(fb, eye, target, up, fov, tunnel_alpha, tunnel_tint);
        if let Some((scene, t, alpha)) = leave {
            visit::render_leave(scene, t, alpha, fb, &mut self.low);
        }

        // Pase 2: la nave, siempre encima del mundo. El buffer de profundidad
        // se limpia para que nada (por ejemplo la lente del agujero negro
        // cuando la nave ya esta adentro) la tape; sus caras se siguen
        // ordenando entre si con la prueba de profundidad.
        fb.depth.fill(0.0);
        let mut frame = self.renderer.begin();
        frame.draw(
            &self.ship_mesh,
            base.with_model(self.ship.model_matrix()).with_params([self.ship.thrust, self.ship.boost, 0.0, 0.0]),
            shaders::ship_shader,
            CullMode::Back,
        );
        let ship_stats = frame.finish(fb);
        stats.objects += ship_stats.objects;
        stats.tris_submitted += ship_stats.tris_submitted;
        stats.tris_backface += ship_stats.tris_backface;
        stats.tris_clipped += ship_stats.tris_clipped;
        stats.tris_raster += ship_stats.tris_raster;
        stats.fragments += ship_stats.fragments;
        stats.objects_culled += ship_stats.objects_culled;
        stats.tris_culled_frustum += ship_stats.tris_culled_frustum;
        self.stats = stats;

        if self.map_blend > 0.5 {
            self.draw_map(fb, smoothstep(0.5, 1.0, self.map_blend));
        }
        self.draw_hud(fb);
        self.draw_transition(fb);
    }

    /// Mezcla el tunel hiperespacial sobre la imagen con opacidad `alpha`.
    /// Se calcula la direccion de cada pixel con la base de la camara y el
    /// FOV, y se reparte por filas entre los hilos.
    #[allow(clippy::too_many_arguments)]
    fn hyperspace_overlay(&self, fb: &mut Framebuffer, eye: Vec3, target: Vec3, up: Vec3, fov: f32, alpha: f32, tint: Vec3) {
        if alpha <= 0.001 {
            return;
        }
        let forward = (target - eye).normalize();
        let right = forward.cross(&up).normalize();
        let cam_up = right.cross(&forward);
        let tan_y = (fov.to_radians() * 0.5).tan();
        let tan_x = tan_y * WIDTH as f32 / HEIGHT as f32;
        let time = self.time;
        fb.par_rows(|y, row| {
            let ny = 1.0 - (y as f32 + 0.5) / HEIGHT as f32 * 2.0;
            for (x, pixel) in row.iter_mut().enumerate() {
                let nx = (x as f32 + 0.5) / WIDTH as f32 * 2.0 - 1.0;
                let dir = (forward + right * (nx * tan_x) + cam_up * (ny * tan_y)).normalize();
                let tunnel = hyperspace(&dir, &forward, time, &tint);
                let old = crate::framebuffer::unpack_color(*pixel);
                *pixel = pack_color(&(old + (tunnel - old) * alpha));
            }
        });
    }

    /// Halo del sol: se suma luz alrededor del disco solo en pixeles que estan
    /// mas lejos que el sol (asi un planeta delante tapa el halo).
    fn sun_glow(&self, fb: &mut Framebuffer) {
        let sun = &self.galaxy.bodies[0];
        let Some((sx, sy, w)) = self.project(&self.positions[0]) else { return };
        let r = sun.radius * self.pixels_per_unit(w);
        let sun_depth = 1.0 / w;
        let extent = r * 3.5 + 30.0;
        let x0 = ((sx - extent).max(0.0)) as usize;
        let x1 = ((sx + extent).min(WIDTH as f32 - 1.0)).max(0.0) as usize;
        let y0 = ((sy - extent).max(0.0)) as usize;
        let y1 = ((sy + extent).min(HEIGHT as f32 - 1.0)).max(0.0) as usize;
        if x0 >= x1 || y0 >= y1 {
            return;
        }
        // Color del halo segun la paleta de la estrella
        let tint = match sun.params[1] as i32 {
            0 => Vec3::new(1.0, 0.3, 0.15),
            2 => Vec3::new(0.4, 0.6, 1.0),
            3 => Vec3::new(0.45, 1.0, 0.6),
            _ => Vec3::new(1.0, 0.55, 0.2),
        };
        for y in y0..=y1 {
            for x in x0..=x1 {
                let i = y * WIDTH + x;
                if fb.depth[i] >= sun_depth {
                    continue;
                }
                let d = ((x as f32 - sx).powi(2) + (y as f32 - sy).powi(2)).sqrt();
                if d < r * 0.9 {
                    continue;
                }
                let g = (-(d - r).max(0.0) / (r * 0.7 + 10.0)).exp() * 0.65;
                if g > 0.004 {
                    fb.add(i, tint * g);
                }
            }
        }
    }

    fn draw_map(&self, fb: &mut Framebuffer, alpha: f32) {
        let dim = |c: u32, k: f32| {
            let r = (((c >> 16) & 0xff) as f32 * k) as u32;
            let g = (((c >> 8) & 0xff) as f32 * k) as u32;
            let b = ((c & 0xff) as f32 * k) as u32;
            (r << 16) | (g << 8) | b
        };

        // Orbitas punteadas
        for b in &self.galaxy.bodies {
            if b.is_sun() || b.is_portal() {
                continue;
            }
            let center = b.parent.map(|p| self.positions[p]).unwrap_or_default();
            let segments = 120;
            for s in (0..segments).step_by(2) {
                let a0 = s as f32 / segments as f32 * std::f32::consts::TAU;
                let a1 = (s + 1) as f32 / segments as f32 * std::f32::consts::TAU;
                let p0 = center + Vec3::new(a0.cos(), 0.0, a0.sin()) * b.orbit_radius;
                let p1 = center + Vec3::new(a1.cos(), 0.0, a1.sin()) * b.orbit_radius;
                if let (Some(q0), Some(q1)) = (self.project(&p0), self.project(&p1)) {
                    draw_line(fb, q0.0 as i32, q0.1 as i32, q1.0 as i32, q1.1 as i32, dim(b.label_color, 0.45 * alpha));
                }
            }
        }

        // Marcadores y nombres
        for (i, b) in self.galaxy.bodies.iter().enumerate() {
            let Some((sx, sy, w)) = self.project(&self.positions[i]) else { continue };
            let r = b.radius * self.pixels_per_unit(w);
            let color = dim(b.label_color, alpha);
            hud::draw_circle(fb, sx, sy, r + 5.0, color);
            if self.hover == Some(i) {
                hud::draw_circle(fb, sx, sy, r + 8.0, 0xffffff);
                hud::draw_circle(fb, sx, sy, r + 9.0, 0xffffff);
            }
            // Las lunas solo llevan su numero para no tapar el nombre de su planeta
            if b.is_portal() {
                let label = format!("9 PORTAL > {}", galaxy::name(self.galaxy.next_index()));
                hud::draw_text_shadow(fb, (sx + r + 10.0) as i32, (sy - 7.0) as i32, &label, 2, color);
            } else if b.is_sun() {
                let label = format!("{} {}", b.warp_key, b.name);
                let x = sx as i32 - hud::text_width(&label, 2) / 2;
                hud::draw_text_shadow(fb, x, (sy - r - 26.0) as i32, &label, 2, color);
            } else if b.parent.is_some() {
                let label = b.warp_key.to_string();
                hud::draw_text_shadow(fb, (sx - r - 18.0) as i32, (sy + r) as i32, &label, 2, color);
            } else {
                let label = format!("{} {}", b.warp_key, b.name);
                hud::draw_text_shadow(fb, (sx + r + 10.0) as i32, (sy - 7.0) as i32, &label, 2, color);
            }
        }

        // La nave: una flecha en la direccion en que apunta
        if let (Some((sx, sy, _)), Some((fx, fy, _))) =
            (self.project(&self.ship.position), self.project(&(self.ship.position + self.ship.forward() * 25.0)))
        {
            let (dx, dy) = (fx - sx, fy - sy);
            let len = (dx * dx + dy * dy).sqrt().max(1e-3);
            let (ux, uy) = (dx / len, dy / len);
            let tip = (sx + ux * 12.0, sy + uy * 12.0);
            let left = (sx - ux * 7.0 - uy * 7.0, sy - uy * 7.0 + ux * 7.0);
            let right = (sx - ux * 7.0 + uy * 7.0, sy - uy * 7.0 - ux * 7.0);
            let c = 0x40ffb0;
            for (a, b) in [(tip, left), (left, right), (right, tip)] {
                draw_line(fb, a.0 as i32, a.1 as i32, b.0 as i32, b.1 as i32, c);
            }
            hud::draw_text_shadow(fb, sx as i32 + 14, sy as i32 + 6, "NAVE", 1, c);
        }

        let bar = HEIGHT as i32 - 30;
        hud::fill_rect(fb, 0, bar, WIDTH as i32, 30, Vec3::zeros(), 0.6 * alpha);
        let title = match self.hover {
            Some(i) => format!("CLICK PARA VIAJAR A {}", self.galaxy.bodies[i].name),
            None => "CLICK EN UN PLANETA O EN EL ESPACIO PARA VIAJAR - M: VOLVER".to_string(),
        };
        let header = format!("GALAXIA {}/{}: {}", self.galaxy.index + 1, galaxy::GALAXY_COUNT, self.galaxy.name);
        let hx = (WIDTH as i32 - hud::text_width(&header, 2)) / 2;
        hud::draw_text_shadow(fb, hx, 12, &header, 2, dim(0xffffff, alpha));
        let x = (WIDTH as i32 - hud::text_width(&title, 2)) / 2;
        hud::draw_text_shadow(fb, x.max(4), bar + 8, &title, 2, dim(0xffffff, alpha));
    }

    fn draw_hud(&self, fb: &mut Framebuffer) {
        let s = &self.stats;
        let on_off = |b: bool| if b { "ON" } else { "OFF" };

        // Contadores del renderer (para comparar culling y LOD)
        let lines = [
            format!("FPS {:.0}", self.fps),
            format!("TRIANGULOS ENVIADOS {}", s.tris_submitted),
            format!("DESCARTE FRUSTUM {} OBJ / {} TRI", s.objects_culled, s.tris_culled_frustum),
            format!("CARAS TRASERAS {}", s.tris_backface),
            format!("FUERA DE PANTALLA {}", s.tris_clipped),
            format!("RASTERIZADOS {}", s.tris_raster),
            format!("FRAGMENTOS {}K", s.fragments / 1000),
            format!("LOD {} (L)  CULLING {} (V)", on_off(self.lod_enabled), on_off(self.renderer.frustum_culling)),
        ];
        if self.show_stats {
            hud::fill_rect(fb, 6, 6, 250, lines.len() as i32 * 11 + 8, Vec3::zeros(), 0.45);
            for (k, line) in lines.iter().enumerate() {
                hud::draw_text(fb, 12, 11 + k as i32 * 11, line, 1, if k == 0 { 0x80ff80 } else { 0xd0d8e8 });
            }
        }

        if self.map_blend > 0.5 {
            return;
        }

        // Velocidad y planeta mas cercano
        let speed = format!("VEL {:.0}", self.ship.speed());
        let x = WIDTH as i32 - hud::text_width(&speed, 2) - 12;
        hud::draw_text_shadow(fb, x, 10, &speed, 2, if self.ship.boost > 0.5 { 0xff80e0 } else { 0xffffff });
        let nearest = (0..self.galaxy.bodies.len()).min_by(|&a, &b| {
            let da = (self.ship.position - self.positions[a]).norm() - self.galaxy.bodies[a].radius;
            let db = (self.ship.position - self.positions[b]).norm() - self.galaxy.bodies[b].radius;
            da.total_cmp(&db)
        });
        if let Some(i) = nearest {
            let d = (self.ship.position - self.positions[i]).norm() - self.galaxy.bodies[i].radius;
            let text = format!("{} A {:.0}", self.galaxy.bodies[i].name, d.max(0.0));
            let x = WIDTH as i32 - hud::text_width(&text, 1) - 12;
            hud::draw_text_shadow(fb, x, 32, &text, 1, self.galaxy.bodies[i].label_color);
        }
        if let Some(i) = self.local_body {
            let text = format!("ORBITA LOCAL: {}", self.galaxy.bodies[i].name);
            let x = WIDTH as i32 - hud::text_width(&text, 1) - 12;
            hud::draw_text_shadow(fb, x, 46, &text, 1, 0x60ffb0);
        }

        self.draw_bubble(fb);
        if let Some(v) = self.visit.filter(|v| v.phase == VisitPhase::Leave) {
            let text = format!("SALIENDO DE {}", self.galaxy.bodies[v.body].name);
            let k = smoothstep(0.0, 0.5, v.t);
            let x = (WIDTH as i32 - hud::text_width(&text, 2)) / 2;
            hud::draw_text_shadow(fb, x, HEIGHT as i32 - 60, &text, 2, dim_color(self.galaxy.bodies[v.body].label_color, k));
        }

        // Nombre de la galaxia al llegar, se desvanece
        if self.banner > 0.0 && self.bubble_k <= 0.0 && self.visit.is_none() && self.jump.as_ref().is_none_or(|j| j.phase == JumpPhase::Exit) {
            let k = smoothstep(0.0, 1.0, self.banner).min(1.0);
            let title = self.galaxy.name;
            let sub = format!("GALAXIA {} DE {}", self.galaxy.index + 1, galaxy::GALAXY_COUNT);
            let dimk = |c: u32| {
                let f = |sh: u32| ((((c >> sh) & 0xff) as f32 * k) as u32) << sh;
                f(16) | f(8) | f(0)
            };
            let x = (WIDTH as i32 - hud::text_width(title, 4)) / 2;
            hud::draw_text_shadow(fb, x, 70, title, 4, dimk(0xffffff));
            let x = (WIDTH as i32 - hud::text_width(&sub, 2)) / 2;
            hud::draw_text_shadow(fb, x, 108, &sub, 2, dimk(0xa0c8ff));
        }

        if self.show_help {
            let mut keys: Vec<_> = self.galaxy.bodies.iter().map(|b| (b.warp_key, b.name)).collect();
            keys.sort_by_key(|&(k, _)| if k == 0 { 10 } else { k });
            let warp: Vec<String> = keys.iter().map(|(k, n)| format!("{} {}", k, n)).collect();
            let line2 = format!("WARP: {}   M: MAPA", warp.join("  "));
            let help = [
                "W/S O FLECHAS: ACELERAR / FRENAR   A/D: GIRAR   ESPACIO/C: SUBIR/BAJAR   SHIFT: TURBO   F: EXPLORAR   H: OCULTAR",
                line2.as_str(),
            ];
            hud::fill_rect(fb, 0, HEIGHT as i32 - 34, WIDTH as i32, 34, Vec3::zeros(), 0.45);
            for (k, line) in help.iter().enumerate() {
                let x = (WIDTH as i32 - hud::text_width(line, 1)) / 2;
                hud::draw_text(fb, x, HEIGHT as i32 - 28 + k as i32 * 13, line, 1, 0xe0e8ff);
            }
        } else {
            // Pista discreta para saber como abrir los paneles
            hud::draw_text_shadow(fb, 10, HEIGHT as i32 - 16, "H: CONTROLES   TAB: DETALLES", 1, 0x8890a8);
        }
    }

    /// Burbuja sobre el planeta cercano: "EXPLORAR". Aparece con un pequeno
    /// rebote y se ilumina cuando el mouse esta encima.
    fn draw_bubble(&self, fb: &mut Framebuffer) {
        if self.bubble_k <= 0.0 {
            return;
        }
        let (Some(i), Some((cx, cy, ax, ay))) = (self.bubble_body, self.bubble_rect()) else { return };
        let b = &self.galaxy.bodies[i];
        let s = ease_out_back(self.bubble_k) * if self.bubble_hover { 1.06 } else { 1.0 };
        let accent = unpack_color(b.label_color);
        let fill = if self.bubble_hover { lerp3(&Vec3::new(0.06, 0.09, 0.16), &accent, 0.25) } else { Vec3::new(0.04, 0.06, 0.12) };
        let border = if self.bubble_hover { Vec3::repeat(1.0) } else { accent };
        let tip = (ay - 4.0).min(cy + BUBBLE_H * 0.5 * s + 14.0);
        hud::bubble(fb, cx, cy, BUBBLE_W * s, BUBBLE_H * s, (ax.clamp(cx - 60.0, cx + 60.0), tip), fill, border, self.bubble_k.min(1.0));
        if self.bubble_k < 0.8 {
            return;
        }
        // Icono: el planeta con su atmosfera
        let (ix, iy) = (cx - BUBBLE_W * 0.5 + 24.0, cy);
        hud::fill_circle(fb, ix, iy, 13.0, accent, 0.35);
        hud::fill_circle(fb, ix, iy, 9.0, accent, 1.0);
        hud::fill_circle(fb, ix - 3.0, iy - 3.0, 3.0, Vec3::repeat(1.0), 0.45);
        let tx = (cx - BUBBLE_W * 0.5 + 46.0) as i32;
        hud::draw_text(fb, tx, cy as i32 - 15, "EXPLORAR", 2, 0xffffff);
        let sub = format!("{} - CLICK / F", b.name);
        hud::draw_text(fb, tx, cy as i32 + 5, &sub, 1, pack_color(&lerp3(&accent, &Vec3::repeat(1.0), 0.35)));
    }

    /// Textos y barras de cine sobre la animacion de la atmosfera.
    fn draw_visit_hud(&self, fb: &mut Framebuffer, v: &Visit) {
        let dur = v.scene.duration();
        let b = &self.galaxy.bodies[v.body];
        let bars = (46.0 * smoothstep(0.0, 0.9, v.t)) as i32;
        hud::fill_rect(fb, 0, 0, WIDTH as i32, bars, Vec3::zeros(), 1.0);
        hud::fill_rect(fb, 0, HEIGHT as i32 - bars, WIDTH as i32, bars, Vec3::zeros(), 1.0);

        // Solo el nombre del planeta: entra, se queda unos segundos y se va
        let k = smoothstep(0.5, 1.3, v.t) * (1.0 - smoothstep(4.5, 5.5, v.t));
        if k > 0.01 {
            let slide = ((1.0 - smoothstep(0.5, 1.3, v.t)) * 20.0) as i32;
            hud::draw_text_shadow(fb, 40 - slide, 70, b.name, 4, dim_color(b.label_color, k));
        }

        // Barra de progreso y como salir
        let y = HEIGHT as i32 - bars + 20;
        let w = WIDTH as i32 - 80;
        hud::fill_rect(fb, 40, y, w, 2, Vec3::repeat(0.25), 1.0);
        hud::fill_rect(fb, 40, y, (w as f32 * (v.t / dur).clamp(0.0, 1.0)) as i32, 2, unpack_color(b.label_color), 1.0);
        let hint = "CLICK: VOLVER A LA NAVE";
        hud::draw_text(fb, WIDTH as i32 - 40 - hud::text_width(hint, 1), y + 8, hint, 1, 0x8890a8);
    }

    fn draw_transition(&self, fb: &mut Framebuffer) {
        let Some(tr) = &self.transition else { return };
        let max_r = ((WIDTH * WIDTH + HEIGHT * HEIGHT) as f32).sqrt() * 0.6;
        // El iris se cierra sobre la nave (si se ve) o sobre el centro
        let (cx, cy) = self
            .project(&self.ship.position)
            .map(|(x, y, _)| (x.clamp(0.0, WIDTH as f32), y.clamp(0.0, HEIGHT as f32)))
            .unwrap_or((WIDTH as f32 * 0.5, HEIGHT as f32 * 0.5));
        let radius = match tr.phase {
            Phase::Closing => {
                let k = (tr.t / CLOSE_TIME).min(1.0);
                hud::warp_streaks(fb, cx, cy, k);
                max_r * 1.4 * (1.0 - ease_in_cubic(k))
            }
            Phase::Hold => 0.0,
            Phase::Opening => max_r * 1.4 * ease_out_cubic((tr.t / OPEN_TIME).min(1.0)),
        };
        hud::iris(fb, cx, cy, radius, self.time);

        if tr.phase != Phase::Opening || tr.t < OPEN_TIME * 0.5 {
            let text = format!("WARP > {}", tr.label);
            let x = (WIDTH as i32 - hud::text_width(&text, 3)) / 2;
            hud::draw_text_shadow(fb, x, HEIGHT as i32 / 2 - 10, &text, 3, pack_color(&Vec3::new(0.6, 0.95, 1.0)));
        }
    }
}
