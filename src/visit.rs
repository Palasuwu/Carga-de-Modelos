//! "Visitas" a los planetas de la primera galaxia: al acercarse a un planeta
//! (o al sol) aparece una burbuja; al hacerle click se entra con el turbo y
//! se ve una animacion corta de su atmosfera.
//!
//! Cada animacion es un shader de pantalla completa (un fragment shader por
//! pixel, como en Shadertoy): la camara vuela sobre la superficie del planeta
//! y para cada pixel se lanza un rayo que se intersecta con el suelo (plano
//! y = 0) o con capas de nubes (planos horizontales). Todo depende del tiempo
//! de la escena, asi que son animaciones procedurales, no videos.
//!
//! Se calculan a media resolucion (480x270) en paralelo y se escalan con
//! interpolacion bilineal: asi cuestan pocos milisegundos por frame.

use nalgebra_glm::{Vec2, Vec3};

use crate::framebuffer::{Framebuffer, pack_color, unpack_color};
use crate::math::smoothstep;
use crate::noise::{fbm, hash3, mix_bits, perlin, unit, voronoi};
use crate::sky::{SkyBox, plasma};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Scene {
    // Galaxia 1
    Sun,
    Magma,
    Verdalia,
    Luna,
    Duna,
    Bandus,
    Glacia,
    // Galaxia 2
    BlueStar,
    Cubix,
    Espejo,
    Toroide,
    Cristal,
    // Galaxia 3
    GreenStar,
    Cacahuate,
    Ojo,
    Burbuja,
    Coral,
    // Galaxia 4
    RedGiant,
    Vacio,
    Tablero,
    Erizo,
    Halo,
}

/// Que escena tiene cada cuerpo (el portal no tiene).
pub fn scene_for(name: &str) -> Option<Scene> {
    Some(match name {
        "SOL" => Scene::Sun,
        "MAGMA" => Scene::Magma,
        "VERDALIA" => Scene::Verdalia,
        "LUNA" => Scene::Luna,
        "DUNA" => Scene::Duna,
        "BANDUS" => Scene::Bandus,
        "GLACIA" => Scene::Glacia,
        "ESTRELLA AZUL" => Scene::BlueStar,
        "CUBIX" => Scene::Cubix,
        "ESPEJO" => Scene::Espejo,
        "TOROIDE" => Scene::Toroide,
        "CRISTAL" => Scene::Cristal,
        "ESTRELLA VERDE" => Scene::GreenStar,
        "CACAHUATE" => Scene::Cacahuate,
        "OJO" => Scene::Ojo,
        "BURBUJA" => Scene::Burbuja,
        "CORAL" => Scene::Coral,
        "GIGANTE ROJA" => Scene::RedGiant,
        "VACIO" => Scene::Vacio,
        "TABLERO" => Scene::Tablero,
        "ERIZO" => Scene::Erizo,
        "HALO" => Scene::Halo,
        _ => return None,
    })
}

impl Scene {
    /// Duracion de la animacion en segundos.
    pub fn duration(self) -> f32 {
        match self {
            Scene::Sun | Scene::Magma | Scene::Bandus | Scene::Glacia => 11.0,
            _ => 10.0,
        }
    }

    /// Camara de cada escena: altura sobre el suelo, velocidad de avance,
    /// inclinacion (radianes, + mira arriba) y vaiven lateral.
    fn camera(self) -> (f32, f32, f32, f32) {
        match self {
            Scene::Sun | Scene::BlueStar | Scene::GreenStar | Scene::RedGiant => (12.0, 7.0, 0.08, 3.0),
            Scene::Magma => (6.0, 8.0, 0.02, 2.0),
            Scene::Verdalia => (14.0, 18.0, -0.06, 6.0),
            Scene::Luna => (3.0, 4.0, 0.07, 1.0),
            Scene::Duna => (5.0, 10.0, -0.01, 2.0),
            Scene::Bandus => (0.0, 30.0, -0.14, 8.0),
            Scene::Glacia => (4.0, 6.0, 0.16, 1.5),
            Scene::Cubix => (22.0, 14.0, -0.24, 5.0),
            Scene::Espejo => (5.0, 8.0, 0.04, 2.0),
            Scene::Toroide => (7.0, 9.0, 0.1, 2.0),
            Scene::Cristal => (5.0, 7.0, 0.06, 2.0),
            Scene::Cacahuate => (16.0, 12.0, 0.02, 4.0),
            Scene::Ojo => (5.0, 3.0, 0.2, 1.5),
            Scene::Burbuja => (6.0, 5.0, 0.05, 2.0),
            Scene::Coral => (4.0, 5.0, 0.04, 2.0),
            Scene::Vacio => (0.0, 20.0, 0.0, 2.0),
            Scene::Tablero => (5.0, 14.0, 0.03, 2.0),
            Scene::Erizo => (6.0, 12.0, 0.05, 2.0),
            Scene::Halo => (5.0, 0.0, 0.14, 4.0),
        }
    }
}

pub const LOW_W: usize = crate::app::WIDTH / 2;
pub const LOW_H: usize = crate::app::HEIGHT / 2;
const TAN_Y: f32 = 0.70; // ~70 grados de campo vertical

/// Datos de un pixel para el shader de la escena.
struct Px {
    /// coordenadas de pantalla a media resolucion (y hacia abajo)
    x: f32,
    y: f32,
    /// coordenadas normalizadas -1..1 (y hacia arriba)
    nx: f32,
    ny: f32,
    dir: Vec3,
    eye: Vec3,
    /// tiempo de la escena (0..duracion)
    t: f32,
    dur: f32,
}

/// Dibuja la escena `scene` en el tiempo `t` sobre `fb`, usando `low` como
/// buffer de media resolucion.
pub fn render(scene: Scene, t: f32, fb: &mut Framebuffer, low: &mut Framebuffer) {
    let (height, speed, pitch0, sway) = scene.camera();
    let dur = scene.duration();
    let eye = Vec3::new((t * 0.23).sin() * sway, height + (t * 0.4).sin() * (height * 0.06 + 0.2), -speed * t);
    let yaw = (t * 0.17).sin() * 0.06;
    let pitch = pitch0 + (t * 0.31).sin() * 0.015;
    let forward = Vec3::new(yaw.sin() * pitch.cos(), pitch.sin(), -yaw.cos() * pitch.cos());
    let right = forward.cross(&Vec3::y()).normalize();
    let up = right.cross(&forward);
    let tan_x = TAN_Y * LOW_W as f32 / LOW_H as f32;

    low.par_rows(|y, row| {
        for (x, pixel) in row.iter_mut().enumerate() {
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            let nx = fx / LOW_W as f32 * 2.0 - 1.0;
            let mut ny = 1.0 - fy / LOW_H as f32 * 2.0;
            // Calor que ondula el aire en los mundos ardientes
            if matches!(scene, Scene::Magma | Scene::Sun | Scene::BlueStar | Scene::GreenStar | Scene::RedGiant | Scene::Erizo) {
                ny += (nx * 9.0 + ny * 40.0 + t * 7.0).sin() * 0.0025 * smoothstep(0.3, -0.4, ny);
            }
            let dir = (forward + right * (nx * tan_x) + up * (ny * TAN_Y)).normalize();
            let p = Px { x: fx, y: fy, nx, ny, dir, eye, t, dur };
            let mut c = match scene {
                Scene::Sun => star(&p, 0),
                Scene::BlueStar => star(&p, 1),
                Scene::GreenStar => star(&p, 2),
                Scene::RedGiant => star(&p, 3),
                Scene::Magma => magma(&p),
                Scene::Verdalia => verdalia(&p),
                Scene::Luna => luna(&p),
                Scene::Duna => duna(&p),
                Scene::Bandus => bandus(&p),
                Scene::Glacia => glacia(&p),
                Scene::Cubix => cubix(&p),
                Scene::Espejo => espejo(&p),
                Scene::Toroide => toroide(&p),
                Scene::Cristal => cristal(&p),
                Scene::Cacahuate => cacahuate(&p),
                Scene::Ojo => ojo(&p),
                Scene::Burbuja => burbuja(&p),
                Scene::Coral => coral(&p),
                Scene::Vacio => vacio(&p),
                Scene::Tablero => tablero(&p),
                Scene::Erizo => erizo(&p),
                Scene::Halo => halo(&p),
            };
            // Vineta de cine
            c *= 1.0 - 0.32 * (nx * nx * 0.6 + ny * ny);
            *pixel = pack_color(&c);
        }
    });

    upscale(low, fb, 1.0);
}

/// Colores del despegue: (aire del planeta, superficie, borde de la atmosfera).
fn palette(scene: Scene) -> (Vec3, Vec3, Vec3) {
    match scene {
        Scene::Sun => (rgb(1.0, 0.55, 0.15), rgb(1.0, 0.8, 0.3), rgb(1.0, 0.9, 0.6)),
        Scene::Magma => (rgb(0.35, 0.1, 0.05), rgb(0.25, 0.08, 0.04), rgb(1.0, 0.4, 0.1)),
        Scene::Verdalia => (rgb(0.45, 0.65, 0.95), rgb(0.1, 0.35, 0.6), rgb(0.6, 0.8, 1.0)),
        Scene::Luna => (rgb(0.0, 0.0, 0.0), rgb(0.5, 0.5, 0.5), rgb(0.25, 0.25, 0.25)),
        Scene::Duna => (rgb(0.75, 0.48, 0.28), rgb(0.8, 0.5, 0.28), rgb(1.0, 0.75, 0.5)),
        Scene::Bandus => (rgb(0.8, 0.68, 0.5), rgb(0.75, 0.58, 0.4), rgb(1.0, 0.9, 0.75)),
        Scene::Glacia => (rgb(0.1, 0.18, 0.32), rgb(0.6, 0.75, 0.9), rgb(0.7, 0.9, 1.0)),
        Scene::BlueStar => (rgb(0.4, 0.65, 1.0), rgb(0.75, 0.9, 1.0), rgb(0.85, 0.95, 1.0)),
        Scene::Cubix => (rgb(0.2, 0.06, 0.3), rgb(0.1, 0.12, 0.22), rgb(0.15, 0.9, 1.0)),
        Scene::Espejo => (rgb(0.3, 0.2, 0.45), rgb(0.6, 0.6, 0.7), rgb(0.9, 0.9, 1.0)),
        Scene::Toroide => (rgb(1.0, 0.7, 0.82), rgb(1.0, 0.45, 0.65), rgb(1.0, 0.9, 0.5)),
        Scene::Cristal => (rgb(0.3, 0.12, 0.5), rgb(0.4, 0.2, 0.65), rgb(0.8, 0.6, 1.0)),
        Scene::GreenStar => (rgb(0.35, 0.9, 0.45), rgb(0.7, 1.0, 0.6), rgb(0.85, 1.0, 0.8)),
        Scene::Cacahuate => (rgb(0.55, 0.85, 0.6), rgb(0.12, 0.4, 0.14), rgb(0.6, 1.0, 0.65)),
        Scene::Ojo => (rgb(0.4, 0.1, 0.18), rgb(0.5, 0.26, 0.45), rgb(0.95, 0.5, 0.5)),
        Scene::Burbuja => (rgb(0.9, 0.78, 0.95), rgb(0.85, 0.6, 0.9), rgb(1.0, 0.85, 1.0)),
        Scene::Coral => (rgb(0.03, 0.3, 0.4), rgb(0.05, 0.25, 0.35), rgb(0.3, 1.0, 0.9)),
        Scene::RedGiant => (rgb(0.85, 0.2, 0.08), rgb(1.0, 0.4, 0.15), rgb(1.0, 0.6, 0.35)),
        Scene::Vacio => (rgb(0.1, 0.01, 0.03), rgb(0.03, 0.0, 0.02), rgb(1.0, 0.28, 0.4)),
        Scene::Tablero => (rgb(0.5, 0.08, 0.2), rgb(0.5, 0.45, 0.4), rgb(1.0, 0.75, 0.3)),
        Scene::Erizo => (rgb(0.55, 0.2, 0.08), rgb(0.3, 0.18, 0.15), rgb(1.0, 0.45, 0.2)),
        Scene::Halo => (rgb(0.45, 0.1, 0.05), rgb(0.95, 0.35, 0.1), rgb(1.0, 0.85, 0.55)),
    }
}

/// Color del aire del planeta: la animacion termina fundiendose a este color
/// y el despegue empieza desde el.
pub fn air_color(scene: Scene) -> Vec3 {
    palette(scene).0
}

/// Segundos que tarda la nave en salir de la atmosfera.
pub const ASCENT_TIME: f32 = 1.9;

/// Fondo del despegue: la nave sube, el aire del planeta se va quedando
/// atras, el horizonte curvo cae y aparecen las estrellas. Es la "pantalla
/// de carga" mientras el mundo se vuelve a cargar en otro hilo, por eso es
/// barato (media resolucion, sin geometria) y puede durar lo que haga falta:
/// despues del ascenso se queda en el espacio con estelas de velocidad.
/// `alpha` < 1 lo mezcla sobre lo que ya hay en `fb` (para el fundido final).
pub fn render_leave(scene: Scene, t: f32, alpha: f32, fb: &mut Framebuffer, low: &mut Framebuffer) {
    let (air_col, surface, rim) = palette(scene);
    let a = smoothstep(0.0, ASCENT_TIME, t);
    let air = (1.0 - a).powf(1.5);
    let aspect = LOW_W as f32 / LOW_H as f32;
    low.par_rows(|y, row| {
        for (x, pixel) in row.iter_mut().enumerate() {
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            let nx = (fx / LOW_W as f32 * 2.0 - 1.0) * aspect;
            let ny = 1.0 - fy / LOW_H as f32 * 2.0;

            // Espacio con estrellas, que aparecen al salir del aire
            let dir = Vec3::new(nx * 0.7, ny * 0.7, -1.0).normalize();
            let mut c = SkyBox::stars(&dir, t) * a * 1.2;

            // El planeta: un disco enorme cuyo horizonte baja y se curva
            let radius = 3.6 - 1.5 * a;
            let cy = -radius + 0.15 - 0.95 * a;
            let d = (nx * nx + (ny - cy).powi(2)).sqrt() - radius;
            if d < 0.0 {
                let tex = n2(nx * 3.0, (ny - cy) * 3.0 + t * 0.4, 152) * 0.5 + 0.5;
                c = surface * (0.55 + 0.45 * tex) * (0.35 + 0.65 * (1.0 + d * 2.0).clamp(0.0, 1.0));
                c += rim * (d * 14.0).exp() * 0.8;
            } else {
                c += rim * (-d * 9.0).exp() * 0.7;
            }

            // Aire del planeta y jirones de nube que pasan hacia abajo
            c = mix(c, air_col * (0.85 - 0.2 * ny), air);
            if air > 0.02 {
                let wisp = fbm(Vec3::new(nx * 2.0, ny * 0.6 + t * 3.5, t * 0.3), 3, 153);
                c = mix(c, air_col * 1.35 + Vec3::repeat(0.08), smoothstep(0.05, 0.5, wisp) * air * 0.7);
            }

            // Estelas de velocidad que salen del punto de fuga
            c += rgb(0.85, 0.92, 1.0) * radial_streaks(nx, ny, t) * (0.3 + 0.6 * a);
            *pixel = pack_color(&c);
        }
    });
    upscale(low, fb, alpha);
}

/// Estelas radiales: la pantalla se divide en carriles por angulo y cada
/// carril en celdas a lo largo del radio (en escala logaritmica, asi las
/// estelas se aceleran al alejarse del centro).
fn radial_streaks(nx: f32, ny: f32, t: f32) -> f32 {
    let r = (nx * nx + ny * ny).sqrt();
    if r < 0.05 {
        return 0.0;
    }
    let lanes = 90.0;
    let fp = (ny.atan2(nx) / std::f32::consts::TAU + 0.5) * lanes;
    let lane = fp.floor();
    let lateral = (fp - lane - 0.5).abs() * std::f32::consts::TAU / lanes * r;
    let line = 1.0 - smoothstep(0.004, 0.011, lateral);
    if line <= 0.0 {
        return 0.0;
    }
    let h = unit(hash3(lane as i32, 0, 0, 154));
    let u = r.ln() * 3.0 - t * (2.5 + 2.5 * h) + h * 13.0;
    let cell = u.floor();
    let along = u - cell;
    let ch = hash3(lane as i32, cell as i32, 1, 155);
    if unit(ch) > 0.5 {
        return 0.0;
    }
    let len = 0.3 + 0.4 * unit(mix_bits(ch));
    if along > len {
        return 0.0;
    }
    (along / len).powi(2) * line * smoothstep(0.05, 0.35, r)
}

/// Escalado bilineal de `low` a la resolucion de la ventana, mezclado con
/// opacidad `alpha` sobre lo que ya hay.
fn upscale(low: &Framebuffer, fb: &mut Framebuffer, alpha: f32) {
    let (w, h) = (fb.width, fb.height);
    let src = &low.color;
    fb.par_rows(|y, row| {
        let sy = ((y as f32 + 0.5) * LOW_H as f32 / h as f32 - 0.5).clamp(0.0, (LOW_H - 1) as f32);
        let y0 = sy as usize;
        let y1 = (y0 + 1).min(LOW_H - 1);
        let fy = sy - y0 as f32;
        for (x, pixel) in row.iter_mut().enumerate() {
            let sx = ((x as f32 + 0.5) * LOW_W as f32 / w as f32 - 0.5).clamp(0.0, (LOW_W - 1) as f32);
            let x0 = sx as usize;
            let x1 = (x0 + 1).min(LOW_W - 1);
            let fx = sx - x0 as f32;
            let at = |xx: usize, yy: usize| unpack_color(src[yy * LOW_W + xx]);
            let top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
            let bottom = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
            let c = top * (1.0 - fy) + bottom * fy;
            *pixel = if alpha >= 0.999 {
                pack_color(&c)
            } else {
                let old = unpack_color(*pixel);
                pack_color(&(old + (c - old) * alpha))
            };
        }
    });
}

// ------------------------------------------------------------ utilidades

fn rgb(r: f32, g: f32, b: f32) -> Vec3 {
    Vec3::new(r, g, b)
}

fn mix(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

/// Ruido 2D (un corte del Perlin 3D).
fn n2(x: f32, z: f32, seed: u32) -> f32 {
    perlin(Vec3::new(x, 0.43, z), seed)
}

fn f2(x: f32, z: f32, octaves: u32, seed: u32) -> f32 {
    fbm(Vec3::new(x, 0.43, z), octaves, seed)
}

fn n1(x: f32, seed: u32) -> f32 {
    perlin(Vec3::new(x, 0.43, 0.71), seed)
}

/// Azimut (0 = al frente) y elevacion de una direccion.
fn angles(d: &Vec3) -> (f32, f32) {
    (d.x.atan2(-d.z), d.y.clamp(-1.0, 1.0).asin())
}

/// Punto donde el rayo toca el plano y = `plane` (y su distancia).
fn hit_plane(p: &Px, plane: f32) -> Option<(Vec3, f32)> {
    let dy = plane - p.eye.y;
    if p.dir.y.abs() < 1e-4 || dy * p.dir.y <= 0.0 {
        return None;
    }
    let dist = (dy / p.dir.y).min(6000.0);
    Some((p.eye + p.dir * dist, dist))
}

/// Capa de particulas en pantalla (nieve, chispas, arena...). La pantalla se
/// divide en celdas; cada celda puede tener una particula en una posicion al
/// azar. Moviendo las coordenadas con el tiempo las particulas viajan.
struct Particles {
    cell: f32,
    vx: f32,
    vy: f32,
    /// radio en pixeles
    size: f32,
    /// alargamiento horizontal (estelas de arena)
    stretch: f32,
    density: f32,
    seed: u32,
}

impl Particles {
    fn sample(&self, x: f32, y: f32, t: f32) -> f32 {
        let px = (x - self.vx * t) / self.cell;
        let py = (y - self.vy * t) / self.cell;
        let (cx, cy) = (px.floor(), py.floor());
        let h = hash3(cx as i32, cy as i32, 0, self.seed);
        if unit(h) > self.density {
            return 0.0;
        }
        let ox = 0.2 + 0.6 * unit(mix_bits(h));
        let oy = 0.2 + 0.6 * unit(mix_bits(h ^ 0x51ed));
        let dx = (px - cx - ox) * self.cell / self.stretch;
        let dy = (py - cy - oy) * self.cell;
        let r = self.size * (0.6 + 0.8 * unit(mix_bits(h ^ 0x2c1b)));
        let k = 1.0 - (dx * dx + dy * dy) / (r * r);
        if k <= 0.0 { 0.0 } else { k * k }
    }
}

/// Destello que decae despues de `start` (con parpadeo).
fn flash(t: f32, start: f32, decay: f32) -> f32 {
    if t < start {
        return 0.0;
    }
    let k = t - start;
    (-k * decay).exp() * (0.65 + 0.35 * (k * 70.0).sin())
}

// ------------------------------------------------------------ SOL

/// Superficie de una estrella. `kind`: 0 sol, 1 azul, 2 verde, 3 gigante roja.
/// Paleta = (lineas oscuras entre granulos, granulo, limbo/corona, protuberancias).
fn star(p: &Px, kind: usize) -> Vec3 {
    let (lane, cell_col, limb, prom) = match kind {
        1 => (rgb(0.08, 0.22, 0.8), rgb(0.82, 0.93, 1.0), rgb(0.45, 0.7, 1.0), rgb(0.5, 0.75, 1.0)),
        2 => (rgb(0.04, 0.42, 0.14), rgb(0.78, 1.0, 0.62), rgb(0.4, 0.95, 0.5), rgb(0.5, 1.0, 0.55)),
        3 => (rgb(0.3, 0.02, 0.01), rgb(1.0, 0.42, 0.16), rgb(0.85, 0.2, 0.08), rgb(1.0, 0.25, 0.1)),
        _ => (rgb(0.7, 0.18, 0.02), rgb(1.0, 0.8, 0.35), rgb(1.0, 0.62, 0.22), rgb(1.0, 0.38, 0.08)),
    };
    // La gigante roja tiene granulos mas grandes y lentos
    let scale = if kind == 3 { 0.06 } else { 0.12 };
    let t = p.t;
    let d = p.dir;
    let mut c = if let Some((g, dist)) = hit_plane(p, 0.0) {
        // Granulacion: celdas de conveccion (Voronoi animado en el tiempo)
        let (f1, f2v, id) = voronoi(Vec3::new(g.x * scale, t * 0.35, g.z * scale), 131);
        let gran = smoothstep(0.0, 0.32, f2v - f1);
        let cell = 0.8 + 0.25 * unit(id);
        let sup = f2(g.x * 0.01 + t * 0.02, g.z * 0.01, 3, 132);
        let mut s = mix(lane, cell_col * cell, gran) * (0.85 + 0.35 * sup);
        s = mix(mix(lane, cell_col, 0.6), s, (-dist * 0.005).exp());
        // Manchas
        let spot = smoothstep(0.32, 0.48, f2(g.x * 0.004 + 3.0, g.z * 0.004, 3, 133));
        s *= 1.0 - 0.75 * spot;
        mix(s, limb, 1.0 - (-dist * 0.0025).exp())
    } else {
        let (a, e) = angles(&d);
        // Corona: brillo que cae con la altura, con serpentinas
        let mut sky = mix(limb, limb * 0.05, smoothstep(0.0, 0.4, e));
        let streak = 0.5 + 0.5 * (a * 14.0 + n1(a * 3.0 + t * 0.05, 134) * 3.0).sin();
        sky += limb * streak * 0.3 * (-e * 5.0).exp();
        sky
    };

    // Protuberancias: arcos de plasma que salen del horizonte
    let (a, e) = angles(&d);
    if e > -0.02 {
        let erupt = smoothstep(5.0, 8.5, t);
        let loops = [
            (-0.38, 0.13, 0.17 + 0.01 * (t * 0.8).sin(), 0.0),
            (0.32, 0.09, 0.12 + 0.35 * erupt * erupt, erupt),
            (0.05, 0.05, 0.07, 0.0),
        ];
        for (k, &(ac, ra, re, fade)) in loops.iter().enumerate() {
            let x = (a - ac) / ra;
            let y = e / re;
            if y < -0.05 || x.abs() > 1.6 {
                continue;
            }
            let r = (x * x + y * y).sqrt();
            let dd = (r - 1.0).abs() * re;
            let thick = 0.007 + 0.004 * n1(x * 4.0 + k as f32, 135).abs();
            let glow = (-(dd / thick).powi(2)).exp() + 0.3 * (-(dd / (thick * 4.0)).powi(2)).exp();
            let fil = 0.55 + 0.45 * n1(y.atan2(x) * 9.0 - t * 1.6 + k as f32 * 5.0, 136);
            c += prom * glow * fil * 1.7 * (1.0 - fade * 0.7);
        }
    }

    // Chispas que suben y destello de la eyeccion de masa coronal
    let sparks = Particles { cell: 16.0, vx: 8.0, vy: -45.0, size: 1.2, stretch: 1.0, density: 0.25, seed: 137 };
    c += mix(cell_col, limb, 0.4) * sparks.sample(p.x, p.y, t);
    c += mix(cell_col, Vec3::repeat(1.0), 0.5) * flash(t, 6.2, 2.2) * 0.55;
    c
}

// ------------------------------------------------------------ MAGMA

/// Rayos dentro de la nube de ceniza: (tiempo, azimut)
const MAGMA_BOLTS: [(f32, f32); 6] = [(1.6, 0.35), (3.4, -0.42), (5.2, 0.12), (6.9, -0.18), (8.4, 0.48), (9.6, -0.05)];

fn magma(p: &Px) -> Vec3 {
    let t = p.t;
    let d = p.dir;
    let glow = rgb(0.6, 0.15, 0.03);
    let (a, e) = angles(&d);

    if let Some((g, dist)) = hit_plane(p, 0.0) {
        // Roca basaltica con rios de lava (donde el ruido cruza cero)
        let rock = f2(g.x * 0.08, g.z * 0.08, 3, 11) * 0.5 + 0.5;
        let mut c = rgb(0.075, 0.045, 0.04) * (0.5 + 0.9 * rock);
        let flow = n2(g.x * 0.012 + t * 0.01, g.z * 0.012, 12) + 0.35 * n2(g.x * 0.05, g.z * 0.05 - t * 0.08, 13);
        let width = 0.05 + dist * 0.0002;
        let river = 1.0 - smoothstep(0.0, width, flow.abs());
        let pulse = 0.8 + 0.2 * (t * 2.0 + g.x * 0.1 + g.z * 0.05).sin();
        let lava = rgb(1.0, 0.33, 0.04) * pulse + rgb(0.7, 0.55, 0.1) * river * river;
        c = mix(c, lava, river);
        c += rgb(0.35, 0.07, 0.0) * (1.0 - smoothstep(0.0, width * 5.0, flow.abs())) * 0.5;
        // Brillo de los rayos sobre el suelo
        for &(tb, _) in &MAGMA_BOLTS {
            c += rgb(0.25, 0.25, 0.35) * flash(t, tb, 9.0) * 0.4;
        }
        return mix(c, glow, 1.0 - (-dist * 0.006).exp());
    }

    // Volcan (cono con crater) y cordillera en el horizonte
    let volcano = (0.23 - 0.55 * a.abs()).min(0.195 + 0.6 * a.abs());
    let range = 0.03 + 0.05 * (n1(a * 5.0, 14) * 0.5 + 0.5) + 0.015 * n1(a * 23.0, 15);
    let profile = volcano.max(range);
    if e < profile {
        let mut c = mix(glow * 0.55, rgb(0.05, 0.03, 0.03), smoothstep(0.0, 0.06, e));
        if volcano >= range {
            // Coladas de lava que bajan por las laderas
            let s = perlin(Vec3::new(a * 40.0, e * 3.0, 0.3), 16).abs();
            let stream = 1.0 - smoothstep(0.0, 0.07, s);
            let beat = 0.6 + 0.4 * (e * 90.0 + t * 5.0).sin();
            c += rgb(1.0, 0.35, 0.05) * stream * beat * (0.3 + 0.7 * e / 0.21);
        }
        c += rgb(1.0, 0.5, 0.1) * (-(a / 0.05).powi(2) - ((e - 0.2) / 0.03).powi(2)).exp() * 1.5;
        return c;
    }

    // Cielo de ceniza iluminado desde abajo por la lava
    let mut c = mix(glow, rgb(0.04, 0.015, 0.015), smoothstep(0.0, 0.5, e));
    let mut density = 0.0;
    if let Some((cp, cdist)) = hit_plane(p, 90.0) {
        density = smoothstep(-0.05, 0.45, f2(cp.x * 0.008 + t * 0.02, cp.z * 0.008, 4, 31)) * (-cdist * 0.0006).exp();
        let lit = rgb(0.11, 0.06, 0.06) + glow * 0.4 * (1.0 - smoothstep(0.0, 0.4, e));
        c = mix(c, lit, density);
    }

    // Columna de ceniza que sale del crater, empujada por el viento
    if e > 0.18 {
        let w = 0.03 + (e - 0.19) * 0.5;
        let q = (a - (e - 0.19) * 0.35) / w;
        if q.abs() < 1.5 {
            let n = fbm(Vec3::new(a * 10.0, e * 6.0 - t * 0.5, t * 0.1), 4, 17) * 0.5 + 0.6;
            let plume = (1.0 - q * q / 2.25) * n.clamp(0.0, 1.0);
            let col = rgb(0.1, 0.065, 0.06) + rgb(0.9, 0.3, 0.05) * (-(e - 0.2) * 9.0).exp();
            c = mix(c, col, plume);
        }
    }

    // Relampagos volcanicos: iluminan la nube y dibujan un rayo quebrado
    for (k, &(tb, ab)) in MAGMA_BOLTS.iter().enumerate() {
        let f = flash(t, tb, 7.0);
        if f < 0.01 {
            continue;
        }
        c += rgb(0.6, 0.6, 0.9) * f * (-((a - ab) / 0.35).powi(2)).exp() * (0.3 + density);
        if e > 0.04 && e < 0.42 && f > 0.25 {
            let path = ab + 0.035 * n1(e * 25.0 + k as f32 * 7.0, 18) + 0.012 * n1(e * 90.0 + k as f32, 19);
            let bolt = (-(a - path).abs() * 500.0).exp();
            c += rgb(0.85, 0.85, 1.0) * bolt * f * 2.0;
        }
    }

    // Bombas de lava lanzadas por el crater
    if a.abs() < 0.4 && e > 0.12 && e < 0.5 {
        for i in 0..28u32 {
            let h = hash3(i as i32, 0, 0, 20);
            let period = 1.6 + 1.2 * unit(h);
            let tau = (t + unit(mix_bits(h)) * period) % period;
            let pa = (unit(mix_bits(h ^ 0x11)) - 0.5) * 0.16 * tau;
            let pe = 0.2 + (0.22 + 0.15 * unit(mix_bits(h ^ 0x22))) * tau - 0.3 * tau * tau;
            if pe < 0.16 {
                continue;
            }
            let dd = ((a - pa).powi(2) + (e - pe).powi(2)) / (0.006 * 0.006);
            c += rgb(1.0, 0.55, 0.15) * (-dd).exp() * 2.0;
        }
    }

    // Brasas que suben
    let embers = Particles { cell: 20.0, vx: 6.0, vy: -22.0, size: 1.1, stretch: 1.0, density: 0.3, seed: 21 };
    c + rgb(1.0, 0.45, 0.1) * embers.sample(p.x + (t + p.y * 0.05).sin() * 3.0, p.y, t)
}

// ------------------------------------------------------------ VERDALIA

fn verdalia_sky(d: &Vec3, sun: &Vec3) -> Vec3 {
    let horizon = rgb(0.72, 0.84, 0.95);
    let zenith = rgb(0.2, 0.43, 0.85);
    let mut c = mix(horizon, zenith, smoothstep(0.0, 0.6, d.y));
    let s = d.dot(sun).max(0.0);
    c += rgb(1.0, 0.9, 0.7) * (s.powi(400) * 3.0 + s.powi(12) * 0.25);
    c
}

fn verdalia(p: &Px) -> Vec3 {
    let t = p.t;
    let d = p.dir;
    let sun = Vec3::new(0.35, 0.22, -1.0).normalize();
    let horizon = rgb(0.72, 0.84, 0.95);
    let cloud_h = 160.0;
    let cloud = |x: f32, z: f32| smoothstep(0.05, 0.35, f2(x * 0.004 + t * 0.01, z * 0.004, 5, 41));

    let mut c = if let Some((g, dist)) = hit_plane(p, 0.0) {
        let island = f2(g.x * 0.003, g.z * 0.003 + 0.7, 5, 51);
        let mut col = if island > 0.12 {
            let h = island - 0.12;
            let forest = f2(g.x * 0.03, g.z * 0.03, 3, 52) * 0.5 + 0.5;
            let mut land = mix(rgb(0.2, 0.48, 0.16), rgb(0.07, 0.28, 0.1), forest);
            land = mix(rgb(0.86, 0.8, 0.6), land, smoothstep(0.0, 0.02, h));
            land = mix(land, rgb(0.42, 0.38, 0.34), smoothstep(0.12, 0.18, h));
            land * 0.9
        } else {
            // Olas: la normal se perturba con ruido que se mueve
            let wx = n2(g.x * 0.15 + t * 0.6, g.z * 0.15, 61) * 0.07 + n2(g.x * 0.5, g.z * 0.5 - t * 1.2, 62) * 0.03;
            let wz = n2(g.x * 0.15, g.z * 0.15 + t * 0.5, 63) * 0.07 + n2(g.x * 0.5 - t, g.z * 0.5, 64) * 0.03;
            let n = Vec3::new(wx, 1.0, wz).normalize();
            let r = d - n * 2.0 * d.dot(&n);
            let fres = 0.03 + 0.97 * (1.0 - (-d.dot(&n)).max(0.0)).powi(5);
            let water = mix(rgb(0.02, 0.12, 0.26), rgb(0.1, 0.45, 0.5), smoothstep(-0.05, 0.12, island));
            let refl = verdalia_sky(&Vec3::new(r.x, r.y.abs(), r.z), &sun) + rgb(1.0, 0.95, 0.8) * r.dot(&sun).max(0.0).powi(300) * 4.0;
            let mut col = mix(water, refl, fres);
            let foam = smoothstep(0.09, 0.12, island) * (0.5 + 0.5 * (island * 300.0 - t * 3.0).sin());
            col = mix(col, rgb(0.95, 0.97, 1.0), foam * 0.7);
            col
        };
        // Sombra de las nubes
        let s = g + sun * ((cloud_h - g.y) / sun.y);
        col *= 1.0 - 0.45 * cloud(s.x, s.z);
        mix(col, horizon, 1.0 - (-dist * 0.0012).exp())
    } else {
        let mut sky = verdalia_sky(&d, &sun);
        // La luna de Verdalia, palida de dia
        let moon = Vec3::new(-0.5, 0.3, -1.0).normalize();
        if d.dot(&moon) > (0.035f32).cos() {
            let k = fbm(d * 60.0, 3, 65) * 0.5 + 0.5;
            sky = mix(sky, rgb(0.85, 0.86, 0.9) * (0.7 + 0.3 * k), 0.55);
        }
        if let Some((cp, cdist)) = hit_plane(p, cloud_h) {
            let den = f2(cp.x * 0.004 + t * 0.01, cp.z * 0.004, 5, 41);
            let cov = smoothstep(0.05, 0.35, den) * (-cdist * 0.00035).exp();
            let shade = 0.78 + 0.22 * smoothstep(0.1, 0.5, den);
            sky = mix(sky, rgb(1.0, 1.0, 1.0) * shade, cov);
        }
        sky
    };

    // Atravesamos un banco de nubes a mitad del vuelo
    let bank = smoothstep(5.8, 6.8, t) * (1.0 - smoothstep(7.2, 8.4, t));
    if bank > 0.0 {
        let wisps = fbm(Vec3::new(p.nx * 2.0, p.ny * 2.0 + t * 1.5, t * 0.6), 3, 66) * 0.5 + 0.5;
        c = mix(c, rgb(0.93, 0.95, 0.98), bank * (0.55 + 0.45 * wisps));
    }
    c
}

// ------------------------------------------------------------ LUNA

/// Sombreado de un crater (multiplicador de brillo) en celdas de `cell`.
fn craters(x: f32, z: f32, cell: f32, seed: u32, sun: (f32, f32)) -> f32 {
    let (cx, cz) = ((x / cell).floor() as i32, (z / cell).floor() as i32);
    let mut shade = 1.0;
    for dz in -1..=1 {
        for dx in -1..=1 {
            let h = hash3(cx + dx, cz + dz, 0, seed);
            if unit(h) > 0.7 {
                continue;
            }
            let ox = (cx + dx) as f32 + unit(mix_bits(h ^ 1));
            let oz = (cz + dz) as f32 + unit(mix_bits(h ^ 2));
            let r = (0.15 + 0.3 * unit(mix_bits(h ^ 3))) * cell;
            let (vx, vz) = (x - ox * cell, z - oz * cell);
            let dist = (vx * vx + vz * vz).sqrt();
            let q = dist / r;
            if q > 1.3 {
                continue;
            }
            let side = if dist > 1e-4 { (vx * sun.0 + vz * sun.1) / dist } else { 0.0 };
            if q < 1.0 {
                // Fondo del crater: la pared del lado del sol queda en sombra
                shade *= 0.8 - 0.35 * side * q * q;
            }
            shade *= 1.0 + 0.35 * (-((q - 1.0) * 6.0).powi(2)).exp() * (0.4 + 0.6 * side);
        }
    }
    shade
}

fn luna(p: &Px) -> Vec3 {
    let t = p.t;
    let d = p.dir;
    let sun = Vec3::new(0.8, 0.35, -0.5).normalize();
    let sun_xz = {
        let l = (sun.x * sun.x + sun.z * sun.z).sqrt();
        (sun.x / l, sun.z / l)
    };
    let (a, e) = angles(&d);

    let mut c = if let Some((g, dist)) = hit_plane(p, 0.0) {
        let detail = (-dist * 0.02).exp();
        let base = 0.4 + 0.1 * f2(g.x * 0.05, g.z * 0.05, 4, 81) + 0.06 * n2(g.x * 0.6, g.z * 0.6, 82) * detail;
        let shade = craters(g.x, g.z, 9.0, 83, sun_xz) * craters(g.x, g.z, 34.0, 84, sun_xz);
        let mut col = rgb(1.0, 0.98, 0.95) * base * shade * (0.75 + 0.25 * (-dist * 0.003).exp());
        // Polvo del impacto del meteorito
        let k = flash(t, 5.3, 1.5);
        col += rgb(1.0, 0.85, 0.6) * k * (-((a + 0.15) / 0.08).powi(2)).exp() * (-(e / 0.03).powi(2)).exp() * 2.0;
        col
    } else {
        let mut sky = SkyBox::stars(&d, t) * 1.2;
        // El sol sin atmosfera: un disco blanco sin halo azul
        let s = d.dot(&sun);
        if s > (0.012f32).cos() {
            sky = rgb(1.0, 1.0, 0.97) * 3.0;
        }
        sky += rgb(1.0, 0.95, 0.9) * s.max(0.0).powi(2000) * 1.5;

        // Verdalia sale lentamente por el horizonte
        let rise = -0.03 + 0.2 * smoothstep(0.0, p.dur, t);
        let v = Vec3::new(0.15, rise, -1.0).normalize();
        let radius = 0.13;
        let vr = v.cross(&Vec3::y()).normalize();
        let vu = vr.cross(&v);
        let cosv = d.dot(&v);
        if cosv > 0.0 {
            let off = d - v * cosv;
            let (u, w) = (off.dot(&vr) / radius, off.dot(&vu) / radius);
            let r2 = u * u + w * w;
            if r2 < 1.0 {
                let z = (1.0 - r2).sqrt();
                let n = vr * u + vu * w - v * z;
                let lambert = n.dot(&sun).max(0.0);
                let (sa, ca) = (t * 0.05).sin_cos();
                let q = Vec3::new(n.x * ca - n.z * sa, n.y, n.x * sa + n.z * ca);
                let land = smoothstep(0.02, 0.08, fbm(q * 2.2, 4, 71));
                let mut surface = mix(rgb(0.05, 0.2, 0.5), rgb(0.2, 0.45, 0.18), land);
                let clouds = smoothstep(0.1, 0.5, fbm(q * 3.0 + Vec3::new(t * 0.02, 0.0, 0.0), 4, 72));
                surface = mix(surface, rgb(0.95, 0.96, 1.0), clouds);
                sky = surface * (lambert * 1.15 + 0.02) + rgb(0.3, 0.55, 1.0) * (1.0 - z).powi(3) * (0.2 + lambert);
            } else if r2 < 1.12 {
                sky += rgb(0.3, 0.55, 1.0) * (1.0 - (r2 - 1.0) / 0.12).powi(2) * 0.35;
            }
        }

        // Meteorito cruzando el cielo hasta el horizonte
        let m = smoothstep(4.6, 5.3, t);
        if t > 4.6 && t < 5.4 {
            let head = (0.55 - 0.7 * m, 0.35 * (1.0 - m));
            let tail = (head.0 + 0.12, head.1 + 0.06);
            let (vx, vy) = (tail.0 - head.0, tail.1 - head.1);
            let s = (((a - head.0) * vx + (e - head.1) * vy) / (vx * vx + vy * vy)).clamp(0.0, 1.0);
            let dist = ((a - head.0 - vx * s).powi(2) + (e - head.1 - vy * s).powi(2)).sqrt();
            sky += rgb(1.0, 0.9, 0.7) * (-dist * 900.0).exp() * (1.0 - s) * 2.5;
        }
        sky
    };

    // Destello del impacto
    c += rgb(1.0, 0.9, 0.75) * flash(t, 5.3, 4.0) * 0.25;
    c
}

// ------------------------------------------------------------ DUNA

fn duna(p: &Px) -> Vec3 {
    let t = p.t;
    let d = p.dir;
    let sun = Vec3::new(-0.4, 0.3, -1.0).normalize();
    let storm = smoothstep(3.0, 8.5, t);
    let haze = mix(rgb(0.8, 0.52, 0.3), rgb(0.62, 0.38, 0.22), storm);

    let mut c = if let Some((g, dist)) = hit_plane(p, 0.0) {
        let ph = g.x * 0.035 + g.z * 0.012 + f2(g.x * 0.004, g.z * 0.004, 3, 92) * 6.0;
        let shade = 0.38 + 0.62 * (0.5 + 0.5 * ph.sin()).powf(0.8);
        let ripples = (g.z * 1.3 + g.x * 0.4 + n2(g.x * 0.1, g.z * 0.1, 93) * 3.0).sin() * 0.07 * (-dist * 0.03).exp();
        let mut col = rgb(0.9, 0.58, 0.32) * (shade + ripples);
        // Arena que corre sobre el suelo
        let drift = n2(g.x * 0.15 - t * 4.0, g.z * 1.0, 94).max(0.0);
        col += rgb(0.3, 0.2, 0.1) * drift * (0.3 + storm) * (-dist * 0.02).exp();
        mix(col, haze, 1.0 - (-dist * (0.003 + 0.03 * storm)).exp())
    } else {
        let mut sky = mix(haze, rgb(0.5, 0.35, 0.3) * (1.0 - 0.35 * storm), smoothstep(0.0, 0.5, d.y));
        let s = d.dot(&sun).max(0.0);
        sky += rgb(1.0, 0.85, 0.6) * (s.powi(600) * 2.0 * (1.0 - 0.8 * storm) + s.powi(8) * 0.25);
        if let Some((cp, cdist)) = hit_plane(p, 45.0) {
            let den = smoothstep(0.0, 0.5, f2(cp.x * 0.01 - t * 0.3, cp.z * 0.01, 4, 91)) * storm;
            sky = mix(sky, haze * 0.85, den * (-cdist * 0.002).exp());
        }
        sky
    };

    // Polvo suspendido en el aire que avanza con el viento
    let dust = fbm(Vec3::new(p.nx * 2.5 + t * 1.6, p.ny * 3.5, t * 0.25), 3, 95) * 0.5 + 0.5;
    c = mix(c, haze * 1.05, dust * storm * 0.55 + 0.08 * storm);

    // Estelas de arena
    let near = Particles { cell: 22.0, vx: 420.0, vy: 25.0, size: 1.1, stretch: 9.0, density: 0.35, seed: 96 };
    let far = Particles { cell: 14.0, vx: 260.0, vy: 12.0, size: 0.8, stretch: 6.0, density: 0.3, seed: 97 };
    let sand = near.sample(p.x, p.y, t) + far.sample(p.x, p.y, t) * 0.6;
    c + rgb(1.0, 0.8, 0.55) * sand * (0.12 + 0.5 * storm)
}

// ------------------------------------------------------------ BANDUS

/// Relampagos dentro de la capa profunda: cada celda destella a su ritmo.
fn bandus_lightning(x: f32, z: f32, t: f32) -> f32 {
    let cell = 80.0;
    let (cx, cz) = ((x / cell).floor() as i32, (z / cell).floor() as i32);
    let mut total = 0.0;
    for dz in -1..=1 {
        for dx in -1..=1 {
            let h = hash3(cx + dx, cz + dz, 0, 141);
            let period = 2.5 + 4.0 * unit(h);
            let tau = (t + unit(mix_bits(h ^ 5)) * period) % period;
            if tau > 0.5 {
                continue;
            }
            let fx = ((cx + dx) as f32 + unit(mix_bits(h ^ 6))) * cell;
            let fz = ((cz + dz) as f32 + unit(mix_bits(h ^ 7))) * cell;
            let d2 = ((x - fx).powi(2) + (z - fz).powi(2)) / (35.0 * 35.0);
            total += (-tau * 9.0).exp() * (0.6 + 0.4 * (tau * 80.0).sin()) * (-d2).exp();
        }
    }
    total
}

fn bandus(p: &Px) -> Vec3 {
    let t = p.t;
    let d = p.dir;
    let haze = rgb(0.86, 0.74, 0.58);
    let sun = Vec3::new(0.5, 0.35, -1.0).normalize();

    let mut c = if let Some((g, dist)) = hit_plane(p, -45.0) {
        // Gran tormenta: los puntos cercanos al vortice se giran
        let (vcx, vcz) = (90.0, -1100.0);
        let (rx, rz) = (g.x - vcx, g.z - vcz);
        let r = (rx * rx + rz * rz).sqrt();
        let ang = 2.8 * (-r / 140.0).exp() + t * 0.5 * (-r / 170.0).exp();
        let (sa, ca) = ang.sin_cos();
        let (wx, wz) = (vcx + rx * ca - rz * sa, vcz + rx * sa + rz * ca);

        // Bandas que corren en sentidos opuestos
        let band = wz * 0.022;
        let flow = (band * 1.3).sin() * 18.0;
        let turb = f2((wx + flow * t) * 0.018, wz * 0.04, 5, 101);
        let v = (band + turb * 1.2).sin() * 0.5 + 0.5;
        let mut col = if v < 0.5 {
            mix(rgb(0.5, 0.33, 0.22), rgb(0.76, 0.56, 0.36), v * 2.0)
        } else {
            mix(rgb(0.76, 0.56, 0.36), rgb(0.94, 0.85, 0.66), v * 2.0 - 1.0)
        };
        // Relieve de las cimas de las nubes
        let billow = f2((g.x + flow * t) * 0.07, g.z * 0.07, 3, 105) * (-dist * 0.004).exp();
        col *= 0.82 + 0.3 * turb + 0.28 * billow;
        col = mix(col, rgb(0.78, 0.36, 0.25), (-r / 110.0).exp() * 0.8);

        // Huecos que dejan ver la capa profunda con relampagos
        let hole = smoothstep(0.25, 0.45, f2(g.x * 0.006 + t * 0.05, g.z * 0.006, 3, 102));
        if hole > 0.0 {
            let mut deep = rgb(0.16, 0.15, 0.2);
            if let Some((g2, _)) = hit_plane(p, -130.0) {
                deep *= 0.6 + 0.4 * (f2(g2.x * 0.01, g2.z * 0.01, 2, 103) * 0.5 + 0.5);
                deep += rgb(0.8, 0.85, 1.0) * bandus_lightning(g2.x, g2.z, t) * 2.0;
            }
            col = mix(col, deep, hole * 0.85);
        }
        col += rgb(0.5, 0.5, 0.6) * bandus_lightning(g.x, g.z, t) * 0.3;
        mix(col, haze, 1.0 - (-dist * 0.0012).exp())
    } else {
        let (a, e) = angles(&d);
        let mut sky = mix(haze, rgb(0.33, 0.38, 0.55), smoothstep(0.0, 0.6, d.y));
        let s = d.dot(&sun).max(0.0);
        sky += rgb(1.0, 0.9, 0.75) * (s.powi(500) * 2.5 + s.powi(10) * 0.2);
        // El anillo de Bandus cruzando el cielo
        let er = 0.3 + 0.12 * (a * 0.8).cos();
        let dr = e - er;
        if dr.abs() < 0.07 {
            let stripes = 0.55 + 0.45 * (dr * 260.0).sin() * (dr * 90.0).cos();
            let edge = 1.0 - smoothstep(0.05, 0.07, dr.abs());
            sky = mix(sky, rgb(0.92, 0.86, 0.75) * stripes, edge * 0.45);
        }
        sky
    };

    // Jirones de nube que pasan muy cerca (dan sensacion de velocidad)
    if let Some((w, dist)) = hit_plane(p, -9.0) {
        let wisp = smoothstep(0.3, 0.6, f2(w.x * 0.03, w.z * 0.015 + t * 0.2, 3, 104));
        c = mix(c, rgb(0.95, 0.9, 0.8), wisp * 0.5 * (-dist * 0.004).exp());
    }
    c
}

// ------------------------------------------------------------ GLACIA

/// Aurora: cortinas sobre varias capas de altura (verde abajo, violeta arriba).
fn aurora(eye: &Vec3, d: &Vec3, t: f32, layers: usize) -> Vec3 {
    if d.y < 0.01 {
        return Vec3::zeros();
    }
    let mut acc = Vec3::zeros();
    let gain = 5.0 / layers as f32;
    for k in 0..layers {
        let h = 140.0 + k as f32 * 112.0 / (layers - 1) as f32;
        let dist = (h - eye.y) / d.y;
        if dist > 5000.0 {
            continue;
        }
        let ap = eye + d * dist;
        let (rx, rz) = (ap.x, ap.z - eye.z);
        let rays = 0.45 + 0.55 * n2(rx * 0.05 + t * 0.6, t * 0.2, 112).abs() * 1.6;
        let kf = k as f32 / (layers - 1) as f32;
        let color = mix(rgb(0.2, 1.0, 0.5), rgb(0.6, 0.3, 1.0), kf);
        for (offset, weight) in [(420.0, 1.0), (760.0, 0.6)] {
            let wave = 70.0 * (rx * 0.006 + t * 0.35 + offset).sin() + 40.0 * (rx * 0.0023 - t * 0.21 + offset * 0.3).sin();
            let dz = rz + offset - wave;
            let band = (-(dz / 45.0).powi(2)).exp();
            acc += color * band * rays * (1.0 - kf * 0.75) * weight * 0.32 * gain;
        }
    }
    acc
}

fn glacia(p: &Px) -> Vec3 {
    let t = p.t;
    let d = p.dir;
    let horizon = rgb(0.05, 0.09, 0.16);

    let mut c = if let Some((g, dist)) = hit_plane(p, 0.0) {
        let detail = (-dist * 0.02).exp();
        let base = mix(rgb(0.5, 0.65, 0.82), rgb(0.72, 0.83, 0.95), f2(g.x * 0.02, g.z * 0.06, 3, 121) * 0.5 + 0.5) * 0.4;
        let sastrugi = n2(g.x * 0.05, g.z * 0.6, 122) * 0.08 * detail;
        let crack = (1.0 - smoothstep(0.0, 0.03, n2(g.x * 0.03, g.z * 0.03, 123).abs())) * (-dist * 0.01).exp();
        let mut col = base * (1.0 + sastrugi) * (1.0 - 0.6 * crack);
        // El hielo pulido refleja la aurora
        col += aurora(&Vec3::new(p.eye.x, -p.eye.y, p.eye.z), &Vec3::new(d.x, -d.y, d.z), t, 3) * 0.35;
        // Destellos de cristales de hielo
        let h = hash3((g.x * 5.0).floor() as i32, (g.z * 5.0).floor() as i32, 0, 124);
        if unit(h) > 0.992 {
            col += Vec3::repeat(((t * 6.0 + unit(mix_bits(h)) * 10.0).sin() * 0.5 + 0.5).powi(8) * detail * 1.2);
        }
        mix(col, horizon, 1.0 - (-dist * 0.003).exp())
    } else {
        let sky = mix(horizon, rgb(0.005, 0.01, 0.03), smoothstep(0.0, 0.5, d.y));
        sky + SkyBox::stars(&d, t) * 0.9 + aurora(&p.eye, &d, t, 8)
    };

    // Nevada en tres capas (las cercanas caen mas rapido y son mas grandes)
    let sway = (t * 0.9 + p.y * 0.02).sin() * 6.0;
    let layers = [
        Particles { cell: 34.0, vx: -25.0, vy: 60.0, size: 1.8, stretch: 1.0, density: 0.35, seed: 125 },
        Particles { cell: 22.0, vx: -18.0, vy: 38.0, size: 1.2, stretch: 1.0, density: 0.35, seed: 126 },
        Particles { cell: 14.0, vx: -10.0, vy: 22.0, size: 0.8, stretch: 1.0, density: 0.3, seed: 127 },
    ];
    for (k, layer) in layers.iter().enumerate() {
        c += Vec3::repeat(0.8 - k as f32 * 0.2) * layer.sample(p.x + sway, p.y, t);
    }
    c
}

// ============================================================ utilidades (galaxias 2 a 4)

/// Disco de un astro en el cielo, centrado en la direccion `v` (unitaria)
/// con radio angular `radius`. Devuelve (u, w, r^2) dentro del disco, con
/// u hacia la derecha y w hacia arriba en -1..1.
fn sky_disc(d: &Vec3, v: &Vec3, radius: f32) -> Option<(f32, f32, f32)> {
    let cosv = d.dot(v);
    if cosv <= 0.0 {
        return None;
    }
    let vr = v.cross(&Vec3::y()).normalize();
    let vu = vr.cross(v);
    let off = d - v * cosv;
    let (u, w) = (off.dot(&vr) / radius, off.dot(&vu) / radius);
    let r2 = u * u + w * w;
    (r2 < 1.0).then_some((u, w, r2))
}

/// Colores de pelicula delgada (arcoiris) segun la fase `x`.
fn iri(x: f32) -> Vec3 {
    let tau = std::f32::consts::TAU;
    Vec3::new(0.5 + 0.5 * (tau * x).cos(), 0.5 + 0.5 * (tau * (x + 0.33)).cos(), 0.5 + 0.5 * (tau * (x + 0.67)).cos())
}

fn reflect(d: &Vec3, n: &Vec3) -> Vec3 {
    d - n * 2.0 * d.dot(n)
}

/// Picos en el horizonte (cristales, agujas, algas): el azimut se divide en
/// celdas y cada celda puede tener un pico triangular. Devuelve (lado -1..1,
/// distancia vertical a la punta) si el pixel cae dentro de un pico.
fn spike(a: f32, e: f32, density: f32, width: f32, height: f32, seed: u32) -> Option<(f32, f32)> {
    let c0 = (a * density).floor() as i32;
    for i in c0 - 1..=c0 + 1 {
        let h = hash3(i, 0, 0, seed);
        if unit(h) < 0.25 {
            continue;
        }
        let c = (i as f32 + 0.2 + 0.6 * unit(mix_bits(h))) / density;
        let w = width * (0.6 + 0.8 * unit(mix_bits(h ^ 1)));
        let hg = height * (0.35 + 0.65 * unit(mix_bits(h ^ 2)));
        if e < hg * (1.0 - (a - c).abs() / w) {
            return Some(((a - c) / w, hg - e));
        }
    }
    None
}

// ------------------------------------------------------------ CUBIX

/// Ciudad de bloques: el rayo recorre la cuadricula celda por celda (DDA) y
/// en cada una compara su altura con la del bloque. Las alturas laten.
fn cubix(p: &Px) -> Vec3 {
    const CELL: f32 = 7.0;
    let t = p.t;
    let d = p.dir;
    let fog = rgb(0.2, 0.06, 0.3);
    let neon = |i: i32, j: i32| mix(rgb(0.15, 0.9, 1.0), rgb(1.0, 0.25, 0.85), (i as f32 * 0.15 + j as f32 * 0.1 + t * 0.3).sin() * 0.5 + 0.5);
    let height = |i: i32, j: i32| {
        let h = hash3(i, j, 0, 201);
        let b = unit(h);
        if b < 0.4 { 0.0 } else { (b - 0.4) * 20.0 * (0.6 + 0.4 * (t * 1.5 + unit(mix_bits(h)) * 6.28).sin()) }
    };

    if d.y < -0.01 {
        let inv = |v: f32| if v.abs() < 1e-6 { 1e30 } else { 1.0 / v.abs() };
        let (ix0, iz0) = ((p.eye.x / CELL).floor(), (p.eye.z / CELL).floor());
        let (mut ix, mut iz) = (ix0 as i32, iz0 as i32);
        let (sx, sz) = (if d.x > 0.0 { 1 } else { -1 }, if d.z > 0.0 { 1 } else { -1 });
        let (dx, dz) = (CELL * inv(d.x), CELL * inv(d.z));
        let mut tx = ((ix0 + (sx > 0) as i32 as f32) * CELL - p.eye.x).abs() * inv(d.x);
        let mut tz = ((iz0 + (sz > 0) as i32 as f32) * CELL - p.eye.z).abs() * inv(d.z);
        let mut s0 = 0.0f32;
        let mut axis = 0;
        for _ in 0..56 {
            let h = height(ix, iz);
            let s1 = tx.min(tz);
            let hit = if p.eye.y + d.y * s0 < h {
                Some((s0, false)) // pared
            } else if p.eye.y + d.y * s1 < h {
                Some(((h - p.eye.y) / d.y, true)) // techo (o el suelo si h = 0)
            } else {
                None
            };
            if let Some((s, top)) = hit {
                let q = p.eye + d * s;
                let glow = neon(ix, iz);
                let col = if top {
                    let (fx, fz) = ((q.x / CELL).rem_euclid(1.0), (q.z / CELL).rem_euclid(1.0));
                    let edge = fx.min(1.0 - fx).min(fz).min(1.0 - fz);
                    let line = 1.0 - smoothstep(0.03, 0.08 + s * 0.0006, edge);
                    let panel = mix(rgb(0.07, 0.08, 0.14), rgb(0.2, 0.22, 0.35), unit(hash3(ix, iz, 1, 202))) * (0.6 + h * 0.05);
                    mix(panel, glow, line * if h > 0.0 { 1.0 } else { 0.35 })
                } else {
                    // Ventanas de neon en las paredes y borde brillante arriba
                    let along = if axis == 0 { q.z } else { q.x } / CELL * 4.0;
                    let (wy, wx) = ((q.y * 0.7).rem_euclid(1.0), along.rem_euclid(1.0));
                    let on = unit(hash3(ix * 7 + iz * 13, (q.y * 0.7).floor() as i32, along.floor() as i32, 203)) > 0.6;
                    let window = if on && wy > 0.25 && wy < 0.75 && wx > 0.2 && wx < 0.8 { 0.9 } else { 0.0 };
                    let wall = rgb(0.09, 0.09, 0.17) * if axis == 0 { 0.8 } else { 0.5 };
                    wall + glow * (window + (1.0 - smoothstep(0.0, 0.5, h - q.y)) * 0.8)
                };
                return mix(col, fog, 1.0 - (-s * 0.006).exp());
            }
            if tx < tz {
                s0 = tx;
                tx += dx;
                ix += sx;
                axis = 0;
            } else {
                s0 = tz;
                tz += dz;
                iz += sz;
                axis = 1;
            }
        }
        return fog;
    }

    // Cielo con lluvia de datos
    let mut c = mix(fog, rgb(0.02, 0.01, 0.08), smoothstep(0.0, 0.5, d.y)) + SkyBox::stars(&d, t) * 0.5;
    let rain = Particles { cell: 24.0, vx: 150.0, vy: 0.0, size: 1.0, stretch: 10.0, density: 0.3, seed: 204 };
    c += rgb(0.15, 0.9, 1.0) * rain.sample(p.y, p.x, t) * 0.6;
    c
}

// ------------------------------------------------------------ ESPEJO

fn espejo_sky(d: &Vec3, t: f32) -> Vec3 {
    let n = fbm(Vec3::new(d.x * 2.0 + t * 0.03, d.y * 2.0, d.z * 2.0), 4, 211) * 0.5 + 0.5;
    let m = fbm(d * 4.0 + Vec3::new(0.0, t * 0.05, 0.0), 3, 212) * 0.5 + 0.5;
    let c = mix(rgb(0.05, 0.02, 0.12), rgb(0.55, 0.2, 0.75), smoothstep(0.35, 0.8, n)) + rgb(0.1, 0.3, 0.7) * smoothstep(0.5, 0.9, m) * 0.6;
    let s = d.dot(&Vec3::new(-0.3, 0.25, -1.0).normalize()).max(0.0);
    c + rgb(0.7, 0.85, 1.0) * (s.powi(300) * 3.0 + s.powi(10) * 0.3)
}

/// Rayo contra las gotas de metal. Devuelve (distancia, normal).
fn hit_spheres(o: &Vec3, d: &Vec3, spheres: &[(Vec3, f32)]) -> Option<(f32, Vec3)> {
    let mut best: Option<(f32, Vec3)> = None;
    for &(c, r) in spheres {
        let oc = o - c;
        let b = oc.dot(d);
        let disc = b * b - (oc.norm_squared() - r * r);
        if disc < 0.0 {
            continue;
        }
        let s = -b - disc.sqrt();
        if s > 0.01 && best.is_none_or(|(bs, _)| s < bs) {
            best = Some((s, (o + d * s - c) / r));
        }
    }
    best
}

/// Mar de metal liquido que refleja la nebulosa, con gotas de cromo flotando.
fn espejo(p: &Px) -> Vec3 {
    let t = p.t;
    let d = p.dir;
    let layout = [(-7.0, -25.0, 3.0), (6.0, -50.0, 2.5), (-5.0, -75.0, 3.5), (9.0, -100.0, 3.0), (-8.0, -125.0, 4.0)];
    let mut spheres = [(Vec3::zeros(), 0.0); 5];
    for (k, &(x, z, r)) in layout.iter().enumerate() {
        spheres[k] = (Vec3::new(x, 4.5 + 2.0 * (t * 0.8 + k as f32 * 1.7).sin(), z), r);
    }
    let up = |r: Vec3| Vec3::new(r.x, r.y.abs(), r.z);
    let chrome = |d: &Vec3, n: &Vec3| {
        let r = up(reflect(d, n));
        espejo_sky(&r, t) * 0.9 + Vec3::repeat((1.0 - d.dot(n).abs()).powi(3) * 0.4)
    };

    let ground = hit_plane(p, 0.0);
    if let Some((s, n)) = hit_spheres(&p.eye, &d, &spheres) {
        if ground.is_none_or(|(_, gd)| s < gd) {
            return chrome(&d, &n);
        }
    }
    let Some((g, dist)) = ground else { return espejo_sky(&d, t) };
    let n = Vec3::new(n2(g.x * 0.2 + t * 0.3, g.z * 0.2, 213) * 0.05, 1.0, n2(g.x * 0.2, g.z * 0.2 - t * 0.3, 214) * 0.05).normalize();
    let r = up(reflect(&d, &n));
    let col = match hit_spheres(&g, &r, &spheres) {
        Some((_, sn)) => chrome(&r, &sn) * 0.85,
        None => espejo_sky(&r, t),
    };
    mix(col.component_mul(&rgb(0.88, 0.88, 0.97)), rgb(0.25, 0.14, 0.4), 1.0 - (-dist * 0.004).exp())
}

// ------------------------------------------------------------ TOROIDE

fn candy(v: f32) -> Vec3 {
    let pal = [rgb(0.16, 0.82, 0.78), rgb(1.0, 0.43, 0.67), rgb(1.0, 0.86, 0.35)];
    let k = v.rem_euclid(3.0);
    let i = (k as usize).min(2);
    mix(pal[i], pal[(i + 1) % 3], smoothstep(0.8, 1.0, k - k.floor()))
}

/// Mundo de caramelo: desde la superficie del toro, el resto del anillo se
/// ve como un arco de colores que cruza el cielo.
fn toroide(p: &Px) -> Vec3 {
    let t = p.t;
    let d = p.dir;
    let (a, e) = angles(&d);
    let haze = rgb(1.0, 0.78, 0.86);
    let mut c = if let Some((g, dist)) = hit_plane(p, 0.0) {
        let v = g.x * 0.06 + (g.z * 0.04).sin() * 2.0 + f2(g.x * 0.02, g.z * 0.02, 3, 221) * 3.0 + t * 0.15;
        let mut col = candy(v) * (0.8 + 0.2 * f2(g.x * 0.2, g.z * 0.2, 2, 222));
        let h = hash3((g.x * 3.0).floor() as i32, (g.z * 3.0).floor() as i32, 0, 223);
        if unit(h) > 0.99 {
            col += Vec3::repeat(((t * 5.0 + unit(mix_bits(h)) * 9.0).sin() * 0.5 + 0.5).powi(6) * (-dist * 0.03).exp());
        }
        mix(col, haze, 1.0 - (-dist * 0.004).exp())
    } else {
        let mut sky = mix(haze, rgb(0.45, 0.75, 0.95), smoothstep(0.0, 0.6, d.y));
        let x = a / 1.25;
        if x.abs() < 1.0 {
            let q = (e - 0.62 * (1.0 - x * x).sqrt()) / 0.085;
            if q.abs() < 1.0 {
                let band = candy(x.asin() * 6.0 - t * 0.4) * (0.45 + 0.55 * (1.0 - q * q).sqrt());
                sky = mix(sky, band, 0.8);
            }
        }
        if let Some((cp, cd)) = hit_plane(p, 70.0) {
            let den = smoothstep(0.1, 0.45, f2(cp.x * 0.008 + t * 0.02, cp.z * 0.008, 4, 224)) * (-cd * 0.0012).exp();
            sky = mix(sky, rgb(1.0, 0.88, 0.94), den);
        }
        sky
    };

    // Chispas de colores que caen
    let colors = [rgb(0.16, 0.82, 0.78), rgb(1.0, 0.43, 0.67), rgb(1.0, 0.86, 0.35)];
    for (k, color) in colors.iter().enumerate() {
        let layer = Particles { cell: 30.0 - k as f32 * 6.0, vx: 6.0, vy: 40.0 - k as f32 * 9.0, size: 1.4 - k as f32 * 0.2, stretch: 1.6, density: 0.3, seed: 225 + k as u32 };
        c = mix(c, *color, layer.sample(p.x + (t + p.y * 0.03).sin() * 4.0, p.y, t));
    }
    c
}

// ------------------------------------------------------------ CRISTAL

fn cristal(p: &Px) -> Vec3 {
    let t = p.t;
    let d = p.dir;
    let (a, e) = angles(&d);
    let haze = rgb(0.35, 0.15, 0.6);
    let (dark, light) = (rgb(0.18, 0.05, 0.32), rgb(0.72, 0.55, 1.0));

    let mut c = if let Some((g, dist)) = hit_plane(p, 0.0) {
        // Suelo de facetas: cada celda de Voronoi es una cara del cristal
        let (f1, f2v, id) = voronoi(Vec3::new(g.x * 0.14, 0.3, g.z * 0.14), 231);
        let u = unit(id);
        let mut col = mix(dark, light, u) * 0.7;
        let seam = 1.0 - smoothstep(0.0, 0.08 + dist * 0.0006, f2v - f1);
        col = mix(col, rgb(0.75, 0.3, 1.0) * 1.3, seam * (0.6 + 0.4 * (t * 1.5 + u * 6.0).sin()));
        col += Vec3::repeat((t * 1.2 + u * 30.0).sin().max(0.0).powi(40) * 0.8);
        mix(col, haze, 1.0 - (-dist * 0.005).exp())
    } else {
        let mut sky = mix(haze, rgb(0.03, 0.01, 0.1), smoothstep(0.0, 0.6, e)) + SkyBox::stars(&d, t) * 0.7;
        let s = d.dot(&Vec3::new(0.4, 0.3, -1.0).normalize()).max(0.0);
        sky += rgb(0.7, 0.85, 1.0) * (s.powi(400) * 3.0 + s.powi(12) * 0.3);
        // Tres filas de agujas de cristal, de lejos a cerca
        for (k, &(density, width, height, seed)) in [(9.0, 0.035, 0.16, 232), (5.0, 0.06, 0.3, 233), (2.6, 0.1, 0.5, 234)].iter().enumerate() {
            if let Some((side, depth)) = spike(a + k as f32 * 0.37, e, density, width, height, seed) {
                let face = mix(dark, light, if side < 0.0 { 0.85 } else { 0.35 });
                let pulse = 0.5 + 0.5 * (t * 1.3 + k as f32 * 2.0 + a * 5.0).sin();
                let col = face + rgb(0.75, 0.27, 1.0) * (-depth * 9.0).exp() * pulse + Vec3::repeat((-side.abs() * 30.0).exp() * 0.5);
                sky = mix(haze, col, 0.4 + 0.3 * k as f32);
            }
        }
        sky
    };

    // Esquirlas flotantes que destellan
    let shards = Particles { cell: 26.0, vx: 3.0, vy: -9.0, size: 1.2, stretch: 1.0, density: 0.3, seed: 235 };
    c += rgb(0.85, 0.7, 1.0) * shards.sample(p.x, p.y, t) * (0.5 + 0.5 * (t * 4.0 + p.x * 0.3).sin());
    c
}

// ------------------------------------------------------------ CACAHUATE

/// Selva vista desde arriba de las copas; el otro lobulo del planeta se ve
/// enorme en el cielo.
fn cacahuate(p: &Px) -> Vec3 {
    let t = p.t;
    let d = p.dir;
    let haze = rgb(0.7, 0.92, 0.72);
    let sun = Vec3::new(0.5, 0.35, -1.0).normalize();

    let mut c = if let Some((g, dist)) = hit_plane(p, 0.0) {
        // Copas de los arboles: cupulas de Voronoi
        let (f1, _, id) = voronoi(Vec3::new(g.x * 0.12, 0.3, g.z * 0.12), 241);
        let u = unit(id);
        let mut col = mix(rgb(0.04, 0.26, 0.07), rgb(0.35, 0.7, 0.15), u) * (1.05 - f1 * 0.9);
        if u > 0.93 {
            col = mix(col, rgb(1.0, 0.45, 0.7), 0.6 * (1.0 - f1)); // arboles en flor
        }
        // Rios serpenteantes
        let river = 1.0 - smoothstep(0.0, 0.03 + dist * 0.0001, n2(g.x * 0.004, g.z * 0.004, 242).abs());
        col = mix(col, rgb(0.15, 0.5, 0.65) + Vec3::repeat(0.3 * (n2(g.x * 0.3 + t, g.z * 0.3, 243)).max(0.0)), river);
        // Neblina que se mueve entre los arboles
        let mist = smoothstep(0.0, 0.5, f2(g.x * 0.01 + t * 0.05, g.z * 0.01, 3, 244));
        col = mix(col, haze, mist * 0.45);
        mix(col, haze, 1.0 - (-dist * 0.0035).exp())
    } else {
        let mut sky = mix(haze, rgb(0.15, 0.5, 0.4), smoothstep(0.0, 0.6, d.y));
        let s = d.dot(&sun).max(0.0);
        sky += rgb(0.8, 1.0, 0.7) * (s.powi(400) * 3.0 + s.powi(10) * 0.3);
        // El otro lobulo del cacahuate
        let v = Vec3::new(-0.4, 0.42, -1.0).normalize();
        if let Some((u, w, r2)) = sky_disc(&d, &v, 0.33) {
            let z = (1.0 - r2).sqrt();
            let n = Vec3::new(u, w, z);
            let land = fbm(n * 2.5 + Vec3::new(t * 0.02, 0.0, 0.0), 4, 245);
            let mut surface = mix(rgb(0.08, 0.3, 0.1), rgb(0.45, 0.72, 0.2), land + 0.5);
            surface = mix(surface, rgb(0.12, 0.45, 0.6), 1.0 - smoothstep(0.0, 0.05, land.abs()));
            let lambert = (n.dot(&Vec3::new(0.6, 0.5, 0.6).normalize())).max(0.0);
            sky = mix(surface * (0.15 + lambert), sky, 0.35 + 0.4 * (1.0 - z).powi(2));
        }
        sky
    };

    // Esporas luminosas que suben
    let sway = (t * 0.7 + p.y * 0.03).sin() * 5.0;
    for (k, seed) in [246u32, 247].iter().enumerate() {
        let spores = Particles { cell: 30.0 - k as f32 * 10.0, vx: 4.0, vy: -14.0 + k as f32 * 5.0, size: 1.5 - k as f32 * 0.4, stretch: 1.0, density: 0.3, seed: *seed };
        c += rgb(0.8, 1.0, 0.4) * spores.sample(p.x + sway, p.y, t) * 0.8;
    }
    c
}

// ------------------------------------------------------------ OJO

/// Un ojo gigante en el cielo que sigue a la camara y parpadea.
fn ojo(p: &Px) -> Vec3 {
    let t = p.t;
    let d = p.dir;
    let (_, e) = angles(&d);
    let beat = (t * 3.6).sin().powi(2).powi(4);
    let haze = rgb(0.4, 0.1, 0.18);
    let skin = rgb(0.47, 0.24, 0.43);

    let mut c = if let Some((g, dist)) = hit_plane(p, 0.0) {
        // Suelo de carne con venas que laten
        let mut col = mix(skin, rgb(0.63, 0.35, 0.55), f2(g.x * 0.06, g.z * 0.06, 3, 251) + 0.5) * 0.75;
        let vein = 1.0 - smoothstep(0.0, 0.05 + dist * 0.0004, n2(g.x * 0.03, g.z * 0.03, 252).abs());
        let small = 1.0 - smoothstep(0.0, 0.06, n2(g.x * 0.12, g.z * 0.12, 253).abs());
        col = mix(col, rgb(0.85, 0.12, 0.15) * (0.7 + 0.6 * beat), vein.max(small * 0.5 * (-dist * 0.02).exp()));
        mix(col, haze, 1.0 - (-dist * 0.006).exp())
    } else {
        let mut sky = mix(haze, rgb(0.08, 0.02, 0.08), smoothstep(0.0, 0.6, e));
        let wisp = 1.0 - smoothstep(0.0, 0.12, fbm(d * 3.0 + Vec3::new(t * 0.02, 0.0, 0.0), 3, 254).abs());
        sky += rgb(0.5, 0.08, 0.1) * wisp * 0.35;

        let v = Vec3::new(0.0, 0.42, -1.0).normalize();
        let s = d.dot(&v).max(0.0);
        sky += rgb(0.8, 0.2, 0.2) * s.powi(8) * 0.35 * (0.7 + 0.3 * beat);
        if let Some((u, w, r2)) = sky_disc(&d, &v, 0.3) {
            // Parpadeos rapidos en t = 3.4 y t = 7.8
            let blink = (1.0 - ((t - 3.4) / 0.14).powi(2)).max(0.0).max((1.0 - ((t - 7.8) / 0.14).powi(2)).max(0.0));
            let open = 0.78 * (1.0 - blink) * (1.0 - u * u).sqrt();
            let ball = 0.55 + 0.45 * (1.0 - r2).sqrt();
            sky = if w.abs() > open {
                skin * ball * 0.8 // parpado
            } else {
                // El iris sigue el vaiven de la camara y mira hacia abajo
                let (ix, iy) = ((t * 0.23).sin() * 0.3, -0.22 + (t * 0.45).cos() * 0.08);
                let ri = ((u - ix).powi(2) + (w - iy).powi(2)).sqrt();
                let pupil = 0.15 * (1.0 + 0.2 * (t * 3.0).sin());
                let col = if ri < pupil {
                    rgb(0.02, 0.02, 0.03)
                } else if ri < 0.42 {
                    let k = (ri - pupil) / (0.42 - pupil);
                    let fibers = 0.6 + 0.4 * n1((w - iy).atan2(u - ix) * 12.0, 255);
                    mix(mix(rgb(0.08, 0.63, 0.35), rgb(0.9, 0.75, 0.24), k) * fibers, rgb(0.04, 0.16, 0.1), smoothstep(0.85, 1.0, k))
                } else {
                    let veins = 1.0 - smoothstep(0.0, 0.1, perlin(Vec3::new(u * 5.0, w * 5.0, 0.4), 256).abs());
                    mix(rgb(0.94, 0.92, 0.9), rgb(0.8, 0.15, 0.15), veins * smoothstep(0.45, 1.0, r2.sqrt()))
                };
                let shine = (-((u - ix + 0.1).powi(2) + (w - iy - 0.1).powi(2)) * 300.0).exp();
                col * ball + Vec3::repeat(shine)
            };
        }
        sky
    };

    let motes = Particles { cell: 28.0, vx: 5.0, vy: -6.0, size: 1.3, stretch: 1.0, density: 0.3, seed: 257 };
    c += rgb(1.0, 0.6, 0.7) * motes.sample(p.x, p.y, t) * 0.5;
    c * (1.0 - 0.12 * beat * (p.nx * p.nx + p.ny * p.ny))
}

// ------------------------------------------------------------ BURBUJA

/// Capa de burbujas de jabon en pantalla. Devuelve (color, opacidad).
fn bubbles(x: f32, y: f32, t: f32, cell: f32, rise: f32, seed: u32) -> (Vec3, f32) {
    let px = (x + (t * 0.8 + y * 0.01).sin() * cell * 0.08) / cell;
    let py = (y + rise * t) / cell;
    let (cx, cy) = (px.floor(), py.floor());
    let h = hash3(cx as i32, cy as i32, 0, seed);
    if unit(h) > 0.45 {
        return (Vec3::zeros(), 0.0);
    }
    let r = 0.16 + 0.2 * unit(mix_bits(h ^ 3));
    let (dx, dy) = (px - cx - 0.5, py - cy - 0.5);
    let q = (dx * dx + dy * dy).sqrt() / r;
    if q > 1.0 {
        return (Vec3::zeros(), 0.0);
    }
    let rim = q.powi(5);
    let film = iri(dy.atan2(dx) * 0.3 + q * 0.8 + unit(mix_bits(h)) + t * 0.1);
    let shine = (-((dx + r * 0.35).powi(2) + (dy + r * 0.35).powi(2)) / (r * r) * 40.0).exp();
    (film * (0.25 + 0.75 * rim) + Vec3::repeat(shine), (0.08 + 0.7 * rim + shine).min(1.0) * smoothstep(1.0, 0.94, q))
}

fn burbuja(p: &Px) -> Vec3 {
    let t = p.t;
    let d = p.dir;
    let haze = rgb(1.0, 0.86, 0.95);
    let sky_at = |d: &Vec3| {
        let s = d.dot(&Vec3::new(0.3, 0.3, -1.0).normalize()).max(0.0);
        mix(haze, rgb(0.6, 0.75, 1.0), smoothstep(0.0, 0.6, d.y)) + Vec3::repeat(s.powi(200) * 1.5 + s.powi(8) * 0.2)
    };
    let mut c = if let Some((g, dist)) = hit_plane(p, 0.0) {
        // Agua jabonosa: pelicula de colores sobre el reflejo del cielo
        let n = Vec3::new(n2(g.x * 0.15 + t * 0.3, g.z * 0.15, 261) * 0.06, 1.0, n2(g.x * 0.15, g.z * 0.15 + t * 0.3, 262) * 0.06).normalize();
        let r = reflect(&d, &n);
        let thick = f2(g.x * 0.04 + t * 0.04, g.z * 0.04, 3, 263);
        let mut col = mix(sky_at(&Vec3::new(r.x, r.y.abs(), r.z)), iri(thick * 2.5 + (1.0 + d.y) * 1.5), 0.4);
        // Espuma
        let (f1, f2v, _) = voronoi(Vec3::new(g.x * 0.25, t * 0.05, g.z * 0.25), 264);
        col = mix(col, Vec3::repeat(1.0), (1.0 - smoothstep(0.0, 0.07, f2v - f1)) * 0.6 * (-dist * 0.015).exp());
        mix(col, haze, 1.0 - (-dist * 0.004).exp())
    } else {
        sky_at(&d)
    };
    for &(cell, rise, seed) in &[(48.0, 14.0, 265u32), (78.0, 22.0, 266), (120.0, 32.0, 267)] {
        let (col, alpha) = bubbles(p.x, p.y, t, cell, rise, seed);
        c = mix(c, col, alpha);
    }
    c
}

// ------------------------------------------------------------ CORAL

fn caustics(x: f32, z: f32, t: f32) -> f32 {
    let v = n2(x * 0.25 + t * 0.3, z * 0.25, 271) + n2(x * 0.25 - t * 0.2, z * 0.25 + t * 0.25, 272);
    (1.0 - v.abs() * 2.5).max(0.0).powi(3)
}

/// Arrecife bajo el agua: luz que baila sobre la arena, corales que laten,
/// algas, peces y burbujas.
fn coral(p: &Px) -> Vec3 {
    let t = p.t;
    let d = p.dir;
    let (a, e) = angles(&d);
    let water = rgb(0.02, 0.25, 0.35);

    let mut c = if let Some((g, dist)) = hit_plane(p, 0.0) {
        let mut col = rgb(0.75, 0.7, 0.5) * 0.55;
        let reef = f2(g.x * 0.03, g.z * 0.03, 3, 273);
        if reef > 0.0 {
            let (f1, _, id) = voronoi(Vec3::new(g.x * 0.5, 0.3, g.z * 0.5), 274);
            let u = unit(id);
            let kind = if u < 0.4 { rgb(0.9, 0.43, 0.47) } else if u < 0.7 { rgb(1.0, 0.6, 0.25) } else { rgb(0.6, 0.35, 0.85) };
            let mut reef_col = kind * (1.0 - f1 * 0.8);
            // Celdas bioluminiscentes, cada una a su ritmo
            if u > 0.8 {
                let glow = if unit(mix_bits(id)) < 0.5 { rgb(0.25, 1.0, 0.9) } else { rgb(1.0, 0.35, 0.85) };
                reef_col += glow * (0.5 + 0.5 * (t * 2.0 + u * 40.0).sin()) * (1.0 - f1) * 0.9;
            }
            col = mix(col, reef_col, smoothstep(0.0, 0.12, reef));
        }
        col += rgb(0.6, 0.9, 0.8) * caustics(g.x, g.z, t) * 0.5;
        mix(col, water, 1.0 - (-dist * 0.022).exp())
    } else {
        // Hacia arriba: la superficie brillante y rayos de luz
        let mut up = mix(water, rgb(0.1, 0.6, 0.7), smoothstep(0.0, 0.7, e));
        up += rgb(0.5, 0.9, 0.85) * n1((a + e * 0.4) * 9.0 + t * 0.25, 275).max(0.0) * e * 1.2;
        // Algas que se mecen
        for (k, &(density, width, height, seed)) in [(7.0, 0.03, 0.3, 276), (3.5, 0.05, 0.55, 277)].iter().enumerate() {
            let sway = (t * 1.2 + e * 14.0 + k as f32).sin() * 0.03 * e;
            if spike(a + sway, e, density, width, height, seed).is_some() {
                up = mix(water, rgb(0.02, 0.2, 0.12), 0.45 + 0.35 * k as f32);
            }
        }
        up
    };

    // Banco de peces y burbujas
    let fish = Particles { cell: 26.0, vx: -55.0, vy: 4.0, size: 1.0, stretch: 3.5, density: 0.3, seed: 278 };
    c = mix(c, rgb(1.0, 0.65, 0.2), fish.sample(p.x, p.y + (t * 2.0 + p.x * 0.05).sin() * 3.0, t) * 0.9);
    let air = Particles { cell: 36.0, vx: 0.0, vy: -30.0, size: 1.6, stretch: 1.0, density: 0.3, seed: 279 };
    c + rgb(0.8, 1.0, 1.0) * air.sample(p.x + (t * 2.0 + p.y * 0.08).sin() * 3.0, p.y, t) * 0.5
}

// ------------------------------------------------------------ VACIO

/// Dentro del planeta hecho de espacio: un pasillo entre dos planos pintados
/// con el mismo shader de la nebulosa, hacia un sol negro.
fn vacio(p: &Px) -> Vec3 {
    let t = p.t;
    let d = p.dir;
    let crimson = [rgb(0.5, 0.08, 0.1), rgb(0.85, 0.25, 0.15), rgb(0.03, 0.0, 0.02)];
    let violet = [rgb(0.3, 0.06, 0.45), rgb(0.7, 0.2, 0.6), rgb(0.02, 0.0, 0.04)];
    let mut c = Vec3::zeros();
    if d.y.abs() > 1e-3 {
        let dist = 14.0 / d.y.abs();
        let g = p.eye + d * dist;
        let uv = Vec2::new(g.x * 0.004, (g.z - p.eye.z * 0.75) * 0.004 + 0.3);
        let colours = if d.y < 0.0 { &crimson } else { &violet };
        c = plasma(uv, t * 2.5 + if d.y < 0.0 { 0.0 } else { 40.0 }, colours) * 1.6 * (-dist * 0.01).exp();
    }
    // Sol negro en el punto de fuga
    let v = Vec3::new(0.0, 0.0, -1.0);
    let ang = d.dot(&v).clamp(-1.0, 1.0).acos();
    let r = 0.11;
    if ang < r {
        c = Vec3::zeros();
    } else {
        let rays = 0.7 + 0.3 * n1(d.y.atan2(d.x) * 6.0 + t * 0.3, 281);
        c += rgb(1.0, 0.28, 0.4) * (-(ang - r) * 14.0).exp() * rays * 1.3 + Vec3::repeat((-(ang - r) * 90.0).exp());
    }
    c + rgb(1.0, 0.8, 0.85) * radial_streaks(p.nx * 1.78, p.ny, t * 0.6) * 0.5
}

// ------------------------------------------------------------ TABLERO

/// Sol retro de franjas (se usa en el cielo y reflejado en el piso).
fn retro_sun(d: &Vec3, t: f32) -> Option<Vec3> {
    let (_, w, _) = sky_disc(d, &Vec3::new(0.0, 0.2, -1.0).normalize(), 0.22)?;
    // Franjas que bajan y se hacen mas anchas hacia abajo
    if w < 0.25 && (w * 7.0 + t * 0.4).rem_euclid(1.0) < 0.2 + 0.35 * (0.25 - w) {
        return None;
    }
    Some(mix(rgb(1.0, 0.15, 0.5), rgb(1.0, 0.85, 0.3), w * 0.5 + 0.5) * 1.2)
}

fn tablero(p: &Px) -> Vec3 {
    let t = p.t;
    let d = p.dir;
    let (a, e) = angles(&d);
    let haze = rgb(0.55, 0.08, 0.25);
    let (cream, dark) = (rgb(0.91, 0.86, 0.78), rgb(0.11, 0.08, 0.125));

    if let Some((g, dist)) = hit_plane(p, 0.0) {
        let (cx, cz) = ((g.x / 4.0).floor(), (g.z / 4.0).floor());
        let parity = (cx as i32 + cz as i32).rem_euclid(2);
        let mut col = if parity == 0 { cream } else { dark };
        let (fx, fz) = ((g.x / 4.0).rem_euclid(1.0), (g.z / 4.0).rem_euclid(1.0));
        let seam = fx.min(1.0 - fx).min(fz).min(1.0 - fz);
        col = mix(rgb(0.86, 0.67, 0.24), col, smoothstep(0.02, 0.05 + dist * 0.0008, seam));
        // Ola de casillas que se encienden
        let wave = smoothstep(0.85, 1.0, (cz * 0.4 + cx * 0.25 + t * 3.0).sin());
        col = mix(col, rgb(1.0, 0.8, 0.3), wave * 0.8);
        // Lejos, las casillas se funden en su color promedio (sin parpadeo)
        col = mix((cream + dark) * 0.5, col, (-dist * 0.009).exp());
        // Piso pulido: refleja el sol
        if let Some(sun) = retro_sun(&Vec3::new(d.x, -d.y, d.z), t) {
            col = mix(col, sun, 0.4);
        }
        return mix(col, haze, 1.0 - (-dist * 0.004).exp());
    }

    let mut c = mix(haze, rgb(0.05, 0.0, 0.08), smoothstep(0.0, 0.5, e)) + SkyBox::stars(&d, t) * smoothstep(0.1, 0.4, e);
    c += rgb(1.0, 0.3, 0.5) * d.dot(&Vec3::new(0.0, 0.2, -1.0).normalize()).max(0.0).powi(20) * 0.4;
    if let Some(sun) = retro_sun(&d, t) {
        c = sun;
    }
    // Fila de peones en el horizonte, uno claro y uno oscuro
    let density = 5.0;
    let i = (a * density).floor();
    let ctr = (i + 0.5) / density;
    let (w, hg) = (0.055, 0.15);
    let (x, y) = ((a - ctr) / w, e / hg);
    let head = (a - ctr).powi(2) + (e - 0.78 * hg).powi(2) < (0.4 * w).powi(2);
    if head || (y < 0.14 && x.abs() < 0.9) || (y < 0.7 && x.abs() < 0.55 - y * 0.45) {
        let piece = if (i as i32).rem_euclid(2) == 0 { cream } else { dark };
        c = piece * (0.55 + 0.45 * (1.0 - x.abs())) + rgb(1.0, 0.2, 0.5) * x.abs().powi(3) * 0.5;
    }
    c
}

// ------------------------------------------------------------ ERIZO

fn erizo(p: &Px) -> Vec3 {
    let t = p.t;
    let d = p.dir;
    let (a, e) = angles(&d);
    let haze = rgb(0.55, 0.2, 0.08);
    let pulse = 0.6 + 0.4 * (t * 2.0).sin();

    let mut c = if let Some((g, dist)) = hit_plane(p, 0.0) {
        let mut col = mix(rgb(0.37, 0.24, 0.22), rgb(0.68, 0.47, 0.37), f2(g.x * 0.08, g.z * 0.08, 3, 291) + 0.5) * 0.45;
        let crack = 1.0 - smoothstep(0.0, 0.04 + dist * 0.0004, n2(g.x * 0.04, g.z * 0.04, 292).abs());
        col = mix(col, rgb(1.0, 0.3, 0.02) * (0.6 + 0.6 * pulse), crack);
        mix(col, haze, 1.0 - (-dist * 0.006).exp())
    } else {
        let mut sky = mix(rgb(0.75, 0.28, 0.1), rgb(0.1, 0.02, 0.02), smoothstep(0.0, 0.5, e));
        // La gigante roja ocupa media vista
        let v = Vec3::new(0.3, 0.12, -1.0).normalize();
        sky += rgb(1.0, 0.3, 0.1) * d.dot(&v).max(0.0).powi(6) * 0.4;
        if let Some((u, w, r2)) = sky_disc(&d, &v, 0.36) {
            let gran = fbm(Vec3::new(u * 5.0, w * 5.0, t * 0.15), 3, 293) * 0.5 + 0.5;
            sky = mix(rgb(0.7, 0.1, 0.03), rgb(1.0, 0.5, 0.2), gran) * (0.5 + 0.6 * (1.0 - r2).sqrt());
        }
        // Agujas de roca con la punta incandescente
        for (k, &(density, width, height, seed)) in [(14.0, 0.016, 0.2, 294), (8.0, 0.025, 0.38, 295), (4.0, 0.04, 0.62, 296)].iter().enumerate() {
            if let Some((side, depth)) = spike(a + k as f32 * 0.21, e, density, width, height, seed) {
                let rock = rgb(0.1, 0.05, 0.05) * (if side < 0.0 { 1.4 } else { 0.7 });
                let tip = (-depth * 22.0).exp();
                let col = rock + mix(rgb(1.0, 0.3, 0.04), rgb(1.0, 0.92, 0.55), tip) * tip * (0.8 + 0.8 * pulse);
                sky = mix(haze, col, 0.45 + 0.27 * k as f32);
            }
        }
        sky
    };

    // Brasas arrastradas por el viento
    let embers = Particles { cell: 22.0, vx: 130.0, vy: -18.0, size: 1.1, stretch: 3.0, density: 0.3, seed: 297 };
    c += rgb(1.0, 0.5, 0.12) * embers.sample(p.x, p.y + (t * 3.0 + p.x * 0.04).sin() * 2.0, t);
    c
}

// ------------------------------------------------------------ HALO

/// Dentro del anillo de plasma: la camara va sobre la cara interna de un
/// cilindro (eje X, radio R). Cada rayo se intersecta con ese cilindro, asi
/// el suelo se curva hacia arriba y el anillo cruza el cielo por encima.
fn halo(p: &Px) -> Vec3 {
    const R: f32 = 380.0;
    const HALF_W: f32 = 85.0;
    let t = p.t;
    let d = p.dir;
    let space = SkyBox::stars(&d, t) * 0.8 + rgb(0.08, 0.01, 0.01);

    // Circulo en el plano YZ: |o + s*dd|^2 = R^2 (la camara esta adentro)
    let (oy, oz) = (p.eye.y - R, p.eye.z);
    let aa = d.y * d.y + d.z * d.z;
    if aa < 1e-6 {
        return space;
    }
    let b = oy * d.y + oz * d.z;
    let s = (-b + (b * b - aa * (oy * oy + oz * oz - R * R)).max(0.0).sqrt()) / aa;
    let hit = p.eye + d * s;
    if hit.x.abs() > HALF_W {
        return space;
    }

    // Coordenadas sobre el anillo: a lo ancho (x) y a lo largo (arco)
    let phi = hit.z.atan2(R - hit.y);
    let arc = phi * R - t * 22.0; // la camara avanza por el anillo
    let heat = fbm(Vec3::new(hit.x * 0.02, arc * 0.012 - t * 0.2, t * 0.1), 4, 301) * 0.5 + 0.5;
    let heat = (heat * 1.5 - 0.2).clamp(0.0, 1.0);
    let mut col = if heat < 0.5 {
        mix(rgb(0.24, 0.02, 0.02), rgb(0.94, 0.31, 0.08), heat * 2.0)
    } else {
        mix(rgb(0.94, 0.31, 0.08), rgb(1.0, 0.92, 0.63), heat * 2.0 - 1.0)
    };
    // Placas de corteza oscura flotando sobre el plasma
    let (f1, f2v, id) = voronoi(Vec3::new(hit.x * 0.06, 0.3, arc * 0.06), 302);
    if unit(id) < 0.55 {
        let plate = smoothstep(0.03, 0.12, f2v - f1);
        col = mix(col, rgb(0.07, 0.03, 0.03) * (1.0 + f1), plate * 0.92);
    }
    // Bordes del anillo
    let edge = smoothstep(HALF_W - 7.0, HALF_W, hit.x.abs());
    col = mix(col, rgb(1.0, 0.8, 0.45), edge);
    // Bruma solo cerca: la parte lejana del anillo se ve nitida en el cielo
    let col = mix(col, rgb(0.5, 0.12, 0.05), 1.0 - (-s.min(260.0) * 0.004).exp());
    let embers = Particles { cell: 24.0, vx: 5.0, vy: -26.0, size: 1.1, stretch: 1.0, density: 0.3, seed: 303 };
    col + rgb(1.0, 0.6, 0.2) * embers.sample(p.x, p.y, t)
}
