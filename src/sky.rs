//! Fondo espacial: el shader de nebulosa (port de nebula.frag / nebula.wgsl)
//! mas dos capas de estrellas que parpadean.
//!
//! El shader original es de pantalla completa y corre en GPU. Aqui va sobre
//! una esfera gigante que envuelve el sistema solar, asi que se evalua por
//! direccion en lugar de por pixel de pantalla:
//!
//! * La nebulosa (`plasma`) es cara (~30 funciones trigonometricas), asi que se
//!   hornea en un cubemap de baja resolucion. Para animarla se guardan dos
//!   cuadros clave (A en t, B en t+I) y se mezclan; mientras tanto el cuadro
//!   siguiente se calcula poco a poco, unas filas por frame y en varios hilos.
//!   La nebulosa es suave, asi que con filtrado bilineal no se nota la resolucion.
//! * Las estrellas se calculan por fragmento (son baratas y necesitan nitidez),
//!   con la misma logica del shader original: hash por celda, umbral, tres
//!   colores posibles y parpadeo con frecuencia y fase aleatorias.
//!
//! No se aplica iluminacion: las estrellas no dependen de la luz del sol.

use std::thread;

use nalgebra_glm::{Vec2, Vec3};

use crate::math::smoothstep;
use crate::noise::{hash3, mix_bits, unit};

// Constantes del shader original
const SPIN_ROTATION: f32 = -2.0;
const SPIN_SPEED: f32 = 0.9;
const COLOUR_1: Vec3 = Vec3::new(0.04, 0.08, 0.14);
const COLOUR_2: Vec3 = Vec3::new(0.08, 0.14, 0.22);
const COLOUR_3: Vec3 = Vec3::new(0.01, 0.01, 0.02);
const CONTRAST: f32 = 2.0;
const LIGHTING: f32 = 0.15;
const SPIN_AMOUNT: f32 = 0.25;
const SPIN_EASE: f32 = 0.1;

/// Velocidad de la animacion de la nebulosa respecto al tiempo real.
pub const NEBULA_SPEED: f32 = 0.5;
/// Segundos entre cuadros clave de la nebulosa.
const KEYFRAME_INTERVAL: f32 = 0.4;

/// `get_plasma` del shader original. `uv` es la coordenada ya centrada
/// (equivalente a (fragCoord - 0.5*res) / length(res)).
pub fn plasma(uv: Vec2, time: f32) -> Vec3 {
    let uv_len = uv.norm();
    let speed = SPIN_ROTATION * SPIN_EASE * 0.2 * time + 302.2;
    let angle = uv.y.atan2(uv.x) + speed
        - SPIN_EASE * 20.0 * (SPIN_AMOUNT * uv_len + (1.0 - SPIN_AMOUNT));
    // En el original se suma y luego se resta `mid`, asi que se cancela
    let mut uv = Vec2::new(uv_len * angle.cos(), uv_len * angle.sin()) * 30.0;

    let speed = time * SPIN_SPEED;
    let mut uv2 = Vec2::repeat(uv.x + uv.y);
    for _ in 0..5 {
        uv2 += Vec2::repeat(uv.x.max(uv.y).sin()) + uv;
        uv += Vec2::new(
            (5.1123314 + 0.353 * uv2.y + speed * 0.131121).cos(),
            (uv2.x - 0.113 * speed).sin(),
        ) * 0.5;
        uv -= Vec2::repeat((uv.x + uv.y).cos() - (uv.x * 0.711 - uv.y).sin());
    }

    let contrast_mod = 0.25 * CONTRAST + 0.5 * SPIN_AMOUNT + 1.2;
    let paint_res = (uv.norm() * 0.035 * contrast_mod).clamp(0.0, 2.0);
    let c1p = (1.0 - contrast_mod * (1.0 - paint_res).abs()).max(0.0);
    let c2p = (1.0 - contrast_mod * paint_res.abs()).max(0.0);
    let c3p = 1.0 - (c1p + c2p).min(1.0);
    let light = (LIGHTING - 0.2) * (c1p * 5.0 - 4.0).max(0.0) + LIGHTING * (c2p * 5.0 - 4.0).max(0.0);

    COLOUR_1 * (0.3 / CONTRAST)
        + (COLOUR_1 * c1p + COLOUR_2 * c2p + COLOUR_3 * c3p) * (1.0 - 0.3 / CONTRAST)
        + Vec3::repeat(light)
}

/// Nebulosa en una direccion. La direccion 3D se convierte en una coordenada
/// 2D mezclando las proyecciones sobre los planos YZ, XZ y XY (triplanar).
/// Se mezclan las coordenadas y no los colores: asi el plasma se evalua una
/// sola vez y no pierde contraste donde se juntan dos proyecciones. Cada
/// hemisferio recibe un desplazamiento distinto para que el cielo no se vea
/// en espejo; el cambio ocurre donde el peso de ese plano es 0, sin costura.
fn nebula_at(d: Vec3, time: f32) -> Vec3 {
    let mut w = Vec3::new(d.x.powi(4), d.y.powi(4), d.z.powi(4));
    w /= w.x + w.y + w.z;
    let side = |v: f32| if v >= 0.0 { 1.0 } else { -1.0 };
    let uv = (Vec2::new(d.y, d.z) + Vec2::new(0.9, 0.0) * side(d.x)) * w.x
        + (Vec2::new(d.x, d.z) + Vec2::new(0.0, 0.9) * side(d.y)) * w.y
        + (Vec2::new(d.x, d.y) + Vec2::new(-0.6, 0.6) * side(d.z)) * w.z;
    plasma(uv * 0.5, time)
}

/// Cara del cubo y coordenadas (u, v) en [-1, 1] para una direccion.
#[inline]
fn face_uv(d: &Vec3) -> (usize, f32, f32) {
    let (ax, ay, az) = (d.x.abs(), d.y.abs(), d.z.abs());
    if ax >= ay && ax >= az {
        if d.x > 0.0 { (0, -d.z / ax, d.y / ax) } else { (1, d.z / ax, d.y / ax) }
    } else if ay >= az {
        if d.y > 0.0 { (2, d.x / ay, -d.z / ay) } else { (3, d.x / ay, d.z / ay) }
    } else if d.z > 0.0 {
        (4, d.x / az, d.y / az)
    } else {
        (5, -d.x / az, d.y / az)
    }
}

/// Inversa de `face_uv`.
fn face_dir(face: usize, u: f32, v: f32) -> Vec3 {
    match face {
        0 => Vec3::new(1.0, v, -u),
        1 => Vec3::new(-1.0, v, u),
        2 => Vec3::new(u, 1.0, -v),
        3 => Vec3::new(u, -1.0, v),
        4 => Vec3::new(u, v, 1.0),
        _ => Vec3::new(-u, v, -1.0),
    }
    .normalize()
}

pub struct SkyBox {
    n: usize,
    a: Vec<Vec3>,
    b: Vec<Vec3>,
    next: Vec<Vec3>,
    t_a: f32,
    rows_done: usize,
}

impl SkyBox {
    /// `n` = resolucion de cada cara del cubemap de la nebulosa.
    pub fn new(n: usize, time: f32) -> Self {
        let size = 6 * n * n;
        let mut sky = SkyBox {
            n,
            a: vec![Vec3::zeros(); size],
            b: vec![Vec3::zeros(); size],
            next: vec![Vec3::zeros(); size],
            t_a: time,
            rows_done: 0,
        };
        sky.reset(time);
        sky
    }

    fn reset(&mut self, time: f32) {
        let rows = 6 * self.n;
        self.t_a = time;
        bake(&mut self.a, self.n, 0..rows, time);
        bake(&mut self.b, self.n, 0..rows, time + KEYFRAME_INTERVAL);
        self.rows_done = 0;
    }

    /// Avanza la animacion: hornea las filas del siguiente cuadro clave que
    /// tocan en este frame y rota los cuadros cuando se cumple el intervalo.
    pub fn update(&mut self, time: f32) {
        let rows = 6 * self.n;
        if time < self.t_a || time > self.t_a + 2.0 * KEYFRAME_INTERVAL {
            self.reset(time);
            return;
        }
        let next_time = self.t_a + 2.0 * KEYFRAME_INTERVAL;
        if time >= self.t_a + KEYFRAME_INTERVAL {
            bake(&mut self.next, self.n, self.rows_done..rows, next_time);
            std::mem::swap(&mut self.a, &mut self.b);
            std::mem::swap(&mut self.b, &mut self.next);
            self.t_a += KEYFRAME_INTERVAL;
            self.rows_done = 0;
            return;
        }
        let progress = (time - self.t_a) / KEYFRAME_INTERVAL;
        let target = ((progress * rows as f32).ceil() as usize).min(rows);
        if target > self.rows_done {
            bake(&mut self.next, self.n, self.rows_done..target, next_time);
            self.rows_done = target;
        }
    }

    fn sample_tex(&self, tex: &[Vec3], face: usize, u: f32, v: f32) -> Vec3 {
        let n = self.n;
        let fx = ((u * 0.5 + 0.5) * n as f32 - 0.5).clamp(0.0, (n - 1) as f32);
        let fy = ((v * 0.5 + 0.5) * n as f32 - 0.5).clamp(0.0, (n - 1) as f32);
        let (x0, y0) = (fx as usize, fy as usize);
        let (x1, y1) = ((x0 + 1).min(n - 1), (y0 + 1).min(n - 1));
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let base = face * n * n;
        let p = |x: usize, y: usize| tex[base + y * n + x];
        let top = p(x0, y0) + (p(x1, y0) - p(x0, y0)) * tx;
        let bottom = p(x0, y1) + (p(x1, y1) - p(x0, y1)) * tx;
        top + (bottom - top) * ty
    }

    /// Nebulosa animada: mezcla de los dos cuadros clave.
    fn nebula(&self, dir: &Vec3, time: f32) -> Vec3 {
        let (face, u, v) = face_uv(dir);
        let f = ((time - self.t_a) / KEYFRAME_INTERVAL).clamp(0.0, 1.0);
        let ca = self.sample_tex(&self.a, face, u, v);
        let cb = self.sample_tex(&self.b, face, u, v);
        ca + (cb - ca) * f
    }

    /// Las dos capas de estrellas en una direccion.
    fn stars(dir: &Vec3, time: f32) -> Vec3 {
        let (face, u, v) = face_uv(dir);
        // Cubemap "equi-angular": reparte las celdas de estrellas de forma pareja
        let k = 4.0 / std::f32::consts::PI;
        let (ea, eb) = (u.atan() * k, v.atan() * k);
        let small = star_layer(face, ea, eb, 340.0, 0.9935, 0.42, 0, time);
        let large = star_layer(face, ea, eb, 110.0, 0.9940, 0.30, 1, time);
        small.sup(&large)
    }

    /// Color del cielo en la direccion `dir` (unitaria) en el instante `time`.
    pub fn color(&self, dir: &Vec3, time: f32) -> Vec3 {
        self.nebula(dir, time) + Self::stars(dir, time)
    }

    /// Cielo con efecto de velocidad (turbo). `forward` es hacia donde mira
    /// la camara y `amount` (0..1) la intensidad del efecto.
    ///
    /// * Estelas: cada estrella se muestrea varias veces a lo largo del arco
    ///   que va hacia el punto de fuga (`forward`) y se queda el maximo, asi
    ///   cada estrella se estira en una linea que sale del centro. El largo
    ///   crece con el angulo, como en un desenfoque de movimiento real.
    /// * Tunel: la direccion con que se lee la nebulosa se desplaza con una
    ///   onda que viaja hacia afuera con el tiempo, y se agregan anillos de
    ///   luz que avanzan; los bordes se tinen de azul.
    pub fn color_warped(&self, dir: &Vec3, time: f32, forward: &Vec3, amount: f32) -> Vec3 {
        if amount < 0.01 {
            return self.color(dir, time);
        }
        let c = dir.dot(forward).clamp(-1.0, 1.0);
        let theta = c.acos(); // angulo respecto al punto de fuga
        let side = (1.0 - c * c).sqrt(); // = sin(theta), sin otra funcion trigonometrica

        // Una sola onda que viaja hacia afuera: deforma la nebulosa y sus
        // crestas se ven como anillos de luz
        let ripple = (theta * 12.0 - time * 12.0).sin();
        let pull = amount * side * (0.22 + 0.07 * ripple);
        let nebula_dir = (dir - forward * pull).normalize();
        let mut nebula = self.nebula(&nebula_dir, time);
        let tint = Vec3::new(1.0, 1.0, 1.0) + Vec3::new(-0.25, 0.05, 0.6) * (amount * side);
        nebula = nebula.component_mul(&tint);
        let crest = ripple.max(0.0);
        let rings = crest * crest * crest * crest;
        nebula += Vec3::new(0.04, 0.08, 0.18) * rings * amount * side;

        // Coordenadas polares alrededor del punto de fuga: phi = angulo
        // alrededor del eje, theta = distancia angular al centro
        let helper = if forward.y.abs() < 0.9 { Vec3::y() } else { Vec3::x() };
        let t1 = forward.cross(&helper).normalize();
        let t2 = forward.cross(&t1);
        let phi = dir.dot(&t2).atan2(dir.dot(&t1));
        let fade_center = smoothstep(0.04, 0.35, theta);
        let mut streaks = Vec3::zeros();
        if fade_center > 0.0 {
            let log_theta = theta.ln();
            streaks = (streak_layer(phi, side, log_theta, time, 260.0, 0.22, 0)
                + streak_layer(phi, side, log_theta, time, 140.0, 0.16, 1))
                * (amount * fade_center);
        }

        // Las estrellas reales se atenuan un poco para que dominen las estelas
        nebula + Self::stars(dir, time) * (1.0 - 0.5 * amount) + streaks
    }
}

/// Estelas de velocidad. El cielo se divide en `lanes` carriles que salen
/// del punto de fuga (por angulo `phi`); cada carril se corta en celdas a lo
/// largo de `theta` en escala logaritmica, y la coordenada se desplaza con el
/// tiempo, asi que cada estela viaja hacia afuera y se acelera al alejarse
/// del centro (como cuando uno avanza muy rapido). Cada celda decide con un
/// hash si tiene estela, su largo, velocidad y color.
fn streak_layer(phi: f32, sin_theta: f32, log_theta: f32, time: f32, lanes: f32, density: f32, layer: i32) -> Vec3 {
    let fp = (phi / std::f32::consts::TAU + 0.5) * lanes;
    let lane = fp.floor();

    // Ancho angular: la linea mide ~1.5 px sin importar que tan lejos del centro.
    // La mayoria de los pixeles caen entre lineas y salen aqui, antes de los hash.
    let lateral = (fp - lane - 0.5).abs() * std::f32::consts::TAU / lanes * sin_theta;
    let line = 1.0 - smoothstep(0.0012, 0.0032, lateral);
    if line <= 0.0 {
        return Vec3::zeros();
    }

    let lane_id = lane as i32;
    let lane_hash = unit(hash3(lane_id, layer, 0, 40));
    let speed = 3.0 + 3.0 * lane_hash;
    let r = log_theta * 5.0 - time * speed + lane_hash * 17.0;
    let cell = r.floor();
    let along = r - cell;
    let cell_hash = hash3(lane_id, cell as i32, layer, 41);
    if unit(cell_hash) > density {
        return Vec3::zeros();
    }
    let length = 0.25 + 0.5 * unit(mix_bits(cell_hash));
    if along > length {
        return Vec3::zeros();
    }
    // Cola tenue, cabeza brillante (la cabeza va por delante, mas afuera)
    let head = (along / length).powi(2);
    let color = if unit(mix_bits(cell_hash ^ 0x5bd1)) < 0.6 {
        Vec3::new(0.75, 0.88, 1.0)
    } else {
        Vec3::new(1.0, 0.85, 1.0)
    };
    color * head * line * 1.4
}

/// `get_star_layer` del shader original, en celdas sobre la cara del cubo.
#[allow(clippy::too_many_arguments)]
fn star_layer(face: usize, a: f32, b: f32, cells: f32, threshold: f32, radius: f32, layer: u32, time: f32) -> Vec3 {
    let x = (a * 0.5 + 0.5) * cells;
    let y = (b * 0.5 + 0.5) * cells;
    let (cx, cy) = (x.floor(), y.floor());
    let (ix, iy, iz) = (cx as i32, cy as i32, (face as u32 * 4 + layer) as i32);
    if unit(hash3(ix, iy, iz, 1)) <= threshold {
        return Vec3::zeros();
    }
    // Cada estrella es un punto redondo en una posicion aleatoria de su celda
    let jx = 0.3 + 0.4 * unit(hash3(ix, iy, iz, 2));
    let jy = 0.3 + 0.4 * unit(hash3(ix, iy, iz, 3));
    let d = ((x - cx - jx).powi(2) + (y - cy - jy).powi(2)).sqrt() / radius;
    if d >= 1.0 {
        return Vec3::zeros();
    }
    let shape = (1.0 - d) * (1.0 - d);

    let color_seed = unit(hash3(ix, iy, iz, 10));
    let star_color = if color_seed < 0.33 {
        Vec3::new(1.0, 1.0, 1.0)
    } else if color_seed < 0.66 {
        Vec3::new(1.0, 0.95, 0.5)
    } else {
        Vec3::new(0.6, 0.85, 1.0)
    };
    let freq = 1.0 + unit(hash3(ix, iy, iz, 20)) * 3.0;
    let phase = unit(hash3(ix, iy, iz, 30)) * 6.2831;
    let flicker = ((time * freq + phase).sin() * 1.5 - 0.5).clamp(0.0, 1.0);
    // Piso de brillo para que el cielo nunca quede vacio
    let twinkle = 0.35 + 0.65 * smoothstep(0.0, 1.0, flicker);
    star_color * shape * twinkle * 1.3
}

/// Hornea las filas `rows` (contando las 6 caras seguidas) en paralelo.
fn bake(tex: &mut [Vec3], n: usize, rows: std::ops::Range<usize>, time: f32) {
    if rows.is_empty() {
        return;
    }
    let start = rows.start;
    let slice = &mut tex[rows.start * n..rows.end * n];
    let threads = thread::available_parallelism().map(|t| t.get()).unwrap_or(4);
    let rows_per_thread = rows.len().div_ceil(threads);
    let t = time * NEBULA_SPEED;
    thread::scope(|s| {
        for (i, chunk) in slice.chunks_mut(rows_per_thread * n).enumerate() {
            s.spawn(move || {
                for (j, texel) in chunk.iter_mut().enumerate() {
                    let row = start + i * rows_per_thread + j / n;
                    let (face, y, x) = (row / n, row % n, j % n);
                    let u = (x as f32 + 0.5) / n as f32 * 2.0 - 1.0;
                    let v = (y as f32 + 0.5) / n as f32 * 2.0 - 1.0;
                    *texel = nebula_at(face_dir(face, u, v), t);
                }
            });
        }
    });
}
