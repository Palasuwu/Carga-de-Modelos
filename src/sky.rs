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
//! * Cada galaxia tiene su propio estilo (`SkyStyle`): los 3 colores del shader
//!   y nubes de nebulosa brillantes en algunas direcciones. Las nubes no se
//!   animan, asi que se hornean una sola vez en otra textura.
//! * Las estrellas se calculan por fragmento (son baratas y necesitan nitidez),
//!   con la misma logica del shader original: hash por celda, umbral, tres
//!   colores posibles y parpadeo con frecuencia y fase aleatorias.
//!
//! No se aplica iluminacion: las estrellas no dependen de la luz del sol.

use std::thread;

use nalgebra_glm::{Vec2, Vec3};

use crate::math::smoothstep;
use crate::noise::{fbm, hash3, mix_bits, unit};

// Constantes del shader original
const SPIN_ROTATION: f32 = -2.0;
const SPIN_SPEED: f32 = 0.9;
const CONTRAST: f32 = 2.0;
const LIGHTING: f32 = 0.15;
const SPIN_AMOUNT: f32 = 0.25;
const SPIN_EASE: f32 = 0.1;

/// Velocidad de la animacion de la nebulosa respecto al tiempo real.
pub const NEBULA_SPEED: f32 = 0.5;
/// Segundos entre cuadros clave de la nebulosa.
const KEYFRAME_INTERVAL: f32 = 0.4;
/// Resolucion de cada cara del cubemap.
pub const SKY_RESOLUTION: usize = 192;

/// Nube de nebulosa en el fondo de una galaxia.
#[derive(Clone)]
pub struct Cloud {
    pub dir: Vec3,
    pub color: Vec3,
    /// radio angular (radianes)
    pub size: f32,
    pub seed: u32,
}

/// Apariencia del cielo de una galaxia.
#[derive(Clone)]
pub struct SkyStyle {
    /// COLOUR_1, COLOUR_2 y COLOUR_3 del shader original
    pub colours: [Vec3; 3],
    pub clouds: Vec<Cloud>,
    /// Color dominante: tine el tunel hiperespacial al viajar a esta galaxia
    pub tint: Vec3,
}

/// `get_plasma` del shader original. `uv` es la coordenada ya centrada
/// (equivalente a (fragCoord - 0.5*res) / length(res)).
pub fn plasma(uv: Vec2, time: f32, colours: &[Vec3; 3]) -> Vec3 {
    let [colour_1, colour_2, colour_3] = *colours;
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

    colour_1 * (0.3 / CONTRAST)
        + (colour_1 * c1p + colour_2 * c2p + colour_3 * c3p) * (1.0 - 0.3 / CONTRAST)
        + Vec3::repeat(light)
}

/// Nebulosa en una direccion. La direccion 3D se convierte en una coordenada
/// 2D mezclando las proyecciones sobre los planos YZ, XZ y XY (triplanar).
/// Se mezclan las coordenadas y no los colores: asi el plasma se evalua una
/// sola vez y no pierde contraste donde se juntan dos proyecciones. Cada
/// hemisferio recibe un desplazamiento distinto para que el cielo no se vea
/// en espejo; el cambio ocurre donde el peso de ese plano es 0, sin costura.
fn nebula_at(d: Vec3, time: f32, colours: &[Vec3; 3]) -> Vec3 {
    let mut w = Vec3::new(d.x.powi(4), d.y.powi(4), d.z.powi(4));
    w /= w.x + w.y + w.z;
    let side = |v: f32| if v >= 0.0 { 1.0 } else { -1.0 };
    let uv = (Vec2::new(d.y, d.z) + Vec2::new(0.9, 0.0) * side(d.x)) * w.x
        + (Vec2::new(d.x, d.z) + Vec2::new(0.0, 0.9) * side(d.y)) * w.y
        + (Vec2::new(d.x, d.y) + Vec2::new(-0.6, 0.6) * side(d.z)) * w.z;
    plasma(uv * 0.5, time, colours)
}

/// Nubes de nebulosa: fBm recortado por una mascara angular alrededor de
/// la direccion de cada nube, con un nucleo mas brillante.
fn clouds_at(d: Vec3, clouds: &[Cloud]) -> Vec3 {
    let mut color = Vec3::zeros();
    for cloud in clouds {
        let c = d.dot(&cloud.dir);
        let mask = smoothstep(cloud.size.cos(), (cloud.size * 0.25).cos(), c);
        if mask <= 0.0 {
            continue;
        }
        let offset = Vec3::new(cloud.seed as f32 * 1.7, 3.1, cloud.seed as f32 * 0.9);
        let n = fbm(d * 3.5 + offset, 5, cloud.seed) * 0.5 + 0.5;
        let wisps = fbm(d * 9.0 - offset, 3, cloud.seed + 7) * 0.5 + 0.5;
        let density = (mask * (n * 1.5 - 0.45 + wisps * 0.3)).max(0.0).powf(1.6);
        color += cloud.color * density + Vec3::repeat(density.powi(3) * 0.25);
    }
    color
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
    style: SkyStyle,
    a: Vec<Vec3>,
    b: Vec<Vec3>,
    next: Vec<Vec3>,
    clouds: Vec<Vec3>,
    t_a: f32,
    /// Se suma al reloj del juego para obtener el tiempo del patron. Permite
    /// "reenganchar" un cielo horneado antes (en otro hilo) sin saltos.
    offset: f32,
    rows_done: usize,
}

impl SkyBox {
    /// `threads` = hilos para el horneado inicial (1 si se carga en segundo plano).
    pub fn new(style: SkyStyle, time: f32, threads: usize) -> Self {
        let n = SKY_RESOLUTION;
        let size = 6 * n * n;
        let mut sky = SkyBox {
            n,
            a: vec![Vec3::zeros(); size],
            b: vec![Vec3::zeros(); size],
            next: vec![Vec3::zeros(); size],
            clouds: vec![Vec3::zeros(); size],
            t_a: time,
            offset: 0.0,
            rows_done: 0,
            style,
        };
        let rows = 0..6 * n;
        let style = sky.style.clone();
        bake(&mut sky.clouds, n, rows.clone(), threads, |d| clouds_at(d, &style.clouds));
        sky.bake_keyframes(time, threads);
        sky
    }

    /// Libera las texturas del cielo (mientras se ve la atmosfera de un
    /// planeta no hacen falta). Despues de esto el cielo no se puede usar:
    /// hay que construir uno nuevo.
    pub fn unload(&mut self) {
        self.a = Vec::new();
        self.b = Vec::new();
        self.next = Vec::new();
        self.clouds = Vec::new();
    }

    pub fn tint(&self) -> Vec3 {
        self.style.tint
    }

    fn bake_keyframes(&mut self, time: f32, threads: usize) {
        let rows = 0..6 * self.n;
        self.t_a = time;
        let colours = self.style.colours;
        let t0 = (time + self.offset) * NEBULA_SPEED;
        let t1 = (time + KEYFRAME_INTERVAL + self.offset) * NEBULA_SPEED;
        bake(&mut self.a, self.n, rows.clone(), threads, |d| nebula_at(d, t0, &colours));
        bake(&mut self.b, self.n, rows, threads, |d| nebula_at(d, t1, &colours));
        self.rows_done = 0;
    }

    /// Ajusta el reloj para que este cielo continue en `now` sin rehornear:
    /// el cuadro A pasa a empezar ahora y el patron sigue donde iba.
    pub fn rebase(&mut self, now: f32) {
        self.offset += self.t_a - now;
        self.t_a = now;
        self.rows_done = 0;
    }

    /// Avanza la animacion: hornea las filas del siguiente cuadro clave que
    /// tocan en este frame y rota los cuadros cuando se cumple el intervalo.
    pub fn update(&mut self, time: f32) {
        let rows = 6 * self.n;
        let threads = thread::available_parallelism().map(|t| t.get()).unwrap_or(4);
        if time < self.t_a || time > self.t_a + 2.0 * KEYFRAME_INTERVAL {
            self.bake_keyframes(time, threads);
            return;
        }
        let colours = self.style.colours;
        let next_time = (self.t_a + 2.0 * KEYFRAME_INTERVAL + self.offset) * NEBULA_SPEED;
        let shader = |d| nebula_at(d, next_time, &colours);
        if time >= self.t_a + KEYFRAME_INTERVAL {
            bake(&mut self.next, self.n, self.rows_done..rows, threads, shader);
            std::mem::swap(&mut self.a, &mut self.b);
            std::mem::swap(&mut self.b, &mut self.next);
            self.t_a += KEYFRAME_INTERVAL;
            self.rows_done = 0;
            return;
        }
        let progress = (time - self.t_a) / KEYFRAME_INTERVAL;
        let target = ((progress * rows as f32).ceil() as usize).min(rows);
        if target > self.rows_done {
            bake(&mut self.next, self.n, self.rows_done..target, threads, shader);
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

    /// Nebulosa animada (mezcla de los dos cuadros clave) mas las nubes.
    pub fn nebula(&self, dir: &Vec3, time: f32) -> Vec3 {
        let (face, u, v) = face_uv(dir);
        let f = ((time - self.t_a) / KEYFRAME_INTERVAL).clamp(0.0, 1.0);
        let ca = self.sample_tex(&self.a, face, u, v);
        let cb = self.sample_tex(&self.b, face, u, v);
        ca + (cb - ca) * f + self.sample_tex(&self.clouds, face, u, v)
    }

    /// Las dos capas de estrellas en una direccion.
    pub fn stars(dir: &Vec3, time: f32) -> Vec3 {
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
    /// * Estelas: lineas de luz que salen del punto de fuga y viajan hacia
    ///   afuera con el tiempo (ver `streak_layer`).
    /// * Tunel: la direccion con que se lee la nebulosa se desplaza con una
    ///   onda que viaja hacia afuera con el tiempo, y se agregan anillos de
    ///   luz que avanzan; los bordes se tinen de azul.
    pub fn color_warped(&self, dir: &Vec3, time: f32, forward: &Vec3, amount: f32) -> Vec3 {
        if amount < 0.01 {
            return self.color(dir, time);
        }
        let p = Polar::new(dir, forward);

        // Una sola onda que viaja hacia afuera: deforma la nebulosa y sus
        // crestas se ven como anillos de luz
        let ripple = (p.theta * 12.0 - time * 12.0).sin();
        let pull = amount * p.side * (0.22 + 0.07 * ripple);
        let nebula_dir = (dir - forward * pull).normalize();
        let mut nebula = self.nebula(&nebula_dir, time);
        let tint = Vec3::new(1.0, 1.0, 1.0) + Vec3::new(-0.25, 0.05, 0.6) * (amount * p.side);
        nebula = nebula.component_mul(&tint);
        let crest = ripple.max(0.0);
        let rings = crest * crest * crest * crest;
        nebula += Vec3::new(0.04, 0.08, 0.18) * rings * amount * p.side;

        let fade_center = smoothstep(0.04, 0.35, p.theta);
        let mut streaks = Vec3::zeros();
        if fade_center > 0.0 {
            streaks = (streak_layer(&p, time, 260.0, 0.22, 0) + streak_layer(&p, time, 140.0, 0.16, 1))
                * (amount * fade_center);
        }

        // Las estrellas reales se atenuan un poco para que dominen las estelas
        nebula + Self::stars(dir, time) * (1.0 - 0.5 * amount) + streaks
    }
}

/// Coordenadas polares de `dir` alrededor del eje `forward`.
pub struct Polar {
    /// angulo al punto de fuga
    theta: f32,
    side: f32,
    log_theta: f32,
    /// angulo alrededor del eje
    phi: f32,
}

impl Polar {
    pub fn new(dir: &Vec3, forward: &Vec3) -> Self {
        let c = dir.dot(forward).clamp(-1.0, 1.0);
        let theta = c.acos().max(1e-4);
        let helper = if forward.y.abs() < 0.9 { Vec3::y() } else { Vec3::x() };
        let t1 = forward.cross(&helper).normalize();
        let t2 = forward.cross(&t1);
        Polar {
            theta,
            side: (1.0 - c * c).sqrt(),
            log_theta: theta.ln(),
            phi: dir.dot(&t2).atan2(dir.dot(&t1)),
        }
    }
}

/// Tunel hiperespacial (viaje entre galaxias). Mismas coordenadas polares
/// que las estelas: paredes que giran en espiral y bandas que avanzan hacia
/// la camara, un nucleo blanco en el punto de fuga y estelas muy densas.
/// `tint` es el color de la galaxia (se mezcla entre origen y destino).
pub fn hyperspace(dir: &Vec3, forward: &Vec3, time: f32, tint: &Vec3) -> Vec3 {
    let p = Polar::new(dir, forward);
    let swirl = (p.phi * 4.0 + p.log_theta * 2.5 - time * 5.0).sin() * 0.5 + 0.5;
    let bands = (p.log_theta * 7.0 - time * 16.0).sin() * 0.5 + 0.5;
    let wall = tint * (0.12 + 0.9 * swirl * bands * bands) * smoothstep(0.03, 0.6, p.theta);
    let core = Vec3::new(1.0, 0.95, 0.9) * (-p.theta * 6.0).exp() * 1.8;
    let streaks = (streak_layer(&p, time * 2.0, 260.0, 0.45, 0) + streak_layer(&p, time * 2.0, 140.0, 0.35, 1))
        * smoothstep(0.03, 0.3, p.theta);
    wall + core + streaks
}

/// Estelas de velocidad. El cielo se divide en `lanes` carriles que salen
/// del punto de fuga (por angulo `phi`); cada carril se corta en celdas a lo
/// largo de `theta` en escala logaritmica, y la coordenada se desplaza con el
/// tiempo, asi que cada estela viaja hacia afuera y se acelera al alejarse
/// del centro (como cuando uno avanza muy rapido). Cada celda decide con un
/// hash si tiene estela, su largo, velocidad y color.
fn streak_layer(p: &Polar, time: f32, lanes: f32, density: f32, layer: i32) -> Vec3 {
    let fp = (p.phi / std::f32::consts::TAU + 0.5) * lanes;
    let lane = fp.floor();

    // Ancho angular: la linea mide ~1.5 px sin importar que tan lejos del centro.
    // La mayoria de los pixeles caen entre lineas y salen aqui, antes de los hash.
    let lateral = (fp - lane - 0.5).abs() * std::f32::consts::TAU / lanes * p.side;
    let line = 1.0 - smoothstep(0.0012, 0.0032, lateral);
    if line <= 0.0 {
        return Vec3::zeros();
    }

    let lane_id = lane as i32;
    let lane_hash = unit(hash3(lane_id, layer, 0, 40));
    let speed = 3.0 + 3.0 * lane_hash;
    let r = p.log_theta * 5.0 - time * speed + lane_hash * 17.0;
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

/// Hornea las filas `rows` (contando las 6 caras seguidas) con `threads`
/// hilos, evaluando `shader` en la direccion del centro de cada texel.
fn bake<F>(tex: &mut [Vec3], n: usize, rows: std::ops::Range<usize>, threads: usize, shader: F)
where
    F: Fn(Vec3) -> Vec3 + Sync,
{
    if rows.is_empty() {
        return;
    }
    let start = rows.start;
    let slice = &mut tex[rows.start * n..rows.end * n];
    let rows_per_thread = rows.len().div_ceil(threads.max(1));
    let shader = &shader;
    let texel_dir = move |row: usize, x: usize| {
        let (face, y) = (row / n, row % n);
        let u = (x as f32 + 0.5) / n as f32 * 2.0 - 1.0;
        let v = (y as f32 + 0.5) / n as f32 * 2.0 - 1.0;
        face_dir(face, u, v)
    };
    if threads <= 1 {
        for (j, texel) in slice.iter_mut().enumerate() {
            *texel = shader(texel_dir(start + j / n, j % n));
        }
        return;
    }
    thread::scope(|s| {
        for (i, chunk) in slice.chunks_mut(rows_per_thread * n).enumerate() {
            s.spawn(move || {
                for (j, texel) in chunk.iter_mut().enumerate() {
                    *texel = shader(texel_dir(start + i * rows_per_thread + j / n, j % n));
                }
            });
        }
    });
}
