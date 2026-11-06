//! Shaders de las galaxias nuevas: planetas raros por su forma o su textura,
//! y el agujero negro que sirve de portal entre galaxias.

use std::f32::consts::{PI, TAU};

use nalgebra_glm::Vec3;

use crate::math::smoothstep;
use crate::noise::{fbm, hash3, mix_bits, perlin, unit, voronoi};
use crate::renderer::{Fragment, Uniforms};
use crate::shaders::{atmosphere, diffuse, lit, mix, rgb, specular, view_dir};

fn reflect(i: &Vec3, n: &Vec3) -> Vec3 {
    i - n * (2.0 * n.dot(i))
}

fn sky(u: &Uniforms, dir: &Vec3) -> Vec3 {
    u.sky.map(|s| s.color(dir, u.time)).unwrap_or_default()
}

fn model_center(u: &Uniforms) -> Vec3 {
    Vec3::new(u.model[(0, 3)], u.model[(1, 3)], u.model[(2, 3)])
}

// ================================================================ Agujero negro

/// Agujero negro con lente gravitacional, dibujado sobre una esfera que lo
/// envuelve. Por cada fragmento se traza un rayo desde la camara (en unidades
/// locales, radio de la esfera = 1):
///  1. Si el rayo toca el disco de acrecion antes de su punto mas cercano al
///     centro, se ve esa parte del disco (la de enfrente).
///  2. Si pasa a menos del horizonte de eventos, la luz no escapa: negro.
///  3. Si no, el rayo se dobla hacia el centro (mas mientras mas cerca pasa,
///     y 0 en el borde de la esfera para que no haya costura) y con ese rayo
///     doblado se busca la parte de atras del disco (que aparece "por encima"
///     del agujero) o se lee el cielo deformado.
///
/// params = [radio del horizonte, radio interno del disco, radio externo, inclinacion]
pub fn black_hole_shader(f: &Fragment, u: &Uniforms) -> Option<Vec3> {
    let center = model_center(u);
    let scale = u.model[(0, 0)];
    let (horizon, inner, outer) = (u.params[0], u.params[1], u.params[2]);
    let (s, c) = u.params[3].sin_cos();
    // Del mundo al marco del disco (rotado sobre X) y de regreso
    let to_disk = |v: Vec3| Vec3::new(v.x, v.y * c + v.z * s, -v.y * s + v.z * c);
    let to_world = |v: Vec3| Vec3::new(v.x, v.y * c - v.z * s, v.y * s + v.z * c);

    let origin = to_disk((u.camera_pos - center) / scale);
    let dir = to_disk((f.world_pos - u.camera_pos).normalize());
    let t_closest = (-origin.dot(&dir)).max(0.0);
    let closest = origin + dir * t_closest;
    let b = closest.norm().max(1e-4);

    let disk = |o: Vec3, d: Vec3, t_max: f32| disk_sample(o, d, t_max, inner, outer, u.time);

    let background = if b < horizon {
        Vec3::zeros()
    } else {
        let bend = (0.085 * (1.0 / b - 1.0)).clamp(0.0, 2.6);
        let toward = -closest / b;
        let bent = (dir * bend.cos() + toward * bend.sin()).normalize();
        // Anillo de fotones: brillo fino justo afuera del horizonte
        let photon_ring = (-(b - horizon * 1.18).abs() * 45.0).exp() * 0.9;
        let behind = sky(u, &to_world(bent)) + rgb(255, 190, 120) * photon_ring;
        match disk(closest, bent, 4.0) {
            Some((color, alpha)) => behind * (1.0 - alpha) + color * alpha,
            None => behind,
        }
    };
    Some(match disk(origin, dir, t_closest) {
        Some((color, alpha)) => background * (1.0 - alpha) + color * alpha,
        None => background,
    })
}

/// Disco de acrecion: interseccion del rayo con el plano y = 0. Gira mas
/// rapido cerca del centro (como una orbita de Kepler), es mas caliente
/// adentro, y el lado que viene hacia la camara se ve mas brillante (Doppler).
fn disk_sample(o: Vec3, d: Vec3, t_max: f32, inner: f32, outer: f32, time: f32) -> Option<(Vec3, f32)> {
    if d.y.abs() < 1e-5 {
        return None;
    }
    let t = -o.y / d.y;
    if t <= 0.0 || t >= t_max {
        return None;
    }
    let p = o + d * t;
    let r = (p.x * p.x + p.z * p.z).sqrt();
    if r < inner || r > outer {
        return None;
    }
    let x = (r - inner) / (outer - inner);
    let angle = p.z.atan2(p.x) - time * 0.9 / r.powf(1.5);
    let swirl = perlin(Vec3::new(angle.cos() * r * 5.0, angle.sin() * r * 5.0, r * 14.0), 101) * 0.5 + 0.5;
    let heat = (1.0 - x).powf(1.3);
    let color = mix(rgb(200, 50, 20), rgb(255, 240, 200), heat) * (0.55 + 0.8 * swirl);
    let orbit_dir = Vec3::new(-p.z, 0.0, p.x) / r;
    let doppler = 1.0 + 0.7 * orbit_dir.dot(&-d);
    let alpha = smoothstep(0.0, 0.12, x) * smoothstep(1.0, 0.6, x) * (0.65 + 0.35 * swirl);
    Some((color * doppler * 1.5, alpha.clamp(0.0, 1.0)))
}

// ================================================================ Galaxia violeta

/// Cubix (cubo redondeado): paneles metalicos con una reticula de neon por
/// cara; pulsos de luz recorren la reticula.
pub fn cubix_shader(f: &Fragment, u: &Uniforms) -> Option<Vec3> {
    let p = f.local_pos;
    let a = p.abs();
    let (face, s, t) = if a.x >= a.y && a.x >= a.z {
        (0, p.y, p.z)
    } else if a.y >= a.z {
        (1, p.x, p.z)
    } else {
        (2, p.x, p.y)
    };
    let (cu, cv) = (s * 5.0, t * 5.0);
    let line = (cu - cu.round()).abs().min((cv - cv.round()).abs());
    let glow = (-line * 28.0).exp();
    let cell = unit(hash3(cu.floor() as i32, cv.floor() as i32, face, 111));
    let panel = mix(rgb(18, 20, 34), rgb(52, 56, 88), cell);
    let pulse = ((cu + cv) * 0.5 - u.time * 3.0).sin().max(0.0).powi(8);
    let neon = mix(rgb(40, 230, 255), rgb(255, 60, 220), (cu * 0.15 + cv * 0.1 + u.time * 0.3).sin() * 0.5 + 0.5);
    Some(lit(panel, f, u) + Vec3::repeat(specular(f, u, 40.0) * 0.4) + neon * glow * (0.45 + 1.6 * pulse))
}

/// Espejo: esfera de cromo que refleja el cielo de la galaxia (el shader
/// del espacio usado como textura) y el brillo de la estrella.
pub fn mirror_shader(f: &Fragment, u: &Uniforms) -> Option<Vec3> {
    let v = view_dir(f, u);
    let r = reflect(&-v, &f.normal);
    let fresnel = 0.6 + 0.4 * (1.0 - f.normal.dot(&v).max(0.0)).powi(2);
    // El cielo es oscuro: se amplifica y se suma un cromo base iluminado
    // por la estrella para que la esfera se lea como metal pulido
    let env = sky(u, &r) * 4.5 * fresnel;
    let chrome = lit(rgb(150, 150, 175), f, u) * 0.45;
    let rim = rgb(170, 140, 255) * (1.0 - f.normal.dot(&v).max(0.0)).powi(4) * 0.8;
    Some(env + chrome + rim + Vec3::repeat(specular(f, u, 300.0) * 2.5))
}

/// Toroide: bandas de colores tipo caramelo que fluyen alrededor del toro.
pub fn torus_candy_shader(f: &Fragment, u: &Uniforms) -> Option<Vec3> {
    let p = f.local_pos;
    let around = p.z.atan2(p.x);
    let tube = p.y.atan2((p.x * p.x + p.z * p.z).sqrt() - 1.0);
    let k = (around * 3.0 / PI + tube / PI - u.time * 0.35).rem_euclid(3.0);
    let palette = [rgb(40, 210, 200), rgb(255, 110, 170), rgb(255, 220, 90)];
    let i = k as usize % 3;
    let edge = smoothstep(0.85, 1.0, k.fract());
    let mut albedo = mix(palette[i], palette[(i + 1) % 3], edge);
    albedo *= 0.9 + 0.1 * fbm(p * 6.0, 2, 121);
    Some(lit(albedo, f, u) + Vec3::repeat(specular(f, u, 30.0) * 0.5))
}

/// Cristal (planeta de picos): facetas violetas; los valles brillan con un
/// pulso y las puntas reflejan la estrella. params[1] = altura de los picos.
pub fn crystal_shader(f: &Fragment, u: &Uniforms) -> Option<Vec3> {
    let k = ((f.local_pos.norm() - 1.0) / u.params[1]).clamp(0.0, 1.0);
    let base = mix(rgb(45, 12, 80), rgb(185, 140, 255), k);
    let pulse = 0.6 + 0.4 * (u.time * 2.0 + f.local_pos.y * 6.0).sin();
    let glow = rgb(190, 70, 255) * (1.0 - k).powi(3) * pulse * 0.9;
    Some(lit(base, f, u) + Vec3::repeat(specular(f, u, 120.0) * 1.2) + glow + atmosphere(f, u, rgb(200, 120, 255), 0.6))
}

// ================================================================ Galaxia esmeralda

/// Cacahuate: selva con rios serpenteantes (cruces por cero del ruido).
pub fn jungle_shader(f: &Fragment, u: &Uniforms) -> Option<Vec3> {
    let p = f.local_pos;
    let n = fbm(p * 2.5, 5, 81);
    let warp = Vec3::new(fbm(p * 1.5, 2, 83), 0.0, fbm(p * 1.5 + Vec3::repeat(4.0), 2, 84));
    let river = 1.0 - smoothstep(0.0, 0.04, perlin(p * 2.2 + warp, 82).abs());
    let mut albedo = mix(rgb(20, 70, 25), rgb(120, 190, 50), n * 0.9 + 0.5);
    albedo = mix(albedo, rgb(90, 70, 40), smoothstep(-0.15, -0.35, n));
    albedo = mix(albedo, rgb(30, 120, 160), river);
    let mut color = lit(albedo, f, u) + Vec3::repeat(specular(f, u, 50.0) * river * 0.5);
    let clouds = smoothstep(0.15, 0.45, fbm(p * 3.0 + Vec3::new(u.time * 0.03, 0.0, 0.0), 3, 85));
    color = mix(color, lit(Vec3::repeat(0.9), f, u), clouds * 0.6);
    Some(color + atmosphere(f, u, rgb(140, 255, 160), 0.5))
}

/// Ojo: un planeta-ojo que siempre mira hacia la camara. El iris se
/// calcula con el angulo entre la normal y la direccion hacia la camara
/// (no con la posicion local), por eso te sigue aunque el planeta gire.
/// La pupila late y cada cierto tiempo parpadea.
pub fn eye_shader(f: &Fragment, u: &Uniforms) -> Option<Vec3> {
    let to_cam = (u.camera_pos - model_center(u)).normalize();
    let n = f.normal;
    let angle = n.dot(&to_cam).clamp(-1.0, 1.0).acos();
    let helper = if to_cam.y.abs() < 0.95 { Vec3::y() } else { Vec3::x() };
    let side = to_cam.cross(&helper).normalize();
    let up = side.cross(&to_cam);

    // Parpado: cada 5 s se cierra durante un instante
    let cycle = (u.time % 5.0) / 5.0;
    let blink = smoothstep(0.92, 0.96, cycle) * smoothstep(1.0, 0.96, cycle);
    let openness = 1.0 - blink;
    if n.dot(&up).abs() > 0.15 + 0.85 * openness && n.dot(&to_cam) > -0.2 {
        let skin = mix(rgb(120, 60, 110), rgb(160, 90, 140), fbm(f.local_pos * 6.0, 2, 131) + 0.5);
        return Some(lit(skin, f, u));
    }

    let iris = 0.45;
    let pupil = 0.14 + 0.04 * (u.time * 0.9).sin();
    let around = n.dot(&up).atan2(n.dot(&side));
    let albedo = if angle < pupil {
        rgb(5, 5, 8)
    } else if angle < iris {
        let fibers = (around * 34.0 + perlin(n * 8.0, 132) * 4.0).sin() * 0.5 + 0.5;
        let k = (angle - pupil) / (iris - pupil);
        let iris_color = mix(rgb(20, 160, 90), rgb(230, 190, 60), k) * (0.6 + 0.4 * fibers);
        mix(iris_color, rgb(10, 40, 25), smoothstep(0.85, 1.0, k))
    } else {
        let veins = 1.0 - smoothstep(0.0, 0.025, perlin(f.local_pos * 5.0, 133).abs());
        mix(rgb(240, 236, 228), rgb(200, 40, 40), veins * smoothstep(iris * 1.6, PI * 0.6, angle))
    };
    Some(lit(albedo, f, u) * 1.1 + Vec3::repeat(specular(f, u, 90.0) * 1.2))
}

/// Burbuja: pelicula delgada iridiscente (el color depende del grosor de la
/// pelicula y del angulo de vision) sobre un reflejo del cielo.
pub fn bubble_shader(f: &Fragment, u: &Uniforms) -> Option<Vec3> {
    let v = view_dir(f, u);
    let ndv = f.normal.dot(&v).max(0.0);
    let film = fbm(f.local_pos * 2.0 + Vec3::new(0.0, u.time * 0.15, 0.0), 3, 91) * 0.5 + 0.5;
    let phase = (1.5 + film * 2.0) * (1.0 + (1.0 - ndv) * 1.5);
    let rainbow = Vec3::new(
        0.5 + 0.5 * (TAU * phase).cos(),
        0.5 + 0.5 * (TAU * (phase + 0.33)).cos(),
        0.5 + 0.5 * (TAU * (phase + 0.67)).cos(),
    );
    let fresnel = 0.15 + 0.85 * (1.0 - ndv).powi(2);
    let env = sky(u, &reflect(&-v, &f.normal));
    Some(env * 0.6 + rainbow * fresnel * 0.9 + Vec3::repeat(specular(f, u, 150.0) * 1.8))
}

/// Coral: oceano con arrecifes y celdas bioluminiscentes (Voronoi) que
/// laten cada una a su ritmo; brillan mas en el lado de noche.
pub fn coral_shader(f: &Fragment, u: &Uniforms) -> Option<Vec3> {
    let p = f.local_pos.normalize();
    let reef = fbm(p * 3.0, 4, 141);
    let albedo = mix(rgb(10, 60, 80), rgb(230, 110, 120), smoothstep(0.05, 0.3, reef));
    let (f1, _, id) = voronoi(p * 6.0, 142);
    let pulse = 0.5 + 0.5 * (u.time * 2.0 + unit(id) * TAU).sin();
    let cell = 1.0 - smoothstep(0.0, 0.35, f1);
    let glow_color = if unit(mix_bits(id)) < 0.5 { rgb(60, 255, 230) } else { rgb(255, 90, 220) };
    let night = 1.0 - diffuse(f, u);
    let glow = glow_color * cell * cell * pulse * (0.25 + 0.75 * night) * 1.3;
    Some(lit(albedo, f, u) + Vec3::repeat(specular(f, u, 40.0) * 0.3) + glow + atmosphere(f, u, rgb(80, 220, 255), 0.5))
}

// ================================================================ Galaxia carmesi

/// Vacio: un planeta hecho de espacio. Usa el mismo shader del cielo como
/// textura, leido con la posicion local (gira con el planeta) y con los ejes
/// cambiados para que no coincida con el fondo; borde brillante.
pub fn void_shader(f: &Fragment, u: &Uniforms) -> Option<Vec3> {
    let p = f.local_pos.normalize();
    let inside = Vec3::new(p.z, p.x, p.y);
    let space = sky(u, &inside) * 2.4;
    let rim = (1.0 - f.normal.dot(&view_dir(f, u)).max(0.0)).powi(2);
    Some(space + rgb(255, 70, 100) * rim * 1.1)
}

/// Tablero: casillas de ajedrez que se tuercen con el tiempo, con juntas
/// doradas.
pub fn checker_shader(f: &Fragment, u: &Uniforms) -> Option<Vec3> {
    let p = f.local_pos.normalize();
    let lat = p.y.asin();
    let lon = p.z.atan2(p.x) + lat * 2.5 * (u.time * 0.4).sin();
    let (cu, cv) = (lon * 8.0 / PI, lat * 8.0 / PI);
    let parity = (cu.floor() as i32 + cv.floor() as i32).rem_euclid(2);
    let seam = (cu - cu.round()).abs().min((cv - cv.round()).abs());
    let mut albedo = if parity == 0 { rgb(232, 220, 200) } else { rgb(28, 20, 32) };
    albedo = mix(rgb(220, 170, 60), albedo, smoothstep(0.02, 0.06, seam));
    Some(lit(albedo, f, u) + Vec3::repeat(specular(f, u, 60.0) * 0.6))
}

/// Erizo (planeta de picos): roca oscura con puntas incandescentes que
/// laten. params[1] = altura de los picos.
pub fn urchin_shader(f: &Fragment, u: &Uniforms) -> Option<Vec3> {
    let k = ((f.local_pos.norm() - 1.0) / u.params[1]).clamp(0.0, 1.0);
    let rock = mix(rgb(95, 60, 55), rgb(175, 120, 95), fbm(f.local_pos * 4.0, 3, 151) + 0.5);
    let tips = smoothstep(0.35, 1.0, k);
    let pulse = 0.65 + 0.35 * (u.time * 3.0 + f.local_pos.x * 8.0).sin();
    let ember = mix(rgb(255, 80, 10), rgb(255, 235, 140), tips) * tips * pulse * 2.0;
    let cracks = rgb(255, 60, 0) * (1.0 - k).powi(6) * 0.35 * pulse;
    Some(lit(rock, f, u) * (1.0 - tips * 0.5) + ember + cracks + atmosphere(f, u, rgb(255, 120, 60), 0.5))
}

/// Halo: anillo de plasma ardiente que fluye alrededor del toro (emisivo).
pub fn halo_shader(f: &Fragment, u: &Uniforms) -> Option<Vec3> {
    let p = f.local_pos;
    let (s, c) = (u.time * 0.5).sin_cos();
    let q = Vec3::new(p.x * c - p.z * s, p.y, p.x * s + p.z * c);
    let around = p.z.atan2(p.x);
    let heat = (fbm(q * 3.0, 4, 161) * 0.7 + 0.5 + 0.15 * (u.time * 2.0 + around * 5.0).sin()).clamp(0.0, 1.0);
    let color = if heat < 0.5 {
        mix(rgb(60, 5, 5), rgb(240, 80, 20), heat * 2.0)
    } else {
        mix(rgb(240, 80, 20), rgb(255, 235, 160), heat * 2.0 - 1.0)
    };
    Some(color * 1.2 + lit(rgb(40, 10, 10), f, u))
}
