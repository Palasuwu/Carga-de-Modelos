//! Fragment shaders. Cada uno es una funcion con la firma `FragmentShader`
//! y se asigna a un objeto de la escena como "shader activo".
//!
//! Todos los patrones se calculan con `local_pos` (posicion sobre la esfera
//! unitaria del modelo), asi que giran junto con el planeta.

use nalgebra_glm::Vec3;

use crate::math::smoothstep;
use crate::noise::{fbm, mix_bits, perlin, unit, voronoi};
use crate::renderer::{Fragment, Uniforms};

pub(crate) const AMBIENT: f32 = 0.04;

#[inline]
pub(crate) fn rgb(r: u8, g: u8, b: u8) -> Vec3 {
    Vec3::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0)
}

#[inline]
pub(crate) fn mix(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

#[inline]
pub(crate) fn light_dir(f: &Fragment, u: &Uniforms) -> Vec3 {
    (u.light_pos - f.world_pos).normalize()
}

#[inline]
pub(crate) fn view_dir(f: &Fragment, u: &Uniforms) -> Vec3 {
    (u.camera_pos - f.world_pos).normalize()
}

/// Lambert: cuanta luz del sol recibe la superficie.
#[inline]
pub(crate) fn diffuse(f: &Fragment, u: &Uniforms) -> f32 {
    f.normal.dot(&light_dir(f, u)).max(0.0)
}

#[inline]
pub(crate) fn lit(albedo: Vec3, f: &Fragment, u: &Uniforms) -> Vec3 {
    albedo * (AMBIENT + diffuse(f, u) * (1.0 - AMBIENT))
}

/// Brillo especular de Blinn-Phong.
#[inline]
pub(crate) fn specular(f: &Fragment, u: &Uniforms, power: f32) -> f32 {
    let l = light_dir(f, u);
    if f.normal.dot(&l) <= 0.0 {
        return 0.0;
    }
    let h = (l + view_dir(f, u)).normalize();
    f.normal.dot(&h).max(0.0).powf(power)
}

/// Borde iluminado tipo atmosfera (fresnel), solo del lado de dia.
#[inline]
pub(crate) fn atmosphere(f: &Fragment, u: &Uniforms, color: Vec3, strength: f32) -> Vec3 {
    let rim = (1.0 - f.normal.dot(&view_dir(f, u)).max(0.0)).powi(3);
    let day = smoothstep(-0.25, 0.4, f.normal.dot(&light_dir(f, u)));
    color * rim * day * strength
}

// ---------------------------------------------------------------- Cielo

/// Esfera gigante del fondo: nebulosa + estrellas, sin iluminacion.
/// params[0] = intensidad del efecto de velocidad (turbo),
/// params[1..4] = direccion hacia donde mira la camara.
pub fn sky_shader(f: &Fragment, u: &Uniforms) -> Option<Vec3> {
    let dir = f.local_pos.normalize();
    let forward = Vec3::new(u.params[1], u.params[2], u.params[3]);
    Some(u.sky.map(|s| s.color_warped(&dir, u.time, &forward, u.params[0])).unwrap_or_default())
}

// ---------------------------------------------------------------- Sol

/// Superficie de plasma animada con dos capas de fBm que se desplazan en el
/// tiempo, oscurecimiento hacia el borde (limb darkening). Es emisivo.
/// params[0] = velocidad de la animacion, params[1] = paleta.
pub fn sun_shader(f: &Fragment, u: &Uniforms) -> Option<Vec3> {
    let p = f.local_pos.normalize();
    let t = u.time * u.params[0];
    let n1 = fbm(p * 2.5 + Vec3::new(t * 0.12, t * 0.05, 0.0), 4, 11);
    let n2 = fbm(p * 7.0 - Vec3::new(0.0, t * 0.3, t * 0.12), 3, 12);
    let heat = (0.55 + n1 * 1.1 + n2 * 0.45).clamp(0.0, 1.0);

    // params[1] elige la paleta: 0 gigante roja, 1 amarilla, 2 azul, 3 verde
    let [cool, mid, hot] = match u.params[1] as i32 {
        0 => [rgb(110, 8, 4), rgb(225, 50, 20), rgb(255, 150, 90)],
        2 => [rgb(30, 60, 200), rgb(110, 165, 255), rgb(230, 245, 255)],
        3 => [rgb(10, 110, 60), rgb(110, 225, 140), rgb(235, 255, 220)],
        _ => [rgb(200, 40, 10), rgb(255, 140, 20), rgb(255, 235, 150)],
    };
    let color = if heat < 0.5 { mix(cool, mid, heat * 2.0) } else { mix(mid, hot, heat * 2.0 - 1.0) };

    let mu = f.normal.dot(&view_dir(f, u)).max(0.0);
    let limb = 0.55 + 0.45 * mu.sqrt();
    Some(color * limb * 1.2 + mid * (1.0 - mu).powi(3) * 0.6)
}

// ---------------------------------------------------------------- Magma

/// Roca oscura con grietas de lava: las grietas salen de la "cresta" del
/// ruido de Perlin (1 - |n|), y la lava pulsa con el tiempo. La lava es
/// emisiva, por eso se ve tambien en el lado de noche.
/// params[0] = intensidad del brillo de la lava.
pub fn magma_shader(f: &Fragment, u: &Uniforms) -> Option<Vec3> {
    let p = f.local_pos.normalize();
    let rock_n = fbm(p * 3.0, 4, 21);
    let rock = mix(rgb(38, 28, 26), rgb(95, 72, 60), rock_n * 0.8 + 0.5);

    let warp = Vec3::new(fbm(p * 2.0, 2, 24), 0.0, fbm(p * 2.0 + Vec3::repeat(5.0), 2, 25)) * 0.6;
    let ridge = 1.0 - perlin((p + warp) * 4.0, 22).abs();
    let ridge2 = 1.0 - perlin(p * 9.0, 23).abs();
    // Las grietas se adelgazan a lo lejos para que no se vean como ruido
    let dist = (u.camera_pos - f.world_pos).norm();
    let thin = smoothstep(40.0, 200.0, dist) * 0.05;
    let crack = smoothstep(0.92 + thin, 0.985, ridge).max(smoothstep(0.95 + thin, 0.995, ridge2) * 0.5);

    let pulse = 0.7 + 0.3 * (u.time * 2.2 + rock_n * 9.0).sin();
    let lava = mix(rgb(255, 50, 0), rgb(255, 210, 60), crack) * crack * pulse * u.params[0];
    Some(lit(rock, f, u) * (1.0 - crack * 0.7) + lava)
}

// ---------------------------------------------------------------- Verdalia

/// Planeta tipo Tierra: oceanos y continentes por umbral de fBm, casquetes
/// polares, nubes animadas, brillo especular solo en el agua y atmosfera.
pub fn verdalia_shader(f: &Fragment, u: &Uniforms) -> Option<Vec3> {
    let p = f.local_pos.normalize();
    let h = fbm(p * 1.8, 5, 31);
    let ocean = h < 0.02;

    let mut albedo = if ocean {
        mix(rgb(8, 30, 90), rgb(30, 110, 170), smoothstep(-0.3, 0.02, h))
    } else {
        let land = mix(rgb(60, 130, 55), rgb(140, 120, 70), smoothstep(0.06, 0.28, h));
        mix(land, rgb(225, 225, 230), smoothstep(0.33, 0.4, h))
    };
    let polar = p.y.abs() + fbm(p * 6.0, 2, 32) * 0.12;
    albedo = mix(albedo, rgb(235, 240, 250), smoothstep(0.8, 0.86, polar));

    let mut color = lit(albedo, f, u);
    if ocean {
        color += Vec3::repeat(specular(f, u, 40.0) * 0.6);
    }
    let drift = Vec3::new(u.time * 0.025, 0.0, u.time * 0.01);
    let clouds = smoothstep(0.08, 0.38, fbm(p * 2.8 + drift, 4, 33));
    color = mix(color, lit(Vec3::repeat(0.95), f, u), clouds * 0.85);
    Some(color + atmosphere(f, u, rgb(90, 160, 255), 0.8))
}

// ---------------------------------------------------------------- Duna

/// Desierto de mesetas: la altura de fBm se "terracea" (floor) en niveles,
/// cada nivel tiene su color, los bordes de los escalones son acantilados
/// oscuros y hay ondulaciones de dunas.
pub fn duna_shader(f: &Fragment, u: &Uniforms) -> Option<Vec3> {
    let p = f.local_pos.normalize();
    let h = fbm(p * 2.2, 5, 41) * 0.9 + 0.5;
    let steps = 6.0;
    let level = (h * steps).floor() / steps;
    let frac = h * steps - (h * steps).floor();

    let palette = [rgb(230, 185, 120), rgb(215, 150, 90), rgb(190, 110, 60), rgb(150, 75, 45), rgb(110, 55, 40)];
    let idx = ((level * palette.len() as f32) as usize).min(palette.len() - 1);
    let mut albedo = palette[idx];

    let ripples = (p.y * 70.0 + perlin(p * 6.0, 42) * 9.0).sin() * 0.05;
    albedo *= 1.0 + ripples;
    albedo *= 1.0 - smoothstep(0.82, 0.98, frac) * 0.45; // acantilados
    Some(lit(albedo, f, u) + atmosphere(f, u, rgb(255, 170, 110), 0.35))
}

// ---------------------------------------------------------------- Bandus

/// Gigante gaseoso: bandas por latitud deformadas con turbulencia, rotacion
/// diferencial animada (cada latitud gira a distinta velocidad) y una
/// tormenta ovalada.
pub fn bandus_shader(f: &Fragment, u: &Uniforms) -> Option<Vec3> {
    let p = f.local_pos.normalize();
    let lat = p.y;
    let angle = u.time * 0.06 * (1.0 + 0.6 * (lat * 7.0).sin());
    let (s, c) = angle.sin_cos();
    let q = Vec3::new(p.x * c - p.z * s, p.y, p.x * s + p.z * c);

    let turb = fbm(Vec3::new(q.x * 1.5, q.y * 5.0, q.z * 1.5), 4, 51);
    let b1 = (lat * 9.0 + turb * 1.8).sin() * 0.5 + 0.5;
    let b2 = (lat * 23.0 + turb * 3.5).sin() * 0.5 + 0.5;
    let mut albedo = mix(rgb(235, 215, 175), rgb(190, 130, 80), b1);
    albedo = mix(albedo, rgb(120, 75, 50), b2 * 0.35);

    let storm_center = Vec3::new(0.75, -0.3, 0.59).normalize();
    let d = (p - storm_center).norm();
    let storm = 1.0 - smoothstep(0.12, 0.24, d + turb * 0.05);
    albedo = mix(albedo, rgb(200, 80, 50), storm * (0.7 + 0.3 * (d * 60.0 - u.time * 2.0).sin()));

    Some(lit(albedo, f, u) + atmosphere(f, u, rgb(255, 220, 170), 0.4))
}

/// Anillos del gigante gaseoso: densidad por radio con huecos descartados
/// (el fragmento no se dibuja) y sombra del planeta calculada con un rayo
/// hacia el sol. params[0] = radio del planeta en el mundo.
pub fn ring_shader(f: &Fragment, u: &Uniforms) -> Option<Vec3> {
    let r = (f.local_pos.x * f.local_pos.x + f.local_pos.z * f.local_pos.z).sqrt();
    let x = (r - 1.35) / (2.3 - 1.35);
    if !(0.0..=1.0).contains(&x) || (0.56..0.62).contains(&x) {
        return None;
    }
    let density = perlin(Vec3::new(r * 22.0, 0.5, 0.5), 52) * 0.5 + 0.5 + (r * 95.0).sin() * 0.15;
    if density < 0.3 {
        return None;
    }
    let albedo = mix(rgb(205, 185, 150), rgb(140, 120, 100), perlin(Vec3::new(r * 60.0, 1.5, 0.5), 53) + 0.5);

    let l = light_dir(f, u);
    let mut light = 0.25 + 0.75 * f.normal.dot(&l).abs();
    // Sombra del planeta sobre el anillo (interseccion rayo-esfera)
    let center = Vec3::new(u.model[(0, 3)], u.model[(1, 3)], u.model[(2, 3)]);
    let oc = f.world_pos - center;
    let b = oc.dot(&l);
    let c = oc.norm_squared() - u.params[0] * u.params[0];
    if b < 0.0 && b * b - c > 0.0 {
        light *= 0.12;
    }
    let edge_fade = smoothstep(0.0, 0.08, x) * smoothstep(1.0, 0.9, x);
    Some(albedo * (AMBIENT + light) * (0.5 + 0.5 * edge_fade))
}

// ---------------------------------------------------------------- Glacia

/// Planeta de hielo: celdas de Voronoi como placas, grietas donde la
/// distancia al segundo punto es casi igual a la del primero, brillo
/// especular fuerte y borde fresnel azulado.
pub fn glacia_shader(f: &Fragment, u: &Uniforms) -> Option<Vec3> {
    let p = f.local_pos.normalize();
    let (f1, f2, id) = voronoi(p * 5.0, 61);
    let crack = 1.0 - smoothstep(0.02, 0.08, f2 - f1);
    let tint = unit(mix_bits(id));
    let mut albedo = mix(rgb(210, 235, 248), rgb(150, 200, 232), tint);
    albedo *= 0.92 + 0.08 * fbm(p * 12.0, 2, 62);
    albedo = mix(albedo, rgb(25, 70, 120), crack * 0.85);

    let spec = specular(f, u, 80.0) * (1.0 - crack);
    Some(lit(albedo, f, u) * 0.92 + Vec3::repeat(spec * 0.45) + atmosphere(f, u, rgb(140, 220, 255), 0.6))
}

// ---------------------------------------------------------------- Luna

/// Luna gris con crateres: cada celda de Voronoi tiene un crater de radio
/// aleatorio, con el fondo oscuro y el borde mas claro.
pub fn moon_shader(f: &Fragment, u: &Uniforms) -> Option<Vec3> {
    let p = f.local_pos.normalize();
    let base = 0.5 + 0.15 * fbm(p * 5.0, 4, 71);
    let (d, _, id) = voronoi(p * 4.0, 72);
    let radius = 0.2 + 0.25 * unit(mix_bits(id));
    let floor = 1.0 - smoothstep(radius * 0.7, radius, d);
    let rim = smoothstep(radius * 0.75, radius, d) * (1.0 - smoothstep(radius, radius * 1.25, d));
    let g = base * (1.0 - floor * 0.3) + rim * 0.18;
    Some(lit(Vec3::new(g, g * 0.98, g * 0.95), f, u))
}

// ---------------------------------------------------------------- Nave

/// Nave: color del material del .obj con Blinn-Phong, una luz de borde
/// para que la silueta se lea aun de noche, y motores emisivos que brillan
/// mas cuando se acelera. params[0] = empuje (0..1), params[1] = turbo.
pub fn ship_shader(f: &Fragment, u: &Uniforms) -> Option<Vec3> {
    let n = f.normal;
    let v = view_dir(f, u);
    let base = f.color;
    if f.extra > 0.5 {
        let flicker = 0.9 + 0.1 * (u.time * 40.0).sin();
        let power = 0.9 + u.params[0] * 0.8 + u.params[1] * 0.8;
        return Some((base + Vec3::new(0.3, 0.1, 0.3) * u.params[0]) * power * flicker);
    }
    let rim = (1.0 - n.dot(&v).abs()).powi(3);
    let color = base * (0.18 + 0.82 * diffuse(f, u))
        + Vec3::repeat(specular(f, u, 48.0) * 0.6)
        + rgb(90, 200, 255) * rim * 0.35;
    Some(color)
}
