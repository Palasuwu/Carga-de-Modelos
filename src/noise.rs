//! Ruido procedural propio: hash entero, ruido de Perlin 3D, fBm y Voronoi.
//! Todo es deterministico: la misma semilla da siempre el mismo resultado.

use nalgebra_glm::Vec3;

#[inline]
pub fn mix_bits(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^= h >> 16;
    h
}

#[inline]
pub fn hash3(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    mix_bits(
        (x as u32).wrapping_mul(0x8da6_b343)
            ^ (y as u32).wrapping_mul(0xd816_3841)
            ^ (z as u32).wrapping_mul(0xcb1a_b31f)
            ^ seed.wrapping_mul(0x9e37_79b9),
    )
}

/// Convierte un hash a flotante en [0, 1).
#[inline]
pub fn unit(h: u32) -> f32 {
    (h >> 8) as f32 / 16_777_216.0
}

#[inline]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Producto punto con uno de los 12 gradientes clasicos de Perlin.
#[inline]
fn grad(h: u32, x: f32, y: f32, z: f32) -> f32 {
    match h & 15 {
        0 | 12 => x + y,
        1 | 13 => -x + y,
        2 => x - y,
        3 => -x - y,
        4 => x + z,
        5 => -x + z,
        6 => x - z,
        7 => -x - z,
        8 => y + z,
        9 | 14 => -y + z,
        10 => y - z,
        _ => -y - z,
    }
}

/// Ruido de Perlin 3D, resultado aproximado en [-1, 1].
pub fn perlin(p: Vec3, seed: u32) -> f32 {
    let (fx, fy, fz) = (p.x.floor(), p.y.floor(), p.z.floor());
    let (x0, y0, z0) = (fx as i32, fy as i32, fz as i32);
    let (x, y, z) = (p.x - fx, p.y - fy, p.z - fz);
    let (u, v, w) = (fade(x), fade(y), fade(z));

    let g = |i: i32, j: i32, k: i32| {
        grad(hash3(x0 + i, y0 + j, z0 + k, seed), x - i as f32, y - j as f32, z - k as f32)
    };
    let l = |a: f32, b: f32, t: f32| a + (b - a) * t;

    l(
        l(l(g(0, 0, 0), g(1, 0, 0), u), l(g(0, 1, 0), g(1, 1, 0), u), v),
        l(l(g(0, 0, 1), g(1, 0, 1), u), l(g(0, 1, 1), g(1, 1, 1), u), v),
        w,
    )
}

/// Fractal Brownian motion: suma de octavas de Perlin, cada una con el doble
/// de frecuencia y la mitad de amplitud. Resultado aprox. en [-1, 1].
pub fn fbm(p: Vec3, octaves: u32, seed: u32) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut norm = 0.0;
    let mut q = p;
    for o in 0..octaves {
        sum += perlin(q, seed.wrapping_add(o * 101)) * amp;
        norm += amp;
        amp *= 0.5;
        q *= 2.03;
    }
    sum / norm
}

/// Ruido celular (Voronoi / Worley) 3D. Devuelve la distancia al punto
/// caracteristico mas cercano (f1), al segundo (f2) y un id de la celda.
pub fn voronoi(p: Vec3, seed: u32) -> (f32, f32, u32) {
    let (cx, cy, cz) = (p.x.floor() as i32, p.y.floor() as i32, p.z.floor() as i32);
    let mut f1 = f32::MAX;
    let mut f2 = f32::MAX;
    let mut id = 0;
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (i, j, k) = (cx + dx, cy + dy, cz + dz);
                let h = hash3(i, j, k, seed);
                let jitter = Vec3::new(unit(h), unit(mix_bits(h ^ 0x68e3_1da4)), unit(mix_bits(h ^ 0xb529_7a4d)));
                let feature = Vec3::new(i as f32, j as f32, k as f32) + jitter;
                let d = (feature - p).norm_squared();
                if d < f1 {
                    f2 = f1;
                    f1 = d;
                    id = h;
                } else if d < f2 {
                    f2 = d;
                }
            }
        }
    }
    (f1.sqrt(), f2.sqrt(), id)
}
