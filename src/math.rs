//! Matrices de vista, proyeccion y viewport escritas a mano, mas utilidades.

use nalgebra_glm::{Mat4, Vec3};

/// Matriz de vista "look at": coloca la camara en `eye` mirando a `target`.
/// Construye una base ortonormal (s = derecha, u = arriba, f = adelante) y
/// la usa para pasar de coordenadas de mundo a coordenadas de camara.
pub fn look_at(eye: &Vec3, target: &Vec3, up: &Vec3) -> Mat4 {
    let f = (target - eye).normalize();
    let s = f.cross(up).normalize();
    let u = s.cross(&f);
    Mat4::new(
        s.x, s.y, s.z, -s.dot(eye),
        u.x, u.y, u.z, -u.dot(eye),
        -f.x, -f.y, -f.z, f.dot(eye),
        0.0, 0.0, 0.0, 1.0,
    )
}

/// Proyeccion en perspectiva (convencion OpenGL: la camara mira hacia -Z y
/// el volumen visible queda en -w <= x, y, z <= w).
pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov_y * 0.5).tan();
    Mat4::new(
        f / aspect, 0.0, 0.0, 0.0,
        0.0, f, 0.0, 0.0,
        0.0, 0.0, (far + near) / (near - far), 2.0 * far * near / (near - far),
        0.0, 0.0, -1.0, 0.0,
    )
}

/// Viewport: de coordenadas normalizadas (-1..1) a pixeles. La y se invierte
/// porque en la ventana crece hacia abajo.
pub fn viewport(width: f32, height: f32) -> Mat4 {
    Mat4::new(
        width * 0.5, 0.0, 0.0, width * 0.5,
        0.0, -height * 0.5, 0.0, height * 0.5,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
}

#[inline]
pub fn lerp3(a: &Vec3, b: &Vec3, t: f32) -> Vec3 {
    a + (b - a) * t
}

#[inline]
pub fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Factor de suavizado independiente de los FPS: con `rate` alto se acerca
/// mas rapido al objetivo. Uso: x += (objetivo - x) * approach(rate, dt).
#[inline]
pub fn approach(rate: f32, dt: f32) -> f32 {
    1.0 - (-rate * dt).exp()
}

/// Diferencia angular en (-PI, PI].
pub fn angle_diff(target: f32, current: f32) -> f32 {
    let mut d = (target - current) % std::f32::consts::TAU;
    if d > std::f32::consts::PI {
        d -= std::f32::consts::TAU;
    } else if d < -std::f32::consts::PI {
        d += std::f32::consts::TAU;
    }
    d
}

pub fn ease_in_cubic(t: f32) -> f32 {
    t * t * t
}

pub fn ease_out_cubic(t: f32) -> f32 {
    1.0 - (1.0 - t).powi(3)
}
