use nalgebra_glm::Vec3;

use crate::framebuffer::Framebuffer;
use crate::line::draw_line;

/// Dibuja el contorno de un triangulo uniendo sus tres vertices con lineas.
/// Solo se usan x, y (ya en coordenadas de pantalla); z se ignora.
pub fn triangle(fb: &mut Framebuffer, a: &Vec3, b: &Vec3, c: &Vec3) {
    let (ax, ay) = (a.x.round() as i32, a.y.round() as i32);
    let (bx, by) = (b.x.round() as i32, b.y.round() as i32);
    let (cx, cy) = (c.x.round() as i32, c.y.round() as i32);

    draw_line(fb, ax, ay, bx, by);
    draw_line(fb, bx, by, cx, cy);
    draw_line(fb, cx, cy, ax, ay);
}
