use crate::framebuffer::Framebuffer;

/// Dibuja una linea de (x0, y0) a (x1, y1) con el algoritmo de Bresenham.
/// Funciona en todos los octantes (cualquier pendiente y direccion).
/// Se usa para el HUD: orbitas del mapa, marcadores y estelas del warp.
pub fn draw_line(fb: &mut Framebuffer, x0: i32, y0: i32, x1: i32, y1: i32, color: u32) {
    // Lineas absurdamente largas (puntos proyectados muy cerca de la camara) se ignoran
    const LIMIT: i32 = 20_000;
    if x0.abs() > LIMIT || y0.abs() > LIMIT || x1.abs() > LIMIT || y1.abs() > LIMIT {
        return;
    }
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let (mut x, mut y) = (x0, y0);

    loop {
        fb.set(x, y, color);
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}
