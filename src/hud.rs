//! Interfaz 2D dibujada directamente en el framebuffer: texto con una fuente
//! bitmap de 5x7, paneles semitransparentes, circulos y la transicion de iris.

use nalgebra_glm::Vec3;

use crate::framebuffer::{Framebuffer, pack_color};
use crate::line::draw_line;
use crate::noise::{hash3, unit};

fn glyph(c: char) -> Option<[&'static str; 7]> {
    Some(match c {
        'A' => [" ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"],
        'B' => ["#### ", "#   #", "#   #", "#### ", "#   #", "#   #", "#### "],
        'C' => [" ### ", "#   #", "#    ", "#    ", "#    ", "#   #", " ### "],
        'D' => ["#### ", "#   #", "#   #", "#   #", "#   #", "#   #", "#### "],
        'E' => ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#####"],
        'F' => ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#    "],
        'G' => [" ### ", "#   #", "#    ", "# ###", "#   #", "#   #", " ####"],
        'H' => ["#   #", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"],
        'I' => [" ### ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "],
        'J' => ["  ###", "   # ", "   # ", "   # ", "   # ", "#  # ", " ##  "],
        'K' => ["#   #", "#  # ", "# #  ", "##   ", "# #  ", "#  # ", "#   #"],
        'L' => ["#    ", "#    ", "#    ", "#    ", "#    ", "#    ", "#####"],
        'M' => ["#   #", "## ##", "# # #", "# # #", "#   #", "#   #", "#   #"],
        'N' => ["#   #", "#   #", "##  #", "# # #", "#  ##", "#   #", "#   #"],
        'O' => [" ### ", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "],
        'P' => ["#### ", "#   #", "#   #", "#### ", "#    ", "#    ", "#    "],
        'Q' => [" ### ", "#   #", "#   #", "#   #", "# # #", "#  # ", " ## #"],
        'R' => ["#### ", "#   #", "#   #", "#### ", "# #  ", "#  # ", "#   #"],
        'S' => [" ####", "#    ", "#    ", " ### ", "    #", "    #", "#### "],
        'T' => ["#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "],
        'U' => ["#   #", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "],
        'V' => ["#   #", "#   #", "#   #", "#   #", "#   #", " # # ", "  #  "],
        'W' => ["#   #", "#   #", "#   #", "# # #", "# # #", "# # #", " # # "],
        'X' => ["#   #", "#   #", " # # ", "  #  ", " # # ", "#   #", "#   #"],
        'Y' => ["#   #", "#   #", " # # ", "  #  ", "  #  ", "  #  ", "  #  "],
        'Z' => ["#####", "    #", "   # ", "  #  ", " #   ", "#    ", "#####"],
        '0' => [" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "],
        '1' => ["  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "],
        '2' => [" ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####"],
        '3' => ["#####", "   # ", "  #  ", "   # ", "    #", "#   #", " ### "],
        '4' => ["   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # "],
        '5' => ["#####", "#    ", "#### ", "    #", "    #", "#   #", " ### "],
        '6' => ["  ## ", " #   ", "#    ", "#### ", "#   #", "#   #", " ### "],
        '7' => ["#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   "],
        '8' => [" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "],
        '9' => [" ### ", "#   #", "#   #", " ####", "    #", "   # ", " ##  "],
        '.' => ["     ", "     ", "     ", "     ", "     ", "     ", "  #  "],
        ',' => ["     ", "     ", "     ", "     ", "     ", "  #  ", " #   "],
        ':' => ["     ", "     ", "  #  ", "     ", "  #  ", "     ", "     "],
        '-' => ["     ", "     ", "     ", " ### ", "     ", "     ", "     "],
        '+' => ["     ", "  #  ", "  #  ", "#####", "  #  ", "  #  ", "     "],
        '=' => ["     ", "     ", "#####", "     ", "#####", "     ", "     "],
        '/' => ["    #", "    #", "   # ", "  #  ", " #   ", "#    ", "#    "],
        '(' => ["   # ", "  #  ", " #   ", " #   ", " #   ", "  #  ", "   # "],
        ')' => [" #   ", "  #  ", "   # ", "   # ", "   # ", "  #  ", " #   "],
        '[' => [" ### ", " #   ", " #   ", " #   ", " #   ", " #   ", " ### "],
        ']' => [" ### ", "   # ", "   # ", "   # ", "   # ", "   # ", " ### "],
        '%' => ["##   ", "##  #", "   # ", "  #  ", " #   ", "#  ##", "   ##"],
        '!' => ["  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "     ", "  #  "],
        '?' => [" ### ", "#   #", "    #", "   # ", "  #  ", "     ", "  #  "],
        '>' => [" #   ", "  #  ", "   # ", "    #", "   # ", "  #  ", " #   "],
        '<' => ["   # ", "  #  ", " #   ", "#    ", " #   ", "  #  ", "   # "],
        '|' => ["  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "],
        '\'' => ["  #  ", "  #  ", "     ", "     ", "     ", "     ", "     "],
        _ => return None,
    })
}

pub fn text_width(text: &str, scale: i32) -> i32 {
    text.chars().count() as i32 * 6 * scale
}

pub fn draw_text(fb: &mut Framebuffer, x: i32, y: i32, text: &str, scale: i32, color: u32) {
    let mut cx = x;
    for ch in text.chars() {
        // Sin acentos en la fuente: se dibuja la letra base
        let ch = match ch.to_ascii_uppercase() {
            'Á' | 'á' => 'A',
            'É' | 'é' => 'E',
            'Í' | 'í' => 'I',
            'Ó' | 'ó' => 'O',
            'Ú' | 'ú' => 'U',
            'Ñ' | 'ñ' => 'N',
            c => c,
        };
        if let Some(rows) = glyph(ch) {
            for (ry, row) in rows.iter().enumerate() {
                for (rx, b) in row.bytes().enumerate() {
                    if b == b'#' {
                        for sy in 0..scale {
                            for sx in 0..scale {
                                fb.set(cx + rx as i32 * scale + sx, y + ry as i32 * scale + sy, color);
                            }
                        }
                    }
                }
            }
        }
        cx += 6 * scale;
    }
}

/// Texto con sombra para que se lea sobre cualquier fondo.
pub fn draw_text_shadow(fb: &mut Framebuffer, x: i32, y: i32, text: &str, scale: i32, color: u32) {
    draw_text(fb, x + scale, y + scale, text, scale, 0x000000);
    draw_text(fb, x, y, text, scale, color);
}

pub fn fill_rect(fb: &mut Framebuffer, x: i32, y: i32, w: i32, h: i32, color: Vec3, alpha: f32) {
    for py in y.max(0)..(y + h).min(fb.height as i32) {
        for px in x.max(0)..(x + w).min(fb.width as i32) {
            fb.blend(px, py, color, alpha);
        }
    }
}

pub fn draw_circle(fb: &mut Framebuffer, cx: f32, cy: f32, r: f32, color: u32) {
    let segments = ((r * 0.8) as i32).clamp(12, 96);
    let mut prev = (cx + r, cy);
    for i in 1..=segments {
        let a = i as f32 / segments as f32 * std::f32::consts::TAU;
        let p = (cx + r * a.cos(), cy + r * a.sin());
        draw_line(fb, prev.0 as i32, prev.1 as i32, p.0 as i32, p.1 as i32, color);
        prev = p;
    }
}

/// Estelas radiales de "salto": lineas que salen del centro y se alargan
/// conforme avanza `t` (0..1).
pub fn warp_streaks(fb: &mut Framebuffer, cx: f32, cy: f32, t: f32) {
    let max_r = ((fb.width * fb.width + fb.height * fb.height) as f32).sqrt();
    for i in 0..140 {
        let a = unit(hash3(i, 0, 0, 91)) * std::f32::consts::TAU;
        let start = (0.1 + 0.9 * unit(hash3(i, 1, 0, 91))) * max_r * 0.6 * (1.0 - t * 0.5);
        let len = t * t * (60.0 + 260.0 * unit(hash3(i, 2, 0, 91)));
        let (s, c) = a.sin_cos();
        let color = if i % 3 == 0 { pack_color(&Vec3::new(0.7, 0.85, 1.0)) } else { 0xffffff };
        draw_line(
            fb,
            (cx + c * start) as i32,
            (cy + s * start) as i32,
            (cx + c * (start + len)) as i32,
            (cy + s * (start + len)) as i32,
            color,
        );
    }
}

/// Transicion de iris: todo lo que queda fuera del circulo de radio
/// `radius` se pinta de negro, con un borde de color animado.
pub fn iris(fb: &mut Framebuffer, cx: f32, cy: f32, radius: f32, time: f32) {
    let rim = 7.0;
    let (w, h) = (fb.width, fb.height);
    for y in 0..h {
        let dy = y as f32 + 0.5 - cy;
        for x in 0..w {
            let dx = x as f32 + 0.5 - cx;
            let d2 = dx * dx + dy * dy;
            if d2 < (radius - rim).max(0.0).powi(2) {
                continue;
            }
            let i = y * w + x;
            let d = d2.sqrt();
            if d > radius {
                fb.color[i] = 0x05030a;
            } else {
                let a = dy.atan2(dx) + time * 3.0;
                let k = 0.5 + 0.5 * (a * 3.0).sin();
                let c = Vec3::new(0.2, 0.9, 1.0) * (1.0 - k) + Vec3::new(1.0, 0.3, 0.8) * k;
                fb.color[i] = pack_color(&c);
            }
        }
    }
}
