pub type Color = (u8, u8, u8);

pub struct Framebuffer {
    pub width: usize,
    pub height: usize,
    buffer: Vec<Color>,
    background_color: Color,
    current_color: Color,
}

impl Framebuffer {
    pub fn new(width: usize, height: usize) -> Self {
        let background_color = (0, 0, 0);
        Framebuffer {
            width,
            height,
            buffer: vec![background_color; width * height],
            background_color,
            current_color: (255, 255, 255),
        }
    }

    pub fn clear(&mut self) {
        self.buffer.fill(self.background_color);
    }

    pub fn set_background_color(&mut self, color: Color) {
        self.background_color = color;
    }

    pub fn set_current_color(&mut self, color: Color) {
        self.current_color = color;
    }

    /// Pinta el pixel (x, y) con el color actual. Ignora coordenadas fuera del buffer.
    pub fn point(&mut self, x: i32, y: i32) {
        if x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height {
            self.buffer[y as usize * self.width + x as usize] = self.current_color;
        }
    }

    pub fn get_pixel(&self, x: usize, y: usize) -> Color {
        self.buffer[y * self.width + x]
    }

    /// Convierte el buffer al formato 0RGB que usa la ventana de minifb.
    pub fn to_u32_buffer(&self) -> Vec<u32> {
        self.buffer
            .iter()
            .map(|&(r, g, b)| ((r as u32) << 16) | ((g as u32) << 8) | b as u32)
            .collect()
    }
}
