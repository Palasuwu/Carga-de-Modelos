use std::fs::File;
use std::io::{BufWriter, Result, Write};

use crate::framebuffer::Framebuffer;

/// Escribe el framebuffer como un archivo BMP de 24 bits.
/// En el framebuffer y = 0 es la fila de arriba (como en la ventana), y el BMP
/// guarda las filas de abajo hacia arriba, asi que se recorren al reves.
pub fn write_bmp_file(fb: &Framebuffer, file_path: &str) -> Result<()> {
    let mut writer = BufWriter::new(File::create(file_path)?);

    let width = fb.width;
    let height = fb.height;
    let row_padding = (4 - (width * 3) % 4) % 4;
    let row_size = width * 3 + row_padding;
    let pixel_data_size = row_size * height;
    let file_size = 54 + pixel_data_size;

    // BMP header (14 bytes)
    writer.write_all(b"BM")?;
    writer.write_all(&(file_size as u32).to_le_bytes())?;
    writer.write_all(&0u32.to_le_bytes())?; // reservado
    writer.write_all(&54u32.to_le_bytes())?; // offset de los datos de pixel

    // DIB header - BITMAPINFOHEADER (40 bytes)
    writer.write_all(&40u32.to_le_bytes())?;
    writer.write_all(&(width as i32).to_le_bytes())?;
    writer.write_all(&(height as i32).to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?; // planos
    writer.write_all(&24u16.to_le_bytes())?; // bits por pixel
    writer.write_all(&0u32.to_le_bytes())?; // sin compresion
    writer.write_all(&(pixel_data_size as u32).to_le_bytes())?;
    writer.write_all(&2835u32.to_le_bytes())?; // resolucion x (72 dpi)
    writer.write_all(&2835u32.to_le_bytes())?; // resolucion y
    writer.write_all(&0u32.to_le_bytes())?; // colores en paleta
    writer.write_all(&0u32.to_le_bytes())?; // colores importantes

    // Datos de pixel: filas de abajo hacia arriba, en orden BGR
    let padding = [0u8; 3];
    for y in (0..height).rev() {
        for x in 0..width {
            let (r, g, b) = fb.get_pixel(x, y);
            writer.write_all(&[b, g, r])?;
        }
        writer.write_all(&padding[..row_padding])?;
    }

    writer.flush()
}
