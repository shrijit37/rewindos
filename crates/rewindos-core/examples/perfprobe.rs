// Throwaway probe: measure per-stage CPU cost of the capture hash/save path.
// Not part of the build. Delete after use.
use std::time::Instant;

use image::{DynamicImage, RgbaImage};
use rewindos_core::hasher::{self, PerceptualHasher};

fn bench<T>(label: &str, iters: u32, mut f: impl FnMut() -> T) {
    // warmup
    f();
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    let per = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("{label:<46} {per:>8.1} ms");
}

fn main() {
    let src = std::env::args()
        .nth(1)
        .unwrap_or_else(|| panic!("usage: perfprobe <screenshot.webp>"));

    let decoded = image::open(&src).expect("decode");
    let rgba = decoded.to_rgba8();
    println!("source: {}x{}", rgba.width(), rgba.height());
    // Simulate the PipeWire RGBx buffer at native 2560x1600 capture size.
    // Triangle, not Nearest: nearest-upscaled content is blocky noise and
    // inflates Lanczos3/encode cost well above real screen content.
    let big: RgbaImage =
        image::imageops::resize(&rgba, 2560, 1600, image::imageops::FilterType::Triangle);

    let hasher = PerceptualHasher::new();
    let pixels = big.as_raw();
    println!("frame: {}x{} RGBA ({} MB)", big.width(), big.height(), pixels.len() / 1_000_000);

    bench("image_from_rgba (copy into DynamicImage)", 20, || {
        hasher::image_from_rgba(pixels, 2560, 1600).unwrap()
    });

    let img = hasher::image_from_rgba(pixels, 2560, 1600).unwrap();
    bench("hash_image @2560x1600 (full res)", 20, || {
        hasher.hash_image(&img)
    });

    let down = hasher::downscale_for_storage(&img, 1920);
    bench("hash_image @1920x1200 (post-downscale)", 20, || {
        hasher.hash_image(&down)
    });

    bench("downscale_for_storage Lanczos3 ->1920", 20, || {
        hasher::downscale_for_storage(&img, 1920)
    });
    bench("downscale_for_storage Triangle ->1920", 20, || {
        let (w, h) = (img.width(), img.height());
        img.resize(1920, (h as f64 * 1920.0 / w as f64) as u32, image::imageops::FilterType::Triangle)
    });
    bench("clone of full-res DynamicImage", 20, || img.clone());
    bench("downscale CatmullRom ->1920", 20, || {
        let (w, h) = (img.width(), img.height());
        img.resize(1920, (h as f64 * 1920.0 / w as f64) as u32, image::imageops::FilterType::CatmullRom)
    });

    let tmp = std::env::temp_dir();
    let out = tmp.join("perfprobe.webp");
    bench("save_webp q80 @1920x1200", 10, || {
        hasher::save_webp(&down, &out, 80).unwrap()
    });
    let thumb = hasher::create_thumbnail(&down, 320);
    let tout = tmp.join("perfprobe-thumb.webp");
    bench("thumbnail resize+save q75 @320", 10, || {
        hasher::save_webp(&hasher::create_thumbnail(&down, 320), &tout, 75).unwrap()
    });

    // Decode side: what the UI/browser pays per WebP it opens.
    let encoded = std::fs::read(&out).unwrap();
    bench("image crate WebP decode (1920x1200)", 10, || {
        image::load_from_memory_with_format(&encoded, image::ImageFormat::WebP).unwrap()
    });
    let tencoded = std::fs::read(&tout).unwrap();
    bench("image crate WebP decode (320x200 thumb)", 20, || {
        image::load_from_memory_with_format(&tencoded, image::ImageFormat::WebP).unwrap()
    });

    // Real-content encode: re-encode the decoded stored frame, no rescale.
    let real = image::DynamicImage::ImageRgba8(rgba.clone());
    let stored = tmp.join("perfprobe-real.webp");
    bench("save_webp q80 @stored res (real content)", 10, || {
        hasher::save_webp(&real, &stored, 80).unwrap()
    });

    // Exact gateway-OCR upload path (ocr.rs:484-495 read_upload_bytes):
    // image::open(WebP) -> Lanczos3 -> 1602 -> libwebp q80.
    bench("OCR upload: open+downscale1602+encode", 5, || {
        let img2 = image::open(&stored).unwrap();
        let small = hasher::downscale_for_storage(&img2, 1602);
        webp::Encoder::from_image(&small).unwrap().encode(80.0).to_vec()
    });
    bench("OCR upload with max_width=1920 (raw read)", 50, || {
        std::fs::read(&stored).unwrap()
    });
    let _ = std::fs::remove_file(&stored);

    let _ = DynamicImage::ImageRgba8(big);
    let _ = std::fs::remove_file(&out);
    let _ = std::fs::remove_file(&tout);
}