import numpy as np
from openslide.deepzoom import DeepZoomGenerator


class DeepZoomStaticTiler:
    """Handles generation of tiles and metadata for all images in a slide."""

    def __init__(self, tile_size, overlap, quality, background_limit, limit_bounds):
        self.tile_size = tile_size
        self.overlap = overlap
        self.quality = quality
        self.background_limit = background_limit
        self.limit_bounds = limit_bounds

    def convert_and_save_tile(self, tile, outfile):
        gray = tile.convert("L")

        bw = gray.point(lambda x: 0 if x < 230 else 1, "F")
        avg_background_limit = np.average(bw)

        # do not save non-square tiles near the edges
        nb_px = (tile.height * tile.width) - (self.tile_size + 2 * self.overlap) * (self.tile_size + 2 * self.overlap)

        if avg_background_limit <= (self.background_limit / 100.0) and nb_px == 0:
            tile.save(outfile, quality=self.quality)

    def process(self, slide, output_dir):
        """Process single slide."""
        tiles_generator = DeepZoomGenerator(slide, self.tile_size, self.overlap, self.limit_bounds)

        for level in range(tiles_generator.level_count - 1, -1, -1):
            cols, rows = tiles_generator.level_tiles[level]
            for row in range(rows):
                for col in range(cols):
                    address = (col, row)
                    tile = tiles_generator.get_tile(level, address)

                    self.convert_and_save_tile(tile, str(output_dir / f"row{row}-col{col}.jpg"))
