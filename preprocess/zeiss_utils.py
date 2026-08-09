"""
Zeiss CZI Slide Reading Backend for Computational Pathology
Supports mosaic CZI images using aicspylibczi.
"""

from typing import Dict, Tuple
import cv2
import numpy
import aicspylibczi


class Backend:
    def __init__(self):
        pass

    @property
    def dimensions(self) -> Tuple[int, int]:
        return self.get_dimensions()

    @property
    def level_downsamples(self) -> Dict[int, Tuple[float, float]]:
        return self.get_level_downsamples()

    @property
    def level_dimensions(self) -> Dict[int, Tuple[int, int]]:
        return self.get_level_dimensions()

    def get_dimensions(self) -> Tuple[int, int]:
        raise NotImplementedError()

    def get_level_downsamples(self) -> Dict[int, Tuple[float, float]]:
        raise NotImplementedError()

    def get_level_dimensions(self) -> Dict[int, Tuple[int, int]]:
        raise NotImplementedError()

    def get_thumbnail(self, level: int) -> numpy.ndarray:
        raise NotImplementedError()

    def read_region(self, coord: Tuple[int, int], level: int, patch_size: Tuple[int, int]) -> numpy.ndarray:
        raise NotImplementedError()


class ZeissBackend(Backend):
    def __init__(self, path: str):
        super().__init__()
        self.__path = path
        self.reader = aicspylibczi.CziFile(path)
        if not self.reader.is_mosaic():
            raise NotImplementedError("Non-mosaic Zeiss CZI files are not supported.")
        
        bbox = self.reader.get_mosaic_bounding_box()
        self.__dimensions = (bbox.h, bbox.w)
        self.__origo = (bbox.x, bbox.y)
        self.__level_dimensions = None
        self.__level_downsamples = None

    def get_dimensions(self) -> Tuple[int, int]:
        return self.__dimensions

    def get_level_dimensions(self) -> Dict[int, Tuple[int, int]]:
        if self.__level_dimensions is None:
            level_dimensions = {0: self.__dimensions}
            downsample = 1
            while True:
                if max(self.dimensions) // (2 ** downsample) < 512:
                    break
                level_dimensions[downsample] = tuple(
                    x // (2 ** downsample) for x in self.dimensions
                )
                downsample += 1
            self.__level_dimensions = level_dimensions
        return self.__level_dimensions

    def get_level_downsamples(self) -> Dict[int, Tuple[float, float]]:
        if self.__level_downsamples is None:
            level_downsamples = {}
            for level, (y, x) in self.level_dimensions.items():
                level_downsamples[level] = (
                    self.dimensions[0] / float(y),
                    self.dimensions[1] / float(x),
                )
            self.__level_downsamples = level_downsamples
        return self.__level_downsamples

    def get_thumbnail(self, level: int) -> numpy.ndarray:
        scale_factor = 1.0 / (2 ** level)
        thumbnail = self.reader.read_mosaic(
            scale_factor=scale_factor,
            C=0,
            background_color=(1.0, 1.0, 1.0),
        )[0]
        return cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)

    def read_region(self, coord: Tuple[int, int], level: int, patch_size: Tuple[int, int]) -> numpy.ndarray:
        x, y, w, h = coord[0], coord[1], patch_size[0], patch_size[1]
        scale_factor = 1.0 / (2 ** level)
        
        xywh = (
            self.__origo[0] + int(x / scale_factor),
            self.__origo[1] + int(y / scale_factor),
            int(w / scale_factor),
            int(h / scale_factor),
        )
        
        tile = self.reader.read_mosaic(
            region=xywh,
            scale_factor=scale_factor,
            C=0,
            background_color=(1.0, 1.0, 1.0),
        )[0]
        return cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

    def __repr__(self) -> str:
        return f"ZeissBackend(path='{self.__path}')"