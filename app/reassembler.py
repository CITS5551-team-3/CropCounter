import cv2

import time
from typing import Tuple, Optional, List, Dict

import numpy as np


def vegetation_index(rgba):
    r = rgba[:, :, 0].astype(np.int32)
    g = rgba[:, :, 1].astype(np.int32)
    b = rgba[:, :, 2].astype(np.int32)

    exg = g + g - r - b
    np.clip(exg, 0, 255, out=exg)

    return exg.astype(np.uint8)


def vegetation_index_bgr(bgr):
    blue, green, red = cv2.split(bgr)
    NGRDI_mask=np.where(green>red,255,0).astype(np.uint8)

    img = cv2.bitwise_or(bgr, bgr, mask=NGRDI_mask)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)

    return img

    r = bgr[:, :, 2].astype(np.int32)
    g = bgr[:, :, 1].astype(np.int32)
    b = bgr[:, :, 0].astype(np.int32)

    exg = g + g - r - b
    np.clip(exg, 0, 255, out=exg)

    return exg.astype(np.uint8)

def box_ioa1(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Args:
        box1 (numpy.ndarray): A 2D array of shape (N, 4) representing N bounding boxes.
        box2 (numpy.ndarray): A 2D array of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (numpy.ndarray): An NxM array containing the pairwise IoA values for every element in box1 and box2.
    """
    if len(box1) == 0 or len(box2) == 0:
        return np.array([])

    # Split box1 and box2 into coordinates (x1, y1) and (x2, y2)
    a1, a2 = np.split(box1[:, np.newaxis, :], 2, axis=2)
    b1, b2 = np.split(box2[np.newaxis, :, :], 2, axis=2)

    # Calculate the intersection area
    inter = np.prod(np.maximum(0, np.minimum(a2, b2) - np.maximum(a1, b1)), axis=2)

    # Calculate the IoA1: inter / area1
    area1 = np.prod(a2 - a1, axis=2)
    return inter / (area1 + eps)



class Rect:
    __slots__ = ('x', 'y', 'w', 'h')

    def __init__(self, x: int, y: int, w: int, h: int):
        self.x: int = x
        self.y: int = y
        self.w: int = w
        self.h: int = h

    def contains(self, x0: int, y0: int) -> bool:
        return self.x <= x0 <= self.x + self.w and self.y <= y0 <= self.y + self.h

    def contains_rect(self, other: 'Rect') -> bool:
        return self.contains(other.x, other.y) and self.contains(other.x + other.w, other.y + other.h)

    def intersects(self, other: 'Rect') -> bool:
        return not (self.x + self.w < other.x or
                    other.x + other.w < self.x or
                    self.y + self.h < other.y or
                    other.y + other.h < self.y)

    def __repr__(self):
        return f'({self.x}, {self.y}, {self.w}, {self.h})'

    @property
    def pt1(self) -> Tuple[int, int]:
        return self.x, self.y

    @property
    def pt2(self) -> Tuple[int, int]:
        return self.x + self.w, self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    @classmethod
    def from_tuple(cls, t: Tuple[int, int, int, int]):
        return Rect(t[0], t[1], t[2], t[3])


class WeedLabel:
    __slots__ = ('rect', 'cls', 'conf')

    def __init__(self, rect: Rect, cls: int, conf: float = 1.0):
        self.rect = rect
        self.cls = cls
        self.conf = conf

    def __repr__(self):
        return f"[{self.rect} cls={self.cls} conf={self.conf}]"


class RectMapping:
    __slots__ = ('src', 'dst')

    def __init__(self, src: Rect, dst: Optional[Rect]):
        self.src = src
        self.dst = dst

    def __repr__(self):
        return f'({self.src} -> {self.dst})'


def draw_boxes(img: np.ndarray, rects: List[Rect], color=(255, 0, 255)):
    for rect in rects:
        p1 = (rect.x, rect.y)
        p2 = (rect.x + rect.w, rect.y + rect.h)
        cv2.rectangle(img, p1, p2, color, 2)


def draw_labels(img: np.ndarray, wls: List[WeedLabel], color=(255, 0, 255)):
    for wl in wls:
        rect = wl.rect
        p1 = (rect.x, rect.y)
        p2 = (rect.x + rect.w, rect.y + rect.h)
        cv2.putText(img, f"{wl.cls}", p1, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
        cv2.rectangle(img, p1, p2, color, 2)


def split_list_randomly(lst, n: int):
    np.random.shuffle(lst)
    sublists = np.array_split(lst, n)
    sublists = [sublist.tolist() for sublist in sublists]
    return sublists


def sample_rects(image_width: int, image_height: int, forbidden_area: List[Rect], scale=20, num_samples=50):
    """
    Sample a number of rects without overlapping with the given ones.
    Args:
        image_width:
        image_height:
        forbidden_area:
        scale: mean value of the length distribution
        num_samples:

    Returns:

    """
    sampled_rects = []

    while len(sampled_rects) < num_samples:
        w = max(min(int(np.random.exponential(scale=scale)), 120), 6)
        h = max(min(int(np.random.exponential(scale=scale)), 120), 6)

        # Ensure the rectangle fits within the image boundaries
        max_x = image_width - w
        max_y = image_height - h

        if max_x <= 0 or max_y <= 0:
            continue

        # Sample a random top-left corner within the allowed range
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

        new_rect = Rect(x, y, w, h)

        # Check if the new rectangle intersects with any given rectangle
        if all(not new_rect.intersects(rect) for rect in forbidden_area):
            sampled_rects.append(new_rect)

    return sampled_rects


class Stopwatch:
    def __init__(self):
        self.startT = 0

    def start(self):
        self.startT = time.time()

    def stop(self, name: str):
        print(f"[{name}] {int(1000 * (time.time() - self.startT))}ms")
        self.start()


def calculate_iou(bbox1: Rect, bbox2: Rect) -> float:
    x1_min = bbox1.x
    y1_min = bbox1.y
    x1_max = bbox1.x + bbox1.w
    y1_max = bbox1.y + bbox1.h

    x2_min = bbox2.x
    y2_min = bbox2.y
    x2_max = bbox2.x + bbox2.w
    y2_max = bbox2.y + bbox2.h

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)

    iou = inter_area / (bbox1_area + bbox2_area - inter_area)
    return iou



class RectSorting:
    WIDTH_DESC = lambda rect: rect.w
    HEIGHT_DESC = lambda rect: rect.h
    AREA_DESC = lambda rect: rect.w * rect.h
    MAXSIDE_DESC = lambda rect: max(rect.w, rect.h)


class Block:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.used = False
        self.down = None
        self.right = None


class Packer:
    def __init__(self, w, h):
        self.root: Block = Block(0, 0, w, h)

    def fit(self, rects: List[Rect]):
        result: List[RectMapping] = []
        for rect in rects:
            node = self.find_node(self.root, rect.w, rect.h)
            if node:
                fit = self.split_node(node, rect.w, rect.h)
                fit_rect = Rect(fit.x, fit.y, rect.w, rect.h)
                mapping = RectMapping(rect, fit_rect)
                result.append(mapping)
            else:
                mapping = RectMapping(rect, None)
                result.append(mapping)
        return result

    def find_node(self, root, w, h) -> Optional[Block]:
        if root.used:
            return self.find_node(root.right, w, h) or self.find_node(root.down, w, h)
        elif w <= root.w and h <= root.h:
            return root
        else:
            return None

    def split_node(self, node: Block, w: int, h: int) -> Block:
        node.used = True
        node.down = Block(node.x, node.y + h, node.w, node.h - h)
        node.right = Block(node.x + w, node.y, node.w - w, h)
        return node


class ResizablePacker:
    def __init__(self):
        self.root = None

    def fit(self, rects: List[Rect]):
        if not rects or len(rects) == 0:
            return []

        w = rects[0].w
        h = rects[0].h
        self.root = Block(0, 0, w, h)
        result: List[RectMapping] = []

        for rect in rects:
            node = self.find_node(self.root, rect.w, rect.h)
            fit_rect: Optional[Rect] = None
            if node:
                fit = self.split_node(node, rect.w, rect.h)
                fit_rect = Rect(fit.x, fit.y, rect.w, rect.h)
            else:
                fit = self.grow_node(rect.w, rect.h)
                if fit:
                    fit_rect = Rect(fit.x, fit.y, rect.w, rect.h)
            mapping = RectMapping(rect, fit_rect)
            result.append(mapping)
        return result

    def find_node(self, root, w, h) -> Optional[Block]:
        if root.used:
            return self.find_node(root.right, w, h) or self.find_node(root.down, w, h)
        elif w <= root.w and h <= root.h:
            return root
        else:
            return None

    def split_node(self, node: Block, w: int, h: int) -> Block:
        node.used = True
        node.down = Block(node.x, node.y + h, node.w, node.h - h)
        node.right = Block(node.x + w, node.y, node.w - w, h)
        return node

    def grow_node(self, w: int, h: int) -> Optional[Block]:
        can_grow_down = w <= self.root.w
        can_grow_right = h <= self.root.h

        should_grow_right = can_grow_right and (self.root.h >= self.root.w + w)
        should_grow_down = can_grow_down and (self.root.w >= self.root.h + h)

        if should_grow_right:
            return self.grow_right(w, h)
        elif should_grow_down:
            return self.grow_down(w, h)
        elif can_grow_right:
            return self.grow_right(w, h)
        elif can_grow_down:
            return self.grow_down(w, h)
        else:
            return None

    def grow_right(self, w, h) -> Optional[Block]:
        root_old = self.root
        self.root = Block(0, 0, root_old.w + w, root_old.h)
        self.root.used = True
        self.root.down = root_old
        self.root.right = Block(root_old.w, 0, w, root_old.h)
        node = self.find_node(self.root, w, h)
        if node:
            return self.split_node(node, w, h)
        else:
            return None

    def grow_down(self, w, h) -> Optional[Block]:
        root_old = self.root
        self.root = Block(0, 0, root_old.w, root_old.h + h)
        self.root.used = True
        self.root.down = Block(0, root_old.h, root_old.w, h)
        self.root.right = root_old
        node = self.find_node(self.root, w, h)
        if node:
            return self.split_node(node, w, h)
        else:
            return None


class Reassembler:
    def __init__(self):
        self.rects: List[Rect] = []
        self.mappings: List[RectMapping] = []
        self.reassembled = False

    def addRect(self, rect: Rect) -> None:
        if self.reassembled:
            raise Exception("Cannot add more rectangles after reassembly.")
        self.rects.append(rect)

    def reassemble(self, srcImg: np.ndarray,
                   initial_width: int = 640,
                   sorting_method: RectSorting = RectSorting.HEIGHT_DESC,
                   autosize: bool = False,
                   border: int = 3,
                   margin: int | Tuple[int, int] = 8) -> np.ndarray:
        """

        Args:
            srcImg:
            initial_width: A sensible value is the square root of total area over all rects
            sorting_method: Choose from RectSorting
            autosize: Frame size are automatically adjusted during fitting. initial_with does not take effect when autosize is on.
            border: The amount of black borders to add around each rectangle (in pixels)
            margin: The number of pixels by which to move each side of the rectangles away from their center. Passing a 2-tuple
            enables randomised margin and specifies the lower bound and upper bound of the margin width.

        Returns:
            The reassembled image.

        """
        if self.reassembled:
            raise Exception("Already reassembled")

        # Preprocessing
        green_rects = extract_green_regions_bgr(srcImg)
        for rect in green_rects:
            self.addRect(rect)

        # Step 1: Finalise rectangles (add margins and borders)
        imgh, imgw = srcImg.shape[:2]
        expanded_rects: List[Rect] = []
        if isinstance(margin, int):  # Constant size margin
            extra = margin + border
            for rect in self.rects:
                x1, y1, w, h = rect.x, rect.y, rect.w, rect.h
                x2, y2 = x1 + w, y1 + h
                x1, y1 = max(x1 - extra, 0), max(y1 - extra, 0)
                x2, y2 = min(x2 + extra, imgw), min(y2 + extra, imgh)
                expanded_rects.append(Rect(x1, y1, x2 - x1, y2 - y1))
        else:  # Margin with size ranging from the given range
            margin_lo, margin_hi = margin
            for rect in self.rects:
                x1, y1, w, h = rect.x, rect.y, rect.w, rect.h
                x2, y2 = x1 + w, y1 + h
                mL, mR, mT, mB = np.random.randint(margin_lo, margin_hi, size=4)
                x1, y1 = max(x1 - mL, 0), max(y1 - mT, 0)
                x2, y2 = min(x2 + mR, imgw), min(y2 + mB, imgh)
                expanded_rects.append(Rect(x1, y1, x2 - x1, y2 - y1))

        # Step 2: sort rects and pack
        start_t = time.time()
        sorted_rects = sorted(expanded_rects, key=sorting_method, reverse=True)
        packer = None
        if autosize:
            packer = ResizablePacker()
        else:
            packer = Packer(initial_width, initial_width)
        mappings = packer.fit(sorted_rects)
        self.mappings = mappings
        fit_el = int((time.time() - start_t) * 1000)

        # Step 3: construct the reassembled image and collect stats
        start_t = time.time()
        result_img = self.draw_rects(srcImg, packer.root.w, packer.root.h, mappings, border)
        draw_el = int((time.time() - start_t) * 1000)

        effective_area = np.sum([i.dst.w * i.dst.h if i.dst else 0 for i in mappings])
        n_fail = np.sum([0 if i.dst else 1 for i in mappings])
        total_area = result_img.shape[0] * result_img.shape[1]
        print(f"{result_img.shape}   areaR={total_area / 640 / 640}   utilisation={effective_area / total_area}"
              f"  n_fail={n_fail}   fit_el={fit_el}ms   draw_el={draw_el}ms")

        self.reassembled = True
        return result_img

    def draw_rects(self, srcImg: np.ndarray,
                   canvas_width: int, canvas_height: int,
                   rect_mappings: List[RectMapping],
                   border: int = 0) -> np.ndarray:
        canvas = np.full((canvas_height, canvas_width, 3), 0, dtype=np.uint8)

        for rect_map in rect_mappings:
            if rect_map.dst:
                canvas[rect_map.dst.y + border:rect_map.dst.y + rect_map.dst.h - border,
                rect_map.dst.x + border:rect_map.dst.x + rect_map.dst.w - border] = srcImg[
                                                                                    rect_map.src.y + border:rect_map.src.y + rect_map.src.h - border,
                                                                                    rect_map.src.x + border:rect_map.src.x + rect_map.src.w - border]

        return canvas

    def reverse_mapping(self, rects: List[Rect]) -> List[RectMapping]:
        """
        Map the rects in mapped image space back to original image space
        Args:
            rects:

        Returns:

        """
        if not self.reassembled:
            raise Exception("Not reassembled")

        rect_mapped_space = [(r.x, r.y, r.x + r.w, r.y + r.h) for r in rects]
        rect_reference = [(r.dst.x, r.dst.y, r.dst.x + r.dst.w, r.dst.y + r.dst.h) for r in self.mappings]

        result: List[RectMapping] = self.__map0(rect_mapped_space, rect_reference, rects, rev=True)
        return result

    def mapping(self, rects: List[Rect]) -> List[RectMapping]:
        """
        Map the rects in original image space to mapped image space
        Args:
            rects:

        Returns:

        """
        if not self.reassembled:
            raise Exception("Not reassembled")

        rect_mapped_space = [(r.x, r.y, r.x + r.w, r.y + r.h) for r in rects]
        rect_reference = [(r.src.x, r.src.y, r.src.x + r.src.w, r.src.y + r.src.h) for r in self.mappings]

        result: List[RectMapping] = self.__map0(rect_mapped_space, rect_reference, rects, rev=False)
        return result

    def __map0(self, rect_mapped_space: List[Tuple], rect_reference: List[Tuple], rects: List[Rect], rev: bool):
        result: List[RectMapping] = []
        ious = box_ioa1(np.array(rect_mapped_space), np.array(rect_reference))
        for i in range(len(rect_mapped_space)):
            rect = rects[i]
            ci = np.argmax(ious[i])
            candidate = self.mappings[ci]
            iou = ious[i][ci]

            for j, rm in enumerate(self.mappings):
                if rm.dst.contains_rect(rect):
                    candidate = rm
                    iou = 1
                    ci = j
                    break

            if False:# not candidate or iou < 0.1:
                result.append(RectMapping(rect, None))
            else:
                dst = candidate.dst if rev else candidate.src
                src = candidate.src if rev else candidate.dst
                dx = rect.x - dst.x
                dy = rect.y - dst.y
                mapped_rect = Rect(src.x + dx, src.y + dy, rect.w, rect.h)
                result.append(RectMapping(rect, mapped_rect))

        return result


def draw_rects_bound(canvas_width: int, canvas_height: int, rect_mappings: List[RectMapping]) -> np.ndarray:
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    for rect_map in rect_mappings:
        if rect_map.dst:
            top_left = (rect_map.dst.x, rect_map.dst.y)
            bottom_right = (rect_map.dst.x + rect_map.dst.w, rect_map.dst.y + rect_map.dst.h)

            col = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

            cv2.rectangle(canvas, top_left, bottom_right, col, 1)

    return canvas


def extract_green_regions(image) -> List[Rect]:
    exg = vegetation_index(image)
    _, b1 = cv2.threshold(exg, 15, 255, cv2.THRESH_BINARY)

    # b, g, r = cv2.split(image)
    # b1 = np.where(np.right_shift(g,1) > np.right_shift(b,2)+np.right_shift(r,2)+7, 0, 255).astype(np.uint8)

    b2 = cv2.morphologyEx(b1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(b2, connectivity=8)

    result: List[Rect] = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        result.append(Rect(x, y, w, h))
    return result


def extract_green_regions_bgr(image) -> List[Rect]:
    b1 = vegetation_index_bgr(image)

    b2 = cv2.erode(b1, np.ones((10, 10), np.uint8))
    b2 = cv2.dilate(b2, np.ones((10, 10), np.uint8))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(b2, connectivity=8)

    result: List[Rect] = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if w < 10 or h < 10:
            continue
        result.append(Rect(x, y, w, h))
    return result

