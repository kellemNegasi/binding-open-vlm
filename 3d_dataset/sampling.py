from __future__ import annotations

import itertools
import random
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


@dataclass
class ObjectSpec:
    shape: str
    color: str
    size: str = "small"
    material: str = "rubber"
    rotation: float | None = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "shape": self.shape,
            "color": self.color,
            "size": self.size,
            "material": self.material,
        }
        if self.rotation is not None:
            payload["rotation"] = self.rotation
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass
class SceneBlueprint:
    task_name: str
    slug: str
    objects: List[ObjectSpec]
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class SceneVocabulary:
    shapes: List[str]
    colors: List[str]
    materials: List[str]
    sizes: List[str]

    @classmethod
    def from_properties(cls, properties: Mapping[str, Mapping[str, object]]) -> "SceneVocabulary":
        return cls(
            shapes=list(properties.get("shapes", {}).keys()),
            colors=list(properties.get("colors", {}).keys()),
            materials=list(properties.get("materials", {}).keys()),
            sizes=list(properties.get("sizes", {}).keys()),
        )

    def ensure_shapes(self, required: Iterable[str]) -> None:
        missing = sorted(set(required) - set(self.shapes))
        if missing:
            raise ValueError(f"Missing shapes in Blender assets: {missing}")

    def ensure_colors(self, required: Iterable[str]) -> None:
        missing = sorted(set(required) - set(self.colors))
        if missing:
            raise ValueError(f"Missing colors in Blender assets: {missing}")


def _balanced_booleans(rng: random.Random, count: int) -> List[bool]:
    midpoint = count // 2
    flags = [True] * midpoint + [False] * (count - midpoint)
    rng.shuffle(flags)
    return flags


def make_disjunctive_scenes(
    *,
    num_images: int,
    distractor_range: Sequence[int],
    vocab: SceneVocabulary,
    rng: random.Random,
    target_shape: str = "sphere",
    distractor_color: str = "green",
    target_color: str = "red",
    object_size: str = "small",
) -> List[SceneBlueprint]:
    vocab.ensure_shapes([target_shape])
    vocab.ensure_colors([distractor_color, target_color])
    blueprints: List[SceneBlueprint] = []
    distractor_choices = list(distractor_range)
    if not distractor_choices:
        raise ValueError("distractor_range must contain at least one value")
    target_flags = _balanced_booleans(rng, num_images)
    for idx in range(num_images):
        n_distractors = rng.choice(distractor_choices)
        objects = [
            ObjectSpec(shape=target_shape, color=distractor_color, size=object_size)
            for _ in range(n_distractors)
        ]
        if target_flags[idx]:
            objects.append(
                ObjectSpec(
                    shape=target_shape,
                    color=target_color,
                    size=object_size,
                    metadata={"is_target": True},
                )
            )
        slug = f"{'popout' if target_flags[idx] else 'uniform'}_nd={n_distractors}"
        metadata = {
            "popout": target_flags[idx],
            "target_present": target_flags[idx],
            "n_distractors": n_distractors,
            "n_objects": len(objects),
            "answer": target_flags[idx],
        }
        blueprints.append(
            SceneBlueprint(
                task_name="disjunctive_search",
                slug=slug,
                objects=objects,
                metadata=metadata,
            )
        )
    return blueprints


def make_conjunctive_scenes(
    *,
    num_images: int,
    distractor_range: Sequence[int],
    vocab: SceneVocabulary,
    rng: random.Random,
    sphere_color: str = "green",
    cube_color: str = "red",
    target_shape: str = "sphere",
    target_color: str = "red",
    object_size: str = "small",
) -> List[SceneBlueprint]:
    vocab.ensure_shapes(["sphere", "cube", target_shape])
    vocab.ensure_colors([sphere_color, cube_color, target_color])
    blueprints: List[SceneBlueprint] = []
    target_flags = _balanced_booleans(rng, num_images)
    for idx in range(num_images):
        n_distractors = rng.choice(list(distractor_range))
        objects: List[ObjectSpec] = []
        for _ in range(n_distractors):
            if rng.random() < 0.5:
                objects.append(ObjectSpec(shape="sphere", color=sphere_color, size=object_size))
            else:
                objects.append(ObjectSpec(shape="cube", color=cube_color, size=object_size))
        if target_flags[idx]:
            objects.append(
                ObjectSpec(
                    shape=target_shape,
                    color=target_color,
                    size=object_size,
                    metadata={"is_target": True},
                )
            )
        slug = f"{'incongruent' if target_flags[idx] else 'congruent'}_nd={n_distractors}"
        metadata = {
            "incongruent": target_flags[idx],
            "target_present": target_flags[idx],
            "n_distractors": n_distractors,
            "n_objects": len(objects),
            "answer": target_flags[idx],
        }
        blueprints.append(
            SceneBlueprint(
                task_name="conjunctive_search",
                slug=slug,
                objects=objects,
                metadata=metadata,
            )
        )
    return blueprints


COUNTING_MODES = {
    "low_entropy": {"unique_shapes": False, "unique_colors": False},
    "medium_shape": {"unique_shapes": False, "unique_colors": True},
    "medium_color": {"unique_shapes": True, "unique_colors": False},
    "high_entropy": {"unique_shapes": True, "unique_colors": True},
}


def _draw_values(rng: random.Random, pool: Sequence[str], num: int, unique: bool) -> List[str]:
    values = list(pool)
    if not values:
        raise ValueError("Pool must contain at least one value.")
    if unique and num > len(values):
        warnings.warn(
            f"Requested {num} unique samples but only {len(values)} values are available; "
            "objects will contain repeated entries.",
            RuntimeWarning,
        )
    if unique and num <= len(values):
        return rng.sample(values, num)
    result = []
    for _ in range(num):
        if unique and len(result) < len(values):
            candidate_pool = [v for v in values if v not in result]
            candidate = rng.choice(candidate_pool)
        else:
            candidate = rng.choice(values)
        result.append(candidate)
    return result


def make_counting_scenes(
    *,
    num_images: int,
    count_range: Sequence[int],
    mode: str,
    vocab: SceneVocabulary,
    rng: random.Random,
    object_size: str = "small",
) -> List[SceneBlueprint]:
    if mode not in COUNTING_MODES:
        raise ValueError(f"Unsupported counting mode '{mode}'")
    specs = COUNTING_MODES[mode]
    counts = list(count_range)
    blueprints: List[SceneBlueprint] = []
    for idx in range(num_images):
        n_objects = rng.choice(counts)
        shapes = _draw_values(rng, vocab.shapes, n_objects, unique=specs["unique_shapes"])
        colors = _draw_values(rng, vocab.colors, n_objects, unique=specs["unique_colors"])
        objects = [
            ObjectSpec(shape=shape, color=color, size=object_size)
            for shape, color in zip(shapes, colors)
        ]
        slug = f"{mode}_n={n_objects}"
        metadata = {
            "n_objects": n_objects,
            "condition": mode,
            "answer": n_objects,
        }
        blueprints.append(
            SceneBlueprint(task_name=f"counting_{mode}", slug=slug, objects=objects, metadata=metadata)
        )
    return blueprints


def count_feature_triplets(objects: Sequence[ObjectSpec]) -> int:
    total = 0
    for triple in itertools.combinations(objects, 3):
        colors = [obj.color for obj in triple]
        shapes = [obj.shape for obj in triple]
        color_pair = len(set(colors)) < len(colors)
        shape_pair = len(set(shapes)) < len(shapes)
        if color_pair and shape_pair:
            total += 1
    return total


def make_scene_description_scenes(
    *,
    triplet_targets: Mapping[int, int],
    object_range: Tuple[int, int],
    vocab: SceneVocabulary,
    rng: random.Random,
    object_size: str = "small",
    max_attempts: int = 2000,
) -> List[SceneBlueprint]:
    min_objects, max_objects = object_range
    if min_objects > max_objects:
        raise ValueError("object_range must be (min, max)")
    blueprints: List[SceneBlueprint] = []
    for triplets, n_required in triplet_targets.items():
        for idx in range(n_required):
            success = False
            for _ in range(max_attempts):
                n_objects = rng.randint(min_objects, max_objects)
                shapes = [rng.choice(vocab.shapes) for _ in range(n_objects)]
                colors = [rng.choice(vocab.colors) for _ in range(n_objects)]
                objects = [
                    ObjectSpec(shape=shape, color=color, size=object_size)
                    for shape, color in zip(shapes, colors)
                ]
                triplet_count = count_feature_triplets(objects)
                if triplet_count == triplets:
                    slug = f"triplets={triplets}_n={n_objects}"
                    metadata = {
                        "n_objects": n_objects,
                        "triplet_count": triplets,
                        "features": [{"shape": obj.shape, "color": obj.color} for obj in objects],
                    }
                    blueprints.append(
                        SceneBlueprint(
                            task_name="scene_description",
                            slug=slug,
                            objects=objects,
                            metadata=metadata,
                        )
                    )
                    success = True
                    break
            if not success:
                raise RuntimeError(
                    f"Unable to sample scene with {triplets} feature triplets after {max_attempts} attempts. "
                    "Try relaxing the target histogram or expanding the vocabulary."
                )
    rng.shuffle(blueprints)
    return blueprints
