# Genesis

This document summarizes known issues and limitations of the Genesis simulator that users may encounter when using LeggedGym-Ex.

---

## 1. Abnormal Collision Behavior with Mesh Terrain

### Description

When importing mesh files (e.g., STL) directly as fixed terrain in Genesis, collision detection between robots and the mesh terrain behaves abnormally. This limits the ability to use custom mesh terrains for training.

### Related Issue

[GitHub Issue #2426](https://github.com/Genesis-Embodied-AI/Genesis/issues/2426)

### Symptoms

1. **When `convexify=False`**: The robot passes through the mesh terrain and keeps falling, indicating that collision detection is not working properly.

2. **When `convexify=True`**: The details of the mesh are eliminated. Complex terrains are not applicable.

3. **`mesh_to_heightfield()` limitation**: Converting mesh to heightfield using this method works well. But as we know, the converted mesh has many unnecessary vertices, adding the burden of computation.

### Impact

- Direct mesh terrain import cannot be reliably used for training legged robots
- Users are limited to using heightfield terrain generation methods provided by Genesis

### Workaround

For now, use the built-in terrain generation methods:
- `mesh_type = "plane"` for flat terrain
- `mesh_type = "heightfield"` for heightfield-based terrain

Avoid importing custom mesh files directly as terrain until this issue is resolved.

### Status

This issue has been reported to the Genesis team but remains unresolved at the time of writing.
