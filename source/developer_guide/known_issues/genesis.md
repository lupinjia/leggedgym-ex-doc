# Genesis 相关问题

本文档总结了 Genesis 仿真器（simulator）的已知问题和限制，用户在使用 LeggedGym-Ex 时可能会遇到这些问题。

---

## 1. 网格地形（Mesh Terrain）的异常碰撞行为

### 描述

当直接导入网格文件（如 STL）作为 Genesis 中的固定地形时，机器人与网格地形之间的碰撞检测（collision detection）会出现异常行为。这限制了使用自定义网格地形进行训练的能力。

### 相关问题

[GitHub Issue #2426](https://github.com/Genesis-Embodied-AI/Genesis/issues/2426)

### 症状

1. **当 `convexify=False` 时**：机器人会穿过网格地形并持续下落，表明碰撞检测未正常工作。

2. **当 `convexify=True` 时**：网格的细节信息会被消除，复杂地形无法适用。

3. **`mesh_to_heightfield()` 限制**：使用此方法将网格转换为高度场（heightfield）运行良好。但转换后的网格包含许多不必要的顶点（vertices），增加了计算负担。

### 影响

- 直接导入的网格地形无法可靠地用于足式机器人（legged robots）训练
- 用户只能使用 Genesis 提供的高度场地形生成方法

### 解决方法

目前请使用内置的地形生成方法：
- `mesh_type = "plane"` 用于平面地形
- `mesh_type = "heightfield"` 用于基于高度场的地形

在此问题解决之前，避免直接导入自定义网格文件作为地形。

### 状态

此问题已向 Genesis 团队报告，但在撰写本文时仍未解决。

## 2. Genesis 新增内容（英文原文合并翻译）

### 1. 网格地形的异常碰撞行为（Abnormal Collision Behavior with Mesh Terrain）
#### 描述
当直接将网格文件（如 STL）导入 Genesis 作为固定地形时，机器人与网格地形之间的碰撞检测会异常。这限制了使用自定义网格地形进行训练的能力。
#### 相关问题
[GitHub Issue #2426](https://github.com/Genesis-Embodied-AI/Genesis/issues/2426)
#### 症状
1. 当 convexify=False 时：机器人会穿过网格地形并持续下落，表明碰撞检测未正常工作。
2. 当 convexify=True 时：网格的细节信息会被消除，复杂地形无法适用。
3.  限制：将网格转换为高度场运行良好，但转换后的网格包含许多不必要的顶点，增加了计算负担。
#### 影响
- 直接导入网格地形无法可靠用于足式机器人训练
- 用户只能使用 Genesis 提供的高度场地形生成方法
#### 解决方法
目前请使用内置的地形生成方法：
-  用于平面地形
-  用于高度场地形
在问题解决之前，避免直接导入自定义网格文件作为地形。
#### 状态
此问题已向 Genesis 团队报告，但在撰写本文时仍未解决。
