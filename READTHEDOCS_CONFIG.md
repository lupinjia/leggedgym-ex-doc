# Read the Docs 多语言配置指南

本指南说明如何在 Read the Docs 上配置 LeggedGym-Ex 文档的中英文版本。

## 架构说明

我们使用 **子项目方式** (方案A) 来实现多语言支持：

- **主项目** (英文): `leggedgym-ex-doc` 
- **子项目** (中文): `leggedgym-ex-doc-zh`

两个项目共享同一个 Git 仓库，但使用不同的源码目录：
- 英文: `source/`
- 中文: `source_zh/`

## 配置步骤

### 1. 导入英文主项目

1. 登录 [Read the Docs](https://readthedocs.org/)
2. 点击 "Import a Project"
3. 选择 GitHub 上的 `LeggedGym-Ex` 仓库
4. 项目名称填写: `leggedgym-ex-doc`
5. 在 "Advanced Settings" 中:
   - **Language**: 选择 "English"
   - **Documentation type**: 选择 "Sphinx"
   - **Python configuration file**: 填写 `LeggedGym-Ex_doc/source/conf.py`
6. 点击 "Build"

### 2. 导入中文子项目

1. 再次点击 "Import a Project"
2. 选择同一个 GitHub 仓库
3. 项目名称填写: `leggedgym-ex-doc-zh`
4. 在 "Advanced Settings" 中:
   - **Language**: 选择 "Simplified Chinese" (简体中文)
   - **Documentation type**: 选择 "Sphinx"
   - **Python configuration file**: 填写 `LeggedGym-Ex_doc/source_zh/conf.py`
5. 点击 "Build"

### 3. 关联子项目到主项目

1. 进入英文主项目 `leggedgym-ex-doc` 的管理页面
2. 点击左侧菜单 "Translations"
3. 在 "Project" 下拉菜单中选择 `leggedgym-ex-doc-zh`
4. 点击 "Add Translation"

完成！现在访问英文文档时，会在左下角看到语言切换器，可以在中英文之间切换。

## 本地构建测试

### 构建英文文档

```bash
cd LeggedGym-Ex_doc

# 安装依赖
pip install -r requirements.txt

# 构建英文文档
make html

# 查看结果
open build/html/index.html
```

### 构建中文文档

```bash
cd LeggedGym-Ex_doc

# 构建中文文档
make html-zh

# 查看结果
open build/zh_CN/html/index.html
```

### 同时构建两种语言

```bash
make html-all
```

## 文件结构

```
LeggedGym-Ex_doc/
├── .readthedocs.yaml       # 英文项目配置
├── .readthedocs-zh.yaml    # 中文项目配置
├── Makefile                # 构建脚本（支持多语言）
├── requirements.txt        # Python 依赖
├── source/                 # 英文文档源码
│   ├── conf.py
│   ├── index.md
│   ├── user_guide/
│   ├── developer_guide/
│   └── _static/
├── source_zh/              # 中文文档源码
│   ├── conf.py            # 中文配置（language = 'zh_CN'）
│   ├── index.md           # 中文首页
│   ├── user_guide/        # 中文用户指南
│   ├── developer_guide/   # 中文开发者指南
│   └── _static/           # 静态资源（与英文共享）
└── docs/                   # 额外文档
    ├── design.md
    └── overview.md
```

## 维护说明

### 文档更新流程

由于采用手动维护（选项A），当需要更新文档时：

1. **先更新英文文档** (`source/`)
2. **同步更新中文文档** (`source_zh/`)
3. **保持内容同步**，确保中英文文档描述的是相同的功能

### 翻译规范

- 技术术语首次出现时提供中英对照（如：Teacher-Student → 教师-学生 (Teacher-Student)）
- 代码、命令、文件名、URL 保持不变
- 配置参数名保持不变
- 论文标题和作者名保持不变

### 自动化检查（可选）

可以创建 GitHub Action 来检查中英文文档的同步状态：

```yaml
# .github/workflows/check-translations.yml
name: Check Translation Sync

on: [push, pull_request]

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Check if Chinese docs are in sync
        run: |
          # 检查 source/ 和 source_zh/ 中的文件是否一一对应
          # 可以添加更复杂的检查逻辑
```

## 注意事项

1. **不要删除** `.readthedocs.yaml` 或 `.readthedocs-zh.yaml`
2. **保持目录结构一致**：`source/` 和 `source_zh/` 应该有相同的文件结构
3. **静态资源**：中英文文档共享相同的 `_static/` 目录内容
4. **URL 结构**：
   - 英文: `https://leggedgym-ex-doc.readthedocs.io/en/latest/`
   - 中文: `https://leggedgym-ex-doc.readthedocs.io/zh_CN/latest/`

## 故障排除

### 中文显示乱码

确保 `source_zh/conf.py` 中设置了正确的语言：

```python
language = 'zh_CN'
```

### 构建失败

检查配置文件路径是否正确：
- 英文: `source/conf.py`
- 中文: `source_zh/conf.py`

### 语言切换器不显示

确保子项目已正确关联到主项目的 "Translations" 设置中。

## 参考链接

- [Read the Docs 多语言文档](https://docs.readthedocs.io/en/stable/localization.html)
- [Read the Docs 配置文件参考](https://docs.readthedocs.io/en/stable/config-file/v2.html)
- [Sphinx 国际化文档](https://www.sphinx-doc.org/en/master/usage/advanced/intl.html)
