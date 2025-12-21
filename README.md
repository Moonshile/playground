# playground

my playground

## uv 环境管理说明

### 在已有项目上迁移到 uv

如果你已经有一个使用 pip 和 requirements.txt 的项目，可以按照以下步骤迁移到 uv：

#### 1. 安装 uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或者使用 pip 安装
pip install uv
```

#### 2. 初始化 uv 项目（如果还没有 pyproject.toml）

如果项目还没有 `pyproject.toml`，可以运行：

```bash
uv init --no-readme
```

如果已经有 `pyproject.toml`，可以跳过此步骤。

#### 3. 自动从 pip freeze 添加依赖（推荐方法）

如果你想从当前 pip 环境自动迁移所有依赖：

```bash
# 1. 导出当前 pip 环境的依赖
pip freeze > requirements-current.txt

# 2. 使用 uv 从 requirements 文件添加依赖到 pyproject.toml
uv pip compile requirements-current.txt -o pyproject.toml

# 或者手动编辑 pyproject.toml，将依赖添加到 dependencies 字段：
# dependencies = [
#     "package1==1.0.0",
#     "package2==2.0.0",
#     ...
# ]

# 3. 同步依赖到虚拟环境
uv sync
```

#### 4. 验证迁移

```bash
# 检查依赖是否正确安装
uv pip list

# 运行项目验证是否正常工作
uv run python your_script.py
```

#### 5. 后续使用

迁移完成后，可以：

- **添加新依赖**: `uv add package-name`
- **移除依赖**: `uv remove package-name`
- **同步环境**: `uv sync`
- **运行命令**: `uv run python script.py`
- **激活虚拟环境**: `source .venv/bin/activate` (Linux/macOS) 或 `.venv\Scripts\activate` (Windows)

#### 注意事项

- `requirements.txt` 可以保留作为备份，但后续建议使用 `pyproject.toml` 和 `uv.lock` 管理依赖
- 如果依赖中有 Git URL 或本地路径，需要在 `pyproject.toml` 中手动配置
- 确保 `pyproject.toml` 中的 `requires-python` 版本与项目兼容
