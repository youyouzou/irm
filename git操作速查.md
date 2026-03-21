# IRM仓库 Git 操作全量速查
```powershell
# ==================== 一、基础配置（首次使用必做） ====================
# 1. 配置 Git 用户名/邮箱（与 GitHub 账号一致）
git config --global user.name "youyouzou"
git config --global user.email "3360932990@qq.com"
# 验证配置是否生效
git config --global --list

# 2. 查看当前分支名（核心，所有推送命令需匹配）
git branch  # 带*号的行即为当前分支（如 master/main）

# 补充：VS Code 实用快捷键
# 保存当前文件：Windows/Linux 按 Ctrl + S；macOS 按 ⌘ + S
# 保存所有文件：Windows/Linux 按 Ctrl + Shift + S；macOS 按 ⌘ + Shift + S
# 打开设置：Windows/Linux 按 Ctrl + ,；macOS 按 ⌘ + ,

# ==================== 二、SSH 密钥配置（必做，避开 443 端口拦截） ====================
# 1. 生成 SSH 密钥（全程按回车，不设置密码）
ssh-keygen -t ed25519 -C "3360932990@qq.com"

# 2. 查看/复制 SSH 公钥（复制后添加到 GitHub）
# Windows 系统执行以下命令
notepad $HOME\.ssh\id_ed25519.pub
# Mac/Linux 系统执行（备用）：cat ~/.ssh/id_ed25519.pub

# 3. 验证 SSH 连接是否成功
ssh -T git@github.com
# 成功提示：Hi youyouzou! You've successfully authenticated, but GitHub does not provide shell access.

# ==================== 三、核心操作：上传 IRM1 到 GitHub 仓库 ====================
# 1. 初始化本地仓库（仅首次执行）
# 进入 IRM1 文件夹（替换为实际路径）
cd D:\IRM1
# 初始化 Git 仓库（仅创建.git目录，不修改代码）
git init

# 2. 清理冗余/错误的远程仓库关联（避免冲突）
git remote rm IRM1
git remote rm IRM5300
git remote rm origin

# 3. 关联 GitHub 的 IRM 仓库（SSH 协议）
git remote add origin git@github.com:youyouzou/IRM.git
# 验证关联是否正确（输出应为 SSH 地址）
git remote -v

# 4. 提交并推送 IRM1 代码
# 暂存所有文件
git add .
# 提交代码（备注必填，清晰描述操作）
git commit -m "init: 上传 IRM1 初始代码"
# 首次推送（分支名替换为实际的 master/main）
git push -u origin master

# ==================== 四、日常同步：本地修改 IRM1 后推送到 GitHub ====================
# 1. 先拉取远程最新代码（推荐，避免冲突）
git pull origin master  # 分支名替换为 master/main

# 2. 暂存 + 提交 + 推送修改
# 暂存所有修改
git add .
# 提交修改（备注需描述具体改动）
git commit -m "update: 修复IRM1/main.py逻辑错误 / 新增XX功能"
# 推送到 GitHub
git push origin master  # 分支名替换为 master/main

# ==================== 五、版本恢复：修改后恢复到原始代码结构 ====================
# 场景1：改了代码但未执行 git add（最易恢复）
git checkout .  # 一键恢复所有未暂存的修改

# 场景2：已 git add 暂存，但未 git commit 提交
git reset HEAD .  # 取消暂存
git checkout .    # 恢复修改

# 场景3：已 git commit 提交（本地），未推送到 GitHub
git log --oneline  # 查看版本记录，复制要恢复的版本号（如 abc123）
git reset --hard abc123  # 回滚到指定版本

# 场景4：已推送到 GitHub，需恢复远程版本
git log --oneline  # 查看版本记录，复制目标版本号
git reset --hard abc123  # 本地回滚到指定版本
git push -f origin master  # 强制推送覆盖远程（单人使用放心，多人协作需谨慎）

# ==================== 六、多电脑协作：另一台电脑上传 IRM2 到 IRM 仓库 ====================
# 步骤1：另一台电脑先完成 SSH 配置（参考第二部分所有命令）

# 步骤2：克隆 GitHub 的 IRM 仓库到本地
git clone git@github.com:youyouzou/IRM.git  # 克隆后生成 IRM 文件夹

# 步骤3：添加 IRM2 并推送
cd 桌面/IRM  # 进入克隆的 IRM 文件夹（替换为实际路径，如桌面/IRM）
# 先把 IRM2 文件夹复制到当前目录，再执行以下命令
git add .
git commit -m "add: 上传 IRM2 文件夹"
git push origin master  # 分支名替换为 master/main

# ==================== 七、常见问题解决 ====================
# 问题1：提示 "remote origin already exists"（远程已存在）
git remote rm origin  # 删除旧关联，再重新执行关联命令

# 问题2：推送提示 "RPC failed; curl 55 Send failure"（443 端口拦截）
git remote -v  # 确认远程地址为 SSH 协议（非 HTTPS）
# 若为 HTTPS 地址，删除后重新关联 SSH
git remote rm origin
git remote add origin git@github.com:youyouzou/IRM.git

# 问题3：推送提示 "src refspec master does not match any"（分支名错误）
git branch  # 确认实际分支名（如 main）
git push -u origin main  # 替换为正确分支名

# 问题4：VS Code 「发布 Branch」按钮转圈卡死
git push -u origin master  # 绕过 GUI，直接用终端推送（替换分支名）


