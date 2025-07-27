# UPMS
# 用户权限管理系统（UPMS）

## 项目概述

用户权限管理系统（UPMS，User Permission Management System）是一款基于 Django 构建的通用型权限管理解决方案。该系统旨在解决各类软件应用中权限管理混乱、重复开发等问题，通过提供统一的用户、角色、权限管理功能，帮助开发者快速集成权限控制能力，提升系统安全性与可维护性。

系统支持独立部署使用，也可无缝集成到企业资源规划（ERP）、客户关系管理（CRM）等大型系统中，为不同规模的应用场景提供灵活的权限管控支持。

## 功能特点

### 核心功能模块



*   **用户管理**：支持用户注册、登录、信息编辑、状态管理（启用 / 禁用）及批量操作。

*   **角色管理**：可自定义角色（如管理员、普通用户），并为角色分配权限集合，实现权限的批量管控。

*   **权限控制**：基于角色的权限分配（RBAC），支持模型级、操作级、菜单级权限控制，精确到按钮级别的权限校验。

*   **菜单管理**：动态生成前端菜单，根据用户权限显示可访问的功能入口，实现界面级权限隔离。

### 系统优势



*   **通用性强**：适配各类 Web 应用，无需重复开发权限模块。

*   **灵活扩展**：支持自定义用户字段、权限类型，满足个性化业务需求。

*   **安全可靠**：严格的权限校验机制，防止未授权访问，保障数据安全。

*   **易于集成**：提供标准化接口，可快速与 Django 项目或其他系统对接。

## 快速开始

### 环境要求



*   Python 3.8+

*   Django 3.2+

*   MySQL 5.7+ 或 PostgreSQL 12+

### 安装步骤



1.  **克隆仓库**



```
git clone https://gitee.com/liyu33/UPMS02.git

cd UPMS02
```



1.  **创建虚拟环境并安装依赖**



```
python -m venv myenv

source myenv/bin/activate  # Linux/Mac

\# 或 myenv\Scripts\activate  # Windows

pip install -r requirements.txt
```



1.  **配置数据库**

    修改 `settings.py` 中的数据库配置：



```
DATABASES = {

&#x20;   'default': {

&#x20;       'ENGINE': 'django.db.backends.mysql',

&#x20;       'NAME': 'upms\_db',

&#x20;       'USER': 'root',

&#x20;       'PASSWORD': 'your\_password',

&#x20;       'HOST': 'localhost',

&#x20;       'PORT': '3306',

&#x20;   }

}
```



1.  **初始化数据库**



```
python manage.py makemigrations

python manage.py migrate

python manage.py createsuperuser  # 创建管理员账户
```



1.  **启动服务**



```
python manage.py runserver 0.0.0.0:8000
```



1.  **访问系统**

    打开浏览器访问 `http://localhost:8000/admin`，使用管理员账户登录后即可进入后台管理界面。

## 系统架构

### 处理流程



1.  用户通过 Web 界面发起操作请求（如访问菜单、提交表单）。

2.  服务器接收请求后，调用权限校验模块，根据用户角色及权限配置进行验证。

3.  验证通过则执行操作并返回结果；验证失败则返回 "无此权限" 提示。

### 数据流程



*   用户操作数据 → 权限校验引擎 → 数据库交互 → 结果反馈给用户

*   核心数据表：`User`（用户）、`Group`（角色）、`Permission`（权限）、`Menu`（菜单）

## 技术栈



*   **后端框架**：Django + Django REST Framework

*   **认证机制**：Django 内置认证系统 + JWT（可选）

*   **前端组件**：基于 Bootstrap/Django SimpleUI（管理界面）

*   **数据库**：ORM 映射支持多数据库类型

*   **权限模型**：RBAC（基于角色的访问控制）

## 许可证

本项目采用 MIT 许可证，详情参见 [LICEN](LICENSE)[SE](LICENSE) 文件。

## 联系方式



*   项目地址：[https://gitee.com](https://gitee.com/liyu33/UPMS02)[/liyu](https://gitee.com/liyu33/UPMS02)[33/UP](https://gitee.com/liyu33/UPMS02)[MS02](https://gitee.com/liyu33/UPMS02)

*   问题反馈：提交 Issue 或联系开发者



<img width="406" height="291" alt="image" src="https://github.com/user-attachments/assets/a8e117e3-f980-4c10-93de-22ed5a44f69e" />

<img width="663" height="342" alt="image" src="https://github.com/user-attachments/assets/07569ed4-75aa-4338-964d-4f12d85ac7a3" />

图 1用户权限系统结构图

