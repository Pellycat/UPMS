"""
URL configuration for UPMS project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
#hello

from app import views
from analysis import views as analysis_views
urlpatterns = [
    #登陆验证
    path('login/', views.login, name='login'),

    path("main/", views.main),

    path("user/", views.user),
    path("user/add/", views.useradd),
    path("user/delete/", views.usrdelete),
    path('user/<int:nid>/edit/', views.useredit, name='edit_user'),

    path("role/", views.role),
    path("role/add", views.roleadd),
    path('role/delete/', views.roledelete, name='role_delete'),

    path("function/", views.function),
    path("function/add/",views.addfunction),
    path('function/<int:f_id>/edit/', views.edit_function, name='edit_function'),
    path('function/delete/', views.delete_function, name='delete_function'),

    path('user/<int:nid>/assign/', views.assignrole, name='assign_role'),
    path('role/<int:nid>/assignfunction/', views.assignfunction, name='assign_function'),
    path('<int:nid>/permission/', views.checkpermission, name='check_permission'),
    # path('analysis/',analysis_views.choice_gupiao),  # 用户输入
    path('portfolio/input',analysis_views.choice_gupiao),  # 用户输入
    path('portfolio/label/min', analysis_views.weights_min_var_label),  # 权重表展示
    path('portfolio/label/max', analysis_views.weights_util_max_label),
    path('portfolio/label/equal', analysis_views.weights_equal_label),
    path('portfolio/cumulative_excess_returns', analysis_views.cumulative_excess_returns),  # 累积收益率绘图展示
    path('portfolio/monthly_excess_returns', analysis_views.monthly_excess_returns),  # 超额收益率绘图展示
    path('portfolio/calculate_and_return_metrics_html', analysis_views.calculate_and_return_metrics_html),  # 收益概述表展示
    path('portfolio/thermodynamic_chart/min', analysis_views.thermodynamic_chart_min),  # 热力图展示
    path('portfolio/thermodynamic_chart/max', analysis_views.thermodynamic_chart_max),
    path('portfolio/thermodynamic_chart/equal', analysis_views.thermodynamic_chart_equal),
    path('portfolio/stocks_choose', analysis_views.stocks_choose)  # 股票选择展示
]
