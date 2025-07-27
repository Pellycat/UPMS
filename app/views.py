
from django.contrib import messages
from .models import UserInfo,RoleInfo,FunctionInfo

from django.views.decorators.http import require_http_methods
from django.shortcuts import render, redirect, get_object_or_404
from analysis import templates as analysis_templates
global_name="管理员"
def login(request):
    if request.method == 'POST':
        user_id = request.POST.get("user")
        password = request.POST.get("pwd")
        identity = request.POST.get("identity")

        try:
            user = UserInfo.objects.get(user_id=user_id)
            if user.password == password:
                if user.identity == '管理员' and identity == 'user':
                    return redirect('/main/')  # 管理员页面
                elif user.identity == '普通用户' and identity == 'member':
                    return redirect('/portfolio/input')
                else:
                    messages.error(request, "身份不匹配。")
            else:
                messages.error(request, "密码错误。")
        except UserInfo.DoesNotExist:
            messages.error(request, "用户不存在。")

    return render(request, 'login.html')


def main(request):
    return render(request,
                  'main.html',
                  {"n1":global_name})

def user(request):
    #获取数据库的所有用户
    data_list=UserInfo.objects.all()
    print(data_list)
    return render(request, 'user.html', {'data_list': data_list})


def useradd(request):
    if request.method == 'GET':
        return render(request, 'add.html')
    if request.method == 'POST':
        id=request.POST.get("add_id")
        name = request.POST.get("add_name")
        password = request.POST.get("add_pwd")
        email=request.POST.get("add_email")
        #添加到数据库
        UserInfo.objects.create(user_id=id, user_name=name, password=password, email=email)
        return redirect("/user/")

def usrdelete(request):
    nid=request.GET.get("nid")
    UserInfo.objects.filter(user_id=nid).delete()
    return redirect("/user/")


def useredit(request, nid):
    # 获得要编辑的用户对象
    user_object = UserInfo.objects.filter(user_id=nid).first()
    if request.method == 'POST':
        # 表单数据处理
        user_name = request.POST.get('user_name')
        password = request.POST.get('password')
        email = request.POST.get('email')
        # 更新数据
        if user_object:
            user_object.user_name = user_name
            user_object.password = password
            user_object.email = email
            user_object.save()
            return redirect('/user/')  # 重定向到其他页面，如用户列表页
    # GET 请求，显示现有数据
    return render(request, 'edit.html', {'user_object': user_object})



def role(request):
    data_list = RoleInfo.objects.all()
    print(data_list)
    return render(request, 'role_converted.html', {'data_list': data_list})

def roleadd(request):
    if request.method == 'GET':
        return render(request, 'addrole.html')
    if request.method == 'POST':
        id = request.POST.get("add_id")
        name = request.POST.get("add_name")
        # 添加到数据库
        RoleInfo.objects.create(role_id=id, role_name=name)
        return redirect("/role/")


@require_http_methods(["POST"])
def roledelete(request):
    nid = request.POST.get("nid")
    RoleInfo.objects.filter(role_id=nid).delete()
    return redirect("/role/")




def function(request):
    data_list = FunctionInfo.objects.all()
    print(data_list)
    return render(request, 'function.html', {'data_list': data_list})

def addfunction(request):
    if request.method == 'GET':
        return render(request, 'addfunction.html')
    if request.method == 'POST':
        id=request.POST.get("add_id")
        name = request.POST.get("add_name")
        #添加到数据库
        FunctionInfo.objects.create(f_id=id, f_name=name)
        return redirect("/function/")
def delete_function(request):
    nid = request.GET.get("nid")
    FunctionInfo.objects.filter(f_id=nid).delete()
    return redirect("/function/")


def edit_function(request, f_id):
    # 获得要编辑的用户对象
    f_object = FunctionInfo.objects.filter(f_id=f_id).first()
    if request.method == 'POST':
        # 表单数据处理
        f_id=request.POST.get("f_id")
        f_name = request.POST.get('f_name')
        # 更新数据
        if f_object:
            f_object.f_id = f_id
            f_object.f_name = f_name
            f_object.save()
            return redirect('/function/')  # 重定向到其他页面，如用户列表页
    # GET 请求，显示现有数据
    return render(request, 'functionedit.html', {'f_object': f_object})



#分配角色
def assignrole(request, nid):
    user = get_object_or_404(UserInfo, user_id=nid)
    if request.method == 'POST':
        selected_role_ids = request.POST.getlist('roles')
        print("Selected roles:", selected_role_ids)  # Debugging line
        selected_roles = RoleInfo.objects.filter(role_id__in=selected_role_ids)
        user.roles.set(selected_roles)
        return redirect(f'/user/{nid}/assign/')
    else:
        all_roles = RoleInfo.objects.all()
        assigned_roles = user.roles.all()
        unassigned_roles = all_roles.difference(assigned_roles)

    return render(request, 'assignRole.html', {
        'user': user,
        'assigned_roles': assigned_roles,
        'unassigned_roles': unassigned_roles
    })

#给角色分配功能
def assignfunction(request, nid):
    role = get_object_or_404(RoleInfo, role_id=nid)  # 获取指定角色
    all_functions = FunctionInfo.objects.all()  # 获取所有功能
    assigned_functions = role.functions.all()  # 获取角色已分配的功能
    unassigned_functions = all_functions.difference(assigned_functions)  # 计算未分配的功能

    if request.method == 'POST':
        # 从提交的表单中获取选中的功能ID
        selected_function_ids = request.POST.getlist('functions')
        print("selected functions:", selected_function_ids)
        # 根据ID获取功能对象
        selected_functions = FunctionInfo.objects.filter(f_id__in=selected_function_ids)
        # 更新角色的功能
        role.functions.set(selected_functions)
        return redirect(f'/role/{nid}/assignfunction')  # 提交后重定向到角色列表

    return render(request, 'assignPermission.html', {
        'role': role,
        'assigned_functions': assigned_functions,
        'unassigned_functions': unassigned_functions
    })


from django.http import JsonResponse


def checkpermission(request, nid):
    user = get_object_or_404(UserInfo, user_id=nid)
    function_id = request.GET.get('function_id')

    # 检查用户是否有这个功能的权限
    if function_id:
        has_permission = user.roles.filter(functions__f_id=function_id).exists()
        if has_permission:
            return JsonResponse({"message": "您拥有相关权限，可以使用该功能。"}, safe=False)
        else:
            return JsonResponse({"message": "您没有相关权限。"}, safe=False)

    roles = user.roles.all()
    functions = FunctionInfo.objects.filter(roleinfo__in=roles).distinct()
    all_functions = FunctionInfo.objects.all()  # 获取数据库中所有的功能
    return render(request, 'member.html', {
        'user': user,
        'roles': roles,
        'functions': functions,
        'nid': nid,
        'user_name':user.user_name,
        'all_functions': all_functions
    })

