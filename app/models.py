from django.db import models


#FunctionInfo.objects.create(f_id="1",f_name="用户查看")
#RoleInfo.objects.create(f_id="2",role_name="用户修改")
#RoleInfo.objects.create(role_id="1",role_name="管理员")
#RoleInfo.objects.create(role_id="2",role_name="普通用户")
class FunctionInfo(models.Model):
    f_id=models.IntegerField(primary_key=True)
    f_name = models.CharField(max_length=32)

class RoleInfo(models.Model):
    role_id = models.IntegerField(primary_key=True)
    role_name = models.CharField(max_length=16)
    functions = models.ManyToManyField(FunctionInfo)


class UserInfo(models.Model):
    user_id = models.IntegerField(primary_key=True)
    user_name = models.CharField(max_length=16)
    password = models.CharField(max_length=16)
    identity= models.CharField(max_length=16,default="普通用户")
    email = models.CharField(max_length=16,null=True,blank=True)
    roles = models.ManyToManyField(RoleInfo)

