from django.db import models

#Lasso方法的预测收益率
class Lasso_Predicted_Return(models.Model):
    Date=models.DateField()
    Stock=models.CharField(max_length=100)
    Predicted_Return=models.FloatField(max_length=100)

#Ridge方法的预测收益率
class Rd_Predicted_Return(models.Model):
    Date=models.DateField()
    Stock=models.CharField(max_length=100)
    Predicted_Return=models.FloatField(max_length=100)

#EN方法的预测收益率
class EN_Predicted_Return(models.Model):
    Date=models.DateField()
    Stock=models.CharField(max_length=100)
    Predicted_Return=models.FloatField(max_length=100)

#实际收益率表
class Real_Return(models.Model):
    Date=models.DateField()
    Stock=models.CharField(max_length=100)
    Ex_Return=models.FloatField(max_length=100)

class Excess_Return_All(models.Model):
    """超额收益率和累计超额收益率表"""
    rf = models.CharField(max_length=100)
    Index_ExRet = models.CharField(max_length=100)
    Min_Var_Excess_Returns = models.CharField(max_length=100)
    Util_Max_Excess_Returns = models.CharField(max_length=100)
    Equal_Excess_Returns = models.CharField(max_length=100)
    Cumulative_Index_ExRet = models.CharField(max_length=100)
    Cumulative_Min_Var_ExRet = models.CharField(max_length=100)
    Cumulative_Util_Max_ExRet = models.CharField(max_length=100)
    Cumulative_equal = models.CharField(max_length=100)
    Cumulative_rf = models.CharField(max_length=100)
    date = models.DateField()
