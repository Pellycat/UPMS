from django import forms

class LoginForm(forms.Form):
    username = forms.CharField()
    password = forms.CharField(widget=forms.PasswordInput)
    identity = forms.ChoiceField(choices=[('member', '会员'), ('user', '管理')], widget=forms.RadioSelect)