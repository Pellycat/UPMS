$(function(){
   var userId = window.euid;

   if(userId == ""){
       return false;
   }
   Cy.ajax('/community/post/userInfo', {
     type: 'GET',
     success: function(res) {
       var mob = '';
       var alias = '';

       if (res.data && !Array.isArray(res.data)) {
         mob = res.data.mob;
         alias = res.data.alias;
       }

       if(typeof(userId)=="undefined"){
         return false;
       }
       setCS(userId,mob,alias);
     }
   })
   // var mob = $("#userInfoForCs").attr("_mob");
   // var alias = $("#userInfoForCs").attr("_alias");
   //
   // if(typeof(userId)=="undefined" || typeof(alias)=="undefined"){
   //     return false;
   // }
   // setCS(userId,mob,alias);
   //setCookie("isSetCs",true);
});
function setCookie(name,value,min, noExpires){

    if (!min) {
      var min = 30;
    }
    var exp = new Date();
    exp.setTime(exp.getTime() + min*60*1000);

    if (noExpires) {
      document.cookie = name + "="+ escape (value) + ";path=/";
    } else {
      document.cookie = name + "="+ escape (value) + ";expires=" +  exp.toGMTString()+";path=/";
    }
}
function getCookie(name){
    var arr,reg=new RegExp("(^| )"+name+"=([^;]*)(;|$)");
    if(arr=document.cookie.match(reg))
    return unescape(arr[2]);
    else
    return null;
    }
function delCookie(name)
{
    var exp = new Date();
    exp.setTime(exp.getTime() - 1);
    var cval=getCookie(name);
    if(cval!=null)
    document.cookie= name + "="+cval+";expires="+exp.toGMTString()+";path=/";
    }
function setCS(user,mobile,nickName){
        _vds.push(['setCS1', 'user_id', user]);
        _vds.push(['setCS2', 'nickName', nickName]);
  //      setCookie("isSetCs","true-"+user);
}

var setRegister = function (nickName) {
    _vds.track('register',{userAlias:nickName});
};

window.jqLog = {};
window.jqLog.setRegister = setRegister;
