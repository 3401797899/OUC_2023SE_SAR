(function(global){

    "use strict";

    var 
        image, first, second, result, form, upload, process, flag

    image = document.querySelector("#upload-img"), 
    first = document.querySelector("#first"),
    second = document.querySelector("#second"),
    result = document.querySelector("#result"),
    form = document.querySelector("#form"),
    upload = document.querySelector("#upload"),
    process = document.querySelector("progress"),
    flag = false//box边框的可见性

    
    function onMouseOver() {
        var 
            box = document.getElementsByClassName("box")

        for (var i = 0; i < box.length; ++i) {
            box[i].onmouseover = function (e) {

                e = global.event || e;

                var
                //get target dom object reference
                    obj =  e.currentTarget || e.target
                
                if (obj && obj.classList && obj.classList.contains("box"))
                {
                    if (flag)
                        obj.style.border = "2px dashed skyblue"
                    obj.style.color = "skyblue"
                    obj.style.backgroundImage = "url(./images/img_skyblue.png)"
                }
            }
        }
    }

    function onMouseLeave() {
        //getElementsByClassName("box")
        var 
            box = document.getElementsByClassName("box")
            
        for (var i = 0; i < box.length; ++i)
            box[i].onmouseleave = function (e) {

                e = global.event || e;
                var
                    obj = e.currentTarget || e.target
                if (obj && obj.classList)
                {
                    if (flag)
                        obj.style.border = "2px dashed"
                    obj.style.color = "black"
                    obj.style.backgroundImage = "url(./images/img_black.png)"
                }
            }
    }

    function download(link){
        const xhr = new XMLHttpRequest();
        xhr.open('GET',link)
        xhr.responseType = 'blob'
        xhr.send()
        xhr.onload = function () {
            const fileBlob = xhr.response;
            console.log(fileBlob)
            const fileUrl = URL.createObjectURL(fileBlob)
            console.log(fileUrl)
            const ele = document.createElement('a')
            ele.setAttribute('href',fileUrl)
            ele.setAttribute('download',"")
            ele.click()
        }
    }

    function onClick() 
    {

        document.body.onclick = function (e) 
        {

            //get event object (window.event for IE compatibility)
            e = global.event || e;

            var
                //get target dom object reference
                targetDomObject = e.target || e.srcElement,

                parentNode = targetDomObject.parentNode,
                //other variables
                i, boxs, img, results, input, fstVal, secVal, rstVal, formData, id, close;

            //extra checks to make sure object exists and contains the class of interest
            if (targetDomObject && targetDomObject.classList)
            {
                if (targetDomObject.classList.contains("download") || parentNode.getAttribute("class") == "download") 
                {
                    results = document.querySelectorAll(".submit")
                    for (var res of results) {
                        download(res.getAttribute("src"));
                    }
                }
                //extra checks to make sure object exists and contains the class of interest
                if (targetDomObject.classList.contains("refresh") || parentNode.getAttribute("class") == "refresh") 
                {
                    boxs = document.querySelectorAll(".box")
                    for(var i = boxs.length - 1; i >= 0; i--) {
                        let box = boxs[i];
                        img = box.querySelector(".submit")
                        close = box.querySelector(".close")
                        input = box.querySelector(".input")
                        if (img != null) {
                            img.remove()
                            close && close.remove()
                        }
                        if (input != null)
                            input.value = ""
                        box.style.border = "2px dashed"
                    }
                    
                    flag = false
                }

                if (targetDomObject.getAttribute("class") == "accuracy btn") 
                {
                    document.getElementsByClassName("loading")[0].style.display = "block";
                    fstVal = document.forms["form"].elements["first"].value
                    secVal = document.forms["form"].elements["second"].value
                    rstVal = document.forms["form"].elements["result"].value
        
                    if (fstVal == "" || secVal == "" || rstVal == "") {
                        alert("需要输入三张图片")
                        return false
                    }
                    
                    formData = new FormData()
            
                    formData.append('img1', $('#first')[0].files[0]);  //添加图片信息的参数
                    formData.append('img2', $('#second')[0].files[0]);
                    formData.append('img_gt', $('#result')[0].files[0]);

                    $.ajax({
                        url: 'https://sar.imbai.cn/accuracy/',
                        type: 'POST',
                        cache: false, //上传文件不需要缓存
                        data: formData,
                        processData: false, // 告诉jQuery不要去处理发送的数据
                        contentType: false, // 告诉jQuery不要去设置Content-Type请求头
                        success: function (data) {
                            document.getElementsByClassName("loading")[0].style.display = "none";
                            if (data.status == "ok") {
                                var accuracy = document.getElementById("accuracy");
                                accuracy.innerHTML = data.msg;
                            }else{
                                alert(data.msg);
                            }
                        },
                        error: function (data) {
                            alert("上传失败");
                        }
                    })     
                    
                }

                if (targetDomObject.getAttribute("class") == "upload btn") 
                {
                    fstVal = document.forms["form"].elements["first"].value
                    secVal = document.forms["form"].elements["second"].value
        
                    if (fstVal == "" || secVal == "") {
                        alert("需要输入两张图片")
                        return false
                    }
            
                    formData = new FormData()
            
                    formData.append('img1', $('#first')[0].files[0]);  //添加图片信息的参数
                    formData.append('img2', $('#second')[0].files[0]);
                    $.ajax({
                        url: 'https://sar.imbai.cn/sar/',
                        type: 'POST',
                        cache: false, //上传文件不需要缓存
                        data: formData,
                        processData: false, // 告诉jQuery不要去处理发送的数据
                        contentType: false, // 告诉jQuery不要去设置Content-Type请求头
                        success: function (data) {
                            try {
                                if (data.status == "ok") {
                                    id = data.id
                                    setTimeout(getApi, 5000, id)
                                } else { 
                                    alert(data.status)
                                }
                            } catch (error) {
                                console.error(error)
                            }
                        },
                        error: function (data) {
                            alert("上传失败");
                        }
                    })     
                }
            }
        }
    }
      

    image.addEventListener("click", (e) => {
        let 
            obj = e.target
            
        if (obj.getAttribute("class") === "close") {
            let 
                box = obj.parentNode,
                img = box.querySelector("img"),
                file = box.querySelector("input")
            
            flag = true
            img.remove()
            obj.remove()
            file.value = "" //清除input的file
            box.style.border = "2px dashed"
        }
    })

    /**
     * 
     * @param {*} e 
     */
    function fileReader(e) {
        var 
            div, img, box

        for (let item of e.target.files) {
            //新建 FileReader 对象
            let reader = new FileReader();
            reader.readAsDataURL(item);

            reader.onload = function() {
                // 当 FileReader 读取文件时候，读取的结果会放在 FileReader.result 属性中
                div = document.createElement("div")
                img = document.createElement("img")
                box = e.target.parentNode

                flag = false
                div.innerHTML = "✕"
                div.setAttribute("class", "close")
                img.setAttribute("src", this.result)
                img.setAttribute("class", "submit")
                img.style.display = 'block';
                box.appendChild(div)
                box.appendChild(img)
                box.style.border = "0px"
            }
        }
    }

    first.addEventListener("change", (e) => {
        fileReader(e)
    })

    second.addEventListener("change", (e) => {
        fileReader(e)
    })

    result && result.addEventListener("change", (e) => {
        fileReader(e)
    })

    function getCurrentDateTime() {
        const now = new Date();
        const year = now.getFullYear();
        const month = String(now.getMonth() + 1).padStart(2, '0');
        const day = String(now.getDate()).padStart(2, '0');
        const hours = String(now.getHours()).padStart(2, '0');
        const minutes = String(now.getMinutes()).padStart(2, '0');
      
        const formattedDateTime = `${year}-${month}-${day} ${hours}:${minutes}`;
        return formattedDateTime;
      }

    function getApi(id) {
        //设置时间 5-秒  1000-毫秒  这里设置你自己想要的时间 
        $.ajax({
            url: 'https://sar.imbai.cn/sar/',
            type: 'GET',
            data: {
                id:id
            },
            success: function (data) {
                // console.log(typeof data.msg)
                let history = localStorage.getItem('history') || '[]';
                history = JSON.parse(history);
                let id_elm = history.filter((item) => item.id === id)[0];
                let flag = history.filter((item) => item.id === id)[0] ? true : false;
                if (typeof data.msg == 'number') {  
                    document.getElementsByClassName('process')[0].innerHTML='loading...';
                    process.value = data.msg
                    setTimeout(getApi, 1000, id)
                    if(id_elm){
                        id_elm.msg = data.msg;
                    }else{
                        id_elm = {
                            id:id,
                            msg:data.msg,
                            time: getCurrentDateTime()
                        }
                    }
                }else if(data.msg == '等待队列中...'){
                    document.getElementsByClassName('process')[0].innerHTML='等待中';
                    setTimeout(getApi, 1000, id)
                    return;
                } else {
                    document.getElementsByClassName('process')[0].innerHTML='ok';
                    var image = document.getElementsByClassName("submit")
                    image[0].src = data.msg[0]
                    image[1].src = data.msg[1]
                    image[0].style.display = 'block';
                    image[1].style.display = 'block';
                    id_elm.msg = data.msg;
                }
                if(!flag){
                    history.push(id_elm);
                }
                localStorage.setItem('history', JSON.stringify(history));
            }
        })
    }

    upload.addEventListener("click", (e) => {   
        form.dispatchEvent(new Event("submit"))
    })

    document.getElementsByClassName('nav-link')[0].addEventListener("click", (e) => {
        document.querySelectorAll(".menu-links li a").forEach((item) => {
            item.style.background = "transparent";
        });
        let target = e.target;
        while(target.tagName != 'LI'){
            target = target.parentElement;
        }
        target.getElementsByTagName('a')[0].style.background = "skyblue";
        var image = document.getElementsByClassName("submit");
        if(image.length){
            for(var i = image.length - 1; i >= 0; i--){
                image[i].parentElement.style.border = "2px dashed"
                image[i].parentElement.getElementsByClassName("close").length && image[i].parentElement.getElementsByClassName("close")[0].remove();
                image[i].remove();
            }
        }
    })

    function showhistory() {
        let history = localStorage.getItem('history') || '[]';
        history = JSON.parse(history);
        if (history.length > 0) {
            var 
                idx, id, time, i, span,
                ul = document.querySelector(".menu-links")

            for (idx = 0; idx < history.length; ++idx) {
                id = history[idx].id
                time = history[idx].time
                let a = document.createElement("a")
                i = document.createElement("i")
                let li = document.createElement("li")
                span = document.createElement("span")

                i.setAttribute("class", "iconfont")
                span.setAttribute("class", "text")
                span.innerHTML = time
                li.setAttribute("value", id)

                li.addEventListener("click", (e) => {
                    var 
                        target = e.target, 
                        id = null;
                    while(target.tagName != 'LI'){
                        target = target.parentElement;
                    }
                    id = target.getAttribute("value");

                    document.querySelectorAll(".menu-links li a").forEach((item) => {
                        item.style.background = "transparent";
                    });
                    li.getElementsByTagName('a')[0].style.background = "skyblue";
                    var image = document.getElementsByClassName("submit");
                    if(image.length != 2){
                        for(var i = image.length - 1; i >= 0; i--){
                            image[i].remove();
                        }
                        let box = document.getElementsByClassName('box');
                        for(let item of box){
                            let img = document.createElement("img");
                            img.setAttribute("class", "submit");
                            img.setAttribute("src", "");
                            item.appendChild(img);
                        }
                        image = document.getElementsByClassName("submit");
                    }
                    image[0].setAttribute('src',`https://sar.imbai.cn/result/${id}-1.png`);
                    image[1].setAttribute('src',`https://sar.imbai.cn/result/${id}-2.png`);
                    
                })

                a.appendChild(i)
                a.appendChild(span)
                li.appendChild(a)
                ul.appendChild(li)
            }
        }
    }

    if (upload.getAttribute("class") === "upload btn") {
        showhistory();
    }
    onClick()
    onMouseOver()
    onMouseLeave()
    
})(typeof window !== "undefined" ? window : this);

//表单验证
// form.addEventListener("submit", (e) => {
//     var 
//         fstVal = document.forms["form"].elements["first"].value,
//         secVal = document.forms["form"].elements["second"].value

//     if (fstVal == "" || secVal == "") {
//         alert("需要输入两个图片")
//         return false
//     }

//     var 
//         formData = new FormData(), id

//     formData.append('img1', $('#first')[0].files[0]);  //添加图片信息的参数
//     formData.append('img2', $('#second')[0].files[0]);

//     $.ajax({
//         url: 'https://sar.imbai.cn/sar/',
//         type: 'POST',
//         cache: false, //上传文件不需要缓存
//         data: formData,
//         processData: false, // 告诉jQuery不要去处理发送的数据
//         contentType: false, // 告诉jQuery不要去设置Content-Type请求头
//         success: function (data) {
//             try {
//                 if (data.status == "ok") {
//                     id = data.id
//                     const moment = require('moment');
//                     const formattedTime = moment().format('YYYYMMDDHHmmss');
//                     localStorage.setItem(id,formattedTime)
//                     setTimeout(getApi, 5000, id)
//                 } else { 
//                     alert(data.status)
//                 }
//             } catch (error) {
//                 console.error(error)
//             }
//         },
//         error: function (data) {
//             alert("上传失败");
//         }
//     })     
// })

// var
//                             div = document.createElement("div"),
//                             img = document.createElement("image"),
//                             fstDiv = document.querySelector(".first")
//                             secDiv = document.querySelector(".second")

//                         //把元素移动到指定的目标位置
//                         animate(fstDiv, 0);    
//                         animate(secDiv, 285);

//                         img.setAttribute("src", data)
//                         div.appendChild(img)
//                         form.appendChild(div)

// if(obj.getAttribute("class") === "box") {
        //     for (let item of obj.files){
        //         //新建 FileReader 对象
        //         let reader = new FileReader();
        //         // 设置以什么方式读取文件，这里以base64方式
        //         reader.readAsDataURL(item);
        //         reader.onload = function(){
        //             // 当 FileReader 读取文件时候，读取的结果会放在 FileReader.result 属性中
        //             let 
        //                 div = document.createElement("div") ,
        //                 img = document.createElement("img")

        //             div.innerHTML = "✕"
        //             div.setAttribute("class", "close")
        //             img.setAttribute("src", this.result)
        //             img.setAttribute("class", "submit")
        //             obj.appendChild(div)
        //             obj.appendChild(img)
        //         };
        //     }
        // }