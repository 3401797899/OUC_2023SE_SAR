(function (global)
{
    "use strict";

    var 
        image = document.querySelector("#image") , 
        first = document.querySelector("#first") ,
        second = document.querySelector("#second") ,
        form = document.querySelector("#form") ,
        upload = document.querySelector("#upload") ,
        flag //box边框的可见性

    function onMouseOver() {
        var 
            box = document.getElementsByClassName("box")

        for (var i = 0; i < box.length; ++i) {
            box[i].onmouseover = function (e){

                e = global.event || e;

                var
                //get target dom object reference
                    obj =  e.currentTarget || e.target
                
                if (obj && obj.classList && obj.classList.contains("box"))
                {
                    if(flag)
                        obj.style.border = "2px dashed skyblue"
                    obj.style.color = "skyblue"
                    obj.style.backgroundImage = "url(./images/img_blue.png)"
                }
            }
        }
    }

    function onMouseLeave() {
        //getElementsByClassName("box")
        var 
            box = document.getElementsByClassName("box")
            
        for (var i = 0; i < box.length; ++i)
            box[i].onmouseleave = function (e){

                e = global.event || e;
                var
                    obj = e.currentTarget || e.target
                if (obj && obj.classList)
                {
                    if(flag)
                        obj.style.border = "2px dashed"
                    obj.style.color = "black"
                    obj.style.backgroundImage = "url(./images/img_black.png)"
                }
            }
    }

    image.addEventListener("click", (e) => {
        let 
            obj = e.target
            
        if(obj.getAttribute("class") === "close"){
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

    function fileReader(e) {
        for(let item of e.target.files){
            //新建 FileReader 对象
            let reader = new FileReader();
            // 设置以什么方式读取文件，这里以base64方式
            reader.readAsDataURL(item);

            reader.onload = function(){
                // 当 FileReader 读取文件时候，读取的结果会放在 FileReader.result 属性中
                let 
                    div = document.createElement("div") ,
                    img = document.createElement("img"),
                    box = e.target.parentNode

                flag = false
                div.innerHTML = "✕"
                div.setAttribute("class", "close")
                img.setAttribute("src", this.result)
                img.setAttribute("class", "submit")
                box.appendChild(div)
                box.appendChild(img)
                box.style.border = "0px"
            };
        }
    }

    first.addEventListener("change", (e)=>{
        fileReader(e)
    })

    second.addEventListener("change", (e)=>{
        fileReader(e)
    })

    function animate(element, target){
        clearInterval(element.timer1);
        element.timer1 = setInterval( function(){
            //element是一个对象，对象点出来的属性有且只有一个，避免多次点击按钮产生多个定时器
            var current = element.offsetLeft;    //获取当前位置，数字类型，没有px。
            //不可以用element.style.left，因为该写法只能获取到行内样式，不能获取到<style></style>标签内的样式
            var step = 10;//每次移动的距离
            step = current < target ? step : -step; //step的正负表示了向左或是向右移动
            current += step;    //计算移动到的位置，数字类型，没有px                
            if ( Math.abs( target - current) > Math.abs(step) ){//当离目标位置的距离大于一步移动的距离
                element.style.left = current + "px";//移动
        
            } else {//当间距小于一步的距离，则清理定时器，把元素直接拿到目标位置
                clearInterval(element.timer1);
                element.style.left = target + "px";
            }
        },200);
    }

    //表单验证
    form.addEventListener("submit", (e) => {
        var 
            fstVal = document.forms["form"].elements["first"].value,
            secVal = document.forms["form"].elements["second"].value

        if (fstVal == "" || secVal == "") {
            alert("需要输入两个图片")
            return false
        }

        var 
            formData = new FormData(),
            form = document.querySelector("form")

        formData.append('file', $('#first')[0].files[0]);  //添加图片信息的参数
        formData.append('file', $('#second')[0].files[0]);
        // formData.append('sizeid', 123);  //添加其他参数

        $.ajax({
            url: 'https://github.com/Aurorsa/aurorsa.github.io',
            type: 'POST',
            cache: false, //上传文件不需要缓存
            data: formData,
            processData: false, // 告诉jQuery不要去处理发送的数据
            contentType: false, // 告诉jQuery不要去设置Content-Type请求头
            success: function (result) {
                var
                    data

                try {
                    data = JSON.parse(result);
                    if (typeof data.msg === "object") {
                        var
                            div = document.createElement("div"),
                            img = document.createElement("image"),
                            fstDiv = document.querySelector(".first")
                            secDiv = document.querySelector(".second")

                        //把元素移动到指定的目标位置
                        animate(fstDiv, 0);    
                        animate(secDiv, 285);

                        img.setAttribute("src", data)
                        div.appendChild(img)
                        form.appendChild(div)
                    } else { 
                        alert(data.msg)
                    }
                    console.log(data);
                } catch (error) {
                    console.error(error);
                }
            },
            error: function (data) {
                alert("上传失败");
            }
        }) 
    })

    upload.addEventListener("click", (e) => {   
        form.dispatchEvent(new Event("submit"))
    })

    onMouseOver()
    onMouseLeave()
}
)(typeof window !== "undefined" ? window : this);


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