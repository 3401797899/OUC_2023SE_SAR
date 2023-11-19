(function (global)
{
    "use strict";

    function onMouseOver() {
    //getElementsByClassName("box")
        document.body.onmouseover = function (e){

            e = global.event || e;

            var
            //get target dom object reference
                targetDomObject = e.target || e.srcElement,
                src = "./images/img_blue.png"
            
            if (targetDomObject && targetDomObject.classList)
            {
                if (targetDomObject.classList.contains("box")) {
                    if(targetDomObject.classList.contains("first")) {
                        var 
                            box = document.getElementById("first"),
                            img = box.getElementsByTagName("img"),
                            span = box.getElementsByTagName("span")

                        box.style.border = "2px dashed skyblue"
                        img[0].src = src
                        span[0].style.color = "skyblue"

                    } else if (targetDomObject.classList.contains("second")) {
                        var 
                            box = document.getElementById("second"),
                            img = box.getElementsByTagName("img"),
                            span = box.getElementsByTagName("span")

                        box.style.border = "2px dashed skyblue"
                        img[0].src = src
                        span[0].style.color = "skyblue"
                    }
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
                //get target dom object reference
                    targetDomObject = e.target || e.srcElement,
                    src = "./images/img_black.png"
                
                if (targetDomObject && targetDomObject.classList)
                {
                    if(targetDomObject.classList.contains("first")) {
                        var 
                            box = document.getElementById("first"),
                            img = box.getElementsByTagName("img"),
                            span = box.getElementsByTagName("span")

                        box.style.border = "2px dashed"
                        img[0].src = src
                        span[0].style.color = "black"

                    } else if (targetDomObject.classList.contains("second")) {
                        var 
                            box = document.getElementById("second"),
                            img = box.getElementsByTagName("img"),
                            span = box.getElementsByTagName("span")

                        box.style.border = "2px dashed"
                        img[0].src = src
                        span[0].style.color = "black"
                    }
                }
            }
    }
    
    onMouseOver()
    onMouseLeave()
}
)(typeof window !== "undefined" ? window : this);
