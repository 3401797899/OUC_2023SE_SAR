# Backend相关介绍

## 部署方式
1. 下载项目源代码后，执行
   ```bash
   pip install -r requirements.txt
   ```
   安装项目相关依赖。

2. 确保redis正常运行并启动celery
   ```bash
   celery -A OUC_SE_G3_SAR worker -l info
   ```
   如果是windows请加上`--pool=solo`
   ```bash
   celery -A OUC_SE_G3_SAR worker -l info --pool=solo
   ```
   
3. 运行项目
   ```bash
   python manage.py runserver
   ```
   
4. 默认端口为8000，访问http://localhost:8000/ 即可。

## 项目接口文档

链接: https://apifox.com/apidoc/shared-7d2aa47f-db2b-4b2c-be12-54b9a80153c6

访问密码 : KA4qgpRs

