# 部署到 Vercel（免费，不用银行卡）

HF 的 Docker Space 现在要 Pro；Render 要绑卡。Vercel 免费版支持 Python Serverless，不用卡。

## 这个版本能做什么

- 单票七维打分卡（手机全屏）
- 输入代码/名称切换、30 秒自动刷新、添加到主屏幕
- 不包含：全市场扫描、板块扫描、相似库（那些需要常驻服务）

## 1. 上传到 GitHub

新建仓库 `stockcheck-vercel`（Public/Private 都行），把**本文件夹里的内容**传上去：

```
api/index.py
public/hud.html
public/index.html
public/manifest.webmanifest
public/icon-192.png
public/icon-512.png
stockcheck/            Python 包
requirements.txt
README.md
VERCEL.md
```

## 2. Vercel 导入

1. [vercel.com](https://vercel.com) → GitHub 登录（免费 Hobby，不用卡）；
2. **Add New → Project** → 选 `stockcheck-vercel` 仓库；
3. **Root Directory** 保持默认（仓库根）；
4. Framework Preset：Other；
5. 点 **Deploy**。

第一次构建约 2~4 分钟（装 pandas/numpy）。

## 3. 验证

```
https://你的项目.vercel.app/api/index?action=health
```

返回 `{"ok":true,...}` 即可。

手机打开：

```
https://你的项目.vercel.app/hud.html?code=600519
```

分享 → 添加到主屏幕。

## 注意

- Serverless 冷启动约 1~3 秒，正常；
- 每次请求实时拉东财，不存数据，没有数据库；
- 地址公开，但只有行情打分，没有个人数据；
- 免费额度：Hobby 100GB 带宽/月，个人用足够。
