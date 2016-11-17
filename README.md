README.mdの書き方(マークダウン記法について)
=====================

参考文献
* <https://ja.wikipedia.org/wiki/Markdown>
* <http://qiita.com/Qiita/items/c686397e4a0f4f11683d#2-9>
* <http://codechord.com/2012/01/readme-markdown/>
* <http://qiita.com/oreo/items/82183bfbaac69971917f>

--------------------------

gitの基本操作
====================

参考文献
* <http://www.yoheim.net/blog.php?q=20140104>

--------------------------

Command line instructions
==================

## Git global setup

```
git config --global user.name "Ammae"
git config --global user.email "ammae.osamu@ist.osaka-u.ac.jp"
```

## Create a new repository

```
git clone https://luna.ise.eng.osaka-u.ac.jp/ammae/learningbasedphotometricstereo.git
cd tmp
touch README.md
git add README.md
git commit -m "add README"
git push -u origin master
```

## Existing folder or Git repository

```
cd existing_folder
git init
git remote add origin https://luna.ise.eng.osaka-u.ac.jp/ammae/learningbasedphotometricstereo.git
git add .
git commit
git push -u origin master
```
