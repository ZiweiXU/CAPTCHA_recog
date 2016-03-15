# CAPTCHA_recog
This a toy project I wrote to practice using Theano to create a lenet-like CNN, which recognizes naive CAPTCHAs 
used by my school's online system. Just a toy project to practice CNN and github :)

** If you just want to have a look at this project, downloading  ==test.py== and  ==model.dat== will be enough for a demo.**

Open your terminal, and cd to the project directory, enter
```
	python test.py
```
it will load a CAPTCHA from http://mis.teach.ustc.edu.cn/userinit.do (this is my school's online teaching system) and show recognition results.

Prerequisites for **test.py**:
- Python 2.7 (ipython is recommended for usability)
- Theano-v0.7 (no guaranteen for any other version, for Theano always changes when a new version is released).
- model.dat, this file is essential for test.py to reconstruct the trained model.
