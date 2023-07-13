#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 13:47:52 2023

@author: insanesac
"""

from flask import Flask, request

app = Flask(__name__)

@app.route('/', methods = ['POST'])
def add_two():
    # x = int(request.args.get('a'))
    # y = int(request.args.get('b'))
    
    x = int(request.form["x"])
    y = int(request.form["y"])
    return str(x+y)

# print(add_two(10, 20))

if __name__ == '__main__':
    app.run(port=8080)