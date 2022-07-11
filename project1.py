from flask import Flask, jsonify, request
from classifier import  get_prediction

app = Flask(__name__)

