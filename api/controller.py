from api.general import build_response
from flask import Blueprint, request, jsonify
from api.service import filter_uri

simple_page = Blueprint('simple_page', __name__, template_folder='templates')


@simple_page.route('/api/filter/uri', methods=['POST'])
def post_filter_uri():
    data = request.get_json()
    predictions = filter_uri(data['images'])
    return jsonify(build_response(data['images'], predictions))


@simple_page.route('/api/filter/uri', methods=['GET'])
def get_filter_uri():
    data = {"images": [{"image": request.args['image']}]}
    predictions = filter_uri(data["images"])
    return jsonify(build_response(predictions))