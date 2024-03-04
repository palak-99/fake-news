from flask import Blueprint, request, jsonify, Response, render_template, current_app

overlay_bp = Blueprint('overlay', __name__)


@overlay_bp.route('/overlay', methods=['POST'])
def create_overlay():
    data = request.json
    overlay = Overlay.from_dict(data)
    dictt = overlay.to_dict()
    print("Overlay dict: ", dictt)
    print(mongo.db, " ttttttttttttt", mongo.db.list_collection_names())
    mongo.db.overlays.insert_one(overlay.to_dict())
    return jsonify({'message': 'Overlay created successfully'}), 201


@overlay_bp.route('/overlay', methods=['GET'])
def get_overlays():
    # overlays = mongo.db.overlays.find()
    # if overlays:
    #     return jsonify([overlay for overlay in overlays]), 200
    return jsonify({'message': 'Overlay not found'}), 404


@overlay_bp.route('/overlay/<overlay_id>', methods=['GET'])
def get_overlay(overlay_id):
    overlay = mongo.db.overlays.find_one({"_id": overlay_id})
    if overlay:
        return jsonify(overlay), 200
    return jsonify({'message': 'Overlay not found'}), 404


@overlay_bp.route('/overlay/<overlay_id>', methods=['PUT'])
def update_overlay(overlay_id):
    data = request.json
    mongo.db.overlays.update_one({"_id": overlay_id}, {"$set": data})
    return jsonify({'message': 'Overlay updated successfully'}), 200


@overlay_bp.route('/overlay/<overlay_id>', methods=['DELETE'])
def delete_overlay(overlay_id):
    mongo.db.overlays.delete_one({"_id": overlay_id})
    return jsonify({'message': 'Overlay deleted successfully'}), 200


@overlay_bp.route('/')
def index():
    return render_template('index.html')
