from flask import Blueprint, render_template, request
from .utils.data_preprocessing import load_and_prepare_data
from .utils.clustering import perform_clustering
from .utils.insights import generate_insights

main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def home():
    segmentation_types = [
        'Demographic',
        'Geographic',
        'Behavioral',
        'Psychographic'
    ]
    if request.method == 'POST':
        chosen_type = request.form.get('segmentation_type')
        data = load_and_prepare_data(chosen_type)
        cluster_labels, cluster_info = perform_clustering(data)
        insights = generate_insights(data, cluster_labels)
        return render_template('results.html',
                               chosen_type=chosen_type,
                               cluster_info=cluster_info,
                               insights=insights)
    return render_template('home.html', segmentation_types=segmentation_types)
