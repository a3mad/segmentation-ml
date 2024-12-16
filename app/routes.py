from flask import Blueprint, render_template, request, session, redirect, url_for, current_app
from .utils.data_preprocessing import load_and_prepare_data, REQUIRED_COLUMNS_MAP
from .utils.clustering import perform_clustering
from .utils.cluster_labeling import generate_cluster_labels
from .utils.graph_utils import generate_graphs
import pandas as pd
import os
import uuid

main = Blueprint('main', __name__)


@main.before_request
def make_session_permanent():
    session.permanent = True


@main.route('/', methods=['GET', 'POST'])
def step1_select_type():
    segmentation_types = list(REQUIRED_COLUMNS_MAP.keys())
    if request.method == 'POST':
        chosen_type = request.form.get('segmentation_type')
        if chosen_type in segmentation_types:
            session['chosen_type'] = chosen_type
            required_cols = REQUIRED_COLUMNS_MAP[chosen_type]
            session['required_columns'] = required_cols
            return redirect(url_for('main.step2_upload_data'))
    return render_template('step1_select_type.html', segmentation_types=segmentation_types)


@main.route('/step2_upload_data', methods=['GET', 'POST'])
def step2_upload_data():
    if 'chosen_type' not in session:
        return redirect(url_for('main.step1_select_type'))

    # Retrieve the required columns info for the chosen segmentation type
    chosen_type = session['chosen_type']
    required_columns_info = REQUIRED_COLUMNS_MAP.get(chosen_type, {}).get('columns', {})

    if request.method == 'POST':
        if 'data_file' in request.files:
            file = request.files['data_file']
            if file.filename != '':
                filename = f"upload_{uuid.uuid4().hex}.csv"
                upload_path = os.path.join('data', filename)
                file.save(upload_path)
                session['uploaded_file'] = upload_path
                return redirect(url_for('main.step3_match_columns'))

    # Pass required columns info to the template
    return render_template('step2_upload_data.html', required_columns_info=required_columns_info)


@main.route('/step3_match_columns', methods=['GET', 'POST'])
def step3_match_columns():
    if 'uploaded_file' not in session or 'chosen_type' not in session:
        return redirect(url_for('main.step1_select_type'))

    chosen_type = session['chosen_type']
    required_cols = list(REQUIRED_COLUMNS_MAP[chosen_type]['columns'].keys())

    try:
        df = pd.read_csv(session['uploaded_file'])
        if current_app.config.get('DEBUG_MODE', False):
            df = df.head(100)
    except Exception as e:
        return f"Error loading dataset: {str(e)}"

    dataset_columns = df.columns.tolist()

    # Autoload matched columns
    matched = {req: next((col for col in dataset_columns if col.lower() == req.lower()), None) for req in required_cols}
    additional_columns = []

    if request.method == 'POST':
        # Handle default column mapping
        for req_col in required_cols:
            chosen_col = request.form.get(req_col)
            matched[req_col] = chosen_col if chosen_col != 'None' else None

        # Handle additional columns selected by the user
        additional_columns = request.form.getlist('additional_columns')
        session['additional_columns'] = additional_columns

        # Store column mapping and redirect to confirmation step
        session['column_mapping'] = matched
        return redirect(url_for('main.step3_confirm_columns'))

    return render_template(
        'step3_match_columns.html',
        required_cols=required_cols,
        dataset_columns=dataset_columns,
        matched=matched,
        segmentation_info=REQUIRED_COLUMNS_MAP[chosen_type],
        additional_columns=additional_columns
    )



@main.route('/step3_confirm_columns', methods=['GET', 'POST'])
def step3_confirm_columns():
    if 'column_mapping' not in session or 'uploaded_file' not in session:
        return redirect(url_for('main.step1_select_type'))

    column_mapping = session['column_mapping']
    additional_columns = session.get('additional_columns', [])

    if request.method == 'POST':
        return redirect(url_for('main.step4_results'))

    return render_template(
        'step3_confirm_columns.html',
        column_mapping=column_mapping,
        additional_columns=additional_columns
    )



@main.route('/step4_results', methods=['GET'])
def step4_results():
    if 'column_mapping' not in session or 'uploaded_file' not in session:
        return redirect(url_for('main.step1_select_type'))

    chosen_type = session['chosen_type']
    column_mapping = session['column_mapping']
    upload_path = session['uploaded_file']

    try:
        # Load the dataset
        df = pd.read_csv(upload_path)
        if current_app.config.get('DEBUG_MODE', False):
            df = df.head(100)

        # Prepare the data
        df_scaled, df_original, encoders = load_and_prepare_data(chosen_type, df, column_mapping, session)

    except ValueError as e:
        return f"Error during data preparation: {str(e)}"
    except Exception as e:
        return f"Unexpected error during data preparation: {str(e)}"

    try:
        # Perform clustering
        cluster_labels, cluster_info = perform_clustering(df_scaled)

        # Generate meaningful cluster labels
        cluster_descriptions = generate_cluster_labels(
            cluster_data=df_original,
            cluster_labels=cluster_labels,
            metadata={
                "encoders": encoders,
                "additional_columns": session.get('additional_columns', []),
                "required_columns": REQUIRED_COLUMNS_MAP[chosen_type]['columns']
            }
        )

        # Add cluster labels to the data
        df_original['Cluster'] = cluster_labels
        df_original['Cluster Label'] = df_original['Cluster'].map(cluster_descriptions)

        # Generate graphs using the new utility function
        graph_save_path = os.path.join(current_app.root_path, 'static', 'images')
        graph_files = generate_graphs(df_original, graph_save_path)

        # Prepare cluster info for the template
        cluster_info = {
            "n_clusters": cluster_info['n_clusters'],
            "silhouette_score": cluster_info['silhouette_score'],
            "clusters": [
                {
                    "id": cluster_id,
                    "label": cluster_descriptions[cluster_id],
                    "num_customers": len(df_original[df_original['Cluster'] == cluster_id])
                }
                for cluster_id in sorted(cluster_descriptions.keys())
            ],
            "columns_used": list(column_mapping.values()) + session.get('additional_columns', [])
        }

    except Exception as e:
        return f"Error during clustering or insights generation: {str(e)}"

    return render_template(
        'step4_results.html',
        chosen_type=chosen_type,
        cluster_info=cluster_info,
        graph_files=graph_files
    )
