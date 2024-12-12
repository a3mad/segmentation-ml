from flask import Blueprint, render_template, request, session, redirect, url_for, current_app
from .utils.data_preprocessing import load_and_prepare_data, REQUIRED_COLUMNS_MAP
from .utils.clustering import perform_clustering
from .utils.insights import generate_insights
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
    if request.method == 'POST':
        if 'data_file' in request.files:
            file = request.files['data_file']
            if file.filename != '':
                filename = f"upload_{uuid.uuid4().hex}.csv"
                upload_path = os.path.join('data', filename)
                file.save(upload_path)
                session['uploaded_file'] = upload_path
                return redirect(url_for('main.step3_match_columns'))
    return render_template('step2_upload_data.html')


@main.route('/step3_match_columns', methods=['GET', 'POST'])
def step3_match_columns():
    if 'uploaded_file' not in session or 'chosen_type' not in session:
        return redirect(url_for('main.step1_select_type'))

    chosen_type = session['chosen_type']
    required_cols = session['required_columns']

    try:
        df = pd.read_csv(session['uploaded_file'])
        if current_app.config.get('DEBUG_MODE', False):  # Limit rows in debug mode
            df = df.head(100)
    except Exception as e:
        return f"Error loading dataset: {str(e)}"

    dataset_columns = df.columns.tolist()  # Get column names from the dataset

    # Autoload matched columns
    matched = {req: next((col for col in dataset_columns if col.lower() == req.lower()), None) for req in required_cols}

    if request.method == 'POST':
        # Store user selections for column mapping
        for req_col in required_cols:
            chosen_col = request.form.get(req_col)
            matched[req_col] = chosen_col if chosen_col != 'None' else None

        session['column_mapping'] = matched
        return redirect(url_for('main.step3_confirm_columns'))  # Redirect to confirmation step

    return render_template('step3_match_columns.html', required_cols=required_cols, dataset_columns=dataset_columns,
                           matched=matched)


@main.route('/step3_confirm_columns', methods=['GET', 'POST'])
def step3_confirm_columns():
    if 'column_mapping' not in session or 'uploaded_file' not in session:
        return redirect(url_for('main.step1_select_type'))

    column_mapping = session['column_mapping']

    if request.method == 'POST':
        return redirect(url_for('main.step4_results'))  # Proceed to the results step

    return render_template('step3_confirm_columns.html', column_mapping=column_mapping)


@main.route('/step4_results', methods=['GET'])
def step4_results():
    if 'column_mapping' not in session or 'uploaded_file' not in session:
        return redirect(url_for('main.step1_select_type'))

    chosen_type = session['chosen_type']
    column_mapping = session['column_mapping']
    upload_path = session['uploaded_file']

    try:
        # Load and preprocess the data
        df = pd.read_csv(upload_path)
        if current_app.config.get('DEBUG_MODE', False):
            df = df.head(100)
        df_scaled, df_original = load_and_prepare_data(chosen_type, df, column_mapping)
    except ValueError as e:
        return f"Error during data preparation: {str(e)}"
    except Exception as e:
        return f"Unexpected error during data preparation: {str(e)}"

    try:
        # Perform clustering
        cluster_labels, cluster_info = perform_clustering(df_scaled)

        # Generate insights
        insights = generate_insights(df_original, cluster_labels)
    except Exception as e:
        return f"Error during clustering or insights generation: {str(e)}"

    return render_template(
        'step4_results.html',
        chosen_type=chosen_type,
        cluster_info=cluster_info,
        insights=insights
    )
