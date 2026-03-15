from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, accuracy_score
from scipy.stats import pearsonr


def average_scores_by_subject_id(all_subject_ids, all_true, all_pred, all_pred_probas=None):
    """
    Average predictions and true values by subject ID, with consistent sorting.

    Args:
        all_subject_ids: List of subject IDs (may contain duplicates)
        all_true: List of true values corresponding to all_subject_ids
        all_pred: List of predictions corresponding to all_subject_ids
        all_pred_probas: Optional list of probability predictions

    Returns:
        unique_ids: Array of unique subject IDs (sorted)
        avg_true: Array of averaged true values per unique ID
        avg_pred: Array of averaged predictions per unique ID
        avg_pred_probas: Array of averaged probability predictions (if provided)
    """
    # Convert inputs to numpy arrays
    all_subject_ids = np.array(all_subject_ids)
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    # Get unique subject IDs and sort them for consistency
    unique_ids = np.unique(all_subject_ids)
    # Sort the unique IDs to ensure consistent ordering
    unique_ids = np.sort(unique_ids)

    # Initialize arrays for averaged values
    avg_true = np.zeros(len(unique_ids))
    avg_pred = np.zeros(len(unique_ids))

    # Average values for each unique ID
    for i, uid in enumerate(unique_ids):
        # Find all occurrences of this ID
        mask = (all_subject_ids == uid)

        # Average true values and predictions
        avg_true[i] = np.mean(all_true[mask])
        avg_pred[i] = np.mean(all_pred[mask])

    return unique_ids, avg_true, avg_pred



def run_cross_validated_ridge_probe(
        embeddings: np.ndarray,
        labels: list,
        ids: list,
        activities: list,
        label_idx: int,
        task_type: str = 'reg',
        activity_to_run: str = None,
        n_folds: int = 3,
        alpha: float = 1.0,
):
    """
    Run k-fold cross-validated ridge regression probe on embeddings.

    Splits are stratified by subject ID to avoid data leakage.

    Args:
        embeddings: Array of shape (N, D) containing pooled embeddings
        labels: List of label arrays, one per label type
        ids: List of subject IDs corresponding to each embedding
        activities: List of activity names corresponding to each embedding
        label_idx: Index into labels list for the target label
        activities_to_run: Optional list of activities to filter by
        n_folds: Number of cross-validation folds
        alpha: Ridge regularization parameter

    Returns:
        Dictionary with mean and std of MAE, MSE, R2, and Pearson correlation
    """
    # Filter by activity if specified
    if activity_to_run is not None:
        mask = np.array([act == activity_to_run for act in activities])
        embeddings = embeddings[mask]
        ids = [id_ for id_, m in zip(ids, mask) if m]
        activities = [act for act, m in zip(activities, mask) if m]
        label_values = labels[label_idx]
        y = np.array([label for label, m in zip(label_values, mask) if m])
    else:
        y = np.array(labels[label_idx])

    # Remove NaN labels
    valid_mask = ~np.isnan(y)
    embeddings = embeddings[valid_mask]
    y = y[valid_mask]
    ids = [id_ for id_, m in zip(ids, valid_mask) if m]
    activities = [act for act, m in zip(activities, valid_mask) if m]

    if len(embeddings) == 0:
        raise ValueError("No embeddings to run probe on.")
    if len(np.unique(y)) < 2:
        print(f"Not enough unique labels for {label_idx} type {task_type}, {activity_to_run}.")
        return None

    # Group indices by subject ID
    unique_ids = sorted(set(ids))
    id_to_indices = {id_: [] for id_ in unique_ids}
    for idx, id_ in enumerate(ids):
        id_to_indices[id_].append(idx)

    # Create splits based on subject IDs
    id_array = np.array(unique_ids)
    
    # For classification, use stratified k-fold and check minimum samples per class
    if task_type == 'clas':
        # Get one label per subject (take first occurrence)
        id_to_label = {}
        for id_, label in zip(ids, y):
            if id_ not in id_to_label:
                id_to_label[id_] = label
        y_per_subject = np.array([id_to_label[id_] for id_ in unique_ids])
        
        # Check if we have enough samples of each class for k-fold
        unique_classes, class_counts = np.unique(y_per_subject, return_counts=True)
        min_class_count = class_counts.min()
        if min_class_count < n_folds:
            print(f"Not enough samples per class for {n_folds}-fold CV: min={min_class_count}, {activity_to_run}.")
            return None
        
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        split_iter = kf.split(id_array, y_per_subject)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        split_iter = kf.split(id_array)

    predictions = {"y_test": [], "y_pred": []}
    pred_indices = []

    for train_id_idx, test_id_idx in split_iter:
        train_ids = id_array[train_id_idx]
        test_ids = id_array[test_id_idx]

        # Get indices for train and test samples
        train_indices = []
        test_indices = []
        for id_ in train_ids:
            train_indices.extend(id_to_indices[id_])
        for id_ in test_ids:
            test_indices.extend(id_to_indices[id_])

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        X_train = embeddings[train_indices]
        X_test = embeddings[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Fit ridge regression
        if task_type == 'clas':
            ridge = LogisticRegression(C=1.0/alpha, max_iter=1000)
        else:
            ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)

        # Predict and compute metrics
        if task_type == 'clas':
            y_pred = ridge.predict_proba(X_test)[:, 1]
        else:
            y_pred = ridge.predict(X_test)

        # Store predictions for this fold
        predictions["y_test"].extend(y_test)
        predictions["y_pred"].extend(y_pred)
        pred_indices.extend(test_indices)

    unique_ids, avg_y_test, avg_y_pred = average_scores_by_subject_id(
        np.array(ids)[pred_indices],
        predictions["y_test"],
        predictions["y_pred"]
    )
    avg_predictions = {"y_test": avg_y_test, "y_pred": avg_y_pred, "unique_ids": unique_ids}

    if task_type == 'clas':
        # make sure there are at exactly two classes in avg_y_test
        if len(np.unique(avg_y_test)) != 2:
            raise ValueError ("Not 2 classes in test for classification metrics.")
        avg_y_test = avg_y_test.astype(int)

        auc = roc_auc_score(avg_y_test, avg_y_pred)
        acc = accuracy_score(avg_y_test, np.round(avg_y_pred))
        fold_metrics = {
            'auc': auc,
            'accuracy': acc,
            "predictions": avg_predictions,
        }
        return fold_metrics

    mae = mean_absolute_error(avg_y_test, avg_y_pred)
    mse = mean_squared_error(avg_y_test, avg_y_pred)
    r2 = r2_score(avg_y_test, avg_y_pred)

    pearson, _ = pearsonr(avg_y_test, avg_y_pred)

    fold_metrics = {
        'mae': mae,
        'mse': mse,
        'r2': r2,
        'pearson': pearson,
        "predictions": avg_predictions,
    }
    return fold_metrics



def run_cross_validated_ridge_probe_with_ensemble(
        embeddings: np.ndarray,
        labels: list,
        ids: list,
        activities: list,
        label_idx: int,
        task_type: str = 'reg',
        n_folds: int = 3,
        alpha: float = 1.0,):
    """
        return a dict with all activity score and the ensemble score which is the average of the activities
    :param embeddings:
    :param labels:
    :param ids:
    :param activities:
    :param label_idx:
    :param n_folds:
    :param alpha:
    :return:
    """
    predictions_per_activity = {}
    metrics = {}
    for activity in set(activities):
        activity_metrics = run_cross_validated_ridge_probe(
            embeddings,
            labels,
            ids,
            activities,
            label_idx,
            task_type=task_type,
            activity_to_run=activity,
            n_folds=n_folds,
            alpha=alpha,
        )
        # Skip activity if insufficient label variance (returns None)
        if activity_metrics is None:
            continue
        predictions_per_activity[activity] = activity_metrics["predictions"]
        if task_type == 'reg':
            metrics[activity] = {
                'mae': activity_metrics['mae'],
                'mse': activity_metrics['mse'],
                'r2': activity_metrics['r2'],
                'pearson': activity_metrics['pearson'],
            }
        elif task_type == 'clas':
            metrics[activity] = {
                'auc': activity_metrics['auc'],
                'accuracy': activity_metrics['accuracy'],
            }
    
    # Handle case where no activities had sufficient label variance
    if not predictions_per_activity:
        return None
    
    all_ids = set()
    # Collect all unique IDs across activities
    for preds in predictions_per_activity.values():
        all_ids.update(preds['unique_ids'])
    all_ids = sorted(all_ids)
    # now for each activity put corresponding predictions in the ensemble arrays
    # if the id is not present put nan
    all_preds = []
    all_test = []
    for preds in predictions_per_activity.values():
        id_to_pred = dict(zip(preds['unique_ids'], zip(preds['y_pred'], preds['y_test'])))
        y_test_activity = []
        y_pred_activity = []
        for id_ in all_ids:
            if id_ in id_to_pred:
                y_test_activity.append(id_to_pred[id_][1])
                y_pred_activity.append(id_to_pred[id_][0])
            else:
                y_test_activity.append(np.nan)
                y_pred_activity.append(np.nan)
        all_test.append(y_test_activity)
        all_preds.append(y_pred_activity)
    all_test = np.array(all_test)
    all_preds = np.array(all_preds)
    # Find subjects with complete data across all activities (no NaNs)
    valid_subjects_mask = ~np.any(np.isnan(all_test), axis=0) & ~np.any(np.isnan(all_preds), axis=0)

    # Filter to only subjects with complete data
    all_test_complete = all_test[:, valid_subjects_mask]
    all_preds_complete = all_preds[:, valid_subjects_mask]
    all_ids_complete = [id_ for id_, valid in zip(all_ids, valid_subjects_mask) if valid]
    print(f"subjects all activities: {len(all_ids_complete)}")
    if len(all_ids_complete) == 0:
        metrics['ensemble'] = {
            'mae': 1,
            'mse': 1,
            'r2': 0,
            'pearson': 0,
            'auc': 0,
            'accuracy': 0,
        }
        return metrics
    # Now average across activities (axis=0) for subjects with complete data
    y_test_ensemble = np.mean(all_test_complete, axis=0)
    y_pred_ensemble = np.mean(all_preds_complete, axis=0)

    if task_type == 'clas':
        auc_ens = roc_auc_score(y_test_ensemble, y_pred_ensemble)
        acc_ens = accuracy_score(y_test_ensemble, np.round(y_pred_ensemble))
        metrics['ensemble'] = {
            'auc': auc_ens,
            'accuracy': acc_ens,
        }
        return metrics


    mae_ens = mean_absolute_error(y_test_ensemble, y_pred_ensemble)
    mse_ens = mean_squared_error(y_test_ensemble, y_pred_ensemble)
    r2_ens = r2_score(y_test_ensemble, y_pred_ensemble)
    pearson_ens, _ = pearsonr(y_test_ensemble, y_pred_ensemble)
    metrics['ensemble'] = {
        'mae': mae_ens,
        'mse': mse_ens,
        'r2': r2_ens,
        'pearson': pearson_ens,
        'predictions': {
            'y_test': y_test_ensemble,
            'y_pred': y_pred_ensemble,
            'unique_ids': all_ids_complete,
        }
    }
    return metrics


def probe_eval(label_names, task_types, run_labels, train_outputs, all_metrics=False, wandb_prefix="final_eval"):
    """

    :param label_names: All the labels in the outputs
    :param task_types: All the task types corresponding to the labels
    :param run_labels: which labels to run probes on
    :param train_outputs:
    :param all_metrics:
    :return:
    """
    probe_metrics = {}
    for name in run_labels:
        for embedding_type in [a for a in train_outputs.keys() if "embedding" in a]:
            label_idx = label_names.index(name) if name in label_names else None
            if label_idx is None:
                continue
            result = run_cross_validated_ridge_probe_with_ensemble(
                embeddings=train_outputs[embedding_type],
                labels=train_outputs['labels'],
                ids=train_outputs['ids'],
                activities=train_outputs['activities'],
                label_idx=label_idx,
                task_type=task_types[label_idx],
                n_folds=5,
                alpha=1.0,
            )
            # Skip if insufficient label variance across all activities
            if result is None:
                continue
            for activity in result:
                for metric_name, value in result[activity].items():
                    if metric_name != 'predictions':
                        if all_metrics:
                            probe_metrics[f"{wandb_prefix}/{embedding_type}/{name}/{activity}/{metric_name}"] = value
                        else:
                            # only log auc and pearson if all_metrics is False
                            if metric_name in ['auc', 'pearson']:
                                probe_metrics[
                                    f"{wandb_prefix}/{embedding_type}/{name}/{activity}/{metric_name}"] = value

    weighted_metric = probe_metrics[f"{wandb_prefix}/block_embeddings/age/ensemble/pearson"] / 0.5 + probe_metrics[
        f"{wandb_prefix}/block_embeddings/age/ensemble/pearson"] / 0.4 + probe_metrics[
                          f"{wandb_prefix}/block_embeddings/age/ensemble/pearson"] / 0.6
    weighted_metric = weighted_metric / 3.0
    probe_metrics[f"{wandb_prefix}/block_embeddings/weighted_metric"] = weighted_metric
    return probe_metrics