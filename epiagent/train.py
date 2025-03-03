import os
import torch
import logging
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

def fine_tune_epiagent_for_UFE(
    model,
    train_dataloader,
    num_steps=200000,
    save_dir='../model/fine_tune/UFE/demo_dataset/',
    device=None,
    learning_rate=5e-5,
    save_steps=20000,
    log_steps=500,
    warmup_steps=10000,
    is_logging=True
):
    """
    Fine-tunes a pre-trained EpiAgent model using the provided DataLoader for unsupervised feature extraction.

    Args:
        model (nn.Module): The pre-trained EpiAgent model to be fine-tuned.
        train_dataloader (DataLoader): DataLoader for the fine-tuning dataset.
        num_steps (int): Number of steps to train (default is 200000).
        save_dir (str): Directory to save the fine-tuned model checkpoints (default is '../model/fine_tune/UFE/demo_dataset/').
        device (torch.device): The device to run the training on (default is 'cuda' if available).
        learning_rate (float): Learning rate for the optimizer (default is 5e-5).
        save_steps (int): Save the model every `save_steps` steps (default is 20000).
        log_steps (int): Log training details every `log_steps` steps (default is 500).
        warmup_steps (int): Number of warm-up steps for the learning rate scheduler (default is 10000).
        is_logging (bool): Whether to log and display training details (default is True).
    
    Returns:
        nn.Module: The fine-tuned EpiAgent model.
     """

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    print(f"Model directory created at: {save_dir}")

    # Set device to 'cuda' if available
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Set up logging if required
    if is_logging:
        log_file_path = os.path.join(save_dir, "log.txt")
        logging.basicConfig(
            filename=log_file_path,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        # Add a StreamHandler to output logs to console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)

    # Initialize optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Define a Noam learning rate scheduler
    def noam_scheduler(step):
        return min((step+1)**-0.5, (step+1)*(warmup_steps**-1.5))

    scheduler = LambdaLR(optimizer, lr_lambda=noam_scheduler)

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()

    # Training variables
    global_step = 0
    steps_in_epoch = len(train_dataloader)
    model.train()

    # Initialize loss tracking variables
    if is_logging:
        total_cca_loss = 0.0
        total_sr_loss = 0.0
        step_cca_loss = 0.0
        step_sr_loss = 0.0
        cell_ccre_accessibility_list = []
        predicted_ccre_accessibility_list = []

    num_epochs = int(num_steps/len(train_dataloader.dataset)*train_dataloader.batch_size)
    # Training loop
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            # Move data to the appropriate device
            cell_input_ids = batch[0].to(device)
            signals = batch[1].to(device)
            ex_cell_ccre_ids_for_CCA = [item.to(device) for item in batch[2]]
            ex_cell_ccre_accessibility_for_CCA = batch[3].to(device)

            with autocast():
                # Forward pass through the model
                outputs = model(
                    input_ids=cell_input_ids,
                    return_transformer_output=True,
                    calculate_rlm_loss=False,
                    calculate_cca_loss=True,
                    calculate_sr_loss=True,
                    ex_cell_ccre_ids=ex_cell_ccre_ids_for_CCA,
                    ex_cell_ccre_accessibility=ex_cell_ccre_accessibility_for_CCA,
                    signals=signals
                )

                # Retrieve losses and outputs
                cca_loss = outputs['cca_loss']
                sr_loss = outputs['sr_loss']
                predicted_ccre_accessibility = outputs['predicted_cca_accessibility']

                # Total loss is the sum of CCA loss and SR loss
                loss = cca_loss + sr_loss

            # Backpropagation with mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1

            # Update loss tracking variables if logging is enabled
            if is_logging:
                # Accumulate losses
                total_cca_loss += cca_loss.item()
                total_sr_loss += sr_loss.item()
                step_cca_loss += cca_loss.item()
                step_sr_loss += sr_loss.item()

                # Collect accessibility labels and predictions (limit to 10,000 for efficiency)
                if len(cell_ccre_accessibility_list) < 10000:
                    cell_ccre_accessibility_list.extend(
                        ex_cell_ccre_accessibility_for_CCA.cpu().numpy().tolist()
                    )
                    predicted_ccre_accessibility_list.extend(predicted_ccre_accessibility)

                # Log training details every `log_steps` steps
                if (global_step % log_steps) == 0:

                    avg_cca_loss = step_cca_loss / log_steps
                    avg_sr_loss = step_sr_loss / log_steps
                    avg_total_loss = avg_cca_loss + avg_sr_loss

                    logging.info(
                        f"Epoch [{epoch + 1}/{num_epochs}], Step [{step + 1}/{steps_in_epoch}], "
                        f"Total Loss: {avg_total_loss:.4f}, CCA Loss: {avg_cca_loss:.4f}, "
                        f"SR Loss: {avg_sr_loss:.4f}"
                    )
                    
                    def _calculate_metrics(true_labels, predicted_scores):
                        """
                        Calculates performance metrics for the Cell-cCRE Alignment (CCA) task.
                    
                        Args:
                            true_labels (list): True accessibility labels (0 or 1).
                            predicted_scores (list): Predicted accessibility scores.
                    
                        Returns:
                            tuple: Positive accuracy, Negative accuracy, AUROC, AUPRC.
                        """
                        true_labels = np.array(true_labels)
                        predicted_scores = np.array(predicted_scores)
                    
                        # Positive samples (accessible cCREs)
                        ps_indices = true_labels > 0
                        # Negative samples (inaccessible cCREs)
                        ns_indices = true_labels == 0
                    
                        # Compute accuracies
                        ps_acc = np.sum(predicted_scores[ps_indices] > 0) / np.sum(ps_indices)
                        ns_acc = np.sum(predicted_scores[ns_indices] < 0) / np.sum(ns_indices)
                    
                        # Compute AUROC and AUPRC
                        auroc = roc_auc_score(true_labels, predicted_scores)
                        auprc = average_precision_score(true_labels, predicted_scores)
                    
                        return ps_acc, ns_acc, auroc, auprc
                    
                    # Calculate model performance metrics
                    ps_acc, ns_acc, auroc, auprc = _calculate_metrics(
                        cell_ccre_accessibility_list,
                        predicted_ccre_accessibility_list
                    )
                    logging.info(
                        f"CCA Metrics - Positive Acc: {ps_acc:.4f}, Negative Acc: {ns_acc:.4f}, "
                        f"AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}"
                    )

                    # Reset step losses and lists
                    step_cca_loss = 0.0
                    step_sr_loss = 0.0
                    cell_ccre_accessibility_list = []
                    predicted_ccre_accessibility_list = []

            # Save the model checkpoint every `save_steps` steps
            if (global_step % save_steps) == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_step_{global_step}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                if is_logging:
                    logging.info(f"Model checkpoint saved at step {global_step} to {checkpoint_path}")

        # End of epoch logging
        if is_logging:
            epoch_total_loss = total_cca_loss + total_sr_loss
            logging.info(
                f"End of Epoch {epoch + 1}/{num_epochs}, Total Loss: {epoch_total_loss:.4f}, "
                f"CCA Loss: {total_cca_loss:.4f}, SR Loss: {total_sr_loss:.4f}"
            )
            # Reset epoch losses
            total_cca_loss = 0.0
            total_sr_loss = 0.0

    return model



def fine_tune_epiagent_for_SCA(
    model,
    train_dataloader,
    num_steps=100000,
    save_dir='../model/fine_tune/SCA/demo_dataset/',
    device=None,
    learning_rate=5e-5,
    save_steps=20000,
    log_steps=500,
    warmup_steps=10000,
    is_logging=True
):
    """
    Fine-tunes a pre-trained supervised EpiAgent model for Supervised Cell Type Annotation (SCA).

    Args:
        model (nn.Module): The pre-trained supervised EpiAgent model to be fine-tuned.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        num_steps (int): Total number of training steps (default is 200000).
        save_dir (str): Directory to save the fine-tuned model checkpoints (default is '../model/fine_tune/SCA/demo_dataset/').
        device (torch.device): The device to run the training on (default is 'cuda' if available).
        learning_rate (float): Learning rate for the optimizer (default is 5e-5).
        save_steps (int): Save the model every `save_steps` steps (default is 20000).
        log_steps (int): Log training details every `log_steps` steps (default is 500).
        warmup_steps (int): Number of warm-up steps for the learning rate scheduler (default is 10000).
        is_logging (bool): Whether to log and display training details (default is True).

    Returns:
        nn.Module: The fine-tuned supervised EpiAgent model.
    """

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    print(f"Model directory created at: {save_dir}")

    # Set device to 'cuda' if available
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Set up logging if required
    if is_logging:
        log_file_path = os.path.join(save_dir, "log.txt")
        logging.basicConfig(
            filename=log_file_path,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        # Add a StreamHandler to output logs to console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)

    # Initialize optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Define a Noam learning rate scheduler
    def noam_scheduler(step):
        return min((step + 1) ** -0.5, (step + 1) * (warmup_steps ** -1.5))

    scheduler = LambdaLR(optimizer, lr_lambda=noam_scheduler)

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()

    # Training variables
    global_step = 0
    total_steps = num_steps
    model.train()

    # Initialize loss tracking variables
    if is_logging:
        total_classification_loss = 0.0
        step_classification_loss = 0.0
        cell_type_labels_list = []
        predicted_cell_type_list = []

    # Training loop
    while global_step < total_steps:
        for batch in train_dataloader:
            if global_step >= total_steps:
                break  # Exit loop if we've reached the desired number of steps

            optimizer.zero_grad()

            # Move data to the appropriate device
            cell_input_ids = batch[0].to(device)
            cell_type_labels = batch[1].to(device)

            with autocast():
                # Forward pass through the model
                # For supervised EpiAgent, assume the model outputs predictions and loss directly
                predicted_logits, classification_loss = model(
                    input_ids=cell_input_ids,
                    need_cell_embeddings=False,
                    calculate_loss=True,
                    target=cell_type_labels
                )

            # Backpropagation with mixed precision
            scaler.scale(classification_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1

            # Update loss tracking variables if logging is enabled
            if is_logging:
                # Accumulate losses
                total_classification_loss += classification_loss.item()
                step_classification_loss += classification_loss.item()

                # Collect true labels and predicted labels
                cell_type_labels_list.extend(cell_type_labels.cpu().numpy().tolist())
                # Obtain predicted class indices
                _, predicted_classes = torch.max(predicted_logits, dim=1)
                predicted_cell_type_list.extend(predicted_classes.cpu().numpy().tolist())

                # Log training details every `log_steps` steps
                if (global_step % log_steps) == 0:
                    avg_classification_loss = step_classification_loss / log_steps

                    # Calculate performance metrics
                    classification_acc, classification_kappa, classification_macro_f1 = _calculate_classification_metrics(
                        cell_type_labels_list,
                        predicted_cell_type_list
                    )

                    logging.info(
                        f"Step [{global_step}/{total_steps}], "
                        f"Classification Loss: {avg_classification_loss:.4f}, "
                        f"Accuracy: {classification_acc:.4f}, "
                        f"Cohen's Kappa: {classification_kappa:.4f}, "
                        f"Macro F1: {classification_macro_f1:.4f}"
                    )

                    # Reset step losses and lists
                    step_classification_loss = 0.0
                    cell_type_labels_list = []
                    predicted_cell_type_list = []

            # Save the model checkpoint every `save_steps` steps
            if (global_step % save_steps) == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_step_{global_step}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                if is_logging:
                    logging.info(f"Model checkpoint saved at step {global_step} to {checkpoint_path}")

    return model

def _calculate_classification_metrics(true_labels, predicted_labels):
    """
    Calculates performance metrics for classification tasks.

    Args:
        true_labels (list): True cell type labels.
        predicted_labels (list): Predicted cell type labels.

    Returns:
        tuple: Accuracy, Cohen's Kappa, Macro F1 score.
    """
    # Convert lists to numpy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Calculate metrics
    acc = accuracy_score(true_labels, predicted_labels)
    kappa = cohen_kappa_score(true_labels, predicted_labels)
    macro_f1 = f1_score(true_labels, predicted_labels, average='macro')

    return acc, kappa, macro_f1
    

def fine_tune_epiagent_for_OSP(
    model,
    classifier_model,
    train_dataloader,
    num_steps=500000,
    save_dir='../model/fine_tune/OSP/demo_dataset/',
    device=None,
    learning_rate=4e-5,
    save_steps=20000,
    log_steps=500,
    warmup_steps=30000,
    is_logging=True
):
    """
    Fine-tunes a pre-trained EpiAgent model for the task of cellular response prediction under 
    Out-of-Sample Stimulated Perturbation (OSP). In addition to optimizing the Cell-cCRE Alignment (CCA)
    and Signal Reconstruction (SR) losses, a classifier model is employed to encourage the separation of
    cell embeddings corresponding to pre-perturbation cells and those that have been mapped (using the [PER]
    token) to the post-perturbation state. The classification loss is added (with a weight of 0.01) to the 
    overall loss to pull the mapped embeddings closer to the post-perturbation distribution while pushing 
    apart those of the pre-perturbation cells.
    
    Args:
        model (nn.Module): The pre-trained EpiAgent model to be fine-tuned.
        classifier_model (nn.Module): A classifier model that computes classification loss and outputs predictions
            based on cell embeddings. It must support a forward pass with parameters (input, calculate_loss, target).
        train_dataloader (DataLoader): DataLoader for the fine-tuning dataset. Each batch should return a tuple:
            (padded_cells_batch, signals_batch, classification_labels_batch, ex_cell_ccre_ids_for_CCA_batch, 
             ex_cell_ccre_accessibility_for_CCA_batch).
        num_steps (int): Total number of training steps (default is 400000).
        save_dir (str): Directory to save model checkpoints (default is '../model/fine_tune/OSP/demo_dataset/').
        device (torch.device): Device to run training on (defaults to 'cuda' if available).
        learning_rate (float): Learning rate for the optimizer (default is 5e-5).
        save_steps (int): Save a model checkpoint every `save_steps` steps (default is 20000).
        log_steps (int): Log training details every `log_steps` steps (default is 500).
        warmup_steps (int): Number of warm-up steps for the learning rate scheduler (default is 10000).
        is_logging (bool): Whether to log training details (default is True).
    
    Returns:
        nn.Module: The fine-tuned EpiAgent model.
    """

    # Ensure the save directory exists.
    os.makedirs(save_dir, exist_ok=True)
    print(f"Model directory created at: {save_dir}")

    # Set device to 'cuda' if available and not provided.
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move both the EpiAgent model and the classifier model to the specified device.
    model.to(device)
    classifier_model.to(device)
    
    # Set up logging if required.
    if is_logging:
        log_file_path = os.path.join(save_dir, "log.txt")
        logging.basicConfig(
            filename=log_file_path,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        # Also output logs to console.
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)

    # Initialize the optimizer with parameters from both the main model and the classifier model.
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier_model.parameters()), lr=learning_rate, weight_decay=0.01)

    # Define a Noam-style learning rate scheduler.
    def noam_scheduler(step):
        return min((step+1)**-0.5, (step+1) * (warmup_steps**-1.5))
    scheduler = LambdaLR(optimizer, lr_lambda=noam_scheduler)

    # Initialize gradient scaler for mixed precision training.
    scaler = GradScaler()

    # Training variables.
    global_step = 0
    steps_in_epoch = len(train_dataloader)
    model.train()
    classifier_model.train()

    # Initialize loss tracking variables for logging.
    if is_logging:
        total_cca_loss = 0.0
        total_sr_loss = 0.0
        total_class_loss = 0.0
        step_cca_loss = 0.0
        step_sr_loss = 0.0
        step_class_loss = 0.0
        # Lists for collecting classification true and predicted labels.
        classification_true_labels = []
        classification_predicted_labels = []
        # Also lists to track accessibility labels for CCA (as before).
        cell_ccre_accessibility_list = []
        predicted_ccre_accessibility_list = []

    
    num_epochs = int(num_steps/len(train_dataloader.dataset)*train_dataloader.batch_size)
    # Training loop
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            # Unpack the batch.
            # Batch returns: (padded_cells_batch, signals_batch, classification_labels_batch, ex_cell_ccre_ids_for_CCA_batch, ex_cell_ccre_accessibility_for_CCA_batch)
            cell_input_ids = batch[0].to(device)
            signals = batch[1].to(device)
            label_for_classification_batch = batch[2].to(device)
            ex_cell_ccre_ids_for_CCA = [item.to(device) for item in batch[3]]
            ex_cell_ccre_accessibility_for_CCA = batch[4].to(device)

            # Mixed precision forward pass.
            with autocast():
                # Forward pass through the EpiAgent model.
                outputs = model(
                    input_ids=cell_input_ids,
                    return_transformer_output=True,
                    calculate_rlm_loss=False,
                    calculate_cca_loss=True,
                    calculate_sr_loss=True,
                    ex_cell_ccre_ids=ex_cell_ccre_ids_for_CCA,
                    ex_cell_ccre_accessibility=ex_cell_ccre_accessibility_for_CCA,
                    signals=signals
                )

                # Retrieve CCA and SR losses from the model's output.
                cca_loss = outputs['cca_loss']
                sr_loss = outputs['sr_loss']

                # Forward pass through the classifier model.
                # We use the transformer outputs' [CLS] token (first token) as input for classification.
                # The classifier returns the predicted label distribution and the classification loss.
                predicted_label, loss_classification = classifier_model(
                    outputs['transformer_outputs'][:, 0, :],
                    calculate_loss=True,
                    target=label_for_classification_batch
                )
                # Retrieve the predicted class by taking the argmax.
                _, predicted_classes = torch.max(predicted_label, dim=1)

                # Combine losses: The total loss is the sum of CCA loss, SR loss, and 0.01 weighted classification loss.
                loss = cca_loss + sr_loss + 0.01 * loss_classification

            # Backpropagation using mixed precision.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1

            # Update loss tracking variables if logging is enabled.
            if is_logging:
                total_cca_loss += cca_loss.item()
                total_sr_loss += sr_loss.item()
                total_class_loss += loss_classification.item()
                step_cca_loss += cca_loss.item()
                step_sr_loss += sr_loss.item()
                step_class_loss += loss_classification.item()
                
                # Collect accessibility labels (for CCA monitoring) with a limit for efficiency.
                predicted_ccre_accessibility = outputs['predicted_cca_accessibility']
                if len(cell_ccre_accessibility_list) < 10000:
                    cell_ccre_accessibility_list.extend(
                        ex_cell_ccre_accessibility_for_CCA.cpu().numpy().tolist()
                    )
                    predicted_ccre_accessibility_list.extend(predicted_ccre_accessibility)

                # Accumulate classification true and predicted labels.
                classification_true_labels.extend(label_for_classification_batch.cpu().numpy().tolist())
                classification_predicted_labels.extend(predicted_classes.cpu().numpy().tolist())

                # Log training details every 'log_steps' steps.
                if (global_step % log_steps) == 0:
                    avg_cca_loss = step_cca_loss / log_steps
                    avg_sr_loss = step_sr_loss / log_steps
                    avg_class_loss = step_class_loss / log_steps
                    avg_total_loss = avg_cca_loss + avg_sr_loss + 0.01 * avg_class_loss

                    # Calculate classification accuracy.
                    classification_true = np.array(classification_true_labels)
                    classification_pred = np.array(classification_predicted_labels)
                    classification_accuracy = np.mean(classification_true == classification_pred)

                    logging.info(
                        f"Epoch [{epoch + 1}/{num_epochs}], Step [{step + 1}/{steps_in_epoch}], "
                        f"Total Loss: {avg_total_loss:.4f}, CCA Loss: {avg_cca_loss:.4f}, "
                        f"SR Loss: {avg_sr_loss:.4f}, Class Loss: {avg_class_loss:.4f} (w=0.01), "
                        f"Classification Accuracy: {classification_accuracy:.4f}"
                    )

                    # Additional metrics for CCA task.
                    def _calculate_metrics(true_labels, predicted_scores):
                        """
                        Calculates performance metrics for the Cell-cCRE Alignment (CCA) task.
                    
                        Args:
                            true_labels (list): True accessibility labels (0 or 1).
                            predicted_scores (list): Predicted accessibility scores.
                    
                        Returns:
                            tuple: Positive accuracy, Negative accuracy, AUROC, AUPRC.
                        """
                        true_labels = np.array(true_labels)
                        predicted_scores = np.array(predicted_scores)
                    
                        ps_indices = true_labels > 0
                        ns_indices = true_labels == 0
                    
                        ps_acc = np.sum(predicted_scores[ps_indices] > 0) / np.sum(ps_indices) if np.sum(ps_indices) > 0 else 0
                        ns_acc = np.sum(predicted_scores[ns_indices] < 0) / np.sum(ns_indices) if np.sum(ns_indices) > 0 else 0
                    
                        auroc = roc_auc_score(true_labels, predicted_scores) if len(np.unique(true_labels)) > 1 else 0.0
                        auprc = average_precision_score(true_labels, predicted_scores) if len(np.unique(true_labels)) > 1 else 0.0
                    
                        return ps_acc, ns_acc, auroc, auprc
                    
                    ps_acc, ns_acc, auroc, auprc = _calculate_metrics(
                        cell_ccre_accessibility_list,
                        predicted_ccre_accessibility_list
                    )
                    logging.info(
                        f"CCA Metrics - Positive Acc: {ps_acc:.4f}, Negative Acc: {ns_acc:.4f}, "
                        f"AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}"
                    )

                    # Reset step loss accumulators and classification lists.
                    step_cca_loss = 0.0
                    step_sr_loss = 0.0
                    step_class_loss = 0.0
                    cell_ccre_accessibility_list = []
                    predicted_ccre_accessibility_list = []
                    classification_true_labels = []
                    classification_predicted_labels = []

            # Save a model checkpoint every 'save_steps' steps.
            if (global_step % save_steps) == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_step_{global_step}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                if is_logging:
                    logging.info(f"Model checkpoint saved at step {global_step} to {checkpoint_path}")

        # End-of-epoch logging.
        if is_logging:
            epoch_total_loss = total_cca_loss + total_sr_loss + 0.01 * total_class_loss
            logging.info(
                f"End of Epoch {epoch + 1}/{num_epochs}, Total Loss: {epoch_total_loss:.4f}, "
                f"CCA Loss: {total_cca_loss:.4f}, SR Loss: {total_sr_loss:.4f}, "
                f"Class Loss: {total_class_loss:.4f}"
            )
            total_cca_loss = 0.0
            total_sr_loss = 0.0
            total_class_loss = 0.0

    return model


def fine_tune_epiagent_for_RDI(
    model,
    classifier_model,
    train_dataloader,
    num_steps=500000,
    save_dir='../model/fine_tune/RDI/demo_dataset/',
    device=None,
    learning_rate=4e-5,
    save_steps=20000,
    log_steps=500,
    warmup_steps=30000,
    is_logging=True
):
    """
    Fine-tunes a pre-trained EpiAgent_BC model for the task of Reference Data Integration (RDI). 
    This function also uses a classifier model to separate embeddings corresponding to cells 
    from different batches by encouraging the model to push apart embeddings from different 
    batches and bring embeddings of the same batch closer together.
    
    Args:
        model (nn.Module): The pre-trained EpiAgent_BC model to be fine-tuned.
        classifier_model (nn.Module): A classifier model to enforce batch separation during training.
        train_dataloader (DataLoader): DataLoader for the fine-tuning dataset.
        num_steps (int): Total number of training steps (default is 500000).
        save_dir (str): Directory to save model checkpoints (default is '../model/fine_tune/RDI/demo_dataset/').
        device (torch.device): Device to run training on (defaults to 'cuda' if available).
        learning_rate (float): Learning rate for the optimizer (default is 4e-5).
        save_steps (int): Save a model checkpoint every `save_steps` steps (default is 20000).
        log_steps (int): Log training details every `log_steps` steps (default is 500).
        warmup_steps (int): Number of warm-up steps for the learning rate scheduler (default is 10000).
        is_logging (bool): Whether to log training details (default is True).
    
    Returns:
        nn.Module: The fine-tuned EpiAgent_BC model.
    """

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    print(f"Model directory created at: {save_dir}")

    # Set device to 'cuda' if available
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    classifier_model.to(device)

    # Set up logging if required
    if is_logging:
        log_file_path = os.path.join(save_dir, "log.txt")
        logging.basicConfig(
            filename=log_file_path,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        # Add a StreamHandler to output logs to console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)

    # Initialize the optimizers
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    optimizer_classifier = torch.optim.AdamW(classifier_model.parameters(), lr=0.00001, weight_decay=0.01)

    # Define a Noam learning rate scheduler
    def noam_scheduler(step):
        return min((step+1)**-0.5, (step+1)*(warmup_steps**-1.5))
    scheduler = LambdaLR(optimizer, lr_lambda=noam_scheduler)

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()

    # Training variables
    global_step = 0
    steps_in_epoch = len(train_dataloader)
    model.train()
    classifier_model.train()

    # Initialize loss tracking variables for logging.
    if is_logging:
        total_cca_loss = 0.0
        total_sr_loss = 0.0
        total_class_loss = 0.0
        step_cca_loss = 0.0
        step_sr_loss = 0.0
        step_class_loss = 0.0
        # Lists for collecting classification true and predicted labels.
        classification_true_labels = []
        classification_predicted_labels = []
        # Also lists to track accessibility labels for CCA (as before).
        cell_ccre_accessibility_list = []
        predicted_ccre_accessibility_list = []

    num_epochs = int(num_steps / len(train_dataloader.dataset) * train_dataloader.batch_size)

    # Training loop
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            optimizer_classifier.zero_grad()

            # Unpack the batch.
            # Batch returns: (padded_cells_batch, signals_batch, batch_ids_batch, ex_cell_ccre_ids_for_CCA_batch, ex_cell_ccre_accessibility_for_CCA_batch)
            cell_input_ids = batch[0].to(device)
            signals = batch[1].to(device)
            cell_batch_ids = batch[2].to(device)
            ex_cell_ccre_ids_for_CCA = [item.to(device) for item in batch[3]]
            ex_cell_ccre_accessibility_for_CCA = batch[4].to(device)
            
            set_requires_grad(model, False)
            set_requires_grad(classifier_model, True)
            # First forward pass: Only train classifier (model is frozen)
            with autocast():
                outputs = model(
                    input_ids=cell_input_ids,
                    batch_ids=cell_batch_ids,
                    return_transformer_output=True
                )
                
                # Classifier forward pass
                _, loss_classification = classifier_model(
                    outputs['transformer_outputs'][:, 0, :],
                    calculate_loss=True,
                    target=cell_batch_ids
                )
                # Backpropagate classification loss (only classifier is updated)
                scaler.scale(loss_classification).backward()
                scaler.step(optimizer_classifier)
                scaler.update()

            
            # Second forward pass: Train model (classifier is frozen)
            optimizer.zero_grad()
            optimizer_classifier.zero_grad()
            set_requires_grad(model, True)
            set_requires_grad(classifier_model, False)
            with autocast():
                # Forward pass through the EpiAgent model.
                outputs = model(
                    input_ids=cell_input_ids,
                    batch_ids=cell_batch_ids,
                    return_transformer_output=True,
                    calculate_rlm_loss=False,
                    calculate_cca_loss=True,
                    calculate_sr_loss=True,
                    ex_cell_ccre_ids=ex_cell_ccre_ids_for_CCA,
                    ex_cell_ccre_accessibility=ex_cell_ccre_accessibility_for_CCA,
                    signals=signals
                )

                # Retrieve CCA and SR losses
                cca_loss = outputs['cca_loss']
                sr_loss = outputs['sr_loss']

                # Classifier forward pass (we use the transformer outputs' [CLS] token as input)
                predicted_label, loss_classification = classifier_model(
                    outputs['transformer_outputs'][:, 0, :],
                    calculate_loss=True,
                    target=cell_batch_ids
                )

                # Retrieve predicted classes
                _, predicted_classes = torch.max(predicted_label, dim=1)

                # Combine losses: CCA loss, SR loss, and classification loss (with weight 0.1 for adversarial loss)
                loss = cca_loss + sr_loss - 0.1 * loss_classification

            # Backpropagation (now only model is updated)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1

            # Update loss tracking variables if logging is enabled.
            if is_logging:
                total_cca_loss += cca_loss.item()
                total_sr_loss += sr_loss.item()
                total_class_loss += loss_classification.item()
                step_cca_loss += cca_loss.item()
                step_sr_loss += sr_loss.item()
                step_class_loss += loss_classification.item()

                # Collect accessibility labels (for CCA monitoring) with a limit for efficiency.
                predicted_ccre_accessibility = outputs['predicted_cca_accessibility']
                if len(cell_ccre_accessibility_list) < 10000:
                    cell_ccre_accessibility_list.extend(
                        ex_cell_ccre_accessibility_for_CCA.cpu().numpy().tolist()
                    )
                    predicted_ccre_accessibility_list.extend(predicted_ccre_accessibility)

                # Accumulate classification true and predicted labels.
                classification_true_labels.extend(cell_batch_ids.cpu().numpy().tolist())
                classification_predicted_labels.extend(predicted_classes.cpu().numpy().tolist())


                # Log training details every 'log_steps' steps.
                if (global_step % log_steps) == 0:
                    avg_cca_loss = step_cca_loss / log_steps
                    avg_sr_loss = step_sr_loss / log_steps
                    avg_class_loss = step_class_loss / log_steps
                    avg_total_loss = avg_cca_loss + avg_sr_loss - 0.1 * avg_class_loss

                    # Calculate classification accuracy.
                    classification_true = np.array(classification_true_labels)
                    classification_pred = np.array(classification_predicted_labels)
                    classification_accuracy = np.mean(classification_true == classification_pred)

                    logging.info(
                        f"Epoch [{epoch + 1}/{num_epochs}], Step [{step + 1}/{steps_in_epoch}], "
                        f"Total Loss: {avg_total_loss:.4f}, CCA Loss: {avg_cca_loss:.4f}, "
                        f"SR Loss: {avg_sr_loss:.4f}, Class Loss: {avg_class_loss:.4f} (w=0.1), "
                        f"Classification Accuracy: {classification_accuracy:.4f}"
                    )

                    # Additional metrics for CCA task.
                    def _calculate_metrics(true_labels, predicted_scores):
                        true_labels = np.array(true_labels)
                        predicted_scores = np.array(predicted_scores)
                        ps_indices = true_labels > 0
                        ns_indices = true_labels == 0

                        ps_acc = np.sum(predicted_scores[ps_indices] > 0) / np.sum(ps_indices) if np.sum(ps_indices) > 0 else 0
                        ns_acc = np.sum(predicted_scores[ns_indices] < 0) / np.sum(ns_indices) if np.sum(ns_indices) > 0 else 0
                        auroc = roc_auc_score(true_labels, predicted_scores) if len(np.unique(true_labels)) > 1 else 0.0
                        auprc = average_precision_score(true_labels, predicted_scores) if len(np.unique(true_labels)) > 1 else 0.0

                        return ps_acc, ns_acc, auroc, auprc

                    ps_acc, ns_acc, auroc, auprc = _calculate_metrics(
                        cell_ccre_accessibility_list,
                        predicted_ccre_accessibility_list
                    )
                    logging.info(
                        f"CCA Metrics - Positive Acc: {ps_acc:.4f}, Negative Acc: {ns_acc:.4f}, "
                        f"AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}"
                    )

                    # Reset step loss accumulators and classification lists.
                    step_cca_loss = 0.0
                    step_sr_loss = 0.0
                    step_class_loss = 0.0
                    cell_ccre_accessibility_list = []
                    predicted_ccre_accessibility_list = []
                    classification_true_labels = []
                    classification_predicted_labels = []

            # Save model checkpoint
            if (global_step % save_steps) == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_step_{global_step}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                logging.info(f"Model checkpoint saved at step {global_step} to {checkpoint_path}")

        # End of epoch logging
        if is_logging:
            epoch_total_loss = total_cca_loss + total_sr_loss - 0.01 * total_class_loss
            logging.info(
                f"End of Epoch {epoch + 1}/{num_epochs}, Total Loss: {epoch_total_loss:.4f}, "
                f"CCA Loss: {total_cca_loss:.4f}, SR Loss: {total_sr_loss:.4f}, "
                f"Class Loss: {total_class_loss:.4f}"
            )
            total_cca_loss = 0.0
            total_sr_loss = 0.0
            total_class_loss = 0.0

    return model
def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad