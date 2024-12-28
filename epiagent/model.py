import torch
import torch.nn as nn
from transformers import BertConfig
from flash_attn.models.bert import BertEncoder

class EpiAgent(nn.Module):
    """
    A model that combines the features of Transformer-based architectures and domain-specific layers for cellular 
    regulatory network learning. It performs tasks like cell-cCRE alignment, replacing language modeling (RLM), 
    and signal reconstruction with a transformer-based backbone (BERT).
    
    Args:
        vocab_size (int): The size of the vocabulary (default is 1355449).
        num_layers (int): Number of layers in the transformer encoder (default is 18).
        embedding_dim (int): The dimensionality of the embeddings (default is 512).
        num_attention_heads (int): The number of attention heads in each transformer layer (default is 8).
        max_rank_embeddings (int): The maximum number of rank embeddings for positional encoding (default is 8192).
        MLP_hidden_for_RLM (int): The size of the hidden layer for the Replacing Language Modeling (RLM) task (default is 64).
        MLP_hidden_for_CCA (int): The size of the hidden layer for the Cell-cCRE Alignment (CCA) task (default is 128).
        pos_weight_for_RLM (bool or tensor): Positive weight for RLM loss, if specified (default is False).
        pos_weight_for_CCA (bool or tensor): Positive weight for CCA loss, if specified (default is False).
        pos_weight_signals (tensor): Positive weight for signal reconstruction loss (default is tensor(100)).
        use_flash_attn (bool): Whether to use FlashAttention for transformer encoder (default is True).
    """
    def __init__(self, 
                 vocab_size=1355449, 
                 num_layers=18, 
                 embedding_dim=512, 
                 num_attention_heads=8, 
                 max_rank_embeddings=8192,
                 MLP_hidden_for_RLM=64,
                 MLP_hidden_for_CCA=128,
                 pos_weight_for_RLM=False,
                 pos_weight_for_CCA=False,
                 pos_weight_signals=torch.tensor(100),
                 use_flash_attn=True):
        super(EpiAgent, self).__init__()

        # Model configuration
        self.vocab_size = vocab_size
        self.cCRE_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rank_embedding = nn.Embedding(max_rank_embeddings, embedding_dim)
        self.config = BertConfig(
            vocab_size=vocab_size,
            num_hidden_layers=num_layers,
            hidden_size=embedding_dim,
            num_attention_heads=num_attention_heads,
            intermediate_size=4 * embedding_dim,
            max_position_embeddings=max_rank_embeddings,
            use_flash_attn=use_flash_attn
        )
        self.EpiAgent_transformer = BertEncoder(self.config)

        # Replacing Language Modeling (RLM) components
        self.fc1_for_RLM = nn.Linear(embedding_dim, MLP_hidden_for_RLM)
        self.layer_norm_for_RLM = nn.LayerNorm(MLP_hidden_for_RLM)
        self.dropout_for_RLM = nn.Dropout(0.25)
        self.fc2_for_RLM = nn.Linear(MLP_hidden_for_RLM, 1)

        # Cell-cCRE Alignment (CCA) components
        self.fc1_for_CCA = nn.Linear(embedding_dim * 2, MLP_hidden_for_CCA)
        self.layer_norm_for_CCA = nn.LayerNorm(MLP_hidden_for_CCA)
        self.dropout_for_CCA = nn.Dropout(0.25)
        self.fc2_for_CCA = nn.Linear(MLP_hidden_for_CCA, 1)

        # Activation functions
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        # Signal reconstruction components
        self.signal_decoder = nn.Linear(embedding_dim, vocab_size - 4)
        self.criterion_SR = nn.BCEWithLogitsLoss(pos_weight=pos_weight_signals)

        # RLM loss criterion
        self.criterion_RLM = nn.BCEWithLogitsLoss() if not pos_weight_for_RLM else nn.BCEWithLogitsLoss(pos_weight=pos_weight_for_RLM)

        # CCA loss criterion
        self.criterion_CCA = nn.BCEWithLogitsLoss() if not pos_weight_for_CCA else nn.BCEWithLogitsLoss(pos_weight=pos_weight_for_CCA)

    def forward(self, 
                input_ids, 
                return_transformer_output=True, 
                calculate_rlm_loss=False, 
                calculate_cca_loss=False, 
                calculate_sr_loss=False, 
                ex_cell_ccre_ids=None,
                ex_cell_ccre_accessibility=None,
                signals=None):
        """
        Forward pass for the EpiAgent model. This method computes various losses (RLM, CCA, SR) and returns a dictionary 
        containing the computed outputs.

        Args:
            input_ids (tensor): Input tensor of shape [batch_size, seq_len].
            return_transformer_output (bool): If True, includes transformer outputs in the return dictionary (default is True).
            calculate_rlm_loss (bool): If True, computes the RLM loss (default is False).
            calculate_cca_loss (bool): If True, computes the CCA loss (default is False).
            calculate_sr_loss (bool): If True, computes the Signal Reconstruction (SR) loss (default is False).
            ex_cell_ccre_ids (tensor, optional): External cCRE IDs for the CCA loss (default is None).
            ex_cell_ccre_accessibility (tensor, optional): Accessibility of cCREs for CCA loss (default is None).
            signals (tensor, optional): Signal tensor for SR loss (default is None).

        Returns:
            dict: A dictionary containing the computed losses and outputs.
        """
        outputs = {}

        # Compute embeddings
        ccre_embeddings = self.cCRE_embedding(input_ids)
        rank_indices = torch.arange(input_ids.shape[1]).unsqueeze(0).expand_as(input_ids).to(input_ids.device)
        rank_embeddings = self.rank_embedding(rank_indices)

        # Transformer encoder pass
        transformer_outputs = self.EpiAgent_transformer(ccre_embeddings + rank_embeddings, key_padding_mask=(input_ids != 0))
        outputs['transformer_outputs'] = transformer_outputs if return_transformer_output else None

        # Compute RLM loss if required
        if calculate_rlm_loss:
            if ex_cell_ccre_accessibility is None:
                raise ValueError("`ex_cell_ccre_accessibility` is required for RLM loss computation.")
            rlm_loss, predicted_accessibility = self.RLM_loss(input_ids, transformer_outputs, ex_cell_ccre_accessibility)
            outputs['rlm_loss'] = rlm_loss
            outputs['predicted_accessibility'] = predicted_accessibility

        # Compute CCA loss if required
        if calculate_cca_loss:
            if ex_cell_ccre_ids is None or ex_cell_ccre_accessibility is None:
                raise ValueError("Both `ex_cell_ccre_ids` and `ex_cell_ccre_accessibility` are required for CCA loss computation.")
            cca_loss, predicted_cca_accessibility = self.CCA_loss(transformer_outputs[:, 0, :], ex_cell_ccre_ids, ex_cell_ccre_accessibility)
            outputs['cca_loss'] = cca_loss
            outputs['predicted_cca_accessibility'] = predicted_cca_accessibility

        # Compute SR loss if required
        if calculate_sr_loss:
            if signals is None:
                raise ValueError("`signals` is required for SR loss computation.")
            predicted_signals = self.signal_decoder(transformer_outputs[:, 0, :])
            sr_loss = self.criterion_SR(predicted_signals, signals)
            outputs['sr_loss'] = sr_loss
            outputs['predicted_signals'] = predicted_signals

        return outputs

    def CCA_loss(self, cell_embeddings, ex_cell_ccre_ids, ex_cell_ccre_accessibility):
        """
        Computes the Cell-cCRE Alignment (CCA) loss.

        Args:
            cell_embeddings (tensor): Embeddings for the input cells.
            ex_cell_ccre_ids (tensor): cCRE indices for external cells used in the cell-cCRE alignment task.
            ex_cell_ccre_accessibility (tensor): Binary tensor indicating the accessibility of cCREs in the cells.

        Returns:
            tuple: 
                - loss (tensor): The updated CCA loss.
                - predicted_cca_accessibility (list): Predicted accessibility scores for external cCREs.
        """
        repeat_counts = [len(ids) for ids in ex_cell_ccre_ids]
        expanded_cell_embeddings = torch.repeat_interleave(cell_embeddings, torch.tensor(repeat_counts, device=cell_embeddings.device), dim=0)
        concatenated_ccre_ids = torch.cat(ex_cell_ccre_ids, dim=0).to(cell_embeddings.device)
        flattened_accessibility = ex_cell_ccre_accessibility.view(-1, 1).to(cell_embeddings.device)

        concatenated_embeddings = torch.cat((expanded_cell_embeddings, self.cCRE_embedding(concatenated_ccre_ids)), dim=-1)
        predicted_scores = self.fc2_for_CCA(self.layer_norm_for_CCA(self.gelu(self.fc1_for_CCA(concatenated_embeddings))))
        loss = self.criterion_CCA(predicted_scores, flattened_accessibility)

        return loss, predicted_scores.view(-1).cpu().detach().numpy().tolist()

    def RLM_loss(self, input_ids, contextual_embeddings, ccre_accessibility):
        """
        Computes the Replacing Language Modeling (RLM) loss.

        Args:
            input_ids (tensor): cCRE IDs for the input cells.
            contextual_embeddings (tensor): Contextual embeddings for cCREs in the cells.
            ccre_accessibility (tensor): Binary tensor indicating whether the cCREs are accessible in the corresponding cells.

        Returns:
            tuple:
                - loss (tensor): The RLM loss.
                - predicted_accessibility (list): Predicted accessibility scores for cCREs.
        """
        valid_indices = input_ids > 3
        valid_embeddings = contextual_embeddings[valid_indices, :]
        accessibility_targets = ccre_accessibility.reshape(-1, 1)

        predicted_scores = self.fc2_for_RLM(self.layer_norm_for_RLM(self.gelu(self.fc1_for_RLM(valid_embeddings))))
        loss = self.criterion_RLM(predicted_scores, accessibility_targets.to(predicted_scores.device))

        return loss, predicted_scores.view(-1).cpu().detach().numpy().tolist()

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance by focusing on hard-to-classify examples.

    Args:
        alpha (float): Weighting factor for the focal loss.
        gamma (float): Focusing parameter to reduce the impact of easy examples.
    """
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Forward pass for focal loss computation.

        Args:
            inputs (tensor): Predicted logits.
            targets (tensor): Ground truth labels.

        Returns:
            tensor: Computed focal loss.
        """
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevents nans when probability is 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

class classifier(nn.Module):
    """
    A neural network classifier with multiple layers for cell type classification.

    Args:
        embedding_dim (int): Dimension of the input embeddings.
        num_classes (int): Number of cell type classes to predict.
    """
    def __init__(self, embedding_dim, num_classes):
        super(classifier, self).__init__()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.layer_norm1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.layer_norm2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, num_classes)
        self.loss = FocalLoss()

    def forward(self, X, calculate_loss=False, target=None):
        """
        Forward pass for the classifier.

        Args:
            X (tensor): Input features.
            calculate_loss (bool): Whether to calculate loss (default: False).
            target (tensor): Ground truth labels (required if calculate_loss is True).

        Returns:
            tensor: Predicted labels.
            tensor (optional): Computed loss if calculate_loss is True.
        """
        predicted_label = self.fc3(
            self.layer_norm2(self.gelu(self.fc2(self.dropout(self.layer_norm1(self.fc1(X))))))
        )
        if calculate_loss:
            loss = self.loss(predicted_label, target)
            return predicted_label, loss
        return predicted_label

class EpiAgent_supervised(nn.Module):
    """
    A supervised variant of the EpiAgent model for cell type annotation.

    This model leverages the transformer backbone of the EpiAgent architecture while integrating a classifier
    module for supervised cell type classification. It serves as the foundation for EpiAgent-B and EpiAgent-NT models.

    Args:
        vocab_size (int): The size of the vocabulary (default is 1355449).
        num_layers (int): Number of layers in the transformer encoder (default is 18).
        embedding_dim (int): The dimensionality of the embeddings (default is 512).
        num_attention_heads (int): The number of attention heads in each transformer layer (default is 8).
        max_rank_embeddings (int): The maximum number of rank embeddings for positional encoding (default is 8192).
        num_classes (int): Number of cell type classes for the classifier.
        use_flash_attn (bool): Whether to use FlashAttention for transformer encoder (default is True).
    """
    def __init__(self, 
                 vocab_size=1355449, 
                 num_layers=18, 
                 embedding_dim=512, 
                 num_attention_heads=8, 
                 max_rank_embeddings=8192, 
                 num_classes=10, 
                 use_flash_attn=True):
        super(EpiAgent_supervised, self).__init__()

        # Embedding layers
        self.vocab_size = vocab_size
        self.cCRE_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rank_embedding = nn.Embedding(max_rank_embeddings, embedding_dim)

        # Transformer configuration
        self.config = BertConfig(
            vocab_size=vocab_size,
            num_hidden_layers=num_layers,
            hidden_size=embedding_dim,
            num_attention_heads=num_attention_heads,
            intermediate_size=4 * embedding_dim,
            max_position_embeddings=max_rank_embeddings,
            use_flash_attn=use_flash_attn
        )

        # Transformer encoder
        self.EpiAgent_transformer = BertEncoder(self.config)

        # Classifier module
        self.classifier = classifier(embedding_dim=embedding_dim, num_classes=num_classes)

    def forward(self, input_ids, need_cell_embeddings=False, calculate_loss=False, target=None):
        """
        Forward pass for the EpiAgent_supervised model.

        Args:
            input_ids (tensor): Input tensor of shape [batch_size, seq_len].
            need_cell_embeddings (bool): If True, outputs the cell embeddings (default is False).
            calculate_loss (bool): If True, computes the loss using the provided target (default is False).
            target (tensor): Ground truth labels for loss computation (required if calculate_loss is True).

        Returns:
            tensor: Predicted labels.
            tensor (optional): Computed loss if calculate_loss is True.
            tensor (optional): Cell embeddings if need_cell_embeddings is True.
        """
        # Embedding layers
        word_embedded = self.cCRE_embedding(input_ids)
        position_indices = torch.arange(input_ids.shape[1]).unsqueeze(0).expand_as(input_ids).to(input_ids.device)
        position_embedded = self.rank_embedding(position_indices)

        # Transformer encoder forward pass
        encoder_outputs = self.EpiAgent_transformer(word_embedded + position_embedded, key_padding_mask=(input_ids != 0))

        # Extract cell embeddings (CLS token representation)
        cell_embeddings = encoder_outputs[:, 0, :]

        # Pass embeddings through the classifier
        if calculate_loss:
            predicted_label, loss = self.classifier(cell_embeddings, calculate_loss=True, target=target)
            if need_cell_embeddings:
                return predicted_label, loss, cell_embeddings
            return predicted_label, loss

        predicted_label = self.classifier(cell_embeddings, calculate_loss=False)
        if need_cell_embeddings:
            return predicted_label, cell_embeddings

        return predicted_label

class classifier_simple(nn.Module):
    """
    A simplified neural network classifier for cell type classification.

    Args:
        embedding_dim (int): Dimension of the input embeddings.
        num_classes (int): Number of cell type classes to predict.
    """
    def __init__(self, embedding_dim, num_classes):
        super(classifier_simple, self).__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.loss = FocalLoss()

    def forward(self, X, calculate_loss=False, target=None):
        """
        Forward pass for the simplified classifier.

        Args:
            X (tensor): Input features.
            calculate_loss (bool): Whether to calculate loss (default: False).
            target (tensor): Ground truth labels (required if calculate_loss is True).

        Returns:
            tensor: Predicted labels.
            tensor (optional): Computed loss if calculate_loss is True.
        """
        predicted_label = self.fc(X)
        if calculate_loss:
            loss = self.loss(predicted_label, target)
            return predicted_label, loss
        return predicted_label

class classifier_simple_1(nn.Module):
    """
    A simplified neural network classifier using CrossEntropyLoss for classification tasks.

    Args:
        embedding_dim (int): Dimension of the input embeddings.
        num_classes (int): Number of cell type classes to predict.
    """
    def __init__(self, embedding_dim, num_classes):
        super(classifier_simple_1, self).__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, X, calculate_loss=False, target=None):
        """
        Forward pass for the simplified classifier with CrossEntropyLoss.

        Args:
            X (tensor): Input features.
            calculate_loss (bool): Whether to calculate loss (default: False).
            target (tensor): Ground truth labels (required if calculate_loss is True).

        Returns:
            tensor: Predicted labels.
            tensor (optional): Computed loss if calculate_loss is True.
        """
        predicted_label = self.fc(X)
        if calculate_loss:
            loss = self.loss(predicted_label, target)
            return predicted_label, loss
        return predicted_label