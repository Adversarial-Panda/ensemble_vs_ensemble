import torch
import faiss
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from collections import Counter  
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.special import digamma, gammaln


######################
####### 1) MV-S ###### 
###################### 

class SoftVotingEnsemble:
    def __init__(self, models, device='cpu'):
        self.models = models
        self.device = device
        for model in self.models:
            model.eval().to(device)

    def predict(self, images, return_probs=False):
        probs = []
        with torch.no_grad():
            for model in self.models:
                outputs = model(images.to(self.device))
                softmaxed = torch.softmax(outputs, dim=1)
                probs.append(softmaxed.cpu().numpy())
        mean_probs = np.mean(np.stack(probs), axis=0)
        if return_probs:
            return mean_probs
        return np.argmax(mean_probs, axis=1)

    def predict_single_with_probs(self, image):
        image = image.unsqueeze(0)  # Shape [1, C, H, W]
        with torch.no_grad():
            model_probs = []
            for model in self.models:
                logits = model(image.to(self.device))
                softmaxed = torch.softmax(logits, dim=1)
                model_probs.append(softmaxed.cpu().numpy())
            mean_probs = np.mean(np.stack(model_probs), axis=0)
            probs = mean_probs[0]
            pred = np.argmax(probs)
            return probs, pred


######################
####### 2) MV-H ###### 
###################### 


class HardVotingEnsemble:
    def __init__(self, models, device='cpu'):
        self.models = models
        self.device = device
        for model in self.models:
            model.eval().to(device)

    def predict(self, images):
        all_preds = []
        with torch.no_grad():
            for model in self.models:
                outputs = model(images.to(self.device))
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu().numpy())

        # shape: [n_models, batch_size] → transpose to [batch_size, n_models]
        all_preds = np.stack(all_preds, axis=0).T  

        # majority vote for each sample
        final_preds = []
        for preds in all_preds:
            most_common = Counter(preds).most_common(1)[0][0]
            final_preds.append(most_common)

        return np.array(final_preds)

    def predict_single_with_probs(self, image):
        image = image.unsqueeze(0)  # Shape [1, C, H, W]
        votes = []
        with torch.no_grad():
            for model in self.models:
                logits = model(image.to(self.device))
                pred = torch.argmax(logits, dim=1).item()
                votes.append(pred)

        # majority voting
        final_pred = Counter(votes).most_common(1)[0][0]
        return votes, final_pred
        


######################
###### 3) W-MV-S ##### 
###################### 

class WeightedSoftVotingEnsemble:
    """
    Weighted soft voting ensemble.

    Weights are automatically computed from validation accuracy.
    """

    def __init__(self, models, device='cpu'):
        self.models = models
        self.device = device

        for model in self.models:
            model.eval().to(device)

        self.weights = None

    def fit_weights(self, val_loader):
        """
        Compute model weights using validation accuracy.

        val_loader:
            PyTorch DataLoader returning (images, labels)
        """

        accuracies = []

        for model in self.models:

            correct = 0
            total = 0

            with torch.no_grad():

                for images, labels in val_loader:

                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(images)
                    preds = outputs.argmax(dim=1)

                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            acc = correct / total
            accuracies.append(acc)

        accuracies = np.array(accuracies)

        # Normalize weights
        self.weights = accuracies / accuracies.sum()

        print("Model Weights:")
        for i, w in enumerate(self.weights):
            print(f"Model {i}: {w:.4f}")

    def predict(self, images, return_probs=False):

        if self.weights is None:
            raise ValueError("Call fit_weights() first.")

        weighted_probs = []

        with torch.no_grad():

            for weight, model in zip(self.weights, self.models):

                outputs = model(images.to(self.device))

                probs = F.softmax(outputs, dim=1)

                weighted_probs.append(
                    probs.cpu().numpy() * weight
                )

        final_probs = np.sum(np.stack(weighted_probs), axis=0)

        if return_probs:
            return final_probs

        return np.argmax(final_probs, axis=1)

    def predict_single_with_probs(self, image):

        if self.weights is None:
            raise ValueError("Call fit_weights() first.")

        image = image.unsqueeze(0)

        weighted_probs = []

        with torch.no_grad():

            for weight, model in zip(self.weights, self.models):

                logits = model(image.to(self.device))

                probs = F.softmax(logits, dim=1)

                weighted_probs.append(
                    probs.cpu().numpy() * weight
                )

        final_probs = np.sum(np.stack(weighted_probs), axis=0)

        probs = final_probs[0]
        pred = np.argmax(probs)

        return probs, pred



######################
###### 4) W-MV-H ##### 
######################

class WeightedHardVotingEnsemble:
    """
    Weighted hard voting ensemble.

    Each model vote is weighted using validation accuracy.
    """

    def __init__(self, models, device='cpu'):
        self.models = models
        self.device = device

        for model in self.models:
            model.eval().to(device)

        self.weights = None

    def fit_weights(self, val_loader):
        """
        Compute model weights using validation accuracy.
        """

        accuracies = []

        for model in self.models:

            correct = 0
            total = 0

            with torch.no_grad():

                for images, labels in val_loader:

                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(images)

                    preds = outputs.argmax(dim=1)

                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            acc = correct / total
            accuracies.append(acc)

        accuracies = np.array(accuracies)

        # Normalize weights
        self.weights = accuracies / accuracies.sum()

        print("Model Weights:")
        for i, w in enumerate(self.weights):
            print(f"Model {i}: {w:.4f}")

    def predict(self, images):

        if self.weights is None:
            raise ValueError("Call fit_weights() first.")

        batch_votes = []

        with torch.no_grad():

            for model in self.models:

                outputs = model(images.to(self.device))

                preds = outputs.argmax(dim=1)

                batch_votes.append(preds.cpu().numpy())

        # [M, B] -> [B, M]
        batch_votes = np.stack(batch_votes, axis=0).T

        final_preds = []

        for sample_votes in batch_votes:

            class_scores = defaultdict(float)

            for model_idx, pred_class in enumerate(sample_votes):

                class_scores[pred_class] += self.weights[model_idx]

            final_pred = max(class_scores, key=class_scores.get)

            final_preds.append(final_pred)

        return np.array(final_preds)

    def predict_single_with_probs(self, image):

        if self.weights is None:
            raise ValueError("Call fit_weights() first.")

        image = image.unsqueeze(0)

        votes = []

        with torch.no_grad():

            for model in self.models:

                logits = model(image.to(self.device))

                pred = logits.argmax(dim=1).item()

                votes.append(pred)

        class_scores = defaultdict(float)

        for model_idx, pred_class in enumerate(votes):

            class_scores[pred_class] += self.weights[model_idx]

        final_pred = max(class_scores, key=class_scores.get)

        return votes, final_pred

        
######################
###### 5) Stack ###### 
###################### 
class MetaClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def train_meta_model_fast(
    base_models,
    meta_model,
    dataloader,
    device,
    epochs=20,
    lr=1e-3,
    use_amp=True
):
    base_models = nn.ModuleList(base_models).to(device)
    meta_model.to(device)

    for m in base_models:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    optimizer = torch.optim.Adam(meta_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    meta_model.train()

    for epoch in range(epochs):
        correct, total, loss_sum = 0, 0, 0.0

        for images, labels in tqdm(dataloader, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # ---- FAST base model inference ----
            with torch.inference_mode():
                feats = torch.cat(
                    [torch.softmax(m(images), dim=1) for m in base_models],
                    dim=1
                )

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = meta_model(feats)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        print(
            f"[Meta] Epoch {epoch+1:02d} | "
            f"loss={loss_sum:.3f} | acc={100*correct/total:.2f}%"
        )

    meta_model.eval()


class Stacking(nn.Module):
    def __init__(self, base_models, meta_model):
        super().__init__()
        self.base_models = nn.ModuleList(base_models)
        self.meta_model = meta_model

    @torch.inference_mode()
    def forward(self, images):
        probs = torch.cat(
            [torch.softmax(m(images), dim=1) for m in self.base_models],
            dim=1
        )
        return self.meta_model(probs)

    @torch.inference_mode()
    def predict(self, images, return_probs=False):
        logits = self.forward(images)
        probs = torch.softmax(logits, dim=1)
        if return_probs:
            return probs
        return probs.argmax(dim=1)


######################
###### 6) KNOP ####### 
###################### 

class KNOP:
    def __init__(self, models, k=7, temperature=2.0, device="cuda"):
        """
        models: list of PyTorch models
        k: number of neighbors in decision space
        temperature: for scaling logits
        """
        self.models = models
        self.M = len(models)
        self.k = k
        self.T = temperature
        self.device = device

        self.DSEL_decisions = None
        self.DSEL_labels = None
        self.correctness = None

    # -----------------------------
    # 🔷 Step 1: Get model outputs
    # -----------------------------
    def _get_model_outputs(self, x):
        """
        Returns logits for all models
        Shape: (M, C)
        """
        outputs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(x.to(self.device))
                outputs.append(logits.squeeze(0))
        return torch.stack(outputs)  # (M, C)

    # -----------------------------
    # 🔷 Step 2: Decision vector
    # -----------------------------
    def _decision_vector(self, x):
        """
        Flattened decision profile
        Shape: (M*C,)
        """
        logits = self._get_model_outputs(x)  # (M, C)

        # Temperature scaling
        probs = F.softmax(logits / self.T, dim=1)

        return probs.view(-1)  # flatten

    # -----------------------------
    # 🔷 Step 3: Fit on DSEL
    # -----------------------------
    def fit(self, dataloader):
        """
        dataloader: DSEL loader (batch_size=1 recommended)
        """
        decision_vectors = []
        labels = []
        correctness_matrix = []

        print("Building DSEL decision space...")

        for x, y in tqdm(dataloader):
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self._get_model_outputs(x)  # (M, C)
            probs = F.softmax(logits / self.T, dim=1)

            # Decision vector
            d_vec = probs.view(-1).cpu().numpy()
            decision_vectors.append(d_vec)

            labels.append(y.item())

            # Correctness per model
            preds = torch.argmax(logits, dim=1)
            correct = (preds == y).float().cpu().numpy()
            correctness_matrix.append(correct)

        self.DSEL_decisions = np.array(decision_vectors)   # (N, M*C)
        self.DSEL_labels = np.array(labels)                # (N,)
        self.correctness = np.array(correctness_matrix)    # (N, M)

        # Normalize decision vectors for cosine similarity
        self.DSEL_decisions = self._normalize(self.DSEL_decisions)

    # -----------------------------
    # 🔷 Utility: Normalize
    # -----------------------------
    def _normalize(self, X):
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    # -----------------------------
    # 🔷 Step 4: Predict
    # -----------------------------
    def predict(self, x):
        """
        Strict KNOP prediction
        """
        # -----------------------------
        # Step 1: Decision vector
        # -----------------------------
        d_q = self._decision_vector(x).cpu().numpy()
        d_q = d_q / (np.linalg.norm(d_q) + 1e-8)
    
        # -----------------------------
        # Step 2: Similarity (cosine)
        # -----------------------------
        sims = np.dot(self.DSEL_decisions, d_q)  # (N,)
    
        # Top-k neighbors
        idx = np.argsort(-sims)[:self.k]
    
        # -----------------------------
        # Step 3: KNOP Oracle Selection
        # -----------------------------
        selected_classifiers = []
    
        for m in range(self.M):
            # Check if classifier m is correct on ALL neighbors
            correct_all = True
            for i in idx:
                if self.correctness[i, m] == 0:
                    correct_all = False
                    break
    
            if correct_all:
                selected_classifiers.append(m)
    
        # -----------------------------
        # Step 4: Fallback if empty
        # -----------------------------
        if len(selected_classifiers) == 0:
            selected_classifiers = list(range(self.M))  # use all
    
        # -----------------------------
        # Step 5: Final prediction
        # -----------------------------
        logits = self._get_model_outputs(x)  # (M, C)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    
        selected_probs = probs[selected_classifiers]  # subset
    
        final = np.mean(selected_probs, axis=0)
    
        pred = np.argmax(final)
    
        return pred, final

    # -----------------------------
    # 🔷 Batch prediction (optional)
    # -----------------------------
    def predict_batch(self, dataloader):
        preds = []
        for x, _ in tqdm(dataloader):
            pred, _ = self.predict(x)
            preds.append(pred)
        return np.array(preds)



#########################
####### 7) KNORAE ####### 
#########################

class KNORAE:
    """
    Pure KNORA-E implementation using DINO embeddings + FAISS.

    KNORA-E:
    Keep only classifiers that correctly classify ALL neighbors.
    If none survive, reduce k recursively.
    """

    def __init__(self, dsel_dataset, pool, device):

        self.device = device
        self.pool = pool
        self.dsel_dataset = dsel_dataset

        self.dsel_loader = DataLoader(
            dsel_dataset,
            batch_size=32,
            shuffle=False
        )

        self.dino_model = timm.create_model(
            "vit_base_patch16_224.dino",
            pretrained=True
        ).to(device).eval()

    # ======================================================
    # Build RoC
    # ======================================================
    def fit(self):

        embs = []
        labels = []

        with torch.no_grad():

            for x, y in tqdm(self.dsel_loader):

                x = x.to(self.device)

                f = self.dino_model.forward_features(x)[:, 0, :]

                embs.append(f.cpu())
                labels.append(y)

        self.dsel_embeddings = torch.cat(embs).numpy().astype("float32")
        self.dsel_labels = torch.cat(labels).numpy()

        self.index = faiss.IndexFlatL2(
            self.dsel_embeddings.shape[1]
        )

        self.index.add(self.dsel_embeddings)

    # ======================================================
    # Predict
    # ======================================================
    def predict(self, test_img, k=7):

        with torch.no_grad():

            emb = self.dino_model.forward_features(
                test_img.unsqueeze(0).to(self.device)
            )[:, 0, :].cpu().numpy().astype("float32")

        _, neighbors = self.index.search(emb, k)

        idxs = neighbors[0]

        roc_imgs = torch.stack([
            self.dsel_dataset[i][0]
            for i in idxs
        ]).to(self.device)

        local_labels = self.dsel_labels[idxs]

        selected_models = []

        current_k = k

        # ==================================================
        # KNORA-E elimination
        # ==================================================
        while current_k > 0:

            selected_models = []

            current_imgs = roc_imgs[:current_k]
            current_labels = local_labels[:current_k]

            for clf in self.pool:

                clf.eval()

                with torch.no_grad():

                    out = clf(current_imgs)

                    preds = out.argmax(dim=1).cpu().numpy()

                    correct = (preds == current_labels).sum()

                    # Must classify ALL neighbors correctly
                    if correct == current_k:
                        selected_models.append(clf)

            if len(selected_models) > 0:
                break

            current_k -= 1

        # ==================================================
        # Fallback
        # ==================================================
        if len(selected_models) == 0:
            selected_models = self.pool

        # ==================================================
        # Majority voting
        # ==================================================
        votes = []

        with torch.no_grad():

            for clf in selected_models:

                out = clf(
                    test_img.unsqueeze(0).to(self.device)
                )

                pred = out.argmax(dim=1).item()

                votes.append(pred)

        final_pred = max(set(votes), key=votes.count)

        return final_pred

#########################
####### 8) KNORAU ####### 
#########################

class KNORAU:
    """
    Pure KNORA-U implementation.

    KNORA-U:
    All classifiers with at least ONE correct neighbor survive.
    Votes are repeated according to local competence.
    """

    def __init__(self, dsel_dataset, pool, device):

        self.device = device
        self.pool = pool
        self.dsel_dataset = dsel_dataset

        self.dsel_loader = DataLoader(
            dsel_dataset,
            batch_size=32,
            shuffle=False
        )

        self.dino_model = timm.create_model(
            "vit_base_patch16_224.dino",
            pretrained=True
        ).to(device).eval()

    # ======================================================
    # Build RoC
    # ======================================================
    def fit(self):

        embs = []
        labels = []

        with torch.no_grad():

            for x, y in tqdm(self.dsel_loader):

                x = x.to(self.device)

                f = self.dino_model.forward_features(x)[:, 0, :]

                embs.append(f.cpu())
                labels.append(y)

        self.dsel_embeddings = torch.cat(embs).numpy().astype("float32")
        self.dsel_labels = torch.cat(labels).numpy()

        self.index = faiss.IndexFlatL2(
            self.dsel_embeddings.shape[1]
        )

        self.index.add(self.dsel_embeddings)

    # ======================================================
    # Predict
    # ======================================================
    def predict(self, test_img, k=7):

        with torch.no_grad():

            emb = self.dino_model.forward_features(
                test_img.unsqueeze(0).to(self.device)
            )[:, 0, :].cpu().numpy().astype("float32")

        _, neighbors = self.index.search(emb, k)

        idxs = neighbors[0]

        roc_imgs = torch.stack([
            self.dsel_dataset[i][0]
            for i in idxs
        ]).to(self.device)

        local_labels = self.dsel_labels[idxs]

        votes = []

        # ==================================================
        # KNORA-U union
        # ==================================================
        for clf in self.pool:

            clf.eval()

            with torch.no_grad():

                out = clf(roc_imgs)

                preds = out.argmax(dim=1).cpu().numpy()

                correct = int(
                    (preds == local_labels).sum()
                )

                # Must correctly classify
                # at least one neighbor
                if correct > 0:

                    test_out = clf(
                        test_img.unsqueeze(0).to(self.device)
                    )

                    test_pred = test_out.argmax(dim=1).item()

                    # Repeat vote according to competence
                    votes.extend([test_pred] * correct)

        # ==================================================
        # Fallback
        # ==================================================
        if len(votes) == 0:

            for clf in self.pool:

                with torch.no_grad():

                    out = clf(
                        test_img.unsqueeze(0).to(self.device)
                    )

                    pred = out.argmax(dim=1).item()

                    votes.append(pred)

        final_pred = max(set(votes), key=votes.count)

        return final_pred

#########################
######## 9) DUS ######### 
######################### 


def dirichlet_entropy(alpha):
    alpha = torch.clamp(alpha, min=1e-6)

    alpha0 = alpha.sum(dim=1, keepdim=True)
    K = alpha.size(1)

    log_B = gammaln(alpha).sum(dim=1) - gammaln(alpha0.squeeze(1))

    entropy = (
        log_B
        + (alpha0.squeeze(1) - K) * digamma(alpha0.squeeze(1))
        - ((alpha - 1) * digamma(alpha)).sum(dim=1)
    )

    return entropy


class DUS:

    def __init__(self, sub_models, device='cuda'):
        self.sub_models = sub_models
        self.device = device

        for model in self.sub_models:
            model.eval().to(device)

    def _forward_all(self, images):

        probs = []
        entropies = []

        with torch.no_grad():

            for model in self.sub_models:

                logits = model(images.to(self.device))

                # Evidential Dirichlet
                evidence = F.softplus(logits)
                alpha = evidence + 1.0

                prob = alpha / alpha.sum(dim=1, keepdim=True)

                entropy = dirichlet_entropy(alpha)

                probs.append(prob)
                entropies.append(entropy)

        probs = torch.stack(probs)          # [M,B,K]
        entropies = torch.stack(entropies)  # [M,B]

        return probs, entropies

    def predict(self, images):

        probs, entropies = self._forward_all(images)

        best_model_idx = torch.argmin(entropies, dim=0)

        B = images.size(0)
        batch_idx = torch.arange(B, device=self.device)

        selected_probs = probs[best_model_idx, batch_idx]

        preds = selected_probs.argmax(dim=1)

        return preds.cpu().numpy()


        
#########################
###### 10) CP-DEL ####### 
######################### 


class ClassSpecificDEL:
    def __init__(self, models, class_perf_matrix, threshold=0.9, device='cpu'):
        """
        models: list of trained models
        class_perf_matrix: numpy array [n_models, n_classes]
            -> accuracy of each model for each class
        threshold: selection threshold (e.g., 0.9)
        """
        self.models = models
        self.class_perf = class_perf_matrix
        self.threshold = threshold
        self.device = device

        for model in self.models:
            model.eval().to(device)

    def predict(self, images):
        batch_size = images.size(0)

        all_preds = []
        with torch.no_grad():
            for model in self.models:
                outputs = model(images.to(self.device))
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu().numpy())

        # shape → [batch_size, n_models]
        all_preds = np.stack(all_preds, axis=0).T  

        final_preds = []

        for i, sample_preds in enumerate(all_preds):
            # Step 1: preliminary prediction (majority of ALL models)
            prelim_class = Counter(sample_preds).most_common(1)[0][0]

            # Step 2: select models with high class-specific accuracy
            selected_preds = []
            for m_idx, pred in enumerate(sample_preds):
                if self.class_perf[m_idx, prelim_class] >= self.threshold:
                    selected_preds.append(pred)

            # Step 3: fallback if no model selected
            if len(selected_preds) == 0:
                selected_preds = sample_preds  # fallback to all models

            # Step 4: final majority voting
            final_pred = Counter(selected_preds).most_common(1)[0][0]
            final_preds.append(final_pred)

        return np.array(final_preds)

    def predict_single_with_probs(self, image):
        image = image.unsqueeze(0)

        votes = []
        with torch.no_grad():
            for model in self.models:
                logits = model(image.to(self.device))
                pred = torch.argmax(logits, dim=1).item()
                votes.append(pred)

        # preliminary prediction
        prelim_class = Counter(votes).most_common(1)[0][0]

        # select models
        selected_votes = []
        for m_idx, pred in enumerate(votes):
            if self.class_perf[m_idx, prelim_class] >= self.threshold:
                selected_votes.append(pred)

        if len(selected_votes) == 0:
            selected_votes = votes

        final_pred = Counter(selected_votes).most_common(1)[0][0]

        return {
            "all_votes": votes,
            "selected_votes": selected_votes,
            "final_prediction": final_pred
        }



def compute_class_performance(models, dataloader, num_classes, device):
    class_correct = [np.zeros(num_classes) for _ in models]
    class_total = [np.zeros(num_classes) for _ in models]

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)

            for m_idx, model in enumerate(models):
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

                for c in range(num_classes):
                    mask = (labels == c)
                    class_total[m_idx][c] += mask.sum().item()
                    class_correct[m_idx][c] += (preds[mask] == c).sum().item()

    class_perf = []
    for m_idx in range(len(models)):
        perf = class_correct[m_idx] / (class_total[m_idx] + 1e-8)
        class_perf.append(perf)

    return np.array(class_perf)