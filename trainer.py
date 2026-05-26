import torch
import torch.nn as nn
import copy
import os
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             confusion_matrix, precision_recall_curve, precision_score)
from prettytable import PrettyTable
from tqdm import tqdm
from domain_adaptator import ReverseLayerF


def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss


def cross_entropy_logits(linear_output, label):
    import torch.nn.functional as F
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    y_hat = class_output.max(1)[1]
    loss = nn.NLLLoss()(class_output,
                        label.type_as(y_hat).view(label.size(0)))
    return n, loss


def entropy_logits(linear_output):
    import torch.nn.functional as F
    p = F.softmax(linear_output, dim=1)
    return -torch.sum(p * torch.log(p + 1e-5), dim=1)


# ──────────────────────────────────────────────────────────────────────
class Trainer:
    """Standard trainer (no DA). Mirrors DrugBAN Trainer structure."""

    def __init__(self, model, optim, device,
                 train_dataloader, val_dataloader, test_dataloader,
                 **config):
        self.model  = model
        self.optim  = optim
        self.device = device
        self.epochs = config["SOLVER"]["EPOCHS"]
        self.current_epoch = 0

        self.train_dataloader = train_dataloader
        self.val_dataloader   = val_dataloader
        self.test_dataloader  = test_dataloader

        self.batch_size  = config["SOLVER"]["BATCH_SIZE"]
        self.nb_training = len(self.train_dataloader)
        self.step        = 0

        self.best_model = None
        self.best_epoch = None
        self.best_auroc = 0

        self.train_loss_epoch = []
        self.val_loss_epoch   = []
        self.val_auroc_epoch  = []
        self.test_metrics     = {}
        self.config    = config
        self.output_dir = config["RESULT"]["OUTPUT_DIR"]

        self.train_table = PrettyTable(["# Epoch", "Train_loss"])
        self.val_table   = PrettyTable(["# Epoch", "AUROC", "AUPRC", "Val_loss"])
        self.test_table  = PrettyTable(["# Best Epoch", "AUROC", "AUPRC", "F1",
                                        "Sensitivity", "Specificity", "Accuracy",
                                        "Threshold", "Test_loss"])

    # ── training loop ─────────────────────────────────────────────────
    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1
            train_loss = self.train_epoch()
            self.train_table.add_row(
                ["epoch " + str(self.current_epoch), float2str(train_loss)])
            self.train_loss_epoch.append(train_loss)

            auroc, auprc, val_loss = self.test(dataloader="val")
            self.val_table.add_row(
                ["epoch " + str(self.current_epoch)] +
                list(map(float2str, [auroc, auprc, val_loss])))
            self.val_loss_epoch.append(val_loss)
            self.val_auroc_epoch.append(auroc)
            print(f"Validation at Epoch {self.current_epoch} | "
                  f"val_loss={val_loss:.4f}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")

            if auroc >= self.best_auroc:
                self.best_model = copy.deepcopy(self.model)
                self.best_auroc = auroc
                self.best_epoch = self.current_epoch

        # Final test on best model
        auroc, auprc, f1, sensitivity, specificity, accuracy, \
            test_loss, thred_optim, precision = self.test(dataloader="test")
        self.test_table.add_row(
            ["epoch " + str(self.best_epoch)] +
            list(map(float2str, [auroc, auprc, f1, sensitivity,
                                 specificity, accuracy, thred_optim, test_loss])))
        print(f"\nTest at Best Model (Epoch {self.best_epoch}) | "
              f"test_loss={test_loss:.4f}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
              f"F1={f1:.4f}  Sens={sensitivity:.4f}  Spec={specificity:.4f}  "
              f"Acc={accuracy:.4f}  Threshold={thred_optim:.4f}")

        self.test_metrics = dict(auroc=auroc, auprc=auprc, f1=f1,
                                 sensitivity=sensitivity, specificity=specificity,
                                 accuracy=accuracy, thred_optim=thred_optim,
                                 test_loss=test_loss, best_epoch=self.best_epoch,
                                 precision=precision)
        self.save_result()
        return self.test_metrics

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        for v_peptide, v_mhc, labels in tqdm(self.train_dataloader,
                                              desc=f"Epoch {self.current_epoch}"):
            self.step += 1
            v_peptide = v_peptide.to(self.device)
            v_mhc     = v_mhc.to(self.device)
            labels    = labels.float().to(self.device)
            self.optim.zero_grad()
            _, _, _, score = self.model(v_peptide, v_mhc)
            n, loss = binary_cross_entropy(score, labels)
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()
        loss_epoch /= len(self.train_dataloader)
        print(f"Training at Epoch {self.current_epoch} | train_loss={loss_epoch:.4f}")
        return loss_epoch

    # ── evaluation ────────────────────────────────────────────────────
    def test(self, dataloader="test"):
        if dataloader == "val":
            data_loader = self.val_dataloader
            eval_model  = self.model
        elif dataloader == "test":
            data_loader = self.test_dataloader
            eval_model  = self.best_model
        else:
            raise ValueError(f"Unknown key: {dataloader}")

        test_loss = 0
        y_label, y_pred = [], []
        with torch.no_grad():
            eval_model.eval()
            for v_peptide, v_mhc, labels in data_loader:
                v_peptide = v_peptide.to(self.device)
                v_mhc     = v_mhc.to(self.device)
                labels    = labels.float().to(self.device)
                _, _, _, score = eval_model(v_peptide, v_mhc)
                n, loss = binary_cross_entropy(score, labels)
                test_loss += loss.item()
                y_label += labels.cpu().tolist()
                y_pred  += n.cpu().tolist()

        test_loss /= len(data_loader)
        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)

        if dataloader == "val":
            return auroc, auprc, test_loss

        fpr, tpr, thresholds = roc_curve(y_label, y_pred)
        prec, recall, _      = precision_recall_curve(y_label, y_pred)
        precision            = tpr / (tpr + fpr + 1e-8)
        f1                   = 2 * precision * tpr / (tpr + precision + 1e-5)
        thred_optim          = thresholds[5:][np.argmax(f1[5:])]
        y_pred_s             = [1 if p >= thred_optim else 0 for p in y_pred]
        cm                   = confusion_matrix(y_label, y_pred_s)
        accuracy             = (cm[0, 0] + cm[1, 1]) / cm.sum()
        sensitivity          = cm[0, 0] / (cm[0, 0] + cm[0, 1] + 1e-8)
        specificity          = cm[1, 1] / (cm[1, 0] + cm[1, 1] + 1e-8)
        precision1           = precision_score(y_label, y_pred_s)
        return (auroc, auprc, float(np.max(f1[5:])), sensitivity, specificity,
                accuracy, test_loss, float(thred_optim), precision1)

    # ── save ──────────────────────────────────────────────────────────
    def save_result(self):
        os.makedirs(self.output_dir, exist_ok=True)
        if self.config["RESULT"]["SAVE_MODEL"]:
            torch.save(self.best_model.state_dict(),
                       os.path.join(self.output_dir,
                                    f"best_model_epoch_{self.best_epoch}.pth"))
            torch.save(self.model.state_dict(),
                       os.path.join(self.output_dir,
                                    f"model_epoch_{self.current_epoch}.pth"))
        torch.save({"train_epoch_loss": self.train_loss_epoch,
                    "val_epoch_loss":   self.val_loss_epoch,
                    "test_metrics":     self.test_metrics,
                    "config":           self.config},
                   os.path.join(self.output_dir, "result_metrics.pt"))
        for name, table in [("train", self.train_table),
                             ("valid", self.val_table),
                             ("test",  self.test_table)]:
            with open(os.path.join(self.output_dir,
                                   f"{name}_markdowntable.txt"), "w") as f:
                f.write(table.get_string())
        print(f"\nResults saved to: {self.output_dir}")


# ──────────────────────────────────────────────────────────────────────
class DATrainer(Trainer):
    """
    Domain Adaptation trainer using CDAN (Conditional Domain Adversarial Network).
    Source domain: labeled HLA-X data.
    Target domain: unlabeled HLA-Y data (labels ignored during DA training).

    Inherits Trainer for test() and save_result(); overrides train() and train_epoch().
    """

    def __init__(self, model, optim, device,
                 multi_loader, val_loader, test_loader,
                 discriminator, optim_da, alpha=1.0, **config):
        # Pass multi_loader as train_dataloader so parent test() / save_result() work
        super().__init__(model, optim, device,
                         multi_loader, val_loader, test_loader, **config)
        self.domain_dmm  = discriminator
        self.optim_da    = optim_da
        self.alpha       = alpha
        self.da_init_epoch = config["DA"]["INIT_EPOCH"]
        self.init_lamb_da  = config["DA"]["LAMB_DA"]
        self.use_entropy   = config["DA"]["USE_ENTROPY"]
        self.n_class       = config["DECODER"]["BINARY"]

        # extend tables
        self.train_table = PrettyTable(
            ["# Epoch", "Train_loss", "Model_loss", "DA_loss", "Lambda_DA"])

    # ── main loop (override) ───────────────────────────────────────────
    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1
            train_loss, model_loss, da_loss, lamb = self.train_da_epoch()
            self.train_table.add_row(
                ["epoch " + str(self.current_epoch)] +
                list(map(float2str, [train_loss, model_loss, da_loss, lamb])))
            self.train_loss_epoch.append(train_loss)

            auroc, auprc, val_loss = self.test(dataloader="val")
            self.val_table.add_row(
                ["epoch " + str(self.current_epoch)] +
                list(map(float2str, [auroc, auprc, val_loss])))
            self.val_loss_epoch.append(val_loss)
            self.val_auroc_epoch.append(auroc)
            print(f"Validation at Epoch {self.current_epoch} | "
                  f"val_loss={val_loss:.4f}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
                  f"[DA λ={lamb:.3f}  model_loss={model_loss:.4f}  da_loss={da_loss:.4f}]")

            if auroc >= self.best_auroc:
                self.best_model = copy.deepcopy(self.model)
                self.best_auroc = auroc
                self.best_epoch = self.current_epoch

        # Final test
        auroc, auprc, f1, sensitivity, specificity, accuracy, \
            test_loss, thred_optim, precision = self.test(dataloader="test")
        self.test_table.add_row(
            ["epoch " + str(self.best_epoch)] +
            list(map(float2str, [auroc, auprc, f1, sensitivity,
                                 specificity, accuracy, thred_optim, test_loss])))
        print(f"\nTest at Best Model (Epoch {self.best_epoch}) | "
              f"test_loss={test_loss:.4f}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
              f"F1={f1:.4f}  Sens={sensitivity:.4f}  Spec={specificity:.4f}  "
              f"Acc={accuracy:.4f}  Threshold={thred_optim:.4f}")
        self.test_metrics = dict(auroc=auroc, auprc=auprc, f1=f1,
                                 sensitivity=sensitivity, specificity=specificity,
                                 accuracy=accuracy, thred_optim=thred_optim,
                                 test_loss=test_loss, best_epoch=self.best_epoch,
                                 precision=precision)
        self.save_result()
        return self.test_metrics

    # ── DA epoch ──────────────────────────────────────────────────────
    def train_da_epoch(self):
        self.model.train()
        self.domain_dmm.train()

        total_loss_epoch = model_loss_epoch = da_loss_epoch = 0
        lamb = self.init_lamb_da if self.current_epoch >= self.da_init_epoch else 0.0

        for batch_s, batch_t in tqdm(self.train_dataloader,
                                     desc=f"Epoch {self.current_epoch} [DA]"):
            self.step += 1

            # ── source batch (labeled) ─────────────────────────────────
            v_p_s, v_m_s, labels = (batch_s[0].to(self.device),
                                     batch_s[1].to(self.device),
                                     batch_s[2].float().to(self.device))
            # ── target batch (unlabeled; labels NOT used) ──────────────
            v_p_t, v_m_t = batch_t[0].to(self.device), batch_t[1].to(self.device)

            self.optim.zero_grad()
            self.optim_da.zero_grad()

            # Forward source
            _, _, f_s, score_s = self.model(v_p_s, v_m_s)
            n_s, model_loss = binary_cross_entropy(score_s, labels)

            if self.current_epoch >= self.da_init_epoch:
                # Forward target
                _, _, f_t, score_t = self.model(v_p_t, v_m_t)

                # CDAN: condition on classifier softmax output
                softmax_s = torch.sigmoid(score_s).detach()  # (B,1)
                softmax_t = torch.sigmoid(score_t).detach()

                # Outer product of BAN feature × softmax → discriminator input
                feature_s = f_s * softmax_s          # (B, IN_DIM)
                feature_t = f_t * softmax_t

                # Gradient reversal
                rev_s = ReverseLayerF.apply(feature_s, self.alpha)
                rev_t = ReverseLayerF.apply(feature_t, self.alpha)

                # Discriminator: 0 = source, 1 = target
                disc_s = self.domain_dmm(rev_s)
                disc_t = self.domain_dmm(rev_t)

                if self.use_entropy:
                    # Entropy weighting (higher entropy → uncertain → down-weight)
                    ent_s = torch.sigmoid(score_s) * torch.log(torch.sigmoid(score_s) + 1e-5)
                    ent_t = torch.sigmoid(score_t) * torch.log(torch.sigmoid(score_t) + 1e-5)
                    w_s = (1.0 + torch.exp(ent_s)).squeeze(1).detach()
                    w_t = (1.0 + torch.exp(ent_t)).squeeze(1).detach()
                    w_s = w_s / w_s.sum()
                    w_t = w_t / w_t.sum()
                else:
                    w_s = w_t = None

                domain_label_s = torch.zeros(disc_s.size(0), 1).to(self.device)
                domain_label_t = torch.ones( disc_t.size(0), 1).to(self.device)

                bce = nn.BCEWithLogitsLoss(reduction='none')
                loss_s = bce(disc_s, domain_label_s).squeeze()
                loss_t = bce(disc_t, domain_label_t).squeeze()

                if w_s is not None:
                    da_loss = (w_s * loss_s).sum() + (w_t * loss_t).sum()
                else:
                    da_loss = loss_s.mean() + loss_t.mean()

                loss = model_loss + lamb * da_loss
                da_loss_epoch += da_loss.item()
            else:
                loss = model_loss

            loss.backward()
            self.optim.step()
            if self.current_epoch >= self.da_init_epoch:
                self.optim_da.step()

            total_loss_epoch += loss.item()
            model_loss_epoch += model_loss.item()

        nb = len(self.train_dataloader)
        total_loss_epoch /= nb
        model_loss_epoch /= nb
        da_loss_epoch    /= nb

        print(f"Training at Epoch {self.current_epoch} | "
              f"total={total_loss_epoch:.4f}  model={model_loss_epoch:.4f}  "
              f"da={da_loss_epoch:.4f}  λ={lamb:.3f}")
        return total_loss_epoch, model_loss_epoch, da_loss_epoch, lamb
