from typing import Optional
from cleandiffuser.nn_classifier import BaseNNClassifier
from .base import BaseClassifier
import torch

class RNDClassifier(BaseClassifier):
    def __init__(self, nn_classifier: BaseNNClassifier, target_model: BaseNNClassifier, reward_model: BaseNNClassifier = None, 
                 device: str = "cpu", optim_params: Optional[dict] = None,):
        super().__init__(nn_classifier, 0.995, None, optim_params, device)
        self.target_model = target_model.to(device)
        if reward_model is not None:
            self.reward_model = reward_model.to(device)

    def loss(self, x, noise, val):
        with torch.no_grad():
            rnd_target = self.target_model(x, noise, None)
        pred_R = self.model(x, noise, None)
        return ((pred_R - rnd_target) ** 2).mean()

    def update(self, x, noise, val):
        self.optim.zero_grad()
        loss = self.loss(x, noise, val)
        loss.backward()
        self.optim.step()
        self.ema_update()
        return {"loss": loss.item()}

    def logp(self, x, noise, c=None):
        return  self.reward_model(x, noise)-10*(((self.model_ema(x, noise)-self.target_model(x, noise, None))**2).sum(dim=1, keepdim=True))
