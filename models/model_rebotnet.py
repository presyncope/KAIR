import torch
from models.model_plain import ModelPlain
from typing import cast, override
from utils.utils_regularizers import regularizer_orth, regularizer_clip
import torch.nn.functional as F

class ModelRebotNet(ModelPlain):
    """Train video restoration with pixel loss"""
    def __init__(self, opt: dict):
        super().__init__(opt)

    @override
    def feed_data(self, data): #type: ignore[override]
        self.L = cast(torch.Tensor, data['L'].to(self.device))
        self.H = cast(torch.Tensor, data['H'].to(self.device))

    @override
    def optimize_parameters(self, current_step: int): # type: ignore[override]
        self.G_optimizer.zero_grad(set_to_none=True)

        T = self.H.shape[1]
        prev_lo = self.L[:, 0, ...]
        running_loss = torch.tensor(0.0, device=self.device)

        for tau in range(T):
            x1 = prev_lo.unsqueeze(1)
            x2 = self.L[:, tau, ...].unsqueeze(1)
            x = torch.cat([x1, x2], dim=1)

            recon = self.netG(x)
            with torch.no_grad():
                prev_lo = F.interpolate(recon.detach(), scale_factor=0.25, mode='bilinear')

            G_loss = self.G_lossfn_weight * self.G_lossfn(recon, self.H[:, tau, ...])
            (G_loss / T).backward()
            running_loss += G_loss.detach()

        # clip grad: `clip_grad_norm` helps prevent the exploding gradient problem.
        clip_val = self.opt_train.get("G_optimizer_clipgrad", 0) or 0
        if clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=clip_val, norm_type=2)

        self.G_optimizer.step()

        # regularizer
        orth_step = self.opt_train.get('G_regularizer_orthstep', 0) or 0
        clip_step = self.opt_train.get('G_regularizer_clipstep', 0) or 0
        if orth_step > 0 and current_step % orth_step == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            with torch.no_grad():
                self.netG.apply(regularizer_orth)
        if clip_step > 0 and current_step % clip_step == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            with torch.no_grad():
                self.netG.apply(regularizer_clip)

        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        self.log_dict['G_loss'] = (running_loss / T).item()

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    @override
    def test(self):
        self.netG.eval()
        outputs = []

        with torch.no_grad():
            T = self.L.shape[1]
            prev_lo = self.L[:, 0, ...]

            for tau in range(T):
                x1 = prev_lo.unsqueeze(1)
                x2 = self.L[:, tau, ...].unsqueeze(1)
                x = torch.cat([x1, x2], dim=1)

                recon = self.netG(x)
                outputs.append(recon)
                prev_lo = F.interpolate(recon, scale_factor=0.25, mode="bilinear")
        self.E = torch.stack(outputs, dim=1)
        self.netG.train()

    # ----------------------------------------
    # load the state_dict of the network
    # ----------------------------------------
    def load_network(self, load_path, network, strict=True, param_key='params'):
        network = self.get_bare_model(network)
        state_dict = torch.load(load_path)
        if param_key in state_dict.keys():
            state_dict = state_dict[param_key]
        self._print_different_keys_loading(network, state_dict, strict)
        network.load_state_dict(state_dict, strict=strict)

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        crt_net = self.get_bare_model(crt_net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        if crt_net_keys != load_net_keys:
            print('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                print(f'  {v}')
            print('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                print(f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    print(f'Size different, ignore [{k}]: crt_net: '
                                   f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)
