import torch
from torch.optim import Adam
from models.model_plain import ModelPlain
from typing import cast, override
from utils.utils_regularizers import regularizer_orth, regularizer_clip

class ModelRebotNet(ModelPlain):
    """Train video restoration with pixel loss"""
    def __init__(self, opt: dict):
        super().__init__(opt)
    
    @override
    def feed_data(self, data): #type: ignore[override]
        self.L = cast(torch.Tensor, data['L'].to(self.device))
        self.H = cast(torch.Tensor, data['H'].to(self.device))
    
    def netG_forward(self):
        t = self.L.shape[0]
        prev = self.L[0]
        outputs = []

        for i in range(t):
            recon = self.netG(self.L[i].unsqueeze(0), prev.unsqueeze(0))
            outputs.append(recon)
            prev = recon.squeeze(0).detach()
        
        self.E = torch.cat(outputs, dim=0)
        
    @override
    def optimize_parameters(self, current_step: int): # type: ignore[override]
        self.G_optimizer.zero_grad(set_to_none=True)

        self.netG_forward()
        G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
        G_loss.backward()

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
        self.log_dict['G_loss'] = G_loss.item()

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    @override
    def test(self):
        n = self.L.size(1)
        self.netG.eval()

        with torch.no_grad():
            self.E = self._test_video(self.L)

        self.netG.train()

    def _test_video(self, lq):
        '''test the video as a whole or as clips (divided temporally). '''

        num_frame_testing = self.opt['val'].get('num_frame_testing', 0)

        if num_frame_testing:
            # test as multiple clips if out-of-memory
            sf = self.opt['scale']
            num_frame_overlapping = self.opt['val'].get('num_frame_overlapping', 2)
            not_overlap_border = False
            b, d, c, h, w = lq.size()
            c = c - 1 if self.opt['netG'].get('nonblind_denoising', False) else c
            stride = num_frame_testing - num_frame_overlapping
            d_idx_list = list(range(0, d-num_frame_testing, stride)) + [max(0, d-num_frame_testing)]
            E = torch.zeros(b, d, c, h*sf, w*sf)
            W = torch.zeros(b, d, 1, 1, 1)

            for d_idx in d_idx_list:
                lq_clip = lq[:, d_idx:d_idx+num_frame_testing, ...]
                out_clip = self._test_clip(lq_clip)
                out_clip_mask = torch.ones((b, min(num_frame_testing, d), 1, 1, 1))

                if not_overlap_border:
                    if d_idx < d_idx_list[-1]:
                        out_clip[:, -num_frame_overlapping//2:, ...] *= 0
                        out_clip_mask[:, -num_frame_overlapping//2:, ...] *= 0
                    if d_idx > d_idx_list[0]:
                        out_clip[:, :num_frame_overlapping//2, ...] *= 0
                        out_clip_mask[:, :num_frame_overlapping//2, ...] *= 0

                E[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip)
                W[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip_mask)
            output = E.div_(W)
        else:
            # test as one clip (the whole video) if you have enough memory
            window_size = self.opt['netG'].get('window_size', [6,8,8])
            d_old = lq.size(1)
            d_pad = (d_old// window_size[0]+1)*window_size[0] - d_old
            lq = torch.cat([lq, torch.flip(lq[:, -d_pad:, ...], [1])], 1)
            output = self._test_clip(lq)
            output = output[:, :d_old, :, :, :]

        return output

    def _test_clip(self, lq):
        ''' test the clip as a whole or as patches. '''

        sf = self.opt['scale']
        window_size = self.opt['netG'].get('window_size', [6,8,8])
        size_patch_testing = self.opt['val'].get('size_patch_testing', 0)
        assert size_patch_testing % window_size[-1] == 0, 'testing patch size should be a multiple of window_size.'

        if size_patch_testing:
            # divide the clip to patches (spatially only, tested patch by patch)
            overlap_size = 20
            not_overlap_border = True

            # test patch by patch
            b, d, c, h, w = lq.size()
            c = c - 1 if self.opt['netG'].get('nonblind_denoising', False) else c
            stride = size_patch_testing - overlap_size
            h_idx_list = list(range(0, h-size_patch_testing, stride)) + [max(0, h-size_patch_testing)]
            w_idx_list = list(range(0, w-size_patch_testing, stride)) + [max(0, w-size_patch_testing)]
            E = torch.zeros(b, d, c, h*sf, w*sf)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = lq[..., h_idx:h_idx+size_patch_testing, w_idx:w_idx+size_patch_testing]
                    if hasattr(self, 'netE'):
                        out_patch = self.netE(in_patch).detach().cpu()
                    else:
                        out_patch = self.netG(in_patch).detach().cpu()

                    out_patch_mask = torch.ones_like(out_patch)

                    if not_overlap_border:
                        if h_idx < h_idx_list[-1]:
                            out_patch[..., -overlap_size//2:, :] *= 0
                            out_patch_mask[..., -overlap_size//2:, :] *= 0
                        if w_idx < w_idx_list[-1]:
                            out_patch[..., :, -overlap_size//2:] *= 0
                            out_patch_mask[..., :, -overlap_size//2:] *= 0
                        if h_idx > h_idx_list[0]:
                            out_patch[..., :overlap_size//2, :] *= 0
                            out_patch_mask[..., :overlap_size//2, :] *= 0
                        if w_idx > w_idx_list[0]:
                            out_patch[..., :, :overlap_size//2] *= 0
                            out_patch_mask[..., :, :overlap_size//2] *= 0

                    E[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing)*sf].add_(out_patch)
                    W[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing)*sf].add_(out_patch_mask)
            output = E.div_(W)

        else:
            _, _, _, h_old, w_old = lq.size()
            h_pad = (h_old// window_size[1]+1)*window_size[1] - h_old
            w_pad = (w_old// window_size[2]+1)*window_size[2] - w_old

            lq = torch.cat([lq, torch.flip(lq[:, :, :, -h_pad:, :], [3])], 3)
            lq = torch.cat([lq, torch.flip(lq[:, :, :, :, -w_pad:], [4])], 4)

            if hasattr(self, 'netE'):
                output = self.netE(lq).detach().cpu()
            else:
                output = self.netG(lq).detach().cpu()

            output = output[:, :, :, :h_old*sf, :w_old*sf]

        return output

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

