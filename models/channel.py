
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math 
import random
import sys
sys.path.append('./')

from models.utils import clipping, add_cp, rm_cp, batch_conv1d, PAPR, normalize



# Realization of multipath channel as a nn module
class Channel(nn.Module):
    def __init__(self, opt, device):
        super(Channel, self).__init__()
        self.opt = opt

        # Generate unit power profile
        power = torch.exp(-torch.arange(opt.L).float()/opt.decay).view(1,1,opt.L)     # 1x1xL
        self.power = power/torch.sum(power)   # Normalize the path power to sum to 1
        self.device = device

    def sample(self, N, P, M, L):
        
        # Sample the channel coefficients
        cof = torch.sqrt(self.power/2) * (torch.randn(N, P, L) + 1j*torch.randn(N, P, L))
        # print("shape",cof.shape)
        #print(cof)
        cof_zp = torch.cat((cof, torch.zeros((N,P,M-L))), -1)
        H_t = torch.fft.fft(cof_zp, dim=-1)
        # print("ht_shape",H_t.shape)
        return cof, H_t

    def sampleJakes2(self, N, P, M, L,v):
        carrier_freq = 3e9
        velocity = v
         # Calculate the maximum Doppler shift
        max_doppler_shift = velocity / 3e8 * carrier_freq
         # Generate angles for the scatterers
        angles = torch.linspace(0, 2 * np.pi, L)

         # 计算多普勒频移
        phases = 2 * np.pi * torch.cos(angles) * max_doppler_shift 
        print(phases)
        phases = 1 / 1.41421356 * (torch.cos(phases) + 1j * torch.sin(phases))
        phases = torch.fft.ifft(phases)

        phases = phases.repeat(N,P,1)
        # print("phase",phases.shape)
        
        # Sample the channel coefficients
        cof = torch.sqrt(self.power/2) * (torch.randn(N, P, L) + 1j*torch.randn(N, P, L))
        
        #print("shape",cof.shape)
        phases_real = phases.real.float().view(N*P, -1).to(self.device)       
        phases_imag = phases.imag.float().view(N*P, -1).to(self.device)       

        
        ind = torch.linspace(self.opt.L-1, 0, self.opt.L).long()
        cof_real = cof.real[...,ind].view(N*P, -1).float().to(self.device)  # (NxP)xL
        cof_imag = cof.imag[...,ind].view(N*P, -1).float().to(self.device)  # (NxP)xL
        
        cof2_real = batch_conv1d(phases_real, cof_real) - batch_conv1d(phases_imag, cof_imag)   
        cof2_imag = batch_conv1d(phases_real, cof_imag) + batch_conv1d(phases_imag, cof_real)  
        cof2 = torch.cat((cof2_real.view(N*P,-1), cof2_imag.view(N*P,-1)), -1).view(N,P,L,2)
        cof2 = torch.view_as_complex(cof2)
        # diff = cof2 - cof.to(self.device)
        # print(f'channel diff :{torch.mean(diff.abs()**2).data}')
        # print(cof2)
        # print("shape",cof2.shape)
        cof_zp = torch.cat((cof2, torch.zeros((N,P,M-L)).to(self.device)), -1)
        H_t = torch.fft.fft(cof_zp, dim=-1)
        # print("ht_shape",H_t.shape)
        return cof2, H_t          

    def sampleJakes(self, N, P, M, L,K):
        carrier_freq = 3.6e9
        velocity = 50.0
         # Calculate the maximum Doppler shift
        max_doppler_shift = velocity / 3e8 * carrier_freq

        # Generate angles for the scatterers
        angles = torch.linspace(0, 2 * np.pi, L)

        # 子载波间隔（单位：Hz）
        subcarrier_spacing = 30e3  # 30 kHz

        # 为每个子载波计算中心频率q
        # 假设子载波索引从0开始，到M-1
        subcarrier_freqs = carrier_freq - torch.arange(0, M+K).float() * subcarrier_spacing
        subcarrier_freqs = subcarrier_freqs.view(1,1,M+K) # Shape: [1, 1, M+K]

        # 计算每个子载波的多普勒频移
        phase = 2 * np.pi * subcarrier_freqs * velocity / 3e8

        # 计算每个路径的多普勒频移
        phases = phase * torch.cos(angles).view(L, 1) # Shape: [L, M+K]
        print("phases",phases.shape)

         # 生成基本cof，其形状应为[1, L, M+K]
        base_cof = (torch.cos(phases) + 1j * torch.sin(phases))  # Shape: [L, M+K]
        base_cof = torch.nn.functional.normalize(base_cof,p=2,dim = 1)
        #cof = base_cof.repeat(N, 1, 1).unsqueeze(1).repeat(1, P, 1, 1).view(N, P, L, M+K)  # Shape: [N, P, L, M+K]
        cof = base_cof
        H_t = torch.fft.fft(cof, dim=-1)

        
        # 根据N, P扩展cof
    #     cof = base_cof.repeat(N, 1, 1).unsqueeze(1).repeat(1, P, 1, 1).view(N, P, L, M)  # Shape: [N, P, L, M]
    #     #print("cofshape",cof.shape)
    #    # print(cof)
    #     #cof = cof.view(N, P, L)
    #     # Zero padding and FFT
    #     #cof_zp = torch.cat((cof, torch.zeros((N, P, M - L))), -1)  # Zero padding
    #     H_t = torch.fft.fft(cof, dim=-1)
        #print("ht_shape",H_t.shape)
        return cof, H_t
    def multipathChannel(self, cpSize, delta_f, inSig, velocity):
        # Create N x M channel matrix
        N, M = inSig.shape                                      # Size of inSig is used to create channel model
        n = torch.zeros(N)                                      # delay_Doppler rows (doppler)
        m = torch.zeros(M)                                      # delay_Doppler cols (delay)
        H = torch.outer(n, m)                                   # Create matrix

        # Generate Channel Parameter
        
    
        step = maxDelayspread / L                               # calculate difference between delays
        pathDelays = torch.arange(0, maxDelayspread + step, step)  # Discrete even delays of L-path channel

        avgPathGains_dB = -(torch.randint(3, 8, (L,))).float()  # Generate random path gains in dB
        avgPathGains = 10 ** (0.1 * avgPathGains_dB)            # Convert to linear

        # Calculate Max Doppler Shift
        v = velocity * 1e3 / 3600                                # Mobile speed (m/s)
        fc = 3.5e9                                               # Carrier frequency
        fd = round(v * fc / 299792458)                            # Maximum Doppler shift to nearest Hz

        # Generate doppler spreads w/ Jake's model
        Vi = torch.tensor([fd * math.cos((2 * math.pi * l) / (L - 1)) for l in range(L)])

        # Initialize channel variables
        T = 1 / delta_f                                           # unextended OFDM symbol period
        Ts = (1 + cpSize) / delta_f                               # OFDM symbol period
        Ti = pathDelays                                           # Path delays
        hi = avgPathGains                                         # Path gains

        # Create matrix representation of channel
        for m_idx in range(M):                         # Loop along the rows
            for n_idx in range(N):                     # Loop down the cols
                for x_idx in range(L):                 # Loop to sum terms in the channel memory
                    # Define terms of model
                    expTerm = (-2 * 1j * math.pi) * ((m_idx + M / 2) * delta_f * Ti[x_idx] - Vi[x_idx] * n_idx * Ts)
                    hiPrime = hi[x_idx] * (1 + 1j * math.pi * Vi[x_idx] * T)
                    # Produce channel impulse response model
                    H[n_idx, m_idx] = H[n_idx, m_idx] + torch.exp(expTerm) * hiPrime

        return H.t().reshape(-1)

    def forward(self, input, cof=None,Ns=0,v=100):
        # Input size:   NxPx(Sx(M+K))
        # Output size:  NxPx(Sx(M+K))
        # Also return the true channel
        # Generate Channel Matrix

        N, P, SMK = input.shape
        S = Ns+self.opt.N_pilot
        # If the channel is not given, random sample one from the channel model
        if cof is None:
            cof, H_t = self.sample(N, P, self.opt.M, self.opt.L)
            # cof, H_t = self.sampleJakes2(N, P, self.opt.M, self.opt.L,v)
        else:
            cof_zp = torch.cat((cof, torch.zeros((N,P,self.opt.M-self.opt.L,2))), 2)  
            cof_zp = torch.view_as_complex(cof_zp) 
            H_t = torch.fft.fft(cof_zp, dim=-1)
        
        

        # input = input.repeat(1,1,1,1,self.opt.L) # (NxP)xSx(M+K)xL

        # signal_real = input.real.float().view(N*P*S, self.opt.M+self.opt.K,self.opt.L)       # (NxPxS)x(M+K)xL
        # signal_imag = input.imag.float().view(N*P*S, self.opt.M+self.opt.K,self.opt.L)       # (NxPxS)x(M+K)xL

        # cof_real = cof.real.to(self.device)
        # cof_imag = cof.imag.to(self.device)
        # output_real = torch.matmul(signal_real,cof_real) - torch.matmul(signal_imag,cof_imag) # (NxPxS)x(M+K)xL
        # output_imag = torch.matmul(signal_real,cof_imag) + torch.matmul(signal_imag, cof_real)
        
        # output_real = torch.sum(output_real,dim = 2) # (NxPxS)x(M+K)
        # output_imag = torch.sum(output_imag,dim = 2) 
        # output = torch.cat((output_real.view(N*P,-1,1), output_imag.view(N*P,-1,1)), -1).view(N,P,SMK,2)

       
        
        #input = input.view(N, self.opt.P, S, self.opt.M+self.opt.K) # NxPxSx(M+K)

        
        # signal_real = input.real.float().view(N*P, -1)       
        # signal_imag = input.imag.float().view(N*P, -1)       

        
        # ind = torch.linspace(self.opt.L-1, 0, self.opt.L).long()
        # cof_real = cof.real[...,ind].view(N*P, -1).float().to(self.device)  # (NxP)xL
        # cof_imag = cof.imag[...,ind].view(N*P, -1).float().to(self.device)  # (NxP)xL
        
        # output_real = batch_conv1d(signal_real, cof_real) - batch_conv1d(signal_imag, cof_imag)   # (NxP)x(L+SMK-1)
        # output_imag = batch_conv1d(signal_real, cof_imag) + batch_conv1d(signal_imag, cof_real)   # (NxP)x(L+SMK-1)

        # # output1 = torch.cat((output_real.view(N*P,-1,1), output_imag.view(N*P,-1,1)), -1).view(N,P,SMK,2)   # NxPxSMKx2

        # output = torch.cat((output_real.view(N*P,-1,1), output_imag.view(N*P,-1,1)), -1).view(N,P,S,-1,2) # NxPxSxMKx2
        # output = torch.view_as_complex(output) 
        
        ## test
        output = input.view(N,P,S,-1)

        t = torch.linspace(0,(S-1) * (0.5e-3 / 14),S).to(self.device)
        carrier_freq = 3e9

        velocity = self.opt.V
         # Calculate the maximum Doppler shift
        max_doppler_shift = velocity / 3e8 * carrier_freq

        angles = torch.linspace(0, 2 * np.pi, self.opt.L).to(self.device)

         # 计算多普勒频移
        # phases = 2 * np.pi * torch.cos(angles) * max_doppler_shift # L

        # phases = torch.outer(phases,t)
        # shift = torch.exp(1j * phases).to(self.device)
        
        # input = input.view(N,P,S,-1)
        # out = torch.zeros_like(input)
        # for n in range(N):
        #     for p in range(P):
        #         for l in range(self.opt.L):
        #             for s in range(S):

        #                 out[n,p,s,...] = out[n,p,s,...] + input[n,p,s,...] * cof[n,p,l] * shift[l,s]
        # output = out.view(N,P,SMK)

        # 将输入和系数张量的形状调整为匹配乘法的形状
        input_reshaped = input.view(N, P, S, -1).unsqueeze(2).to(self.device)  # (N, P, 1,S, MK)
        cof_reshaped = cof.view(N, P, self.opt.L, 1,1).to(self.device)  # (N, P, L, 1,1)

        # 将相位张量调整为匹配乘法的形状
        phases = 2 * np.pi * torch.cos(angles) * max_doppler_shift  # (L,)
        phases = torch.outer(phases, t).to(self.device)  # (L, S)
        shift = torch.exp(1j * phases).unsqueeze(2).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, L, S,1)

        # 执行向量化操作
        
        out = torch.sum(input_reshaped * cof_reshaped * shift, dim=2)  # (N, P, MK)
        # print(cof_reshaped.shape)
        # 将输出调整为所需的形状
        output = out.view(N, P, SMK)
        
        
        """
        phases_real =  torch.cos(phases).unsqueeze(1).unsqueeze(0).unsqueeze(0)
        phases_imag =  torch.sin(phases).unsqueeze(1).unsqueeze(0).unsqueeze(0)

        print("phase shape",phases_real.shape)
        output_real = output.real
        output_imag = output.imag.to(self.device)
        # print(phases_real.shape)
        # print(output_real.shape)
        output2_real = output_real * phases_real - output_imag * phases_imag
        output2_imag = output_real * phases_imag + output_imag * phases_real

        output2_real = output2_real.view(N*P,-1)
        output2_imag = output2_imag.view(N*P,-1)
        # output2 = torch.cat((output2_real.view(N*P,-1,1), output2_imag.view(N*P,-1,1)), -1).view(N,P,SMK,2)

        # output = output2.view(N,P,SMK,2)
        # # # print("output",output.shape)
        # output = torch.view_as_complex(output) 

         
        ind = torch.linspace(self.opt.L-1, 0, self.opt.L).long()
        cof_real = cof.real[...,ind].view(N*P, -1).float().to(self.device)  # (NxP)xL
        cof_imag = cof.imag[...,ind].view(N*P, -1).float().to(self.device)  # (NxP)xL
        
        output_real = batch_conv1d(output2_real, cof_real) - batch_conv1d(output2_imag, cof_imag)   # (NxP)x(L+SMK-1)
        output_imag = batch_conv1d(output2_real, cof_imag) + batch_conv1d(output2_imag, cof_real)   # (NxP)x(L+SMK-1)

        # output1 = torch.cat((output_real.view(N*P,-1,1), output_imag.view(N*P,-1,1)), -1).view(N,P,SMK,2)   # NxPxSMKx2

        output = torch.cat((output_real.view(N*P,-1,1), output_imag.view(N*P,-1,1)), -1).view(N,P,SMK,2) # NxPxSxMKx2
        output = torch.view_as_complex(output) 
        """

        return output, H_t


# Realization of OFDM system as a nn module
class OFDM(nn.Module):
    def __init__(self, opt, device, pilot_path):
        super(OFDM, self).__init__()
        self.opt = opt

        # Setup the channel layer
        self.channel = Channel(opt, device)
        
        # Generate the pilot signal
        if not os.path.exists(pilot_path):
            bits = torch.randint(2, (opt.M,2))
            torch.save(bits,pilot_path)
            pilot = (2*bits-1).float()
        else:
            bits = torch.load(pilot_path)
            pilot = (2*bits-1).float()
    
        self.pilot = pilot.to(device)
        self.pilot = torch.view_as_complex(self.pilot)
        self.pilot = normalize(self.pilot, 1)
        # print(self.pilot)
        # print("pilot shape",self.pilot.shape)
        #ISFFT
        if(self.opt.mod=='OTFS'):
            self.pilot_is = self.pilot.repeat(opt.P,opt.N_pilot,1)
        elif(self.opt.mod=='OFDM'):
            self.pilot_cp = add_cp(torch.fft.ifft(self.pilot), self.opt.K).repeat(opt.P, opt.N_pilot,1)  
        # self.pilot_is = np.sqrt(self.pilot_is.shape[-2] / self.pilot_is.shape[-1]) * torch.fft.ifft(self.pilot_is,dim = -2)
        # print(self.pilot_is)
        # self.pilot_cp = add_cp(self.pilot_is, self.opt.K).repeat(opt.P, 1,1)
        
        
        # self.pilot_cp = add_cp(torch.fft.ifft(self.pilot), self.opt.K).repeat(opt.P, opt.N_pilot,1)      
        # print(self.pilot_cp)  
        # print("pilot_cp",self.pilot_cp.shape)

    def forward(self, x, SNR, cof=None, batch_size=None,v=100):
        # Input size: NxPxSxM   The information to be transmitted
        # cof denotes given channel coefficients
                
        # If x is None, we only send the pilots through the channel
        is_pilot = (x == None)
        # print("x",x.shape)
        if not is_pilot:
            
            # Change to new complex representations
            N = x.shape[0]
            
            if(self.opt.mod=='OTFS'):

                # ISFFT
                pilot = self.pilot_is.repeat(N,1,1,1)
                x = torch.cat((pilot, x), 2)
                # print("xshape",x.shape)
                x = np.sqrt(x.shape[-2] / x.shape[-1]) * torch.fft.ifft(x,dim = -2)
                x = add_cp(x, self.opt.K)
            elif(self.opt.mod=='OFDM'):
                # IFFT:                    NxPxSxM  => NxPxSxM
                x = torch.fft.ifft(x, dim=-1)
                 # Add Cyclic Prefix:       NxPxSxM  => NxPxSx(M+K)
                x = add_cp(x, self.opt.K)

                # Add pilot:               NxPxSx(M+K)  => NxPx(S+2)x(M+K)
                pilot = self.pilot_cp.repeat(N,1,1,1)
                x = torch.cat((pilot, x), 2)
              
                      

            Ns = self.opt.S
        else:
            N = batch_size
            x = self.pilot_cp.repeat(N,1,1,1)
            Ns = 0    
        # print("x",x.shape)
        # Reshape:                 NxPx(S+2)x(M+K)  => NxPx(S+2)(M+K)
        x = x.view(N, self.opt.P, (Ns+self.opt.N_pilot)*(self.opt.M+self.opt.K))
        # print("x.v",x.shape)
        # PAPR before clipping
        papr = PAPR(x)
        
        # Clipping (Optional):     NxPx(S+1)(M+K)  => NxPx(S+1)(M+K)
        if self.opt.is_clip:
            x = self.clip(x)
        
        # PAPR after clipping
        papr_cp = PAPR(x)
        
        
        # Pass through the Channel:        NxPx(S+1)(M+K)  =>  NxPx((S+1)(M+K))
        y, H_t = self.channel(x, cof, Ns, v)
        # print("channel diff",torch.sum(x.abs()-y.abs()))
        
        # print("y",y.shape)
        # Calculate the power of received signal        
        pwr = torch.mean(y.abs()**2, -1, True)
        noise_pwr = pwr*10**(-SNR/10)

        # Generate random noise
        noise = torch.sqrt(noise_pwr/2) * (torch.randn_like(y) + 1j*torch.randn_like(y))
        y_noisy = y + noise
        
        # NxPx((S+S')(M+K))  =>  NxPx(S+S')x(M+K)
        output = y_noisy.view(N, self.opt.P, Ns+self.opt.N_pilot, self.opt.M+self.opt.K)

        if(self.opt.mod=='OTFS'):
            #SFFT
            output = rm_cp(output, self.opt.K)
            output = np.sqrt(output.shape[-1] / output.shape[-2]) * torch.fft.fft(output,dim = -2)
            # print(output.shape)
            y_pilot = output[:,:,:self.opt.N_pilot,:]         # NxPxS'x(M+K)
            y_sig = output[:,:,self.opt.N_pilot:,:]           # NxPxSx(M+K)
        elif(self.opt.mod=='OFDM'):
            y_pilot = output[:,:,:self.opt.N_pilot,:]         # NxPxS'x(M+K)
            y_sig = output[:,:,self.opt.N_pilot:,:]           # NxPxSx(M+K)
        if not is_pilot:

            if(self.opt.mod=='OTFS'):
                info_pilot = y_pilot
                info_sig = y_sig
                # print("pilot diff",torch.sum(info_pilot.abs()-pilot.abs()))
            elif(self.opt.mod=='OFDM'):
                    
                # Remove Cyclic Prefix:   
                info_pilot = rm_cp(y_pilot, self.opt.K)    # NxPxS'xM
                info_sig = rm_cp(y_sig, self.opt.K)        # NxPxSxM


                
                # FFT:                     
                info_pilot = torch.fft.fft(info_pilot, dim=-1)
                info_sig = torch.fft.fft(info_sig, dim=-1)
                # print("INFOSIG",info_sig.shape)

            # SFFT
            # # print("infopilot",info_pilot.shape)

            # info_pilot = np.sqrt(info_pilot.shape[-1] / info_pilot.shape[-2]) * torch.fft.fft(info_pilot,dim = -2)
            # print(info_pilot)
            # info_sig = np.sqrt(info_sig.shape[-1] / info_sig.shape[-2]) * torch.fft.fft(info_sig,dim = -2)

          
            return info_pilot, info_sig, H_t, noise_pwr, papr, papr_cp
        else:
            info_pilot = rm_cp(y_pilot, self.opt.K)    # NxPxS'xM
            info_pilot = torch.fft.fft(info_pilot, dim=-1)

            return info_pilot, H_t, noise_pwr


# Realization of direct transmission over the multipath channel
class PLAIN(nn.Module):
    
    def __init__(self, opt, device):
        super(PLAIN, self).__init__()
        self.opt = opt

        # Setup the channel layer
        self.channel = Channel(opt, device)

    def forward(self, x, SNR):

        # Input size: NxPxM   
        N, P, M = x.shape
        y = self.channel(x, None)
        
        # Calculate the power of received signal
        pwr = torch.mean(y.abs()**2, -1, True)      
        noise_pwr = pwr*10**(-SNR/10)
        
        # Generate random noise
        noise = torch.sqrt(noise_pwr/2) * (torch.randn_like(y) + 1j*torch.randn_like(y))
        y_noisy = y + noise                                    # NxPx(M+L-1)
        rx = y_noisy[:, :, :M, :]
        return rx 


if __name__ == "__main__":

    import argparse
    opt = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    opt.P = 1
    opt.S = 6
    opt.M = 64
    opt.K = 16
    opt.L = 8
    opt.decay = 4
    opt.N_pilot = 1
    opt.SNR = 10
    opt.is_clip = False

    ofdm = OFDM(opt, 0, './models/Pilot_bit.pt')

    input_f = torch.randn(32, opt.P, opt.S, opt.M) + 1j*torch.randn(32, opt.P, opt.S, opt.M)
    input_f = normalize(input_f, 1)
    input_f = input_f.cuda()

    info_pilot, info_sig, H_t, noise_pwr, papr, papr_cp = ofdm(input_f, opt.SNR, v=10)
    H_t = H_t.cuda()
    err = input_f*H_t.unsqueeze(0) 
    err = err - info_sig
    print(f'OFDM path error :{torch.mean(err.abs()**2).data}')


    info_pilot, info_sig, H_t, noise_pwr, papr, papr_cp = ofdm(input_f, opt.SNR, v=1000)
    H_t = H_t.cuda()
    err = input_f*H_t.unsqueeze(0) 
    err = err - info_sig
    print(f'OFDM path error :{torch.mean(err.abs()**2).data}')

    from utils import ZF_equalization, MMSE_equalization, LS_channel_est, LMMSE_channel_est

    H_est_LS = LS_channel_est(ofdm.pilot, info_pilot)
    err_LS = torch.mean((H_est_LS.squeeze()-H_t.squeeze()).abs()**2)
    print(f'LS channel estimation error :{err_LS.data}')

    H_est_LMMSE = LMMSE_channel_est(ofdm.pilot, info_pilot, opt.M*noise_pwr)
    err_LMMSE = torch.mean((H_est_LMMSE.squeeze()-H_t.squeeze()).abs()**2)
    print(f'LMMSE channel estimation error :{err_LMMSE.data}')
    
    rx_ZF = ZF_equalization(H_t.unsqueeze(0), info_sig)
    err_ZF = torch.mean((rx_ZF.squeeze()-input_f.squeeze()).abs()**2)
    print(f'ZF error :{err_ZF.data}')

    rx_MMSE = MMSE_equalization(H_t.unsqueeze(0), info_sig, opt.M*noise_pwr)
    err_MMSE = torch.mean((rx_MMSE.squeeze()-input_f.squeeze()).abs()**2)
    print(f'MMSE error :{err_MMSE.data}')







