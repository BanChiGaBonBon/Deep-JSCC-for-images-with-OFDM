
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
        # cof = torch.sqrt(self.power/2) * (torch.randn(N, P, L) + 1j*torch.randn(N, P, L))
        cof = torch.sqrt(self.power/2) * (np.random.rayleigh(size=(N,P,L)))
        # print("shape",cof.shape)
        #print(cof)
        cof_zp = torch.cat((cof, torch.zeros((N,P,M-L))), -1)
        H_t = torch.fft.fft(cof_zp, dim=-1)
        # print("ht_shape",H_t.shape)
        return cof, H_t


    def forward(self, input, cof=None,Ns=0,v=100):
        # Input size:   NxPx(Sx(M+K))
        # Output size:  NxPx(Sx(M+K))
        # Also return the true channel
        # Generate Channel Matrix
        
        N, P, SMK = input.shape
        if(self.opt.mod=='OFDM'):
            M = self.opt.M
            S = Ns+self.opt.N_pilot
        elif(self.opt.mod=='OTFS'):
            S = Ns
            M = self.opt.M+self.opt.N_pilot
        # If the channel is not given, random sample one from the channel model
        if cof is None:
            cof, H_t = self.sample(N, P, M, self.opt.L)
            # cof, H_t = self.sampleJakes2(N, P, self.opt.M, self.opt.L,v)
        else:
            cof_zp = torch.cat((cof, torch.zeros((N,P,M-self.opt.L,2))), 2)  
            cof_zp = torch.view_as_complex(cof_zp) 
            H_t = torch.fft.fft(cof_zp, dim=-1)
        
        # 根据路径衰落生成路径距离
        len = - 100 * np.log(cof).to(self.device) # N, P, L

        delay1 = len / 3e8

        # 符号间的延迟差
        delay2= torch.linspace(0,(S-1) * (0.5e-3 / 14),S).to(self.device)
        delay2 = delay2.repeat(N,P,1)

        delay = delay1.unsqueeze(3) + delay2.unsqueeze(2) # N,P,L,S
        

        carrier_freq = 3e9

        if self.opt.is_random_v:
            velocity = random.uniform(0,self.opt.v_range)
        else:
            velocity = self.opt.V
        # velocity = random.uniform(0,100)
         # Calculate the maximum Doppler shift
        max_doppler_shift = velocity / 3e8 * carrier_freq

        angles = torch.linspace(0, 2 * np.pi, self.opt.L).to(self.device)

       
        # 将输入和系数张量的形状调整为匹配乘法的形状
        input_reshaped = input.view(N, P, S, -1).unsqueeze(2).to(self.device)  # (N, P, 1,S, M+K)
        cof_reshaped = cof.view(N, P, self.opt.L, 1,1).to(self.device)  # (N, P, L, 1,1)

        # 将相位张量调整为匹配乘法的形状
        phases = (2 * np.pi * torch.cos(angles) * max_doppler_shift).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1,1,L,1)

        phases = phases * delay.to(self.device)  # (N,P,L, S)
        shift = torch.exp(-1j * phases).unsqueeze(-1)  # (N,P, L, S,1)

        # 执行向量化操作
        out = torch.sum(input_reshaped * cof_reshaped * shift, dim=2)  # (N, P, M+K)
        # print(cof_reshaped.shape)
        # 将输出调整为所需的形状
        output = out.view(N, P, SMK)
        
       
        return output, H_t

    def sinc_base_FD_FIR(self,x,N,delay):
        FD = delay % 1
        int_delay = int(delay // 1)
        
        ## 分数延时滤波
        if 0 == FD:
            i0 = 0
            delayed_signal = x
        else:
            i0 = int(np.ceil(N/2) - 1)
            win_index_list = np.arange(0,N) - i0
            win_fun = np.hamming(N)
            sinc_filter = np.sinc(win_index_list - FD)
            sinc_filter = sinc_filter * win_fun 
            delayed_signal = np.convolve(x, sinc_filter, mode='FULL')    # 使用卷积来应用sinc滤波器

        
        ## 整数延迟
        t_delay = int_delay - i0
        res = np.zeros(len(delayed_signal) + 2 * np.abs(t_delay))
        if 0 == t_delay:
            res = delayed_signal
        elif 0 < t_delay:
            res[t_delay:t_delay + len(delayed_signal)] = delayed_signal
        else:
            t_delay = np.abs(t_delay)
            res[:len(delayed_signal) - t_delay] = delayed_signal[t_delay:]
            
        return res[:len(x)]

# Realization of OFDM system as a nn module
class OFDM(nn.Module):
    def __init__(self, opt, device, pilot_path):
        super(OFDM, self).__init__()
        self.opt = opt

        # Setup the channel layer
        self.channel = Channel(opt, device)
        
        # Generate the pilot signal
        if(self.opt.mod=='OTFS'):
            if not os.path.exists(pilot_path):
                bits = torch.randint(2, (opt.S,2))
                torch.save(bits,pilot_path)
                pilot = (2*bits-1).float()
                
            else:
                bits = torch.load(pilot_path)
                pilot = (2*bits-1).float()
        elif(self.opt.mod=='OFDM'):
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

            self.pilot_is = self.pilot.view(1,self.opt.S,1).repeat(opt.P,1,opt.N_pilot)

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
                pilot = self.pilot_is.repeat(N,1,1,1) #  NxPxSx2
                
                x = torch.cat((pilot, x), 3)
                
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
        if(self.opt.mod=='OFDM'):
            M = self.opt.M
            S = Ns+self.opt.N_pilot
        elif(self.opt.mod=='OTFS'):
            S = Ns
            M = self.opt.M+self.opt.N_pilot
       
        
        x = x.view(N, self.opt.P, S*(M+self.opt.K))

        

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
        output = y_noisy.view(N, self.opt.P, S, M+self.opt.K)

        if(self.opt.mod=='OTFS'):
            #SFFT
            output = rm_cp(output, self.opt.K)
            output = np.sqrt(output.shape[-1] / output.shape[-2]) * torch.fft.fft(output,dim = -2)
            # print(output.shape)
            y_pilot = output[:,:,:,:self.opt.N_pilot]     
                
            y_sig = output[:,:,:,self.opt.N_pilot:] 
            # print("ypilot",y_pilot.shape)
            # print("ysig",y_sig.shape)          
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







