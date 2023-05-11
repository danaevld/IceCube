import os,time
import matplotlib.pyplot as plt 
import numpy as np
from charon import propa
import charon.physicsconstants as PC
pc = PC.PhysicsConstants()
from charon import profile
from scipy.interpolate import splrep, splev
from ipynb.fs.full.func_plot_histo import plot_projections
from scipy.interpolate import interp1d
import random as rd

font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'normal',
        'size': 11,
        }

color = ['limegreen', 'darkgreen', 'mediumblue', 'indigo','mediumvioletred','green']

def Spectra_generator(mass,channel,process,nodes=300,bins=300,logscale=True,interactions=True):
    theta_12=33.82
    theta_13=8.6
    theta_23=48.6
    delta_m_12=7.39e-5
    delta_m_13=2.528e-3
    delta = 221.
    Emin = 1
    Emax = mass
    nu_flavor = ['nu_e','nu_mu','nu_tau','nu_e_bar','nu_mu_bar','nu_tau_bar']
    channels = [r'$\nu_e$', r'$\nu_\mu$',r'$\nu_\tau$',r'$\bar{\nu_{e}}$',r'$\bar{\nu_{\mu}}$', r'$\bar{\nu_\tau}$']
    Flux = propa.NuFlux(channel,mass,nodes,Emin=Emin,Emax=Emax,bins=bins,
                     process=process,logscale=logscale,interactions=interactions,
                     theta_12=theta_12,theta_13=theta_13,theta_23=theta_23,
                     delta_m_12=delta_m_12,delta_m_13=delta_m_13,delta=delta,
                     pathSunModel='../charon/models/struct_b16_agss09.dat')
    mass_range = Flux.iniE()
    Flux_ini_Halo = Flux.iniFlux('Halo')
    '''
    plt.figure(figsize = (6,4))
    for i, nu_flavor in enumerate(nu_flavor):
        plt.plot(mass_range,Flux_ini_Halo[nu_flavor]/mass,label=channels[i], color=color[i])
    plt.yscale('log')
    #plt.xlim(1,Emax)
    plt.semilogx()
    plt.grid()
    plt.xlabel(r"$E [GeV]$", fontdict = font)
    plt.ylabel(r"$dN_\nu/dE_\nu\;[GeV^{-1}]$", fontdict = font)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncols=6, fancybox=True, shadow=False)'''
    return Flux
    
def Spectra_interpolation(mass, channel, process, source, extrapolation, galactic_profile):
    resp_matrix_data = np.load('Response matrix/Resp_MC1122_logE.pkl',allow_pickle=True, encoding="latin1")
    E_true_center = resp_matrix_data['Bin']['true_energy_center']
    True_spectra_list = []
    nu_flavor = ['nu_e','nu_mu','nu_tau','nu_e_bar','nu_mu_bar','nu_tau_bar']
    if source == True: # no oscillations
        if extrapolation == False: # no extrapolation
            Flux = Spectra_generator(mass=mass,channel=channel,process=process, nodes=300)
            mass_range = Flux.iniE()
            Flux_ini_Halo = Flux.iniFlux('Halo')
            plt.figure(figsize = (6,4))
            for i, nu_flavor in enumerate(nu_flavor):
                if process == 'decay':
                    Spectra_interp = splrep(mass_range, 2*Flux_ini_Halo[nu_flavor]/mass)
                else:
                    Spectra_interp = splrep(mass_range, Flux_ini_Halo[nu_flavor]/mass)
                True_spectra = splev(E_true_center, Spectra_interp, der = 0)
                plt.plot(E_true_center, True_spectra, label = nu_flavor, color=color[i])
                True_spectra_list.append(True_spectra)
            plt.title('Spectra interpolate at values of E true center | channel = {0}'.format(channel))
            plt.yscale('log')
            plt.semilogx()
            plt.grid()
            plt.xlabel("$E [GeV]$", fontdict = font)
            plt.ylabel(r"$dN_\nu/dE_\nu\;[GeV^{-1}]$", fontdict = font)
            plt.legend()
            return True_spectra_list
        else: # no oscillation + extrapolation
            Flux = Spectra_generator(mass=mass,channel=channel,process=process, nodes=300)
            mass_range = Flux.iniE()
            Flux_ini_Halo = Spectra_extrapolate(mass=mass, channel=channel, process=process, source=source)
            plt.figure(figsize = (6,4))
            for i, nu_flavor in enumerate(nu_flavor):
                if process == 'decay':
                    Spectra_interp = splrep(mass_range, Flux_ini_Halo[i])
                else:
                    Spectra_interp = splrep(mass_range, Flux_ini_Halo[i])
                True_spectra = splev(E_true_center, Spectra_interp, der = 0)
                plt.plot(E_true_center, True_spectra, label = nu_flavor, color=color[i])
                True_spectra_list.append(True_spectra)
            plt.title('Spectra interpolate at values of E true center | channel = {0}'.format(channel))
            plt.yscale('log')
            plt.semilogx()
            plt.grid()
            plt.xlabel("$E [GeV]$", fontdict = font)
            plt.ylabel(r"$dN_\nu/dE_\nu\;[GeV^{-1}]$", fontdict = font)
            plt.legend()
            return True_spectra_list
    else: # oscillations
        if extrapolation == False:
            # osc but no extrapolation
            Flux = Spectra_generator(mass=mass,channel=channel,process=process, nodes=300)
            mass_range = Flux.iniE()
            Flux_osc = Flux.Halo('detector',zenith=np.deg2rad(-29.00781+90))
            plt.figure(figsize = (6,4))
            for i, nu_flavor in enumerate(nu_flavor):
                if process == 'decay':
                    Spectra_interp = splrep(mass_range, (2*Flux_osc[nu_flavor])/mass)
                else:
                    Spectra_interp = splrep(mass_range, Flux_osc[nu_flavor]/mass)
                True_spectra = splev(E_true_center, Spectra_interp, der = 0)
                plt.plot(E_true_center, True_spectra, label = nu_flavor, color=color[i])
                True_spectra_list.append(True_spectra)
            plt.title('Spectra interpolate at values of E true center | channel = {0}'.format(channel))
            plt.yscale('log')
            plt.semilogx()
            plt.grid()
            plt.xlabel("$E [GeV]$", fontdict = font)
            plt.ylabel(r"$dN_\nu/dE_\nu\;[GeV^{-1}]$", fontdict = font)
            plt.legend()
            return True_spectra_list
        else: # osc and extrapolation
            Flux = Spectra_generator(mass=mass,channel=channel,process=process, nodes=300)
            mass_range = Flux.iniE()
            Flux_osc = Spectra_extrapolate(mass=mass, channel=channel, process=process, source=source)
            plt.figure(figsize = (6,4))
            for i, nu_flavor in enumerate(nu_flavor):
                if process == 'decay':
                    Spectra_interp = splrep(mass_range, Flux_osc[i])
                else:
                    Spectra_interp = splrep(mass_range, Flux_osc[i])
                True_spectra = splev(E_true_center, Spectra_interp, der = 0)
                plt.plot(E_true_center, True_spectra, label = nu_flavor, color=color[i])
                True_spectra_list.append(True_spectra)
            plt.title('Spectra interpolate at values of E true center | channel = {0}'.format(channel))
            plt.yscale('log')
            plt.semilogx()
            plt.grid()
            plt.xlabel("$E [GeV]$", fontdict = font)
            plt.ylabel(r"$dN_\nu/dE_\nu\;[GeV^{-1}]$", fontdict = font)
            plt.legend()
            return True_spectra_list
        
def Flux_generator(mass, channel, process, galactic_profile, extrapolation, Ntheta, Clumpy = True, Plot = True):
    if extrapolation == False:
        Flux = Spectra_generator(mass=mass, channel=channel, process=process, nodes=300)
        Flux_osc = Flux.Halo('detector',zenith=np.deg2rad(-29.00781+90))
        mass_range = Flux.iniE()
    else:
        Spectra = Spectra_generator(mass=mass, channel=channel, process=process, nodes=300)
        mass_range = Spectra.iniE()
        Flux_osc = Spectra_extrapolate(mass=mass, channel=channel, process=process, source=False)
    nu_flavor = ['nu_e','nu_mu','nu_tau','nu_e_bar','nu_mu_bar','nu_tau_bar'] 
    if Clumpy == True and process == 'decay': #Clumpy + decay
        if galactic_profile == profile.NFW: #Clumpy + decay + NFW
            theta, Jtheta = ClumpyReader('Clumpy_Jfactor/Dfactor_dDdOmega_GeV_cm2_sr_NFW_NestiSalucci.output', header_index = 10)
        else:
            theta, Jtheta = ClumpyReader('Clumpy_Jfactor/Dfactor_dDdOmega_GeV_cm2_sr_Burkert_NestiSalucci.output', header_index = 10) #Clumpy + decay + Burkert
    elif Clumpy == True and process == 'ann':
        if galactic_profile == profile.Burkert: #Clumpy + ann + Burkert
            theta, Jtheta = ClumpyReader('Clumpy_Jfactor/Jfactor_dJdOmega_GeV2_cm5_sr_Burkert_NestiSalucci.output', header_index=3)
        else:
            theta, Jtheta = ClumpyReader('Clumpy_Jfactor/Jfactor_dJdOmega_GeV2_cm5_sr_NFW_NestiSalucci.output', header_index=3) #Clumpy + ann + NFW
    else: # Charon Jprofile
        R       = 100.  
        d       = 8    
        theta = np.linspace(0.,np.pi,Ntheta)
        J      = profile.J(galactic_profile,R,d,process)
        Jtheta = [J.Jtheta(j) for j in theta]
    Transpose_flux, Flux_list = [],[]
    if extrapolation == False: # No extrapolation
        for k, nu_flav in enumerate(nu_flavor):
            if process == 'decay':
                Transpose_flux.append((2*Flux_osc[nu_flav][:,None])/mass) 
            else: 
                Transpose_flux.append((Flux_osc[nu_flav][:,None])/mass)    
            Flux_list.append(Transpose_flux[k]*Jtheta)
    else: # extrapolation
        for k, nu_flav in enumerate(nu_flavor):
            Transpose_flux.append(Flux_osc[k][:,None]) 
            Flux_list.append(Transpose_flux[k]*Jtheta)
    if Plot == True:
        for j, nu_flav in enumerate(nu_flavor):
            plt.figure(figsize = (7,10))
            plt.title(r'Neutrinos flux for $m_X$ = {0}PeV | channel = {1}'.format(int(mass/1e6),channel))
            plt.subplot(211)
            plt.title(r'Neutrinos flux function of opening angle | channel = {0} | $m_X = {1}$PeV'.format(channel,int(mass/1e6)), fontdict = font)
            plt.plot(theta, Flux_list[j][1,:], label = nu_flavor[j], color = color[j])
            plt.yscale('log')
            plt.grid()
            plt.xlabel(r"$\theta \; (deg)$", fontdict=font)
            plt.xlim(0,180)
            plt.ylabel(r"$d\phi_\nu/dE\;[GeV^{-1}]$", fontdict=font)
            plt.legend()
        
            plt.subplot(212)
            plt.title(r'Neutrinos flux function of Energy | channel = {0} | $m_X = {1}$PeV'.format(channel,int(mass/1e6)), fontdict = font)
            plt.plot(mass_range, Flux_list[j][:,1], label = nu_flavor[j], color = color[j])
            plt.yscale('log')
            plt.semilogx()
            plt.grid()
            plt.xlabel("$E [GeV]$", fontdict = font)
            plt.ylabel(r"$d\phi_\nu/dE\;[GeV^{-1}]$", fontdict=font)
            plt.legend()
            
            theta_edges = np.histogram_bin_edges(theta,bins=len(theta))
            mass_range_edges = np.histogram_bin_edges(mass_range,bins=len(mass_range))
            plot_projections(Flux_list[j],(theta_edges,mass_range_edges),'Theta [rad]','E [GeV]','Titre')

    return Flux_list

def J_interpolation(process, galactic_profile, Clumpy=True, Ntheta=100):
    resp_matrix_data = np.load('Response matrix/Resp_MC1122_logE.pkl',allow_pickle=True, encoding="latin1")
    theta_true_center=resp_matrix_data['Bin']['true_psi_center']
    nu_flavor = ['nu_e','nu_mu','nu_tau','nu_e_bar','nu_mu_bar','nu_tau_bar']
    if Clumpy == True and process == 'decay': #Clumpy + decay
        if galactic_profile == profile.NFW: #Clumpy + decay + NFW
            opening_angle, Jtheta = ClumpyReader('Clumpy_Jfactor/Dfactor_dDdOmega_GeV_cm2_sr_NFW_NestiSalucci.output', header_index = 10)
        else:
            opening_angle, Jtheta = ClumpyReader('Clumpy_Jfactor/Dfactor_dDdOmega_GeV_cm2_sr_Burkert_NestiSalucci.output', header_index = 10) #Clumpy + decay + Burkert
    elif Clumpy == True and process == 'ann':
        if galactic_profile == profile.Burkert: #Clumpy + ann + Burkert
            opening_angle, Jtheta = ClumpyReader('Clumpy_Jfactor/Jfactor_dJdOmega_GeV2_cm5_sr_Burkert_NestiSalucci.output', header_index=3)
        else:
            opening_angle, Jtheta = ClumpyReader('Clumpy_Jfactor/Jfactor_dJdOmega_GeV2_cm5_sr_NFW_NestiSalucci.output', header_index=3) #Clumpy + ann + NFW
    else:
        R       = 100.  
        d       = 8     
        opening_angle = np.linspace(0.,np.pi,Ntheta)
        J      = profile.J(galactic_profile,R,d,process)
        Jtheta = [J.Jtheta(j) for j in opening_angle] # (rad)
    J_interp = splrep(np.rad2deg(opening_angle), Jtheta)
    True_J = splev(theta_true_center, J_interp, der = 0) # (deg)
    return True_J
   
def Flux_interpolation(mass,channel,process,extrapolation,galactic_profile,source=False):
    resp_matrix_data = np.load('Response matrix/Resp_MC1122_logE.pkl',allow_pickle=True, encoding="latin1")
    theta_true_center=resp_matrix_data['Bin']['true_psi_center']
    nu_flavor = ['nu_e','nu_mu','nu_tau','nu_e_bar','nu_mu_bar','nu_tau_bar']
    Transpose_flux, True_flux = [],[]
    J_true = J_interpolation(process=process,galactic_profile=galactic_profile)
    Flux_osc = Spectra_interpolation(mass=mass,channel=channel,process=process, source=source, extrapolation=extrapolation,galactic_profile=galactic_profile) #normalized by the mass
    if extrapolation == False:
        for i, nu_flavor in enumerate(nu_flavor):
            if process == 'decay':
                Transpose_flux.append(Flux_osc[i][:,None])
            else:  
                Transpose_flux.append(Flux_osc[i][:,None])
            True_flux.append(Transpose_flux[i]*J_true)
        return True_flux
    else:
        Flux_osc = Spectra_extrapolate(mass=mass, channel=channel, process=process, source=source)
        for k in range(len(Flux_osc)):
            Transpose_flux.append(Flux_osc[k][:,None])
            True_flux.append(Transpose_flux[k]*J_true)
        return True_flux
    
def Signal_PDF(mass, channel, process, extrapolation, galactic_profile, normalize):
    resp_matrix_data = np.load('/home/dvaldenaire/Python/Analysis/Response matrix/Resp_MC1122_logE.pkl',allow_pickle=True, encoding="latin1")
    Resp = resp_matrix_data['Resp']
    True_energy_center = resp_matrix_data['Bin']['true_energy_center']
    True_psi_center = resp_matrix_data['Bin']['true_psi_center']
    Reco_energy_center = resp_matrix_data['Bin']['reco_energy_center']
    Reco_psi_center = resp_matrix_data['Bin']['reco_psi_center']
    True_flux = Flux_interpolation(mass=mass,channel=channel,process=process,extrapolation=extrapolation,galactic_profile=galactic_profile)
    grid = np.meshgrid(True_psi_center, True_energy_center, Reco_psi_center, Reco_energy_center, indexing='ij')
    RecoRate = np.empty((len(Reco_psi_center),len(Reco_energy_center)))
    for i, nu_flavor in enumerate(Resp.keys()):
        TotalWeight = np.sum(Resp[nu_flavor])
        dRdlogE=Resp[nu_flavor]*grid[1]
        RespPdf = dRdlogE/np.sum(dRdlogE)
        RecoRate += np.tensordot(RespPdf*TotalWeight, True_flux[i], axes=([0,1], [1,0]))
    if normalize == True:
        RecoRate = RecoRate/np.sum(RecoRate)
    return RecoRate 

def Background_cut():
    data = np.load('Burnsample/Bkg.pkl',allow_pickle=True, encoding="latin1")
    return data

def Background_PDF(oversample, hist, bins, density):
    data = np.load('Burnsample/Bkg.pkl',allow_pickle=True, encoding="latin1")
    burnsample = data['burnsample']
    DEC_reco = burnsample['reco_Dec']
    RA_reco = burnsample['reco_RA']
    psi_reco = burnsample['reco_psi']
    E_reco = burnsample['reco_TotalEnergy']
    if oversample==True: # OVERSAMPLE 
        RA_random_oversample,DEC_oversample,E_reco_oversample,psi_scrambled = [],[],[],[]
        for i in range(len(DEC_reco)):
            for j in range(10):
                RA_random_oversample.append(rd.randint(0,360))
                DEC_oversample.append(DEC_reco[i])
                E_reco_oversample.append(E_reco[i])
        E_reco_oversample = np.array(E_reco_oversample)
        for i in range(len(RA_random_oversample)):
            psi_scrambled.append(psi_f(RA_random_oversample[i], DEC_oversample[i]))
        if hist==False: 
            return psi_scrambled
        else: # OVERSAMPLE + HIST
            plt.figure(figsize=(7,5))
            psi_scrambled = np.rad2deg(psi_scrambled)
            Background = plt.hist2d(psi_scrambled,np.log10(E_reco_oversample),bins=bins,cmap="plasma", density=density)
            plt.xlabel("$\psi_{reco}$ (deg)", fontdict=font)
            plt.ylabel("$log_{10}(E_{reco})$", fontdict=font)
            plt.colorbar()
            return Background
    else: # NO OVERSAMPLE 
        for i in range(len(DEC_reco)):
            psi = psi_f(RA_reco[i], DEC_reco[i])
        if hist==False:
            return psi
        else: # NO OVERSAMPLE + HIST
            plt.figure(figsize=(7,5))
            psi = np.rad2deg(psi)
            Background = plt.hist2d(psi,np.log10(E_reco_oversample),bins=bins, density=density)
            plt.xlabel("$\psi_{reco}$ (deg)", fontdict=font)
            plt.ylabel("$log_{10}(E_{reco})$", fontdict=font)
            plt.colorbar()
            return Background
    
def psi_f(RA,decl):
    return np.arccos(np.cos(np.pi/2.-(-29.0078*np.pi/180))*np.cos(np.pi/2.-decl)\
                      +np.sin(np.pi/2.-(-29.0078*np.pi/180))*np.sin(np.pi/2.-decl)*\
                       np.cos(RA-266.4168*np.pi/180))

def Spectra_extrapolate(mass, channel, process, source, kind='linear'):
    flux_extrapol = []
    nu_flavor = ['nu_e','nu_mu','nu_tau','nu_e_bar','nu_mu_bar','nu_tau_bar']
    Flux = Spectra_generator(mass=mass, channel=channel, process=process,nodes=300)
    mass_range = Flux.iniE()
    mass_range_log = np.log(mass_range)
    if source == True:
        Flux_Halo = Flux.iniFlux('Halo')
        for i in nu_flavor:
            index = 0
            for k in range(0,len(Flux_Halo[i])-1):
                if round(Flux_Halo[i][k+1],2) == round(Flux_Halo[i][k],2):
                    index = index + 1
            index = index + 5
            if process == 'ann': #source + ann 
                flux_cut = np.log(Flux_Halo[i][index:])
                mass_range_cut = np.log(mass_range[index:])
                func = interp1d(mass_range_cut, flux_cut,
                bounds_error=False,
                kind=kind,
                fill_value='extrapolate')
                newflux_halo = [np.exp(func(i))/mass for i in mass_range_log]
                flux_extrapol.append(newflux_halo)
                if i=='nu_tau_bar':
                    plt.figure(figsize = (6,4))
                    plt.title(r'Energy spectra at source | channel : {0} | $m_X = {1}$PeV'.format(channel,int(mass/1e6)), fontdict = font)
                    plt.yscale('log')
                    plt.semilogx()
                    plt.grid()
                    plt.xlabel(r"$E$ [GeV]", fontdict = font)
                    plt.ylabel(r"$dN_\nu/dE_\nu\;[GeV^{-1}]$", fontdict = font)
                    for j in range(len(flux_extrapol)):
                        plt.plot(mass_range, flux_extrapol[j],color=color[j], label=nu_flavor[j])
                    plt.legend()
            else: # source + decay
                flux_cut = np.log(Flux_Halo[i][index:200])
                mass_range_cut = np.log(mass_range[index:200])
                func = interp1d(mass_range_cut, flux_cut,
                bounds_error=False,
                kind=kind,
                fill_value='extrapolate')
                newflux_halo = np.array([2*np.exp(func(i))/mass for i in mass_range_log[0:index]])
                Flux_halo_tail = np.array(2*Flux_Halo[i][index:]/mass)
                newflux = np.concatenate((newflux_halo, Flux_halo_tail), axis = None)
                flux_extrapol.append(newflux)
                if i=='nu_tau_bar':
                    plt.figure(figsize = (6,4))
                    plt.title(r'Energy spectra at source | channel : {0} | $m_X = {1}$PeV'.format(channel,int(mass/1e6)), fontdict = font)
                    plt.yscale('log')
                    plt.semilogx()
                    plt.grid()
                    plt.xlabel(r"$E$ [GeV]", fontdict = font)
                    plt.ylabel(r"$dN_\nu/dE_\nu\;[GeV^{-1}]$", fontdict = font)
                    for j in range(len(flux_extrapol)):
                        plt.plot(mass_range, flux_extrapol[j],color=color[j], label=nu_flavor[j])
                    plt.legend()       
        return np.array(flux_extrapol)
    else: 
        Flux_osc = Flux.Halo('detector',zenith=np.deg2rad(-29.00781+90))
        for i in nu_flavor:
            index = 0
            for k in range(0,len(Flux_osc[i])-1):
                if round(Flux_osc[i][k+1],2) == round(Flux_osc[i][k],2):
                    index = index + 1
            index = index + 5
            if process == 'ann': #earth + ann
                flux_cut = np.log(Flux_osc[i][index:])
                mass_range_cut = np.log(mass_range[index:])
                func = interp1d(mass_range_cut, flux_cut,
                bounds_error=False,
                kind=kind,
                fill_value='extrapolate')
                newflux_osc = [np.exp(func(i))/mass for i in mass_range_log]
                flux_extrapol.append(newflux_osc)
                if i=='nu_tau_bar':
                    plt.figure(figsize = (6,4))
                    plt.title(r'Energy spectra at Earth | channel : {0} | $m_X = {1}$PeV'.format(channel,int(mass/1e6)), fontdict = font)
                    plt.yscale('log')
                    plt.semilogx()
                    plt.grid()
                    plt.xlabel(r"$E$ [GeV]", fontdict = font)
                    plt.ylabel(r"$dN_\nu/dE_\nu\;[GeV^{-1}]$", fontdict = font)
                    for j in range(len(flux_extrapol)):
                        plt.plot(mass_range, flux_extrapol[j],color=color[j], label=nu_flavor[j])
                    plt.legend()
            else: #earth + decay
                flux_cut = np.log(Flux_osc[i][index:200])
                mass_range_cut = np.log(mass_range[index:200])
                func = interp1d(mass_range_cut, flux_cut,
                bounds_error=False,
                kind=kind,
                fill_value='extrapolate')
                newflux_osc = np.array([(2*np.exp(func(i)))/mass for i in mass_range_log[0:index]])
                Flux_osc_tail = np.array((2*Flux_osc[i][index:])/mass)
                newflux = np.concatenate((newflux_osc, Flux_osc_tail), axis = None)
                flux_extrapol.append(newflux)
                if i=='nu_tau_bar':
                    plt.figure(figsize = (6,4))
                    plt.title(r'Energy spectra at Earth | channel : {0} | $m_X = {1}$PeV'.format(channel,int(mass/1e6)), fontdict = font)
                    plt.yscale('log')
                    plt.semilogx()
                    plt.grid()
                    plt.xlabel(r"$E$ [GeV]", fontdict = font)
                    plt.ylabel(r"$dN_\nu/dE_\nu\;[GeV^{-1}]$", fontdict = font)
                    for j in range(len(flux_extrapol)):
                        plt.plot(mass_range, flux_extrapol[j],color=color[j], label=nu_flavor[j])
                    plt.legend()
        return np.array(flux_extrapol)

def ClumpyReader(file, header_index):
    theta = []
    JfactorTotal = []

    with open(file, 'r') as f:
        for i, line in enumerate(f):
            if i >= header_index:
                data = line.strip().split()
                theta.append(float(data[0]))
                JfactorTotal.append(float(data[-2]))
    return theta, JfactorTotal
    
    
    
    
    