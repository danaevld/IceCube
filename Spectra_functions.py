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

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 12,
        }

color = ['limegreen', 'darkgreen', 'mediumblue', 'indigo','mediumvioletred','green']

def Spectra_generator(mass,channel,process,galactic_profile=profile.NFW,nodes=300,bins=300,logscale=True,interactions=True):
    theta_12=33.82
    theta_13=8.6
    theta_23=48.6
    delta_m_12=7.39e-5
    delta_m_13=2.528e-3
    delta = 221.
    Emin = 1
    Emax = mass
    nu_flavor = ['nu_e','nu_mu','nu_tau','nu_e_bar','nu_mu_bar','nu_tau_bar']
    Flux = propa.NuFlux(channel,mass,nodes,Emin=Emin,Emax=Emax,bins=bins,
                     process=process,logscale=logscale,interactions=interactions,
                     theta_12=theta_12,theta_13=theta_13,theta_23=theta_23,
                     delta_m_12=delta_m_12,delta_m_13=delta_m_13,delta=delta,
                     pathSunModel='../charon/models/struct_b16_agss09.dat')
    mass_range = Flux.iniE()
    Flux_ini_Halo = Flux.iniFlux('Halo')
    plt.figure(figsize = (6,4))
    plt.title(r'Energy spectra at source | channel : {0} | $m_X = {1}$PeV'.format(channel,int(mass/1e6)), fontdict = font)
    for i, nu_flavor in enumerate(nu_flavor):
        plt.plot(mass_range,Flux_ini_Halo[nu_flavor]/mass,label=nu_flavor, color=color[i])
    plt.yscale('log')
    #plt.xlim(1,Emax)
    plt.semilogx()
    plt.grid()
    plt.xlabel("$E [GeV]$", fontdict = font)
    plt.ylabel(r"$dN_\nu/dE_\nu\;[GeV^{-1}]$", fontdict = font)
    plt.legend()
    return Flux
    
def Spectra_interpolation(mass, channel, E_true_center, process, source, extrapolation):
    True_spectra_list = []
    nu_flavor = ['nu_e','nu_mu','nu_tau','nu_e_bar','nu_mu_bar','nu_tau_bar']
    if osc = False: # no oscillations
        if extrapolation = False: # no extrapolation
            Flux = Spectra_generator(mass=mass,channel=channel,process=process)
            mass_range = Flux.iniE()
            Flux_ini_Halo = Flux.iniFlux('Halo')
            plt.figure(figsize = (6,4))
            for i, nu_flavor in enumerate(nu_flavor):
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
            Flux = Spectra_generator(mass=mass,channel=channel,process=process)
            mass_range = Flux.iniE()
            Flux_ini_Halo = Spectra_extrapolate(mass=mass, channel=channel, process=process, source=source)
            plt.figure(figsize = (6,4))
            for i, nu_flavor in enumerate(nu_flavor):
                Spectra_interp = splrep(mass_range, Flux_ini_Halo[i]/mass)
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
        if extrapolation = False:
            # osc but no extrapolation
            Flux = Spectra_generator(mass=mass,channel=channel,process=process)
            mass_range = Flux.iniE()
            Flux_osc = Flux.Halo('detector',zenith=np.deg2rad(-29.00781+90))
            plt.figure(figsize = (6,4))
            for i, nu_flavor in enumerate(nu_flavor):
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
            Flux = Spectra_generator(mass=mass,channel=channel,process=process)
            mass_range = Flux.iniE()
            Flux_osc = Spectra_extrapolate(mass=mass, channel=channel, process=process, source=source)
            plt.figure(figsize = (6,4))
            for i, nu_flavor in enumerate(nu_flavor):
                Spectra_interp = splrep(mass_range, Flux_osc[i]/mass)
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
        
def Flux_generator(mass, channel, process, galactic_profile=profile.NFW, Ntheta=300):
    Flux = Spectra_generator(mass=mass, channel=channel)
    mass_range = Flux.iniE()
    Flux_osc = Flux.Halo('detector',zenith=np.deg2rad(-29.00781+90))
    nu_flavor = ['nu_e','nu_mu','nu_tau','nu_e_bar','nu_mu_bar','nu_tau_bar']
    R       = 100.  
    d       = 8     
    theta = np.linspace(0.,np.pi,Ntheta)
    J      = profile.J(galactic_profile,R,d,process)
    Jtheta = [J.Jtheta(j) for j in theta]
    Transpose_flux, Flux, True_flux_list = [],[],[]
    for k, nu_flav in enumerate(nu_flavor):
        Transpose_flux.append(Flux_osc[nu_flav][:,None])
        Flux.append(Transpose_flux[k]*Jtheta)
    for j, nu_flav in enumerate(nu_flavor):
        plt.figure(figsize = (6,4))
        plt.title(r'Neutrinos flux for $m_X$ = {0}PeV | channel = {1}'.format(int(mass/1e6),channel))
        plt.subplot(211)
        plt.title(r'Neutrinos flux function of opening angle | channel = {0} |$m_X = {1}$PeV'.format(channel,int(mass/1e6)), fontdict = font)
        plt.plot(theta, Flux[j][1,:], label = nu_flavor[j], color = color[j])
        plt.yscale('log')
        plt.grid()
        plt.xlabel(r"$\theta \; (rad)$")
        plt.xticks(ticks=[0,np.pi/2,np.pi],labels=['$0$','$\pi/2$','$\pi$'])
        plt.ylabel(r"$d\phi_\nu/dE\;[GeV^{-1}]$")
        plt.legend()
        
        plt.subplot(212)
        plt.title(r'Neutrinos flux function of Energy | channel = {0} |$m_X = {1}$PeV'.format(channel,int(mass/1e6)), fontdict = font)
        plt.plot(mass_range, Flux[j][:,1], label = nu_flavor[j], color = color[j])
        plt.yscale('log')
        plt.semilogx()
        plt.grid()
        plt.xlabel("$E [GeV]$", fontdict = font)
        plt.ylabel(r"$d\phi_\nu/dE\;[GeV^{-1}]$")
        plt.legend()

def J_interpolation(process, theta_true_center, galactic_profile=profile.NFW, Ntheta = 300):
    nu_flavor = ['nu_e','nu_mu','nu_tau','nu_e_bar','nu_mu_bar','nu_tau_bar']
    R       = 100.  
    d       = 8     
    opening_angle = np.linspace(0.,np.pi,Ntheta)
    J      = profile.J(galactic_profile,R,d,process)
    Jtheta = [J.Jtheta(j) for j in opening_angle] # (rad)
    J_interp = splrep(np.rad2deg(opening_angle), Jtheta)
    True_J = splev(theta_true_center, J_interp, der = 0) # (deg)
    plt.subplot(211)
    plt.plot(np.rad2deg(opening_angle), Jtheta, 'g-+', label = 'J profile')
    plt.xlim(0.,180) 
    plt.yscale('log') 
    plt.xlabel(r'$\theta [^\circ]$')
    plt.ylabel(r'$J(\theta)\;[\mathrm{GeV^2/cm^5}]$')
    plt.legend()
    plt.subplot(212)       
    plt.plot(theta_true_center, True_J,'k-+',label = 'interpolation')
    plt.xlim(0.,180) 
    plt.yscale('log') 
    plt.xlabel(r'$\theta [^\circ]$')
    plt.ylabel(r'$J(\theta)\;[\mathrm{GeV^2/cm^5}]$')
    plt.legend()
    return True_J
   
def Flux_interpolation(mass,channel,process,theta_true_center, E_true_center, source, extrapolation): 
    nu_flavor = ['nu_e','nu_mu','nu_tau','nu_e_bar','nu_mu_bar','nu_tau_bar']
    Transpose_flux, True_flux = [],[]
    J_true = J_interpolation(process=process, theta_true_center=theta_true_center)
    True_spectra_list = Spectra_interpolation(mass=mass, channel=channel,process=process, E_true_center=E_true_center, source=source, extrapolation=extrapolation)
    Flux = Spectra_generator(mass=mass,channel=channel,process=process)
    Flux_osc = Flux.Halo('detector',zenith=np.deg2rad(-29.00781+90))
    for i, nu_flavor in enumerate(nu_flavor):
        Transpose_flux.append(Flux_osc[nu_flavor][:,None])
        True_flux.append(Transpose_flux[i]*J_true)
    return True_flux

def Signal_PDF(mass, channel, process):
    resp_matrix_data = np.load('./Resp_MC1122_logE.pkl',allow_pickle=True, encoding="latin1")
    Resp = resp_matrix_data['Resp']
    True_energy_center = resp_matrix_data['Bin']['true_energy_center']
    True_psi_center = resp_matrix_data['Bin']['true_psi_center']
    Reco_energy_center = resp_matrix_data['Bin']['reco_energy_center']
    Reco_psi_center = resp_matrix_data['Bin']['reco_psi_center']
    
    True_flux = Flux_interpolation(mass=mass,channel=channel,process=process, theta_true_center=True_psi_center, E_true_center=True_energy_center)
    
    grid = np.meshgrid(True_psi_center, True_energy_center, Reco_psi_center, Reco_energy_center, indexing='ij')
    RecoRate = np.zeros((len(Reco_psi_center),len(Reco_energy_center)))

    for i, nu_flavor in enumerate(Resp.keys()):
        TotalWeight = np.sum(Resp[nu_flavor])
        dRdlogE=Resp[nu_flavor]*grid[1]
        RespPdf = dRdlogE/np.sum(dRdlogE)
        RecoRate += np.tensordot(RespPdf*TotalWeight, True_flux, axes=([0,1], [1,0]))
    return RecoRate 

def psi_f(RA,decl):
    return np.arccos(np.cos(np.pi/2.-(-29.0078*np.pi/180))*np.cos(np.pi/2.-decl)\
                      +np.sin(np.pi/2.-(-29.0078*np.pi/180))*np.sin(np.pi/2.-decl)*\
                       np.cos(RA-266.4168*np.pi/180))

def Spectra_extrapolate(mass, channel, process, kind = 'cubic', source=True):
    flux_extrapol = []
    nu_flavor = ['nu_e','nu_mu','nu_tau','nu_e_bar','nu_mu_bar','nu_tau_bar']
    Flux = Spectra_generator(mass = mass, channel = channel, process=process)
    mass_range = Flux.iniE()
    if source == True:
        Flux_Halo = Flux.iniFlux('Halo')
        for i in nu_flavor:
            index = 0
            for k in range(0,len(Flux_Halo[i])-1):
                if round(Flux_Halo[i][k+1],2) == round(Flux_Halo[i][k],2):
                    index = index + 1
            if process == 'ann':
                flux_cut = Flux_Halo[i][index:]
                mass_range_cut = mass_range[index:]
                func = interp1d(mass_range_cut, flux_cut,
                bounds_error=False,
                kind=kind,
                fill_value='extrapolate')
                newflux_halo = [func(i)/mass for i in mass_range]
                flux_extrapol.append(newflux_halo)
                plt.figure(figsize = (6,4))
                plt.title(r'Energy spectra at source | channel : {0} | $m_X = {1}$PeV'.format(channel,int(mass/1e6)), fontdict = font)
                plt.yscale('log')
                plt.semilogx()
                plt.grid()
                plt.xlabel(r"$E$ [GeV]", fontdict = font)
                plt.ylabel(r"$dN_\nu/dE_\nu\;[GeV^{-1}]$", fontdict = font)
                for j in range(len(flux_extrapol)):
                    plt.plot(mass_range, flux_extrapol[j],color=color[j],label=nu_flavor[j])
                plt.legend()
            else:
                flux_cut = Flux_Halo[i][index:200]
                mass_range_cut = mass_range[index:200]
                func = interp1d(mass_range_cut, flux_cut,
                bounds_error=False,
                kind=kind,
                fill_value='extrapolate')
                newflux_halo = np.array([func(i)/mass for i in mass_range[0:index]])
                Flux_halo_tail = np.array(Flux_Halo[i][index:]/mass)
                newflux = np.concatenate((newflux_halo, Flux_halo_tail), axis = None)
                flux_extrapol.append(newflux)
                plt.figure(figsize = (6,4))
                plt.title(r'Energy spectra at source | channel : {0} | $m_X = {1}$PeV'.format(channel,int(mass/1e6)), fontdict = font)
                plt.yscale('log')
                plt.semilogx()
                plt.grid()
                plt.xlabel(r"$E$ [GeV]", fontdict = font)
                plt.ylabel(r"$dN_\nu/dE_\nu\;[GeV^{-1}]$", fontdict = font)
                for j in range(len(flux_extrapol)):
                    plt.plot(mass_range, flux_extrapol[j],color=color[j],label=nu_flavor[j])
                plt.legend()
        return flux_extrapol
    else:   
        Flux_osc = Flux.Halo('detector',zenith=np.deg2rad(-29.00781+90))
        for i in nu_flavor:
            index = 0
            for k in range(0,len(Flux_osc[i])-1):
                if round(Flux_osc[i][k+1],2) == round(Flux_osc[i][k],2):
                    index = index + 1
            if process == 'ann':
                flux_cut = Flux_osc[i][index:]
                mass_range_cut = mass_range[index:]
                func = interp1d(mass_range_cut, flux_cut,
                bounds_error=False,
                kind=kind,
                fill_value='extrapolate')
                newflux_osc = [func(i)/mass for i in mass_range]
                flux_extrapol.append(newflux_osc)
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
            else:
                flux_cut = Flux_osc[i][index:200]
                mass_range_cut = mass_range[index:200]
                func = interp1d(mass_range_cut, flux_cut,
                bounds_error=False,
                kind=kind,
                fill_value='extrapolate')
                newflux_osc = np.array([func(i)/mass for i in mass_range[0:index]])
                Flux_osc_tail = np.array(Flux_osc[i][index:]/mass)
                newflux = np.concatenate((newflux_osc, Flux_osc_tail), axis = None)
                flux_extrapol.append(newflux)
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
        return flux_extrapol

