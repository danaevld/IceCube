import os,time
import matplotlib.pyplot as plt 
import numpy as np
from charon import propa
import charon.physicsconstants as PC
pc = PC.PhysicsConstants()
from charon import profile
from scipy.interpolate import splrep, splev
from ipynb.fs.full.func_plot_histo import plot_projections

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 12,
        }

color = ['limegreen', 'darkgreen', 'mediumblue', 'indigo','mediumvioletred','green']

def Spectra_generator(mass,channel,process='ann',galactic_profile=profile.NFW,nodes=300,bins=300,logscale=True,interactions=True):
    theta_12=33.82
    theta_13=8.6
    theta_23=48.6
    delta_m_12=7.39e-5
    delta_m_13=2.528e-3
    delta = 221.
    #theta = np.linspace(0.,np.pi,100)
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
    plt.xlim(1,Emax)
    plt.semilogx()
    plt.grid()
    plt.xlabel("$E [GeV]$", fontdict = font)
    plt.ylabel(r"$dN_\nu/dE_\nu\;[GeV^{-1}]$", fontdict = font)
    plt.legend()
    return Flux
    
def Spectra_interpolation(mass, channel, E_true_center, process='ann'):
    True_spectra_list = []
    nu_flavor = ['nu_e','nu_mu','nu_tau','nu_e_bar','nu_mu_bar','nu_tau_bar']
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
    
def Flux_generator(mass, channel, process='ann', galactic_profile=profile.NFW, Ntheta=300):
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
        plt.figure(figsize = (8,6))
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
   
def Flux_interpolation(mass,channel,theta_true_center, E_true_center): 
    nu_flavor = ['nu_e','nu_mu','nu_tau','nu_e_bar','nu_mu_bar','nu_tau_bar']
    Transpose_flux, True_flux = [],[]
    J_true = J_interpolation(process = 'ann', theta_true_center=theta_true_center)
    True_spectra_list = Spectra_interpolation(mass=mass, channel=channel, E_true_center=E_true_center)
    Flux = Spectra_generator(mass=mass,channel=channel)
    Flux_osc = Flux.Halo('detector',zenith=np.deg2rad(-29.00781+90))
    for i, nu_flavor in enumerate(nu_flavor):
        Transpose_flux.append(Flux_osc[nu_flavor][:,None])
        True_flux.append(Transpose_flux[i]*J_true)
    return True_flux
    
'''    
resp_matrix_data = np.load('/home/dvaldenaire/Python/Analysis/Resp_MC1122_logE.pkl',allow_pickle=True, encoding="latin1")
True_energy_center = resp_matrix_data['Bin']['true_energy_center']
True_psi_center = resp_matrix_data['Bin']['true_psi_center']
True_Flux = Flux_interpolation(mass=1e7, channel = 'nuenue', theta_true_center=True_psi_center, E_true_center=True_energy_center)'''

