import os,time,sys
import matplotlib.pyplot as plt 
import numpy as np
import charon.physicsconstants as PC
pc = PC.PhysicsConstants()
from charon import profile
from Spectra_functions import Spectra_generator, Spectra_interpolation, Flux_generator, J_interpolation, Flux_interpolation, Signal_PDF
from scipy.interpolate import splrep, splev
from ipynb.fs.full.func_plot_histo import plot_projections
from scipy.interpolate import interp1d
import random as rd
sys.path.append("/home/dvaldenaire/Python/Analysis/DMfit/DMfit")
from modeling import PdfBase, Model, Parameter
from data_DM import DataSet
from llh import LikelihoodRatioTest

def Background_cut():
    data = np.load('Burnsample/Bkg.pkl',allow_pickle=True, encoding="latin1")
    return data

def Signal_injection(mass, channel, process, galactic_profile):
    Reco_rate_NFW = Signal_PDF(mass=1e7, channel='nuenue',process='decay',galactic_profile=profile.NFW, extrapolation=True, normalize=False)
    Background = Background_cut()
    
    
def Sensitivity(mass_range,channel,process,galactic_profile, CL=90,exposure=2933.8*24*60*60):
    Sensitivity, SignalFraction = [],[]
    Background = Background_cut()
    SumBackground = np.sum(Background)
    for mass in mass_range:
        # PARAMETERS:
        if mass == 1e6:
            extrapolation = False
        else:
            extrapolation = True
        RecoRate = Signal_PDF(mass=mass, channel=channel,process=process,galactic_profile=galactic_profile, extrapolation=extrapolation, normalize=False)
        ParameterH1 = Parameter(value=0.5, limits=(0,1), fixed=False, name='H1')
        ParameterH0 = Parameter(value=0., limits=(0,1), fixed=True, name='H0')
        # PDFs:
        SignalPDF = PdfBase(RecoRate.flatten()/np.sum(RecoRate), name='Signal PDF')
        BackgroundPDF = PdfBase((Background.flatten())/np.sum(Background), name='Background PDF')
        PseudoData = Background_cut()
        # HYPOTHESES:
        modelH1 = ParameterH1*SignalPDF + (1-ParameterH1)*BackgroundPDF
        modelH0 = ParameterH0*SignalPDF + (1-ParameterH0)*BackgroundPDF
        #DATA SET:
        ds = DataSet()
        ds.asimov(SumBackground, modelH0)
        #LIKELIHOOD:
        lr = LikelihoodRatioTest(model = modelH1, null_model = modelH0)
        lr.data = ds
        lr.fit("H1")
        lr.fit("H0")
        print('TS value at the output upper limit: {0}'.format(lr.TS))
        #SENSITIVITY:
        xi_CL = lr.upperlimit_llhinterval('H1', 'H0', CL)
        print('Signal fraction Ns/Ntot:', xi_CL)
        Nsignal = xi_CL*SumBackground
        SignalFraction.append([xi_CL, Nsignal])
        print('\n Nb signal:', round(Nsignal))
        # Convert to thermal cross-section:
        if process == 'decay':
            K = 4*np.pi*mass
        else:
            K = 8*np.pi*mass**2
        Sensitivity.append(Nsignal/np.sum((1/K)*RecoRate*exposure))
    return(SignalFraction, Sensitivity)
        