import numpy as np
from numpy import *
import math
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import quad, nquad, odeint, simps
from scipy import optimize, signal, interpolate
import sys
import h5py
import os
import platform

###############################!!!CODE USE INSTRUCTIONS AND TIPS!!!#########################################
############################################################################################################
#THIS CODE MODELS SUPERNOVA SPECTRA LINE PROFILES BY POST-PROCESSING HYDRODYNAMICS DATASETS
#THE SOBOLEV APPROXIMATION IS USED (V>V_TH)
#BEST WORKING VERSION SO FAR: 60-day and 30-day STELLA IIP input with p/z range -4,4.

#NOTES FOR COMPARING STELLA MODEL AGAINST CONTEMPORANEOUS MESA/FLASH MODEL:
#---------------------------------------------------------------------------
#IF THE TEMPERATURES DO NOT MATCH, THE EFFECT ON THE SAHA SOLVER WILL BE LARGE AND IONIZATION
#FRACTIONS WILL DIFFER SIGNIFICANTLY, PRODUCING A DIFFERENT PROFILE WITH SUPPRESSED BLUE SIDE.
#WIGGLES AND LARGE GRADIENTS ALSO HAVE A SECONDARY EFFECT. AS A RESULT, SETTING TAU_PH TO A LARGER
#VALUE IN COMBINATION WITH USING A TEMPERATURE MORE CONSISTENT WITH STELLA'S SEEM TO LEAD TO AGREEMENT.
#FOR EXAMPLE, THE T=30DAY STELLA PROFILE WAS MATCHED WELL BY THE MESA DATA BY USING THE STELLA TEMPERATURES
#SCALED DOWN (0.9*STELLA_TEMP) AND WITH TAU_PHOT SET TO 10 INSTEAD OF 1.
#A SECOND-ORDER EFFECT IS RESAMPLING; MESA COMES WITH MORE DATA AND A DIRECT COMPARISON TO STELLA WOULD
#REQUIRE RE-SAMPLING THE DATA DOWN TO THE STELLA DATA SIZE. HIGH RESOLUTION ALSO ALWAYS HELPS.
#A MORE "CONSISTENT" WAY TO MATCH THE STELLA RESULTS WITH THE MESA OUTPUT IS SIMPLY TO: (A) SET TAU_PH = 50,
#(B) RESAMPLE DOWN TO STELLA DATA SIZE AND, (C) SET MESA TEMP DATA TO 70% OF ORIGINAL VALUES.
#SAME RECIPE WORKS FOR FLASH TOO.

#NOTES FOR COMPARING MODELS WITH/WITHOUT 1/R^2 WIND APPENDED:
#---------------------------------------------------------------------------
#THE CURRENT IMPLEMENTATION OF (1/R^2) WIND APPENDING AT THE END OF THE INPUT PROFILE IS VERY SENSITIVE
#TO THE V_WIND PARAMETER MAINLY BECAUSE IT CONTROLS THE VELOCITY GRADIENT AT THE STAR-WIND INTERFACE (DVDR)
#AND THEREFORE THE SOBOLEV OPTICAL DEPTH. A "FIX" IS TO SET V_W TO A VALUE CLOSE TO THE VELOCITY OF THE OUTER
#CELL OF THE INPUT DATA. IN GENERAL, IF V_W >~ V_IN,OUTER THEN THE PROFILE TENDS TO RESEMBLE THE "NO WIND"
#CASE IN STELLA. IF, ON THE OTHER HAND, V_W << V_IN,OUTER, THEN THE BLUE SIDE IS SEVERELY SUPPRESSED.
#LARGE NEGATIVE VELOCITY GRADIENTS SUPPRESS PROFILE. IF, INSTEAD, ONE USES THE INPUT OPTICAL DEPTH, THE
#PROFILES BETWEEN THE NO-WIND AND WIND CASE MATCH (AS SUGGESTED BY RUNNING EXPERIMENT ON INPUT STELLA
#30-DAY PROFILE). LATEST UPDATE ALLOWS FOR A SMOOTH VELOCITY TRANSITION BETWEEN THE INPUT DATA AND THE WIND
#THAT FOLLOWS AND EXPONENTIAL DECLINE DOWN TO THE DESIRED V_W AT LARGE RADII. THE STEEPNESS OF THIS TRANSITION
#IS CONTROLLED BY THE W_STEEP PARAMETER. SETTING THIS TO A SMALLER VALUE MAKES THE TRANSITION MORE GRADUAL
#AND ALLOWS FOR THE RECOVERY OF THE ORIGINAL "NO-WIND" LINE PROFILE.
#SUMMARY: EITHER MATCH V_W TO OUTER VELOCITY OF DATA OR INPUT A VERY SMOOTH TRANSITION (W_STEEP~0.1, PREFERRED).
#IN ADDITION: IN SITUATIONS WITH STEEP WINDS USING A SMALLER X/P GRID AND RESOLVING MORE HELPS
#UPDATE: WIND ISSUE WITH HIGH W_STEEP FIXED, BY CONSIDERING Z = X_SPACE[I] INSTEAD OF Z = 10000 IN RESONANCE
#SURFACE SOLVER.

############################################################################################################
############################################################################################################

#1. INPUTS & CONSTANTS [cgs]
####CONSTANTS###############
pi     = math.pi        #Pi
mp     = 1.6726231e-24  #Mass of proton
mH     = 1.6605402e-24  #Mass of hydrogen atom
mHe    = 6.6464764e-24  #Mass of helium atom
mC     = 1.9944235e-23  #Mass of carbon atom
mN     = 2.3258671e-23  #Mass of nitrogen atom
mO     = 2.6561e-23     #Mass of oxygen atom
me     = 9.1093897e-28  #Mass of electron
ee     = 4.8032068e-10  #Charge of electron
c      = 2.99792458e+10 #Speed of light
kb     = 1.38065812e-16 #Boltzmann constant
sigma  = 5.6705119e-5   #Stefan-Boltzmann constant
alpha  = 7.5646e-15     #Radiation density constant
hpl    = 6.62607554e-27 #Plack constant
NA     = 6.02213674e+23 #Avogadro's number
Rgas   = 8.314e+7       #Universal gas constant (R)
ev2erg = 1.60218e-12    #eV to erg conversion
msun   = 1.99e+33       #solar mass in cgs
Lsun   = 3.99e+33       #solar mass in cgs
yrtosec= 31622400.0     #seconds in a year
kms2cms= 1.0e+5         #km/s to cm/s conversion factor
####LINE DATA INPUTS########
A32    = 4.4101e+07    #Einstein A coefficient [Hz]
l0     = 6564.64       #Line-center wavelength
f0     = 6.4108e-1     #Line oscillator strength
gl     = 8.0           #Line transition occupation number for lower level
gu     = 18.0          #Line transition occupation number for upper level
xI     = 13.6          #Species ionization potential (eV)
####GRID PARAMETERS#########
rph    = 1.0           #Normalized photosphere radius
tau_ph = 0.666           #Desired photosphere optical depth
Iph    = 1.0           #Normalized photosphere intensity
pmin   = -4            #Minimum of p-space range (impact parameter)
pmax   = 4             #Maximum of p-space range (impact parameter)
p_bins = 200            #Number of bins in the p-space (impact parameter), typically > 100 for good quality
xmin   = -4            #Minimum value of x (frequency) array
xmax   = 4             #Maximum value of x (frequency) array
x_bins = 200            #Number of bins in the x (frequency) array, typically > 100 for good quality
vel_tr = 1.0e+07       #Inner truncation velocity for MESA/FLASH input data only
tr_w_flash = False      #Truncate wind data? (For FLASH input ONLY)
resample = False       #Resample initial data arrays to calculate smoother line profile
sampl_level = 1598      #Resampling level (i.e. 5000 means new arrays will have length of 5000)
append_w = False       #Append wind data at end of input hydro profiles, assuming 1/r^2 wind structure (STELLA/MESA ONLY)
w_bins   = 100         #Size of wind data array
mdot_w   = 0.3         #Wind mass-loss rate in Msun/yr
v_wind   = 100.0       #Wind velocity in km/s
w_steep  = 100.0       #Steepness of transition from outer cell velocity of input data to v_wind (0.1-1.0)
stl_wind_fix = False   #For CSM-STELLA profiles ONLY re-establish that wind above CSI shock has velocity = v_wind
####PHYSICS OPTIONS########
TopLighting = False    #Use the Source function that accounts for CSI Top-Lighting effects (Branch et al. 2000).
                       #Works best for STELLA/FLASH input with CSI data. For other disable TL_Auto_Mode.
                       #DOES NOT WORK with appended wind (for STELLA/MESA).
TL_Auto_Mode = True   #If enabled it automatically detects CSI front for TopLighting and the muting factor, otherwise user inputs the values below
Ics      = 0.3         #Specific Intensity of CSI Top-Lighting front
Rcs      = 2.0         #Normalized radius of CSI shock
sobolev_tau = True     #Use optical depth perscription based on Sobolev formula? If .false. code uses input tau data
nLTE = False            #Enable nLTE treatment for calculation of atomic levels by using Cloudy.
                       #For this to work, the user NEEDS to have a compiled cloudy.exe in the same directory as this code.
nLTE_Atm = True        #If False it post-processes entire input profile with Cloudy for nLTE. If True, it only applies nLTE to the
                       #optically-thin region, above the photosphere. Buggy and needs improvement.
plt_spec = False        #If True, it plots Cloudy-computed optical spectrum for comparison purposes.
#####I/O Controls###########
debug    = True          #De-bug issues by printing out all post-processed data used for calculation of line profile
plot_dbg = False           #Plot processed input hydrodynamic profiles (STELLA ONLY)
db_f1 = 'csm_1msun_'+str(sys.argv[1]).split(".")[1][0:6]+'.dat'        #De-bug output file name for main final profile data
db_f2 = 'csm_1msun_'+str(sys.argv[1]).split(".")[1][0:6]+'b.dat'    #De-bug output file name for Sobolev optical depth data
input_mode = 'stella'     #Read-in input extracted from either FLASH/STELLA or MESA
fname = 'csm_1msun_day40_line.dat'        #Output line profile file name

#2. DATA INPUT FROM EXTRENAL HYDRODYNAMICS FILE (STELLA, MESA OR FLASH)
#STELLA (DEFAULT)
if input_mode == 'stella':
    #DATA INPUT FROM FIDUCIAL MODEL
    """
    velx, radius, h1, tau, temp, rho, nel, lumin, trad, prad, opacit = loadtxt(sys.argv[1], usecols=(3,0,5,4,2,1,6,0,0,0,0), unpack=True, skiprows=6)
    he4 = np.ones(len(velx)) * (1.0 - 0.7381 - 0.0134)
    c12 = np.ones(len(velx)) * 0.0029
    n14 = np.ones(len(velx)) * 0.0009
    o16 = np.ones(len(velx)) * 0.0077
    ro = [radius[0]/2.0,]
    dro = [ro[0],]
    mo = [(4.0*math.pi*ro[0]**2)*dro[0]*rho[0],]
    for i in range(1,len(radius)):
        ro.append(i)
        dro.append(i)
        mo.append(i)
        ro[i] = (radius[i] + radius[i-1])/2.0
        dro[i] = ro[i] - ro[i-1]
        mo[i] = mo[i-1] + (4.0*math.pi*ro[i]**2)*dro[i]*rho[i]

    #[velx, radius, h1, he4, c12, n14, o16, tau, temp, rho, mo, ro, nel, lumin, trad, prad, opacit] = [velx[:-1], radius[:-1], h1[:-1], he4[:-1], c12[:-1], n14[:-1], o16[:-1], tau[:-1], temp[:-1], rho[:-1], mo[:-1], ro[:-1], nel[:-1], lumin[:-1], trad[:-1], prad[:-1], opacit[:-1]]
    """
    try:
        velx, radius, h1, he4, c12, n14, o16, tau, temp, rho, mo, ro, nel, lumin, trad, prad, opacit = loadtxt(sys.argv[1], usecols=(4,3,13,15,16,17,18,10,7,5,11,12,36,34,8,6,9), unpack=True, skiprows=6)
        #Remove last array element corresponding to boundary in STELLA output
        [velx, radius, h1, he4, c12, n14, o16, tau, temp, rho, mo, ro, nel, lumin, trad, prad, opacit] = [velx[:-1], radius[:-1], h1[:-1], he4[:-1], c12[:-1], n14[:-1], o16[:-1], tau[:-1], temp[:-1], rho[:-1], mo[:-1], ro[:-1], nel[:-1], lumin[:-1], trad[:-1], prad[:-1], opacit[:-1]]
    except:
        print("You are not passing the right input to the reader. Expected Input Format: STELLA")
        sys.exit(1)

if input_mode == 'mesa':
    try:
        velx, radius, h1, he4, c12, n14, o16, tau, temp, rho, mo, ro, free_e, ye, lumin = loadtxt(sys.argv[1], usecols=(35,50,87,90,91,92,93,56,52,23,82,50,63,7,17), unpack=True, skiprows=6)
        velx, radius, h1, he4, c12, n14, o16, tau, temp, rho, mo, ro, free_e, ye, lumin = [velx[::-1], radius[::-1], h1[::-1], he4[::-1], c12[::-1], n14[::-1], o16[::-1], tau[::-1], temp[::-1], rho[::-1], mo[::-1], ro[::-1], free_e[::-1], ye[::-1], lumin[::-1]]
        mo = mo*msun
        diff0 = [(abs(vel_tr - x),idx) for (idx,x) in enumerate(velx)]
        diff0.sort()
        ind_trunc = diff0[0][1]
        ind_max = len(radius)-1
        velx, radius, h1, he4, c12, n14, o16, tau, temp, rho, mo, ro, free_e, ye, lumin = [velx[ind_trunc:ind_max], radius[ind_trunc:ind_max], h1[ind_trunc:ind_max], he4[ind_trunc:ind_max], c12[ind_trunc:ind_max], n14[ind_trunc:ind_max], o16[ind_trunc:ind_max], tau[ind_trunc:ind_max], temp[ind_trunc:ind_max], rho[ind_trunc:ind_max], mo[ind_trunc:ind_max], ro[ind_trunc:ind_max], free_e[ind_trunc:ind_max], ye[ind_trunc:ind_max],lumin[ind_trunc:ind_max]]
        lumin = lumin * Lsun
        nel = (free_e * rho) * (h1/1.0 + he4/4.0 + c12/12.0 + n14/14.0 + o16/14.0) / mp
        ###temp = temp*0.7 # FIX USED ONLY WHEN COMPARING CONTEMPORANEOUS MESA/STELLA DATA
    except:
        print("You are not passing the right input to the reader. Expected Input Format: MESA")
        sys.exit(1)

#FLASH
if input_mode == 'flash':
    try:
        file = h5py.File(sys.argv[1],'r')
        rho1 = file['dens']
        velx1 = file['velx']
        temp1 = file['temp']
        coord = file['bounding box']
        node = file['node type']
        h1_i = file['h1  ']
        he4_i = file['he4 ']
        c12_i = file['c12 ']
        n14_i = file['n14 ']
        o16_i = file['o16 ']
        ye1 = file['ye  ']
        blksize=file['block size']
        nblocks = len(coord)
        nzones = rho1.shape[3]

        l = -1
        blk = []
        radius = []
        rho = []
        temp = []
        velx = []
        h1 = []
        he4 = []
        c12 = []
        n14 = []
        o16 = []
        opac = []
        ye = []
        lumin = []
        for i in range(0,nblocks):
            blk.append(i)
            blk[i] = blksize[i][0]
            dd=(blk[i]/16)
            dd = dd/2
            if node[i] == 1:
                for j in range(0,nzones):
                    l = l + 1
                    radius.append(l)
                    rho.append(l)
                    velx.append(l)
                    temp.append(l)
                    opac.append(l)
                    ye.append(l)
                    h1.append(l)
                    he4.append(l)
                    c12.append(l)
                    n14.append(l)
                    o16.append(l)
                    lumin.append(l)

                    radius[l] = coord[()][i][0][0]+(2*j+1)*dd
                    rho[l] = rho1[()][i][0][0][j]
                    velx[l] = velx1[()][i][0][0][j]
                    temp[l] = temp1[()][i][0][0][j] ## * 0.7
                    ye[l] = ye1[()][i][0][0][j]
                    h1[l] = h1_i[()][i][0][0][j]
                    he4[l] = he4_i[()][i][0][0][j]
                    c12[l] = c12_i[()][i][0][0][j]
                    n14[l] = n14_i[()][i][0][0][j]
                    o16[l] = o16_i[()][i][0][0][j]
                    opac[l] = (1.0 +  h1[l]) * 0.2 #Thomspon electron scattering opacity
                    lumin[l] = 4.0*math.pi*(radius[l]**2)*sigma*temp[l]**4

        #Calculate mass coordinate array
        mo1 = [(4.0/3.0)*math.pi*radius[0]**3,]
        dr = [radius[0],]
        nel = [(mo1[0]/mp)*ye[0]/(4.0*math.pi*radius[0]**3.0/3.0),]
        for i in range(1,len(radius)):
            mo1.append(i)
            dr.append(i)
            nel.append(i)
            dr[i] = radius[i]-radius[i-1]
            mo1[i] = rho[i]*4.0*math.pi*dr[i]*radius[i]**2.0
            nel[i] = (ye[i] * rho[i]) * (h1[i]/1.0 + he4[i]/4.0 + c12[i]/12.0 + n14[i]/14.0 + o16[i]/14.0) / mp

        summ = mo1[0]
        mo = [mo1[0],]
        ro = [radius[0],]
        for i in range(1,len(mo1)):
            mo.append(i)
            ro.append(i)
            summ = summ + mo1[i]
            mo[i] = summ
            ro[i] = radius[i] + dr[i]

        #Calculate optical depth array
        opac2 = opac[::-1]
        rho2 = rho[::-1]
        dr2 = dr[::-1]
        tau = []
        summ2 = 0.0
        for i in range(0,len(opac)):
            tau.append(i)
            summ2 = summ2 + opac2[i]*rho2[i]*dr2[i]
            tau[i] = summ2

        tau = tau[::-1]

     #Remove first array element corresponding to inner boundary in FLASH output
        [velx, radius, h1, he4, c12, n14, o16, tau, temp, rho, mo, ro, nel,lumin] = [velx[1:], radius[1:], h1[1:], he4[1:], c12[1:], n14[1:], o16[1:], tau[1:], temp[1:], rho[1:], mo[1:], ro[1:], nel[1:], lumin[1:]]

        diff0 = [(abs(vel_tr - x),idx) for (idx,x) in enumerate(velx)]
        diff0.sort()
        ind_trunc = diff0[0][1]
        ind_max = len(radius)-1

        velx, radius, h1, he4, c12, n14, o16, tau, temp, rho, mo, ro, nel, lumin = [velx[ind_trunc:ind_max], radius[ind_trunc:ind_max], h1[ind_trunc:ind_max],  he4[ind_trunc:ind_max],  c12[ind_trunc:ind_max],  n14[ind_trunc:ind_max],  o16[ind_trunc:ind_max], tau[ind_trunc:ind_max], temp[ind_trunc:ind_max], rho[ind_trunc:ind_max], mo[ind_trunc:ind_max], ro[ind_trunc:ind_max], nel[ind_trunc:ind_max], lumin[ind_trunc:ind_max]]

        if tr_w_flash:
            ind_max_v = np.argmax(velx)
            velx, radius, h1, he4, c12, n14, o16, tau, temp, rho, mo, ro, nel, lumin = [velx[0:ind_max_v], radius[0:ind_max_v], h1[0:ind_max_v], he4[0:ind_max_v], c12[0:ind_max_v], n14[0:ind_max_v], o16[0:ind_max_v], tau[0:ind_max_v], temp[0:ind_max_v], rho[0:ind_max_v], mo[0:ind_max_v], ro[0:ind_max_v], nel[0:ind_max_v], lumin[0:ind_max_v]]


    except:
        print("You are not passing the right input to the reader. Expected Input Format: FLASH")
        sys.exit(1)

#----------------------------------------------------------------------------------------------------------------------------------------------

#3. OPTIONALLY APPEND WIND DATA PAST THE CALCULATED HYDRO INPUT RANGES
if append_w and input_mode in ['stella','mesa']:
    #First calculate last cell width of original array
    dr_last = radius[len(radius)-1] - radius[len(radius)-2]

    r_inner = radius[len(radius)-1]
    r_outer = 2.0 * r_inner
    radius_w = np.linspace(r_inner,r_outer,w_bins)
    dr_w = radius_w[1]-radius_w[0]
    ro_w = radius_w

    [velx_w,h1_w,temp_w,rho_w,mo1_w,nel_w,kappa_w,tau1_w] = [[],[],[],[],[],[],[],[]]
    for i in range(0,len(radius_w)):
        velx_w.append(i)
        h1_w.append(i)
        temp_w.append(i)
        rho_w.append(i)
        mo1_w.append(i)
        nel_w.append(i)
        kappa_w.append(i)
        tau1_w.append(i)

        velx_w[i] = (velx[len(radius)-1] - v_wind * kms2cms)*np.exp(-w_steep*(radius_w[i]-r_inner)/r_inner) + v_wind * kms2cms
        h1_w[i] = h1[len(radius)-1]
        temp_w[i] = temp[len(radius)-1]
        rho_w[i] = mdot_w * (msun/yrtosec) / (4.0*pi*v_wind*kms2cms*radius_w[i]**2)
        nel_w[i] = nel[len(radius)-1] * rho_w[i] / rho[len(radius)-1]
        mo1_w[i] = 4.0*pi*(radius_w[i]**2.0)*dr_w*rho_w[i]
        kappa_w[i] = (1.0 + h1_w[i]) * 0.2
        tau1_w[i] = kappa_w[i] * rho_w[i] * dr_w

    mo_w = [mo[len(mo)-1],]
    sum_m = mo[len(mo)-1]
    for i in range(1,len(radius_w)):
        mo_w.append(i)
        sum_m = sum_m + (mo1_w[i] + mo1_w[i-1])
        mo_w[i] = sum_m

    tau1_w_inv = tau1_w[::-1]
    tau_w = [tau1_w_inv[0],]
    sum_t = 0.0
    for i in range(1,len(tau1_w_inv)):
        tau_w.append(i)
        sum_t = sum_t + (tau1_w_inv[i] + tau1_w_inv[i-1])
        tau_w[i] = sum_t

    tau_w = tau_w[::-1]

    tau_factor = tau_w[0]/tau[len(tau)-1]

    tau_w = tau_w / tau_factor

    [velx, radius, h1, tau, temp, rho, mo, ro, nel] = [np.concatenate((velx,velx_w)),np.concatenate((radius,radius_w)),np.concatenate((h1,h1_w)),np.concatenate((tau,tau_w)),np.concatenate((temp,temp_w)),np.concatenate((rho,rho_w)),np.concatenate((mo,mo_w)),np.concatenate((ro,ro_w)),np.concatenate((nel,nel_w))]

#----------------------------------------------------------------------------------------------------------------------------------------------

#4. OPTIONALLY INTERPOLATE DATA ARRAY VALUES TO ENHANCE RESOLUTION FOR A SMOOTHER LINE PROFILE
if resample:
    #Define new radial array
    r_new  = np.linspace(radius[0],radius[len(radius)-1],sampl_level)
    #Generate interpolation functions
    f_rho  = interpolate.interp1d(radius, rho)
    f_vel  = interpolate.interp1d(radius, velx)
    f_h1   = interpolate.interp1d(radius, h1)
    f_he4  = interpolate.interp1d(radius, he4)
    f_c12  = interpolate.interp1d(radius, c12)
    f_n14  = interpolate.interp1d(radius, n14)
    f_o16  = interpolate.interp1d(radius, o16)
    f_tau  = interpolate.interp1d(radius, tau)
    f_temp = interpolate.interp1d(radius, temp)
    f_mo   = interpolate.interp1d(radius, mo)
    f_ro   = interpolate.interp1d(radius, ro)
    f_nel  = interpolate.interp1d(radius, nel)
    f_lumin = interpolate.interp1d(radius, lumin)
    f_prad = interpolate.interp1d(radius, prad)
    f_opacit = interpolate.interp1d(radius, opacit)
    #Produce interpolated values
    [rho_new,vel_new,h1_new,he4_new,c12_new,n14_new,o16_new,tau_new,lumin_new,prad_new,opacit_new] = [f_rho(r_new),f_vel(r_new),f_h1(r_new),f_he4(r_new),f_c12(r_new),f_n14(r_new),f_o16(r_new),f_tau(r_new),f_lumin(r_new),f_prad(r_new),f_opacit(r_new)]
    [temp_new,mo_new,ro_new,nel_new]  = [f_temp(r_new),f_mo(r_new),f_ro(r_new),f_nel(r_new)]
    #Map original arrays to interpolated arrays
    [velx, radius, h1, he4, c12, n14, o16, tau, temp, rho, mo, ro, nel, lumin, prad, opacit] = [vel_new, r_new, h1_new, he4_new, c12_new, n14_new, o16_new, tau_new, temp_new, rho_new, mo_new, ro_new, nel_new, lumin_new, prad_new, opacit_new]

#----------------------------------------------------------------------------------------------------------------------------------------------

if input_mode == 'stella':
    dens = []
    dm = []
    dvdr = []
    dvol = []
    dr = []
    drho = []
    for i in range(0,len(radius)):
        dm.append(i)
        dens.append(i)
        dvdr.append(i)
        dvol.append(i)
        dr.append(i)
        drho.append(i)
        if i == 0:
            dens[i] = rho[i]
            dm[i] = mo[i]
            dvdr[i] = velx[0]/ro[0]
            dvol[i] = dm[i] / dens[i]
            dr[i] = ro[i]
            drho[i] = dens[i] / ro[i]
        if i != 0:
            dens[i] = (mo[i]-mo[i-1])/(4.0*math.pi*(ro[i]-ro[i-1])*ro[i]**2.0)
            dm[i] = (mo[i]-mo[i-1])
            dvdr[i] = (velx[i] - velx[i-1])/(ro[i] - ro[i-1])
            dvol[i] = dm[i] / dens[i]
            dr[i] = ro[i]-ro[i-1]
            drho[i] = (rho[i] - rho[i-1])/(ro[i]-ro[i-1])

    dens = rho
    if stl_wind_fix:
        diff00 = [(abs(tau_ph - x),idx) for (idx,x) in enumerate(tau)]
        diff00.sort()
        ind_p0 = diff00[0][1]
        dRho_dR0 = abs(diff(dens)/diff(radius))
        dRho_dR0 = np.append(0.0, dRho_dR0)
        dRho_dR0 = dRho_dR0[ind_p0:len(dRho_dR0)]
        csir_ind0 = ind_p0 + np.argmax(dRho_dR0) - 1
        for i in range(0,len(radius)):
            if i < csir_ind0:
                velx[i] = velx[i]
            if i >= csir_ind0:
                velx[i] = v_wind * kms2cms


if input_mode in ['flash','mesa']:
    dens = rho

    dvdr = []
    for i in range(0,len(radius)):
        dvdr.append(i)
        if i == 0:
            dvdr[i] = velx[0]/ro[0]
        if i != 0:
            dvdr[i] = (velx[i] - velx[i-1])/(ro[i] - ro[i-1])

#----------------------------------------------------------------------------------------------------------------------------------------------

#6A. CALCULATE SOBOLEV OPTICAL DEPTH UNDER THE ASSUMPTION OF LTE FOR HYDROGEN
#NOTE: THIS WILL HAVE TO BE MODIFIED FOR SPECIES/LINES OTHER THAN HYDROGEN!
#NOTE (CONT'D): AN IDEA IS TO USE LTE/NLTE OUTPUT FROM CLOUDY OR CMFGEN
#First determine total number density of specific species
nH = np.multiply(dens,h1)/mH
tt = temp #trad
#Determine value of parition function for ground state as function of radius, value for ionized H is just 1.
Z_i = np.ones(len(dens))
E13 = np.asarray([-13.6,-3.4,-1.5])*ev2erg # first three Hydrogen ionization energies
g13 = np.asarray([2.0,8.0,18.0] )          # first three Hydrogen degeneracy parameters
Z_g = []
n2n1 = []
n3n2 = []
for i in range(0,len(dens)):
    Z_g.append(i)
    n2n1.append(i)
    n3n2.append(i)
    Z_g[i] = g13[0]*np.exp(-(E13[0]-E13[0])/(kb*tt[i])) + g13[1]*np.exp(-(E13[1]-E13[0])/(kb*tt[i])) + \
           g13[2]*np.exp(-(E13[2]-E13[0])/(kb*tt[i]))
    n2n1[i] = (g13[1]/g13[0])*np.exp(-(E13[1]-E13[0])/(kb*tt[i]))
    n3n2[i] = (g13[2]/g13[1])*np.exp(-(E13[2]-E13[1])/(kb*tt[i]))


def saha(x,T,Zg,Zi,nHtot,Ne,xI):
    return x - (2*Zi/(Ne*Zg)) * ((2*pi*me*kb*T/(hpl**2))**1.5) * np.exp(-xI*ev2erg/(kb*T))

nHI = []
n2 = []
IonF  = []
nHII = []
n3 = []
jsp = []
Lsp = []
for i in range(0,len(dens)):
    nHI.append(i)
    nHII.append(i)
    n2.append(i)
    n3.append(i)
    IonF.append(i)
    jsp.append(i)
    Lsp.append(i)
    soln = scipy.optimize.toms748(saha,-1e+12,1e+12,args=(tt[i],Z_g[i],Z_i[i],nH[i],nel[i],xI,))
    nHII[i] = soln
    nHI[i] = nH[i]/(1+soln)
    IonF[i] = (nH[i]-nHI[i])/nH[i]
    n2[i] = nH[i] * n2n1[i]/(1.0 + n2n1[i]) * 1.0/(1.0 + ((nH[i]-nHI[i])/nHI[i]))
    n3[i] = n3n2[i] * n2[i]
    jsp[i] = hpl * c * A32 * n3[i]
    Lsp[i] = jsp[i] * dvol[i]

Lsp = np.cumsum(Lsp[::-1])[::-1]

#----------------------------------------------------------------------------------------------------------------------------------------------

#6B. CALCULATE SOBOLEV OPTICAL DEPTH UNDER THE ASSUMPTION OF nLTE FOR HYDROGEN. THE CURRENT IMPLEMENTATION REQUIRES CLOUDY TO BE INSTALLED
#SINCE CLOUDY IS UTILIZED FOR THE NLTE CALCULATION OF THE ATOMIC LEVELS NEEDED FOR DETERMINING THE SOBOLEV OPTICAL DEPTH.
#THERE ARE TWO MODES: ONE ALLOWS THE USER TO ONLY POST-PROCESS THE "WIND" DATA, ABOVE THE EFFECTIVE PHOTOSPHERE, FOR NLTE. THE OTHER MODE
#PROCESSES THE ENTIRE INPUT DATA THROUGH NLTE. A CLOUDY INPUT FILE IS AUTOMATICALLY CREATED, RUN AND THEN THE RESULTS ARE RETURNED FOR
#PROCESSING THROUGH THE REMAINDER OF THE CODE.

if nLTE:
    #First determine inner radius/BB properties of Cloudy simulation depending on nLTE_Atm option
    if nLTE_Atm:
        difff = [(abs(tau_ph - x),idx) for (idx,x) in enumerate(tau)]
        #difff = [(abs(10.0 - x),idx) for (idx,x) in enumerate(tau)]
        difff.sort()
        ind_phot = difff[0][1]
        T_BB = temp[ind_phot]
        L_BB = 4.0*math.pi*(radius[ind_phot]**2)*sigma*T_BB**4
        logLBB = log10(L_BB)
        Inner_Rad = radius[ind_phot]
        Log_IR = log10(Inner_Rad)
    if not nLTE_Atm:
        ind_phot = 0
        T_BB = temp[ind_phot]
        L_BB = 4.0*math.pi*(radius[ind_phot]**2)*sigma*T_BB**4
        logLBB = log10(L_BB)
        Inner_Rad = radius[ind_phot]
        Log_IR = log10(Inner_Rad)

    if nLTE_Atm:
        #First calculate atomic levels with LTE approach as before
        nH = np.multiply(dens,h1)/mH
        nHe = np.multiply(dens,he4)/mHe
        nC = np.multiply(dens,c12)/mC
        nN = np.multiply(dens,n14)/mN
        nO = np.multiply(dens,o16)/mO
        Z_i = np.ones(len(dens))
        E13 = np.asarray([-13.6,-3.4,-1.5])*ev2erg # first three Hydrogen ionization energies
        g13 = np.asarray([2.0,8.0,16.0] )          # first three Hydrogen degeneracy parameters
        Z_g = []
        n2n1 = []
        n3n2 = []
        for i in range(0,len(dens)):
            Z_g.append(i)
            n2n1.append(i)
            n3n2.append(i)
            Z_g[i] = g13[0]*np.exp(-(E13[0]-E13[0])/(kb*temp[i])) + g13[1]*np.exp(-(E13[1]-E13[0])/(kb*temp[i])) + \
            g13[2]*np.exp(-(E13[2]-E13[0])/(kb*temp[i]))
            n2n1[i] = (g13[1]/g13[0])*np.exp(-(E13[1]-E13[0])/(kb*temp[i]))
            n3n2[i] = (g13[2]/g13[1])*np.exp(-(E13[2]-E13[1])/(kb*temp[i]))


        def saha(x,T,Zg,Zi,nHtot,Ne,xI):
            return x - (2*Zi/(Ne*Zg)) * ((2*pi*me*kb*T/(hpl**2))**1.5) * np.exp(-xI*ev2erg/(kb*T))

        nHI = []
        n2 = []
        IonF  = []
        nHII = []
        for i in range(0,len(dens)):
            nHI.append(i)
            nHII.append(i)
            n2.append(i)
            IonF.append(i)
            soln = scipy.optimize.toms748(saha,-1e+12,1e+12,args=(temp[i],Z_g[i],Z_i[i],nH[i],nel[i],xI,))
            nHII[i] = soln
            nHI[i] = nH[i]/(1+soln)
            IonF[i] = (nH[i]-nHI[i])/nH[i]
            n2[i] = nH[i] * n2n1[i]/(1.0 + n2n1[i]) * 1.0/(1.0 + ((nH[i]-nHI[i])/nHI[i]))


        with open('script.in','w') as cloudy_in:
            print("title CSI levels input file", file=cloudy_in)
            print("blackbody",round(T_BB,7),"K", file=cloudy_in)
            print("luminosity total",round(logLBB,7), file=cloudy_in)
            print("radius",round(log10(radius[ind_phot+1]),7), file=cloudy_in)
            print("stop radius",round((log10(radius[len(radius)-1])+log10(radius[len(radius)-2]))/2.0,7), file=cloudy_in)
            print("stop temperature off", file=cloudy_in)
            print("sphere", file=cloudy_in)
            print("dlaw table radius", file=cloudy_in)
            for i in range(ind_phot+1,len(radius)):
                print("continue",round(log10(radius[i]),7), round(log10(nH[i]),7), file=cloudy_in)
            print("end of dlaw", file=cloudy_in)
            print("tlaw table radius", file=cloudy_in)
            for i in range(ind_phot+1,len(radius)):
                print("continue",round(log10(radius[i]),7), round(log10(temp[i]),7), file=cloudy_in)
            print("end of tlaw", file=cloudy_in)
            print("element helium abundance", round(log10(nHe[ind_phot]/nH[ind_phot]),7), file=cloudy_in)
            print("element carbon abundance", round(log10(nC[ind_phot]/nH[ind_phot]),7), file=cloudy_in)
            print("element nitrogen abundance", round(log10(nN[ind_phot]/nH[ind_phot]),7), file=cloudy_in)
            print("element oxygen abundance", round(log10(nO[ind_phot]/nH[ind_phot]),7), file=cloudy_in)
            print('save species densities "species.pop"', file=cloudy_in)
            print('"H"', file=cloudy_in)
            print('"H+"', file=cloudy_in)
            print('"H[1:6]"', file=cloudy_in)
            print('"e-"', file=cloudy_in)
            print("end", file=cloudy_in)
            print('save overview "script.ovr"', file=cloudy_in)
            print('save continuum "spectrum.dat" units angstroms', file=cloudy_in)

        os.system("export CLOUDY_DATA_PATH=/Users/emmanouilchatzopoulos/Desktop/cloudy/c17.02/data")
        os.system("./cloudy.exe -r script")

        rad_cl,n2_2s,n2_2p,n3_3s,n3_3p,n3_3d,nel_cl = np.loadtxt("species.pop",usecols=(0,4,5,6,7,8,9),unpack=True,skiprows=1)
        n2_cl = n2_2s + n2_2p
        n3_cl = n3_3s + n3_3p + n3_3d
        n3n2_cl = n3_cl/n2_cl

        #Cloudy output is now mapped back into original input array
        new_length = ind_phot + len(rad_cl)
        r_new = np.concatenate((radius[0:ind_phot+1],rad_cl + radius[ind_phot+1]))
        r_new = np.append(r_new,radius[len(radius)-1])
        n2_new = np.concatenate((n2[0:ind_phot+1],n2_cl))
        n2_new = np.append(n2_new,n2[len(radius)-1])
        n3n2_new = np.concatenate((n3n2[0:ind_phot+1],n3n2_cl))
        n3n2_new = np.append(n3n2_new,n3n2[len(radius)-1])
        nel_new = np.concatenate((nel[0:ind_phot+1],nel_cl))
        nel_new = np.append(nel_new,nel[len(radius)-1])

        f_n2 = interpolate.interp1d(r_new,n2_new)
        f_n3n2 = interpolate.interp1d(r_new,n3n2_new)
        f_nel = interpolate.interp1d(r_new,nel_new)

        n2_cl_new = f_n2(radius)
        n3n2_cl_new = f_n3n2(radius)
        nel_cl_new = f_nel(radius)

        n2 = n2_cl_new
        n3n2 = n3n2_cl_new
        nel = nel_cl_new

        if plt_spec:
            wavelength,flux = np.loadtxt("spectrum.dat",usecols=(0,4),unpack=True,skiprows=1)
            plt.plot(wavelength,flux,color='k',linewidth=2)
            plt.title("Cloudy-computed optical nLTE spectrum")
            plt.xlabel("Wavelength [A]")
            plt.ylabel("Luminosity")
            plt.xlim([2000.0,10000.0])
            plt.show()

        os.system("rm species.pop")

    if not nLTE_Atm:
        nH = np.multiply(dens,h1)/mH
        nHe = np.multiply(dens,he4)/mHe
        nC = np.multiply(dens,c12)/mC
        nN = np.multiply(dens,n14)/mN
        nO = np.multiply(dens,o16)/mO
        with open('script.in','w') as cloudy_in:

            #First the full data are mapped to a smaller array suitable for CLOUDY post-processing
            ff_nH = interpolate.interp1d(radius,nH)
            ff_temp = interpolate.interp1d(radius,temp)
            ff_nHe = interpolate.interp1d(radius,nHe)
            ff_nC = interpolate.interp1d(radius,nC)
            ff_nN = interpolate.interp1d(radius,nN)
            ff_nO = interpolate.interp1d(radius,nO)

            log_rad_new = np.linspace(round(log10(radius[0]),0),round(log10(radius[len(radius)-1]),0),100)
            rad_new = 10.**log_rad_new

            nH_red = ff_nH(rad_new)
            temp_red = ff_temp(rad_new)
            he_red = ff_nHe(rad_new)
            C_red = ff_nC(rad_new)
            N_red = ff_nN(rad_new)
            O_red = ff_nO(rad_new)

            print("title CSI levels input file", file=cloudy_in)
            print("blackbody",round(T_BB,7),"K", file=cloudy_in)
            print("luminosity total",round(logLBB,7), file=cloudy_in)
            print("radius",round(log10(rad_new[ind_phot+1]) + 0.00001,7), file=cloudy_in)
            print("stop radius",round((log10(rad_new[len(rad_new)-1])+log10(rad_new[len(rad_new)-2]))/2.0,7), file=cloudy_in)
            print("stop temperature off", file=cloudy_in)
            print("set nend",50000, file=cloudy_in)
            print("stop zone",50000, file=cloudy_in)
            print("set save line width",10, file=cloudy_in)
            print("sphere", file=cloudy_in)
            print("dlaw table depth", file=cloudy_in)
            print("continue",-35,round(log10(nH_red[0]),7), file=cloudy_in)
            for i in range(ind_phot+1,len(rad_new)):
                print("continue",round(log10(rad_new[i]),7), round(log10(nH_red[i]),7), file=cloudy_in)
            print("end of dlaw", file=cloudy_in)
            print("tlaw table depth", file=cloudy_in)
            print("continue",-35,round(log10(temp_red[0]),7), file=cloudy_in)
            for i in range(ind_phot+1,len(rad_new)):
                print("continue",round(log10(rad_new[i]),7), round(log10(temp_red[i]),7), file=cloudy_in)
            print("end of tlaw", file=cloudy_in)
            print("element helium table depth", file=cloudy_in)
            print("continue",-35, round(log10(he_red[0]/nH_red[0]),7), file=cloudy_in)
            for i in range(ind_phot+1,len(rad_new)):
                print("continue",round(log10(rad_new[i]),7), round(log10(he_red[i]/nH_red[i]),7), file=cloudy_in)
            print("end of table", file=cloudy_in)
            print("element carbon table depth", file=cloudy_in)
            print("continue",-35, round(log10(C_red[0]/nH_red[0]),7), file=cloudy_in)
            for i in range(ind_phot+1,len(rad_new)):
                print("continue",round(log10(rad_new[i]),7), round(log10(C_red[i]/nH_red[i]),7), file=cloudy_in)
            print("end of table", file=cloudy_in)
            print("element nitrogen table depth", file=cloudy_in)
            print("continue",-35, round(log10(N_red[0]/nH_red[0]),7), file=cloudy_in)
            for i in range(ind_phot+1,len(rad_new)):
                print("continue",round(log10(rad_new[i]),7), round(log10(N_red[i]/nH_red[i]),7), file=cloudy_in)
            print("end of table", file=cloudy_in)
            print("element oxygen table depth", file=cloudy_in)
            print("continue",-35, round(log10(O_red[0]/nH_red[0]),7), file=cloudy_in)
            for i in range(ind_phot+1,len(rad_new)):
                print("continue",round(log10(rad_new[i]),7), round(log10(O_red[i]/nH_red[i]),7), file=cloudy_in)
            print("end of table", file=cloudy_in)
            print('save species densities "species.pop"', file=cloudy_in)
            print('"H"', file=cloudy_in)
            print('"H+"', file=cloudy_in)
            print('"H[1:6]"', file=cloudy_in)
            print('"e-"', file=cloudy_in)
            print("end", file=cloudy_in)
            print('save overview "script.ovr"', file=cloudy_in)
            print('save continuum "spectrum.dat" units angstroms', file=cloudy_in)

        os.system("export CLOUDY_DATA_PATH=/Users/emmanouilchatzopoulos/Desktop/cloudy/c17.02/data")
        os.system("./cloudy.exe -r script")

        rad_cl,n2_2s,n2_2p,n3_3s,n3_3p,n3_3d,nel_cl = np.loadtxt("species.pop",usecols=(0,4,5,6,7,8,9),unpack=True,skiprows=1)
        n2_cl = n2_2s + n2_2p
        n3_cl = n3_3s + n3_3p + n3_3d
        n3n2_cl = n3_cl/n2_cl

        #Cloudy output is now mapped back into original input array

        rad_cl = np.append(rad_cl,radius[len(radius)-1])
        n2_cl = np.append(n2_cl,n2_cl[len(n2_cl)-1])
        n3_cl = np.append(n3_cl,n3_cl[len(n3_cl)-1])
        n3n2_cl = np.append(n3n2_cl,n3n2_cl[len(n3n2_cl)-1])
        nel_cl = np.append(nel_cl,nel_cl[len(nel_cl)-1])
        f_n2 = interpolate.interp1d(rad_cl,n2_cl)
        f_n3 = interpolate.interp1d(rad_cl,n3_cl)
        f_n3n2 = interpolate.interp1d(rad_cl,n3n2_cl)
        f_nel = interpolate.interp1d(rad_cl,nel_cl)
        n2_cl_new = f_n2(radius)
        n3_cl_new = f_n3(radius)
        n3n2_cl_new = f_n3n2(radius)
        nel_cl_new = f_nel(radius)

        n2 = n2_cl_new
        n3n2 = n3n2_cl_new
        nel = nel_cl_new

        if plt_spec:
            wavelength,flux = np.loadtxt("spectrum.dat",usecols=(0,4),unpack=True,skiprows=1)
            plt.plot(wavelength,flux,color='k',linewidth=2)
            plt.title("Cloudy-computed optical nLTE spectrum")
            plt.xlabel("Wavelength [A]")
            plt.ylabel("Luminosity")
            plt.xlim([2000.0,10000.0])
            plt.show()

        os.system("rm species.pop")

#----------------------------------------------------------------------------------------------------------------------------------------------

#7. LOCATE THE INDEX CORRESPONDING TO THE RADIUS OF THE PHOTOSPHERE
diff2 = [(abs(tau_ph - x),idx) for (idx,x) in enumerate(tau)]
diff2.sort()
ind_p = diff2[0][1]

#----------------------------------------------------------------------------------------------------------------------------------------------

#8. CALCULATE VELOCITY, RADIUS AT PHOTOSPHERE AND NORMALIZED VELOCITY, RADIUS ARRAYS WRT PHOTOSPHERIC VALUES
vel_p = velx[ind_p]
rad_p = radius[ind_p]
phot_lum = lumin[ind_p]
phot_lum2 = 4.0*math.pi*((rad_p)**2)*sigma*temp[ind_p]**4
vel2 = velx/vel_p
r2 = radius/rad_p

#----------------------------------------------------------------------------------------------------------------------------------------------
#9. DEFINE OPTICAL DEPTH FUNCTION (BY READING IN DATA OR BY CALCULATED SOBOLEV OPTICAL DEPTH)
#NOTE: A power-law tau yields the regular P-Cygni profile shape expected.
def tau_s(p,z):
    rr = np.sqrt(p**2+z**2)
    if rr <= max(r2):
        diff3 = [(abs(rr - x),idx) for (idx,x) in enumerate(r2)]
        diff3.sort()
        ind_r = diff3[0][1]
    elif rr > max(r2):
        ind_r = len(r2)-1
    return tau[ind_r],ind_r

def tau_sob(p,z):
    rr = np.sqrt(p**2+z**2)
    if rr <= max(r2):
        diff3 = [(abs(rr - x),idx) for (idx,x) in enumerate(r2)]
        diff3.sort()
        ind_r = diff3[0][1]
    elif rr > max(r2):
        ind_r = len(r2)-1

    kap = (pi*ee**2/(me*c))*f0*n2[ind_r]*(1.0 - n3n2[ind_r]*g13[1]/g13[2])*(l0**2/c)
    #The velocity gradient needs to account for negative values and avoid cases where tau_sob -> infinity (Dessart et al. 2015)
    if dvdr[ind_r] >= velx[ind_r]/radius[ind_r]:
        v_grad = 1.0/(((z/rr)**2) * dvdr[ind_r] + (1.0 - ((z/rr)**2)) * velx[ind_r]/radius[ind_r])
    if dvdr[ind_r] < velx[ind_r]/radius[ind_r]:
        v_grad = 1.0/max((((z/rr)**2) * dvdr[ind_r] + (1.0 - ((z/rr)**2)) * velx[ind_r]/radius[ind_r]), -0.2*dvdr[ind_r])

    return kap * v_grad,ind_r

#10. DEFINE SOURCE FUNCTION
def Source(p,z):
    rad = np.sqrt(p**2+z**2)
    if rad > rph:
        return 0.5 * Iph * (1.0 - np.sqrt(1.0 - (rph/rad)**2.0))
    else:
        return Iph

#----------------------------------------------------------------------------------------------------------------------------------------------

#11. DEFINE RESONANCE CONDITION FUNCTION
def resSurf(z,p,x):
    rr = np.sqrt(p**2+z**2)
    if rr <= max(r2):
        diff4 = [(abs(rr - x),idx) for (idx,x) in enumerate(r2)]
        diff4.sort()
        ind_r = diff4[0][1]
        v_r = vel2[ind_r]
    elif rr > max(r2):
        ind_r = len(r2)-1
        v_r = vel2[ind_r]
    #v_r = vel2[ind_r]
    return x - (z/np.sqrt(z**2+p**2))*v_r

#----------------------------------------------------------------------------------------------------------------------------------------------

#12. DEFINE P-SPACE AND DETERMINE INDICES OF IMPACT PARAMETER INTEGRATION LIMITS CORRESPONDING TO P = 0,1 AND P CORRESPONDING TO CSI SHOCK
p_space = np.linspace(pmin,pmax,p_bins)
diff5 = [(abs(1.0 - x),idx) for (idx,x) in enumerate(p_space)]
diff5.sort()
ind_p1 = diff5[0][1]
diff6 = [(abs(0.0 - x),idx) for (idx,x) in enumerate(p_space)]
diff6.sort()
ind_p0 = diff6[0][1]

diff7 = [(abs(Rcs - x),idx) for (idx,x) in enumerate(p_space)]
diff7.sort()
ind_pCSI = diff7[0][1]

#----------------------------------------------------------------------------------------------------------------------------------------------

#13. DEFINE X-SPACE (FREQUENCY)
x_space = np.linspace(xmin,xmax,x_bins)

#----------------------------------------------------------------------------------------------------------------------------------------------

#14. FIND RESONANCE POINTS FOR ALL VALUES OF X
z0_soln = []
for i in range(0,len(x_space)):
    z0_soln.append([])
    for j in range(0,len(p_space)):
        z0_soln[i].append(i*j)
        try:
            z0_soln[i][j] = scipy.optimize.toms748(resSurf,-100,100,args=(p_space[j],x_space[i],))
        except Exception as e:
            #print('Resonance Surface Solver Cannot Find Root')
            #print(sys.stderr, "Exception: %s" % str(e))
            #sys.exit(1)
            z0_soln[i][j] = 10000.0 # x_space[i] ## 0.0

            #if x_space[i] < 0:
            #    z0_soln[i][j] = x_space[i] # 1000.0
            #if x_space[i] >= 0:
            #    z0_soln[i][j] = 1000000.0

                #if p_space[j] >= 1.0 or p_space[j] <= -1.0:
                #    z0_soln[i][j] = x_space[i]
                #if p_space[j] < 1.0 and p_space[j] > -1.0:
                #    z0_soln[i][j] = 1000.0


#----------------------------------------------------------------------------------------------------------------------------------------------

#15. PLOT RESONANCE SURFACES
"""
for i in range(0,len(x_space)):
    plt.plot(z0_soln[i],p_space,linestyle='None',marker='o',markersize=3,label='x = '+str(round(x_space[i],1)))
plt.title("Interaction Surfaces")
plt.xlabel("z")
plt.ylabel("p")
plt.xlim([p_space[0],p_space[len(p_space)-1]])
plt.show()
MANOS
"""

#----------------------------------------------------------------------------------------------------------------------------------------------

#16. CALCULATE INTENSITY INTEGRANDS NEEDED AT THE LOCATIONS OF ALL RESONANCE POINTS
zeta = []
S = []
SnT = []
integrand1 = []
integrand1b = []
integrand2 = []
for i in range(0,len(x_space)):
    zeta.append([])
    S.append([])
    SnT.append([])
    integrand1.append([])
    integrand1b.append([])
    integrand2.append([])
    for j in range(0,len(p_space)):
        zeta[i].append(i*j)
        S[i].append(i*j)
        SnT[i].append(i*j)
        integrand1[i].append(i*j)
        integrand1b[i].append(i*j)
        integrand2[i].append(i*j)

        tau2 = tau_s(p_space[j],z0_soln[i][j])[0]

        if sobolev_tau:
            tau2 = tau_sob(p_space[j],z0_soln[i][j])[0]

        zeta[i][j] = np.exp(-tau2)
        S[i][j] = Source(p_space[j],z0_soln[i][j])

        integrand1[i][j] = S[i][j] * (1.0 - zeta[i][j]) * p_space[j]
        integrand2[i][j] = zeta[i][j] * p_space[j]

#----------------------------------------------------------------------------------------------------------------------------------------------

#17. PERFORM TRAPEZOIDAL INTEGRATION AND CALCULATE NORMALIZED FLUX ALONG LINE PROFILE
F = []
for i in range(0,len(x_space)):
    F.append(i)
    sum1 = 0.0
    sum1b = 0.0
    sum2 = 0.0
    sum3 = 0.0
    if x_space[i] >= 0:
        for j in range(ind_p1,len(p_space)-1):
            sum1 = sum1 + (integrand1[i][j]+integrand1[i][j+1])*(p_space[j+1]-p_space[j]) / 2.0

        F[i] = (1.0 + 2.0 * sum1/Iph)

    if x_space[i] < 0:
        for j in range(ind_p0,ind_p1):
            sum2 = sum2 + (integrand1[i][j]+integrand1[i][j+1])*(p_space[j+1]-p_space[j]) / 2.0
            sum3 = sum3 + (integrand2[i][j]+integrand2[i][j+1])*(p_space[j+1]-p_space[j]) / 2.0

        for j in range(ind_p1,len(p_space)-1):
            sum1b = sum1b + (integrand1[i][j]+integrand1[i][j+1])*(p_space[j+1]-p_space[j]) / 2.0

        F[i] = 2.0 * (sum2/Iph + sum3 + sum1b/Iph)


if TopLighting:

    if not TL_Auto_Mode:
        muting = (1.0 - Ics/Iph)/(1.0 - Ics/Iph + 2.0*(Ics/Iph)*(Rcs/rph)**2)

    if TL_Auto_Mode:
        #First find location of CSIR by looking for maximum density gradient
        dRho_dR = abs(diff(dens)/diff(radius))
        dRho_dR = np.append(0.0, dRho_dR)
        dRho_dR = dRho_dR[ind_p:len(dRho_dR)]
        csir_ind = ind_p + np.argmax(dRho_dR) - 1
        csir_rad = radius[csir_ind]
        csir_lum = lumin[csir_ind]
        csir_temp = temp[csir_ind]
        csir_lum2 = 4.0*math.pi*(csir_rad**2)*sigma*csir_temp**4
        #mute_gamma = csir_lum/phot_lum
        mute_gamma = csir_lum2/phot_lum2
        muting = (2.0*(csir_rad/rad_p)**2 - mute_gamma)/(2.0*(csir_rad/rad_p)**2 - mute_gamma + 2.0*((csir_rad/rad_p)**2)*mute_gamma)

        plt.loglog(radius,dens,linewidth=3,color='r')
        plt.axvline(x=rad_p,linestyle=":",linewidth=2,color='k',label="Rphot")
        plt.axvline(x=csir_rad,linestyle=":",linewidth=2,color='b',label="Rcsir")
        plt.xlabel("Radius [cm]")
        plt.ylabel("Density [g/cm^3]")
        plt.legend()
        plt.show()

        plt.loglog(radius,temp,linewidth=3,color='r')
        plt.axvline(x=rad_p,linestyle=":",linewidth=2,color='k',label="Rphot")
        plt.axvline(x=csir_rad,linestyle=":",linewidth=2,color='b',label="Rcsir")
        plt.xlabel("Radius [cm]")
        plt.ylabel("Temperature [K]")
        plt.legend()
        plt.show()

        plt.loglog(radius,np.asarray(velx)/kms2cms,linewidth=3,color='r')
        plt.axvline(x=rad_p,linestyle=":",linewidth=2,color='k',label="Rphot")
        plt.axvline(x=csir_rad,linestyle=":",linewidth=2,color='b',label="Rcsir")
        plt.xlabel("Radius [cm]")
        plt.ylabel("Velocity [km/s]")
        plt.legend()
        plt.show()

    for i in range(0,len(x_space)):
        F[i] = (F[i] - F[len(F)-1]) * muting + F[len(F) - 1]

#----------------------------------------------------------------------------------------------------------------------------------------------

#18. PLOT FINAL LINE PROFILE IN DIMENSIONLESS FREQUENCY (X) SPACE AND SAVE OUTPUT TO ASCII FILE
#Lambda-array
lambda_arr = l0*(1.0 + (x_space)*(vel_p/c))
vel_arr = c * (lambda_arr - l0)/l0

"""
plt.plot(x_space,F,linestyle='-',)
plt.title("Line Profile: x-space")
plt.xlabel("x")
plt.ylabel("F")
plt.xlim([x_space[0],x_space[len(x_space)-1]])
#plt.ylim([p_space[0],p_space[len(p_space)-1]])
plt.show()

plt.plot(lambda_arr,F,linestyle='-',)
plt.title("Line Profile: Wavelength-space")
plt.xlabel("Wavelength [A]")
plt.ylabel("F")
plt.xlim([lambda_arr[0],lambda_arr[len(x_space)-1]])
#plt.ylim([p_space[0],p_space[len(p_space)-1]])
plt.show()

plt.plot(vel_arr/kms2cms,F,linestyle='-',)
plt.title("Line Profile: Velocity-space")
plt.xlabel("velocity [km/s]")
plt.ylabel("F")
#plt.xlim([vel_arr[0],lambda_arr[len(x_space)-1]])
#plt.ylim([p_space[0],p_space[len(p_space)-1]])
plt.show()
MANOS
"""

with open(fname,'w') as f1:
    for i in range(0,len(F)):
        print('{0:5f} {1:5f} {2:5f} {3:10f}'.format(x_space[i],lambda_arr[i],vel_arr[i]/1e+5,F[i]), file=f1)


#DEBUGGING: OUTPUT OF PROCESSED DATA FILES-------------------------------------------------------------------------------------------------~
if debug:

    #Calculate electron-scattering optical depth
    sigmaT = 6.6524e-25
    nel_inv = nel[::-1]
    dr_inv = dr[::-1]

    tau_e = []
    sum_tau_e = []
    for i in range(0,len(dr_inv)):
        tau_e.append(i)
        sum_tau_e.append(i)
        if i == 0:
            tau_e[i] = sigmaT * nel_inv[i] * dr_inv[i]
            sum_tau_e[i] = tau_e[i]

        if i != 0:
            tau_e[i] = sigmaT * nel_inv[i] * dr_inv[i]
            sum_tau_e[i] = tau_e[i] + sum_tau_e[i-1]

    tau_e = tau_e[::-1]
    sum_tau_e = sum_tau_e[::-1]

    def tau_sob2(p,z):
        rr = np.sqrt(p**2+z**2)
        if rr <= max(r2):
            diff3 = [(abs(rr - x),idx) for (idx,x) in enumerate(r2)]
            diff3.sort()
            ind_r = diff3[0][1]
        elif rr > max(r2):
            ind_r = len(r2)-1

        kap = (pi*ee**2/(me*c))*f0*n2[ind_r]*(1.0 - n3n2[ind_r]*g13[1]/g13[2])*(l0**2/c)
        #The velocity gradient needs to account for negative values and avoid cases where tau_sob -> infinity (Dessart et al. 2015)
        if dvdr[ind_r] >= velx[ind_r]/radius[ind_r]:
            v_grad = 1.0/(((z/rr)**2) * dvdr[ind_r] + (1.0 - ((z/rr)**2)) * velx[ind_r]/radius[ind_r])
        if dvdr[ind_r] < velx[ind_r]/radius[ind_r]:
            v_grad = 1.0/max((((z/rr)**2) * dvdr[ind_r] + (1.0 - ((z/rr)**2)) * velx[ind_r]/radius[ind_r]), -0.2*dvdr[ind_r])

        return kap * v_grad,ind_r,kap,v_grad,dvdr[ind_r],velx[ind_r]/radius[ind_r],rr,max(r2)

    def Source2(rr):
        rad = rr/rad_p
        if rad > rph:
            return 0.5 * Iph * (1.0 - np.sqrt(1.0 - (rph/rad)**2.0))
        else:
            return Iph


    DeltaVD = []
    kappa_line = []
    kappa_scatt = []
    abs_frac = []
    tau_therm = []
    e_ff = []
    L_ff = []
    L_ff_full = []
    L_sh = []
    L_Ha = []
    cs = []
    for i in range(0,len(radius)):
        kappa_line.append(i)
        kappa_scatt.append(i)
        DeltaVD.append(i)
        abs_frac.append(i)
        tau_therm.append(i)
        e_ff.append(i)
        L_ff.append(i)
        L_ff_full.append(i)
        L_sh.append(i)
        L_Ha.append(i)
        cs.append(i)
        DeltaVD[i] = (1.0/(l0*1.0e-8)) * np.sqrt((2.0*kb*temp[i])/mH)
        kappa_line[i] = (pi*ee**2/(me*c))*(1.0/np.sqrt(pi))*f0*n2[i]*(1.0 - (gl/gu) * n3n2[i]) * 1.0/DeltaVD[i]
        kappa_scatt[i] = nel[i] * sigmaT
        abs_frac[i] = kappa_line[i]/(kappa_line[i] + kappa_scatt[i])
        tau_therm[i] = 1.0/np.sqrt(abs_frac[i])
        e_ff[i] = 1.4e-27*np.sqrt(temp[i])*nel[i]*nHII[i]*1.2
        L_ff[i] = e_ff[i]*4.0*math.pi*dr[i]*radius[i]**2
        L_ff_full[i] = e_ff[i]*(4.0/3.0)*math.pi*radius[i]**3
        L_sh[i] = 2.0*math.pi*rho[i]*(radius[i]**2)*velx[i]**3
        cs[i] = np.sqrt(1.6667*prad[i]/rho[i])
        #L_Ha[i] = hpl*(c/(l0*(1e-8)))*A32*n3n2[i]*n2[i]*(radius[i]**2)*dr[i]
        L_Ha[i] = hpl*(c/(l0*(1e-8)))*A32*n3n2[i]*n2[i]*dvol[i]
        #print(radius[i],tau[i],cs[i],velx[i],dvdr[i],prad[i],rho[i],drho[i])



    tau_therm = np.cumsum(tau_therm[::-1])[::-1]
    L_ff_sum = np.cumsum(L_ff)
    L_Ha_sum = np.cumsum(L_Ha)
    L_stella_sum = np.cumsum(lumin)

    max_vel_ind = np.argmax(velx)
    n3n2_out = n3n2[max_vel_ind:len(n3n2)]
    max_n3n2_ind = np.argmax(n3n2_out)
    max_n3n2_ind_full = max_vel_ind + max_n3n2_ind
    radiussquare = np.multiply(radius,radius)
    n3n2radiussquare = np.multiply(n3n2,radiussquare)
    maxn3n2radsquare_ind = np.argmax(n3n2radiussquare)
    min_cs_ind = np.argmin(cs)
    cs_out = cs[min_cs_ind:len(cs)]
    cs_max = np.argmax(cs_out)
    min_cs_ind_full = min_cs_ind + cs_max

    max_kappaline_ind = np.argmax(kappa_line)
    min_kappaline_ind = np.argmin(kappa_line)

    difff0 = [(abs(tau_ph - x),idx) for (idx,x) in enumerate(tau)]
    difff0.sort()
    ind_phot0 = difff0[0][1]

    mult = 2.0
    DVD = (1.0/(l0*1.0e-8)) * np.sqrt((2.0*kb*temp[ind_phot0])/mH)
    dl = mult * (c/DVD)
    Bl = (2.0*hpl*(c**2)/(l0*1e-8)**5)/(math.exp(hpl*c/(kb*temp[ind_phot0]*(l0*1.0e-8))) - 1.0)
    L_BB_dl = Bl*dl*4.0*math.pi*(2.0*math.pi*(radius[ind_phot0]**2))

    #####
    ind_peak_vel = np.argmax(velx)
    ind_rs = np.argmax(dens[ind_peak_vel:len(dens)])
    ind_rs_full = ind_rs + len(dens[0:ind_peak_vel])
    r_rs = radius[ind_rs_full]
    ind_fs = np.argmin(drho)
    r_fs = radius[ind_fs]

    R_csir = np.sum(np.multiply(L_Ha[ind_rs_full:ind_fs],radius[ind_rs_full:ind_fs]))/np.sum(L_Ha[ind_rs_full:ind_fs])
    dif2 = [(abs(R_csir - x),idx) for (idx,x) in enumerate(radius)]
    dif2.sort()
    ind_csir_med = dif2[0][1]
    L_csir = lumin[ind_csir_med]
    R_csir2 = r_fs
    dif3 = [(abs(R_csir2 - x),idx) for (idx,x) in enumerate(radius)]
    dif3.sort()
    ind_csir_med2 = dif3[0][1]
    L_csir2 = lumin[ind_csir_med2]

    L_csir_cumulative = np.sum(L_Ha[ind_rs_full:ind_fs])
    #####

    lambda_delta =  2.0 * l0 * velx[ind_phot0] / c
    low_lambda = l0 - lambda_delta
    high_lambda = l0 + lambda_delta
    resolution = 1000000

    def bb_func(T,lam):
        hpl = 6.6260755e-27
        kb = 1.380658e-16
        c = 2.99792458e10
        return 2.0 * hpl * c**2 / (lam*1.0e-8)**5 /(np.exp(hpl*c/((lam*1.0e-8)*kb*T)) - 1.0)


    lambda_vals = np.linspace(50,100000,resolution)
    dl = ((100000-50)/resolution) * 1.0e-8
    bb_vals = bb_func(temp[ind_phot0],lambda_vals)

    diff0 = [(abs(low_lambda - x),idx) for (idx,x) in enumerate(lambda_vals)]
    diff0.sort()
    ind_low = diff0[0][1]
    diff1 = [(abs(high_lambda - x),idx) for (idx,x) in enumerate(lambda_vals)]
    diff1.sort()
    ind_high = diff1[0][1]

    Lum_specific = bb_func(temp[ind_phot0],l0) * 4.0 * math.pi * (radius[ind_phot0]**2) * 5.0*1e-8

    #####
    print("------------------------------------------------------------------------")
    print("PHASE:",str(sys.argv[1]).split(".")[1][0:6])
    print("INNER RADIUS/TAU/TAU_THOMPSON [USE!]:",radius[max_kappaline_ind],tau[max_kappaline_ind],sum_tau_e[max_kappaline_ind])
    print("TAU = 2/3 RADIUS:",radius[ind_phot0])
    print("REVERSE SHOCK RADIUS/TAU:",r_rs,tau[ind_rs_full])
    print("FORWARD SHOCK RADIUS/TAU:",r_fs,tau[ind_fs])
    print("CSIR AVERAGE RADIUS/TAU [USE!]:",R_csir,tau[ind_csir_med])
    print("INNER LUMUNOSITY:",lumin[max_kappaline_ind])
    print("CSIR LUMINOSITY (LUMIN AT R_CSIR):",L_csir)
    print("CSIR HALPHA LUMINOSITY:",L_Ha[ind_csir_med])
    print("PHOTOSPHERE LUMINOSITY:",lumin[ind_phot0])
    print("BACKGROUND TO LINE RATIO [USE!]:",round(lumin[ind_phot0]/L_Ha[ind_csir_med],5))

    #Muting parameter
    R_mute = R_csir/radius[max_kappaline_ind]
    R_mute2 = R_csir/radius[ind_phot0]
    R_mute_b = R_csir2/radius[max_kappaline_ind]
    R_mute2_b = R_csir2/radius[ind_phot0]

    gamma_mute = L_csir/lumin[max_kappaline_ind] #instead of photo
    gamma_mute2 = L_csir/lumin[ind_phot0]
    gamma_mute_b = L_csir2/lumin[max_kappaline_ind] #instead of photo
    gamma_mute2_b = L_csir2/lumin[ind_phot0]

    if R_csir < radius[ind_phot0]:
        print("CSIR RADIUS IS INTERNAL TO TAU = 2/3 PHOTOSPHERE")

    if R_csir2 < radius[ind_phot0]:
        print("CSIR (FS) RADIUS IS INTERNAL TO TAU = 2/3 PHOTOSPHERE")

    if R_csir > radius[max_kappaline_ind]:
        muting2 = (2.0*R_mute**2-gamma_mute)/(2.0*R_mute**2-gamma_mute+2.0*(R_mute**2)*gamma_mute)
        muting2b = (2.0*R_mute2**2-gamma_mute2)/(2.0*R_mute2**2-gamma_mute2+2.0*(R_mute2**2)*gamma_mute2)
        print("LINE MUTING PARAMETER (R_CSIR wrt R_INNER) [USE!]:",muting2)
        print("LINE MUTING PARAMETER (R_CSIR wrt R_PHOT):",muting2b)
    else:
        muting2 = 0.0
        muting2b = 0.0
        print("NO TOP LIGHTING APPLICABLE")
        print("LINE MUTING PARAMETER (wrt R_INNER) [USE!]:",muting2)

    ###
    if R_csir2 > radius[max_kappaline_ind]:
        muting22 = (2.0*R_mute_b**2-gamma_mute_b)/(2.0*R_mute_b**2-gamma_mute_b+2.0*(R_mute_b**2)*gamma_mute_b)
        muting22b = (2.0*R_mute2_b**2-gamma_mute2_b)/(2.0*R_mute2_b**2-gamma_mute2_b+2.0*(R_mute2_b**2)*gamma_mute2_b)
        print("LINE MUTING PARAMETER (FS wrt R_INNER):",muting22)
        print("LINE MUTING PARAMETER (FS wrt R_PHOT):",muting22b)
    else:
        muting22 = 0.0
        muting22b = 0.0
        print("NO TOP LIGHTING APPLICABLE")
        print("LINE MUTING PARAMETER (wrt R_INNER):",muting22)

    plt.loglog(radius,dens,linewidth=3,color='k')
    plt.axvline(x=radius[max_kappaline_ind],linestyle="-",linewidth=2,color='k',label="R_inner")
    plt.axvline(x=radius[ind_phot0],linestyle="-",linewidth=2,color='r',label="R_phot")
    plt.axvline(x=R_csir,linestyle="-",linewidth=2,color='b',label="R_csir")
    plt.axvline(x=r_rs,linestyle=":",linewidth=1,color='g',label="R_rs")
    plt.axvline(x=r_fs,linestyle=":",linewidth=1,color='m',label="R_fs")
    #plt.ylim([radius[max_kappaline_ind],radius[len(radius)-1]])
    plt.xlabel("Radius [cm]")
    plt.ylabel("Density [g/cm^3]")
    plt.legend()
    #plt.show()
    plt.savefig("radii_locations_"+str(sys.argv[1]).split(".")[1][0:6]+".png")

    """
    plt.loglog(radius,tau,linewidth=3,color='k')
    plt.axvline(x=radius[max_kappaline_ind],linestyle="-",linewidth=2,color='k',label="R_inner")
    plt.axvline(x=radius[ind_phot0],linestyle="-",linewidth=2,color='r',label="R_phot")
    plt.axvline(x=R_csir,linestyle="-",linewidth=2,color='b',label="R_csir")
    plt.axvline(x=r_rs,linestyle=":",linewidth=1,color='g',label="R_rs")
    plt.axvline(x=r_fs,linestyle=":",linewidth=1,color='m',label="R_fs")
    #plt.ylim([radius[max_kappaline_ind],radius[len(radius)-1]])
    plt.xlabel("Radius [cm]")
    plt.ylabel("Optical Depth")
    plt.legend()
    plt.show()
    """

    with open(db_f1,'w') as f0a:
        for i in range(0,len(radius)):
            print(radius[i],rho[i],temp[i],velx[i],tau[i],sum_tau_e[i],h1[i],nel[i],nH[i],nHI[i],nHII[i],n3n2[i],n2[i],dvdr[i],ro[i],kappa_line[i],kappa_scatt[i],tau_therm[i],Source2(radius[i]),ro[i],jsp[i],Lsp[i],e_ff[i],L_ff[i],L_ff_sum[i],L_ff_full[i],L_sh[i],cs[i], file=f0a)

    with open(db_f2,'w') as f0b:
        for i in range(0,len(x_space)):
            for j in range(0,len(p_space)):
                rr1 = np.sqrt(x_space[i]**2+p_space[j]**2)
                rr0a = np.sqrt(z0_soln[i][j]**2+p_space[j]**2)
                rr0 = rr1 * rad_p
                tausb = tau_sob2(p_space[j],z0_soln[i][j])[0]
                ind_used = tau_sob2(p_space[j],z0_soln[i][j])[1]
                kap = tau_sob2(p_space[j],z0_soln[i][j])[2]
                vgrad = tau_sob2(p_space[j],z0_soln[i][j])[3]
                dvdrind = tau_sob2(p_space[j],z0_soln[i][j])[4]
                veldvr = tau_sob2(p_space[j],z0_soln[i][j])[5]
                rr_in = tau_sob2(p_space[j],z0_soln[i][j])[6]
                maxr2 = tau_sob2(p_space[j],z0_soln[i][j])[7]
                print(rr0,rr0a,rr_in,maxr2,ind_used,x_space[i],p_space[j],z0_soln[i][j],tausb,kap,vgrad,dvdrind,veldvr, file=f0b)

#----------------------------------------------------------------------------------------------------------------------------------------------

if input_mode == 'stella':

    if plot_dbg:

        nden = dens/mH
        pgas = kb*np.multiply(nden,temp)
        Pratio = pgas/prad
        radius15 = radius/1.0e+15
        opacit2 = np.multiply(opacit,dens)

        tsobo = []
        for i in range(0,len(radius15)):
            tsobo.append(i)
            Fact1 = (pi*ee**2/(me*c))*f0*n2[i]*(1.0 - n3n2[i]*g13[1]/g13[2])*(l0**2/c)
            if dvdr[i] >= velx[i]/radius[i]:
                Fact2 = 1.0/dvdr[i]
            if dvdr[i] < velx[i]/radius[i]:
                Fact2 = 1.0/max((dvdr[i]), -0.2*dvdr[i])
            tsobo[i] = Fact1 * Fact2

        fig, axs = plt.subplots(4, 2, figsize=(8,10))

        axs[0, 0].semilogy(radius15,temp)
        axs[0, 0].set_ylabel("Temperature [K]")

        axs[0, 1].semilogy(radius15, Pratio)
        axs[0, 1].set_ylabel("Pgas/Prad")

        axs[1, 0].plot(radius15, velx)
        axs[1, 0].set_ylabel("Velocity [km/s]")

        axs[1, 1].semilogy(radius15, lumin)
        axs[1, 1].set_ylabel("Luminosity [erg/s]")

        axs[2, 0].semilogy(radius15, dens)
        axs[2, 0].set_ylabel("Density [g/cm3]")

        axs[2, 1].semilogy(radius15, nel)
        axs[2, 1].set_ylabel("n_e [cm^-3]")

        axs[3, 0].semilogy(radius15, opacit2)
        axs[3, 0].set_ylabel("Opacity [cm^-1]")
        axs[3, 0].set_xlabel("Radius [10^15 cm]")

        axs[3, 1].semilogy(radius15, tau, label='Gray')
        axs[3, 1].semilogy(radius15, tsobo, label='Sobolev')
        axs[3, 1].axhline(y=1.0, color='k', linestyle=':')
        axs[3, 1].set_ylabel("tau")
        axs[3, 1].set_xlabel("Radius [10^15 cm]")

        plt.legend()
        plt.show()
