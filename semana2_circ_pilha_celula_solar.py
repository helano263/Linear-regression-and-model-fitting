#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 20:13:13 2020

@author: helano
"""

import pandas as pd                 #import spreadsheet data
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


#######################################
###### Data
#######################################
# Data is imported from spreadsheet file using pandas, them converted to np array

dt = pd.read_excel('semana2_circ_pilha_celula_solar.xlsx', sheet_name='pilha')


R, sig_R = 4.755, 0.003
#R, sig_R = 5.1, 0.1
x = np.linspace(0, 6, 200)
t = np.linspace(0, 200,1000)

i_pilha = dt['i'].to_numpy()   # convertendo para mili Amper
V_pilha = dt['Vpilha'].to_numpy()
sig_V_pilha = dt['sig y'].to_numpy()
sig_i_pilha = np.sqrt( (sig_V_pilha/R)**2 + (V_pilha*sig_R/R/R)**2 )   # convertendo para mili Amper

dt = pd.read_excel('semana2_circ_pilha_celula_solar.xlsx', sheet_name='celula')
pilha = pd.read_excel('semana2_circ_pilha_celula_solar.xlsx', sheet_name='pilha')

i_pilha = pilha['i'].to_numpy()*1000   # convertendo para mili Amper
V_pilha = pilha['Vpilha'].to_numpy()
sig_V_pilha = pilha['sig y'].to_numpy()
sig_i_pilha = np.sqrt( (sig_V_pilha/R)**2 + (V_pilha*sig_R/R/R)**2 )*1000   # convertendo para mili Amper

i_celula = dt['i'].to_numpy()*1000   # convertendo para mili Amper
V_celula = dt['Vcelula'].to_numpy()
sig_V_celula = dt['sig y'].to_numpy()
sig_i_celula = np.sqrt( (sig_V_celula/R)**2 + (V_celula*sig_R/R/R)**2 )*1000   # convertendo para mili Amper

# %%
#######################################
###### Functions
####################################### 
linear = lambda x, a, b: a*x + b
Ohm = lambda x, a, b: -a*x + b
curve = lambda x, il, i0, a: il - i0*( np.exp(a*x) - 1 )
Pot_C = lambda x, il, i0, a: il*x - i0*x*( np.exp(a*x) - 1 )
Pot_P = lambda x, R, E: E*x - R*x**2

# %%
######################################
##### Plotting
######################################

# ==================== FIGURE 1
fig1 = plt.figure(figsize=(13,13))
# --------------------------- PROPERTIES FIG 1 --------------------------------
frame1_f1 = fig1.add_axes((.1,.5,.8,.8))
frame2_f1 = fig1.add_axes((.1,.3,.8,.2)) 
frame3_f1 = fig1.add_axes((.1,.1,.8,.2)) 
frame1_f1_2 = frame1_f1.twinx()
frame1_f1.set_title("Circuito com bateria", fontsize=25)
frame2_f1.set_xlabel("Corrente [mA]", fontsize=20)
frame1_f1.set_ylabel("Tensão [V]", fontsize=20)
frame2_f1.set_ylabel("Tensão [V]", fontsize=18)
frame3_f1.set_ylabel("Potência [mW]", fontsize=18)
frame1_f1_2.set_ylabel("Potência [mW]", fontsize=20)
frame1_f1.tick_params(axis='both', labelsize=16)
frame2_f1.tick_params(axis='both', labelsize=16)
frame3_f1.tick_params(axis='both', labelsize=16)
frame1_f1.set_xlim(0, 200)
#frame1_f1.set_ylim(0, 1.75)
frame2_f1.set_xlim(0, 200)
frame3_f1.set_xlim(0, 200)
#frame2_f1.set_ylim(-0.003, 0.003)
frame1_f1.set_xticklabels([])
frame2_f1.set_xticklabels([])
frame1_f1.tick_params(direction="in", size=8)
frame2_f1.tick_params(direction="in", size=8)
frame3_f1.tick_params(direction="in", size=8)

#frame2_f1.set_yticks([-0.002, 0, 0.001])
#frame2_f1.grid(which='major', axis='y')
# these are matplotlib.patch.Patch properties
props1 = dict(boxstyle='round', facecolor='tomato', alpha=0.5)
props2 = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
# --------------------------- PLOTTING FIG 1 ----------------------------------
frame1_f1.plot(i_pilha, V_pilha, '^k', label='Medidas')
frame1_f1.errorbar(i_pilha, V_pilha, yerr=sig_V_pilha, ecolor='black',
                   ls='none', lw=1, capsize=2)
#ajuste
popt_p, popv_p = curve_fit(Ohm, i_pilha, V_pilha, p0=[0.851925/1000, 1.22859])
frame1_f1.plot(t, Ohm(t, *popt_p), '-r', label='Ajuste: Tensão')
[sig_a, sig_b] = np.sqrt(np.diag(popv_p))

#valores do webroot
R_wb, E_wb, sig_R_wb, sig_E_wb = 0.851925/1000, 1.22859, 0.00403247/1000, 9.8173e-5
chi2_wb = 61.8793
# residuos
difference_p = V_pilha - Ohm(i_pilha, *[R_wb, E_wb])
#difference_p = V_pilha - Ohm(i_pilha, *popt_p)
frame2_f1.plot(i_pilha, difference_p, '^k', ms=5)
frame2_f1.axhline(y=0, c='r')
frame2_f1.errorbar(i_pilha, difference_p, yerr=sig_V_pilha, ecolor='black',
                   ls='none', lw=1, capsize=2)

# chi² e DOF
chi2_p = np.sum( (difference_p/sig_V_pilha)**2 )
DOF_p = len(V_pilha) - len(popt_p)

# Grafico da potencia
Pot_p = V_pilha*i_pilha
# ajuste
popt_pp, pop_vp = curve_fit(Pot_P, i_pilha, Pot_p)
[R_pp, E_pp] = popt_pp
[sig_R_pp, sig_E_pp] = np.sqrt( np.diag(pop_vp) )
frame1_f1_2.plot(t, Pot_P(t, *popt_pp), '-b', label='Ajuste: Potência') # potencia em Wats
frame1_f1_2.plot(i_pilha, Pot_p, '^', color='darkorange', label='Medidas: Potência') # potencia em Wats

# Sigma da potencia
# propagando as incertezas da equaçao P = Vi
sig_Pot_pilha = Pot_p*np.sqrt( (sig_V_pilha/V_pilha)**2 + (sig_i_pilha/i_pilha)**2 )

# residuos potencia
difference_pp = Pot_p - Pot_P(i_pilha, *popt_pp)
frame3_f1.axhline(y=0, c='b')
frame3_f1.plot(i_pilha, difference_pp, '^', color='darkorange', ms=5)
frame3_f1.errorbar(i_pilha, difference_pp, yerr=sig_Pot_pilha, ecolor='black',
                   ls='none', lw=1, capsize=2)

# chi² e DOF potencia
chi2_pp = np.sum( (difference_pp/sig_Pot_pilha)**2 )
DOF_pp = len(V_pilha) - len(popt_pp)

# caixa de texto
textstr_p = '\n'.join((
    r'Ajuste: $ V = E - Ri $',
    r'$E = (%.4f \pm %.4f) \ V $' % (E_wb, sig_E_wb),
    r'$R = (%.3f \pm %.3f) \ \Omega $' % (R_wb*1000, sig_R_wb*1000), 
    r'$ \chi^2 = %.3f$' % (chi2_wb), 
    r'$GL = %.i$' % (DOF_p) ))
textstr_pp = '\n'.join((
    r'Ajuste: $ Pot = Ei - Ri^2 $',
    r'$E = (%.4f \pm %.4f) \ V $' % (E_pp, sig_E_pp),
    r'$R = (%.3f \pm %.3f) \ \Omega $' % (R_pp*1000, sig_R_pp*1000), 
    r'$ \chi^2 = %.3f$' % (chi2_pp), 
    r'$GL = %.i$' % (DOF_pp) ))
frame1_f1.text(1.1, 0.80, textstr_p, transform=frame1_f1.transAxes, fontsize=20,
        verticalalignment='top', bbox=props1)
frame1_f1.text(1.1, 0.6, textstr_pp, transform=frame1_f1.transAxes, fontsize=20,
        verticalalignment='top', bbox=props2)

fig1.legend(fontsize=18, bbox_to_anchor=(1.225, 1.22))
fig1.savefig('Battery_circuit_curve.png', bbox_inches='tight')


# %%
# ==================== FIGURE 2
fig2 = plt.figure(figsize=(13, 13))
# --------------------------- PROPERTIES FIG 2 --------------------------------
frame1_f2 = fig2.add_axes((.1,.3,.8,.6))
frame2_f2 = fig2.add_axes((.1,.1,.8,.2)) 
# creating a secondary axes
frame1_f2_2 = frame1_f2.twinx()
frame1_f2.set_title("Curva característica da celula solar", fontsize=25)
frame1_f2.set_ylabel("Corrente [mA]", fontsize=20)
frame1_f2_2.set_ylabel("Potencia [mW]", fontsize=20)
frame2_f2.set_xlabel("Tensão [V]", fontsize=20)
frame2_f2.set_ylabel("Corrente [mA]", fontsize=18)
frame1_f2.tick_params(axis='both', labelsize=16)
frame2_f2.tick_params(axis='both', labelsize=16)
frame1_f2.set_xlim(0, 6)
frame1_f2_2.set_xlim(0, 6)
frame1_f2.set_ylim(0, 14)
frame1_f2_2.set_ylim(0, 40)
frame2_f2.set_xlim(0, 6)
frame2_f2.set_ylim(-3, 3)
frame1_f2.set_xticklabels([])
frame1_f2.tick_params(direction="in", size=8)
frame2_f2.tick_params(direction="in", size=8)
# these are matplotlib.patch.Patch properties
props1 = dict(boxstyle='round', facecolor='tomato', alpha=0.5)
props2 = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
# --------------------------- PLOTTING FIG 2 ----------------------------------
# plot da corrente
frame1_f2.plot(V_celula, i_celula, '^k', label='Medidas da corrente')
frame1_f2.errorbar(V_celula, i_celula, yerr=sig_i_celula, ecolor='black',
                   ls='none', lw=1, capsize=2)

# ajuste da curva caracteristica (da corrente)
popt, popv = curve_fit(curve, V_celula, i_celula, p0=[11, 0.2, 0])
sig_celula = np.sqrt(np.diag(popv))


# Potencia
Pot_celula = V_celula*i_celula
sig_Pot_celula = Pot_celula*np.sqrt( (sig_V_celula/V_celula)**2 + (sig_i_celula/i_celula)**2 )

# ajuste Potencia da celula
popt_pc, popv_pc = curve_fit(Pot_C, V_celula, Pot_celula)
[il_pc, i0_pc, a_pc] = popt_pc
[sig_il_pc, sig_i0_pc, sig_a_pc] = np.sqrt(np.diag(popv_pc))

# valores da potencia maxima e corrente maxima
Pot_celula_smoth = Pot_C(x, *popt_pc)
idx_max = Pot_celula_smoth.argmax()
Pot_max = Pot_celula_smoth[idx_max]
iP_max = curve(x[idx_max], *popt)

#chi2 da potencia
difference_pc = Pot_celula - Pot_C(V_celula, *popt_pc)
chi2_pc = np.sum( (difference_pc/sig_Pot_celula)**2 )

# plots da potencia
frame1_f2_2.plot(V_celula, Pot_celula, '^', color='darkorange', label='Medidas da potência')
frame1_f2_2.plot(x, Pot_C(x, *popt_pc), '-b', label='Ajuste da potência')
#frame1_f2_2.errorbar(V_celula, Pot_celula, yerr=sig_Pot_celula, ecolor='black',
#                   ls='none', lw=1, capsize=2)

# plots das retas de potencia maxima e corrente respectiva
frame1_f2_2.vlines(x[idx_max], ymin=0, ymax=Pot_celula_smoth[idx_max], lw=1, linestyle='--')
frame1_f2.hlines(curve(x[idx_max], *popt), xmin=0, xmax=x[idx_max], lw=1, linestyle='--')


difference = i_celula - curve(V_celula, *popt)

frame1_f2.plot(x, curve(x, *popt), '-r', label='Ajuste da corrente')
frame2_f2.plot(V_celula, difference, '^k')
frame2_f2.axhline(y=0, c='r')
frame2_f2.errorbar(V_celula, difference, yerr=sig_i_celula, ecolor='black',
                   ls='none', lw=1, capsize=2)

# Chi2
chi2_celula = np.sum( (difference/sig_i_celula)**2 )
DOF_celula = len(i_celula) - len(popt)
# --------------------------- TEXT FIG 2 ----------------------------------

# caixa de texto
# place a text box in upper left in axes coords
textstr = '\n'.join((
    r'Ajuste: Curva característica',
    r'$ i = i_L - i_0 [e^{\frac{e}{nK_BT}V} - 1] $',
    r'$i_L = (%.2f \pm %.2f) \ mA $' % (popt[0], sig_celula[0]),
    r'$i_0 = (%.2f \pm %.2f) \ mA $' % (popt[1], sig_celula[1]), 
    r'$ \frac{e}{n K_B T} = (%.2f \pm %.2f) \ V^{-1} $' % (popt[2], sig_celula[2]),
    r'$ \chi^2 = %.3f$' % (chi2_celula), 
    r'$GL = %.i$' % (DOF_celula) ))

textstr2 = '\n'.join((
    r'Ajuste: Potência',
    r'$ Pot = V \cdot i_L - V \cdot i_0 [e^{\frac{e}{nK_BT}V} - 1] $',
    r'$P_{max} = %.2f  \ mW $' % (Pot_max),
    r'$i_{max} = %.2f \ mA$' % (iP_max),
    r'$i_L = (%.2f \pm %.2f) \ mA $' % (il_pc, sig_il_pc),
    r'$i_0 = (%.2f \pm %.2f) \ mA $' % (i0_pc, sig_i0_pc),
    r'$ \frac{e}{n K_B T} = (%.2f \pm %.2f) \ V^{-1} $' % (a_pc, sig_a_pc),
    r'$ \chi^2 = %.3f$' % (chi2_pc),
    r'$GL = %.i$' % (DOF_celula)))

frame1_f2.text(1.08, 0.75, textstr, transform=frame1_f2.transAxes, fontsize=20,
        verticalalignment='top', bbox=props1)
frame1_f2.text(1.08, 0.35, textstr2, transform=frame1_f2.transAxes, fontsize=20,
        verticalalignment='top', bbox=props2)

fig2.legend(fontsize=18, bbox_to_anchor=(1.195, 0.86))
fig2.savefig('Solar_cell_curve.png', bbox_inches='tight')











