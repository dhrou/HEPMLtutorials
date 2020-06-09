#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

__doc__="""
ATLAS Higgs Machine Learning Challenge 2014
Read CERN Open Data Portal Dataset http://opendata.cern.ch/record/328
and manipulate it
 - KaggleWeight and KaggleSet are removed
  - Label is changd from charcter to integer 0 or 1
 - DetailLabel is introduced indicating subpopulations
 - systematics effect are simulated
     - bkg_weight_norm : manipulates the background weight of the W background
     - tau_energy_scale : manipulates PRI_had_pt and recompute other quantities accordingly
             Some WARNING : variable DER_mass_MMC is not properly manipulated (modification is linearised), 
             and I advocate to NOT use DER_mass_MMC when doSystTauEnergyScale is enabled
             There is a threshold in the original HiggsML file at 20GeV on PRI_had_energy. 
             This threshold is moved when changing sysTauEnergyScale which is unphysicsal. 
             So if you're going to play with sysTauEnergyScale (within 0.9-1.1), 
             I suggest you remove events below say 22 GeV *after* the manipulation
             applying doSystTauEnergyScale with sysTauENergyScale=1. does NOT yield identical results as not applyield 
             doSystTauEnergyScale, this is because of rounding error and zero mass approximation.
             doSysTauEnerbyScale impacts PRI_had_pt as well as PRI_met and PRI_met_phi
    - so overall I suggest that when playing with doSystTauEnergyScale, the reference is
          - not using DER_mass_MMC
          - applying *after* this manipulation PRI_had_pt>22
          - run with sysTauENergyScale=1. to have the reference
          
Author D. Rousseau LAL, Nov 2016

Modification Dec 2016 (V. Estrade):
- Wrap everything into separated functions.
- V4 class now handle 1D-vector values (to improve computation efficiency).
- Fix compatibility with both python 2 and 3.
- Use pandas.DataFrame to ease computation along columns
- Loading function for the base HiggsML dataset (fetch it on the internet if needed)

Refactor March 2017 (V. Estrade):
- Split load function (cleaner)

July 06 2017 (V. Estrade):
- Add normalization_weight function

May 2019 (D. Rousseau) :
- Major hack, in preparation for Centralesupelec EI,
python syst/datawarehouse/datawarehouse/higgsml.py -i atlas-higgs-challenge-2014-v2.csv.gz -o atlas-higgs-challenge-2014-v2-s0.csv

python higgsml_syst.py -i atlas-higgs-challenge-2014-v2.csv.gz -o atlas-higgs-challenge-2014-v2-syst1.csv --csv -p --BKGnorm 1. --Wnorm 1. --tes 1. --jes 1. --softMET 0. --seed 31415926
python higgsml_syst.py --help # for command help
reasonable values for parameters
BKGnorm : 1.05  
Wnorm : 1.5 
tes : 1.03
jes : 1.03
softMET : 3 GeV




"""
__version__ = "4.0"
__author__ = "David Rousseau, and Victor Estrade "

import sys
import os
import gzip
import copy
import pandas as pd
import numpy as np


def load_higgs(in_file):
    filename=in_file
    print ("filename=",filename)
    data = pd.read_csv(filename)
    return data

# ==================================================================================
#  V4 Class and physic computations
# ==================================================================================

class V4:
    """
    A simple 4-vector class to ease calculation, work easy peasy on numpy vector of 4 vector
    """
    px=0
    py=0
    pz=0
    e=0
    def __init__(self,apx=0., apy=0., apz=0., ae=0.):
        """
        Constructor with 4 coordinates
        """
        self.px = apx
        self.py = apy
        self.pz = apz
        self.e = ae
        if self.e + 1e-3 < self.p():
            raise ValueError("Energy is too small! Energy: {}, p: {}".format(self.e, self.p()))

    def copy(self):
        return copy.deepcopy(self)
    
    def p2(self):
        return self.px**2 + self.py**2 + self.pz**2
    
    def p(self):
        return np.sqrt(self.p2())
    
    def pt2(self):
        return self.px**2 + self.py**2
    
    def pt(self):
        return np.sqrt(self.pt2())
    
    def m(self):
        return np.sqrt( np.abs( self.e**2 - self.p2() ) ) # abs is needed for protection
    
    def eta(self):
        return np.arcsinh( self.pz/self.pt() )
    
    def phi(self):
        return np.arctan2(self.py, self.px)
    
    def deltaPhi(self, v):
        """delta phi with another v"""
        return (self.phi() - v.phi() + 3*np.pi) % (2*np.pi) - np.pi
    
    def deltaEta(self,v):
        """delta eta with another v"""
        return self.eta()-v.eta()
    
    def deltaR(self,v):
        """delta R with another v"""
        return np.sqrt(self.deltaPhi(v)**2+self.deltaEta(v)**2 )

    def eWithM(self,m=0.):
        """recompute e given m"""
        return np.sqrt(self.p2()+m**2)

    # FIXME this gives ugly prints with 1D-arrays
    def __str__(self):
        return "PxPyPzE( %s,%s,%s,%s)<=>PtEtaPhiM( %s,%s,%s,%s) " % (self.px, self.py,self.pz,self.e,self.pt(),self.eta(),self.phi(),self.m())

    def scale(self,factor=1.): # scale
        """Apply a simple scaling"""
        self.px *= factor
        self.py *= factor
        self.pz *= factor
        self.e = np.abs( factor*self.e )
    
    def scaleFixedM(self,factor=1.): 
        """Scale (keeping mass unchanged)"""
        m = self.m()
        self.px *= factor
        self.py *= factor
        self.pz *= factor
        self.e = self.eWithM(m)
    
    def setPtEtaPhiM(self, pt=0., eta=0., phi=0., m=0):
        """Re-initialize with : pt, eta, phi and m"""
        self.px = pt*np.cos(phi)
        self.py = pt*np.sin(phi)
        self.pz = pt*np.sinh(eta)
        self.e = self.eWithM(m)
    
    def sum(self, v):
        """Add another V4 into self"""
        self.px += v.px
        self.py += v.py
        self.pz += v.pz
        self.e += v.e
    
    def __iadd__(self, other):
        """Add another V4 into self"""
        try:
            self.px += other.px
            self.py += other.py
            self.pz += other.pz
            self.e += other.e
        except AttributeError: 
            # If 'other' is not V4 like object then return special NotImplemented error
            return NotImplemented
        return self
    
    def __add__(self, other):
        """Add 2 V4 vectors : v3 = v1 + v2 = v1.__add__(v2)"""
        copy = self.copy()
        try:
            copy.px += other.px
            copy.py += other.py
            copy.pz += other.pz
            copy.e += other.e
        except AttributeError: 
            # If 'other' is not V4 like object then return special NotImplemented error
            return NotImplemented
        return copy




# ==================================================================================
def getDetailLabel(origWeight, Label, num=True):
    """
    Given original weight and label, 
    return more precise label specifying the original simulation type.
    
    Args
    ----
        origWeight: the original weight of the event
        Label : the label of the event (can be {"b", "s"} or {0,1})
        num: (default=True) if True use the numeric detail labels
                else use the string detail labels. You should prefer numeric labels.

    Return
    ------
        detailLabel: the corresponding detail label ("W" is the default if not found)

    Note : Could be better optimized but this is fast enough.
    """
    # prefer numeric detail label
    detail_label_num={
        57207:0, # Signal
        4613:1,
        8145:2,
        4610:3,
        917703: 105, #Z
        5127399:111,
        4435976:112,
        4187604:113,
        2407146:114,
        1307751:115,
        944596:122,
        936590:123,
        1093224:124,
        225326:132,
        217575:133,
        195328:134,
        254338:135,
        2268701:300 #T
        }
    # complementary for W detaillabeldict=200
    #previous alphanumeric detail label    
    detail_label_str={
       57207:"S0",
       4613:"S1",
       8145:"S2",
       4610:"S3",
       917703:"Z05",
       5127399:"Z11",
       4435976:"Z12",
       4187604:"Z13",
       2407146:"Z14",
       1307751:"Z15",
       944596:"Z22",
       936590:"Z23",
       1093224:"Z24",
       225326:"Z32",
       217575:"Z33",
       195328:"Z34",
       254338:"Z35",
       2268701:"W"  # was T
    }

    if num:
        detailLabelDict = detail_label_num
        defaultLabel=400 # 400 "T" is the default value if not found
    else:
        detailLabelDict = detail_label_str
        defaultLabel="T" #"T" is the default value if not found
        
    iWeight=int(1e7*origWeight+0.5)
    detailLabel = detailLabelDict.get(iWeight, defaultLabel) 
    if detailLabel == "T" and (Label != 0 and Label != 'b') :
        raise ValueError("ERROR! if not in detailLabelDict sould have Label==1 ({}, {})".format(iWeight,Label))

    return detailLabel


def w_bkg_weight_norm(data, systBkgNorm):
    """
    Apply a scaling to the weight. For W background

    Args
    ----
        data: the dataset should be a pandas.DataFrame like object.
            This function will modify the given data inplace.

    """
    # scale the weight, arbitrary but reasonable value
    data["Weight"] = ( data["Weight"]*systBkgNorm ).where(data["detailLabel"] == 300, other=data["Weight"])



def all_bkg_weight_norm(data, systBkgNorm):
    """
    Apply a scaling to the weight.

    Args
    ----
        data: the dataset should be a pandas.DataFrame like object.
            This function will modify the given data inplace.

    """
    # scale the weight, arbitrary but reasonable value
    data["Weight"] = ( data["Weight"]*systBkgNorm ).where(data["Label"] == 0, other=data["Weight"])





        
# ==================================================================================
# Manipulate the 4-momenta
# ==================================================================================
def mom4_manipulate (data, systTauEnergyScale, systJetEnergyScale,softMET):
    """
    Manipulate primary inputs : the PRI_had_pt PRI_jet_leading_pt PRI_jet_subleading_pt and recompute the others values accordingly.

    Args
    ----
        data: the dataset should be a pandas.DataFrame like object.
            This function will modify the given data inplace.
        systTauEnergyScale : the factor applied : PRI_had_pt <-- PRI_had_pt * systTauEnergyScale
        systJetEnergyScale : the factor applied : all jet pt  * systJetEnergyScale
        recompute MET accordingly
        Add soft MET gaussian random energy 

        
    Notes :
    -------
        Recompute :
            - PRI_had_pt
            - PRI_jet_leading_pt
            - PRI_jet_subleading_pt            
            - PRI_met
            - PRI_met_phi
            - PRI_met_sumet
        Round up to 3 decimals.

    """


    vmet = V4() # met 4-vector
    vmet.setPtEtaPhiM(data["PRI_met"], 0., data["PRI_met_phi"], 0.) # met mass zero,
    met_sumet=data["PRI_met_sumet"]
    
    if systTauEnergyScale!=1.:
        # scale tau energy scale, arbitrary but reasonable value
        data["PRI_had_pt"] *= systTauEnergyScale 


        # build 4-vectors
        vtau = V4() # tau 4-vector
        vtau.setPtEtaPhiM(data["PRI_had_pt"], data["PRI_had_eta"], data["PRI_had_phi"], 0.8) # tau mass 0.8 like in original

        #vlep = V4() # lepton 4-vector
        #vlep.setPtEtaPhiM(data["PRI_lep_pt"], data["PRI_lep_eta"], data["PRI_lep_phi"], 0.) # lep mass 0 (either 0.106 or 0.0005 but info is lost)


        # fix MET according to tau pt change (minus sign since met is minus sum pt of visible particles
        vtauDeltaMinus = vtau.copy()
        vtauDeltaMinus.scaleFixedM( (1.-systTauEnergyScale)/systTauEnergyScale )
        vmet += vtauDeltaMinus
        vmet.pz = 0.
        vmet.e = vmet.eWithM(0.)

        #met_sum_et is increased if energy scale increased
        tauDeltaMinus=vtau.pt()
        met_sumet+= (systTauEnergyScale-1)/systTauEnergyScale *tauDeltaMinus


    
    
    

    # scale jet energy scale, arbitrary but reasonable value

    if systJetEnergyScale!=1. :
        #data["PRI_jet_leading_pt"]    *= systJetEnergyScale
        data["PRI_jet_leading_pt"] = np.where(data["PRI_jet_num"] >0,
                                           data["PRI_jet_leading_pt"]*systJetEnergyScale,
                                           data["PRI_jet_leading_pt"])
        #data["PRI_jet_subleading_pt"] *= systJetEnergyScale
        data["PRI_jet_subleading_pt"] = np.where(data["PRI_jet_num"] >1,
                                           data["PRI_jet_subleading_pt"]*systJetEnergyScale,
                                           data["PRI_jet_subleading_pt"])

        data["PRI_jet_all_pt"] *= systJetEnergyScale 

        jet_all_pt= data["PRI_jet_all_pt"]
    
        #met_sum_et is increased if energy scale increased
        met_sumet+= (systJetEnergyScale-1)/systJetEnergyScale *jet_all_pt
    

        # first jet if it exists
        vj1 = V4()
        vj1.setPtEtaPhiM(data["PRI_jet_leading_pt"].where( data["PRI_jet_num"] > 0, other=0 ),
                             data["PRI_jet_leading_eta"].where( data["PRI_jet_num"] > 0, other=0 ),
                             data["PRI_jet_leading_phi"].where( data["PRI_jet_num"] > 0, other=0 ),
                             0.) # zero mass
        # fix MET according to leading jet pt change
        vj1DeltaMinus = vj1.copy()
        vj1DeltaMinus.scaleFixedM( (1.-systJetEnergyScale)/systJetEnergyScale )
        vmet += vj1DeltaMinus
        vmet.pz = 0.
        vmet.e = vmet.eWithM(0.)



        # second jet if it exists
        vj2=V4()
        vj2.setPtEtaPhiM(data["PRI_jet_subleading_pt"].where( data["PRI_jet_num"] > 1, other=0 ),
                         data["PRI_jet_subleading_eta"].where( data["PRI_jet_num"] > 1, other=0 ),
                         data["PRI_jet_subleading_phi"].where( data["PRI_jet_num"] > 1, other=0 ),
                         0.) # zero mass

        # fix MET according to leading jet pt change
        vj2DeltaMinus = vj2.copy()
        vj2DeltaMinus.scaleFixedM( (1.-systJetEnergyScale)/systJetEnergyScale )
        vmet += vj2DeltaMinus
        vmet.pz = 0.
        vmet.e = vmet.eWithM(0.)
        
    #note that in principle we should also fix MET for the third jet or more but we do not have enough information

    if softMET>0:
        # add soft met term
        # Compute the missing v4 vector
        random_state = np.random.RandomState(seed=seed)
        SIZE = data.shape[0]
        v4_soft_term = V4()
        v4_soft_term.px = random_state.normal(0, softMET, size=SIZE)
        v4_soft_term.py = random_state.normal(0, softMET, size=SIZE)
        v4_soft_term.pz = np.zeros(SIZE)
        v4_soft_term.e = v4_soft_term.eWithM(0.)
        # fix MET according to soft term
        vmet = vmet + v4_soft_term

    

    data["PRI_met"] = vmet.pt()
    data["PRI_met_phi"] = vmet.phi()
    data["PRI_met_sumet"] = met_sumet
                     
    # Fix precision to 3 decimals
    DECIMALS = 3

    data["PRI_had_pt"] = data["PRI_had_pt"].round(decimals=DECIMALS)
    data["PRI_had_eta"] = data["PRI_had_eta"].round(decimals=DECIMALS)
    data["PRI_had_phi"] = data["PRI_had_phi"].round(decimals=DECIMALS)
    data["PRI_lep_pt"] = data["PRI_lep_pt"].round(decimals=DECIMALS)
    data["PRI_lep_eta"] = data["PRI_lep_eta"].round(decimals=DECIMALS)
    data["PRI_lep_phi"] = data["PRI_lep_phi"].round(decimals=DECIMALS)
    data["PRI_met"] = data["PRI_met"].round(decimals=DECIMALS)
    data["PRI_met_phi"] = data["PRI_met_phi"].round(decimals=DECIMALS)
    data["PRI_met_sumet"] = data["PRI_met_sumet"].round(decimals=DECIMALS)
    data["PRI_jet_leading_pt"] = data["PRI_jet_leading_pt"].round(decimals=DECIMALS)
    data["PRI_jet_leading_eta"] = data["PRI_jet_leading_eta"].round(decimals=DECIMALS)
    data["PRI_jet_leading_phi"] = data["PRI_jet_leading_phi"].round(decimals=DECIMALS)
    data["PRI_jet_subleading_pt"] = data["PRI_jet_subleading_pt"].round(decimals=DECIMALS)
    data["PRI_jet_subleading_eta"] = data["PRI_jet_subleading_eta"].round(decimals=DECIMALS)
    data["PRI_jet_subleading_phi"] = data["PRI_jet_subleading_phi"].round(decimals=DECIMALS)
    data["PRI_jet_all_pt"] = data["PRI_jet_all_pt"].round(decimals=DECIMALS)
    


# ==================================================================================
#  MAIN : here is defined the behaviour of this module as a main script
# ==================================================================================
import argparse
 
def parse_args():
    """
    ArgumentParser.

    Return
    ------
        args: the parsed arguments.
    """
    
    # First create a parser with a short description of the program.
    # The parser will automatically handle the usual stuff like the --help messages.
    parser = argparse.ArgumentParser(
        description="Higgs manipulation script. Can be used to produce new dataset with skewed features.")

    # ====================================================================================
    # Real things here
    # ====================================================================================

    parser.add_argument("--quiet", "-q", help="Verbosity level", action="store_true", dest='quiet')
    parser.add_argument("--debug", "-d", help="If debug", default=False, action="store_true", dest='debug')
    parser.add_argument("--csv", help="Option flag to prevent compression into gzip file.",
        action="store_true", dest='csv')
    parser.add_argument('--BKGnorm', help='All Background Weight norm scale factor. Reasonable value [0.9, 1.1]. Default 1.', default=1.,type=float, dest='bkg_scale')
    parser.add_argument('--Wnorm', help='W Background Weight norm scale factor. Reasonable value [0.2, 5.] Default 1.',default=1., type=float, dest='w_scale')
    parser.add_argument('--tes', help='Tau energy scale factor. Reasonable value [0.9, 1.1]. Default 1.', default=1., type=float, dest='tes')
    parser.add_argument('--jes', help='Jet energy scale factor. Reasonable value [0.9, 1.1]. Default 1.', default=1.,type=float, dest='jes')
    parser.add_argument('--softMET', help='soft MET term. Reasonable value [0.,10.]. Default 0.', default=0., type=float, dest='softMET')        
    parser.add_argument('--seed', help='random seed for soft MET term. Default 31415926 ', default=31415926, type=int, dest='seed')        
    parser.add_argument('-i', help='the name of the input file', default="atlas-higgs-challenge-2014-v2-s0_e500000.csv.gz", dest="in_file")
    parser.add_argument('-o', default="higgs_output.csv", help='the name of the output file', dest="out_file")

    
    # Now do your job and parse my command line !
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    quiet = args.quiet # quiet flag
    csv = args.csv # csv flag
    w_scale = args.w_scale # W bkg Weight scaling factor
    bkg_scale = args.bkg_scale # ALl bkg Weight scaling factor
    tes = args.tes # Tau energy scale factor
    jes = args.jes # ket energy scale factor
    softMET = args.softMET # soft Missing Energy Term
    seed = args.seed # random seed for soft met term
    in_file = args.in_file # input file
    out_file = args.out_file # output file
    debug = args.debug

    columns = [ "EventId",
                "PRI_had_pt",
                "PRI_had_eta",
                "PRI_had_phi",
                "PRI_lep_pt",
                "PRI_lep_eta",
                "PRI_lep_phi",
                "PRI_met",
                "PRI_met_phi",
                "PRI_met_sumet",
                "PRI_jet_num",
                "PRI_jet_leading_pt",
                "PRI_jet_leading_eta",
                "PRI_jet_leading_phi",
                "PRI_jet_subleading_pt",
                "PRI_jet_subleading_eta",
                "PRI_jet_subleading_phi",
                "PRI_jet_all_pt",
                "Weight",
                "Label",                
                "detailLabel",
                ] 

    if not quiet:
        print("Loading the dataset")

    data = load_higgs(in_file)

    
    if w_scale is not None:
        if not quiet:
            print("W bkg weight rescaling :", w_scale)
        w_bkg_weight_norm(data, w_scale)

    if bkg_scale is not None:
        if not quiet:
            print("All bkg weight rescaling :", bkg_scale)
        all_bkg_weight_norm(data, bkg_scale)

    if tes is not None or jes is not None or softMET is not None:
        if not quiet:
            print("Tau energy rescaling :", tes)
            print("Jet energy rescaling :", jes)
            print("Soft MET addition :", softMET)
        mom4_manipulate(data, tes,jes,softMET)

    compression = None if csv else "gzip"
    _, ext = os.path.splitext(out_file)
    if ext != ".csv":
        out_file += ".csv"
    if not csv and ext != ".gz":
        out_file += ".gz"

    if not quiet:
        print("Writing results to :", out_file)


    if debug:
        print ("only take first few rows for debug")
        data=data.head(50)

    data.to_csv(out_file, index=False, compression=compression, columns=columns)
                
    print("Done.")
