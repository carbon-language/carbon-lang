//===-- PPCScheduleG4Plus.td - PPC G4+ Scheduling Defs. ----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the itinerary class data for the G4+ (7450) processor.
//
//===----------------------------------------------------------------------===//

def IU3    : FuncUnit; // integer unit 3 (7450 simple)
def IU4    : FuncUnit; // integer unit 4 (7450 simple)

def G4PlusItineraries : ProcessorItineraries<
  [IU1, IU2, IU3, IU4, BPU, SLU, FPU1, VFPU, VIU1, VIU2, VPU], [], [
  InstrItinData<IIC_IntSimple   , [InstrStage<1, [IU1, IU2, IU3, IU4]>]>,
  InstrItinData<IIC_IntGeneral  , [InstrStage<1, [IU1, IU2, IU3, IU4]>]>,
  InstrItinData<IIC_IntCompare  , [InstrStage<1, [IU1, IU2, IU3, IU4]>]>,
  InstrItinData<IIC_IntDivW     , [InstrStage<23, [IU2]>]>,
  InstrItinData<IIC_IntMFFS     , [InstrStage<5, [FPU1]>]>,
  InstrItinData<IIC_IntMFVSCR   , [InstrStage<2, [VFPU]>]>,
  InstrItinData<IIC_IntMTFSB0   , [InstrStage<5, [FPU1]>]>,
  InstrItinData<IIC_IntMulHW    , [InstrStage<4, [IU2]>]>,
  InstrItinData<IIC_IntMulHWU   , [InstrStage<4, [IU2]>]>,
  InstrItinData<IIC_IntMulLI    , [InstrStage<3, [IU2]>]>,
  InstrItinData<IIC_IntRotate   , [InstrStage<1, [IU1, IU2, IU3, IU4]>]>,
  InstrItinData<IIC_IntShift    , [InstrStage<2, [IU1, IU2, IU3, IU4]>]>,
  InstrItinData<IIC_IntTrapW    , [InstrStage<2, [IU1, IU2, IU3, IU4]>]>,
  InstrItinData<IIC_BrB         , [InstrStage<1, [BPU]>]>,
  InstrItinData<IIC_BrCR        , [InstrStage<2, [IU2]>]>,
  InstrItinData<IIC_BrMCR       , [InstrStage<2, [IU2]>]>,
  InstrItinData<IIC_BrMCRX      , [InstrStage<2, [IU2]>]>,
  InstrItinData<IIC_LdStDCBF    , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_LdStDCBI    , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_LdStLoad    , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_LdStLoadUpd , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_LdStStore   , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_LdStStoreUpd, [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_LdStDSS     , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_LdStICBI    , [InstrStage<3, [IU2]>]>,
  InstrItinData<IIC_LdStSTFD    , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_LdStSTFDU   , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_LdStLFD     , [InstrStage<4, [SLU]>]>,
  InstrItinData<IIC_LdStLFDU    , [InstrStage<4, [SLU]>]>,
  InstrItinData<IIC_LdStLHA     , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_LdStLHAU    , [InstrStage<3, [SLU]>]>,  
  InstrItinData<IIC_LdStLMW     , [InstrStage<37, [SLU]>]>,
  InstrItinData<IIC_LdStLVecX   , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_LdStLWA     , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_LdStLWARX   , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_LdStSTD     , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_LdStSTDCX   , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_LdStSTDU    , [InstrStage<3, [SLU]>]>,  
  InstrItinData<IIC_LdStSTVEBX  , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_LdStSTWCX   , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_LdStSync    , [InstrStage<35, [SLU]>]>,
  InstrItinData<IIC_SprISYNC    , [InstrStage<0, [IU1, IU2, IU3, IU4]>]>,
  InstrItinData<IIC_SprMFSR     , [InstrStage<4, [IU2]>]>,
  InstrItinData<IIC_SprMTMSR    , [InstrStage<2, [IU2]>]>,
  InstrItinData<IIC_SprMTSR     , [InstrStage<2, [IU2]>]>,
  InstrItinData<IIC_SprTLBSYNC  , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_SprMFCR     , [InstrStage<2, [IU2]>]>,
  InstrItinData<IIC_SprMFMSR    , [InstrStage<3, [IU2]>]>,
  InstrItinData<IIC_SprMFSPR    , [InstrStage<4, [IU2]>]>,
  InstrItinData<IIC_SprMFTB     , [InstrStage<5, [IU2]>]>,
  InstrItinData<IIC_SprMTSPR    , [InstrStage<2, [IU2]>]>,
  InstrItinData<IIC_SprMTSRIN   , [InstrStage<2, [IU2]>]>,
  InstrItinData<IIC_SprRFI      , [InstrStage<1, [IU1, IU2, IU3, IU4]>]>,
  InstrItinData<IIC_SprSC       , [InstrStage<0, [IU1, IU2, IU3, IU4]>]>,
  InstrItinData<IIC_FPGeneral   , [InstrStage<5, [FPU1]>]>,
  InstrItinData<IIC_FPAddSub    , [InstrStage<5, [FPU1]>]>,  
  InstrItinData<IIC_FPCompare   , [InstrStage<5, [FPU1]>]>,
  InstrItinData<IIC_FPDivD      , [InstrStage<35, [FPU1]>]>,
  InstrItinData<IIC_FPDivS      , [InstrStage<21, [FPU1]>]>,
  InstrItinData<IIC_FPFused     , [InstrStage<5, [FPU1]>]>,
  InstrItinData<IIC_FPRes       , [InstrStage<14, [FPU1]>]>,
  InstrItinData<IIC_VecGeneral  , [InstrStage<1, [VIU1]>]>,
  InstrItinData<IIC_VecFP       , [InstrStage<4, [VFPU]>]>,
  InstrItinData<IIC_VecFPCompare, [InstrStage<2, [VFPU]>]>,
  InstrItinData<IIC_VecComplex  , [InstrStage<4, [VIU2]>]>,
  InstrItinData<IIC_VecPerm     , [InstrStage<2, [VPU]>]>,
  InstrItinData<IIC_VecFPRound  , [InstrStage<4, [VIU1]>]>,
  InstrItinData<IIC_VecVSL      , [InstrStage<2, [VPU]>]>,
  InstrItinData<IIC_VecVSR      , [InstrStage<2, [VPU]>]>
]>;
