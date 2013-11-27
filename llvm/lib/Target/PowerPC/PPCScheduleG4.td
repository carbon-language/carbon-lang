//===-- PPCScheduleG4.td - PPC G4 Scheduling Definitions ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the itinerary class data for the G4 (7400) processor.
//
//===----------------------------------------------------------------------===//

def G4Itineraries : ProcessorItineraries<
  [IU1, IU2, SLU, SRU, BPU, FPU1, VIU1, VIU2, VPU, VFPU], [], [
  InstrItinData<IIC_IntSimple   , [InstrStage<1, [IU1, IU2]>]>,
  InstrItinData<IIC_IntGeneral  , [InstrStage<1, [IU1, IU2]>]>,
  InstrItinData<IIC_IntCompare  , [InstrStage<1, [IU1, IU2]>]>,
  InstrItinData<IIC_IntDivW     , [InstrStage<19, [IU1]>]>,
  InstrItinData<IIC_IntMFFS     , [InstrStage<3, [FPU1]>]>,
  InstrItinData<IIC_IntMFVSCR   , [InstrStage<1, [VIU1]>]>,
  InstrItinData<IIC_IntMTFSB0   , [InstrStage<3, [FPU1]>]>,
  InstrItinData<IIC_IntMulHW    , [InstrStage<5, [IU1]>]>,
  InstrItinData<IIC_IntMulHWU   , [InstrStage<6, [IU1]>]>,
  InstrItinData<IIC_IntMulLI    , [InstrStage<3, [IU1]>]>,
  InstrItinData<IIC_IntRotate   , [InstrStage<1, [IU1, IU2]>]>,
  InstrItinData<IIC_IntShift    , [InstrStage<1, [IU1, IU2]>]>,
  InstrItinData<IIC_IntTrapW    , [InstrStage<2, [IU1, IU2]>]>,
  InstrItinData<IIC_BrB         , [InstrStage<1, [BPU]>]>,
  InstrItinData<IIC_BrCR        , [InstrStage<1, [SRU]>]>,
  InstrItinData<IIC_BrMCR       , [InstrStage<1, [SRU]>]>,
  InstrItinData<IIC_BrMCRX      , [InstrStage<1, [SRU]>]>,
  InstrItinData<IIC_LdStDCBF    , [InstrStage<2, [SLU]>]>,
  InstrItinData<IIC_LdStDCBI    , [InstrStage<2, [SLU]>]>,
  InstrItinData<IIC_LdStLoad    , [InstrStage<2, [SLU]>]>,
  InstrItinData<IIC_LdStLoadUpd , [InstrStage<2, [SLU]>]>,
  InstrItinData<IIC_LdStStore   , [InstrStage<2, [SLU]>]>,
  InstrItinData<IIC_LdStStoreUpd, [InstrStage<2, [SLU]>]>,
  InstrItinData<IIC_LdStDSS     , [InstrStage<2, [SLU]>]>,
  InstrItinData<IIC_LdStICBI    , [InstrStage<2, [SLU]>]>,
  InstrItinData<IIC_LdStSTFD    , [InstrStage<2, [SLU]>]>,
  InstrItinData<IIC_LdStSTFDU   , [InstrStage<2, [SLU]>]>,
  InstrItinData<IIC_LdStLFD     , [InstrStage<2, [SLU]>]>,
  InstrItinData<IIC_LdStLFDU    , [InstrStage<2, [SLU]>]>,
  InstrItinData<IIC_LdStLHA     , [InstrStage<2, [SLU]>]>,
  InstrItinData<IIC_LdStLHAU    , [InstrStage<2, [SLU]>]>, 
  InstrItinData<IIC_LdStLMW     , [InstrStage<34, [SLU]>]>,
  InstrItinData<IIC_LdStLVecX   , [InstrStage<2, [SLU]>]>,
  InstrItinData<IIC_LdStLWARX   , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_LdStSTVEBX  , [InstrStage<2, [SLU]>]>,
  InstrItinData<IIC_LdStSTWCX   , [InstrStage<5, [SLU]>]>,
  InstrItinData<IIC_LdStSync    , [InstrStage<8, [SLU]>]>,
  InstrItinData<IIC_SprISYNC    , [InstrStage<2, [SRU]>]>,
  InstrItinData<IIC_SprMFSR     , [InstrStage<3, [SRU]>]>,
  InstrItinData<IIC_SprMTMSR    , [InstrStage<1, [SRU]>]>,
  InstrItinData<IIC_SprMTSR     , [InstrStage<2, [SRU]>]>,
  InstrItinData<IIC_SprTLBSYNC  , [InstrStage<8, [SRU]>]>,
  InstrItinData<IIC_SprMFCR     , [InstrStage<1, [SRU]>]>,
  InstrItinData<IIC_SprMFMSR    , [InstrStage<1, [SRU]>]>,
  InstrItinData<IIC_SprMFSPR    , [InstrStage<3, [SRU]>]>,
  InstrItinData<IIC_SprMFTB     , [InstrStage<1, [SRU]>]>,
  InstrItinData<IIC_SprMTSPR    , [InstrStage<2, [SRU]>]>,
  InstrItinData<IIC_SprMTSRIN   , [InstrStage<2, [SRU]>]>,
  InstrItinData<IIC_SprRFI      , [InstrStage<2, [SRU]>]>,
  InstrItinData<IIC_SprSC       , [InstrStage<2, [SRU]>]>,
  InstrItinData<IIC_FPGeneral   , [InstrStage<1, [FPU1]>]>,
  InstrItinData<IIC_FPAddSub    , [InstrStage<1, [FPU1]>]>,
  InstrItinData<IIC_FPCompare   , [InstrStage<1, [FPU1]>]>,
  InstrItinData<IIC_FPDivD      , [InstrStage<31, [FPU1]>]>,
  InstrItinData<IIC_FPDivS      , [InstrStage<17, [FPU1]>]>,
  InstrItinData<IIC_FPFused     , [InstrStage<1, [FPU1]>]>,
  InstrItinData<IIC_FPRes       , [InstrStage<10, [FPU1]>]>,
  InstrItinData<IIC_VecGeneral  , [InstrStage<1, [VIU1]>]>,
  InstrItinData<IIC_VecFP       , [InstrStage<4, [VFPU]>]>,
  InstrItinData<IIC_VecFPCompare, [InstrStage<1, [VIU1]>]>,
  InstrItinData<IIC_VecComplex  , [InstrStage<3, [VIU2]>]>,
  InstrItinData<IIC_VecPerm     , [InstrStage<1, [VPU]>]>,
  InstrItinData<IIC_VecFPRound  , [InstrStage<4, [VFPU]>]>,
  InstrItinData<IIC_VecVSL      , [InstrStage<1, [VIU1]>]>,
  InstrItinData<IIC_VecVSR      , [InstrStage<1, [VIU1]>]>
]>;
