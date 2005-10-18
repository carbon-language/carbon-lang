//===- PPCScheduleG3.td - PPC G3 Scheduling Definitions ----*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the itinerary class data for the G3 (750) processor.
//
//===----------------------------------------------------------------------===//


def G3Itineraries : ProcessorItineraries<G3, [
  InstrItinData<IntGeneral  , [InstrStage<1, [IU1, IU2]>]>,
  InstrItinData<IntCompare  , [InstrStage<1, [IU1, IU2]>]>,
  InstrItinData<IntDivD     , [InstrStage<0, [NoUnit]>]>,
  InstrItinData<IntDivW     , [InstrStage<19, [IU1]>]>,
  InstrItinData<IntMFFS     , [InstrStage<1, [FPU1]>]>,
  InstrItinData<IntMFVSCR   , [InstrStage<0, [NoUnit]>]>,
  InstrItinData<IntMTFSB0   , [InstrStage<3, [FPU1]>]>,
  InstrItinData<IntMTSRD    , [InstrStage<0, [NoUnit]>]>,
  InstrItinData<IntMulHD    , [InstrStage<0, [NoUnit]>]>,
  InstrItinData<IntMulHW    , [InstrStage<5, [IU1]>]>,
  InstrItinData<IntMulHWU   , [InstrStage<6, [IU1]>]>,
  InstrItinData<IntMulLI    , [InstrStage<3, [IU1]>]>,
  InstrItinData<IntRFID     , [InstrStage<0, [NoUnit]>]>,
  InstrItinData<IntRotateD  , [InstrStage<0, [NoUnit]>]>,
  InstrItinData<IntRotate   , [InstrStage<1, [IU1, IU2]>]>,
  InstrItinData<IntShift    , [InstrStage<1, [IU1, IU2]>]>,
  InstrItinData<IntTrapD    , [InstrStage<0, [NoUnit]>]>,
  InstrItinData<IntTrapW    , [InstrStage<2, [IU1, IU2]>]>,
  InstrItinData<BrB         , [InstrStage<1, [BPU]>]>,
  InstrItinData<BrCR        , [InstrStage<1, [SRU]>]>,
  InstrItinData<BrMCR       , [InstrStage<1, [SRU]>]>,
  InstrItinData<BrMCRX      , [InstrStage<1, [SRU]>]>,
  InstrItinData<LdStDCBA    , [InstrStage<2, [SLU]>]>,
  InstrItinData<LdStDCBF    , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStDCBI    , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStDCBT    , [InstrStage<2, [SLU]>]>,
  InstrItinData<LdStDSS     , [InstrStage<0, [NoUnit]>]>,
  InstrItinData<LdStICBI    , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStLBZUX   , [InstrStage<2, [SLU]>]>,
  InstrItinData<LdStLD      , [InstrStage<0, [NoUnit]>]>,
  InstrItinData<LdStLDARX   , [InstrStage<0, [NoUnit]>]>,
  InstrItinData<LdStLFD     , [InstrStage<2, [SLU]>]>,
  InstrItinData<LdStLFDU    , [InstrStage<2, [SLU]>]>,
  InstrItinData<LdStLHA     , [InstrStage<2, [SLU]>]>,
  InstrItinData<LdStLMW     , [InstrStage<34, [SLU]>]>,
  InstrItinData<LdStLVEBX   , [InstrStage<0, [NoUnit]>]>,
  InstrItinData<LdStLWA     , [InstrStage<0, [NoUnit]>]>,
  InstrItinData<LdStLWARX   , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStSLBIA   , [InstrStage<0, [NoUnit]>]>,
  InstrItinData<LdStSLBIE   , [InstrStage<0, [NoUnit]>]>,
  InstrItinData<LdStSTD     , [InstrStage<0, [NoUnit]>]>,
  InstrItinData<LdStSTDCX   , [InstrStage<0, [NoUnit]>]>,
  InstrItinData<LdStSTVEBX  , [InstrStage<0, [NoUnit]>]>,
  InstrItinData<LdStSTWCX   , [InstrStage<8, [SLU]>]>,
  InstrItinData<LdStSync    , [InstrStage<3, [SLU]>]>,
  InstrItinData<SprISYNC    , [InstrStage<2, [SRU]>]>,
  InstrItinData<SprMFSR     , [InstrStage<3, [SRU]>]>,
  InstrItinData<SprMTMSR    , [InstrStage<1, [SRU]>]>,
  InstrItinData<SprMTSR     , [InstrStage<2, [SRU]>]>,
  InstrItinData<SprTLBSYNC  , [InstrStage<3, [SRU]>]>,
  InstrItinData<SprMFCR     , [InstrStage<1, [SRU]>]>,
  InstrItinData<SprMFMSR    , [InstrStage<1, [SRU]>]>,
  InstrItinData<SprMFSPR    , [InstrStage<3, [SRU]>]>,
  InstrItinData<SprMFTB     , [InstrStage<3, [SRU]>]>,
  InstrItinData<SprMTSPR    , [InstrStage<2, [SRU]>]>,
  InstrItinData<SprMTSRIN   , [InstrStage<2, [SRU]>]>,
  InstrItinData<SprRFI      , [InstrStage<2, [SRU]>]>,
  InstrItinData<SprSC       , [InstrStage<2, [SRU]>]>,
  InstrItinData<FPGeneral   , [InstrStage<1, [FPU1]>]>,
  InstrItinData<FPCompare   , [InstrStage<1, [FPU1]>]>,
  InstrItinData<FPDivD      , [InstrStage<31, [FPU1]>]>,
  InstrItinData<FPDivS      , [InstrStage<17, [FPU1]>]>,
  InstrItinData<FPFused     , [InstrStage<2, [FPU1]>]>,
  InstrItinData<FPRes       , [InstrStage<10, [FPU1]>]>,
  InstrItinData<FPSqrt      , [InstrStage<0, [NoUnit]>]>,
  InstrItinData<VecGeneral  , [InstrStage<0, [NoUnit]>]>,
  InstrItinData<VecFP       , [InstrStage<0, [NoUnit]>]>,
  InstrItinData<VecFPCompare, [InstrStage<0, [NoUnit]>]>,
  InstrItinData<VecComplex  , [InstrStage<0, [NoUnit]>]>,
  InstrItinData<VecPerm     , [InstrStage<0, [NoUnit]>]>,
  InstrItinData<VecFPRound  , [InstrStage<0, [NoUnit]>]>,
  InstrItinData<VecVSL      , [InstrStage<0, [NoUnit]>]>,
  InstrItinData<VecVSR      , [InstrStage<0, [NoUnit]>]>
]>;
