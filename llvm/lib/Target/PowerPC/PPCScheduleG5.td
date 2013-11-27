//===-- PPCScheduleG5.td - PPC G5 Scheduling Definitions ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the itinerary class data for the G5 (970) processor.
//
//===----------------------------------------------------------------------===//

def G5Itineraries : ProcessorItineraries<
  [IU1, IU2, SLU, BPU, FPU1, FPU2, VFPU, VIU1, VIU2, VPU], [], [
  InstrItinData<IIC_IntSimple   , [InstrStage<2, [IU1, IU2]>]>,
  InstrItinData<IIC_IntGeneral  , [InstrStage<2, [IU1, IU2]>]>,
  InstrItinData<IIC_IntCompare  , [InstrStage<3, [IU1, IU2]>]>,
  InstrItinData<IIC_IntDivD     , [InstrStage<68, [IU1]>]>,
  InstrItinData<IIC_IntDivW     , [InstrStage<36, [IU1]>]>,
  InstrItinData<IIC_IntMFFS     , [InstrStage<6, [IU2]>]>,
  InstrItinData<IIC_IntMFVSCR   , [InstrStage<1, [VFPU]>]>,
  InstrItinData<IIC_IntMTFSB0   , [InstrStage<6, [FPU1, FPU2]>]>,
  InstrItinData<IIC_IntMulHD    , [InstrStage<7, [IU1, IU2]>]>,
  InstrItinData<IIC_IntMulHW    , [InstrStage<5, [IU1, IU2]>]>,
  InstrItinData<IIC_IntMulHWU   , [InstrStage<5, [IU1, IU2]>]>,
  InstrItinData<IIC_IntMulLI    , [InstrStage<4, [IU1, IU2]>]>,
  InstrItinData<IIC_IntRFID     , [InstrStage<1, [IU2]>]>,
  InstrItinData<IIC_IntRotateD  , [InstrStage<2, [IU1, IU2]>]>,
  InstrItinData<IIC_IntRotateDI , [InstrStage<2, [IU1, IU2]>]>,  
  InstrItinData<IIC_IntRotate   , [InstrStage<4, [IU1, IU2]>]>,
  InstrItinData<IIC_IntShift    , [InstrStage<2, [IU1, IU2]>]>,
  InstrItinData<IIC_IntTrapD    , [InstrStage<1, [IU1, IU2]>]>,
  InstrItinData<IIC_IntTrapW    , [InstrStage<1, [IU1, IU2]>]>,
  InstrItinData<IIC_BrB         , [InstrStage<1, [BPU]>]>,
  InstrItinData<IIC_BrCR        , [InstrStage<4, [BPU]>]>,
  InstrItinData<IIC_BrMCR       , [InstrStage<2, [BPU]>]>,
  InstrItinData<IIC_BrMCRX      , [InstrStage<3, [BPU]>]>,
  InstrItinData<IIC_LdStDCBF    , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_LdStLoad    , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_LdStLoadUpd , [InstrStage<3, [SLU]>]>,  
  InstrItinData<IIC_LdStStore   , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_LdStStoreUpd, [InstrStage<3, [SLU]>]>,  
  InstrItinData<IIC_LdStDSS     , [InstrStage<10, [SLU]>]>,
  InstrItinData<IIC_LdStICBI    , [InstrStage<40, [SLU]>]>,
  InstrItinData<IIC_LdStSTFD    , [InstrStage<4, [SLU]>]>,
  InstrItinData<IIC_LdStSTFDU   , [InstrStage<4, [SLU]>]>,  
  InstrItinData<IIC_LdStLD      , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_LdStLDU     , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_LdStLDARX   , [InstrStage<11, [SLU]>]>,
  InstrItinData<IIC_LdStLFD     , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_LdStLFDU    , [InstrStage<5, [SLU]>]>,
  InstrItinData<IIC_LdStLHA     , [InstrStage<5, [SLU]>]>,
  InstrItinData<IIC_LdStLHAU    , [InstrStage<5, [SLU]>]>,  
  InstrItinData<IIC_LdStLMW     , [InstrStage<64, [SLU]>]>,
  InstrItinData<IIC_LdStLVecX   , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_LdStLWA     , [InstrStage<5, [SLU]>]>,
  InstrItinData<IIC_LdStLWARX   , [InstrStage<11, [SLU]>]>,
  InstrItinData<IIC_LdStSLBIA   , [InstrStage<40, [SLU]>]>, // needs work
  InstrItinData<IIC_LdStSLBIE   , [InstrStage<2, [SLU]>]>,
  InstrItinData<IIC_LdStSTD     , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_LdStSTDU    , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_LdStSTDCX   , [InstrStage<11, [SLU]>]>,
  InstrItinData<IIC_LdStSTVEBX  , [InstrStage<5, [SLU]>]>,
  InstrItinData<IIC_LdStSTWCX   , [InstrStage<11, [SLU]>]>,
  InstrItinData<IIC_LdStSync    , [InstrStage<35, [SLU]>]>,
  InstrItinData<IIC_SprISYNC    , [InstrStage<40, [SLU]>]>, // needs work
  InstrItinData<IIC_SprMFSR     , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_SprMTMSR    , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_SprMTSR     , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_SprTLBSYNC  , [InstrStage<3, [SLU]>]>,
  InstrItinData<IIC_SprMFCR     , [InstrStage<2, [IU2]>]>,
  InstrItinData<IIC_SprMFMSR    , [InstrStage<3, [IU2]>]>,
  InstrItinData<IIC_SprMFSPR    , [InstrStage<3, [IU2]>]>,
  InstrItinData<IIC_SprMFTB     , [InstrStage<10, [IU2]>]>,
  InstrItinData<IIC_SprMTSPR    , [InstrStage<8, [IU2]>]>,
  InstrItinData<IIC_SprSC       , [InstrStage<1, [IU2]>]>,
  InstrItinData<IIC_FPGeneral   , [InstrStage<6, [FPU1, FPU2]>]>,
  InstrItinData<IIC_FPAddSub    , [InstrStage<6, [FPU1, FPU2]>]>,
  InstrItinData<IIC_FPCompare   , [InstrStage<8, [FPU1, FPU2]>]>,
  InstrItinData<IIC_FPDivD      , [InstrStage<33, [FPU1, FPU2]>]>,
  InstrItinData<IIC_FPDivS      , [InstrStage<33, [FPU1, FPU2]>]>,
  InstrItinData<IIC_FPFused     , [InstrStage<6, [FPU1, FPU2]>]>,
  InstrItinData<IIC_FPRes       , [InstrStage<6, [FPU1, FPU2]>]>,
  InstrItinData<IIC_FPSqrt      , [InstrStage<40, [FPU1, FPU2]>]>,
  InstrItinData<IIC_VecGeneral  , [InstrStage<2, [VIU1]>]>,
  InstrItinData<IIC_VecFP       , [InstrStage<8, [VFPU]>]>,
  InstrItinData<IIC_VecFPCompare, [InstrStage<2, [VFPU]>]>,
  InstrItinData<IIC_VecComplex  , [InstrStage<5, [VIU2]>]>,
  InstrItinData<IIC_VecPerm     , [InstrStage<3, [VPU]>]>,
  InstrItinData<IIC_VecFPRound  , [InstrStage<8, [VFPU]>]>,
  InstrItinData<IIC_VecVSL      , [InstrStage<2, [VIU1]>]>,
  InstrItinData<IIC_VecVSR      , [InstrStage<3, [VPU]>]>
]>;

// ===---------------------------------------------------------------------===//
// e5500 machine model for scheduling and other instruction cost heuristics.

def G5Model : SchedMachineModel {
  let IssueWidth = 4;  // 4 (non-branch) instructions are dispatched per cycle.
  let MinLatency = 0;  // Out-of-order dispatch.
  let LoadLatency = 3; // Optimistic load latency assuming bypass.
                       // This is overriden by OperandCycles if the
                       // Itineraries are queried instead.
  let MispredictPenalty = 16;

  let Itineraries = G5Itineraries;
}

