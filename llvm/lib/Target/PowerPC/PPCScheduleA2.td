//===- PPCScheduleA2.td - PPC A2 Scheduling Definitions --*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//

// Primary reference:
// A2 Processor User's Manual.
// IBM (as updated in) 2010.

//===----------------------------------------------------------------------===//
// Functional units on the PowerPC A2 chip sets
//
def XU     : FuncUnit; // XU pipeline
def FU     : FuncUnit; // FI pipeline

//
// This file defines the itinerary class data for the PPC A2 processor.
//
//===----------------------------------------------------------------------===//


def PPCA2Itineraries : ProcessorItineraries<
  [XU, FU], [], [
  InstrItinData<IntSimple   , [InstrStage<1, [XU]>],
                              [1, 1, 1]>,
  InstrItinData<IntGeneral  , [InstrStage<1, [XU]>],
                              [2, 1, 1]>,
  InstrItinData<IntCompare  , [InstrStage<1, [XU]>],
                              [2, 1, 1]>,
  InstrItinData<IntDivW     , [InstrStage<1, [XU]>],
                              [39, 1, 1]>,
  InstrItinData<IntDivD     , [InstrStage<1, [XU]>],
                              [71, 1, 1]>,
  InstrItinData<IntMulHW    , [InstrStage<1, [XU]>],
                              [5, 1, 1]>,
  InstrItinData<IntMulHWU   , [InstrStage<1, [XU]>],
                              [5, 1, 1]>,
  InstrItinData<IntMulLI    , [InstrStage<1, [XU]>],
                              [6, 1, 1]>,
  InstrItinData<IntRotate   , [InstrStage<1, [XU]>],
                              [2, 1, 1]>,
  InstrItinData<IntRotateD  , [InstrStage<1, [XU]>],
                              [2, 1, 1]>,
  InstrItinData<IntRotateDI , [InstrStage<1, [XU]>],
                              [2, 1, 1]>,
  InstrItinData<IntShift    , [InstrStage<1, [XU]>],
                              [2, 1, 1]>,
  InstrItinData<IntTrapW    , [InstrStage<1, [XU]>],
                              [2, 1]>,
  InstrItinData<IntTrapD    , [InstrStage<1, [XU]>],
                              [2, 1]>,
  InstrItinData<BrB         , [InstrStage<1, [XU]>],
                              [6, 1, 1]>,
  InstrItinData<BrCR        , [InstrStage<1, [XU]>],
                              [1, 1, 1]>,
  InstrItinData<BrMCR       , [InstrStage<1, [XU]>],
                              [5, 1, 1]>,
  InstrItinData<BrMCRX      , [InstrStage<1, [XU]>],
                              [1, 1, 1]>,
  InstrItinData<LdStDCBA    , [InstrStage<1, [XU]>],
                              [1, 1, 1]>,
  InstrItinData<LdStDCBF    , [InstrStage<1, [XU]>],
                              [1, 1, 1]>,
  InstrItinData<LdStDCBI    , [InstrStage<1, [XU]>],
                              [1, 1, 1]>,
  InstrItinData<LdStLoad    , [InstrStage<1, [XU]>],
                              [6, 1, 1]>,
  InstrItinData<LdStLoadUpd , [InstrStage<1, [XU]>],
                              [6, 8, 1, 1]>,
  InstrItinData<LdStLDU     , [InstrStage<1, [XU]>],
                              [6, 1, 1]>,
  InstrItinData<LdStStore   , [InstrStage<1, [XU]>],
                              [1, 1, 1]>,
  InstrItinData<LdStStoreUpd, [InstrStage<1, [XU]>],
                              [2, 1, 1, 1]>,
  InstrItinData<LdStICBI,     [InstrStage<1, [XU]>],
                              [16, 1, 1]>,
  InstrItinData<LdStSTFD    , [InstrStage<1, [XU]>],
                              [1, 1, 1]>,
  InstrItinData<LdStSTFDU   , [InstrStage<1, [XU]>],
                              [2, 1, 1, 1]>,
  InstrItinData<LdStLFD     , [InstrStage<1, [XU]>],
                              [7, 1, 1]>,
  InstrItinData<LdStLFDU    , [InstrStage<1, [XU]>],
                              [7, 9, 1, 1]>,
  InstrItinData<LdStLHA     , [InstrStage<1, [XU]>],
                              [6, 1, 1]>,
  InstrItinData<LdStLHAU    , [InstrStage<1, [XU]>],
                              [6, 8, 1, 1]>,
  InstrItinData<LdStLWARX   , [InstrStage<1, [XU]>],
                              [82, 1, 1]>, // L2 latency
  InstrItinData<LdStSTD     , [InstrStage<1, [XU]>],
                              [1, 1, 1]>,
  InstrItinData<LdStSTDU    , [InstrStage<1, [XU]>],
                              [2, 1, 1, 1]>,
  InstrItinData<LdStSTDCX   , [InstrStage<1, [XU]>],
                              [82, 1, 1]>, // L2 latency
  InstrItinData<LdStSTWCX   , [InstrStage<1, [XU]>],
                              [82, 1, 1]>, // L2 latency
  InstrItinData<LdStSync    , [InstrStage<1, [XU]>],
                              [6]>,
  InstrItinData<SprISYNC    , [InstrStage<1, [XU]>],
                              [16]>,
  InstrItinData<SprMTMSR    , [InstrStage<1, [XU]>],
                              [16, 1]>,
  InstrItinData<SprMFCR     , [InstrStage<1, [XU]>],
                              [6, 1]>,
  InstrItinData<SprMFMSR    , [InstrStage<1, [XU]>],
                              [4, 1]>,
  InstrItinData<SprMFSPR    , [InstrStage<1, [XU]>],
                              [6, 1]>,
  InstrItinData<SprMFTB     , [InstrStage<1, [XU]>],
                              [4, 1]>,
  InstrItinData<SprMTSPR    , [InstrStage<1, [XU]>],
                              [6, 1]>,
  InstrItinData<SprRFI      , [InstrStage<1, [XU]>],
                              [16]>,
  InstrItinData<SprSC       , [InstrStage<1, [XU]>],
                              [16]>,
  InstrItinData<FPGeneral   , [InstrStage<1, [FU]>],
                              [6, 1, 1]>,
  InstrItinData<FPAddSub    , [InstrStage<1, [FU]>],
                              [6, 1, 1]>,
  InstrItinData<FPCompare   , [InstrStage<1, [FU]>],
                              [5, 1, 1]>,
  InstrItinData<FPDivD      , [InstrStage<1, [FU]>],
                              [72, 1, 1]>,
  InstrItinData<FPDivS      , [InstrStage<1, [FU]>],
                              [59, 1, 1]>,
  InstrItinData<FPSqrt      , [InstrStage<1, [FU]>],
                              [69, 1, 1]>,
  InstrItinData<FPFused     , [InstrStage<1, [FU]>],
                              [6, 1, 1, 1]>,
  InstrItinData<FPRes       , [InstrStage<1, [FU]>],
                              [6, 1]>
]>;

// ===---------------------------------------------------------------------===//
// A2 machine model for scheduling and other instruction cost heuristics.

def PPCA2Model : SchedMachineModel {
  let IssueWidth = 1;  // 2 micro-ops are dispatched per cycle.
  let MinLatency = -1; // OperandCycles are interpreted as MinLatency.
  let LoadLatency = 6; // Optimistic load latency assuming bypass.
                       // This is overriden by OperandCycles if the
                       // Itineraries are queried instead.
  let MispredictPenalty = 13;

  let Itineraries = PPCA2Itineraries;
}

