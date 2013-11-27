//===-- PPCScheduleE500mc.td - e5500 Scheduling Defs -------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the itinerary class data for the Freescale e5500 64-bit 
// Power processor.
// 
// All information is derived from the "e5500 Core Reference Manual",
// Freescale Document Number e5500RM, Rev. 1, 03/2012.
//
//===----------------------------------------------------------------------===//
// Relevant functional units in the Freescale e5500 core
// (These are the same as for the e500mc)
//
//  * Decode & Dispatch
//    Can dispatch up to 2 instructions per clock cycle to either the GPR Issue
//    queues (GIQx), FP Issue Queue (FIQ), or Branch issue queue (BIQ).
// def DIS0 : FuncUnit;
// def DIS1 : FuncUnit;

//  * Execute
//    6 pipelined execution units: SFX0, SFX1, BU, FPU, LSU, CFX.
//    The CFX has a bypass path, allowing non-divide instructions to execute 
//    while a divide instruction is being executed.
// def SFX0  : FuncUnit; // Simple unit 0
// def SFX1  : FuncUnit; // Simple unit 1
// def BU    : FuncUnit; // Branch unit
// def CFX_DivBypass 
//           : FuncUnit; // CFX divide bypass path
// def CFX_0 : FuncUnit; // CFX pipeline stage 0

def CFX_1 : FuncUnit; // CFX pipeline stage 1 

// def LSU_0 : FuncUnit; // LSU pipeline
// def FPU_0 : FuncUnit; // FPU pipeline

// def CR_Bypass : Bypass;

def PPCE5500Itineraries : ProcessorItineraries<
  [DIS0, DIS1, SFX0, SFX1, BU, CFX_DivBypass, CFX_0, CFX_1,
   LSU_0, FPU_0],
  [CR_Bypass, GPR_Bypass, FPR_Bypass], [
  InstrItinData<IIC_IntSimple,   [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [SFX0, SFX1]>],
                                 [5, 2, 2], // Latency = 1
                                 [GPR_Bypass, GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IIC_IntGeneral,  [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [SFX0, SFX1]>],
                                 [5, 2, 2], // Latency = 1
                                 [GPR_Bypass, GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IIC_IntCompare,  [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [SFX0, SFX1]>],
                                 [6, 2, 2], // Latency = 1 or 2
                                 [CR_Bypass, GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IIC_IntDivD,     [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [CFX_0], 0>,
                                  InstrStage<26, [CFX_DivBypass]>],
                                 [30, 2, 2], // Latency= 4..26, Repeat rate= 4..26
                                 [GPR_Bypass, GPR_Bypass, GPR_Bypass]>,                              
  InstrItinData<IIC_IntDivW,     [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [CFX_0], 0>,
                                  InstrStage<16, [CFX_DivBypass]>],
                                 [20, 2, 2], // Latency= 4..16, Repeat rate= 4..16
                                 [GPR_Bypass, GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IIC_IntMFFS,     [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [FPU_0]>],
                                 [11], // Latency = 7, Repeat rate = 1
                                 [FPR_Bypass]>,
  InstrItinData<IIC_IntMTFSB0,   [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<7, [FPU_0]>],
                                 [11, 2, 2], // Latency = 7, Repeat rate = 7
                                 [NoBypass, NoBypass, NoBypass]>,
  InstrItinData<IIC_IntMulHD,    [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [CFX_0], 0>,
                                  InstrStage<2, [CFX_1]>],
                                 [9, 2, 2], // Latency = 4..7, Repeat rate = 2..4
                                 [GPR_Bypass, GPR_Bypass, GPR_Bypass]>,                              
  InstrItinData<IIC_IntMulHW,    [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [CFX_0], 0>,
                                  InstrStage<1, [CFX_1]>],
                                 [8, 2, 2], // Latency = 4, Repeat rate = 1
                                 [GPR_Bypass, GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IIC_IntMulHWU,   [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [CFX_0], 0>,
                                  InstrStage<1, [CFX_1]>],
                                 [8, 2, 2], // Latency = 4, Repeat rate = 1
                                 [GPR_Bypass, GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IIC_IntMulLI,    [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [CFX_0], 0>,
                                  InstrStage<2, [CFX_1]>],
                                 [8, 2, 2], // Latency = 4 or 5, Repeat = 2
                                 [GPR_Bypass, GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IIC_IntRotate,   [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [SFX0, SFX1]>],
                                 [5, 2, 2], // Latency = 1
                                 [GPR_Bypass, GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IIC_IntRotateD,  [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<2, [SFX0, SFX1]>],
                                 [6, 2, 2], // Latency = 2, Repeat rate = 2
                                 [GPR_Bypass, GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IIC_IntRotateDI, [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [SFX0, SFX1]>],
                                 [5, 2, 2], // Latency = 1, Repeat rate = 1
                                 [GPR_Bypass, GPR_Bypass, GPR_Bypass]>,                                                            
  InstrItinData<IIC_IntShift,    [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<2, [SFX0, SFX1]>],
                                 [6, 2, 2], // Latency = 2, Repeat rate = 2
                                 [GPR_Bypass, GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IIC_IntTrapW,    [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<2, [SFX0]>],
                                 [6, 2], // Latency = 2, Repeat rate = 2
                                 [GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IIC_BrB,         [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [BU]>],
                                 [5, 2], // Latency = 1
                                 [NoBypass, GPR_Bypass]>,
  InstrItinData<IIC_BrCR,        [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [BU]>],
                                 [5, 2, 2], // Latency = 1
                                 [CR_Bypass, CR_Bypass, CR_Bypass]>,
  InstrItinData<IIC_BrMCR,       [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [BU]>],
                                 [5, 2], // Latency = 1
                                 [CR_Bypass, CR_Bypass]>,
  InstrItinData<IIC_BrMCRX,      [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [CFX_0]>],
                                 [5, 2, 2], // Latency = 1
                                 [CR_Bypass, GPR_Bypass]>,
  InstrItinData<IIC_LdStDCBA,    [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [LSU_0]>],
                                 [7, 2], // Latency = 3, Repeat rate = 1
                                 [GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IIC_LdStDCBF,    [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [LSU_0]>],
                                 [7, 2], // Latency = 3, Repeat rate = 1
                                 [GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IIC_LdStDCBI,    [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [LSU_0]>],
                                 [7, 2], // Latency = 3, Repeat rate = 1
                                 [GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IIC_LdStLoad,    [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [LSU_0]>],
                                 [7, 2], // Latency = 3
                                 [GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IIC_LdStLoadUpd, [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [SFX0, SFX1], 0>,
                                  InstrStage<1, [LSU_0]>],
                                 [7, 2], // Latency = 3, Repeat rate = 1
                                 [GPR_Bypass, GPR_Bypass],
                                 2>, // 2 micro-ops
  InstrItinData<IIC_LdStLD,      [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [LSU_0]>],
                                 [7, 2], // Latency = 3, Repeat rate = 1
                                 [GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IIC_LdStLDARX,   [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<3, [LSU_0]>],
                                 [7, 2], // Latency = 3, Repeat rate = 3
                                 [GPR_Bypass, GPR_Bypass]>,                              
  InstrItinData<IIC_LdStLDU,     [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [SFX0, SFX1], 0>,
                                  InstrStage<1, [LSU_0]>],
                                 [7, 2], // Latency = 3, Repeat rate = 1
                                 [GPR_Bypass, GPR_Bypass],
                                 2>, // 2 micro-ops
  InstrItinData<IIC_LdStStore,   [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [LSU_0]>],
                                 [7, 2], // Latency = 3, Repeat rate = 1
                                 [NoBypass, GPR_Bypass]>,
  InstrItinData<IIC_LdStStoreUpd,[InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [SFX0, SFX1], 0>,
                                  InstrStage<1, [LSU_0]>],
                                 [7, 2], // Latency = 3, Repeat rate = 1
                                 [NoBypass, GPR_Bypass],
                                 2>, // 2 micro-ops                              
  InstrItinData<IIC_LdStICBI,    [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [LSU_0]>],
                                 [7, 2], // Latency = 3, Repeat rate = 1
                                 [NoBypass, GPR_Bypass]>,
  InstrItinData<IIC_LdStSTFD,    [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [LSU_0]>],
                                 [7, 2, 2], // Latency = 3, Repeat rate = 1
                                 [GPR_Bypass, GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IIC_LdStSTFDU,   [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [SFX0, SFX1], 0>,
                                  InstrStage<1, [LSU_0]>],
                                 [7, 2, 2], // Latency = 3, Repeat rate = 1
                                 [GPR_Bypass, GPR_Bypass, GPR_Bypass],
                                 2>, // 2 micro-ops                              
  InstrItinData<IIC_LdStLFD,     [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [LSU_0]>],
                                 [8, 2, 2], // Latency = 4, Repeat rate = 1
                                 [FPR_Bypass, GPR_Bypass, GPR_Bypass],
                                 2>, // 2 micro-ops
  InstrItinData<IIC_LdStLFDU,    [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [SFX0, SFX1], 0>,
                                  InstrStage<1, [LSU_0]>],
                                 [8, 2, 2], // Latency = 4, Repeat rate = 1
                                 [FPR_Bypass, GPR_Bypass, GPR_Bypass],
                                 2>, // 2 micro-ops
  InstrItinData<IIC_LdStLHA,     [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [LSU_0]>],
                                 [7, 2], // Latency = 3
                                 [GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IIC_LdStLHAU,    [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [SFX0, SFX1], 0>,
                                  InstrStage<1, [LSU_0]>],
                                 [7, 2], // Latency = 3, Repeat rate = 1
                                 [GPR_Bypass, GPR_Bypass],
                                 2>, // 2 micro-ops                              
  InstrItinData<IIC_LdStLMW,     [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<4, [LSU_0]>],
                                 [8, 2], // Latency = r+3, Repeat rate = r+3
                                 [NoBypass, GPR_Bypass]>,
  InstrItinData<IIC_LdStLWARX,   [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<3, [LSU_0]>],
                                 [7, 2, 2], // Latency = 3, Repeat rate = 3
                                 [GPR_Bypass, GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IIC_LdStSTD,     [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [LSU_0]>],
                                 [7, 2], // Latency = 3, Repeat rate = 1                              
                                 [NoBypass, GPR_Bypass]>,
  InstrItinData<IIC_LdStSTDCX,   [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [LSU_0]>],
                                 [7, 2], // Latency = 3, Repeat rate = 1                              
                                 [NoBypass, GPR_Bypass]>,                              
  InstrItinData<IIC_LdStSTDU,    [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [SFX0, SFX1], 0>,
                                  InstrStage<1, [LSU_0]>],
                                 [7, 2], // Latency = 3, Repeat rate = 1
                                 [NoBypass, GPR_Bypass],
                                 2>, // 2 micro-ops                              
  InstrItinData<IIC_LdStSTWCX,   [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [LSU_0]>],
                                 [7, 2], // Latency = 3, Repeat rate = 1
                                 [NoBypass, GPR_Bypass]>,
  InstrItinData<IIC_LdStSync,    [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [LSU_0]>]>,
  InstrItinData<IIC_SprMTMSR,    [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<2, [CFX_0]>],
                                 [6, 2], // Latency = 2, Repeat rate = 4
                                 [GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IIC_SprTLBSYNC,  [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [LSU_0], 0>]>,
  InstrItinData<IIC_SprMFCR,     [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<5, [CFX_0]>],
                                 [9, 2], // Latency = 5, Repeat rate = 5
                                 [GPR_Bypass, CR_Bypass]>,
  InstrItinData<IIC_SprMFMSR,    [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<4, [SFX0]>],
                                 [8, 2], // Latency = 4, Repeat rate = 4
                                 [GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IIC_SprMFSPR,    [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [CFX_0]>],
                                 [5], // Latency = 1, Repeat rate = 1
                                 [GPR_Bypass]>,
  InstrItinData<IIC_SprMFTB,     [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<4, [CFX_0]>],
                                 [8, 2], // Latency = 4, Repeat rate = 4
                                 [NoBypass, GPR_Bypass]>,
  InstrItinData<IIC_SprMTSPR,    [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [SFX0, SFX1]>],
                                 [5], // Latency = 1, Repeat rate = 1
                                 [GPR_Bypass]>,
  InstrItinData<IIC_FPGeneral,   [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [FPU_0]>],
                                 [11, 2, 2], // Latency = 7, Repeat rate = 1 
                                 [FPR_Bypass, FPR_Bypass, FPR_Bypass]>,
  InstrItinData<IIC_FPAddSub,    [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [FPU_0]>],
                                 [11, 2, 2], // Latency = 7, Repeat rate = 1 
                                 [FPR_Bypass, FPR_Bypass, FPR_Bypass]>,                              
  InstrItinData<IIC_FPCompare,   [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [FPU_0]>],
                                 [11, 2, 2], // Latency = 7, Repeat rate = 1
                                 [CR_Bypass, FPR_Bypass, FPR_Bypass]>,
  InstrItinData<IIC_FPDivD,      [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<31, [FPU_0]>],
                                 [39, 2, 2], // Latency = 35, Repeat rate = 31
                                 [FPR_Bypass, FPR_Bypass, FPR_Bypass]>,
  InstrItinData<IIC_FPDivS,      [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<16, [FPU_0]>],
                                 [24, 2, 2], // Latency = 20, Repeat rate = 16 
                                 [FPR_Bypass, FPR_Bypass, FPR_Bypass]>,
  InstrItinData<IIC_FPFused,     [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<1, [FPU_0]>],
                                 [11, 2, 2, 2], // Latency = 7, Repeat rate = 1
                                 [FPR_Bypass, FPR_Bypass, FPR_Bypass, FPR_Bypass]>,
  InstrItinData<IIC_FPRes,       [InstrStage<1, [DIS0, DIS1], 0>,
                                  InstrStage<2, [FPU_0]>],
                                 [12, 2], // Latency = 8, Repeat rate = 2
                                 [FPR_Bypass, FPR_Bypass]>
]>;

// ===---------------------------------------------------------------------===//
// e5500 machine model for scheduling and other instruction cost heuristics.

def PPCE5500Model : SchedMachineModel {
  let IssueWidth = 2;  // 2 micro-ops are dispatched per cycle.
  let MinLatency = -1; // OperandCycles are interpreted as MinLatency.
  let LoadLatency = 6; // Optimistic load latency assuming bypass.
                       // This is overriden by OperandCycles if the
                       // Itineraries are queried instead.

  let Itineraries = PPCE5500Itineraries;
}
