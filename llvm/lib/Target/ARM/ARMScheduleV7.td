//===- ARMScheduleV7.td - ARM v7 Scheduling Definitions ----*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the itinerary class data for the ARM v7 processors.
//
//===----------------------------------------------------------------------===//

//
// Scheduling information derived from "Cortex-A8 Technical Reference Manual".
//
// Dual issue pipeline represented by FU_Pipe0 | FU_Pipe1
//
def CortexA8Itineraries : ProcessorItineraries<[

  // Two fully-pipelined integer ALU pipelines
  //
  // No operand cycles
  InstrItinData<IIC_iALUx    , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>]>,
  //
  // Binary Instructions that produce a result
  InstrItinData<IIC_iALUi    , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>], [2, 2]>,
  InstrItinData<IIC_iALUr    , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>], [2, 2, 2]>,
  InstrItinData<IIC_iALUsi   , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>], [2, 2, 1]>,
  InstrItinData<IIC_iALUsr   , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>], [2, 2, 1, 1]>,
  //
  // Unary Instructions that produce a result
  InstrItinData<IIC_iUNAr    , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>], [2, 2]>,
  InstrItinData<IIC_iUNAsi   , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>], [2, 1]>,
  InstrItinData<IIC_iUNAsr   , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>], [2, 1, 1]>,
  //
  // Compare instructions
  InstrItinData<IIC_iCMPi    , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>], [2]>,
  InstrItinData<IIC_iCMPr    , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>], [2, 2]>,
  InstrItinData<IIC_iCMPsi   , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>], [2, 1]>,
  InstrItinData<IIC_iCMPsr   , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>], [2, 1, 1]>,
  //
  // Move instructions, unconditional
  InstrItinData<IIC_iMOVi    , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>], [1]>,
  InstrItinData<IIC_iMOVr    , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>], [1, 1]>,
  InstrItinData<IIC_iMOVsi   , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>], [1, 1]>,
  InstrItinData<IIC_iMOVsr   , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>], [1, 1, 1]>,
  //
  // Move instructions, conditional
  InstrItinData<IIC_iCMOVi   , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>], [2]>,
  InstrItinData<IIC_iCMOVr   , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>], [2, 1]>,
  InstrItinData<IIC_iCMOVsi  , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>], [2, 1]>,
  InstrItinData<IIC_iCMOVsr  , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>], [2, 1, 1]>,

  // Integer multiply pipeline
  // Result written in E5, but that is relative to the last cycle of multicycle,
  // so we use 6 for those cases
  //
  InstrItinData<IIC_iMUL16   , [InstrStage<1, [FU_Pipe0]>], [5, 1, 1]>,
  InstrItinData<IIC_iMAC16   , [InstrStage<1, [FU_Pipe1], 0>, 
                                InstrStage<2, [FU_Pipe0]>], [6, 1, 1, 4]>,
  InstrItinData<IIC_iMUL32   , [InstrStage<1, [FU_Pipe1], 0>, 
                                InstrStage<2, [FU_Pipe0]>], [6, 1, 1]>,
  InstrItinData<IIC_iMAC32   , [InstrStage<1, [FU_Pipe1], 0>, 
                                InstrStage<2, [FU_Pipe0]>], [6, 1, 1, 4]>,
  InstrItinData<IIC_iMUL64   , [InstrStage<2, [FU_Pipe1], 0>, 
                                InstrStage<3, [FU_Pipe0]>], [6, 6, 1, 1]>,
  InstrItinData<IIC_iMAC64   , [InstrStage<2, [FU_Pipe1], 0>, 
                                InstrStage<3, [FU_Pipe0]>], [6, 6, 1, 1]>,
  
  // Integer load pipeline
  //
  // loads have an extra cycle of latency, but are fully pipelined
  // use FU_Issue to enforce the 1 load/store per cycle limit
  //
  // Immediate offset
  InstrItinData<IIC_iLoadi   , [InstrStage<1, [FU_Issue], 0>,
                                InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                                InstrStage<1, [FU_LdSt0]>], [3, 1]>,
  //
  // Register offset
  InstrItinData<IIC_iLoadr   , [InstrStage<1, [FU_Issue], 0>,
                                InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                                InstrStage<1, [FU_LdSt0]>], [3, 1, 1]>,
  //
  // Scaled register offset, issues over 2 cycles
  InstrItinData<IIC_iLoadsi  , [InstrStage<2, [FU_Issue], 0>,
                                InstrStage<1, [FU_Pipe0], 0>,
                                InstrStage<1, [FU_Pipe1]>,
                                InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                                InstrStage<1, [FU_LdSt0]>], [4, 1, 1]>,
  //
  // Immediate offset with update
  InstrItinData<IIC_iLoadiu  , [InstrStage<1, [FU_Issue], 0>,
                                InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                                InstrStage<1, [FU_LdSt0]>], [3, 2, 1]>,
  //
  // Register offset with update
  InstrItinData<IIC_iLoadru  , [InstrStage<1, [FU_Issue], 0>,
                                InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                                InstrStage<1, [FU_LdSt0]>], [3, 2, 1, 1]>,
  //
  // Scaled register offset with update, issues over 2 cycles
  InstrItinData<IIC_iLoadsiu , [InstrStage<2, [FU_Issue], 0>,
                                InstrStage<1, [FU_Pipe0], 0>,
                                InstrStage<1, [FU_Pipe1]>,
                                InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                                InstrStage<1, [FU_LdSt0]>], [4, 3, 1, 1]>,
  //
  // Load multiple
  InstrItinData<IIC_iLoadm   , [InstrStage<2, [FU_Issue], 0>,
                                InstrStage<2, [FU_Pipe0], 0>,
                                InstrStage<2, [FU_Pipe1]>,
                                InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                                InstrStage<1, [FU_LdSt0]>]>,

  // Integer store pipeline
  //
  // use FU_Issue to enforce the 1 load/store per cycle limit
  //
  // Immediate offset
  InstrItinData<IIC_iStorei  , [InstrStage<1, [FU_Issue], 0>,
                                InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                                InstrStage<1, [FU_LdSt0]>], [3, 1]>,
  //
  // Register offset
  InstrItinData<IIC_iStorer  , [InstrStage<1, [FU_Issue], 0>,
                                InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                                InstrStage<1, [FU_LdSt0]>], [3, 1, 1]>,
  //
  // Scaled register offset, issues over 2 cycles
  InstrItinData<IIC_iStoresi , [InstrStage<2, [FU_Issue], 0>,
                                InstrStage<1, [FU_Pipe0], 0>,
                                InstrStage<1, [FU_Pipe1]>,
                                InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                                InstrStage<1, [FU_LdSt0]>], [3, 1, 1]>,
  //
  // Immediate offset with update
  InstrItinData<IIC_iStoreiu , [InstrStage<1, [FU_Issue], 0>,
                                InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                                InstrStage<1, [FU_LdSt0]>], [2, 3, 1]>,
  //
  // Register offset with update
  InstrItinData<IIC_iStoreru  , [InstrStage<1, [FU_Issue], 0>,
                                InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                                InstrStage<1, [FU_LdSt0]>], [2, 3, 1, 1]>,
  //
  // Scaled register offset with update, issues over 2 cycles
  InstrItinData<IIC_iStoresiu, [InstrStage<2, [FU_Issue], 0>,
                                InstrStage<1, [FU_Pipe0], 0>,
                                InstrStage<1, [FU_Pipe1]>,
                                InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                                InstrStage<1, [FU_LdSt0]>], [3, 3, 1, 1]>,
  //
  // Store multiple
  InstrItinData<IIC_iStorem  , [InstrStage<2, [FU_Issue], 0>,
                                InstrStage<2, [FU_Pipe0], 0>,
                                InstrStage<2, [FU_Pipe1]>,
                                InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                                InstrStage<1, [FU_LdSt0]>]>,
  
  // Branch
  //
  // no delay slots, so the latency of a branch is unimportant
  InstrItinData<IIC_Br      , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>]>,

  // VFP
  // Issue through integer pipeline, and execute in NEON unit. We assume
  // RunFast mode so that NFP pipeline is used for single-precision when
  // possible.
  //
  // FP Special Register to Integer Register File Move
  InstrItinData<IIC_fpSTAT , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                              InstrStage<1, [FU_NLSPipe], 1>]>,
  //
  // Integer to Single-Precision FP Register File Move
  InstrItinData<IIC_fpMOVIS , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NLSPipe], 1>]>,
  //
  // Integer to Double-Precision FP Register File Move
  InstrItinData<IIC_fpMOVID , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NLSPipe], 1>]>,
  //
  // Single-Precision FP to Integer Register File Move
  InstrItinData<IIC_fpMOVSI , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NLSPipe], 1>], [20, 1]>,
  //
  // Double-Precision FP to Integer Register File Move
  InstrItinData<IIC_fpMOVDI , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NLSPipe], 1>], [20, 20, 1]>,
  //
  // Single-precision FP Unary
  InstrItinData<IIC_fpUNA32 , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>], [7, 1]>,
  //
  // Double-precision FP Unary
  InstrItinData<IIC_fpUNA64 , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<4, [FU_NPipe], 0>,
                               InstrStage<4, [FU_NLSPipe]>]>,
  //
  // Single-precision FP Compare
  InstrItinData<IIC_fpCMP32 , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>], [7, 1]>,
  //
  // Double-precision FP Compare
  InstrItinData<IIC_fpCMP64 , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<4, [FU_NPipe], 0>,
                               InstrStage<4, [FU_NLSPipe]>]>,
  //
  // Single to Double FP Convert
  InstrItinData<IIC_fpCVTSD , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<7, [FU_NPipe], 0>,
                               InstrStage<7, [FU_NLSPipe]>]>,
  //
  // Double to Single FP Convert
  InstrItinData<IIC_fpCVTDS , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<5, [FU_NPipe], 0>,
                               InstrStage<5, [FU_NLSPipe]>]>,
  //
  // Single-Precision FP to Integer Convert
  InstrItinData<IIC_fpCVTSI , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>], [7, 1]>,
  //
  // Double-Precision FP to Integer Convert
  InstrItinData<IIC_fpCVTDI , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<8, [FU_NPipe], 0>,
                               InstrStage<8, [FU_NLSPipe]>]>,
  //
  // Integer to Single-Precision FP Convert
  InstrItinData<IIC_fpCVTIS , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>], [7, 1]>,
  //
  // Integer to Double-Precision FP Convert
  InstrItinData<IIC_fpCVTID , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<8, [FU_NPipe], 0>,
                               InstrStage<8, [FU_NLSPipe]>]>,
  //
  // Single-precision FP ALU
  InstrItinData<IIC_fpALU32 , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>], [7, 1]>,
  //
  // Double-precision FP ALU
  InstrItinData<IIC_fpALU64 , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<9, [FU_NPipe], 0>,
                               InstrStage<9, [FU_NLSPipe]>]>,
  //
  // Single-precision FP Multiply
  InstrItinData<IIC_fpMUL32 , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>], [7, 1]>,
  //
  // Double-precision FP Multiply
  InstrItinData<IIC_fpMUL64 , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<11, [FU_NPipe], 0>,
                               InstrStage<11, [FU_NLSPipe]>]>,
  //
  // Single-precision FP MAC
  InstrItinData<IIC_fpMAC32 , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>], [7, 1]>,
  //
  // Double-precision FP MAC
  InstrItinData<IIC_fpMAC64 , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<19, [FU_NPipe], 0>,
                               InstrStage<19, [FU_NLSPipe]>]>,
  //
  // Single-precision FP DIV
  InstrItinData<IIC_fpDIV32 , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<20, [FU_NPipe], 0>,
                               InstrStage<20, [FU_NLSPipe]>]>,
  //
  // Double-precision FP DIV
  InstrItinData<IIC_fpDIV64 , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<29, [FU_NPipe], 0>,
                               InstrStage<29, [FU_NLSPipe]>]>,
  //
  // Single-precision FP SQRT
  InstrItinData<IIC_fpSQRT32, [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<19, [FU_NPipe], 0>,
                               InstrStage<19, [FU_NLSPipe]>]>,
  //
  // Double-precision FP SQRT
  InstrItinData<IIC_fpSQRT64, [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<29, [FU_NPipe], 0>,
                               InstrStage<29, [FU_NLSPipe]>]>,
  //
  // Single-precision FP Load
  // use FU_Issue to enforce the 1 load/store per cycle limit
  InstrItinData<IIC_fpLoad32, [InstrStage<1, [FU_Issue], 0>, 
                               InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_LdSt0], 0>,
                               InstrStage<1, [FU_NLSPipe]>]>,
  //
  // Double-precision FP Load
  // use FU_Issue to enforce the 1 load/store per cycle limit
  InstrItinData<IIC_fpLoad64, [InstrStage<2, [FU_Issue], 0>, 
                               InstrStage<1, [FU_Pipe0], 0>,
                               InstrStage<1, [FU_Pipe1]>,
                               InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_LdSt0], 0>,
                               InstrStage<1, [FU_NLSPipe]>]>,
  //
  // FP Load Multiple
  // use FU_Issue to enforce the 1 load/store per cycle limit
  InstrItinData<IIC_fpLoadm,  [InstrStage<3, [FU_Issue], 0>, 
                               InstrStage<2, [FU_Pipe0], 0>,
                               InstrStage<2, [FU_Pipe1]>,
                               InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_LdSt0], 0>,
                               InstrStage<1, [FU_NLSPipe]>]>,
  //
  // Single-precision FP Store
  // use FU_Issue to enforce the 1 load/store per cycle limit
  InstrItinData<IIC_fpStore32,[InstrStage<1, [FU_Issue], 0>, 
                               InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_LdSt0], 0>,
                               InstrStage<1, [FU_NLSPipe]>]>,
  //
  // Double-precision FP Store
  // use FU_Issue to enforce the 1 load/store per cycle limit
  InstrItinData<IIC_fpStore64,[InstrStage<2, [FU_Issue], 0>, 
                               InstrStage<1, [FU_Pipe0], 0>,
                               InstrStage<1, [FU_Pipe1]>,
                               InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_LdSt0], 0>,
                               InstrStage<1, [FU_NLSPipe]>]>,
  //
  // FP Store Multiple
  // use FU_Issue to enforce the 1 load/store per cycle limit
  InstrItinData<IIC_fpStore64,[InstrStage<3, [FU_Issue], 0>, 
                               InstrStage<2, [FU_Pipe0], 0>,
                               InstrStage<2, [FU_Pipe1]>,
                               InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_LdSt0], 0>,
                               InstrStage<1, [FU_NLSPipe]>]>
]>;

// FIXME
def CortexA9Itineraries : ProcessorItineraries<[
  InstrItinData<IIC_iALUx   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iALUi   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iALUr   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iALUsi  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iALUsr  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iUNAr   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iUNAsi  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iUNAsr  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iCMPi   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iCMPr   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iCMPsi  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iCMPsr  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iMOVi   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iMOVr   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iMOVsi  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iMOVsr  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iCMOVi  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iCMOVr  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iCMOVsi , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iCMOVsr , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iMUL16  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iMAC16  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iMUL32  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iMAC32  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iMUL64  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iMAC64  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iLoadi  , [InstrStage<1, [FU_Pipe0]>,
                               InstrStage<1, [FU_LdSt0]>]>,
  InstrItinData<IIC_iLoadr  , [InstrStage<1, [FU_Pipe0]>,
                               InstrStage<1, [FU_LdSt0]>]>,
  InstrItinData<IIC_iLoadsi , [InstrStage<1, [FU_Pipe0]>,
                               InstrStage<1, [FU_LdSt0]>]>,
  InstrItinData<IIC_iLoadiu , [InstrStage<1, [FU_Pipe0]>,
                               InstrStage<1, [FU_LdSt0]>]>,
  InstrItinData<IIC_iLoadru , [InstrStage<1, [FU_Pipe0]>,
                               InstrStage<1, [FU_LdSt0]>]>,
  InstrItinData<IIC_iLoadsiu, [InstrStage<1, [FU_Pipe0]>,
                               InstrStage<1, [FU_LdSt0]>]>,
  InstrItinData<IIC_iLoadm  , [InstrStage<2, [FU_Pipe0]>,
                               InstrStage<2, [FU_LdSt0]>]>,
  InstrItinData<IIC_iStorei  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iStorer  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iStoresi , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iStoreiu , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iStoreru , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iStoresiu, [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_iStorem  , [InstrStage<2, [FU_Pipe0]>]>,
  InstrItinData<IIC_Br       , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpSTAT   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpMOVSI  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpMOVDI  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpMOVIS  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpMOVID  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpUNA32  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpUNA64  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpCMP32  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpCMP64  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpCVTSD  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpCVTDS  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpCVTIS  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpCVTID  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpCVTSI  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpCVTDI  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpALU32  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpALU64  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpMUL32  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpMUL64  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpMAC32  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpMAC64  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpDIV32  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpDIV64  , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpSQRT32 , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpSQRT64 , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpLoad32 , [InstrStage<1, [FU_Pipe0]>,
                                InstrStage<1, [FU_LdSt0]>]>,
  InstrItinData<IIC_fpLoad64 , [InstrStage<1, [FU_Pipe0]>,
                                InstrStage<1, [FU_LdSt0]>]>,
  InstrItinData<IIC_fpLoadm  , [InstrStage<1, [FU_Pipe0]>,
                                InstrStage<1, [FU_LdSt0]>]>,
  InstrItinData<IIC_fpStore32, [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpStore64, [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpStorem , [InstrStage<1, [FU_Pipe0]>]>
]>;
