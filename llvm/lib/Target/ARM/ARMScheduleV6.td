//===- ARMScheduleV6.td - ARM v6 Scheduling Definitions ----*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the itinerary class data for the ARM v6 processors.
//
//===----------------------------------------------------------------------===//

// Model based on ARM1176
//
// Scheduling information derived from "ARM1176JZF-S Technical Reference Manual".
//
def ARMV6Itineraries : ProcessorItineraries<[
  //
  // No operand cycles
  InstrItinData<IIC_iALUx    , [InstrStage<1, [FU_Pipe0]>]>,
  //
  // Binary Instructions that produce a result
  InstrItinData<IIC_iALUi    , [InstrStage<1, [FU_Pipe0]>], [2, 2]>,
  InstrItinData<IIC_iALUr    , [InstrStage<1, [FU_Pipe0]>], [2, 2, 2]>,
  InstrItinData<IIC_iALUsi   , [InstrStage<1, [FU_Pipe0]>], [2, 2, 1]>,
  InstrItinData<IIC_iALUsr   , [InstrStage<2, [FU_Pipe0]>], [3, 3, 2, 1]>,
  //
  // Unary Instructions that produce a result
  InstrItinData<IIC_iUNAr    , [InstrStage<1, [FU_Pipe0]>], [2, 2]>,
  InstrItinData<IIC_iUNAsi   , [InstrStage<1, [FU_Pipe0]>], [2, 1]>,
  InstrItinData<IIC_iUNAsr   , [InstrStage<2, [FU_Pipe0]>], [3, 2, 1]>,
  //
  // Compare instructions
  InstrItinData<IIC_iCMPi    , [InstrStage<1, [FU_Pipe0]>], [2]>,
  InstrItinData<IIC_iCMPr    , [InstrStage<1, [FU_Pipe0]>], [2, 2]>,
  InstrItinData<IIC_iCMPsi   , [InstrStage<1, [FU_Pipe0]>], [2, 1]>,
  InstrItinData<IIC_iCMPsr   , [InstrStage<2, [FU_Pipe0]>], [3, 2, 1]>,
  //
  // Move instructions, unconditional
  InstrItinData<IIC_iMOVi    , [InstrStage<1, [FU_Pipe0]>], [2]>,
  InstrItinData<IIC_iMOVr    , [InstrStage<1, [FU_Pipe0]>], [2, 2]>,
  InstrItinData<IIC_iMOVsi   , [InstrStage<1, [FU_Pipe0]>], [2, 1]>,
  InstrItinData<IIC_iMOVsr   , [InstrStage<2, [FU_Pipe0]>], [3, 2, 1]>,
  //
  // Move instructions, conditional
  InstrItinData<IIC_iCMOVi   , [InstrStage<1, [FU_Pipe0]>], [3]>,
  InstrItinData<IIC_iCMOVr   , [InstrStage<1, [FU_Pipe0]>], [3, 2]>,
  InstrItinData<IIC_iCMOVsi  , [InstrStage<1, [FU_Pipe0]>], [3, 1]>,
  InstrItinData<IIC_iCMOVsr  , [InstrStage<1, [FU_Pipe0]>], [4, 2, 1]>,

  // Integer multiply pipeline
  //
  InstrItinData<IIC_iMUL16   , [InstrStage<1, [FU_Pipe0]>], [4, 1, 1]>,
  InstrItinData<IIC_iMAC16   , [InstrStage<1, [FU_Pipe0]>], [4, 1, 1, 2]>,
  InstrItinData<IIC_iMUL32   , [InstrStage<2, [FU_Pipe0]>], [5, 1, 1]>,
  InstrItinData<IIC_iMAC32   , [InstrStage<2, [FU_Pipe0]>], [5, 1, 1, 2]>,
  InstrItinData<IIC_iMUL64   , [InstrStage<3, [FU_Pipe0]>], [6, 1, 1]>,
  InstrItinData<IIC_iMAC64   , [InstrStage<3, [FU_Pipe0]>], [6, 1, 1, 2]>,
  
  // Integer load pipeline
  //
  // Immediate offset
  InstrItinData<IIC_iLoadi   , [InstrStage<1, [FU_Pipe0]>], [4, 1]>,
  //
  // Register offset
  InstrItinData<IIC_iLoadr   , [InstrStage<1, [FU_Pipe0]>], [4, 1, 1]>,
  //
  // Scaled register offset, issues over 2 cycles
  InstrItinData<IIC_iLoadsi  , [InstrStage<2, [FU_Pipe0]>], [5, 2, 1]>,
  //
  // Immediate offset with update
  InstrItinData<IIC_iLoadiu  , [InstrStage<1, [FU_Pipe0]>], [4, 2, 1]>,
  //
  // Register offset with update
  InstrItinData<IIC_iLoadru  , [InstrStage<1, [FU_Pipe0]>], [4, 2, 1, 1]>,
  //
  // Scaled register offset with update, issues over 2 cycles
  InstrItinData<IIC_iLoadsiu , [InstrStage<2, [FU_Pipe0]>], [5, 2, 2, 1]>,

  //
  // Load multiple
  InstrItinData<IIC_iLoadm   , [InstrStage<3, [FU_Pipe0]>]>,

  // Integer store pipeline
  //
  // Immediate offset
  InstrItinData<IIC_iStorei  , [InstrStage<1, [FU_Pipe0]>], [2, 1]>,
  //
  // Register offset
  InstrItinData<IIC_iStorer  , [InstrStage<1, [FU_Pipe0]>], [2, 1, 1]>,

  //
  // Scaled register offset, issues over 2 cycles
  InstrItinData<IIC_iStoresi , [InstrStage<2, [FU_Pipe0]>], [2, 2, 1]>,
  //
  // Immediate offset with update
  InstrItinData<IIC_iStoreiu , [InstrStage<1, [FU_Pipe0]>], [2, 2, 1]>,
  //
  // Register offset with update
  InstrItinData<IIC_iStoreru , [InstrStage<1, [FU_Pipe0]>], [2, 2, 1, 1]>,
  //
  // Scaled register offset with update, issues over 2 cycles
  InstrItinData<IIC_iStoresiu, [InstrStage<2, [FU_Pipe0]>], [2, 2, 2, 1]>,
  //
  // Store multiple
  InstrItinData<IIC_iStorem   , [InstrStage<3, [FU_Pipe0]>]>,
  
  // Branch
  //
  // no delay slots, so the latency of a branch is unimportant
  InstrItinData<IIC_Br      , [InstrStage<1, [FU_Pipe0]>]>,

  // VFP
  // Issue through integer pipeline, and execute in NEON unit. We assume
  // RunFast mode so that NFP pipeline is used for single-precision when
  // possible.
  //
  // FP Special Register to Integer Register File Move
  InstrItinData<IIC_fpSTAT , [InstrStage<1, [FU_Pipe0]>], [3]>,
  //
  // Single-precision FP Unary
  InstrItinData<IIC_fpUNA32 , [InstrStage<1, [FU_Pipe0]>], [5, 2]>,
  //
  // Double-precision FP Unary
  InstrItinData<IIC_fpUNA64 , [InstrStage<1, [FU_Pipe0]>], [5, 2]>,
  //
  // Single-precision FP Compare
  InstrItinData<IIC_fpCMP32 , [InstrStage<1, [FU_Pipe0]>], [2, 2]>,
  //
  // Double-precision FP Compare
  InstrItinData<IIC_fpCMP64 , [InstrStage<1, [FU_Pipe0]>], [2, 2]>,
  //
  // Single to Double FP Convert
  InstrItinData<IIC_fpCVTSD , [InstrStage<1, [FU_Pipe0]>], [5, 2]>,
  //
  // Double to Single FP Convert
  InstrItinData<IIC_fpCVTDS , [InstrStage<1, [FU_Pipe0]>], [5, 2]>,
  //
  // Single-Precision FP to Integer Convert
  InstrItinData<IIC_fpCVTSI , [InstrStage<1, [FU_Pipe0]>], [9, 2]>,
  //
  // Double-Precision FP to Integer Convert
  InstrItinData<IIC_fpCVTDI , [InstrStage<1, [FU_Pipe0]>], [9, 2]>,
  //
  // Integer to Single-Precision FP Convert
  InstrItinData<IIC_fpCVTIS , [InstrStage<1, [FU_Pipe0]>], [9, 2]>,
  //
  // Integer to Double-Precision FP Convert
  InstrItinData<IIC_fpCVTID , [InstrStage<1, [FU_Pipe0]>], [9, 2]>,
  //
  // Single-precision FP ALU
  InstrItinData<IIC_fpALU32 , [InstrStage<1, [FU_Pipe0]>], [9, 2, 2]>,
  //
  // Double-precision FP ALU
  InstrItinData<IIC_fpALU64 , [InstrStage<1, [FU_Pipe0]>], [9, 2, 2]>,
  //
  // Single-precision FP Multiply
  InstrItinData<IIC_fpMUL32 , [InstrStage<1, [FU_Pipe0]>], [9, 2, 2]>,
  //
  // Double-precision FP Multiply
  InstrItinData<IIC_fpMUL64 , [InstrStage<2, [FU_Pipe0]>], [9, 2, 2]>,
  //
  // Single-precision FP MAC
  InstrItinData<IIC_fpMAC32 , [InstrStage<1, [FU_Pipe0]>], [9, 2, 2, 2]>,
  //
  // Double-precision FP MAC
  InstrItinData<IIC_fpMAC64 , [InstrStage<2, [FU_Pipe0]>], [9, 2, 2, 2]>,
  //
  // Single-precision FP DIV
  InstrItinData<IIC_fpDIV32 , [InstrStage<15, [FU_Pipe0]>], [20, 2, 2]>,
  //
  // Double-precision FP DIV
  InstrItinData<IIC_fpDIV64 , [InstrStage<29, [FU_Pipe0]>], [34, 2, 2]>,
  //
  // Single-precision FP SQRT
  InstrItinData<IIC_fpSQRT32 , [InstrStage<15, [FU_Pipe0]>], [20, 2, 2]>,
  //
  // Double-precision FP SQRT
  InstrItinData<IIC_fpSQRT64 , [InstrStage<29, [FU_Pipe0]>], [34, 2, 2]>,
  //
  // Single-precision FP Load
  InstrItinData<IIC_fpLoad32 , [InstrStage<1, [FU_Pipe0]>], [5, 2, 2]>,
  //
  // Double-precision FP Load
  InstrItinData<IIC_fpLoad64 , [InstrStage<1, [FU_Pipe0]>], [5, 2, 2]>,
  //
  // FP Load Multiple
  InstrItinData<IIC_fpLoadm , [InstrStage<3, [FU_Pipe0]>]>,
  //
  // Single-precision FP Store
  InstrItinData<IIC_fpStore32 , [InstrStage<1, [FU_Pipe0]>], [2, 2, 2]>,
  //
  // Double-precision FP Store
  // use FU_Issue to enforce the 1 load/store per cycle limit
  InstrItinData<IIC_fpStore64 , [InstrStage<1, [FU_Pipe0]>], [2, 2, 2]>,
  //
  // FP Store Multiple
  InstrItinData<IIC_fpStorem , [InstrStage<3, [FU_Pipe0]>]>
]>;
