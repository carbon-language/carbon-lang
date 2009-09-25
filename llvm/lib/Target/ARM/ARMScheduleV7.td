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
                              InstrStage<1, [FU_NLSPipe]>]>,
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
  InstrItinData<IIC_fpStorem, [InstrStage<3, [FU_Issue], 0>, 
                               InstrStage<2, [FU_Pipe0], 0>,
                               InstrStage<2, [FU_Pipe1]>,
                               InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_LdSt0], 0>,
                               InstrStage<1, [FU_NLSPipe]>]>,

  // NEON
  // Issue through integer pipeline, and execute in NEON unit.
  //
  // VLD1
  InstrItinData<IIC_VLD1,     [InstrStage<1, [FU_Issue], 0>, 
                               InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_LdSt0], 0>,
                               InstrStage<1, [FU_NLSPipe]>]>,
  //
  // VLD2
  InstrItinData<IIC_VLD2,     [InstrStage<1, [FU_Issue], 0>, 
                               InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_LdSt0], 0>,
                               InstrStage<1, [FU_NLSPipe]>], [2, 2, 1]>,
  //
  // VLD3
  InstrItinData<IIC_VLD3,     [InstrStage<1, [FU_Issue], 0>, 
                               InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_LdSt0], 0>,
                               InstrStage<1, [FU_NLSPipe]>], [2, 2, 2, 1]>,
  //
  // VLD4
  InstrItinData<IIC_VLD4,     [InstrStage<1, [FU_Issue], 0>, 
                               InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_LdSt0], 0>,
                               InstrStage<1, [FU_NLSPipe]>], [2, 2, 2, 2, 1]>,
  //
  // VST
  InstrItinData<IIC_VST,      [InstrStage<1, [FU_Issue], 0>, 
                               InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_LdSt0], 0>,
                               InstrStage<1, [FU_NLSPipe]>]>,
  //
  // Double-register FP Unary
  InstrItinData<IIC_VUNAD,    [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>], [5, 2]>,
  //
  // Quad-register FP Unary
  // Result written in N5, but that is relative to the last cycle of multicycle,
  // so we use 6 for those cases
  InstrItinData<IIC_VUNAQ,    [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<2, [FU_NPipe]>], [6, 2]>,
  //
  // Double-register FP Binary
  InstrItinData<IIC_VBIND,    [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>], [5, 2, 2]>,
  //
  // Quad-register FP Binary
  // Result written in N5, but that is relative to the last cycle of multicycle,
  // so we use 6 for those cases
  InstrItinData<IIC_VBINQ,    [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<2, [FU_NPipe]>], [6, 2, 2]>,
  //
  // Move Immediate
  InstrItinData<IIC_VMOVImm,  [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>], [3]>,
  //
  // Double-register Permute Move
  InstrItinData<IIC_VMOVD,    [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NLSPipe]>], [2, 1]>,
  //
  // Quad-register Permute Move
  // Result written in N2, but that is relative to the last cycle of multicycle,
  // so we use 3 for those cases
  InstrItinData<IIC_VMOVQ,    [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<2, [FU_NLSPipe]>], [3, 1]>,
  //
  // Integer to Single-precision Move
  InstrItinData<IIC_VMOVIS ,  [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NLSPipe]>], [2, 1]>,
  //
  // Integer to Double-precision Move
  InstrItinData<IIC_VMOVID ,  [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NLSPipe]>], [2, 1, 1]>,
  //
  // Single-precision to Integer Move
  InstrItinData<IIC_VMOVSI ,  [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NLSPipe]>], [20, 1]>,
  //
  // Double-precision to Integer Move
  InstrItinData<IIC_VMOVDI ,  [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NLSPipe]>], [20, 20, 1]>,
  //
  // Integer to Lane Move
  InstrItinData<IIC_VMOVISL , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<2, [FU_NLSPipe]>], [3, 1, 1]>,
  //
  // Double-register Permute
  InstrItinData<IIC_VPERMD,   [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NLSPipe]>], [2, 2, 1, 1]>,
  //
  // Quad-register Permute
  // Result written in N2, but that is relative to the last cycle of multicycle,
  // so we use 3 for those cases
  InstrItinData<IIC_VPERMQ,   [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<2, [FU_NLSPipe]>], [3, 3, 1, 1]>,
  //
  // Quad-register Permute (3 cycle issue)
  // Result written in N2, but that is relative to the last cycle of multicycle,
  // so we use 4 for those cases
  InstrItinData<IIC_VPERMQ3,  [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NLSPipe]>,
                               InstrStage<1, [FU_NPipe], 0>,
                               InstrStage<2, [FU_NLSPipe]>], [4, 4, 1, 1]>,
  //
  // Double-register FP Multiple-Accumulate
  InstrItinData<IIC_VMACD,    [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>], [9, 2, 2, 3]>,
  //
  // Quad-register FP Multiple-Accumulate
  // Result written in N9, but that is relative to the last cycle of multicycle,
  // so we use 10 for those cases
  InstrItinData<IIC_VMACQ,    [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<2, [FU_NPipe]>], [10, 2, 2, 3]>,
  //
  // Double-register Reciprical Step
  InstrItinData<IIC_VRECSD,   [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>], [9, 2, 2]>,
  //
  // Quad-register Reciprical Step
  InstrItinData<IIC_VRECSQ,   [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<2, [FU_NPipe]>], [10, 2, 2]>,
  //
  // Double-register Integer Count
  InstrItinData<IIC_VCNTiD,   [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>], [3, 2, 2]>,
  //
  // Quad-register Integer Count
  // Result written in N3, but that is relative to the last cycle of multicycle,
  // so we use 4 for those cases
  InstrItinData<IIC_VCNTiQ,   [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<2, [FU_NPipe]>], [4, 2, 2]>,
  //
  // Double-register Integer Unary
  InstrItinData<IIC_VUNAiD,   [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>], [4, 2]>,
  //
  // Quad-register Integer Unary
  InstrItinData<IIC_VUNAiQ,   [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>], [4, 2]>,
  //
  // Double-register Integer Q-Unary
  InstrItinData<IIC_VQUNAiD,  [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>], [4, 1]>,
  //
  // Quad-register Integer CountQ-Unary
  InstrItinData<IIC_VQUNAiQ,  [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>], [4, 1]>,
  //
  // Double-register Integer Binary
  InstrItinData<IIC_VBINiD,   [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>], [3, 2, 2]>,
  //
  // Quad-register Integer Binary
  InstrItinData<IIC_VBINiQ,   [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>], [3, 2, 2]>,
  //
  // Double-register Integer Binary (4 cycle)
  InstrItinData<IIC_VBINi4D,  [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>], [4, 2, 1]>,
  //
  // Quad-register Integer Binary (4 cycle)
  InstrItinData<IIC_VBINi4Q,  [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>], [4, 2, 1]>,
  //
  // Double-register Integer Subtract
  InstrItinData<IIC_VSUBiD,   [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>], [3, 2, 1]>,
  //
  // Quad-register Integer Subtract
  InstrItinData<IIC_VSUBiQ,   [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>], [3, 2, 1]>,
  //
  // Double-register Integer Shift
  InstrItinData<IIC_VSHLiD,   [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>], [3, 1, 1]>,
  //
  // Quad-register Integer Shift
  InstrItinData<IIC_VSHLiQ,   [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<2, [FU_NPipe]>], [4, 1, 1]>,
  //
  // Double-register Integer Shift (4 cycle)
  InstrItinData<IIC_VSHLi4D,   [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>], [4, 1, 1]>,
  //
  // Quad-register Integer Shift (4 cycle)
  InstrItinData<IIC_VSHLi4Q,   [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<2, [FU_NPipe]>], [5, 1, 1]>,
  //
  // Double-register Integer Pair Add Long
  InstrItinData<IIC_VPALiD,   [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>], [6, 3, 2, 1]>,
  //
  // Quad-register Integer Pair Add Long
  InstrItinData<IIC_VPALiQ,   [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<2, [FU_NPipe]>], [7, 3, 2, 1]>,
  //
  // Double-register Integer Multiply (.8, .16)
  InstrItinData<IIC_VMULi16D, [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>], [6, 2, 2]>,
  //
  // Double-register Integer Multiply (.32)
  InstrItinData<IIC_VMULi32D, [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<2, [FU_NPipe]>], [7, 2, 1]>,
  //
  // Quad-register Integer Multiply (.8, .16)
  InstrItinData<IIC_VMULi16Q, [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<2, [FU_NPipe]>], [7, 2, 2]>,
  //
  // Quad-register Integer Multiply (.32)
  InstrItinData<IIC_VMULi32Q, [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>,
                               InstrStage<2, [FU_NLSPipe], 0>,
                               InstrStage<3, [FU_NPipe]>], [9, 2, 1]>,
  //
  // Double-register Integer Multiply-Accumulate (.8, .16)
  InstrItinData<IIC_VMACi16D, [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>], [6, 2, 2, 3]>,
  //
  // Double-register Integer Multiply-Accumulate (.32)
  InstrItinData<IIC_VMACi32D, [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<2, [FU_NPipe]>], [7, 2, 1, 3]>,
  //
  // Quad-register Integer Multiply-Accumulate (.8, .16)
  InstrItinData<IIC_VMACi16Q, [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<2, [FU_NPipe]>], [7, 2, 2, 3]>,
  //
  // Quad-register Integer Multiply-Accumulate (.32)
  InstrItinData<IIC_VMACi32Q, [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NPipe]>,
                               InstrStage<2, [FU_NLSPipe], 0>,
                               InstrStage<3, [FU_NPipe]>], [9, 2, 1, 3]>,
  //
  // Double-register VEXT
  InstrItinData<IIC_VEXTD,    [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NLSPipe]>], [2, 1, 1]>,
  //
  // Quad-register VEXT
  InstrItinData<IIC_VEXTQ,    [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<2, [FU_NLSPipe]>], [3, 1, 1]>,
  //
  // VTB
  InstrItinData<IIC_VTB1,     [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<2, [FU_NLSPipe]>], [3, 2, 1]>,
  InstrItinData<IIC_VTB2,     [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<2, [FU_NLSPipe]>], [3, 2, 2, 1]>,
  InstrItinData<IIC_VTB3,     [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NLSPipe]>,
                               InstrStage<1, [FU_NPipe], 0>,
                               InstrStage<2, [FU_NLSPipe]>], [4, 2, 2, 3, 1]>,
  InstrItinData<IIC_VTB4,     [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NLSPipe]>,
                               InstrStage<1, [FU_NPipe], 0>,
                               InstrStage<2, [FU_NLSPipe]>], [4, 2, 2, 3, 3, 1]>,
  //
  // VTBX
  InstrItinData<IIC_VTBX1,    [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<2, [FU_NLSPipe]>], [3, 1, 2, 1]>,
  InstrItinData<IIC_VTBX2,    [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<2, [FU_NLSPipe]>], [3, 1, 2, 2, 1]>,
  InstrItinData<IIC_VTBX3,    [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NLSPipe]>,
                               InstrStage<1, [FU_NPipe], 0>,
                               InstrStage<2, [FU_NLSPipe]>], [4, 1, 2, 2, 3, 1]>,
  InstrItinData<IIC_VTBX4,    [InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_NLSPipe]>,
                               InstrStage<1, [FU_NPipe], 0>,
                               InstrStage<2, [FU_NLSPipe]>], [4, 1, 2, 2, 3, 3, 1]>
]>;
