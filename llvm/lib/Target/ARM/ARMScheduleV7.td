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
// Dual issue pipeline so every itinerary starts with FU_Pipe0 | FU_Pipe1
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
                                InstrStage<1, [FU_Pipe1], 0>,
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
                                InstrStage<1, [FU_Pipe1], 0>,
                                InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                                InstrStage<1, [FU_LdSt0]>], [4, 3, 1, 1]>,
  //
  // Load multiple
  InstrItinData<IIC_iLoadm   , [InstrStage<2, [FU_Issue], 0>,
                                InstrStage<2, [FU_Pipe0], 0>,
                                InstrStage<2, [FU_Pipe1], 0>,
                                InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                                InstrStage<1, [FU_LdSt0]>]>,

  // Integer store pipeline
  //
  // use FU_Issue to enforce the 1 load/store per cycle limit
  //
  // Immediate offset
  InstrItinData<IIC_iStorei  , [InstrStage<1, [FU_Issue], 0>,
                                InstrStage<1, [FU_Pipe0, FU_Pipe1]>], [3, 1]>,
  //
  // Register offset
  InstrItinData<IIC_iStorer  , [InstrStage<1, [FU_Issue], 0>,
                                InstrStage<1, [FU_Pipe0, FU_Pipe1]>], [3, 1, 1]>,
  //
  // Scaled register offset, issues over 2 cycles
  InstrItinData<IIC_iStoresi , [InstrStage<2, [FU_Issue], 0>,
                                InstrStage<1, [FU_Pipe0], 0>,
                                InstrStage<1, [FU_Pipe1], 0>,
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
                                InstrStage<1, [FU_Pipe1], 0>,
                                InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                                InstrStage<1, [FU_LdSt0]>], [3, 3, 1, 1]>,
  //
  // Store multiple
  InstrItinData<IIC_iStorem  , [InstrStage<2, [FU_Issue], 0>,
                                InstrStage<2, [FU_Pipe0], 0>,
                                InstrStage<2, [FU_Pipe1], 0>,
                                InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                                InstrStage<1, [FU_LdSt0]>]>,
  
  // Branch
  //
  // no delay slots, so the latency of a branch is unimportant
  InstrItinData<IIC_Br      , [InstrStage<1, [FU_Pipe0, FU_Pipe1]>]>,

  // NFP ALU is not pipelined so stall all issues 
  InstrItinData<IIC_fpALU   , [InstrStage<7, [FU_Pipe0], 0>,
                               InstrStage<7, [FU_Pipe1], 0>]>,
  // VFP MPY is not pipelined so stall all issues 
  InstrItinData<IIC_fpMPY   , [InstrStage<7, [FU_Pipe0], 0>,
                               InstrStage<7, [FU_Pipe1], 0>]>,
  // loads have an extra cycle of latency, but are fully pipelined
  // use FU_Issue to enforce the 1 load/store per cycle limit
  InstrItinData<IIC_fpLoad  , [InstrStage<1, [FU_Issue], 0>, 
                               InstrStage<1, [FU_Pipe0, FU_Pipe1]>,
                               InstrStage<1, [FU_LdSt0]>]>,
  // use FU_Issue to enforce the 1 load/store per cycle limit
  InstrItinData<IIC_fpStore , [InstrStage<1, [FU_Issue], 0>, 
                               InstrStage<1, [FU_Pipe0, FU_Pipe1]>]>
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
  InstrItinData<IIC_Br      , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpALU   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpMPY   , [InstrStage<1, [FU_Pipe0]>]>,
  InstrItinData<IIC_fpLoad  , [InstrStage<1, [FU_Pipe0]>,
                               InstrStage<1, [FU_LdSt0]>]>,
  InstrItinData<IIC_fpStore , [InstrStage<1, [FU_Pipe0]>]>
]>;
