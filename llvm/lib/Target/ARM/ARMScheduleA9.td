//=- ARMScheduleA9.td - ARM Cortex-A9 Scheduling Definitions -*- tablegen -*-=//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the itinerary class data for the ARM Cortex A9 processors.
//
//===----------------------------------------------------------------------===//

//
// Ad-hoc scheduling information derived from pretty vague "Cortex-A9 Technical
// Reference Manual".
//
// Functional units
def A9_Issue   : FuncUnit; // issue
def A9_Pipe0   : FuncUnit; // pipeline 0
def A9_Pipe1   : FuncUnit; // pipeline 1
def A9_LSPipe  : FuncUnit; // LS pipe
def A9_NPipe   : FuncUnit; // NEON ALU/MUL pipe
def A9_DRegsVFP: FuncUnit; // FP register set, VFP side
def A9_DRegsN  : FuncUnit; // FP register set, NEON side

// Dual issue pipeline represented by A9_Pipe0 | A9_Pipe1
//
def CortexA9Itineraries : ProcessorItineraries<
  [A9_NPipe, A9_DRegsN, A9_DRegsVFP, A9_LSPipe, A9_Pipe0, A9_Pipe1, A9_Issue], [
  // VFP and NEON shares the same register file. This means that every VFP
  // instruction should wait for full completion of the consecutive NEON
  // instruction and vice-versa. We model this behavior with two artificial FUs:
  // DRegsVFP and DRegsVFP.
  //
  // Every VFP instruction:
  //  - Acquires DRegsVFP resource for 1 cycle
  //  - Reserves DRegsN resource for the whole duration (including time to
  //    register file writeback!).
  // Every NEON instruction does the same but with FUs swapped.
  //
  // Since the reserved FU cannot be acquired this models precisly "cross-domain"
  // stalls.

  // VFP
  // Issue through integer pipeline, and execute in NEON unit.

  // FP Special Register to Integer Register File Move
  InstrItinData<IIC_fpSTAT , [InstrStage<1, [A9_DRegsVFP], 0, Required>,
                              InstrStage<2, [A9_DRegsN],   0, Reserved>,
                              InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                              InstrStage<1, [A9_NPipe]>]>,
  //
  // Single-precision FP Unary
  InstrItinData<IIC_fpUNA32 , [InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               // Extra latency cycles since wbck is 2 cycles
                               InstrStage<3, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [1, 1]>,
  //
  // Double-precision FP Unary
  InstrItinData<IIC_fpUNA64 , [InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               // Extra latency cycles since wbck is 2 cycles
                               InstrStage<3, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [1, 1]>,

  //
  // Single-precision FP Compare
  InstrItinData<IIC_fpCMP32 , [InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               // Extra latency cycles since wbck is 4 cycles
                               InstrStage<5, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [1, 1]>,
  //
  // Double-precision FP Compare
  InstrItinData<IIC_fpCMP64 , [InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               // Extra latency cycles since wbck is 4 cycles
                               InstrStage<5, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [1, 1]>,
  //
  // Single to Double FP Convert
  InstrItinData<IIC_fpCVTSD , [InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<5, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [4, 1]>,
  //
  // Double to Single FP Convert
  InstrItinData<IIC_fpCVTDS , [InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<5, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [4, 1]>,

  //
  // Single to Half FP Convert
  InstrItinData<IIC_fpCVTSH , [InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<5, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [4, 1]>,
  //
  // Half to Single FP Convert
  InstrItinData<IIC_fpCVTHS , [InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<3, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [2, 1]>,

  //
  // Single-Precision FP to Integer Convert
  InstrItinData<IIC_fpCVTSI , [InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<5, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [4, 1]>,
  //
  // Double-Precision FP to Integer Convert
  InstrItinData<IIC_fpCVTDI , [InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<5, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [4, 1]>,
  //
  // Integer to Single-Precision FP Convert
  InstrItinData<IIC_fpCVTIS , [InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<5, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [4, 1]>,
  //
  // Integer to Double-Precision FP Convert
  InstrItinData<IIC_fpCVTID , [InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<5, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [4, 1]>,
  //
  // Single-precision FP ALU
  InstrItinData<IIC_fpALU32 , [InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<5, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [4, 1, 1]>,
  //
  // Double-precision FP ALU
  InstrItinData<IIC_fpALU64 , [InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<5, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [4, 1, 1]>,
  //
  // Single-precision FP Multiply
  InstrItinData<IIC_fpMUL32 , [InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<6, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [5, 1, 1]>,
  //
  // Double-precision FP Multiply
  InstrItinData<IIC_fpMUL64 , [InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<7, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<2, [A9_NPipe]>], [6, 1, 1]>,
  //
  // Single-precision FP MAC
  InstrItinData<IIC_fpMAC32 , [InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<9, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [8, 0, 1, 1]>,
  //
  // Double-precision FP MAC
  InstrItinData<IIC_fpMAC64 , [InstrStage<1,  [A9_DRegsVFP], 0, Required>,
                               InstrStage<10, [A9_DRegsN],  0, Reserved>,
                               InstrStage<1,  [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<2,  [A9_NPipe]>], [9, 0, 1, 1]>,
  //
  // Single-precision FP DIV
  InstrItinData<IIC_fpDIV32 , [InstrStage<1,  [A9_DRegsVFP], 0, Required>,
                               InstrStage<16, [A9_DRegsN],  0, Reserved>,
                               InstrStage<1,  [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<10, [A9_NPipe]>], [15, 1, 1]>,
  //
  // Double-precision FP DIV
  InstrItinData<IIC_fpDIV64 , [InstrStage<1,  [A9_DRegsVFP], 0, Required>,
                               InstrStage<26, [A9_DRegsN],  0, Reserved>,
                               InstrStage<1,  [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<20, [A9_NPipe]>], [25, 1, 1]>,
  //
  // Single-precision FP SQRT
  InstrItinData<IIC_fpSQRT32, [InstrStage<1,  [A9_DRegsVFP], 0, Required>,
                               InstrStage<18, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1,   [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<13,  [A9_NPipe]>], [17, 1]>,
  //
  // Double-precision FP SQRT
  InstrItinData<IIC_fpSQRT64, [InstrStage<1,  [A9_DRegsVFP], 0, Required>,
                               InstrStage<33, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1,  [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<28, [A9_NPipe]>], [32, 1]>,

  //
  // Integer to Single-precision Move
  InstrItinData<IIC_fpMOVIS,  [InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               // Extra 1 latency cycle since wbck is 2 cycles
                               InstrStage<3, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [1, 1]>,
  //
  // Integer to Double-precision Move
  InstrItinData<IIC_fpMOVID,  [InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               // Extra 1 latency cycle since wbck is 2 cycles
                               InstrStage<3, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [1, 1, 1]>,
  //
  // Single-precision to Integer Move
  InstrItinData<IIC_fpMOVSI,  [InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<2, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [1, 1]>,
  //
  // Double-precision to Integer Move
  InstrItinData<IIC_fpMOVDI,  [InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<2, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [1, 1, 1]>,
  //
  // Single-precision FP Load
  // use A9_Issue to enforce the 1 load/store per cycle limit
  InstrItinData<IIC_fpLoad32, [InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<2, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_Issue], 0>, 
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_LSPipe], 0>,
                               InstrStage<1, [A9_NPipe]>]>,
  //
  // Double-precision FP Load
  // use A9_Issue to enforce the 1 load/store per cycle limit
  InstrItinData<IIC_fpLoad64, [InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<2, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_Issue], 0>, 
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_LSPipe], 0>,
                               InstrStage<1, [A9_NPipe]>]>,
  //
  // FP Load Multiple
  // use A9_Issue to enforce the 1 load/store per cycle limit
  InstrItinData<IIC_fpLoadm,  [InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<2, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_Issue], 0>, 
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_LSPipe], 0>,
                               InstrStage<1, [A9_NPipe]>]>,
  //
  // Single-precision FP Store
  // use A9_Issue to enforce the 1 load/store per cycle limit
  InstrItinData<IIC_fpStore32,[InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<2, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_Issue], 0>, 
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_LSPipe], 0>,
                               InstrStage<1, [A9_NPipe]>]>,
  //
  // Double-precision FP Store
  // use A9_Issue to enforce the 1 load/store per cycle limit
  InstrItinData<IIC_fpStore64,[InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<2, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_Issue], 0>, 
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_LSPipe], 0>,
                               InstrStage<1, [A9_NPipe]>]>,
  //
  // FP Store Multiple
  // use A9_Issue to enforce the 1 load/store per cycle limit
  InstrItinData<IIC_fpStorem, [InstrStage<1, [A9_DRegsVFP], 0, Required>,
                               InstrStage<2, [A9_DRegsN],   0, Reserved>,
                               InstrStage<1, [A9_Issue], 0>, 
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_LSPipe], 0>,
                               InstrStage<1, [A9_NPipe]>]>,
  // NEON
  // Issue through integer pipeline, and execute in NEON unit.
  // FIXME: Neon pipeline and LdSt unit are multiplexed. 
  //        Add some syntactic sugar to model this!
  // VLD1
  // FIXME: We don't model this instruction properly
  InstrItinData<IIC_VLD1,     [InstrStage<1, [A9_DRegsN],   0, Required>,
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Issue], 0>, 
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_LSPipe], 0>,
                               InstrStage<1, [A9_NPipe]>]>,
  //
  // VLD2
  // FIXME: We don't model this instruction properly
  InstrItinData<IIC_VLD2,     [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Issue], 0>, 
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_LSPipe], 0>,
                               InstrStage<1, [A9_NPipe]>], [2, 2, 1]>,
  //
  // VLD3
  // FIXME: We don't model this instruction properly
  InstrItinData<IIC_VLD3,     [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Issue], 0>, 
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_LSPipe], 0>,
                               InstrStage<1, [A9_NPipe]>], [2, 2, 2, 1]>,
  //
  // VLD4
  // FIXME: We don't model this instruction properly
  InstrItinData<IIC_VLD4,     [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Issue], 0>, 
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_LSPipe], 0>,
                               InstrStage<1, [A9_NPipe]>], [2, 2, 2, 2, 1]>,
  //
  // VST
  // FIXME: We don't model this instruction properly
  InstrItinData<IIC_VST,      [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Issue], 0>, 
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_LSPipe], 0>,
                               InstrStage<1, [A9_NPipe]>]>,
  //
  // Double-register Integer Unary
  InstrItinData<IIC_VUNAiD,   [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [4, 2]>,
  //
  // Quad-register Integer Unary
  InstrItinData<IIC_VUNAiQ,   [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [4, 2]>,
  //
  // Double-register Integer Q-Unary
  InstrItinData<IIC_VQUNAiD,  [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [4, 1]>,
  //
  // Quad-register Integer CountQ-Unary
  InstrItinData<IIC_VQUNAiQ,  [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [4, 1]>,
  //
  // Double-register Integer Binary
  InstrItinData<IIC_VBINiD,   [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [3, 2, 2]>,
  //
  // Quad-register Integer Binary
  InstrItinData<IIC_VBINiQ,   [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [3, 2, 2]>,
  //
  // Double-register Integer Subtract
  InstrItinData<IIC_VSUBiD,   [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [3, 2, 1]>,
  //
  // Quad-register Integer Subtract
  InstrItinData<IIC_VSUBiQ,   [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [3, 2, 1]>,
  //
  // Double-register Integer Shift
  InstrItinData<IIC_VSHLiD,   [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [3, 1, 1]>,
  //
  // Quad-register Integer Shift
  InstrItinData<IIC_VSHLiQ,   [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [3, 1, 1]>,
  //
  // Double-register Integer Shift (4 cycle)
  InstrItinData<IIC_VSHLi4D,  [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [4, 1, 1]>,
  //
  // Quad-register Integer Shift (4 cycle)
  InstrItinData<IIC_VSHLi4Q,  [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [4, 1, 1]>,
  //
  // Double-register Integer Binary (4 cycle)
  InstrItinData<IIC_VBINi4D,  [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [4, 2, 2]>,
  //
  // Quad-register Integer Binary (4 cycle)
  InstrItinData<IIC_VBINi4Q,  [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [4, 2, 2]>,
  //
  // Double-register Integer Subtract (4 cycle)
  InstrItinData<IIC_VSUBiD,   [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [4, 2, 1]>,
  //
  // Quad-register Integer Subtract (4 cycle)
  InstrItinData<IIC_VSUBiQ,   [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [4, 2, 1]>,

  //
  // Double-register Integer Count
  InstrItinData<IIC_VCNTiD,   [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [3, 2, 2]>,
  //
  // Quad-register Integer Count
  // Result written in N3, but that is relative to the last cycle of multicycle,
  // so we use 4 for those cases
  InstrItinData<IIC_VCNTiQ,   [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<2, [A9_NPipe]>], [4, 2, 2]>,
  //
  // Double-register Absolute Difference and Accumulate
  InstrItinData<IIC_VABAD,    [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [6, 3, 2, 1]>,
  //
  // Quad-register Absolute Difference and Accumulate
  InstrItinData<IIC_VABAQ,    [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<2, [A9_NPipe]>], [6, 3, 2, 1]>,
  //
  // Double-register Integer Pair Add Long
  InstrItinData<IIC_VPALiD,   [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [6, 3, 1]>,
  //
  // Quad-register Integer Pair Add Long
  InstrItinData<IIC_VPALiQ,   [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<2, [A9_NPipe]>], [6, 3, 1]>,

  //
  // Double-register Integer Multiply (.8, .16)
  InstrItinData<IIC_VMULi16D, [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [6, 2, 2]>,
  //
  // Quad-register Integer Multiply (.8, .16)
  InstrItinData<IIC_VMULi16Q, [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<2, [A9_NPipe]>], [7, 2, 2]>,

  //
  // Double-register Integer Multiply (.32)
  InstrItinData<IIC_VMULi32D, [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<2, [A9_NPipe]>], [7, 2, 1]>,
  //
  // Quad-register Integer Multiply (.32)
  InstrItinData<IIC_VMULi32Q, [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 9 cycles
                               InstrStage<10, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<4, [A9_NPipe]>], [9, 2, 1]>,
  //
  // Double-register Integer Multiply-Accumulate (.8, .16)
  InstrItinData<IIC_VMACi16D, [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [6, 3, 2, 2]>,
  //
  // Double-register Integer Multiply-Accumulate (.32)
  InstrItinData<IIC_VMACi32D, [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<2, [A9_NPipe]>], [7, 3, 2, 1]>,
  //
  // Quad-register Integer Multiply-Accumulate (.8, .16)
  InstrItinData<IIC_VMACi16Q, [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<2, [A9_NPipe]>], [7, 3, 2, 2]>,
  //
  // Quad-register Integer Multiply-Accumulate (.32)
  InstrItinData<IIC_VMACi32Q, [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 9 cycles
                               InstrStage<10, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<4, [A9_NPipe]>], [9, 3, 2, 1]>,
  //
  // Move Immediate
  InstrItinData<IIC_VMOVImm,  [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [3]>,
  //
  // Double-register Permute Move
  InstrItinData<IIC_VMOVD,    [InstrStage<1, [A9_DRegsN],   0, Required>,
  // FIXME: all latencies are arbitrary, no information is available
                               InstrStage<3, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_LSPipe]>], [2, 1]>,
  //
  // Quad-register Permute Move
  // Result written in N2, but that is relative to the last cycle of multicycle,
  // so we use 3 for those cases
  InstrItinData<IIC_VMOVQ,    [InstrStage<1, [A9_DRegsN],   0, Required>,
  // FIXME: all latencies are arbitrary, no information is available
                               InstrStage<4, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<2, [A9_NPipe]>], [3, 1]>,
  //
  // Integer to Single-precision Move
  InstrItinData<IIC_VMOVIS ,  [InstrStage<1, [A9_DRegsN],   0, Required>,
  // FIXME: all latencies are arbitrary, no information is available
                               InstrStage<3, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [2, 1]>,
  //
  // Integer to Double-precision Move
  InstrItinData<IIC_VMOVID ,  [InstrStage<1, [A9_DRegsN],   0, Required>,
  // FIXME: all latencies are arbitrary, no information is available
                               InstrStage<3, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [2, 1, 1]>,
  //
  // Single-precision to Integer Move
  InstrItinData<IIC_VMOVSI ,  [InstrStage<1, [A9_DRegsN],   0, Required>,
  // FIXME: all latencies are arbitrary, no information is available
                               InstrStage<3, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [2, 1]>,
  //
  // Double-precision to Integer Move
  InstrItinData<IIC_VMOVDI ,  [InstrStage<1, [A9_DRegsN],   0, Required>,
  // FIXME: all latencies are arbitrary, no information is available
                               InstrStage<3, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [2, 2, 1]>,
  //
  // Integer to Lane Move
  InstrItinData<IIC_VMOVISL , [InstrStage<1, [A9_DRegsN],   0, Required>,
  // FIXME: all latencies are arbitrary, no information is available
                               InstrStage<4, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<2, [A9_NPipe]>], [3, 1, 1]>,

  //
  // Double-register FP Unary
  InstrItinData<IIC_VUNAD,    [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [5, 2]>,
  //
  // Quad-register FP Unary
  // Result written in N5, but that is relative to the last cycle of multicycle,
  // so we use 6 for those cases
  InstrItinData<IIC_VUNAQ,    [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<2, [A9_NPipe]>], [6, 2]>,
  //
  // Double-register FP Binary
  // FIXME: We're using this itin for many instructions and [2, 2] here is too
  // optimistic.
  InstrItinData<IIC_VBIND,    [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [5, 2, 2]>,
  //
  // Quad-register FP Binary
  // Result written in N5, but that is relative to the last cycle of multicycle,
  // so we use 6 for those cases
  // FIXME: We're using this itin for many instructions and [2, 2] here is too
  // optimistic.
  InstrItinData<IIC_VBINQ,    [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 8 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<2, [A9_NPipe]>], [6, 2, 2]>,
  //
  // Double-register FP Multiple-Accumulate
  InstrItinData<IIC_VMACD,    [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<2, [A9_NPipe]>], [6, 3, 2, 1]>,
  //
  // Quad-register FP Multiple-Accumulate
  // Result written in N9, but that is relative to the last cycle of multicycle,
  // so we use 10 for those cases
  InstrItinData<IIC_VMACQ,    [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 9 cycles
                               InstrStage<10, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<4, [A9_NPipe]>], [8, 4, 2, 1]>,
  //
  // Double-register Reciprical Step
  InstrItinData<IIC_VRECSD,   [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<2, [A9_NPipe]>], [6, 2, 2]>,
  //
  // Quad-register Reciprical Step
  InstrItinData<IIC_VRECSQ,   [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 9 cycles
                               InstrStage<10, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<4, [A9_NPipe]>], [8, 2, 2]>,
  //
  // Double-register Permute
  InstrItinData<IIC_VPERMD,   [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 6 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [2, 2, 1, 1]>,
  //
  // Quad-register Permute
  // Result written in N2, but that is relative to the last cycle of multicycle,
  // so we use 3 for those cases
  InstrItinData<IIC_VPERMQ,   [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<2, [A9_NPipe]>], [3, 3, 1, 1]>,
  //
  // Quad-register Permute (3 cycle issue)
  // Result written in N2, but that is relative to the last cycle of multicycle,
  // so we use 4 for those cases
  InstrItinData<IIC_VPERMQ3,  [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 8 cycles
                               InstrStage<9, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<3, [A9_LSPipe]>], [4, 4, 1, 1]>,

  //
  // Double-register VEXT
  InstrItinData<IIC_VEXTD,    [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<7, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<1, [A9_NPipe]>], [2, 1, 1]>,
  //
  // Quad-register VEXT
  InstrItinData<IIC_VEXTQ,    [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 9 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<2, [A9_NPipe]>], [3, 1, 1]>,
  //
  // VTB
  InstrItinData<IIC_VTB1,     [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<2, [A9_NPipe]>], [3, 2, 1]>,
  InstrItinData<IIC_VTB2,     [InstrStage<2, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<2, [A9_NPipe]>], [3, 2, 2, 1]>,
  InstrItinData<IIC_VTB3,     [InstrStage<2, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 8 cycles
                               InstrStage<9, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<3, [A9_NPipe]>], [4, 2, 2, 3, 1]>,
  InstrItinData<IIC_VTB4,     [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 8 cycles
                               InstrStage<9, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<3, [A9_NPipe]>], [4, 2, 2, 3, 3, 1]>,
  //
  // VTBX
  InstrItinData<IIC_VTBX1,    [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<2, [A9_NPipe]>], [3, 1, 2, 1]>,
  InstrItinData<IIC_VTBX2,    [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 7 cycles
                               InstrStage<8, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<2, [A9_NPipe]>], [3, 1, 2, 2, 1]>,
  InstrItinData<IIC_VTBX3,    [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 8 cycles
                               InstrStage<9, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<3, [A9_NPipe]>], [4, 1, 2, 2, 3, 1]>,
  InstrItinData<IIC_VTBX4,    [InstrStage<1, [A9_DRegsN],   0, Required>,
                               // Extra latency cycles since wbck is 8 cycles
                               InstrStage<9, [A9_DRegsVFP], 0, Reserved>,
                               InstrStage<1, [A9_Pipe0, A9_Pipe1]>,
                               InstrStage<2, [A9_NPipe]>], [4, 1, 2, 2, 3, 3, 1]>
]>;
