//===--- PTXCallingConv.td - Calling Conventions -----------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This describes the calling conventions for the PTX architecture.
//
//===----------------------------------------------------------------------===//

// Currently, we reserve one register of each type for return values and let
// the rest be used for parameters.  This is a dirty hack, but I am not sure
// how to tell LLVM that registers used for parameter passing cannot be used
// for return values.

// PTX Calling Conventions
def CC_PTX : CallingConv<[
  CCIfType<[i1], CCAssignToReg<[P1, P2, P3, P4, P5, P6, P7]>>,
  CCIfType<[i16], CCAssignToReg<[RH1, RH2, RH3, RH4, RH5, RH6, RH7]>>,
  CCIfType<[i32, f32], CCAssignToReg<[R1, R2, R3, R4, R5, R6, R7]>>,
  CCIfType<[i64, f64], CCAssignToReg<[RD1, RD2, RD3, RD4, RD5, RD6, RD7]>>
]>;

//===----------------------------------------------------------------------===//
// Return Value Calling Conventions
//===----------------------------------------------------------------------===//

def RetCC_PTX : CallingConv<[
  CCIfType<[i1], CCAssignToReg<[P0]>>,
  CCIfType<[i16], CCAssignToReg<[RH0]>>,
  CCIfType<[i32, f32], CCAssignToReg<[R0]>>,
  CCIfType<[i64, f64], CCAssignToReg<[RD0]>>
]>;
