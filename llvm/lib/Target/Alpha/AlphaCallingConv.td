//===- AlphaCallingConv.td - Calling Conventions for Alpha -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This describes the calling conventions for Alpha architecture.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Alpha Return Value Calling Convention
//===----------------------------------------------------------------------===//
def RetCC_Alpha : CallingConv<[
  // i64 is returned in register R0
  CCIfType<[i64], CCAssignToReg<[R0]>>,

  // f32 / f64 are returned in F0/F1
  CCIfType<[f32, f64], CCAssignToReg<[F0, F1]>>
]>;

//===----------------------------------------------------------------------===//
// Alpha Argument Calling Conventions
//===----------------------------------------------------------------------===//
def CC_Alpha : CallingConv<[
  // The first 6 arguments are passed in registers, whether integer or
  // floating-point
  CCIfType<[i64], CCAssignToRegWithShadow<[R16, R17, R18, R19, R20, R21],
                                          [F16, F17, F18, F19, F20, F21]>>,

  CCIfType<[f32, f64], CCAssignToRegWithShadow<[F16, F17, F18, F19, F20, F21],
                                               [R16, R17, R18, R19, R20, R21]>>,

  // Stack slots are 8 bytes in size and 8-byte aligned.
  CCIfType<[i64, f32, f64], CCAssignToStack<8, 8>>
]>;
