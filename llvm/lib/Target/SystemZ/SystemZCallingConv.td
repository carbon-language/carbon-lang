//=- SystemZCallingConv.td - Calling Conventions for SystemZ -*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This describes the calling conventions for SystemZ architecture.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SystemZ Return Value Calling Convention
//===----------------------------------------------------------------------===//
def RetCC_SystemZ : CallingConv<[
  // Promote i8/i16/i32 arguments to i64.
  CCIfType<[i8, i16, i32], CCPromoteToType<i64>>,

  // i64 is returned in register R2
  CCIfType<[i64], CCAssignToReg<[R2D, R3D, R4D, R5D]>>,

  // f32 / f64 are returned in F0
  CCIfType<[f32], CCAssignToReg<[F0S, F2S, F4S, F6S]>>,
  CCIfType<[f64], CCAssignToReg<[F0L, F2L, F4L, F6L]>>
]>;

//===----------------------------------------------------------------------===//
// SystemZ Argument Calling Conventions
//===----------------------------------------------------------------------===//
def CC_SystemZ : CallingConv<[
  // Promote i8/i16/i32 arguments to i64.
  CCIfType<[i8, i16, i32], CCPromoteToType<i64>>,

  // The first 5 integer arguments of non-varargs functions are passed in
  // integer registers.
  CCIfType<[i64], CCAssignToReg<[R2D, R3D, R4D, R5D, R6D]>>,

  // The first 4 floating point arguments of non-varargs functions are passed
  // in FP registers.
  CCIfType<[f32], CCAssignToReg<[F0S, F2S, F4S, F6S]>>,
  CCIfType<[f64], CCAssignToReg<[F0L, F2L, F4L, F6L]>>,

  // Integer values get stored in stack slots that are 8 bytes in
  // size and 8-byte aligned.
  CCIfType<[i64, f32, f64], CCAssignToStack<8, 8>>
]>;
