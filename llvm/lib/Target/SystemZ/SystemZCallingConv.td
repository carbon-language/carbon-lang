//=- SystemZCallingConv.td - Calling conventions for SystemZ -*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This describes the calling conventions for the SystemZ ABI.
//===----------------------------------------------------------------------===//

class CCIfExtend<CCAction A>
  : CCIf<"ArgFlags.isSExt() || ArgFlags.isZExt()", A>;

//===----------------------------------------------------------------------===//
// z/Linux return value calling convention
//===----------------------------------------------------------------------===//
def RetCC_SystemZ : CallingConv<[
  // Promote i32 to i64 if it has an explicit extension type.
  CCIfType<[i32], CCIfExtend<CCPromoteToType<i64>>>,

  // ABI-compliant code returns 64-bit integers in R2.  Make the other
  // call-clobbered argument registers available for code that doesn't
  // care about the ABI.  (R6 is an argument register too, but is
  // call-saved and therefore not suitable for return values.)
  CCIfType<[i32], CCAssignToReg<[R2L, R3L, R4L, R5L]>>,
  CCIfType<[i64], CCAssignToReg<[R2D, R3D, R4D, R5D]>>,

  // ABI-complaint code returns float and double in F0.  Make the
  // other floating-point argument registers available for code that
  // doesn't care about the ABI.  All floating-point argument registers
  // are call-clobbered, so we can use all of them here.
  CCIfType<[f32], CCAssignToReg<[F0S, F2S, F4S, F6S]>>,
  CCIfType<[f64], CCAssignToReg<[F0D, F2D, F4D, F6D]>>

  // ABI-compliant code returns long double by reference, but that conversion
  // is left to higher-level code.  Perhaps we could add an f128 definition
  // here for code that doesn't care about the ABI?
]>;

//===----------------------------------------------------------------------===//
// z/Linux argument calling conventions
//===----------------------------------------------------------------------===//
def CC_SystemZ : CallingConv<[
  // Promote i32 to i64 if it has an explicit extension type.
  // The convention is that true integer arguments that are smaller
  // than 64 bits should be marked as extended, but structures that
  // are smaller than 64 bits shouldn't.
  CCIfType<[i32], CCIfExtend<CCPromoteToType<i64>>>,

  // Force long double values to the stack and pass i64 pointers to them.
  CCIfType<[f128], CCPassIndirect<i64>>,

  // The first 5 integer arguments are passed in R2-R6.  Note that R6
  // is call-saved.
  CCIfType<[i32], CCAssignToReg<[R2L, R3L, R4L, R5L, R6L]>>,
  CCIfType<[i64], CCAssignToReg<[R2D, R3D, R4D, R5D, R6D]>>,

  // The first 4 float and double arguments are passed in even registers F0-F6.
  CCIfType<[f32], CCAssignToReg<[F0S, F2S, F4S, F6S]>>,
  CCIfType<[f64], CCAssignToReg<[F0D, F2D, F4D, F6D]>>,

  // Other arguments are passed in 8-byte-aligned 8-byte stack slots.
  CCIfType<[i32, i64, f32, f64], CCAssignToStack<8, 8>>
]>;

//===----------------------------------------------------------------------===//
// z/Linux callee-saved registers
//===----------------------------------------------------------------------===//
def CSR_SystemZ : CalleeSavedRegs<(add (sequence "R%dD", 6, 15),
                                       (sequence "F%dD", 8, 15))>;
