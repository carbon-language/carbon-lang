//===- ARMCallingConv.td - Calling Conventions for ARM ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This describes the calling conventions for ARM architecture.
//===----------------------------------------------------------------------===//

/// CCIfSubtarget - Match if the current subtarget has a feature F.
class CCIfSubtarget<string F, CCAction A>:
  CCIf<!strconcat("State.getTarget().getSubtarget<ARMSubtarget>().", F), A>;

/// CCIfAlign - Match of the original alignment of the arg
class CCIfAlign<string Align, CCAction A>:
  CCIf<!strconcat("ArgFlags.getOrigAlign() == ", Align), A>;

//===----------------------------------------------------------------------===//
// ARM APCS Calling Convention
//===----------------------------------------------------------------------===//
def CC_ARM_APCS : CallingConv<[

  CCIfType<[i8, i16], CCPromoteToType<i32>>,

  // f64 is passed in pairs of GPRs, possibly split onto the stack
  CCIfType<[f64], CCCustom<"CC_ARM_APCS_Custom_f64">>,

  CCIfType<[f32], CCBitConvertToType<i32>>,
  CCIfType<[i32], CCAssignToReg<[R0, R1, R2, R3]>>,

  CCIfType<[i32], CCAssignToStack<4, 4>>,
  CCIfType<[f64], CCAssignToStack<8, 4>>
]>;

def RetCC_ARM_APCS : CallingConv<[
  CCIfType<[f32], CCBitConvertToType<i32>>,
  CCIfType<[f64], CCCustom<"RetCC_ARM_APCS_Custom_f64">>,

  CCIfType<[i32], CCAssignToReg<[R0, R1, R2, R3]>>,
  CCIfType<[i64], CCAssignToRegWithShadow<[R0, R2], [R1, R3]>>
]>;

//===----------------------------------------------------------------------===//
// ARM AAPCS (EABI) Calling Convention
//===----------------------------------------------------------------------===//
def CC_ARM_AAPCS : CallingConv<[

  CCIfType<[i8, i16], CCPromoteToType<i32>>,

  // i64/f64 is passed in even pairs of GPRs
  // i64 is 8-aligned i32 here, so we may need to eat R1 as a pad register
  CCIfType<[i32], CCIfAlign<"8", CCAssignToRegWithShadow<[R0, R2], [R0, R1]>>>,
  CCIfType<[f64], CCCustom<"CC_ARM_AAPCS_Custom_f64">>,

  CCIfType<[f32], CCBitConvertToType<i32>>,
  CCIfType<[i32], CCAssignToReg<[R0, R1, R2, R3]>>,

  CCIfType<[i32], CCAssignToStack<4, 4>>,
  CCIfType<[f64], CCAssignToStack<8, 8>>
]>;

def RetCC_ARM_AAPCS : CallingConv<[
  CCIfType<[f32], CCBitConvertToType<i32>>,
  CCIfType<[f64], CCCustom<"RetCC_ARM_AAPCS_Custom_f64">>,

  CCIfType<[i32], CCAssignToReg<[R0, R1, R2, R3]>>,
  CCIfType<[i64], CCAssignToRegWithShadow<[R0, R2], [R1, R3]>>
]>;

//===----------------------------------------------------------------------===//
// ARM Calling Convention Dispatch
//===----------------------------------------------------------------------===//

def CC_ARM : CallingConv<[
  CCIfSubtarget<"isAAPCS_ABI()", CCDelegateTo<CC_ARM_AAPCS>>,
  CCDelegateTo<CC_ARM_APCS>
]>;

def RetCC_ARM : CallingConv<[
  CCIfSubtarget<"isAAPCS_ABI()", CCDelegateTo<RetCC_ARM_AAPCS>>,
  CCDelegateTo<RetCC_ARM_APCS>
]>;
