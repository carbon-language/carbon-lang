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

/// CCIfFloatABI - Match of the float ABI and the arg. ABIType may be "Hard" or
///                "Soft".
class CCIfFloatABI<string ABIType, CCAction A>:
  CCIf<!strconcat("llvm::FloatABIType == llvm::FloatABI::", ABIType), A>;

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
// ARM AAPCS (EABI) Calling Convention, common parts
//===----------------------------------------------------------------------===//

def CC_ARM_AAPCS_Common : CallingConv<[

  CCIfType<[i8, i16], CCPromoteToType<i32>>,

  // i64/f64 is passed in even pairs of GPRs
  // i64 is 8-aligned i32 here, so we may need to eat R1 as a pad register
  // (and the same is true for f64 if VFP is not enabled)
  CCIfType<[i32], CCIfAlign<"8", CCAssignToRegWithShadow<[R0, R2], [R0, R1]>>>,
  CCIfType<[i32], CCIf<"State.getNextStackOffset() == 0 &&"
                       "ArgFlags.getOrigAlign() != 8",
                       CCAssignToReg<[R0, R1, R2, R3]>>>,

  CCIfType<[i32, f32], CCAssignToStack<4, 4>>,
  CCIfType<[f64], CCAssignToStack<8, 8>>
]>;

def RetCC_ARM_AAPCS_Common : CallingConv<[
  CCIfType<[i32], CCAssignToReg<[R0, R1, R2, R3]>>,
  CCIfType<[i64], CCAssignToRegWithShadow<[R0, R2], [R1, R3]>>
]>;

//===----------------------------------------------------------------------===//
// ARM AAPCS (EABI) Calling Convention
//===----------------------------------------------------------------------===//

def CC_ARM_AAPCS : CallingConv<[
  CCIfType<[f64], CCCustom<"CC_ARM_AAPCS_Custom_f64">>,
  CCIfType<[f32], CCBitConvertToType<i32>>,
  CCDelegateTo<CC_ARM_AAPCS_Common>
]>;

def RetCC_ARM_AAPCS : CallingConv<[
  CCIfType<[f64], CCCustom<"RetCC_ARM_AAPCS_Custom_f64">>,
  CCIfType<[f32], CCBitConvertToType<i32>>,
  CCDelegateTo<RetCC_ARM_AAPCS_Common>
]>;

//===----------------------------------------------------------------------===//
// ARM AAPCS-VFP (EABI) Calling Convention
//===----------------------------------------------------------------------===//

def CC_ARM_AAPCS_VFP : CallingConv<[
  CCIfType<[f64], CCAssignToReg<[D0, D1, D2, D3, D4, D5, D6, D7]>>,
  CCIfType<[f32], CCAssignToReg<[S0, S1, S2, S3, S4, S5, S6, S7, S8,
                                 S9, S10, S11, S12, S13, S14, S15]>>,
  CCDelegateTo<CC_ARM_AAPCS_Common>
]>;

def RetCC_ARM_AAPCS_VFP : CallingConv<[
  CCIfType<[f64], CCAssignToReg<[D0, D1, D2, D3, D4, D5, D6, D7]>>,
  CCIfType<[f32], CCAssignToReg<[S0, S1, S2, S3, S4, S5, S6, S7, S8,
                                 S9, S10, S11, S12, S13, S14, S15]>>,
  CCDelegateTo<RetCC_ARM_AAPCS_Common>
]>;

//===----------------------------------------------------------------------===//
// ARM Calling Convention Dispatch
//===----------------------------------------------------------------------===//

def CC_ARM : CallingConv<[
  CCIfSubtarget<"isAAPCS_ABI()",
                CCIfSubtarget<"hasVFP2()",
                              CCIfFloatABI<"Hard",
                                           CCDelegateTo<CC_ARM_AAPCS_VFP>>>>,
  CCIfSubtarget<"isAAPCS_ABI()", CCDelegateTo<CC_ARM_AAPCS>>,
  CCDelegateTo<CC_ARM_APCS>
]>;

def RetCC_ARM : CallingConv<[
  CCIfSubtarget<"isAAPCS_ABI()",
                CCIfSubtarget<"hasVFP2()",
                              CCIfFloatABI<"Hard",
                                           CCDelegateTo<RetCC_ARM_AAPCS_VFP>>>>,
  CCIfSubtarget<"isAAPCS_ABI()", CCDelegateTo<RetCC_ARM_AAPCS>>,
  CCDelegateTo<RetCC_ARM_APCS>
]>;
