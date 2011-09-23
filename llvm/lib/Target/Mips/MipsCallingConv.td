//===- MipsCallingConv.td - Calling Conventions for Mips ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This describes the calling conventions for Mips architecture.
//===----------------------------------------------------------------------===//

/// CCIfSubtarget - Match if the current subtarget has a feature F.
class CCIfSubtarget<string F, CCAction A>:
  CCIf<!strconcat("State.getTarget().getSubtarget<MipsSubtarget>().", F), A>;

//===----------------------------------------------------------------------===//
// Mips O32 Calling Convention
//===----------------------------------------------------------------------===//

// Only the return rules are defined here for O32. The rules for argument
// passing are defined in MipsISelLowering.cpp.
def RetCC_MipsO32 : CallingConv<[
  // i32 are returned in registers V0, V1, A0, A1
  CCIfType<[i32], CCAssignToReg<[V0, V1, A0, A1]>>,

  // f32 are returned in registers F0, F2
  CCIfType<[f32], CCAssignToReg<[F0, F2]>>,

  // f64 are returned in register D0, D1
  CCIfType<[f64], CCIfSubtarget<"isNotSingleFloat()", CCAssignToReg<[D0, D1]>>>
]>;

//===----------------------------------------------------------------------===//
// Mips N32/64 Calling Convention
//===----------------------------------------------------------------------===//

def CC_MipsN : CallingConv<[
  // FIXME: Handle byval, complex and float double parameters.

  // Promote i8/i16/i32 arguments to i64.
  CCIfType<[i8, i16, i32], CCPromoteToType<i64>>,

  // Integer arguments are passed in integer registers.
  CCIfType<[i64], CCAssignToRegWithShadow<[A0_64, A1_64, A2_64, A3_64,
                                           T0_64, T1_64, T2_64, T3_64],
                                          [D12_64, D13_64, D14_64, D15_64,
                                           D16_64, D17_64, D18_64, D19_64]>>,

  // f32 arguments are passed in single precision FP registers.
  CCIfType<[f32], CCAssignToRegWithShadow<[F12, F13, F14, F15,
                                           F16, F17, F18, F19],
                                          [A0_64, A1_64, A2_64, A3_64,
                                           T0_64, T1_64, T2_64, T3_64]>>,

  // f64 arguments are passed in double precision FP registers.
  CCIfType<[f64], CCAssignToRegWithShadow<[D12_64, D13_64, D14_64, D15_64,
                                           D16_64, D17_64, D18_64, D19_64],
                                          [A0_64, A1_64, A2_64, A3_64,
                                           T0_64, T1_64, T2_64, T3_64]>>,

  // All stack parameter slots become 64-bit doublewords and are 8-byte aligned.
  CCIfType<[i64, f64], CCAssignToStack<8, 8>>,
  CCIfType<[f32], CCAssignToStack<4, 8>>
]>;

def RetCC_MipsN : CallingConv<[
  // FIXME: Handle complex and float double return values.

  // i32 are returned in registers V0, V1
  CCIfType<[i32], CCAssignToReg<[V0, V1]>>,

  // i64 are returned in registers V0_64, V1_64
  CCIfType<[i64], CCAssignToReg<[V0_64, V1_64]>>,

  // f32 are returned in registers F0, F2
  CCIfType<[f32], CCAssignToReg<[F0, F2]>>,

  // f64 are returned in registers D0, D2
  CCIfType<[f64], CCAssignToReg<[D0_64, D2_64]>>
]>;

//===----------------------------------------------------------------------===//
// Mips EABI Calling Convention
//===----------------------------------------------------------------------===//

def CC_MipsEABI : CallingConv<[
  // Promote i8/i16 arguments to i32.
  CCIfType<[i8, i16], CCPromoteToType<i32>>,

  // Integer arguments are passed in integer registers.
  CCIfType<[i32], CCAssignToReg<[A0, A1, A2, A3, T0, T1, T2, T3]>>,

  // Single fp arguments are passed in pairs within 32-bit mode
  CCIfType<[f32], CCIfSubtarget<"isSingleFloat()",
                  CCAssignToReg<[F12, F13, F14, F15, F16, F17, F18, F19]>>>,

  CCIfType<[f32], CCIfSubtarget<"isNotSingleFloat()",
                  CCAssignToReg<[F12, F14, F16, F18]>>>,

  // The first 4 double fp arguments are passed in single fp registers.
  CCIfType<[f64], CCIfSubtarget<"isNotSingleFloat()",
                  CCAssignToReg<[D6, D7, D8, D9]>>>,

  // Integer values get stored in stack slots that are 4 bytes in
  // size and 4-byte aligned.
  CCIfType<[i32, f32], CCAssignToStack<4, 4>>,

  // Integer values get stored in stack slots that are 8 bytes in
  // size and 8-byte aligned.
  CCIfType<[f64], CCIfSubtarget<"isNotSingleFloat()", CCAssignToStack<8, 8>>>
]>;

def RetCC_MipsEABI : CallingConv<[
  // i32 are returned in registers V0, V1
  CCIfType<[i32], CCAssignToReg<[V0, V1]>>,

  // f32 are returned in registers F0, F1
  CCIfType<[f32], CCAssignToReg<[F0, F1]>>,

  // f64 are returned in register D0
  CCIfType<[f64], CCIfSubtarget<"isNotSingleFloat()", CCAssignToReg<[D0]>>>
]>;

//===----------------------------------------------------------------------===//
// Mips Calling Convention Dispatch
//===----------------------------------------------------------------------===//

def CC_Mips : CallingConv<[
  CCIfSubtarget<"isABI_EABI()", CCDelegateTo<CC_MipsEABI>>,
  CCIfSubtarget<"isABI_N32()", CCDelegateTo<CC_MipsN>>,
  CCIfSubtarget<"isABI_N64()", CCDelegateTo<CC_MipsN>>
]>;

def RetCC_Mips : CallingConv<[
  CCIfSubtarget<"isABI_EABI()", CCDelegateTo<RetCC_MipsEABI>>,
  CCIfSubtarget<"isABI_N32()", CCDelegateTo<RetCC_MipsN>>,
  CCIfSubtarget<"isABI_N64()", CCDelegateTo<RetCC_MipsN>>,
  CCDelegateTo<RetCC_MipsO32>
]>;
