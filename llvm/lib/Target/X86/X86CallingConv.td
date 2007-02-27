//===- X86CallingConv.td - Calling Conventions for X86 32/64 ----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This describes the calling conventions for the X86-32 and X86-64
// architectures.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Return Value Calling Conventions
//===----------------------------------------------------------------------===//

// Return-value conventions common to all X86 CC's.
def RetCC_X86Common : CallingConv<[
  // Scalar values are returned in AX first, then DX.
  CCMatchType<[i8] , CCAssignToReg<[AL]>>,
  CCMatchType<[i16], CCAssignToReg<[AX]>>,
  CCMatchType<[i32], CCAssignToReg<[EAX, EDX]>>,
  CCMatchType<[i64], CCAssignToReg<[RAX, RDX]>>,
  
  // Vector types are always returned in XMM0.  If the target doesn't have XMM0,
  // it won't have vector types.
  CCMatchType<[v16i8, v8i16, v4i32, v2i64, v4f32, v2f64], CCAssignToReg<[XMM0]>>
]>;

// X86-32 C return-value convention.
def RetCC_X86_32_C : CallingConv<[
  // The X86-32 calling convention returns FP values in ST0, otherwise it is the
  // same as the common X86 calling conv.
  CCMatchType<[f32], CCAssignToReg<[ST0]>>,
  CCMatchType<[f64], CCAssignToReg<[ST0]>>,
  CCDelegateTo<RetCC_X86Common>
]>;

// X86-32 FastCC return-value convention.
def RetCC_X86_32_Fast : CallingConv<[
  // The X86-32 fastcc returns FP values in XMM0 if the target has SSE2,
  // otherwise it is the the C calling conventions.
  CCMatchType<[f32], CCMatchIf<"Subtarget->hasSSE2()", CCAssignToReg<[XMM0]>>>,
  CCMatchType<[f64], CCMatchIf<"Subtarget->hasSSE2()", CCAssignToReg<[XMM0]>>>,
  CCDelegateTo<RetCC_X86Common>
]>;

// X86-64 C return-value convention.
def RetCC_X86_64_C : CallingConv<[
  // The X86-64 calling convention always returns FP values in XMM0.
  CCMatchType<[f32], CCAssignToReg<[XMM0]>>,
  CCMatchType<[f64], CCAssignToReg<[XMM0]>>,
  CCDelegateTo<RetCC_X86Common>
]>;



// This is the root return-value convention for the X86-32 backend.
def RetCC_X86_32 : CallingConv<[
  // If FastCC, use RetCC_X86_32_Fast.
  CCMatchIfCC<"CallingConv::Fast", CCDelegateTo<RetCC_X86_32_Fast>>,
  // Otherwise, use RetCC_X86_32_C.
  CCDelegateTo<RetCC_X86_32_C>
]>;

// This is the root return-value convention for the X86-64 backend.
def RetCC_X86_64 : CallingConv<[
  // Always just the same as C calling conv for X86-64.
  CCDelegateTo<RetCC_X86_64_C>
]>;

// This is the return-value convention used for the entire X86 backend.
def RetCC_X86 : CallingConv<[
  CCMatchIf<"Subtarget->is64Bit()", CCDelegateTo<RetCC_X86_64>>,
  CCDelegateTo<RetCC_X86_32>
]>;

//===----------------------------------------------------------------------===//
// Argument Calling Conventions
//===----------------------------------------------------------------------===//


def CC_X86_64_C : CallingConv<[
  // Promote i8/i16 arguments to i32.
  CCMatchType<[i8, i16], CCPromoteToType<i32>>,
  
  // The first 6 integer arguments are passed in integer registers.
  CCMatchType<[i32], CCAssignToReg<[EDI, ESI, EDX, ECX, R8D, R9D]>>,
  CCMatchType<[i64], CCAssignToReg<[RDI, RSI, RDX, RCX, R8 , R9 ]>>,
  
  // The first 8 FP/Vector arguments are passed in XMM registers.
  CCMatchType<[f32, f64, v16i8, v8i16, v4i32, v2i64, v4f32, v2f64],
              CCAssignToReg<[XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7]>>,

  // Integer/FP values get stored in stack slots that are 8 bytes in size and
  // 8-byte aligned if there are no more registers to hold them.
  CCMatchType<[i32, i64, f32, f64], CCAssignToStack<8, 8>>,
  
  // Vectors get 16-byte stack slots that are 16-byte aligned.
  CCMatchType<[v16i8, v8i16, v4i32, v2i64, v4f32, v2f64],
              CCAssignToStack<16, 16>>
]>;


