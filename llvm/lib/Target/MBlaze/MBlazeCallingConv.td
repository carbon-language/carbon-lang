//===- MBlazeCallingConv.td - Calling Conventions for MBlaze ----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// This describes the calling conventions for MBlaze architecture.
//===----------------------------------------------------------------------===//

/// CCIfSubtarget - Match if the current subtarget has a feature F.
class CCIfSubtarget<string F, CCAction A>: 
  CCIf<!strconcat("State.getTarget().getSubtarget<MBlazeSubtarget>().", F), A>;

//===----------------------------------------------------------------------===//
// MBlaze ABI Calling Convention
//===----------------------------------------------------------------------===//

def CC_MBlaze : CallingConv<[
  // Promote i8/i16 arguments to i32.
  CCIfType<[i8, i16], CCPromoteToType<i32>>,

  // Integer arguments are passed in integer registers.
  CCIfType<[i32], CCAssignToReg<[R5, R6, R7, R8, R9, R10]>>,

  // Single fp arguments are passed in floating point registers
  CCIfType<[f32], CCAssignToReg<[F5, F6, F7, F8, F9, F10]>>,

  // 32-bit values get stored in stack slots that are 4 bytes in
  // size and 4-byte aligned.
  CCIfType<[i32, f32], CCAssignToStack<4, 4>>
]>;

def RetCC_MBlaze : CallingConv<[
  // i32 are returned in registers R3, R4
  CCIfType<[i32], CCAssignToReg<[R3, R4]>>,

  // f32 are returned in registers F3, F4
  CCIfType<[f32], CCAssignToReg<[F3, F4]>>
]>;
