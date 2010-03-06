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

def RetCC_MBlaze : CallingConv<[
  // i32 are returned in registers R3, R4
  CCIfType<[i32], CCAssignToReg<[R3, R4]>>,

  // f32 are returned in registers F3, F4
  CCIfType<[f32], CCAssignToReg<[F3, F4]>>
]>;
