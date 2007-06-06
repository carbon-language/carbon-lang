//===- MipsCallingConv.td - Calling Conventions for Mips --------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bruno Cardoso Lopes and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// This describes the calling conventions for Mips architecture.
//===----------------------------------------------------------------------===//

/// CCIfSubtarget - Match if the current subtarget has a feature F.
class CCIfSubtarget<string F, CCAction A>: 
  CCIf<!strconcat("State.getTarget().getSubtarget<MipsSubtarget>().", F), A>;

//===----------------------------------------------------------------------===//
// Mips Return Value Calling Convention
//===----------------------------------------------------------------------===//
def RetCC_Mips : CallingConv<[
  // i32 are returned in registers V0, V1
  CCIfType<[i32], CCAssignToReg<[V0, V1]>>
]>;


//===----------------------------------------------------------------------===//
// Mips Argument Calling Conventions
//===----------------------------------------------------------------------===//
def CC_Mips : CallingConv<[
  // Promote i8/i16 arguments to i32.
  CCIfType<[i8, i16], CCPromoteToType<i32>>,

  // The first 4 integer arguments are passed in integer registers.
  CCIfType<[i32], CCAssignToReg<[A0, A1, A2, A3]>>,

  // Integer values get stored in stack slots that are 4 bytes in
  // size and 4-byte aligned.
  CCIfType<[i32], CCAssignToStack<4, 4>>
]>;

