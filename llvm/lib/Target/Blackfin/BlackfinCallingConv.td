//===--- BlackfinCallingConv.td - Calling Conventions ------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This describes the calling conventions for the Blackfin architectures.
//
//===----------------------------------------------------------------------===//

// Blackfin C Calling convention.
def CC_Blackfin : CallingConv<[
  CCIfType<[i16], CCPromoteToType<i32>>,
  CCIfSRet<CCAssignToReg<[P0]>>,
  CCAssignToReg<[R0, R1, R2]>,
  CCAssignToStack<4, 4>
]>;

//===----------------------------------------------------------------------===//
// Return Value Calling Conventions
//===----------------------------------------------------------------------===//

// Blackfin C return-value convention.
def RetCC_Blackfin : CallingConv<[
  CCIfType<[i16], CCPromoteToType<i32>>,
  CCAssignToReg<[R0, R1]>
]>;
