//===- AMDILCallingConv.td - Calling Conventions AMDIL -----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//
// This describes the calling conventions for the AMDIL architectures.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Return Value Calling Conventions
//===----------------------------------------------------------------------===//

// AMDIL 32-bit C return-value convention.
def RetCC_AMDIL32 : CallingConv<[
 // Since IL has no return values, all values can be emulated on the stack
 // The stack can then be mapped to a number of sequential virtual registers
 // in IL

 // Integer and FP scalar values get put on the stack at 16-byte alignment
 // but with a size of 4 bytes
 CCIfType<[i32, f32], CCAssignToReg<
 [
 R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, R17, R18, R19, R20
]> >, CCAssignToStack<16, 16>]>;

// AMDIL 32-bit C Calling convention.
def CC_AMDIL32 : CallingConv<[
  // Since IL has parameter values, all values can be emulated on the stack
 // The stack can then be mapped to a number of sequential virtual registers
 // in IL
 // Integer and FP scalar values get put on the stack at 16-byte alignment
 // but with a size of 4 bytes
 // Integer and FP scalar values get put on the stack at 16-byte alignment
 // but with a size of 4 bytes
 CCIfType<[i32, f32], CCAssignToReg<
[R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, R17, R18, R19, R20
]> >, CCAssignToStack<16, 16>]>;
