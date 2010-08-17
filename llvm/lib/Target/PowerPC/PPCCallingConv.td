//===- PPCCallingConv.td - Calling Conventions for PowerPC -*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This describes the calling conventions for the PowerPC 32- and 64-bit
// architectures.
//
//===----------------------------------------------------------------------===//

/// CCIfSubtarget - Match if the current subtarget has a feature F.
class CCIfSubtarget<string F, CCAction A>
 : CCIf<!strconcat("State.getTarget().getSubtarget<PPCSubtarget>().", F), A>;

//===----------------------------------------------------------------------===//
// Return Value Calling Convention
//===----------------------------------------------------------------------===//

// Return-value convention for PowerPC
def RetCC_PPC : CallingConv<[
  CCIfType<[i32], CCAssignToReg<[R3, R4, R5, R6, R7, R8, R9, R10]>>,
  CCIfType<[i64], CCAssignToReg<[X3, X4, X5, X6]>>,
  
  CCIfType<[f32], CCAssignToReg<[F1]>>,
  CCIfType<[f64], CCAssignToReg<[F1, F2]>>,
  
  // Vector types are always returned in V2.
  CCIfType<[v16i8, v8i16, v4i32, v4f32], CCAssignToReg<[V2]>>
]>;


//===----------------------------------------------------------------------===//
// PowerPC Argument Calling Conventions
//===----------------------------------------------------------------------===//
/*
def CC_PPC : CallingConv<[
  // The first 8 integer arguments are passed in integer registers.
  CCIfType<[i32], CCAssignToReg<[R3, R4, R5, R6, R7, R8, R9, R10]>>,
  CCIfType<[i64], CCAssignToReg<[X3, X4, X5, X6, X7, X8, X9, X10]>>,
  
  // Common sub-targets passes FP values in F1 - F13
  CCIfType<[f32, f64], 
           CCAssignToReg<[F1, F2, F3, F4, F5, F6, F7, F8,F9,F10,F11,F12,F13]>>,
           
  // The first 12 Vector arguments are passed in altivec registers.
  CCIfType<[v16i8, v8i16, v4i32, v4f32],
              CCAssignToReg<[V2, V3, V4, V5, V6, V7, V8, V9, V10,V11,V12,V13]>>

/*
  // Integer/FP values get stored in stack slots that are 8 bytes in size and
  // 8-byte aligned if there are no more registers to hold them.
  CCIfType<[i32, i64, f32, f64], CCAssignToStack<8, 8>>,
  
  // Vectors get 16-byte stack slots that are 16-byte aligned.
  CCIfType<[v16i8, v8i16, v4i32, v2i64, v4f32, v2f64],
              CCAssignToStack<16, 16>>*/
]>;

*/

//===----------------------------------------------------------------------===//
// PowerPC System V Release 4 ABI
//===----------------------------------------------------------------------===//

def CC_PPC_SVR4_Common : CallingConv<[
  // The ABI requires i64 to be passed in two adjacent registers with the first
  // register having an odd register number.
  CCIfType<[i32], CCIfSplit<CCCustom<"CC_PPC_SVR4_Custom_AlignArgRegs">>>,

  // The first 8 integer arguments are passed in integer registers.
  CCIfType<[i32], CCAssignToReg<[R3, R4, R5, R6, R7, R8, R9, R10]>>,

  // Make sure the i64 words from a long double are either both passed in
  // registers or both passed on the stack.
  CCIfType<[f64], CCIfSplit<CCCustom<"CC_PPC_SVR4_Custom_AlignFPArgRegs">>>,
  
  // FP values are passed in F1 - F8.
  CCIfType<[f32, f64], CCAssignToReg<[F1, F2, F3, F4, F5, F6, F7, F8]>>,

  // Split arguments have an alignment of 8 bytes on the stack.
  CCIfType<[i32], CCIfSplit<CCAssignToStack<4, 8>>>,
  
  CCIfType<[i32], CCAssignToStack<4, 4>>,
  
  // Floats are stored in double precision format, thus they have the same
  // alignment and size as doubles.
  CCIfType<[f32,f64], CCAssignToStack<8, 8>>,  

  // Vectors get 16-byte stack slots that are 16-byte aligned.
  CCIfType<[v16i8, v8i16, v4i32, v4f32], CCAssignToStack<16, 16>>
]>;

// This calling convention puts vector arguments always on the stack. It is used
// to assign vector arguments which belong to the variable portion of the
// parameter list of a variable argument function.
def CC_PPC_SVR4_VarArg : CallingConv<[
  CCDelegateTo<CC_PPC_SVR4_Common>
]>;

// In contrast to CC_PPC_SVR4_VarArg, this calling convention first tries to put
// vector arguments in vector registers before putting them on the stack.
def CC_PPC_SVR4 : CallingConv<[
  // The first 12 Vector arguments are passed in AltiVec registers.
  CCIfType<[v16i8, v8i16, v4i32, v4f32],
           CCAssignToReg<[V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13]>>,
           
  CCDelegateTo<CC_PPC_SVR4_Common>
]>;  

// Helper "calling convention" to handle aggregate by value arguments.
// Aggregate by value arguments are always placed in the local variable space
// of the caller. This calling convention is only used to assign those stack
// offsets in the callers stack frame.
//
// Still, the address of the aggregate copy in the callers stack frame is passed
// in a GPR (or in the parameter list area if all GPRs are allocated) from the
// caller to the callee. The location for the address argument is assigned by
// the CC_PPC_SVR4 calling convention.
//
// The only purpose of CC_PPC_SVR4_Custom_Dummy is to skip arguments which are
// not passed by value.
 
def CC_PPC_SVR4_ByVal : CallingConv<[
  CCIfByVal<CCPassByVal<4, 4>>,
  
  CCCustom<"CC_PPC_SVR4_Custom_Dummy">
]>;

