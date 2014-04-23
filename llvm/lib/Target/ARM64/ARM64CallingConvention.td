//===- ARM64CallingConv.td - Calling Conventions for ARM64 -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This describes the calling conventions for ARM64 architecture.
//
//===----------------------------------------------------------------------===//

/// CCIfAlign - Match of the original alignment of the arg
class CCIfAlign<string Align, CCAction A> :
  CCIf<!strconcat("ArgFlags.getOrigAlign() == ", Align), A>;

//===----------------------------------------------------------------------===//
// ARM AAPCS64 Calling Convention
//===----------------------------------------------------------------------===//

def CC_ARM64_AAPCS : CallingConv<[
  CCIfType<[v2f32], CCBitConvertToType<v2i32>>,
  CCIfType<[v2f64, v4f32], CCBitConvertToType<v2i64>>,

  // An SRet is passed in X8, not X0 like a normal pointer parameter.
  CCIfSRet<CCIfType<[i64], CCAssignToRegWithShadow<[X8], [W8]>>>,

  // Put ByVal arguments directly on the stack. Minimum size and alignment of a
  // slot is 64-bit.
  CCIfByVal<CCPassByVal<8, 8>>,

  // Handle i1, i8, i16, i32, i64, f32, f64 and v2f64 by passing in registers,
  // up to eight each of GPR and FPR.
  CCIfType<[i1, i8, i16], CCCustom<"CC_ARM64_Custom_i1i8i16_Reg">>,
  CCIfType<[i32], CCAssignToRegWithShadow<[W0, W1, W2, W3, W4, W5, W6, W7],
                                          [X0, X1, X2, X3, X4, X5, X6, X7]>>,
  // i128 is split to two i64s, we can't fit half to register X7.
  CCIfType<[i64], CCIfSplit<CCAssignToRegWithShadow<[X0, X2, X4, X6],
                                                    [X0, X1, X3, X5]>>>,

  // i128 is split to two i64s, and its stack alignment is 16 bytes.
  CCIfType<[i64], CCIfSplit<CCAssignToStackWithShadow<8, 16, [X7]>>>,

  CCIfType<[i64], CCAssignToRegWithShadow<[X0, X1, X2, X3, X4, X5, X6, X7],
                                          [W0, W1, W2, W3, W4, W5, W6, W7]>>,
  CCIfType<[f32], CCAssignToRegWithShadow<[S0, S1, S2, S3, S4, S5, S6, S7],
                                          [Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7]>>,
  CCIfType<[f64], CCAssignToRegWithShadow<[D0, D1, D2, D3, D4, D5, D6, D7],
                                          [Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7]>>,
  CCIfType<[v1i64, v2i32, v4i16, v8i8, v1f64, v2f32],
           CCAssignToRegWithShadow<[D0, D1, D2, D3, D4, D5, D6, D7],
                                   [Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7]>>,
  CCIfType<[f128, v2i64, v4i32, v8i16, v16i8, v4f32, v2f64],
           CCAssignToReg<[Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7]>>,

  // If more than will fit in registers, pass them on the stack instead.
  CCIfType<[i1, i8, i16], CCAssignToStack<8, 8>>,
  CCIfType<[i32, f32], CCAssignToStack<8, 8>>,
  CCIfType<[i64, f64, v1f64, v2f32, v1i64, v2i32, v4i16, v8i8],
           CCAssignToStack<8, 8>>,
  CCIfType<[v2i64, v4i32, v8i16, v16i8, v4f32, v2f64], CCAssignToStack<16, 16>>
]>;

def RetCC_ARM64_AAPCS : CallingConv<[
  CCIfType<[v2f32], CCBitConvertToType<v2i32>>,
  CCIfType<[v2f64, v4f32], CCBitConvertToType<v2i64>>,

  CCIfType<[i32], CCAssignToRegWithShadow<[W0, W1, W2, W3, W4, W5, W6, W7],
                                          [X0, X1, X2, X3, X4, X5, X6, X7]>>,
  CCIfType<[i64], CCAssignToRegWithShadow<[X0, X1, X2, X3, X4, X5, X6, X7],
                                          [W0, W1, W2, W3, W4, W5, W6, W7]>>,
  CCIfType<[f32], CCAssignToRegWithShadow<[S0, S1, S2, S3, S4, S5, S6, S7],
                                          [Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7]>>,
  CCIfType<[f64], CCAssignToRegWithShadow<[D0, D1, D2, D3, D4, D5, D6, D7],
                                          [Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7]>>,
  CCIfType<[v1i64, v2i32, v4i16, v8i8, v1f64, v2f32],
      CCAssignToRegWithShadow<[D0, D1, D2, D3, D4, D5, D6, D7],
                              [Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7]>>,
  CCIfType<[f128, v2i64, v4i32, v8i16, v16i8, v4f32, v2f64],
      CCAssignToReg<[Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7]>>
]>;


// Darwin uses a calling convention which differs in only two ways
// from the standard one at this level:
//     + i128s (i.e. split i64s) don't need even registers.
//     + Stack slots are sized as needed rather than being at least 64-bit.
def CC_ARM64_DarwinPCS : CallingConv<[
  CCIfType<[v2f32], CCBitConvertToType<v2i32>>,
  CCIfType<[v2f64, v4f32, f128], CCBitConvertToType<v2i64>>,

  // An SRet is passed in X8, not X0 like a normal pointer parameter.
  CCIfSRet<CCIfType<[i64], CCAssignToRegWithShadow<[X8], [W8]>>>,

  // Put ByVal arguments directly on the stack. Minimum size and alignment of a
  // slot is 64-bit.
  CCIfByVal<CCPassByVal<8, 8>>,

  // Handle i1, i8, i16, i32, i64, f32, f64 and v2f64 by passing in registers,
  // up to eight each of GPR and FPR.
  CCIfType<[i1, i8, i16], CCCustom<"CC_ARM64_Custom_i1i8i16_Reg">>,
  CCIfType<[i32], CCAssignToRegWithShadow<[W0, W1, W2, W3, W4, W5, W6, W7],
                                          [X0, X1, X2, X3, X4, X5, X6, X7]>>,
  // i128 is split to two i64s, we can't fit half to register X7.
  CCIfType<[i64],
           CCIfSplit<CCAssignToRegWithShadow<[X0, X1, X2, X3, X4, X5, X6],
                                             [W0, W1, W2, W3, W4, W5, W6]>>>,
  // i128 is split to two i64s, and its stack alignment is 16 bytes.
  CCIfType<[i64], CCIfSplit<CCAssignToStackWithShadow<8, 16, [X7]>>>,

  CCIfType<[i64], CCAssignToRegWithShadow<[X0, X1, X2, X3, X4, X5, X6, X7],
                                          [W0, W1, W2, W3, W4, W5, W6, W7]>>,
  CCIfType<[f32], CCAssignToRegWithShadow<[S0, S1, S2, S3, S4, S5, S6, S7],
                                          [Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7]>>,
  CCIfType<[f64], CCAssignToRegWithShadow<[D0, D1, D2, D3, D4, D5, D6, D7],
                                          [Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7]>>,
  CCIfType<[v1i64, v2i32, v4i16, v8i8, v1f64, v2f32],
           CCAssignToRegWithShadow<[D0, D1, D2, D3, D4, D5, D6, D7],
                                   [Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7]>>,
  CCIfType<[v2i64, v4i32, v8i16, v16i8, v4f32, v2f64],
           CCAssignToReg<[Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7]>>,

  // If more than will fit in registers, pass them on the stack instead.
  CCIfType<[i1, i8, i16], CCCustom<"CC_ARM64_Custom_i1i8i16_Stack">>,
  CCIfType<[i32, f32], CCAssignToStack<4, 4>>,
  CCIfType<[i64, f64, v1f64, v2f32, v1i64, v2i32, v4i16, v8i8],
           CCAssignToStack<8, 8>>,
  CCIfType<[v2i64, v4i32, v8i16, v16i8, v4f32, v2f64], CCAssignToStack<16, 16>>
]>;

def CC_ARM64_DarwinPCS_VarArg : CallingConv<[
  CCIfType<[v2f32], CCBitConvertToType<v2i32>>,
  CCIfType<[v2f64, v4f32, f128], CCBitConvertToType<v2i64>>,

  // Handle all scalar types as either i64 or f64.
  CCIfType<[i8, i16, i32], CCPromoteToType<i64>>,
  CCIfType<[f32],          CCPromoteToType<f64>>,

  // Everything is on the stack.
  // i128 is split to two i64s, and its stack alignment is 16 bytes.
  CCIfType<[i64], CCIfSplit<CCAssignToStack<8, 16>>>,
  CCIfType<[i64, f64, v1i64, v2i32, v4i16, v8i8, v1f64, v2f32], CCAssignToStack<8, 8>>,
  CCIfType<[v2i64, v4i32, v8i16, v16i8, v4f32, v2f64],   CCAssignToStack<16, 16>>
]>;

// The WebKit_JS calling convention only passes the first argument (the callee)
// in register and the remaining arguments on stack. We allow 32bit stack slots,
// so that WebKit can write partial values in the stack and define the other
// 32bit quantity as undef.
def CC_ARM64_WebKit_JS : CallingConv<[
  // Handle i1, i8, i16, i32, and i64 passing in register X0 (W0).
  CCIfType<[i1, i8, i16], CCCustom<"CC_ARM64_WebKit_JS_i1i8i16_Reg">>,
  CCIfType<[i32], CCAssignToRegWithShadow<[W0], [X0]>>,
  CCIfType<[i64], CCAssignToRegWithShadow<[X0], [W0]>>,

  // Pass the remaining arguments on the stack instead.
  CCIfType<[i1, i8, i16], CCAssignToStack<4, 4>>,
  CCIfType<[i32, f32], CCAssignToStack<4, 4>>,
  CCIfType<[i64, f64], CCAssignToStack<8, 8>>
]>;

def RetCC_ARM64_WebKit_JS : CallingConv<[
  CCIfType<[i32], CCAssignToRegWithShadow<[W0, W1, W2, W3, W4, W5, W6, W7],
                                          [X0, X1, X2, X3, X4, X5, X6, X7]>>,
  CCIfType<[i64], CCAssignToRegWithShadow<[X0, X1, X2, X3, X4, X5, X6, X7],
                                          [W0, W1, W2, W3, W4, W5, W6, W7]>>,
  CCIfType<[f32], CCAssignToRegWithShadow<[S0, S1, S2, S3, S4, S5, S6, S7],
                                          [Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7]>>,
  CCIfType<[f64], CCAssignToRegWithShadow<[D0, D1, D2, D3, D4, D5, D6, D7],
                                          [Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7]>>
]>;

// FIXME: LR is only callee-saved in the sense that *we* preserve it and are
// presumably a callee to someone. External functions may not do so, but this
// is currently safe since BL has LR as an implicit-def and what happens after a
// tail call doesn't matter.
//
// It would be better to model its preservation semantics properly (create a
// vreg on entry, use it in RET & tail call generation; make that vreg def if we
// end up saving LR as part of a call frame). Watch this space...
def CSR_ARM64_AAPCS : CalleeSavedRegs<(add LR, FP, X19, X20, X21, X22,
                                           X23, X24, X25, X26, X27, X28,
                                           D8,  D9,  D10, D11,
                                           D12, D13, D14, D15)>;

// Constructors and destructors return 'this' in the iOS 64-bit C++ ABI; since
// 'this' and the pointer return value are both passed in X0 in these cases,
// this can be partially modelled by treating X0 as a callee-saved register;
// only the resulting RegMask is used; the SaveList is ignored
//
// (For generic ARM 64-bit ABI code, clang will not generate constructors or
// destructors with 'this' returns, so this RegMask will not be used in that
// case)
def CSR_ARM64_AAPCS_ThisReturn : CalleeSavedRegs<(add CSR_ARM64_AAPCS, X0)>;

// The function used by Darwin to obtain the address of a thread-local variable
// guarantees more than a normal AAPCS function. x16 and x17 are used on the
// fast path for calculation, but other registers except X0 (argument/return)
// and LR (it is a call, after all) are preserved.
def CSR_ARM64_TLS_Darwin
    : CalleeSavedRegs<(add (sub (sequence "X%u", 1, 28), X16, X17),
                           FP,
                           (sequence "Q%u", 0, 31))>;

// The ELF stub used for TLS-descriptor access saves every feasible
// register. Only X0 and LR are clobbered.
def CSR_ARM64_TLS_ELF
    : CalleeSavedRegs<(add (sequence "X%u", 1, 28), FP,
                           (sequence "Q%u", 0, 31))>;

def CSR_ARM64_AllRegs
    : CalleeSavedRegs<(add (sequence "W%u", 0, 30), WSP,
                           (sequence "X%u", 0, 28), FP, LR, SP,
                           (sequence "B%u", 0, 31), (sequence "H%u", 0, 31),
                           (sequence "S%u", 0, 31), (sequence "D%u", 0, 31),
                           (sequence "Q%u", 0, 31))>;

