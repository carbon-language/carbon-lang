//===- AMDILOperands.td - AMD IL Operands ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// Custom memory operand
//===----------------------------------------------------------------------===//

def MEMI32  : Operand<i32> {
    let PrintMethod = "printMemOperand";
    let MIOperandInfo = (ops GPRI32, GPRI32);
}

// Call target types
def calltarget   : Operand<i32>;
def brtarget   : Operand<OtherVT>;

// def v2i8imm : Operand<v2i8>;
// def v4i8imm : Operand<v4i8>;
// def v2i16imm : Operand<v2i16>;
// def v4i16imm : Operand<v4i16>;
// def v2i32imm : Operand<v2i32>;
// def v4i32imm : Operand<v4i32>;
// def v2i64imm : Operand<v2i64>;
// def v2f32imm : Operand<v2f32>;
// def v4f32imm : Operand<v4f32>;
// def v2f64imm : Operand<v2f64>;

