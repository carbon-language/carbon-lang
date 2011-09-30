//===- Mips64InstrInfo.td - Mips64 Instruction Information -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes Mips64 instructions.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Mips64 Instruction Predicate Definitions.
//===----------------------------------------------------------------------===//
def HasMips64    : Predicate<"Subtarget.hasMips64()">;
def HasMips64r2  : Predicate<"Subtarget.hasMips64r2()">;

//===----------------------------------------------------------------------===//
// Mips Operand, Complex Patterns and Transformations Definitions.
//===----------------------------------------------------------------------===//

// Instruction operand types
def simm16_64      : Operand<i64>;

// Unsigned Operand
def uimm16_64      : Operand<i64> {
  let PrintMethod = "printUnsignedImm";
}

//===----------------------------------------------------------------------===//
// Instructions specific format
//===----------------------------------------------------------------------===//

// Arithmetic 3 register operands
class ArithR64<bits<6> op, bits<6> func, string instr_asm, SDNode OpNode,
               InstrItinClass itin, bit isComm = 0>:
  FR<op, func, (outs CPU64Regs:$dst), (ins CPU64Regs:$b, CPU64Regs:$c),
     !strconcat(instr_asm, "\t$dst, $b, $c"),
     [(set CPU64Regs:$dst, (OpNode CPU64Regs:$b, CPU64Regs:$c))], itin> {
  let isCommutable = isComm;
}

// Arithmetic 2 register operands
class ArithI64<bits<6> op, string instr_asm, SDNode OpNode,
               Operand Od, PatLeaf imm_type> :
  FI<op, (outs CPU64Regs:$dst), (ins CPU64Regs:$b, Od:$c),
     !strconcat(instr_asm, "\t$dst, $b, $c"),
     [(set CPU64Regs:$dst, (OpNode CPU64Regs:$b, imm_type:$c))], IIAlu>;

//  Logical
let isCommutable = 1 in
class LogicR64<bits<6> func, string instr_asm, SDNode OpNode>:
  FR<0x00, func, (outs CPU64Regs:$dst), (ins CPU64Regs:$b, CPU64Regs:$c),
     !strconcat(instr_asm, "\t$dst, $b, $c"),
     [(set CPU64Regs:$dst, (OpNode CPU64Regs:$b, CPU64Regs:$c))], IIAlu>;

class LogicI64<bits<6> op, string instr_asm, SDNode OpNode>:
  FI<op, (outs CPU64Regs:$dst), (ins CPU64Regs:$b, uimm16_64:$c),
     !strconcat(instr_asm, "\t$dst, $b, $c"),
     [(set CPU64Regs:$dst, (OpNode CPU64Regs:$b, immZExt16:$c))], IIAlu>;

//===----------------------------------------------------------------------===//
// Instruction definition
//===----------------------------------------------------------------------===//

/// Arithmetic Instructions (ALU Immediate)
def DADDiu   : ArithI64<0x19, "daddiu", add, simm16_64, immSExt16>;
def DANDi    : LogicI64<0x0c, "andi", and>;
def DORi     : LogicI64<0x0d, "ori",  or>;
def DXORi    : LogicI64<0x0e, "xori",  xor>;

/// Arithmetic Instructions (3-Operand, R-Type)
def DADDu    : ArithR64<0x00, 0x2d, "daddu", add, IIAlu, 1>;
def DSUBu    : ArithR64<0x00, 0x2f, "dsubu", sub, IIAlu, 1>;
def DAND     : LogicR64<0x24, "and", and>;
def DOR      : LogicR64<0x25, "or", or>;
def DXOR     : LogicR64<0x26, "xor", xor>;
