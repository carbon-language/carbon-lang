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

//  Logical
let isCommutable = 1 in
class LogicR64<bits<6> func, string instr_asm, SDNode OpNode>:
  FR<0x00, func, (outs CPU64Regs:$dst), (ins CPU64Regs:$b, CPU64Regs:$c),
     !strconcat(instr_asm, "\t$dst, $b, $c"),
     [(set CPU64Regs:$dst, (OpNode CPU64Regs:$b, CPU64Regs:$c))], IIAlu>;

//===----------------------------------------------------------------------===//
// Instruction definition
//===----------------------------------------------------------------------===//

/// Arithmetic Instructions (3-Operand, R-Type)
def DADDu    : ArithR64<0x00, 0x2d, "daddu", add, IIAlu, 1>;
def DSUBu    : ArithR64<0x00, 0x2f, "dsubu", sub, IIAlu, 1>;
def DAND     : LogicR64<0x24, "and", and>;
def DOR      : LogicR64<0x25, "or", or>;
def DXOR     : LogicR64<0x26, "xor", xor>;
