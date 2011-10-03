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
def shamt_64       : Operand<i64>;

// Unsigned Operand
def uimm16_64      : Operand<i64> {
  let PrintMethod = "printUnsignedImm";
}

// Transformation Function - get Imm - 32.
def Subtract32 : SDNodeXForm<imm, [{
  return getI32Imm((unsigned)N->getZExtValue() - 32);
}]>;

// imm32_63 predicate - True if imm is in range [32, 63].
def imm32_63 : ImmLeaf<i64,
                       [{return (int32_t)Imm >= 32 && (int32_t)Imm < 64;}],
                       Subtract32>;

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

let isCommutable = 1 in
class LogicNOR64<bits<6> op, bits<6> func, string instr_asm>:
  FR<op, func, (outs CPU64Regs:$dst), (ins CPU64Regs:$b, CPU64Regs:$c),
     !strconcat(instr_asm, "\t$dst, $b, $c"),
     [(set CPU64Regs:$dst, (not (or CPU64Regs:$b, CPU64Regs:$c)))], IIAlu>;

// Shifts
class LogicR_shift_rotate_imm64<bits<6> func, bits<5> _rs, string instr_asm,
                                SDNode OpNode, PatFrag PF>:
  FR<0x00, func, (outs CPU64Regs:$dst), (ins CPU64Regs:$b, shamt_64:$c),
     !strconcat(instr_asm, "\t$dst, $b, $c"),
     [(set CPU64Regs:$dst, (OpNode CPU64Regs:$b, (i64 PF:$c)))],
     IIAlu> {
  let rs = _rs;
}

class LogicR_shift_rotate_reg64<bits<6> func, bits<5> _shamt, string instr_asm,
                                SDNode OpNode>:
  FR<0x00, func, (outs CPU64Regs:$dst), (ins CPU64Regs:$c, CPU64Regs:$b),
     !strconcat(instr_asm, "\t$dst, $b, $c"),
     [(set CPU64Regs:$dst, (OpNode CPU64Regs:$b, CPU64Regs:$c))], IIAlu> {
  let shamt = _shamt;
}

// Mul, Div
let Defs = [HI64, LO64] in {
  let isCommutable = 1 in
  class Mul64<bits<6> func, string instr_asm, InstrItinClass itin>:
    FR<0x00, func, (outs), (ins CPU64Regs:$a, CPU64Regs:$b),
       !strconcat(instr_asm, "\t$a, $b"), [], itin>;

  class Div64<SDNode op, bits<6> func, string instr_asm, InstrItinClass itin>:
              FR<0x00, func, (outs), (ins CPU64Regs:$a, CPU64Regs:$b),
              !strconcat(instr_asm, "\t$$zero, $a, $b"),
              [(op CPU64Regs:$a, CPU64Regs:$b)], itin>;
}

// Move from Hi/Lo
let shamt = 0 in {
let rs = 0, rt = 0 in
class MoveFromLOHI64<bits<6> func, string instr_asm>:
  FR<0x00, func, (outs CPU64Regs:$dst), (ins),
     !strconcat(instr_asm, "\t$dst"), [], IIHiLo>;

let rt = 0, rd = 0 in
class MoveToLOHI64<bits<6> func, string instr_asm>:
  FR<0x00, func, (outs), (ins CPU64Regs:$src),
     !strconcat(instr_asm, "\t$src"), [], IIHiLo>;
}

// Count Leading Ones/Zeros in Word
class CountLeading64<bits<6> func, string instr_asm, list<dag> pattern>:
  FR<0x1c, func, (outs CPU64Regs:$dst), (ins CPU64Regs:$src),
     !strconcat(instr_asm, "\t$dst, $src"), pattern, IIAlu>,
     Requires<[HasBitCount]> {
  let shamt = 0;
  let rt = rd;
}

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
def DSUBu    : ArithR64<0x00, 0x2f, "dsubu", sub, IIAlu>;
def DAND     : LogicR64<0x24, "and", and>;
def DOR      : LogicR64<0x25, "or", or>;
def DXOR     : LogicR64<0x26, "xor", xor>;
def DNOR     : LogicNOR64<0x00, 0x27, "nor">;

/// Shift Instructions
def DSLL     : LogicR_shift_rotate_imm64<0x38, 0x00, "dsll", shl, immZExt5>;
def DSRL     : LogicR_shift_rotate_imm64<0x3a, 0x00, "dsrl", srl, immZExt5>;
def DSRA     : LogicR_shift_rotate_imm64<0x3b, 0x00, "dsra", sra, immZExt5>;
def DSLL32   : LogicR_shift_rotate_imm64<0x3c, 0x00, "dsll32", shl, imm32_63>;
def DSRL32   : LogicR_shift_rotate_imm64<0x3e, 0x00, "dsrl32", srl, imm32_63>;
def DSRA32   : LogicR_shift_rotate_imm64<0x3f, 0x00, "dsra32", sra, imm32_63>;
def DSLLV    : LogicR_shift_rotate_reg64<0x24, 0x00, "dsllv", shl>;
def DSRLV    : LogicR_shift_rotate_reg64<0x26, 0x00, "dsrlv", srl>;
def DSRAV    : LogicR_shift_rotate_reg64<0x27, 0x00, "dsrav", sra>;

// Rotate Instructions
let Predicates = [HasMips64r2] in {
  def DROTR    : LogicR_shift_rotate_imm64<0x3a, 0x01, "drotr", rotr, immZExt5>;
  def DROTR32  : LogicR_shift_rotate_imm64<0x3e, 0x01, "drotr32", rotr,
                                           imm32_63>;
  def DROTRV   : LogicR_shift_rotate_reg64<0x16, 0x01, "drotrv", rotr>;
}

/// Multiply and Divide Instructions.
def DMULT    : Mul64<0x1c, "dmult", IIImul>;
def DMULTu   : Mul64<0x1d, "dmultu", IIImul>;
def DSDIV    : Div64<MipsDivRem, 0x1e, "ddiv", IIIdiv>;
def DUDIV    : Div64<MipsDivRemU, 0x1f, "ddivu", IIIdiv>;

let Defs = [HI64] in
  def MTHI64  : MoveToLOHI64<0x11, "mthi">;
let Defs = [LO64] in
  def MTLO64  : MoveToLOHI64<0x13, "mtlo">;

let Uses = [HI64] in
  def MFHI64  : MoveFromLOHI64<0x10, "mfhi">;
let Uses = [LO64] in
  def MFLO64  : MoveFromLOHI64<0x12, "mflo">;

/// Count Leading
def DCLZ : CountLeading64<0x24, "dclz",
                          [(set CPU64Regs:$dst, (ctlz CPU64Regs:$src))]>;
def DCLO : CountLeading64<0x25, "dclo",
                          [(set CPU64Regs:$dst, (ctlz (not CPU64Regs:$src)))]>;

//===----------------------------------------------------------------------===//
//  Arbitrary patterns that map to one or more instructions
//===----------------------------------------------------------------------===//

// Small immediates
def : Pat<(i64 immSExt16:$in),
          (DADDiu ZERO_64, imm:$in)>;
def : Pat<(i64 immZExt16:$in),
          (DORi ZERO_64, imm:$in)>;
