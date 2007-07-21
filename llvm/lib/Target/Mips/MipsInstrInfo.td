//===- MipsInstrInfo.td - Mips Register defs --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bruno Cardoso Lopes and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction format superclass
//===----------------------------------------------------------------------===//

include "MipsInstrFormats.td"

//===----------------------------------------------------------------------===//
// Mips profiles and nodes
//===----------------------------------------------------------------------===//

// Call
def SDT_MipsJmpLink : SDTypeProfile<0, 1, [SDTCisVT<0, iPTR>]>;
def MipsJmpLink     : SDNode<"MipsISD::JmpLink",SDT_MipsJmpLink, [SDNPHasChain,
                             SDNPOutFlag]>;

// Hi and Lo nodes are created to let easy manipulation of 16-bit when 
// handling 32-bit immediates. They are used on MipsISelLowering to 
// lower stuff like GlobalAddress, ExternalSymbol, ...
// This two nodes have nothing to do with Mips Registers Hi and Lo.
def MipsHi : SDNode<"MipsISD::Hi", SDTIntUnaryOp>;
def MipsLo : SDNode<"MipsISD::Lo", SDTIntUnaryOp>; 

// Return 
def SDT_MipsRet : SDTypeProfile<0, 1, [SDTCisInt<0>]>; 
def MipsRet     : SDNode<"MipsISD::Ret", SDT_MipsRet, [SDNPHasChain, 
                             SDNPOptInFlag]>;

// These are target-independent nodes, but have target-specific formats.
def SDT_MipsCallSeq : SDTypeProfile<0, 1, [SDTCisVT<0, i32>]>;
def callseq_start   : SDNode<"ISD::CALLSEQ_START", SDT_MipsCallSeq, 
                             [SDNPHasChain, SDNPOutFlag]>;
def callseq_end     : SDNode<"ISD::CALLSEQ_END", SDT_MipsCallSeq, 
                             [SDNPHasChain, SDNPOutFlag]>;

// Instruction operand types
def brtarget    : Operand<OtherVT>;
def calltarget  : Operand<i32>;
def uimm16      : Operand<i32>;
def simm16      : Operand<i32>;
def shamt       : Operand<i32>; 

// Address operand
def mem : Operand<i32> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops simm16, CPURegs);
}

//===----------------------------------------------------------------------===//
// Mips Patterns and Transformations
//===----------------------------------------------------------------------===//

// Transformation Function - get the lower 16 bits.
def LO16 : SDNodeXForm<imm, [{
  return getI32Imm((unsigned)N->getValue() & 0xFFFF);
}]>;

// Transformation Function - get the higher 16 bits.
def HI16 : SDNodeXForm<imm, [{
  return getI32Imm((unsigned)N->getValue() >> 16);
}]>;

// Node immediate fits as 16-bit sign extended on target immediate.
// e.g. addi, andi
def immSExt16  : PatLeaf<(imm), [{
  if (N->getValueType(0) == MVT::i32)
    return (int32_t)N->getValue() == (short)N->getValue();
  else    
    return (int64_t)N->getValue() == (short)N->getValue();
}]>;

// Node immediate fits as 16-bit zero extended on target immediate.
// The LO16 param means that only the lower 16 bits of the node
// immediate are caught.
// e.g. addiu, sltiu
def immZExt16  : PatLeaf<(imm), [{
  if (N->getValueType(0) == MVT::i32)
    return (uint32_t)N->getValue() == (unsigned short)N->getValue();
  else    
    return (uint64_t)N->getValue() == (unsigned short)N->getValue();
}], LO16>;

// shamt field must fit in 5 bits.
def immZExt5 : PatLeaf<(imm), [{
  return N->getValue() == ((N->getValue()) & 0x1f) ;
}]>;

// Mips Address Mode! SDNode frameindex could possibily be a match 
// since load and store instructions from stack used it.
def addr : ComplexPattern<i32, 2, "SelectAddr", [frameindex], []>;

//===----------------------------------------------------------------------===//
// Instructions specific format
//===----------------------------------------------------------------------===//

// Arithmetic 3 register operands
let isCommutable = 1 in 
class ArithR< bits<6> op, bits<6> func, string instr_asm, SDNode OpNode>: 
  FR< op, 
      func, 
      (outs CPURegs:$dst), 
      (ins CPURegs:$b, CPURegs:$c), 
      !strconcat(instr_asm, " $dst, $b, $c"), 
      [(set CPURegs:$dst, (OpNode CPURegs:$b, CPURegs:$c))] >;

let isCommutable = 1 in 
class ArithOverflowR< bits<6> op, bits<6> func, string instr_asm>: 
  FR< op, 
      func, 
      (outs CPURegs:$dst), 
      (ins CPURegs:$b, CPURegs:$c), 
      !strconcat(instr_asm, " $dst, $b, $c"), 
      []>;

// Arithmetic 2 register operands
let isCommutable = 1 in
class ArithI<bits<6> op, string instr_asm, SDNode OpNode, 
             Operand Od, PatLeaf imm_type> : 
  FI< op, 
      (outs CPURegs:$dst), 
      (ins CPURegs:$b, Od:$c), 
      !strconcat(instr_asm, " $dst, $b, $c"), 
      [(set CPURegs:$dst, (OpNode CPURegs:$b, imm_type:$c))] >;

// Arithmetic Multiply ADD/SUB
let rd=0 in
class MArithR<bits<6> func, string instr_asm> : 
  FR< 0x1c, 
      func,
      (outs CPURegs:$rs), 
      (ins CPURegs:$rt), 
      !strconcat(instr_asm, " $rs, $rt"), 
      []>;

//  Logical
class LogicR<bits<6> func, string instr_asm, SDNode OpNode>:
  FR< 0x00, 
      func, 
      (outs CPURegs:$dst), 
      (ins CPURegs:$b, CPURegs:$c), 
      !strconcat(instr_asm, " $dst, $b, $c"), 
      [(set CPURegs:$dst, (OpNode CPURegs:$b, CPURegs:$c))] >;

class LogicI<bits<6> op, string instr_asm, SDNode OpNode>:
  FI< op,
      (outs CPURegs:$dst),
      (ins CPURegs:$b, uimm16:$c),
      !strconcat(instr_asm, " $dst, $b, $c"),
      [(set CPURegs:$dst, (OpNode CPURegs:$b, immSExt16:$c))]>;

class LogicNOR<bits<6> op, bits<6> func, string instr_asm>:
  FR< op, 
      func, 
      (outs CPURegs:$dst), 
      (ins CPURegs:$b, CPURegs:$c), 
      !strconcat(instr_asm, " $dst, $b, $c"), 
      [(set CPURegs:$dst, (not (or CPURegs:$b, CPURegs:$c)))] >;

// Shifts
let rt = 0 in
class LogicR_shift_imm<bits<6> func, string instr_asm, SDNode OpNode>:
  FR< 0x00, 
      func, 
      (outs CPURegs:$dst),
      (ins CPURegs:$b, shamt:$c),
      !strconcat(instr_asm, " $dst, $b, $c"), 
      [(set CPURegs:$dst, (OpNode CPURegs:$b, immZExt5:$c))] >;

class LogicR_shift_reg<bits<6> func, string instr_asm, SDNode OpNode>:
  FR< 0x00, 
      func, 
      (outs CPURegs:$dst),
      (ins CPURegs:$b, CPURegs:$c),
      !strconcat(instr_asm, " $dst, $b, $c"), 
      [(set CPURegs:$dst, (OpNode CPURegs:$b, CPURegs:$c))] >;

// Load Upper Imediate
class LoadUpper<bits<6> op, string instr_asm>:
  FI< op,
      (outs CPURegs:$dst),
      (ins uimm16:$imm),
      !strconcat(instr_asm, " $dst, $imm"),
      []>;

// Memory Load/Store 
let isLoad = 1 in
class LoadM<bits<6> op, string instr_asm, PatFrag OpNode>:
  FI< op,
      (outs CPURegs:$dst),
      (ins mem:$addr),
      !strconcat(instr_asm, " $dst, $addr"),
      [(set CPURegs:$dst, (OpNode addr:$addr))]>;

let isStore = 1 in
class StoreM<bits<6> op, string instr_asm, PatFrag OpNode>:
  FI< op,
      (outs),
      (ins CPURegs:$dst, mem:$addr),
      !strconcat(instr_asm, " $dst, $addr"),
      [(OpNode CPURegs:$dst, addr:$addr)]>;

// Conditional Branch
let isBranch = 1, isTerminator=1 in
class CBranch<bits<6> op, string instr_asm, PatFrag cond_op>:
  FI< op,
      (outs),
      (ins CPURegs:$a, CPURegs:$b, brtarget:$offset),
      !strconcat(instr_asm, " $a, $b, $offset"),
      [(brcond (cond_op CPURegs:$a, CPURegs:$b), bb:$offset)]>;

class SetCC_R<bits<6> op, bits<6> func, string instr_asm,
      PatFrag cond_op>:
  FR< op,
      func,
      (outs CPURegs:$dst),
      (ins CPURegs:$b, CPURegs:$c),
      !strconcat(instr_asm, " $dst, $b, $c"),
      [(set CPURegs:$dst, (cond_op CPURegs:$b, CPURegs:$c))]>;

class SetCC_I<bits<6> op, string instr_asm, PatFrag cond_op,
      Operand Od, PatLeaf imm_type>:
  FI< op,
      (outs CPURegs:$dst),
      (ins CPURegs:$b, Od:$c),
      !strconcat(instr_asm, " $dst, $b, $c"),
      [(set CPURegs:$dst, (cond_op CPURegs:$b, imm_type:$c))]>;

// Unconditional branch
let hasCtrlDep=1, isTerminator=1 in
class JumpFJ<bits<6> op, string instr_asm>:
  FJ< op,
      (outs),
      (ins brtarget:$target),
      !strconcat(instr_asm, " $target"),
      [(br bb:$target)]>;

let hasCtrlDep=1, isTerminator=1, rd=0 in
class JumpFR<bits<6> op, bits<6> func, string instr_asm>:
  FR< op,
      func,
      (outs),
      (ins CPURegs:$target),
      !strconcat(instr_asm, " $target"),
      []>;

// Jump and Link (Call)
let isCall=1 in
class JumpLink<bits<6> op, string instr_asm>: 
  FJ< op,
      (outs),
      (ins calltarget:$target),
      !strconcat(instr_asm, " $target"),
      [(MipsJmpLink imm:$target)]>;

let isCall=1 in
class JumpLinkReg<bits<6> op, bits<6> func, string instr_asm>:
  FR< op,
      func,
      (outs),
      (ins CPURegs:$rd, CPURegs:$rs),
      !strconcat(instr_asm, " $rs, $rd"),
      []>;

// Mul, Div 
class MulDiv<bits<6> func, string instr_asm>: 
  FR< 0x00, 
      func, 
      (outs),
      (ins CPURegs:$a, CPURegs:$b), 
      !strconcat(instr_asm, " $a, $b"), 
      []>;

// Move from Hi/Lo 
class MoveFromTo<bits<6> func, string instr_asm>:
  FR< 0x00, 
      func, 
      (outs CPURegs:$dst), 
      (ins),
      !strconcat(instr_asm, " $dst"), 
      []>;

// Count Leading Ones/Zeros in Word
class CountLeading<bits<6> func, string instr_asm>:
  FR< 0x1c, 
      func, 
      (outs CPURegs:$dst), 
      (ins CPURegs:$src), 
      !strconcat(instr_asm, " $dst, $src"), 
      []>;


//===----------------------------------------------------------------------===//
// Pseudo instructions
//===----------------------------------------------------------------------===//

class Pseudo<dag outs, dag ins, string asmstr, list<dag> pattern>: 
      MipsInst<outs, ins, asmstr, pattern>;

// As stack alignment is always done with addiu, we need a 16-bit immediate
def ADJCALLSTACKDOWN : Pseudo<(outs), (ins uimm16:$amt),
                              "!ADJCALLSTACKDOWN $amt",
                              [(callseq_start imm:$amt)]>, Imp<[SP],[SP]>;
def ADJCALLSTACKUP   : Pseudo<(outs), (ins uimm16:$amt),
                              "!ADJCALLSTACKUP $amt",
                              [(callseq_end imm:$amt)]>, Imp<[SP],[SP]>;

def IMPLICIT_DEF_CPURegs : Pseudo<(outs CPURegs:$dst), (ins),
                                  "!IMPLICIT_DEF $dst",
                                  [(set CPURegs:$dst, (undef))]>;

//===----------------------------------------------------------------------===//
// Instruction definition
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Mips32 I
//===----------------------------------------------------------------------===//

// Arithmetic
def ADDiu   : ArithI<0x09, "addiu", add, uimm16, immSExt16>;
def ADDi    : ArithI<0x08, "addi",  add, simm16, immZExt16>;
def MUL     : ArithR<0x1c, 0x02, "mul", mul>;
def ADDu    : ArithR<0x00, 0x21, "addu", add>;
def SUBu    : ArithR<0x00, 0x23, "subu", sub>;
def ADD     : ArithOverflowR<0x00, 0x20, "add">;
def SUB     : ArithOverflowR<0x00, 0x22, "sub">;
def MADD    : MArithR<0x00, "madd">;
def MADDU   : MArithR<0x01, "maddu">;
def MSUB    : MArithR<0x04, "msub">;
def MSUBU   : MArithR<0x05, "msubu">;

// Logical
def AND     : LogicR<0x24, "and", and>;
def OR      : LogicR<0x25, "or",  or>;
def XOR     : LogicR<0x26, "xor", xor>;
def ANDi    : LogicI<0x0c, "andi", and>;
def ORi     : LogicI<0x0d, "ori",  or>;
def XORi    : LogicI<0x0e, "xori",  xor>;
def NOR     : LogicNOR<0x00, 0x27, "nor">;

// Shifts 
def SLL     : LogicR_shift_imm<0x00, "sll", shl>;
def SRL     : LogicR_shift_imm<0x02, "srl", srl>;
def SRA     : LogicR_shift_imm<0x03, "sra", sra>;
def SLLV    : LogicR_shift_reg<0x04, "sllv", shl>;
def SRLV    : LogicR_shift_reg<0x06, "srlv", srl>;
def SRAV    : LogicR_shift_reg<0x07, "srav", sra>;

// Load Upper Immediate
def LUi     : LoadUpper<0x0f, "lui">;

// Load/Store
def LB      : LoadM<0x20, "lb",  sextloadi8>;
def LBu     : LoadM<0x24, "lbu", zextloadi8>;
def LH      : LoadM<0x21, "lh",  sextloadi16>;
def LHu     : LoadM<0x25, "lhu", zextloadi16>;
def LW      : LoadM<0x23, "lw",  load>;
def SB      : StoreM<0x28, "sb", truncstorei8>;
def SH      : StoreM<0x29, "sh", truncstorei16>;
def SW      : StoreM<0x2b, "sw", store>;

// Conditional Branch
def BEQ     : CBranch<0x04, "beq", seteq>;
def BNE     : CBranch<0x05, "bne", setne>;
def SLT     : SetCC_R<0x00, 0x2a, "slt", setlt>;
def SLTu    : SetCC_R<0x00, 0x2b, "sltu", setult>;
def SLTi    : SetCC_I<0x0a, "slti", setlt, simm16, immSExt16>;
def SLTiu   : SetCC_I<0x0b, "sltiu", setult, uimm16, immZExt16>;

// Unconditional jump
def J       : JumpFJ<0x02, "j">;
def JR      : JumpFR<0x00, 0x08, "jr">;

// Jump and Link (Call)
def JAL     : JumpLink<0x03, "jal">;
def JALR    : JumpLinkReg<0x00, 0x09, "jalr">;

// MulDiv and Move From Hi/Lo operations, have
// their correpondent SDNodes created on ISelDAG.
// Special Mul, Div operations
def MULT    : MulDiv<0x18, "mult">;
def MULTu   : MulDiv<0x19, "multu">;
def DIV     : MulDiv<0x1a, "div">;
def DIVu    : MulDiv<0x1b, "divu">;

// Move From Hi/Lo 
def MFHI    : MoveFromTo<0x10, "mfhi">;
def MFLO    : MoveFromTo<0x12, "mflo">;
def MTHI    : MoveFromTo<0x11, "mthi">;
def MTLO    : MoveFromTo<0x13, "mtlo">;

// Count Leading
def CLO     : CountLeading<0x21, "clo">;
def CLZ     : CountLeading<0x20, "clz">;

// No operation
let addr=0 in
def NOOP :  FJ<0, (outs), (ins), "nop", []>;

// Ret instruction - as mips does not have "ret" a 
// jr $ra must be generated.
let isReturn=1, isTerminator=1, hasDelaySlot=1,
    isBarrier=1, hasCtrlDep=1, rs=0, rt=0, shamt=0 in 
{
  def RET : FR <0x00, 0x02, (outs), (ins CPURegs:$target),
                "jr $target", [(MipsRet CPURegs:$target)]>;
}

//===----------------------------------------------------------------------===//
//  Arbitrary patterns that map to one or more instructions
//===----------------------------------------------------------------------===//

// Small immediates
def : Pat<(i32 immSExt16:$in), 
          (ADDiu ZERO, imm:$in)>;
def : Pat<(i32 immZExt16:$in), 
          (ORi ZERO, imm:$in)>;

// Arbitrary immediates
def : Pat<(i32 imm:$imm),
          (ORi (LUi (HI16 imm:$imm)), (LO16 imm:$imm))>;

// Call
def : Pat<(MipsJmpLink (i32 tglobaladdr:$dst)),
          (JAL tglobaladdr:$dst)>;
def : Pat<(MipsJmpLink (i32 texternalsym:$dst)),
          (JAL texternalsym:$dst)>;

// GlobalAddress, Constant Pool, ExternalSymbol, and JumpTable
def : Pat<(MipsHi tglobaladdr:$in), (LUi tglobaladdr:$in)>;
def : Pat<(MipsLo tglobaladdr:$in), (ADDiu ZERO, tglobaladdr:$in)>;

// When extracting the address from GlobalAddress we
// need something of the form "addiu $reg, %lo(addr)"
def : Pat<(add CPURegs:$a, (MipsLo tglobaladdr:$in)),
          (ADDiu CPURegs:$a, tglobaladdr:$in)>;

// Mips does not have not, so we increase the operation  
def : Pat<(not CPURegs:$in),
          (NOR CPURegs:$in, CPURegs:$in)>;

// extended load and stores 
def : Pat<(i32 (extloadi8  addr:$src)), (LBu addr:$src)>;
def : Pat<(i32 (extloadi16 addr:$src)), (LHu addr:$src)>;
def : Pat<(truncstorei1 CPURegs:$src, addr:$addr), 
           (SB CPURegs:$src, addr:$src)>;

def : Pat<(brcond (setne CPURegs:$lhs, (add ZERO, 0)), bb:$dst),
          (BNE CPURegs:$lhs, ZERO, bb:$dst)>;


// Conditional branch patterns.
// cond branches patterns, 2 register operands signed.
def : Pat<(brcond (setlt CPURegs:$lhs, CPURegs:$rhs), bb:$dst),
          (BNE (SLT CPURegs:$lhs, CPURegs:$rhs), ZERO, bb:$dst)>;
def : Pat<(brcond (setle CPURegs:$lhs, CPURegs:$rhs), bb:$dst),
          (BEQ (SLT CPURegs:$rhs, CPURegs:$lhs), ZERO, bb:$dst)>;
def : Pat<(brcond (setgt CPURegs:$lhs, CPURegs:$rhs), bb:$dst),
          (BNE (SLT CPURegs:$rhs, CPURegs:$lhs), ZERO, bb:$dst)>;
def : Pat<(brcond (setge CPURegs:$lhs, CPURegs:$rhs), bb:$dst),
          (BEQ (SLT CPURegs:$lhs, CPURegs:$rhs), ZERO, bb:$dst)>;

// cond branches patterns, 2 register operands unsigned.
def : Pat<(brcond (setult CPURegs:$lhs, CPURegs:$rhs), bb:$dst),
          (BNE (SLTu CPURegs:$lhs, CPURegs:$rhs), ZERO, bb:$dst)>;
def : Pat<(brcond (setule CPURegs:$lhs, CPURegs:$rhs), bb:$dst),
          (BEQ (SLTu CPURegs:$rhs, CPURegs:$lhs), ZERO, bb:$dst)>;
def : Pat<(brcond (setugt CPURegs:$lhs, CPURegs:$rhs), bb:$dst),
          (BNE (SLTu CPURegs:$rhs, CPURegs:$lhs), ZERO, bb:$dst)>;
def : Pat<(brcond (setuge CPURegs:$lhs, CPURegs:$rhs), bb:$dst),
          (BEQ (SLTu CPURegs:$lhs, CPURegs:$rhs), ZERO, bb:$dst)>;

// cond branches patterns, reg/imm operands signed.
def : Pat<(brcond (setult CPURegs:$lhs, immSExt16:$rhs), bb:$dst),
          (BNE (SLTi CPURegs:$lhs, immSExt16:$rhs), ZERO, bb:$dst)>;
def : Pat<(brcond (setuge CPURegs:$lhs, immSExt16:$rhs), bb:$dst),
          (BEQ (SLTi CPURegs:$lhs, immSExt16:$rhs), ZERO, bb:$dst)>;

// cond branches patterns, reg/imm operands unsigned.
def : Pat<(brcond (setult CPURegs:$lhs, immZExt16:$rhs), bb:$dst),
          (BNE (SLTiu CPURegs:$lhs, immZExt16:$rhs), ZERO, bb:$dst)>;
def : Pat<(brcond (setuge CPURegs:$lhs, immZExt16:$rhs), bb:$dst),
          (BEQ (SLTiu CPURegs:$lhs, immZExt16:$rhs), ZERO, bb:$dst)>;
