//===- MBlazeInstrInfo.td - MBlaze Instruction defs --------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction format superclass
//===----------------------------------------------------------------------===//
include "MBlazeInstrFormats.td"

//===----------------------------------------------------------------------===//
// MBlaze type profiles
//===----------------------------------------------------------------------===//

// def SDTMBlazeSelectCC : SDTypeProfile<1, 3, [SDTCisSameAs<0, 1>]>;
def SDT_MBlazeRet     : SDTypeProfile<0, 1, [SDTCisInt<0>]>;
def SDT_MBlazeJmpLink : SDTypeProfile<0, 1, [SDTCisVT<0, i32>]>;
def SDT_MBCallSeqStart : SDCallSeqStart<[SDTCisVT<0, i32>]>;
def SDT_MBCallSeqEnd   : SDCallSeqEnd<[SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;

//===----------------------------------------------------------------------===//
// MBlaze specific nodes
//===----------------------------------------------------------------------===//

def MBlazeRet     : SDNode<"MBlazeISD::Ret", SDT_MBlazeRet,
                           [SDNPHasChain, SDNPOptInFlag]>;

def MBlazeJmpLink : SDNode<"MBlazeISD::JmpLink",SDT_MBlazeJmpLink,
                           [SDNPHasChain,SDNPOptInFlag,SDNPOutFlag]>;

def MBWrapper   : SDNode<"MBlazeISD::Wrap", SDTIntUnaryOp>;

def callseq_start : SDNode<"ISD::CALLSEQ_START", SDT_MBCallSeqStart,
                           [SDNPHasChain, SDNPOutFlag]>;

def callseq_end   : SDNode<"ISD::CALLSEQ_END", SDT_MBCallSeqEnd,
                           [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;

//===----------------------------------------------------------------------===//
// MBlaze Instruction Predicate Definitions.
//===----------------------------------------------------------------------===//
def HasPipe3     : Predicate<"Subtarget.hasPipe3()">;
def HasBarrel    : Predicate<"Subtarget.hasBarrel()">;
def NoBarrel     : Predicate<"!Subtarget.hasBarrel()">;
def HasDiv       : Predicate<"Subtarget.hasDiv()">;
def HasMul       : Predicate<"Subtarget.hasMul()">;
def HasFSL       : Predicate<"Subtarget.hasFSL()">;
def HasEFSL      : Predicate<"Subtarget.hasEFSL()">;
def HasMSRSet    : Predicate<"Subtarget.hasMSRSet()">;
def HasException : Predicate<"Subtarget.hasException()">;
def HasPatCmp    : Predicate<"Subtarget.hasPatCmp()">;
def HasFPU       : Predicate<"Subtarget.hasFPU()">;
def HasESR       : Predicate<"Subtarget.hasESR()">;
def HasPVR       : Predicate<"Subtarget.hasPVR()">;
def HasMul64     : Predicate<"Subtarget.hasMul64()">;
def HasSqrt      : Predicate<"Subtarget.hasSqrt()">;
def HasMMU       : Predicate<"Subtarget.hasMMU()">;

//===----------------------------------------------------------------------===//
// MBlaze Operand, Complex Patterns and Transformations Definitions.
//===----------------------------------------------------------------------===//

def MBlazeMemAsmOperand : AsmOperandClass {
  let Name = "Mem";
  let SuperClasses = [];
}

def MBlazeFslAsmOperand : AsmOperandClass {
  let Name = "Fsl";
  let SuperClasses = [];
}

// Instruction operand types
def brtarget    : Operand<OtherVT>;
def calltarget  : Operand<i32>;
def simm16      : Operand<i32>;
def uimm5       : Operand<i32>;
def uimm14      : Operand<i32>;
def uimm15      : Operand<i32>;
def fimm        : Operand<f32>;

// Unsigned Operand
def uimm16      : Operand<i32> {
  let PrintMethod = "printUnsignedImm";
}

// FSL Operand
def fslimm      : Operand<i32> {
  let PrintMethod = "printFSLImm";
  let ParserMatchClass = MBlazeFslAsmOperand;
}

// Address operand
def memri : Operand<i32> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops GPR, simm16);
  let ParserMatchClass = MBlazeMemAsmOperand;
}

def memrr : Operand<i32> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops GPR, GPR);
  let ParserMatchClass = MBlazeMemAsmOperand;
}

// Node immediate fits as 16-bit sign extended on target immediate.
def immSExt16  : PatLeaf<(imm), [{
  return (N->getZExtValue() >> 16) == 0;
}]>;

// Node immediate fits as 16-bit zero extended on target immediate.
// The LO16 param means that only the lower 16 bits of the node
// immediate are caught.
// e.g. addiu, sltiu
def immZExt16  : PatLeaf<(imm), [{
  return (N->getZExtValue() >> 16) == 0;
}]>;

// FSL immediate field must fit in 4 bits.
def immZExt4 : PatLeaf<(imm), [{
  return N->getZExtValue() == ((N->getZExtValue()) & 0xf) ;
}]>;

// shamt field must fit in 5 bits.
def immZExt5 : PatLeaf<(imm), [{
  return N->getZExtValue() == ((N->getZExtValue()) & 0x1f) ;
}]>;

// MBlaze Address Mode. SDNode frameindex could possibily be a match
// since load and store instructions from stack used it.
def iaddr : ComplexPattern<i32, 2, "SelectAddrRegImm", [frameindex], []>;
def xaddr : ComplexPattern<i32, 2, "SelectAddrRegReg", [], []>;

//===----------------------------------------------------------------------===//
// Pseudo instructions
//===----------------------------------------------------------------------===//

// As stack alignment is always done with addiu, we need a 16-bit immediate
let Defs = [R1], Uses = [R1] in {
def ADJCALLSTACKDOWN : MBlazePseudo<(outs), (ins simm16:$amt),
                                  "#ADJCALLSTACKDOWN $amt",
                                  [(callseq_start timm:$amt)]>;
def ADJCALLSTACKUP   : MBlazePseudo<(outs),
                                  (ins uimm16:$amt1, simm16:$amt2),
                                  "#ADJCALLSTACKUP $amt1",
                                  [(callseq_end timm:$amt1, timm:$amt2)]>;
}

//===----------------------------------------------------------------------===//
// Instructions specific format
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Arithmetic Instructions
//===----------------------------------------------------------------------===//
class Arith<bits<6> op, bits<11> flags, string instr_asm, SDNode OpNode,
            InstrItinClass itin> :
            TA<op, flags, (outs GPR:$dst), (ins GPR:$b, GPR:$c),
               !strconcat(instr_asm, "   $dst, $b, $c"),
               [(set GPR:$dst, (OpNode GPR:$b, GPR:$c))], itin>;

class ArithI<bits<6> op, string instr_asm, SDNode OpNode,
             Operand Od, PatLeaf imm_type> :
             TB<op, (outs GPR:$dst), (ins GPR:$b, Od:$c),
                !strconcat(instr_asm, "   $dst, $b, $c"),
                [(set GPR:$dst, (OpNode GPR:$b, imm_type:$c))], IIAlu>;

class ArithI32<bits<6> op, string instr_asm,Operand Od, PatLeaf imm_type> :
               TB<op, (outs GPR:$dst), (ins GPR:$b, Od:$c),
                  !strconcat(instr_asm, "   $dst, $b, $c"),
                  [], IIAlu>;

class ShiftI<bits<6> op, bits<2> flags, string instr_asm, SDNode OpNode,
             Operand Od, PatLeaf imm_type> :
             SHT<op, flags, (outs GPR:$dst), (ins GPR:$b, Od:$c),
                 !strconcat(instr_asm, "   $dst, $b, $c"),
                 [(set GPR:$dst, (OpNode GPR:$b, imm_type:$c))], IIAlu>;

class ArithR<bits<6> op, bits<11> flags, string instr_asm, SDNode OpNode,
            InstrItinClass itin> :
            TAR<op, flags, (outs GPR:$dst), (ins GPR:$b, GPR:$c),
                !strconcat(instr_asm, "   $dst, $c, $b"),
                [(set GPR:$dst, (OpNode GPR:$b, GPR:$c))], itin>;

class ArithRI<bits<6> op, string instr_asm, SDNode OpNode,
             Operand Od, PatLeaf imm_type> :
             TBR<op, (outs GPR:$dst), (ins Od:$b, GPR:$c),
                 !strconcat(instr_asm, "   $dst, $c, $b"),
                 [(set GPR:$dst, (OpNode imm_type:$b, GPR:$c))], IIAlu>;

class ArithN<bits<6> op, bits<11> flags, string instr_asm,
            InstrItinClass itin> :
            TA<op, flags, (outs GPR:$dst), (ins GPR:$b, GPR:$c),
               !strconcat(instr_asm, "   $dst, $b, $c"),
               [], itin>;

class ArithNI<bits<6> op, string instr_asm,Operand Od, PatLeaf imm_type> :
             TB<op, (outs GPR:$dst), (ins GPR:$b, Od:$c),
                !strconcat(instr_asm, "   $dst, $b, $c"),
                [], IIAlu>;

class ArithRN<bits<6> op, bits<11> flags, string instr_asm,
            InstrItinClass itin> :
            TAR<op, flags, (outs GPR:$dst), (ins GPR:$c, GPR:$b),
                !strconcat(instr_asm, "   $dst, $b, $c"),
                [], itin>;

class ArithRNI<bits<6> op, string instr_asm,Operand Od, PatLeaf imm_type> :
             TBR<op, (outs GPR:$dst), (ins Od:$c, GPR:$b),
                 !strconcat(instr_asm, "   $dst, $b, $c"),
                 [], IIAlu>;

//===----------------------------------------------------------------------===//
// Misc Arithmetic Instructions
//===----------------------------------------------------------------------===//

class Logic<bits<6> op, bits<11> flags, string instr_asm, SDNode OpNode> :
            TA<op, flags, (outs GPR:$dst), (ins GPR:$b, GPR:$c),
               !strconcat(instr_asm, "   $dst, $b, $c"),
               [(set GPR:$dst, (OpNode GPR:$b, GPR:$c))], IIAlu>;

class LogicI<bits<6> op, string instr_asm, SDNode OpNode> :
             TB<op, (outs GPR:$dst), (ins GPR:$b, uimm16:$c),
                !strconcat(instr_asm, "   $dst, $b, $c"),
                [(set GPR:$dst, (OpNode GPR:$b, immZExt16:$c))],
                IIAlu>;

class LogicI32<bits<6> op, string instr_asm> :
               TB<op, (outs GPR:$dst), (ins GPR:$b, uimm16:$c),
                  !strconcat(instr_asm, "   $dst, $b, $c"),
                  [], IIAlu>;

class PatCmp<bits<6> op, bits<11> flags, string instr_asm> :
             TA<op, flags, (outs GPR:$dst), (ins GPR:$b, GPR:$c),
                !strconcat(instr_asm, "   $dst, $b, $c"),
                 [], IIAlu>;

//===----------------------------------------------------------------------===//
// Memory Access Instructions
//===----------------------------------------------------------------------===//
class LoadM<bits<6> op, bits<11> flags, string instr_asm> :
            TA<op, flags, (outs GPR:$dst), (ins memrr:$addr),
               !strconcat(instr_asm, "   $dst, $addr"),
               [], IILoad>;

class LoadMI<bits<6> op, string instr_asm, PatFrag OpNode> :
             TB<op, (outs GPR:$dst), (ins memri:$addr),
                !strconcat(instr_asm, "   $dst, $addr"),
                [(set (i32 GPR:$dst), (OpNode iaddr:$addr))], IILoad>;

class StoreM<bits<6> op, bits<11> flags, string instr_asm> :
             TA<op, flags, (outs), (ins GPR:$dst, memrr:$addr),
                !strconcat(instr_asm, "   $dst, $addr"),
                [], IIStore>;

class StoreMI<bits<6> op, string instr_asm, PatFrag OpNode> :
              TB<op, (outs), (ins GPR:$dst, memri:$addr),
                 !strconcat(instr_asm, "   $dst, $addr"),
                 [(OpNode (i32 GPR:$dst), iaddr:$addr)], IIStore>;

//===----------------------------------------------------------------------===//
// Branch Instructions
//===----------------------------------------------------------------------===//
class Branch<bits<6> op, bits<5> br, bits<11> flags, string instr_asm> :
             TA<op, flags, (outs), (ins GPR:$target),
                !strconcat(instr_asm, "   $target"),
                [], IIBranch> {
  let rd = 0x0;
  let ra = br;
  let Form = FCCR;
}

class BranchI<bits<6> op, bits<5> br, string instr_asm> :
              TB<op, (outs), (ins brtarget:$target),
                 !strconcat(instr_asm, "   $target"),
                 [], IIBranch> {
  let rd = 0;
  let ra = br;
  let Form = FCCI;
}

//===----------------------------------------------------------------------===//
// Branch and Link Instructions
//===----------------------------------------------------------------------===//
class BranchL<bits<6> op, bits<5> br, bits<11> flags, string instr_asm> :
              TA<op, flags, (outs), (ins GPR:$link, GPR:$target),
                 !strconcat(instr_asm, "   $link, $target"),
                 [], IIBranch> {
  let ra = br;
  let Form = FRCR;
}

class BranchLI<bits<6> op, bits<5> br, string instr_asm> :
               TB<op, (outs), (ins GPR:$link, calltarget:$target),
                  !strconcat(instr_asm, "   $link, $target"),
                  [], IIBranch> {
  let ra = br;
  let Form = FRCI;
}

//===----------------------------------------------------------------------===//
// Conditional Branch Instructions
//===----------------------------------------------------------------------===//
class BranchC<bits<6> op, bits<5> br, bits<11> flags, string instr_asm> :
              TA<op, flags, (outs),
                 (ins GPR:$a, GPR:$b),
                 !strconcat(instr_asm, "   $a, $b"),
                 [], IIBranch> {
  let rd = br;
  let Form = FCRR;
}

class BranchCI<bits<6> op, bits<5> br, string instr_asm> :
               TB<op, (outs), (ins GPR:$a, brtarget:$offset),
                  !strconcat(instr_asm, "   $a, $offset"),
                  [], IIBranch> {
  let rd = br;
  let Form = FCRI;
}

//===----------------------------------------------------------------------===//
// MBlaze arithmetic instructions
//===----------------------------------------------------------------------===//

let isCommutable = 1, isAsCheapAsAMove = 1 in {
  def ADD    :  Arith<0x00, 0x000, "add    ", add,  IIAlu>;
  def ADDC   :  Arith<0x02, 0x000, "addc   ", adde, IIAlu>;
  def ADDK   :  Arith<0x04, 0x000, "addk   ", addc, IIAlu>;
  def ADDKC  : ArithN<0x06, 0x000, "addkc  ", IIAlu>;
  def AND    :  Logic<0x21, 0x000, "and    ", and>;
  def OR     :  Logic<0x20, 0x000, "or     ", or>;
  def XOR    :  Logic<0x22, 0x000, "xor    ", xor>;
  def PCMPBF : PatCmp<0x20, 0x400, "pcmpbf ">;
  def PCMPEQ : PatCmp<0x22, 0x400, "pcmpeq ">;
  def PCMPNE : PatCmp<0x23, 0x400, "pcmpne ">;
}

let isAsCheapAsAMove = 1 in {
  def ANDN   :  ArithN<0x23, 0x000, "andn   ", IIAlu>;
  def CMP    :  ArithN<0x05, 0x001, "cmp    ", IIAlu>;
  def CMPU   :  ArithN<0x05, 0x003, "cmpu   ", IIAlu>;
  def RSUB   :  ArithR<0x01, 0x000, "rsub   ", sub,  IIAlu>;
  def RSUBC  :  ArithR<0x03, 0x000, "rsubc  ", sube, IIAlu>;
  def RSUBK  :  ArithR<0x05, 0x000, "rsubk  ", subc, IIAlu>;
  def RSUBKC : ArithRN<0x07, 0x000, "rsubkc ", IIAlu>;
}

let isCommutable = 1, Predicates=[HasMul] in {
  def MUL    : Arith<0x10, 0x000, "mul    ", mul,   IIAlu>;
}

let isCommutable = 1, Predicates=[HasMul,HasMul64] in {
  def MULH   : Arith<0x10, 0x001, "mulh   ", mulhs, IIAlu>;
  def MULHU  : Arith<0x10, 0x003, "mulhu  ", mulhu, IIAlu>;
}

let Predicates=[HasMul,HasMul64] in {
  def MULHSU : ArithN<0x10, 0x002, "mulhsu ", IIAlu>;
}

let Predicates=[HasBarrel] in {
  def BSRL   :   Arith<0x11, 0x000, "bsrl   ", srl, IIAlu>;
  def BSRA   :   Arith<0x11, 0x200, "bsra   ", sra, IIAlu>;
  def BSLL   :   Arith<0x11, 0x400, "bsll   ", shl, IIAlu>;
  def BSRLI  :  ShiftI<0x19, 0x0, "bsrli  ", srl, uimm5, immZExt5>;
  def BSRAI  :  ShiftI<0x19, 0x1, "bsrai  ", sra, uimm5, immZExt5>;
  def BSLLI  :  ShiftI<0x19, 0x2, "bslli  ", shl, uimm5, immZExt5>;
}

let Predicates=[HasDiv] in {
  def IDIV   :  ArithR<0x12, 0x000, "idiv   ", sdiv, IIAlu>;
  def IDIVU  :  ArithR<0x12, 0x002, "idivu  ", udiv, IIAlu>;
}

//===----------------------------------------------------------------------===//
// MBlaze immediate mode arithmetic instructions
//===----------------------------------------------------------------------===//

let isAsCheapAsAMove = 1 in {
  def ADDI    :   ArithI<0x08, "addi   ", add,  simm16, immSExt16>;
  def ADDIC   :  ArithNI<0x0A, "addic  ", simm16, immSExt16>;
  def ADDIK   :  ArithNI<0x0C, "addik  ", simm16, immSExt16>;
  def ADDIKC  :   ArithI<0x0E, "addikc ", addc, simm16, immSExt16>;
  def RSUBI   :  ArithRI<0x09, "rsubi  ", sub,  simm16, immSExt16>;
  def RSUBIC  : ArithRNI<0x0B, "rsubic ", simm16, immSExt16>;
  def RSUBIK  : ArithRNI<0x0D, "rsubik ", simm16, immSExt16>;
  def RSUBIKC :  ArithRI<0x0F, "rsubikc", subc, simm16, immSExt16>;
  def ANDNI   :  ArithNI<0x2B, "andni  ", uimm16, immZExt16>;
  def ANDI    :   LogicI<0x29, "andi   ", and>;
  def ORI     :   LogicI<0x28, "ori    ", or>;
  def XORI    :   LogicI<0x2A, "xori   ", xor>;
}

let Predicates=[HasMul] in {
  def MULI    :   ArithI<0x18, "muli   ", mul, simm16, immSExt16>;
}

//===----------------------------------------------------------------------===//
// MBlaze memory access instructions
//===----------------------------------------------------------------------===//

let canFoldAsLoad = 1, isReMaterializable = 1 in {
  def LBU  :  LoadM<0x30, 0x000, "lbu    ">;
  def LBUR :  LoadM<0x30, 0x200, "lbur   ">;

  def LHU  :  LoadM<0x31, 0x000, "lhu    ">;
  def LHUR :  LoadM<0x31, 0x200, "lhur   ">;

  def LW   :  LoadM<0x32, 0x000, "lw     ">;
  def LWR  :  LoadM<0x32, 0x200, "lwr    ">;
  def LWX  :  LoadM<0x32, 0x400, "lwx    ">;

  def LBUI : LoadMI<0x38, "lbui   ", zextloadi8>;
  def LHUI : LoadMI<0x39, "lhui   ", zextloadi16>;
  def LWI  : LoadMI<0x3A, "lwi    ", load>;
}

  def SB  :  StoreM<0x34, 0x000, "sb     ">;
  def SBR :  StoreM<0x34, 0x200, "sbr    ">;

  def SH  :  StoreM<0x35, 0x000, "sh     ">;
  def SHR :  StoreM<0x35, 0x200, "shr    ">;

  def SW  :  StoreM<0x36, 0x000, "sw     ">;
  def SWR :  StoreM<0x36, 0x200, "swr    ">;
  def SWX :  StoreM<0x36, 0x400, "swx    ">;

  def SBI : StoreMI<0x3C, "sbi    ", truncstorei8>;
  def SHI : StoreMI<0x3D, "shi    ", truncstorei16>;
  def SWI : StoreMI<0x3E, "swi    ", store>;

//===----------------------------------------------------------------------===//
// MBlaze branch instructions
//===----------------------------------------------------------------------===//

let isBranch = 1, isTerminator = 1, hasCtrlDep = 1, isBarrier = 1 in {
  def BRI    :  BranchI<0x2E, 0x00, "bri    ">;
  def BRAI   :  BranchI<0x2E, 0x08, "brai   ">;
}

let isBranch = 1, isTerminator = 1, hasCtrlDep = 1 in {
  def BEQI   : BranchCI<0x2F, 0x00, "beqi   ">;
  def BNEI   : BranchCI<0x2F, 0x01, "bnei   ">;
  def BLTI   : BranchCI<0x2F, 0x02, "blti   ">;
  def BLEI   : BranchCI<0x2F, 0x03, "blei   ">;
  def BGTI   : BranchCI<0x2F, 0x04, "bgti   ">;
  def BGEI   : BranchCI<0x2F, 0x05, "bgei   ">;
}

let isBranch = 1, isIndirectBranch = 1, isTerminator = 1, hasCtrlDep = 1,
    isBarrier = 1 in {
  def BR     :   Branch<0x26, 0x00, 0x000, "br     ">;
  def BRA    :   Branch<0x26, 0x08, 0x000, "bra    ">;
}

let isBranch = 1, isIndirectBranch = 1, isTerminator = 1, hasCtrlDep = 1 in {
  def BEQ    :  BranchC<0x27, 0x00, 0x000, "beq    ">;
  def BNE    :  BranchC<0x27, 0x01, 0x000, "bne    ">;
  def BLT    :  BranchC<0x27, 0x02, 0x000, "blt    ">;
  def BLE    :  BranchC<0x27, 0x03, 0x000, "ble    ">;
  def BGT    :  BranchC<0x27, 0x04, 0x000, "bgt    ">;
  def BGE    :  BranchC<0x27, 0x05, 0x000, "bge    ">;
}

let isBranch = 1, isTerminator = 1, hasDelaySlot = 1, hasCtrlDep = 1,
    isBarrier = 1 in {
  def BRID   :  BranchI<0x2E, 0x10, "brid   ">;
  def BRAID  :  BranchI<0x2E, 0x18, "braid  ">;
}

let isBranch = 1, isTerminator = 1, hasDelaySlot = 1, hasCtrlDep = 1 in {
  def BEQID  : BranchCI<0x2F, 0x10, "beqid  ">;
  def BNEID  : BranchCI<0x2F, 0x11, "bneid  ">;
  def BLTID  : BranchCI<0x2F, 0x12, "bltid  ">;
  def BLEID  : BranchCI<0x2F, 0x13, "bleid  ">;
  def BGTID  : BranchCI<0x2F, 0x14, "bgtid  ">;
  def BGEID  : BranchCI<0x2F, 0x15, "bgeid  ">;
}

let isBranch = 1, isIndirectBranch = 1, isTerminator = 1,
    hasDelaySlot = 1, hasCtrlDep = 1, isBarrier = 1 in {
  def BRD    :   Branch<0x26, 0x10, 0x000, "brd    ">;
  def BRAD   :   Branch<0x26, 0x18, 0x000, "brad   ">;
}

let isBranch = 1, isIndirectBranch = 1, isTerminator = 1,
    hasDelaySlot = 1, hasCtrlDep = 1 in {
  def BEQD   :  BranchC<0x27, 0x10, 0x000, "beqd   ">;
  def BNED   :  BranchC<0x27, 0x11, 0x000, "bned   ">;
  def BLTD   :  BranchC<0x27, 0x12, 0x000, "bltd   ">;
  def BLED   :  BranchC<0x27, 0x13, 0x000, "bled   ">;
  def BGTD   :  BranchC<0x27, 0x14, 0x000, "bgtd   ">;
  def BGED   :  BranchC<0x27, 0x15, 0x000, "bged   ">;
}

let isCall = 1, hasDelaySlot = 1, hasCtrlDep = 1, isBarrier = 1,
    Defs = [R3,R4,R5,R6,R7,R8,R9,R10,R11,R12],
    Uses = [R1,R5,R6,R7,R8,R9,R10] in {
  def BRLID  : BranchLI<0x2E, 0x14, "brlid  ">;
  def BRALID : BranchLI<0x2E, 0x1C, "bralid ">;
}

let isCall = 1, hasDelaySlot = 1, hasCtrlDep = 1, isIndirectBranch = 1,
    isBarrier = 1,
    Defs = [R3,R4,R5,R6,R7,R8,R9,R10,R11,R12],
    Uses = [R1,R5,R6,R7,R8,R9,R10] in {
  def BRLD   : BranchL<0x26, 0x14, 0x000, "brld   ">;
  def BRALD  : BranchL<0x26, 0x1C, 0x000, "brald  ">;
}

let isReturn=1, isTerminator=1, hasDelaySlot=1, isBarrier=1,
    hasCtrlDep=1, rd=0x10, Form=FCRI in {
  def RTSD   : TB<0x2D, (outs), (ins GPR:$target, simm16:$imm),
                  "rtsd      $target, $imm",
                  [],
                  IIBranch>;
}

let isReturn=1, isTerminator=1, hasDelaySlot=1, isBarrier=1,
    hasCtrlDep=1, rd=0x11, Form=FCRI in {
  def RTID   : TB<0x2D, (outs), (ins GPR:$target, simm16:$imm),
                  "rtid      $target, $imm",
                  [],
                  IIBranch>;
}

let isReturn=1, isTerminator=1, hasDelaySlot=1, isBarrier=1,
    hasCtrlDep=1, rd=0x12, Form=FCRI in {
  def RTBD   : TB<0x2D, (outs), (ins GPR:$target, simm16:$imm),
                  "rtbd      $target, $imm",
                  [],
                  IIBranch>;
}

let isReturn=1, isTerminator=1, hasDelaySlot=1, isBarrier=1,
    hasCtrlDep=1, rd=0x14, Form=FCRI in {
  def RTED   : TB<0x2D, (outs), (ins GPR:$target, simm16:$imm),
                  "rted      $target, $imm",
                  [],
                  IIBranch>;
}

//===----------------------------------------------------------------------===//
// MBlaze misc instructions
//===----------------------------------------------------------------------===//

let neverHasSideEffects = 1 in {
  def NOP :  MBlazeInst< 0x20, FC, (outs), (ins), "nop    ", [], IIAlu>;
}

let usesCustomInserter = 1 in {
  def Select_CC : MBlazePseudo<(outs GPR:$dst),
    (ins GPR:$T, GPR:$F, GPR:$CMP, i32imm:$CC), // F T reversed
    "; SELECT_CC PSEUDO!",
    []>;

  def ShiftL : MBlazePseudo<(outs GPR:$dst),
    (ins GPR:$L, GPR:$R),
    "; ShiftL PSEUDO!",
    []>;

  def ShiftRA : MBlazePseudo<(outs GPR:$dst),
    (ins GPR:$L, GPR:$R),
    "; ShiftRA PSEUDO!",
    []>;

  def ShiftRL : MBlazePseudo<(outs GPR:$dst),
    (ins GPR:$L, GPR:$R),
    "; ShiftRL PSEUDO!",
    []>;
}


let rb = 0 in {
  def SEXT16 : TA<0x24, 0x061, (outs GPR:$dst), (ins GPR:$src),
                  "sext16    $dst, $src", [], IIAlu>;
  def SEXT8  : TA<0x24, 0x060, (outs GPR:$dst), (ins GPR:$src),
                  "sext8     $dst, $src", [], IIAlu>;
  def SRL    : TA<0x24, 0x041, (outs GPR:$dst), (ins GPR:$src),
                  "srl       $dst, $src", [], IIAlu>;
  def SRA    : TA<0x24, 0x001, (outs GPR:$dst), (ins GPR:$src),
                  "sra       $dst, $src", [], IIAlu>;
  def SRC    : TA<0x24, 0x021, (outs GPR:$dst), (ins GPR:$src),
                  "src       $dst, $src", [], IIAlu>;
}

let isCodeGenOnly=1 in {
  def ADDI32  : ArithI32<0x08, "addi   ", simm16, immSExt16>;
  def ORI32   : LogicI32<0x28, "ori    ">;
  def BRLID32 : BranchLI<0x2E, 0x14, "brlid  ">;
}

//===----------------------------------------------------------------------===//
// Misc. instructions
//===----------------------------------------------------------------------===//
def MFS : SPC<0x25, 0x2, (outs GPR:$dst), (ins uimm14:$rg),
              "mfs       $dst, $rg", [], IIAlu>;

def MTS : SPC<0x25, 0x3, (outs), (ins uimm14:$dst, GPR:$rg),
              "mts       $dst, $rg", [], IIAlu>;

def MSRSET : MSR<0x25, 0x20, (outs GPR:$dst), (ins uimm15:$set),
                 "msrset    $dst, $set", [], IIAlu>;

def MSRCLR : MSR<0x25, 0x22, (outs GPR:$dst), (ins uimm15:$clr),
                 "msrclr    $dst, $clr", [], IIAlu>;

let rd=0x0, Form=FCRR in {
  def WDC  : TA<0x24, 0x64, (outs), (ins GPR:$a, GPR:$b),
                "wdc       $a, $b", [], IIAlu>;
  def WDCF : TA<0x24, 0x74, (outs), (ins GPR:$a, GPR:$b),
                "wdc.flush $a, $b", [], IIAlu>;
  def WDCC : TA<0x24, 0x66, (outs), (ins GPR:$a, GPR:$b),
                "wdc.clear $a, $b", [], IIAlu>;
  def WIC  : TA<0x24, 0x68, (outs), (ins GPR:$a, GPR:$b),
                "wic       $a, $b", [], IIAlu>;
}

def BRK  :  BranchL<0x26, 0x0C, 0x000, "brk    ">;
def BRKI : BranchLI<0x2E, 0x0C, "brki   ">;

def IMM : MBlazeInst<0x2C, FCCI, (outs), (ins simm16:$imm),
                     "imm       $imm", [], IIAlu>;

//===----------------------------------------------------------------------===//
//  Arbitrary patterns that map to one or more instructions
//===----------------------------------------------------------------------===//

// Small immediates
def : Pat<(i32 0), (ADD (i32 R0), (i32 R0))>;
def : Pat<(i32 immSExt16:$imm), (ADDI (i32 R0), imm:$imm)>;
def : Pat<(i32 immZExt16:$imm), (ORI (i32 R0), imm:$imm)>;

// Arbitrary immediates
def : Pat<(i32 imm:$imm), (ADDI (i32 R0), imm:$imm)>;

// In register sign extension
def : Pat<(sext_inreg GPR:$src, i16), (SEXT16 GPR:$src)>;
def : Pat<(sext_inreg GPR:$src, i8),  (SEXT8 GPR:$src)>;

// Call
def : Pat<(MBlazeJmpLink (i32 tglobaladdr:$dst)),
          (BRLID (i32 R15), tglobaladdr:$dst)>;

def : Pat<(MBlazeJmpLink (i32 texternalsym:$dst)),
          (BRLID (i32 R15), texternalsym:$dst)>;

def : Pat<(MBlazeJmpLink GPR:$dst),
          (BRALD (i32 R15), GPR:$dst)>;

// Shift Instructions
def : Pat<(shl GPR:$L, GPR:$R), (ShiftL GPR:$L, GPR:$R)>;
def : Pat<(sra GPR:$L, GPR:$R), (ShiftRA GPR:$L, GPR:$R)>;
def : Pat<(srl GPR:$L, GPR:$R), (ShiftRL GPR:$L, GPR:$R)>;

// SET_CC operations
def : Pat<(setcc (i32 GPR:$L), (i32 GPR:$R), SETEQ),
          (Select_CC (ADDI (i32 R0), 1), (ADDI (i32 R0), 0),
                     (CMP GPR:$R, GPR:$L), 1)>;
def : Pat<(setcc (i32 GPR:$L), (i32 GPR:$R), SETNE),
          (Select_CC (ADDI (i32 R0), 1), (ADDI (i32 R0), 0),
                     (CMP GPR:$R, GPR:$L), 2)>;
def : Pat<(setcc (i32 GPR:$L), (i32 GPR:$R), SETGT),
          (Select_CC (ADDI (i32 R0), 1), (ADDI (i32 R0), 0),
                     (CMP GPR:$R, GPR:$L), 3)>;
def : Pat<(setcc (i32 GPR:$L), (i32 GPR:$R), SETLT),
          (Select_CC (ADDI (i32 R0), 1), (ADDI (i32 R0), 0),
                     (CMP GPR:$R, GPR:$L), 4)>;
def : Pat<(setcc (i32 GPR:$L), (i32 GPR:$R), SETGE),
          (Select_CC (ADDI (i32 R0), 1), (ADDI (i32 R0), 0),
                     (CMP GPR:$R, GPR:$L), 5)>;
def : Pat<(setcc (i32 GPR:$L), (i32 GPR:$R), SETLE),
          (Select_CC (ADDI (i32 R0), 1), (ADDI (i32 R0), 0),
                     (CMP GPR:$R, GPR:$L), 6)>;
def : Pat<(setcc (i32 GPR:$L), (i32 GPR:$R), SETUGT),
          (Select_CC (ADDI (i32 R0), 1), (ADDI (i32 R0), 0),
                     (CMPU GPR:$R, GPR:$L), 3)>;
def : Pat<(setcc (i32 GPR:$L), (i32 GPR:$R), SETULT),
          (Select_CC (ADDI (i32 R0), 1), (ADDI (i32 R0), 0),
                     (CMPU GPR:$R, GPR:$L), 4)>;
def : Pat<(setcc (i32 GPR:$L), (i32 GPR:$R), SETUGE),
          (Select_CC (ADDI (i32 R0), 1), (ADDI (i32 R0), 0),
                     (CMPU GPR:$R, GPR:$L), 5)>;
def : Pat<(setcc (i32 GPR:$L), (i32 GPR:$R), SETULE),
          (Select_CC (ADDI (i32 R0), 1), (ADDI (i32 R0), 0),
                     (CMPU GPR:$R, GPR:$L), 6)>;

// SELECT operations
def : Pat<(select (i32 GPR:$C), (i32 GPR:$T), (i32 GPR:$F)),
          (Select_CC GPR:$T, GPR:$F, GPR:$C, 2)>;

// SELECT_CC
def : Pat<(selectcc (i32 GPR:$L), (i32 GPR:$R),
                    (i32 GPR:$T), (i32 GPR:$F), SETEQ),
          (Select_CC GPR:$T, GPR:$F, (CMP GPR:$R, GPR:$L), 1)>;
def : Pat<(selectcc (i32 GPR:$L), (i32 GPR:$R),
                    (i32 GPR:$T), (i32 GPR:$F), SETNE),
          (Select_CC GPR:$T, GPR:$F, (CMP GPR:$R, GPR:$L), 2)>;
def : Pat<(selectcc (i32 GPR:$L), (i32 GPR:$R),
                    (i32 GPR:$T), (i32 GPR:$F), SETGT),
          (Select_CC GPR:$T, GPR:$F, (CMP GPR:$R, GPR:$L), 3)>;
def : Pat<(selectcc (i32 GPR:$L), (i32 GPR:$R),
                    (i32 GPR:$T), (i32 GPR:$F), SETLT),
          (Select_CC GPR:$T, GPR:$F, (CMP GPR:$R, GPR:$L), 4)>;
def : Pat<(selectcc (i32 GPR:$L), (i32 GPR:$R),
                    (i32 GPR:$T), (i32 GPR:$F), SETGE),
          (Select_CC GPR:$T, GPR:$F, (CMP GPR:$R, GPR:$L), 5)>;
def : Pat<(selectcc (i32 GPR:$L), (i32 GPR:$R),
                    (i32 GPR:$T), (i32 GPR:$F), SETLE),
          (Select_CC GPR:$T, GPR:$F, (CMP GPR:$R, GPR:$L), 6)>;
def : Pat<(selectcc (i32 GPR:$L), (i32 GPR:$R),
                    (i32 GPR:$T), (i32 GPR:$F), SETUGT),
          (Select_CC GPR:$T, GPR:$F, (CMPU GPR:$R, GPR:$L), 3)>;
def : Pat<(selectcc (i32 GPR:$L), (i32 GPR:$R),
                    (i32 GPR:$T), (i32 GPR:$F), SETULT),
          (Select_CC GPR:$T, GPR:$F, (CMPU GPR:$R, GPR:$L), 4)>;
def : Pat<(selectcc (i32 GPR:$L), (i32 GPR:$R),
                    (i32 GPR:$T), (i32 GPR:$F), SETUGE),
          (Select_CC GPR:$T, GPR:$F, (CMPU GPR:$R, GPR:$L), 5)>;
def : Pat<(selectcc (i32 GPR:$L), (i32 GPR:$R),
                    (i32 GPR:$T), (i32 GPR:$F), SETULE),
          (Select_CC GPR:$T, GPR:$F, (CMPU GPR:$R, GPR:$L), 6)>;

// Ret instructions
def : Pat<(MBlazeRet GPR:$target), (RTSD GPR:$target, 0x8)>;

// BR instructions
def : Pat<(br bb:$T), (BRID bb:$T)>;
def : Pat<(brind GPR:$T), (BRAD GPR:$T)>;

// BRCOND instructions
def : Pat<(brcond (setcc (i32 GPR:$L), (i32 GPR:$R), SETEQ), bb:$T),
          (BEQID (CMP GPR:$R, GPR:$L), bb:$T)>;
def : Pat<(brcond (setcc (i32 GPR:$L), (i32 GPR:$R), SETNE), bb:$T),
          (BNEID (CMP GPR:$R, GPR:$L), bb:$T)>;
def : Pat<(brcond (setcc (i32 GPR:$L), (i32 GPR:$R), SETGT), bb:$T),
          (BGTID (CMP GPR:$R, GPR:$L), bb:$T)>;
def : Pat<(brcond (setcc (i32 GPR:$L), (i32 GPR:$R), SETLT), bb:$T),
          (BLTID (CMP GPR:$R, GPR:$L), bb:$T)>;
def : Pat<(brcond (setcc (i32 GPR:$L), (i32 GPR:$R), SETGE), bb:$T),
          (BGEID (CMP GPR:$R, GPR:$L), bb:$T)>;
def : Pat<(brcond (setcc (i32 GPR:$L), (i32 GPR:$R), SETLE), bb:$T),
          (BLEID (CMP GPR:$R, GPR:$L), bb:$T)>;
def : Pat<(brcond (setcc (i32 GPR:$L), (i32 GPR:$R), SETUGT), bb:$T),
          (BGTID (CMPU GPR:$R, GPR:$L), bb:$T)>;
def : Pat<(brcond (setcc (i32 GPR:$L), (i32 GPR:$R), SETULT), bb:$T),
          (BLTID (CMPU GPR:$R, GPR:$L), bb:$T)>;
def : Pat<(brcond (setcc (i32 GPR:$L), (i32 GPR:$R), SETUGE), bb:$T),
          (BGEID (CMPU GPR:$R, GPR:$L), bb:$T)>;
def : Pat<(brcond (setcc (i32 GPR:$L), (i32 GPR:$R), SETULE), bb:$T),
          (BLEID (CMPU GPR:$R, GPR:$L), bb:$T)>;
def : Pat<(brcond (i32 GPR:$C), bb:$T),
          (BNEID GPR:$C, bb:$T)>;

// Jump tables, global addresses, and constant pools
def : Pat<(MBWrapper tglobaladdr:$in), (ORI (i32 R0), tglobaladdr:$in)>;
def : Pat<(MBWrapper tjumptable:$in),  (ORI (i32 R0), tjumptable:$in)>;
def : Pat<(MBWrapper tconstpool:$in),  (ORI (i32 R0), tconstpool:$in)>;

// Misc instructions
def : Pat<(and (i32 GPR:$lh), (not (i32 GPR:$rh))),(ANDN GPR:$lh, GPR:$rh)>;

// Arithmetic with immediates
def : Pat<(add (i32 GPR:$in), imm:$imm),(ADDI GPR:$in, imm:$imm)>;
def : Pat<(or (i32 GPR:$in), imm:$imm),(ORI GPR:$in, imm:$imm)>;
def : Pat<(xor (i32 GPR:$in), imm:$imm),(XORI GPR:$in, imm:$imm)>;

// Convert any extend loads into zero extend loads
def : Pat<(extloadi8  iaddr:$src), (i32 (LBUI iaddr:$src))>;
def : Pat<(extloadi16 iaddr:$src), (i32 (LHUI iaddr:$src))>;
def : Pat<(extloadi8  xaddr:$src), (i32 (LBU xaddr:$src))>;
def : Pat<(extloadi16 xaddr:$src), (i32 (LHU xaddr:$src))>;

// 32-bit load and store
def : Pat<(store (i32 GPR:$dst), xaddr:$addr), (SW GPR:$dst, xaddr:$addr)>;
def : Pat<(load xaddr:$addr), (i32 (LW xaddr:$addr))>;

// 16-bit load and store
def : Pat<(truncstorei16 (i32 GPR:$dst), xaddr:$addr), (SH GPR:$dst, xaddr:$addr)>;
def : Pat<(zextloadi16 xaddr:$addr), (i32 (LHU xaddr:$addr))>;

// 8-bit load and store
def : Pat<(truncstorei8 (i32 GPR:$dst), xaddr:$addr), (SB GPR:$dst, xaddr:$addr)>;
def : Pat<(zextloadi8 xaddr:$addr), (i32 (LBU xaddr:$addr))>;

// Peepholes
def : Pat<(store (i32 0), iaddr:$dst), (SWI (i32 R0), iaddr:$dst)>;

//===----------------------------------------------------------------------===//
// Floating Point Support
//===----------------------------------------------------------------------===//
include "MBlazeInstrFSL.td"
include "MBlazeInstrFPU.td"
