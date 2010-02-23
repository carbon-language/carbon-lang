//===- MBlazeInstrInfo.td - MBlaze Instruction defs -------------*- C++ -*-===//
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
// MBlaze profiles and nodes
//===----------------------------------------------------------------------===//
def SDT_MBlazeRet     : SDTypeProfile<0, 1, [SDTCisInt<0>]>;
def SDT_MBlazeJmpLink : SDTypeProfile<0, 1, [SDTCisVT<0, i32>]>;

// Call
def MBlazeJmpLink : SDNode<"MBlazeISD::JmpLink",SDT_MBlazeJmpLink,
                               [SDNPHasChain,SDNPOptInFlag,SDNPOutFlag]>;

// Return
def MBlazeRet : SDNode<"MBlazeISD::Ret", SDT_MBlazeRet,
                           [SDNPHasChain, SDNPOptInFlag]>;

// Hi and Lo nodes are used to handle global addresses. Used on 
// MBlazeISelLowering to lower stuff like GlobalAddress, ExternalSymbol 
// static model.
def MBWrapper   : SDNode<"MBlazeISD::Wrap", SDTIntUnaryOp>;
def MBlazeGPRel : SDNode<"MBlazeISD::GPRel", SDTIntUnaryOp>;

def SDT_MBCallSeqStart : SDCallSeqStart<[SDTCisVT<0, i32>]>;
def SDT_MBCallSeqEnd   : SDCallSeqEnd<[SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;

// These are target-independent nodes, but have target-specific formats.
def callseq_start : SDNode<"ISD::CALLSEQ_START", SDT_MBCallSeqStart,
                           [SDNPHasChain, SDNPOutFlag]>;
def callseq_end   : SDNode<"ISD::CALLSEQ_END", SDT_MBCallSeqEnd,
                           [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;

def SDTMBlazeSelectCC : SDTypeProfile<1, 3, [SDTCisSameAs<0, 1>]>;

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

// Instruction operand types
def brtarget    : Operand<OtherVT>;
def calltarget  : Operand<i32>;
def simm16      : Operand<i32>;
def uimm5       : Operand<i32>;
def fimm        : Operand<f32>;

// Unsigned Operand
def uimm16      : Operand<i32> {
  let PrintMethod = "printUnsignedImm";
}

// FSL Operand
def fslimm      : Operand<i32> {
  let PrintMethod = "printFSLImm";
}

// Address operand
def memri : Operand<i32> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops simm16, CPURegs);
}

def memrr : Operand<i32> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops CPURegs, CPURegs);
}

// Transformation Function - get the lower 16 bits.
def LO16 : SDNodeXForm<imm, [{
  return getI32Imm((unsigned)N->getZExtValue() & 0xFFFF);
}]>;

// Transformation Function - get the higher 16 bits.
def HI16 : SDNodeXForm<imm, [{
  return getI32Imm((unsigned)N->getZExtValue() >> 16);
}]>;

// Node immediate fits as 16-bit sign extended on target immediate.
// e.g. addi, andi
def immSExt16  : PatLeaf<(imm), [{
  return (N->getZExtValue() >> 16) == 0;
}]>;

// Node immediate fits as 16-bit zero extended on target immediate.
// The LO16 param means that only the lower 16 bits of the node
// immediate are caught.
// e.g. addiu, sltiu
def immZExt16  : PatLeaf<(imm), [{
  return (N->getZExtValue() >> 16) == 0;
}], LO16>;

// FSL immediate field must fit in 4 bits.
def immZExt4 : PatLeaf<(imm), [{
      return N->getZExtValue() == ((N->getZExtValue()) & 0xf) ;
}]>;

// shamt field must fit in 5 bits.
def immZExt5 : PatLeaf<(imm), [{
      return N->getZExtValue() == ((N->getZExtValue()) & 0x1f) ;
}]>;

// MBlaze Address Mode! SDNode frameindex could possibily be a match
// since load and store instructions from stack used it.
def iaddr : ComplexPattern<i32, 2, "SelectAddrRegImm", [frameindex], []>;
def xaddr : ComplexPattern<i32, 2, "SelectAddrRegReg", [], []>;

//===----------------------------------------------------------------------===//
// Pseudo instructions
//===----------------------------------------------------------------------===//

// As stack alignment is always done with addiu, we need a 16-bit immediate
let Defs = [R1], Uses = [R1] in {
def ADJCALLSTACKDOWN : MBlazePseudo<(outs), (ins simm16:$amt),
                                  "${:comment} ADJCALLSTACKDOWN $amt",
                                  [(callseq_start timm:$amt)]>;
def ADJCALLSTACKUP   : MBlazePseudo<(outs),
                                  (ins uimm16:$amt1, simm16:$amt2),
                                  "${:comment} ADJCALLSTACKUP $amt1",
                                  [(callseq_end timm:$amt1, timm:$amt2)]>;
}

// Some assembly macros need to avoid pseudoinstructions and assembler
// automatic reodering, we should reorder ourselves.
def MACRO     : MBlazePseudo<(outs), (ins), ".set macro",     []>;
def REORDER   : MBlazePseudo<(outs), (ins), ".set reorder",   []>;
def NOMACRO   : MBlazePseudo<(outs), (ins), ".set nomacro",   []>;
def NOREORDER : MBlazePseudo<(outs), (ins), ".set noreorder", []>;

// When handling PIC code the assembler needs .cpload and .cprestore
// directives. If the real instructions corresponding these directives
// are used, we have the same behavior, but get also a bunch of warnings
// from the assembler.
def CPLOAD : MBlazePseudo<(outs), (ins CPURegs:$reg), ".cpload $reg", []>;
def CPRESTORE : MBlazePseudo<(outs), (ins uimm16:$l), ".cprestore $l\n", []>;

//===----------------------------------------------------------------------===//
// Instructions specific format
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Arithmetic Instructions
//===----------------------------------------------------------------------===//
class Arith<bits<6> op, bits<11> flags, string instr_asm, SDNode OpNode,
            InstrItinClass itin> :
            TA<op, flags, (outs CPURegs:$dst), (ins CPURegs:$b, CPURegs:$c),
               !strconcat(instr_asm, "   $dst, $b, $c"),
               [(set CPURegs:$dst, (OpNode CPURegs:$b, CPURegs:$c))], itin>;

class ArithI<bits<6> op, string instr_asm, SDNode OpNode,
             Operand Od, PatLeaf imm_type> :
             TAI<op, (outs CPURegs:$dst), (ins CPURegs:$b, Od:$c),
                 !strconcat(instr_asm, "   $dst, $b, $c"),
                 [(set CPURegs:$dst, (OpNode CPURegs:$b, imm_type:$c))], IIAlu>;

class ArithR<bits<6> op, bits<11> flags, string instr_asm, SDNode OpNode,
            InstrItinClass itin> :
            TA<op, flags, (outs CPURegs:$dst), (ins CPURegs:$c, CPURegs:$b),
               !strconcat(instr_asm, "   $dst, $c, $b"),
               [(set CPURegs:$dst, (OpNode CPURegs:$b, CPURegs:$c))], itin>;

class ArithRI<bits<6> op, string instr_asm, SDNode OpNode,
             Operand Od, PatLeaf imm_type> :
             TAI<op, (outs CPURegs:$dst), (ins Od:$b, CPURegs:$c),
                 !strconcat(instr_asm, "   $dst, $c, $b"),
                 [(set CPURegs:$dst, (OpNode imm_type:$b, CPURegs:$c))], IIAlu>;

class ArithN<bits<6> op, bits<11> flags, string instr_asm,
            InstrItinClass itin> :
            TA<op, flags, (outs CPURegs:$dst), (ins CPURegs:$b, CPURegs:$c),
               !strconcat(instr_asm, "   $dst, $b, $c"),
               [], itin>;

class ArithNI<bits<6> op, string instr_asm,Operand Od, PatLeaf imm_type> :
             TAI<op, (outs CPURegs:$dst), (ins CPURegs:$b, Od:$c),
                 !strconcat(instr_asm, "   $dst, $b, $c"),
                 [], IIAlu>;

class ArithRN<bits<6> op, bits<11> flags, string instr_asm,
            InstrItinClass itin> :
            TA<op, flags, (outs CPURegs:$dst), (ins CPURegs:$c, CPURegs:$b),
               !strconcat(instr_asm, "   $dst, $b, $c"),
               [], itin>;

class ArithRNI<bits<6> op, string instr_asm,Operand Od, PatLeaf imm_type> :
             TAI<op, (outs CPURegs:$dst), (ins Od:$c, CPURegs:$b),
                 !strconcat(instr_asm, "   $dst, $b, $c"),
                 [], IIAlu>;

//===----------------------------------------------------------------------===//
// Misc Arithmetic Instructions
//===----------------------------------------------------------------------===//

class Logic<bits<6> op, bits<11> flags, string instr_asm, SDNode OpNode> :
            TA<op, flags, (outs CPURegs:$dst), (ins CPURegs:$b, CPURegs:$c),
               !strconcat(instr_asm, "   $dst, $b, $c"),
               [(set CPURegs:$dst, (OpNode CPURegs:$b, CPURegs:$c))], IIAlu>;

class LogicI<bits<6> op, string instr_asm, SDNode OpNode> :
             TAI<op, (outs CPURegs:$dst), (ins CPURegs:$b, uimm16:$c),
                 !strconcat(instr_asm, "   $dst, $b, $c"),
                 [(set CPURegs:$dst, (OpNode CPURegs:$b, immZExt16:$c))],
                 IIAlu>;

class EffectiveAddress<string instr_asm> :
          TAI<0x08, (outs CPURegs:$dst), (ins memri:$addr),
              instr_asm, [(set CPURegs:$dst, iaddr:$addr)], IIAlu>;

//===----------------------------------------------------------------------===//
// Memory Access Instructions
//===----------------------------------------------------------------------===//
class LoadM<bits<6> op, string instr_asm, PatFrag OpNode> :
            TA<op, 0x000, (outs CPURegs:$dst), (ins memrr:$addr),
               !strconcat(instr_asm, "   $dst, $addr"),
               [(set CPURegs:$dst, (OpNode xaddr:$addr))], IILoad>;

class LoadMI<bits<6> op, string instr_asm, PatFrag OpNode> :
             TAI<op, (outs CPURegs:$dst), (ins memri:$addr),
                 !strconcat(instr_asm, "   $dst, $addr"),
                 [(set CPURegs:$dst, (OpNode iaddr:$addr))], IILoad>;

class StoreM<bits<6> op, string instr_asm, PatFrag OpNode> :
             TA<op, 0x000, (outs), (ins CPURegs:$dst, memrr:$addr),
                !strconcat(instr_asm, "   $dst, $addr"),
                [(OpNode CPURegs:$dst, xaddr:$addr)], IIStore>;

class StoreMI<bits<6> op, string instr_asm, PatFrag OpNode> :
              TAI<op, (outs), (ins CPURegs:$dst, memri:$addr),
                  !strconcat(instr_asm, "   $dst, $addr"),
                  [(OpNode CPURegs:$dst, iaddr:$addr)], IIStore>;

//===----------------------------------------------------------------------===//
// Branch Instructions
//===----------------------------------------------------------------------===//
class Branch<bits<6> op, bits<5> br, bits<11> flags, string instr_asm> :
             TBR<op, br, flags, (outs), (ins CPURegs:$target),
                 !strconcat(instr_asm, "   $target"),
                 [(brind CPURegs:$target)], IIBranch>;

class BranchI<bits<6> op, bits<5> brf, string instr_asm> :
              TBRI<op, brf, (outs), (ins brtarget:$target),
                   !strconcat(instr_asm, "   $target"),
                   [(br bb:$target)], IIBranch>;

//===----------------------------------------------------------------------===//
// Branch and Link Instructions
//===----------------------------------------------------------------------===//
class BranchL<bits<6> op, bits<5> br, bits<11> flags, string instr_asm> :
              TBRL<op, br, flags, (outs), (ins CPURegs:$target),
                   !strconcat(instr_asm, "   r15, $target"),
                   [], IIBranch>;

class BranchLI<bits<6> op, bits<5> br, string instr_asm> :
               TBRLI<op, br, (outs), (ins calltarget:$target),
                     !strconcat(instr_asm, "   r15, $target"),
                     [], IIBranch>;

//===----------------------------------------------------------------------===//
// Conditional Branch Instructions
//===----------------------------------------------------------------------===//
class BranchC<bits<6> op, bits<5> br, bits<11> flags, string instr_asm,
              PatFrag cond_op> :
              TBRC<op, br, flags, (outs),
                   (ins CPURegs:$a, CPURegs:$b, brtarget:$offset),
                   !strconcat(instr_asm, "   $a, $b, $offset"),
                   [], IIBranch>; 
                   //(brcond (cond_op CPURegs:$a, CPURegs:$b), bb:$offset)],
                   //IIBranch>;

class BranchCI<bits<6> op, bits<5> br, string instr_asm, PatFrag cond_op> :
               TBRCI<op, br, (outs), (ins CPURegs:$a, brtarget:$offset),
                     !strconcat(instr_asm, "   $a, $offset"),
                     [], IIBranch>;

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
    def BSRLI  :  ArithI<0x11, "bsrli  ", srl, uimm5, immZExt5>;
    def BSRAI  :  ArithI<0x11, "bsrai  ", sra, uimm5, immZExt5>;
    def BSLLI  :  ArithI<0x11, "bslli  ", shl, uimm5, immZExt5>;
}

let Predicates=[HasDiv] in {
    def IDIV   :  Arith<0x12, 0x000, "idiv   ", sdiv, IIAlu>;
    def IDIVU  :  Arith<0x12, 0x002, "idivu  ", udiv, IIAlu>;
}

//===----------------------------------------------------------------------===//
// MBlaze immediate mode arithmetic instructions
//===----------------------------------------------------------------------===//

let isAsCheapAsAMove = 1 in {
    def ADDI    :   ArithI<0x08, "addi   ", add,  simm16, immSExt16>;
    def ADDIC   :  ArithNI<0x0A, "addic  ", simm16, immSExt16>;
    def ADDIK   :  ArithNI<0x0C, "addik  ", simm16, immSExt16>;
    def ADDIKC  :   ArithI<0x0E, "addikc ", addc, simm16, immSExt16>;
    def RSUBI   :   ArithRI<0x09, "rsubi  ", sub,  simm16, immSExt16>;
    def RSUBIC  :  ArithRNI<0x0B, "rsubi  ", simm16, immSExt16>;
    def RSUBIK  :  ArithRNI<0x0E, "rsubic ", simm16, immSExt16>;
    def RSUBIKC :   ArithRI<0x0F, "rsubikc", subc, simm16, immSExt16>;
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
    def LBU  :  LoadM<0x30, "lbu    ", zextloadi8>;
    def LHU  :  LoadM<0x31, "lhu    ", zextloadi16>;
    def LW   :  LoadM<0x32, "lw     ", load>;

    def LBUI : LoadMI<0x30, "lbui   ", zextloadi8>;
    def LHUI : LoadMI<0x31, "lhui   ", zextloadi16>;
    def LWI  : LoadMI<0x32, "lwi    ", load>;
}

    def SB  :  StoreM<0x34, "sb     ", truncstorei8>;
    def SH  :  StoreM<0x35, "sh     ", truncstorei16>;
    def SW  :  StoreM<0x36, "sw     ", store>;

    def SBI : StoreMI<0x34, "sbi    ", truncstorei8>;
    def SHI : StoreMI<0x35, "shi    ", truncstorei16>;
    def SWI : StoreMI<0x36, "swi    ", store>;

//===----------------------------------------------------------------------===//
// MBlaze branch instructions
//===----------------------------------------------------------------------===//

let isBranch = 1, isTerminator = 1, hasCtrlDep = 1 in {
    def BRI    :  BranchI<0x2E, 0x00, "bri    ">;
    def BRAI   :  BranchI<0x2E, 0x08, "brai   ">;
    def BEQI   : BranchCI<0x2F, 0x00, "beqi   ", seteq>;
    def BNEI   : BranchCI<0x2F, 0x01, "bnei   ", setne>;
    def BLTI   : BranchCI<0x2F, 0x02, "blti   ", setlt>;
    def BLEI   : BranchCI<0x2F, 0x03, "blei   ", setle>;
    def BGTI   : BranchCI<0x2F, 0x04, "bgti   ", setgt>;
    def BGEI   : BranchCI<0x2F, 0x05, "bgei   ", setge>;
}

let isBranch = 1, isIndirectBranch = 1, isTerminator = 1, hasCtrlDep = 1 in {
    def BR     :   Branch<0x26, 0x00, 0x000, "br     ">;
    def BRA    :   Branch<0x26, 0x08, 0x000, "bra    ">;
    def BEQ    :  BranchC<0x27, 0x00, 0x000, "beq    ", seteq>;
    def BNE    :  BranchC<0x27, 0x01, 0x000, "bne    ", setne>;
    def BLT    :  BranchC<0x27, 0x02, 0x000, "blt    ", setlt>;
    def BLE    :  BranchC<0x27, 0x03, 0x000, "ble    ", setle>;
    def BGT    :  BranchC<0x27, 0x04, 0x000, "bgt    ", setgt>;
    def BGE    :  BranchC<0x27, 0x05, 0x000, "bge    ", setge>;
}

let isBranch = 1, isTerminator = 1, hasDelaySlot = 1, hasCtrlDep = 1 in {
    def BRID   :  BranchI<0x2E, 0x10, "brid   ">;
    def BRAID  :  BranchI<0x2E, 0x18, "braid  ">;
    def BEQID  : BranchCI<0x2F, 0x10, "beqid  ", seteq>;
    def BNEID  : BranchCI<0x2F, 0x11, "bneid  ", setne>;
    def BLTID  : BranchCI<0x2F, 0x12, "bltid  ", setlt>;
    def BLEID  : BranchCI<0x2F, 0x13, "bleid  ", setle>;
    def BGTID  : BranchCI<0x2F, 0x14, "bgtid  ", setgt>;
    def BGEID  : BranchCI<0x2F, 0x15, "bgeid  ", setge>;
}

let isBranch = 1, isIndirectBranch = 1, isTerminator = 1,
    hasDelaySlot = 1, hasCtrlDep = 1 in {
    def BRD    :   Branch<0x26, 0x10, 0x000, "brd    ">;
    def BRAD   :   Branch<0x26, 0x18, 0x000, "brad   ">;
    def BEQD   :  BranchC<0x27, 0x10, 0x000, "beqd   ", seteq>;
    def BNED   :  BranchC<0x27, 0x11, 0x000, "bned   ", setne>;
    def BLTD   :  BranchC<0x27, 0x12, 0x000, "bltd   ", setlt>;
    def BLED   :  BranchC<0x27, 0x13, 0x000, "bled   ", setle>;
    def BGTD   :  BranchC<0x27, 0x14, 0x000, "bgtd   ", setgt>;
    def BGED   :  BranchC<0x27, 0x15, 0x000, "bged   ", setge>;
}

let isCall = 1, hasCtrlDep = 1, isIndirectBranch = 1,
    Defs = [R3,R4,R5,R6,R7,R8,R9,R10,R11,R12],
    Uses = [R1,R5,R6,R7,R8,R9,R10] in {
    def BRL    : BranchL<0x26, 0x04, 0x000, "brl    ">;
    def BRAL   : BranchL<0x26, 0x0C, 0x000, "bral   ">;
}

let isCall = 1, hasDelaySlot = 1, hasCtrlDep = 1,
    Defs = [R3,R4,R5,R6,R7,R8,R9,R10,R11,R12],
    Uses = [R1,R5,R6,R7,R8,R9,R10] in {
    def BRLID  : BranchLI<0x2E, 0x14, "brlid  ">;
    def BRALID : BranchLI<0x2E, 0x1C, "bralid ">;
}

let isCall = 1, hasDelaySlot = 1, hasCtrlDep = 1, isIndirectBranch = 1,
    Defs = [R3,R4,R5,R6,R7,R8,R9,R10,R11,R12],
    Uses = [R1,R5,R6,R7,R8,R9,R10] in {
    def BRLD   : BranchL<0x26, 0x14, 0x000, "brld   ">;
    def BRALD  : BranchL<0x26, 0x1C, 0x000, "brald  ">;
}

let isReturn=1, isTerminator=1, hasDelaySlot=1,
    isBarrier=1, hasCtrlDep=1, imm16=0x8 in {
    def RTSD   : TRET<0x2D, (outs), (ins CPURegs:$target),
                      "rtsd      $target, 8",
                      [(MBlazeRet CPURegs:$target)],
                      IIBranch>;
}

//===----------------------------------------------------------------------===//
// MBlaze misc instructions
//===----------------------------------------------------------------------===//

let addr = 0 in {
    def NOP :  TADDR<0x00, (outs), (ins), "nop    ", [], IIAlu>;
}

let usesCustomInserter = 1 in {
  //class PseudoSelCC<RegisterClass RC, string asmstr>:
  //  MBlazePseudo<(outs RC:$D), (ins RC:$T, RC:$F, CPURegs:$CMP), asmstr,
  //  [(set RC:$D, (MBlazeSelectCC RC:$T, RC:$F, CPURegs:$CMP))]>;
  //def Select_CC : PseudoSelCC<CPURegs, "# MBlazeSelect_CC">;

  def Select_CC : MBlazePseudo<(outs CPURegs:$dst),
    (ins CPURegs:$T, CPURegs:$F, CPURegs:$CMP, i32imm:$CC),
    "; SELECT_CC PSEUDO!",
    []>;

  def ShiftL : MBlazePseudo<(outs CPURegs:$dst),
    (ins CPURegs:$L, CPURegs:$R),
    "; ShiftL PSEUDO!",
    []>;

  def ShiftRA : MBlazePseudo<(outs CPURegs:$dst),
    (ins CPURegs:$L, CPURegs:$R),
    "; ShiftRA PSEUDO!",
    []>;

  def ShiftRL : MBlazePseudo<(outs CPURegs:$dst),
    (ins CPURegs:$L, CPURegs:$R),
    "; ShiftRL PSEUDO!",
    []>;
}


let rb = 0 in {
    def SEXT16 : TA<0x24, 0x061, (outs CPURegs:$dst), (ins CPURegs:$src),
                    "sext16  $dst, $src", [], IIAlu>;
    def SEXT8  : TA<0x24, 0x060, (outs CPURegs:$dst), (ins CPURegs:$src),
                    "sext8   $dst, $src", [], IIAlu>;
    def SRL    : TA<0x24, 0x041, (outs CPURegs:$dst), (ins CPURegs:$src),
                    "srl     $dst, $src", [], IIAlu>;
    def SRA    : TA<0x24, 0x001, (outs CPURegs:$dst), (ins CPURegs:$src),
                    "sra     $dst, $src", [], IIAlu>;
    def SRC    : TA<0x24, 0x021, (outs CPURegs:$dst), (ins CPURegs:$src),
                    "src     $dst, $src", [], IIAlu>;
}

def LEA_ADDI : EffectiveAddress<"addi    $dst, ${addr:stackloc}">;

//===----------------------------------------------------------------------===//
//  Arbitrary patterns that map to one or more instructions
//===----------------------------------------------------------------------===//

// Small immediates
def : Pat<(i32 0), (ADD R0, R0)>;
def : Pat<(i32 immSExt16:$imm), (ADDI R0, imm:$imm)>;
def : Pat<(i32 immZExt16:$imm), (ORI R0, imm:$imm)>;

// Arbitrary immediates
def : Pat<(i32 imm:$imm), (ADDI R0, imm:$imm)>;

// In register sign extension
def : Pat<(sext_inreg CPURegs:$src, i16), (SEXT16 CPURegs:$src)>;
def : Pat<(sext_inreg CPURegs:$src, i8),  (SEXT8 CPURegs:$src)>;

// Call
def : Pat<(MBlazeJmpLink (i32 tglobaladdr:$dst)), (BRLID tglobaladdr:$dst)>;
def : Pat<(MBlazeJmpLink (i32 texternalsym:$dst)),(BRLID texternalsym:$dst)>;
def : Pat<(MBlazeJmpLink CPURegs:$dst), (BRLD CPURegs:$dst)>;

// Shift Instructions
def : Pat<(shl CPURegs:$L, CPURegs:$R), (ShiftL CPURegs:$L, CPURegs:$R)>;
def : Pat<(sra CPURegs:$L, CPURegs:$R), (ShiftRA CPURegs:$L, CPURegs:$R)>;
def : Pat<(srl CPURegs:$L, CPURegs:$R), (ShiftRL CPURegs:$L, CPURegs:$R)>;

// SET_CC operations
def : Pat<(setcc CPURegs:$L, CPURegs:$R, SETEQ),
          (Select_CC (ADDI R0, 1), (ADDI R0, 0), 
                     (CMP CPURegs:$L, CPURegs:$R), 1)>;
def : Pat<(setcc CPURegs:$L, CPURegs:$R, SETNE),
          (Select_CC (ADDI R0, 1), (ADDI R0, 0), 
                     (CMP CPURegs:$L, CPURegs:$R), 2)>;
def : Pat<(setcc CPURegs:$L, CPURegs:$R, SETGT),
          (Select_CC (ADDI R0, 1), (ADDI R0, 0), 
                     (CMP CPURegs:$L, CPURegs:$R), 3)>;
def : Pat<(setcc CPURegs:$L, CPURegs:$R, SETLT),
          (Select_CC (ADDI R0, 1), (ADDI R0, 0), 
                     (CMP CPURegs:$L, CPURegs:$R), 4)>;
def : Pat<(setcc CPURegs:$L, CPURegs:$R, SETGE),
          (Select_CC (ADDI R0, 1), (ADDI R0, 0), 
                     (CMP CPURegs:$L, CPURegs:$R), 5)>;
def : Pat<(setcc CPURegs:$L, CPURegs:$R, SETLE),
          (Select_CC (ADDI R0, 1), (ADDI R0, 0), 
                     (CMP CPURegs:$L, CPURegs:$R), 6)>;
def : Pat<(setcc CPURegs:$L, CPURegs:$R, SETUGT),
          (Select_CC (ADDI R0, 1), (ADDI R0, 0), 
                     (CMPU CPURegs:$L, CPURegs:$R), 3)>;
def : Pat<(setcc CPURegs:$L, CPURegs:$R, SETULT),
          (Select_CC (ADDI R0, 1), (ADDI R0, 0), 
                     (CMPU CPURegs:$L, CPURegs:$R), 4)>;
def : Pat<(setcc CPURegs:$L, CPURegs:$R, SETUGE),
          (Select_CC (ADDI R0, 1), (ADDI R0, 0), 
                     (CMPU CPURegs:$L, CPURegs:$R), 5)>;
def : Pat<(setcc CPURegs:$L, CPURegs:$R, SETULE),
          (Select_CC (ADDI R0, 1), (ADDI R0, 0), 
                     (CMPU CPURegs:$L, CPURegs:$R), 6)>;

// SELECT operations
def : Pat<(select CPURegs:$C, CPURegs:$T, CPURegs:$F),
          (Select_CC CPURegs:$T, CPURegs:$F, CPURegs:$C, 2)>;

// SELECT_CC 
def : Pat<(selectcc CPURegs:$L, CPURegs:$R, CPURegs:$T, CPURegs:$F, SETEQ),
          (Select_CC CPURegs:$T, CPURegs:$F, (CMP CPURegs:$L, CPURegs:$R), 1)>;
def : Pat<(selectcc CPURegs:$L, CPURegs:$R, CPURegs:$T, CPURegs:$F, SETNE),
          (Select_CC CPURegs:$T, CPURegs:$F, (CMP CPURegs:$L, CPURegs:$R), 2)>;
def : Pat<(selectcc CPURegs:$L, CPURegs:$R, CPURegs:$T, CPURegs:$F, SETGT),
          (Select_CC CPURegs:$T, CPURegs:$F, (CMP CPURegs:$L, CPURegs:$R), 3)>;
def : Pat<(selectcc CPURegs:$L, CPURegs:$R, CPURegs:$T, CPURegs:$F, SETLT),
          (Select_CC CPURegs:$T, CPURegs:$F, (CMP CPURegs:$L, CPURegs:$R), 4)>;
def : Pat<(selectcc CPURegs:$L, CPURegs:$R, CPURegs:$T, CPURegs:$F, SETGE),
          (Select_CC CPURegs:$T, CPURegs:$F, (CMP CPURegs:$L, CPURegs:$R), 5)>;
def : Pat<(selectcc CPURegs:$L, CPURegs:$R, CPURegs:$T, CPURegs:$F, SETLE),
          (Select_CC CPURegs:$T, CPURegs:$F, (CMP CPURegs:$L, CPURegs:$R), 6)>;
def : Pat<(selectcc CPURegs:$L, CPURegs:$R, CPURegs:$T, CPURegs:$F, SETUGT),
          (Select_CC CPURegs:$T, CPURegs:$F, (CMPU CPURegs:$L, CPURegs:$R), 3)>;
def : Pat<(selectcc CPURegs:$L, CPURegs:$R, CPURegs:$T, CPURegs:$F, SETULT),
          (Select_CC CPURegs:$T, CPURegs:$F, (CMPU CPURegs:$L, CPURegs:$R), 4)>;
def : Pat<(selectcc CPURegs:$L, CPURegs:$R, CPURegs:$T, CPURegs:$F, SETUGE),
          (Select_CC CPURegs:$T, CPURegs:$F, (CMPU CPURegs:$L, CPURegs:$R), 5)>;
def : Pat<(selectcc CPURegs:$L, CPURegs:$R, CPURegs:$T, CPURegs:$F, SETULE),
          (Select_CC CPURegs:$T, CPURegs:$F, (CMPU CPURegs:$L, CPURegs:$R), 6)>;

// BRCOND instructions
def : Pat<(brcond (setcc CPURegs:$L, CPURegs:$R, SETEQ), bb:$T),
          (BEQID (CMP CPURegs:$R, CPURegs:$L), bb:$T)>;
def : Pat<(brcond (setcc CPURegs:$L, CPURegs:$R, SETNE), bb:$T),
          (BNEID (CMP CPURegs:$R, CPURegs:$L), bb:$T)>;
def : Pat<(brcond (setcc CPURegs:$L, CPURegs:$R, SETGT), bb:$T),
          (BGTID (CMP CPURegs:$R, CPURegs:$L), bb:$T)>;
def : Pat<(brcond (setcc CPURegs:$L, CPURegs:$R, SETLT), bb:$T),
          (BLTID (CMP CPURegs:$R, CPURegs:$L), bb:$T)>;
def : Pat<(brcond (setcc CPURegs:$L, CPURegs:$R, SETGE), bb:$T),
          (BGEID (CMP CPURegs:$R, CPURegs:$L), bb:$T)>;
def : Pat<(brcond (setcc CPURegs:$L, CPURegs:$R, SETLE), bb:$T),
          (BLEID (CMP CPURegs:$R, CPURegs:$L), bb:$T)>;
def : Pat<(brcond (setcc CPURegs:$L, CPURegs:$R, SETUGT), bb:$T),
          (BGTID (CMPU CPURegs:$R, CPURegs:$L), bb:$T)>;
def : Pat<(brcond (setcc CPURegs:$L, CPURegs:$R, SETULT), bb:$T),
          (BLTID (CMPU CPURegs:$R, CPURegs:$L), bb:$T)>;
def : Pat<(brcond (setcc CPURegs:$L, CPURegs:$R, SETUGE), bb:$T),
          (BGEID (CMPU CPURegs:$R, CPURegs:$L), bb:$T)>;
def : Pat<(brcond (setcc CPURegs:$L, CPURegs:$R, SETULE), bb:$T),
          (BLEID (CMPU CPURegs:$R, CPURegs:$L), bb:$T)>;
def : Pat<(brcond CPURegs:$C, bb:$T),
          (BNEID CPURegs:$C, bb:$T)>;

// Jump tables, global addresses, and constant pools
def : Pat<(MBWrapper tglobaladdr:$in), (ORI R0, tglobaladdr:$in)>;
def : Pat<(MBWrapper tjumptable:$in),  (ORI R0, tjumptable:$in)>;
def : Pat<(MBWrapper tconstpool:$in),  (ORI R0, tconstpool:$in)>;

// Misc instructions
def : Pat<(and CPURegs:$lh, (not CPURegs:$rh)),(ANDN CPURegs:$lh, CPURegs:$rh)>;

// Arithmetic with immediates
def : Pat<(add CPURegs:$in, imm:$imm),(ADDI CPURegs:$in, imm:$imm)>;
def : Pat<(or CPURegs:$in, imm:$imm),(ORI CPURegs:$in, imm:$imm)>;
def : Pat<(xor CPURegs:$in, imm:$imm),(XORI CPURegs:$in, imm:$imm)>;

// extended load and stores
def : Pat<(extloadi1  iaddr:$src), (LBUI iaddr:$src)>;
def : Pat<(extloadi8  iaddr:$src), (LBUI iaddr:$src)>;
def : Pat<(extloadi16 iaddr:$src), (LHUI iaddr:$src)>;
def : Pat<(extloadi1  xaddr:$src), (LBU  xaddr:$src)>;
def : Pat<(extloadi8  xaddr:$src), (LBU  xaddr:$src)>;
def : Pat<(extloadi16 xaddr:$src), (LHU  xaddr:$src)>;

def : Pat<(sextloadi1  iaddr:$src), (SEXT8  (LBUI iaddr:$src))>;
def : Pat<(sextloadi8  iaddr:$src), (SEXT8  (LBUI iaddr:$src))>;
def : Pat<(sextloadi16 iaddr:$src), (SEXT16 (LHUI iaddr:$src))>;
def : Pat<(sextloadi1  xaddr:$src), (SEXT8  (LBU xaddr:$src))>;
def : Pat<(sextloadi8  xaddr:$src), (SEXT8  (LBU xaddr:$src))>;
def : Pat<(sextloadi16 xaddr:$src), (SEXT16 (LHU xaddr:$src))>;

// peepholes
def : Pat<(store (i32 0), iaddr:$dst), (SWI R0, iaddr:$dst)>;

//===----------------------------------------------------------------------===//
// Floating Point Support
//===----------------------------------------------------------------------===//
include "MBlazeInstrFSL.td"
include "MBlazeInstrFPU.td"
