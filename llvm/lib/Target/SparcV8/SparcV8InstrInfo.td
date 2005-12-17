//===- SparcV8Instrs.td - Target Description for SparcV8 Target -----------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the SparcV8 instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction format superclass
//===----------------------------------------------------------------------===//

class InstV8 : Instruction {          // SparcV8 instruction baseline
  field bits<32> Inst;

  let Namespace = "V8";

  bits<2> op;
  let Inst{31-30} = op;               // Top two bits are the 'op' field

  // Bit attributes specific to SparcV8 instructions
  bit isPasi       = 0; // Does this instruction affect an alternate addr space?
  bit isPrivileged = 0; // Is this a privileged instruction?
}

include "SparcV8InstrFormats.td"

//===----------------------------------------------------------------------===//
// Instruction Pattern Stuff
//===----------------------------------------------------------------------===//

def simm13  : PatLeaf<(imm), [{
  // simm13 predicate - True if the imm fits in a 13-bit sign extended field.
  return (((int)N->getValue() << (32-13)) >> (32-13)) == (int)N->getValue();
}]>;

def LO10 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant((unsigned)N->getValue() & 1023, MVT::i32);
}]>;

def HI22 : SDNodeXForm<imm, [{
  // Transformation function: shift the immediate value down into the low bits.
  return CurDAG->getTargetConstant((unsigned)N->getValue() >> 10, MVT::i32);
}]>;

def SETHIimm : PatLeaf<(imm), [{
  return (((unsigned)N->getValue() >> 10) << 10) == (unsigned)N->getValue();
}], HI22>;

// Addressing modes.
def ADDRrr : ComplexPattern<i32, 2, "SelectADDRrr", []>;
def ADDRri : ComplexPattern<i32, 2, "SelectADDRri", []>;

// Address operands
def MEMrr : Operand<i32> {
  let PrintMethod = "printMemOperand";
  let NumMIOperands = 2;
  let MIOperandInfo = (ops IntRegs, IntRegs);
}
def MEMri : Operand<i32> {
  let PrintMethod = "printMemOperand";
  let NumMIOperands = 2;
  let MIOperandInfo = (ops IntRegs, i32imm);
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

// Pseudo instructions.
class PseudoInstV8<string asmstr, dag ops> : InstV8  {
  let AsmString = asmstr;
  dag OperandList = ops;
}
def PHI : PseudoInstV8<"PHI", (ops variable_ops)>;
def ADJCALLSTACKDOWN : PseudoInstV8<"!ADJCALLSTACKDOWN $amt",
                                    (ops i32imm:$amt)>;
def ADJCALLSTACKUP : PseudoInstV8<"!ADJCALLSTACKUP $amt",
                                  (ops i32imm:$amt)>;
//def IMPLICIT_USE : PseudoInstV8<"!IMPLICIT_USE",(ops variable_ops)>;
def IMPLICIT_DEF : PseudoInstV8<"!IMPLICIT_DEF $dst", 
                                (ops IntRegs:$dst)>;
def FpMOVD : PseudoInstV8<"!FpMOVD", (ops)>; // pseudo 64-bit double move

// Section A.3 - Synthetic Instructions, p. 85
// special cases of JMPL:
let isReturn = 1, isTerminator = 1, hasDelaySlot = 1 in {
  let rd = I7.Num, rs1 = G0.Num, simm13 = 8 in
    def RET : F3_2<2, 0b111000,
                   (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                   "ret $b, $c, $dst", []>;
  let rd = O7.Num, rs1 = G0.Num, simm13 = 8 in
    def RETL: F3_2<2, 0b111000, (ops),
                   "retl", [(ret)]>;
}
// CMP is a special case of SUBCC where destination is ignored, by setting it to
// %g0 (hardwired zero).
// FIXME: should keep track of the fact that it defs the integer condition codes
let rd = 0 in
  def CMPri: F3_2<2, 0b010100,
                  (ops IntRegs:$b, i32imm:$c),
                  "cmp $b, $c", []>;

// Section B.1 - Load Integer Instructions, p. 90
def LDSBrr : F3_1<3, 0b001001,
                  (ops IntRegs:$dst, MEMrr:$addr),
                  "ldsb [$addr], $dst",
                  [(set IntRegs:$dst, (sextload ADDRrr:$addr, i8))]>;
def LDSBri : F3_2<3, 0b001001,
                  (ops IntRegs:$dst, MEMri:$addr),
                  "ldsb [$addr], $dst",
                  [(set IntRegs:$dst, (sextload ADDRri:$addr, i8))]>;
def LDSHrr : F3_1<3, 0b001010,
                  (ops IntRegs:$dst, MEMrr:$addr),
                  "ldsh [$addr], $dst",
                  [(set IntRegs:$dst, (sextload ADDRrr:$addr, i16))]>;
def LDSHri : F3_2<3, 0b001010,
                  (ops IntRegs:$dst, MEMri:$addr),
                  "ldsh [$addr], $dst",
                  [(set IntRegs:$dst, (sextload ADDRri:$addr, i16))]>;
def LDUBrr : F3_1<3, 0b000001,
                  (ops IntRegs:$dst, MEMrr:$addr),
                  "ldub [$addr], $dst",
                  [(set IntRegs:$dst, (zextload ADDRrr:$addr, i8))]>;
def LDUBri : F3_2<3, 0b000001,
                  (ops IntRegs:$dst, MEMri:$addr),
                  "ldub [$addr], $dst",
                  [(set IntRegs:$dst, (zextload ADDRri:$addr, i8))]>;
def LDUHrr : F3_1<3, 0b000010,
                  (ops IntRegs:$dst, MEMrr:$addr),
                  "lduh [$addr], $dst",
                  [(set IntRegs:$dst, (zextload ADDRrr:$addr, i16))]>;
def LDUHri : F3_2<3, 0b000010,
                  (ops IntRegs:$dst, MEMri:$addr),
                  "lduh [$addr], $dst",
                  [(set IntRegs:$dst, (zextload ADDRri:$addr, i16))]>;
def LDrr   : F3_1<3, 0b000000,
                  (ops IntRegs:$dst, MEMrr:$addr),
                  "ld [$addr], $dst",
                  [(set IntRegs:$dst, (load ADDRrr:$addr))]>;
def LDri   : F3_2<3, 0b000000,
                  (ops IntRegs:$dst, MEMri:$addr),
                  "ld [$addr], $dst",
                  [(set IntRegs:$dst, (load ADDRri:$addr))]>;
def LDDrr  : F3_1<3, 0b000011,
                  (ops IntRegs:$dst, MEMrr:$addr),
                  "ldd [$addr], $dst", []>;
def LDDri  : F3_2<3, 0b000011,
                  (ops IntRegs:$dst, MEMri:$addr),
                  "ldd [$addr], $dst", []>;

// Section B.2 - Load Floating-point Instructions, p. 92
def LDFrr  : F3_1<3, 0b100000,
                  (ops FPRegs:$dst, MEMrr:$addr),
                  "ld [$addr], $dst",
                  [(set FPRegs:$dst, (load ADDRrr:$addr))]>;
def LDFri  : F3_2<3, 0b100000,
                  (ops FPRegs:$dst, MEMri:$addr),
                  "ld [$addr], $dst",
                  [(set FPRegs:$dst, (load ADDRri:$addr))]>;
def LDDFrr : F3_1<3, 0b100011,
                  (ops DFPRegs:$dst, MEMrr:$addr),
                  "ldd [$addr], $dst",
                  [(set DFPRegs:$dst, (load ADDRrr:$addr))]>;
def LDDFri : F3_2<3, 0b100011,
                  (ops DFPRegs:$dst, MEMri:$addr),
                  "ldd [$addr], $dst",
                  [(set DFPRegs:$dst, (load ADDRri:$addr))]>;

// Section B.4 - Store Integer Instructions, p. 95
def STBrr : F3_1<3, 0b000101,
                 (ops MEMrr:$addr, IntRegs:$src),
                 "stb $src, [$addr]",
                 [(truncstore IntRegs:$src, ADDRrr:$addr, i8)]>;
def STBri : F3_2<3, 0b000101,
                 (ops MEMri:$addr, IntRegs:$src),
                 "stb $src, [$addr]",
                 [(truncstore IntRegs:$src, ADDRri:$addr, i8)]>;
def STHrr : F3_1<3, 0b000110,
                 (ops MEMrr:$addr, IntRegs:$src),
                 "sth $src, [$addr]",
                 [(truncstore IntRegs:$src, ADDRrr:$addr, i16)]>;
def STHri : F3_2<3, 0b000110,
                 (ops MEMri:$addr, IntRegs:$src),
                 "sth $src, [$addr]",
                 [(truncstore IntRegs:$src, ADDRri:$addr, i16)]>;
def STrr  : F3_1<3, 0b000100,
                 (ops MEMrr:$addr, IntRegs:$src),
                 "st $src, [$addr]",
                 [(store IntRegs:$src, ADDRrr:$addr)]>;
def STri  : F3_2<3, 0b000100,
                 (ops MEMri:$addr, IntRegs:$src),
                 "st $src, [$addr]",
                 [(store IntRegs:$src, ADDRri:$addr)]>;
def STDrr : F3_1<3, 0b000111,
                 (ops MEMrr:$addr, IntRegs:$src),
                 "std $src, [$addr]", []>;
def STDri : F3_2<3, 0b000111,
                 (ops MEMri:$addr, IntRegs:$src),
                 "std $src, [$addr]", []>;

// Section B.5 - Store Floating-point Instructions, p. 97
def STFrr   : F3_1<3, 0b100100,
                   (ops MEMrr:$addr, FPRegs:$src),
                   "st $src, [$addr]",
                   [(store FPRegs:$src, ADDRrr:$addr)]>;
def STFri   : F3_2<3, 0b100100,
                   (ops MEMri:$addr, FPRegs:$src),
                   "st $src, [$addr]",
                   [(store FPRegs:$src, ADDRri:$addr)]>;
def STDFrr  : F3_1<3, 0b100111,
                   (ops MEMrr:$addr, DFPRegs:$src),
                   "std  $src, [$addr]",
                   [(store DFPRegs:$src, ADDRrr:$addr)]>;
def STDFri  : F3_2<3, 0b100111,
                   (ops MEMri:$addr, DFPRegs:$src),
                   "std $src, [$addr]",
                   [(store DFPRegs:$src, ADDRri:$addr)]>;

// Section B.9 - SETHI Instruction, p. 104
def SETHIi: F2_1<0b100,
                 (ops IntRegs:$dst, i32imm:$src),
                 "sethi $src, $dst",
                 [(set IntRegs:$dst, SETHIimm:$src)]>;

// Section B.10 - NOP Instruction, p. 105
// (It's a special case of SETHI)
let rd = 0, imm22 = 0 in
  def NOP : F2_1<0b100, (ops), "nop", []>;

// Section B.11 - Logical Instructions, p. 106
def ANDrr   : F3_1<2, 0b000001,
                   (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                   "and $b, $c, $dst",
                   [(set IntRegs:$dst, (and IntRegs:$b, IntRegs:$c))]>;
def ANDri   : F3_2<2, 0b000001,
                   (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                   "and $b, $c, $dst",
                   [(set IntRegs:$dst, (and IntRegs:$b, simm13:$c))]>;
def ANDNrr  : F3_1<2, 0b000101,
                   (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                   "andn $b, $c, $dst",
                   [(set IntRegs:$dst, (and IntRegs:$b, (not IntRegs:$c)))]>;
def ANDNri  : F3_2<2, 0b000101,
                   (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                   "andn $b, $c, $dst", []>;
def ORrr    : F3_1<2, 0b000010,
                   (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                   "or $b, $c, $dst",
                   [(set IntRegs:$dst, (or IntRegs:$b, IntRegs:$c))]>;
def ORri    : F3_2<2, 0b000010,
                   (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                   "or $b, $c, $dst",
                   [(set IntRegs:$dst, (or IntRegs:$b, simm13:$c))]>;
def ORNrr   : F3_1<2, 0b000110,
                   (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                   "orn $b, $c, $dst",
                   [(set IntRegs:$dst, (or IntRegs:$b, (not IntRegs:$c)))]>;
def ORNri   : F3_2<2, 0b000110,
                   (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                   "orn $b, $c, $dst", []>;
def XORrr   : F3_1<2, 0b000011,
                   (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                   "xor $b, $c, $dst",
                   [(set IntRegs:$dst, (xor IntRegs:$b, IntRegs:$c))]>;
def XORri   : F3_2<2, 0b000011,
                   (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                   "xor $b, $c, $dst",
                   [(set IntRegs:$dst, (xor IntRegs:$b, simm13:$c))]>;
def XNORrr  : F3_1<2, 0b000111,
                   (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                   "xnor $b, $c, $dst",
                   [(set IntRegs:$dst, (xor IntRegs:$b, (not IntRegs:$c)))]>;
def XNORri  : F3_2<2, 0b000111,
                   (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                   "xnor $b, $c, $dst", []>;

// Section B.12 - Shift Instructions, p. 107
def SLLrr : F3_1<2, 0b100101,
                 (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                 "sll $b, $c, $dst",
                 [(set IntRegs:$dst, (shl IntRegs:$b, IntRegs:$c))]>;
def SLLri : F3_2<2, 0b100101,
                 (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                 "sll $b, $c, $dst",
                 [(set IntRegs:$dst, (shl IntRegs:$b, simm13:$c))]>;
def SRLrr : F3_1<2, 0b100110, 
                 (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                  "srl $b, $c, $dst",
                  [(set IntRegs:$dst, (srl IntRegs:$b, IntRegs:$c))]>;
def SRLri : F3_2<2, 0b100110,
                 (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                 "srl $b, $c, $dst", 
                 [(set IntRegs:$dst, (srl IntRegs:$b, simm13:$c))]>;
def SRArr : F3_1<2, 0b100111, 
                 (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                  "sra $b, $c, $dst",
                  [(set IntRegs:$dst, (sra IntRegs:$b, IntRegs:$c))]>;
def SRAri : F3_2<2, 0b100111,
                 (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                 "sra $b, $c, $dst",
                 [(set IntRegs:$dst, (sra IntRegs:$b, simm13:$c))]>;

// Section B.13 - Add Instructions, p. 108
def ADDrr   : F3_1<2, 0b000000, 
                  (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                  "add $b, $c, $dst",
                   [(set IntRegs:$dst, (add IntRegs:$b, IntRegs:$c))]>;
def ADDri   : F3_2<2, 0b000000,
                   (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                   "add $b, $c, $dst",
                   [(set IntRegs:$dst, (add IntRegs:$b, simm13:$c))]>;
def ADDCCrr : F3_1<2, 0b010000, 
                   (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                   "addcc $b, $c, $dst", []>;
def ADDCCri : F3_2<2, 0b010000,
                   (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                   "addcc $b, $c, $dst", []>;
def ADDXrr  : F3_1<2, 0b001000, 
                   (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                   "addx $b, $c, $dst", []>;
def ADDXri  : F3_2<2, 0b001000,
                   (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                   "addx $b, $c, $dst", []>;

// Section B.15 - Subtract Instructions, p. 110
def SUBrr   : F3_1<2, 0b000100, 
                   (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                   "sub $b, $c, $dst",
                   [(set IntRegs:$dst, (sub IntRegs:$b, IntRegs:$c))]>;
def SUBri   : F3_2<2, 0b000100,
                   (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                   "sub $b, $c, $dst",
                   [(set IntRegs:$dst, (sub IntRegs:$b, simm13:$c))]>;
def SUBXrr  : F3_1<2, 0b001100, 
                   (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                   "subx $b, $c, $dst", []>;
def SUBXri  : F3_2<2, 0b001100,
                   (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                   "subx $b, $c, $dst", []>;
def SUBCCrr : F3_1<2, 0b010100, 
                   (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                   "subcc $b, $c, $dst", []>;
def SUBCCri : F3_2<2, 0b010100,
                   (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                   "subcc $b, $c, $dst", []>;
def SUBXCCrr: F3_1<2, 0b011100, 
                   (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                   "subxcc $b, $c, $dst", []>;

// Section B.18 - Multiply Instructions, p. 113
def UMULrr  : F3_1<2, 0b001010, 
                   (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                   "umul $b, $c, $dst", []>;
def UMULri  : F3_2<2, 0b001010,
                   (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                   "umul $b, $c, $dst", []>;
def SMULrr  : F3_1<2, 0b001011, 
                   (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                   "smul $b, $c, $dst", []>;
def SMULri  : F3_2<2, 0b001011,
                   (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                   "smul $b, $c, $dst", []>;

// Section B.19 - Divide Instructions, p. 115
def UDIVrr   : F3_1<2, 0b001110, 
                    (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                    "udiv $b, $c, $dst", []>;
def UDIVri   : F3_2<2, 0b001110,
                    (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                    "udiv $b, $c, $dst", []>;
def SDIVrr   : F3_1<2, 0b001111,
                    (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                    "sdiv $b, $c, $dst", []>;
def SDIVri   : F3_2<2, 0b001111,
                    (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                    "sdiv $b, $c, $dst", []>;

// Section B.20 - SAVE and RESTORE, p. 117
def SAVErr    : F3_1<2, 0b111100,
                     (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                     "save $b, $c, $dst", []>;
def SAVEri    : F3_2<2, 0b111100,
                     (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                     "save $b, $c, $dst", []>;
def RESTORErr : F3_1<2, 0b111101,
                     (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                     "restore $b, $c, $dst", []>;
def RESTOREri : F3_2<2, 0b111101,
                     (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                     "restore $b, $c, $dst", []>;

// Section B.21 - Branch on Integer Condition Codes Instructions, p. 119

// conditional branch class:
class BranchV8<bits<4> cc, dag ops, string asmstr>
 : F2_2<cc, 0b010, ops, asmstr> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 1;
}

let isBarrier = 1 in
  def BA   : BranchV8<0b1000, (ops IntRegs:$dst), "ba $dst">;
def BN   : BranchV8<0b0000, (ops IntRegs:$dst), "bn $dst">;
def BNE  : BranchV8<0b1001, (ops IntRegs:$dst), "bne $dst">;
def BE   : BranchV8<0b0001, (ops IntRegs:$dst), "be $dst">;
def BG   : BranchV8<0b1010, (ops IntRegs:$dst), "bg $dst">;
def BLE  : BranchV8<0b0010, (ops IntRegs:$dst), "ble $dst">;
def BGE  : BranchV8<0b1011, (ops IntRegs:$dst), "bge $dst">;
def BL   : BranchV8<0b0011, (ops IntRegs:$dst), "bl $dst">;
def BGU  : BranchV8<0b1100, (ops IntRegs:$dst), "bgu $dst">;
def BLEU : BranchV8<0b0100, (ops IntRegs:$dst), "bleu $dst">;
def BCC  : BranchV8<0b1101, (ops IntRegs:$dst), "bcc $dst">;
def BCS  : BranchV8<0b0101, (ops IntRegs:$dst), "bcs $dst">;

// Section B.22 - Branch on Floating-point Condition Codes Instructions, p. 121

// floating-point conditional branch class:
class FPBranchV8<bits<4> cc, dag ops, string asmstr>
 : F2_2<cc, 0b110, ops, asmstr> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 1;
}

def FBA  : FPBranchV8<0b1000, (ops IntRegs:$dst), "fba $dst">;
def FBN  : FPBranchV8<0b0000, (ops IntRegs:$dst), "fbn $dst">;
def FBU  : FPBranchV8<0b0111, (ops IntRegs:$dst), "fbu $dst">;
def FBG  : FPBranchV8<0b0110, (ops IntRegs:$dst), "fbg $dst">;
def FBUG : FPBranchV8<0b0101, (ops IntRegs:$dst), "fbug $dst">;
def FBL  : FPBranchV8<0b0100, (ops IntRegs:$dst), "fbl $dst">;
def FBUL : FPBranchV8<0b0011, (ops IntRegs:$dst), "fbul $dst">;
def FBLG : FPBranchV8<0b0010, (ops IntRegs:$dst), "fblg $dst">;
def FBNE : FPBranchV8<0b0001, (ops IntRegs:$dst), "fbne $dst">;
def FBE  : FPBranchV8<0b1001, (ops IntRegs:$dst), "fbe $dst">;
def FBUE : FPBranchV8<0b1010, (ops IntRegs:$dst), "fbue $dst">;
def FBGE : FPBranchV8<0b1011, (ops IntRegs:$dst), "fbge $dst">;
def FBUGE: FPBranchV8<0b1100, (ops IntRegs:$dst), "fbuge $dst">;
def FBLE : FPBranchV8<0b1101, (ops IntRegs:$dst), "fble $dst">;
def FBULE: FPBranchV8<0b1110, (ops IntRegs:$dst), "fbule $dst">;
def FBO  : FPBranchV8<0b1111, (ops IntRegs:$dst), "fbo $dst">;



// Section B.24 - Call and Link Instruction, p. 125
// This is the only Format 1 instruction
let Uses = [O0, O1, O2, O3, O4, O5], hasDelaySlot = 1, isCall = 1 in { 
  // pc-relative call:
  let Defs = [O0, O1, O2, O3, O4, O5, O7, G1, G2, G3, G4, G5, G6, G7,
    D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14, D15] in
  def CALL : InstV8 {
    let OperandList = (ops IntRegs:$dst);
    bits<30> disp;
    let op = 1;
    let Inst{29-0} = disp;
    let AsmString = "call $dst";
  }

  // indirect call (O7 is an EXPLICIT def in indirect calls, so it cannot also
  // be an implicit def):
  let Defs = [O0, O1, O2, O3, O4, O5, G1, G2, G3, G4, G5, G6, G7,
    D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14, D15] in
  def JMPLrr : F3_1<2, 0b111000,
                    (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                    "jmpl $b+$c, $dst", []>;
}

// Section B.29 - Write State Register Instructions
def WRrr : F3_1<2, 0b110000,
                (ops IntRegs:$dst, IntRegs:$b, IntRegs:$c),
                "wr $b, $c, $dst", []>;
def WRri : F3_2<2, 0b110000,
                (ops IntRegs:$dst, IntRegs:$b, i32imm:$c),
                "wr $b, $c, $dst", []>;

// Convert Integer to Floating-point Instructions, p. 141
def FITOS : F3_3<2, 0b110100, 0b011000100,
                 (ops FPRegs:$dst, FPRegs:$src),
                 "fitos $src, $dst">;
def FITOD : F3_3<2, 0b110100, 0b011001000, 
                 (ops DFPRegs:$dst, DFPRegs:$src),
                 "fitod $src, $dst">;

// Convert Floating-point to Integer Instructions, p. 142
def FSTOI : F3_3<2, 0b110100, 0b011010001,
                 (ops FPRegs:$dst, FPRegs:$src),
                 "fstoi $src, $dst">;
def FDTOI : F3_3<2, 0b110100, 0b011010010,
                 (ops DFPRegs:$dst, DFPRegs:$src),
                 "fdtoi $src, $dst">;

// Convert between Floating-point Formats Instructions, p. 143
def FSTOD : F3_3<2, 0b110100, 0b011001001, 
                 (ops DFPRegs:$dst, FPRegs:$src),
                 "fstod $src, $dst">;
def FDTOS : F3_3<2, 0b110100, 0b011000110,
                 (ops FPRegs:$dst, DFPRegs:$src),
                 "fdtos $src, $dst">;

// Floating-point Move Instructions, p. 144
def FMOVS : F3_3<2, 0b110100, 0b000000001,
                 (ops FPRegs:$dst, FPRegs:$src),
                 "fmovs $src, $dst">;
def FNEGS : F3_3<2, 0b110100, 0b000000101, 
                 (ops FPRegs:$dst, FPRegs:$src),
                 "fnegs $src, $dst">;
def FABSS : F3_3<2, 0b110100, 0b000001001, 
                 (ops FPRegs:$dst, FPRegs:$src),
                 "fabss $src, $dst">;

// Floating-point Add and Subtract Instructions, p. 146
def FADDS  : F3_3<2, 0b110100, 0b001000001,
                  (ops FPRegs:$dst, FPRegs:$src1, FPRegs:$src2),
                  "fadds $src1, $src2, $dst">;
def FADDD  : F3_3<2, 0b110100, 0b001000010,
                  (ops DFPRegs:$dst, DFPRegs:$src1, DFPRegs:$src2),
                  "faddd $src1, $src2, $dst">;
def FSUBS  : F3_3<2, 0b110100, 0b001000101,
                  (ops FPRegs:$dst, FPRegs:$src1, FPRegs:$src2),
                  "fsubs $src1, $src2, $dst">;
def FSUBD  : F3_3<2, 0b110100, 0b001000110,
                  (ops DFPRegs:$dst, DFPRegs:$src1, DFPRegs:$src2),
                  "fsubd $src1, $src2, $dst">;

// Floating-point Multiply and Divide Instructions, p. 147
def FMULS  : F3_3<2, 0b110100, 0b001001001,
                  (ops FPRegs:$dst, FPRegs:$src1, FPRegs:$src2),
                  "fmuls $src1, $src2, $dst">;
def FMULD  : F3_3<2, 0b110100, 0b001001010,
                  (ops DFPRegs:$dst, DFPRegs:$src1, DFPRegs:$src2),
                  "fmuld $src1, $src2, $dst">;
def FSMULD : F3_3<2, 0b110100, 0b001101001,
                  (ops DFPRegs:$dst, FPRegs:$src1, FPRegs:$src2),
                  "fsmuld $src1, $src2, $dst">;
def FDIVS  : F3_3<2, 0b110100, 0b001001101,
                 (ops FPRegs:$dst, FPRegs:$src1, FPRegs:$src2),
                 "fdivs $src1, $src2, $dst">;
def FDIVD  : F3_3<2, 0b110100, 0b001001110,
                 (ops DFPRegs:$dst, DFPRegs:$src1, DFPRegs:$src2),
                 "fdivd $src1, $src2, $dst">;

// Floating-point Compare Instructions, p. 148
// Note: the 2nd template arg is different for these guys.
// Note 2: the result of a FCMP is not available until the 2nd cycle
// after the instr is retired, but there is no interlock. This behavior
// is modelled with a forced noop after the instruction.
def FCMPS  : F3_3<2, 0b110101, 0b001010001,
                  (ops FPRegs:$src1, FPRegs:$src2),
                  "fcmps $src1, $src2\n\tnop">;
def FCMPD  : F3_3<2, 0b110101, 0b001010010,
                  (ops DFPRegs:$src1, DFPRegs:$src2),
                  "fcmpd $src1, $src2\n\tnop">;

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//===----------------------------------------------------------------------===//

// Small immediates.
def : Pat<(i32 simm13:$val),
          (ORri G0, imm:$val)>;
// Arbitrary immediates.
def : Pat<(i32 imm:$val),
          (ORri (SETHIi (HI22 imm:$val)), (LO10 imm:$val))>;
