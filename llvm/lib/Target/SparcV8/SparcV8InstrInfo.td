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

include "SparcV8InstrInfo_F2.td"
include "SparcV8InstrInfo_F3.td"

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

// Pseudo instructions.
def PHI : InstV8 {
  let Name = "PHI";
}
def ADJCALLSTACKDOWN : InstV8 {
  let Name = "ADJCALLSTACKDOWN";
}
def ADJCALLSTACKUP : InstV8 {
  let Name = "ADJCALLSTACKUP";
}

// Section A.3 - Synthetic Instructions, p. 85
// special cases of JMPL:
let isReturn = 1, isTerminator = 1, simm13 = 8 in
  def RET : F3_2<2, 0b111000, "ret">;
let isReturn = 1, isTerminator = 1, simm13 = 8 in
  def RETL: F3_2<2, 0b111000, "retl">;
// CMP is a special case of SUBCC where destination is ignored, by setting it to
// %g0 (hardwired zero).
// FIXME: should keep track of the fact that it defs the integer condition codes
let rd = 0 in
  def CMPri: F3_2<2, 0b010100, "cmp">;

// Section B.1 - Load Integer Instructions, p. 90
def LDSBmr: F3_2<3, 0b001001, "ldsb">;
def LDSHmr: F3_2<3, 0b001010, "ldsh">;
def LDUBmr: F3_2<3, 0b000001, "ldub">;
def LDUHmr: F3_2<3, 0b000010, "lduh">;
def LDmr  : F3_2<3, 0b000000, "ld">;
def LDDmr : F3_2<3, 0b000011, "ldd">;

// Section B.2 - Load Floating-point Instructions, p. 92
def LDFmr  : F3_2<3, 0b100000, "ld">;
def LDDFmr : F3_2<3, 0b100011, "ldd">;
def LDFSRmr: F3_2<3, 0b100001, "ld">;

// Section B.4 - Store Integer Instructions, p. 95
def STBrm : F3_2<3, 0b000101, "stb">;
def STHrm : F3_2<3, 0b000110, "sth">;
def STrm  : F3_2<3, 0b000100, "st">;
def STDrm : F3_2<3, 0b000111, "std">;

// Section B.9 - SETHI Instruction, p. 104
def SETHIi: F2_1<0b100, "sethi">;

// Section B.10 - NOP Instruction, p. 105
// (It's a special case of SETHI)
let rd = 0, imm = 0 in
  def NOP : F2_1<0b100, "nop">;

// Section B.11 - Logical Instructions, p. 106
def ANDrr : F3_1<2, 0b000001, "and">;
def ANDri : F3_2<2, 0b000001, "and">;
def ORrr  : F3_1<2, 0b000010, "or">;
def ORri  : F3_2<2, 0b000010, "or">;
def XORrr : F3_1<2, 0b000011, "xor">;
def XORri : F3_2<2, 0b000011, "xor">;

// Section B.12 - Shift Instructions, p. 107
def SLLrr : F3_1<2, 0b100101, "sll">;
def SLLri : F3_2<2, 0b100101, "sll">;
def SRLrr : F3_1<2, 0b100110, "srl">;
def SRLri : F3_2<2, 0b100110, "srl">;
def SRArr : F3_1<2, 0b100111, "sra">;
def SRAri : F3_2<2, 0b100111, "sra">;

// Section B.13 - Add Instructions, p. 108
def ADDrr : F3_1<2, 0b000000, "add">;
def ADDri : F3_2<2, 0b000000, "add">;

// Section B.15 - Subtract Instructions, p. 110
def SUBrr   : F3_1<2, 0b000100, "sub">;
def SUBCCrr : F3_1<2, 0b010100, "subcc">;
def SUBCCri : F3_2<2, 0b010100, "subcc">;

// Section B.18 - Multiply Instructions, p. 113
def UMULrr : F3_1<2, 0b001010, "umul">;
def SMULrr : F3_1<2, 0b001011, "smul">;

// Section B.19 - Divide Instructions, p. 115
def UDIVrr   : F3_1<2, 0b001110, "udiv">;
def UDIVri   : F3_2<2, 0b001110, "udiv">;
def SDIVrr   : F3_1<2, 0b001111, "sdiv">;
def SDIVri   : F3_2<2, 0b001111, "sdiv">;
def UDIVCCrr : F3_1<2, 0b011110, "udivcc">;
def UDIVCCri : F3_2<2, 0b011110, "udivcc">;
def SDIVCCrr : F3_1<2, 0b011111, "sdivcc">;
def SDIVCCri : F3_2<2, 0b011111, "sdivcc">;

// Section B.20 - SAVE and RESTORE, p. 117
def SAVErr    : F3_1<2, 0b111100, "save">;           // save    r, r, r
def SAVEri    : F3_2<2, 0b111100, "save">;           // save    r, i, r
def RESTORErr : F3_1<2, 0b111101, "restore">;        // restore r, r, r
def RESTOREri : F3_2<2, 0b111101, "restore">;        // restore r, i, r

// Section B.21 - Branch on Integer Condition Codes Instructions, p. 119

// conditional branch class:
class BranchV8<bits<4> cc, string nm> : F2_2<cc, 0b010, nm> {
  let isBranch = 1;
  let isTerminator = 1;
}
def BA   : BranchV8<0b1000, "ba">;
def BN   : BranchV8<0b0000, "bn">;
def BNE  : BranchV8<0b1001, "bne">;
def BE   : BranchV8<0b0001, "be">;
def BG   : BranchV8<0b1010, "bg">;
def BLE  : BranchV8<0b0010, "ble">;
def BGE  : BranchV8<0b1011, "bge">;
def BL   : BranchV8<0b0011, "bl">;
def BGU  : BranchV8<0b1100, "bgu">;
def BLEU : BranchV8<0b0100, "bleu">;
def BCC  : BranchV8<0b1101, "bcc">;
def BCS  : BranchV8<0b0101, "bcs">;

// Section B.24 - Call and Link Instruction, p. 125
// This is the only Format 1 instruction
def CALL : InstV8 {
  bits<30> disp;
  let op = 1;
  let Inst{29-0} = disp;
  let Name = "call";
  let isCall = 1;
}

// Section B.25 - Jump and Link, p. 126
// FIXME: are they to be used as branches, calls, or returns? (not used now)
def JMPLrr : F3_1<2, 0b111000, "jmpl">;              // jmpl [rs1+rs2], rd
def JMPLri : F3_2<2, 0b111000, "jmpl">;              // jmpl [rs1+imm], rd

// Section B.29 - Write State Register Instructions
def WRrr : F3_1<2, 0b110000, "wr">;                    // wr rs1, rs2, rd
def WRri : F3_2<2, 0b110000, "wr">;                    // wr rs1, imm, rd

