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

// Section B.20: SAVE and RESTORE - p117
def SAVEr    : F3_1<2, 0b111100, "save">;           // save    r, r, r
def SAVEi    : F3_2<2, 0b111100, "save">;           // save    r, i, r
def RESTOREr : F3_1<2, 0b111101, "restore">;        // restore r, r, r
def RESTOREi : F3_2<2, 0b111101, "restore">;        // restore r, i, r

// Section B.24: Call and Link - p125
// This is the only Format 1 instruction
def CALL : InstV8 {
  bits<30> disp;

  let op = 1;
  let Inst{29-0} = disp;
  let Name = "call";
}

// Section B.9 - SETHI Instruction, p. 102
def SETHIi: F2_1<0b100, "sethi">;

// Section B.11 - Logical Instructions, p. 106
def ANDri : F3_2<2, 0b000001, "and">;
def ORrr  : F3_1<2, 0b000010, "or">;
def ORri  : F3_2<2, 0b000010, "or">;

// Section B.12 - Shift Instructions, p. 107
def SLLri : F3_1<2, 0b100101, "sll">;
def SRLri : F3_1<2, 0b100110, "srl">;
def SRAri : F3_1<2, 0b100111, "sra">;

// Section B.13 - Add Instructions, p. 108
def ADDrr : F3_1<2, 0b000000, "add">;

// Section B.25 - Jump and Link, p. 126
def JMPLr : F3_1<2, 0b111000, "jmpl">;              // jmpl [rs1+rs2], rd
def JMPLi : F3_2<2, 0b111000, "jmpl">;              // jmpl [rs1+imm], rd

