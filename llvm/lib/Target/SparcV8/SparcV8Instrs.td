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

include "../Target.td"
include "SparcV8Reg.td"

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

include "SparcV8Instrs_F2.td"
include "SparcV8Instrs_F3.td"

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

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

// Section B.25: Jump and Link - p126
def JMPLr : F3_1<2, 0b111000, "jmpl">;              // jmpl [rs1+rs2], rd
def JMPLi : F3_2<2, 0b111000, "jmpl">;              // jmpl [rs1+imm], rd

