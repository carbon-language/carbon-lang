//===- SparcV8InstrInfo.td - Describe the SparcV8 Instruction Set -*- C++ -*-=//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

class Format<bits<4> val> {
  bits<4> Value = val;
}

// All of the SparcV8 instruction formats, plus a pseudo-instruction format:
def Pseudo : Format<0>;
def IForm : Format<1>;
def BForm : Format<2>;
def SCForm : Format<3>;
def DForm : Format<4>;
def XForm : Format<5>;
def XLForm : Format<6>;
def XFXForm : Format<7>;
def XFLForm : Format<8>;
def XOForm : Format<9>;
def AForm : Format<10>;
def MForm : Format<11>;

class PPCInst<string nm, bits<6> opcd, Format f> : Instruction {
  let Namespace = "SparcV8";

  let Name = nm;
  bits<6> Opcode = opcd;
  Format Form = f;
  bits<4> FormBits = Form.Value;
}

// Pseudo-instructions:
def PHI : PPCInst<"PHI", 0, Pseudo>;          // PHI node...
def NOP : PPCInst<"NOP", 0, Pseudo>;          // No-op
def ADJCALLSTACKDOWN : PPCInst<"ADJCALLSTACKDOWN", 0, Pseudo>;
def ADJCALLSTACKUP : PPCInst<"ADJCALLSTACKUP", 0, Pseudo>;


