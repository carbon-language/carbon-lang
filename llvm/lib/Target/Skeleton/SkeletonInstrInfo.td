//===- SkeletonInstrInfo.td - Describe the Instruction Set ------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Skeleton instruction information.  Fill in stuff here.
//
//===----------------------------------------------------------------------===//

class Format<bits<4> val> {
  bits<4> Value = val;
}

// All of the PowerPC instruction formats, plus a pseudo-instruction format:
def Pseudo : Format<0>;
def IForm : Format<1>;
def BForm : Format<2>;

// Look at how other targets factor commonality between instructions.
class SkelInst<string nm, bits<6> opcd, Format f> : Instruction {
  let Namespace = "Skeleton";

  let Name = nm;
  bits<6> Opcode = opcd;
  Format Form = f;
  bits<4> FormBits = Form.Value;
}

// Pseudo-instructions:
def PHI : SkelInst<"PHI", 0, Pseudo>;          // PHI node...
def NOP : SkelInst<"NOP", 0, Pseudo>;          // No-op
def ADJCALLSTACKDOWN : SkelInst<"ADJCALLSTACKDOWN", 0, Pseudo>;
def ADJCALLSTACKUP : SkelInst<"ADJCALLSTACKUP", 0, Pseudo>;


