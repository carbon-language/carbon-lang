//===- PTXInstrInfo.td - PTX Instruction defs -----------------*- tblgen-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the PTX instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction format superclass
//===----------------------------------------------------------------------===//

include "PTXInstrFormats.td"

//===----------------------------------------------------------------------===//
// PTX Specific Node Definitions
//===----------------------------------------------------------------------===//

def PTXexit
  : SDNode<"PTXISD::EXIT", SDTNone, [SDNPHasChain]>;
def PTXret
  : SDNode<"PTXISD::RET",  SDTNone, [SDNPHasChain]>;

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

///===- Data Movement and Conversion Instructions -------------------------===//

let neverHasSideEffects = 1 in {
  // rely on isMoveInstr to separate MOVpp, MOVrr, etc.
  def MOVpp
    : InstPTX<(outs Preds:$d), (ins Preds:$a), "mov.pred\t$d, $a", []>;
  def MOVrr
    : InstPTX<(outs RRegs32:$d), (ins RRegs32:$a), "mov.s32\t$d, $a", []>;
}

let isReMaterializable = 1, isAsCheapAsAMove = 1 in {
  def MOVpi
    : InstPTX<(outs Preds:$d), (ins i1imm:$a), "mov.pred\t$d, $a",
              [(set Preds:$d, imm:$a)]>;
  def MOVri
    : InstPTX<(outs RRegs32:$d), (ins i32imm:$a), "mov.s32\t$d, $a",
              [(set RRegs32:$d, imm:$a)]>;
}

///===- Control Flow Instructions -----------------------------------------===//

let isReturn = 1, isTerminator = 1, isBarrier = 1 in {
  def EXIT : InstPTX<(outs), (ins), "exit", [(PTXexit)]>;
  def RET  : InstPTX<(outs), (ins), "ret",  [(PTXret)]>;
}
