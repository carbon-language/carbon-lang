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
// Instruction Pattern Stuff
//===----------------------------------------------------------------------===//

def load_global : PatFrag<(ops node:$ptr), (load node:$ptr), [{
  if (const Value *Src = cast<LoadSDNode>(N)->getSrcValue())
    if (const PointerType *PT = dyn_cast<PointerType>(Src->getType()))
      return PT->getAddressSpace() <= 255;
  return false;
}]>;

// Addressing modes.
def ADDRri : ComplexPattern<i32, 2, "SelectADDRri", [], []>;
def ADDRii : ComplexPattern<i32, 2, "SelectADDRii", [], []>;

// Address operands
def MEMri : Operand<i32> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops RRegs32, i32imm);
}
def MEMii : Operand<i32> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops i32imm, i32imm);
}

//===----------------------------------------------------------------------===//
// PTX Specific Node Definitions
//===----------------------------------------------------------------------===//

def PTXexit
  : SDNode<"PTXISD::EXIT", SDTNone, [SDNPHasChain]>;
def PTXret
  : SDNode<"PTXISD::RET",  SDTNone, [SDNPHasChain]>;

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//

multiclass INT3<string opcstr, SDNode opnode> {
  def rr : InstPTX<(outs RRegs32:$d),
                   (ins RRegs32:$a, RRegs32:$b),
                   !strconcat(opcstr, ".%type\t$d, $a, $b"),
                   [(set RRegs32:$d, (opnode RRegs32:$a, RRegs32:$b))]>;
  def ri : InstPTX<(outs RRegs32:$d),
                   (ins RRegs32:$a, i32imm:$b),
                   !strconcat(opcstr, ".%type\t$d, $a, $b"),
                   [(set RRegs32:$d, (opnode RRegs32:$a, imm:$b))]>;
}

multiclass PTX_LD<string opstr, RegisterClass RC, PatFrag pat_load> {
  def ri : InstPTX<(outs RC:$d),
                   (ins MEMri:$a),
                   !strconcat(opstr, ".%type\t$d, [$a]"),
                   [(set RC:$d, (pat_load ADDRri:$a))]>;
  def ii : InstPTX<(outs RC:$d),
                   (ins MEMii:$a),
                   !strconcat(opstr, ".%type\t$d, [$a]"),
                   [(set RC:$d, (pat_load ADDRii:$a))]>;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

///===- Integer Arithmetic Instructions -----------------------------------===//

defm ADD : INT3<"add", add>;
defm SUB : INT3<"sub", sub>;

///===- Data Movement and Conversion Instructions -------------------------===//

let neverHasSideEffects = 1 in {
  // rely on isMoveInstr to separate MOVpp, MOVrr, etc.
  def MOVpp
    : InstPTX<(outs Preds:$d), (ins Preds:$a), "mov.pred\t$d, $a", []>;
  def MOVrr
    : InstPTX<(outs RRegs32:$d), (ins RRegs32:$a), "mov.%type\t$d, $a", []>;
}

let isReMaterializable = 1, isAsCheapAsAMove = 1 in {
  def MOVpi
    : InstPTX<(outs Preds:$d), (ins i1imm:$a), "mov.pred\t$d, $a",
              [(set Preds:$d, imm:$a)]>;
  def MOVri
    : InstPTX<(outs RRegs32:$d), (ins i32imm:$a), "mov.s32\t$d, $a",
              [(set RRegs32:$d, imm:$a)]>;
}

defm LDg : PTX_LD<"ld.global", RRegs32, load_global>;

///===- Control Flow Instructions -----------------------------------------===//

let isReturn = 1, isTerminator = 1, isBarrier = 1 in {
  def EXIT : InstPTX<(outs), (ins), "exit", [(PTXexit)]>;
  def RET  : InstPTX<(outs), (ins), "ret",  [(PTXret)]>;
}
