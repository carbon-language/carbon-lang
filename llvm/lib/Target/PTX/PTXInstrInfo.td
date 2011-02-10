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
  const Value *Src;
  const PointerType *PT;
  if ((Src = cast<LoadSDNode>(N)->getSrcValue()) &&
      (PT = dyn_cast<PointerType>(Src->getType())))
    return PT->getAddressSpace() == PTX::GLOBAL;
  return false;
}]>;

def load_constant : PatFrag<(ops node:$ptr), (load node:$ptr), [{
  const Value *Src;
  const PointerType *PT;
  if ((Src = cast<LoadSDNode>(N)->getSrcValue()) &&
      (PT = dyn_cast<PointerType>(Src->getType())))
    return PT->getAddressSpace() == PTX::CONSTANT;
  return false;
}]>;

def load_local : PatFrag<(ops node:$ptr), (load node:$ptr), [{
  const Value *Src;
  const PointerType *PT;
  if ((Src = cast<LoadSDNode>(N)->getSrcValue()) &&
      (PT = dyn_cast<PointerType>(Src->getType())))
    return PT->getAddressSpace() == PTX::LOCAL;
  return false;
}]>;

def load_parameter : PatFrag<(ops node:$ptr), (load node:$ptr), [{
  const Value *Src;
  const PointerType *PT;
  if ((Src = cast<LoadSDNode>(N)->getSrcValue()) &&
      (PT = dyn_cast<PointerType>(Src->getType())))
    return PT->getAddressSpace() == PTX::PARAMETER;
  return false;
}]>;

def load_shared : PatFrag<(ops node:$ptr), (load node:$ptr), [{
  const Value *Src;
  const PointerType *PT;
  if ((Src = cast<LoadSDNode>(N)->getSrcValue()) &&
      (PT = dyn_cast<PointerType>(Src->getType())))
    return PT->getAddressSpace() == PTX::SHARED;
  return false;
}]>;

def store_global
  : PatFrag<(ops node:$d, node:$ptr), (store node:$d, node:$ptr), [{
  const Value *Src;
  const PointerType *PT;
  if ((Src = cast<StoreSDNode>(N)->getSrcValue()) &&
      (PT = dyn_cast<PointerType>(Src->getType())))
    return PT->getAddressSpace() == PTX::GLOBAL;
  return false;
}]>;

def store_local
  : PatFrag<(ops node:$d, node:$ptr), (store node:$d, node:$ptr), [{
  const Value *Src;
  const PointerType *PT;
  if ((Src = cast<StoreSDNode>(N)->getSrcValue()) &&
      (PT = dyn_cast<PointerType>(Src->getType())))
    return PT->getAddressSpace() == PTX::LOCAL;
  return false;
}]>;

def store_parameter
  : PatFrag<(ops node:$d, node:$ptr), (store node:$d, node:$ptr), [{
  const Value *Src;
  const PointerType *PT;
  if ((Src = cast<StoreSDNode>(N)->getSrcValue()) &&
      (PT = dyn_cast<PointerType>(Src->getType())))
    return PT->getAddressSpace() == PTX::PARAMETER;
  return false;
}]>;

def store_shared
  : PatFrag<(ops node:$d, node:$ptr), (store node:$d, node:$ptr), [{
  const Value *Src;
  const PointerType *PT;
  if ((Src = cast<StoreSDNode>(N)->getSrcValue()) &&
      (PT = dyn_cast<PointerType>(Src->getType())))
    return PT->getAddressSpace() == PTX::SHARED;
  return false;
}]>;

// Addressing modes.
def ADDRrr : ComplexPattern<i32, 2, "SelectADDRrr", [], []>;
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
def MEMpi : Operand<i32> {
  let PrintMethod = "printParamOperand";
  let MIOperandInfo = (ops i32imm);
}

//===----------------------------------------------------------------------===//
// PTX Specific Node Definitions
//===----------------------------------------------------------------------===//

// PTX allow generic 3-reg shifts like shl r0, r1, r2
def PTXshl : SDNode<"ISD::SHL", SDTIntBinOp>;
def PTXsrl : SDNode<"ISD::SRL", SDTIntBinOp>;
def PTXsra : SDNode<"ISD::SRA", SDTIntBinOp>;

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

// no %type directive, non-communtable
multiclass INT3ntnc<string opcstr, SDNode opnode> {
  def rr : InstPTX<(outs RRegs32:$d),
                   (ins RRegs32:$a, RRegs32:$b),
                   !strconcat(opcstr, "\t$d, $a, $b"),
                   [(set RRegs32:$d, (opnode RRegs32:$a, RRegs32:$b))]>;
  def ri : InstPTX<(outs RRegs32:$d),
                   (ins RRegs32:$a, i32imm:$b),
                   !strconcat(opcstr, "\t$d, $a, $b"),
                   [(set RRegs32:$d, (opnode RRegs32:$a, imm:$b))]>;
  def ir : InstPTX<(outs RRegs32:$d),
                   (ins i32imm:$a, RRegs32:$b),
                   !strconcat(opcstr, "\t$d, $a, $b"),
                   [(set RRegs32:$d, (opnode imm:$a, RRegs32:$b))]>;
}

multiclass PTX_LD<string opstr, RegisterClass RC, PatFrag pat_load> {
  def rr : InstPTX<(outs RC:$d),
                   (ins MEMri:$a),
                   !strconcat(opstr, ".%type\t$d, [$a]"),
                   [(set RC:$d, (pat_load ADDRrr:$a))]>;
  def ri : InstPTX<(outs RC:$d),
                   (ins MEMri:$a),
                   !strconcat(opstr, ".%type\t$d, [$a]"),
                   [(set RC:$d, (pat_load ADDRri:$a))]>;
  def ii : InstPTX<(outs RC:$d),
                   (ins MEMii:$a),
                   !strconcat(opstr, ".%type\t$d, [$a]"),
                   [(set RC:$d, (pat_load ADDRii:$a))]>;
}

multiclass PTX_ST<string opstr, RegisterClass RC, PatFrag pat_store> {
  def rr : InstPTX<(outs),
                   (ins RC:$d, MEMri:$a),
                   !strconcat(opstr, ".%type\t[$a], $d"),
                   [(pat_store RC:$d, ADDRrr:$a)]>;
  def ri : InstPTX<(outs),
                   (ins RC:$d, MEMri:$a),
                   !strconcat(opstr, ".%type\t[$a], $d"),
                   [(pat_store RC:$d, ADDRri:$a)]>;
  def ii : InstPTX<(outs),
                   (ins RC:$d, MEMii:$a),
                   !strconcat(opstr, ".%type\t[$a], $d"),
                   [(pat_store RC:$d, ADDRii:$a)]>;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

///===- Integer Arithmetic Instructions -----------------------------------===//

defm ADD : INT3<"add", add>;
defm SUB : INT3<"sub", sub>;

///===- Logic and Shift Instructions --------------------------------------===//

defm SHL : INT3ntnc<"shl.b32", PTXshl>;
defm SRL : INT3ntnc<"shr.u32", PTXsrl>;
defm SRA : INT3ntnc<"shr.s32", PTXsra>;

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
defm LDc : PTX_LD<"ld.const",  RRegs32, load_constant>;
defm LDl : PTX_LD<"ld.local",  RRegs32, load_local>;
defm LDp : PTX_LD<"ld.param",  RRegs32, load_parameter>;
defm LDs : PTX_LD<"ld.shared", RRegs32, load_shared>;

def LDpi : InstPTX<(outs RRegs32:$d), (ins MEMpi:$a),
                   "ld.param.%type\t$d, [$a]", []>;

defm STg : PTX_ST<"st.global", RRegs32, store_global>;
defm STl : PTX_ST<"st.local",  RRegs32, store_local>;
// Store to parameter state space requires PTX 2.0 or higher?
// defm STp : PTX_ST<"st.param",  RRegs32, store_parameter>;
defm STs : PTX_ST<"st.shared", RRegs32, store_shared>;

///===- Control Flow Instructions -----------------------------------------===//

let isReturn = 1, isTerminator = 1, isBarrier = 1 in {
  def EXIT : InstPTX<(outs), (ins), "exit", [(PTXexit)]>;
  def RET  : InstPTX<(outs), (ins), "ret",  [(PTXret)]>;
}
