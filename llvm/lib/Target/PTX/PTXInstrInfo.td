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
// Code Generation Predicates
//===----------------------------------------------------------------------===//

def Use32BitAddresses : Predicate<"!getSubtarget().use64BitAddresses()">;
def Use64BitAddresses : Predicate<"getSubtarget().use64BitAddresses()">;

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
def ADDRrr32 : ComplexPattern<i32, 2, "SelectADDRrr", [], []>;
def ADDRrr64 : ComplexPattern<i64, 2, "SelectADDRrr", [], []>;
def ADDRri32 : ComplexPattern<i32, 2, "SelectADDRri", [], []>;
def ADDRri64 : ComplexPattern<i64, 2, "SelectADDRri", [], []>;
def ADDRii32 : ComplexPattern<i32, 2, "SelectADDRii", [], []>;
def ADDRii64 : ComplexPattern<i64, 2, "SelectADDRii", [], []>;


// Address operands
def MEMri32 : Operand<i32> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops RRegu32, i32imm);
}
def MEMri64 : Operand<i64> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops RRegu64, i64imm);
}
def MEMii32 : Operand<i32> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops i32imm, i32imm);
}
def MEMii64 : Operand<i64> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops i64imm, i64imm);
}
// The operand here does not correspond to an actual address, so we
// can use i32 in 64-bit address modes.
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

// Three-operand floating-point instruction template
multiclass FLOAT3<string opcstr, SDNode opnode> {
  def rr32 : InstPTX<(outs RRegf32:$d),
                     (ins RRegf32:$a, RRegf32:$b),
                     !strconcat(opcstr, ".f32\t$d, $a, $b"),
                     [(set RRegf32:$d, (opnode RRegf32:$a, RRegf32:$b))]>;
  def ri32 : InstPTX<(outs RRegf32:$d),
                     (ins RRegf32:$a, f32imm:$b),
                     !strconcat(opcstr, ".f32\t$d, $a, $b"),
                     [(set RRegf32:$d, (opnode RRegf32:$a, fpimm:$b))]>;
  def rr64 : InstPTX<(outs RRegf64:$d),
                     (ins RRegf64:$a, RRegf64:$b),
                     !strconcat(opcstr, ".f64\t$d, $a, $b"),
                     [(set RRegf64:$d, (opnode RRegf64:$a, RRegf64:$b))]>;
  def ri64 : InstPTX<(outs RRegf64:$d),
                     (ins RRegf64:$a, f64imm:$b),
                     !strconcat(opcstr, ".f64\t$d, $a, $b"),
                     [(set RRegf64:$d, (opnode RRegf64:$a, fpimm:$b))]>;
}

multiclass INT3<string opcstr, SDNode opnode> {
  def rr16 : InstPTX<(outs RRegu16:$d),
                     (ins RRegu16:$a, RRegu16:$b),
                     !strconcat(opcstr, ".u16\t$d, $a, $b"),
                     [(set RRegu16:$d, (opnode RRegu16:$a, RRegu16:$b))]>;
  def ri16 : InstPTX<(outs RRegu16:$d),
                     (ins RRegu16:$a, i16imm:$b),
                     !strconcat(opcstr, ".u16\t$d, $a, $b"),
                     [(set RRegu16:$d, (opnode RRegu16:$a, imm:$b))]>;
  def rr32 : InstPTX<(outs RRegu32:$d),
                     (ins RRegu32:$a, RRegu32:$b),
                     !strconcat(opcstr, ".u32\t$d, $a, $b"),
                     [(set RRegu32:$d, (opnode RRegu32:$a, RRegu32:$b))]>;
  def ri32 : InstPTX<(outs RRegu32:$d),
                     (ins RRegu32:$a, i32imm:$b),
                     !strconcat(opcstr, ".u32\t$d, $a, $b"),
                     [(set RRegu32:$d, (opnode RRegu32:$a, imm:$b))]>;
  def rr64 : InstPTX<(outs RRegu64:$d),
                     (ins RRegu64:$a, RRegu64:$b),
                     !strconcat(opcstr, ".u64\t$d, $a, $b"),
                     [(set RRegu64:$d, (opnode RRegu64:$a, RRegu64:$b))]>;
  def ri64 : InstPTX<(outs RRegu64:$d),
                     (ins RRegu64:$a, i64imm:$b),
                     !strconcat(opcstr, ".u64\t$d, $a, $b"),
                     [(set RRegu64:$d, (opnode RRegu64:$a, imm:$b))]>;
}

// no %type directive, non-communtable
multiclass INT3ntnc<string opcstr, SDNode opnode> {
  def rr : InstPTX<(outs RRegu32:$d),
                   (ins RRegu32:$a, RRegu32:$b),
                   !strconcat(opcstr, "\t$d, $a, $b"),
                   [(set RRegu32:$d, (opnode RRegu32:$a, RRegu32:$b))]>;
  def ri : InstPTX<(outs RRegu32:$d),
                   (ins RRegu32:$a, i32imm:$b),
                   !strconcat(opcstr, "\t$d, $a, $b"),
                   [(set RRegu32:$d, (opnode RRegu32:$a, imm:$b))]>;
  def ir : InstPTX<(outs RRegu32:$d),
                   (ins i32imm:$a, RRegu32:$b),
                   !strconcat(opcstr, "\t$d, $a, $b"),
                   [(set RRegu32:$d, (opnode imm:$a, RRegu32:$b))]>;
}

multiclass PTX_LD<string opstr, string typestr, RegisterClass RC, PatFrag pat_load> {
  def rr32 : InstPTX<(outs RC:$d),
                     (ins MEMri32:$a),
                     !strconcat(opstr, !strconcat(typestr, "\t$d, [$a]")),
                     [(set RC:$d, (pat_load ADDRrr32:$a))]>, Requires<[Use32BitAddresses]>;
  def rr64 : InstPTX<(outs RC:$d),
                     (ins MEMri64:$a),
                     !strconcat(opstr, !strconcat(typestr, "\t$d, [$a]")),
                     [(set RC:$d, (pat_load ADDRrr64:$a))]>, Requires<[Use64BitAddresses]>;
  def ri32 : InstPTX<(outs RC:$d),
                     (ins MEMri32:$a),
                     !strconcat(opstr, !strconcat(typestr, "\t$d, [$a]")),
                     [(set RC:$d, (pat_load ADDRri32:$a))]>, Requires<[Use32BitAddresses]>;
  def ri64 : InstPTX<(outs RC:$d),
                     (ins MEMri64:$a),
                     !strconcat(opstr, !strconcat(typestr, "\t$d, [$a]")),
                     [(set RC:$d, (pat_load ADDRri64:$a))]>, Requires<[Use64BitAddresses]>;
  def ii32 : InstPTX<(outs RC:$d),
                     (ins MEMii32:$a),
                     !strconcat(opstr, !strconcat(typestr, "\t$d, [$a]")),
                     [(set RC:$d, (pat_load ADDRii32:$a))]>, Requires<[Use32BitAddresses]>;
  def ii64 : InstPTX<(outs RC:$d),
                     (ins MEMii64:$a),
                     !strconcat(opstr, !strconcat(typestr, "\t$d, [$a]")),
                     [(set RC:$d, (pat_load ADDRii64:$a))]>, Requires<[Use64BitAddresses]>;
}

multiclass PTX_LD_ALL<string opstr, PatFrag pat_load> {
  defm u16 : PTX_LD<opstr, ".u16", RRegu16, pat_load>;
  defm u32 : PTX_LD<opstr, ".u32", RRegu32, pat_load>;
  defm u64 : PTX_LD<opstr, ".u64", RRegu64, pat_load>;
  defm f32 : PTX_LD<opstr, ".f32", RRegf32, pat_load>;
  defm f64 : PTX_LD<opstr, ".f64", RRegf64, pat_load>;
}

multiclass PTX_ST<string opstr, string typestr, RegisterClass RC, PatFrag pat_store> {
  def rr32 : InstPTX<(outs),
                     (ins RC:$d, MEMri32:$a),
                     !strconcat(opstr, !strconcat(typestr, "\t[$a], $d")),
                     [(pat_store RC:$d, ADDRrr32:$a)]>, Requires<[Use32BitAddresses]>;
  def rr64 : InstPTX<(outs),
                     (ins RC:$d, MEMri64:$a),
                     !strconcat(opstr, !strconcat(typestr, "\t[$a], $d")),
                     [(pat_store RC:$d, ADDRrr64:$a)]>, Requires<[Use64BitAddresses]>;
  def ri32 : InstPTX<(outs),
                   (ins RC:$d, MEMri32:$a),
                   !strconcat(opstr, !strconcat(typestr, "\t[$a], $d")),
                   [(pat_store RC:$d, ADDRri32:$a)]>, Requires<[Use32BitAddresses]>;
  def ri64 : InstPTX<(outs),
                   (ins RC:$d, MEMri64:$a),
                   !strconcat(opstr, !strconcat(typestr, "\t[$a], $d")),
                   [(pat_store RC:$d, ADDRri64:$a)]>, Requires<[Use64BitAddresses]>;
  def ii32 : InstPTX<(outs),
                   (ins RC:$d, MEMii32:$a),
                   !strconcat(opstr, !strconcat(typestr, "\t[$a], $d")),
                   [(pat_store RC:$d, ADDRii32:$a)]>, Requires<[Use32BitAddresses]>;
  def ii64 : InstPTX<(outs),
                   (ins RC:$d, MEMii64:$a),
                   !strconcat(opstr, !strconcat(typestr, "\t[$a], $d")),
                   [(pat_store RC:$d, ADDRii64:$a)]>, Requires<[Use64BitAddresses]>;
}

multiclass PTX_ST_ALL<string opstr, PatFrag pat_store> {
  defm u16 : PTX_ST<opstr, ".u16", RRegu16, pat_store>;
  defm u32 : PTX_ST<opstr, ".u32", RRegu32, pat_store>;
  defm u64 : PTX_ST<opstr, ".u64", RRegu64, pat_store>;
  defm f32 : PTX_ST<opstr, ".f32", RRegf32, pat_store>;
  defm f64 : PTX_ST<opstr, ".f64", RRegf64, pat_store>;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

///===- Floating-Point Arithmetic Instructions ----------------------------===//

defm FADD : FLOAT3<"add", fadd>;
defm FSUB : FLOAT3<"sub", fsub>;
defm FMUL : FLOAT3<"mul", fmul>;

///===- Integer Arithmetic Instructions -----------------------------------===//

defm ADD : INT3<"add", add>;
defm SUB : INT3<"sub", sub>;

///===- Logic and Shift Instructions --------------------------------------===//

defm SHL : INT3ntnc<"shl.b32", PTXshl>;
defm SRL : INT3ntnc<"shr.u32", PTXsrl>;
defm SRA : INT3ntnc<"shr.s32", PTXsra>;

///===- Data Movement and Conversion Instructions -------------------------===//

let neverHasSideEffects = 1 in {
  def MOVPREDrr
    : InstPTX<(outs Preds:$d), (ins Preds:$a), "mov.pred\t$d, $a", []>;
  def MOVU16rr
    : InstPTX<(outs RRegu16:$d), (ins RRegu16:$a), "mov.u16\t$d, $a", []>;
  def MOVU32rr
    : InstPTX<(outs RRegu32:$d), (ins RRegu32:$a), "mov.u32\t$d, $a", []>;
  def MOVU64rr
    : InstPTX<(outs RRegu64:$d), (ins RRegu64:$a), "mov.u64\t$d, $a", []>;
  def MOVF32rr
    : InstPTX<(outs RRegf32:$d), (ins RRegf32:$a), "mov.f32\t$d, $a", []>;
  def MOVF64rr
    : InstPTX<(outs RRegf64:$d), (ins RRegf64:$a), "mov.f64\t$d, $a", []>;
}

let isReMaterializable = 1, isAsCheapAsAMove = 1 in {
  def MOVPREDri
    : InstPTX<(outs Preds:$d), (ins i1imm:$a), "mov.pred\t$d, $a",
              [(set Preds:$d, imm:$a)]>;
  def MOVU16ri
    : InstPTX<(outs RRegu16:$d), (ins i16imm:$a), "mov.u16\t$d, $a",
              [(set RRegu16:$d, imm:$a)]>;
  def MOVU32ri
    : InstPTX<(outs RRegu32:$d), (ins i32imm:$a), "mov.u32\t$d, $a",
              [(set RRegu32:$d, imm:$a)]>;
  def MOVU164ri
    : InstPTX<(outs RRegu64:$d), (ins i64imm:$a), "mov.u64\t$d, $a",
              [(set RRegu64:$d, imm:$a)]>;
  def MOVF32ri
    : InstPTX<(outs RRegf32:$d), (ins f32imm:$a), "mov.f32\t$d, $a",
              [(set RRegf32:$d, fpimm:$a)]>;
  def MOVF64ri
    : InstPTX<(outs RRegf64:$d), (ins f64imm:$a), "mov.f64\t$d, $a",
              [(set RRegf64:$d, fpimm:$a)]>;
}

// Loads
defm LDg : PTX_LD_ALL<"ld.global", load_global>;
defm LDc : PTX_LD_ALL<"ld.const",  load_constant>;
defm LDl : PTX_LD_ALL<"ld.local",  load_local>;
defm LDs : PTX_LD_ALL<"ld.shared", load_shared>;

// This is a special instruction that is manually inserted for kernel parameters
def LDpiU16 : InstPTX<(outs RRegu16:$d), (ins MEMpi:$a),
                      "ld.param.u16\t$d, [$a]", []>;
def LDpiU32 : InstPTX<(outs RRegu32:$d), (ins MEMpi:$a),
                      "ld.param.u32\t$d, [$a]", []>;
def LDpiU64 : InstPTX<(outs RRegu64:$d), (ins MEMpi:$a),
                      "ld.param.u64\t$d, [$a]", []>;
def LDpiF32 : InstPTX<(outs RRegf32:$d), (ins MEMpi:$a),
                      "ld.param.f32\t$d, [$a]", []>;
def LDpiF64 : InstPTX<(outs RRegf64:$d), (ins MEMpi:$a),
                      "ld.param.f64\t$d, [$a]", []>;

// Stores
defm STg : PTX_ST_ALL<"st.global", store_global>;
defm STl : PTX_ST_ALL<"st.local",  store_local>;
defm STs : PTX_ST_ALL<"st.shared", store_shared>;

// defm STp : PTX_ST_ALL<"st.param",  store_parameter>;
// defm LDp : PTX_LD_ALL<"ld.param",  load_parameter>;
// TODO: Do something with st.param if/when it is needed.

///===- Control Flow Instructions -----------------------------------------===//

let isReturn = 1, isTerminator = 1, isBarrier = 1 in {
  def EXIT : InstPTX<(outs), (ins), "exit", [(PTXexit)]>;
  def RET  : InstPTX<(outs), (ins), "ret",  [(PTXret)]>;
}
