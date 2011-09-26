//===- PTXInstrLoadStore.td - PTX Load/Store Instruction Defs -*- tblgen-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the PTX load/store instructions in TableGen format.
//
//===----------------------------------------------------------------------===//


// Addressing Predicates
// We have to differentiate between 32- and 64-bit pointer types
def Use32BitAddresses : Predicate<"!getSubtarget().is64Bit()">;
def Use64BitAddresses : Predicate<"getSubtarget().is64Bit()">;

//===----------------------------------------------------------------------===//
// Pattern Fragments for Loads/Stores
//===----------------------------------------------------------------------===//

def load_global : PatFrag<(ops node:$ptr), (load node:$ptr), [{
  const Value *Src;
  const PointerType *PT;
  const SDValue &MemOp = N->getOperand(1);
  if ((MemOp.getOpcode() != ISD::FrameIndex) &&
      (Src = cast<LoadSDNode>(N)->getSrcValue()) &&
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
  const SDValue &MemOp = N->getOperand(1);
  return MemOp.getOpcode() == ISD::FrameIndex;
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
  const SDValue &MemOp = N->getOperand(2);
  if ((MemOp.getOpcode() != ISD::FrameIndex) &&
      (Src = cast<StoreSDNode>(N)->getSrcValue()) &&
      (PT = dyn_cast<PointerType>(Src->getType())))
    return PT->getAddressSpace() == PTX::GLOBAL;
  return false;
}]>;

def store_local
  : PatFrag<(ops node:$d, node:$ptr), (store node:$d, node:$ptr), [{
  const SDValue &MemOp = N->getOperand(2);
  return MemOp.getOpcode() == ISD::FrameIndex;
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
def ADDRrr32    : ComplexPattern<i32, 2, "SelectADDRrr", [], []>;
def ADDRrr64    : ComplexPattern<i64, 2, "SelectADDRrr", [], []>;
def ADDRri32    : ComplexPattern<i32, 2, "SelectADDRri", [], []>;
def ADDRri64    : ComplexPattern<i64, 2, "SelectADDRri", [], []>;
def ADDRii32    : ComplexPattern<i32, 2, "SelectADDRii", [], []>;
def ADDRii64    : ComplexPattern<i64, 2, "SelectADDRii", [], []>;
def ADDRlocal32 : ComplexPattern<i32, 2, "SelectADDRlocal", [], []>;
def ADDRlocal64 : ComplexPattern<i64, 2, "SelectADDRlocal", [], []>;

// Address operands
def MEMri32 : Operand<i32> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops RegI32, i32imm);
}
def MEMri64 : Operand<i64> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops RegI64, i64imm);
}
def LOCALri32 : Operand<i32> {
  let PrintMethod = "printLocalOperand";
  let MIOperandInfo = (ops RegI32, i32imm);
}
def LOCALri64 : Operand<i64> {
  let PrintMethod = "printLocalOperand";
  let MIOperandInfo = (ops RegI64, i64imm);
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
def MEMret : Operand<i32> {
  let PrintMethod = "printReturnOperand";
  let MIOperandInfo = (ops i32imm);
}


// Load/store .param space
def PTXloadparam
  : SDNode<"PTXISD::LOAD_PARAM", SDTypeProfile<1, 1, [SDTCisVT<1, i32>]>,
           [SDNPHasChain, SDNPOutGlue, SDNPOptInGlue]>;
def PTXstoreparam
  : SDNode<"PTXISD::STORE_PARAM", SDTypeProfile<0, 2, [SDTCisVT<0, i32>]>,
           [SDNPHasChain, SDNPOutGlue, SDNPOptInGlue]>;

def PTXreadparam
  : SDNode<"PTXISD::READ_PARAM", SDTypeProfile<1, 1, [SDTCisVT<1, i32>]>,
      [SDNPHasChain, SDNPOutGlue, SDNPOptInGlue]>;
def PTXwriteparam
  : SDNode<"PTXISD::WRITE_PARAM", SDTypeProfile<0, 1, []>,
      [SDNPHasChain, SDNPOutGlue, SDNPOptInGlue]>;



//===----------------------------------------------------------------------===//
// Classes for loads/stores
//===----------------------------------------------------------------------===//
multiclass PTX_LD<string opstr, string typestr,
           RegisterClass RC, PatFrag pat_load> {
  def rr32 : InstPTX<(outs RC:$d),
                     (ins MEMri32:$a),
                     !strconcat(opstr, !strconcat(typestr, "\t$d, [$a]")),
                     [(set RC:$d, (pat_load ADDRrr32:$a))]>,
                     Requires<[Use32BitAddresses]>;
  def rr64 : InstPTX<(outs RC:$d),
                     (ins MEMri64:$a),
                     !strconcat(opstr, !strconcat(typestr, "\t$d, [$a]")),
                     [(set RC:$d, (pat_load ADDRrr64:$a))]>,
                     Requires<[Use64BitAddresses]>;
  def ri32 : InstPTX<(outs RC:$d),
                     (ins MEMri32:$a),
                     !strconcat(opstr, !strconcat(typestr, "\t$d, [$a]")),
                     [(set RC:$d, (pat_load ADDRri32:$a))]>,
                     Requires<[Use32BitAddresses]>;
  def ri64 : InstPTX<(outs RC:$d),
                     (ins MEMri64:$a),
                     !strconcat(opstr, !strconcat(typestr, "\t$d, [$a]")),
                     [(set RC:$d, (pat_load ADDRri64:$a))]>,
                     Requires<[Use64BitAddresses]>;
  def ii32 : InstPTX<(outs RC:$d),
                     (ins MEMii32:$a),
                     !strconcat(opstr, !strconcat(typestr, "\t$d, [$a]")),
                     [(set RC:$d, (pat_load ADDRii32:$a))]>,
                     Requires<[Use32BitAddresses]>;
  def ii64 : InstPTX<(outs RC:$d),
                     (ins MEMii64:$a),
                     !strconcat(opstr, !strconcat(typestr, "\t$d, [$a]")),
                     [(set RC:$d, (pat_load ADDRii64:$a))]>,
                     Requires<[Use64BitAddresses]>;
}

multiclass PTX_ST<string opstr, string typestr, RegisterClass RC,
                  PatFrag pat_store> {
  def rr32 : InstPTX<(outs),
                     (ins RC:$d, MEMri32:$a),
                     !strconcat(opstr, !strconcat(typestr, "\t[$a], $d")),
                     [(pat_store RC:$d, ADDRrr32:$a)]>,
                     Requires<[Use32BitAddresses]>;
  def rr64 : InstPTX<(outs),
                     (ins RC:$d, MEMri64:$a),
                     !strconcat(opstr, !strconcat(typestr, "\t[$a], $d")),
                     [(pat_store RC:$d, ADDRrr64:$a)]>,
                     Requires<[Use64BitAddresses]>;
  def ri32 : InstPTX<(outs),
                   (ins RC:$d, MEMri32:$a),
                   !strconcat(opstr, !strconcat(typestr, "\t[$a], $d")),
                   [(pat_store RC:$d, ADDRri32:$a)]>,
                   Requires<[Use32BitAddresses]>;
  def ri64 : InstPTX<(outs),
                   (ins RC:$d, MEMri64:$a),
                   !strconcat(opstr, !strconcat(typestr, "\t[$a], $d")),
                   [(pat_store RC:$d, ADDRri64:$a)]>,
                   Requires<[Use64BitAddresses]>;
  def ii32 : InstPTX<(outs),
                   (ins RC:$d, MEMii32:$a),
                   !strconcat(opstr, !strconcat(typestr, "\t[$a], $d")),
                   [(pat_store RC:$d, ADDRii32:$a)]>,
                   Requires<[Use32BitAddresses]>;
  def ii64 : InstPTX<(outs),
                   (ins RC:$d, MEMii64:$a),
                   !strconcat(opstr, !strconcat(typestr, "\t[$a], $d")),
                   [(pat_store RC:$d, ADDRii64:$a)]>,
                   Requires<[Use64BitAddresses]>;
}

multiclass PTX_LOCAL_LD_ST<string typestr, RegisterClass RC> {
  def LDri32 : InstPTX<(outs RC:$d), (ins LOCALri32:$a),
                      !strconcat("ld.local", !strconcat(typestr, "\t$d, [$a]")),
                       [(set RC:$d, (load_local ADDRlocal32:$a))]>;
  def LDri64 : InstPTX<(outs RC:$d), (ins LOCALri64:$a),
                      !strconcat("ld.local", !strconcat(typestr, "\t$d, [$a]")),
                       [(set RC:$d, (load_local ADDRlocal64:$a))]>;
  def STri32 : InstPTX<(outs), (ins RC:$d, LOCALri32:$a),
                      !strconcat("st.local", !strconcat(typestr, "\t[$a], $d")),
                       [(store_local RC:$d, ADDRlocal32:$a)]>;
  def STri64 : InstPTX<(outs), (ins RC:$d, LOCALri64:$a),
                      !strconcat("st.local", !strconcat(typestr, "\t[$a], $d")),
                       [(store_local RC:$d, ADDRlocal64:$a)]>;
}

multiclass PTX_PARAM_LD_ST<string typestr, RegisterClass RC> {
  let hasSideEffects = 1 in {
  def LDpi : InstPTX<(outs RC:$d), (ins MEMpi:$a),
                     !strconcat("ld.param", !strconcat(typestr, "\t$d, [$a]")),
                     [(set RC:$d, (PTXloadparam timm:$a))]>;
  def STpi : InstPTX<(outs), (ins MEMpi:$d, RC:$a),
                     !strconcat("st.param", !strconcat(typestr, "\t[$d], $a")),
                     [(PTXstoreparam timm:$d, RC:$a)]>;
  }
}

multiclass PTX_LD_ALL<string opstr, PatFrag pat_load> {
  defm u16 : PTX_LD<opstr, ".u16", RegI16, pat_load>;
  defm u32 : PTX_LD<opstr, ".u32", RegI32, pat_load>;
  defm u64 : PTX_LD<opstr, ".u64", RegI64, pat_load>;
  defm f32 : PTX_LD<opstr, ".f32", RegF32, pat_load>;
  defm f64 : PTX_LD<opstr, ".f64", RegF64, pat_load>;
}

multiclass PTX_ST_ALL<string opstr, PatFrag pat_store> {
  defm u16 : PTX_ST<opstr, ".u16", RegI16, pat_store>;
  defm u32 : PTX_ST<opstr, ".u32", RegI32, pat_store>;
  defm u64 : PTX_ST<opstr, ".u64", RegI64, pat_store>;
  defm f32 : PTX_ST<opstr, ".f32", RegF32, pat_store>;
  defm f64 : PTX_ST<opstr, ".f64", RegF64, pat_store>;
}



//===----------------------------------------------------------------------===//
// Instruction definitions for loads/stores
//===----------------------------------------------------------------------===//

// Global/shared stores
defm STg : PTX_ST_ALL<"st.global", store_global>;
defm STs : PTX_ST_ALL<"st.shared", store_shared>;

// Global/shared/constant loads
defm LDg : PTX_LD_ALL<"ld.global", load_global>;
defm LDc : PTX_LD_ALL<"ld.const",  load_constant>;
defm LDs : PTX_LD_ALL<"ld.shared", load_shared>;

// Param loads/stores
defm PARAMPRED : PTX_PARAM_LD_ST<".pred", RegPred>;
defm PARAMU16  : PTX_PARAM_LD_ST<".u16", RegI16>;
defm PARAMU32  : PTX_PARAM_LD_ST<".u32", RegI32>;
defm PARAMU64  : PTX_PARAM_LD_ST<".u64", RegI64>;
defm PARAMF32  : PTX_PARAM_LD_ST<".f32", RegF32>;
defm PARAMF64  : PTX_PARAM_LD_ST<".f64", RegF64>;

// Local loads/stores
defm LOCALPRED : PTX_LOCAL_LD_ST<".pred", RegPred>;
defm LOCALU16  : PTX_LOCAL_LD_ST<".u16", RegI16>;
defm LOCALU32  : PTX_LOCAL_LD_ST<".u32", RegI32>;
defm LOCALU64  : PTX_LOCAL_LD_ST<".u64", RegI64>;
defm LOCALF32  : PTX_LOCAL_LD_ST<".f32", RegF32>;
defm LOCALF64  : PTX_LOCAL_LD_ST<".f64", RegF64>;

