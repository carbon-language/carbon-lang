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

// Addressing
def Use32BitAddresses : Predicate<"!getSubtarget().use64BitAddresses()">;
def Use64BitAddresses : Predicate<"getSubtarget().use64BitAddresses()">;

// Shader Model Support
def SupportsSM13       : Predicate<"getSubtarget().supportsSM13()">;
def DoesNotSupportSM13 : Predicate<"!getSubtarget().supportsSM13()">;
def SupportsSM20       : Predicate<"getSubtarget().supportsSM20()">;
def DoesNotSupportSM20 : Predicate<"!getSubtarget().supportsSM20()">;

// PTX Version Support
def SupportsPTX21       : Predicate<"getSubtarget().supportsPTX21()">;
def DoesNotSupportPTX21 : Predicate<"!getSubtarget().supportsPTX21()">;
def SupportsPTX22       : Predicate<"getSubtarget().supportsPTX22()">;
def DoesNotSupportPTX22 : Predicate<"!getSubtarget().supportsPTX22()">;

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

// Branch & call targets have OtherVT type.
def brtarget   : Operand<OtherVT>;
def calltarget : Operand<i32>;

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
def PTXcopyaddress
  : SDNode<"PTXISD::COPY_ADDRESS", SDTypeProfile<1, 1, []>, []>;

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//

//===- Floating-Point Instructions - 3 Operand Form -----------------------===//
multiclass PTX_FLOAT_3OP<string opcstr, SDNode opnode> {
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

//===- Floating-Point Instructions - 4 Operand Form -----------------------===//
multiclass PTX_FLOAT_4OP<string opcstr, SDNode opnode1, SDNode opnode2> {
  def rrr32 : InstPTX<(outs RRegf32:$d),
                      (ins RRegf32:$a, RRegf32:$b, RRegf32:$c),
                      !strconcat(opcstr, ".f32\t$d, $a, $b, $c"),
                      [(set RRegf32:$d, (opnode2 (opnode1 RRegf32:$a,
                                                          RRegf32:$b),
                                                 RRegf32:$c))]>;
  def rri32 : InstPTX<(outs RRegf32:$d),
                      (ins RRegf32:$a, RRegf32:$b, f32imm:$c),
                      !strconcat(opcstr, ".f32\t$d, $a, $b, $c"),
                      [(set RRegf32:$d, (opnode2 (opnode1 RRegf32:$a,
                                                          RRegf32:$b),
                                                 fpimm:$c))]>;
  def rrr64 : InstPTX<(outs RRegf64:$d),
                      (ins RRegf64:$a, RRegf64:$b, RRegf64:$c),
                      !strconcat(opcstr, ".f64\t$d, $a, $b, $c"),
                      [(set RRegf64:$d, (opnode2 (opnode1 RRegf64:$a,
                                                          RRegf64:$b),
                                                 RRegf64:$c))]>;
  def rri64 : InstPTX<(outs RRegf64:$d),
                      (ins RRegf64:$a, RRegf64:$b, f64imm:$c),
                      !strconcat(opcstr, ".f64\t$d, $a, $b, $c"),
                      [(set RRegf64:$d, (opnode2 (opnode1 RRegf64:$a,
                                                          RRegf64:$b),
                                                 fpimm:$c))]>;
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

multiclass PTX_LOGIC<string opcstr, SDNode opnode> {
  def rr16 : InstPTX<(outs RRegu16:$d),
                     (ins RRegu16:$a, RRegu16:$b),
                     !strconcat(opcstr, ".b16\t$d, $a, $b"),
                     [(set RRegu16:$d, (opnode RRegu16:$a, RRegu16:$b))]>;
  def ri16 : InstPTX<(outs RRegu16:$d),
                     (ins RRegu16:$a, i16imm:$b),
                     !strconcat(opcstr, ".b16\t$d, $a, $b"),
                     [(set RRegu16:$d, (opnode RRegu16:$a, imm:$b))]>;
  def rr32 : InstPTX<(outs RRegu32:$d),
                     (ins RRegu32:$a, RRegu32:$b),
                     !strconcat(opcstr, ".b32\t$d, $a, $b"),
                     [(set RRegu32:$d, (opnode RRegu32:$a, RRegu32:$b))]>;
  def ri32 : InstPTX<(outs RRegu32:$d),
                     (ins RRegu32:$a, i32imm:$b),
                     !strconcat(opcstr, ".b32\t$d, $a, $b"),
                     [(set RRegu32:$d, (opnode RRegu32:$a, imm:$b))]>;
  def rr64 : InstPTX<(outs RRegu64:$d),
                     (ins RRegu64:$a, RRegu64:$b),
                     !strconcat(opcstr, ".b64\t$d, $a, $b"),
                     [(set RRegu64:$d, (opnode RRegu64:$a, RRegu64:$b))]>;
  def ri64 : InstPTX<(outs RRegu64:$d),
                     (ins RRegu64:$a, i64imm:$b),
                     !strconcat(opcstr, ".b64\t$d, $a, $b"),
                     [(set RRegu64:$d, (opnode RRegu64:$a, imm:$b))]>;
}

multiclass INT3ntnc<string opcstr, SDNode opnode> {
  def rr16 : InstPTX<(outs RRegu16:$d),
                     (ins RRegu16:$a, RRegu16:$b),
                     !strconcat(opcstr, "16\t$d, $a, $b"),
                     [(set RRegu16:$d, (opnode RRegu16:$a, RRegu16:$b))]>;
  def rr32 : InstPTX<(outs RRegu32:$d),
                     (ins RRegu32:$a, RRegu32:$b),
                     !strconcat(opcstr, "32\t$d, $a, $b"),
                     [(set RRegu32:$d, (opnode RRegu32:$a, RRegu32:$b))]>;
  def rr64 : InstPTX<(outs RRegu64:$d),
                     (ins RRegu64:$a, RRegu64:$b),
                     !strconcat(opcstr, "64\t$d, $a, $b"),
                     [(set RRegu64:$d, (opnode RRegu64:$a, RRegu64:$b))]>;
  def ri16 : InstPTX<(outs RRegu16:$d),
                     (ins RRegu16:$a, i16imm:$b),
                     !strconcat(opcstr, "16\t$d, $a, $b"),
                     [(set RRegu16:$d, (opnode RRegu16:$a, imm:$b))]>;
  def ri32 : InstPTX<(outs RRegu32:$d),
                     (ins RRegu32:$a, i32imm:$b),
                     !strconcat(opcstr, "32\t$d, $a, $b"),
                     [(set RRegu32:$d, (opnode RRegu32:$a, imm:$b))]>;
  def ri64 : InstPTX<(outs RRegu64:$d),
                     (ins RRegu64:$a, i64imm:$b),
                     !strconcat(opcstr, "64\t$d, $a, $b"),
                     [(set RRegu64:$d, (opnode RRegu64:$a, imm:$b))]>;
  def ir16 : InstPTX<(outs RRegu16:$d),
                     (ins i16imm:$a, RRegu16:$b),
                     !strconcat(opcstr, "16\t$d, $a, $b"),
                     [(set RRegu16:$d, (opnode imm:$a, RRegu16:$b))]>;
  def ir32 : InstPTX<(outs RRegu32:$d),
                     (ins i32imm:$a, RRegu32:$b),
                     !strconcat(opcstr, "32\t$d, $a, $b"),
                     [(set RRegu32:$d, (opnode imm:$a, RRegu32:$b))]>;
  def ir64 : InstPTX<(outs RRegu64:$d),
                     (ins i64imm:$a, RRegu64:$b),
                     !strconcat(opcstr, "64\t$d, $a, $b"),
                     [(set RRegu64:$d, (opnode imm:$a, RRegu64:$b))]>;
}

multiclass PTX_SETP_I<RegisterClass RC, string regclsname, Operand immcls,
                        CondCode cmp, string cmpstr> {
  // TODO support 5-operand format: p|q, a, b, c

  def rr
    : InstPTX<(outs Preds:$p), (ins RC:$a, RC:$b),
              !strconcat("setp.", cmpstr, ".", regclsname, "\t$p, $a, $b"),
              [(set Preds:$p, (setcc RC:$a, RC:$b, cmp))]>;
  def ri
    : InstPTX<(outs Preds:$p), (ins RC:$a, immcls:$b),
              !strconcat("setp.", cmpstr, ".", regclsname, "\t$p, $a, $b"),
              [(set Preds:$p, (setcc RC:$a, imm:$b, cmp))]>;

  def rr_and_r
    : InstPTX<(outs Preds:$p), (ins RC:$a, RC:$b, Preds:$c),
              !strconcat("setp.", cmpstr, ".and.", regclsname, "\t$p, $a, $b, $c"),
              [(set Preds:$p, (and (setcc RC:$a, RC:$b, cmp), Preds:$c))]>;
  def ri_and_r
    : InstPTX<(outs Preds:$p), (ins RC:$a, immcls:$b, Preds:$c),
              !strconcat("setp.", cmpstr, ".and.", regclsname, "\t$p, $a, $b, $c"),
              [(set Preds:$p, (and (setcc RC:$a, imm:$b, cmp), Preds:$c))]>;
  def rr_or_r
    : InstPTX<(outs Preds:$p), (ins RC:$a, RC:$b, Preds:$c),
              !strconcat("setp.", cmpstr, ".or.", regclsname, "\t$p, $a, $b, $c"),
              [(set Preds:$p, (or (setcc RC:$a, RC:$b, cmp), Preds:$c))]>;
  def ri_or_r
    : InstPTX<(outs Preds:$p), (ins RC:$a, immcls:$b, Preds:$c),
              !strconcat("setp.", cmpstr, ".or.", regclsname, "\t$p, $a, $b, $c"),
              [(set Preds:$p, (or (setcc RC:$a, imm:$b, cmp), Preds:$c))]>;
  def rr_xor_r
    : InstPTX<(outs Preds:$p), (ins RC:$a, RC:$b, Preds:$c),
              !strconcat("setp.", cmpstr, ".xor.", regclsname, "\t$p, $a, $b, $c"),
              [(set Preds:$p, (xor (setcc RC:$a, RC:$b, cmp), Preds:$c))]>;
  def ri_xor_r
    : InstPTX<(outs Preds:$p), (ins RC:$a, immcls:$b, Preds:$c),
              !strconcat("setp.", cmpstr, ".xor.", regclsname, "\t$p, $a, $b, $c"),
              [(set Preds:$p, (xor (setcc RC:$a, imm:$b, cmp), Preds:$c))]>;

  def rr_and_not_r
    : InstPTX<(outs Preds:$p), (ins RC:$a, RC:$b, Preds:$c),
              !strconcat("setp.", cmpstr, ".and.", regclsname, "\t$p, $a, $b, !$c"),
              [(set Preds:$p, (and (setcc RC:$a, RC:$b, cmp), (not Preds:$c)))]>;
  def ri_and_not_r
    : InstPTX<(outs Preds:$p), (ins RC:$a, immcls:$b, Preds:$c),
              !strconcat("setp.", cmpstr, ".and.", regclsname, "\t$p, $a, $b, !$c"),
              [(set Preds:$p, (and (setcc RC:$a, imm:$b, cmp), (not Preds:$c)))]>;
  def rr_or_not_r
    : InstPTX<(outs Preds:$p), (ins RC:$a, RC:$b, Preds:$c),
              !strconcat("setp.", cmpstr, ".or.", regclsname, "\t$p, $a, $b, !$c"),
              [(set Preds:$p, (or (setcc RC:$a, RC:$b, cmp), (not Preds:$c)))]>;
  def ri_or_not_r
    : InstPTX<(outs Preds:$p), (ins RC:$a, immcls:$b, Preds:$c),
              !strconcat("setp.", cmpstr, ".or.", regclsname, "\t$p, $a, $b, !$c"),
              [(set Preds:$p, (or (setcc RC:$a, imm:$b, cmp), (not Preds:$c)))]>;
  def rr_xor_not_r
    : InstPTX<(outs Preds:$p), (ins RC:$a, RC:$b, Preds:$c),
              !strconcat("setp.", cmpstr, ".xor.", regclsname, "\t$p, $a, $b, !$c"),
              [(set Preds:$p, (xor (setcc RC:$a, RC:$b, cmp), (not Preds:$c)))]>;
  def ri_xor_not_r
    : InstPTX<(outs Preds:$p), (ins RC:$a, immcls:$b, Preds:$c),
              !strconcat("setp.", cmpstr, ".xor.", regclsname, "\t$p, $a, $b, !$c"),
              [(set Preds:$p, (xor (setcc RC:$a, imm:$b, cmp), (not Preds:$c)))]>;
}

multiclass PTX_SETP_FP<RegisterClass RC, string regclsname,
                        CondCode ucmp, CondCode ocmp, string cmpstr> {
  // TODO support 5-operand format: p|q, a, b, c

  def rr_u
    : InstPTX<(outs Preds:$p), (ins RC:$a, RC:$b),
              !strconcat("setp.", cmpstr, "u.", regclsname, "\t$p, $a, $b"),
              [(set Preds:$p, (setcc RC:$a, RC:$b, ucmp))]>;
  def rr_o
    : InstPTX<(outs Preds:$p), (ins RC:$a, RC:$b),
              !strconcat("setp.", cmpstr, ".", regclsname, "\t$p, $a, $b"),
              [(set Preds:$p, (setcc RC:$a, RC:$b, ocmp))]>;

  def rr_and_r_u
    : InstPTX<(outs Preds:$p), (ins RC:$a, RC:$b, Preds:$c),
              !strconcat("setp.", cmpstr, "u.and.", regclsname, "\t$p, $a, $b, $c"),
              [(set Preds:$p, (and (setcc RC:$a, RC:$b, ucmp), Preds:$c))]>;
  def rr_and_r_o
    : InstPTX<(outs Preds:$p), (ins RC:$a, RC:$b, Preds:$c),
              !strconcat("setp.", cmpstr, ".and.", regclsname, "\t$p, $a, $b, $c"),
              [(set Preds:$p, (and (setcc RC:$a, RC:$b, ocmp), Preds:$c))]>;

  def rr_or_r_u
    : InstPTX<(outs Preds:$p), (ins RC:$a, RC:$b, Preds:$c),
              !strconcat("setp.", cmpstr, "u.or.", regclsname, "\t$p, $a, $b, $c"),
              [(set Preds:$p, (or (setcc RC:$a, RC:$b, ucmp), Preds:$c))]>;
  def rr_or_r_o
    : InstPTX<(outs Preds:$p), (ins RC:$a, RC:$b, Preds:$c),
              !strconcat("setp.", cmpstr, ".or.", regclsname, "\t$p, $a, $b, $c"),
              [(set Preds:$p, (or (setcc RC:$a, RC:$b, ocmp), Preds:$c))]>;

  def rr_xor_r_u
    : InstPTX<(outs Preds:$p), (ins RC:$a, RC:$b, Preds:$c),
              !strconcat("setp.", cmpstr, "u.xor.", regclsname, "\t$p, $a, $b, $c"),
              [(set Preds:$p, (xor (setcc RC:$a, RC:$b, ucmp), Preds:$c))]>;
  def rr_xor_r_o
    : InstPTX<(outs Preds:$p), (ins RC:$a, RC:$b, Preds:$c),
              !strconcat("setp.", cmpstr, ".xor.", regclsname, "\t$p, $a, $b, $c"),
              [(set Preds:$p, (xor (setcc RC:$a, RC:$b, ocmp), Preds:$c))]>;

  def rr_and_not_r_u
    : InstPTX<(outs Preds:$p), (ins RC:$a, RC:$b, Preds:$c),
              !strconcat("setp.", cmpstr, "u.and.", regclsname, "\t$p, $a, $b, !$c"),
              [(set Preds:$p, (and (setcc RC:$a, RC:$b, ucmp), (not Preds:$c)))]>;
  def rr_and_not_r_o
    : InstPTX<(outs Preds:$p), (ins RC:$a, RC:$b, Preds:$c),
              !strconcat("setp.", cmpstr, ".and.", regclsname, "\t$p, $a, $b, !$c"),
              [(set Preds:$p, (and (setcc RC:$a, RC:$b, ocmp), (not Preds:$c)))]>;

  def rr_or_not_r_u
    : InstPTX<(outs Preds:$p), (ins RC:$a, RC:$b, Preds:$c),
              !strconcat("setp.", cmpstr, "u.or.", regclsname, "\t$p, $a, $b, !$c"),
              [(set Preds:$p, (or (setcc RC:$a, RC:$b, ucmp), (not Preds:$c)))]>;
  def rr_or_not_r_o
    : InstPTX<(outs Preds:$p), (ins RC:$a, RC:$b, Preds:$c),
              !strconcat("setp.", cmpstr, ".or.", regclsname, "\t$p, $a, $b, !$c"),
              [(set Preds:$p, (or (setcc RC:$a, RC:$b, ocmp), (not Preds:$c)))]>;

  def rr_xor_not_r_u
    : InstPTX<(outs Preds:$p), (ins RC:$a, RC:$b, Preds:$c),
              !strconcat("setp.", cmpstr, "u.xor.", regclsname, "\t$p, $a, $b, !$c"),
              [(set Preds:$p, (xor (setcc RC:$a, RC:$b, ucmp), (not Preds:$c)))]>;
  def rr_xor_not_r_o
    : InstPTX<(outs Preds:$p), (ins RC:$a, RC:$b, Preds:$c),
              !strconcat("setp.", cmpstr, ".xor.", regclsname, "\t$p, $a, $b, !$c"),
              [(set Preds:$p, (xor (setcc RC:$a, RC:$b, ocmp), (not Preds:$c)))]>;
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

///===- Integer Arithmetic Instructions -----------------------------------===//

defm ADD : INT3<"add", add>;
defm SUB : INT3<"sub", sub>;
defm MUL : INT3<"mul.lo", mul>; // FIXME: Allow 32x32 -> 64 multiplies
defm DIV : INT3<"div", udiv>;
defm REM : INT3<"rem", urem>;

///===- Floating-Point Arithmetic Instructions ----------------------------===//

// Standard Binary Operations
defm FADD : PTX_FLOAT_3OP<"add", fadd>;
defm FSUB : PTX_FLOAT_3OP<"sub", fsub>;
defm FMUL : PTX_FLOAT_3OP<"mul", fmul>;

// TODO: Allow user selection of rounding modes for fdiv.
// For division, we need to have f32 and f64 differently.
// For f32, we just always use .approx since it is supported on all hardware
// for PTX 1.4+, which is our minimum target.
def FDIVrr32 : InstPTX<(outs RRegf32:$d),
                       (ins RRegf32:$a, RRegf32:$b),
                       "div.approx.f32\t$d, $a, $b",
                       [(set RRegf32:$d, (fdiv RRegf32:$a, RRegf32:$b))]>;
def FDIVri32 : InstPTX<(outs RRegf32:$d),
                       (ins RRegf32:$a, f32imm:$b),
                       "div.approx.f32\t$d, $a, $b",
                       [(set RRegf32:$d, (fdiv RRegf32:$a, fpimm:$b))]>;

// For f64, we must specify a rounding for sm 1.3+ but *not* for sm 1.0.
def FDIVrr64SM13 : InstPTX<(outs RRegf64:$d),
                           (ins RRegf64:$a, RRegf64:$b),
                           "div.rn.f64\t$d, $a, $b",
                           [(set RRegf64:$d, (fdiv RRegf64:$a, RRegf64:$b))]>,
                   Requires<[SupportsSM13]>;
def FDIVri64SM13 : InstPTX<(outs RRegf64:$d),
                           (ins RRegf64:$a, f64imm:$b),
                           "div.rn.f64\t$d, $a, $b",
                           [(set RRegf64:$d, (fdiv RRegf64:$a, fpimm:$b))]>,
                   Requires<[SupportsSM13]>;
def FDIVrr64SM10 : InstPTX<(outs RRegf64:$d),
                           (ins RRegf64:$a, RRegf64:$b),
                           "div.f64\t$d, $a, $b",
                           [(set RRegf64:$d, (fdiv RRegf64:$a, RRegf64:$b))]>,
                   Requires<[DoesNotSupportSM13]>;
def FDIVri64SM10 : InstPTX<(outs RRegf64:$d),
                           (ins RRegf64:$a, f64imm:$b),
                           "div.f64\t$d, $a, $b",
                           [(set RRegf64:$d, (fdiv RRegf64:$a, fpimm:$b))]>,
                   Requires<[DoesNotSupportSM13]>;



// Multi-operation hybrid instructions

// The selection of mad/fma is tricky.  In some cases, they are the *same*
// instruction, but in other cases we may prefer one or the other.  Also,
// different PTX versions differ on whether rounding mode flags are required.
// In the short term, mad is supported on all PTX versions and we use a
// default rounding mode no matter what shader model or PTX version.
// TODO: Allow the rounding mode to be selectable through llc.
defm FMADSM13 : PTX_FLOAT_4OP<"mad.rn", fmul, fadd>, Requires<[SupportsSM13]>;
defm FMAD : PTX_FLOAT_4OP<"mad", fmul, fadd>, Requires<[DoesNotSupportSM13]>;

///===- Floating-Point Intrinsic Instructions -----------------------------===//

def FSQRT32 : InstPTX<(outs RRegf32:$d),
                      (ins RRegf32:$a),
                      "sqrt.rn.f32\t$d, $a",
                      [(set RRegf32:$d, (fsqrt RRegf32:$a))]>;

def FSQRT64 : InstPTX<(outs RRegf64:$d),
                      (ins RRegf64:$a),
                      "sqrt.rn.f64\t$d, $a",
                      [(set RRegf64:$d, (fsqrt RRegf64:$a))]>;

def FSIN32 : InstPTX<(outs RRegf32:$d),
                     (ins RRegf32:$a),
                     "sin.approx.f32\t$d, $a",
                     [(set RRegf32:$d, (fsin RRegf32:$a))]>;

def FSIN64 : InstPTX<(outs RRegf64:$d),
                     (ins RRegf64:$a),
                     "sin.approx.f64\t$d, $a",
                     [(set RRegf64:$d, (fsin RRegf64:$a))]>;

def FCOS32 : InstPTX<(outs RRegf32:$d),
                     (ins RRegf32:$a),
                     "cos.approx.f32\t$d, $a",
                     [(set RRegf32:$d, (fcos RRegf32:$a))]>;

def FCOS64 : InstPTX<(outs RRegf64:$d),
                     (ins RRegf64:$a),
                     "cos.approx.f64\t$d, $a",
                     [(set RRegf64:$d, (fcos RRegf64:$a))]>;


///===- Comparison and Selection Instructions -----------------------------===//

// Compare u16

defm SETPEQu16 : PTX_SETP_I<RRegu16, "u16", i16imm, SETEQ,  "eq">;
defm SETPNEu16 : PTX_SETP_I<RRegu16, "u16", i16imm, SETNE,  "ne">;
defm SETPLTu16 : PTX_SETP_I<RRegu16, "u16", i16imm, SETULT, "lt">;
defm SETPLEu16 : PTX_SETP_I<RRegu16, "u16", i16imm, SETULE, "le">;
defm SETPGTu16 : PTX_SETP_I<RRegu16, "u16", i16imm, SETUGT, "gt">;
defm SETPGEu16 : PTX_SETP_I<RRegu16, "u16", i16imm, SETUGE, "ge">;

// Compare u32

defm SETPEQu32 : PTX_SETP_I<RRegu32, "u32", i32imm, SETEQ,  "eq">;
defm SETPNEu32 : PTX_SETP_I<RRegu32, "u32", i32imm, SETNE,  "ne">;
defm SETPLTu32 : PTX_SETP_I<RRegu32, "u32", i32imm, SETULT, "lt">;
defm SETPLEu32 : PTX_SETP_I<RRegu32, "u32", i32imm, SETULE, "le">;
defm SETPGTu32 : PTX_SETP_I<RRegu32, "u32", i32imm, SETUGT, "gt">;
defm SETPGEu32 : PTX_SETP_I<RRegu32, "u32", i32imm, SETUGE, "ge">;

// Compare u64

defm SETPEQu64 : PTX_SETP_I<RRegu64, "u64", i64imm, SETEQ,  "eq">;
defm SETPNEu64 : PTX_SETP_I<RRegu64, "u64", i64imm, SETNE,  "ne">;
defm SETPLTu64 : PTX_SETP_I<RRegu64, "u64", i64imm, SETULT, "lt">;
defm SETPLEu64 : PTX_SETP_I<RRegu64, "u64", i64imm, SETULE, "le">;
defm SETPGTu64 : PTX_SETP_I<RRegu64, "u64", i64imm, SETUGT, "gt">;
defm SETPGEu64 : PTX_SETP_I<RRegu64, "u64", i64imm, SETUGE, "ge">;

// Compare f32

defm SETPEQf32 : PTX_SETP_FP<RRegf32, "f32", SETUEQ, SETOEQ, "eq">;
defm SETPNEf32 : PTX_SETP_FP<RRegf32, "f32", SETUNE, SETONE, "ne">;
defm SETPLTf32 : PTX_SETP_FP<RRegf32, "f32", SETULT, SETOLT, "lt">;
defm SETPLEf32 : PTX_SETP_FP<RRegf32, "f32", SETULE, SETOLE, "le">;
defm SETPGTf32 : PTX_SETP_FP<RRegf32, "f32", SETUGT, SETOGT, "gt">;
defm SETPGEf32 : PTX_SETP_FP<RRegf32, "f32", SETUGE, SETOGE, "ge">;

// Compare f64

defm SETPEQf64 : PTX_SETP_FP<RRegf64, "f64", SETUEQ, SETOEQ, "eq">;
defm SETPNEf64 : PTX_SETP_FP<RRegf64, "f64", SETUNE, SETONE, "ne">;
defm SETPLTf64 : PTX_SETP_FP<RRegf64, "f64", SETULT, SETOLT, "lt">;
defm SETPLEf64 : PTX_SETP_FP<RRegf64, "f64", SETULE, SETOLE, "le">;
defm SETPGTf64 : PTX_SETP_FP<RRegf64, "f64", SETUGT, SETOGT, "gt">;
defm SETPGEf64 : PTX_SETP_FP<RRegf64, "f64", SETUGE, SETOGE, "ge">;

///===- Logic and Shift Instructions --------------------------------------===//

defm SHL : INT3ntnc<"shl.b", PTXshl>;
defm SRL : INT3ntnc<"shr.u", PTXsrl>;
defm SRA : INT3ntnc<"shr.s", PTXsra>;

defm AND : PTX_LOGIC<"and", and>;
defm OR  : PTX_LOGIC<"or",  or>;
defm XOR : PTX_LOGIC<"xor", xor>;

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

let isReMaterializable = 1, isAsCheapAsAMove = 1 in {
  def MOVaddr32
    : InstPTX<(outs RRegu32:$d), (ins i32imm:$a), "mov.u32\t$d, $a",
              [(set RRegu32:$d, (PTXcopyaddress tglobaladdr:$a))]>;
  def MOVaddr64
    : InstPTX<(outs RRegu64:$d), (ins i64imm:$a), "mov.u64\t$d, $a",
              [(set RRegu64:$d, (PTXcopyaddress tglobaladdr:$a))]>;
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

// Conversion to pred

def CVT_pred_u16
  : InstPTX<(outs Preds:$d), (ins RRegu16:$a), "cvt.pred.u16\t$d, $a",
            [(set Preds:$d, (trunc RRegu16:$a))]>;

def CVT_pred_u32
  : InstPTX<(outs Preds:$d), (ins RRegu32:$a), "cvt.pred.u32\t$d, $a",
            [(set Preds:$d, (trunc RRegu32:$a))]>;

def CVT_pred_u64
  : InstPTX<(outs Preds:$d), (ins RRegu64:$a), "cvt.pred.u64\t$d, $a",
            [(set Preds:$d, (trunc RRegu64:$a))]>;

def CVT_pred_f32
  : InstPTX<(outs Preds:$d), (ins RRegf32:$a), "cvt.pred.f32\t$d, $a",
            [(set Preds:$d, (fp_to_uint RRegf32:$a))]>;

def CVT_pred_f64
  : InstPTX<(outs Preds:$d), (ins RRegf64:$a), "cvt.pred.f64\t$d, $a",
            [(set Preds:$d, (fp_to_uint RRegf64:$a))]>;

// Conversion to u16

def CVT_u16_pred
  : InstPTX<(outs RRegu16:$d), (ins Preds:$a), "cvt.u16.pred\t$d, $a",
            [(set RRegu16:$d, (zext Preds:$a))]>;

def CVT_u16_u32
  : InstPTX<(outs RRegu16:$d), (ins RRegu32:$a), "cvt.u16.u32\t$d, $a",
            [(set RRegu16:$d, (trunc RRegu32:$a))]>;

def CVT_u16_u64
  : InstPTX<(outs RRegu16:$d), (ins RRegu64:$a), "cvt.u16.u64\t$d, $a",
            [(set RRegu16:$d, (trunc RRegu64:$a))]>;

def CVT_u16_f32
  : InstPTX<(outs RRegu16:$d), (ins RRegf32:$a), "cvt.u16.f32\t$d, $a",
            [(set RRegu16:$d, (fp_to_uint RRegf32:$a))]>;

def CVT_u16_f64
  : InstPTX<(outs RRegu16:$d), (ins RRegf64:$a), "cvt.u16.f64\t$d, $a",
            [(set RRegu16:$d, (fp_to_uint RRegf64:$a))]>;

// Conversion to u32

def CVT_u32_pred
  : InstPTX<(outs RRegu32:$d), (ins Preds:$a), "cvt.u32.pred\t$d, $a",
            [(set RRegu32:$d, (zext Preds:$a))]>;

def CVT_u32_u16
  : InstPTX<(outs RRegu32:$d), (ins RRegu16:$a), "cvt.u32.u16\t$d, $a",
            [(set RRegu32:$d, (zext RRegu16:$a))]>;

def CVT_u32_u64
  : InstPTX<(outs RRegu32:$d), (ins RRegu64:$a), "cvt.u32.u64\t$d, $a",
            [(set RRegu32:$d, (trunc RRegu64:$a))]>;

def CVT_u32_f32
  : InstPTX<(outs RRegu32:$d), (ins RRegf32:$a), "cvt.u32.f32\t$d, $a",
            [(set RRegu32:$d, (fp_to_uint RRegf32:$a))]>;

def CVT_u32_f64
  : InstPTX<(outs RRegu32:$d), (ins RRegf64:$a), "cvt.u32.f64\t$d, $a",
            [(set RRegu32:$d, (fp_to_uint RRegf64:$a))]>;

// Conversion to u64

def CVT_u64_pred
  : InstPTX<(outs RRegu64:$d), (ins Preds:$a), "cvt.u64.pred\t$d, $a",
            [(set RRegu64:$d, (zext Preds:$a))]>;

def CVT_u64_u16
  : InstPTX<(outs RRegu64:$d), (ins RRegu16:$a), "cvt.u64.u16\t$d, $a",
            [(set RRegu64:$d, (zext RRegu16:$a))]>;

def CVT_u64_u32
  : InstPTX<(outs RRegu64:$d), (ins RRegu32:$a), "cvt.u64.u32\t$d, $a",
            [(set RRegu64:$d, (zext RRegu32:$a))]>;

def CVT_u64_f32
  : InstPTX<(outs RRegu64:$d), (ins RRegf32:$a), "cvt.u64.f32\t$d, $a",
            [(set RRegu64:$d, (fp_to_uint RRegf32:$a))]>;

def CVT_u64_f64
  : InstPTX<(outs RRegu64:$d), (ins RRegf64:$a), "cvt.u64.f32\t$d, $a",
            [(set RRegu64:$d, (fp_to_uint RRegf64:$a))]>;

// Conversion to f32

def CVT_f32_pred
  : InstPTX<(outs RRegf32:$d), (ins Preds:$a), "cvt.f32.pred\t$d, $a",
            [(set RRegf32:$d, (uint_to_fp Preds:$a))]>;

def CVT_f32_u16
  : InstPTX<(outs RRegf32:$d), (ins RRegu16:$a), "cvt.f32.u16\t$d, $a",
            [(set RRegf32:$d, (uint_to_fp RRegu16:$a))]>;

def CVT_f32_u32
  : InstPTX<(outs RRegf32:$d), (ins RRegu32:$a), "cvt.f32.u32\t$d, $a",
            [(set RRegf32:$d, (uint_to_fp RRegu32:$a))]>;

def CVT_f32_u64
  : InstPTX<(outs RRegf32:$d), (ins RRegu64:$a), "cvt.f32.u64\t$d, $a",
            [(set RRegf32:$d, (uint_to_fp RRegu64:$a))]>;

def CVT_f32_f64
  : InstPTX<(outs RRegf32:$d), (ins RRegf64:$a), "cvt.f32.f64\t$d, $a",
            [(set RRegf32:$d, (fround RRegf64:$a))]>;

// Conversion to f64

def CVT_f64_pred
  : InstPTX<(outs RRegf64:$d), (ins Preds:$a), "cvt.f64.pred\t$d, $a",
            [(set RRegf64:$d, (uint_to_fp Preds:$a))]>;

def CVT_f64_u16
  : InstPTX<(outs RRegf64:$d), (ins RRegu16:$a), "cvt.f64.u16\t$d, $a",
            [(set RRegf64:$d, (uint_to_fp RRegu16:$a))]>;

def CVT_f64_u32
  : InstPTX<(outs RRegf64:$d), (ins RRegu32:$a), "cvt.f64.u32\t$d, $a",
            [(set RRegf64:$d, (uint_to_fp RRegu32:$a))]>;

def CVT_f64_u64
  : InstPTX<(outs RRegf64:$d), (ins RRegu64:$a), "cvt.f64.u64\t$d, $a",
            [(set RRegf64:$d, (uint_to_fp RRegu64:$a))]>;

def CVT_f64_f32
  : InstPTX<(outs RRegf64:$d), (ins RRegf32:$a), "cvt.f64.f32\t$d, $a",
            [(set RRegf64:$d, (fextend RRegf32:$a))]>;

///===- Control Flow Instructions -----------------------------------------===//

let isBranch = 1, isTerminator = 1, isBarrier = 1 in {
  def BRAd
    : InstPTX<(outs), (ins brtarget:$d), "bra\t$d", [(br bb:$d)]>;
}

let isBranch = 1, isTerminator = 1 in {
  // FIXME: The pattern part is blank because I cannot (or do not yet know
  // how to) use the first operand of PredicateOperand (a Preds register) here
  def BRAdp
    : InstPTX<(outs), (ins brtarget:$d), "bra\t$d",
              [/*(brcond pred:$_p, bb:$d)*/]>;
}

let isReturn = 1, isTerminator = 1, isBarrier = 1 in {
  def EXIT : InstPTX<(outs), (ins), "exit", [(PTXexit)]>;
  def RET  : InstPTX<(outs), (ins), "ret",  [(PTXret)]>;
}

///===- Intrinsic Instructions --------------------------------------------===//

include "PTXIntrinsicInstrInfo.td"
