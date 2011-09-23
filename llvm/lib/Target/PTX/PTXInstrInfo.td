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
def Use32BitAddresses : Predicate<"!getSubtarget().is64Bit()">;
def Use64BitAddresses : Predicate<"getSubtarget().is64Bit()">;

// Shader Model Support
def FDivNeedsRoundingMode : Predicate<"getSubtarget().fdivNeedsRoundingMode()">;
def FDivNoRoundingMode : Predicate<"!getSubtarget().fdivNeedsRoundingMode()">;
def FMadNeedsRoundingMode : Predicate<"getSubtarget().fmadNeedsRoundingMode()">;
def FMadNoRoundingMode : Predicate<"!getSubtarget().fmadNeedsRoundingMode()">;

// PTX Version Support
def SupportsPTX21       : Predicate<"getSubtarget().supportsPTX21()">;
def DoesNotSupportPTX21 : Predicate<"!getSubtarget().supportsPTX21()">;
def SupportsPTX22       : Predicate<"getSubtarget().supportsPTX22()">;
def DoesNotSupportPTX22 : Predicate<"!getSubtarget().supportsPTX22()">;
def SupportsPTX23       : Predicate<"getSubtarget().supportsPTX23()">;
def DoesNotSupportPTX23 : Predicate<"!getSubtarget().supportsPTX23()">;

// Fused-Multiply Add
def SupportsFMA         : Predicate<"getSubtarget().supportsFMA()">;
def DoesNotSupportFMA   : Predicate<"!getSubtarget().supportsFMA()">;

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
  let MIOperandInfo = (ops RegI32, i32imm);
}
def MEMri64 : Operand<i64> {
  let PrintMethod = "printMemOperand";
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

// def SDT_PTXCallSeqStart : SDCallSeqStart<[SDTCisVT<0, i32>]>;
// def SDT_PTXCallSeqEnd   : SDCallSeqEnd<[SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;

// def PTXcallseq_start : SDNode<"ISD::CALLSEQ_START", SDT_PTXCallSeqStart,
//                               [SDNPHasChain, SDNPOutGlue]>;
// def PTXcallseq_end   : SDNode<"ISD::CALLSEQ_END", SDT_PTXCallSeqEnd,
//                               [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;

def PTXcall : SDNode<"PTXISD::CALL", SDTNone,
                     [SDNPHasChain, SDNPVariadic, SDNPOptInGlue, SDNPOutGlue]>;


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
  : SDNode<"PTXISD::RET",  SDTNone,
           [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;
def PTXcopyaddress
  : SDNode<"PTXISD::COPY_ADDRESS", SDTypeProfile<1, 1, []>, []>;

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
// Instruction Class Templates
//===----------------------------------------------------------------------===//

//===- Floating-Point Instructions - 2 Operand Form -----------------------===//
multiclass PTX_FLOAT_2OP<string opcstr, SDNode opnode> {
  def rr32 : InstPTX<(outs RegF32:$d),
                     (ins RegF32:$a),
                     !strconcat(opcstr, ".f32\t$d, $a"),
                     [(set RegF32:$d, (opnode RegF32:$a))]>;
  def ri32 : InstPTX<(outs RegF32:$d),
                     (ins f32imm:$a),
                     !strconcat(opcstr, ".f32\t$d, $a"),
                     [(set RegF32:$d, (opnode fpimm:$a))]>;
  def rr64 : InstPTX<(outs RegF64:$d),
                     (ins RegF64:$a),
                     !strconcat(opcstr, ".f64\t$d, $a"),
                     [(set RegF64:$d, (opnode RegF64:$a))]>;
  def ri64 : InstPTX<(outs RegF64:$d),
                     (ins f64imm:$a),
                     !strconcat(opcstr, ".f64\t$d, $a"),
                     [(set RegF64:$d, (opnode fpimm:$a))]>;
}

//===- Floating-Point Instructions - 3 Operand Form -----------------------===//
multiclass PTX_FLOAT_3OP<string opcstr, SDNode opnode> {
  def rr32 : InstPTX<(outs RegF32:$d),
                     (ins RegF32:$a, RegF32:$b),
                     !strconcat(opcstr, ".f32\t$d, $a, $b"),
                     [(set RegF32:$d, (opnode RegF32:$a, RegF32:$b))]>;
  def ri32 : InstPTX<(outs RegF32:$d),
                     (ins RegF32:$a, f32imm:$b),
                     !strconcat(opcstr, ".f32\t$d, $a, $b"),
                     [(set RegF32:$d, (opnode RegF32:$a, fpimm:$b))]>;
  def rr64 : InstPTX<(outs RegF64:$d),
                     (ins RegF64:$a, RegF64:$b),
                     !strconcat(opcstr, ".f64\t$d, $a, $b"),
                     [(set RegF64:$d, (opnode RegF64:$a, RegF64:$b))]>;
  def ri64 : InstPTX<(outs RegF64:$d),
                     (ins RegF64:$a, f64imm:$b),
                     !strconcat(opcstr, ".f64\t$d, $a, $b"),
                     [(set RegF64:$d, (opnode RegF64:$a, fpimm:$b))]>;
}

//===- Floating-Point Instructions - 4 Operand Form -----------------------===//
multiclass PTX_FLOAT_4OP<string opcstr, SDNode opnode1, SDNode opnode2> {
  def rrr32 : InstPTX<(outs RegF32:$d),
                      (ins RegF32:$a, RegF32:$b, RegF32:$c),
                      !strconcat(opcstr, ".f32\t$d, $a, $b, $c"),
                      [(set RegF32:$d, (opnode2 (opnode1 RegF32:$a,
                                                          RegF32:$b),
                                                 RegF32:$c))]>;
  def rri32 : InstPTX<(outs RegF32:$d),
                      (ins RegF32:$a, RegF32:$b, f32imm:$c),
                      !strconcat(opcstr, ".f32\t$d, $a, $b, $c"),
                      [(set RegF32:$d, (opnode2 (opnode1 RegF32:$a,
                                                          RegF32:$b),
                                                 fpimm:$c))]>;
  def rrr64 : InstPTX<(outs RegF64:$d),
                      (ins RegF64:$a, RegF64:$b, RegF64:$c),
                      !strconcat(opcstr, ".f64\t$d, $a, $b, $c"),
                      [(set RegF64:$d, (opnode2 (opnode1 RegF64:$a,
                                                          RegF64:$b),
                                                 RegF64:$c))]>;
  def rri64 : InstPTX<(outs RegF64:$d),
                      (ins RegF64:$a, RegF64:$b, f64imm:$c),
                      !strconcat(opcstr, ".f64\t$d, $a, $b, $c"),
                      [(set RegF64:$d, (opnode2 (opnode1 RegF64:$a,
                                                          RegF64:$b),
                                                 fpimm:$c))]>;
}

multiclass INT3<string opcstr, SDNode opnode> {
  def rr16 : InstPTX<(outs RegI16:$d),
                     (ins RegI16:$a, RegI16:$b),
                     !strconcat(opcstr, ".u16\t$d, $a, $b"),
                     [(set RegI16:$d, (opnode RegI16:$a, RegI16:$b))]>;
  def ri16 : InstPTX<(outs RegI16:$d),
                     (ins RegI16:$a, i16imm:$b),
                     !strconcat(opcstr, ".u16\t$d, $a, $b"),
                     [(set RegI16:$d, (opnode RegI16:$a, imm:$b))]>;
  def rr32 : InstPTX<(outs RegI32:$d),
                     (ins RegI32:$a, RegI32:$b),
                     !strconcat(opcstr, ".u32\t$d, $a, $b"),
                     [(set RegI32:$d, (opnode RegI32:$a, RegI32:$b))]>;
  def ri32 : InstPTX<(outs RegI32:$d),
                     (ins RegI32:$a, i32imm:$b),
                     !strconcat(opcstr, ".u32\t$d, $a, $b"),
                     [(set RegI32:$d, (opnode RegI32:$a, imm:$b))]>;
  def rr64 : InstPTX<(outs RegI64:$d),
                     (ins RegI64:$a, RegI64:$b),
                     !strconcat(opcstr, ".u64\t$d, $a, $b"),
                     [(set RegI64:$d, (opnode RegI64:$a, RegI64:$b))]>;
  def ri64 : InstPTX<(outs RegI64:$d),
                     (ins RegI64:$a, i64imm:$b),
                     !strconcat(opcstr, ".u64\t$d, $a, $b"),
                     [(set RegI64:$d, (opnode RegI64:$a, imm:$b))]>;
}

multiclass PTX_LOGIC<string opcstr, SDNode opnode> {
  def ripreds : InstPTX<(outs RegPred:$d),
                     (ins RegPred:$a, i1imm:$b),
                     !strconcat(opcstr, ".pred\t$d, $a, $b"),
                     [(set RegPred:$d, (opnode RegPred:$a, imm:$b))]>;
  def rrpreds : InstPTX<(outs RegPred:$d),
                     (ins RegPred:$a, RegPred:$b),
                     !strconcat(opcstr, ".pred\t$d, $a, $b"),
                     [(set RegPred:$d, (opnode RegPred:$a, RegPred:$b))]>;
  def rr16 : InstPTX<(outs RegI16:$d),
                     (ins RegI16:$a, RegI16:$b),
                     !strconcat(opcstr, ".b16\t$d, $a, $b"),
                     [(set RegI16:$d, (opnode RegI16:$a, RegI16:$b))]>;
  def ri16 : InstPTX<(outs RegI16:$d),
                     (ins RegI16:$a, i16imm:$b),
                     !strconcat(opcstr, ".b16\t$d, $a, $b"),
                     [(set RegI16:$d, (opnode RegI16:$a, imm:$b))]>;
  def rr32 : InstPTX<(outs RegI32:$d),
                     (ins RegI32:$a, RegI32:$b),
                     !strconcat(opcstr, ".b32\t$d, $a, $b"),
                     [(set RegI32:$d, (opnode RegI32:$a, RegI32:$b))]>;
  def ri32 : InstPTX<(outs RegI32:$d),
                     (ins RegI32:$a, i32imm:$b),
                     !strconcat(opcstr, ".b32\t$d, $a, $b"),
                     [(set RegI32:$d, (opnode RegI32:$a, imm:$b))]>;
  def rr64 : InstPTX<(outs RegI64:$d),
                     (ins RegI64:$a, RegI64:$b),
                     !strconcat(opcstr, ".b64\t$d, $a, $b"),
                     [(set RegI64:$d, (opnode RegI64:$a, RegI64:$b))]>;
  def ri64 : InstPTX<(outs RegI64:$d),
                     (ins RegI64:$a, i64imm:$b),
                     !strconcat(opcstr, ".b64\t$d, $a, $b"),
                     [(set RegI64:$d, (opnode RegI64:$a, imm:$b))]>;
}

multiclass INT3ntnc<string opcstr, SDNode opnode> {
  def rr16 : InstPTX<(outs RegI16:$d),
                     (ins RegI16:$a, RegI16:$b),
                     !strconcat(opcstr, "16\t$d, $a, $b"),
                     [(set RegI16:$d, (opnode RegI16:$a, RegI16:$b))]>;
  def rr32 : InstPTX<(outs RegI32:$d),
                     (ins RegI32:$a, RegI32:$b),
                     !strconcat(opcstr, "32\t$d, $a, $b"),
                     [(set RegI32:$d, (opnode RegI32:$a, RegI32:$b))]>;
  def rr64 : InstPTX<(outs RegI64:$d),
                     (ins RegI64:$a, RegI64:$b),
                     !strconcat(opcstr, "64\t$d, $a, $b"),
                     [(set RegI64:$d, (opnode RegI64:$a, RegI64:$b))]>;
  def ri16 : InstPTX<(outs RegI16:$d),
                     (ins RegI16:$a, i16imm:$b),
                     !strconcat(opcstr, "16\t$d, $a, $b"),
                     [(set RegI16:$d, (opnode RegI16:$a, imm:$b))]>;
  def ri32 : InstPTX<(outs RegI32:$d),
                     (ins RegI32:$a, i32imm:$b),
                     !strconcat(opcstr, "32\t$d, $a, $b"),
                     [(set RegI32:$d, (opnode RegI32:$a, imm:$b))]>;
  def ri64 : InstPTX<(outs RegI64:$d),
                     (ins RegI64:$a, i64imm:$b),
                     !strconcat(opcstr, "64\t$d, $a, $b"),
                     [(set RegI64:$d, (opnode RegI64:$a, imm:$b))]>;
  def ir16 : InstPTX<(outs RegI16:$d),
                     (ins i16imm:$a, RegI16:$b),
                     !strconcat(opcstr, "16\t$d, $a, $b"),
                     [(set RegI16:$d, (opnode imm:$a, RegI16:$b))]>;
  def ir32 : InstPTX<(outs RegI32:$d),
                     (ins i32imm:$a, RegI32:$b),
                     !strconcat(opcstr, "32\t$d, $a, $b"),
                     [(set RegI32:$d, (opnode imm:$a, RegI32:$b))]>;
  def ir64 : InstPTX<(outs RegI64:$d),
                     (ins i64imm:$a, RegI64:$b),
                     !strconcat(opcstr, "64\t$d, $a, $b"),
                     [(set RegI64:$d, (opnode imm:$a, RegI64:$b))]>;
}

multiclass PTX_SETP_I<RegisterClass RC, string regclsname, Operand immcls,
                        CondCode cmp, string cmpstr> {
  // TODO support 5-operand format: p|q, a, b, c

  def rr
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b),
              !strconcat("setp.", cmpstr, ".", regclsname, "\t$p, $a, $b"),
              [(set RegPred:$p, (setcc RC:$a, RC:$b, cmp))]>;
  def ri
    : InstPTX<(outs RegPred:$p), (ins RC:$a, immcls:$b),
              !strconcat("setp.", cmpstr, ".", regclsname, "\t$p, $a, $b"),
              [(set RegPred:$p, (setcc RC:$a, imm:$b, cmp))]>;

  def rr_and_r
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".and.", regclsname, "\t$p, $a, $b, $c"),
              [(set RegPred:$p, (and (setcc RC:$a, RC:$b, cmp), RegPred:$c))]>;
  def ri_and_r
    : InstPTX<(outs RegPred:$p), (ins RC:$a, immcls:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".and.", regclsname, "\t$p, $a, $b, $c"),
              [(set RegPred:$p, (and (setcc RC:$a, imm:$b, cmp), RegPred:$c))]>;
  def rr_or_r
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".or.", regclsname, "\t$p, $a, $b, $c"),
              [(set RegPred:$p, (or (setcc RC:$a, RC:$b, cmp), RegPred:$c))]>;
  def ri_or_r
    : InstPTX<(outs RegPred:$p), (ins RC:$a, immcls:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".or.", regclsname, "\t$p, $a, $b, $c"),
              [(set RegPred:$p, (or (setcc RC:$a, imm:$b, cmp), RegPred:$c))]>;
  def rr_xor_r
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".xor.", regclsname, "\t$p, $a, $b, $c"),
              [(set RegPred:$p, (xor (setcc RC:$a, RC:$b, cmp), RegPred:$c))]>;
  def ri_xor_r
    : InstPTX<(outs RegPred:$p), (ins RC:$a, immcls:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".xor.", regclsname, "\t$p, $a, $b, $c"),
              [(set RegPred:$p, (xor (setcc RC:$a, imm:$b, cmp), RegPred:$c))]>;

  def rr_and_not_r
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".and.", regclsname, "\t$p, $a, $b, !$c"),
              [(set RegPred:$p, (and (setcc RC:$a, RC:$b, cmp), (not RegPred:$c)))]>;
  def ri_and_not_r
    : InstPTX<(outs RegPred:$p), (ins RC:$a, immcls:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".and.", regclsname, "\t$p, $a, $b, !$c"),
              [(set RegPred:$p, (and (setcc RC:$a, imm:$b, cmp), (not RegPred:$c)))]>;
  def rr_or_not_r
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".or.", regclsname, "\t$p, $a, $b, !$c"),
              [(set RegPred:$p, (or (setcc RC:$a, RC:$b, cmp), (not RegPred:$c)))]>;
  def ri_or_not_r
    : InstPTX<(outs RegPred:$p), (ins RC:$a, immcls:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".or.", regclsname, "\t$p, $a, $b, !$c"),
              [(set RegPred:$p, (or (setcc RC:$a, imm:$b, cmp), (not RegPred:$c)))]>;
  def rr_xor_not_r
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".xor.", regclsname, "\t$p, $a, $b, !$c"),
              [(set RegPred:$p, (xor (setcc RC:$a, RC:$b, cmp), (not RegPred:$c)))]>;
  def ri_xor_not_r
    : InstPTX<(outs RegPred:$p), (ins RC:$a, immcls:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".xor.", regclsname, "\t$p, $a, $b, !$c"),
              [(set RegPred:$p, (xor (setcc RC:$a, imm:$b, cmp), (not RegPred:$c)))]>;
}

multiclass PTX_SETP_FP<RegisterClass RC, string regclsname,
                        CondCode ucmp, CondCode ocmp, string cmpstr> {
  // TODO support 5-operand format: p|q, a, b, c

  def rr_u
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b),
              !strconcat("setp.", cmpstr, "u.", regclsname, "\t$p, $a, $b"),
              [(set RegPred:$p, (setcc RC:$a, RC:$b, ucmp))]>;
  def rr_o
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b),
              !strconcat("setp.", cmpstr, ".", regclsname, "\t$p, $a, $b"),
              [(set RegPred:$p, (setcc RC:$a, RC:$b, ocmp))]>;

  def rr_and_r_u
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, "u.and.", regclsname, "\t$p, $a, $b, $c"),
              [(set RegPred:$p, (and (setcc RC:$a, RC:$b, ucmp), RegPred:$c))]>;
  def rr_and_r_o
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".and.", regclsname, "\t$p, $a, $b, $c"),
              [(set RegPred:$p, (and (setcc RC:$a, RC:$b, ocmp), RegPred:$c))]>;

  def rr_or_r_u
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, "u.or.", regclsname, "\t$p, $a, $b, $c"),
              [(set RegPred:$p, (or (setcc RC:$a, RC:$b, ucmp), RegPred:$c))]>;
  def rr_or_r_o
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".or.", regclsname, "\t$p, $a, $b, $c"),
              [(set RegPred:$p, (or (setcc RC:$a, RC:$b, ocmp), RegPred:$c))]>;

  def rr_xor_r_u
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, "u.xor.", regclsname, "\t$p, $a, $b, $c"),
              [(set RegPred:$p, (xor (setcc RC:$a, RC:$b, ucmp), RegPred:$c))]>;
  def rr_xor_r_o
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".xor.", regclsname, "\t$p, $a, $b, $c"),
              [(set RegPred:$p, (xor (setcc RC:$a, RC:$b, ocmp), RegPred:$c))]>;

  def rr_and_not_r_u
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, "u.and.", regclsname, "\t$p, $a, $b, !$c"),
              [(set RegPred:$p, (and (setcc RC:$a, RC:$b, ucmp), (not RegPred:$c)))]>;
  def rr_and_not_r_o
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".and.", regclsname, "\t$p, $a, $b, !$c"),
              [(set RegPred:$p, (and (setcc RC:$a, RC:$b, ocmp), (not RegPred:$c)))]>;

  def rr_or_not_r_u
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, "u.or.", regclsname, "\t$p, $a, $b, !$c"),
              [(set RegPred:$p, (or (setcc RC:$a, RC:$b, ucmp), (not RegPred:$c)))]>;
  def rr_or_not_r_o
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".or.", regclsname, "\t$p, $a, $b, !$c"),
              [(set RegPred:$p, (or (setcc RC:$a, RC:$b, ocmp), (not RegPred:$c)))]>;

  def rr_xor_not_r_u
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, "u.xor.", regclsname, "\t$p, $a, $b, !$c"),
              [(set RegPred:$p, (xor (setcc RC:$a, RC:$b, ucmp), (not RegPred:$c)))]>;
  def rr_xor_not_r_o
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".xor.", regclsname, "\t$p, $a, $b, !$c"),
              [(set RegPred:$p, (xor (setcc RC:$a, RC:$b, ocmp), (not RegPred:$c)))]>;
}

multiclass PTX_SELP<RegisterClass RC, string regclsname> {
  def rr
    : InstPTX<(outs RC:$r), (ins RegPred:$a, RC:$b, RC:$c),
              !strconcat("selp.", regclsname, "\t$r, $b, $c, $a"),
              [(set RC:$r, (select RegPred:$a, RC:$b, RC:$c))]>;
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
  defm u16 : PTX_LD<opstr, ".u16", RegI16, pat_load>;
  defm u32 : PTX_LD<opstr, ".u32", RegI32, pat_load>;
  defm u64 : PTX_LD<opstr, ".u64", RegI64, pat_load>;
  defm f32 : PTX_LD<opstr, ".f32", RegF32, pat_load>;
  defm f64 : PTX_LD<opstr, ".f64", RegF64, pat_load>;
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
  defm u16 : PTX_ST<opstr, ".u16", RegI16, pat_store>;
  defm u32 : PTX_ST<opstr, ".u32", RegI32, pat_store>;
  defm u64 : PTX_ST<opstr, ".u64", RegI64, pat_store>;
  defm f32 : PTX_ST<opstr, ".f32", RegF32, pat_store>;
  defm f64 : PTX_ST<opstr, ".f64", RegF64, pat_store>;
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

// Standard Unary Operations
defm FNEG : PTX_FLOAT_2OP<"neg", fneg>;

// Standard Binary Operations
defm FADD : PTX_FLOAT_3OP<"add.rn", fadd>;
defm FSUB : PTX_FLOAT_3OP<"sub.rn", fsub>;
defm FMUL : PTX_FLOAT_3OP<"mul.rn", fmul>;

// For floating-point division:
// SM_13+ defaults to .rn for f32 and f64,
// SM10 must *not* provide a rounding

// TODO:
//     - Allow user selection of rounding modes for fdiv
//     - Add support for -prec-div=false (.approx)

def FDIVrr32SM13 : InstPTX<(outs RegF32:$d),
                       (ins RegF32:$a, RegF32:$b),
                       "div.rn.f32\t$d, $a, $b",
                       [(set RegF32:$d, (fdiv RegF32:$a, RegF32:$b))]>,
                   Requires<[FDivNeedsRoundingMode]>;
def FDIVri32SM13 : InstPTX<(outs RegF32:$d),
                       (ins RegF32:$a, f32imm:$b),
                       "div.rn.f32\t$d, $a, $b",
                       [(set RegF32:$d, (fdiv RegF32:$a, fpimm:$b))]>,
                   Requires<[FDivNeedsRoundingMode]>;
def FDIVrr32SM10 : InstPTX<(outs RegF32:$d),
                       (ins RegF32:$a, RegF32:$b),
                       "div.f32\t$d, $a, $b",
                       [(set RegF32:$d, (fdiv RegF32:$a, RegF32:$b))]>,
                   Requires<[FDivNoRoundingMode]>;
def FDIVri32SM10 : InstPTX<(outs RegF32:$d),
                       (ins RegF32:$a, f32imm:$b),
                       "div.f32\t$d, $a, $b",
                       [(set RegF32:$d, (fdiv RegF32:$a, fpimm:$b))]>,
                   Requires<[FDivNoRoundingMode]>;

def FDIVrr64SM13 : InstPTX<(outs RegF64:$d),
                           (ins RegF64:$a, RegF64:$b),
                           "div.rn.f64\t$d, $a, $b",
                           [(set RegF64:$d, (fdiv RegF64:$a, RegF64:$b))]>,
                   Requires<[FDivNeedsRoundingMode]>;
def FDIVri64SM13 : InstPTX<(outs RegF64:$d),
                           (ins RegF64:$a, f64imm:$b),
                           "div.rn.f64\t$d, $a, $b",
                           [(set RegF64:$d, (fdiv RegF64:$a, fpimm:$b))]>,
                   Requires<[FDivNeedsRoundingMode]>;
def FDIVrr64SM10 : InstPTX<(outs RegF64:$d),
                           (ins RegF64:$a, RegF64:$b),
                           "div.f64\t$d, $a, $b",
                           [(set RegF64:$d, (fdiv RegF64:$a, RegF64:$b))]>,
                   Requires<[FDivNoRoundingMode]>;
def FDIVri64SM10 : InstPTX<(outs RegF64:$d),
                           (ins RegF64:$a, f64imm:$b),
                           "div.f64\t$d, $a, $b",
                           [(set RegF64:$d, (fdiv RegF64:$a, fpimm:$b))]>,
                   Requires<[FDivNoRoundingMode]>;



// Multi-operation hybrid instructions

// The selection of mad/fma is tricky.  In some cases, they are the *same*
// instruction, but in other cases we may prefer one or the other.  Also,
// different PTX versions differ on whether rounding mode flags are required.
// In the short term, mad is supported on all PTX versions and we use a
// default rounding mode no matter what shader model or PTX version.
// TODO: Allow the rounding mode to be selectable through llc.
defm FMADSM13 : PTX_FLOAT_4OP<"mad.rn", fmul, fadd>,
                Requires<[FMadNeedsRoundingMode, SupportsFMA]>;
defm FMAD : PTX_FLOAT_4OP<"mad", fmul, fadd>,
            Requires<[FMadNoRoundingMode, SupportsFMA]>;

///===- Floating-Point Intrinsic Instructions -----------------------------===//

def FSQRT32 : InstPTX<(outs RegF32:$d),
                      (ins RegF32:$a),
                      "sqrt.rn.f32\t$d, $a",
                      [(set RegF32:$d, (fsqrt RegF32:$a))]>;

def FSQRT64 : InstPTX<(outs RegF64:$d),
                      (ins RegF64:$a),
                      "sqrt.rn.f64\t$d, $a",
                      [(set RegF64:$d, (fsqrt RegF64:$a))]>;

def FSIN32 : InstPTX<(outs RegF32:$d),
                     (ins RegF32:$a),
                     "sin.approx.f32\t$d, $a",
                     [(set RegF32:$d, (fsin RegF32:$a))]>;

def FSIN64 : InstPTX<(outs RegF64:$d),
                     (ins RegF64:$a),
                     "sin.approx.f64\t$d, $a",
                     [(set RegF64:$d, (fsin RegF64:$a))]>;

def FCOS32 : InstPTX<(outs RegF32:$d),
                     (ins RegF32:$a),
                     "cos.approx.f32\t$d, $a",
                     [(set RegF32:$d, (fcos RegF32:$a))]>;

def FCOS64 : InstPTX<(outs RegF64:$d),
                     (ins RegF64:$a),
                     "cos.approx.f64\t$d, $a",
                     [(set RegF64:$d, (fcos RegF64:$a))]>;


///===- Comparison and Selection Instructions -----------------------------===//

// .setp

// Compare u16

defm SETPEQu16 : PTX_SETP_I<RegI16, "u16", i16imm, SETEQ,  "eq">;
defm SETPNEu16 : PTX_SETP_I<RegI16, "u16", i16imm, SETNE,  "ne">;
defm SETPLTu16 : PTX_SETP_I<RegI16, "u16", i16imm, SETULT, "lt">;
defm SETPLEu16 : PTX_SETP_I<RegI16, "u16", i16imm, SETULE, "le">;
defm SETPGTu16 : PTX_SETP_I<RegI16, "u16", i16imm, SETUGT, "gt">;
defm SETPGEu16 : PTX_SETP_I<RegI16, "u16", i16imm, SETUGE, "ge">;
defm SETPLTs16 : PTX_SETP_I<RegI16, "s16", i16imm, SETLT,  "lt">;
defm SETPLEs16 : PTX_SETP_I<RegI16, "s16", i16imm, SETLE,  "le">;
defm SETPGTs16 : PTX_SETP_I<RegI16, "s16", i16imm, SETGT,  "gt">;
defm SETPGEs16 : PTX_SETP_I<RegI16, "s16", i16imm, SETGE,  "ge">;

// Compare u32

defm SETPEQu32 : PTX_SETP_I<RegI32, "u32", i32imm, SETEQ,  "eq">;
defm SETPNEu32 : PTX_SETP_I<RegI32, "u32", i32imm, SETNE,  "ne">;
defm SETPLTu32 : PTX_SETP_I<RegI32, "u32", i32imm, SETULT, "lt">;
defm SETPLEu32 : PTX_SETP_I<RegI32, "u32", i32imm, SETULE, "le">;
defm SETPGTu32 : PTX_SETP_I<RegI32, "u32", i32imm, SETUGT, "gt">;
defm SETPGEu32 : PTX_SETP_I<RegI32, "u32", i32imm, SETUGE, "ge">;
defm SETPLTs32 : PTX_SETP_I<RegI32, "s32", i32imm, SETLT,  "lt">;
defm SETPLEs32 : PTX_SETP_I<RegI32, "s32", i32imm, SETLE,  "le">;
defm SETPGTs32 : PTX_SETP_I<RegI32, "s32", i32imm, SETGT,  "gt">;
defm SETPGEs32 : PTX_SETP_I<RegI32, "s32", i32imm, SETGE,  "ge">;

// Compare u64

defm SETPEQu64 : PTX_SETP_I<RegI64, "u64", i64imm, SETEQ,  "eq">;
defm SETPNEu64 : PTX_SETP_I<RegI64, "u64", i64imm, SETNE,  "ne">;
defm SETPLTu64 : PTX_SETP_I<RegI64, "u64", i64imm, SETULT, "lt">;
defm SETPLEu64 : PTX_SETP_I<RegI64, "u64", i64imm, SETULE, "le">;
defm SETPGTu64 : PTX_SETP_I<RegI64, "u64", i64imm, SETUGT, "gt">;
defm SETPGEu64 : PTX_SETP_I<RegI64, "u64", i64imm, SETUGE, "ge">;
defm SETPLTs64 : PTX_SETP_I<RegI64, "s64", i64imm, SETLT,  "lt">;
defm SETPLEs64 : PTX_SETP_I<RegI64, "s64", i64imm, SETLE,  "le">;
defm SETPGTs64 : PTX_SETP_I<RegI64, "s64", i64imm, SETGT,  "gt">;
defm SETPGEs64 : PTX_SETP_I<RegI64, "s64", i64imm, SETGE,  "ge">;

// Compare f32

defm SETPEQf32 : PTX_SETP_FP<RegF32, "f32", SETUEQ, SETOEQ, "eq">;
defm SETPNEf32 : PTX_SETP_FP<RegF32, "f32", SETUNE, SETONE, "ne">;
defm SETPLTf32 : PTX_SETP_FP<RegF32, "f32", SETULT, SETOLT, "lt">;
defm SETPLEf32 : PTX_SETP_FP<RegF32, "f32", SETULE, SETOLE, "le">;
defm SETPGTf32 : PTX_SETP_FP<RegF32, "f32", SETUGT, SETOGT, "gt">;
defm SETPGEf32 : PTX_SETP_FP<RegF32, "f32", SETUGE, SETOGE, "ge">;

// Compare f64

defm SETPEQf64 : PTX_SETP_FP<RegF64, "f64", SETUEQ, SETOEQ, "eq">;
defm SETPNEf64 : PTX_SETP_FP<RegF64, "f64", SETUNE, SETONE, "ne">;
defm SETPLTf64 : PTX_SETP_FP<RegF64, "f64", SETULT, SETOLT, "lt">;
defm SETPLEf64 : PTX_SETP_FP<RegF64, "f64", SETULE, SETOLE, "le">;
defm SETPGTf64 : PTX_SETP_FP<RegF64, "f64", SETUGT, SETOGT, "gt">;
defm SETPGEf64 : PTX_SETP_FP<RegF64, "f64", SETUGE, SETOGE, "ge">;

// .selp

defm PTX_SELPu16 : PTX_SELP<RegI16, "u16">;
defm PTX_SELPu32 : PTX_SELP<RegI32, "u32">;
defm PTX_SELPu64 : PTX_SELP<RegI64, "u64">;
defm PTX_SELPf32 : PTX_SELP<RegF32, "f32">;
defm PTX_SELPf64 : PTX_SELP<RegF64, "f64">;

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
    : InstPTX<(outs RegPred:$d), (ins RegPred:$a), "mov.pred\t$d, $a", []>;
  def MOVU16rr
    : InstPTX<(outs RegI16:$d), (ins RegI16:$a), "mov.u16\t$d, $a", []>;
  def MOVU32rr
    : InstPTX<(outs RegI32:$d), (ins RegI32:$a), "mov.u32\t$d, $a", []>;
  def MOVU64rr
    : InstPTX<(outs RegI64:$d), (ins RegI64:$a), "mov.u64\t$d, $a", []>;
  def MOVF32rr
    : InstPTX<(outs RegF32:$d), (ins RegF32:$a), "mov.f32\t$d, $a", []>;
  def MOVF64rr
    : InstPTX<(outs RegF64:$d), (ins RegF64:$a), "mov.f64\t$d, $a", []>;
}

let isReMaterializable = 1, isAsCheapAsAMove = 1 in {
  def MOVPREDri
    : InstPTX<(outs RegPred:$d), (ins i1imm:$a), "mov.pred\t$d, $a",
              [(set RegPred:$d, imm:$a)]>;
  def MOVU16ri
    : InstPTX<(outs RegI16:$d), (ins i16imm:$a), "mov.u16\t$d, $a",
              [(set RegI16:$d, imm:$a)]>;
  def MOVU32ri
    : InstPTX<(outs RegI32:$d), (ins i32imm:$a), "mov.u32\t$d, $a",
              [(set RegI32:$d, imm:$a)]>;
  def MOVU64ri
    : InstPTX<(outs RegI64:$d), (ins i64imm:$a), "mov.u64\t$d, $a",
              [(set RegI64:$d, imm:$a)]>;
  def MOVF32ri
    : InstPTX<(outs RegF32:$d), (ins f32imm:$a), "mov.f32\t$d, $a",
              [(set RegF32:$d, fpimm:$a)]>;
  def MOVF64ri
    : InstPTX<(outs RegF64:$d), (ins f64imm:$a), "mov.f64\t$d, $a",
              [(set RegF64:$d, fpimm:$a)]>;
}

let isReMaterializable = 1, isAsCheapAsAMove = 1 in {
  def MOVaddr32
    : InstPTX<(outs RegI32:$d), (ins i32imm:$a), "mov.u32\t$d, $a",
              [(set RegI32:$d, (PTXcopyaddress tglobaladdr:$a))]>;
  def MOVaddr64
    : InstPTX<(outs RegI64:$d), (ins i64imm:$a), "mov.u64\t$d, $a",
              [(set RegI64:$d, (PTXcopyaddress tglobaladdr:$a))]>;
}

// Loads
defm LDg : PTX_LD_ALL<"ld.global", load_global>;
defm LDc : PTX_LD_ALL<"ld.const",  load_constant>;
defm LDl : PTX_LD_ALL<"ld.local",  load_local>;
defm LDs : PTX_LD_ALL<"ld.shared", load_shared>;

// These instructions are used to load/store from the .param space for
// device and kernel parameters

let hasSideEffects = 1 in {
  def LDpiPred : InstPTX<(outs RegPred:$d), (ins MEMpi:$a),
                         "ld.param.pred\t$d, [$a]",
                         [(set RegPred:$d, (PTXloadparam timm:$a))]>;
  def LDpiU16  : InstPTX<(outs RegI16:$d), (ins MEMpi:$a),
                         "ld.param.u16\t$d, [$a]",
                         [(set RegI16:$d, (PTXloadparam timm:$a))]>;
  def LDpiU32  : InstPTX<(outs RegI32:$d), (ins MEMpi:$a),
                         "ld.param.u32\t$d, [$a]",
                         [(set RegI32:$d, (PTXloadparam timm:$a))]>;
  def LDpiU64  : InstPTX<(outs RegI64:$d), (ins MEMpi:$a),
                         "ld.param.u64\t$d, [$a]",
                         [(set RegI64:$d, (PTXloadparam timm:$a))]>;
  def LDpiF32  : InstPTX<(outs RegF32:$d), (ins MEMpi:$a),
                         "ld.param.f32\t$d, [$a]",
                         [(set RegF32:$d, (PTXloadparam timm:$a))]>;
  def LDpiF64  : InstPTX<(outs RegF64:$d), (ins MEMpi:$a),
                         "ld.param.f64\t$d, [$a]",
                         [(set RegF64:$d, (PTXloadparam timm:$a))]>;

  def STpiPred : InstPTX<(outs), (ins MEMpi:$d, RegPred:$a),
                         "st.param.pred\t[$d], $a",
                         [(PTXstoreparam timm:$d, RegPred:$a)]>;
  def STpiU16  : InstPTX<(outs), (ins MEMpi:$d, RegI16:$a),
                         "st.param.u16\t[$d], $a",
                         [(PTXstoreparam timm:$d, RegI16:$a)]>;
  def STpiU32  : InstPTX<(outs), (ins MEMpi:$d, RegI32:$a),
                         "st.param.u32\t[$d], $a",
                         [(PTXstoreparam timm:$d, RegI32:$a)]>;
  def STpiU64  : InstPTX<(outs), (ins MEMpi:$d, RegI64:$a),
                         "st.param.u64\t[$d], $a",
                         [(PTXstoreparam timm:$d, RegI64:$a)]>;
  def STpiF32  : InstPTX<(outs), (ins MEMpi:$d, RegF32:$a),
                         "st.param.f32\t[$d], $a",
                         [(PTXstoreparam timm:$d, RegF32:$a)]>;
  def STpiF64  : InstPTX<(outs), (ins MEMpi:$d, RegF64:$a),
                         "st.param.f64\t[$d], $a",
                         [(PTXstoreparam timm:$d, RegF64:$a)]>;
}

// Stores
defm STg : PTX_ST_ALL<"st.global", store_global>;
defm STl : PTX_ST_ALL<"st.local",  store_local>;
defm STs : PTX_ST_ALL<"st.shared", store_shared>;

// defm STp : PTX_ST_ALL<"st.param",  store_parameter>;
// defm LDp : PTX_LD_ALL<"ld.param",  load_parameter>;
// TODO: Do something with st.param if/when it is needed.

// Conversion to pred
// PTX does not directly support converting to a predicate type, so we fake it
// by performing a greater-than test between the value and zero.  This follows
// the C convention that any non-zero value is equivalent to 'true'.
def CVT_pred_u16
  : InstPTX<(outs RegPred:$d), (ins RegI16:$a), "setp.gt.u16\t$d, $a, 0",
            [(set RegPred:$d, (trunc RegI16:$a))]>;

def CVT_pred_u32
  : InstPTX<(outs RegPred:$d), (ins RegI32:$a), "setp.gt.u32\t$d, $a, 0",
            [(set RegPred:$d, (trunc RegI32:$a))]>;

def CVT_pred_u64
  : InstPTX<(outs RegPred:$d), (ins RegI64:$a), "setp.gt.u64\t$d, $a, 0",
            [(set RegPred:$d, (trunc RegI64:$a))]>;

def CVT_pred_f32
  : InstPTX<(outs RegPred:$d), (ins RegF32:$a), "setp.gt.f32\t$d, $a, 0",
            [(set RegPred:$d, (fp_to_uint RegF32:$a))]>;

def CVT_pred_f64
  : InstPTX<(outs RegPred:$d), (ins RegF64:$a), "setp.gt.f64\t$d, $a, 0",
            [(set RegPred:$d, (fp_to_uint RegF64:$a))]>;

// Conversion to u16
// PTX does not directly support converting a predicate to a value, so we
// use a select instruction to select either 0 or 1 (integer or fp) based
// on the truth value of the predicate.
def CVT_u16_preda
  : InstPTX<(outs RegI16:$d), (ins RegPred:$a), "selp.u16\t$d, 1, 0, $a",
            [(set RegI16:$d, (anyext RegPred:$a))]>;

def CVT_u16_pred
  : InstPTX<(outs RegI16:$d), (ins RegPred:$a), "selp.u16\t$d, 1, 0, $a",
            [(set RegI16:$d, (zext RegPred:$a))]>;

def CVT_u16_preds
  : InstPTX<(outs RegI16:$d), (ins RegPred:$a), "selp.u16\t$d, 1, 0, $a",
            [(set RegI16:$d, (sext RegPred:$a))]>;

def CVT_u16_u32
  : InstPTX<(outs RegI16:$d), (ins RegI32:$a), "cvt.u16.u32\t$d, $a",
            [(set RegI16:$d, (trunc RegI32:$a))]>;

def CVT_u16_u64
  : InstPTX<(outs RegI16:$d), (ins RegI64:$a), "cvt.u16.u64\t$d, $a",
            [(set RegI16:$d, (trunc RegI64:$a))]>;

def CVT_u16_f32
  : InstPTX<(outs RegI16:$d), (ins RegF32:$a), "cvt.rzi.u16.f32\t$d, $a",
            [(set RegI16:$d, (fp_to_uint RegF32:$a))]>;

def CVT_u16_f64
  : InstPTX<(outs RegI16:$d), (ins RegF64:$a), "cvt.rzi.u16.f64\t$d, $a",
            [(set RegI16:$d, (fp_to_uint RegF64:$a))]>;

// Conversion to u32

def CVT_u32_pred
  : InstPTX<(outs RegI32:$d), (ins RegPred:$a), "selp.u32\t$d, 1, 0, $a",
            [(set RegI32:$d, (zext RegPred:$a))]>;

def CVT_u32_b16
  : InstPTX<(outs RegI32:$d), (ins RegI16:$a), "cvt.u32.u16\t$d, $a",
            [(set RegI32:$d, (anyext RegI16:$a))]>;

def CVT_u32_u16
  : InstPTX<(outs RegI32:$d), (ins RegI16:$a), "cvt.u32.u16\t$d, $a",
            [(set RegI32:$d, (zext RegI16:$a))]>;

def CVT_u32_preds
  : InstPTX<(outs RegI32:$d), (ins RegPred:$a), "selp.u32\t$d, 1, 0, $a",
            [(set RegI32:$d, (sext RegPred:$a))]>;

def CVT_u32_s16
  : InstPTX<(outs RegI32:$d), (ins RegI16:$a), "cvt.u32.s16\t$d, $a",
            [(set RegI32:$d, (sext RegI16:$a))]>;

def CVT_u32_u64
  : InstPTX<(outs RegI32:$d), (ins RegI64:$a), "cvt.u32.u64\t$d, $a",
            [(set RegI32:$d, (trunc RegI64:$a))]>;

def CVT_u32_f32
  : InstPTX<(outs RegI32:$d), (ins RegF32:$a), "cvt.rzi.u32.f32\t$d, $a",
            [(set RegI32:$d, (fp_to_uint RegF32:$a))]>;

def CVT_u32_f64
  : InstPTX<(outs RegI32:$d), (ins RegF64:$a), "cvt.rzi.u32.f64\t$d, $a",
            [(set RegI32:$d, (fp_to_uint RegF64:$a))]>;

// Conversion to u64

def CVT_u64_pred
  : InstPTX<(outs RegI64:$d), (ins RegPred:$a), "selp.u64\t$d, 1, 0, $a",
            [(set RegI64:$d, (zext RegPred:$a))]>;

def CVT_u64_preds
  : InstPTX<(outs RegI64:$d), (ins RegPred:$a), "selp.u64\t$d, 1, 0, $a",
            [(set RegI64:$d, (sext RegPred:$a))]>;

def CVT_u64_u16
  : InstPTX<(outs RegI64:$d), (ins RegI16:$a), "cvt.u64.u16\t$d, $a",
            [(set RegI64:$d, (zext RegI16:$a))]>;

def CVT_u64_s16
  : InstPTX<(outs RegI64:$d), (ins RegI16:$a), "cvt.u64.s16\t$d, $a",
            [(set RegI64:$d, (sext RegI16:$a))]>;

def CVT_u64_u32
  : InstPTX<(outs RegI64:$d), (ins RegI32:$a), "cvt.u64.u32\t$d, $a",
            [(set RegI64:$d, (zext RegI32:$a))]>;

def CVT_u64_s32
  : InstPTX<(outs RegI64:$d), (ins RegI32:$a), "cvt.u64.s32\t$d, $a",
            [(set RegI64:$d, (sext RegI32:$a))]>;

def CVT_u64_f32
  : InstPTX<(outs RegI64:$d), (ins RegF32:$a), "cvt.rzi.u64.f32\t$d, $a",
            [(set RegI64:$d, (fp_to_uint RegF32:$a))]>;

def CVT_u64_f64
  : InstPTX<(outs RegI64:$d), (ins RegF64:$a), "cvt.rzi.u64.f64\t$d, $a",
            [(set RegI64:$d, (fp_to_uint RegF64:$a))]>;

// Conversion to f32

def CVT_f32_pred
  : InstPTX<(outs RegF32:$d), (ins RegPred:$a),
            "selp.f32\t$d, 0F3F800000, 0F00000000, $a",  // 1.0
            [(set RegF32:$d, (uint_to_fp RegPred:$a))]>;

def CVT_f32_u16
  : InstPTX<(outs RegF32:$d), (ins RegI16:$a), "cvt.rn.f32.u16\t$d, $a",
            [(set RegF32:$d, (uint_to_fp RegI16:$a))]>;

def CVT_f32_u32
  : InstPTX<(outs RegF32:$d), (ins RegI32:$a), "cvt.rn.f32.u32\t$d, $a",
            [(set RegF32:$d, (uint_to_fp RegI32:$a))]>;

def CVT_f32_u64
  : InstPTX<(outs RegF32:$d), (ins RegI64:$a), "cvt.rn.f32.u64\t$d, $a",
            [(set RegF32:$d, (uint_to_fp RegI64:$a))]>;

def CVT_f32_f64
  : InstPTX<(outs RegF32:$d), (ins RegF64:$a), "cvt.rn.f32.f64\t$d, $a",
            [(set RegF32:$d, (fround RegF64:$a))]>;

// Conversion to f64

def CVT_f64_pred
  : InstPTX<(outs RegF64:$d), (ins RegPred:$a),
            "selp.f64\t$d, 0D3F80000000000000, 0D0000000000000000, $a",  // 1.0
            [(set RegF64:$d, (uint_to_fp RegPred:$a))]>;

def CVT_f64_u16
  : InstPTX<(outs RegF64:$d), (ins RegI16:$a), "cvt.rn.f64.u16\t$d, $a",
            [(set RegF64:$d, (uint_to_fp RegI16:$a))]>;

def CVT_f64_u32
  : InstPTX<(outs RegF64:$d), (ins RegI32:$a), "cvt.rn.f64.u32\t$d, $a",
            [(set RegF64:$d, (uint_to_fp RegI32:$a))]>;

def CVT_f64_u64
  : InstPTX<(outs RegF64:$d), (ins RegI64:$a), "cvt.rn.f64.u64\t$d, $a",
            [(set RegF64:$d, (uint_to_fp RegI64:$a))]>;

def CVT_f64_f32
  : InstPTX<(outs RegF64:$d), (ins RegF32:$a), "cvt.f64.f32\t$d, $a",
            [(set RegF64:$d, (fextend RegF32:$a))]>;

///===- Control Flow Instructions -----------------------------------------===//

let isBranch = 1, isTerminator = 1, isBarrier = 1 in {
  def BRAd
    : InstPTX<(outs), (ins brtarget:$d), "bra\t$d", [(br bb:$d)]>;
}

let isBranch = 1, isTerminator = 1 in {
  // FIXME: The pattern part is blank because I cannot (or do not yet know
  // how to) use the first operand of PredicateOperand (a RegPred register) here
  def BRAdp
    : InstPTX<(outs), (ins brtarget:$d), "bra\t$d",
              [/*(brcond pred:$_p, bb:$d)*/]>;
}

let isReturn = 1, isTerminator = 1, isBarrier = 1 in {
  def EXIT : InstPTX<(outs), (ins), "exit", [(PTXexit)]>;
  def RET  : InstPTX<(outs), (ins), "ret",  [(PTXret)]>;
}

let hasSideEffects = 1 in {
  def CALL : InstPTX<(outs), (ins), "call", [(PTXcall)]>;
}


///===- Spill Instructions ------------------------------------------------===//
// Special instructions used for stack spilling
def STACKSTOREI16 : InstPTX<(outs), (ins i32imm:$d, RegI16:$a),
                            "mov.u16\ts$d, $a", []>;
def STACKSTOREI32 : InstPTX<(outs), (ins i32imm:$d, RegI32:$a),
                            "mov.u32\ts$d, $a", []>;
def STACKSTOREI64 : InstPTX<(outs), (ins i32imm:$d, RegI64:$a),
                            "mov.u64\ts$d, $a", []>;
def STACKSTOREF32 : InstPTX<(outs), (ins i32imm:$d, RegF32:$a),
                            "mov.f32\ts$d, $a", []>;
def STACKSTOREF64 : InstPTX<(outs), (ins i32imm:$d, RegF64:$a),
                            "mov.f64\ts$d, $a", []>;

def STACKLOADI16 : InstPTX<(outs), (ins RegI16:$d, i32imm:$a),
                           "mov.u16\t$d, s$a", []>;
def STACKLOADI32 : InstPTX<(outs), (ins RegI32:$d, i32imm:$a),
                           "mov.u32\t$d, s$a", []>;
def STACKLOADI64 : InstPTX<(outs), (ins RegI64:$d, i32imm:$a),
                           "mov.u64\t$d, s$a", []>;
def STACKLOADF32 : InstPTX<(outs), (ins RegF32:$d, i32imm:$a),
                           "mov.f32\t$d, s$a", []>;
def STACKLOADF64 : InstPTX<(outs), (ins RegF64:$d, i32imm:$a),
                           "mov.f64\t$d, s$a", []>;

///===- Parameter Passing Pseudo-Instructions -----------------------------===//

def READPARAMPRED : InstPTX<(outs RegPred:$a), (ins i32imm:$b),
                            "mov.pred\t$a, %param$b", []>;
def READPARAMI16  : InstPTX<(outs RegI16:$a), (ins i32imm:$b),
                            "mov.b16\t$a, %param$b", []>;
def READPARAMI32  : InstPTX<(outs RegI32:$a), (ins i32imm:$b),
                            "mov.b32\t$a, %param$b", []>;
def READPARAMI64  : InstPTX<(outs RegI64:$a), (ins i32imm:$b),
                            "mov.b64\t$a, %param$b", []>;
def READPARAMF32  : InstPTX<(outs RegF32:$a), (ins i32imm:$b),
                            "mov.f32\t$a, %param$b", []>;
def READPARAMF64  : InstPTX<(outs RegF64:$a), (ins i32imm:$b),
                            "mov.f64\t$a, %param$b", []>;

def WRITEPARAMPRED : InstPTX<(outs), (ins RegPred:$a), "//w", []>;
def WRITEPARAMI16  : InstPTX<(outs), (ins RegI16:$a), "//w", []>;
def WRITEPARAMI32  : InstPTX<(outs), (ins RegI32:$a), "//w", []>;
def WRITEPARAMI64  : InstPTX<(outs), (ins RegI64:$a), "//w", []>;
def WRITEPARAMF32  : InstPTX<(outs), (ins RegF32:$a), "//w", []>;
def WRITEPARAMF64  : InstPTX<(outs), (ins RegF64:$a), "//w", []>;

// Call handling
// def ADJCALLSTACKUP :
//   InstPTX<(outs), (ins i32imm:$amt1, i32imm:$amt2), "",
//           [(PTXcallseq_end timm:$amt1, timm:$amt2)]>;
// def ADJCALLSTACKDOWN :
//   InstPTX<(outs), (ins i32imm:$amt), "",
//           [(PTXcallseq_start timm:$amt)]>;

///===- Intrinsic Instructions --------------------------------------------===//

include "PTXIntrinsicInstrInfo.td"
