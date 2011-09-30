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



//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//

// For floating-point instructions, we cannot just embed the pattern into the
// instruction definition since we need to muck around with the rounding mode,
// and I do not know how to insert constants into instructions directly from
// pattern matches.

//===- Floating-Point Instructions - 2 Operand Form -----------------------===//
multiclass PTX_FLOAT_2OP<string opcstr> {
  def rr32 : InstPTX<(outs RegF32:$d),
                     (ins RndMode:$r, RegF32:$a),
                     !strconcat(opcstr, "$r.f32\t$d, $a"), []>;
  def ri32 : InstPTX<(outs RegF32:$d),
                     (ins RndMode:$r, f32imm:$a),
                     !strconcat(opcstr, "$r.f32\t$d, $a"), []>;
  def rr64 : InstPTX<(outs RegF64:$d),
                     (ins RndMode:$r, RegF64:$a),
                     !strconcat(opcstr, "$r.f64\t$d, $a"), []>;
  def ri64 : InstPTX<(outs RegF64:$d),
                     (ins RndMode:$r, f64imm:$a),
                     !strconcat(opcstr, "$r.f64\t$d, $a"), []>;
}

//===- Floating-Point Instructions - 3 Operand Form -----------------------===//
multiclass PTX_FLOAT_3OP<string opcstr> {
  def rr32 : InstPTX<(outs RegF32:$d),
                     (ins RndMode:$r, RegF32:$a, RegF32:$b),
                     !strconcat(opcstr, "$r.f32\t$d, $a, $b"), []>;
  def ri32 : InstPTX<(outs RegF32:$d),
                     (ins RndMode:$r, RegF32:$a, f32imm:$b),
                     !strconcat(opcstr, "$r.f32\t$d, $a, $b"), []>;
  def rr64 : InstPTX<(outs RegF64:$d),
                     (ins RndMode:$r, RegF64:$a, RegF64:$b),
                     !strconcat(opcstr, "$r.f64\t$d, $a, $b"), []>;
  def ri64 : InstPTX<(outs RegF64:$d),
                     (ins RndMode:$r, RegF64:$a, f64imm:$b),
                     !strconcat(opcstr, "$r.f64\t$d, $a, $b"), []>;
}

//===- Floating-Point Instructions - 4 Operand Form -----------------------===//
multiclass PTX_FLOAT_4OP<string opcstr> {
  def rrr32 : InstPTX<(outs RegF32:$d),
                      (ins RndMode:$r, RegF32:$a, RegF32:$b, RegF32:$c),
                      !strconcat(opcstr, "$r.f32\t$d, $a, $b, $c"), []>;
  def rri32 : InstPTX<(outs RegF32:$d),
                      (ins RndMode:$r, RegF32:$a, RegF32:$b, f32imm:$c),
                      !strconcat(opcstr, "$r.f32\t$d, $a, $b, $c"), []>;
  def rii32 : InstPTX<(outs RegF32:$d),
                      (ins RndMode:$r, RegF32:$a, f32imm:$b, f32imm:$c),
                      !strconcat(opcstr, "$r.f32\t$d, $a, $b, $c"), []>;
  def rrr64 : InstPTX<(outs RegF64:$d),
                      (ins RndMode:$r, RegF64:$a, RegF64:$b, RegF64:$c),
                      !strconcat(opcstr, "$r.f64\t$d, $a, $b, $c"), []>;
  def rri64 : InstPTX<(outs RegF64:$d),
                      (ins RndMode:$r, RegF64:$a, RegF64:$b, f64imm:$c),
                      !strconcat(opcstr, "$r.f64\t$d, $a, $b, $c"), []>;
  def rii64 : InstPTX<(outs RegF64:$d),
                      (ins RndMode:$r, RegF64:$a, f64imm:$b, f64imm:$c),
                      !strconcat(opcstr, "$r.f64\t$d, $a, $b, $c"), []>;
}

//===- Integer Instructions - 3 Operand Form ------------------------------===//
multiclass PTX_INT3<string opcstr, SDNode opnode> {
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

//===- Bitwise Logic Instructions - 3 Operand Form ------------------------===//
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

//===- Integer Shift Instructions - 3 Operand Form ------------------------===//
multiclass PTX_INT3ntnc<string opcstr, SDNode opnode> {
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

//===- Set Predicate Instructions (Int) - 3/4 Operand Forms ---------------===//
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
              !strconcat("setp.", cmpstr, ".and.", regclsname,
                         "\t$p, $a, $b, $c"),
              [(set RegPred:$p, (and (setcc RC:$a, RC:$b, cmp), RegPred:$c))]>;
  def ri_and_r
    : InstPTX<(outs RegPred:$p), (ins RC:$a, immcls:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".and.", regclsname,
                         "\t$p, $a, $b, $c"),
              [(set RegPred:$p, (and (setcc RC:$a, imm:$b, cmp),
                                     RegPred:$c))]>;
  def rr_or_r
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".or.", regclsname,
                         "\t$p, $a, $b, $c"),
              [(set RegPred:$p, (or (setcc RC:$a, RC:$b, cmp), RegPred:$c))]>;
  def ri_or_r
    : InstPTX<(outs RegPred:$p), (ins RC:$a, immcls:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".or.", regclsname,
                         "\t$p, $a, $b, $c"),
              [(set RegPred:$p, (or (setcc RC:$a, imm:$b, cmp), RegPred:$c))]>;
  def rr_xor_r
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".xor.", regclsname,
                         "\t$p, $a, $b, $c"),
              [(set RegPred:$p, (xor (setcc RC:$a, RC:$b, cmp), RegPred:$c))]>;
  def ri_xor_r
    : InstPTX<(outs RegPred:$p), (ins RC:$a, immcls:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".xor.", regclsname,
                         "\t$p, $a, $b, $c"),
              [(set RegPred:$p, (xor (setcc RC:$a, imm:$b, cmp),
                                     RegPred:$c))]>;

  def rr_and_not_r
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".and.", regclsname,
                         "\t$p, $a, $b, !$c"),
              [(set RegPred:$p, (and (setcc RC:$a, RC:$b, cmp),
                                     (not RegPred:$c)))]>;
  def ri_and_not_r
    : InstPTX<(outs RegPred:$p), (ins RC:$a, immcls:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".and.", regclsname,
                         "\t$p, $a, $b, !$c"),
              [(set RegPred:$p, (and (setcc RC:$a, imm:$b, cmp),
                                     (not RegPred:$c)))]>;
  def rr_or_not_r
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".or.", regclsname,
                         "\t$p, $a, $b, !$c"),
              [(set RegPred:$p, (or (setcc RC:$a, RC:$b, cmp),
                                    (not RegPred:$c)))]>;
  def ri_or_not_r
    : InstPTX<(outs RegPred:$p), (ins RC:$a, immcls:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".or.", regclsname,
                         "\t$p, $a, $b, !$c"),
              [(set RegPred:$p, (or (setcc RC:$a, imm:$b, cmp),
                                    (not RegPred:$c)))]>;
  def rr_xor_not_r
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".xor.", regclsname,
                         "\t$p, $a, $b, !$c"),
              [(set RegPred:$p, (xor (setcc RC:$a, RC:$b, cmp),
                                     (not RegPred:$c)))]>;
  def ri_xor_not_r
    : InstPTX<(outs RegPred:$p), (ins RC:$a, immcls:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".xor.", regclsname,
                         "\t$p, $a, $b, !$c"),
              [(set RegPred:$p, (xor (setcc RC:$a, imm:$b, cmp),
                                     (not RegPred:$c)))]>;
}

//===- Set Predicate Instructions (FP) - 3/4 Operand Form -----------------===//
multiclass PTX_SETP_FP<RegisterClass RC, string regclsname, Operand immcls,
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

  def ri_u
    : InstPTX<(outs RegPred:$p), (ins RC:$a, immcls:$b),
              !strconcat("setp.", cmpstr, "u.", regclsname, "\t$p, $a, $b"),
              [(set RegPred:$p, (setcc RC:$a, fpimm:$b, ucmp))]>;
  def ri_o
    : InstPTX<(outs RegPred:$p), (ins RC:$a, immcls:$b),
              !strconcat("setp.", cmpstr, ".", regclsname, "\t$p, $a, $b"),
              [(set RegPred:$p, (setcc RC:$a, fpimm:$b, ocmp))]>;

  def rr_and_r_u
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, "u.and.", regclsname,
                         "\t$p, $a, $b, $c"),
              [(set RegPred:$p, (and (setcc RC:$a, RC:$b, ucmp),
                                     RegPred:$c))]>;
  def rr_and_r_o
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".and.", regclsname,
                         "\t$p, $a, $b, $c"),
              [(set RegPred:$p, (and (setcc RC:$a, RC:$b, ocmp),
                                     RegPred:$c))]>;

  def rr_or_r_u
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, "u.or.", regclsname,
                         "\t$p, $a, $b, $c"),
              [(set RegPred:$p, (or (setcc RC:$a, RC:$b, ucmp), RegPred:$c))]>;
  def rr_or_r_o
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".or.", regclsname,
                         "\t$p, $a, $b, $c"),
              [(set RegPred:$p, (or (setcc RC:$a, RC:$b, ocmp), RegPred:$c))]>;

  def rr_xor_r_u
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, "u.xor.", regclsname,
                         "\t$p, $a, $b, $c"),
              [(set RegPred:$p, (xor (setcc RC:$a, RC:$b, ucmp),
                                     RegPred:$c))]>;
  def rr_xor_r_o
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".xor.", regclsname,
                         "\t$p, $a, $b, $c"),
              [(set RegPred:$p, (xor (setcc RC:$a, RC:$b, ocmp),
                                     RegPred:$c))]>;

  def rr_and_not_r_u
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, "u.and.", regclsname,
                         "\t$p, $a, $b, !$c"),
              [(set RegPred:$p, (and (setcc RC:$a, RC:$b, ucmp),
                                     (not RegPred:$c)))]>;
  def rr_and_not_r_o
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".and.", regclsname,
                         "\t$p, $a, $b, !$c"),
              [(set RegPred:$p, (and (setcc RC:$a, RC:$b, ocmp),
                                     (not RegPred:$c)))]>;

  def rr_or_not_r_u
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, "u.or.", regclsname,
                         "\t$p, $a, $b, !$c"),
              [(set RegPred:$p, (or (setcc RC:$a, RC:$b, ucmp),
                                    (not RegPred:$c)))]>;
  def rr_or_not_r_o
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".or.", regclsname,
                         "\t$p, $a, $b, !$c"),
              [(set RegPred:$p, (or (setcc RC:$a, RC:$b, ocmp),
                                    (not RegPred:$c)))]>;

  def rr_xor_not_r_u
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, "u.xor.", regclsname,
                         "\t$p, $a, $b, !$c"),
              [(set RegPred:$p, (xor (setcc RC:$a, RC:$b, ucmp),
                                     (not RegPred:$c)))]>;
  def rr_xor_not_r_o
    : InstPTX<(outs RegPred:$p), (ins RC:$a, RC:$b, RegPred:$c),
              !strconcat("setp.", cmpstr, ".xor.", regclsname,
                         "\t$p, $a, $b, !$c"),
              [(set RegPred:$p, (xor (setcc RC:$a, RC:$b, ocmp),
                                     (not RegPred:$c)))]>;
}

//===- Select Predicate Instructions - 4 Operand Form ---------------------===//
multiclass PTX_SELP<RegisterClass RC, string regclsname, Operand immcls,
                    SDNode immnode> {
  def rr
    : InstPTX<(outs RC:$r), (ins RegPred:$a, RC:$b, RC:$c),
              !strconcat("selp.", regclsname, "\t$r, $b, $c, $a"),
              [(set RC:$r, (select RegPred:$a, RC:$b, RC:$c))]>;
  def ri
    : InstPTX<(outs RC:$r), (ins RegPred:$a, RC:$b, immcls:$c),
              !strconcat("selp.", regclsname, "\t$r, $b, $c, $a"),
              [(set RC:$r, (select RegPred:$a, RC:$b, immnode:$c))]>;
  def ii
    : InstPTX<(outs RC:$r), (ins RegPred:$a, immcls:$b, immcls:$c),
              !strconcat("selp.", regclsname, "\t$r, $b, $c, $a"),
              [(set RC:$r, (select RegPred:$a, immnode:$b, immnode:$c))]>;
}



//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

///===- Integer Arithmetic Instructions -----------------------------------===//

defm ADD : PTX_INT3<"add", add>;
defm SUB : PTX_INT3<"sub", sub>;
defm MUL : PTX_INT3<"mul.lo", mul>; // FIXME: Allow 32x32 -> 64 multiplies
defm DIV : PTX_INT3<"div", udiv>;
defm REM : PTX_INT3<"rem", urem>;

///===- Floating-Point Arithmetic Instructions ----------------------------===//

// FNEG
defm FNEG : PTX_FLOAT_2OP<"neg">;

// Standard Binary Operations
defm FADD : PTX_FLOAT_3OP<"add">;
defm FSUB : PTX_FLOAT_3OP<"sub">;
defm FMUL : PTX_FLOAT_3OP<"mul">;
defm FDIV : PTX_FLOAT_3OP<"div">;

// Multi-operation hybrid instructions
defm FMAD : PTX_FLOAT_4OP<"mad">, Requires<[SupportsFMA]>;


///===- Floating-Point Intrinsic Instructions -----------------------------===//

// SQRT
def FSQRTrr32 : InstPTX<(outs RegF32:$d), (ins RndMode:$r, RegF32:$a),
                        "sqrt$r.f32\t$d, $a", []>;
def FSQRTri32 : InstPTX<(outs RegF32:$d), (ins RndMode:$r, f32imm:$a),
                        "sqrt$r.f32\t$d, $a", []>;
def FSQRTrr64 : InstPTX<(outs RegF64:$d), (ins RndMode:$r, RegF64:$a),
                        "sqrt$r.f64\t$d, $a", []>;
def FSQRTri64 : InstPTX<(outs RegF64:$d), (ins RndMode:$r, f64imm:$a),
                        "sqrt$r.f64\t$d, $a", []>;

// SIN
def FSINrr32 : InstPTX<(outs RegF32:$d), (ins RndMode:$r, RegF32:$a),
                       "sin$r.f32\t$d, $a", []>;
def FSINri32 : InstPTX<(outs RegF32:$d), (ins RndMode:$r, f32imm:$a),
                       "sin$r.f32\t$d, $a", []>;
def FSINrr64 : InstPTX<(outs RegF64:$d), (ins RndMode:$r, RegF64:$a),
                       "sin$r.f64\t$d, $a", []>;
def FSINri64 : InstPTX<(outs RegF64:$d), (ins RndMode:$r, f64imm:$a),
                       "sin$r.f64\t$d, $a", []>;

// COS
def FCOSrr32 : InstPTX<(outs RegF32:$d), (ins RndMode:$r, RegF32:$a),
                       "cos$r.f32\t$d, $a", []>;
def FCOSri32 : InstPTX<(outs RegF32:$d), (ins RndMode:$r, f32imm:$a),
                       "cos$r.f32\t$d, $a", []>;
def FCOSrr64 : InstPTX<(outs RegF64:$d), (ins RndMode:$r, RegF64:$a),
                       "cos$r.f64\t$d, $a", []>;
def FCOSri64 : InstPTX<(outs RegF64:$d), (ins RndMode:$r, f64imm:$a),
                       "cos$r.f64\t$d, $a", []>;




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

defm SETPEQf32 : PTX_SETP_FP<RegF32, "f32", f32imm, SETUEQ, SETOEQ, "eq">;
defm SETPNEf32 : PTX_SETP_FP<RegF32, "f32", f32imm, SETUNE, SETONE, "ne">;
defm SETPLTf32 : PTX_SETP_FP<RegF32, "f32", f32imm, SETULT, SETOLT, "lt">;
defm SETPLEf32 : PTX_SETP_FP<RegF32, "f32", f32imm, SETULE, SETOLE, "le">;
defm SETPGTf32 : PTX_SETP_FP<RegF32, "f32", f32imm, SETUGT, SETOGT, "gt">;
defm SETPGEf32 : PTX_SETP_FP<RegF32, "f32", f32imm, SETUGE, SETOGE, "ge">;

// Compare f64

defm SETPEQf64 : PTX_SETP_FP<RegF64, "f64", f64imm, SETUEQ, SETOEQ, "eq">;
defm SETPNEf64 : PTX_SETP_FP<RegF64, "f64", f64imm, SETUNE, SETONE, "ne">;
defm SETPLTf64 : PTX_SETP_FP<RegF64, "f64", f64imm, SETULT, SETOLT, "lt">;
defm SETPLEf64 : PTX_SETP_FP<RegF64, "f64", f64imm, SETULE, SETOLE, "le">;
defm SETPGTf64 : PTX_SETP_FP<RegF64, "f64", f64imm, SETUGT, SETOGT, "gt">;
defm SETPGEf64 : PTX_SETP_FP<RegF64, "f64", f64imm, SETUGE, SETOGE, "ge">;

// .selp

defm SELPi16 : PTX_SELP<RegI16, "u16", i16imm, imm>;
defm SELPi32 : PTX_SELP<RegI32, "u32", i32imm, imm>;
defm SELPi64 : PTX_SELP<RegI64, "u64", i64imm, imm>;
defm SELPf32 : PTX_SELP<RegF32, "f32", f32imm, fpimm>;
defm SELPf64 : PTX_SELP<RegF64, "f64", f64imm, fpimm>;

///===- Logic and Shift Instructions --------------------------------------===//

defm SHL : PTX_INT3ntnc<"shl.b", PTXshl>;
defm SRL : PTX_INT3ntnc<"shr.u", PTXsrl>;
defm SRA : PTX_INT3ntnc<"shr.s", PTXsra>;

defm AND : PTX_LOGIC<"and", and>;
defm OR  : PTX_LOGIC<"or",  or>;
defm XOR : PTX_LOGIC<"xor", xor>;

///===- Data Movement and Conversion Instructions -------------------------===//

// any_extend
// Implement the anyext instruction in terms of the PTX cvt instructions.
//def : Pat<(i32 (anyext RegI16:$a)), (CVT_u32_u16 RegI16:$a)>;
//def : Pat<(i64 (anyext RegI16:$a)), (CVT_u64_u16 RegI16:$a)>;
//def : Pat<(i64 (anyext RegI32:$a)), (CVT_u64_u32 RegI32:$a)>;

// bitconvert
// These instructions implement the bit-wise conversion between integer and
// floating-point types.
def MOVi32f32
  : InstPTX<(outs RegI32:$d), (ins RegF32:$a), "mov.b32\t$d, $a", []>;
def MOVf32i32
  : InstPTX<(outs RegF32:$d), (ins RegI32:$a), "mov.b32\t$d, $a", []>;
def MOVi64f64
  : InstPTX<(outs RegI64:$d), (ins RegF64:$a), "mov.b64\t$d, $a", []>;
def MOVf64i64
  : InstPTX<(outs RegF64:$d), (ins RegI64:$a), "mov.b64\t$d, $a", []>;

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

// PTX cvt instructions
// Note all of these may actually be used, we just define all possible patterns
// here (that make sense).
// FIXME: Can we collapse this somehow into a multiclass def?

// To i16
def CVTu16u32
  : InstPTX<(outs RegI16:$d), (ins RegI32:$a), "cvt.u16.u32\t$d, $a", []>;
def CVTu16u64
  : InstPTX<(outs RegI16:$d), (ins RegI64:$a), "cvt.u16.u64\t$d, $a", []>;
def CVTu16f32
  : InstPTX<(outs RegI16:$d), (ins RndMode:$r, RegF32:$a),
            "cvt$r.u16.f32\t$d, $a", []>;
def CVTs16f32
  : InstPTX<(outs RegI16:$d), (ins RndMode:$r, RegF32:$a),
            "cvt$r.s16.f32\t$d, $a", []>;
def CVTu16f64
  : InstPTX<(outs RegI16:$d), (ins RndMode:$r, RegF64:$a),
            "cvt$r.u16.f64\t$d, $a", []>;
def CVTs16f64
  : InstPTX<(outs RegI16:$d), (ins RndMode:$r, RegF64:$a),
            "cvt$r.s16.f64\t$d, $a", []>;

// To i32
def CVTu32u16
  : InstPTX<(outs RegI32:$d), (ins RegI16:$a), "cvt.u32.u16\t$d, $a", []>;
def CVTs32s16
  : InstPTX<(outs RegI32:$d), (ins RegI16:$a), "cvt.s32.s16\t$d, $a", []>;
def CVTu32u64
  : InstPTX<(outs RegI32:$d), (ins RegI64:$a), "cvt.u32.u64\t$d, $a", []>;
def CVTu32f32
  : InstPTX<(outs RegI32:$d), (ins RndMode:$r, RegF32:$a),
            "cvt$r.u32.f32\t$d, $a", []>;
def CVTs32f32
  : InstPTX<(outs RegI32:$d), (ins RndMode:$r, RegF32:$a),
            "cvt$r.s32.f32\t$d, $a", []>;
def CVTu32f64
  : InstPTX<(outs RegI32:$d), (ins RndMode:$r, RegF64:$a),
            "cvt$r.u32.f64\t$d, $a", []>;
def CVTs32f64
  : InstPTX<(outs RegI32:$d), (ins RndMode:$r, RegF64:$a),
            "cvt$r.s32.f64\t$d, $a", []>;

// To i64
def CVTu64u16
  : InstPTX<(outs RegI64:$d), (ins RegI16:$a), "cvt.u64.u16\t$d, $a", []>;
def CVTs64s16
  : InstPTX<(outs RegI64:$d), (ins RegI16:$a), "cvt.s64.s16\t$d, $a", []>;
def CVTu64u32
  : InstPTX<(outs RegI64:$d), (ins RegI32:$a), "cvt.u64.u32\t$d, $a", []>;
def CVTs64s32
  : InstPTX<(outs RegI64:$d), (ins RegI32:$a), "cvt.s64.s32\t$d, $a", []>;
def CVTu64f32
  : InstPTX<(outs RegI64:$d), (ins RndMode:$r, RegF32:$a),
            "cvt$r.u64.f32\t$d, $a", []>;
def CVTs64f32
  : InstPTX<(outs RegI64:$d), (ins RndMode:$r, RegF32:$a),
            "cvt$r.s64.f32\t$d, $a", []>;
def CVTu64f64
  : InstPTX<(outs RegI64:$d), (ins RndMode:$r, RegF64:$a),
            "cvt$r.u64.f64\t$d, $a", []>;
def CVTs64f64
  : InstPTX<(outs RegI64:$d), (ins RndMode:$r, RegF64:$a),
            "cvt$r.s64.f64\t$d, $a", []>;

// To f32
def CVTf32u16
  : InstPTX<(outs RegF32:$d), (ins RndMode:$r, RegI16:$a),
            "cvt$r.f32.u16\t$d, $a", []>;
def CVTf32s16
  : InstPTX<(outs RegF32:$d), (ins RndMode:$r, RegI16:$a),
            "cvt$r.f32.s16\t$d, $a", []>;
def CVTf32u32
  : InstPTX<(outs RegF32:$d), (ins RndMode:$r, RegI32:$a),
            "cvt$r.f32.u32\t$d, $a", []>;
def CVTf32s32
  : InstPTX<(outs RegF32:$d), (ins RndMode:$r, RegI32:$a),
            "cvt$r.f32.s32\t$d, $a", []>;
def CVTf32u64
  : InstPTX<(outs RegF32:$d), (ins RndMode:$r, RegI64:$a),
            "cvt$r.f32.u64\t$d, $a", []>;
def CVTf32s64
  : InstPTX<(outs RegF32:$d), (ins RndMode:$r, RegI64:$a),
            "cvt$r.f32.s64\t$d, $a", []>;
def CVTf32f64
  : InstPTX<(outs RegF32:$d), (ins RndMode:$r, RegF64:$a),
            "cvt$r.f32.f64\t$d, $a", []>;

// To f64
def CVTf64u16
  : InstPTX<(outs RegF64:$d), (ins RndMode:$r, RegI16:$a),
            "cvt$r.f64.u16\t$d, $a", []>;
def CVTf64s16
  : InstPTX<(outs RegF64:$d), (ins RndMode:$r, RegI16:$a),
            "cvt$r.f64.s16\t$d, $a", []>;
def CVTf64u32
  : InstPTX<(outs RegF64:$d), (ins RndMode:$r, RegI32:$a),
            "cvt$r.f64.u32\t$d, $a", []>;
def CVTf64s32
  : InstPTX<(outs RegF64:$d), (ins RndMode:$r, RegI32:$a),
            "cvt$r.f64.s32\t$d, $a", []>;
def CVTf64u64
  : InstPTX<(outs RegF64:$d), (ins RndMode:$r, RegI64:$a),
            "cvt$r.f64.u64\t$d, $a", []>;
def CVTf64s64
  : InstPTX<(outs RegF64:$d), (ins RndMode:$r, RegI64:$a),
            "cvt$r.f64.s64\t$d, $a", []>;
def CVTf64f32
  : InstPTX<(outs RegF64:$d), (ins RegF32:$a), "cvt.f64.f32\t$d, $a", []>;

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


//===----------------------------------------------------------------------===//
// Instruction Selection Patterns
//===----------------------------------------------------------------------===//

// FADD
def : Pat<(f32 (fadd RegF32:$a, RegF32:$b)),
          (FADDrr32 RndDefault, RegF32:$a, RegF32:$b)>;
def : Pat<(f32 (fadd RegF32:$a, fpimm:$b)),
          (FADDri32 RndDefault, RegF32:$a, fpimm:$b)>;
def : Pat<(f64 (fadd RegF64:$a, RegF64:$b)),
          (FADDrr64 RndDefault, RegF64:$a, RegF64:$b)>;
def : Pat<(f64 (fadd RegF64:$a, fpimm:$b)),
          (FADDri64 RndDefault, RegF64:$a, fpimm:$b)>;

// FSUB
def : Pat<(f32 (fsub RegF32:$a, RegF32:$b)),
          (FSUBrr32 RndDefault, RegF32:$a, RegF32:$b)>;
def : Pat<(f32 (fsub RegF32:$a, fpimm:$b)),
          (FSUBri32 RndDefault, RegF32:$a, fpimm:$b)>;
def : Pat<(f64 (fsub RegF64:$a, RegF64:$b)),
          (FSUBrr64 RndDefault, RegF64:$a, RegF64:$b)>;
def : Pat<(f64 (fsub RegF64:$a, fpimm:$b)),
          (FSUBri64 RndDefault, RegF64:$a, fpimm:$b)>;

// FMUL
def : Pat<(f32 (fmul RegF32:$a, RegF32:$b)),
          (FMULrr32 RndDefault, RegF32:$a, RegF32:$b)>;
def : Pat<(f32 (fmul RegF32:$a, fpimm:$b)),
          (FMULri32 RndDefault, RegF32:$a, fpimm:$b)>;
def : Pat<(f64 (fmul RegF64:$a, RegF64:$b)),
          (FMULrr64 RndDefault, RegF64:$a, RegF64:$b)>;
def : Pat<(f64 (fmul RegF64:$a, fpimm:$b)),
          (FMULri64 RndDefault, RegF64:$a, fpimm:$b)>;

// FDIV
def : Pat<(f32 (fdiv RegF32:$a, RegF32:$b)),
          (FDIVrr32 RndDefault, RegF32:$a, RegF32:$b)>;
def : Pat<(f32 (fdiv RegF32:$a, fpimm:$b)),
          (FDIVri32 RndDefault, RegF32:$a, fpimm:$b)>;
def : Pat<(f64 (fdiv RegF64:$a, RegF64:$b)),
          (FDIVrr64 RndDefault, RegF64:$a, RegF64:$b)>;
def : Pat<(f64 (fdiv RegF64:$a, fpimm:$b)),
          (FDIVri64 RndDefault, RegF64:$a, fpimm:$b)>;

// FMUL+FADD
def : Pat<(f32 (fadd (fmul RegF32:$a, RegF32:$b), RegF32:$c)),
          (FMADrrr32 RndDefault, RegF32:$a, RegF32:$b, RegF32:$c)>;
def : Pat<(f32 (fadd (fmul RegF32:$a, RegF32:$b), fpimm:$c)),
          (FMADrri32 RndDefault, RegF32:$a, RegF32:$b, fpimm:$c)>;
def : Pat<(f32 (fadd (fmul RegF32:$a, fpimm:$b), fpimm:$c)),
          (FMADrrr32 RndDefault, RegF32:$a, fpimm:$b, fpimm:$c)>;
def : Pat<(f32 (fadd (fmul RegF32:$a, RegF32:$b), fpimm:$c)),
          (FMADrri32 RndDefault, RegF32:$a, RegF32:$b, fpimm:$c)>;
def : Pat<(f64 (fadd (fmul RegF64:$a, RegF64:$b), RegF64:$c)),
          (FMADrrr64 RndDefault, RegF64:$a, RegF64:$b, RegF64:$c)>;
def : Pat<(f64 (fadd (fmul RegF64:$a, RegF64:$b), fpimm:$c)),
          (FMADrri64 RndDefault, RegF64:$a, RegF64:$b, fpimm:$c)>;
def : Pat<(f64 (fadd (fmul RegF64:$a, fpimm:$b), fpimm:$c)),
          (FMADrri64 RndDefault, RegF64:$a, fpimm:$b, fpimm:$c)>;

// FNEG
def : Pat<(f32 (fneg RegF32:$a)), (FNEGrr32 RndDefault, RegF32:$a)>;
def : Pat<(f32 (fneg fpimm:$a)), (FNEGri32 RndDefault, fpimm:$a)>;
def : Pat<(f64 (fneg RegF64:$a)), (FNEGrr64 RndDefault, RegF64:$a)>;
def : Pat<(f64 (fneg fpimm:$a)), (FNEGri64 RndDefault, fpimm:$a)>;

// FSQRT
def : Pat<(f32 (fsqrt RegF32:$a)), (FSQRTrr32 RndDefault, RegF32:$a)>;
def : Pat<(f32 (fsqrt fpimm:$a)), (FSQRTri32 RndDefault, fpimm:$a)>;
def : Pat<(f64 (fsqrt RegF64:$a)), (FSQRTrr64 RndDefault, RegF64:$a)>;
def : Pat<(f64 (fsqrt fpimm:$a)), (FSQRTri64 RndDefault, fpimm:$a)>;

// FSIN
def : Pat<(f32 (fsin RegF32:$a)), (FSINrr32 RndDefault, RegF32:$a)>;
def : Pat<(f32 (fsin fpimm:$a)), (FSINri32 RndDefault, fpimm:$a)>;
def : Pat<(f64 (fsin RegF64:$a)), (FSINrr64 RndDefault, RegF64:$a)>;
def : Pat<(f64 (fsin fpimm:$a)), (FSINri64 RndDefault, fpimm:$a)>;

// FCOS
def : Pat<(f32 (fcos RegF32:$a)), (FCOSrr32 RndDefault, RegF32:$a)>;
def : Pat<(f32 (fcos fpimm:$a)), (FCOSri32 RndDefault, fpimm:$a)>;
def : Pat<(f64 (fcos RegF64:$a)), (FCOSrr64 RndDefault, RegF64:$a)>;
def : Pat<(f64 (fcos fpimm:$a)), (FCOSri64 RndDefault, fpimm:$a)>;

// Type conversion notes:
// - PTX does not directly support converting a predicate to a value, so we
//   use a select instruction to select either 0 or 1 (integer or fp) based
//   on the truth value of the predicate.
// - PTX does not directly support converting to a predicate type, so we fake it
//   by performing a greater-than test between the value and zero.  This follows
//   the C convention that any non-zero value is equivalent to 'true'.

// Conversion to pred
def : Pat<(i1 (trunc RegI16:$a)),      (SETPGTu16ri RegI16:$a, 0)>;
def : Pat<(i1 (trunc RegI32:$a)),      (SETPGTu32ri RegI32:$a, 0)>;
def : Pat<(i1 (trunc RegI64:$a)),      (SETPGTu64ri RegI64:$a, 0)>;
def : Pat<(i1 (fp_to_uint RegF32:$a)), (SETPGTu32ri (MOVi32f32 RegF32:$a), 0)>;
def : Pat<(i1 (fp_to_uint RegF64:$a)), (SETPGTu64ri (MOVi64f64 RegF64:$a), 0)>;

// Conversion to u16
def : Pat<(i16 (anyext RegPred:$a)),    (SELPi16ii RegPred:$a, 1, 0)>;
def : Pat<(i16 (sext RegPred:$a)),      (SELPi16ii RegPred:$a, 0xFFFF, 0)>;
def : Pat<(i16 (zext RegPred:$a)),      (SELPi16ii RegPred:$a, 1, 0)>;
def : Pat<(i16 (trunc RegI32:$a)),      (CVTu16u32 RegI32:$a)>;
def : Pat<(i16 (trunc RegI64:$a)),      (CVTu16u64 RegI64:$a)>;
def : Pat<(i16 (fp_to_uint RegF32:$a)), (CVTu16f32 RndDefault, RegF32:$a)>;
def : Pat<(i16 (fp_to_sint RegF32:$a)), (CVTs16f32 RndDefault, RegF32:$a)>;
def : Pat<(i16 (fp_to_uint RegF64:$a)), (CVTu16f64 RndDefault, RegF64:$a)>;
def : Pat<(i16 (fp_to_sint RegF64:$a)), (CVTs16f64 RndDefault, RegF64:$a)>;

// Conversion to u32
def : Pat<(i32 (anyext RegPred:$a)),    (SELPi32ii RegPred:$a, 1, 0)>;
def : Pat<(i32 (sext RegPred:$a)),      (SELPi32ii RegPred:$a, 0xFFFFFFFF, 0)>;
def : Pat<(i32 (zext RegPred:$a)),      (SELPi32ii RegPred:$a, 1, 0)>;
def : Pat<(i32 (anyext RegI16:$a)),     (CVTu32u16 RegI16:$a)>;
def : Pat<(i32 (sext RegI16:$a)),       (CVTs32s16 RegI16:$a)>;
def : Pat<(i32 (zext RegI16:$a)),       (CVTu32u16 RegI16:$a)>;
def : Pat<(i32 (trunc RegI64:$a)),      (CVTu32u64 RegI64:$a)>;
def : Pat<(i32 (fp_to_uint RegF32:$a)), (CVTu32f32 RndDefault, RegF32:$a)>;
def : Pat<(i32 (fp_to_sint RegF32:$a)), (CVTs32f32 RndDefault, RegF32:$a)>;
def : Pat<(i32 (fp_to_uint RegF64:$a)), (CVTu32f64 RndDefault, RegF64:$a)>;
def : Pat<(i32 (fp_to_sint RegF64:$a)), (CVTs32f64 RndDefault, RegF64:$a)>;
def : Pat<(i32 (bitconvert RegF32:$a)), (MOVi32f32 RegF32:$a)>;

// Conversion to u64
def : Pat<(i64 (anyext RegPred:$a)),    (SELPi64ii RegPred:$a, 1, 0)>;
def : Pat<(i64 (sext RegPred:$a)),      (SELPi64ii RegPred:$a,
                                         0xFFFFFFFFFFFFFFFF, 0)>;
def : Pat<(i64 (zext RegPred:$a)),      (SELPi64ii RegPred:$a, 1, 0)>;
def : Pat<(i64 (anyext RegI16:$a)),     (CVTu64u16 RegI16:$a)>;
def : Pat<(i64 (sext RegI16:$a)),       (CVTs64s16 RegI16:$a)>;
def : Pat<(i64 (zext RegI16:$a)),       (CVTu64u16 RegI16:$a)>;
def : Pat<(i64 (anyext RegI32:$a)),     (CVTu64u32 RegI32:$a)>;
def : Pat<(i64 (sext RegI32:$a)),       (CVTs64s32 RegI32:$a)>;
def : Pat<(i64 (zext RegI32:$a)),       (CVTu64u32 RegI32:$a)>;
def : Pat<(i64 (fp_to_uint RegF32:$a)), (CVTu64f32 RndDefault, RegF32:$a)>;
def : Pat<(i64 (fp_to_sint RegF32:$a)), (CVTs64f32 RndDefault, RegF32:$a)>;
def : Pat<(i64 (fp_to_uint RegF64:$a)), (CVTu64f64 RndDefault, RegF64:$a)>;
def : Pat<(i64 (fp_to_sint RegF64:$a)), (CVTs64f64 RndDefault, RegF64:$a)>;
def : Pat<(i64 (bitconvert RegF64:$a)), (MOVi64f64 RegF64:$a)>;

// Conversion to f32
def : Pat<(f32 (uint_to_fp RegPred:$a)), (SELPf32rr RegPred:$a,
                                        (MOVf32i32 0x3F800000), (MOVf32i32 0))>;
def : Pat<(f32 (uint_to_fp RegI16:$a)),  (CVTf32u16 RndDefault, RegI16:$a)>;
def : Pat<(f32 (sint_to_fp RegI16:$a)),  (CVTf32s16 RndDefault, RegI16:$a)>;
def : Pat<(f32 (uint_to_fp RegI32:$a)),  (CVTf32u32 RndDefault, RegI32:$a)>;
def : Pat<(f32 (sint_to_fp RegI32:$a)),  (CVTf32s32 RndDefault, RegI32:$a)>;
def : Pat<(f32 (uint_to_fp RegI64:$a)),  (CVTf32u64 RndDefault, RegI64:$a)>;
def : Pat<(f32 (sint_to_fp RegI64:$a)),  (CVTf32s64 RndDefault, RegI64:$a)>;
def : Pat<(f32 (fround RegF64:$a)),      (CVTf32f64 RndDefault, RegF64:$a)>;
def : Pat<(f32 (bitconvert RegI32:$a)),  (MOVf32i32 RegI32:$a)>;

// Conversion to f64
def : Pat<(f64 (uint_to_fp RegPred:$a)), (SELPf64rr RegPred:$a,
                                (MOVf64i64 0x3F80000000000000), (MOVf64i64 0))>;
def : Pat<(f64 (uint_to_fp RegI16:$a)), (CVTf64u16 RndDefault, RegI16:$a)>;
def : Pat<(f64 (sint_to_fp RegI16:$a)), (CVTf64s16 RndDefault, RegI16:$a)>;
def : Pat<(f64 (uint_to_fp RegI32:$a)), (CVTf64u32 RndDefault, RegI32:$a)>;
def : Pat<(f64 (sint_to_fp RegI32:$a)), (CVTf64s32 RndDefault, RegI32:$a)>;
def : Pat<(f64 (uint_to_fp RegI64:$a)), (CVTf64u64 RndDefault, RegI64:$a)>;
def : Pat<(f64 (sint_to_fp RegI64:$a)), (CVTf64s64 RndDefault, RegI64:$a)>;
def : Pat<(f64 (fextend RegF32:$a)),    (CVTf64f32 RegF32:$a)>;
def : Pat<(f64 (bitconvert RegI64:$a)), (MOVf64i64 RegI64:$a)>;


///===- Intrinsic Instructions --------------------------------------------===//
include "PTXIntrinsicInstrInfo.td"

///===- Load/Store Instructions -------------------------------------------===//
include "PTXInstrLoadStore.td"

