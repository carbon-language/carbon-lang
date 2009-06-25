//===- ARMInstrThumb2.td - Thumb2 support for ARM -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Thumb2 instruction set.
//
//===----------------------------------------------------------------------===//

// Shifted operands. No register controlled shifts for Thumb2.
// Note: We do not support rrx shifted operands yet.
def t2_so_reg : Operand<i32>,    // reg imm
                ComplexPattern<i32, 2, "SelectThumb2ShifterOperandReg",
                               [shl,srl,sra,rotr]> {
  let PrintMethod = "printSOOperand";
  let MIOperandInfo = (ops GPR, i32imm);
}

// t2_so_imm_XFORM - Return a t2_so_imm value packed into the format 
// described for t2_so_imm def below.
def t2_so_imm_XFORM : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(
        ARM_AM::getT2SOImmVal(N->getZExtValue()), MVT::i32);
}]>;

// t2_so_imm_not_XFORM - Return the complement of a t2_so_imm value
def t2_so_imm_not_XFORM : SDNodeXForm<imm, [{
    return CurDAG->getTargetConstant(
        ARM_AM::getT2SOImmVal(~((uint32_t)N->getZExtValue())), MVT::i32);
}]>;

// t2_so_imm_neg_XFORM - Return the negation of a t2_so_imm value
def t2_so_imm_neg_XFORM : SDNodeXForm<imm, [{
    return CurDAG->getTargetConstant(
        ARM_AM::getT2SOImmVal(-((int)N->getZExtValue())), MVT::i32);
}]>;

// t2_so_imm - Match a 32-bit immediate operand, which is an
// 8-bit immediate rotated by an arbitrary number of bits, or an 8-bit
// immediate splatted into multiple bytes of the word. t2_so_imm values are
// represented in the imm field in the same 12-bit form that they are encoded
// into t2_so_imm instructions: the 8-bit immediate is the least significant bits
// [bits 0-7], the 4-bit shift/splat amount is the next 4 bits [bits 8-11].
def t2_so_imm : Operand<i32>,
                PatLeaf<(imm), [{
       return ARM_AM::getT2SOImmVal((uint32_t)N->getZExtValue()) != -1;
     }], t2_so_imm_XFORM> {
  let PrintMethod = "printT2SOImmOperand";
}

// t2_so_imm_not - Match an immediate that is a complement 
// of a t2_so_imm.
def t2_so_imm_not : Operand<i32>,
                    PatLeaf<(imm), [{
       return ARM_AM::getT2SOImmVal(~((uint32_t)N->getZExtValue())) != -1;
     }], t2_so_imm_not_XFORM> {
  let PrintMethod = "printT2SOImmOperand";
}

// t2_so_imm_neg - Match an immediate that is a negation of a t2_so_imm.
def t2_so_imm_neg : Operand<i32>,
                    PatLeaf<(imm), [{
       return ARM_AM::getT2SOImmVal(-((int)N->getZExtValue())) != -1;
     }], t2_so_imm_neg_XFORM> {
  let PrintMethod = "printT2SOImmOperand";
}

/// imm1_31 predicate - True if the 32-bit immediate is in the range [1,31].
def imm1_31 : PatLeaf<(i32 imm), [{
  return (int32_t)N->getZExtValue() >= 1 && (int32_t)N->getZExtValue() < 32;
}]>;

/// imm0_4095 predicate - True if the 32-bit immediate is in the range [0.4095].
def imm0_4095 : PatLeaf<(i32 imm), [{
  return (uint32_t)N->getZExtValue() < 4096;
}]>;

def imm0_4095_neg : PatLeaf<(i32 imm), [{ 
 return (uint32_t)(-N->getZExtValue()) < 4096; 
}], imm_neg_XFORM>; 

/// imm0_65535 predicate - True if the 32-bit immediate is in the range 
/// [0.65535].
def imm0_65535 : PatLeaf<(i32 imm), [{
  return (uint32_t)N->getZExtValue() < 65536;
}]>;


/// bf_inv_mask_imm predicate - An AND mask to clear an arbitrary width bitfield
/// e.g., 0xf000ffff
def bf_inv_mask_imm : Operand<i32>,
                      PatLeaf<(imm), [{ 
  uint32_t v = (uint32_t)N->getZExtValue();
  if (v == 0xffffffff)
    return 0;
  // naive checker. should do better, but simple is best for now since it's
  // more likely to be correct.
  while (v & 1) v >>= 1;    // shift off the leading 1's
  if (v)
    {
      while (!(v & 1)) v >>=1;  // shift off the mask
      while (v & 1) v >>= 1;    // shift off the trailing 1's
    }
  // if this is a mask for clearing a bitfield, what's left should be zero.
  return (v == 0);
}] > {
  let PrintMethod = "printBitfieldInvMaskImmOperand";
}

/// Split a 32-bit immediate into two 16 bit parts.
def t2_lo16 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant((uint32_t)N->getZExtValue() & 0xffff,
                                   MVT::i32);
}]>;

def t2_hi16 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant((uint32_t)N->getZExtValue() >> 16, MVT::i32);
}]>;

def t2_lo16AllZero : PatLeaf<(i32 imm), [{
  // Returns true if all low 16-bits are 0.
  return (((uint32_t)N->getZExtValue()) & 0xFFFFUL) == 0;
  }], t2_hi16>;

//===----------------------------------------------------------------------===//
//  Thumb2 to cover the functionality of the ARM instruction set.
//

/// T2I_un_irs - Defines a set of (op reg, {so_imm|r|so_reg}) patterns for a
//  unary operation that produces a value.
multiclass T2I_un_irs<string opc, PatFrag opnode, bit Cheap = 0, bit ReMat = 0>{
   // shifted imm
   def i : T2I<(outs GPR:$dst), (ins t2_so_imm:$src),
                !strconcat(opc, " $dst, $src"),
                [(set GPR:$dst, (opnode t2_so_imm:$src))]> {
     let isAsCheapAsAMove = Cheap;
     let isReMaterializable = ReMat;
   }
   // register
   def r : T2I<(outs GPR:$dst), (ins GPR:$src),
                !strconcat(opc, " $dst, $src"),
                [(set GPR:$dst, (opnode GPR:$src))]>;
   // shifted register
   def s : T2I<(outs GPR:$dst), (ins t2_so_reg:$src),
                !strconcat(opc, " $dst, $src"),
                [(set GPR:$dst, (opnode t2_so_reg:$src))]>;
}

/// T2I_bin_irs - Defines a set of (op reg, {so_imm|r|so_reg}) patterns for a
//  binary operation that produces a value.
multiclass T2I_bin_irs<string opc, PatFrag opnode> {
   // shifted imm
   def ri : T2I<(outs GPR:$dst), (ins GPR:$lhs, t2_so_imm:$rhs),
                !strconcat(opc, " $dst, $lhs, $rhs"),
                [(set GPR:$dst, (opnode GPR:$lhs, t2_so_imm:$rhs))]>;
   // register
   def rr : T2I<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs),
                !strconcat(opc, " $dst, $lhs, $rhs"),
                [(set GPR:$dst, (opnode GPR:$lhs, GPR:$rhs))]>;
   // shifted register
   def rs : T2I<(outs GPR:$dst), (ins GPR:$lhs, t2_so_reg:$rhs),
                !strconcat(opc, " $dst, $lhs, $rhs"),
                [(set GPR:$dst, (opnode GPR:$lhs, t2_so_reg:$rhs))]>;
}

/// T2I_rbin_is - Same as T2I_bin_irs except the order of operands are reversed.
multiclass T2I_rbin_irs<string opc, PatFrag opnode> {
   // shifted imm
   def ri : T2I<(outs GPR:$dst), (ins GPR:$rhs, t2_so_imm:$lhs),
                !strconcat(opc, " $dst, $rhs, $lhs"),
                [(set GPR:$dst, (opnode t2_so_imm:$lhs, GPR:$rhs))]>;
   // register
   def rr : T2I<(outs GPR:$dst), (ins GPR:$rhs, GPR:$lhs),
                !strconcat(opc, " $dst, $rhs, $lhs"),
                [(set GPR:$dst, (opnode GPR:$lhs, GPR:$rhs))]>;
   // shifted register
   def rs : T2I<(outs GPR:$dst), (ins GPR:$rhs, t2_so_reg:$lhs),
                !strconcat(opc, " $dst, $rhs, $lhs"),
                [(set GPR:$dst, (opnode t2_so_reg:$lhs, GPR:$rhs))]>;
}

/// T2I_bin_s_irs - Similar to T2I_bin_irs except it sets the 's' bit so the
/// instruction modifies the CPSR register.
let Defs = [CPSR] in {
multiclass T2I_bin_s_irs<string opc, PatFrag opnode> {
   // shifted imm
   def ri : T2I<(outs GPR:$dst), (ins GPR:$lhs, t2_so_imm:$rhs),
                !strconcat(opc, "s $dst, $lhs, $rhs"),
                [(set GPR:$dst, (opnode GPR:$lhs, t2_so_imm:$rhs))]>;
   // register
   def rr : T2I<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs),
                !strconcat(opc, " $dst, $lhs, $rhs"),
                [(set GPR:$dst, (opnode GPR:$lhs, GPR:$rhs))]>;
   // shifted register
   def rs : T2I<(outs GPR:$dst), (ins GPR:$lhs, t2_so_reg:$rhs),
                !strconcat(opc, "s $dst, $lhs, $rhs"),
                [(set GPR:$dst, (opnode GPR:$lhs, t2_so_reg:$rhs))]>;
}
}

/// T2I_rbin_s_irs - Same as T2I_bin_s_irs except the order of operands are
/// reversed.
let Defs = [CPSR] in {
multiclass T2I_rbin_s_irs<string opc, PatFrag opnode> {
   // shifted imm
   def ri : T2I<(outs GPR:$dst), (ins GPR:$rhs, t2_so_imm:$lhs),
                !strconcat(opc, "s $dst, $rhs, $lhs"),
                [(set GPR:$dst, (opnode t2_so_imm:$lhs, GPR:$rhs))]>;
   // register
   def rr : T2I<(outs GPR:$dst), (ins GPR:$rhs, GPR:$lhs),
                !strconcat(opc, " $dst, $rhs, $lhs"),
                [(set GPR:$dst, (opnode GPR:$lhs, GPR:$rhs))]>;
   // shifted register
   def rs : T2I<(outs GPR:$dst), (ins GPR:$rhs, t2_so_reg:$lhs),
                !strconcat(opc, "s $dst, $rhs, $lhs"),
                [(set GPR:$dst, (opnode t2_so_reg:$lhs, GPR:$rhs))]>;
}
}

/// T2I_bin_ii12rs - Defines a set of (op reg, {so_imm|imm0_4095|r|so_reg})
/// patterns for a binary operation that produces a value.
multiclass T2I_bin_ii12rs<string opc, PatFrag opnode> {
   // shifted imm
   def ri : T2I<(outs GPR:$dst), (ins GPR:$lhs, t2_so_imm:$rhs),
                !strconcat(opc, " $dst, $lhs, $rhs"),
                [(set GPR:$dst, (opnode GPR:$lhs, t2_so_imm:$rhs))]>;
   // 12-bit imm
   def ri12 : T2I<(outs GPR:$dst), (ins GPR:$lhs, i32imm:$rhs),
                !strconcat(opc, "w $dst, $lhs, $rhs"),
                [(set GPR:$dst, (opnode GPR:$lhs, imm0_4095:$rhs))]>;
   // register
   def rr : T2I<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs),
                !strconcat(opc, " $dst, $lhs, $rhs"),
                [(set GPR:$dst, (opnode GPR:$lhs, GPR:$rhs))]>;
   // shifted register
   def rs : T2I<(outs GPR:$dst), (ins GPR:$lhs, t2_so_reg:$rhs),
                !strconcat(opc, " $dst, $lhs, $rhs"),
                [(set GPR:$dst, (opnode GPR:$lhs, t2_so_reg:$rhs))]>;
}

/// T2I_bin_c_irs - Defines a set of (op reg, {so_imm|r|so_reg}) patterns for a
//  binary operation that produces a value and set the carry bit. It can also
/// optionally set CPSR.
let Uses = [CPSR] in {
multiclass T2I_bin_c_irs<string opc, PatFrag opnode> {
   // shifted imm
   def ri : T2I<(outs GPR:$dst), (ins GPR:$lhs, t2_so_imm:$rhs, cc_out:$s),
                !strconcat(opc, "${s} $dst, $lhs, $rhs"),
                [(set GPR:$dst, (opnode GPR:$lhs, t2_so_imm:$rhs))]>;
   // register
   def rr : T2I<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs, cc_out:$s),
                !strconcat(opc, "${s} $dst, $lhs, $rhs"),
                [(set GPR:$dst, (opnode GPR:$lhs, GPR:$rhs))]>;
   // shifted register
   def rs : T2I<(outs GPR:$dst), (ins GPR:$lhs, t2_so_reg:$rhs, cc_out:$s),
                !strconcat(opc, "${s} $dst, $lhs, $rhs"),
                [(set GPR:$dst, (opnode GPR:$lhs, t2_so_reg:$rhs))]>;
}
}

/// T2I_rbin_c_irs - Same as T2I_bin_c_irs except the order of operands are
/// reversed.
let Uses = [CPSR] in {
multiclass T2I_rbin_c_irs<string opc, PatFrag opnode> {
   // shifted imm
   def ri : T2I<(outs GPR:$dst), (ins GPR:$rhs, t2_so_imm:$lhs, cc_out:$s),
                !strconcat(opc, "${s} $dst, $rhs, $lhs"),
                [(set GPR:$dst, (opnode t2_so_imm:$lhs, GPR:$rhs))]>;
   // register
   def rr : T2I<(outs GPR:$dst), (ins GPR:$rhs, GPR:$lhs, cc_out:$s),
                !strconcat(opc, "${s} $dst, $rhs, $lhs"),
                [(set GPR:$dst, (opnode GPR:$lhs, GPR:$rhs))]>;
   // shifted register
   def rs : T2I<(outs GPR:$dst), (ins GPR:$rhs, t2_so_reg:$lhs, cc_out:$s),
                !strconcat(opc, "${s} $dst, $rhs, $lhs"),
                [(set GPR:$dst, (opnode t2_so_reg:$lhs, GPR:$rhs))]>;
}
}

/// T2I_sh_ir - Defines a set of (op reg, {so_imm|r}) patterns for a shift /
//  rotate operation that produces a value.
multiclass T2I_sh_ir<string opc, PatFrag opnode> {
   // 5-bit imm
   def ri : T2I<(outs GPR:$dst), (ins GPR:$lhs, i32imm:$rhs),
                !strconcat(opc, " $dst, $lhs, $rhs"),
                [(set GPR:$dst, (opnode GPR:$lhs, imm1_31:$rhs))]>;
   // register
   def rr : T2I<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs),
                !strconcat(opc, " $dst, $lhs, $rhs"),
                [(set GPR:$dst, (opnode GPR:$lhs, GPR:$rhs))]>;
}

/// T21_cmp_irs - Defines a set of (op r, {so_imm|r|so_reg}) cmp / test
/// patterns. Similar to T2I_bin_irs except the instruction does not produce
/// a explicit result, only implicitly set CPSR.
let Uses = [CPSR] in {
multiclass T2I_cmp_is<string opc, PatFrag opnode> {
   // shifted imm
   def ri : T2I<(outs), (ins GPR:$lhs, t2_so_imm:$rhs),
                !strconcat(opc, " $lhs, $rhs"),
                [(opnode GPR:$lhs, t2_so_imm:$rhs)]>;
   // register
   def rr : T2I<(outs), (ins GPR:$lhs, GPR:$rhs),
                !strconcat(opc, " $lhs, $rhs"),
                [(opnode GPR:$lhs, GPR:$rhs)]>;
   // shifted register
   def rs : T2I<(outs), (ins GPR:$lhs, t2_so_reg:$rhs),
                !strconcat(opc, " $lhs, $rhs"),
                [(opnode GPR:$lhs, t2_so_reg:$rhs)]>;
}
}

//===----------------------------------------------------------------------===//
//  Miscellaneous Instructions.
//

let isNotDuplicable = 1 in
def t2PICADD : T2I<(outs tGPR:$dst), (ins tGPR:$lhs, pclabel:$cp),
                   "$cp:\n\tadd $dst, pc",
                   [(set tGPR:$dst, (ARMpic_add tGPR:$lhs, imm:$cp))]>;


// LEApcrel - Load a pc-relative address into a register without offending the
// assembler.
def t2LEApcrel : T2I<(outs GPR:$dst), (ins i32imm:$label, pred:$p),
                   !strconcat(!strconcat(".set PCRELV${:uid}, ($label-(",
                                         "${:private}PCRELL${:uid}+8))\n"),
                              !strconcat("${:private}PCRELL${:uid}:\n\t",
                                         "add$p $dst, pc, #PCRELV${:uid}")),
                   []>;

def t2LEApcrelJT : T2I<(outs GPR:$dst),
                       (ins i32imm:$label, i32imm:$id, pred:$p),
          !strconcat(!strconcat(".set PCRELV${:uid}, (${label}_${id:no_hash}-(",
                                         "${:private}PCRELL${:uid}+8))\n"),
                              !strconcat("${:private}PCRELL${:uid}:\n\t",
                                         "add$p $dst, pc, #PCRELV${:uid}")),
                   []>;

// ADD rd, sp, #so_imm
def t2ADDrSPi : T2I<(outs GPR:$dst), (ins GPR:$sp, t2_so_imm:$imm),
                    "add $dst, $sp, $imm",
                    []>;

// ADD rd, sp, #imm12
def t2ADDrSPi12 : T2I<(outs GPR:$dst), (ins GPR:$sp, i32imm:$imm),
                    "addw $dst, $sp, $imm",
                    []>;

def t2ADDrSPs : T2I<(outs GPR:$dst), (ins GPR:$sp, t2_so_reg:$rhs),
                    "addw $dst, $sp, $rhs",
                    []>;


//===----------------------------------------------------------------------===//
//  Arithmetic Instructions.
//

//===----------------------------------------------------------------------===//
//  Move Instructions.
//

let neverHasSideEffects = 1 in
def t2MOVr : T2I<(outs GPR:$dst), (ins GPR:$src),
                  "mov $dst, $src", []>;

let isReMaterializable = 1, isAsCheapAsAMove = 1 in
def t2MOVi16 : T2I<(outs GPR:$dst), (ins i32imm:$src),
                   "movw $dst, $src",
                   [(set GPR:$dst, imm0_65535:$src)]>;

// FIXME: Also available in ARM mode.
let Constraints = "$src = $dst" in
def t2MOVTi16 : T2I<(outs GPR:$dst), (ins GPR:$src, i32imm:$imm),
                   "movt $dst, $imm",
                   [(set GPR:$dst,
                         (or (and GPR:$src, 0xffff), t2_lo16AllZero:$imm))]>;

//===----------------------------------------------------------------------===//
//  Arithmetic Instructions.
//

defm t2ADD  : T2I_bin_ii12rs<"add", BinOpFrag<(add  node:$LHS, node:$RHS)>>;
defm t2SUB  : T2I_bin_ii12rs<"sub", BinOpFrag<(sub  node:$LHS, node:$RHS)>>;

// ADD and SUB with 's' bit set. No 12-bit immediate (T4) variants.
defm t2ADDS : T2I_bin_s_irs<"add",  BinOpFrag<(addc node:$LHS, node:$RHS)>>;
defm t2SUBS : T2I_bin_s_irs<"sub",  BinOpFrag<(subc node:$LHS, node:$RHS)>>;

// FIXME: predication support
defm t2ADC  : T2I_bin_c_irs<"adc",  BinOpFrag<(adde node:$LHS, node:$RHS)>>;
defm t2SBC  : T2I_bin_c_irs<"sbc",  BinOpFrag<(sube node:$LHS, node:$RHS)>>;

// RSB, RSC
defm t2RSB  : T2I_rbin_irs  <"rsb", BinOpFrag<(sub  node:$LHS, node:$RHS)>>;
defm t2RSBS : T2I_rbin_c_irs<"rsb", BinOpFrag<(subc node:$LHS, node:$RHS)>>;
defm t2RSC  : T2I_rbin_s_irs<"rsc", BinOpFrag<(sube node:$LHS, node:$RHS)>>;

// (sub X, imm) gets canonicalized to (add X, -imm).  Match this form.
def : Thumb2Pat<(add       GPR:$src, t2_so_imm_neg:$imm),
                (t2SUBri   GPR:$src, t2_so_imm_neg:$imm)>;
def : Thumb2Pat<(add       GPR:$src, imm0_4095_neg:$imm),
                (t2SUBri12 GPR:$src, imm0_4095_neg:$imm)>;


//===----------------------------------------------------------------------===//
//  Shift and rotate Instructions.
//

defm t2LSL  : T2I_sh_ir<"lsl", BinOpFrag<(shl  node:$LHS, node:$RHS)>>;
defm t2LSR  : T2I_sh_ir<"lsr", BinOpFrag<(srl  node:$LHS, node:$RHS)>>;
defm t2ASR  : T2I_sh_ir<"asr", BinOpFrag<(sra  node:$LHS, node:$RHS)>>;
defm t2ROR  : T2I_sh_ir<"ror", BinOpFrag<(rotr node:$LHS, node:$RHS)>>;

def t2MOVrx : T2I<(outs GPR:$dst), (ins GPR:$src),
                  "mov $dst, $src, rrx",
                  [(set GPR:$dst, (ARMrrx GPR:$src))]>;

//===----------------------------------------------------------------------===//
//  Bitwise Instructions.
//

defm t2AND  : T2I_bin_irs<"and", BinOpFrag<(and node:$LHS, node:$RHS)>>;
defm t2ORR  : T2I_bin_irs<"orr", BinOpFrag<(or  node:$LHS, node:$RHS)>>;
defm t2EOR  : T2I_bin_irs<"eor", BinOpFrag<(xor node:$LHS, node:$RHS)>>;

defm t2BIC  : T2I_bin_irs<"bic", BinOpFrag<(and node:$LHS, (not node:$RHS))>>;

def : Thumb2Pat<(and     GPR:$src, t2_so_imm_not:$imm),
                (t2BICri GPR:$src, t2_so_imm_not:$imm)>;

defm t2ORN  : T2I_bin_irs<"orn", BinOpFrag<(or  node:$LHS, (not node:$RHS))>>;

def : Thumb2Pat<(or      GPR:$src, t2_so_imm_not:$imm),
                (t2ORNri GPR:$src, t2_so_imm_not:$imm)>;

defm t2MVN  : T2I_un_irs  <"mvn", UnOpFrag<(not node:$Src)>, 1, 1>;

// A8.6.17  BFC - Bitfield clear
// FIXME: Also available in ARM mode.
let Constraints = "$src = $dst" in
def t2BFC : T2I<(outs GPR:$dst), (ins GPR:$src, bf_inv_mask_imm:$imm),
                "bfc $dst, $imm",
                [(set GPR:$dst, (and GPR:$src, bf_inv_mask_imm:$imm))]>;

// FIXME: A8.6.18  BFI - Bitfield insert (Encoding T1)

//===----------------------------------------------------------------------===//
//  Multiply Instructions.
//
def t2MUL: T2I<(outs GPR:$dst), (ins GPR:$a, GPR:$b),
                "mul $dst, $a, $b",
                [(set GPR:$dst, (mul GPR:$a, GPR:$b))]>;

def t2MLA: T2I<(outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$c),
		"mla $dst, $a, $b, $c",
		[(set GPR:$dst, (add (mul GPR:$a, GPR:$b), GPR:$c))]>;

def t2MLS: T2I<(outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$c),
		"mls $dst, $a, $b, $c",
                [(set GPR:$dst, (sub GPR:$c, (mul GPR:$a, GPR:$b)))]>;

// FIXME: SMULL, etc.

//===----------------------------------------------------------------------===//
//  Misc. Arithmetic Instructions.
//

/////
/// A8.6.31  CLZ
/////
// FIXME not firing? but ARM version does...
def t2CLZ : T2I<(outs GPR:$dst), (ins GPR:$src),
                "clz $dst, $src",
                [(set GPR:$dst, (ctlz GPR:$src))]>;

def t2REV : T2I<(outs GPR:$dst), (ins GPR:$src),
                "rev $dst, $src",
                [(set GPR:$dst, (bswap GPR:$src))]>;

def t2REV16 : T2I<(outs GPR:$dst), (ins GPR:$src),
                "rev16 $dst, $src",
                [(set GPR:$dst,
                    (or (and (srl GPR:$src, (i32 8)), 0xFF),
                        (or (and (shl GPR:$src, (i32 8)), 0xFF00),
                            (or (and (srl GPR:$src, (i32 8)), 0xFF0000),
                                (and (shl GPR:$src, (i32 8)), 0xFF000000)))))]>;

/////
/// A8.6.137  REVSH
/////
def t2REVSH : T2I<(outs GPR:$dst), (ins GPR:$src),
                 "revsh $dst, $src",
                 [(set GPR:$dst,
                    (sext_inreg
                      (or (srl (and GPR:$src, 0xFFFF), (i32 8)),
                          (shl GPR:$src, (i32 8))), i16))]>;

// FIXME: PKHxx etc.

//===----------------------------------------------------------------------===//
//  Comparison Instructions...
//

defm t2CMP   : T2I_cmp_is<"cmp",
                          BinOpFrag<(ARMcmp node:$LHS, node:$RHS)>>;
defm t2CMPnz : T2I_cmp_is<"cmp",
                          BinOpFrag<(ARMcmpNZ node:$LHS, node:$RHS)>>;

defm t2CMN   : T2I_cmp_is<"cmn",
                          BinOpFrag<(ARMcmp node:$LHS,(ineg node:$RHS))>>;
defm t2CMNnz : T2I_cmp_is<"cmn",
                          BinOpFrag<(ARMcmpNZ node:$LHS,(ineg node:$RHS))>>;

def : Thumb2Pat<(ARMcmp  GPR:$src, t2_so_imm_neg:$imm),
                (t2CMNri GPR:$src, t2_so_imm_neg:$imm)>;

def : Thumb2Pat<(ARMcmpNZ  GPR:$src, t2_so_imm_neg:$imm),
                (t2CMNri   GPR:$src, t2_so_imm_neg:$imm)>;

// FIXME: TST, TEQ, etc.

// A8.6.27  CBNZ, CBZ - Compare and branch on (non)zero.
// Short range conditional branch. Looks awesome for loops. Need to figure
// out how to use this one.

// FIXME: Conditional moves


//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//

// ConstantPool, GlobalAddress, and JumpTable
def : Thumb2Pat<(ARMWrapper  tglobaladdr :$dst), (t2LEApcrel tglobaladdr :$dst)>;
def : Thumb2Pat<(ARMWrapper  tconstpool  :$dst), (t2LEApcrel tconstpool  :$dst)>;
def : Thumb2Pat<(ARMWrapperJT tjumptable:$dst, imm:$id),
                (t2LEApcrelJT tjumptable:$dst, imm:$id)>;

// Large immediate handling.

def : Thumb2Pat<(i32 imm:$src),
                (t2MOVTi16 (t2MOVi16 (t2_lo16 imm:$src)),
                           (t2_hi16 imm:$src))>;
