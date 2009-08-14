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

// IT block predicate field
def it_pred : Operand<i32> {
  let PrintMethod = "printPredicateOperand";
}

// IT block condition mask
def it_mask : Operand<i32> {
  let PrintMethod = "printThumbITMask";
}

// Table branch address
def tb_addrmode : Operand<i32> {
  let PrintMethod = "printTBAddrMode";
}

// Shifted operands. No register controlled shifts for Thumb2.
// Note: We do not support rrx shifted operands yet.
def t2_so_reg : Operand<i32>,    // reg imm
                ComplexPattern<i32, 2, "SelectT2ShifterOperandReg",
                               [shl,srl,sra,rotr]> {
  let PrintMethod = "printT2SOOperand";
  let MIOperandInfo = (ops GPR, i32imm);
}

// t2_so_imm_not_XFORM - Return the complement of a t2_so_imm value
def t2_so_imm_not_XFORM : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(~((uint32_t)N->getZExtValue()), MVT::i32);
}]>;

// t2_so_imm_neg_XFORM - Return the negation of a t2_so_imm value
def t2_so_imm_neg_XFORM : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(-((int)N->getZExtValue()), MVT::i32);
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
}]>;

// t2_so_imm_not - Match an immediate that is a complement 
// of a t2_so_imm.
def t2_so_imm_not : Operand<i32>,
                    PatLeaf<(imm), [{
  return ARM_AM::getT2SOImmVal(~((uint32_t)N->getZExtValue())) != -1;
}], t2_so_imm_not_XFORM>;

// t2_so_imm_neg - Match an immediate that is a negation of a t2_so_imm.
def t2_so_imm_neg : Operand<i32>,
                    PatLeaf<(imm), [{
  return ARM_AM::getT2SOImmVal(-((int)N->getZExtValue())) != -1;
}], t2_so_imm_neg_XFORM>;

/// imm1_31 predicate - True if the 32-bit immediate is in the range [1,31].
def imm1_31 : PatLeaf<(i32 imm), [{
  return (int32_t)N->getZExtValue() >= 1 && (int32_t)N->getZExtValue() < 32;
}]>;

/// imm0_4095 predicate - True if the 32-bit immediate is in the range [0.4095].
def imm0_4095 : Operand<i32>,
                PatLeaf<(i32 imm), [{
  return (uint32_t)N->getZExtValue() < 4096;
}]>;

def imm0_4095_neg : PatLeaf<(i32 imm), [{ 
 return (uint32_t)(-N->getZExtValue()) < 4096; 
}], imm_neg_XFORM>; 

def imm0_255_neg : PatLeaf<(i32 imm), [{
  return (uint32_t)(-N->getZExtValue()) < 255;
}], imm_neg_XFORM>; 

/// imm0_65535 predicate - True if the 32-bit immediate is in the range 
/// [0.65535].
def imm0_65535 : PatLeaf<(i32 imm), [{
  return (uint32_t)N->getZExtValue() < 65536;
}]>;

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


// Define Thumb2 specific addressing modes.

// t2addrmode_imm12  := reg + imm12
def t2addrmode_imm12 : Operand<i32>,
                       ComplexPattern<i32, 2, "SelectT2AddrModeImm12", []> {
  let PrintMethod = "printT2AddrModeImm12Operand";
  let MIOperandInfo = (ops GPR:$base, i32imm:$offsimm);
}

// t2addrmode_imm8  := reg - imm8
def t2addrmode_imm8 : Operand<i32>,
                      ComplexPattern<i32, 2, "SelectT2AddrModeImm8", []> {
  let PrintMethod = "printT2AddrModeImm8Operand";
  let MIOperandInfo = (ops GPR:$base, i32imm:$offsimm);
}

def t2am_imm8_offset : Operand<i32>,
                       ComplexPattern<i32, 1, "SelectT2AddrModeImm8Offset", []>{
  let PrintMethod = "printT2AddrModeImm8OffsetOperand";
}

// t2addrmode_imm8s4  := reg +/- (imm8 << 2)
def t2addrmode_imm8s4 : Operand<i32>,
                        ComplexPattern<i32, 2, "SelectT2AddrModeImm8s4", []> {
  let PrintMethod = "printT2AddrModeImm8s4Operand";
  let MIOperandInfo = (ops GPR:$base, i32imm:$offsimm);
}

// t2addrmode_so_reg  := reg + (reg << imm2)
def t2addrmode_so_reg : Operand<i32>,
                        ComplexPattern<i32, 3, "SelectT2AddrModeSoReg", []> {
  let PrintMethod = "printT2AddrModeSoRegOperand";
  let MIOperandInfo = (ops GPR:$base, GPR:$offsreg, i32imm:$offsimm);
}


//===----------------------------------------------------------------------===//
// Multiclass helpers...
//

/// T2I_un_irs - Defines a set of (op reg, {so_imm|r|so_reg}) patterns for a
/// unary operation that produces a value. These are predicable and can be
/// changed to modify CPSR.
multiclass T2I_un_irs<string opc, PatFrag opnode, bit Cheap = 0, bit ReMat = 0>{
   // shifted imm
   def i : T2sI<(outs GPR:$dst), (ins t2_so_imm:$src), IIC_iALU,
                opc, " $dst, $src",
                [(set GPR:$dst, (opnode t2_so_imm:$src))]> {
     let isAsCheapAsAMove = Cheap;
     let isReMaterializable = ReMat;
   }
   // register
   def r : T2I<(outs GPR:$dst), (ins GPR:$src), IIC_iALU,
               opc, ".w $dst, $src",
                [(set GPR:$dst, (opnode GPR:$src))]>;
   // shifted register
   def s : T2I<(outs GPR:$dst), (ins t2_so_reg:$src), IIC_iALU,
               opc, ".w $dst, $src",
               [(set GPR:$dst, (opnode t2_so_reg:$src))]>;
}

/// T2I_bin_irs - Defines a set of (op reg, {so_imm|r|so_reg}) patterns for a
//  binary operation that produces a value. These are predicable and can be
/// changed to modify CPSR.
multiclass T2I_bin_irs<string opc, PatFrag opnode, 
                       bit Commutable = 0, string wide =""> {
   // shifted imm
   def ri : T2sI<(outs GPR:$dst), (ins GPR:$lhs, t2_so_imm:$rhs), IIC_iALU,
                 opc, " $dst, $lhs, $rhs",
                 [(set GPR:$dst, (opnode GPR:$lhs, t2_so_imm:$rhs))]>;
   // register
   def rr : T2sI<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs), IIC_iALU,
                 opc, !strconcat(wide, " $dst, $lhs, $rhs"),
                 [(set GPR:$dst, (opnode GPR:$lhs, GPR:$rhs))]> {
     let isCommutable = Commutable;
   }
   // shifted register
   def rs : T2sI<(outs GPR:$dst), (ins GPR:$lhs, t2_so_reg:$rhs), IIC_iALU,
                 opc, !strconcat(wide, " $dst, $lhs, $rhs"),
                 [(set GPR:$dst, (opnode GPR:$lhs, t2_so_reg:$rhs))]>;
}

/// T2I_bin_w_irs - Same as T2I_bin_irs except these operations need
//  the ".w" prefix to indicate that they are wide.
multiclass T2I_bin_w_irs<string opc, PatFrag opnode, bit Commutable = 0> :
    T2I_bin_irs<opc, opnode, Commutable, ".w">;

/// T2I_rbin_is - Same as T2I_bin_irs except the order of operands are
/// reversed. It doesn't define the 'rr' form since it's handled by its
/// T2I_bin_irs counterpart.
multiclass T2I_rbin_is<string opc, PatFrag opnode> {
   // shifted imm
   def ri : T2I<(outs GPR:$dst), (ins GPR:$rhs, t2_so_imm:$lhs), IIC_iALU,
                opc, ".w $dst, $rhs, $lhs",
                [(set GPR:$dst, (opnode t2_so_imm:$lhs, GPR:$rhs))]>;
   // shifted register
   def rs : T2I<(outs GPR:$dst), (ins GPR:$rhs, t2_so_reg:$lhs), IIC_iALU,
                opc, " $dst, $rhs, $lhs",
                [(set GPR:$dst, (opnode t2_so_reg:$lhs, GPR:$rhs))]>;
}

/// T2I_bin_s_irs - Similar to T2I_bin_irs except it sets the 's' bit so the
/// instruction modifies the CPSR register.
let Defs = [CPSR] in {
multiclass T2I_bin_s_irs<string opc, PatFrag opnode, bit Commutable = 0> {
   // shifted imm
   def ri : T2I<(outs GPR:$dst), (ins GPR:$lhs, t2_so_imm:$rhs), IIC_iALU,
                !strconcat(opc, "s"), ".w $dst, $lhs, $rhs",
                [(set GPR:$dst, (opnode GPR:$lhs, t2_so_imm:$rhs))]>;
   // register
   def rr : T2I<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs), IIC_iALU,
                !strconcat(opc, "s"), ".w $dst, $lhs, $rhs",
                [(set GPR:$dst, (opnode GPR:$lhs, GPR:$rhs))]> {
     let isCommutable = Commutable;
   }
   // shifted register
   def rs : T2I<(outs GPR:$dst), (ins GPR:$lhs, t2_so_reg:$rhs), IIC_iALU,
                !strconcat(opc, "s"), ".w $dst, $lhs, $rhs",
                [(set GPR:$dst, (opnode GPR:$lhs, t2_so_reg:$rhs))]>;
}
}

/// T2I_bin_ii12rs - Defines a set of (op reg, {so_imm|imm0_4095|r|so_reg})
/// patterns for a binary operation that produces a value.
multiclass T2I_bin_ii12rs<string opc, PatFrag opnode, bit Commutable = 0> {
   // shifted imm
   def ri : T2sI<(outs GPR:$dst), (ins GPR:$lhs, t2_so_imm:$rhs), IIC_iALU,
                 opc, ".w $dst, $lhs, $rhs",
                 [(set GPR:$dst, (opnode GPR:$lhs, t2_so_imm:$rhs))]>;
   // 12-bit imm
   def ri12 : T2sI<(outs GPR:$dst), (ins GPR:$lhs, imm0_4095:$rhs), IIC_iALU,
                   !strconcat(opc, "w"), " $dst, $lhs, $rhs",
                   [(set GPR:$dst, (opnode GPR:$lhs, imm0_4095:$rhs))]>;
   // register
   def rr : T2sI<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs), IIC_iALU,
                 opc, ".w $dst, $lhs, $rhs",
                 [(set GPR:$dst, (opnode GPR:$lhs, GPR:$rhs))]> {
     let isCommutable = Commutable;
   }
   // shifted register
   def rs : T2sI<(outs GPR:$dst), (ins GPR:$lhs, t2_so_reg:$rhs), IIC_iALU,
                 opc, ".w $dst, $lhs, $rhs",
                 [(set GPR:$dst, (opnode GPR:$lhs, t2_so_reg:$rhs))]>;
}

/// T2I_adde_sube_irs - Defines a set of (op reg, {so_imm|r|so_reg}) patterns for a
/// binary operation that produces a value and use and define the carry bit.
/// It's not predicable.
let Uses = [CPSR] in {
multiclass T2I_adde_sube_irs<string opc, PatFrag opnode, bit Commutable = 0> {
   // shifted imm
   def ri : T2sI<(outs GPR:$dst), (ins GPR:$lhs, t2_so_imm:$rhs), IIC_iALU,
                 opc, " $dst, $lhs, $rhs",
                 [(set GPR:$dst, (opnode GPR:$lhs, t2_so_imm:$rhs))]>,
                 Requires<[IsThumb2, CarryDefIsUnused]>;
   // register
   def rr : T2sI<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs), IIC_iALU,
                 opc, ".w $dst, $lhs, $rhs",
                 [(set GPR:$dst, (opnode GPR:$lhs, GPR:$rhs))]>,
                 Requires<[IsThumb2, CarryDefIsUnused]> {
     let isCommutable = Commutable;
   }
   // shifted register
   def rs : T2sI<(outs GPR:$dst), (ins GPR:$lhs, t2_so_reg:$rhs), IIC_iALU,
                 opc, ".w $dst, $lhs, $rhs",
                 [(set GPR:$dst, (opnode GPR:$lhs, t2_so_reg:$rhs))]>,
                 Requires<[IsThumb2, CarryDefIsUnused]>;
   // Carry setting variants
   // shifted imm
   def Sri : T2XI<(outs GPR:$dst), (ins GPR:$lhs, t2_so_imm:$rhs), IIC_iALU,
                  !strconcat(opc, "s $dst, $lhs, $rhs"),
                  [(set GPR:$dst, (opnode GPR:$lhs, t2_so_imm:$rhs))]>,
                  Requires<[IsThumb2, CarryDefIsUsed]> {
                    let Defs = [CPSR];
                  }
   // register
   def Srr : T2XI<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs), IIC_iALU,
                  !strconcat(opc, "s.w $dst, $lhs, $rhs"),
                  [(set GPR:$dst, (opnode GPR:$lhs, GPR:$rhs))]>,
                  Requires<[IsThumb2, CarryDefIsUsed]> {
                    let Defs = [CPSR];
                    let isCommutable = Commutable;
   }
   // shifted register
   def Srs : T2XI<(outs GPR:$dst), (ins GPR:$lhs, t2_so_reg:$rhs), IIC_iALU,
                  !strconcat(opc, "s.w $dst, $lhs, $rhs"),
                  [(set GPR:$dst, (opnode GPR:$lhs, t2_so_reg:$rhs))]>,
                  Requires<[IsThumb2, CarryDefIsUsed]> {
                    let Defs = [CPSR];
   }
}
}

/// T2I_rbin_s_is - Same as T2I_rbin_is except sets 's' bit.
let Defs = [CPSR] in {
multiclass T2I_rbin_s_is<string opc, PatFrag opnode> {
   // shifted imm
   def ri : T2XI<(outs GPR:$dst), (ins GPR:$rhs, t2_so_imm:$lhs, cc_out:$s),
                 IIC_iALU,
                 !strconcat(opc, "${s}.w $dst, $rhs, $lhs"),
                 [(set GPR:$dst, (opnode t2_so_imm:$lhs, GPR:$rhs))]>;
   // shifted register
   def rs : T2XI<(outs GPR:$dst), (ins GPR:$rhs, t2_so_reg:$lhs, cc_out:$s),
                 IIC_iALU,
                 !strconcat(opc, "${s} $dst, $rhs, $lhs"),
                 [(set GPR:$dst, (opnode t2_so_reg:$lhs, GPR:$rhs))]>;
}
}

/// T2I_sh_ir - Defines a set of (op reg, {so_imm|r}) patterns for a shift /
//  rotate operation that produces a value.
multiclass T2I_sh_ir<string opc, PatFrag opnode> {
   // 5-bit imm
   def ri : T2sI<(outs GPR:$dst), (ins GPR:$lhs, i32imm:$rhs), IIC_iALU,
                 opc, ".w $dst, $lhs, $rhs",
                 [(set GPR:$dst, (opnode GPR:$lhs, imm1_31:$rhs))]>;
   // register
   def rr : T2sI<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs), IIC_iALU,
                 opc, ".w $dst, $lhs, $rhs",
                 [(set GPR:$dst, (opnode GPR:$lhs, GPR:$rhs))]>;
}

/// T21_cmp_irs - Defines a set of (op r, {so_imm|r|so_reg}) cmp / test
/// patterns. Similar to T2I_bin_irs except the instruction does not produce
/// a explicit result, only implicitly set CPSR.
let Defs = [CPSR] in {
multiclass T2I_cmp_is<string opc, PatFrag opnode> {
   // shifted imm
   def ri : T2I<(outs), (ins GPR:$lhs, t2_so_imm:$rhs), IIC_iALU,
                opc, ".w $lhs, $rhs",
                [(opnode GPR:$lhs, t2_so_imm:$rhs)]>;
   // register
   def rr : T2I<(outs), (ins GPR:$lhs, GPR:$rhs), IIC_iALU,
                opc, ".w $lhs, $rhs",
                [(opnode GPR:$lhs, GPR:$rhs)]>;
   // shifted register
   def rs : T2I<(outs), (ins GPR:$lhs, t2_so_reg:$rhs), IIC_iALU,
                opc, ".w $lhs, $rhs",
                [(opnode GPR:$lhs, t2_so_reg:$rhs)]>;
}
}

/// T2I_ld - Defines a set of (op r, {imm12|imm8|so_reg}) load patterns.
multiclass T2I_ld<string opc, PatFrag opnode> {
  def i12 : T2Ii12<(outs GPR:$dst), (ins t2addrmode_imm12:$addr), IIC_iLoad,
                   opc, ".w $dst, $addr",
                   [(set GPR:$dst, (opnode t2addrmode_imm12:$addr))]>;
  def i8  : T2Ii8 <(outs GPR:$dst), (ins t2addrmode_imm8:$addr), IIC_iLoad,
                   opc, " $dst, $addr",
                   [(set GPR:$dst, (opnode t2addrmode_imm8:$addr))]>;
  def s   : T2Iso <(outs GPR:$dst), (ins t2addrmode_so_reg:$addr), IIC_iLoad,
                   opc, ".w $dst, $addr",
                   [(set GPR:$dst, (opnode t2addrmode_so_reg:$addr))]>;
  def pci : T2Ipc <(outs GPR:$dst), (ins i32imm:$addr), IIC_iLoad,
                   opc, ".w $dst, $addr",
                   [(set GPR:$dst, (opnode (ARMWrapper tconstpool:$addr)))]>;
}

/// T2I_st - Defines a set of (op r, {imm12|imm8|so_reg}) store patterns.
multiclass T2I_st<string opc, PatFrag opnode> {
  def i12 : T2Ii12<(outs), (ins GPR:$src, t2addrmode_imm12:$addr), IIC_iStore,
                   opc, ".w $src, $addr",
                   [(opnode GPR:$src, t2addrmode_imm12:$addr)]>;
  def i8  : T2Ii8 <(outs), (ins GPR:$src, t2addrmode_imm8:$addr), IIC_iStore,
                   opc, " $src, $addr",
                   [(opnode GPR:$src, t2addrmode_imm8:$addr)]>;
  def s   : T2Iso <(outs), (ins GPR:$src, t2addrmode_so_reg:$addr), IIC_iStore,
                   opc, ".w $src, $addr",
                   [(opnode GPR:$src, t2addrmode_so_reg:$addr)]>;
}

/// T2I_picld - Defines the PIC load pattern.
class T2I_picld<string opc, PatFrag opnode> :
      T2I<(outs GPR:$dst), (ins addrmodepc:$addr), IIC_iLoad,
          !strconcat("${addr:label}:\n\t", opc), " $dst, $addr",
          [(set GPR:$dst, (opnode addrmodepc:$addr))]>;

/// T2I_picst - Defines the PIC store pattern.
class T2I_picst<string opc, PatFrag opnode> :
      T2I<(outs), (ins GPR:$src, addrmodepc:$addr), IIC_iStore,
          !strconcat("${addr:label}:\n\t", opc), " $src, $addr",
          [(opnode GPR:$src, addrmodepc:$addr)]>;


/// T2I_unary_rrot - A unary operation with two forms: one whose operand is a
/// register and one whose operand is a register rotated by 8/16/24.
multiclass T2I_unary_rrot<string opc, PatFrag opnode> {
  def r     : T2I<(outs GPR:$dst), (ins GPR:$Src), IIC_iALU,
                  opc, ".w $dst, $Src",
                 [(set GPR:$dst, (opnode GPR:$Src))]>;
  def r_rot : T2I<(outs GPR:$dst), (ins GPR:$Src, i32imm:$rot), IIC_iALU,
                  opc, ".w $dst, $Src, ror $rot",
                 [(set GPR:$dst, (opnode (rotr GPR:$Src, rot_imm:$rot)))]>;
}

/// T2I_bin_rrot - A binary operation with two forms: one whose operand is a
/// register and one whose operand is a register rotated by 8/16/24.
multiclass T2I_bin_rrot<string opc, PatFrag opnode> {
  def rr     : T2I<(outs GPR:$dst), (ins GPR:$LHS, GPR:$RHS), IIC_iALU,
                  opc, " $dst, $LHS, $RHS",
                  [(set GPR:$dst, (opnode GPR:$LHS, GPR:$RHS))]>;
  def rr_rot : T2I<(outs GPR:$dst), (ins GPR:$LHS, GPR:$RHS, i32imm:$rot),
                  IIC_iALU, opc, " $dst, $LHS, $RHS, ror $rot",
                  [(set GPR:$dst, (opnode GPR:$LHS,
                                          (rotr GPR:$RHS, rot_imm:$rot)))]>;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Miscellaneous Instructions.
//

// LEApcrel - Load a pc-relative address into a register without offending the
// assembler.
def t2LEApcrel : T2XI<(outs GPR:$dst), (ins i32imm:$label, pred:$p), IIC_iALU,
                      "adr$p.w $dst, #$label", []>;

def t2LEApcrelJT : T2XI<(outs GPR:$dst),
                        (ins i32imm:$label, lane_cst:$id, pred:$p), IIC_iALU,
                        "adr$p.w $dst, #${label}_${id}", []>;

// ADD r, sp, {so_imm|i12}
def t2ADDrSPi   : T2sI<(outs GPR:$dst), (ins GPR:$sp, t2_so_imm:$imm), IIC_iALU,
                       "add", ".w $dst, $sp, $imm", []>;
def t2ADDrSPi12 : T2I<(outs GPR:$dst), (ins GPR:$sp, imm0_4095:$imm), IIC_iALU,
                       "addw", " $dst, $sp, $imm", []>;

// ADD r, sp, so_reg
def t2ADDrSPs   : T2sI<(outs GPR:$dst), (ins GPR:$sp, t2_so_reg:$rhs), IIC_iALU,
                       "add", ".w $dst, $sp, $rhs", []>;

// SUB r, sp, {so_imm|i12}
def t2SUBrSPi   : T2sI<(outs GPR:$dst), (ins GPR:$sp, t2_so_imm:$imm), IIC_iALU,
                       "sub", ".w $dst, $sp, $imm", []>;
def t2SUBrSPi12 : T2I<(outs GPR:$dst), (ins GPR:$sp, imm0_4095:$imm), IIC_iALU,
                       "subw", " $dst, $sp, $imm", []>;

// SUB r, sp, so_reg
def t2SUBrSPs   : T2sI<(outs GPR:$dst), (ins GPR:$sp, t2_so_reg:$rhs), IIC_iALU,
                       "sub", " $dst, $sp, $rhs", []>;


// Pseudo instruction that will expand into a t2SUBrSPi + a copy.
let usesCustomDAGSchedInserter = 1 in { // Expanded by the scheduler.
def t2SUBrSPi_   : PseudoInst<(outs GPR:$dst), (ins GPR:$sp, t2_so_imm:$imm),
                   NoItinerary, "@ sub.w $dst, $sp, $imm", []>;
def t2SUBrSPi12_ : PseudoInst<(outs GPR:$dst), (ins GPR:$sp, imm0_4095:$imm),
                   NoItinerary, "@ subw $dst, $sp, $imm", []>;
def t2SUBrSPs_   : PseudoInst<(outs GPR:$dst), (ins GPR:$sp, t2_so_reg:$rhs),
                   NoItinerary, "@ sub $dst, $sp, $rhs", []>;
} // usesCustomDAGSchedInserter


//===----------------------------------------------------------------------===//
//  Load / store Instructions.
//

// Load
let canFoldAsLoad = 1 in
defm t2LDR   : T2I_ld<"ldr",  UnOpFrag<(load node:$Src)>>;

// Loads with zero extension
defm t2LDRH  : T2I_ld<"ldrh", UnOpFrag<(zextloadi16 node:$Src)>>;
defm t2LDRB  : T2I_ld<"ldrb", UnOpFrag<(zextloadi8  node:$Src)>>;

// Loads with sign extension
defm t2LDRSH : T2I_ld<"ldrsh", UnOpFrag<(sextloadi16 node:$Src)>>;
defm t2LDRSB : T2I_ld<"ldrsb", UnOpFrag<(sextloadi8  node:$Src)>>;

let mayLoad = 1 in {
// Load doubleword
def t2LDRDi8 : T2Ii8s4<(outs GPR:$dst), (ins t2addrmode_imm8s4:$addr),
                        IIC_iLoad, "ldrd", " $dst, $addr", []>;
def t2LDRDpci : T2Ii8s4<(outs GPR:$dst), (ins i32imm:$addr), IIC_iLoad,
                       "ldrd", " $dst, $addr", []>;
}

// zextload i1 -> zextload i8
def : T2Pat<(zextloadi1 t2addrmode_imm12:$addr),
            (t2LDRBi12  t2addrmode_imm12:$addr)>;
def : T2Pat<(zextloadi1 t2addrmode_imm8:$addr),
            (t2LDRBi8   t2addrmode_imm8:$addr)>;
def : T2Pat<(zextloadi1 t2addrmode_so_reg:$addr),
            (t2LDRBs    t2addrmode_so_reg:$addr)>;
def : T2Pat<(zextloadi1 (ARMWrapper tconstpool:$addr)),
            (t2LDRBpci  tconstpool:$addr)>;

// extload -> zextload
// FIXME: Reduce the number of patterns by legalizing extload to zextload
// earlier?
def : T2Pat<(extloadi1  t2addrmode_imm12:$addr),
            (t2LDRBi12  t2addrmode_imm12:$addr)>;
def : T2Pat<(extloadi1  t2addrmode_imm8:$addr),
            (t2LDRBi8   t2addrmode_imm8:$addr)>;
def : T2Pat<(extloadi1  t2addrmode_so_reg:$addr),
            (t2LDRBs    t2addrmode_so_reg:$addr)>;
def : T2Pat<(extloadi1  (ARMWrapper tconstpool:$addr)),
            (t2LDRBpci  tconstpool:$addr)>;

def : T2Pat<(extloadi8  t2addrmode_imm12:$addr),
            (t2LDRBi12  t2addrmode_imm12:$addr)>;
def : T2Pat<(extloadi8  t2addrmode_imm8:$addr),
            (t2LDRBi8   t2addrmode_imm8:$addr)>;
def : T2Pat<(extloadi8  t2addrmode_so_reg:$addr),
            (t2LDRBs    t2addrmode_so_reg:$addr)>;
def : T2Pat<(extloadi8  (ARMWrapper tconstpool:$addr)),
            (t2LDRBpci  tconstpool:$addr)>;

def : T2Pat<(extloadi16 t2addrmode_imm12:$addr),
            (t2LDRHi12  t2addrmode_imm12:$addr)>;
def : T2Pat<(extloadi16 t2addrmode_imm8:$addr),
            (t2LDRHi8   t2addrmode_imm8:$addr)>;
def : T2Pat<(extloadi16 t2addrmode_so_reg:$addr),
            (t2LDRHs    t2addrmode_so_reg:$addr)>;
def : T2Pat<(extloadi16 (ARMWrapper tconstpool:$addr)),
            (t2LDRHpci  tconstpool:$addr)>;

// Indexed loads
let mayLoad = 1 in {
def t2LDR_PRE  : T2Iidxldst<(outs GPR:$dst, GPR:$base_wb),
                            (ins t2addrmode_imm8:$addr),
                            AddrModeT2_i8, IndexModePre, IIC_iLoad,
                            "ldr", " $dst, $addr!", "$addr.base = $base_wb",
                            []>;

def t2LDR_POST : T2Iidxldst<(outs GPR:$dst, GPR:$base_wb),
                            (ins GPR:$base, t2am_imm8_offset:$offset),
                            AddrModeT2_i8, IndexModePost, IIC_iLoad,
                           "ldr", " $dst, [$base], $offset", "$base = $base_wb",
                            []>;

def t2LDRB_PRE : T2Iidxldst<(outs GPR:$dst, GPR:$base_wb),
                            (ins t2addrmode_imm8:$addr),
                            AddrModeT2_i8, IndexModePre, IIC_iLoad,
                            "ldrb", " $dst, $addr!", "$addr.base = $base_wb",
                            []>;
def t2LDRB_POST : T2Iidxldst<(outs GPR:$dst, GPR:$base_wb),
                            (ins GPR:$base, t2am_imm8_offset:$offset),
                            AddrModeT2_i8, IndexModePost, IIC_iLoad,
                          "ldrb", " $dst, [$base], $offset", "$base = $base_wb",
                            []>;

def t2LDRH_PRE : T2Iidxldst<(outs GPR:$dst, GPR:$base_wb),
                            (ins t2addrmode_imm8:$addr),
                            AddrModeT2_i8, IndexModePre, IIC_iLoad,
                            "ldrh", " $dst, $addr!", "$addr.base = $base_wb",
                            []>;
def t2LDRH_POST : T2Iidxldst<(outs GPR:$dst, GPR:$base_wb),
                            (ins GPR:$base, t2am_imm8_offset:$offset),
                            AddrModeT2_i8, IndexModePost, IIC_iLoad,
                          "ldrh", " $dst, [$base], $offset", "$base = $base_wb",
                            []>;

def t2LDRSB_PRE : T2Iidxldst<(outs GPR:$dst, GPR:$base_wb),
                            (ins t2addrmode_imm8:$addr),
                            AddrModeT2_i8, IndexModePre, IIC_iLoad,
                            "ldrsb", " $dst, $addr!", "$addr.base = $base_wb",
                            []>;
def t2LDRSB_POST : T2Iidxldst<(outs GPR:$dst, GPR:$base_wb),
                            (ins GPR:$base, t2am_imm8_offset:$offset),
                            AddrModeT2_i8, IndexModePost, IIC_iLoad,
                         "ldrsb", " $dst, [$base], $offset", "$base = $base_wb",
                            []>;

def t2LDRSH_PRE : T2Iidxldst<(outs GPR:$dst, GPR:$base_wb),
                            (ins t2addrmode_imm8:$addr),
                            AddrModeT2_i8, IndexModePre, IIC_iLoad,
                            "ldrsh", " $dst, $addr!", "$addr.base = $base_wb",
                            []>;
def t2LDRSH_POST : T2Iidxldst<(outs GPR:$dst, GPR:$base_wb),
                            (ins GPR:$base, t2am_imm8_offset:$offset),
                            AddrModeT2_i8, IndexModePost, IIC_iLoad,
                         "ldrsh", " $dst, [$base], $offset", "$base = $base_wb",
                            []>;
}

// Store
defm t2STR   : T2I_st<"str",  BinOpFrag<(store node:$LHS, node:$RHS)>>;
defm t2STRB  : T2I_st<"strb", BinOpFrag<(truncstorei8 node:$LHS, node:$RHS)>>;
defm t2STRH  : T2I_st<"strh", BinOpFrag<(truncstorei16 node:$LHS, node:$RHS)>>;

// Store doubleword
let mayLoad = 1 in
def t2STRDi8 : T2Ii8s4<(outs), (ins GPR:$src, t2addrmode_imm8s4:$addr),
               IIC_iStore, "strd", " $src, $addr", []>;

// Indexed stores
def t2STR_PRE  : T2Iidxldst<(outs GPR:$base_wb),
                            (ins GPR:$src, GPR:$base, t2am_imm8_offset:$offset),
                            AddrModeT2_i8, IndexModePre, IIC_iStore,
                          "str", " $src, [$base, $offset]!", "$base = $base_wb",
             [(set GPR:$base_wb,
                   (pre_store GPR:$src, GPR:$base, t2am_imm8_offset:$offset))]>;

def t2STR_POST : T2Iidxldst<(outs GPR:$base_wb),
                            (ins GPR:$src, GPR:$base, t2am_imm8_offset:$offset),
                            AddrModeT2_i8, IndexModePost, IIC_iStore,
                           "str", " $src, [$base], $offset", "$base = $base_wb",
             [(set GPR:$base_wb,
                   (post_store GPR:$src, GPR:$base, t2am_imm8_offset:$offset))]>;

def t2STRH_PRE  : T2Iidxldst<(outs GPR:$base_wb),
                            (ins GPR:$src, GPR:$base, t2am_imm8_offset:$offset),
                            AddrModeT2_i8, IndexModePre, IIC_iStore,
                         "strh", " $src, [$base, $offset]!", "$base = $base_wb",
        [(set GPR:$base_wb,
              (pre_truncsti16 GPR:$src, GPR:$base, t2am_imm8_offset:$offset))]>;

def t2STRH_POST : T2Iidxldst<(outs GPR:$base_wb),
                            (ins GPR:$src, GPR:$base, t2am_imm8_offset:$offset),
                            AddrModeT2_i8, IndexModePost, IIC_iStore,
                          "strh", " $src, [$base], $offset", "$base = $base_wb",
       [(set GPR:$base_wb,
             (post_truncsti16 GPR:$src, GPR:$base, t2am_imm8_offset:$offset))]>;

def t2STRB_PRE  : T2Iidxldst<(outs GPR:$base_wb),
                            (ins GPR:$src, GPR:$base, t2am_imm8_offset:$offset),
                            AddrModeT2_i8, IndexModePre, IIC_iStore,
                         "strb", " $src, [$base, $offset]!", "$base = $base_wb",
         [(set GPR:$base_wb,
               (pre_truncsti8 GPR:$src, GPR:$base, t2am_imm8_offset:$offset))]>;

def t2STRB_POST : T2Iidxldst<(outs GPR:$base_wb),
                            (ins GPR:$src, GPR:$base, t2am_imm8_offset:$offset),
                            AddrModeT2_i8, IndexModePost, IIC_iStore,
                          "strb", " $src, [$base], $offset", "$base = $base_wb",
        [(set GPR:$base_wb,
              (post_truncsti8 GPR:$src, GPR:$base, t2am_imm8_offset:$offset))]>;


// FIXME: ldrd / strd pre / post variants

//===----------------------------------------------------------------------===//
//  Load / store multiple Instructions.
//

let mayLoad = 1 in
def t2LDM : T2XI<(outs),
                 (ins addrmode4:$addr, pred:$p, reglist:$dst1, variable_ops),
              IIC_iLoad, "ldm${addr:submode}${p}${addr:wide} $addr, $dst1", []>;

let mayStore = 1 in
def t2STM : T2XI<(outs),
                 (ins addrmode4:$addr, pred:$p, reglist:$src1, variable_ops),
              IIC_iStore, "stm${addr:submode}${p}${addr:wide} $addr, $src1", []>;

//===----------------------------------------------------------------------===//
//  Move Instructions.
//

let neverHasSideEffects = 1 in
def t2MOVr : T2sI<(outs GPR:$dst), (ins GPR:$src), IIC_iALU,
                   "mov", ".w $dst, $src", []>;

let isReMaterializable = 1, isAsCheapAsAMove = 1 in
def t2MOVi : T2sI<(outs GPR:$dst), (ins t2_so_imm:$src), IIC_iALU,
                   "mov", ".w $dst, $src",
                   [(set GPR:$dst, t2_so_imm:$src)]>;

let isReMaterializable = 1, isAsCheapAsAMove = 1 in
def t2MOVi16 : T2I<(outs GPR:$dst), (ins i32imm:$src), IIC_iALU,
                   "movw", " $dst, $src",
                   [(set GPR:$dst, imm0_65535:$src)]>;

// FIXME: Also available in ARM mode.
let Constraints = "$src = $dst" in
def t2MOVTi16 : T2sI<(outs GPR:$dst), (ins GPR:$src, i32imm:$imm), IIC_iALU,
                     "movt", " $dst, $imm",
                     [(set GPR:$dst,
                           (or (and GPR:$src, 0xffff), t2_lo16AllZero:$imm))]>;

//===----------------------------------------------------------------------===//
//  Extend Instructions.
//

// Sign extenders

defm t2SXTB  : T2I_unary_rrot<"sxtb", UnOpFrag<(sext_inreg node:$Src, i8)>>;
defm t2SXTH  : T2I_unary_rrot<"sxth", UnOpFrag<(sext_inreg node:$Src, i16)>>;

defm t2SXTAB : T2I_bin_rrot<"sxtab",
                        BinOpFrag<(add node:$LHS, (sext_inreg node:$RHS, i8))>>;
defm t2SXTAH : T2I_bin_rrot<"sxtah",
                        BinOpFrag<(add node:$LHS, (sext_inreg node:$RHS,i16))>>;

// TODO: SXT(A){B|H}16

// Zero extenders

let AddedComplexity = 16 in {
defm t2UXTB   : T2I_unary_rrot<"uxtb"  , UnOpFrag<(and node:$Src, 0x000000FF)>>;
defm t2UXTH   : T2I_unary_rrot<"uxth"  , UnOpFrag<(and node:$Src, 0x0000FFFF)>>;
defm t2UXTB16 : T2I_unary_rrot<"uxtb16", UnOpFrag<(and node:$Src, 0x00FF00FF)>>;

def : T2Pat<(and (shl GPR:$Src, (i32 8)), 0xFF00FF),
            (t2UXTB16r_rot GPR:$Src, 24)>;
def : T2Pat<(and (srl GPR:$Src, (i32 8)), 0xFF00FF),
            (t2UXTB16r_rot GPR:$Src, 8)>;

defm t2UXTAB : T2I_bin_rrot<"uxtab",
                            BinOpFrag<(add node:$LHS, (and node:$RHS, 0x00FF))>>;
defm t2UXTAH : T2I_bin_rrot<"uxtah",
                            BinOpFrag<(add node:$LHS, (and node:$RHS, 0xFFFF))>>;
}

//===----------------------------------------------------------------------===//
//  Arithmetic Instructions.
//

defm t2ADD  : T2I_bin_ii12rs<"add", BinOpFrag<(add  node:$LHS, node:$RHS)>, 1>;
defm t2SUB  : T2I_bin_ii12rs<"sub", BinOpFrag<(sub  node:$LHS, node:$RHS)>>;

// ADD and SUB with 's' bit set. No 12-bit immediate (T4) variants.
defm t2ADDS : T2I_bin_s_irs <"add",  BinOpFrag<(addc node:$LHS, node:$RHS)>, 1>;
defm t2SUBS : T2I_bin_s_irs <"sub",  BinOpFrag<(subc node:$LHS, node:$RHS)>>;

defm t2ADC  : T2I_adde_sube_irs<"adc",BinOpFrag<(adde node:$LHS, node:$RHS)>,1>;
defm t2SBC  : T2I_adde_sube_irs<"sbc",BinOpFrag<(sube node:$LHS, node:$RHS)>>;

// RSB
defm t2RSB  : T2I_rbin_is   <"rsb", BinOpFrag<(sub  node:$LHS, node:$RHS)>>;
defm t2RSBS : T2I_rbin_s_is <"rsb", BinOpFrag<(subc node:$LHS, node:$RHS)>>;

// (sub X, imm) gets canonicalized to (add X, -imm).  Match this form.
let AddedComplexity = 1 in
def : T2Pat<(add       GPR:$src, imm0_255_neg:$imm),
            (t2SUBri   GPR:$src, imm0_255_neg:$imm)>;
def : T2Pat<(add       GPR:$src, t2_so_imm_neg:$imm),
            (t2SUBri   GPR:$src, t2_so_imm_neg:$imm)>;
def : T2Pat<(add       GPR:$src, imm0_4095_neg:$imm),
            (t2SUBri12 GPR:$src, imm0_4095_neg:$imm)>;


//===----------------------------------------------------------------------===//
//  Shift and rotate Instructions.
//

defm t2LSL  : T2I_sh_ir<"lsl", BinOpFrag<(shl  node:$LHS, node:$RHS)>>;
defm t2LSR  : T2I_sh_ir<"lsr", BinOpFrag<(srl  node:$LHS, node:$RHS)>>;
defm t2ASR  : T2I_sh_ir<"asr", BinOpFrag<(sra  node:$LHS, node:$RHS)>>;
defm t2ROR  : T2I_sh_ir<"ror", BinOpFrag<(rotr node:$LHS, node:$RHS)>>;

def t2MOVrx : T2sI<(outs GPR:$dst), (ins GPR:$src), IIC_iALU,
                   "rrx", ".w $dst, $src",
                   [(set GPR:$dst, (ARMrrx GPR:$src))]>;

let Defs = [CPSR] in {
def t2MOVsrl_flag : T2XI<(outs GPR:$dst), (ins GPR:$src), IIC_iALU,
                         "lsrs.w $dst, $src, #1",
                         [(set GPR:$dst, (ARMsrl_flag GPR:$src))]>;
def t2MOVsra_flag : T2XI<(outs GPR:$dst), (ins GPR:$src), IIC_iALU,
                         "asrs.w $dst, $src, #1",
                         [(set GPR:$dst, (ARMsra_flag GPR:$src))]>;
}

//===----------------------------------------------------------------------===//
//  Bitwise Instructions.
//

defm t2AND  : T2I_bin_w_irs<"and", BinOpFrag<(and node:$LHS, node:$RHS)>, 1>;
defm t2ORR  : T2I_bin_w_irs<"orr", BinOpFrag<(or  node:$LHS, node:$RHS)>, 1>;
defm t2EOR  : T2I_bin_w_irs<"eor", BinOpFrag<(xor node:$LHS, node:$RHS)>, 1>;

defm t2BIC  : T2I_bin_w_irs<"bic", BinOpFrag<(and node:$LHS, (not node:$RHS))>>;

let Constraints = "$src = $dst" in
def t2BFC : T2I<(outs GPR:$dst), (ins GPR:$src, bf_inv_mask_imm:$imm), IIC_iALU,
                "bfc", " $dst, $imm",
                [(set GPR:$dst, (and GPR:$src, bf_inv_mask_imm:$imm))]>;

// FIXME: A8.6.18  BFI - Bitfield insert (Encoding T1)

defm t2ORN  : T2I_bin_irs<"orn", BinOpFrag<(or  node:$LHS, (not node:$RHS))>>;

// Prefer over of t2EORri ra, rb, -1 because mvn has 16-bit version
let AddedComplexity = 1 in
defm t2MVN  : T2I_un_irs  <"mvn", UnOpFrag<(not node:$Src)>, 1, 1>;


def : T2Pat<(and     GPR:$src, t2_so_imm_not:$imm),
            (t2BICri GPR:$src, t2_so_imm_not:$imm)>;

// FIXME: Disable this pattern on Darwin to workaround an assembler bug.
def : T2Pat<(or      GPR:$src, t2_so_imm_not:$imm),
            (t2ORNri GPR:$src, t2_so_imm_not:$imm)>,
            Requires<[IsThumb2]>;

def : T2Pat<(t2_so_imm_not:$src),
            (t2MVNi t2_so_imm_not:$src)>;

//===----------------------------------------------------------------------===//
//  Multiply Instructions.
//
let isCommutable = 1 in
def t2MUL: T2I<(outs GPR:$dst), (ins GPR:$a, GPR:$b), IIC_iMPYw,
                "mul", " $dst, $a, $b",
                [(set GPR:$dst, (mul GPR:$a, GPR:$b))]>;

def t2MLA: T2I<(outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$c), IIC_iMPYw,
		"mla", " $dst, $a, $b, $c",
		[(set GPR:$dst, (add (mul GPR:$a, GPR:$b), GPR:$c))]>;

def t2MLS: T2I<(outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$c), IIC_iMPYw,
		"mls", " $dst, $a, $b, $c",
                [(set GPR:$dst, (sub GPR:$c, (mul GPR:$a, GPR:$b)))]>;

// Extra precision multiplies with low / high results
let neverHasSideEffects = 1 in {
let isCommutable = 1 in {
def t2SMULL : T2I<(outs GPR:$ldst, GPR:$hdst), (ins GPR:$a, GPR:$b), IIC_iMPYl,
                   "smull", " $ldst, $hdst, $a, $b", []>;

def t2UMULL : T2I<(outs GPR:$ldst, GPR:$hdst), (ins GPR:$a, GPR:$b), IIC_iMPYl,
                   "umull", " $ldst, $hdst, $a, $b", []>;
}

// Multiply + accumulate
def t2SMLAL : T2I<(outs GPR:$ldst, GPR:$hdst), (ins GPR:$a, GPR:$b), IIC_iMPYl,
                  "smlal", " $ldst, $hdst, $a, $b", []>;

def t2UMLAL : T2I<(outs GPR:$ldst, GPR:$hdst), (ins GPR:$a, GPR:$b), IIC_iMPYl,
                  "umlal", " $ldst, $hdst, $a, $b", []>;

def t2UMAAL : T2I<(outs GPR:$ldst, GPR:$hdst), (ins GPR:$a, GPR:$b), IIC_iMPYl,
                  "umaal", " $ldst, $hdst, $a, $b", []>;
} // neverHasSideEffects

// Most significant word multiply
def t2SMMUL : T2I<(outs GPR:$dst), (ins GPR:$a, GPR:$b), IIC_iMPYw,
                  "smmul", " $dst, $a, $b",
                  [(set GPR:$dst, (mulhs GPR:$a, GPR:$b))]>;

def t2SMMLA : T2I<(outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$c), IIC_iMPYw,
                  "smmla", " $dst, $a, $b, $c",
                  [(set GPR:$dst, (add (mulhs GPR:$a, GPR:$b), GPR:$c))]>;


def t2SMMLS : T2I <(outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$c), IIC_iMPYw,
                   "smmls", " $dst, $a, $b, $c",
                   [(set GPR:$dst, (sub GPR:$c, (mulhs GPR:$a, GPR:$b)))]>;

multiclass T2I_smul<string opc, PatFrag opnode> {
  def BB : T2I<(outs GPR:$dst), (ins GPR:$a, GPR:$b), IIC_iMPYw,
              !strconcat(opc, "bb"), " $dst, $a, $b",
              [(set GPR:$dst, (opnode (sext_inreg GPR:$a, i16),
                                      (sext_inreg GPR:$b, i16)))]>;

  def BT : T2I<(outs GPR:$dst), (ins GPR:$a, GPR:$b), IIC_iMPYw,
              !strconcat(opc, "bt"), " $dst, $a, $b",
              [(set GPR:$dst, (opnode (sext_inreg GPR:$a, i16),
                                      (sra GPR:$b, (i32 16))))]>;

  def TB : T2I<(outs GPR:$dst), (ins GPR:$a, GPR:$b), IIC_iMPYw,
              !strconcat(opc, "tb"), " $dst, $a, $b",
              [(set GPR:$dst, (opnode (sra GPR:$a, (i32 16)),
                                      (sext_inreg GPR:$b, i16)))]>;

  def TT : T2I<(outs GPR:$dst), (ins GPR:$a, GPR:$b), IIC_iMPYw,
              !strconcat(opc, "tt"), " $dst, $a, $b",
              [(set GPR:$dst, (opnode (sra GPR:$a, (i32 16)),
                                      (sra GPR:$b, (i32 16))))]>;

  def WB : T2I<(outs GPR:$dst), (ins GPR:$a, GPR:$b), IIC_iMPYh,
              !strconcat(opc, "wb"), " $dst, $a, $b",
              [(set GPR:$dst, (sra (opnode GPR:$a,
                                    (sext_inreg GPR:$b, i16)), (i32 16)))]>;

  def WT : T2I<(outs GPR:$dst), (ins GPR:$a, GPR:$b), IIC_iMPYh,
              !strconcat(opc, "wt"), " $dst, $a, $b",
              [(set GPR:$dst, (sra (opnode GPR:$a,
                                    (sra GPR:$b, (i32 16))), (i32 16)))]>;
}


multiclass T2I_smla<string opc, PatFrag opnode> {
  def BB : T2I<(outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$acc), IIC_iMPYw,
              !strconcat(opc, "bb"), " $dst, $a, $b, $acc",
              [(set GPR:$dst, (add GPR:$acc,
                               (opnode (sext_inreg GPR:$a, i16),
                                       (sext_inreg GPR:$b, i16))))]>;

  def BT : T2I<(outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$acc), IIC_iMPYw,
             !strconcat(opc, "bt"), " $dst, $a, $b, $acc",
             [(set GPR:$dst, (add GPR:$acc, (opnode (sext_inreg GPR:$a, i16),
                                                    (sra GPR:$b, (i32 16)))))]>;

  def TB : T2I<(outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$acc), IIC_iMPYw,
              !strconcat(opc, "tb"), " $dst, $a, $b, $acc",
              [(set GPR:$dst, (add GPR:$acc, (opnode (sra GPR:$a, (i32 16)),
                                                 (sext_inreg GPR:$b, i16))))]>;

  def TT : T2I<(outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$acc), IIC_iMPYw,
              !strconcat(opc, "tt"), " $dst, $a, $b, $acc",
             [(set GPR:$dst, (add GPR:$acc, (opnode (sra GPR:$a, (i32 16)),
                                                    (sra GPR:$b, (i32 16)))))]>;

  def WB : T2I<(outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$acc), IIC_iMPYw,
              !strconcat(opc, "wb"), " $dst, $a, $b, $acc",
              [(set GPR:$dst, (add GPR:$acc, (sra (opnode GPR:$a,
                                       (sext_inreg GPR:$b, i16)), (i32 16))))]>;

  def WT : T2I<(outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$acc), IIC_iMPYw,
              !strconcat(opc, "wt"), " $dst, $a, $b, $acc",
              [(set GPR:$dst, (add GPR:$acc, (sra (opnode GPR:$a,
                                         (sra GPR:$b, (i32 16))), (i32 16))))]>;
}

defm t2SMUL : T2I_smul<"smul", BinOpFrag<(mul node:$LHS, node:$RHS)>>;
defm t2SMLA : T2I_smla<"smla", BinOpFrag<(mul node:$LHS, node:$RHS)>>;

// TODO: Halfword multiple accumulate long: SMLAL<x><y>
// TODO: Dual halfword multiple: SMUAD, SMUSD, SMLAD, SMLSD, SMLALD, SMLSLD


//===----------------------------------------------------------------------===//
//  Misc. Arithmetic Instructions.
//

def t2CLZ : T2I<(outs GPR:$dst), (ins GPR:$src), IIC_iALU,
                "clz", " $dst, $src",
                [(set GPR:$dst, (ctlz GPR:$src))]>;

def t2REV : T2I<(outs GPR:$dst), (ins GPR:$src), IIC_iALU,
                "rev", ".w $dst, $src",
                [(set GPR:$dst, (bswap GPR:$src))]>;

def t2REV16 : T2I<(outs GPR:$dst), (ins GPR:$src), IIC_iALU,
                "rev16", ".w $dst, $src",
                [(set GPR:$dst,
                    (or (and (srl GPR:$src, (i32 8)), 0xFF),
                        (or (and (shl GPR:$src, (i32 8)), 0xFF00),
                            (or (and (srl GPR:$src, (i32 8)), 0xFF0000),
                                (and (shl GPR:$src, (i32 8)), 0xFF000000)))))]>;

def t2REVSH : T2I<(outs GPR:$dst), (ins GPR:$src), IIC_iALU,
                 "revsh", ".w $dst, $src",
                 [(set GPR:$dst,
                    (sext_inreg
                      (or (srl (and GPR:$src, 0xFFFF), (i32 8)),
                          (shl GPR:$src, (i32 8))), i16))]>;

def t2PKHBT : T2I<(outs GPR:$dst), (ins GPR:$src1, GPR:$src2, i32imm:$shamt),
                  IIC_iALU, "pkhbt", " $dst, $src1, $src2, LSL $shamt",
                  [(set GPR:$dst, (or (and GPR:$src1, 0xFFFF),
                                      (and (shl GPR:$src2, (i32 imm:$shamt)),
                                           0xFFFF0000)))]>;

// Alternate cases for PKHBT where identities eliminate some nodes.
def : T2Pat<(or (and GPR:$src1, 0xFFFF), (and GPR:$src2, 0xFFFF0000)),
            (t2PKHBT GPR:$src1, GPR:$src2, 0)>;
def : T2Pat<(or (and GPR:$src1, 0xFFFF), (shl GPR:$src2, imm16_31:$shamt)),
            (t2PKHBT GPR:$src1, GPR:$src2, imm16_31:$shamt)>;

def t2PKHTB : T2I<(outs GPR:$dst), (ins GPR:$src1, GPR:$src2, i32imm:$shamt),
                  IIC_iALU, "pkhtb", " $dst, $src1, $src2, ASR $shamt",
                  [(set GPR:$dst, (or (and GPR:$src1, 0xFFFF0000),
                                      (and (sra GPR:$src2, imm16_31:$shamt),
                                           0xFFFF)))]>;

// Alternate cases for PKHTB where identities eliminate some nodes.  Note that
// a shift amount of 0 is *not legal* here, it is PKHBT instead.
def : T2Pat<(or (and GPR:$src1, 0xFFFF0000), (srl GPR:$src2, (i32 16))),
            (t2PKHTB GPR:$src1, GPR:$src2, 16)>;
def : T2Pat<(or (and GPR:$src1, 0xFFFF0000),
                     (and (srl GPR:$src2, imm1_15:$shamt), 0xFFFF)),
            (t2PKHTB GPR:$src1, GPR:$src2, imm1_15:$shamt)>;

//===----------------------------------------------------------------------===//
//  Comparison Instructions...
//

defm t2CMP  : T2I_cmp_is<"cmp",
                         BinOpFrag<(ARMcmp node:$LHS, node:$RHS)>>;
defm t2CMPz : T2I_cmp_is<"cmp",
                         BinOpFrag<(ARMcmpZ node:$LHS, node:$RHS)>>;

defm t2CMN  : T2I_cmp_is<"cmn",
                         BinOpFrag<(ARMcmp node:$LHS,(ineg node:$RHS))>>;
defm t2CMNz : T2I_cmp_is<"cmn",
                         BinOpFrag<(ARMcmpZ node:$LHS,(ineg node:$RHS))>>;

def : T2Pat<(ARMcmp  GPR:$src, t2_so_imm_neg:$imm),
            (t2CMNri GPR:$src, t2_so_imm_neg:$imm)>;

def : T2Pat<(ARMcmpZ  GPR:$src, t2_so_imm_neg:$imm),
            (t2CMNri   GPR:$src, t2_so_imm_neg:$imm)>;

defm t2TST  : T2I_cmp_is<"tst",
                         BinOpFrag<(ARMcmpZ (and node:$LHS, node:$RHS), 0)>>;
defm t2TEQ  : T2I_cmp_is<"teq",
                         BinOpFrag<(ARMcmpZ (xor node:$LHS, node:$RHS), 0)>>;

// A8.6.27  CBNZ, CBZ - Compare and branch on (non)zero.
// Short range conditional branch. Looks awesome for loops. Need to figure
// out how to use this one.


// Conditional moves
// FIXME: should be able to write a pattern for ARMcmov, but can't use
// a two-value operand where a dag node expects two operands. :( 
def t2MOVCCr : T2I<(outs GPR:$dst), (ins GPR:$false, GPR:$true), IIC_iALU,
                   "mov", ".w $dst, $true",
      [/*(set GPR:$dst, (ARMcmov GPR:$false, GPR:$true, imm:$cc, CCR:$ccr))*/]>,
                RegConstraint<"$false = $dst">;

def t2MOVCCi : T2I<(outs GPR:$dst), (ins GPR:$false, t2_so_imm:$true), IIC_iALU,
                   "mov", ".w $dst, $true",
[/*(set GPR:$dst, (ARMcmov GPR:$false, t2_so_imm:$true, imm:$cc, CCR:$ccr))*/]>,
                   RegConstraint<"$false = $dst">;

def t2MOVCClsl : T2I<(outs GPR:$dst), (ins GPR:$false, GPR:$true, i32imm:$rhs),
                   IIC_iALU, "lsl", ".w $dst, $true, $rhs", []>,
                   RegConstraint<"$false = $dst">;
def t2MOVCClsr : T2I<(outs GPR:$dst), (ins GPR:$false, GPR:$true, i32imm:$rhs),
                   IIC_iALU, "lsr", ".w $dst, $true, $rhs", []>,
                   RegConstraint<"$false = $dst">;
def t2MOVCCasr : T2I<(outs GPR:$dst), (ins GPR:$false, GPR:$true, i32imm:$rhs),
                   IIC_iALU, "asr", ".w $dst, $true, $rhs", []>,
                   RegConstraint<"$false = $dst">;
def t2MOVCCror : T2I<(outs GPR:$dst), (ins GPR:$false, GPR:$true, i32imm:$rhs),
                   IIC_iALU, "ror", ".w $dst, $true, $rhs", []>,
                   RegConstraint<"$false = $dst">;

//===----------------------------------------------------------------------===//
// TLS Instructions
//

// __aeabi_read_tp preserves the registers r1-r3.
let isCall = 1,
  Defs = [R0, R12, LR, CPSR] in {
  def t2TPsoft : T2XI<(outs), (ins), IIC_Br,
                     "bl __aeabi_read_tp",
                     [(set R0, ARMthread_pointer)]>;
}

//===----------------------------------------------------------------------===//
// SJLJ Exception handling intrinsics
//   eh_sjlj_setjmp() is an instruction sequence to store the return
//   address and save #0 in R0 for the non-longjmp case.
//   Since by its nature we may be coming from some other function to get
//   here, and we're using the stack frame for the containing function to
//   save/restore registers, we can't keep anything live in regs across
//   the eh_sjlj_setjmp(), else it will almost certainly have been tromped upon
//   when we get here from a longjmp(). We force everthing out of registers
//   except for our own input by listing the relevant registers in Defs. By
//   doing so, we also cause the prologue/epilogue code to actively preserve
//   all of the callee-saved resgisters, which is exactly what we want.
let Defs = 
  [ R0,  R1,  R2,  R3,  R4,  R5,  R6,  R7,  R8,  R9,  R10, R11, R12, LR,  D0,
    D1,  D2,  D3,  D4,  D5,  D6,  D7,  D8,  D9,  D10, D11, D12, D13, D14, D15,
    D16, D17, D18, D19, D20, D21, D22, D23, D24, D25, D26, D27, D28, D29, D30,
    D31 ] in {
  def t2Int_eh_sjlj_setjmp : Thumb2XI<(outs), (ins GPR:$src),
                               AddrModeNone, SizeSpecial, NoItinerary,
                               "str.w sp, [$src, #+8] @ eh_setjmp begin\n"
                               "\tadr r12, 0f\n"
                               "\torr r12, #1\n"
                               "\tstr.w r12, [$src, #+4]\n"
                               "\tmovs r0, #0\n"
                               "\tb 1f\n"
                               "0:\tmovs r0, #1 @ eh_setjmp end\n"
                               "1:", "",
                               [(set R0, (ARMeh_sjlj_setjmp GPR:$src))]>;
}



//===----------------------------------------------------------------------===//
// Control-Flow Instructions
//

// FIXME: remove when we have a way to marking a MI with these properties.
// FIXME: $dst1 should be a def. But the extra ops must be in the end of the
// operand list.
// FIXME: Should pc be an implicit operand like PICADD, etc?
let isReturn = 1, isTerminator = 1, mayLoad = 1 in
  def t2LDM_RET : T2XI<(outs),
                    (ins addrmode4:$addr, pred:$p, reglist:$dst1, variable_ops),
                    IIC_Br, "ldm${addr:submode}${p}${addr:wide} $addr, $dst1",
                    []>;

let isBranch = 1, isTerminator = 1, isBarrier = 1 in {
let isPredicable = 1 in
def t2B   : T2XI<(outs), (ins brtarget:$target), IIC_Br,
                 "b.w $target",
                 [(br bb:$target)]>;

let isNotDuplicable = 1, isIndirectBranch = 1 in {
def t2BR_JT :
    T2JTI<(outs),
          (ins GPR:$target, GPR:$index, jt2block_operand:$jt, i32imm:$id),
           IIC_Br, "mov pc, $target\n$jt",
          [(ARMbr2jt GPR:$target, GPR:$index, tjumptable:$jt, imm:$id)]>;

// FIXME: Add a non-pc based case that can be predicated.
def t2TBB :
    T2JTI<(outs),
        (ins tb_addrmode:$index, jt2block_operand:$jt, i32imm:$id),
         IIC_Br, "tbb $index\n$jt", []>;

def t2TBH :
    T2JTI<(outs),
        (ins tb_addrmode:$index, jt2block_operand:$jt, i32imm:$id),
         IIC_Br, "tbh $index\n$jt", []>;
} // isNotDuplicable, isIndirectBranch

} // isBranch, isTerminator, isBarrier

// FIXME: should be able to write a pattern for ARMBrcond, but can't use
// a two-value operand where a dag node expects two operands. :(
let isBranch = 1, isTerminator = 1 in
def t2Bcc : T2I<(outs), (ins brtarget:$target), IIC_Br,
                "b", ".w $target",
                [/*(ARMbrcond bb:$target, imm:$cc)*/]>;


// IT block
def t2IT : Thumb2XI<(outs), (ins it_pred:$cc, it_mask:$mask),
                    AddrModeNone, Size2Bytes,  IIC_iALU,
                    "it$mask $cc", "", []>;

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//

// ConstantPool, GlobalAddress, and JumpTable
def : T2Pat<(ARMWrapper  tglobaladdr :$dst), (t2LEApcrel tglobaladdr :$dst)>;
def : T2Pat<(ARMWrapper  tconstpool  :$dst), (t2LEApcrel tconstpool  :$dst)>;
def : T2Pat<(ARMWrapperJT tjumptable:$dst, imm:$id),
            (t2LEApcrelJT tjumptable:$dst, imm:$id)>;

// Large immediate handling.

def : T2Pat<(i32 imm:$src),
            (t2MOVTi16 (t2MOVi16 (t2_lo16 imm:$src)), (t2_hi16 imm:$src))>;
