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
                ComplexPattern<i32, 2, "SelectShifterOperand",
                               [shl,srl,sra,rotr]> {
  let PrintMethod = "printSOOperand";
  let MIOperandInfo = (ops GPR, i32imm);
}

def LO16 : SDNodeXForm<imm, [{
  // Transformation function: shift the immediate value down into the low bits.
  return getI32Imm((unsigned short)N->getZExtValue());
}]>;

def HI16 : SDNodeXForm<imm, [{
  // Transformation function: shift the immediate value down into the low bits.
  return getI32Imm((unsigned)N->getZExtValue() >> 16);
}]>;

def imm16high : PatLeaf<(i32 imm), [{
  // Returns true if all bits out of the [31..16] range are 0.
  return ((N->getZExtValue() & 0xFFFF0000ULL) == N->getZExtValue());
}], HI16>;

def imm16high0xffff : PatLeaf<(i32 imm), [{
  // Returns true if lo 16 bits are set and this is a 32-bit value. 
  return ((N->getZExtValue() & 0x0000FFFFULL) == 0xFFFFULL);
}], HI16>;

def imm0_4095 : PatLeaf<(i32 imm), [{ 
  return (uint32_t)N->getZExtValue() < 4096; 
}]>; 

def imm0_4095_neg : PatLeaf<(i32 imm), [{ 
 return (uint32_t)-N->getZExtValue() < 4096; 
}], imm_neg_XFORM>; 

def imm0_65535 : PatLeaf<(i32 imm), [{ 
  return N->getZExtValue() < 65536; 
}]>; 

// A6.3.2 Modified immediate constants in Thumb instructions (#<const>)
// FIXME: Move it the the addrmode matcher code.
def t2_so_imm : PatLeaf<(i32 imm), [{
  uint64_t v = N->getZExtValue();
  if (v == 0 || v > 0xffffffffUL) return false;
  // variant1 - 0b0000x - 8-bit which could be zero (not supported for now)

  // variant2 - 0b00nnx - 8-bit repeated inside the 32-bit room
  unsigned hi16 = (unsigned)(v >> 16);
  unsigned lo16 = (unsigned)(v & 0xffffUL);
  bool valid = (hi16 == lo16) && (
    (v & 0x00ff00ffUL) == 0 ||        // type 0001x
    (v & 0xff00ff00UL) == 0 ||        // type 0010x
    ((lo16 >> 8) == (lo16 & 0xff)));  // type 0011x
  if (valid) return true;

  // variant3 - 0b01000..0b11111 - 8-bit shifted inside the 32-bit room
  unsigned shift = CountLeadingZeros_32(v);
  uint64_t mask = (0xff000000ULL >> shift);
  // If valid, it is type 01000 + shift
  return ((shift < 24) && (v & mask) > 0) && ((v & (~mask)) == 0);
}]>;


//===----------------------------------------------------------------------===//
//  Thumb-2 to cover the functionality of the ARM instruction set.
//

/// T2I_bin_irs - Defines a set of (op reg, {so_imm|reg|so_reg}) patterns for a
//  binary operation that produces a value.
multiclass T2I_bin_irs<string opc, PatFrag opnode> {
   // shifted imm
   def ri : PseudoInst<(outs GPR:$dst), (ins GPR:$lhs, i32imm:$rhs),
                       !strconcat(opc, " $dst, $lhs, $rhs"),
                       [(set GPR:$dst, (opnode GPR:$lhs, t2_so_imm:$rhs))]>,
                      Requires<[HasThumb2]>;
   // register
   def rr : PseudoInst<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs),
                       !strconcat(opc, " $dst, $lhs, $rhs"),
                       [(set GPR:$dst, (opnode GPR:$lhs, GPR:$rhs))]>, 
                      Requires<[HasThumb2]>;
   // shifted register
   def rs : PseudoInst<(outs GPR:$dst), (ins GPR:$lhs, t2_so_reg:$rhs),
                       !strconcat(opc, " $dst, $lhs, $rhs"),
                       [(set GPR:$dst, (opnode GPR:$lhs, t2_so_reg:$rhs))]>, 
                      Requires<[HasThumb2]>;
}

/// T2I_bin_s_irs - Similar to T2I_bin_irs except it sets the 's' bit so the
/// instruction modifies the CPSR register.
let Defs = [CPSR] in {
multiclass T2I_bin_s_irs<string opc, PatFrag opnode> {
   // shifted imm
   def ri : PseudoInst<(outs GPR:$dst), (ins GPR:$lhs, i32imm:$rhs),
                       !strconcat(opc, "s $dst, $lhs, $rhs"),
                       [(set GPR:$dst, (opnode GPR:$lhs, t2_so_imm:$rhs))]>,
                      Requires<[HasThumb2]>;

   // register
   def rr : PseudoInst<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs),
                       !strconcat(opc, "s $dst, $lhs, $rhs"),
                       [(set GPR:$dst, (opnode GPR:$lhs, GPR:$rhs))]>, 
                      Requires<[HasThumb2]>;

   // shifted register
   def rs : PseudoInst<(outs GPR:$dst), (ins GPR:$lhs, t2_so_reg:$rhs),
                       !strconcat(opc, "s $dst, $lhs, $rhs"),
                       [(set GPR:$dst, (opnode GPR:$lhs, t2_so_reg:$rhs))]>, 
                      Requires<[HasThumb2]>;
}
}

/// T2I_bin_c_irs - Similar to T2I_bin_irs except it uses the 's' bit. Also the
/// instruction can optionally set the CPSR register.
let Uses = [CPSR] in {
multiclass T2I_bin_c_irs<string opc, PatFrag opnode> {
   // shifted imm
   def ri : PseudoInst<(outs GPR:$dst), (ins GPR:$lhs, i32imm:$rhs, cc_out:$s),
                       !strconcat(opc, "${s} $dst, $lhs, $rhs"),
                       [(set GPR:$dst, (opnode GPR:$lhs, t2_so_imm:$rhs))]>,
                      Requires<[HasThumb2]>;

   // register
   def rr : PseudoInst<(outs GPR:$dst), (ins GPR:$lhs, GPR:$rhs, cc_out:$s),
                       !strconcat(opc, "${s} $dst, $lhs, $rhs"),
                       [(set GPR:$dst, (opnode GPR:$lhs, GPR:$rhs))]>, 
                      Requires<[HasThumb2]>;

   // shifted register
   def rs : PseudoInst<(outs GPR:$dst), (ins GPR:$lhs, t2_so_reg:$rhs, cc_out:$s),
                       !strconcat(opc, "${s} $dst, $lhs, $rhs"),
                       [(set GPR:$dst, (opnode GPR:$lhs, t2_so_reg:$rhs))]>, 
                      Requires<[HasThumb2]>;
}
}

//===----------------------------------------------------------------------===//
//  Arithmetic Instructions.
//

//===----------------------------------------------------------------------===//
//  Move Instructions.
//
def tMOVi16  : PseudoInst<(outs GPR:$dst), (ins i32imm:$src),
                          "movw $dst, $src",
                          [(set GPR:$dst, imm0_65535:$src)]>, 
                         Requires<[HasThumb2]>;

let Constraints = "$src = $dst" in
def tMOVTi16 : PseudoInst<(outs GPR:$dst), (ins GPR:$src, i32imm:$imm),
                          "movt $dst, $imm",
                          [(set GPR:$dst, (or (and GPR:$src, 0xffff), 
                                              imm16high:$imm))]>,
                         Requires<[HasThumb2]>;

def : Pat<(and (or GPR:$src, imm16high:$imm1), imm16high0xffff:$imm2),
          (tMOVTi16 GPR:$src, (HI16 imm16high:$imm1))>,
         Requires<[HasThumb2]>;

def : Pat<(i32 imm:$imm),
          (tMOVTi16 (tMOVi16 (LO16 imm:$imm)),(HI16 imm:$imm))>,
         Requires<[HasThumb2]>;

//===----------------------------------------------------------------------===//
//  Arithmetic Instructions.
//
defm t2ADD  : T2I_bin_irs <"add", BinOpFrag<(add node:$LHS, node:$RHS)>>;
defm t2SUB  : T2I_bin_irs <"sub", BinOpFrag<(sub node:$LHS, node:$RHS)>>;

def tADDri12 : PseudoInst<(outs GPR:$dst), (ins GPR:$lhs, i32imm:$rhs),
                          "add $dst, $lhs, $rhs", 
                          [(set GPR:$dst, (add GPR:$lhs, imm0_4095:$rhs))]>,
                         Requires<[HasThumb2]>; 
def tSUBri12 : PseudoInst<(outs GPR:$dst), (ins GPR:$lhs, i32imm:$rhs), 
                          "sub $dst, $lhs, $rhs",
                          [(set GPR:$dst, (add GPR:$lhs, imm0_4095_neg:$rhs))]>,
                         Requires<[HasThumb2]>;

defm t2ADDS : T2I_bin_s_irs<"add", BinOpFrag<(addc node:$LHS, node:$RHS)>>;
defm t2SUBS : T2I_bin_s_irs<"sub", BinOpFrag<(subc node:$LHS, node:$RHS)>>;

defm t2ADC : T2I_bin_c_irs<"adc", BinOpFrag<(adde node:$LHS, node:$RHS)>>;
defm t2SBC : T2I_bin_c_irs<"sbc", BinOpFrag<(sube node:$LHS, node:$RHS)>>;


def tMLS : PseudoInst<(outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$c), 
                      "mls $dst, $a, $b, $c", 
                      [(set GPR:$dst, (sub GPR:$c, (mul GPR:$a, GPR:$b)))]>,
                     Requires<[HasThumb2]>;

def tORNrs : PseudoInst<(outs GPR:$dst), (ins GPR:$src1, t2_so_reg:$src2),
                        "orn $dst, $src1, $src2",
                        [(set GPR:$dst, (or GPR:$src1, (not t2_so_reg: $src2)))]>,
                       Requires<[HasThumb2]>;
