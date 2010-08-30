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
  let PrintMethod = "printMandatoryPredicateOperand";
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
  let MIOperandInfo = (ops rGPR, i32imm);
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
// into t2_so_imm instructions: the 8-bit immediate is the least significant
// bits [bits 0-7], the 4-bit shift/splat amount is the next 4 bits [bits 8-11].
def t2_so_imm : Operand<i32>, PatLeaf<(imm), [{ return Pred_t2_so_imm(N); }]>;

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

// Break t2_so_imm's up into two pieces.  This handles immediates with up to 16
// bits set in them.  This uses t2_so_imm2part to match and t2_so_imm2part_[12]
// to get the first/second pieces.
def t2_so_imm2part : Operand<i32>,
                  PatLeaf<(imm), [{
      return ARM_AM::isT2SOImmTwoPartVal((unsigned)N->getZExtValue());
    }]> {
}

def t2_so_imm2part_1 : SDNodeXForm<imm, [{
  unsigned V = ARM_AM::getT2SOImmTwoPartFirst((unsigned)N->getZExtValue());
  return CurDAG->getTargetConstant(V, MVT::i32);
}]>;

def t2_so_imm2part_2 : SDNodeXForm<imm, [{
  unsigned V = ARM_AM::getT2SOImmTwoPartSecond((unsigned)N->getZExtValue());
  return CurDAG->getTargetConstant(V, MVT::i32);
}]>;

def t2_so_neg_imm2part : Operand<i32>, PatLeaf<(imm), [{
      return ARM_AM::isT2SOImmTwoPartVal(-(int)N->getZExtValue());
    }]> {
}

def t2_so_neg_imm2part_1 : SDNodeXForm<imm, [{
  unsigned V = ARM_AM::getT2SOImmTwoPartFirst(-(int)N->getZExtValue());
  return CurDAG->getTargetConstant(V, MVT::i32);
}]>;

def t2_so_neg_imm2part_2 : SDNodeXForm<imm, [{
  unsigned V = ARM_AM::getT2SOImmTwoPartSecond(-(int)N->getZExtValue());
  return CurDAG->getTargetConstant(V, MVT::i32);
}]>;

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

def imm0_255_not : PatLeaf<(i32 imm), [{
  return (uint32_t)(~N->getZExtValue()) < 255;
}], imm_comp_XFORM>;

// Define Thumb2 specific addressing modes.

// t2addrmode_imm12  := reg + imm12
def t2addrmode_imm12 : Operand<i32>,
                       ComplexPattern<i32, 2, "SelectT2AddrModeImm12", []> {
  let PrintMethod = "printT2AddrModeImm12Operand";
  let MIOperandInfo = (ops GPR:$base, i32imm:$offsimm);
}

// t2addrmode_imm8  := reg +/- imm8
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

def t2am_imm8s4_offset : Operand<i32> {
  let PrintMethod = "printT2AddrModeImm8s4OffsetOperand";
}

// t2addrmode_so_reg  := reg + (reg << imm2)
def t2addrmode_so_reg : Operand<i32>,
                        ComplexPattern<i32, 3, "SelectT2AddrModeSoReg", []> {
  let PrintMethod = "printT2AddrModeSoRegOperand";
  let MIOperandInfo = (ops GPR:$base, rGPR:$offsreg, i32imm:$offsimm);
}


//===----------------------------------------------------------------------===//
// Multiclass helpers...
//

/// T2I_un_irs - Defines a set of (op reg, {so_imm|r|so_reg}) patterns for a
/// unary operation that produces a value. These are predicable and can be
/// changed to modify CPSR.
multiclass T2I_un_irs<bits<4> opcod, string opc, PatFrag opnode,
                      bit Cheap = 0, bit ReMat = 0> {
   // shifted imm
   def i : T2sI<(outs rGPR:$dst), (ins t2_so_imm:$src), IIC_iMOVi,
                opc, "\t$dst, $src",
                [(set rGPR:$dst, (opnode t2_so_imm:$src))]> {
     let isAsCheapAsAMove = Cheap;
     let isReMaterializable = ReMat;
     let Inst{31-27} = 0b11110;
     let Inst{25} = 0;
     let Inst{24-21} = opcod;
     let Inst{20} = ?; // The S bit.
     let Inst{19-16} = 0b1111; // Rn
     let Inst{15} = 0;
   }
   // register
   def r : T2sI<(outs rGPR:$dst), (ins rGPR:$src), IIC_iMOVr,
                opc, ".w\t$dst, $src",
                [(set rGPR:$dst, (opnode rGPR:$src))]> {
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24-21} = opcod;
     let Inst{20} = ?; // The S bit.
     let Inst{19-16} = 0b1111; // Rn
     let Inst{14-12} = 0b000; // imm3
     let Inst{7-6} = 0b00; // imm2
     let Inst{5-4} = 0b00; // type
   }
   // shifted register
   def s : T2sI<(outs rGPR:$dst), (ins t2_so_reg:$src), IIC_iMOVsi,
                opc, ".w\t$dst, $src",
                [(set rGPR:$dst, (opnode t2_so_reg:$src))]> {
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24-21} = opcod;
     let Inst{20} = ?; // The S bit.
     let Inst{19-16} = 0b1111; // Rn
   }
}

/// T2I_bin_irs - Defines a set of (op reg, {so_imm|r|so_reg}) patterns for a
/// binary operation that produces a value. These are predicable and can be
/// changed to modify CPSR.
multiclass T2I_bin_irs<bits<4> opcod, string opc, PatFrag opnode,
                       bit Commutable = 0, string wide = ""> {
   // shifted imm
   def ri : T2sI<(outs rGPR:$dst), (ins rGPR:$lhs, t2_so_imm:$rhs), IIC_iALUi,
                 opc, "\t$dst, $lhs, $rhs",
                 [(set rGPR:$dst, (opnode rGPR:$lhs, t2_so_imm:$rhs))]> {
     let Inst{31-27} = 0b11110;
     let Inst{25} = 0;
     let Inst{24-21} = opcod;
     let Inst{20} = ?; // The S bit.
     let Inst{15} = 0;
   }
   // register
   def rr : T2sI<(outs rGPR:$dst), (ins rGPR:$lhs, rGPR:$rhs), IIC_iALUr,
                 opc, !strconcat(wide, "\t$dst, $lhs, $rhs"),
                 [(set rGPR:$dst, (opnode rGPR:$lhs, rGPR:$rhs))]> {
     let isCommutable = Commutable;
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24-21} = opcod;
     let Inst{20} = ?; // The S bit.
     let Inst{14-12} = 0b000; // imm3
     let Inst{7-6} = 0b00; // imm2
     let Inst{5-4} = 0b00; // type
   }
   // shifted register
   def rs : T2sI<(outs rGPR:$dst), (ins rGPR:$lhs, t2_so_reg:$rhs), IIC_iALUsi,
                 opc, !strconcat(wide, "\t$dst, $lhs, $rhs"),
                 [(set rGPR:$dst, (opnode rGPR:$lhs, t2_so_reg:$rhs))]> {
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24-21} = opcod;
     let Inst{20} = ?; // The S bit.
   }
}

/// T2I_bin_cpsr_irs - Defines a set of (op reg, {so_imm|r|so_reg}) patterns for
/// a binary operation that produces a value. These are predicable and modify
/// CPSR.
multiclass T2I_bin_cpsr_irs<bits<4> opcod, string opc, PatFrag opnode,
                            bit Commutable = 0, string wide = ""> {
   // shifted imm
   def ri : T2sI_cpsr<(outs rGPR:$dst), (ins rGPR:$lhs, t2_so_imm:$rhs),
                      IIC_iALUi, opc, "\t$dst, $lhs, $rhs",
                 [(set rGPR:$dst, (opnode rGPR:$lhs, t2_so_imm:$rhs))]> {
     let Inst{31-27} = 0b11110;
     let Inst{25} = 0;
     let Inst{24-21} = opcod;
     let Inst{20} = 1; // The S bit.
     let Inst{15} = 0;
   }
   // register
   def rr : T2sI_cpsr<(outs rGPR:$dst), (ins rGPR:$lhs, rGPR:$rhs),
                      IIC_iALUr, opc, !strconcat(wide, "\t$dst, $lhs, $rhs"),
                 [(set rGPR:$dst, (opnode rGPR:$lhs, rGPR:$rhs))]> {
     let isCommutable = Commutable;
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24-21} = opcod;
     let Inst{20} = 1; // The S bit.
     let Inst{14-12} = 0b000; // imm3
     let Inst{7-6} = 0b00; // imm2
     let Inst{5-4} = 0b00; // type
   }
   // shifted register
   def rs : T2sI_cpsr<(outs rGPR:$dst), (ins rGPR:$lhs, t2_so_reg:$rhs),
                      IIC_iALUsi, opc, !strconcat(wide, "\t$dst, $lhs, $rhs"),
                 [(set rGPR:$dst, (opnode rGPR:$lhs, t2_so_reg:$rhs))]> {
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24-21} = opcod;
     let Inst{20} = 1; // The S bit.
   }
}

/// T2I_bin_w_irs - Same as T2I_bin_irs except these operations need
//  the ".w" prefix to indicate that they are wide.
multiclass T2I_bin_w_irs<bits<4> opcod, string opc, PatFrag opnode,
                         bit Commutable = 0> :
    T2I_bin_irs<opcod, opc, opnode, Commutable, ".w">;

/// T2I_bin_sw_irs - Same as T2I_bin_w_irs except these operations set
//  the 'S' bit.
multiclass T2I_bin_sw_irs<bits<4> opcod, string opc, PatFrag opnode,
                         bit Commutable = 0> :
    T2I_bin_cpsr_irs<opcod, opc, opnode, Commutable, ".w">;

/// T2I_rbin_is - Same as T2I_bin_irs except the order of operands are
/// reversed.  The 'rr' form is only defined for the disassembler; for codegen
/// it is equivalent to the T2I_bin_irs counterpart.
multiclass T2I_rbin_irs<bits<4> opcod, string opc, PatFrag opnode> {
   // shifted imm
   def ri : T2sI<(outs rGPR:$dst), (ins rGPR:$rhs, t2_so_imm:$lhs), IIC_iALUi,
                 opc, ".w\t$dst, $rhs, $lhs",
                 [(set rGPR:$dst, (opnode t2_so_imm:$lhs, rGPR:$rhs))]> {
     let Inst{31-27} = 0b11110;
     let Inst{25} = 0;
     let Inst{24-21} = opcod;
     let Inst{20} = ?; // The S bit.
     let Inst{15} = 0;
   }
   // register
   def rr : T2sI<(outs rGPR:$dst), (ins rGPR:$rhs, rGPR:$lhs), IIC_iALUr,
                 opc, "\t$dst, $rhs, $lhs",
                 [/* For disassembly only; pattern left blank */]> {
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24-21} = opcod;
     let Inst{20} = ?; // The S bit.
     let Inst{14-12} = 0b000; // imm3
     let Inst{7-6} = 0b00; // imm2
     let Inst{5-4} = 0b00; // type
   }
   // shifted register
   def rs : T2sI<(outs rGPR:$dst), (ins rGPR:$rhs, t2_so_reg:$lhs), IIC_iALUsi,
                 opc, "\t$dst, $rhs, $lhs",
                 [(set rGPR:$dst, (opnode t2_so_reg:$lhs, rGPR:$rhs))]> {
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24-21} = opcod;
     let Inst{20} = ?; // The S bit.
   }
}

/// T2I_bin_s_irs - Similar to T2I_bin_irs except it sets the 's' bit so the
/// instruction modifies the CPSR register.
let Defs = [CPSR] in {
multiclass T2I_bin_s_irs<bits<4> opcod, string opc, PatFrag opnode,
                         bit Commutable = 0> {
   // shifted imm
   def ri : T2I<(outs rGPR:$dst), (ins GPR:$lhs, t2_so_imm:$rhs), IIC_iALUi,
                !strconcat(opc, "s"), ".w\t$dst, $lhs, $rhs",
                [(set rGPR:$dst, (opnode GPR:$lhs, t2_so_imm:$rhs))]> {
     let Inst{31-27} = 0b11110;
     let Inst{25} = 0;
     let Inst{24-21} = opcod;
     let Inst{20} = 1; // The S bit.
     let Inst{15} = 0;
   }
   // register
   def rr : T2I<(outs rGPR:$dst), (ins GPR:$lhs, rGPR:$rhs), IIC_iALUr,
                !strconcat(opc, "s"), ".w\t$dst, $lhs, $rhs",
                [(set rGPR:$dst, (opnode GPR:$lhs, rGPR:$rhs))]> {
     let isCommutable = Commutable;
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24-21} = opcod;
     let Inst{20} = 1; // The S bit.
     let Inst{14-12} = 0b000; // imm3
     let Inst{7-6} = 0b00; // imm2
     let Inst{5-4} = 0b00; // type
   }
   // shifted register
   def rs : T2I<(outs rGPR:$dst), (ins GPR:$lhs, t2_so_reg:$rhs), IIC_iALUsi,
                !strconcat(opc, "s"), ".w\t$dst, $lhs, $rhs",
                [(set rGPR:$dst, (opnode GPR:$lhs, t2_so_reg:$rhs))]> {
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24-21} = opcod;
     let Inst{20} = 1; // The S bit.
   }
}
}

/// T2I_bin_ii12rs - Defines a set of (op reg, {so_imm|imm0_4095|r|so_reg})
/// patterns for a binary operation that produces a value.
multiclass T2I_bin_ii12rs<bits<3> op23_21, string opc, PatFrag opnode,
                          bit Commutable = 0> {
   // shifted imm
   def ri : T2sI<(outs rGPR:$dst), (ins GPR:$lhs, t2_so_imm:$rhs), IIC_iALUi,
                 opc, ".w\t$dst, $lhs, $rhs",
                 [(set rGPR:$dst, (opnode GPR:$lhs, t2_so_imm:$rhs))]> {
     let Inst{31-27} = 0b11110;
     let Inst{25} = 0;
     let Inst{24} = 1;
     let Inst{23-21} = op23_21;
     let Inst{20} = 0; // The S bit.
     let Inst{15} = 0;
   }
   // 12-bit imm
   def ri12 : T2I<(outs rGPR:$dst), (ins GPR:$lhs, imm0_4095:$rhs), IIC_iALUi,
                  !strconcat(opc, "w"), "\t$dst, $lhs, $rhs",
                  [(set rGPR:$dst, (opnode GPR:$lhs, imm0_4095:$rhs))]> {
     let Inst{31-27} = 0b11110;
     let Inst{25} = 1;
     let Inst{24} = 0;
     let Inst{23-21} = op23_21;
     let Inst{20} = 0; // The S bit.
     let Inst{15} = 0;
   }
   // register
   def rr : T2sI<(outs rGPR:$dst), (ins GPR:$lhs, rGPR:$rhs), IIC_iALUr,
                 opc, ".w\t$dst, $lhs, $rhs",
                 [(set rGPR:$dst, (opnode GPR:$lhs, rGPR:$rhs))]> {
     let isCommutable = Commutable;
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24} = 1;
     let Inst{23-21} = op23_21;
     let Inst{20} = 0; // The S bit.
     let Inst{14-12} = 0b000; // imm3
     let Inst{7-6} = 0b00; // imm2
     let Inst{5-4} = 0b00; // type
   }
   // shifted register
   def rs : T2sI<(outs rGPR:$dst), (ins GPR:$lhs, t2_so_reg:$rhs), IIC_iALUsi,
                 opc, ".w\t$dst, $lhs, $rhs",
                 [(set rGPR:$dst, (opnode GPR:$lhs, t2_so_reg:$rhs))]> {
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24} = 1;
     let Inst{23-21} = op23_21;
     let Inst{20} = 0; // The S bit.
   }
}

/// T2I_adde_sube_irs - Defines a set of (op reg, {so_imm|r|so_reg}) patterns
/// for a binary operation that produces a value and use the carry
/// bit. It's not predicable.
let Uses = [CPSR] in {
multiclass T2I_adde_sube_irs<bits<4> opcod, string opc, PatFrag opnode,
                             bit Commutable = 0> {
   // shifted imm
   def ri : T2sI<(outs rGPR:$dst), (ins rGPR:$lhs, t2_so_imm:$rhs), IIC_iALUi,
                 opc, "\t$dst, $lhs, $rhs",
                 [(set rGPR:$dst, (opnode rGPR:$lhs, t2_so_imm:$rhs))]>,
                 Requires<[IsThumb2]> {
     let Inst{31-27} = 0b11110;
     let Inst{25} = 0;
     let Inst{24-21} = opcod;
     let Inst{20} = 0; // The S bit.
     let Inst{15} = 0;
   }
   // register
   def rr : T2sI<(outs rGPR:$dst), (ins rGPR:$lhs, rGPR:$rhs), IIC_iALUr,
                 opc, ".w\t$dst, $lhs, $rhs",
                 [(set rGPR:$dst, (opnode rGPR:$lhs, rGPR:$rhs))]>,
                 Requires<[IsThumb2]> {
     let isCommutable = Commutable;
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24-21} = opcod;
     let Inst{20} = 0; // The S bit.
     let Inst{14-12} = 0b000; // imm3
     let Inst{7-6} = 0b00; // imm2
     let Inst{5-4} = 0b00; // type
   }
   // shifted register
   def rs : T2sI<(outs rGPR:$dst), (ins rGPR:$lhs, t2_so_reg:$rhs), IIC_iALUsi,
                 opc, ".w\t$dst, $lhs, $rhs",
                 [(set rGPR:$dst, (opnode rGPR:$lhs, t2_so_reg:$rhs))]>,
                 Requires<[IsThumb2]> {
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24-21} = opcod;
     let Inst{20} = 0; // The S bit.
   }
}

// Carry setting variants
let Defs = [CPSR] in {
multiclass T2I_adde_sube_s_irs<bits<4> opcod, string opc, PatFrag opnode,
                               bit Commutable = 0> {
   // shifted imm
   def ri : T2sI<(outs rGPR:$dst), (ins rGPR:$lhs, t2_so_imm:$rhs), IIC_iALUi,
                 opc, "\t$dst, $lhs, $rhs",
                 [(set rGPR:$dst, (opnode rGPR:$lhs, t2_so_imm:$rhs))]>,
                 Requires<[IsThumb2]> {
     let Inst{31-27} = 0b11110;
     let Inst{25} = 0;
     let Inst{24-21} = opcod;
     let Inst{20} = 1; // The S bit.
     let Inst{15} = 0;
   }
   // register
   def rr : T2sI<(outs rGPR:$dst), (ins rGPR:$lhs, rGPR:$rhs), IIC_iALUr,
                 opc, ".w\t$dst, $lhs, $rhs",
                 [(set rGPR:$dst, (opnode rGPR:$lhs, rGPR:$rhs))]>,
                 Requires<[IsThumb2]> {
     let isCommutable = Commutable;
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24-21} = opcod;
     let Inst{20} = 1; // The S bit.
     let Inst{14-12} = 0b000; // imm3
     let Inst{7-6} = 0b00; // imm2
     let Inst{5-4} = 0b00; // type
   }
   // shifted register
   def rs : T2sI<(outs rGPR:$dst), (ins rGPR:$lhs, t2_so_reg:$rhs), IIC_iALUsi,
                 opc, ".w\t$dst, $lhs, $rhs",
                 [(set rGPR:$dst, (opnode rGPR:$lhs, t2_so_reg:$rhs))]>,
                 Requires<[IsThumb2]> {
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24-21} = opcod;
     let Inst{20} = 1; // The S bit.
   }
}
}
}

/// T2I_rbin_s_is - Same as T2I_rbin_irs except sets 's' bit and the register
/// version is not needed since this is only for codegen.
let Defs = [CPSR] in {
multiclass T2I_rbin_s_is<bits<4> opcod, string opc, PatFrag opnode> {
   // shifted imm
   def ri : T2I<(outs rGPR:$dst), (ins rGPR:$rhs, t2_so_imm:$lhs), IIC_iALUi,
                !strconcat(opc, "s"), ".w\t$dst, $rhs, $lhs",
                [(set rGPR:$dst, (opnode t2_so_imm:$lhs, rGPR:$rhs))]> {
     let Inst{31-27} = 0b11110;
     let Inst{25} = 0;
     let Inst{24-21} = opcod;
     let Inst{20} = 1; // The S bit.
     let Inst{15} = 0;
   }
   // shifted register
   def rs : T2I<(outs rGPR:$dst), (ins rGPR:$rhs, t2_so_reg:$lhs), IIC_iALUsi,
                !strconcat(opc, "s"), "\t$dst, $rhs, $lhs",
                [(set rGPR:$dst, (opnode t2_so_reg:$lhs, rGPR:$rhs))]> {
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24-21} = opcod;
     let Inst{20} = 1; // The S bit.
   }
}
}

/// T2I_sh_ir - Defines a set of (op reg, {so_imm|r}) patterns for a shift /
//  rotate operation that produces a value.
multiclass T2I_sh_ir<bits<2> opcod, string opc, PatFrag opnode> {
   // 5-bit imm
   def ri : T2sI<(outs rGPR:$dst), (ins rGPR:$lhs, i32imm:$rhs), IIC_iMOVsi,
                 opc, ".w\t$dst, $lhs, $rhs",
                 [(set rGPR:$dst, (opnode rGPR:$lhs, imm1_31:$rhs))]> {
     let Inst{31-27} = 0b11101;
     let Inst{26-21} = 0b010010;
     let Inst{19-16} = 0b1111; // Rn
     let Inst{5-4} = opcod;
   }
   // register
   def rr : T2sI<(outs rGPR:$dst), (ins rGPR:$lhs, rGPR:$rhs), IIC_iMOVsr,
                 opc, ".w\t$dst, $lhs, $rhs",
                 [(set rGPR:$dst, (opnode rGPR:$lhs, rGPR:$rhs))]> {
     let Inst{31-27} = 0b11111;
     let Inst{26-23} = 0b0100;
     let Inst{22-21} = opcod;
     let Inst{15-12} = 0b1111;
     let Inst{7-4} = 0b0000;
   }
}

/// T2I_cmp_irs - Defines a set of (op r, {so_imm|r|so_reg}) cmp / test
/// patterns. Similar to T2I_bin_irs except the instruction does not produce
/// a explicit result, only implicitly set CPSR.
let isCompare = 1, Defs = [CPSR] in {
multiclass T2I_cmp_irs<bits<4> opcod, string opc, PatFrag opnode> {
   // shifted imm
   def ri : T2I<(outs), (ins GPR:$lhs, t2_so_imm:$rhs), IIC_iCMPi,
                opc, ".w\t$lhs, $rhs",
                [(opnode GPR:$lhs, t2_so_imm:$rhs)]> {
     let Inst{31-27} = 0b11110;
     let Inst{25} = 0;
     let Inst{24-21} = opcod;
     let Inst{20} = 1; // The S bit.
     let Inst{15} = 0;
     let Inst{11-8} = 0b1111; // Rd
   }
   // register
   def rr : T2I<(outs), (ins GPR:$lhs, rGPR:$rhs), IIC_iCMPr,
                opc, ".w\t$lhs, $rhs",
                [(opnode GPR:$lhs, rGPR:$rhs)]> {
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24-21} = opcod;
     let Inst{20} = 1; // The S bit.
     let Inst{14-12} = 0b000; // imm3
     let Inst{11-8} = 0b1111; // Rd
     let Inst{7-6} = 0b00; // imm2
     let Inst{5-4} = 0b00; // type
   }
   // shifted register
   def rs : T2I<(outs), (ins GPR:$lhs, t2_so_reg:$rhs), IIC_iCMPsi,
                opc, ".w\t$lhs, $rhs",
                [(opnode GPR:$lhs, t2_so_reg:$rhs)]> {
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24-21} = opcod;
     let Inst{20} = 1; // The S bit.
     let Inst{11-8} = 0b1111; // Rd
   }
}
}

/// T2I_ld - Defines a set of (op r, {imm12|imm8|so_reg}) load patterns.
multiclass T2I_ld<bit signed, bits<2> opcod, string opc, PatFrag opnode> {
  def i12 : T2Ii12<(outs GPR:$dst), (ins t2addrmode_imm12:$addr), IIC_iLoadi,
                   opc, ".w\t$dst, $addr",
                   [(set GPR:$dst, (opnode t2addrmode_imm12:$addr))]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-25} = 0b00;
    let Inst{24} = signed;
    let Inst{23} = 1;
    let Inst{22-21} = opcod;
    let Inst{20} = 1; // load
  }
  def i8  : T2Ii8 <(outs GPR:$dst), (ins t2addrmode_imm8:$addr), IIC_iLoadi,
                   opc, "\t$dst, $addr",
                   [(set GPR:$dst, (opnode t2addrmode_imm8:$addr))]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-25} = 0b00;
    let Inst{24} = signed;
    let Inst{23} = 0;
    let Inst{22-21} = opcod;
    let Inst{20} = 1; // load
    let Inst{11} = 1;
    // Offset: index==TRUE, wback==FALSE
    let Inst{10} = 1; // The P bit.
    let Inst{8} = 0; // The W bit.
  }
  def s   : T2Iso <(outs GPR:$dst), (ins t2addrmode_so_reg:$addr), IIC_iLoadr,
                   opc, ".w\t$dst, $addr",
                   [(set GPR:$dst, (opnode t2addrmode_so_reg:$addr))]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-25} = 0b00;
    let Inst{24} = signed;
    let Inst{23} = 0;
    let Inst{22-21} = opcod;
    let Inst{20} = 1; // load
    let Inst{11-6} = 0b000000;
  }
  def pci : T2Ipc <(outs GPR:$dst), (ins i32imm:$addr), IIC_iLoadi,
                   opc, ".w\t$dst, $addr",
                   [(set GPR:$dst, (opnode (ARMWrapper tconstpool:$addr)))]> {
    let isReMaterializable = 1;
    let Inst{31-27} = 0b11111;
    let Inst{26-25} = 0b00;
    let Inst{24} = signed;
    let Inst{23} = ?; // add = (U == '1')
    let Inst{22-21} = opcod;
    let Inst{20} = 1; // load
    let Inst{19-16} = 0b1111; // Rn
  }
}

/// T2I_st - Defines a set of (op r, {imm12|imm8|so_reg}) store patterns.
multiclass T2I_st<bits<2> opcod, string opc, PatFrag opnode> {
  def i12 : T2Ii12<(outs), (ins GPR:$src, t2addrmode_imm12:$addr), IIC_iStorei,
                   opc, ".w\t$src, $addr",
                   [(opnode GPR:$src, t2addrmode_imm12:$addr)]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0001;
    let Inst{22-21} = opcod;
    let Inst{20} = 0; // !load
  }
  def i8  : T2Ii8 <(outs), (ins GPR:$src, t2addrmode_imm8:$addr), IIC_iStorei,
                   opc, "\t$src, $addr",
                   [(opnode GPR:$src, t2addrmode_imm8:$addr)]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0000;
    let Inst{22-21} = opcod;
    let Inst{20} = 0; // !load
    let Inst{11} = 1;
    // Offset: index==TRUE, wback==FALSE
    let Inst{10} = 1; // The P bit.
    let Inst{8} = 0; // The W bit.
  }
  def s   : T2Iso <(outs), (ins GPR:$src, t2addrmode_so_reg:$addr), IIC_iStorer,
                   opc, ".w\t$src, $addr",
                   [(opnode GPR:$src, t2addrmode_so_reg:$addr)]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0000;
    let Inst{22-21} = opcod;
    let Inst{20} = 0; // !load
    let Inst{11-6} = 0b000000;
  }
}

/// T2I_unary_rrot - A unary operation with two forms: one whose operand is a
/// register and one whose operand is a register rotated by 8/16/24.
multiclass T2I_unary_rrot<bits<3> opcod, string opc, PatFrag opnode> {
  def r     : T2I<(outs rGPR:$dst), (ins rGPR:$src), IIC_iUNAr,
                  opc, ".w\t$dst, $src",
                 [(set rGPR:$dst, (opnode rGPR:$src))]> {
     let Inst{31-27} = 0b11111;
     let Inst{26-23} = 0b0100;
     let Inst{22-20} = opcod;
     let Inst{19-16} = 0b1111; // Rn
     let Inst{15-12} = 0b1111;
     let Inst{7} = 1;
     let Inst{5-4} = 0b00; // rotate
   }
  def r_rot : T2I<(outs rGPR:$dst), (ins rGPR:$src, i32imm:$rot), IIC_iUNAsi,
                  opc, ".w\t$dst, $src, ror $rot",
                 [(set rGPR:$dst, (opnode (rotr rGPR:$src, rot_imm:$rot)))]> {
     let Inst{31-27} = 0b11111;
     let Inst{26-23} = 0b0100;
     let Inst{22-20} = opcod;
     let Inst{19-16} = 0b1111; // Rn
     let Inst{15-12} = 0b1111;
     let Inst{7} = 1;
     let Inst{5-4} = {?,?}; // rotate
   }
}

// UXTB16 - Requres T2ExtractPack, does not need the .w qualifier.
multiclass T2I_unary_rrot_uxtb16<bits<3> opcod, string opc, PatFrag opnode> {
  def r     : T2I<(outs rGPR:$dst), (ins rGPR:$src), IIC_iUNAr,
                  opc, "\t$dst, $src",
                 [(set rGPR:$dst, (opnode rGPR:$src))]>,
                 Requires<[HasT2ExtractPack]> {
     let Inst{31-27} = 0b11111;
     let Inst{26-23} = 0b0100;
     let Inst{22-20} = opcod;
     let Inst{19-16} = 0b1111; // Rn
     let Inst{15-12} = 0b1111;
     let Inst{7} = 1;
     let Inst{5-4} = 0b00; // rotate
   }
  def r_rot : T2I<(outs rGPR:$dst), (ins rGPR:$src, i32imm:$rot), IIC_iUNAsi,
                  opc, "\t$dst, $src, ror $rot",
                 [(set rGPR:$dst, (opnode (rotr rGPR:$src, rot_imm:$rot)))]>,
                 Requires<[HasT2ExtractPack]> {
     let Inst{31-27} = 0b11111;
     let Inst{26-23} = 0b0100;
     let Inst{22-20} = opcod;
     let Inst{19-16} = 0b1111; // Rn
     let Inst{15-12} = 0b1111;
     let Inst{7} = 1;
     let Inst{5-4} = {?,?}; // rotate
   }
}

// SXTB16 - Requres T2ExtractPack, does not need the .w qualifier, no pattern
// supported yet.
multiclass T2I_unary_rrot_sxtb16<bits<3> opcod, string opc> {
  def r     : T2I<(outs rGPR:$dst), (ins rGPR:$src), IIC_iUNAr,
                  opc, "\t$dst, $src", []> {
     let Inst{31-27} = 0b11111;
     let Inst{26-23} = 0b0100;
     let Inst{22-20} = opcod;
     let Inst{19-16} = 0b1111; // Rn
     let Inst{15-12} = 0b1111;
     let Inst{7} = 1;
     let Inst{5-4} = 0b00; // rotate
   }
  def r_rot : T2I<(outs rGPR:$dst), (ins rGPR:$src, i32imm:$rot), IIC_iUNAsi,
                  opc, "\t$dst, $src, ror $rot", []> {
     let Inst{31-27} = 0b11111;
     let Inst{26-23} = 0b0100;
     let Inst{22-20} = opcod;
     let Inst{19-16} = 0b1111; // Rn
     let Inst{15-12} = 0b1111;
     let Inst{7} = 1;
     let Inst{5-4} = {?,?}; // rotate
   }
}

/// T2I_bin_rrot - A binary operation with two forms: one whose operand is a
/// register and one whose operand is a register rotated by 8/16/24.
multiclass T2I_bin_rrot<bits<3> opcod, string opc, PatFrag opnode> {
  def rr     : T2I<(outs rGPR:$dst), (ins rGPR:$LHS, rGPR:$RHS), IIC_iALUr,
                  opc, "\t$dst, $LHS, $RHS",
                  [(set rGPR:$dst, (opnode rGPR:$LHS, rGPR:$RHS))]>,
                  Requires<[HasT2ExtractPack]> {
     let Inst{31-27} = 0b11111;
     let Inst{26-23} = 0b0100;
     let Inst{22-20} = opcod;
     let Inst{15-12} = 0b1111;
     let Inst{7} = 1;
     let Inst{5-4} = 0b00; // rotate
   }
  def rr_rot : T2I<(outs rGPR:$dst), (ins rGPR:$LHS, rGPR:$RHS, i32imm:$rot),
                  IIC_iALUsr, opc, "\t$dst, $LHS, $RHS, ror $rot",
                  [(set rGPR:$dst, (opnode rGPR:$LHS,
                                          (rotr rGPR:$RHS, rot_imm:$rot)))]>,
                  Requires<[HasT2ExtractPack]> {
     let Inst{31-27} = 0b11111;
     let Inst{26-23} = 0b0100;
     let Inst{22-20} = opcod;
     let Inst{15-12} = 0b1111;
     let Inst{7} = 1;
     let Inst{5-4} = {?,?}; // rotate
   }
}

// DO variant - disassembly only, no pattern

multiclass T2I_bin_rrot_DO<bits<3> opcod, string opc> {
  def rr     : T2I<(outs rGPR:$dst), (ins rGPR:$LHS, rGPR:$RHS), IIC_iALUr,
                  opc, "\t$dst, $LHS, $RHS", []> {
     let Inst{31-27} = 0b11111;
     let Inst{26-23} = 0b0100;
     let Inst{22-20} = opcod;
     let Inst{15-12} = 0b1111;
     let Inst{7} = 1;
     let Inst{5-4} = 0b00; // rotate
   }
  def rr_rot : T2I<(outs rGPR:$dst), (ins rGPR:$LHS, rGPR:$RHS, i32imm:$rot),
                  IIC_iALUsr, opc, "\t$dst, $LHS, $RHS, ror $rot", []> {
     let Inst{31-27} = 0b11111;
     let Inst{26-23} = 0b0100;
     let Inst{22-20} = opcod;
     let Inst{15-12} = 0b1111;
     let Inst{7} = 1;
     let Inst{5-4} = {?,?}; // rotate
   }
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Miscellaneous Instructions.
//

// LEApcrel - Load a pc-relative address into a register without offending the
// assembler.
let neverHasSideEffects = 1 in {
let isReMaterializable = 1 in
def t2LEApcrel : T2XI<(outs rGPR:$dst), (ins i32imm:$label, pred:$p), IIC_iALUi,
                      "adr${p}.w\t$dst, #$label", []> {
  let Inst{31-27} = 0b11110;
  let Inst{25-24} = 0b10;
  // Inst{23:21} = '11' (add = FALSE) or '00' (add = TRUE)
  let Inst{22} = 0;
  let Inst{20} = 0;
  let Inst{19-16} = 0b1111; // Rn
  let Inst{15} = 0;
}
} // neverHasSideEffects
def t2LEApcrelJT : T2XI<(outs rGPR:$dst),
                        (ins i32imm:$label, nohash_imm:$id, pred:$p), IIC_iALUi,
                        "adr${p}.w\t$dst, #${label}_${id}", []> {
  let Inst{31-27} = 0b11110;
  let Inst{25-24} = 0b10;
  // Inst{23:21} = '11' (add = FALSE) or '00' (add = TRUE)
  let Inst{22} = 0;
  let Inst{20} = 0;
  let Inst{19-16} = 0b1111; // Rn
  let Inst{15} = 0;
}

// ADD r, sp, {so_imm|i12}
def t2ADDrSPi   : T2sI<(outs GPR:$dst), (ins GPR:$sp, t2_so_imm:$imm),
                        IIC_iALUi, "add", ".w\t$dst, $sp, $imm", []> {
  let Inst{31-27} = 0b11110;
  let Inst{25} = 0;
  let Inst{24-21} = 0b1000;
  let Inst{20} = ?; // The S bit.
  let Inst{19-16} = 0b1101; // Rn = sp
  let Inst{15} = 0;
}
def t2ADDrSPi12 : T2I<(outs GPR:$dst), (ins GPR:$sp, imm0_4095:$imm),
                       IIC_iALUi, "addw", "\t$dst, $sp, $imm", []> {
  let Inst{31-27} = 0b11110;
  let Inst{25} = 1;
  let Inst{24-21} = 0b0000;
  let Inst{20} = 0; // The S bit.
  let Inst{19-16} = 0b1101; // Rn = sp
  let Inst{15} = 0;
}

// ADD r, sp, so_reg
def t2ADDrSPs   : T2sI<(outs GPR:$dst), (ins GPR:$sp, t2_so_reg:$rhs),
                        IIC_iALUsi, "add", ".w\t$dst, $sp, $rhs", []> {
  let Inst{31-27} = 0b11101;
  let Inst{26-25} = 0b01;
  let Inst{24-21} = 0b1000;
  let Inst{20} = ?; // The S bit.
  let Inst{19-16} = 0b1101; // Rn = sp
  let Inst{15} = 0;
}

// SUB r, sp, {so_imm|i12}
def t2SUBrSPi   : T2sI<(outs GPR:$dst), (ins GPR:$sp, t2_so_imm:$imm),
                        IIC_iALUi, "sub", ".w\t$dst, $sp, $imm", []> {
  let Inst{31-27} = 0b11110;
  let Inst{25} = 0;
  let Inst{24-21} = 0b1101;
  let Inst{20} = ?; // The S bit.
  let Inst{19-16} = 0b1101; // Rn = sp
  let Inst{15} = 0;
}
def t2SUBrSPi12 : T2I<(outs GPR:$dst), (ins GPR:$sp, imm0_4095:$imm),
                       IIC_iALUi, "subw", "\t$dst, $sp, $imm", []> {
  let Inst{31-27} = 0b11110;
  let Inst{25} = 1;
  let Inst{24-21} = 0b0101;
  let Inst{20} = 0; // The S bit.
  let Inst{19-16} = 0b1101; // Rn = sp
  let Inst{15} = 0;
}

// SUB r, sp, so_reg
def t2SUBrSPs   : T2sI<(outs GPR:$dst), (ins GPR:$sp, t2_so_reg:$rhs),
                       IIC_iALUsi,
                       "sub", "\t$dst, $sp, $rhs", []> {
  let Inst{31-27} = 0b11101;
  let Inst{26-25} = 0b01;
  let Inst{24-21} = 0b1101;
  let Inst{20} = ?; // The S bit.
  let Inst{19-16} = 0b1101; // Rn = sp
  let Inst{15} = 0;
}

// Signed and unsigned division on v7-M
def t2SDIV : T2I<(outs rGPR:$dst), (ins rGPR:$a, rGPR:$b), IIC_iALUi, 
                 "sdiv", "\t$dst, $a, $b",
                 [(set rGPR:$dst, (sdiv rGPR:$a, rGPR:$b))]>,
                 Requires<[HasDivide]> {
  let Inst{31-27} = 0b11111;
  let Inst{26-21} = 0b011100;
  let Inst{20} = 0b1;
  let Inst{15-12} = 0b1111;
  let Inst{7-4} = 0b1111;
}

def t2UDIV : T2I<(outs rGPR:$dst), (ins rGPR:$a, rGPR:$b), IIC_iALUi, 
                 "udiv", "\t$dst, $a, $b",
                 [(set rGPR:$dst, (udiv rGPR:$a, rGPR:$b))]>,
                 Requires<[HasDivide]> {
  let Inst{31-27} = 0b11111;
  let Inst{26-21} = 0b011101;
  let Inst{20} = 0b1;
  let Inst{15-12} = 0b1111;
  let Inst{7-4} = 0b1111;
}

//===----------------------------------------------------------------------===//
//  Load / store Instructions.
//

// Load
let canFoldAsLoad = 1, isReMaterializable = 1  in
defm t2LDR   : T2I_ld<0, 0b10, "ldr",  UnOpFrag<(load node:$Src)>>;

// Loads with zero extension
defm t2LDRH  : T2I_ld<0, 0b01, "ldrh", UnOpFrag<(zextloadi16 node:$Src)>>;
defm t2LDRB  : T2I_ld<0, 0b00, "ldrb", UnOpFrag<(zextloadi8  node:$Src)>>;

// Loads with sign extension
defm t2LDRSH : T2I_ld<1, 0b01, "ldrsh", UnOpFrag<(sextloadi16 node:$Src)>>;
defm t2LDRSB : T2I_ld<1, 0b00, "ldrsb", UnOpFrag<(sextloadi8  node:$Src)>>;

let mayLoad = 1, neverHasSideEffects = 1, hasExtraDefRegAllocReq = 1 in {
// Load doubleword
def t2LDRDi8  : T2Ii8s4<1, 0, 1, (outs rGPR:$dst1, rGPR:$dst2),
                        (ins t2addrmode_imm8s4:$addr),
                        IIC_iLoadi, "ldrd", "\t$dst1, $addr", []>;
def t2LDRDpci : T2Ii8s4<1, 0, 1, (outs rGPR:$dst1, rGPR:$dst2),
                        (ins i32imm:$addr), IIC_iLoadi,
                       "ldrd", "\t$dst1, $addr", []> {
  let Inst{19-16} = 0b1111; // Rn
}
} // mayLoad = 1, neverHasSideEffects = 1, hasExtraDefRegAllocReq = 1

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

// FIXME: The destination register of the loads and stores can't be PC, but
//        can be SP. We need another regclass (similar to rGPR) to represent
//        that. Not a pressing issue since these are selected manually,
//        not via pattern.

// Indexed loads
let mayLoad = 1, neverHasSideEffects = 1 in {
def t2LDR_PRE  : T2Iidxldst<0, 0b10, 1, 1, (outs GPR:$dst, GPR:$base_wb),
                            (ins t2addrmode_imm8:$addr),
                            AddrModeT2_i8, IndexModePre, IIC_iLoadiu,
                            "ldr", "\t$dst, $addr!", "$addr.base = $base_wb",
                            []>;

def t2LDR_POST : T2Iidxldst<0, 0b10, 1, 0, (outs GPR:$dst, GPR:$base_wb),
                            (ins GPR:$base, t2am_imm8_offset:$offset),
                            AddrModeT2_i8, IndexModePost, IIC_iLoadiu,
                          "ldr", "\t$dst, [$base], $offset", "$base = $base_wb",
                            []>;

def t2LDRB_PRE : T2Iidxldst<0, 0b00, 1, 1, (outs GPR:$dst, GPR:$base_wb),
                            (ins t2addrmode_imm8:$addr),
                            AddrModeT2_i8, IndexModePre, IIC_iLoadiu,
                            "ldrb", "\t$dst, $addr!", "$addr.base = $base_wb",
                            []>;
def t2LDRB_POST : T2Iidxldst<0, 0b00, 1, 0, (outs GPR:$dst, GPR:$base_wb),
                            (ins GPR:$base, t2am_imm8_offset:$offset),
                            AddrModeT2_i8, IndexModePost, IIC_iLoadiu,
                         "ldrb", "\t$dst, [$base], $offset", "$base = $base_wb",
                            []>;

def t2LDRH_PRE : T2Iidxldst<0, 0b01, 1, 1, (outs GPR:$dst, GPR:$base_wb),
                            (ins t2addrmode_imm8:$addr),
                            AddrModeT2_i8, IndexModePre, IIC_iLoadiu,
                            "ldrh", "\t$dst, $addr!", "$addr.base = $base_wb",
                            []>;
def t2LDRH_POST : T2Iidxldst<0, 0b01, 1, 0, (outs GPR:$dst, GPR:$base_wb),
                            (ins GPR:$base, t2am_imm8_offset:$offset),
                            AddrModeT2_i8, IndexModePost, IIC_iLoadiu,
                         "ldrh", "\t$dst, [$base], $offset", "$base = $base_wb",
                            []>;

def t2LDRSB_PRE : T2Iidxldst<1, 0b00, 1, 1, (outs GPR:$dst, GPR:$base_wb),
                            (ins t2addrmode_imm8:$addr),
                            AddrModeT2_i8, IndexModePre, IIC_iLoadiu,
                            "ldrsb", "\t$dst, $addr!", "$addr.base = $base_wb",
                            []>;
def t2LDRSB_POST : T2Iidxldst<1, 0b00, 1, 0, (outs GPR:$dst, GPR:$base_wb),
                            (ins GPR:$base, t2am_imm8_offset:$offset),
                            AddrModeT2_i8, IndexModePost, IIC_iLoadiu,
                        "ldrsb", "\t$dst, [$base], $offset", "$base = $base_wb",
                            []>;

def t2LDRSH_PRE : T2Iidxldst<1, 0b01, 1, 1, (outs GPR:$dst, GPR:$base_wb),
                            (ins t2addrmode_imm8:$addr),
                            AddrModeT2_i8, IndexModePre, IIC_iLoadiu,
                            "ldrsh", "\t$dst, $addr!", "$addr.base = $base_wb",
                            []>;
def t2LDRSH_POST : T2Iidxldst<1, 0b01, 1, 0, (outs GPR:$dst, GPR:$base_wb),
                            (ins GPR:$base, t2am_imm8_offset:$offset),
                            AddrModeT2_i8, IndexModePost, IIC_iLoadiu,
                        "ldrsh", "\t$dst, [$base], $offset", "$base = $base_wb",
                            []>;
} // mayLoad = 1, neverHasSideEffects = 1 

// LDRT, LDRBT, LDRHT, LDRSBT, LDRSHT all have offset mode (PUW=0b110) and are
// for disassembly only.
// Ref: A8.6.57 LDR (immediate, Thumb) Encoding T4
class T2IldT<bit signed, bits<2> type, string opc>
  : T2Ii8<(outs GPR:$dst), (ins t2addrmode_imm8:$addr), IIC_iLoadi, opc,
          "\t$dst, $addr", []> {
  let Inst{31-27} = 0b11111;
  let Inst{26-25} = 0b00;
  let Inst{24} = signed;
  let Inst{23} = 0;
  let Inst{22-21} = type;
  let Inst{20} = 1; // load
  let Inst{11} = 1;
  let Inst{10-8} = 0b110; // PUW.
}

def t2LDRT   : T2IldT<0, 0b10, "ldrt">;
def t2LDRBT  : T2IldT<0, 0b00, "ldrbt">;
def t2LDRHT  : T2IldT<0, 0b01, "ldrht">;
def t2LDRSBT : T2IldT<1, 0b00, "ldrsbt">;
def t2LDRSHT : T2IldT<1, 0b01, "ldrsht">;

// Store
defm t2STR :T2I_st<0b10,"str", BinOpFrag<(store node:$LHS, node:$RHS)>>;
defm t2STRB:T2I_st<0b00,"strb",BinOpFrag<(truncstorei8 node:$LHS, node:$RHS)>>;
defm t2STRH:T2I_st<0b01,"strh",BinOpFrag<(truncstorei16 node:$LHS, node:$RHS)>>;

// Store doubleword
let mayLoad = 1, neverHasSideEffects = 1, hasExtraSrcRegAllocReq = 1 in
def t2STRDi8 : T2Ii8s4<1, 0, 0, (outs),
                       (ins GPR:$src1, GPR:$src2, t2addrmode_imm8s4:$addr),
               IIC_iStorer, "strd", "\t$src1, $addr", []>;

// Indexed stores
def t2STR_PRE  : T2Iidxldst<0, 0b10, 0, 1, (outs GPR:$base_wb),
                            (ins GPR:$src, GPR:$base, t2am_imm8_offset:$offset),
                            AddrModeT2_i8, IndexModePre, IIC_iStoreiu,
                         "str", "\t$src, [$base, $offset]!", "$base = $base_wb",
             [(set GPR:$base_wb,
                   (pre_store GPR:$src, GPR:$base, t2am_imm8_offset:$offset))]>;

def t2STR_POST : T2Iidxldst<0, 0b10, 0, 0, (outs GPR:$base_wb),
                            (ins GPR:$src, GPR:$base, t2am_imm8_offset:$offset),
                            AddrModeT2_i8, IndexModePost, IIC_iStoreiu,
                          "str", "\t$src, [$base], $offset", "$base = $base_wb",
             [(set GPR:$base_wb,
                  (post_store GPR:$src, GPR:$base, t2am_imm8_offset:$offset))]>;

def t2STRH_PRE  : T2Iidxldst<0, 0b01, 0, 1, (outs GPR:$base_wb),
                            (ins GPR:$src, GPR:$base, t2am_imm8_offset:$offset),
                            AddrModeT2_i8, IndexModePre, IIC_iStoreiu,
                        "strh", "\t$src, [$base, $offset]!", "$base = $base_wb",
        [(set GPR:$base_wb,
              (pre_truncsti16 GPR:$src, GPR:$base, t2am_imm8_offset:$offset))]>;

def t2STRH_POST : T2Iidxldst<0, 0b01, 0, 0, (outs GPR:$base_wb),
                            (ins GPR:$src, GPR:$base, t2am_imm8_offset:$offset),
                            AddrModeT2_i8, IndexModePost, IIC_iStoreiu,
                         "strh", "\t$src, [$base], $offset", "$base = $base_wb",
       [(set GPR:$base_wb,
             (post_truncsti16 GPR:$src, GPR:$base, t2am_imm8_offset:$offset))]>;

def t2STRB_PRE  : T2Iidxldst<0, 0b00, 0, 1, (outs GPR:$base_wb),
                            (ins GPR:$src, GPR:$base, t2am_imm8_offset:$offset),
                            AddrModeT2_i8, IndexModePre, IIC_iStoreiu,
                        "strb", "\t$src, [$base, $offset]!", "$base = $base_wb",
         [(set GPR:$base_wb,
               (pre_truncsti8 GPR:$src, GPR:$base, t2am_imm8_offset:$offset))]>;

def t2STRB_POST : T2Iidxldst<0, 0b00, 0, 0, (outs GPR:$base_wb),
                            (ins GPR:$src, GPR:$base, t2am_imm8_offset:$offset),
                            AddrModeT2_i8, IndexModePost, IIC_iStoreiu,
                         "strb", "\t$src, [$base], $offset", "$base = $base_wb",
        [(set GPR:$base_wb,
              (post_truncsti8 GPR:$src, GPR:$base, t2am_imm8_offset:$offset))]>;

// STRT, STRBT, STRHT all have offset mode (PUW=0b110) and are for disassembly
// only.
// Ref: A8.6.193 STR (immediate, Thumb) Encoding T4
class T2IstT<bits<2> type, string opc>
  : T2Ii8<(outs GPR:$src), (ins t2addrmode_imm8:$addr), IIC_iStorei, opc,
          "\t$src, $addr", []> {
  let Inst{31-27} = 0b11111;
  let Inst{26-25} = 0b00;
  let Inst{24} = 0; // not signed
  let Inst{23} = 0;
  let Inst{22-21} = type;
  let Inst{20} = 0; // store
  let Inst{11} = 1;
  let Inst{10-8} = 0b110; // PUW
}

def t2STRT   : T2IstT<0b10, "strt">;
def t2STRBT  : T2IstT<0b00, "strbt">;
def t2STRHT  : T2IstT<0b01, "strht">;

// ldrd / strd pre / post variants
// For disassembly only.

def t2LDRD_PRE  : T2Ii8s4<1, 1, 1, (outs GPR:$dst1, GPR:$dst2),
                 (ins GPR:$base, t2am_imm8s4_offset:$imm), NoItinerary,
                 "ldrd", "\t$dst1, $dst2, [$base, $imm]!", []>;

def t2LDRD_POST : T2Ii8s4<0, 1, 1, (outs GPR:$dst1, GPR:$dst2),
                 (ins GPR:$base, t2am_imm8s4_offset:$imm), NoItinerary,
                 "ldrd", "\t$dst1, $dst2, [$base], $imm", []>;

def t2STRD_PRE  : T2Ii8s4<1, 1, 0, (outs),
                 (ins GPR:$src1, GPR:$src2, GPR:$base, t2am_imm8s4_offset:$imm),
                 NoItinerary, "strd", "\t$src1, $src2, [$base, $imm]!", []>;

def t2STRD_POST : T2Ii8s4<0, 1, 0, (outs),
                 (ins GPR:$src1, GPR:$src2, GPR:$base, t2am_imm8s4_offset:$imm),
                 NoItinerary, "strd", "\t$src1, $src2, [$base], $imm", []>;

// T2Ipl (Preload Data/Instruction) signals the memory system of possible future
// data/instruction access.  These are for disassembly only.
//
// A8.6.117, A8.6.118.  Different instructions are generated for #0 and #-0.
// The neg_zero operand translates -0 to -1, -1 to -2, ..., etc.
multiclass T2Ipl<bit instr, bit write, string opc> {

  def i12 : T2I<(outs), (ins GPR:$base, i32imm:$imm), IIC_iLoadi, opc,
                "\t[$base, $imm]", []> {
    let Inst{31-25} = 0b1111100;
    let Inst{24} = instr;
    let Inst{23} = 1; // U = 1
    let Inst{22} = 0;
    let Inst{21} = write;
    let Inst{20} = 1;
    let Inst{15-12} = 0b1111;
  }

  def i8 : T2I<(outs), (ins GPR:$base, neg_zero:$imm), IIC_iLoadi, opc,
                "\t[$base, $imm]", []> {
    let Inst{31-25} = 0b1111100;
    let Inst{24} = instr;
    let Inst{23} = 0; // U = 0
    let Inst{22} = 0;
    let Inst{21} = write;
    let Inst{20} = 1;
    let Inst{15-12} = 0b1111;
    let Inst{11-8} = 0b1100;
  }

  def pci : T2I<(outs), (ins GPR:$base, neg_zero:$imm), IIC_iLoadi, opc,
                "\t[pc, $imm]", []> {
    let Inst{31-25} = 0b1111100;
    let Inst{24} = instr;
    let Inst{23} = ?; // add = (U == 1)
    let Inst{22} = 0;
    let Inst{21} = write;
    let Inst{20} = 1;
    let Inst{19-16} = 0b1111; // Rn = 0b1111
    let Inst{15-12} = 0b1111;
  }

  def r   : T2I<(outs), (ins GPR:$base, GPR:$a), IIC_iLoadi, opc,
                "\t[$base, $a]", []> {
    let Inst{31-25} = 0b1111100;
    let Inst{24} = instr;
    let Inst{23} = 0; // add = TRUE for T1
    let Inst{22} = 0;
    let Inst{21} = write;
    let Inst{20} = 1;
    let Inst{15-12} = 0b1111;
    let Inst{11-6} = 0000000;
    let Inst{5-4} = 0b00; // no shift is applied
  }

  def s   : T2I<(outs), (ins GPR:$base, GPR:$a, i32imm:$shamt), IIC_iLoadi, opc,
                "\t[$base, $a, lsl $shamt]", []> {
    let Inst{31-25} = 0b1111100;
    let Inst{24} = instr;
    let Inst{23} = 0; // add = TRUE for T1
    let Inst{22} = 0;
    let Inst{21} = write;
    let Inst{20} = 1;
    let Inst{15-12} = 0b1111;
    let Inst{11-6} = 0000000;
  }
}

defm t2PLD  : T2Ipl<0, 0, "pld">;
defm t2PLDW : T2Ipl<0, 1, "pldw">;
defm t2PLI  : T2Ipl<1, 0, "pli">;

//===----------------------------------------------------------------------===//
//  Load / store multiple Instructions.
//

let mayLoad = 1, neverHasSideEffects = 1, hasExtraDefRegAllocReq = 1 in {
def t2LDM : T2XI<(outs), (ins addrmode4:$addr, pred:$p,
                          reglist:$dsts, variable_ops), IIC_iLoadm,
                 "ldm${addr:submode}${p}${addr:wide}\t$addr, $dsts", []> {
  let Inst{31-27} = 0b11101;
  let Inst{26-25} = 0b00;
  let Inst{24-23} = {?, ?}; // IA: '01', DB: '10'
  let Inst{22} = 0;
  let Inst{21} = 0; // The W bit.
  let Inst{20} = 1; // Load
}

def t2LDM_UPD : T2XIt<(outs GPR:$wb), (ins addrmode4:$addr, pred:$p,
                                       reglist:$dsts, variable_ops), IIC_iLoadm,
                      "ldm${addr:submode}${p}${addr:wide}\t$addr!, $dsts",
                      "$addr.addr = $wb", []> {
  let Inst{31-27} = 0b11101;
  let Inst{26-25} = 0b00;
  let Inst{24-23} = {?, ?}; // IA: '01', DB: '10'
  let Inst{22} = 0;
  let Inst{21} = 1; // The W bit.
  let Inst{20} = 1; // Load
}
} // mayLoad, neverHasSideEffects, hasExtraDefRegAllocReq

let mayStore = 1, neverHasSideEffects = 1, hasExtraSrcRegAllocReq = 1 in {
def t2STM : T2XI<(outs), (ins addrmode4:$addr, pred:$p,
                          reglist:$srcs, variable_ops), IIC_iStorem,
                 "stm${addr:submode}${p}${addr:wide}\t$addr, $srcs", []> {
  let Inst{31-27} = 0b11101;
  let Inst{26-25} = 0b00;
  let Inst{24-23} = {?, ?}; // IA: '01', DB: '10'
  let Inst{22} = 0;
  let Inst{21} = 0; // The W bit.
  let Inst{20} = 0; // Store
}

def t2STM_UPD : T2XIt<(outs GPR:$wb), (ins addrmode4:$addr, pred:$p,
                                       reglist:$srcs, variable_ops),
                      IIC_iStorem,
                      "stm${addr:submode}${p}${addr:wide}\t$addr!, $srcs",
                      "$addr.addr = $wb", []> {
  let Inst{31-27} = 0b11101;
  let Inst{26-25} = 0b00;
  let Inst{24-23} = {?, ?}; // IA: '01', DB: '10'
  let Inst{22} = 0;
  let Inst{21} = 1; // The W bit.
  let Inst{20} = 0; // Store
}
} // mayStore, neverHasSideEffects, hasExtraSrcRegAllocReq

//===----------------------------------------------------------------------===//
//  Move Instructions.
//

let neverHasSideEffects = 1 in
def t2MOVr : T2sI<(outs GPR:$dst), (ins GPR:$src), IIC_iMOVr,
                   "mov", ".w\t$dst, $src", []> {
  let Inst{31-27} = 0b11101;
  let Inst{26-25} = 0b01;
  let Inst{24-21} = 0b0010;
  let Inst{20} = ?; // The S bit.
  let Inst{19-16} = 0b1111; // Rn
  let Inst{14-12} = 0b000;
  let Inst{7-4} = 0b0000;
}

// AddedComplexity to ensure isel tries t2MOVi before t2MOVi16.
let isReMaterializable = 1, isAsCheapAsAMove = 1, AddedComplexity = 1 in
def t2MOVi : T2sI<(outs rGPR:$dst), (ins t2_so_imm:$src), IIC_iMOVi,
                   "mov", ".w\t$dst, $src",
                   [(set rGPR:$dst, t2_so_imm:$src)]> {
  let Inst{31-27} = 0b11110;
  let Inst{25} = 0;
  let Inst{24-21} = 0b0010;
  let Inst{20} = ?; // The S bit.
  let Inst{19-16} = 0b1111; // Rn
  let Inst{15} = 0;
}

let isReMaterializable = 1, isAsCheapAsAMove = 1 in
def t2MOVi16 : T2I<(outs rGPR:$dst), (ins i32imm:$src), IIC_iMOVi,
                   "movw", "\t$dst, $src",
                   [(set rGPR:$dst, imm0_65535:$src)]> {
  let Inst{31-27} = 0b11110;
  let Inst{25} = 1;
  let Inst{24-21} = 0b0010;
  let Inst{20} = 0; // The S bit.
  let Inst{15} = 0;
}

let Constraints = "$src = $dst" in
def t2MOVTi16 : T2I<(outs rGPR:$dst), (ins rGPR:$src, i32imm:$imm), IIC_iMOVi,
                    "movt", "\t$dst, $imm",
                    [(set rGPR:$dst,
                          (or (and rGPR:$src, 0xffff), lo16AllZero:$imm))]> {
  let Inst{31-27} = 0b11110;
  let Inst{25} = 1;
  let Inst{24-21} = 0b0110;
  let Inst{20} = 0; // The S bit.
  let Inst{15} = 0;
}

def : T2Pat<(or rGPR:$src, 0xffff0000), (t2MOVTi16 rGPR:$src, 0xffff)>;

//===----------------------------------------------------------------------===//
//  Extend Instructions.
//

// Sign extenders

defm t2SXTB  : T2I_unary_rrot<0b100, "sxtb",
                              UnOpFrag<(sext_inreg node:$Src, i8)>>;
defm t2SXTH  : T2I_unary_rrot<0b000, "sxth",
                              UnOpFrag<(sext_inreg node:$Src, i16)>>;
defm t2SXTB16 : T2I_unary_rrot_sxtb16<0b010, "sxtb16">;

defm t2SXTAB : T2I_bin_rrot<0b100, "sxtab",
                        BinOpFrag<(add node:$LHS, (sext_inreg node:$RHS, i8))>>;
defm t2SXTAH : T2I_bin_rrot<0b000, "sxtah",
                        BinOpFrag<(add node:$LHS, (sext_inreg node:$RHS,i16))>>;
defm t2SXTAB16 : T2I_bin_rrot_DO<0b010, "sxtab16">;

// TODO: SXT(A){B|H}16 - done for disassembly only

// Zero extenders

let AddedComplexity = 16 in {
defm t2UXTB   : T2I_unary_rrot<0b101, "uxtb",
                               UnOpFrag<(and node:$Src, 0x000000FF)>>;
defm t2UXTH   : T2I_unary_rrot<0b001, "uxth",
                               UnOpFrag<(and node:$Src, 0x0000FFFF)>>;
defm t2UXTB16 : T2I_unary_rrot_uxtb16<0b011, "uxtb16",
                               UnOpFrag<(and node:$Src, 0x00FF00FF)>>;

// FIXME: This pattern incorrectly assumes the shl operator is a rotate.
//        The transformation should probably be done as a combiner action
//        instead so we can include a check for masking back in the upper
//        eight bits of the source into the lower eight bits of the result.
//def : T2Pat<(and (shl rGPR:$Src, (i32 8)), 0xFF00FF),
//            (t2UXTB16r_rot rGPR:$Src, 24)>, Requires<[HasT2ExtractPack]>;
def : T2Pat<(and (srl rGPR:$Src, (i32 8)), 0xFF00FF),
            (t2UXTB16r_rot rGPR:$Src, 8)>, Requires<[HasT2ExtractPack]>;

defm t2UXTAB : T2I_bin_rrot<0b101, "uxtab",
                           BinOpFrag<(add node:$LHS, (and node:$RHS, 0x00FF))>>;
defm t2UXTAH : T2I_bin_rrot<0b001, "uxtah",
                           BinOpFrag<(add node:$LHS, (and node:$RHS, 0xFFFF))>>;
defm t2UXTAB16 : T2I_bin_rrot_DO<0b011, "uxtab16">;
}

//===----------------------------------------------------------------------===//
//  Arithmetic Instructions.
//

defm t2ADD  : T2I_bin_ii12rs<0b000, "add",
                             BinOpFrag<(add  node:$LHS, node:$RHS)>, 1>;
defm t2SUB  : T2I_bin_ii12rs<0b101, "sub",
                             BinOpFrag<(sub  node:$LHS, node:$RHS)>>;

// ADD and SUB with 's' bit set. No 12-bit immediate (T4) variants.
defm t2ADDS : T2I_bin_s_irs <0b1000, "add",
                             BinOpFrag<(addc node:$LHS, node:$RHS)>, 1>;
defm t2SUBS : T2I_bin_s_irs <0b1101, "sub",
                             BinOpFrag<(subc node:$LHS, node:$RHS)>>;

defm t2ADC  : T2I_adde_sube_irs<0b1010, "adc",
                          BinOpFrag<(adde_dead_carry node:$LHS, node:$RHS)>, 1>;
defm t2SBC  : T2I_adde_sube_irs<0b1011, "sbc",
                          BinOpFrag<(sube_dead_carry node:$LHS, node:$RHS)>>;
defm t2ADCS : T2I_adde_sube_s_irs<0b1010, "adc",
                          BinOpFrag<(adde_live_carry node:$LHS, node:$RHS)>, 1>;
defm t2SBCS : T2I_adde_sube_s_irs<0b1011, "sbc",
                          BinOpFrag<(sube_live_carry node:$LHS, node:$RHS)>>;

// RSB
defm t2RSB  : T2I_rbin_irs  <0b1110, "rsb",
                             BinOpFrag<(sub  node:$LHS, node:$RHS)>>;
defm t2RSBS : T2I_rbin_s_is <0b1110, "rsb",
                             BinOpFrag<(subc node:$LHS, node:$RHS)>>;

// (sub X, imm) gets canonicalized to (add X, -imm).  Match this form.
// The assume-no-carry-in form uses the negation of the input since add/sub
// assume opposite meanings of the carry flag (i.e., carry == !borrow).
// See the definition of AddWithCarry() in the ARM ARM A2.2.1 for the gory
// details.
// The AddedComplexity preferences the first variant over the others since
// it can be shrunk to a 16-bit wide encoding, while the others cannot.
let AddedComplexity = 1 in
def : T2Pat<(add        GPR:$src, imm0_255_neg:$imm),
            (t2SUBri    GPR:$src, imm0_255_neg:$imm)>;
def : T2Pat<(add        GPR:$src, t2_so_imm_neg:$imm),
            (t2SUBri    GPR:$src, t2_so_imm_neg:$imm)>;
def : T2Pat<(add        GPR:$src, imm0_4095_neg:$imm),
            (t2SUBri12  GPR:$src, imm0_4095_neg:$imm)>;
let AddedComplexity = 1 in
def : T2Pat<(addc       rGPR:$src, imm0_255_neg:$imm),
            (t2SUBSri   rGPR:$src, imm0_255_neg:$imm)>;
def : T2Pat<(addc       rGPR:$src, t2_so_imm_neg:$imm),
            (t2SUBSri   rGPR:$src, t2_so_imm_neg:$imm)>;
// The with-carry-in form matches bitwise not instead of the negation.
// Effectively, the inverse interpretation of the carry flag already accounts
// for part of the negation.
let AddedComplexity = 1 in
def : T2Pat<(adde       rGPR:$src, imm0_255_not:$imm),
            (t2SBCSri   rGPR:$src, imm0_255_not:$imm)>;
def : T2Pat<(adde       rGPR:$src, t2_so_imm_not:$imm),
            (t2SBCSri   rGPR:$src, t2_so_imm_not:$imm)>;

// Select Bytes -- for disassembly only

def t2SEL : T2I<(outs GPR:$dst), (ins GPR:$a, GPR:$b), NoItinerary, "sel",
                "\t$dst, $a, $b", []> {
  let Inst{31-27} = 0b11111;
  let Inst{26-24} = 0b010;
  let Inst{23} = 0b1;
  let Inst{22-20} = 0b010;
  let Inst{15-12} = 0b1111;
  let Inst{7} = 0b1;
  let Inst{6-4} = 0b000;
}

// A6.3.13, A6.3.14, A6.3.15 Parallel addition and subtraction (signed/unsigned)
// And Miscellaneous operations -- for disassembly only
class T2I_pam<bits<3> op22_20, bits<4> op7_4, string opc,
              list<dag> pat = [/* For disassembly only; pattern left blank */]>
  : T2I<(outs rGPR:$dst), (ins rGPR:$a, rGPR:$b), NoItinerary, opc,
        "\t$dst, $a, $b", pat> {
  let Inst{31-27} = 0b11111;
  let Inst{26-23} = 0b0101;
  let Inst{22-20} = op22_20;
  let Inst{15-12} = 0b1111;
  let Inst{7-4} = op7_4;
}

// Saturating add/subtract -- for disassembly only

def t2QADD    : T2I_pam<0b000, 0b1000, "qadd",
                        [(set rGPR:$dst, (int_arm_qadd rGPR:$a, rGPR:$b))]>;
def t2QADD16  : T2I_pam<0b001, 0b0001, "qadd16">;
def t2QADD8   : T2I_pam<0b000, 0b0001, "qadd8">;
def t2QASX    : T2I_pam<0b010, 0b0001, "qasx">;
def t2QDADD   : T2I_pam<0b000, 0b1001, "qdadd">;
def t2QDSUB   : T2I_pam<0b000, 0b1011, "qdsub">;
def t2QSAX    : T2I_pam<0b110, 0b0001, "qsax">;
def t2QSUB    : T2I_pam<0b000, 0b1010, "qsub",
                        [(set rGPR:$dst, (int_arm_qsub rGPR:$a, rGPR:$b))]>;
def t2QSUB16  : T2I_pam<0b101, 0b0001, "qsub16">;
def t2QSUB8   : T2I_pam<0b100, 0b0001, "qsub8">;
def t2UQADD16 : T2I_pam<0b001, 0b0101, "uqadd16">;
def t2UQADD8  : T2I_pam<0b000, 0b0101, "uqadd8">;
def t2UQASX   : T2I_pam<0b010, 0b0101, "uqasx">;
def t2UQSAX   : T2I_pam<0b110, 0b0101, "uqsax">;
def t2UQSUB16 : T2I_pam<0b101, 0b0101, "uqsub16">;
def t2UQSUB8  : T2I_pam<0b100, 0b0101, "uqsub8">;

// Signed/Unsigned add/subtract -- for disassembly only

def t2SASX    : T2I_pam<0b010, 0b0000, "sasx">;
def t2SADD16  : T2I_pam<0b001, 0b0000, "sadd16">;
def t2SADD8   : T2I_pam<0b000, 0b0000, "sadd8">;
def t2SSAX    : T2I_pam<0b110, 0b0000, "ssax">;
def t2SSUB16  : T2I_pam<0b101, 0b0000, "ssub16">;
def t2SSUB8   : T2I_pam<0b100, 0b0000, "ssub8">;
def t2UASX    : T2I_pam<0b010, 0b0100, "uasx">;
def t2UADD16  : T2I_pam<0b001, 0b0100, "uadd16">;
def t2UADD8   : T2I_pam<0b000, 0b0100, "uadd8">;
def t2USAX    : T2I_pam<0b110, 0b0100, "usax">;
def t2USUB16  : T2I_pam<0b101, 0b0100, "usub16">;
def t2USUB8   : T2I_pam<0b100, 0b0100, "usub8">;

// Signed/Unsigned halving add/subtract -- for disassembly only

def t2SHASX   : T2I_pam<0b010, 0b0010, "shasx">;
def t2SHADD16 : T2I_pam<0b001, 0b0010, "shadd16">;
def t2SHADD8  : T2I_pam<0b000, 0b0010, "shadd8">;
def t2SHSAX   : T2I_pam<0b110, 0b0010, "shsax">;
def t2SHSUB16 : T2I_pam<0b101, 0b0010, "shsub16">;
def t2SHSUB8  : T2I_pam<0b100, 0b0010, "shsub8">;
def t2UHASX   : T2I_pam<0b010, 0b0110, "uhasx">;
def t2UHADD16 : T2I_pam<0b001, 0b0110, "uhadd16">;
def t2UHADD8  : T2I_pam<0b000, 0b0110, "uhadd8">;
def t2UHSAX   : T2I_pam<0b110, 0b0110, "uhsax">;
def t2UHSUB16 : T2I_pam<0b101, 0b0110, "uhsub16">;
def t2UHSUB8  : T2I_pam<0b100, 0b0110, "uhsub8">;

// Unsigned Sum of Absolute Differences [and Accumulate] -- for disassembly only

def t2USAD8   : T2I_mac<0, 0b111, 0b0000, (outs rGPR:$dst),
                                           (ins rGPR:$a, rGPR:$b),
                        NoItinerary, "usad8", "\t$dst, $a, $b", []> {
  let Inst{15-12} = 0b1111;
}
def t2USADA8  : T2I_mac<0, 0b111, 0b0000, (outs rGPR:$dst),
                       (ins rGPR:$a, rGPR:$b, rGPR:$acc), NoItinerary, "usada8",
                        "\t$dst, $a, $b, $acc", []>;

// Signed/Unsigned saturate -- for disassembly only

def t2SSAT: T2I<(outs rGPR:$dst), (ins i32imm:$bit_pos, rGPR:$a, shift_imm:$sh),
                NoItinerary, "ssat", "\t$dst, $bit_pos, $a$sh",
                [/* For disassembly only; pattern left blank */]> {
  let Inst{31-27} = 0b11110;
  let Inst{25-22} = 0b1100;
  let Inst{20} = 0;
  let Inst{15} = 0;
}

def t2SSAT16: T2I<(outs rGPR:$dst), (ins i32imm:$bit_pos, rGPR:$a), NoItinerary,
                   "ssat16", "\t$dst, $bit_pos, $a",
                   [/* For disassembly only; pattern left blank */]> {
  let Inst{31-27} = 0b11110;
  let Inst{25-22} = 0b1100;
  let Inst{20} = 0;
  let Inst{15} = 0;
  let Inst{21} = 1;        // sh = '1'
  let Inst{14-12} = 0b000; // imm3 = '000'
  let Inst{7-6} = 0b00;    // imm2 = '00'
}

def t2USAT: T2I<(outs rGPR:$dst), (ins i32imm:$bit_pos, rGPR:$a, shift_imm:$sh),
                NoItinerary, "usat", "\t$dst, $bit_pos, $a$sh",
                [/* For disassembly only; pattern left blank */]> {
  let Inst{31-27} = 0b11110;
  let Inst{25-22} = 0b1110;
  let Inst{20} = 0;
  let Inst{15} = 0;
}

def t2USAT16: T2I<(outs rGPR:$dst), (ins i32imm:$bit_pos, rGPR:$a), NoItinerary,
                   "usat16", "\t$dst, $bit_pos, $a",
                   [/* For disassembly only; pattern left blank */]> {
  let Inst{31-27} = 0b11110;
  let Inst{25-22} = 0b1110;
  let Inst{20} = 0;
  let Inst{15} = 0;
  let Inst{21} = 1;        // sh = '1'
  let Inst{14-12} = 0b000; // imm3 = '000'
  let Inst{7-6} = 0b00;    // imm2 = '00'
}

def : T2Pat<(int_arm_ssat GPR:$a, imm:$pos), (t2SSAT imm:$pos, GPR:$a, 0)>;
def : T2Pat<(int_arm_usat GPR:$a, imm:$pos), (t2USAT imm:$pos, GPR:$a, 0)>;

//===----------------------------------------------------------------------===//
//  Shift and rotate Instructions.
//

defm t2LSL  : T2I_sh_ir<0b00, "lsl", BinOpFrag<(shl  node:$LHS, node:$RHS)>>;
defm t2LSR  : T2I_sh_ir<0b01, "lsr", BinOpFrag<(srl  node:$LHS, node:$RHS)>>;
defm t2ASR  : T2I_sh_ir<0b10, "asr", BinOpFrag<(sra  node:$LHS, node:$RHS)>>;
defm t2ROR  : T2I_sh_ir<0b11, "ror", BinOpFrag<(rotr node:$LHS, node:$RHS)>>;

let Uses = [CPSR] in {
def t2MOVrx : T2sI<(outs rGPR:$dst), (ins rGPR:$src), IIC_iMOVsi,
                   "rrx", "\t$dst, $src",
                   [(set rGPR:$dst, (ARMrrx rGPR:$src))]> {
  let Inst{31-27} = 0b11101;
  let Inst{26-25} = 0b01;
  let Inst{24-21} = 0b0010;
  let Inst{20} = ?; // The S bit.
  let Inst{19-16} = 0b1111; // Rn
  let Inst{14-12} = 0b000;
  let Inst{7-4} = 0b0011;
}
}

let Defs = [CPSR] in {
def t2MOVsrl_flag : T2I<(outs rGPR:$dst), (ins rGPR:$src), IIC_iMOVsi,
                        "lsrs", ".w\t$dst, $src, #1",
                        [(set rGPR:$dst, (ARMsrl_flag rGPR:$src))]> {
  let Inst{31-27} = 0b11101;
  let Inst{26-25} = 0b01;
  let Inst{24-21} = 0b0010;
  let Inst{20} = 1; // The S bit.
  let Inst{19-16} = 0b1111; // Rn
  let Inst{5-4} = 0b01; // Shift type.
  // Shift amount = Inst{14-12:7-6} = 1.
  let Inst{14-12} = 0b000;
  let Inst{7-6} = 0b01;
}
def t2MOVsra_flag : T2I<(outs rGPR:$dst), (ins rGPR:$src), IIC_iMOVsi,
                        "asrs", ".w\t$dst, $src, #1",
                        [(set rGPR:$dst, (ARMsra_flag rGPR:$src))]> {
  let Inst{31-27} = 0b11101;
  let Inst{26-25} = 0b01;
  let Inst{24-21} = 0b0010;
  let Inst{20} = 1; // The S bit.
  let Inst{19-16} = 0b1111; // Rn
  let Inst{5-4} = 0b10; // Shift type.
  // Shift amount = Inst{14-12:7-6} = 1.
  let Inst{14-12} = 0b000;
  let Inst{7-6} = 0b01;
}
}

//===----------------------------------------------------------------------===//
//  Bitwise Instructions.
//

defm t2AND  : T2I_bin_w_irs<0b0000, "and",
                            BinOpFrag<(and node:$LHS, node:$RHS)>, 1>;
defm t2ORR  : T2I_bin_w_irs<0b0010, "orr",
                            BinOpFrag<(or  node:$LHS, node:$RHS)>, 1>;
defm t2EOR  : T2I_bin_w_irs<0b0100, "eor",
                            BinOpFrag<(xor node:$LHS, node:$RHS)>, 1>;

defm t2BIC  : T2I_bin_w_irs<0b0001, "bic",
                            BinOpFrag<(and node:$LHS, (not node:$RHS))>>;

//let Defs = [CPSR] in
defm t2ANDS : T2I_bin_sw_irs<0b0000, "and",
                             BinOpFrag<(ARMand node:$LHS, node:$RHS)>, 1>;

let Constraints = "$src = $dst" in
def t2BFC : T2I<(outs rGPR:$dst), (ins rGPR:$src, bf_inv_mask_imm:$imm),
                IIC_iUNAsi, "bfc", "\t$dst, $imm",
                [(set rGPR:$dst, (and rGPR:$src, bf_inv_mask_imm:$imm))]> {
  let Inst{31-27} = 0b11110;
  let Inst{25} = 1;
  let Inst{24-20} = 0b10110;
  let Inst{19-16} = 0b1111; // Rn
  let Inst{15} = 0;
}

def t2SBFX: T2I<(outs rGPR:$dst), (ins rGPR:$src, imm0_31:$lsb, imm0_31:$width),
                 IIC_iALUi, "sbfx", "\t$dst, $src, $lsb, $width", []> {
  let Inst{31-27} = 0b11110;
  let Inst{25} = 1;
  let Inst{24-20} = 0b10100;
  let Inst{15} = 0;
}

def t2UBFX: T2I<(outs rGPR:$dst), (ins rGPR:$src, imm0_31:$lsb, imm0_31:$width),
                 IIC_iALUi, "ubfx", "\t$dst, $src, $lsb, $width", []> {
  let Inst{31-27} = 0b11110;
  let Inst{25} = 1;
  let Inst{24-20} = 0b11100;
  let Inst{15} = 0;
}

// A8.6.18  BFI - Bitfield insert (Encoding T1)
let Constraints = "$src = $dst" in
def t2BFI : T2I<(outs rGPR:$dst),
                (ins rGPR:$src, rGPR:$val, bf_inv_mask_imm:$imm),
                IIC_iALUi, "bfi", "\t$dst, $val, $imm",
                [(set rGPR:$dst, (ARMbfi rGPR:$src, rGPR:$val,
                                 bf_inv_mask_imm:$imm))]> {
  let Inst{31-27} = 0b11110;
  let Inst{25} = 1;
  let Inst{24-20} = 0b10110;
  let Inst{15} = 0;
}

defm t2ORN  : T2I_bin_irs<0b0011, "orn", BinOpFrag<(or  node:$LHS,
                          (not node:$RHS))>, 0, "">;

// Prefer over of t2EORri ra, rb, -1 because mvn has 16-bit version
let AddedComplexity = 1 in
defm t2MVN  : T2I_un_irs <0b0011, "mvn", UnOpFrag<(not node:$Src)>, 1, 1>;


let AddedComplexity = 1 in
def : T2Pat<(and     rGPR:$src, t2_so_imm_not:$imm),
            (t2BICri rGPR:$src, t2_so_imm_not:$imm)>;

// FIXME: Disable this pattern on Darwin to workaround an assembler bug.
def : T2Pat<(or      rGPR:$src, t2_so_imm_not:$imm),
            (t2ORNri rGPR:$src, t2_so_imm_not:$imm)>,
            Requires<[IsThumb2]>;

def : T2Pat<(t2_so_imm_not:$src),
            (t2MVNi t2_so_imm_not:$src)>;

//===----------------------------------------------------------------------===//
//  Multiply Instructions.
//
let isCommutable = 1 in
def t2MUL: T2I<(outs rGPR:$dst), (ins rGPR:$a, rGPR:$b), IIC_iMUL32,
                "mul", "\t$dst, $a, $b",
                [(set rGPR:$dst, (mul rGPR:$a, rGPR:$b))]> {
  let Inst{31-27} = 0b11111;
  let Inst{26-23} = 0b0110;
  let Inst{22-20} = 0b000;
  let Inst{15-12} = 0b1111; // Ra = 0b1111 (no accumulate)
  let Inst{7-4} = 0b0000; // Multiply
}

def t2MLA: T2I<(outs rGPR:$dst), (ins rGPR:$a, rGPR:$b, rGPR:$c), IIC_iMAC32,
		"mla", "\t$dst, $a, $b, $c",
		[(set rGPR:$dst, (add (mul rGPR:$a, rGPR:$b), rGPR:$c))]> {
  let Inst{31-27} = 0b11111;
  let Inst{26-23} = 0b0110;
  let Inst{22-20} = 0b000;
  let Inst{15-12} = {?, ?, ?, ?}; // Ra
  let Inst{7-4} = 0b0000; // Multiply
}

def t2MLS: T2I<(outs rGPR:$dst), (ins rGPR:$a, rGPR:$b, rGPR:$c), IIC_iMAC32,
		"mls", "\t$dst, $a, $b, $c",
                [(set rGPR:$dst, (sub rGPR:$c, (mul rGPR:$a, rGPR:$b)))]> {
  let Inst{31-27} = 0b11111;
  let Inst{26-23} = 0b0110;
  let Inst{22-20} = 0b000;
  let Inst{15-12} = {?, ?, ?, ?}; // Ra
  let Inst{7-4} = 0b0001; // Multiply and Subtract
}

// Extra precision multiplies with low / high results
let neverHasSideEffects = 1 in {
let isCommutable = 1 in {
def t2SMULL : T2I<(outs rGPR:$ldst, rGPR:$hdst),
                  (ins rGPR:$a, rGPR:$b), IIC_iMUL64,
                   "smull", "\t$ldst, $hdst, $a, $b", []> {
  let Inst{31-27} = 0b11111;
  let Inst{26-23} = 0b0111;
  let Inst{22-20} = 0b000;
  let Inst{7-4} = 0b0000;
}

def t2UMULL : T2I<(outs rGPR:$ldst, rGPR:$hdst),
                  (ins rGPR:$a, rGPR:$b), IIC_iMUL64,
                   "umull", "\t$ldst, $hdst, $a, $b", []> {
  let Inst{31-27} = 0b11111;
  let Inst{26-23} = 0b0111;
  let Inst{22-20} = 0b010;
  let Inst{7-4} = 0b0000;
}
} // isCommutable

// Multiply + accumulate
def t2SMLAL : T2I<(outs rGPR:$ldst, rGPR:$hdst),
                  (ins rGPR:$a, rGPR:$b), IIC_iMAC64,
                  "smlal", "\t$ldst, $hdst, $a, $b", []>{
  let Inst{31-27} = 0b11111;
  let Inst{26-23} = 0b0111;
  let Inst{22-20} = 0b100;
  let Inst{7-4} = 0b0000;
}

def t2UMLAL : T2I<(outs rGPR:$ldst, rGPR:$hdst),
                  (ins rGPR:$a, rGPR:$b), IIC_iMAC64,
                  "umlal", "\t$ldst, $hdst, $a, $b", []>{
  let Inst{31-27} = 0b11111;
  let Inst{26-23} = 0b0111;
  let Inst{22-20} = 0b110;
  let Inst{7-4} = 0b0000;
}

def t2UMAAL : T2I<(outs rGPR:$ldst, rGPR:$hdst),
                  (ins rGPR:$a, rGPR:$b), IIC_iMAC64,
                  "umaal", "\t$ldst, $hdst, $a, $b", []>{
  let Inst{31-27} = 0b11111;
  let Inst{26-23} = 0b0111;
  let Inst{22-20} = 0b110;
  let Inst{7-4} = 0b0110;
}
} // neverHasSideEffects

// Rounding variants of the below included for disassembly only

// Most significant word multiply
def t2SMMUL : T2I<(outs rGPR:$dst), (ins rGPR:$a, rGPR:$b), IIC_iMUL32,
                  "smmul", "\t$dst, $a, $b",
                  [(set rGPR:$dst, (mulhs rGPR:$a, rGPR:$b))]> {
  let Inst{31-27} = 0b11111;
  let Inst{26-23} = 0b0110;
  let Inst{22-20} = 0b101;
  let Inst{15-12} = 0b1111; // Ra = 0b1111 (no accumulate)
  let Inst{7-4} = 0b0000; // No Rounding (Inst{4} = 0)
}

def t2SMMULR : T2I<(outs rGPR:$dst), (ins rGPR:$a, rGPR:$b), IIC_iMUL32,
                  "smmulr", "\t$dst, $a, $b", []> {
  let Inst{31-27} = 0b11111;
  let Inst{26-23} = 0b0110;
  let Inst{22-20} = 0b101;
  let Inst{15-12} = 0b1111; // Ra = 0b1111 (no accumulate)
  let Inst{7-4} = 0b0001; // Rounding (Inst{4} = 1)
}

def t2SMMLA : T2I<(outs rGPR:$dst), (ins rGPR:$a, rGPR:$b, rGPR:$c), IIC_iMAC32,
                  "smmla", "\t$dst, $a, $b, $c",
                  [(set rGPR:$dst, (add (mulhs rGPR:$a, rGPR:$b), rGPR:$c))]> {
  let Inst{31-27} = 0b11111;
  let Inst{26-23} = 0b0110;
  let Inst{22-20} = 0b101;
  let Inst{15-12} = {?, ?, ?, ?}; // Ra
  let Inst{7-4} = 0b0000; // No Rounding (Inst{4} = 0)
}

def t2SMMLAR: T2I<(outs rGPR:$dst), (ins rGPR:$a, rGPR:$b, rGPR:$c), IIC_iMAC32,
                  "smmlar", "\t$dst, $a, $b, $c", []> {
  let Inst{31-27} = 0b11111;
  let Inst{26-23} = 0b0110;
  let Inst{22-20} = 0b101;
  let Inst{15-12} = {?, ?, ?, ?}; // Ra
  let Inst{7-4} = 0b0001; // Rounding (Inst{4} = 1)
}

def t2SMMLS: T2I <(outs rGPR:$dst), (ins rGPR:$a, rGPR:$b, rGPR:$c), IIC_iMAC32,
                   "smmls", "\t$dst, $a, $b, $c",
                   [(set rGPR:$dst, (sub rGPR:$c, (mulhs rGPR:$a, rGPR:$b)))]> {
  let Inst{31-27} = 0b11111;
  let Inst{26-23} = 0b0110;
  let Inst{22-20} = 0b110;
  let Inst{15-12} = {?, ?, ?, ?}; // Ra
  let Inst{7-4} = 0b0000; // No Rounding (Inst{4} = 0)
}

def t2SMMLSR:T2I <(outs rGPR:$dst), (ins rGPR:$a, rGPR:$b, rGPR:$c), IIC_iMAC32,
                   "smmlsr", "\t$dst, $a, $b, $c", []> {
  let Inst{31-27} = 0b11111;
  let Inst{26-23} = 0b0110;
  let Inst{22-20} = 0b110;
  let Inst{15-12} = {?, ?, ?, ?}; // Ra
  let Inst{7-4} = 0b0001; // Rounding (Inst{4} = 1)
}

multiclass T2I_smul<string opc, PatFrag opnode> {
  def BB : T2I<(outs rGPR:$dst), (ins rGPR:$a, rGPR:$b), IIC_iMUL32,
              !strconcat(opc, "bb"), "\t$dst, $a, $b",
              [(set rGPR:$dst, (opnode (sext_inreg rGPR:$a, i16),
                                      (sext_inreg rGPR:$b, i16)))]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0110;
    let Inst{22-20} = 0b001;
    let Inst{15-12} = 0b1111; // Ra = 0b1111 (no accumulate)
    let Inst{7-6} = 0b00;
    let Inst{5-4} = 0b00;
  }

  def BT : T2I<(outs rGPR:$dst), (ins rGPR:$a, rGPR:$b), IIC_iMUL32,
              !strconcat(opc, "bt"), "\t$dst, $a, $b",
              [(set rGPR:$dst, (opnode (sext_inreg rGPR:$a, i16),
                                      (sra rGPR:$b, (i32 16))))]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0110;
    let Inst{22-20} = 0b001;
    let Inst{15-12} = 0b1111; // Ra = 0b1111 (no accumulate)
    let Inst{7-6} = 0b00;
    let Inst{5-4} = 0b01;
  }

  def TB : T2I<(outs rGPR:$dst), (ins rGPR:$a, rGPR:$b), IIC_iMUL32,
              !strconcat(opc, "tb"), "\t$dst, $a, $b",
              [(set rGPR:$dst, (opnode (sra rGPR:$a, (i32 16)),
                                      (sext_inreg rGPR:$b, i16)))]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0110;
    let Inst{22-20} = 0b001;
    let Inst{15-12} = 0b1111; // Ra = 0b1111 (no accumulate)
    let Inst{7-6} = 0b00;
    let Inst{5-4} = 0b10;
  }

  def TT : T2I<(outs rGPR:$dst), (ins rGPR:$a, rGPR:$b), IIC_iMUL32,
              !strconcat(opc, "tt"), "\t$dst, $a, $b",
              [(set rGPR:$dst, (opnode (sra rGPR:$a, (i32 16)),
                                      (sra rGPR:$b, (i32 16))))]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0110;
    let Inst{22-20} = 0b001;
    let Inst{15-12} = 0b1111; // Ra = 0b1111 (no accumulate)
    let Inst{7-6} = 0b00;
    let Inst{5-4} = 0b11;
  }

  def WB : T2I<(outs rGPR:$dst), (ins rGPR:$a, rGPR:$b), IIC_iMUL16,
              !strconcat(opc, "wb"), "\t$dst, $a, $b",
              [(set rGPR:$dst, (sra (opnode rGPR:$a,
                                    (sext_inreg rGPR:$b, i16)), (i32 16)))]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0110;
    let Inst{22-20} = 0b011;
    let Inst{15-12} = 0b1111; // Ra = 0b1111 (no accumulate)
    let Inst{7-6} = 0b00;
    let Inst{5-4} = 0b00;
  }

  def WT : T2I<(outs rGPR:$dst), (ins rGPR:$a, rGPR:$b), IIC_iMUL16,
              !strconcat(opc, "wt"), "\t$dst, $a, $b",
              [(set rGPR:$dst, (sra (opnode rGPR:$a,
                                    (sra rGPR:$b, (i32 16))), (i32 16)))]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0110;
    let Inst{22-20} = 0b011;
    let Inst{15-12} = 0b1111; // Ra = 0b1111 (no accumulate)
    let Inst{7-6} = 0b00;
    let Inst{5-4} = 0b01;
  }
}


multiclass T2I_smla<string opc, PatFrag opnode> {
  def BB : T2I<(outs rGPR:$dst), (ins rGPR:$a, rGPR:$b, rGPR:$acc), IIC_iMAC16,
              !strconcat(opc, "bb"), "\t$dst, $a, $b, $acc",
              [(set rGPR:$dst, (add rGPR:$acc,
                               (opnode (sext_inreg rGPR:$a, i16),
                                       (sext_inreg rGPR:$b, i16))))]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0110;
    let Inst{22-20} = 0b001;
    let Inst{15-12} = {?, ?, ?, ?}; // Ra
    let Inst{7-6} = 0b00;
    let Inst{5-4} = 0b00;
  }

  def BT : T2I<(outs rGPR:$dst), (ins rGPR:$a, rGPR:$b, rGPR:$acc), IIC_iMAC16,
             !strconcat(opc, "bt"), "\t$dst, $a, $b, $acc",
             [(set rGPR:$dst, (add rGPR:$acc, (opnode (sext_inreg rGPR:$a, i16),
                                                  (sra rGPR:$b, (i32 16)))))]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0110;
    let Inst{22-20} = 0b001;
    let Inst{15-12} = {?, ?, ?, ?}; // Ra
    let Inst{7-6} = 0b00;
    let Inst{5-4} = 0b01;
  }

  def TB : T2I<(outs rGPR:$dst), (ins rGPR:$a, rGPR:$b, rGPR:$acc), IIC_iMAC16,
              !strconcat(opc, "tb"), "\t$dst, $a, $b, $acc",
              [(set rGPR:$dst, (add rGPR:$acc, (opnode (sra rGPR:$a, (i32 16)),
                                                (sext_inreg rGPR:$b, i16))))]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0110;
    let Inst{22-20} = 0b001;
    let Inst{15-12} = {?, ?, ?, ?}; // Ra
    let Inst{7-6} = 0b00;
    let Inst{5-4} = 0b10;
  }

  def TT : T2I<(outs rGPR:$dst), (ins rGPR:$a, rGPR:$b, rGPR:$acc), IIC_iMAC16,
              !strconcat(opc, "tt"), "\t$dst, $a, $b, $acc",
             [(set rGPR:$dst, (add rGPR:$acc, (opnode (sra rGPR:$a, (i32 16)),
                                                  (sra rGPR:$b, (i32 16)))))]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0110;
    let Inst{22-20} = 0b001;
    let Inst{15-12} = {?, ?, ?, ?}; // Ra
    let Inst{7-6} = 0b00;
    let Inst{5-4} = 0b11;
  }

  def WB : T2I<(outs rGPR:$dst), (ins rGPR:$a, rGPR:$b, rGPR:$acc), IIC_iMAC16,
              !strconcat(opc, "wb"), "\t$dst, $a, $b, $acc",
              [(set rGPR:$dst, (add rGPR:$acc, (sra (opnode rGPR:$a,
                                     (sext_inreg rGPR:$b, i16)), (i32 16))))]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0110;
    let Inst{22-20} = 0b011;
    let Inst{15-12} = {?, ?, ?, ?}; // Ra
    let Inst{7-6} = 0b00;
    let Inst{5-4} = 0b00;
  }

  def WT : T2I<(outs rGPR:$dst), (ins rGPR:$a, rGPR:$b, rGPR:$acc), IIC_iMAC16,
              !strconcat(opc, "wt"), "\t$dst, $a, $b, $acc",
              [(set rGPR:$dst, (add rGPR:$acc, (sra (opnode rGPR:$a,
                                       (sra rGPR:$b, (i32 16))), (i32 16))))]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0110;
    let Inst{22-20} = 0b011;
    let Inst{15-12} = {?, ?, ?, ?}; // Ra
    let Inst{7-6} = 0b00;
    let Inst{5-4} = 0b01;
  }
}

defm t2SMUL : T2I_smul<"smul", BinOpFrag<(mul node:$LHS, node:$RHS)>>;
defm t2SMLA : T2I_smla<"smla", BinOpFrag<(mul node:$LHS, node:$RHS)>>;

// Halfword multiple accumulate long: SMLAL<x><y> -- for disassembly only
def t2SMLALBB : T2I_mac<1, 0b100, 0b1000, (outs rGPR:$ldst,rGPR:$hdst),
         (ins rGPR:$a,rGPR:$b), IIC_iMAC64, "smlalbb", "\t$ldst, $hdst, $a, $b",
           [/* For disassembly only; pattern left blank */]>;
def t2SMLALBT : T2I_mac<1, 0b100, 0b1001, (outs rGPR:$ldst,rGPR:$hdst),
         (ins rGPR:$a,rGPR:$b), IIC_iMAC64, "smlalbt", "\t$ldst, $hdst, $a, $b",
           [/* For disassembly only; pattern left blank */]>;
def t2SMLALTB : T2I_mac<1, 0b100, 0b1010, (outs rGPR:$ldst,rGPR:$hdst),
         (ins rGPR:$a,rGPR:$b), IIC_iMAC64, "smlaltb", "\t$ldst, $hdst, $a, $b",
           [/* For disassembly only; pattern left blank */]>;
def t2SMLALTT : T2I_mac<1, 0b100, 0b1011, (outs rGPR:$ldst,rGPR:$hdst),
         (ins rGPR:$a,rGPR:$b), IIC_iMAC64, "smlaltt", "\t$ldst, $hdst, $a, $b",
           [/* For disassembly only; pattern left blank */]>;

// Dual halfword multiple: SMUAD, SMUSD, SMLAD, SMLSD, SMLALD, SMLSLD
// These are for disassembly only.

def t2SMUAD: T2I_mac<0, 0b010, 0b0000, (outs rGPR:$dst), (ins rGPR:$a, rGPR:$b),
                     IIC_iMAC32, "smuad", "\t$dst, $a, $b", []> {
  let Inst{15-12} = 0b1111;
}
def t2SMUADX:T2I_mac<0, 0b010, 0b0001, (outs rGPR:$dst), (ins rGPR:$a, rGPR:$b),
                     IIC_iMAC32, "smuadx", "\t$dst, $a, $b", []> {
  let Inst{15-12} = 0b1111;
}
def t2SMUSD: T2I_mac<0, 0b100, 0b0000, (outs rGPR:$dst), (ins rGPR:$a, rGPR:$b),
                     IIC_iMAC32, "smusd", "\t$dst, $a, $b", []> {
  let Inst{15-12} = 0b1111;
}
def t2SMUSDX:T2I_mac<0, 0b100, 0b0001, (outs rGPR:$dst), (ins rGPR:$a, rGPR:$b),
                     IIC_iMAC32, "smusdx", "\t$dst, $a, $b", []> {
  let Inst{15-12} = 0b1111;
}
def t2SMLAD   : T2I_mac<0, 0b010, 0b0000, (outs rGPR:$dst),
                        (ins rGPR:$a, rGPR:$b, rGPR:$acc), IIC_iMAC32, "smlad",
                        "\t$dst, $a, $b, $acc", []>;
def t2SMLADX  : T2I_mac<0, 0b010, 0b0001, (outs rGPR:$dst),
                        (ins rGPR:$a, rGPR:$b, rGPR:$acc), IIC_iMAC32, "smladx",
                        "\t$dst, $a, $b, $acc", []>;
def t2SMLSD   : T2I_mac<0, 0b100, 0b0000, (outs rGPR:$dst),
                        (ins rGPR:$a, rGPR:$b, rGPR:$acc), IIC_iMAC32, "smlsd",
                        "\t$dst, $a, $b, $acc", []>;
def t2SMLSDX  : T2I_mac<0, 0b100, 0b0001, (outs rGPR:$dst),
                        (ins rGPR:$a, rGPR:$b, rGPR:$acc), IIC_iMAC32, "smlsdx",
                        "\t$dst, $a, $b, $acc", []>;
def t2SMLALD  : T2I_mac<1, 0b100, 0b1100, (outs rGPR:$ldst,rGPR:$hdst),
                        (ins rGPR:$a,rGPR:$b), IIC_iMAC64, "smlald",
                        "\t$ldst, $hdst, $a, $b", []>;
def t2SMLALDX : T2I_mac<1, 0b100, 0b1101, (outs rGPR:$ldst,rGPR:$hdst),
                        (ins rGPR:$a,rGPR:$b), IIC_iMAC64, "smlaldx",
                        "\t$ldst, $hdst, $a, $b", []>;
def t2SMLSLD  : T2I_mac<1, 0b101, 0b1100, (outs rGPR:$ldst,rGPR:$hdst),
                        (ins rGPR:$a,rGPR:$b), IIC_iMAC64, "smlsld",
                        "\t$ldst, $hdst, $a, $b", []>;
def t2SMLSLDX : T2I_mac<1, 0b101, 0b1101, (outs rGPR:$ldst,rGPR:$hdst),
                        (ins rGPR:$a,rGPR:$b), IIC_iMAC64, "smlsldx",
                        "\t$ldst, $hdst, $a, $b", []>;

//===----------------------------------------------------------------------===//
//  Misc. Arithmetic Instructions.
//

class T2I_misc<bits<2> op1, bits<2> op2, dag oops, dag iops,
      InstrItinClass itin, string opc, string asm, list<dag> pattern>
  : T2I<oops, iops, itin, opc, asm, pattern> {
  let Inst{31-27} = 0b11111;
  let Inst{26-22} = 0b01010;
  let Inst{21-20} = op1;
  let Inst{15-12} = 0b1111;
  let Inst{7-6} = 0b10;
  let Inst{5-4} = op2;
}

def t2CLZ : T2I_misc<0b11, 0b00, (outs rGPR:$dst), (ins rGPR:$src), IIC_iUNAr,
                    "clz", "\t$dst, $src", [(set rGPR:$dst, (ctlz rGPR:$src))]>;

def t2RBIT : T2I_misc<0b01, 0b10, (outs rGPR:$dst), (ins rGPR:$src), IIC_iUNAr,
                      "rbit", "\t$dst, $src",
                      [(set rGPR:$dst, (ARMrbit rGPR:$src))]>;

def t2REV : T2I_misc<0b01, 0b00, (outs rGPR:$dst), (ins rGPR:$src), IIC_iUNAr,
                 "rev", ".w\t$dst, $src", [(set rGPR:$dst, (bswap rGPR:$src))]>;

def t2REV16 : T2I_misc<0b01, 0b01, (outs rGPR:$dst), (ins rGPR:$src), IIC_iUNAr,
                       "rev16", ".w\t$dst, $src",
                [(set rGPR:$dst,
                    (or (and (srl rGPR:$src, (i32 8)), 0xFF),
                        (or (and (shl rGPR:$src, (i32 8)), 0xFF00),
                            (or (and (srl rGPR:$src, (i32 8)), 0xFF0000),
                               (and (shl rGPR:$src, (i32 8)), 0xFF000000)))))]>;

def t2REVSH : T2I_misc<0b01, 0b11, (outs rGPR:$dst), (ins rGPR:$src), IIC_iUNAr,
                       "revsh", ".w\t$dst, $src",
                 [(set rGPR:$dst,
                    (sext_inreg
                      (or (srl (and rGPR:$src, 0xFF00), (i32 8)),
                          (shl rGPR:$src, (i32 8))), i16))]>;

def t2PKHBT : T2I<(outs rGPR:$dst), (ins rGPR:$src1, rGPR:$src2, shift_imm:$sh),
                  IIC_iALUsi, "pkhbt", "\t$dst, $src1, $src2$sh",
                  [(set rGPR:$dst, (or (and rGPR:$src1, 0xFFFF),
                                      (and (shl rGPR:$src2, lsl_amt:$sh),
                                           0xFFFF0000)))]>,
                  Requires<[HasT2ExtractPack]> {
  let Inst{31-27} = 0b11101;
  let Inst{26-25} = 0b01;
  let Inst{24-20} = 0b01100;
  let Inst{5} = 0; // BT form
  let Inst{4} = 0;
}

// Alternate cases for PKHBT where identities eliminate some nodes.
def : T2Pat<(or (and rGPR:$src1, 0xFFFF), (and rGPR:$src2, 0xFFFF0000)),
            (t2PKHBT rGPR:$src1, rGPR:$src2, 0)>,
            Requires<[HasT2ExtractPack]>;
def : T2Pat<(or (and rGPR:$src1, 0xFFFF), (shl rGPR:$src2, imm16_31:$sh)),
            (t2PKHBT rGPR:$src1, rGPR:$src2, (lsl_shift_imm imm16_31:$sh))>,
            Requires<[HasT2ExtractPack]>;

// Note: Shifts of 1-15 bits will be transformed to srl instead of sra and
// will match the pattern below.
def t2PKHTB : T2I<(outs rGPR:$dst), (ins rGPR:$src1, rGPR:$src2, shift_imm:$sh),
                  IIC_iALUsi, "pkhtb", "\t$dst, $src1, $src2$sh",
                  [(set rGPR:$dst, (or (and rGPR:$src1, 0xFFFF0000),
                                       (and (sra rGPR:$src2, asr_amt:$sh),
                                            0xFFFF)))]>,
                  Requires<[HasT2ExtractPack]> {
  let Inst{31-27} = 0b11101;
  let Inst{26-25} = 0b01;
  let Inst{24-20} = 0b01100;
  let Inst{5} = 1; // TB form
  let Inst{4} = 0;
}

// Alternate cases for PKHTB where identities eliminate some nodes.  Note that
// a shift amount of 0 is *not legal* here, it is PKHBT instead.
def : T2Pat<(or (and rGPR:$src1, 0xFFFF0000), (srl rGPR:$src2, imm16_31:$sh)),
            (t2PKHTB rGPR:$src1, rGPR:$src2, (asr_shift_imm imm16_31:$sh))>,
            Requires<[HasT2ExtractPack]>;
def : T2Pat<(or (and rGPR:$src1, 0xFFFF0000),
                (and (srl rGPR:$src2, imm1_15:$sh), 0xFFFF)),
            (t2PKHTB rGPR:$src1, rGPR:$src2, (asr_shift_imm imm1_15:$sh))>,
            Requires<[HasT2ExtractPack]>;

//===----------------------------------------------------------------------===//
//  Comparison Instructions...
//
defm t2CMP  : T2I_cmp_irs<0b1101, "cmp",
                          BinOpFrag<(ARMcmp node:$LHS, node:$RHS)>>;
defm t2CMPz : T2I_cmp_irs<0b1101, "cmp",
                          BinOpFrag<(ARMcmpZ node:$LHS, node:$RHS)>>;

//FIXME: Disable CMN, as CCodes are backwards from compare expectations
//       Compare-to-zero still works out, just not the relationals
//defm t2CMN  : T2I_cmp_irs<0b1000, "cmn",
//                          BinOpFrag<(ARMcmp node:$LHS,(ineg node:$RHS))>>;
defm t2CMNz : T2I_cmp_irs<0b1000, "cmn",
                          BinOpFrag<(ARMcmpZ node:$LHS,(ineg node:$RHS))>>;

//def : T2Pat<(ARMcmp  GPR:$src, t2_so_imm_neg:$imm),
//            (t2CMNri GPR:$src, t2_so_imm_neg:$imm)>;

def : T2Pat<(ARMcmpZ  GPR:$src, t2_so_imm_neg:$imm),
            (t2CMNzri GPR:$src, t2_so_imm_neg:$imm)>;

defm t2TST  : T2I_cmp_irs<0b0000, "tst",
                          BinOpFrag<(ARMcmpZ (and node:$LHS, node:$RHS), 0)>>;
defm t2TEQ  : T2I_cmp_irs<0b0100, "teq",
                          BinOpFrag<(ARMcmpZ (xor node:$LHS, node:$RHS), 0)>>;

// Conditional moves
// FIXME: should be able to write a pattern for ARMcmov, but can't use
// a two-value operand where a dag node expects two operands. :(
let neverHasSideEffects = 1 in {
def t2MOVCCr : T2I<(outs rGPR:$dst), (ins rGPR:$false, rGPR:$true), IIC_iCMOVr,
                   "mov", ".w\t$dst, $true",
   [/*(set rGPR:$dst, (ARMcmov rGPR:$false, rGPR:$true, imm:$cc, CCR:$ccr))*/]>,
                RegConstraint<"$false = $dst"> {
  let Inst{31-27} = 0b11101;
  let Inst{26-25} = 0b01;
  let Inst{24-21} = 0b0010;
  let Inst{20} = 0; // The S bit.
  let Inst{19-16} = 0b1111; // Rn
  let Inst{14-12} = 0b000;
  let Inst{7-4} = 0b0000;
}

def t2MOVCCi : T2I<(outs rGPR:$dst), (ins rGPR:$false, t2_so_imm:$true),
                   IIC_iCMOVi, "mov", ".w\t$dst, $true",
[/*(set rGPR:$dst,(ARMcmov rGPR:$false,t2_so_imm:$true, imm:$cc, CCR:$ccr))*/]>,
                   RegConstraint<"$false = $dst"> {
  let Inst{31-27} = 0b11110;
  let Inst{25} = 0;
  let Inst{24-21} = 0b0010;
  let Inst{20} = 0; // The S bit.
  let Inst{19-16} = 0b1111; // Rn
  let Inst{15} = 0;
}

class T2I_movcc_sh<bits<2> opcod, dag oops, dag iops, InstrItinClass itin,
                   string opc, string asm, list<dag> pattern>
  : T2I<oops, iops, itin, opc, asm, pattern> {
  let Inst{31-27} = 0b11101;
  let Inst{26-25} = 0b01;
  let Inst{24-21} = 0b0010;
  let Inst{20} = 0; // The S bit.
  let Inst{19-16} = 0b1111; // Rn
  let Inst{5-4} = opcod; // Shift type.
}
def t2MOVCClsl : T2I_movcc_sh<0b00, (outs rGPR:$dst),
                             (ins rGPR:$false, rGPR:$true, i32imm:$rhs),
                             IIC_iCMOVsi, "lsl", ".w\t$dst, $true, $rhs", []>,
                 RegConstraint<"$false = $dst">;
def t2MOVCClsr : T2I_movcc_sh<0b01, (outs rGPR:$dst),
                             (ins rGPR:$false, rGPR:$true, i32imm:$rhs),
                             IIC_iCMOVsi, "lsr", ".w\t$dst, $true, $rhs", []>,
                 RegConstraint<"$false = $dst">;
def t2MOVCCasr : T2I_movcc_sh<0b10, (outs rGPR:$dst),
                             (ins rGPR:$false, rGPR:$true, i32imm:$rhs),
                             IIC_iCMOVsi, "asr", ".w\t$dst, $true, $rhs", []>,
                 RegConstraint<"$false = $dst">;
def t2MOVCCror : T2I_movcc_sh<0b11, (outs rGPR:$dst),
                             (ins rGPR:$false, rGPR:$true, i32imm:$rhs),
                             IIC_iCMOVsi, "ror", ".w\t$dst, $true, $rhs", []>,
                 RegConstraint<"$false = $dst">;
} // neverHasSideEffects

//===----------------------------------------------------------------------===//
// Atomic operations intrinsics
//

// memory barriers protect the atomic sequences
let hasSideEffects = 1 in {
def t2DMBsy : AInoP<(outs), (ins), ThumbFrm, NoItinerary, "dmb", "",
                    [(ARMMemBarrier)]>, Requires<[IsThumb, HasDB]> {
  let Inst{31-4} = 0xF3BF8F5;
  // FIXME: add support for options other than a full system DMB
  let Inst{3-0} = 0b1111;
}

def t2DSBsy : AInoP<(outs), (ins), ThumbFrm, NoItinerary, "dsb", "",
                    [(ARMSyncBarrier)]>, Requires<[IsThumb, HasDB]> {
  let Inst{31-4} = 0xF3BF8F4;
  // FIXME: add support for options other than a full system DSB
  let Inst{3-0} = 0b1111;
}
}

// Helper class for multiclass T2MemB -- for disassembly only
class T2I_memb<string opc, string asm>
  : T2I<(outs), (ins), NoItinerary, opc, asm,
        [/* For disassembly only; pattern left blank */]>,
    Requires<[IsThumb2, HasV7]> {
  let Inst{31-20} = 0xf3b;
  let Inst{15-14} = 0b10;
  let Inst{12} = 0;
}

multiclass T2MemB<bits<4> op7_4, string opc> {

  def st : T2I_memb<opc, "\tst"> {
    let Inst{7-4} = op7_4;
    let Inst{3-0} = 0b1110;
  }

  def ish : T2I_memb<opc, "\tish"> {
    let Inst{7-4} = op7_4;
    let Inst{3-0} = 0b1011;
  }

  def ishst : T2I_memb<opc, "\tishst"> {
    let Inst{7-4} = op7_4;
    let Inst{3-0} = 0b1010;
  }

  def nsh : T2I_memb<opc, "\tnsh"> {
    let Inst{7-4} = op7_4;
    let Inst{3-0} = 0b0111;
  }

  def nshst : T2I_memb<opc, "\tnshst"> {
    let Inst{7-4} = op7_4;
    let Inst{3-0} = 0b0110;
  }

  def osh : T2I_memb<opc, "\tosh"> {
    let Inst{7-4} = op7_4;
    let Inst{3-0} = 0b0011;
  }

  def oshst : T2I_memb<opc, "\toshst"> {
    let Inst{7-4} = op7_4;
    let Inst{3-0} = 0b0010;
  }
}

// These DMB variants are for disassembly only.
defm t2DMB : T2MemB<0b0101, "dmb">;

// These DSB variants are for disassembly only.
defm t2DSB : T2MemB<0b0100, "dsb">;

// ISB has only full system option -- for disassembly only
def t2ISBsy : T2I_memb<"isb", ""> {
  let Inst{7-4} = 0b0110;
  let Inst{3-0} = 0b1111;
}

class T2I_ldrex<bits<2> opcod, dag oops, dag iops, AddrMode am, SizeFlagVal sz,
                InstrItinClass itin, string opc, string asm, string cstr,
                list<dag> pattern, bits<4> rt2 = 0b1111>
  : Thumb2I<oops, iops, am, sz, itin, opc, asm, cstr, pattern> {
  let Inst{31-27} = 0b11101;
  let Inst{26-20} = 0b0001101;
  let Inst{11-8} = rt2;
  let Inst{7-6} = 0b01;
  let Inst{5-4} = opcod;
  let Inst{3-0} = 0b1111;
}
class T2I_strex<bits<2> opcod, dag oops, dag iops, AddrMode am, SizeFlagVal sz,
                InstrItinClass itin, string opc, string asm, string cstr,
                list<dag> pattern, bits<4> rt2 = 0b1111>
  : Thumb2I<oops, iops, am, sz, itin, opc, asm, cstr, pattern> {
  let Inst{31-27} = 0b11101;
  let Inst{26-20} = 0b0001100;
  let Inst{11-8} = rt2;
  let Inst{7-6} = 0b01;
  let Inst{5-4} = opcod;
}

let mayLoad = 1 in {
def t2LDREXB : T2I_ldrex<0b00, (outs rGPR:$dest), (ins rGPR:$ptr), AddrModeNone,
                         Size4Bytes, NoItinerary, "ldrexb", "\t$dest, [$ptr]",
                         "", []>;
def t2LDREXH : T2I_ldrex<0b01, (outs rGPR:$dest), (ins rGPR:$ptr), AddrModeNone,
                         Size4Bytes, NoItinerary, "ldrexh", "\t$dest, [$ptr]",
                         "", []>;
def t2LDREX  : Thumb2I<(outs rGPR:$dest), (ins rGPR:$ptr), AddrModeNone,
                       Size4Bytes, NoItinerary,
                       "ldrex", "\t$dest, [$ptr]", "",
                      []> {
  let Inst{31-27} = 0b11101;
  let Inst{26-20} = 0b0000101;
  let Inst{11-8} = 0b1111;
  let Inst{7-0} = 0b00000000; // imm8 = 0
}
def t2LDREXD : T2I_ldrex<0b11, (outs rGPR:$dest, rGPR:$dest2), (ins rGPR:$ptr),
                         AddrModeNone, Size4Bytes, NoItinerary,
                         "ldrexd", "\t$dest, $dest2, [$ptr]", "",
                         [], {?, ?, ?, ?}>;
}

let mayStore = 1, Constraints = "@earlyclobber $success" in {
def t2STREXB : T2I_strex<0b00, (outs rGPR:$success), (ins rGPR:$src, rGPR:$ptr),
                         AddrModeNone, Size4Bytes, NoItinerary,
                         "strexb", "\t$success, $src, [$ptr]", "", []>;
def t2STREXH : T2I_strex<0b01, (outs rGPR:$success), (ins rGPR:$src, rGPR:$ptr),
                         AddrModeNone, Size4Bytes, NoItinerary,
                         "strexh", "\t$success, $src, [$ptr]", "", []>;
def t2STREX  : Thumb2I<(outs rGPR:$success), (ins rGPR:$src, rGPR:$ptr),
                       AddrModeNone, Size4Bytes, NoItinerary,
                       "strex", "\t$success, $src, [$ptr]", "",
                      []> {
  let Inst{31-27} = 0b11101;
  let Inst{26-20} = 0b0000100;
  let Inst{7-0} = 0b00000000; // imm8 = 0
}
def t2STREXD : T2I_strex<0b11, (outs rGPR:$success),
                         (ins rGPR:$src, rGPR:$src2, rGPR:$ptr),
                         AddrModeNone, Size4Bytes, NoItinerary,
                         "strexd", "\t$success, $src, $src2, [$ptr]", "", [],
                         {?, ?, ?, ?}>;
}

// Clear-Exclusive is for disassembly only.
def t2CLREX : T2I<(outs), (ins), NoItinerary, "clrex", "",
                  [/* For disassembly only; pattern left blank */]>,
            Requires<[IsARM, HasV7]>  {
  let Inst{31-20} = 0xf3b;
  let Inst{15-14} = 0b10;
  let Inst{12} = 0;
  let Inst{7-4} = 0b0010;
}

//===----------------------------------------------------------------------===//
// TLS Instructions
//

// __aeabi_read_tp preserves the registers r1-r3.
let isCall = 1,
  Defs = [R0, R12, LR, CPSR] in {
  def t2TPsoft : T2XI<(outs), (ins), IIC_Br,
                     "bl\t__aeabi_read_tp",
                     [(set R0, ARMthread_pointer)]> {
    let Inst{31-27} = 0b11110;
    let Inst{15-14} = 0b11;
    let Inst{12} = 1;
  }
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
//   $val is a scratch register for our use.
let Defs =
  [ R0,  R1,  R2,  R3,  R4,  R5,  R6,  R7,  R8,  R9,  R10, R11, R12, LR,  D0,
    D1,  D2,  D3,  D4,  D5,  D6,  D7,  D8,  D9,  D10, D11, D12, D13, D14, D15,
    D16, D17, D18, D19, D20, D21, D22, D23, D24, D25, D26, D27, D28, D29, D30,
    D31 ], hasSideEffects = 1, isBarrier = 1 in {
  def t2Int_eh_sjlj_setjmp : Thumb2XI<(outs), (ins tGPR:$src, tGPR:$val),
                               AddrModeNone, SizeSpecial, NoItinerary,
                               "mov\t$val, pc\t${:comment} begin eh.setjmp\n\t"
                               "adds\t$val, #7\n\t"
                               "str\t$val, [$src, #4]\n\t"
                               "movs\tr0, #0\n\t"
                               "b\t1f\n\t"
                               "movs\tr0, #1\t${:comment} end eh.setjmp\n\t"
                               "1:", "",
                          [(set R0, (ARMeh_sjlj_setjmp tGPR:$src, tGPR:$val))]>,
                             Requires<[IsThumb2, HasVFP2]>;
}

let Defs =
  [ R0,  R1,  R2,  R3,  R4,  R5,  R6,  R7,  R8,  R9,  R10, R11, R12, LR ],
  hasSideEffects = 1, isBarrier = 1 in {
  def t2Int_eh_sjlj_setjmp_nofp : Thumb2XI<(outs), (ins tGPR:$src, tGPR:$val),
                               AddrModeNone, SizeSpecial, NoItinerary,
                               "mov\t$val, pc\t${:comment} begin eh.setjmp\n\t"
                               "adds\t$val, #7\n\t"
                               "str\t$val, [$src, #4]\n\t"
                               "movs\tr0, #0\n\t"
                               "b\t1f\n\t"
                               "movs\tr0, #1\t${:comment} end eh.setjmp\n\t"
                               "1:", "",
                          [(set R0, (ARMeh_sjlj_setjmp tGPR:$src, tGPR:$val))]>,
                                  Requires<[IsThumb2, NoVFP]>;
}


//===----------------------------------------------------------------------===//
// Control-Flow Instructions
//

// FIXME: remove when we have a way to marking a MI with these properties.
// FIXME: $dst1 should be a def. But the extra ops must be in the end of the
// operand list.
// FIXME: Should pc be an implicit operand like PICADD, etc?
let isReturn = 1, isTerminator = 1, isBarrier = 1, mayLoad = 1,
    hasExtraDefRegAllocReq = 1 in
  def t2LDM_RET : T2XIt<(outs GPR:$wb), (ins addrmode4:$addr, pred:$p,
                                         reglist:$dsts, variable_ops), IIC_Br,
                        "ldm${addr:submode}${p}${addr:wide}\t$addr!, $dsts",
                        "$addr.addr = $wb", []> {
  let Inst{31-27} = 0b11101;
  let Inst{26-25} = 0b00;
  let Inst{24-23} = {?, ?}; // IA: '01', DB: '10'
  let Inst{22} = 0;
  let Inst{21} = 1; // The W bit.
  let Inst{20} = 1; // Load
}

let isBranch = 1, isTerminator = 1, isBarrier = 1 in {
let isPredicable = 1 in
def t2B   : T2XI<(outs), (ins brtarget:$target), IIC_Br,
                 "b.w\t$target",
                 [(br bb:$target)]> {
  let Inst{31-27} = 0b11110;
  let Inst{15-14} = 0b10;
  let Inst{12} = 1;
}

let isNotDuplicable = 1, isIndirectBranch = 1 in {
def t2BR_JT :
    T2JTI<(outs),
          (ins GPR:$target, GPR:$index, jt2block_operand:$jt, i32imm:$id),
           IIC_Br, "mov\tpc, $target$jt",
          [(ARMbr2jt GPR:$target, GPR:$index, tjumptable:$jt, imm:$id)]> {
  let Inst{31-27} = 0b11101;
  let Inst{26-20} = 0b0100100;
  let Inst{19-16} = 0b1111;
  let Inst{14-12} = 0b000;
  let Inst{11-8} = 0b1111; // Rd = pc
  let Inst{7-4} = 0b0000;
}

// FIXME: Add a non-pc based case that can be predicated.
def t2TBB :
    T2JTI<(outs),
        (ins tb_addrmode:$index, jt2block_operand:$jt, i32imm:$id),
         IIC_Br, "tbb\t$index$jt", []> {
  let Inst{31-27} = 0b11101;
  let Inst{26-20} = 0b0001101;
  let Inst{19-16} = 0b1111; // Rn = pc (table follows this instruction)
  let Inst{15-8} = 0b11110000;
  let Inst{7-4} = 0b0000; // B form
}

def t2TBH :
    T2JTI<(outs),
        (ins tb_addrmode:$index, jt2block_operand:$jt, i32imm:$id),
         IIC_Br, "tbh\t$index$jt", []> {
  let Inst{31-27} = 0b11101;
  let Inst{26-20} = 0b0001101;
  let Inst{19-16} = 0b1111; // Rn = pc (table follows this instruction)
  let Inst{15-8} = 0b11110000;
  let Inst{7-4} = 0b0001; // H form
}

// Generic versions of the above two instructions, for disassembly only

def t2TBBgen : T2I<(outs), (ins GPR:$a, GPR:$b), IIC_Br,
                    "tbb", "\t[$a, $b]", []>{
  let Inst{31-27} = 0b11101;
  let Inst{26-20} = 0b0001101;
  let Inst{15-8} = 0b11110000;
  let Inst{7-4} = 0b0000; // B form
}

def t2TBHgen : T2I<(outs), (ins GPR:$a, GPR:$b), IIC_Br,
                   "tbh", "\t[$a, $b, lsl #1]", []> {
  let Inst{31-27} = 0b11101;
  let Inst{26-20} = 0b0001101;
  let Inst{15-8} = 0b11110000;
  let Inst{7-4} = 0b0001; // H form
}
} // isNotDuplicable, isIndirectBranch

} // isBranch, isTerminator, isBarrier

// FIXME: should be able to write a pattern for ARMBrcond, but can't use
// a two-value operand where a dag node expects two operands. :(
let isBranch = 1, isTerminator = 1 in
def t2Bcc : T2I<(outs), (ins brtarget:$target), IIC_Br,
                "b", ".w\t$target",
                [/*(ARMbrcond bb:$target, imm:$cc)*/]> {
  let Inst{31-27} = 0b11110;
  let Inst{15-14} = 0b10;
  let Inst{12} = 0;
}


// IT block
let Defs = [ITSTATE] in
def t2IT : Thumb2XI<(outs), (ins it_pred:$cc, it_mask:$mask),
                    AddrModeNone, Size2Bytes,  IIC_iALUx,
                    "it$mask\t$cc", "", []> {
  // 16-bit instruction.
  let Inst{31-16} = 0x0000;
  let Inst{15-8} = 0b10111111;
}

// Branch and Exchange Jazelle -- for disassembly only
// Rm = Inst{19-16}
def t2BXJ : T2I<(outs), (ins rGPR:$func), NoItinerary, "bxj", "\t$func",
              [/* For disassembly only; pattern left blank */]> {
  let Inst{31-27} = 0b11110;
  let Inst{26} = 0;
  let Inst{25-20} = 0b111100;
  let Inst{15-14} = 0b10;
  let Inst{12} = 0;
}

// Change Processor State is a system instruction -- for disassembly only.
// The singleton $opt operand contains the following information:
// opt{4-0} = mode from Inst{4-0}
// opt{5} = changemode from Inst{17}
// opt{8-6} = AIF from Inst{8-6}
// opt{10-9} = imod from Inst{19-18} with 0b10 as enable and 0b11 as disable
def t2CPS : T2XI<(outs),(ins cps_opt:$opt), NoItinerary, "cps$opt",
                 [/* For disassembly only; pattern left blank */]> {
  let Inst{31-27} = 0b11110;
  let Inst{26} = 0;
  let Inst{25-20} = 0b111010;
  let Inst{15-14} = 0b10;
  let Inst{12} = 0;
}

// A6.3.4 Branches and miscellaneous control
// Table A6-14 Change Processor State, and hint instructions
// Helper class for disassembly only.
class T2I_hint<bits<8> op7_0, string opc, string asm>
  : T2I<(outs), (ins), NoItinerary, opc, asm,
        [/* For disassembly only; pattern left blank */]> {
  let Inst{31-20} = 0xf3a;
  let Inst{15-14} = 0b10;
  let Inst{12} = 0;
  let Inst{10-8} = 0b000;
  let Inst{7-0} = op7_0;
}

def t2NOP   : T2I_hint<0b00000000, "nop",   ".w">;
def t2YIELD : T2I_hint<0b00000001, "yield", ".w">;
def t2WFE   : T2I_hint<0b00000010, "wfe",   ".w">;
def t2WFI   : T2I_hint<0b00000011, "wfi",   ".w">;
def t2SEV   : T2I_hint<0b00000100, "sev",   ".w">;

def t2DBG : T2I<(outs),(ins i32imm:$opt), NoItinerary, "dbg", "\t$opt",
                [/* For disassembly only; pattern left blank */]> {
  let Inst{31-20} = 0xf3a;
  let Inst{15-14} = 0b10;
  let Inst{12} = 0;
  let Inst{10-8} = 0b000;
  let Inst{7-4} = 0b1111;
}

// Secure Monitor Call is a system instruction -- for disassembly only
// Option = Inst{19-16}
def t2SMC : T2I<(outs), (ins i32imm:$opt), NoItinerary, "smc", "\t$opt",
                [/* For disassembly only; pattern left blank */]> {
  let Inst{31-27} = 0b11110;
  let Inst{26-20} = 0b1111111;
  let Inst{15-12} = 0b1000;
}

// Store Return State is a system instruction -- for disassembly only
def t2SRSDBW : T2I<(outs),(ins i32imm:$mode),NoItinerary,"srsdb","\tsp!, $mode",
                   [/* For disassembly only; pattern left blank */]> {
  let Inst{31-27} = 0b11101;
  let Inst{26-20} = 0b0000010; // W = 1
}

def t2SRSDB  : T2I<(outs),(ins i32imm:$mode),NoItinerary,"srsdb","\tsp, $mode",
                   [/* For disassembly only; pattern left blank */]> {
  let Inst{31-27} = 0b11101;
  let Inst{26-20} = 0b0000000; // W = 0
}

def t2SRSIAW : T2I<(outs),(ins i32imm:$mode),NoItinerary,"srsia","\tsp!, $mode",
                   [/* For disassembly only; pattern left blank */]> {
  let Inst{31-27} = 0b11101;
  let Inst{26-20} = 0b0011010; // W = 1
}

def t2SRSIA  : T2I<(outs), (ins i32imm:$mode),NoItinerary,"srsia","\tsp, $mode",
                   [/* For disassembly only; pattern left blank */]> {
  let Inst{31-27} = 0b11101;
  let Inst{26-20} = 0b0011000; // W = 0
}

// Return From Exception is a system instruction -- for disassembly only
def t2RFEDBW : T2I<(outs), (ins rGPR:$base), NoItinerary, "rfedb", "\t$base!",
                   [/* For disassembly only; pattern left blank */]> {
  let Inst{31-27} = 0b11101;
  let Inst{26-20} = 0b0000011; // W = 1
}

def t2RFEDB  : T2I<(outs), (ins rGPR:$base), NoItinerary, "rfeab", "\t$base",
                   [/* For disassembly only; pattern left blank */]> {
  let Inst{31-27} = 0b11101;
  let Inst{26-20} = 0b0000001; // W = 0
}

def t2RFEIAW : T2I<(outs), (ins rGPR:$base), NoItinerary, "rfeia", "\t$base!",
                   [/* For disassembly only; pattern left blank */]> {
  let Inst{31-27} = 0b11101;
  let Inst{26-20} = 0b0011011; // W = 1
}

def t2RFEIA  : T2I<(outs), (ins rGPR:$base), NoItinerary, "rfeia", "\t$base",
                   [/* For disassembly only; pattern left blank */]> {
  let Inst{31-27} = 0b11101;
  let Inst{26-20} = 0b0011001; // W = 0
}

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//

// Two piece so_imms.
def : T2Pat<(or rGPR:$LHS, t2_so_imm2part:$RHS),
             (t2ORRri (t2ORRri rGPR:$LHS, (t2_so_imm2part_1 imm:$RHS)),
                    (t2_so_imm2part_2 imm:$RHS))>;
def : T2Pat<(xor rGPR:$LHS, t2_so_imm2part:$RHS),
             (t2EORri (t2EORri rGPR:$LHS, (t2_so_imm2part_1 imm:$RHS)),
                    (t2_so_imm2part_2 imm:$RHS))>;
def : T2Pat<(add rGPR:$LHS, t2_so_imm2part:$RHS),
             (t2ADDri (t2ADDri rGPR:$LHS, (t2_so_imm2part_1 imm:$RHS)),
                    (t2_so_imm2part_2 imm:$RHS))>;
def : T2Pat<(add rGPR:$LHS, t2_so_neg_imm2part:$RHS),
             (t2SUBri (t2SUBri rGPR:$LHS, (t2_so_neg_imm2part_1 imm:$RHS)),
                    (t2_so_neg_imm2part_2 imm:$RHS))>;

// 32-bit immediate using movw + movt.
// This is a single pseudo instruction to make it re-materializable. Remove
// when we can do generalized remat.
let isReMaterializable = 1 in
def t2MOVi32imm : T2Ix2<(outs rGPR:$dst), (ins i32imm:$src), IIC_iMOVi,
                   "movw", "\t$dst, ${src:lo16}\n\tmovt${p}\t$dst, ${src:hi16}",
                     [(set rGPR:$dst, (i32 imm:$src))]>;

// ConstantPool, GlobalAddress, and JumpTable
def : T2Pat<(ARMWrapper  tglobaladdr :$dst), (t2LEApcrel tglobaladdr :$dst)>,
           Requires<[IsThumb2, DontUseMovt]>;
def : T2Pat<(ARMWrapper  tconstpool  :$dst), (t2LEApcrel tconstpool  :$dst)>;
def : T2Pat<(ARMWrapper  tglobaladdr :$dst), (t2MOVi32imm tglobaladdr :$dst)>,
           Requires<[IsThumb2, UseMovt]>;

def : T2Pat<(ARMWrapperJT tjumptable:$dst, imm:$id),
            (t2LEApcrelJT tjumptable:$dst, imm:$id)>;

// Pseudo instruction that combines ldr from constpool and add pc. This should
// be expanded into two instructions late to allow if-conversion and
// scheduling.
let canFoldAsLoad = 1, isReMaterializable = 1 in
def t2LDRpci_pic : PseudoInst<(outs GPR:$dst), (ins i32imm:$addr, pclabel:$cp),
                   NoItinerary,
                   "${:comment} ldr.w\t$dst, $addr\n$cp:\n\tadd\t$dst, pc",
               [(set GPR:$dst, (ARMpic_add (load (ARMWrapper tconstpool:$addr)),
                                           imm:$cp))]>,
               Requires<[IsThumb2]>;

//===----------------------------------------------------------------------===//
// Move between special register and ARM core register -- for disassembly only
//

// Rd = Instr{11-8}
def t2MRS : T2I<(outs rGPR:$dst), (ins), NoItinerary, "mrs", "\t$dst, cpsr",
                [/* For disassembly only; pattern left blank */]> {
  let Inst{31-27} = 0b11110;
  let Inst{26} = 0;
  let Inst{25-21} = 0b11111;
  let Inst{20} = 0; // The R bit.
  let Inst{15-14} = 0b10;
  let Inst{12} = 0;
}

// Rd = Instr{11-8}
def t2MRSsys : T2I<(outs rGPR:$dst), (ins), NoItinerary, "mrs", "\t$dst, spsr",
                   [/* For disassembly only; pattern left blank */]> {
  let Inst{31-27} = 0b11110;
  let Inst{26} = 0;
  let Inst{25-21} = 0b11111;
  let Inst{20} = 1; // The R bit.
  let Inst{15-14} = 0b10;
  let Inst{12} = 0;
}

// Rn = Inst{19-16}
def t2MSR : T2I<(outs), (ins rGPR:$src, msr_mask:$mask), NoItinerary, "msr",
                "\tcpsr$mask, $src",
                [/* For disassembly only; pattern left blank */]> {
  let Inst{31-27} = 0b11110;
  let Inst{26} = 0;
  let Inst{25-21} = 0b11100;
  let Inst{20} = 0; // The R bit.
  let Inst{15-14} = 0b10;
  let Inst{12} = 0;
}

// Rn = Inst{19-16}
def t2MSRsys : T2I<(outs), (ins rGPR:$src, msr_mask:$mask), NoItinerary, "msr",
                   "\tspsr$mask, $src",
                   [/* For disassembly only; pattern left blank */]> {
  let Inst{31-27} = 0b11110;
  let Inst{26} = 0;
  let Inst{25-21} = 0b11100;
  let Inst{20} = 1; // The R bit.
  let Inst{15-14} = 0b10;
  let Inst{12} = 0;
}
