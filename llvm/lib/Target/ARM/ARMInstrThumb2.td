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

// Shifted operands. No register controlled shifts for Thumb2.
// Note: We do not support rrx shifted operands yet.
def t2_so_reg : Operand<i32>,    // reg imm
                ComplexPattern<i32, 2, "SelectT2ShifterOperandReg",
                               [shl,srl,sra,rotr]> {
  let EncoderMethod = "getT2SORegOpValue";
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
def t2_so_imm : Operand<i32>, PatLeaf<(imm), [{ return Pred_t2_so_imm(N); }]> {
  let EncoderMethod = "getT2SOImmOpValue";
}

// t2_so_imm_not - Match an immediate that is a complement
// of a t2_so_imm.
def t2_so_imm_not : Operand<i32>,
                    PatLeaf<(imm), [{
  return ARM_AM::getT2SOImmVal(~((uint32_t)N->getZExtValue())) != -1;
}], t2_so_imm_not_XFORM>;

// t2_so_imm_neg - Match an immediate that is a negation of a t2_so_imm.
def t2_so_imm_neg : Operand<i32>,
                    PatLeaf<(imm), [{
  return ARM_AM::getT2SOImmVal(-((uint32_t)N->getZExtValue())) != -1;
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
  let PrintMethod = "printAddrModeImm12Operand";
  let EncoderMethod = "getAddrModeImm12OpValue";
  let MIOperandInfo = (ops GPR:$base, i32imm:$offsimm);
  let ParserMatchClass = MemMode5AsmOperand;
}

// ADR instruction labels.
def t2adrlabel : Operand<i32> {
  let EncoderMethod = "getT2AdrLabelOpValue";
}


// t2addrmode_imm8  := reg +/- imm8
def t2addrmode_imm8 : Operand<i32>,
                      ComplexPattern<i32, 2, "SelectT2AddrModeImm8", []> {
  let PrintMethod = "printT2AddrModeImm8Operand";
  let EncoderMethod = "getT2AddrModeImm8OpValue";
  let MIOperandInfo = (ops GPR:$base, i32imm:$offsimm);
  let ParserMatchClass = MemMode5AsmOperand;
}

def t2am_imm8_offset : Operand<i32>,
                       ComplexPattern<i32, 1, "SelectT2AddrModeImm8Offset",
                                      [], [SDNPWantRoot]> {
  let PrintMethod = "printT2AddrModeImm8OffsetOperand";
  let EncoderMethod = "getT2AddrModeImm8OffsetOpValue";
  let ParserMatchClass = MemMode5AsmOperand;
}

// t2addrmode_imm8s4  := reg +/- (imm8 << 2)
def t2addrmode_imm8s4 : Operand<i32> {
  let PrintMethod = "printT2AddrModeImm8s4Operand";
  let EncoderMethod = "getT2AddrModeImm8s4OpValue";
  let MIOperandInfo = (ops GPR:$base, i32imm:$offsimm);
  let ParserMatchClass = MemMode5AsmOperand;
}

def t2am_imm8s4_offset : Operand<i32> {
  let PrintMethod = "printT2AddrModeImm8s4OffsetOperand";
}

// t2addrmode_so_reg  := reg + (reg << imm2)
def t2addrmode_so_reg : Operand<i32>,
                        ComplexPattern<i32, 3, "SelectT2AddrModeSoReg", []> {
  let PrintMethod = "printT2AddrModeSoRegOperand";
  let EncoderMethod = "getT2AddrModeSORegOpValue";
  let MIOperandInfo = (ops GPR:$base, rGPR:$offsreg, i32imm:$offsimm);
  let ParserMatchClass = MemMode5AsmOperand;
}


//===----------------------------------------------------------------------===//
// Multiclass helpers...
//


class T2OneRegImm<dag oops, dag iops, InstrItinClass itin,
           string opc, string asm, list<dag> pattern>
  : T2I<oops, iops, itin, opc, asm, pattern> {
  bits<4> Rd;
  bits<12> imm;

  let Inst{11-8}  = Rd;
  let Inst{26}    = imm{11};
  let Inst{14-12} = imm{10-8};
  let Inst{7-0}   = imm{7-0};
}


class T2sOneRegImm<dag oops, dag iops, InstrItinClass itin,
           string opc, string asm, list<dag> pattern>
  : T2sI<oops, iops, itin, opc, asm, pattern> {
  bits<4> Rd;
  bits<4> Rn;
  bits<12> imm;

  let Inst{11-8}  = Rd;
  let Inst{26}    = imm{11};
  let Inst{14-12} = imm{10-8};
  let Inst{7-0}   = imm{7-0};
}

class T2OneRegCmpImm<dag oops, dag iops, InstrItinClass itin,
           string opc, string asm, list<dag> pattern>
  : T2I<oops, iops, itin, opc, asm, pattern> {
  bits<4> Rn;
  bits<12> imm;

  let Inst{19-16}  = Rn;
  let Inst{26}    = imm{11};
  let Inst{14-12} = imm{10-8};
  let Inst{7-0}   = imm{7-0};
}


class T2OneRegShiftedReg<dag oops, dag iops, InstrItinClass itin,
           string opc, string asm, list<dag> pattern>
  : T2I<oops, iops, itin, opc, asm, pattern> {
  bits<4> Rd;
  bits<12> ShiftedRm;

  let Inst{11-8}  = Rd;
  let Inst{3-0}   = ShiftedRm{3-0};
  let Inst{5-4}   = ShiftedRm{6-5};
  let Inst{14-12} = ShiftedRm{11-9};
  let Inst{7-6}   = ShiftedRm{8-7};
}

class T2sOneRegShiftedReg<dag oops, dag iops, InstrItinClass itin,
           string opc, string asm, list<dag> pattern>
  : T2sI<oops, iops, itin, opc, asm, pattern> {
  bits<4> Rd;
  bits<12> ShiftedRm;

  let Inst{11-8}  = Rd;
  let Inst{3-0}   = ShiftedRm{3-0};
  let Inst{5-4}   = ShiftedRm{6-5};
  let Inst{14-12} = ShiftedRm{11-9};
  let Inst{7-6}   = ShiftedRm{8-7};
}

class T2OneRegCmpShiftedReg<dag oops, dag iops, InstrItinClass itin,
           string opc, string asm, list<dag> pattern>
  : T2I<oops, iops, itin, opc, asm, pattern> {
  bits<4> Rn;
  bits<12> ShiftedRm;

  let Inst{19-16} = Rn;
  let Inst{3-0}   = ShiftedRm{3-0};
  let Inst{5-4}   = ShiftedRm{6-5};
  let Inst{14-12} = ShiftedRm{11-9};
  let Inst{7-6}   = ShiftedRm{8-7};
}

class T2TwoReg<dag oops, dag iops, InstrItinClass itin,
           string opc, string asm, list<dag> pattern>
  : T2I<oops, iops, itin, opc, asm, pattern> {
  bits<4> Rd;
  bits<4> Rm;

  let Inst{11-8}  = Rd;
  let Inst{3-0}   = Rm;
}

class T2sTwoReg<dag oops, dag iops, InstrItinClass itin,
           string opc, string asm, list<dag> pattern>
  : T2sI<oops, iops, itin, opc, asm, pattern> {
  bits<4> Rd;
  bits<4> Rm;

  let Inst{11-8}  = Rd;
  let Inst{3-0}   = Rm;
}

class T2TwoRegCmp<dag oops, dag iops, InstrItinClass itin,
           string opc, string asm, list<dag> pattern>
  : T2I<oops, iops, itin, opc, asm, pattern> {
  bits<4> Rn;
  bits<4> Rm;

  let Inst{19-16} = Rn;
  let Inst{3-0}   = Rm;
}


class T2TwoRegImm<dag oops, dag iops, InstrItinClass itin,
           string opc, string asm, list<dag> pattern>
  : T2I<oops, iops, itin, opc, asm, pattern> {
  bits<4> Rd;
  bits<4> Rn;
  bits<12> imm;

  let Inst{11-8}  = Rd;
  let Inst{19-16} = Rn;
  let Inst{26}    = imm{11};
  let Inst{14-12} = imm{10-8};
  let Inst{7-0}   = imm{7-0};
}

class T2sTwoRegImm<dag oops, dag iops, InstrItinClass itin,
           string opc, string asm, list<dag> pattern>
  : T2sI<oops, iops, itin, opc, asm, pattern> {
  bits<4> Rd;
  bits<4> Rn;
  bits<12> imm;

  let Inst{11-8}  = Rd;
  let Inst{19-16} = Rn;
  let Inst{26}    = imm{11};
  let Inst{14-12} = imm{10-8};
  let Inst{7-0}   = imm{7-0};
}

class T2TwoRegShiftImm<dag oops, dag iops, InstrItinClass itin,
           string opc, string asm, list<dag> pattern>
  : T2I<oops, iops, itin, opc, asm, pattern> {
  bits<4> Rd;
  bits<4> Rm;
  bits<5> imm;

  let Inst{11-8}  = Rd;
  let Inst{3-0}   = Rm;
  let Inst{14-12} = imm{4-2};
  let Inst{7-6}   = imm{1-0};
}

class T2sTwoRegShiftImm<dag oops, dag iops, InstrItinClass itin,
           string opc, string asm, list<dag> pattern>
  : T2sI<oops, iops, itin, opc, asm, pattern> {
  bits<4> Rd;
  bits<4> Rm;
  bits<5> imm;

  let Inst{11-8}  = Rd;
  let Inst{3-0}   = Rm;
  let Inst{14-12} = imm{4-2};
  let Inst{7-6}   = imm{1-0};
}

class T2ThreeReg<dag oops, dag iops, InstrItinClass itin,
           string opc, string asm, list<dag> pattern>
  : T2I<oops, iops, itin, opc, asm, pattern> {
  bits<4> Rd;
  bits<4> Rn;
  bits<4> Rm;

  let Inst{11-8}  = Rd;
  let Inst{19-16} = Rn;
  let Inst{3-0}   = Rm;
}

class T2sThreeReg<dag oops, dag iops, InstrItinClass itin,
           string opc, string asm, list<dag> pattern>
  : T2sI<oops, iops, itin, opc, asm, pattern> {
  bits<4> Rd;
  bits<4> Rn;
  bits<4> Rm;

  let Inst{11-8}  = Rd;
  let Inst{19-16} = Rn;
  let Inst{3-0}   = Rm;
}

class T2TwoRegShiftedReg<dag oops, dag iops, InstrItinClass itin,
           string opc, string asm, list<dag> pattern>
  : T2I<oops, iops, itin, opc, asm, pattern> {
  bits<4> Rd;
  bits<4> Rn;
  bits<12> ShiftedRm;

  let Inst{11-8}  = Rd;
  let Inst{19-16} = Rn;
  let Inst{3-0}   = ShiftedRm{3-0};
  let Inst{5-4}   = ShiftedRm{6-5};
  let Inst{14-12} = ShiftedRm{11-9};
  let Inst{7-6}   = ShiftedRm{8-7};
}

class T2sTwoRegShiftedReg<dag oops, dag iops, InstrItinClass itin,
           string opc, string asm, list<dag> pattern>
  : T2sI<oops, iops, itin, opc, asm, pattern> {
  bits<4> Rd;
  bits<4> Rn;
  bits<12> ShiftedRm;

  let Inst{11-8}  = Rd;
  let Inst{19-16} = Rn;
  let Inst{3-0}   = ShiftedRm{3-0};
  let Inst{5-4}   = ShiftedRm{6-5};
  let Inst{14-12} = ShiftedRm{11-9};
  let Inst{7-6}   = ShiftedRm{8-7};
}

class T2FourReg<dag oops, dag iops, InstrItinClass itin,
           string opc, string asm, list<dag> pattern>
  : T2I<oops, iops, itin, opc, asm, pattern> {
  bits<4> Rd;
  bits<4> Rn;
  bits<4> Rm;
  bits<4> Ra;

  let Inst{19-16} = Rn;
  let Inst{15-12} = Ra;
  let Inst{11-8}  = Rd;
  let Inst{3-0}   = Rm;
}

class T2MulLong<bits<3> opc22_20, bits<4> opc7_4,
                dag oops, dag iops, InstrItinClass itin,
                string opc, string asm, list<dag> pattern>
  : T2I<oops, iops, itin, opc, asm, pattern> {
  bits<4> RdLo;
  bits<4> RdHi;
  bits<4> Rn;
  bits<4> Rm;

  let Inst{31-23} = 0b111110111;
  let Inst{22-20} = opc22_20;
  let Inst{19-16} = Rn;
  let Inst{15-12} = RdLo;
  let Inst{11-8}  = RdHi;
  let Inst{7-4}   = opc7_4;
  let Inst{3-0}   = Rm;
}


/// T2I_un_irs - Defines a set of (op reg, {so_imm|r|so_reg}) patterns for a
/// unary operation that produces a value. These are predicable and can be
/// changed to modify CPSR.
multiclass T2I_un_irs<bits<4> opcod, string opc,
                     InstrItinClass iii, InstrItinClass iir, InstrItinClass iis,
                      PatFrag opnode, bit Cheap = 0, bit ReMat = 0> {
   // shifted imm
   def i : T2sOneRegImm<(outs rGPR:$Rd), (ins t2_so_imm:$imm), iii,
                opc, "\t$Rd, $imm",
                [(set rGPR:$Rd, (opnode t2_so_imm:$imm))]> {
     let isAsCheapAsAMove = Cheap;
     let isReMaterializable = ReMat;
     let Inst{31-27} = 0b11110;
     let Inst{25} = 0;
     let Inst{24-21} = opcod;
     let Inst{19-16} = 0b1111; // Rn
     let Inst{15} = 0;
   }
   // register
   def r : T2sTwoReg<(outs rGPR:$Rd), (ins rGPR:$Rm), iir,
                opc, ".w\t$Rd, $Rm",
                [(set rGPR:$Rd, (opnode rGPR:$Rm))]> {
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24-21} = opcod;
     let Inst{19-16} = 0b1111; // Rn
     let Inst{14-12} = 0b000; // imm3
     let Inst{7-6} = 0b00; // imm2
     let Inst{5-4} = 0b00; // type
   }
   // shifted register
   def s : T2sOneRegShiftedReg<(outs rGPR:$Rd), (ins t2_so_reg:$ShiftedRm), iis,
                opc, ".w\t$Rd, $ShiftedRm",
                [(set rGPR:$Rd, (opnode t2_so_reg:$ShiftedRm))]> {
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24-21} = opcod;
     let Inst{19-16} = 0b1111; // Rn
   }
}

/// T2I_bin_irs - Defines a set of (op reg, {so_imm|r|so_reg}) patterns for a
/// binary operation that produces a value. These are predicable and can be
/// changed to modify CPSR.
multiclass T2I_bin_irs<bits<4> opcod, string opc,
                     InstrItinClass iii, InstrItinClass iir, InstrItinClass iis,
                       PatFrag opnode, bit Commutable = 0, string wide = ""> {
   // shifted imm
   def ri : T2sTwoRegImm<
                (outs rGPR:$Rd), (ins rGPR:$Rn, t2_so_imm:$imm), iii,
                 opc, "\t$Rd, $Rn, $imm",
                 [(set rGPR:$Rd, (opnode rGPR:$Rn, t2_so_imm:$imm))]> {
     let Inst{31-27} = 0b11110;
     let Inst{25} = 0;
     let Inst{24-21} = opcod;
     let Inst{15} = 0;
   }
   // register
   def rr : T2sThreeReg<(outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm), iir,
                 opc, !strconcat(wide, "\t$Rd, $Rn, $Rm"),
                 [(set rGPR:$Rd, (opnode rGPR:$Rn, rGPR:$Rm))]> {
     let isCommutable = Commutable;
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24-21} = opcod;
     let Inst{14-12} = 0b000; // imm3
     let Inst{7-6} = 0b00; // imm2
     let Inst{5-4} = 0b00; // type
   }
   // shifted register
   def rs : T2sTwoRegShiftedReg<
                 (outs rGPR:$Rd), (ins rGPR:$Rn, t2_so_reg:$ShiftedRm), iis,
                 opc, !strconcat(wide, "\t$Rd, $Rn, $ShiftedRm"),
                 [(set rGPR:$Rd, (opnode rGPR:$Rn, t2_so_reg:$ShiftedRm))]> {
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24-21} = opcod;
   }
}

/// T2I_bin_w_irs - Same as T2I_bin_irs except these operations need
//  the ".w" prefix to indicate that they are wide.
multiclass T2I_bin_w_irs<bits<4> opcod, string opc,
                     InstrItinClass iii, InstrItinClass iir, InstrItinClass iis,
                         PatFrag opnode, bit Commutable = 0> :
    T2I_bin_irs<opcod, opc, iii, iir, iis, opnode, Commutable, ".w">;

/// T2I_rbin_is - Same as T2I_bin_irs except the order of operands are
/// reversed.  The 'rr' form is only defined for the disassembler; for codegen
/// it is equivalent to the T2I_bin_irs counterpart.
multiclass T2I_rbin_irs<bits<4> opcod, string opc, PatFrag opnode> {
   // shifted imm
   def ri : T2sTwoRegImm<
                 (outs rGPR:$Rd), (ins rGPR:$Rn, t2_so_imm:$imm), IIC_iALUi,
                 opc, ".w\t$Rd, $Rn, $imm",
                 [(set rGPR:$Rd, (opnode t2_so_imm:$imm, rGPR:$Rn))]> {
     let Inst{31-27} = 0b11110;
     let Inst{25} = 0;
     let Inst{24-21} = opcod;
     let Inst{15} = 0;
   }
   // register
   def rr : T2sThreeReg<
                 (outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm), IIC_iALUr,
                 opc, "\t$Rd, $Rn, $Rm",
                 [/* For disassembly only; pattern left blank */]> {
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24-21} = opcod;
     let Inst{14-12} = 0b000; // imm3
     let Inst{7-6} = 0b00; // imm2
     let Inst{5-4} = 0b00; // type
   }
   // shifted register
   def rs : T2sTwoRegShiftedReg<
                 (outs rGPR:$Rd), (ins rGPR:$Rn, t2_so_reg:$ShiftedRm),
                 IIC_iALUsir, opc, "\t$Rd, $Rn, $ShiftedRm",
                 [(set rGPR:$Rd, (opnode t2_so_reg:$ShiftedRm, rGPR:$Rn))]> {
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24-21} = opcod;
   }
}

/// T2I_bin_s_irs - Similar to T2I_bin_irs except it sets the 's' bit so the
/// instruction modifies the CPSR register.
let isCodeGenOnly = 1, Defs = [CPSR] in {
multiclass T2I_bin_s_irs<bits<4> opcod, string opc,
                     InstrItinClass iii, InstrItinClass iir, InstrItinClass iis,
                         PatFrag opnode, bit Commutable = 0> {
   // shifted imm
   def ri : T2TwoRegImm<
                (outs rGPR:$Rd), (ins GPR:$Rn, t2_so_imm:$imm), iii,
                !strconcat(opc, "s"), ".w\t$Rd, $Rn, $imm",
                [(set rGPR:$Rd, (opnode GPR:$Rn, t2_so_imm:$imm))]> {
     let Inst{31-27} = 0b11110;
     let Inst{25} = 0;
     let Inst{24-21} = opcod;
     let Inst{20} = 1; // The S bit.
     let Inst{15} = 0;
   }
   // register
   def rr : T2ThreeReg<
                (outs rGPR:$Rd), (ins GPR:$Rn, rGPR:$Rm), iir,
                !strconcat(opc, "s"), ".w\t$Rd, $Rn, $Rm",
                [(set rGPR:$Rd, (opnode GPR:$Rn, rGPR:$Rm))]> {
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
   def rs : T2TwoRegShiftedReg<
                (outs rGPR:$Rd), (ins GPR:$Rn, t2_so_reg:$ShiftedRm), iis,
                !strconcat(opc, "s"), ".w\t$Rd, $Rn, $ShiftedRm",
                [(set rGPR:$Rd, (opnode GPR:$Rn, t2_so_reg:$ShiftedRm))]> {
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
   // The register-immediate version is re-materializable. This is useful
   // in particular for taking the address of a local.
   let isReMaterializable = 1 in {
   def ri : T2sTwoRegImm<
                 (outs rGPR:$Rd), (ins GPR:$Rn, t2_so_imm:$imm), IIC_iALUi,
                 opc, ".w\t$Rd, $Rn, $imm",
                 [(set rGPR:$Rd, (opnode GPR:$Rn, t2_so_imm:$imm))]> {
     let Inst{31-27} = 0b11110;
     let Inst{25} = 0;
     let Inst{24} = 1;
     let Inst{23-21} = op23_21;
     let Inst{15} = 0;
   }
   }
   // 12-bit imm
   def ri12 : T2I<
                  (outs rGPR:$Rd), (ins GPR:$Rn, imm0_4095:$imm), IIC_iALUi,
                  !strconcat(opc, "w"), "\t$Rd, $Rn, $imm",
                  [(set rGPR:$Rd, (opnode GPR:$Rn, imm0_4095:$imm))]> {
     bits<4> Rd;
     bits<4> Rn;
     bits<12> imm;
     let Inst{31-27} = 0b11110;
     let Inst{26} = imm{11};
     let Inst{25-24} = 0b10;
     let Inst{23-21} = op23_21;
     let Inst{20} = 0; // The S bit.
     let Inst{19-16} = Rn;
     let Inst{15} = 0;
     let Inst{14-12} = imm{10-8};
     let Inst{11-8} = Rd;
     let Inst{7-0} = imm{7-0};
   }
   // register
   def rr : T2sThreeReg<(outs rGPR:$Rd), (ins GPR:$Rn, rGPR:$Rm), IIC_iALUr,
                 opc, ".w\t$Rd, $Rn, $Rm",
                 [(set rGPR:$Rd, (opnode GPR:$Rn, rGPR:$Rm))]> {
     let isCommutable = Commutable;
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24} = 1;
     let Inst{23-21} = op23_21;
     let Inst{14-12} = 0b000; // imm3
     let Inst{7-6} = 0b00; // imm2
     let Inst{5-4} = 0b00; // type
   }
   // shifted register
   def rs : T2sTwoRegShiftedReg<
                 (outs rGPR:$Rd), (ins GPR:$Rn, t2_so_reg:$ShiftedRm),
                 IIC_iALUsi, opc, ".w\t$Rd, $Rn, $ShiftedRm",
                 [(set rGPR:$Rd, (opnode GPR:$Rn, t2_so_reg:$ShiftedRm))]> {
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24} = 1;
     let Inst{23-21} = op23_21;
   }
}

/// T2I_adde_sube_irs - Defines a set of (op reg, {so_imm|r|so_reg}) patterns
/// for a binary operation that produces a value and use the carry
/// bit. It's not predicable.
let Uses = [CPSR] in {
multiclass T2I_adde_sube_irs<bits<4> opcod, string opc, PatFrag opnode,
                             bit Commutable = 0> {
   // shifted imm
   def ri : T2sTwoRegImm<(outs rGPR:$Rd), (ins rGPR:$Rn, t2_so_imm:$imm),
                 IIC_iALUi, opc, "\t$Rd, $Rn, $imm",
                 [(set rGPR:$Rd, (opnode rGPR:$Rn, t2_so_imm:$imm))]>,
                 Requires<[IsThumb2]> {
     let Inst{31-27} = 0b11110;
     let Inst{25} = 0;
     let Inst{24-21} = opcod;
     let Inst{15} = 0;
   }
   // register
   def rr : T2sThreeReg<(outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm), IIC_iALUr,
                 opc, ".w\t$Rd, $Rn, $Rm",
                 [(set rGPR:$Rd, (opnode rGPR:$Rn, rGPR:$Rm))]>,
                 Requires<[IsThumb2]> {
     let isCommutable = Commutable;
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24-21} = opcod;
     let Inst{14-12} = 0b000; // imm3
     let Inst{7-6} = 0b00; // imm2
     let Inst{5-4} = 0b00; // type
   }
   // shifted register
   def rs : T2sTwoRegShiftedReg<
                 (outs rGPR:$Rd), (ins rGPR:$Rn, t2_so_reg:$ShiftedRm),
                 IIC_iALUsi, opc, ".w\t$Rd, $Rn, $ShiftedRm",
                 [(set rGPR:$Rd, (opnode rGPR:$Rn, t2_so_reg:$ShiftedRm))]>,
                 Requires<[IsThumb2]> {
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24-21} = opcod;
   }
}

// Carry setting variants
let isCodeGenOnly = 1, Defs = [CPSR] in {
multiclass T2I_adde_sube_s_irs<bits<4> opcod, string opc, PatFrag opnode,
                               bit Commutable = 0> {
   // shifted imm
   def ri : T2sTwoRegImm<
                 (outs rGPR:$Rd), (ins rGPR:$Rn, t2_so_imm:$imm), IIC_iALUi,
                 opc, "\t$Rd, $Rn, $imm",
                 [(set rGPR:$Rd, (opnode rGPR:$Rn, t2_so_imm:$imm))]>,
                 Requires<[IsThumb2]> {
     let Inst{31-27} = 0b11110;
     let Inst{25} = 0;
     let Inst{24-21} = opcod;
     let Inst{20} = 1; // The S bit.
     let Inst{15} = 0;
   }
   // register
   def rr : T2sThreeReg<(outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm), IIC_iALUr,
                 opc, ".w\t$Rd, $Rn, $Rm",
                 [(set rGPR:$Rd, (opnode rGPR:$Rn, rGPR:$Rm))]>,
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
   def rs : T2sTwoRegShiftedReg<
                 (outs rGPR:$Rd), (ins rGPR:$Rn, t2_so_reg:$ShiftedRm),
                 IIC_iALUsi, opc, ".w\t$Rd, $Rn, $ShiftedRm",
                 [(set rGPR:$Rd, (opnode rGPR:$Rn, t2_so_reg:$ShiftedRm))]>,
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
let isCodeGenOnly = 1, Defs = [CPSR] in {
multiclass T2I_rbin_s_is<bits<4> opcod, string opc, PatFrag opnode> {
   // shifted imm
   def ri : T2TwoRegImm<
                (outs rGPR:$Rd), (ins rGPR:$Rn, t2_so_imm:$imm), IIC_iALUi,
                !strconcat(opc, "s"), ".w\t$Rd, $Rn, $imm",
                [(set rGPR:$Rd, (opnode t2_so_imm:$imm, rGPR:$Rn))]> {
     let Inst{31-27} = 0b11110;
     let Inst{25} = 0;
     let Inst{24-21} = opcod;
     let Inst{20} = 1; // The S bit.
     let Inst{15} = 0;
   }
   // shifted register
   def rs : T2TwoRegShiftedReg<
                (outs rGPR:$Rd), (ins rGPR:$Rn, t2_so_reg:$ShiftedRm),
                IIC_iALUsi, !strconcat(opc, "s"), "\t$Rd, $Rn, $ShiftedRm",
                [(set rGPR:$Rd, (opnode t2_so_reg:$ShiftedRm, rGPR:$Rn))]> {
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
   def ri : T2sTwoRegShiftImm<
                 (outs rGPR:$Rd), (ins rGPR:$Rm, i32imm:$imm), IIC_iMOVsi,
                 opc, ".w\t$Rd, $Rm, $imm",
                 [(set rGPR:$Rd, (opnode rGPR:$Rm, imm1_31:$imm))]> {
     let Inst{31-27} = 0b11101;
     let Inst{26-21} = 0b010010;
     let Inst{19-16} = 0b1111; // Rn
     let Inst{5-4} = opcod;
   }
   // register
   def rr : T2sThreeReg<
                 (outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm), IIC_iMOVsr,
                 opc, ".w\t$Rd, $Rn, $Rm",
                 [(set rGPR:$Rd, (opnode rGPR:$Rn, rGPR:$Rm))]> {
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
multiclass T2I_cmp_irs<bits<4> opcod, string opc,
                     InstrItinClass iii, InstrItinClass iir, InstrItinClass iis,
                       PatFrag opnode> {
   // shifted imm
   def ri : T2OneRegCmpImm<
                (outs), (ins GPR:$Rn, t2_so_imm:$imm), iii,
                opc, ".w\t$Rn, $imm",
                [(opnode GPR:$Rn, t2_so_imm:$imm)]> {
     let Inst{31-27} = 0b11110;
     let Inst{25} = 0;
     let Inst{24-21} = opcod;
     let Inst{20} = 1; // The S bit.
     let Inst{15} = 0;
     let Inst{11-8} = 0b1111; // Rd
   }
   // register
   def rr : T2TwoRegCmp<
                (outs), (ins GPR:$lhs, rGPR:$rhs), iir,
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
   def rs : T2OneRegCmpShiftedReg<
                (outs), (ins GPR:$Rn, t2_so_reg:$ShiftedRm), iis,
                opc, ".w\t$Rn, $ShiftedRm",
                [(opnode GPR:$Rn, t2_so_reg:$ShiftedRm)]> {
     let Inst{31-27} = 0b11101;
     let Inst{26-25} = 0b01;
     let Inst{24-21} = opcod;
     let Inst{20} = 1; // The S bit.
     let Inst{11-8} = 0b1111; // Rd
   }
}
}

/// T2I_ld - Defines a set of (op r, {imm12|imm8|so_reg}) load patterns.
multiclass T2I_ld<bit signed, bits<2> opcod, string opc,
                  InstrItinClass iii, InstrItinClass iis, PatFrag opnode> {
  def i12 : T2Ii12<(outs GPR:$Rt), (ins t2addrmode_imm12:$addr), iii,
                   opc, ".w\t$Rt, $addr",
                   [(set GPR:$Rt, (opnode t2addrmode_imm12:$addr))]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-25} = 0b00;
    let Inst{24} = signed;
    let Inst{23} = 1;
    let Inst{22-21} = opcod;
    let Inst{20} = 1; // load

    bits<4> Rt;
    let Inst{15-12} = Rt;

    bits<17> addr;
    let Inst{19-16} = addr{16-13}; // Rn
    let Inst{23}    = addr{12};    // U
    let Inst{11-0}  = addr{11-0};  // imm
  }
  def i8  : T2Ii8 <(outs GPR:$Rt), (ins t2addrmode_imm8:$addr), iii,
                   opc, "\t$Rt, $addr",
                   [(set GPR:$Rt, (opnode t2addrmode_imm8:$addr))]> {
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

    bits<4> Rt;
    let Inst{15-12} = Rt;

    bits<13> addr;
    let Inst{19-16} = addr{12-9}; // Rn
    let Inst{9}     = addr{8};    // U
    let Inst{7-0}   = addr{7-0};  // imm
  }
  def s   : T2Iso <(outs GPR:$Rt), (ins t2addrmode_so_reg:$addr), iis,
                   opc, ".w\t$Rt, $addr",
                   [(set GPR:$Rt, (opnode t2addrmode_so_reg:$addr))]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-25} = 0b00;
    let Inst{24} = signed;
    let Inst{23} = 0;
    let Inst{22-21} = opcod;
    let Inst{20} = 1; // load
    let Inst{11-6} = 0b000000;

    bits<4> Rt;
    let Inst{15-12} = Rt;

    bits<10> addr;
    let Inst{19-16} = addr{9-6}; // Rn
    let Inst{3-0}   = addr{5-2}; // Rm
    let Inst{5-4}   = addr{1-0}; // imm
  }

  def pci : t2PseudoInst<(outs GPR:$Rt), (ins i32imm:$addr), Size4Bytes, iis,
                      [(set GPR:$Rt, (opnode (ARMWrapper tconstpool:$addr)))]>;
}

/// T2I_st - Defines a set of (op r, {imm12|imm8|so_reg}) store patterns.
multiclass T2I_st<bits<2> opcod, string opc,
                  InstrItinClass iii, InstrItinClass iis, PatFrag opnode> {
  def i12 : T2Ii12<(outs), (ins GPR:$Rt, t2addrmode_imm12:$addr), iii,
                   opc, ".w\t$Rt, $addr",
                   [(opnode GPR:$Rt, t2addrmode_imm12:$addr)]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0001;
    let Inst{22-21} = opcod;
    let Inst{20} = 0; // !load

    bits<4> Rt;
    let Inst{15-12} = Rt;

    bits<17> addr;
    let Inst{19-16} = addr{16-13}; // Rn
    let Inst{23}    = addr{12};    // U
    let Inst{11-0}  = addr{11-0};  // imm
  }
  def i8  : T2Ii8 <(outs), (ins GPR:$Rt, t2addrmode_imm8:$addr), iii,
                   opc, "\t$Rt, $addr",
                   [(opnode GPR:$Rt, t2addrmode_imm8:$addr)]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0000;
    let Inst{22-21} = opcod;
    let Inst{20} = 0; // !load
    let Inst{11} = 1;
    // Offset: index==TRUE, wback==FALSE
    let Inst{10} = 1; // The P bit.
    let Inst{8} = 0; // The W bit.

    bits<4> Rt;
    let Inst{15-12} = Rt;

    bits<13> addr;
    let Inst{19-16} = addr{12-9}; // Rn
    let Inst{9}     = addr{8};    // U
    let Inst{7-0}   = addr{7-0};  // imm
  }
  def s   : T2Iso <(outs), (ins GPR:$Rt, t2addrmode_so_reg:$addr), iis,
                   opc, ".w\t$Rt, $addr",
                   [(opnode GPR:$Rt, t2addrmode_so_reg:$addr)]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0000;
    let Inst{22-21} = opcod;
    let Inst{20} = 0; // !load
    let Inst{11-6} = 0b000000;

    bits<4> Rt;
    let Inst{15-12} = Rt;

    bits<10> addr;
    let Inst{19-16}   = addr{9-6}; // Rn
    let Inst{3-0} = addr{5-2}; // Rm
    let Inst{5-4}   = addr{1-0}; // imm
  }
}

/// T2I_ext_rrot - A unary operation with two forms: one whose operand is a
/// register and one whose operand is a register rotated by 8/16/24.
multiclass T2I_ext_rrot<bits<3> opcod, string opc, PatFrag opnode> {
  def r     : T2TwoReg<(outs rGPR:$Rd), (ins rGPR:$Rm), IIC_iEXTr,
                  opc, ".w\t$Rd, $Rm",
                 [(set rGPR:$Rd, (opnode rGPR:$Rm))]> {
     let Inst{31-27} = 0b11111;
     let Inst{26-23} = 0b0100;
     let Inst{22-20} = opcod;
     let Inst{19-16} = 0b1111; // Rn
     let Inst{15-12} = 0b1111;
     let Inst{7} = 1;
     let Inst{5-4} = 0b00; // rotate
   }
  def r_rot : T2TwoReg<(outs rGPR:$Rd), (ins rGPR:$Rm, rot_imm:$rot), IIC_iEXTr,
                  opc, ".w\t$Rd, $Rm, ror $rot",
                 [(set rGPR:$Rd, (opnode (rotr rGPR:$Rm, rot_imm:$rot)))]> {
     let Inst{31-27} = 0b11111;
     let Inst{26-23} = 0b0100;
     let Inst{22-20} = opcod;
     let Inst{19-16} = 0b1111; // Rn
     let Inst{15-12} = 0b1111;
     let Inst{7} = 1;

     bits<2> rot;
     let Inst{5-4} = rot{1-0}; // rotate
   }
}

// UXTB16 - Requres T2ExtractPack, does not need the .w qualifier.
multiclass T2I_ext_rrot_uxtb16<bits<3> opcod, string opc, PatFrag opnode> {
  def r     : T2TwoReg<(outs rGPR:$Rd), (ins rGPR:$Rm), IIC_iEXTr,
                  opc, "\t$Rd, $Rm",
                 [(set rGPR:$Rd, (opnode rGPR:$Rm))]>,
                 Requires<[HasT2ExtractPack, IsThumb2]> {
     let Inst{31-27} = 0b11111;
     let Inst{26-23} = 0b0100;
     let Inst{22-20} = opcod;
     let Inst{19-16} = 0b1111; // Rn
     let Inst{15-12} = 0b1111;
     let Inst{7} = 1;
     let Inst{5-4} = 0b00; // rotate
   }
  def r_rot : T2TwoReg<(outs rGPR:$dst), (ins rGPR:$Rm, rot_imm:$rot),
                  IIC_iEXTr, opc, "\t$dst, $Rm, ror $rot",
                 [(set rGPR:$dst, (opnode (rotr rGPR:$Rm, rot_imm:$rot)))]>,
                 Requires<[HasT2ExtractPack, IsThumb2]> {
     let Inst{31-27} = 0b11111;
     let Inst{26-23} = 0b0100;
     let Inst{22-20} = opcod;
     let Inst{19-16} = 0b1111; // Rn
     let Inst{15-12} = 0b1111;
     let Inst{7} = 1;

     bits<2> rot;
     let Inst{5-4} = rot{1-0}; // rotate
   }
}

// SXTB16 - Requres T2ExtractPack, does not need the .w qualifier, no pattern
// supported yet.
multiclass T2I_ext_rrot_sxtb16<bits<3> opcod, string opc> {
  def r     : T2TwoReg<(outs rGPR:$Rd), (ins rGPR:$Rm), IIC_iEXTr,
                  opc, "\t$Rd, $Rm", []> {
     let Inst{31-27} = 0b11111;
     let Inst{26-23} = 0b0100;
     let Inst{22-20} = opcod;
     let Inst{19-16} = 0b1111; // Rn
     let Inst{15-12} = 0b1111;
     let Inst{7} = 1;
     let Inst{5-4} = 0b00; // rotate
   }
  def r_rot : T2TwoReg<(outs rGPR:$Rd), (ins rGPR:$Rm, i32imm:$rot), IIC_iEXTr,
                  opc, "\t$Rd, $Rm, ror $rot", []> {
     let Inst{31-27} = 0b11111;
     let Inst{26-23} = 0b0100;
     let Inst{22-20} = opcod;
     let Inst{19-16} = 0b1111; // Rn
     let Inst{15-12} = 0b1111;
     let Inst{7} = 1;

      bits<2> rot;
      let Inst{5-4} = rot{1-0}; // rotate
   }
}

/// T2I_exta_rrot - A binary operation with two forms: one whose operand is a
/// register and one whose operand is a register rotated by 8/16/24.
multiclass T2I_exta_rrot<bits<3> opcod, string opc, PatFrag opnode> {
  def rr     : T2ThreeReg<(outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm), IIC_iEXTAr,
                  opc, "\t$Rd, $Rn, $Rm",
                  [(set rGPR:$Rd, (opnode rGPR:$Rn, rGPR:$Rm))]>,
                  Requires<[HasT2ExtractPack, IsThumb2]> {
     let Inst{31-27} = 0b11111;
     let Inst{26-23} = 0b0100;
     let Inst{22-20} = opcod;
     let Inst{15-12} = 0b1111;
     let Inst{7} = 1;
     let Inst{5-4} = 0b00; // rotate
   }
  def rr_rot : T2ThreeReg<(outs rGPR:$Rd),
                  (ins rGPR:$Rn, rGPR:$Rm, rot_imm:$rot),
                  IIC_iEXTAsr, opc, "\t$Rd, $Rn, $Rm, ror $rot",
                  [(set rGPR:$Rd, (opnode rGPR:$Rn,
                                          (rotr rGPR:$Rm, rot_imm:$rot)))]>,
                  Requires<[HasT2ExtractPack, IsThumb2]> {
     let Inst{31-27} = 0b11111;
     let Inst{26-23} = 0b0100;
     let Inst{22-20} = opcod;
     let Inst{15-12} = 0b1111;
     let Inst{7} = 1;

     bits<2> rot;
     let Inst{5-4} = rot{1-0}; // rotate
   }
}

// DO variant - disassembly only, no pattern

multiclass T2I_exta_rrot_DO<bits<3> opcod, string opc> {
  def rr     : T2ThreeReg<(outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm), IIC_iEXTAr,
                  opc, "\t$Rd, $Rn, $Rm", []> {
     let Inst{31-27} = 0b11111;
     let Inst{26-23} = 0b0100;
     let Inst{22-20} = opcod;
     let Inst{15-12} = 0b1111;
     let Inst{7} = 1;
     let Inst{5-4} = 0b00; // rotate
   }
  def rr_rot : T2ThreeReg<(outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm, i32imm:$rot),
                  IIC_iEXTAsr, opc, "\t$Rd, $Rn, $Rm, ror $rot", []> {
     let Inst{31-27} = 0b11111;
     let Inst{26-23} = 0b0100;
     let Inst{22-20} = opcod;
     let Inst{15-12} = 0b1111;
     let Inst{7} = 1;

     bits<2> rot;
     let Inst{5-4} = rot{1-0}; // rotate
   }
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Miscellaneous Instructions.
//

class T2PCOneRegImm<dag oops, dag iops, InstrItinClass itin,
           string asm, list<dag> pattern>
  : T2XI<oops, iops, itin, asm, pattern> {
  bits<4> Rd;
  bits<12> label;

  let Inst{11-8}  = Rd;
  let Inst{26}    = label{11};
  let Inst{14-12} = label{10-8};
  let Inst{7-0}   = label{7-0};
}

// LEApcrel - Load a pc-relative address into a register without offending the
// assembler.
def t2ADR : T2PCOneRegImm<(outs rGPR:$Rd),
              (ins t2adrlabel:$addr, pred:$p),
              IIC_iALUi, "adr{$p}.w\t$Rd, #$addr", []> {
  let Inst{31-27} = 0b11110;
  let Inst{25-24} = 0b10;
  // Inst{23:21} = '11' (add = FALSE) or '00' (add = TRUE)
  let Inst{22} = 0;
  let Inst{20} = 0;
  let Inst{19-16} = 0b1111; // Rn
  let Inst{15} = 0;

  bits<4> Rd;
  bits<13> addr;
  let Inst{11-8} = Rd;
  let Inst{23}    = addr{12};
  let Inst{21}    = addr{12};
  let Inst{26}    = addr{11};
  let Inst{14-12} = addr{10-8};
  let Inst{7-0}   = addr{7-0};
}

let neverHasSideEffects = 1, isReMaterializable = 1 in
def t2LEApcrel   : t2PseudoInst<(outs rGPR:$Rd), (ins i32imm:$label, pred:$p),
                                Size4Bytes, IIC_iALUi, []>;
def t2LEApcrelJT : t2PseudoInst<(outs rGPR:$Rd),
                                (ins i32imm:$label, nohash_imm:$id, pred:$p),
                                Size4Bytes, IIC_iALUi,
                                []>;


// FIXME: None of these add/sub SP special instructions should be necessary
// at all for thumb2 since they use the same encodings as the generic
// add/sub instructions. In thumb1 we need them since they have dedicated
// encodings. At the least, they should be pseudo instructions.
// ADD r, sp, {so_imm|i12}
let isCodeGenOnly = 1 in {
def t2ADDrSPi   : T2sTwoRegImm<(outs GPR:$Rd), (ins GPR:$Rn, t2_so_imm:$imm),
                        IIC_iALUi, "add", ".w\t$Rd, $Rn, $imm", []> {
  let Inst{31-27} = 0b11110;
  let Inst{25} = 0;
  let Inst{24-21} = 0b1000;
  let Inst{15} = 0;
}
def t2ADDrSPi12 : T2TwoRegImm<(outs GPR:$Rd), (ins GPR:$Rn, imm0_4095:$imm),
                       IIC_iALUi, "addw", "\t$Rd, $Rn, $imm", []> {
  let Inst{31-27} = 0b11110;
  let Inst{25-20} = 0b100000;
  let Inst{15} = 0;
}

// ADD r, sp, so_reg
def t2ADDrSPs   : T2sTwoRegShiftedReg<
                        (outs GPR:$Rd), (ins GPR:$Rn, t2_so_reg:$ShiftedRm),
                        IIC_iALUsi, "add", ".w\t$Rd, $Rn, $ShiftedRm", []> {
  let Inst{31-27} = 0b11101;
  let Inst{26-25} = 0b01;
  let Inst{24-21} = 0b1000;
  let Inst{15} = 0;
}

// SUB r, sp, {so_imm|i12}
def t2SUBrSPi   : T2sTwoRegImm<(outs GPR:$Rd), (ins GPR:$Rn, t2_so_imm:$imm),
                        IIC_iALUi, "sub", ".w\t$Rd, $Rn, $imm", []> {
  let Inst{31-27} = 0b11110;
  let Inst{25} = 0;
  let Inst{24-21} = 0b1101;
  let Inst{15} = 0;
}
def t2SUBrSPi12 : T2TwoRegImm<(outs GPR:$Rd), (ins GPR:$Rn, imm0_4095:$imm),
                       IIC_iALUi, "subw", "\t$Rd, $Rn, $imm", []> {
  let Inst{31-27} = 0b11110;
  let Inst{25-20} = 0b101010;
  let Inst{15} = 0;
}

// SUB r, sp, so_reg
def t2SUBrSPs   : T2sTwoRegImm<(outs GPR:$Rd), (ins GPR:$Rn, t2_so_reg:$imm),
                       IIC_iALUsi,
                       "sub", "\t$Rd, $Rn, $imm", []> {
  let Inst{31-27} = 0b11101;
  let Inst{26-25} = 0b01;
  let Inst{24-21} = 0b1101;
  let Inst{19-16} = 0b1101; // Rn = sp
  let Inst{15} = 0;
}
} // end isCodeGenOnly = 1

// Signed and unsigned division on v7-M
def t2SDIV : T2ThreeReg<(outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm), IIC_iALUi,
                 "sdiv", "\t$Rd, $Rn, $Rm",
                 [(set rGPR:$Rd, (sdiv rGPR:$Rn, rGPR:$Rm))]>,
                 Requires<[HasDivide, IsThumb2]> {
  let Inst{31-27} = 0b11111;
  let Inst{26-21} = 0b011100;
  let Inst{20} = 0b1;
  let Inst{15-12} = 0b1111;
  let Inst{7-4} = 0b1111;
}

def t2UDIV : T2ThreeReg<(outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm), IIC_iALUi,
                 "udiv", "\t$Rd, $Rn, $Rm",
                 [(set rGPR:$Rd, (udiv rGPR:$Rn, rGPR:$Rm))]>,
                 Requires<[HasDivide, IsThumb2]> {
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
defm t2LDR   : T2I_ld<0, 0b10, "ldr", IIC_iLoad_i, IIC_iLoad_si,
                      UnOpFrag<(load node:$Src)>>;

// Loads with zero extension
defm t2LDRH  : T2I_ld<0, 0b01, "ldrh", IIC_iLoad_bh_i, IIC_iLoad_bh_si,
                      UnOpFrag<(zextloadi16 node:$Src)>>;
defm t2LDRB  : T2I_ld<0, 0b00, "ldrb", IIC_iLoad_bh_i, IIC_iLoad_bh_si,
                      UnOpFrag<(zextloadi8  node:$Src)>>;

// Loads with sign extension
defm t2LDRSH : T2I_ld<1, 0b01, "ldrsh", IIC_iLoad_bh_i, IIC_iLoad_bh_si,
                      UnOpFrag<(sextloadi16 node:$Src)>>;
defm t2LDRSB : T2I_ld<1, 0b00, "ldrsb", IIC_iLoad_bh_i, IIC_iLoad_bh_si,
                      UnOpFrag<(sextloadi8  node:$Src)>>;

let mayLoad = 1, neverHasSideEffects = 1, hasExtraDefRegAllocReq = 1 in {
// Load doubleword
def t2LDRDi8  : T2Ii8s4<1, 0, 1, (outs rGPR:$Rt, rGPR:$Rt2),
                        (ins t2addrmode_imm8s4:$addr),
                        IIC_iLoad_d_i, "ldrd", "\t$Rt, $Rt2, $addr", []>;
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
def t2LDR_PRE  : T2Iidxldst<0, 0b10, 1, 1, (outs GPR:$Rt, GPR:$Rn),
                            (ins t2addrmode_imm8:$addr),
                            AddrModeT2_i8, IndexModePre, IIC_iLoad_iu,
                            "ldr", "\t$Rt, $addr!", "$addr.base = $Rn",
                            []>;

def t2LDR_POST : T2Iidxldst<0, 0b10, 1, 0, (outs GPR:$Rt, GPR:$Rn),
                            (ins GPR:$base, t2am_imm8_offset:$addr),
                            AddrModeT2_i8, IndexModePost, IIC_iLoad_iu,
                          "ldr", "\t$Rt, [$Rn], $addr", "$base = $Rn",
                            []>;

def t2LDRB_PRE : T2Iidxldst<0, 0b00, 1, 1, (outs GPR:$Rt, GPR:$Rn),
                            (ins t2addrmode_imm8:$addr),
                            AddrModeT2_i8, IndexModePre, IIC_iLoad_bh_iu,
                            "ldrb", "\t$Rt, $addr!", "$addr.base = $Rn",
                            []>;
def t2LDRB_POST : T2Iidxldst<0, 0b00, 1, 0, (outs GPR:$Rt, GPR:$Rn),
                            (ins GPR:$base, t2am_imm8_offset:$addr),
                            AddrModeT2_i8, IndexModePost, IIC_iLoad_bh_iu,
                         "ldrb", "\t$Rt, [$Rn], $addr", "$base = $Rn",
                            []>;

def t2LDRH_PRE : T2Iidxldst<0, 0b01, 1, 1, (outs GPR:$Rt, GPR:$Rn),
                            (ins t2addrmode_imm8:$addr),
                            AddrModeT2_i8, IndexModePre, IIC_iLoad_bh_iu,
                            "ldrh", "\t$Rt, $addr!", "$addr.base = $Rn",
                            []>;
def t2LDRH_POST : T2Iidxldst<0, 0b01, 1, 0, (outs GPR:$Rt, GPR:$Rn),
                            (ins GPR:$base, t2am_imm8_offset:$addr),
                            AddrModeT2_i8, IndexModePost, IIC_iLoad_bh_iu,
                         "ldrh", "\t$Rt, [$Rn], $addr", "$base = $Rn",
                            []>;

def t2LDRSB_PRE : T2Iidxldst<1, 0b00, 1, 1, (outs GPR:$Rt, GPR:$Rn),
                            (ins t2addrmode_imm8:$addr),
                            AddrModeT2_i8, IndexModePre, IIC_iLoad_bh_iu,
                            "ldrsb", "\t$Rt, $addr!", "$addr.base = $Rn",
                            []>;
def t2LDRSB_POST : T2Iidxldst<1, 0b00, 1, 0, (outs GPR:$Rt, GPR:$Rn),
                            (ins GPR:$base, t2am_imm8_offset:$addr),
                            AddrModeT2_i8, IndexModePost, IIC_iLoad_bh_iu,
                        "ldrsb", "\t$Rt, [$Rn], $addr", "$base = $Rn",
                            []>;

def t2LDRSH_PRE : T2Iidxldst<1, 0b01, 1, 1, (outs GPR:$Rt, GPR:$Rn),
                            (ins t2addrmode_imm8:$addr),
                            AddrModeT2_i8, IndexModePre, IIC_iLoad_bh_iu,
                            "ldrsh", "\t$Rt, $addr!", "$addr.base = $Rn",
                            []>;
def t2LDRSH_POST : T2Iidxldst<1, 0b01, 1, 0, (outs GPR:$dst, GPR:$Rn),
                            (ins GPR:$base, t2am_imm8_offset:$addr),
                            AddrModeT2_i8, IndexModePost, IIC_iLoad_bh_iu,
                        "ldrsh", "\t$dst, [$Rn], $addr", "$base = $Rn",
                            []>;
} // mayLoad = 1, neverHasSideEffects = 1

// LDRT, LDRBT, LDRHT, LDRSBT, LDRSHT all have offset mode (PUW=0b110) and are
// for disassembly only.
// Ref: A8.6.57 LDR (immediate, Thumb) Encoding T4
class T2IldT<bit signed, bits<2> type, string opc, InstrItinClass ii>
  : T2Ii8<(outs GPR:$Rt), (ins t2addrmode_imm8:$addr), ii, opc,
          "\t$Rt, $addr", []> {
  let Inst{31-27} = 0b11111;
  let Inst{26-25} = 0b00;
  let Inst{24} = signed;
  let Inst{23} = 0;
  let Inst{22-21} = type;
  let Inst{20} = 1; // load
  let Inst{11} = 1;
  let Inst{10-8} = 0b110; // PUW.

  bits<4> Rt;
  bits<13> addr;
  let Inst{15-12} = Rt;
  let Inst{19-16} = addr{12-9};
  let Inst{7-0}   = addr{7-0};
}

def t2LDRT   : T2IldT<0, 0b10, "ldrt", IIC_iLoad_i>;
def t2LDRBT  : T2IldT<0, 0b00, "ldrbt", IIC_iLoad_bh_i>;
def t2LDRHT  : T2IldT<0, 0b01, "ldrht", IIC_iLoad_bh_i>;
def t2LDRSBT : T2IldT<1, 0b00, "ldrsbt", IIC_iLoad_bh_i>;
def t2LDRSHT : T2IldT<1, 0b01, "ldrsht", IIC_iLoad_bh_i>;

// Store
defm t2STR :T2I_st<0b10,"str", IIC_iStore_i, IIC_iStore_si,
                   BinOpFrag<(store node:$LHS, node:$RHS)>>;
defm t2STRB:T2I_st<0b00,"strb", IIC_iStore_bh_i, IIC_iStore_bh_si,
                   BinOpFrag<(truncstorei8 node:$LHS, node:$RHS)>>;
defm t2STRH:T2I_st<0b01,"strh", IIC_iStore_bh_i, IIC_iStore_bh_si,
                   BinOpFrag<(truncstorei16 node:$LHS, node:$RHS)>>;

// Store doubleword
let mayLoad = 1, neverHasSideEffects = 1, hasExtraSrcRegAllocReq = 1 in
def t2STRDi8 : T2Ii8s4<1, 0, 0, (outs),
                       (ins GPR:$Rt, GPR:$Rt2, t2addrmode_imm8s4:$addr),
               IIC_iStore_d_r, "strd", "\t$Rt, $Rt2, $addr", []>;

// Indexed stores
def t2STR_PRE  : T2Iidxldst<0, 0b10, 0, 1, (outs GPR:$base_wb),
                            (ins GPR:$Rt, GPR:$Rn, t2am_imm8_offset:$addr),
                            AddrModeT2_i8, IndexModePre, IIC_iStore_iu,
                         "str", "\t$Rt, [$Rn, $addr]!", "$Rn = $base_wb",
             [(set GPR:$base_wb,
                   (pre_store GPR:$Rt, GPR:$Rn, t2am_imm8_offset:$addr))]>;

def t2STR_POST : T2Iidxldst<0, 0b10, 0, 0, (outs GPR:$base_wb),
                            (ins GPR:$Rt, GPR:$Rn, t2am_imm8_offset:$addr),
                            AddrModeT2_i8, IndexModePost, IIC_iStore_iu,
                          "str", "\t$Rt, [$Rn], $addr", "$Rn = $base_wb",
             [(set GPR:$base_wb,
                  (post_store GPR:$Rt, GPR:$Rn, t2am_imm8_offset:$addr))]>;

def t2STRH_PRE  : T2Iidxldst<0, 0b01, 0, 1, (outs GPR:$base_wb),
                            (ins GPR:$Rt, GPR:$Rn, t2am_imm8_offset:$addr),
                            AddrModeT2_i8, IndexModePre, IIC_iStore_iu,
                        "strh", "\t$Rt, [$Rn, $addr]!", "$Rn = $base_wb",
        [(set GPR:$base_wb,
              (pre_truncsti16 GPR:$Rt, GPR:$Rn, t2am_imm8_offset:$addr))]>;

def t2STRH_POST : T2Iidxldst<0, 0b01, 0, 0, (outs GPR:$base_wb),
                            (ins GPR:$Rt, GPR:$Rn, t2am_imm8_offset:$addr),
                            AddrModeT2_i8, IndexModePost, IIC_iStore_bh_iu,
                         "strh", "\t$Rt, [$Rn], $addr", "$Rn = $base_wb",
       [(set GPR:$base_wb,
             (post_truncsti16 GPR:$Rt, GPR:$Rn, t2am_imm8_offset:$addr))]>;

def t2STRB_PRE  : T2Iidxldst<0, 0b00, 0, 1, (outs GPR:$base_wb),
                            (ins GPR:$Rt, GPR:$Rn, t2am_imm8_offset:$addr),
                            AddrModeT2_i8, IndexModePre, IIC_iStore_bh_iu,
                        "strb", "\t$Rt, [$Rn, $addr]!", "$Rn = $base_wb",
         [(set GPR:$base_wb,
               (pre_truncsti8 GPR:$Rt, GPR:$Rn, t2am_imm8_offset:$addr))]>;

def t2STRB_POST : T2Iidxldst<0, 0b00, 0, 0, (outs GPR:$base_wb),
                            (ins GPR:$Rt, GPR:$Rn, t2am_imm8_offset:$addr),
                            AddrModeT2_i8, IndexModePost, IIC_iStore_bh_iu,
                         "strb", "\t$Rt, [$Rn], $addr", "$Rn = $base_wb",
        [(set GPR:$base_wb,
              (post_truncsti8 GPR:$Rt, GPR:$Rn, t2am_imm8_offset:$addr))]>;

// STRT, STRBT, STRHT all have offset mode (PUW=0b110) and are for disassembly
// only.
// Ref: A8.6.193 STR (immediate, Thumb) Encoding T4
class T2IstT<bits<2> type, string opc, InstrItinClass ii>
  : T2Ii8<(outs GPR:$Rt), (ins t2addrmode_imm8:$addr), ii, opc,
          "\t$Rt, $addr", []> {
  let Inst{31-27} = 0b11111;
  let Inst{26-25} = 0b00;
  let Inst{24} = 0; // not signed
  let Inst{23} = 0;
  let Inst{22-21} = type;
  let Inst{20} = 0; // store
  let Inst{11} = 1;
  let Inst{10-8} = 0b110; // PUW

  bits<4> Rt;
  bits<13> addr;
  let Inst{15-12} = Rt;
  let Inst{19-16} = addr{12-9};
  let Inst{7-0}   = addr{7-0};
}

def t2STRT   : T2IstT<0b10, "strt", IIC_iStore_i>;
def t2STRBT  : T2IstT<0b00, "strbt", IIC_iStore_bh_i>;
def t2STRHT  : T2IstT<0b01, "strht", IIC_iStore_bh_i>;

// ldrd / strd pre / post variants
// For disassembly only.

def t2LDRD_PRE  : T2Ii8s4<1, 1, 1, (outs GPR:$Rt, GPR:$Rt2),
                 (ins GPR:$base, t2am_imm8s4_offset:$imm), IIC_iLoad_d_ru,
                 "ldrd", "\t$Rt, $Rt2, [$base, $imm]!", []>;

def t2LDRD_POST : T2Ii8s4<0, 1, 1, (outs GPR:$Rt, GPR:$Rt2),
                 (ins GPR:$base, t2am_imm8s4_offset:$imm), IIC_iLoad_d_ru,
                 "ldrd", "\t$Rt, $Rt2, [$base], $imm", []>;

def t2STRD_PRE  : T2Ii8s4<1, 1, 0, (outs),
                 (ins GPR:$Rt, GPR:$Rt2, GPR:$base, t2am_imm8s4_offset:$imm),
                 IIC_iStore_d_ru, "strd", "\t$Rt, $Rt2, [$base, $imm]!", []>;

def t2STRD_POST : T2Ii8s4<0, 1, 0, (outs),
                 (ins GPR:$Rt, GPR:$Rt2, GPR:$base, t2am_imm8s4_offset:$imm),
                 IIC_iStore_d_ru, "strd", "\t$Rt, $Rt2, [$base], $imm", []>;

// T2Ipl (Preload Data/Instruction) signals the memory system of possible future
// data/instruction access.  These are for disassembly only.
// instr_write is inverted for Thumb mode: (prefetch 3) -> (preload 0),
// (prefetch 1) -> (preload 2),  (prefetch 2) -> (preload 1).
multiclass T2Ipl<bits<1> write, bits<1> instr, string opc> {

  def i12 : T2Ii12<(outs), (ins t2addrmode_imm12:$addr), IIC_Preload, opc,
                "\t$addr",
              [(ARMPreload t2addrmode_imm12:$addr, (i32 write), (i32 instr))]> {
    let Inst{31-25} = 0b1111100;
    let Inst{24} = instr;
    let Inst{22} = 0;
    let Inst{21} = write;
    let Inst{20} = 1;
    let Inst{15-12} = 0b1111;

    bits<17> addr;
    let Inst{19-16} = addr{16-13}; // Rn
    let Inst{23}    = addr{12};    // U
    let Inst{11-0}  = addr{11-0};  // imm12
  }

  def i8 : T2Ii8<(outs), (ins t2addrmode_imm8:$addr), IIC_Preload, opc,
                "\t$addr",
               [(ARMPreload t2addrmode_imm8:$addr, (i32 write), (i32 instr))]> {
    let Inst{31-25} = 0b1111100;
    let Inst{24} = instr;
    let Inst{23} = 0; // U = 0
    let Inst{22} = 0;
    let Inst{21} = write;
    let Inst{20} = 1;
    let Inst{15-12} = 0b1111;
    let Inst{11-8} = 0b1100;

    bits<13> addr;
    let Inst{19-16} = addr{12-9}; // Rn
    let Inst{7-0}   = addr{7-0};  // imm8
  }

  def s : T2Iso<(outs), (ins t2addrmode_so_reg:$addr), IIC_Preload, opc,
               "\t$addr",
             [(ARMPreload t2addrmode_so_reg:$addr, (i32 write), (i32 instr))]> {
    let Inst{31-25} = 0b1111100;
    let Inst{24} = instr;
    let Inst{23} = 0; // add = TRUE for T1
    let Inst{22} = 0;
    let Inst{21} = write;
    let Inst{20} = 1;
    let Inst{15-12} = 0b1111;
    let Inst{11-6} = 0000000;

    bits<10> addr;
    let Inst{19-16} = addr{9-6}; // Rn
    let Inst{3-0}   = addr{5-2}; // Rm
    let Inst{5-4}   = addr{1-0}; // imm2
  }
}

defm t2PLD  : T2Ipl<0, 0, "pld">,  Requires<[IsThumb2]>;
defm t2PLDW : T2Ipl<1, 0, "pldw">, Requires<[IsThumb2,HasV7,HasMP]>;
defm t2PLI  : T2Ipl<0, 1, "pli">,  Requires<[IsThumb2,HasV7]>;

//===----------------------------------------------------------------------===//
//  Load / store multiple Instructions.
//

multiclass thumb2_ldst_mult<string asm, InstrItinClass itin,
                            InstrItinClass itin_upd, bit L_bit> {
  def IA :
    T2XI<(outs), (ins GPR:$Rn, pred:$p, reglist:$regs, variable_ops),
         itin, !strconcat(asm, "ia${p}.w\t$Rn, $regs"), []> {
    bits<4>  Rn;
    bits<16> regs;

    let Inst{31-27} = 0b11101;
    let Inst{26-25} = 0b00;
    let Inst{24-23} = 0b01;     // Increment After
    let Inst{22}    = 0;
    let Inst{21}    = 0;        // No writeback
    let Inst{20}    = L_bit;
    let Inst{19-16} = Rn;
    let Inst{15-0}  = regs;
  }
  def IA_UPD :
    T2XIt<(outs GPR:$wb), (ins GPR:$Rn, pred:$p, reglist:$regs, variable_ops),
          itin_upd, !strconcat(asm, "ia${p}.w\t$Rn!, $regs"), "$Rn = $wb", []> {
    bits<4>  Rn;
    bits<16> regs;

    let Inst{31-27} = 0b11101;
    let Inst{26-25} = 0b00;
    let Inst{24-23} = 0b01;     // Increment After
    let Inst{22}    = 0;
    let Inst{21}    = 1;        // Writeback
    let Inst{20}    = L_bit;
    let Inst{19-16} = Rn;
    let Inst{15-0}  = regs;
  }
  def DB :
    T2XI<(outs), (ins GPR:$Rn, pred:$p, reglist:$regs, variable_ops),
         itin, !strconcat(asm, "db${p}.w\t$Rn, $regs"), []> {
    bits<4>  Rn;
    bits<16> regs;

    let Inst{31-27} = 0b11101;
    let Inst{26-25} = 0b00;
    let Inst{24-23} = 0b10;     // Decrement Before
    let Inst{22}    = 0;
    let Inst{21}    = 0;        // No writeback
    let Inst{20}    = L_bit;
    let Inst{19-16} = Rn;
    let Inst{15-0}  = regs;
  }
  def DB_UPD :
    T2XIt<(outs GPR:$wb), (ins GPR:$Rn, pred:$p, reglist:$regs, variable_ops),
          itin_upd, !strconcat(asm, "db${p}.w\t$Rn, $regs"), "$Rn = $wb", []> {
    bits<4>  Rn;
    bits<16> regs;

    let Inst{31-27} = 0b11101;
    let Inst{26-25} = 0b00;
    let Inst{24-23} = 0b10;     // Decrement Before
    let Inst{22}    = 0;
    let Inst{21}    = 1;        // Writeback
    let Inst{20}    = L_bit;
    let Inst{19-16} = Rn;
    let Inst{15-0}  = regs;
  }
}

let neverHasSideEffects = 1 in {

let mayLoad = 1, hasExtraDefRegAllocReq = 1 in
defm t2LDM : thumb2_ldst_mult<"ldm", IIC_iLoad_m, IIC_iLoad_mu, 1>;

let mayStore = 1, hasExtraSrcRegAllocReq = 1 in
defm t2STM : thumb2_ldst_mult<"stm", IIC_iStore_m, IIC_iStore_mu, 0>;

} // neverHasSideEffects


//===----------------------------------------------------------------------===//
//  Move Instructions.
//

let neverHasSideEffects = 1 in
def t2MOVr : T2sTwoReg<(outs GPR:$Rd), (ins GPR:$Rm), IIC_iMOVr,
                   "mov", ".w\t$Rd, $Rm", []> {
  let Inst{31-27} = 0b11101;
  let Inst{26-25} = 0b01;
  let Inst{24-21} = 0b0010;
  let Inst{19-16} = 0b1111; // Rn
  let Inst{14-12} = 0b000;
  let Inst{7-4} = 0b0000;
}

// AddedComplexity to ensure isel tries t2MOVi before t2MOVi16.
let isReMaterializable = 1, isAsCheapAsAMove = 1, isMoveImm = 1,
    AddedComplexity = 1 in
def t2MOVi : T2sOneRegImm<(outs rGPR:$Rd), (ins t2_so_imm:$imm), IIC_iMOVi,
                   "mov", ".w\t$Rd, $imm",
                   [(set rGPR:$Rd, t2_so_imm:$imm)]> {
  let Inst{31-27} = 0b11110;
  let Inst{25} = 0;
  let Inst{24-21} = 0b0010;
  let Inst{19-16} = 0b1111; // Rn
  let Inst{15} = 0;
}

let isReMaterializable = 1, isAsCheapAsAMove = 1, isMoveImm = 1 in
def t2MOVi16 : T2I<(outs rGPR:$Rd), (ins i32imm_hilo16:$imm), IIC_iMOVi,
                   "movw", "\t$Rd, $imm",
                   [(set rGPR:$Rd, imm0_65535:$imm)]> {
  let Inst{31-27} = 0b11110;
  let Inst{25} = 1;
  let Inst{24-21} = 0b0010;
  let Inst{20} = 0; // The S bit.
  let Inst{15} = 0;

  bits<4> Rd;
  bits<16> imm;

  let Inst{11-8}  = Rd;
  let Inst{19-16} = imm{15-12};
  let Inst{26}    = imm{11};
  let Inst{14-12} = imm{10-8};
  let Inst{7-0}   = imm{7-0};
}

def t2MOVi16_ga_pcrel : PseudoInst<(outs rGPR:$Rd),
                                (ins i32imm:$addr, pclabel:$id), IIC_iMOVi, []>;

let Constraints = "$src = $Rd" in {
def t2MOVTi16 : T2I<(outs rGPR:$Rd),
                    (ins rGPR:$src, i32imm_hilo16:$imm), IIC_iMOVi,
                    "movt", "\t$Rd, $imm",
                    [(set rGPR:$Rd,
                          (or (and rGPR:$src, 0xffff), lo16AllZero:$imm))]> {
  let Inst{31-27} = 0b11110;
  let Inst{25} = 1;
  let Inst{24-21} = 0b0110;
  let Inst{20} = 0; // The S bit.
  let Inst{15} = 0;

  bits<4> Rd;
  bits<16> imm;

  let Inst{11-8}  = Rd;
  let Inst{19-16} = imm{15-12};
  let Inst{26}    = imm{11};
  let Inst{14-12} = imm{10-8};
  let Inst{7-0}   = imm{7-0};
}

def t2MOVTi16_ga_pcrel : PseudoInst<(outs rGPR:$Rd),
                     (ins rGPR:$src, i32imm:$addr, pclabel:$id), IIC_iMOVi, []>;
} // Constraints

def : T2Pat<(or rGPR:$src, 0xffff0000), (t2MOVTi16 rGPR:$src, 0xffff)>;

//===----------------------------------------------------------------------===//
//  Extend Instructions.
//

// Sign extenders

defm t2SXTB  : T2I_ext_rrot<0b100, "sxtb",
                              UnOpFrag<(sext_inreg node:$Src, i8)>>;
defm t2SXTH  : T2I_ext_rrot<0b000, "sxth",
                              UnOpFrag<(sext_inreg node:$Src, i16)>>;
defm t2SXTB16 : T2I_ext_rrot_sxtb16<0b010, "sxtb16">;

defm t2SXTAB : T2I_exta_rrot<0b100, "sxtab",
                        BinOpFrag<(add node:$LHS, (sext_inreg node:$RHS, i8))>>;
defm t2SXTAH : T2I_exta_rrot<0b000, "sxtah",
                        BinOpFrag<(add node:$LHS, (sext_inreg node:$RHS,i16))>>;
defm t2SXTAB16 : T2I_exta_rrot_DO<0b010, "sxtab16">;

// TODO: SXT(A){B|H}16 - done for disassembly only

// Zero extenders

let AddedComplexity = 16 in {
defm t2UXTB   : T2I_ext_rrot<0b101, "uxtb",
                               UnOpFrag<(and node:$Src, 0x000000FF)>>;
defm t2UXTH   : T2I_ext_rrot<0b001, "uxth",
                               UnOpFrag<(and node:$Src, 0x0000FFFF)>>;
defm t2UXTB16 : T2I_ext_rrot_uxtb16<0b011, "uxtb16",
                               UnOpFrag<(and node:$Src, 0x00FF00FF)>>;

// FIXME: This pattern incorrectly assumes the shl operator is a rotate.
//        The transformation should probably be done as a combiner action
//        instead so we can include a check for masking back in the upper
//        eight bits of the source into the lower eight bits of the result.
//def : T2Pat<(and (shl rGPR:$Src, (i32 8)), 0xFF00FF),
//            (t2UXTB16r_rot rGPR:$Src, 24)>,
//          Requires<[HasT2ExtractPack, IsThumb2]>;
def : T2Pat<(and (srl rGPR:$Src, (i32 8)), 0xFF00FF),
            (t2UXTB16r_rot rGPR:$Src, 8)>,
        Requires<[HasT2ExtractPack, IsThumb2]>;

defm t2UXTAB : T2I_exta_rrot<0b101, "uxtab",
                           BinOpFrag<(add node:$LHS, (and node:$RHS, 0x00FF))>>;
defm t2UXTAH : T2I_exta_rrot<0b001, "uxtah",
                           BinOpFrag<(add node:$LHS, (and node:$RHS, 0xFFFF))>>;
defm t2UXTAB16 : T2I_exta_rrot_DO<0b011, "uxtab16">;
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
                             IIC_iALUi, IIC_iALUr, IIC_iALUsi,
                             BinOpFrag<(addc node:$LHS, node:$RHS)>, 1>;
defm t2SUBS : T2I_bin_s_irs <0b1101, "sub",
                             IIC_iALUi, IIC_iALUr, IIC_iALUsi,
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

def t2SEL : T2ThreeReg<(outs GPR:$Rd), (ins GPR:$Rn, GPR:$Rm),
                NoItinerary, "sel", "\t$Rd, $Rn, $Rm", []> {
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
              list<dag> pat = [/* For disassembly only; pattern left blank */],
              dag iops = (ins rGPR:$Rn, rGPR:$Rm),
              string asm = "\t$Rd, $Rn, $Rm">
  : T2I<(outs rGPR:$Rd), iops, NoItinerary, opc, asm, pat> {
  let Inst{31-27} = 0b11111;
  let Inst{26-23} = 0b0101;
  let Inst{22-20} = op22_20;
  let Inst{15-12} = 0b1111;
  let Inst{7-4} = op7_4;

  bits<4> Rd;
  bits<4> Rn;
  bits<4> Rm;

  let Inst{11-8}  = Rd;
  let Inst{19-16} = Rn;
  let Inst{3-0}   = Rm;
}

// Saturating add/subtract -- for disassembly only

def t2QADD    : T2I_pam<0b000, 0b1000, "qadd",
                        [(set rGPR:$Rd, (int_arm_qadd rGPR:$Rn, rGPR:$Rm))],
                        (ins rGPR:$Rm, rGPR:$Rn), "\t$Rd, $Rm, $Rn">;
def t2QADD16  : T2I_pam<0b001, 0b0001, "qadd16">;
def t2QADD8   : T2I_pam<0b000, 0b0001, "qadd8">;
def t2QASX    : T2I_pam<0b010, 0b0001, "qasx">;
def t2QDADD   : T2I_pam<0b000, 0b1001, "qdadd", [],
                        (ins rGPR:$Rm, rGPR:$Rn), "\t$Rd, $Rm, $Rn">;
def t2QDSUB   : T2I_pam<0b000, 0b1011, "qdsub", [],
                        (ins rGPR:$Rm, rGPR:$Rn), "\t$Rd, $Rm, $Rn">;
def t2QSAX    : T2I_pam<0b110, 0b0001, "qsax">;
def t2QSUB    : T2I_pam<0b000, 0b1010, "qsub",
                        [(set rGPR:$Rd, (int_arm_qsub rGPR:$Rn, rGPR:$Rm))],
                        (ins rGPR:$Rm, rGPR:$Rn), "\t$Rd, $Rm, $Rn">;
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

// Helper class for disassembly only
// A6.3.16 & A6.3.17
// T2Imac - Thumb2 multiply [accumulate, and absolute difference] instructions.
class T2ThreeReg_mac<bit long, bits<3> op22_20, bits<4> op7_4, dag oops,
  dag iops, InstrItinClass itin, string opc, string asm, list<dag> pattern>
  : T2ThreeReg<oops, iops, itin, opc, asm, pattern> {
  let Inst{31-27} = 0b11111;
  let Inst{26-24} = 0b011;
  let Inst{23}    = long;
  let Inst{22-20} = op22_20;
  let Inst{7-4}   = op7_4;
}

class T2FourReg_mac<bit long, bits<3> op22_20, bits<4> op7_4, dag oops,
  dag iops, InstrItinClass itin, string opc, string asm, list<dag> pattern>
  : T2FourReg<oops, iops, itin, opc, asm, pattern> {
  let Inst{31-27} = 0b11111;
  let Inst{26-24} = 0b011;
  let Inst{23}    = long;
  let Inst{22-20} = op22_20;
  let Inst{7-4}   = op7_4;
}

// Unsigned Sum of Absolute Differences [and Accumulate] -- for disassembly only

def t2USAD8   : T2ThreeReg_mac<0, 0b111, 0b0000, (outs rGPR:$Rd),
                                           (ins rGPR:$Rn, rGPR:$Rm),
                        NoItinerary, "usad8", "\t$Rd, $Rn, $Rm", []> {
  let Inst{15-12} = 0b1111;
}
def t2USADA8  : T2FourReg_mac<0, 0b111, 0b0000, (outs rGPR:$Rd),
                       (ins rGPR:$Rn, rGPR:$Rm, rGPR:$Ra), NoItinerary,
                        "usada8", "\t$Rd, $Rn, $Rm, $Ra", []>;

// Signed/Unsigned saturate -- for disassembly only

class T2SatI<dag oops, dag iops, InstrItinClass itin,
           string opc, string asm, list<dag> pattern>
  : T2I<oops, iops, itin, opc, asm, pattern> {
  bits<4> Rd;
  bits<4> Rn;
  bits<5> sat_imm;
  bits<7> sh;

  let Inst{11-8}  = Rd;
  let Inst{19-16} = Rn;
  let Inst{4-0}   = sat_imm{4-0};
  let Inst{21}    = sh{6};
  let Inst{14-12} = sh{4-2};
  let Inst{7-6}   = sh{1-0};
}

def t2SSAT: T2SatI<
                (outs rGPR:$Rd), (ins i32imm:$sat_imm, rGPR:$Rn, shift_imm:$sh),
                NoItinerary, "ssat", "\t$Rd, $sat_imm, $Rn$sh",
                [/* For disassembly only; pattern left blank */]> {
  let Inst{31-27} = 0b11110;
  let Inst{25-22} = 0b1100;
  let Inst{20} = 0;
  let Inst{15} = 0;
}

def t2SSAT16: T2SatI<
                   (outs rGPR:$Rd), (ins i32imm:$sat_imm, rGPR:$Rn), NoItinerary,
                   "ssat16", "\t$Rd, $sat_imm, $Rn",
                   [/* For disassembly only; pattern left blank */]> {
  let Inst{31-27} = 0b11110;
  let Inst{25-22} = 0b1100;
  let Inst{20} = 0;
  let Inst{15} = 0;
  let Inst{21} = 1;        // sh = '1'
  let Inst{14-12} = 0b000; // imm3 = '000'
  let Inst{7-6} = 0b00;    // imm2 = '00'
}

def t2USAT: T2SatI<
                (outs rGPR:$Rd), (ins i32imm:$sat_imm, rGPR:$Rn, shift_imm:$sh),
                NoItinerary, "usat", "\t$Rd, $sat_imm, $Rn$sh",
                [/* For disassembly only; pattern left blank */]> {
  let Inst{31-27} = 0b11110;
  let Inst{25-22} = 0b1110;
  let Inst{20} = 0;
  let Inst{15} = 0;
}

def t2USAT16: T2SatI<
                    (outs rGPR:$dst), (ins i32imm:$sat_imm, rGPR:$Rn), NoItinerary,
                   "usat16", "\t$dst, $sat_imm, $Rn",
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
def t2RRX : T2sTwoReg<(outs rGPR:$Rd), (ins rGPR:$Rm), IIC_iMOVsi,
                   "rrx", "\t$Rd, $Rm",
                   [(set rGPR:$Rd, (ARMrrx rGPR:$Rm))]> {
  let Inst{31-27} = 0b11101;
  let Inst{26-25} = 0b01;
  let Inst{24-21} = 0b0010;
  let Inst{19-16} = 0b1111; // Rn
  let Inst{14-12} = 0b000;
  let Inst{7-4} = 0b0011;
}
}

let isCodeGenOnly = 1, Defs = [CPSR] in {
def t2MOVsrl_flag : T2TwoRegShiftImm<
                        (outs rGPR:$Rd), (ins rGPR:$Rm), IIC_iMOVsi,
                        "lsrs", ".w\t$Rd, $Rm, #1",
                        [(set rGPR:$Rd, (ARMsrl_flag rGPR:$Rm))]> {
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
def t2MOVsra_flag : T2TwoRegShiftImm<
                        (outs rGPR:$Rd), (ins rGPR:$Rm), IIC_iMOVsi,
                        "asrs", ".w\t$Rd, $Rm, #1",
                        [(set rGPR:$Rd, (ARMsra_flag rGPR:$Rm))]> {
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
                            IIC_iBITi, IIC_iBITr, IIC_iBITsi,
                            BinOpFrag<(and node:$LHS, node:$RHS)>, 1>;
defm t2ORR  : T2I_bin_w_irs<0b0010, "orr",
                            IIC_iBITi, IIC_iBITr, IIC_iBITsi,
                            BinOpFrag<(or  node:$LHS, node:$RHS)>, 1>;
defm t2EOR  : T2I_bin_w_irs<0b0100, "eor",
                            IIC_iBITi, IIC_iBITr, IIC_iBITsi,
                            BinOpFrag<(xor node:$LHS, node:$RHS)>, 1>;

defm t2BIC  : T2I_bin_w_irs<0b0001, "bic",
                            IIC_iBITi, IIC_iBITr, IIC_iBITsi,
                            BinOpFrag<(and node:$LHS, (not node:$RHS))>>;

class T2BitFI<dag oops, dag iops, InstrItinClass itin,
              string opc, string asm, list<dag> pattern>
    : T2I<oops, iops, itin, opc, asm, pattern> {
  bits<4> Rd;
  bits<5> msb;
  bits<5> lsb;

  let Inst{11-8}  = Rd;
  let Inst{4-0}   = msb{4-0};
  let Inst{14-12} = lsb{4-2};
  let Inst{7-6}   = lsb{1-0};
}

class T2TwoRegBitFI<dag oops, dag iops, InstrItinClass itin,
              string opc, string asm, list<dag> pattern>
    : T2BitFI<oops, iops, itin, opc, asm, pattern> {
  bits<4> Rn;

  let Inst{19-16} = Rn;
}

let Constraints = "$src = $Rd" in
def t2BFC : T2BitFI<(outs rGPR:$Rd), (ins rGPR:$src, bf_inv_mask_imm:$imm),
                IIC_iUNAsi, "bfc", "\t$Rd, $imm",
                [(set rGPR:$Rd, (and rGPR:$src, bf_inv_mask_imm:$imm))]> {
  let Inst{31-27} = 0b11110;
  let Inst{25} = 1;
  let Inst{24-20} = 0b10110;
  let Inst{19-16} = 0b1111; // Rn
  let Inst{15} = 0;

  bits<10> imm;
  let msb{4-0} = imm{9-5};
  let lsb{4-0} = imm{4-0};
}

def t2SBFX: T2TwoRegBitFI<
                (outs rGPR:$Rd), (ins rGPR:$Rn, imm0_31:$lsb, imm0_31_m1:$msb),
                 IIC_iUNAsi, "sbfx", "\t$Rd, $Rn, $lsb, $msb", []> {
  let Inst{31-27} = 0b11110;
  let Inst{25} = 1;
  let Inst{24-20} = 0b10100;
  let Inst{15} = 0;
}

def t2UBFX: T2TwoRegBitFI<
                (outs rGPR:$Rd), (ins rGPR:$Rn, imm0_31:$lsb, imm0_31_m1:$msb),
                 IIC_iUNAsi, "ubfx", "\t$Rd, $Rn, $lsb, $msb", []> {
  let Inst{31-27} = 0b11110;
  let Inst{25} = 1;
  let Inst{24-20} = 0b11100;
  let Inst{15} = 0;
}

// A8.6.18  BFI - Bitfield insert (Encoding T1)
let Constraints = "$src = $Rd" in {
  def t2BFI : T2TwoRegBitFI<(outs rGPR:$Rd),
                  (ins rGPR:$src, rGPR:$Rn, bf_inv_mask_imm:$imm),
                  IIC_iBITi, "bfi", "\t$Rd, $Rn, $imm",
                  [(set rGPR:$Rd, (ARMbfi rGPR:$src, rGPR:$Rn,
                                   bf_inv_mask_imm:$imm))]> {
    let Inst{31-27} = 0b11110;
    let Inst{25} = 1;
    let Inst{24-20} = 0b10110;
    let Inst{15} = 0;

    bits<10> imm;
    let msb{4-0} = imm{9-5};
    let lsb{4-0} = imm{4-0};
  }

  // GNU as only supports this form of bfi (w/ 4 arguments)
  let isAsmParserOnly = 1 in
  def t2BFI4p : T2TwoRegBitFI<(outs rGPR:$Rd),
                  (ins rGPR:$src, rGPR:$Rn, lsb_pos_imm:$lsbit,
                       width_imm:$width),
                  IIC_iBITi, "bfi", "\t$Rd, $Rn, $lsbit, $width",
                  []> {
    let Inst{31-27} = 0b11110;
    let Inst{25} = 1;
    let Inst{24-20} = 0b10110;
    let Inst{15} = 0;

    bits<5> lsbit;
    bits<5> width;
    let msb{4-0} = width; // Custom encoder => lsb+width-1
    let lsb{4-0} = lsbit;
  }
}

defm t2ORN  : T2I_bin_irs<0b0011, "orn",
                          IIC_iBITi, IIC_iBITr, IIC_iBITsi,
                          BinOpFrag<(or  node:$LHS, (not node:$RHS))>, 0, "">;

// Prefer over of t2EORri ra, rb, -1 because mvn has 16-bit version
let AddedComplexity = 1 in
defm t2MVN  : T2I_un_irs <0b0011, "mvn",
                          IIC_iMVNi, IIC_iMVNr, IIC_iMVNsi,
                          UnOpFrag<(not node:$Src)>, 1, 1>;


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
def t2MUL: T2ThreeReg<(outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm), IIC_iMUL32,
                "mul", "\t$Rd, $Rn, $Rm",
                [(set rGPR:$Rd, (mul rGPR:$Rn, rGPR:$Rm))]> {
  let Inst{31-27} = 0b11111;
  let Inst{26-23} = 0b0110;
  let Inst{22-20} = 0b000;
  let Inst{15-12} = 0b1111; // Ra = 0b1111 (no accumulate)
  let Inst{7-4} = 0b0000; // Multiply
}

def t2MLA: T2FourReg<
                (outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm, rGPR:$Ra), IIC_iMAC32,
                "mla", "\t$Rd, $Rn, $Rm, $Ra",
                [(set rGPR:$Rd, (add (mul rGPR:$Rn, rGPR:$Rm), rGPR:$Ra))]> {
  let Inst{31-27} = 0b11111;
  let Inst{26-23} = 0b0110;
  let Inst{22-20} = 0b000;
  let Inst{7-4} = 0b0000; // Multiply
}

def t2MLS: T2FourReg<
                (outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm, rGPR:$Ra), IIC_iMAC32,
                "mls", "\t$Rd, $Rn, $Rm, $Ra",
                [(set rGPR:$Rd, (sub rGPR:$Ra, (mul rGPR:$Rn, rGPR:$Rm)))]> {
  let Inst{31-27} = 0b11111;
  let Inst{26-23} = 0b0110;
  let Inst{22-20} = 0b000;
  let Inst{7-4} = 0b0001; // Multiply and Subtract
}

// Extra precision multiplies with low / high results
let neverHasSideEffects = 1 in {
let isCommutable = 1 in {
def t2SMULL : T2MulLong<0b000, 0b0000,
                  (outs rGPR:$Rd, rGPR:$Ra),
                  (ins rGPR:$Rn, rGPR:$Rm), IIC_iMUL64,
                   "smull", "\t$Rd, $Ra, $Rn, $Rm", []>;

def t2UMULL : T2MulLong<0b010, 0b0000,
                  (outs rGPR:$RdLo, rGPR:$RdHi),
                  (ins rGPR:$Rn, rGPR:$Rm), IIC_iMUL64,
                   "umull", "\t$RdLo, $RdHi, $Rn, $Rm", []>;
} // isCommutable

// Multiply + accumulate
def t2SMLAL : T2MulLong<0b100, 0b0000,
                  (outs rGPR:$RdLo, rGPR:$RdHi),
                  (ins rGPR:$Rn, rGPR:$Rm), IIC_iMAC64,
                  "smlal", "\t$RdLo, $RdHi, $Rn, $Rm", []>;

def t2UMLAL : T2MulLong<0b110, 0b0000,
                  (outs rGPR:$RdLo, rGPR:$RdHi),
                  (ins rGPR:$Rn, rGPR:$Rm), IIC_iMAC64,
                  "umlal", "\t$RdLo, $RdHi, $Rn, $Rm", []>;

def t2UMAAL : T2MulLong<0b110, 0b0110,
                  (outs rGPR:$RdLo, rGPR:$RdHi),
                  (ins rGPR:$Rn, rGPR:$Rm), IIC_iMAC64,
                  "umaal", "\t$RdLo, $RdHi, $Rn, $Rm", []>;
} // neverHasSideEffects

// Rounding variants of the below included for disassembly only

// Most significant word multiply
def t2SMMUL : T2ThreeReg<(outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm), IIC_iMUL32,
                  "smmul", "\t$Rd, $Rn, $Rm",
                  [(set rGPR:$Rd, (mulhs rGPR:$Rn, rGPR:$Rm))]> {
  let Inst{31-27} = 0b11111;
  let Inst{26-23} = 0b0110;
  let Inst{22-20} = 0b101;
  let Inst{15-12} = 0b1111; // Ra = 0b1111 (no accumulate)
  let Inst{7-4} = 0b0000; // No Rounding (Inst{4} = 0)
}

def t2SMMULR : T2ThreeReg<(outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm), IIC_iMUL32,
                  "smmulr", "\t$Rd, $Rn, $Rm", []> {
  let Inst{31-27} = 0b11111;
  let Inst{26-23} = 0b0110;
  let Inst{22-20} = 0b101;
  let Inst{15-12} = 0b1111; // Ra = 0b1111 (no accumulate)
  let Inst{7-4} = 0b0001; // Rounding (Inst{4} = 1)
}

def t2SMMLA : T2FourReg<
        (outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm, rGPR:$Ra), IIC_iMAC32,
                "smmla", "\t$Rd, $Rn, $Rm, $Ra",
                [(set rGPR:$Rd, (add (mulhs rGPR:$Rm, rGPR:$Rn), rGPR:$Ra))]> {
  let Inst{31-27} = 0b11111;
  let Inst{26-23} = 0b0110;
  let Inst{22-20} = 0b101;
  let Inst{7-4} = 0b0000; // No Rounding (Inst{4} = 0)
}

def t2SMMLAR: T2FourReg<
        (outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm, rGPR:$Ra), IIC_iMAC32,
                  "smmlar", "\t$Rd, $Rn, $Rm, $Ra", []> {
  let Inst{31-27} = 0b11111;
  let Inst{26-23} = 0b0110;
  let Inst{22-20} = 0b101;
  let Inst{7-4} = 0b0001; // Rounding (Inst{4} = 1)
}

def t2SMMLS: T2FourReg<
        (outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm, rGPR:$Ra), IIC_iMAC32,
                "smmls", "\t$Rd, $Rn, $Rm, $Ra",
                [(set rGPR:$Rd, (sub rGPR:$Ra, (mulhs rGPR:$Rn, rGPR:$Rm)))]> {
  let Inst{31-27} = 0b11111;
  let Inst{26-23} = 0b0110;
  let Inst{22-20} = 0b110;
  let Inst{7-4} = 0b0000; // No Rounding (Inst{4} = 0)
}

def t2SMMLSR:T2FourReg<
        (outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm, rGPR:$Ra), IIC_iMAC32,
                "smmlsr", "\t$Rd, $Rn, $Rm, $Ra", []> {
  let Inst{31-27} = 0b11111;
  let Inst{26-23} = 0b0110;
  let Inst{22-20} = 0b110;
  let Inst{7-4} = 0b0001; // Rounding (Inst{4} = 1)
}

multiclass T2I_smul<string opc, PatFrag opnode> {
  def BB : T2ThreeReg<(outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm), IIC_iMUL16,
              !strconcat(opc, "bb"), "\t$Rd, $Rn, $Rm",
              [(set rGPR:$Rd, (opnode (sext_inreg rGPR:$Rn, i16),
                                      (sext_inreg rGPR:$Rm, i16)))]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0110;
    let Inst{22-20} = 0b001;
    let Inst{15-12} = 0b1111; // Ra = 0b1111 (no accumulate)
    let Inst{7-6} = 0b00;
    let Inst{5-4} = 0b00;
  }

  def BT : T2ThreeReg<(outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm), IIC_iMUL16,
              !strconcat(opc, "bt"), "\t$Rd, $Rn, $Rm",
              [(set rGPR:$Rd, (opnode (sext_inreg rGPR:$Rn, i16),
                                      (sra rGPR:$Rm, (i32 16))))]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0110;
    let Inst{22-20} = 0b001;
    let Inst{15-12} = 0b1111; // Ra = 0b1111 (no accumulate)
    let Inst{7-6} = 0b00;
    let Inst{5-4} = 0b01;
  }

  def TB : T2ThreeReg<(outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm), IIC_iMUL16,
              !strconcat(opc, "tb"), "\t$Rd, $Rn, $Rm",
              [(set rGPR:$Rd, (opnode (sra rGPR:$Rn, (i32 16)),
                                      (sext_inreg rGPR:$Rm, i16)))]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0110;
    let Inst{22-20} = 0b001;
    let Inst{15-12} = 0b1111; // Ra = 0b1111 (no accumulate)
    let Inst{7-6} = 0b00;
    let Inst{5-4} = 0b10;
  }

  def TT : T2ThreeReg<(outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm), IIC_iMUL16,
              !strconcat(opc, "tt"), "\t$Rd, $Rn, $Rm",
              [(set rGPR:$Rd, (opnode (sra rGPR:$Rn, (i32 16)),
                                      (sra rGPR:$Rm, (i32 16))))]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0110;
    let Inst{22-20} = 0b001;
    let Inst{15-12} = 0b1111; // Ra = 0b1111 (no accumulate)
    let Inst{7-6} = 0b00;
    let Inst{5-4} = 0b11;
  }

  def WB : T2ThreeReg<(outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm), IIC_iMUL16,
              !strconcat(opc, "wb"), "\t$Rd, $Rn, $Rm",
              [(set rGPR:$Rd, (sra (opnode rGPR:$Rn,
                                    (sext_inreg rGPR:$Rm, i16)), (i32 16)))]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0110;
    let Inst{22-20} = 0b011;
    let Inst{15-12} = 0b1111; // Ra = 0b1111 (no accumulate)
    let Inst{7-6} = 0b00;
    let Inst{5-4} = 0b00;
  }

  def WT : T2ThreeReg<(outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm), IIC_iMUL16,
              !strconcat(opc, "wt"), "\t$Rd, $Rn, $Rm",
              [(set rGPR:$Rd, (sra (opnode rGPR:$Rn,
                                    (sra rGPR:$Rm, (i32 16))), (i32 16)))]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0110;
    let Inst{22-20} = 0b011;
    let Inst{15-12} = 0b1111; // Ra = 0b1111 (no accumulate)
    let Inst{7-6} = 0b00;
    let Inst{5-4} = 0b01;
  }
}


multiclass T2I_smla<string opc, PatFrag opnode> {
  def BB : T2FourReg<
        (outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm, rGPR:$Ra), IIC_iMAC16,
              !strconcat(opc, "bb"), "\t$Rd, $Rn, $Rm, $Ra",
              [(set rGPR:$Rd, (add rGPR:$Ra,
                               (opnode (sext_inreg rGPR:$Rn, i16),
                                       (sext_inreg rGPR:$Rm, i16))))]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0110;
    let Inst{22-20} = 0b001;
    let Inst{7-6} = 0b00;
    let Inst{5-4} = 0b00;
  }

  def BT : T2FourReg<
       (outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm, rGPR:$Ra), IIC_iMAC16,
             !strconcat(opc, "bt"), "\t$Rd, $Rn, $Rm, $Ra",
             [(set rGPR:$Rd, (add rGPR:$Ra, (opnode (sext_inreg rGPR:$Rn, i16),
                                                 (sra rGPR:$Rm, (i32 16)))))]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0110;
    let Inst{22-20} = 0b001;
    let Inst{7-6} = 0b00;
    let Inst{5-4} = 0b01;
  }

  def TB : T2FourReg<
        (outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm, rGPR:$Ra), IIC_iMAC16,
              !strconcat(opc, "tb"), "\t$Rd, $Rn, $Rm, $Ra",
              [(set rGPR:$Rd, (add rGPR:$Ra, (opnode (sra rGPR:$Rn, (i32 16)),
                                               (sext_inreg rGPR:$Rm, i16))))]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0110;
    let Inst{22-20} = 0b001;
    let Inst{7-6} = 0b00;
    let Inst{5-4} = 0b10;
  }

  def TT : T2FourReg<
        (outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm, rGPR:$Ra), IIC_iMAC16,
              !strconcat(opc, "tt"), "\t$Rd, $Rn, $Rm, $Ra",
             [(set rGPR:$Rd, (add rGPR:$Ra, (opnode (sra rGPR:$Rn, (i32 16)),
                                                 (sra rGPR:$Rm, (i32 16)))))]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0110;
    let Inst{22-20} = 0b001;
    let Inst{7-6} = 0b00;
    let Inst{5-4} = 0b11;
  }

  def WB : T2FourReg<
        (outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm, rGPR:$Ra), IIC_iMAC16,
              !strconcat(opc, "wb"), "\t$Rd, $Rn, $Rm, $Ra",
              [(set rGPR:$Rd, (add rGPR:$Ra, (sra (opnode rGPR:$Rn,
                                    (sext_inreg rGPR:$Rm, i16)), (i32 16))))]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0110;
    let Inst{22-20} = 0b011;
    let Inst{7-6} = 0b00;
    let Inst{5-4} = 0b00;
  }

  def WT : T2FourReg<
        (outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm, rGPR:$Ra), IIC_iMAC16,
              !strconcat(opc, "wt"), "\t$Rd, $Rn, $Rm, $Ra",
              [(set rGPR:$Rd, (add rGPR:$Ra, (sra (opnode rGPR:$Rn,
                                      (sra rGPR:$Rm, (i32 16))), (i32 16))))]> {
    let Inst{31-27} = 0b11111;
    let Inst{26-23} = 0b0110;
    let Inst{22-20} = 0b011;
    let Inst{7-6} = 0b00;
    let Inst{5-4} = 0b01;
  }
}

defm t2SMUL : T2I_smul<"smul", BinOpFrag<(mul node:$LHS, node:$RHS)>>;
defm t2SMLA : T2I_smla<"smla", BinOpFrag<(mul node:$LHS, node:$RHS)>>;

// Halfword multiple accumulate long: SMLAL<x><y> -- for disassembly only
def t2SMLALBB : T2FourReg_mac<1, 0b100, 0b1000, (outs rGPR:$Ra,rGPR:$Rd),
         (ins rGPR:$Rn,rGPR:$Rm), IIC_iMAC64, "smlalbb", "\t$Ra, $Rd, $Rn, $Rm",
           [/* For disassembly only; pattern left blank */]>;
def t2SMLALBT : T2FourReg_mac<1, 0b100, 0b1001, (outs rGPR:$Ra,rGPR:$Rd),
         (ins rGPR:$Rn,rGPR:$Rm), IIC_iMAC64, "smlalbt", "\t$Ra, $Rd, $Rn, $Rm",
           [/* For disassembly only; pattern left blank */]>;
def t2SMLALTB : T2FourReg_mac<1, 0b100, 0b1010, (outs rGPR:$Ra,rGPR:$Rd),
         (ins rGPR:$Rn,rGPR:$Rm), IIC_iMAC64, "smlaltb", "\t$Ra, $Rd, $Rn, $Rm",
           [/* For disassembly only; pattern left blank */]>;
def t2SMLALTT : T2FourReg_mac<1, 0b100, 0b1011, (outs rGPR:$Ra,rGPR:$Rd),
         (ins rGPR:$Rn,rGPR:$Rm), IIC_iMAC64, "smlaltt", "\t$Ra, $Rd, $Rn, $Rm",
           [/* For disassembly only; pattern left blank */]>;

// Dual halfword multiple: SMUAD, SMUSD, SMLAD, SMLSD, SMLALD, SMLSLD
// These are for disassembly only.

def t2SMUAD: T2ThreeReg_mac<
            0, 0b010, 0b0000, (outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm),
            IIC_iMAC32, "smuad", "\t$Rd, $Rn, $Rm", []> {
  let Inst{15-12} = 0b1111;
}
def t2SMUADX:T2ThreeReg_mac<
            0, 0b010, 0b0001, (outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm),
            IIC_iMAC32, "smuadx", "\t$Rd, $Rn, $Rm", []> {
  let Inst{15-12} = 0b1111;
}
def t2SMUSD: T2ThreeReg_mac<
            0, 0b100, 0b0000, (outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm),
            IIC_iMAC32, "smusd", "\t$Rd, $Rn, $Rm", []> {
  let Inst{15-12} = 0b1111;
}
def t2SMUSDX:T2ThreeReg_mac<
            0, 0b100, 0b0001, (outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm),
            IIC_iMAC32, "smusdx", "\t$Rd, $Rn, $Rm", []> {
  let Inst{15-12} = 0b1111;
}
def t2SMLAD   : T2ThreeReg_mac<
            0, 0b010, 0b0000, (outs rGPR:$Rd),
            (ins rGPR:$Rn, rGPR:$Rm, rGPR:$Ra), IIC_iMAC32, "smlad",
            "\t$Rd, $Rn, $Rm, $Ra", []>;
def t2SMLADX  : T2FourReg_mac<
            0, 0b010, 0b0001, (outs rGPR:$Rd),
            (ins rGPR:$Rn, rGPR:$Rm, rGPR:$Ra), IIC_iMAC32, "smladx",
            "\t$Rd, $Rn, $Rm, $Ra", []>;
def t2SMLSD   : T2FourReg_mac<0, 0b100, 0b0000, (outs rGPR:$Rd),
            (ins rGPR:$Rn, rGPR:$Rm, rGPR:$Ra), IIC_iMAC32, "smlsd",
            "\t$Rd, $Rn, $Rm, $Ra", []>;
def t2SMLSDX  : T2FourReg_mac<0, 0b100, 0b0001, (outs rGPR:$Rd),
            (ins rGPR:$Rn, rGPR:$Rm, rGPR:$Ra), IIC_iMAC32, "smlsdx",
            "\t$Rd, $Rn, $Rm, $Ra", []>;
def t2SMLALD  : T2FourReg_mac<1, 0b100, 0b1100, (outs rGPR:$Ra,rGPR:$Rd),
                        (ins rGPR:$Rm, rGPR:$Rn), IIC_iMAC64, "smlald",
                        "\t$Ra, $Rd, $Rm, $Rn", []>;
def t2SMLALDX : T2FourReg_mac<1, 0b100, 0b1101, (outs rGPR:$Ra,rGPR:$Rd),
                        (ins rGPR:$Rm,rGPR:$Rn), IIC_iMAC64, "smlaldx",
                        "\t$Ra, $Rd, $Rm, $Rn", []>;
def t2SMLSLD  : T2FourReg_mac<1, 0b101, 0b1100, (outs rGPR:$Ra,rGPR:$Rd),
                        (ins rGPR:$Rm,rGPR:$Rn), IIC_iMAC64, "smlsld",
                        "\t$Ra, $Rd, $Rm, $Rn", []>;
def t2SMLSLDX : T2FourReg_mac<1, 0b101, 0b1101, (outs rGPR:$Ra,rGPR:$Rd),
                        (ins rGPR:$Rm,rGPR:$Rn), IIC_iMAC64, "smlsldx",
                        "\t$Ra, $Rd, $Rm, $Rn", []>;

//===----------------------------------------------------------------------===//
//  Misc. Arithmetic Instructions.
//

class T2I_misc<bits<2> op1, bits<2> op2, dag oops, dag iops,
      InstrItinClass itin, string opc, string asm, list<dag> pattern>
  : T2ThreeReg<oops, iops, itin, opc, asm, pattern> {
  let Inst{31-27} = 0b11111;
  let Inst{26-22} = 0b01010;
  let Inst{21-20} = op1;
  let Inst{15-12} = 0b1111;
  let Inst{7-6} = 0b10;
  let Inst{5-4} = op2;
  let Rn{3-0} = Rm;
}

def t2CLZ : T2I_misc<0b11, 0b00, (outs rGPR:$Rd), (ins rGPR:$Rm), IIC_iUNAr,
                    "clz", "\t$Rd, $Rm", [(set rGPR:$Rd, (ctlz rGPR:$Rm))]>;

def t2RBIT : T2I_misc<0b01, 0b10, (outs rGPR:$Rd), (ins rGPR:$Rm), IIC_iUNAr,
                      "rbit", "\t$Rd, $Rm",
                      [(set rGPR:$Rd, (ARMrbit rGPR:$Rm))]>;

def t2REV : T2I_misc<0b01, 0b00, (outs rGPR:$Rd), (ins rGPR:$Rm), IIC_iUNAr,
                 "rev", ".w\t$Rd, $Rm", [(set rGPR:$Rd, (bswap rGPR:$Rm))]>;

def t2REV16 : T2I_misc<0b01, 0b01, (outs rGPR:$Rd), (ins rGPR:$Rm), IIC_iUNAr,
                       "rev16", ".w\t$Rd, $Rm",
                [(set rGPR:$Rd,
                    (or (and (srl rGPR:$Rm, (i32 8)), 0xFF),
                        (or (and (shl rGPR:$Rm, (i32 8)), 0xFF00),
                            (or (and (srl rGPR:$Rm, (i32 8)), 0xFF0000),
                               (and (shl rGPR:$Rm, (i32 8)), 0xFF000000)))))]>;

def t2REVSH : T2I_misc<0b01, 0b11, (outs rGPR:$Rd), (ins rGPR:$Rm), IIC_iUNAr,
                       "revsh", ".w\t$Rd, $Rm",
                 [(set rGPR:$Rd,
                    (sext_inreg
                      (or (srl (and rGPR:$Rm, 0xFF00), (i32 8)),
                          (shl rGPR:$Rm, (i32 8))), i16))]>;

def t2PKHBT : T2ThreeReg<
            (outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm, shift_imm:$sh),
                  IIC_iBITsi, "pkhbt", "\t$Rd, $Rn, $Rm$sh",
                  [(set rGPR:$Rd, (or (and rGPR:$Rn, 0xFFFF),
                                      (and (shl rGPR:$Rm, lsl_amt:$sh),
                                           0xFFFF0000)))]>,
                  Requires<[HasT2ExtractPack, IsThumb2]> {
  let Inst{31-27} = 0b11101;
  let Inst{26-25} = 0b01;
  let Inst{24-20} = 0b01100;
  let Inst{5} = 0; // BT form
  let Inst{4} = 0;

  bits<8> sh;
  let Inst{14-12} = sh{7-5};
  let Inst{7-6}   = sh{4-3};
}

// Alternate cases for PKHBT where identities eliminate some nodes.
def : T2Pat<(or (and rGPR:$src1, 0xFFFF), (and rGPR:$src2, 0xFFFF0000)),
            (t2PKHBT rGPR:$src1, rGPR:$src2, 0)>,
            Requires<[HasT2ExtractPack, IsThumb2]>;
def : T2Pat<(or (and rGPR:$src1, 0xFFFF), (shl rGPR:$src2, imm16_31:$sh)),
            (t2PKHBT rGPR:$src1, rGPR:$src2, (lsl_shift_imm imm16_31:$sh))>,
            Requires<[HasT2ExtractPack, IsThumb2]>;

// Note: Shifts of 1-15 bits will be transformed to srl instead of sra and
// will match the pattern below.
def t2PKHTB : T2ThreeReg<
                  (outs rGPR:$Rd), (ins rGPR:$Rn, rGPR:$Rm, shift_imm:$sh),
                  IIC_iBITsi, "pkhtb", "\t$Rd, $Rn, $Rm$sh",
                  [(set rGPR:$Rd, (or (and rGPR:$Rn, 0xFFFF0000),
                                       (and (sra rGPR:$Rm, asr_amt:$sh),
                                            0xFFFF)))]>,
                  Requires<[HasT2ExtractPack, IsThumb2]> {
  let Inst{31-27} = 0b11101;
  let Inst{26-25} = 0b01;
  let Inst{24-20} = 0b01100;
  let Inst{5} = 1; // TB form
  let Inst{4} = 0;

  bits<8> sh;
  let Inst{14-12} = sh{7-5};
  let Inst{7-6}   = sh{4-3};
}

// Alternate cases for PKHTB where identities eliminate some nodes.  Note that
// a shift amount of 0 is *not legal* here, it is PKHBT instead.
def : T2Pat<(or (and rGPR:$src1, 0xFFFF0000), (srl rGPR:$src2, imm16_31:$sh)),
            (t2PKHTB rGPR:$src1, rGPR:$src2, (asr_shift_imm imm16_31:$sh))>,
            Requires<[HasT2ExtractPack, IsThumb2]>;
def : T2Pat<(or (and rGPR:$src1, 0xFFFF0000),
                (and (srl rGPR:$src2, imm1_15:$sh), 0xFFFF)),
            (t2PKHTB rGPR:$src1, rGPR:$src2, (asr_shift_imm imm1_15:$sh))>,
            Requires<[HasT2ExtractPack, IsThumb2]>;

//===----------------------------------------------------------------------===//
//  Comparison Instructions...
//
defm t2CMP  : T2I_cmp_irs<0b1101, "cmp",
                          IIC_iCMPi, IIC_iCMPr, IIC_iCMPsi,
                          BinOpFrag<(ARMcmp node:$LHS, node:$RHS)>>;

def : T2Pat<(ARMcmpZ  GPR:$lhs, t2_so_imm:$imm),
            (t2CMPri  GPR:$lhs, t2_so_imm:$imm)>;
def : T2Pat<(ARMcmpZ  GPR:$lhs, rGPR:$rhs),
            (t2CMPrr  GPR:$lhs, rGPR:$rhs)>;
def : T2Pat<(ARMcmpZ  GPR:$lhs, t2_so_reg:$rhs),
            (t2CMPrs  GPR:$lhs, t2_so_reg:$rhs)>;

//FIXME: Disable CMN, as CCodes are backwards from compare expectations
//       Compare-to-zero still works out, just not the relationals
//defm t2CMN  : T2I_cmp_irs<0b1000, "cmn",
//                          BinOpFrag<(ARMcmp node:$LHS,(ineg node:$RHS))>>;
defm t2CMNz : T2I_cmp_irs<0b1000, "cmn",
                          IIC_iCMPi, IIC_iCMPr, IIC_iCMPsi,
                          BinOpFrag<(ARMcmpZ node:$LHS,(ineg node:$RHS))>>;

//def : T2Pat<(ARMcmp  GPR:$src, t2_so_imm_neg:$imm),
//            (t2CMNri GPR:$src, t2_so_imm_neg:$imm)>;

def : T2Pat<(ARMcmpZ  GPR:$src, t2_so_imm_neg:$imm),
            (t2CMNzri GPR:$src, t2_so_imm_neg:$imm)>;

defm t2TST  : T2I_cmp_irs<0b0000, "tst",
                          IIC_iTSTi, IIC_iTSTr, IIC_iTSTsi,
                         BinOpFrag<(ARMcmpZ (and_su node:$LHS, node:$RHS), 0)>>;
defm t2TEQ  : T2I_cmp_irs<0b0100, "teq",
                          IIC_iTSTi, IIC_iTSTr, IIC_iTSTsi,
                         BinOpFrag<(ARMcmpZ (xor_su node:$LHS, node:$RHS), 0)>>;

// Conditional moves
// FIXME: should be able to write a pattern for ARMcmov, but can't use
// a two-value operand where a dag node expects two operands. :(
let neverHasSideEffects = 1 in {
def t2MOVCCr : T2TwoReg<
                   (outs rGPR:$Rd), (ins rGPR:$false, rGPR:$Rm), IIC_iCMOVr,
                   "mov", ".w\t$Rd, $Rm",
   [/*(set rGPR:$Rd, (ARMcmov rGPR:$false, rGPR:$Rm, imm:$cc, CCR:$ccr))*/]>,
                RegConstraint<"$false = $Rd"> {
  let Inst{31-27} = 0b11101;
  let Inst{26-25} = 0b01;
  let Inst{24-21} = 0b0010;
  let Inst{20} = 0; // The S bit.
  let Inst{19-16} = 0b1111; // Rn
  let Inst{14-12} = 0b000;
  let Inst{7-4} = 0b0000;
}

let isMoveImm = 1 in
def t2MOVCCi : T2OneRegImm<(outs rGPR:$Rd), (ins rGPR:$false, t2_so_imm:$imm),
                   IIC_iCMOVi, "mov", ".w\t$Rd, $imm",
[/*(set rGPR:$Rd,(ARMcmov rGPR:$false,t2_so_imm:$imm, imm:$cc, CCR:$ccr))*/]>,
                   RegConstraint<"$false = $Rd"> {
  let Inst{31-27} = 0b11110;
  let Inst{25} = 0;
  let Inst{24-21} = 0b0010;
  let Inst{20} = 0; // The S bit.
  let Inst{19-16} = 0b1111; // Rn
  let Inst{15} = 0;
}

let isMoveImm = 1 in
def t2MOVCCi16 : T2I<(outs rGPR:$Rd), (ins rGPR:$false, i32imm_hilo16:$imm),
                      IIC_iCMOVi,
                      "movw", "\t$Rd, $imm", []>,
                      RegConstraint<"$false = $Rd"> {
  let Inst{31-27} = 0b11110;
  let Inst{25} = 1;
  let Inst{24-21} = 0b0010;
  let Inst{20} = 0; // The S bit.
  let Inst{15} = 0;

  bits<4> Rd;
  bits<16> imm;

  let Inst{11-8}  = Rd;
  let Inst{19-16} = imm{15-12};
  let Inst{26}    = imm{11};
  let Inst{14-12} = imm{10-8};
  let Inst{7-0}   = imm{7-0};
}

let isMoveImm = 1 in
def t2MOVCCi32imm : PseudoInst<(outs rGPR:$dst),
                               (ins rGPR:$false, i32imm:$src, pred:$p),
                    IIC_iCMOVix2, []>, RegConstraint<"$false = $dst">;

let isMoveImm = 1 in
def t2MVNCCi : T2OneRegImm<(outs rGPR:$Rd), (ins rGPR:$false, t2_so_imm:$imm),
                   IIC_iCMOVi, "mvn", ".w\t$Rd, $imm",
[/*(set rGPR:$Rd,(ARMcmov rGPR:$false,t2_so_imm_not:$imm,
                   imm:$cc, CCR:$ccr))*/]>,
                   RegConstraint<"$false = $Rd"> {
  let Inst{31-27} = 0b11110;
  let Inst{25} = 0;
  let Inst{24-21} = 0b0011;
  let Inst{20} = 0; // The S bit.
  let Inst{19-16} = 0b1111; // Rn
  let Inst{15} = 0;
}

class T2I_movcc_sh<bits<2> opcod, dag oops, dag iops, InstrItinClass itin,
                   string opc, string asm, list<dag> pattern>
  : T2TwoRegShiftImm<oops, iops, itin, opc, asm, pattern> {
  let Inst{31-27} = 0b11101;
  let Inst{26-25} = 0b01;
  let Inst{24-21} = 0b0010;
  let Inst{20} = 0; // The S bit.
  let Inst{19-16} = 0b1111; // Rn
  let Inst{5-4} = opcod; // Shift type.
}
def t2MOVCClsl : T2I_movcc_sh<0b00, (outs rGPR:$Rd),
                             (ins rGPR:$false, rGPR:$Rm, i32imm:$imm),
                             IIC_iCMOVsi, "lsl", ".w\t$Rd, $Rm, $imm", []>,
                 RegConstraint<"$false = $Rd">;
def t2MOVCClsr : T2I_movcc_sh<0b01, (outs rGPR:$Rd),
                             (ins rGPR:$false, rGPR:$Rm, i32imm:$imm),
                             IIC_iCMOVsi, "lsr", ".w\t$Rd, $Rm, $imm", []>,
                 RegConstraint<"$false = $Rd">;
def t2MOVCCasr : T2I_movcc_sh<0b10, (outs rGPR:$Rd),
                             (ins rGPR:$false, rGPR:$Rm, i32imm:$imm),
                             IIC_iCMOVsi, "asr", ".w\t$Rd, $Rm, $imm", []>,
                 RegConstraint<"$false = $Rd">;
def t2MOVCCror : T2I_movcc_sh<0b11, (outs rGPR:$Rd),
                             (ins rGPR:$false, rGPR:$Rm, i32imm:$imm),
                             IIC_iCMOVsi, "ror", ".w\t$Rd, $Rm, $imm", []>,
                 RegConstraint<"$false = $Rd">;
} // neverHasSideEffects

//===----------------------------------------------------------------------===//
// Atomic operations intrinsics
//

// memory barriers protect the atomic sequences
let hasSideEffects = 1 in {
def t2DMB : AInoP<(outs), (ins memb_opt:$opt), ThumbFrm, NoItinerary,
                  "dmb", "\t$opt", [(ARMMemBarrier (i32 imm:$opt))]>,
                  Requires<[IsThumb, HasDB]> {
  bits<4> opt;
  let Inst{31-4} = 0xf3bf8f5;
  let Inst{3-0} = opt;
}
}

def t2DSB : AInoP<(outs), (ins memb_opt:$opt), ThumbFrm, NoItinerary,
                  "dsb", "\t$opt",
                  [/* For disassembly only; pattern left blank */]>,
                  Requires<[IsThumb, HasDB]> {
  bits<4> opt;
  let Inst{31-4} = 0xf3bf8f4;
  let Inst{3-0} = opt;
}

// ISB has only full system option -- for disassembly only
def t2ISB : AInoP<(outs), (ins), ThumbFrm, NoItinerary, "isb", "",
                  [/* For disassembly only; pattern left blank */]>,
                  Requires<[IsThumb2, HasV7]> {
  let Inst{31-4} = 0xf3bf8f6;
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

  bits<4> Rn;
  bits<4> Rt;
  let Inst{19-16} = Rn;
  let Inst{15-12} = Rt;
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

  bits<4> Rd;
  bits<4> Rn;
  bits<4> Rt;
  let Inst{11-8}  = Rd;
  let Inst{19-16} = Rn;
  let Inst{15-12} = Rt;
}

let mayLoad = 1 in {
def t2LDREXB : T2I_ldrex<0b00, (outs rGPR:$Rt), (ins rGPR:$Rn), AddrModeNone,
                         Size4Bytes, NoItinerary, "ldrexb", "\t$Rt, [$Rn]",
                         "", []>;
def t2LDREXH : T2I_ldrex<0b01, (outs rGPR:$Rt), (ins rGPR:$Rn), AddrModeNone,
                         Size4Bytes, NoItinerary, "ldrexh", "\t$Rt, [$Rn]",
                         "", []>;
def t2LDREX  : Thumb2I<(outs rGPR:$Rt), (ins rGPR:$Rn), AddrModeNone,
                       Size4Bytes, NoItinerary,
                       "ldrex", "\t$Rt, [$Rn]", "",
                      []> {
  let Inst{31-27} = 0b11101;
  let Inst{26-20} = 0b0000101;
  let Inst{11-8} = 0b1111;
  let Inst{7-0} = 0b00000000; // imm8 = 0

  bits<4> Rn;
  bits<4> Rt;
  let Inst{19-16} = Rn;
  let Inst{15-12} = Rt;
}
def t2LDREXD : T2I_ldrex<0b11, (outs rGPR:$Rt, rGPR:$Rt2), (ins rGPR:$Rn),
                         AddrModeNone, Size4Bytes, NoItinerary,
                         "ldrexd", "\t$Rt, $Rt2, [$Rn]", "",
                         [], {?, ?, ?, ?}> {
  bits<4> Rt2;
  let Inst{11-8} = Rt2;
}
}

let mayStore = 1, Constraints = "@earlyclobber $Rd" in {
def t2STREXB : T2I_strex<0b00, (outs rGPR:$Rd), (ins rGPR:$Rt, rGPR:$Rn),
                         AddrModeNone, Size4Bytes, NoItinerary,
                         "strexb", "\t$Rd, $Rt, [$Rn]", "", []>;
def t2STREXH : T2I_strex<0b01, (outs rGPR:$Rd), (ins rGPR:$Rt, rGPR:$Rn),
                         AddrModeNone, Size4Bytes, NoItinerary,
                         "strexh", "\t$Rd, $Rt, [$Rn]", "", []>;
def t2STREX  : Thumb2I<(outs rGPR:$Rd), (ins rGPR:$Rt, rGPR:$Rn),
                       AddrModeNone, Size4Bytes, NoItinerary,
                       "strex", "\t$Rd, $Rt, [$Rn]", "",
                      []> {
  let Inst{31-27} = 0b11101;
  let Inst{26-20} = 0b0000100;
  let Inst{7-0} = 0b00000000; // imm8 = 0

  bits<4> Rd;
  bits<4> Rn;
  bits<4> Rt;
  let Inst{11-8}  = Rd;
  let Inst{19-16} = Rn;
  let Inst{15-12} = Rt;
}
def t2STREXD : T2I_strex<0b11, (outs rGPR:$Rd),
                         (ins rGPR:$Rt, rGPR:$Rt2, rGPR:$Rn),
                         AddrModeNone, Size4Bytes, NoItinerary,
                         "strexd", "\t$Rd, $Rt, $Rt2, [$Rn]", "", [],
                         {?, ?, ?, ?}> {
  bits<4> Rt2;
  let Inst{11-8} = Rt2;
}
}

// Clear-Exclusive is for disassembly only.
def t2CLREX : T2XI<(outs), (ins), NoItinerary, "clrex",
                   [/* For disassembly only; pattern left blank */]>,
            Requires<[IsThumb2, HasV7]>  {
  let Inst{31-16} = 0xf3bf;
  let Inst{15-14} = 0b10;
  let Inst{13} = 0;
  let Inst{12} = 0;
  let Inst{11-8} = 0b1111;
  let Inst{7-4} = 0b0010;
  let Inst{3-0} = 0b1111;
}

//===----------------------------------------------------------------------===//
// TLS Instructions
//

// __aeabi_read_tp preserves the registers r1-r3.
let isCall = 1,
  Defs = [R0, R12, LR, CPSR], Uses = [SP] in {
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
    D31 ], hasSideEffects = 1, isBarrier = 1, isCodeGenOnly = 1 in {
  def t2Int_eh_sjlj_setjmp : Thumb2XI<(outs), (ins tGPR:$src, tGPR:$val),
                               AddrModeNone, SizeSpecial, NoItinerary, "", "",
                          [(set R0, (ARMeh_sjlj_setjmp tGPR:$src, tGPR:$val))]>,
                             Requires<[IsThumb2, HasVFP2]>;
}

let Defs =
  [ R0,  R1,  R2,  R3,  R4,  R5,  R6,  R7,  R8,  R9,  R10, R11, R12, LR ],
  hasSideEffects = 1, isBarrier = 1, isCodeGenOnly = 1 in {
  def t2Int_eh_sjlj_setjmp_nofp : Thumb2XI<(outs), (ins tGPR:$src, tGPR:$val),
                               AddrModeNone, SizeSpecial, NoItinerary, "", "",
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
    hasExtraDefRegAllocReq = 1, isCodeGenOnly = 1 in
def t2LDMIA_RET: T2XIt<(outs GPR:$wb), (ins GPR:$Rn, pred:$p,
                                        reglist:$regs, variable_ops),
                        IIC_iLoad_mBr,
                        "ldmia${p}.w\t$Rn!, $regs",
                        "$Rn = $wb", []> {
  bits<4>  Rn;
  bits<16> regs;

  let Inst{31-27} = 0b11101;
  let Inst{26-25} = 0b00;
  let Inst{24-23} = 0b01;     // Increment After
  let Inst{22}    = 0;
  let Inst{21}    = 1;        // Writeback
  let Inst{20}    = 1;
  let Inst{19-16} = Rn;
  let Inst{15-0}  = regs;
}

let isBranch = 1, isTerminator = 1, isBarrier = 1 in {
let isPredicable = 1 in
def t2B   : T2XI<(outs), (ins uncondbrtarget:$target), IIC_Br,
                 "b.w\t$target",
                 [(br bb:$target)]> {
  let Inst{31-27} = 0b11110;
  let Inst{15-14} = 0b10;
  let Inst{12} = 1;

  bits<20> target;
  let Inst{26} = target{19};
  let Inst{11} = target{18};
  let Inst{13} = target{17};
  let Inst{21-16} = target{16-11};
  let Inst{10-0} = target{10-0};
}

let isNotDuplicable = 1, isIndirectBranch = 1 in {
def t2BR_JT : t2PseudoInst<(outs),
          (ins GPR:$target, GPR:$index, i32imm:$jt, i32imm:$id),
           SizeSpecial, IIC_Br,
          [(ARMbr2jt GPR:$target, GPR:$index, tjumptable:$jt, imm:$id)]>;

// FIXME: Add a non-pc based case that can be predicated.
def t2TBB_JT : t2PseudoInst<(outs),
        (ins GPR:$index, i32imm:$jt, i32imm:$id),
         SizeSpecial, IIC_Br, []>;

def t2TBH_JT : t2PseudoInst<(outs),
        (ins GPR:$index, i32imm:$jt, i32imm:$id),
         SizeSpecial, IIC_Br, []>;

def t2TBB : T2I<(outs), (ins GPR:$Rn, GPR:$Rm), IIC_Br,
                    "tbb", "\t[$Rn, $Rm]", []> {
  bits<4> Rn;
  bits<4> Rm;
  let Inst{31-20} = 0b111010001101;
  let Inst{19-16} = Rn;
  let Inst{15-5} = 0b11110000000;
  let Inst{4} = 0; // B form
  let Inst{3-0} = Rm;
}

def t2TBH : T2I<(outs), (ins GPR:$Rn, GPR:$Rm), IIC_Br,
                   "tbh", "\t[$Rn, $Rm, lsl #1]", []> {
  bits<4> Rn;
  bits<4> Rm;
  let Inst{31-20} = 0b111010001101;
  let Inst{19-16} = Rn;
  let Inst{15-5} = 0b11110000000;
  let Inst{4} = 1; // H form
  let Inst{3-0} = Rm;
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

  bits<4> p;
  let Inst{25-22} = p;

  bits<21> target;
  let Inst{26} = target{20};
  let Inst{11} = target{19};
  let Inst{13} = target{18};
  let Inst{21-16} = target{17-12};
  let Inst{10-0} = target{11-1};
}


// IT block
let Defs = [ITSTATE] in
def t2IT : Thumb2XI<(outs), (ins it_pred:$cc, it_mask:$mask),
                    AddrModeNone, Size2Bytes,  IIC_iALUx,
                    "it$mask\t$cc", "", []> {
  // 16-bit instruction.
  let Inst{31-16} = 0x0000;
  let Inst{15-8} = 0b10111111;

  bits<4> cc;
  bits<4> mask;
  let Inst{7-4} = cc;
  let Inst{3-0} = mask;
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

  bits<4> func;
  let Inst{19-16} = func;
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

  bits<11> opt;

  // mode number
  let Inst{4-0} = opt{4-0};

  // M flag
  let Inst{8} = opt{5};

  // F flag
  let Inst{5} = opt{6};

  // I flag
  let Inst{6} = opt{7};

  // A flag
  let Inst{7} = opt{8};

  // imod flag
  let Inst{10-9} = opt{10-9};
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

  bits<4> opt;
  let Inst{3-0} = opt;
}

// Secure Monitor Call is a system instruction -- for disassembly only
// Option = Inst{19-16}
def t2SMC : T2I<(outs), (ins i32imm:$opt), NoItinerary, "smc", "\t$opt",
                [/* For disassembly only; pattern left blank */]> {
  let Inst{31-27} = 0b11110;
  let Inst{26-20} = 0b1111111;
  let Inst{15-12} = 0b1000;

  bits<4> opt;
  let Inst{19-16} = opt;
}

class T2SRS<bits<12> op31_20,
           dag oops, dag iops, InstrItinClass itin,
          string opc, string asm, list<dag> pattern>
  : T2I<oops, iops, itin, opc, asm, pattern> {
  let Inst{31-20} = op31_20{11-0};

  bits<5> mode;
  let Inst{4-0} = mode{4-0};
}

// Store Return State is a system instruction -- for disassembly only
def t2SRSDBW : T2SRS<0b111010000010,
                   (outs),(ins i32imm:$mode),NoItinerary,"srsdb","\tsp!, $mode",
                   [/* For disassembly only; pattern left blank */]>;
def t2SRSDB  : T2SRS<0b111010000000,
                   (outs),(ins i32imm:$mode),NoItinerary,"srsdb","\tsp, $mode",
                   [/* For disassembly only; pattern left blank */]>;
def t2SRSIAW : T2SRS<0b111010011010,
                   (outs),(ins i32imm:$mode),NoItinerary,"srsia","\tsp!, $mode",
                   [/* For disassembly only; pattern left blank */]>;
def t2SRSIA  : T2SRS<0b111010011000,
                   (outs), (ins i32imm:$mode),NoItinerary,"srsia","\tsp, $mode",
                   [/* For disassembly only; pattern left blank */]>;

// Return From Exception is a system instruction -- for disassembly only

class T2RFE<bits<12> op31_20, dag oops, dag iops, InstrItinClass itin,
          string opc, string asm, list<dag> pattern>
  : T2I<oops, iops, itin, opc, asm, pattern> {
  let Inst{31-20} = op31_20{11-0};

  bits<4> Rn;
  let Inst{19-16} = Rn;
}

def t2RFEDBW : T2RFE<0b111010000011,
                   (outs), (ins rGPR:$Rn), NoItinerary, "rfedb", "\t$Rn!",
                   [/* For disassembly only; pattern left blank */]>;
def t2RFEDB  : T2RFE<0b111010000001,
                   (outs), (ins rGPR:$Rn), NoItinerary, "rfeab", "\t$Rn",
                   [/* For disassembly only; pattern left blank */]>;
def t2RFEIAW : T2RFE<0b111010011011,
                   (outs), (ins rGPR:$Rn), NoItinerary, "rfeia", "\t$Rn!",
                   [/* For disassembly only; pattern left blank */]>;
def t2RFEIA  : T2RFE<0b111010011001,
                   (outs), (ins rGPR:$Rn), NoItinerary, "rfeia", "\t$Rn",
                   [/* For disassembly only; pattern left blank */]>;

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//

// 32-bit immediate using movw + movt.
// This is a single pseudo instruction to make it re-materializable.
// FIXME: Remove this when we can do generalized remat.
let isReMaterializable = 1, isMoveImm = 1 in
def t2MOVi32imm : PseudoInst<(outs rGPR:$dst), (ins i32imm:$src), IIC_iMOVix2,
                            [(set rGPR:$dst, (i32 imm:$src))]>,
                            Requires<[IsThumb, HasV6T2]>;

// Pseudo instruction that combines movw + movt + add pc (if pic).
// It also makes it possible to rematerialize the instructions.
// FIXME: Remove this when we can do generalized remat and when machine licm
// can properly the instructions.
let isReMaterializable = 1 in {
def t2MOV_ga_pcrel : PseudoInst<(outs rGPR:$dst), (ins i32imm:$addr),
                                IIC_iMOVix2addpc,
                          [(set rGPR:$dst, (ARMWrapperPIC tglobaladdr:$addr))]>,
                          Requires<[IsThumb2, UseMovt]>;

def t2MOV_ga_dyn : PseudoInst<(outs rGPR:$dst), (ins i32imm:$addr),
                              IIC_iMOVix2,
                          [(set rGPR:$dst, (ARMWrapperDYN tglobaladdr:$addr))]>,
                          Requires<[IsThumb2, UseMovt]>;
}

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
def t2LDRpci_pic : PseudoInst<(outs rGPR:$dst), (ins i32imm:$addr, pclabel:$cp),
                   IIC_iLoadiALU,
              [(set rGPR:$dst, (ARMpic_add (load (ARMWrapper tconstpool:$addr)),
                                           imm:$cp))]>,
               Requires<[IsThumb2]>;

//===----------------------------------------------------------------------===//
// Move between special register and ARM core register -- for disassembly only
//

class T2SpecialReg<bits<12> op31_20, bits<2> op15_14, bits<1> op12,
          dag oops, dag iops, InstrItinClass itin,
          string opc, string asm, list<dag> pattern>
  : T2I<oops, iops, itin, opc, asm, pattern> {
  let Inst{31-20} = op31_20{11-0};
  let Inst{15-14} = op15_14{1-0};
  let Inst{12} = op12{0};
}

class T2MRS<bits<12> op31_20, bits<2> op15_14, bits<1> op12,
          dag oops, dag iops, InstrItinClass itin,
          string opc, string asm, list<dag> pattern>
  : T2SpecialReg<op31_20, op15_14, op12, oops, iops, itin, opc, asm, pattern> {
  bits<4> Rd;
  let Inst{11-8} = Rd;
  let Inst{19-16} = 0b1111;
}

def t2MRS : T2MRS<0b111100111110, 0b10, 0,
                (outs rGPR:$Rd), (ins), NoItinerary, "mrs", "\t$Rd, cpsr",
                [/* For disassembly only; pattern left blank */]>;
def t2MRSsys : T2MRS<0b111100111111, 0b10, 0,
                   (outs rGPR:$Rd), (ins), NoItinerary, "mrs", "\t$Rd, spsr",
                   [/* For disassembly only; pattern left blank */]>;

class T2MSR<bits<12> op31_20, bits<2> op15_14, bits<1> op12,
            dag oops, dag iops, InstrItinClass itin,
          string opc, string asm, list<dag> pattern>
  : T2SpecialReg<op31_20, op15_14, op12, oops, iops, itin, opc, asm, pattern> {
  bits<4> Rn;
  bits<4> mask;
  let Inst{19-16} = Rn;
  let Inst{11-8} = mask;
}

def t2MSR : T2MSR<0b111100111000, 0b10, 0,
                (outs), (ins rGPR:$Rn, msr_mask:$mask), NoItinerary, "msr",
                "\tcpsr$mask, $Rn",
                [/* For disassembly only; pattern left blank */]>;
def t2MSRsys : T2MSR<0b111100111001, 0b10, 0,
                   (outs), (ins rGPR:$Rn, msr_mask:$mask), NoItinerary, "msr",
                   "\tspsr$mask, $Rn",
                   [/* For disassembly only; pattern left blank */]>;

//===----------------------------------------------------------------------===//
// Move between coprocessor and ARM core register -- for disassembly only
//

class t2MovRCopro<string opc, bit direction>
  : T2Cop<(outs), (ins p_imm:$cop, i32imm:$opc1,
                       GPR:$Rt, c_imm:$CRn, c_imm:$CRm, i32imm:$opc2),
          !strconcat(opc, "\t$cop, $opc1, $Rt, $CRn, $CRm, $opc2"),
          [/* For disassembly only; pattern left blank */]> {
  let Inst{27-24} = 0b1110;
  let Inst{20} = direction;
  let Inst{4} = 1;

  bits<4> Rt;
  bits<4> cop;
  bits<3> opc1;
  bits<3> opc2;
  bits<4> CRm;
  bits<4> CRn;

  let Inst{15-12} = Rt;
  let Inst{11-8}  = cop;
  let Inst{23-21} = opc1;
  let Inst{7-5}   = opc2;
  let Inst{3-0}   = CRm;
  let Inst{19-16} = CRn;
}

def t2MCR2 : t2MovRCopro<"mcr2", 0 /* from ARM core register to coprocessor */>;
def t2MRC2 : t2MovRCopro<"mrc2", 1 /* from coprocessor to ARM core register */>;

class t2MovRRCopro<string opc, bit direction>
  : T2Cop<(outs), (ins p_imm:$cop, i32imm:$opc1, GPR:$Rt, GPR:$Rt2, c_imm:$CRm),
          !strconcat(opc, "\t$cop, $opc1, $Rt, $Rt2, $CRm"),
          [/* For disassembly only; pattern left blank */]> {
  let Inst{27-24} = 0b1100;
  let Inst{23-21} = 0b010;
  let Inst{20} = direction;

  bits<4> Rt;
  bits<4> Rt2;
  bits<4> cop;
  bits<4> opc1;
  bits<4> CRm;

  let Inst{15-12} = Rt;
  let Inst{19-16} = Rt2;
  let Inst{11-8}  = cop;
  let Inst{7-4}   = opc1;
  let Inst{3-0}   = CRm;
}

def t2MCRR2 : t2MovRRCopro<"mcrr2",
                           0 /* from ARM core register to coprocessor */>;
def t2MRRC2 : t2MovRRCopro<"mrrc2",
                           1 /* from coprocessor to ARM core register */>;

//===----------------------------------------------------------------------===//
// Other Coprocessor Instructions.  For disassembly only.
//

def t2CDP2 : T2Cop<(outs), (ins p_imm:$cop, i32imm:$opc1,
                   c_imm:$CRd, c_imm:$CRn, c_imm:$CRm, i32imm:$opc2),
                   "cdp2\t$cop, $opc1, $CRd, $CRn, $CRm, $opc2",
                   [/* For disassembly only; pattern left blank */]> {
  let Inst{27-24} = 0b1110;

  bits<4> opc1;
  bits<4> CRn;
  bits<4> CRd;
  bits<4> cop;
  bits<3> opc2;
  bits<4> CRm;

  let Inst{3-0}   = CRm;
  let Inst{4}     = 0;
  let Inst{7-5}   = opc2;
  let Inst{11-8}  = cop;
  let Inst{15-12} = CRd;
  let Inst{19-16} = CRn;
  let Inst{23-20} = opc1;
}
