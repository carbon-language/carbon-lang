//===- ARMInstrInfo.td - Target Description for ARM Target -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the ARM instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ARM specific DAG Nodes.
//

// Type profiles.
def SDT_ARMCallSeqStart : SDCallSeqStart<[ SDTCisVT<0, i32> ]>;
def SDT_ARMCallSeqEnd   : SDCallSeqEnd<[ SDTCisVT<0, i32>, SDTCisVT<1, i32> ]>;

def SDT_ARMSaveCallPC : SDTypeProfile<0, 1, []>;

def SDT_ARMcall    : SDTypeProfile<0, -1, [SDTCisPtrTy<0>]>;

def SDT_ARMCMov    : SDTypeProfile<1, 3,
                                   [SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>,
                                    SDTCisVT<3, i32>]>;

def SDT_ARMBrcond  : SDTypeProfile<0, 2,
                                   [SDTCisVT<0, OtherVT>, SDTCisVT<1, i32>]>;

def SDT_ARMBrJT    : SDTypeProfile<0, 3,
                                  [SDTCisPtrTy<0>, SDTCisVT<1, i32>,
                                   SDTCisVT<2, i32>]>;

def SDT_ARMBr2JT   : SDTypeProfile<0, 4,
                                  [SDTCisPtrTy<0>, SDTCisVT<1, i32>,
                                   SDTCisVT<2, i32>, SDTCisVT<3, i32>]>;

def SDT_ARMBCC_i64 : SDTypeProfile<0, 6,
                                  [SDTCisVT<0, i32>,
                                   SDTCisVT<1, i32>, SDTCisVT<2, i32>,
                                   SDTCisVT<3, i32>, SDTCisVT<4, i32>,
                                   SDTCisVT<5, OtherVT>]>;

def SDT_ARMAnd     : SDTypeProfile<1, 2,
                                   [SDTCisVT<0, i32>, SDTCisVT<1, i32>,
                                    SDTCisVT<2, i32>]>;

def SDT_ARMCmp     : SDTypeProfile<0, 2, [SDTCisSameAs<0, 1>]>;

def SDT_ARMPICAdd  : SDTypeProfile<1, 2, [SDTCisSameAs<0, 1>,
                                          SDTCisPtrTy<1>, SDTCisVT<2, i32>]>;

def SDT_ARMThreadPointer : SDTypeProfile<1, 0, [SDTCisPtrTy<0>]>;
def SDT_ARMEH_SJLJ_Setjmp : SDTypeProfile<1, 2, [SDTCisInt<0>, SDTCisPtrTy<1>,
                                                 SDTCisInt<2>]>;
def SDT_ARMEH_SJLJ_Longjmp: SDTypeProfile<0, 2, [SDTCisPtrTy<0>, SDTCisInt<1>]>;

def SDT_ARMEH_SJLJ_DispatchSetup: SDTypeProfile<0, 1, [SDTCisPtrTy<0>]>;

def SDT_ARMMEMBARRIER     : SDTypeProfile<0, 1, [SDTCisInt<0>]>;

def SDT_ARMTCRET : SDTypeProfile<0, 1, [SDTCisPtrTy<0>]>;

def SDT_ARMBFI : SDTypeProfile<1, 3, [SDTCisVT<0, i32>, SDTCisVT<1, i32>,
                                      SDTCisVT<2, i32>, SDTCisVT<3, i32>]>;

// Node definitions.
def ARMWrapper       : SDNode<"ARMISD::Wrapper",     SDTIntUnaryOp>;
def ARMWrapperJT     : SDNode<"ARMISD::WrapperJT",   SDTIntBinOp>;

def ARMcallseq_start : SDNode<"ISD::CALLSEQ_START", SDT_ARMCallSeqStart,
                              [SDNPHasChain, SDNPOutFlag]>;
def ARMcallseq_end   : SDNode<"ISD::CALLSEQ_END",   SDT_ARMCallSeqEnd,
                              [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;

def ARMcall          : SDNode<"ARMISD::CALL", SDT_ARMcall,
                              [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag,
                               SDNPVariadic]>;
def ARMcall_pred    : SDNode<"ARMISD::CALL_PRED", SDT_ARMcall,
                              [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag,
                               SDNPVariadic]>;
def ARMcall_nolink   : SDNode<"ARMISD::CALL_NOLINK", SDT_ARMcall,
                              [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag,
                               SDNPVariadic]>;

def ARMretflag       : SDNode<"ARMISD::RET_FLAG", SDTNone,
                              [SDNPHasChain, SDNPOptInFlag]>;

def ARMcmov          : SDNode<"ARMISD::CMOV", SDT_ARMCMov,
                              [SDNPInFlag]>;
def ARMcneg          : SDNode<"ARMISD::CNEG", SDT_ARMCMov,
                              [SDNPInFlag]>;

def ARMbrcond        : SDNode<"ARMISD::BRCOND", SDT_ARMBrcond,
                              [SDNPHasChain, SDNPInFlag, SDNPOutFlag]>;

def ARMbrjt          : SDNode<"ARMISD::BR_JT", SDT_ARMBrJT,
                              [SDNPHasChain]>;
def ARMbr2jt         : SDNode<"ARMISD::BR2_JT", SDT_ARMBr2JT,
                              [SDNPHasChain]>;

def ARMBcci64        : SDNode<"ARMISD::BCC_i64", SDT_ARMBCC_i64,
                              [SDNPHasChain]>;

def ARMcmp           : SDNode<"ARMISD::CMP", SDT_ARMCmp,
                              [SDNPOutFlag]>;

def ARMcmpZ          : SDNode<"ARMISD::CMPZ", SDT_ARMCmp,
                              [SDNPOutFlag, SDNPCommutative]>;

def ARMpic_add       : SDNode<"ARMISD::PIC_ADD", SDT_ARMPICAdd>;

def ARMsrl_flag      : SDNode<"ARMISD::SRL_FLAG", SDTIntUnaryOp, [SDNPOutFlag]>;
def ARMsra_flag      : SDNode<"ARMISD::SRA_FLAG", SDTIntUnaryOp, [SDNPOutFlag]>;
def ARMrrx           : SDNode<"ARMISD::RRX"     , SDTIntUnaryOp, [SDNPInFlag ]>;

def ARMthread_pointer: SDNode<"ARMISD::THREAD_POINTER", SDT_ARMThreadPointer>;
def ARMeh_sjlj_setjmp: SDNode<"ARMISD::EH_SJLJ_SETJMP",
                               SDT_ARMEH_SJLJ_Setjmp, [SDNPHasChain]>;
def ARMeh_sjlj_longjmp: SDNode<"ARMISD::EH_SJLJ_LONGJMP",
                               SDT_ARMEH_SJLJ_Longjmp, [SDNPHasChain]>;
def ARMeh_sjlj_dispatchsetup: SDNode<"ARMISD::EH_SJLJ_DISPATCHSETUP",
                               SDT_ARMEH_SJLJ_DispatchSetup, [SDNPHasChain]>;


def ARMMemBarrier     : SDNode<"ARMISD::MEMBARRIER", SDT_ARMMEMBARRIER,
                               [SDNPHasChain]>;
def ARMMemBarrierMCR  : SDNode<"ARMISD::MEMBARRIER_MCR", SDT_ARMMEMBARRIER,
                               [SDNPHasChain]>;
def ARMPreload        : SDNode<"ARMISD::PRELOAD", SDTPrefetch,
                               [SDNPHasChain, SDNPMayLoad, SDNPMayStore]>;

def ARMrbit          : SDNode<"ARMISD::RBIT", SDTIntUnaryOp>;

def ARMtcret         : SDNode<"ARMISD::TC_RETURN", SDT_ARMTCRET,
                        [SDNPHasChain,  SDNPOptInFlag, SDNPVariadic]>;


def ARMbfi           : SDNode<"ARMISD::BFI", SDT_ARMBFI>;

//===----------------------------------------------------------------------===//
// ARM Instruction Predicate Definitions.
//
def HasV4T           : Predicate<"Subtarget->hasV4TOps()">, AssemblerPredicate;
def NoV4T            : Predicate<"!Subtarget->hasV4TOps()">;
def HasV5T           : Predicate<"Subtarget->hasV5TOps()">;
def HasV5TE          : Predicate<"Subtarget->hasV5TEOps()">, AssemblerPredicate;
def HasV6            : Predicate<"Subtarget->hasV6Ops()">, AssemblerPredicate;
def HasV6T2          : Predicate<"Subtarget->hasV6T2Ops()">, AssemblerPredicate;
def NoV6T2           : Predicate<"!Subtarget->hasV6T2Ops()">;
def HasV7            : Predicate<"Subtarget->hasV7Ops()">, AssemblerPredicate;
def NoVFP            : Predicate<"!Subtarget->hasVFP2()">;
def HasVFP2          : Predicate<"Subtarget->hasVFP2()">, AssemblerPredicate;
def HasVFP3          : Predicate<"Subtarget->hasVFP3()">, AssemblerPredicate;
def HasNEON          : Predicate<"Subtarget->hasNEON()">, AssemblerPredicate;
def HasDivide        : Predicate<"Subtarget->hasDivide()">, AssemblerPredicate;
def HasT2ExtractPack : Predicate<"Subtarget->hasT2ExtractPack()">,
                                 AssemblerPredicate;
def HasDB            : Predicate<"Subtarget->hasDataBarrier()">,
                                 AssemblerPredicate;
def HasMP            : Predicate<"Subtarget->hasMPExtension()">,
                                 AssemblerPredicate;
def UseNEONForFP     : Predicate<"Subtarget->useNEONForSinglePrecisionFP()">;
def DontUseNEONForFP : Predicate<"!Subtarget->useNEONForSinglePrecisionFP()">;
def IsThumb          : Predicate<"Subtarget->isThumb()">, AssemblerPredicate;
def IsThumb1Only     : Predicate<"Subtarget->isThumb1Only()">;
def IsThumb2         : Predicate<"Subtarget->isThumb2()">, AssemblerPredicate;
def IsARM            : Predicate<"!Subtarget->isThumb()">, AssemblerPredicate;
def IsDarwin         : Predicate<"Subtarget->isTargetDarwin()">;
def IsNotDarwin      : Predicate<"!Subtarget->isTargetDarwin()">;

// FIXME: Eventually this will be just "hasV6T2Ops".
def UseMovt          : Predicate<"Subtarget->useMovt()">;
def DontUseMovt      : Predicate<"!Subtarget->useMovt()">;
def UseVMLx          : Predicate<"Subtarget->useVMLx()">;

//===----------------------------------------------------------------------===//
// ARM Flag Definitions.

class RegConstraint<string C> {
  string Constraints = C;
}

//===----------------------------------------------------------------------===//
//  ARM specific transformation functions and pattern fragments.
//

// so_imm_neg_XFORM - Return a so_imm value packed into the format described for
// so_imm_neg def below.
def so_imm_neg_XFORM : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(-(int)N->getZExtValue(), MVT::i32);
}]>;

// so_imm_not_XFORM - Return a so_imm value packed into the format described for
// so_imm_not def below.
def so_imm_not_XFORM : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(~(int)N->getZExtValue(), MVT::i32);
}]>;

/// imm1_15 predicate - True if the 32-bit immediate is in the range [1,15].
def imm1_15 : PatLeaf<(i32 imm), [{
  return (int32_t)N->getZExtValue() >= 1 && (int32_t)N->getZExtValue() < 16;
}]>;

/// imm16_31 predicate - True if the 32-bit immediate is in the range [16,31].
def imm16_31 : PatLeaf<(i32 imm), [{
  return (int32_t)N->getZExtValue() >= 16 && (int32_t)N->getZExtValue() < 32;
}]>;

def so_imm_neg :
  PatLeaf<(imm), [{
    return ARM_AM::getSOImmVal(-(int)N->getZExtValue()) != -1;
  }], so_imm_neg_XFORM>;

def so_imm_not :
  PatLeaf<(imm), [{
    return ARM_AM::getSOImmVal(~(int)N->getZExtValue()) != -1;
  }], so_imm_not_XFORM>;

// sext_16_node predicate - True if the SDNode is sign-extended 16 or more bits.
def sext_16_node : PatLeaf<(i32 GPR:$a), [{
  return CurDAG->ComputeNumSignBits(SDValue(N,0)) >= 17;
}]>;

/// bf_inv_mask_imm predicate - An AND mask to clear an arbitrary width bitfield
/// e.g., 0xf000ffff
def bf_inv_mask_imm : Operand<i32>,
                      PatLeaf<(imm), [{
  return ARM::isBitFieldInvertedMask(N->getZExtValue());
}] > {
  string EncoderMethod = "getBitfieldInvertedMaskOpValue";
  let PrintMethod = "printBitfieldInvMaskImmOperand";
}

/// Split a 32-bit immediate into two 16 bit parts.
def hi16 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant((uint32_t)N->getZExtValue() >> 16, MVT::i32);
}]>;

def lo16AllZero : PatLeaf<(i32 imm), [{
  // Returns true if all low 16-bits are 0.
  return (((uint32_t)N->getZExtValue()) & 0xFFFFUL) == 0;
}], hi16>;

/// imm0_65535 predicate - True if the 32-bit immediate is in the range
/// [0.65535].
def imm0_65535 : PatLeaf<(i32 imm), [{
  return (uint32_t)N->getZExtValue() < 65536;
}]>;

class BinOpFrag<dag res> : PatFrag<(ops node:$LHS, node:$RHS), res>;
class UnOpFrag <dag res> : PatFrag<(ops node:$Src), res>;

/// adde and sube predicates - True based on whether the carry flag output
/// will be needed or not.
def adde_dead_carry :
  PatFrag<(ops node:$LHS, node:$RHS), (adde node:$LHS, node:$RHS),
  [{return !N->hasAnyUseOfValue(1);}]>;
def sube_dead_carry :
  PatFrag<(ops node:$LHS, node:$RHS), (sube node:$LHS, node:$RHS),
  [{return !N->hasAnyUseOfValue(1);}]>;
def adde_live_carry :
  PatFrag<(ops node:$LHS, node:$RHS), (adde node:$LHS, node:$RHS),
  [{return N->hasAnyUseOfValue(1);}]>;
def sube_live_carry :
  PatFrag<(ops node:$LHS, node:$RHS), (sube node:$LHS, node:$RHS),
  [{return N->hasAnyUseOfValue(1);}]>;

//===----------------------------------------------------------------------===//
// Operand Definitions.
//

// Branch target.
def brtarget : Operand<OtherVT>;

// A list of registers separated by comma. Used by load/store multiple.
def reglist : Operand<i32> {
  string EncoderMethod = "getRegisterListOpValue";
  let PrintMethod = "printRegisterList";
}

def RegListAsmOperand : AsmOperandClass {
  let Name = "RegList";
  let SuperClasses = [];
}

// An operand for the CONSTPOOL_ENTRY pseudo-instruction.
def cpinst_operand : Operand<i32> {
  let PrintMethod = "printCPInstOperand";
}

def jtblock_operand : Operand<i32> {
  let PrintMethod = "printJTBlockOperand";
}
def jt2block_operand : Operand<i32> {
  let PrintMethod = "printJT2BlockOperand";
}

// Local PC labels.
def pclabel : Operand<i32> {
  let PrintMethod = "printPCLabel";
}

def neon_vcvt_imm32 : Operand<i32> {
  string EncoderMethod = "getNEONVcvtImm32OpValue";
}

// rot_imm: An integer that encodes a rotate amount. Must be 8, 16, or 24.
def rot_imm : Operand<i32>, PatLeaf<(i32 imm), [{
  int32_t v = (int32_t)N->getZExtValue();
  return v == 8 || v == 16 || v == 24; }]> {
  string EncoderMethod = "getRotImmOpValue";
}

// shift_imm: An integer that encodes a shift amount and the type of shift
// (currently either asr or lsl) using the same encoding used for the
// immediates in so_reg operands.
def shift_imm : Operand<i32> {
  let PrintMethod = "printShiftImmOperand";
}

// shifter_operand operands: so_reg and so_imm.
def so_reg : Operand<i32>,    // reg reg imm
             ComplexPattern<i32, 3, "SelectShifterOperandReg",
                            [shl,srl,sra,rotr]> {
  string EncoderMethod = "getSORegOpValue";
  let PrintMethod = "printSORegOperand";
  let MIOperandInfo = (ops GPR, GPR, i32imm);
}
def shift_so_reg : Operand<i32>,    // reg reg imm
                   ComplexPattern<i32, 3, "SelectShiftShifterOperandReg",
                                  [shl,srl,sra,rotr]> {
  string EncoderMethod = "getSORegOpValue";
  let PrintMethod = "printSORegOperand";
  let MIOperandInfo = (ops GPR, GPR, i32imm);
}

// so_imm - Match a 32-bit shifter_operand immediate operand, which is an
// 8-bit immediate rotated by an arbitrary number of bits.  so_imm values are
// represented in the imm field in the same 12-bit form that they are encoded
// into so_imm instructions: the 8-bit immediate is the least significant bits
// [bits 0-7], the 4-bit shift amount is the next 4 bits [bits 8-11].
def so_imm : Operand<i32>, PatLeaf<(imm), [{ return Pred_so_imm(N); }]> {
  string EncoderMethod = "getSOImmOpValue";
  let PrintMethod = "printSOImmOperand";
}

// Break so_imm's up into two pieces.  This handles immediates with up to 16
// bits set in them.  This uses so_imm2part to match and so_imm2part_[12] to
// get the first/second pieces.
def so_imm2part : Operand<i32>,
                  PatLeaf<(imm), [{
      return ARM_AM::isSOImmTwoPartVal((unsigned)N->getZExtValue());
    }]> {
  let PrintMethod = "printSOImm2PartOperand";
}

def so_imm2part_1 : SDNodeXForm<imm, [{
  unsigned V = ARM_AM::getSOImmTwoPartFirst((unsigned)N->getZExtValue());
  return CurDAG->getTargetConstant(V, MVT::i32);
}]>;

def so_imm2part_2 : SDNodeXForm<imm, [{
  unsigned V = ARM_AM::getSOImmTwoPartSecond((unsigned)N->getZExtValue());
  return CurDAG->getTargetConstant(V, MVT::i32);
}]>;

def so_neg_imm2part : Operand<i32>, PatLeaf<(imm), [{
      return ARM_AM::isSOImmTwoPartVal(-(int)N->getZExtValue());
    }]> {
  let PrintMethod = "printSOImm2PartOperand";
}

def so_neg_imm2part_1 : SDNodeXForm<imm, [{
  unsigned V = ARM_AM::getSOImmTwoPartFirst(-(int)N->getZExtValue());
  return CurDAG->getTargetConstant(V, MVT::i32);
}]>;

def so_neg_imm2part_2 : SDNodeXForm<imm, [{
  unsigned V = ARM_AM::getSOImmTwoPartSecond(-(int)N->getZExtValue());
  return CurDAG->getTargetConstant(V, MVT::i32);
}]>;

/// imm0_31 predicate - True if the 32-bit immediate is in the range [0,31].
def imm0_31 : Operand<i32>, PatLeaf<(imm), [{
  return (int32_t)N->getZExtValue() < 32;
}]>;

/// imm0_31_m1 - Matches and prints like imm0_31, but encodes as 'value - 1'.
def imm0_31_m1 : Operand<i32>, PatLeaf<(imm), [{
  return (int32_t)N->getZExtValue() < 32;
}]> {
  string EncoderMethod = "getImmMinusOneOpValue";
}

// Define ARM specific addressing modes.


// addrmode_imm12 := reg +/- imm12
//
def addrmode_imm12 : Operand<i32>,
                     ComplexPattern<i32, 2, "SelectAddrModeImm12", []> {
  // 12-bit immediate operand. Note that instructions using this encode
  // #0 and #-0 differently. We flag #-0 as the magic value INT32_MIN. All other
  // immediate values are as normal.

  string EncoderMethod = "getAddrModeImm12OpValue";
  let PrintMethod = "printAddrModeImm12Operand";
  let MIOperandInfo = (ops GPR:$base, i32imm:$offsimm);
}
// ldst_so_reg := reg +/- reg shop imm
//
def ldst_so_reg : Operand<i32>,
                  ComplexPattern<i32, 3, "SelectLdStSOReg", []> {
  string EncoderMethod = "getLdStSORegOpValue";
  // FIXME: Simplify the printer
  let PrintMethod = "printAddrMode2Operand";
  let MIOperandInfo = (ops GPR:$base, GPR:$offsreg, i32imm:$offsimm);
}

// addrmode2 := reg +/- imm12
//           := reg +/- reg shop imm
//
def addrmode2 : Operand<i32>,
                ComplexPattern<i32, 3, "SelectAddrMode2", []> {
  let PrintMethod = "printAddrMode2Operand";
  let MIOperandInfo = (ops GPR:$base, GPR:$offsreg, i32imm:$offsimm);
}

def am2offset : Operand<i32>,
                ComplexPattern<i32, 2, "SelectAddrMode2Offset",
                [], [SDNPWantRoot]> {
  let PrintMethod = "printAddrMode2OffsetOperand";
  let MIOperandInfo = (ops GPR, i32imm);
}

// addrmode3 := reg +/- reg
// addrmode3 := reg +/- imm8
//
def addrmode3 : Operand<i32>,
                ComplexPattern<i32, 3, "SelectAddrMode3", []> {
  let PrintMethod = "printAddrMode3Operand";
  let MIOperandInfo = (ops GPR:$base, GPR:$offsreg, i32imm:$offsimm);
}

def am3offset : Operand<i32>,
                ComplexPattern<i32, 2, "SelectAddrMode3Offset",
                               [], [SDNPWantRoot]> {
  let PrintMethod = "printAddrMode3OffsetOperand";
  let MIOperandInfo = (ops GPR, i32imm);
}

// ldstm_mode := {ia, ib, da, db}
//
def ldstm_mode : OptionalDefOperand<OtherVT, (ops i32), (ops (i32 1))> {
  let PrintMethod = "printLdStmModeOperand";
}

def MemMode5AsmOperand : AsmOperandClass {
  let Name = "MemMode5";
  let SuperClasses = [];
}

// addrmode5 := reg +/- imm8*4
//
def addrmode5 : Operand<i32>,
                ComplexPattern<i32, 2, "SelectAddrMode5", []> {
  let PrintMethod = "printAddrMode5Operand";
  let MIOperandInfo = (ops GPR:$base, i32imm);
  let ParserMatchClass = MemMode5AsmOperand;
  string EncoderMethod = "getAddrMode5OpValue";
}

// addrmode6 := reg with optional writeback
//
def addrmode6 : Operand<i32>,
                ComplexPattern<i32, 2, "SelectAddrMode6", [], [SDNPWantParent]>{
  let PrintMethod = "printAddrMode6Operand";
  let MIOperandInfo = (ops GPR:$addr, i32imm);
  string EncoderMethod = "getAddrMode6AddressOpValue";
}

def am6offset : Operand<i32> {
  let PrintMethod = "printAddrMode6OffsetOperand";
  let MIOperandInfo = (ops GPR);
  string EncoderMethod = "getAddrMode6OffsetOpValue";
}

// addrmodepc := pc + reg
//
def addrmodepc : Operand<i32>,
                 ComplexPattern<i32, 2, "SelectAddrModePC", []> {
  let PrintMethod = "printAddrModePCOperand";
  let MIOperandInfo = (ops GPR, i32imm);
}

def nohash_imm : Operand<i32> {
  let PrintMethod = "printNoHashImmediate";
}

//===----------------------------------------------------------------------===//

include "ARMInstrFormats.td"

//===----------------------------------------------------------------------===//
// Multiclass helpers...
//

/// AsI1_bin_irs - Defines a set of (op r, {so_imm|r|so_reg}) patterns for a
/// binop that produces a value.
multiclass AsI1_bin_irs<bits<4> opcod, string opc,
                     InstrItinClass iii, InstrItinClass iir, InstrItinClass iis,
                        PatFrag opnode, bit Commutable = 0> {
  // The register-immediate version is re-materializable. This is useful
  // in particular for taking the address of a local.
  let isReMaterializable = 1 in {
  def ri : AsI1<opcod, (outs GPR:$Rd), (ins GPR:$Rn, so_imm:$imm), DPFrm,
               iii, opc, "\t$Rd, $Rn, $imm",
               [(set GPR:$Rd, (opnode GPR:$Rn, so_imm:$imm))]> {
    bits<4> Rd;
    bits<4> Rn;
    bits<12> imm;
    let Inst{25} = 1;
    let Inst{19-16} = Rn;
    let Inst{15-12} = Rd;
    let Inst{11-0} = imm;
  }
  }
  def rr : AsI1<opcod, (outs GPR:$Rd), (ins GPR:$Rn, GPR:$Rm), DPFrm,
               iir, opc, "\t$Rd, $Rn, $Rm",
               [(set GPR:$Rd, (opnode GPR:$Rn, GPR:$Rm))]> {
    bits<4> Rd;
    bits<4> Rn;
    bits<4> Rm;
    let Inst{25} = 0;
    let isCommutable = Commutable;
    let Inst{19-16} = Rn;
    let Inst{15-12} = Rd;
    let Inst{11-4} = 0b00000000;
    let Inst{3-0} = Rm;
  }
  def rs : AsI1<opcod, (outs GPR:$Rd), (ins GPR:$Rn, so_reg:$shift), DPSoRegFrm,
               iis, opc, "\t$Rd, $Rn, $shift",
               [(set GPR:$Rd, (opnode GPR:$Rn, so_reg:$shift))]> {
    bits<4> Rd;
    bits<4> Rn;
    bits<12> shift;
    let Inst{25} = 0;
    let Inst{19-16} = Rn;
    let Inst{15-12} = Rd;
    let Inst{11-0} = shift;
  }
}

/// AI1_bin_s_irs - Similar to AsI1_bin_irs except it sets the 's' bit so the
/// instruction modifies the CPSR register.
let Defs = [CPSR] in {
multiclass AI1_bin_s_irs<bits<4> opcod, string opc,
                     InstrItinClass iii, InstrItinClass iir, InstrItinClass iis,
                         PatFrag opnode, bit Commutable = 0> {
  def ri : AI1<opcod, (outs GPR:$Rd), (ins GPR:$Rn, so_imm:$imm), DPFrm,
               iii, opc, "\t$Rd, $Rn, $imm",
               [(set GPR:$Rd, (opnode GPR:$Rn, so_imm:$imm))]> {
    bits<4> Rd;
    bits<4> Rn;
    bits<12> imm;
    let Inst{25} = 1;
    let Inst{20} = 1;
    let Inst{19-16} = Rn;
    let Inst{15-12} = Rd;
    let Inst{11-0} = imm;
  }
  def rr : AI1<opcod, (outs GPR:$Rd), (ins GPR:$Rn, GPR:$Rm), DPFrm,
               iir, opc, "\t$Rd, $Rn, $Rm",
               [(set GPR:$Rd, (opnode GPR:$Rn, GPR:$Rm))]> {
    bits<4> Rd;
    bits<4> Rn;
    bits<4> Rm;
    let isCommutable = Commutable;
    let Inst{25} = 0;
    let Inst{20} = 1;
    let Inst{19-16} = Rn;
    let Inst{15-12} = Rd;
    let Inst{11-4} = 0b00000000;
    let Inst{3-0} = Rm;
  }
  def rs : AI1<opcod, (outs GPR:$Rd), (ins GPR:$Rn, so_reg:$shift), DPSoRegFrm,
               iis, opc, "\t$Rd, $Rn, $shift",
               [(set GPR:$Rd, (opnode GPR:$Rn, so_reg:$shift))]> {
    bits<4> Rd;
    bits<4> Rn;
    bits<12> shift;
    let Inst{25} = 0;
    let Inst{20} = 1;
    let Inst{19-16} = Rn;
    let Inst{15-12} = Rd;
    let Inst{11-0} = shift;
  }
}
}

/// AI1_cmp_irs - Defines a set of (op r, {so_imm|r|so_reg}) cmp / test
/// patterns. Similar to AsI1_bin_irs except the instruction does not produce
/// a explicit result, only implicitly set CPSR.
let isCompare = 1, Defs = [CPSR] in {
multiclass AI1_cmp_irs<bits<4> opcod, string opc,
                     InstrItinClass iii, InstrItinClass iir, InstrItinClass iis,
                       PatFrag opnode, bit Commutable = 0> {
  def ri : AI1<opcod, (outs), (ins GPR:$Rn, so_imm:$imm), DPFrm, iii,
               opc, "\t$Rn, $imm",
               [(opnode GPR:$Rn, so_imm:$imm)]> {
    bits<4> Rn;
    bits<12> imm;
    let Inst{25} = 1;
    let Inst{20} = 1;
    let Inst{19-16} = Rn;
    let Inst{15-12} = 0b0000;
    let Inst{11-0} = imm;
  }
  def rr : AI1<opcod, (outs), (ins GPR:$Rn, GPR:$Rm), DPFrm, iir,
               opc, "\t$Rn, $Rm",
               [(opnode GPR:$Rn, GPR:$Rm)]> {
    bits<4> Rn;
    bits<4> Rm;
    let isCommutable = Commutable;
    let Inst{25} = 0;
    let Inst{20} = 1;
    let Inst{19-16} = Rn;
    let Inst{15-12} = 0b0000;
    let Inst{11-4} = 0b00000000;
    let Inst{3-0} = Rm;
  }
  def rs : AI1<opcod, (outs), (ins GPR:$Rn, so_reg:$shift), DPSoRegFrm, iis,
               opc, "\t$Rn, $shift",
               [(opnode GPR:$Rn, so_reg:$shift)]> {
    bits<4> Rn;
    bits<12> shift;
    let Inst{25} = 0;
    let Inst{20} = 1;
    let Inst{19-16} = Rn;
    let Inst{15-12} = 0b0000;
    let Inst{11-0} = shift;
  }
}
}

/// AI_ext_rrot - A unary operation with two forms: one whose operand is a
/// register and one whose operand is a register rotated by 8/16/24.
/// FIXME: Remove the 'r' variant. Its rot_imm is zero.
multiclass AI_ext_rrot<bits<8> opcod, string opc, PatFrag opnode> {
  def r     : AExtI<opcod, (outs GPR:$Rd), (ins GPR:$Rm),
                 IIC_iEXTr, opc, "\t$Rd, $Rm",
                 [(set GPR:$Rd, (opnode GPR:$Rm))]>,
              Requires<[IsARM, HasV6]> {
    bits<4> Rd;
    bits<4> Rm;
    let Inst{19-16} = 0b1111;
    let Inst{15-12} = Rd;
    let Inst{11-10} = 0b00;
    let Inst{3-0}   = Rm;
  }
  def r_rot : AExtI<opcod, (outs GPR:$Rd), (ins GPR:$Rm, rot_imm:$rot),
                 IIC_iEXTr, opc, "\t$Rd, $Rm, ror $rot",
                 [(set GPR:$Rd, (opnode (rotr GPR:$Rm, rot_imm:$rot)))]>,
              Requires<[IsARM, HasV6]> {
    bits<4> Rd;
    bits<4> Rm;
    bits<2> rot;
    let Inst{19-16} = 0b1111;
    let Inst{15-12} = Rd;
    let Inst{11-10} = rot;
    let Inst{3-0}   = Rm;
  }
}

multiclass AI_ext_rrot_np<bits<8> opcod, string opc> {
  def r     : AExtI<opcod, (outs GPR:$Rd), (ins GPR:$Rm),
                 IIC_iEXTr, opc, "\t$Rd, $Rm",
                 [/* For disassembly only; pattern left blank */]>,
              Requires<[IsARM, HasV6]> {
    let Inst{19-16} = 0b1111;
    let Inst{11-10} = 0b00;
  }
  def r_rot : AExtI<opcod, (outs GPR:$Rd), (ins GPR:$Rm, rot_imm:$rot),
                 IIC_iEXTr, opc, "\t$Rd, $Rm, ror $rot",
                 [/* For disassembly only; pattern left blank */]>,
              Requires<[IsARM, HasV6]> {
    bits<2> rot;
    let Inst{19-16} = 0b1111;
    let Inst{11-10} = rot;
  }
}

/// AI_exta_rrot - A binary operation with two forms: one whose operand is a
/// register and one whose operand is a register rotated by 8/16/24.
multiclass AI_exta_rrot<bits<8> opcod, string opc, PatFrag opnode> {
  def rr     : AExtI<opcod, (outs GPR:$Rd), (ins GPR:$Rn, GPR:$Rm),
                  IIC_iEXTAr, opc, "\t$Rd, $Rn, $Rm",
                  [(set GPR:$Rd, (opnode GPR:$Rn, GPR:$Rm))]>,
               Requires<[IsARM, HasV6]> {
    let Inst{11-10} = 0b00;
  }
  def rr_rot : AExtI<opcod, (outs GPR:$Rd), (ins GPR:$Rn, GPR:$Rm,
                                             rot_imm:$rot),
                  IIC_iEXTAr, opc, "\t$Rd, $Rn, $Rm, ror $rot",
                  [(set GPR:$Rd, (opnode GPR:$Rn,
                                          (rotr GPR:$Rm, rot_imm:$rot)))]>,
                  Requires<[IsARM, HasV6]> {
    bits<4> Rn;
    bits<2> rot;
    let Inst{19-16} = Rn;
    let Inst{11-10} = rot;
  }
}

// For disassembly only.
multiclass AI_exta_rrot_np<bits<8> opcod, string opc> {
  def rr     : AExtI<opcod, (outs GPR:$Rd), (ins GPR:$Rn, GPR:$Rm),
                  IIC_iEXTAr, opc, "\t$Rd, $Rn, $Rm",
                  [/* For disassembly only; pattern left blank */]>,
               Requires<[IsARM, HasV6]> {
    let Inst{11-10} = 0b00;
  }
  def rr_rot : AExtI<opcod, (outs GPR:$Rd), (ins GPR:$Rn, GPR:$Rm,
                                             rot_imm:$rot),
                  IIC_iEXTAr, opc, "\t$Rd, $Rn, $Rm, ror $rot",
                  [/* For disassembly only; pattern left blank */]>,
                  Requires<[IsARM, HasV6]> {
    bits<4> Rn;
    bits<2> rot;
    let Inst{19-16} = Rn;
    let Inst{11-10} = rot;
  }
}

/// AI1_adde_sube_irs - Define instructions and patterns for adde and sube.
let Uses = [CPSR] in {
multiclass AI1_adde_sube_irs<bits<4> opcod, string opc, PatFrag opnode,
                             bit Commutable = 0> {
  def ri : AsI1<opcod, (outs GPR:$Rd), (ins GPR:$Rn, so_imm:$imm),
                DPFrm, IIC_iALUi, opc, "\t$Rd, $Rn, $imm",
               [(set GPR:$Rd, (opnode GPR:$Rn, so_imm:$imm))]>,
               Requires<[IsARM]> {
    bits<4> Rd;
    bits<4> Rn;
    bits<12> imm;
    let Inst{25} = 1;
    let Inst{15-12} = Rd;
    let Inst{19-16} = Rn;
    let Inst{11-0} = imm;
  }
  def rr : AsI1<opcod, (outs GPR:$Rd), (ins GPR:$Rn, GPR:$Rm),
                DPFrm, IIC_iALUr, opc, "\t$Rd, $Rn, $Rm",
               [(set GPR:$Rd, (opnode GPR:$Rn, GPR:$Rm))]>,
               Requires<[IsARM]> {
    bits<4> Rd;
    bits<4> Rn;
    bits<4> Rm;
    let Inst{11-4} = 0b00000000;
    let Inst{25} = 0;
    let isCommutable = Commutable;
    let Inst{3-0} = Rm;
    let Inst{15-12} = Rd;
    let Inst{19-16} = Rn;
  }
  def rs : AsI1<opcod, (outs GPR:$Rd), (ins GPR:$Rn, so_reg:$shift),
                DPSoRegFrm, IIC_iALUsr, opc, "\t$Rd, $Rn, $shift",
               [(set GPR:$Rd, (opnode GPR:$Rn, so_reg:$shift))]>,
               Requires<[IsARM]> {
    bits<4> Rd;
    bits<4> Rn;
    bits<12> shift;
    let Inst{25} = 0;
    let Inst{11-0} = shift;
    let Inst{15-12} = Rd;
    let Inst{19-16} = Rn;
  }
}
// Carry setting variants
let Defs = [CPSR] in {
multiclass AI1_adde_sube_s_irs<bits<4> opcod, string opc, PatFrag opnode,
                             bit Commutable = 0> {
  def Sri : AXI1<opcod, (outs GPR:$Rd), (ins GPR:$Rn, so_imm:$imm),
                DPFrm, IIC_iALUi, !strconcat(opc, "\t$Rd, $Rn, $imm"),
               [(set GPR:$Rd, (opnode GPR:$Rn, so_imm:$imm))]>,
               Requires<[IsARM]> {
    bits<4> Rd;
    bits<4> Rn;
    bits<12> imm;
    let Inst{15-12} = Rd;
    let Inst{19-16} = Rn;
    let Inst{11-0} = imm;
    let Inst{20} = 1;
    let Inst{25} = 1;
  }
  def Srr : AXI1<opcod, (outs GPR:$Rd), (ins GPR:$Rn, GPR:$Rm),
                DPFrm, IIC_iALUr, !strconcat(opc, "\t$Rd, $Rn, $Rm"),
               [(set GPR:$Rd, (opnode GPR:$Rn, GPR:$Rm))]>,
               Requires<[IsARM]> {
    bits<4> Rd;
    bits<4> Rn;
    bits<4> Rm;
    let Inst{11-4} = 0b00000000;
    let isCommutable = Commutable;
    let Inst{3-0} = Rm;
    let Inst{15-12} = Rd;
    let Inst{19-16} = Rn;
    let Inst{20} = 1;
    let Inst{25} = 0;
  }
  def Srs : AXI1<opcod, (outs GPR:$Rd), (ins GPR:$Rn, so_reg:$shift),
                DPSoRegFrm, IIC_iALUsr, !strconcat(opc, "\t$Rd, $Rn, $shift"),
               [(set GPR:$Rd, (opnode GPR:$Rn, so_reg:$shift))]>,
               Requires<[IsARM]> {
    bits<4> Rd;
    bits<4> Rn;
    bits<12> shift;
    let Inst{11-0} = shift;
    let Inst{15-12} = Rd;
    let Inst{19-16} = Rn;
    let Inst{20} = 1;
    let Inst{25} = 0;
  }
}
}
}

let canFoldAsLoad = 1, isReMaterializable = 1 in {
multiclass AI_ldr1<bit opc22, string opc, InstrItinClass iii,
           InstrItinClass iir, PatFrag opnode> {
  // Note: We use the complex addrmode_imm12 rather than just an input
  // GPR and a constrained immediate so that we can use this to match
  // frame index references and avoid matching constant pool references.
  def i12 : AIldst1<0b010, opc22, 1, (outs GPR:$Rt), (ins addrmode_imm12:$addr),
                   AddrMode_i12, LdFrm, iii, opc, "\t$Rt, $addr",
                  [(set GPR:$Rt, (opnode addrmode_imm12:$addr))]> {
    bits<4>  Rt;
    bits<17> addr;
    let Inst{23}    = addr{12};     // U (add = ('U' == 1))
    let Inst{19-16} = addr{16-13};  // Rn
    let Inst{15-12} = Rt;
    let Inst{11-0}  = addr{11-0};   // imm12
  }
  def rs : AIldst1<0b011, opc22, 1, (outs GPR:$Rt), (ins ldst_so_reg:$shift),
                  AddrModeNone, LdFrm, iir, opc, "\t$Rt, $shift",
                 [(set GPR:$Rt, (opnode ldst_so_reg:$shift))]> {
    bits<4>  Rt;
    bits<17> shift;
    let Inst{23}    = shift{12};    // U (add = ('U' == 1))
    let Inst{19-16} = shift{16-13}; // Rn
    let Inst{11-0}  = shift{11-0};
  }
}
}

multiclass AI_str1<bit opc22, string opc, InstrItinClass iii,
           InstrItinClass iir, PatFrag opnode> {
  // Note: We use the complex addrmode_imm12 rather than just an input
  // GPR and a constrained immediate so that we can use this to match
  // frame index references and avoid matching constant pool references.
  def i12 : AIldst1<0b010, opc22, 0, (outs),
                   (ins GPR:$Rt, addrmode_imm12:$addr),
                   AddrMode_i12, StFrm, iii, opc, "\t$Rt, $addr",
                  [(opnode GPR:$Rt, addrmode_imm12:$addr)]> {
    bits<4> Rt;
    bits<17> addr;
    let Inst{23}    = addr{12};     // U (add = ('U' == 1))
    let Inst{19-16} = addr{16-13};  // Rn
    let Inst{15-12} = Rt;
    let Inst{11-0}  = addr{11-0};   // imm12
  }
  def rs : AIldst1<0b011, opc22, 0, (outs), (ins GPR:$Rt, ldst_so_reg:$shift),
                  AddrModeNone, StFrm, iir, opc, "\t$Rt, $shift",
                 [(opnode GPR:$Rt, ldst_so_reg:$shift)]> {
    bits<4> Rt;
    bits<17> shift;
    let Inst{23}    = shift{12};    // U (add = ('U' == 1))
    let Inst{19-16} = shift{16-13}; // Rn
    let Inst{11-0}  = shift{11-0};
  }
}
//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Miscellaneous Instructions.
//

/// CONSTPOOL_ENTRY - This instruction represents a floating constant pool in
/// the function.  The first operand is the ID# for this instruction, the second
/// is the index into the MachineConstantPool that this is, the third is the
/// size in bytes of this constant pool entry.
let neverHasSideEffects = 1, isNotDuplicable = 1 in
def CONSTPOOL_ENTRY :
PseudoInst<(outs), (ins cpinst_operand:$instid, cpinst_operand:$cpidx,
                    i32imm:$size), NoItinerary, "", []>;

// FIXME: Marking these as hasSideEffects is necessary to prevent machine DCE
// from removing one half of the matched pairs. That breaks PEI, which assumes
// these will always be in pairs, and asserts if it finds otherwise. Better way?
let Defs = [SP], Uses = [SP], hasSideEffects = 1 in {
def ADJCALLSTACKUP :
PseudoInst<(outs), (ins i32imm:$amt1, i32imm:$amt2, pred:$p), NoItinerary, "",
           [(ARMcallseq_end timm:$amt1, timm:$amt2)]>;

def ADJCALLSTACKDOWN :
PseudoInst<(outs), (ins i32imm:$amt, pred:$p), NoItinerary, "",
           [(ARMcallseq_start timm:$amt)]>;
}

def NOP : AI<(outs), (ins), MiscFrm, NoItinerary, "nop", "",
             [/* For disassembly only; pattern left blank */]>,
          Requires<[IsARM, HasV6T2]> {
  let Inst{27-16} = 0b001100100000;
  let Inst{15-8} = 0b11110000;
  let Inst{7-0} = 0b00000000;
}

def YIELD : AI<(outs), (ins), MiscFrm, NoItinerary, "yield", "",
             [/* For disassembly only; pattern left blank */]>,
          Requires<[IsARM, HasV6T2]> {
  let Inst{27-16} = 0b001100100000;
  let Inst{15-8} = 0b11110000;
  let Inst{7-0} = 0b00000001;
}

def WFE : AI<(outs), (ins), MiscFrm, NoItinerary, "wfe", "",
             [/* For disassembly only; pattern left blank */]>,
          Requires<[IsARM, HasV6T2]> {
  let Inst{27-16} = 0b001100100000;
  let Inst{15-8} = 0b11110000;
  let Inst{7-0} = 0b00000010;
}

def WFI : AI<(outs), (ins), MiscFrm, NoItinerary, "wfi", "",
             [/* For disassembly only; pattern left blank */]>,
          Requires<[IsARM, HasV6T2]> {
  let Inst{27-16} = 0b001100100000;
  let Inst{15-8} = 0b11110000;
  let Inst{7-0} = 0b00000011;
}

def SEL : AI<(outs GPR:$dst), (ins GPR:$a, GPR:$b), DPFrm, NoItinerary, "sel",
             "\t$dst, $a, $b",
             [/* For disassembly only; pattern left blank */]>,
          Requires<[IsARM, HasV6]> {
  bits<4> Rd;
  bits<4> Rn;
  bits<4> Rm;
  let Inst{3-0} = Rm;
  let Inst{15-12} = Rd;
  let Inst{19-16} = Rn;
  let Inst{27-20} = 0b01101000;
  let Inst{7-4} = 0b1011;
  let Inst{11-8} = 0b1111;
}

def SEV : AI<(outs), (ins), MiscFrm, NoItinerary, "sev", "",
             [/* For disassembly only; pattern left blank */]>,
          Requires<[IsARM, HasV6T2]> {
  let Inst{27-16} = 0b001100100000;
  let Inst{15-8} = 0b11110000;
  let Inst{7-0} = 0b00000100;
}

// The i32imm operand $val can be used by a debugger to store more information
// about the breakpoint.
def BKPT : AI<(outs), (ins i32imm:$val), MiscFrm, NoItinerary, "bkpt", "\t$val",
              [/* For disassembly only; pattern left blank */]>,
           Requires<[IsARM]> {
  bits<16> val;
  let Inst{3-0} = val{3-0};
  let Inst{19-8} = val{15-4};
  let Inst{27-20} = 0b00010010;
  let Inst{7-4} = 0b0111;
}

// Change Processor State is a system instruction -- for disassembly only.
// The singleton $opt operand contains the following information:
// opt{4-0} = mode from Inst{4-0}
// opt{5} = changemode from Inst{17}
// opt{8-6} = AIF from Inst{8-6}
// opt{10-9} = imod from Inst{19-18} with 0b10 as enable and 0b11 as disable
// FIXME: Integrated assembler will need these split out.
def CPS : AXI<(outs), (ins cps_opt:$opt), MiscFrm, NoItinerary, "cps$opt",
              [/* For disassembly only; pattern left blank */]>,
          Requires<[IsARM]> {
  let Inst{31-28} = 0b1111;
  let Inst{27-20} = 0b00010000;
  let Inst{16} = 0;
  let Inst{5} = 0;
}

// Preload signals the memory system of possible future data/instruction access.
// These are for disassembly only.
multiclass APreLoad<bits<1> read, bits<1> data, string opc> {

  def i12 : AXI<(outs), (ins addrmode_imm12:$addr), MiscFrm, IIC_Preload,
                !strconcat(opc, "\t$addr"),
                [(ARMPreload addrmode_imm12:$addr, (i32 read), (i32 data))]> {
    bits<4> Rt;
    bits<17> addr;
    let Inst{31-26} = 0b111101;
    let Inst{25} = 0; // 0 for immediate form
    let Inst{24} = data;
    let Inst{23} = addr{12};        // U (add = ('U' == 1))
    let Inst{22} = read;
    let Inst{21-20} = 0b01;
    let Inst{19-16} = addr{16-13};  // Rn
    let Inst{15-12} = Rt;
    let Inst{11-0}  = addr{11-0};   // imm12
  }

  def rs : AXI<(outs), (ins ldst_so_reg:$shift), MiscFrm, IIC_Preload,
               !strconcat(opc, "\t$shift"),
               [(ARMPreload ldst_so_reg:$shift, (i32 read), (i32 data))]> {
    bits<4> Rt;
    bits<17> shift;
    let Inst{31-26} = 0b111101;
    let Inst{25} = 1; // 1 for register form
    let Inst{24} = data;
    let Inst{23} = shift{12};    // U (add = ('U' == 1))
    let Inst{22} = read;
    let Inst{21-20} = 0b01;
    let Inst{19-16} = shift{16-13}; // Rn
    let Inst{11-0}  = shift{11-0};
  }
}

defm PLD  : APreLoad<1, 1, "pld">,  Requires<[IsARM]>;
defm PLDW : APreLoad<0, 1, "pldw">, Requires<[IsARM,HasV7,HasMP]>;
defm PLI  : APreLoad<1, 0, "pli">,  Requires<[IsARM,HasV7]>;

def SETEND : AXI<(outs),(ins setend_op:$end), MiscFrm, NoItinerary,
                 "setend\t$end",
                 [/* For disassembly only; pattern left blank */]>,
               Requires<[IsARM]> {
  bits<1> end;
  let Inst{31-10} = 0b1111000100000001000000;
  let Inst{9} = end;
  let Inst{8-0} = 0;
}

def DBG : AI<(outs), (ins i32imm:$opt), MiscFrm, NoItinerary, "dbg", "\t$opt",
             [/* For disassembly only; pattern left blank */]>,
          Requires<[IsARM, HasV7]> {
  bits<4> opt;
  let Inst{27-4} = 0b001100100000111100001111;
  let Inst{3-0} = opt;
}

// A5.4 Permanently UNDEFINED instructions.
let isBarrier = 1, isTerminator = 1 in
def TRAP : AXI<(outs), (ins), MiscFrm, NoItinerary,
               "trap", [(trap)]>,
           Requires<[IsARM]> {
  let Inst{27-25} = 0b011;
  let Inst{24-20} = 0b11111;
  let Inst{7-5} = 0b111;
  let Inst{4} = 0b1;
}

// Address computation and loads and stores in PIC mode.
// FIXME: These PIC insn patterns are pseudos, but derive from the normal insn
//        classes (AXI1, et.al.) and so have encoding information and such,
//        which is suboptimal. Once the rest of the code emitter (including
//        JIT) is MC-ized we should look at refactoring these into true
//        pseudos. As is, the encoding information ends up being ignored,
//        as these instructions are lowered to individual MC-insts.
let isNotDuplicable = 1 in {
def PICADD : AXI1<0b0100, (outs GPR:$dst), (ins GPR:$a, pclabel:$cp, pred:$p),
                  Pseudo, IIC_iALUr, "",
                   [(set GPR:$dst, (ARMpic_add GPR:$a, imm:$cp))]>;

let AddedComplexity = 10 in {
def PICLDR  : AXI2ldw<(outs GPR:$dst), (ins addrmodepc:$addr, pred:$p),
                  Pseudo, IIC_iLoad_r, "",
                  [(set GPR:$dst, (load addrmodepc:$addr))]>;

def PICLDRH : AXI3ldh<(outs GPR:$dst), (ins addrmodepc:$addr, pred:$p),
            Pseudo, IIC_iLoad_bh_r, "",
                  [(set GPR:$dst, (zextloadi16 addrmodepc:$addr))]>;

def PICLDRB : AXI2ldb<(outs GPR:$dst), (ins addrmodepc:$addr, pred:$p),
            Pseudo, IIC_iLoad_bh_r, "",
                  [(set GPR:$dst, (zextloadi8 addrmodepc:$addr))]>;

def PICLDRSH : AXI3ldsh<(outs GPR:$dst), (ins addrmodepc:$addr, pred:$p),
           Pseudo, IIC_iLoad_bh_r, "",
                  [(set GPR:$dst, (sextloadi16 addrmodepc:$addr))]>;

def PICLDRSB : AXI3ldsb<(outs GPR:$dst), (ins addrmodepc:$addr, pred:$p),
           Pseudo, IIC_iLoad_bh_r, "",
                  [(set GPR:$dst, (sextloadi8 addrmodepc:$addr))]>;
}
let AddedComplexity = 10 in {
def PICSTR  : AXI2stw<(outs), (ins GPR:$src, addrmodepc:$addr, pred:$p),
               Pseudo, IIC_iStore_r, "",
               [(store GPR:$src, addrmodepc:$addr)]>;

def PICSTRH : AXI3sth<(outs), (ins GPR:$src, addrmodepc:$addr, pred:$p),
           Pseudo, IIC_iStore_bh_r, "",
               [(truncstorei16 GPR:$src, addrmodepc:$addr)]>;

def PICSTRB : AXI2stb<(outs), (ins GPR:$src, addrmodepc:$addr, pred:$p),
           Pseudo, IIC_iStore_bh_r, "",
               [(truncstorei8 GPR:$src, addrmodepc:$addr)]>;
}
} // isNotDuplicable = 1


// LEApcrel - Load a pc-relative address into a register without offending the
// assembler.
// FIXME: These are marked as pseudos, but they're really not(?). They're just
// the ADR instruction. Is this the right way to handle that? They need
// encoding information regardless.
let neverHasSideEffects = 1 in {
let isReMaterializable = 1 in
def LEApcrel : AXI1<0x0, (outs GPR:$dst), (ins i32imm:$label, pred:$p),
                    Pseudo, IIC_iALUi,
                    "adr$p\t$dst, #$label", []>;

} // neverHasSideEffects
def LEApcrelJT : AXI1<0x0, (outs GPR:$dst),
                           (ins i32imm:$label, nohash_imm:$id, pred:$p),
                      Pseudo, IIC_iALUi,
                      "adr$p\t$dst, #${label}_${id}", []> {
    let Inst{25} = 1;
}

//===----------------------------------------------------------------------===//
//  Control Flow Instructions.
//

let isReturn = 1, isTerminator = 1, isBarrier = 1 in {
  // ARMV4T and above
  def BX_RET : AI<(outs), (ins), BrMiscFrm, IIC_Br,
                  "bx", "\tlr", [(ARMretflag)]>,
               Requires<[IsARM, HasV4T]> {
    let Inst{27-0}  = 0b0001001011111111111100011110;
  }

  // ARMV4 only
  def MOVPCLR : AI<(outs), (ins), BrMiscFrm, IIC_Br,
                  "mov", "\tpc, lr", [(ARMretflag)]>,
               Requires<[IsARM, NoV4T]> {
    let Inst{27-0} = 0b0001101000001111000000001110;
  }
}

// Indirect branches
let isBranch = 1, isTerminator = 1, isBarrier = 1, isIndirectBranch = 1 in {
  // ARMV4T and above
  def BRIND : AXI<(outs), (ins GPR:$dst), BrMiscFrm, IIC_Br, "bx\t$dst",
                  [(brind GPR:$dst)]>,
              Requires<[IsARM, HasV4T]> {
    bits<4> dst;
    let Inst{31-4} = 0b1110000100101111111111110001;
    let Inst{3-0}  = dst;
  }

  // ARMV4 only
  def MOVPCRX : AXI<(outs), (ins GPR:$dst), BrMiscFrm, IIC_Br, "mov\tpc, $dst",
                  [(brind GPR:$dst)]>,
              Requires<[IsARM, NoV4T]> {
    bits<4> dst;
    let Inst{31-4} = 0b1110000110100000111100000000;
    let Inst{3-0}   = dst;
  }
}

// FIXME: remove when we have a way to marking a MI with these properties.
// FIXME: Should pc be an implicit operand like PICADD, etc?
let isReturn = 1, isTerminator = 1, isBarrier = 1, mayLoad = 1,
    hasExtraDefRegAllocReq = 1, isCodeGenOnly = 1 in
  def LDM_RET : AXI4ld<(outs GPR:$wb), (ins GPR:$Rn, ldstm_mode:$mode, pred:$p,
                                        reglist:$dsts, variable_ops),
                       IndexModeUpd, LdStMulFrm, IIC_iLoad_mBr,
                       "ldm${mode}${p}\t$Rn!, $dsts",
                       "$Rn = $wb", []>;

// On non-Darwin platforms R9 is callee-saved.
let isCall = 1,
  Defs = [R0,  R1,  R2,  R3,  R12, LR,
          D0,  D1,  D2,  D3,  D4,  D5,  D6,  D7,
          D16, D17, D18, D19, D20, D21, D22, D23,
          D24, D25, D26, D27, D28, D29, D30, D31, CPSR, FPSCR] in {
  def BL  : ABXI<0b1011, (outs), (ins i32imm:$func, variable_ops),
                IIC_Br, "bl\t$func",
                [(ARMcall tglobaladdr:$func)]>,
            Requires<[IsARM, IsNotDarwin]> {
    let Inst{31-28} = 0b1110;
    // FIXME: Encoding info for $func. Needs fixups bits.
  }

  def BL_pred : ABI<0b1011, (outs), (ins i32imm:$func, variable_ops),
                   IIC_Br, "bl", "\t$func",
                   [(ARMcall_pred tglobaladdr:$func)]>,
                Requires<[IsARM, IsNotDarwin]>;

  // ARMv5T and above
  def BLX : AXI<(outs), (ins GPR:$func, variable_ops), BrMiscFrm,
                IIC_Br, "blx\t$func",
                [(ARMcall GPR:$func)]>,
            Requires<[IsARM, HasV5T, IsNotDarwin]> {
    bits<4> func;
    let Inst{27-4} = 0b000100101111111111110011;
    let Inst{3-0}   = func;
  }

  // ARMv4T
  // Note: Restrict $func to the tGPR regclass to prevent it being in LR.
  def BX : ABXIx2<(outs), (ins tGPR:$func, variable_ops),
                  IIC_Br, "mov\tlr, pc\n\tbx\t$func",
                  [(ARMcall_nolink tGPR:$func)]>,
           Requires<[IsARM, HasV4T, IsNotDarwin]> {
    bits<4> func;
    let Inst{27-4} = 0b000100101111111111110001;
    let Inst{3-0}   = func;
  }

  // ARMv4
  def BMOVPCRX : ABXIx2<(outs), (ins tGPR:$func, variable_ops),
                 IIC_Br, "mov\tlr, pc\n\tmov\tpc, $func",
                 [(ARMcall_nolink tGPR:$func)]>,
           Requires<[IsARM, NoV4T, IsNotDarwin]> {
    bits<4> func;
    let Inst{27-4} = 0b000110100000111100000000;
    let Inst{3-0}   = func;
  }
}

// On Darwin R9 is call-clobbered.
let isCall = 1,
  Defs = [R0,  R1,  R2,  R3,  R9,  R12, LR,
          D0,  D1,  D2,  D3,  D4,  D5,  D6,  D7,
          D16, D17, D18, D19, D20, D21, D22, D23,
          D24, D25, D26, D27, D28, D29, D30, D31, CPSR, FPSCR] in {
  def BLr9  : ABXI<0b1011, (outs), (ins i32imm:$func, variable_ops),
                IIC_Br, "bl\t$func",
                [(ARMcall tglobaladdr:$func)]>, Requires<[IsARM, IsDarwin]> {
    let Inst{31-28} = 0b1110;
    // FIXME: Encoding info for $func. Needs fixups bits.
  }

  def BLr9_pred : ABI<0b1011, (outs), (ins i32imm:$func, variable_ops),
                   IIC_Br, "bl", "\t$func",
                   [(ARMcall_pred tglobaladdr:$func)]>,
                  Requires<[IsARM, IsDarwin]>;

  // ARMv5T and above
  def BLXr9 : AXI<(outs), (ins GPR:$func, variable_ops), BrMiscFrm,
                IIC_Br, "blx\t$func",
                [(ARMcall GPR:$func)]>, Requires<[IsARM, HasV5T, IsDarwin]> {
    bits<4> func;
    let Inst{27-4} = 0b000100101111111111110011;
    let Inst{3-0}   = func;
  }

  // ARMv4T
  // Note: Restrict $func to the tGPR regclass to prevent it being in LR.
  def BXr9 : ABXIx2<(outs), (ins tGPR:$func, variable_ops),
                  IIC_Br, "mov\tlr, pc\n\tbx\t$func",
                  [(ARMcall_nolink tGPR:$func)]>,
             Requires<[IsARM, HasV4T, IsDarwin]> {
    bits<4> func;
    let Inst{27-4} = 0b000100101111111111110001;
    let Inst{3-0}   = func;
  }

  // ARMv4
  def BMOVPCRXr9 : ABXIx2<(outs), (ins tGPR:$func, variable_ops),
                 IIC_Br, "mov\tlr, pc\n\tmov\tpc, $func",
                 [(ARMcall_nolink tGPR:$func)]>,
           Requires<[IsARM, NoV4T, IsDarwin]> {
    bits<4> func;
    let Inst{27-4} = 0b000110100000111100000000;
    let Inst{3-0}   = func;
  }
}

// Tail calls.

// FIXME: These should probably be xformed into the non-TC versions of the
// instructions as part of MC lowering.
let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1 in {
  // Darwin versions.
  let Defs = [R0, R1, R2, R3, R9, R12,
              D0, D1, D2, D3, D4, D5, D6, D7,
              D16, D17, D18, D19, D20, D21, D22, D23, D24, D25, D26,
              D27, D28, D29, D30, D31, PC],
      Uses = [SP] in {
    def TCRETURNdi : AInoP<(outs), (ins i32imm:$dst, variable_ops),
                       Pseudo, IIC_Br,
                       "@TC_RETURN","\t$dst", []>, Requires<[IsDarwin]>;

    def TCRETURNri : AInoP<(outs), (ins tcGPR:$dst, variable_ops),
                       Pseudo, IIC_Br,
                       "@TC_RETURN","\t$dst", []>, Requires<[IsDarwin]>;

    def TAILJMPd : ABXI<0b1010, (outs), (ins brtarget:$dst, variable_ops),
                   IIC_Br, "b\t$dst  @ TAILCALL",
                   []>, Requires<[IsDarwin]>;

    def TAILJMPdt: ABXI<0b1010, (outs), (ins brtarget:$dst, variable_ops),
                   IIC_Br, "b.w\t$dst  @ TAILCALL",
                   []>, Requires<[IsDarwin]>;

    def TAILJMPr : AXI<(outs), (ins tcGPR:$dst, variable_ops),
                     BrMiscFrm, IIC_Br, "bx\t$dst  @ TAILCALL",
                   []>, Requires<[IsDarwin]> {
      bits<4> dst;
      let Inst{31-4} = 0b1110000100101111111111110001;
      let Inst{3-0}  = dst;
    }
  }

  // Non-Darwin versions (the difference is R9).
  let Defs = [R0, R1, R2, R3, R12,
              D0, D1, D2, D3, D4, D5, D6, D7,
              D16, D17, D18, D19, D20, D21, D22, D23, D24, D25, D26,
              D27, D28, D29, D30, D31, PC],
      Uses = [SP] in {
    def TCRETURNdiND : AInoP<(outs), (ins i32imm:$dst, variable_ops),
                       Pseudo, IIC_Br,
                       "@TC_RETURN","\t$dst", []>, Requires<[IsNotDarwin]>;

    def TCRETURNriND : AInoP<(outs), (ins tcGPR:$dst, variable_ops),
                       Pseudo, IIC_Br,
                       "@TC_RETURN","\t$dst", []>, Requires<[IsNotDarwin]>;

    def TAILJMPdND : ABXI<0b1010, (outs), (ins brtarget:$dst, variable_ops),
                   IIC_Br, "b\t$dst  @ TAILCALL",
                   []>, Requires<[IsARM, IsNotDarwin]>;

    def TAILJMPdNDt : ABXI<0b1010, (outs), (ins brtarget:$dst, variable_ops),
                   IIC_Br, "b.w\t$dst  @ TAILCALL",
                   []>, Requires<[IsThumb, IsNotDarwin]>;

    def TAILJMPrND : AXI<(outs), (ins tcGPR:$dst, variable_ops),
                     BrMiscFrm, IIC_Br, "bx\t$dst  @ TAILCALL",
                   []>, Requires<[IsNotDarwin]> {
      bits<4> dst;
      let Inst{31-4} = 0b1110000100101111111111110001;
      let Inst{3-0}  = dst;
    }
  }
}

let isBranch = 1, isTerminator = 1 in {
  // B is "predicable" since it can be xformed into a Bcc.
  let isBarrier = 1 in {
    let isPredicable = 1 in
    def B : ABXI<0b1010, (outs), (ins brtarget:$target), IIC_Br,
                "b\t$target", [(br bb:$target)]>;

    let isNotDuplicable = 1, isIndirectBranch = 1,
        // FIXME: $imm field is not specified by asm string.  Mark as cgonly.
        isCodeGenOnly = 1 in {
    def BR_JTr : JTI<(outs), (ins GPR:$target, jtblock_operand:$jt, i32imm:$id),
                      IIC_Br, "mov\tpc, $target$jt",
                      [(ARMbrjt GPR:$target, tjumptable:$jt, imm:$id)]> {
      let Inst{11-4}  = 0b00000000;
      let Inst{15-12} = 0b1111;
      let Inst{20}    = 0; // S Bit
      let Inst{24-21} = 0b1101;
      let Inst{27-25} = 0b000;
    }
    def BR_JTm : JTI<(outs),
                     (ins addrmode2:$target, jtblock_operand:$jt, i32imm:$id),
                     IIC_Br, "ldr\tpc, $target$jt",
                     [(ARMbrjt (i32 (load addrmode2:$target)), tjumptable:$jt,
                       imm:$id)]> {
      let Inst{15-12} = 0b1111;
      let Inst{20}    = 1; // L bit
      let Inst{21}    = 0; // W bit
      let Inst{22}    = 0; // B bit
      let Inst{24}    = 1; // P bit
      let Inst{27-25} = 0b011;
    }
    def BR_JTadd : JTI<(outs),
                     (ins GPR:$target, GPR:$idx, jtblock_operand:$jt, i32imm:$id),
                      IIC_Br, "add\tpc, $target, $idx$jt",
                      [(ARMbrjt (add GPR:$target, GPR:$idx), tjumptable:$jt,
                        imm:$id)]> {
      let Inst{15-12} = 0b1111;
      let Inst{20}    = 0; // S bit
      let Inst{24-21} = 0b0100;
      let Inst{27-25} = 0b000;
    }
    } // isNotDuplicable = 1, isIndirectBranch = 1
  } // isBarrier = 1

  // FIXME: should be able to write a pattern for ARMBrcond, but can't use
  // a two-value operand where a dag node expects two operands. :(
  def Bcc : ABI<0b1010, (outs), (ins brtarget:$target),
               IIC_Br, "b", "\t$target",
               [/*(ARMbrcond bb:$target, imm:$cc, CCR:$ccr)*/]>;
}

// Branch and Exchange Jazelle -- for disassembly only
def BXJ : ABI<0b0001, (outs), (ins GPR:$func), NoItinerary, "bxj", "\t$func",
              [/* For disassembly only; pattern left blank */]> {
  let Inst{23-20} = 0b0010;
  //let Inst{19-8} = 0xfff;
  let Inst{7-4} = 0b0010;
}

// Secure Monitor Call is a system instruction -- for disassembly only
def SMC : ABI<0b0001, (outs), (ins i32imm:$opt), NoItinerary, "smc", "\t$opt",
              [/* For disassembly only; pattern left blank */]> {
  bits<4> opt;
  let Inst{23-4} = 0b01100000000000000111;
  let Inst{3-0} = opt;
}

// Supervisor Call (Software Interrupt) -- for disassembly only
let isCall = 1 in {
def SVC : ABI<0b1111, (outs), (ins i32imm:$svc), IIC_Br, "svc", "\t$svc",
              [/* For disassembly only; pattern left blank */]> {
  bits<24> svc;
  let Inst{23-0} = svc;
}
}

// Store Return State is a system instruction -- for disassembly only
let isCodeGenOnly = 1 in {  // FIXME: This should not use submode!
def SRSW : ABXI<{1,0,0,?}, (outs), (ins ldstm_mode:$amode, i32imm:$mode),
                NoItinerary, "srs${amode}\tsp!, $mode",
                [/* For disassembly only; pattern left blank */]> {
  let Inst{31-28} = 0b1111;
  let Inst{22-20} = 0b110; // W = 1
}

def SRS  : ABXI<{1,0,0,?}, (outs), (ins ldstm_mode:$amode, i32imm:$mode),
                NoItinerary, "srs${amode}\tsp, $mode",
                [/* For disassembly only; pattern left blank */]> {
  let Inst{31-28} = 0b1111;
  let Inst{22-20} = 0b100; // W = 0
}

// Return From Exception is a system instruction -- for disassembly only
def RFEW : ABXI<{1,0,0,?}, (outs), (ins ldstm_mode:$amode, GPR:$base),
                NoItinerary, "rfe${amode}\t$base!",
                [/* For disassembly only; pattern left blank */]> {
  let Inst{31-28} = 0b1111;
  let Inst{22-20} = 0b011; // W = 1
}

def RFE  : ABXI<{1,0,0,?}, (outs), (ins ldstm_mode:$amode, GPR:$base),
                NoItinerary, "rfe${amode}\t$base",
                [/* For disassembly only; pattern left blank */]> {
  let Inst{31-28} = 0b1111;
  let Inst{22-20} = 0b001; // W = 0
}
} // isCodeGenOnly = 1

//===----------------------------------------------------------------------===//
//  Load / store Instructions.
//

// Load


defm LDR  : AI_ldr1<0, "ldr", IIC_iLoad_r, IIC_iLoad_si,
                    UnOpFrag<(load node:$Src)>>;
defm LDRB : AI_ldr1<1, "ldrb", IIC_iLoad_bh_r, IIC_iLoad_bh_si,
                    UnOpFrag<(zextloadi8 node:$Src)>>;
defm STR  : AI_str1<0, "str", IIC_iStore_r, IIC_iStore_si,
                   BinOpFrag<(store node:$LHS, node:$RHS)>>;
defm STRB : AI_str1<1, "strb", IIC_iStore_bh_r, IIC_iStore_bh_si,
                   BinOpFrag<(truncstorei8 node:$LHS, node:$RHS)>>;

// Special LDR for loads from non-pc-relative constpools.
let canFoldAsLoad = 1, mayLoad = 1, neverHasSideEffects = 1,
    isReMaterializable = 1 in
def LDRcp : AIldst1<0b010, 0, 1, (outs GPR:$Rt), (ins addrmode_imm12:$addr),
                 AddrMode_i12, LdFrm, IIC_iLoad_r, "ldr", "\t$Rt, $addr", []> {
  bits<4> Rt;
  bits<17> addr;
  let Inst{23}    = addr{12};     // U (add = ('U' == 1))
  let Inst{19-16} = 0b1111;
  let Inst{15-12} = Rt;
  let Inst{11-0}  = addr{11-0};   // imm12
}

// Loads with zero extension
def LDRH  : AI3ldh<(outs GPR:$dst), (ins addrmode3:$addr), LdMiscFrm,
                  IIC_iLoad_bh_r, "ldrh", "\t$dst, $addr",
                  [(set GPR:$dst, (zextloadi16 addrmode3:$addr))]>;

// Loads with sign extension
def LDRSH : AI3ldsh<(outs GPR:$dst), (ins addrmode3:$addr), LdMiscFrm,
                   IIC_iLoad_bh_r, "ldrsh", "\t$dst, $addr",
                   [(set GPR:$dst, (sextloadi16 addrmode3:$addr))]>;

def LDRSB : AI3ldsb<(outs GPR:$dst), (ins addrmode3:$addr), LdMiscFrm,
                   IIC_iLoad_bh_r, "ldrsb", "\t$dst, $addr",
                   [(set GPR:$dst, (sextloadi8 addrmode3:$addr))]>;

let mayLoad = 1, neverHasSideEffects = 1, hasExtraDefRegAllocReq = 1,
    isCodeGenOnly = 1 in { // $dst2 doesn't exist in asmstring?
// Load doubleword
def LDRD : AI3ldd<(outs GPR:$dst1, GPR:$dst2), (ins addrmode3:$addr), LdMiscFrm,
                 IIC_iLoad_d_r, "ldrd", "\t$dst1, $addr",
                 []>, Requires<[IsARM, HasV5TE]>;

// Indexed loads
def LDR_PRE  : AI2ldwpr<(outs GPR:$dst, GPR:$base_wb),
                     (ins addrmode2:$addr), LdFrm, IIC_iLoad_ru,
                     "ldr", "\t$dst, $addr!", "$addr.base = $base_wb", []>;

def LDR_POST : AI2ldwpo<(outs GPR:$dst, GPR:$base_wb),
                     (ins GPR:$base, am2offset:$offset), LdFrm, IIC_iLoad_ru,
                     "ldr", "\t$dst, [$base], $offset", "$base = $base_wb", []>;

def LDRH_PRE  : AI3ldhpr<(outs GPR:$dst, GPR:$base_wb),
                     (ins addrmode3:$addr), LdMiscFrm, IIC_iLoad_bh_ru,
                     "ldrh", "\t$dst, $addr!", "$addr.base = $base_wb", []>;

def LDRH_POST : AI3ldhpo<(outs GPR:$dst, GPR:$base_wb),
                  (ins GPR:$base,am3offset:$offset), LdMiscFrm, IIC_iLoad_bh_ru,
                    "ldrh", "\t$dst, [$base], $offset", "$base = $base_wb", []>;

def LDRB_PRE  : AI2ldbpr<(outs GPR:$dst, GPR:$base_wb),
                     (ins addrmode2:$addr), LdFrm, IIC_iLoad_bh_ru,
                     "ldrb", "\t$dst, $addr!", "$addr.base = $base_wb", []>;

def LDRB_POST : AI2ldbpo<(outs GPR:$dst, GPR:$base_wb),
                     (ins GPR:$base,am2offset:$offset), LdFrm, IIC_iLoad_bh_ru,
                    "ldrb", "\t$dst, [$base], $offset", "$base = $base_wb", []>;

def LDRSH_PRE : AI3ldshpr<(outs GPR:$dst, GPR:$base_wb),
                      (ins addrmode3:$addr), LdMiscFrm, IIC_iLoad_bh_ru,
                      "ldrsh", "\t$dst, $addr!", "$addr.base = $base_wb", []>;

def LDRSH_POST: AI3ldshpo<(outs GPR:$dst, GPR:$base_wb),
                  (ins GPR:$base,am3offset:$offset), LdMiscFrm, IIC_iLoad_bh_ru,
                   "ldrsh", "\t$dst, [$base], $offset", "$base = $base_wb", []>;

def LDRSB_PRE : AI3ldsbpr<(outs GPR:$dst, GPR:$base_wb),
                      (ins addrmode3:$addr), LdMiscFrm, IIC_iLoad_bh_ru,
                      "ldrsb", "\t$dst, $addr!", "$addr.base = $base_wb", []>;

def LDRSB_POST: AI3ldsbpo<(outs GPR:$dst, GPR:$base_wb),
                    (ins GPR:$base,am3offset:$offset), LdMiscFrm, IIC_iLoad_ru,
                   "ldrsb", "\t$dst, [$base], $offset", "$base = $base_wb", []>;

// For disassembly only
def LDRD_PRE : AI3lddpr<(outs GPR:$dst1, GPR:$dst2, GPR:$base_wb),
                        (ins addrmode3:$addr), LdMiscFrm, IIC_iLoad_d_ru,
                 "ldrd", "\t$dst1, $dst2, $addr!", "$addr.base = $base_wb", []>,
                Requires<[IsARM, HasV5TE]>;

// For disassembly only
def LDRD_POST : AI3lddpo<(outs GPR:$dst1, GPR:$dst2, GPR:$base_wb),
                   (ins GPR:$base,am3offset:$offset), LdMiscFrm, IIC_iLoad_d_ru,
            "ldrd", "\t$dst1, $dst2, [$base], $offset", "$base = $base_wb", []>,
                Requires<[IsARM, HasV5TE]>;

} // mayLoad = 1, neverHasSideEffects = 1, hasExtraDefRegAllocReq = 1

// LDRT, LDRBT, LDRSBT, LDRHT, LDRSHT are for disassembly only.

def LDRT : AI2ldwpo<(outs GPR:$dst, GPR:$base_wb),
                   (ins GPR:$base, am2offset:$offset), LdFrm, IIC_iLoad_ru,
                   "ldrt", "\t$dst, [$base], $offset", "$base = $base_wb", []> {
  let Inst{21} = 1; // overwrite
}

def LDRBT : AI2ldbpo<(outs GPR:$dst, GPR:$base_wb),
                  (ins GPR:$base,am2offset:$offset), LdFrm, IIC_iLoad_bh_ru,
                  "ldrbt", "\t$dst, [$base], $offset", "$base = $base_wb", []> {
  let Inst{21} = 1; // overwrite
}

def LDRSBT : AI3ldsbpo<(outs GPR:$dst, GPR:$base_wb),
                 (ins GPR:$base,am3offset:$offset), LdMiscFrm, IIC_iLoad_bh_ru,
                 "ldrsbt", "\t$dst, [$base], $offset", "$base = $base_wb", []> {
  let Inst{21} = 1; // overwrite
}

def LDRHT : AI3ldhpo<(outs GPR:$dst, GPR:$base_wb),
                 (ins GPR:$base, am3offset:$offset), LdMiscFrm, IIC_iLoad_bh_ru,
                  "ldrht", "\t$dst, [$base], $offset", "$base = $base_wb", []> {
  let Inst{21} = 1; // overwrite
}

def LDRSHT : AI3ldshpo<(outs GPR:$dst, GPR:$base_wb),
                 (ins GPR:$base,am3offset:$offset), LdMiscFrm, IIC_iLoad_bh_ru,
                 "ldrsht", "\t$dst, [$base], $offset", "$base = $base_wb", []> {
  let Inst{21} = 1; // overwrite
}

// Store

// Stores with truncate
def STRH : AI3sth<(outs), (ins GPR:$src, addrmode3:$addr), StMiscFrm,
               IIC_iStore_bh_r, "strh", "\t$src, $addr",
               [(truncstorei16 GPR:$src, addrmode3:$addr)]>;

// Store doubleword
let mayStore = 1, neverHasSideEffects = 1, hasExtraSrcRegAllocReq = 1,
    isCodeGenOnly = 1 in  // $src2 doesn't exist in asm string
def STRD : AI3std<(outs), (ins GPR:$src1, GPR:$src2, addrmode3:$addr),
               StMiscFrm, IIC_iStore_d_r,
               "strd", "\t$src1, $addr", []>, Requires<[IsARM, HasV5TE]>;

// Indexed stores
def STR_PRE  : AI2stwpr<(outs GPR:$base_wb),
                     (ins GPR:$src, GPR:$base, am2offset:$offset),
                     StFrm, IIC_iStore_ru,
                    "str", "\t$src, [$base, $offset]!", "$base = $base_wb",
                    [(set GPR:$base_wb,
                      (pre_store GPR:$src, GPR:$base, am2offset:$offset))]>;

def STR_POST : AI2stwpo<(outs GPR:$base_wb),
                     (ins GPR:$src, GPR:$base,am2offset:$offset),
                     StFrm, IIC_iStore_ru,
                    "str", "\t$src, [$base], $offset", "$base = $base_wb",
                    [(set GPR:$base_wb,
                      (post_store GPR:$src, GPR:$base, am2offset:$offset))]>;

def STRH_PRE : AI3sthpr<(outs GPR:$base_wb),
                     (ins GPR:$src, GPR:$base,am3offset:$offset),
                     StMiscFrm, IIC_iStore_ru,
                     "strh", "\t$src, [$base, $offset]!", "$base = $base_wb",
                    [(set GPR:$base_wb,
                      (pre_truncsti16 GPR:$src, GPR:$base,am3offset:$offset))]>;

def STRH_POST: AI3sthpo<(outs GPR:$base_wb),
                     (ins GPR:$src, GPR:$base,am3offset:$offset),
                     StMiscFrm, IIC_iStore_bh_ru,
                     "strh", "\t$src, [$base], $offset", "$base = $base_wb",
                    [(set GPR:$base_wb, (post_truncsti16 GPR:$src,
                                         GPR:$base, am3offset:$offset))]>;

def STRB_PRE : AI2stbpr<(outs GPR:$base_wb),
                     (ins GPR:$src, GPR:$base,am2offset:$offset),
                     StFrm, IIC_iStore_bh_ru,
                     "strb", "\t$src, [$base, $offset]!", "$base = $base_wb",
                    [(set GPR:$base_wb, (pre_truncsti8 GPR:$src,
                                         GPR:$base, am2offset:$offset))]>;

def STRB_POST: AI2stbpo<(outs GPR:$base_wb),
                     (ins GPR:$src, GPR:$base,am2offset:$offset),
                     StFrm, IIC_iStore_bh_ru,
                     "strb", "\t$src, [$base], $offset", "$base = $base_wb",
                    [(set GPR:$base_wb, (post_truncsti8 GPR:$src,
                                         GPR:$base, am2offset:$offset))]>;

// For disassembly only
def STRD_PRE : AI3stdpr<(outs GPR:$base_wb),
                     (ins GPR:$src1, GPR:$src2, GPR:$base, am3offset:$offset),
                     StMiscFrm, IIC_iStore_d_ru,
                     "strd", "\t$src1, $src2, [$base, $offset]!",
                     "$base = $base_wb", []>;

// For disassembly only
def STRD_POST: AI3stdpo<(outs GPR:$base_wb),
                     (ins GPR:$src1, GPR:$src2, GPR:$base, am3offset:$offset),
                     StMiscFrm, IIC_iStore_d_ru,
                     "strd", "\t$src1, $src2, [$base], $offset",
                     "$base = $base_wb", []>;

// STRT, STRBT, and STRHT are for disassembly only.

def STRT : AI2stwpo<(outs GPR:$base_wb),
                    (ins GPR:$src, GPR:$base,am2offset:$offset),
                    StFrm, IIC_iStore_ru,
                    "strt", "\t$src, [$base], $offset", "$base = $base_wb",
                    [/* For disassembly only; pattern left blank */]> {
  let Inst{21} = 1; // overwrite
}

def STRBT : AI2stbpo<(outs GPR:$base_wb),
                     (ins GPR:$src, GPR:$base,am2offset:$offset),
                     StFrm, IIC_iStore_bh_ru,
                     "strbt", "\t$src, [$base], $offset", "$base = $base_wb",
                     [/* For disassembly only; pattern left blank */]> {
  let Inst{21} = 1; // overwrite
}

def STRHT: AI3sthpo<(outs GPR:$base_wb),
                    (ins GPR:$src, GPR:$base,am3offset:$offset),
                    StMiscFrm, IIC_iStore_bh_ru,
                    "strht", "\t$src, [$base], $offset", "$base = $base_wb",
                    [/* For disassembly only; pattern left blank */]> {
  let Inst{21} = 1; // overwrite
}

//===----------------------------------------------------------------------===//
//  Load / store multiple Instructions.
//

let mayLoad = 1, neverHasSideEffects = 1, hasExtraDefRegAllocReq = 1,
    isCodeGenOnly = 1 in {
def LDM : AXI4ld<(outs), (ins GPR:$Rn, ldstm_mode:$amode, pred:$p,
                          reglist:$dsts, variable_ops),
                 IndexModeNone, LdStMulFrm, IIC_iLoad_m,
                 "ldm${amode}${p}\t$Rn, $dsts", "", []>;

def LDM_UPD : AXI4ld<(outs GPR:$wb), (ins GPR:$Rn, ldstm_mode:$amode, pred:$p,
                                      reglist:$dsts, variable_ops),
                     IndexModeUpd, LdStMulFrm, IIC_iLoad_mu,
                     "ldm${amode}${p}\t$Rn!, $dsts",
                     "$Rn = $wb", []>;
} // mayLoad, neverHasSideEffects, hasExtraDefRegAllocReq

let mayStore = 1, neverHasSideEffects = 1, hasExtraSrcRegAllocReq = 1,
    isCodeGenOnly = 1 in {
def STM : AXI4st<(outs), (ins GPR:$Rn, ldstm_mode:$amode, pred:$p,
                          reglist:$srcs, variable_ops),
                 IndexModeNone, LdStMulFrm, IIC_iStore_m,
                 "stm${amode}${p}\t$Rn, $srcs", "", []>;

def STM_UPD : AXI4st<(outs GPR:$wb), (ins GPR:$Rn, ldstm_mode:$amode, pred:$p,
                                      reglist:$srcs, variable_ops),
                     IndexModeUpd, LdStMulFrm, IIC_iStore_mu,
                     "stm${amode}${p}\t$Rn!, $srcs",
                     "$Rn = $wb", []>;
} // mayStore, neverHasSideEffects, hasExtraSrcRegAllocReq

//===----------------------------------------------------------------------===//
//  Move Instructions.
//

let neverHasSideEffects = 1 in
def MOVr : AsI1<0b1101, (outs GPR:$Rd), (ins GPR:$Rm), DPFrm, IIC_iMOVr,
                "mov", "\t$Rd, $Rm", []>, UnaryDP {
  bits<4> Rd;
  bits<4> Rm;

  let Inst{11-4} = 0b00000000;
  let Inst{25} = 0;
  let Inst{3-0} = Rm;
  let Inst{15-12} = Rd;
}

// A version for the smaller set of tail call registers.
let neverHasSideEffects = 1 in
def MOVr_TC : AsI1<0b1101, (outs tcGPR:$Rd), (ins tcGPR:$Rm), DPFrm,
                IIC_iMOVr, "mov", "\t$Rd, $Rm", []>, UnaryDP {
  bits<4> Rd;
  bits<4> Rm;

  let Inst{11-4} = 0b00000000;
  let Inst{25} = 0;
  let Inst{3-0} = Rm;
  let Inst{15-12} = Rd;
}

def MOVs : AsI1<0b1101, (outs GPR:$Rd), (ins shift_so_reg:$src),
                DPSoRegFrm, IIC_iMOVsr,
                "mov", "\t$Rd, $src", [(set GPR:$Rd, shift_so_reg:$src)]>,
                UnaryDP {
  bits<4> Rd;
  bits<12> src;
  let Inst{15-12} = Rd;
  let Inst{11-0} = src;
  let Inst{25} = 0;
}

let isReMaterializable = 1, isAsCheapAsAMove = 1 in
def MOVi : AsI1<0b1101, (outs GPR:$Rd), (ins so_imm:$imm), DPFrm, IIC_iMOVi,
                "mov", "\t$Rd, $imm", [(set GPR:$Rd, so_imm:$imm)]>, UnaryDP {
  bits<4> Rd;
  bits<12> imm;
  let Inst{25} = 1;
  let Inst{15-12} = Rd;
  let Inst{19-16} = 0b0000;
  let Inst{11-0} = imm;
}

let isReMaterializable = 1, isAsCheapAsAMove = 1 in
def MOVi16 : AI1<0b1000, (outs GPR:$Rd), (ins i32imm:$imm),
                 DPFrm, IIC_iMOVi,
                 "movw", "\t$Rd, $imm",
                 [(set GPR:$Rd, imm0_65535:$imm)]>,
                 Requires<[IsARM, HasV6T2]>, UnaryDP {
  bits<4> Rd;
  bits<16> imm;
  let Inst{15-12} = Rd;
  let Inst{11-0}  = imm{11-0};
  let Inst{19-16} = imm{15-12};
  let Inst{20} = 0;
  let Inst{25} = 1;
}

let Constraints = "$src = $Rd" in
def MOVTi16 : AI1<0b1010, (outs GPR:$Rd), (ins GPR:$src, i32imm:$imm),
                  DPFrm, IIC_iMOVi,
                  "movt", "\t$Rd, $imm",
                  [(set GPR:$Rd,
                        (or (and GPR:$src, 0xffff),
                            lo16AllZero:$imm))]>, UnaryDP,
                  Requires<[IsARM, HasV6T2]> {
  bits<4> Rd;
  bits<16> imm;
  let Inst{15-12} = Rd;
  let Inst{11-0}  = imm{11-0};
  let Inst{19-16} = imm{15-12};
  let Inst{20} = 0;
  let Inst{25} = 1;
}

def : ARMPat<(or GPR:$src, 0xffff0000), (MOVTi16 GPR:$src, 0xffff)>,
      Requires<[IsARM, HasV6T2]>;

let Uses = [CPSR] in
def RRX: PseudoInst<(outs GPR:$Rd), (ins GPR:$Rm), IIC_iMOVsi, "",
                    [(set GPR:$Rd, (ARMrrx GPR:$Rm))]>, UnaryDP,
                    Requires<[IsARM]>;

// These aren't really mov instructions, but we have to define them this way
// due to flag operands.

let Defs = [CPSR] in {
def MOVsrl_flag : PseudoInst<(outs GPR:$dst), (ins GPR:$src), IIC_iMOVsi, "",
                      [(set GPR:$dst, (ARMsrl_flag GPR:$src))]>, UnaryDP,
                      Requires<[IsARM]>;
def MOVsra_flag : PseudoInst<(outs GPR:$dst), (ins GPR:$src), IIC_iMOVsi, "",
                      [(set GPR:$dst, (ARMsra_flag GPR:$src))]>, UnaryDP,
                      Requires<[IsARM]>;
}

//===----------------------------------------------------------------------===//
//  Extend Instructions.
//

// Sign extenders

defm SXTB  : AI_ext_rrot<0b01101010,
                         "sxtb", UnOpFrag<(sext_inreg node:$Src, i8)>>;
defm SXTH  : AI_ext_rrot<0b01101011,
                         "sxth", UnOpFrag<(sext_inreg node:$Src, i16)>>;

defm SXTAB : AI_exta_rrot<0b01101010,
               "sxtab", BinOpFrag<(add node:$LHS, (sext_inreg node:$RHS, i8))>>;
defm SXTAH : AI_exta_rrot<0b01101011,
               "sxtah", BinOpFrag<(add node:$LHS, (sext_inreg node:$RHS,i16))>>;

// For disassembly only
defm SXTB16  : AI_ext_rrot_np<0b01101000, "sxtb16">;

// For disassembly only
defm SXTAB16 : AI_exta_rrot_np<0b01101000, "sxtab16">;

// Zero extenders

let AddedComplexity = 16 in {
defm UXTB   : AI_ext_rrot<0b01101110,
                          "uxtb"  , UnOpFrag<(and node:$Src, 0x000000FF)>>;
defm UXTH   : AI_ext_rrot<0b01101111,
                          "uxth"  , UnOpFrag<(and node:$Src, 0x0000FFFF)>>;
defm UXTB16 : AI_ext_rrot<0b01101100,
                          "uxtb16", UnOpFrag<(and node:$Src, 0x00FF00FF)>>;

// FIXME: This pattern incorrectly assumes the shl operator is a rotate.
//        The transformation should probably be done as a combiner action
//        instead so we can include a check for masking back in the upper
//        eight bits of the source into the lower eight bits of the result.
//def : ARMV6Pat<(and (shl GPR:$Src, (i32 8)), 0xFF00FF),
//               (UXTB16r_rot GPR:$Src, 24)>;
def : ARMV6Pat<(and (srl GPR:$Src, (i32 8)), 0xFF00FF),
               (UXTB16r_rot GPR:$Src, 8)>;

defm UXTAB : AI_exta_rrot<0b01101110, "uxtab",
                        BinOpFrag<(add node:$LHS, (and node:$RHS, 0x00FF))>>;
defm UXTAH : AI_exta_rrot<0b01101111, "uxtah",
                        BinOpFrag<(add node:$LHS, (and node:$RHS, 0xFFFF))>>;
}

// This isn't safe in general, the add is two 16-bit units, not a 32-bit add.
// For disassembly only
defm UXTAB16 : AI_exta_rrot_np<0b01101100, "uxtab16">;


def SBFX  : I<(outs GPR:$Rd),
              (ins GPR:$Rn, imm0_31:$lsb, imm0_31_m1:$width),
               AddrMode1, Size4Bytes, IndexModeNone, DPFrm, IIC_iUNAsi,
               "sbfx", "\t$Rd, $Rn, $lsb, $width", "", []>,
               Requires<[IsARM, HasV6T2]> {
  bits<4> Rd;
  bits<4> Rn;
  bits<5> lsb;
  bits<5> width;
  let Inst{27-21} = 0b0111101;
  let Inst{6-4}   = 0b101;
  let Inst{20-16} = width;
  let Inst{15-12} = Rd;
  let Inst{11-7}  = lsb;
  let Inst{3-0}   = Rn;
}

def UBFX  : I<(outs GPR:$Rd),
              (ins GPR:$Rn, imm0_31:$lsb, imm0_31_m1:$width),
               AddrMode1, Size4Bytes, IndexModeNone, DPFrm, IIC_iUNAsi,
               "ubfx", "\t$Rd, $Rn, $lsb, $width", "", []>,
               Requires<[IsARM, HasV6T2]> {
  bits<4> Rd;
  bits<4> Rn;
  bits<5> lsb;
  bits<5> width;
  let Inst{27-21} = 0b0111111;
  let Inst{6-4}   = 0b101;
  let Inst{20-16} = width;
  let Inst{15-12} = Rd;
  let Inst{11-7}  = lsb;
  let Inst{3-0}   = Rn;
}

//===----------------------------------------------------------------------===//
//  Arithmetic Instructions.
//

defm ADD  : AsI1_bin_irs<0b0100, "add",
                         IIC_iALUi, IIC_iALUr, IIC_iALUsr,
                         BinOpFrag<(add  node:$LHS, node:$RHS)>, 1>;
defm SUB  : AsI1_bin_irs<0b0010, "sub",
                         IIC_iALUi, IIC_iALUr, IIC_iALUsr,
                         BinOpFrag<(sub  node:$LHS, node:$RHS)>>;

// ADD and SUB with 's' bit set.
defm ADDS : AI1_bin_s_irs<0b0100, "adds",
                          IIC_iALUi, IIC_iALUr, IIC_iALUsr,
                          BinOpFrag<(addc node:$LHS, node:$RHS)>, 1>;
defm SUBS : AI1_bin_s_irs<0b0010, "subs",
                          IIC_iALUi, IIC_iALUr, IIC_iALUsr,
                          BinOpFrag<(subc node:$LHS, node:$RHS)>>;

defm ADC : AI1_adde_sube_irs<0b0101, "adc",
                          BinOpFrag<(adde_dead_carry node:$LHS, node:$RHS)>, 1>;
defm SBC : AI1_adde_sube_irs<0b0110, "sbc",
                          BinOpFrag<(sube_dead_carry node:$LHS, node:$RHS)>>;
defm ADCS : AI1_adde_sube_s_irs<0b0101, "adcs",
                          BinOpFrag<(adde_live_carry node:$LHS, node:$RHS)>, 1>;
defm SBCS : AI1_adde_sube_s_irs<0b0110, "sbcs",
                          BinOpFrag<(sube_live_carry node:$LHS, node:$RHS) >>;

def RSBri : AsI1<0b0011, (outs GPR:$Rd), (ins GPR:$Rn, so_imm:$imm), DPFrm,
                 IIC_iALUi, "rsb", "\t$Rd, $Rn, $imm",
                 [(set GPR:$Rd, (sub so_imm:$imm, GPR:$Rn))]> {
  bits<4> Rd;
  bits<4> Rn;
  bits<12> imm;
  let Inst{25} = 1;
  let Inst{15-12} = Rd;
  let Inst{19-16} = Rn;
  let Inst{11-0} = imm;
}

// The reg/reg form is only defined for the disassembler; for codegen it is
// equivalent to SUBrr.
def RSBrr : AsI1<0b0011, (outs GPR:$Rd), (ins GPR:$Rn, GPR:$Rm), DPFrm,
                 IIC_iALUr, "rsb", "\t$Rd, $Rn, $Rm",
                 [/* For disassembly only; pattern left blank */]> {
  bits<4> Rd;
  bits<4> Rn;
  bits<4> Rm;
  let Inst{11-4} = 0b00000000;
  let Inst{25} = 0;
  let Inst{3-0} = Rm;
  let Inst{15-12} = Rd;
  let Inst{19-16} = Rn;
}

def RSBrs : AsI1<0b0011, (outs GPR:$Rd), (ins GPR:$Rn, so_reg:$shift),
                 DPSoRegFrm, IIC_iALUsr, "rsb", "\t$Rd, $Rn, $shift",
                 [(set GPR:$Rd, (sub so_reg:$shift, GPR:$Rn))]> {
  bits<4> Rd;
  bits<4> Rn;
  bits<12> shift;
  let Inst{25} = 0;
  let Inst{11-0} = shift;
  let Inst{15-12} = Rd;
  let Inst{19-16} = Rn;
}

// RSB with 's' bit set.
let Defs = [CPSR] in {
def RSBSri : AI1<0b0011, (outs GPR:$Rd), (ins GPR:$Rn, so_imm:$imm), DPFrm,
                 IIC_iALUi, "rsbs", "\t$Rd, $Rn, $imm",
                 [(set GPR:$Rd, (subc so_imm:$imm, GPR:$Rn))]> {
  bits<4> Rd;
  bits<4> Rn;
  bits<12> imm;
  let Inst{25} = 1;
  let Inst{20} = 1;
  let Inst{15-12} = Rd;
  let Inst{19-16} = Rn;
  let Inst{11-0} = imm;
}
def RSBSrs : AI1<0b0011, (outs GPR:$Rd), (ins GPR:$Rn, so_reg:$shift),
                 DPSoRegFrm, IIC_iALUsr, "rsbs", "\t$Rd, $Rn, $shift",
                 [(set GPR:$Rd, (subc so_reg:$shift, GPR:$Rn))]> {
  bits<4> Rd;
  bits<4> Rn;
  bits<12> shift;
  let Inst{25} = 0;
  let Inst{20} = 1;
  let Inst{11-0} = shift;
  let Inst{15-12} = Rd;
  let Inst{19-16} = Rn;
}
}

let Uses = [CPSR] in {
def RSCri : AsI1<0b0111, (outs GPR:$Rd), (ins GPR:$Rn, so_imm:$imm),
                 DPFrm, IIC_iALUi, "rsc", "\t$Rd, $Rn, $imm",
                 [(set GPR:$Rd, (sube_dead_carry so_imm:$imm, GPR:$Rn))]>,
                 Requires<[IsARM]> {
  bits<4> Rd;
  bits<4> Rn;
  bits<12> imm;
  let Inst{25} = 1;
  let Inst{15-12} = Rd;
  let Inst{19-16} = Rn;
  let Inst{11-0} = imm;
}
// The reg/reg form is only defined for the disassembler; for codegen it is
// equivalent to SUBrr.
def RSCrr : AsI1<0b0111, (outs GPR:$Rd), (ins GPR:$Rn, GPR:$Rm),
                 DPFrm, IIC_iALUr, "rsc", "\t$Rd, $Rn, $Rm",
                 [/* For disassembly only; pattern left blank */]> {
  bits<4> Rd;
  bits<4> Rn;
  bits<4> Rm;
  let Inst{11-4} = 0b00000000;
  let Inst{25} = 0;
  let Inst{3-0} = Rm;
  let Inst{15-12} = Rd;
  let Inst{19-16} = Rn;
}
def RSCrs : AsI1<0b0111, (outs GPR:$Rd), (ins GPR:$Rn, so_reg:$shift),
                 DPSoRegFrm, IIC_iALUsr, "rsc", "\t$Rd, $Rn, $shift",
                 [(set GPR:$Rd, (sube_dead_carry so_reg:$shift, GPR:$Rn))]>,
                 Requires<[IsARM]> {
  bits<4> Rd;
  bits<4> Rn;
  bits<12> shift;
  let Inst{25} = 0;
  let Inst{11-0} = shift;
  let Inst{15-12} = Rd;
  let Inst{19-16} = Rn;
}
}

// FIXME: Allow these to be predicated.
let Defs = [CPSR], Uses = [CPSR] in {
def RSCSri : AXI1<0b0111, (outs GPR:$Rd), (ins GPR:$Rn, so_imm:$imm),
                  DPFrm, IIC_iALUi, "rscs\t$Rd, $Rn, $imm",
                  [(set GPR:$Rd, (sube_dead_carry so_imm:$imm, GPR:$Rn))]>,
                  Requires<[IsARM]> {
  bits<4> Rd;
  bits<4> Rn;
  bits<12> imm;
  let Inst{25} = 1;
  let Inst{20} = 1;
  let Inst{15-12} = Rd;
  let Inst{19-16} = Rn;
  let Inst{11-0} = imm;
}
def RSCSrs : AXI1<0b0111, (outs GPR:$Rd), (ins GPR:$Rn, so_reg:$shift),
                  DPSoRegFrm, IIC_iALUsr, "rscs\t$Rd, $Rn, $shift",
                  [(set GPR:$Rd, (sube_dead_carry so_reg:$shift, GPR:$Rn))]>,
                  Requires<[IsARM]> {
  bits<4> Rd;
  bits<4> Rn;
  bits<12> shift;
  let Inst{25} = 0;
  let Inst{20} = 1;
  let Inst{11-0} = shift;
  let Inst{15-12} = Rd;
  let Inst{19-16} = Rn;
}
}

// (sub X, imm) gets canonicalized to (add X, -imm).  Match this form.
// The assume-no-carry-in form uses the negation of the input since add/sub
// assume opposite meanings of the carry flag (i.e., carry == !borrow).
// See the definition of AddWithCarry() in the ARM ARM A2.2.1 for the gory
// details.
def : ARMPat<(add    GPR:$src, so_imm_neg:$imm),
             (SUBri  GPR:$src, so_imm_neg:$imm)>;
def : ARMPat<(addc   GPR:$src, so_imm_neg:$imm),
             (SUBSri GPR:$src, so_imm_neg:$imm)>;
// The with-carry-in form matches bitwise not instead of the negation.
// Effectively, the inverse interpretation of the carry flag already accounts
// for part of the negation.
def : ARMPat<(adde   GPR:$src, so_imm_not:$imm),
             (SBCri  GPR:$src, so_imm_not:$imm)>;

// Note: These are implemented in C++ code, because they have to generate
// ADD/SUBrs instructions, which use a complex pattern that a xform function
// cannot produce.
// (mul X, 2^n+1) -> (add (X << n), X)
// (mul X, 2^n-1) -> (rsb X, (X << n))

// ARM Arithmetic Instruction -- for disassembly only
// GPR:$dst = GPR:$a op GPR:$b
class AAI<bits<8> op27_20, bits<8> op11_4, string opc,
          list<dag> pattern = [/* For disassembly only; pattern left blank */]>
  : AI<(outs GPR:$Rd), (ins GPR:$Rn, GPR:$Rm), DPFrm, IIC_iALUr,
       opc, "\t$Rd, $Rn, $Rm", pattern> {
  bits<4> Rd;
  bits<4> Rn;
  bits<4> Rm;
  let Inst{27-20} = op27_20;
  let Inst{11-4} = op11_4;
  let Inst{19-16} = Rn;
  let Inst{15-12} = Rd;
  let Inst{3-0}   = Rm;
}

// Saturating add/subtract -- for disassembly only

def QADD    : AAI<0b00010000, 0b00000101, "qadd",
                  [(set GPR:$Rd, (int_arm_qadd GPR:$Rn, GPR:$Rm))]>;
def QSUB    : AAI<0b00010010, 0b00000101, "qsub",
                  [(set GPR:$Rd, (int_arm_qsub GPR:$Rn, GPR:$Rm))]>;
def QDADD   : AAI<0b00010100, 0b00000101, "qdadd">;
def QDSUB   : AAI<0b00010110, 0b00000101, "qdsub">;

def QADD16  : AAI<0b01100010, 0b11110001, "qadd16">;
def QADD8   : AAI<0b01100010, 0b11111001, "qadd8">;
def QASX    : AAI<0b01100010, 0b11110011, "qasx">;
def QSAX    : AAI<0b01100010, 0b11110101, "qsax">;
def QSUB16  : AAI<0b01100010, 0b11110111, "qsub16">;
def QSUB8   : AAI<0b01100010, 0b11111111, "qsub8">;
def UQADD16 : AAI<0b01100110, 0b11110001, "uqadd16">;
def UQADD8  : AAI<0b01100110, 0b11111001, "uqadd8">;
def UQASX   : AAI<0b01100110, 0b11110011, "uqasx">;
def UQSAX   : AAI<0b01100110, 0b11110101, "uqsax">;
def UQSUB16 : AAI<0b01100110, 0b11110111, "uqsub16">;
def UQSUB8  : AAI<0b01100110, 0b11111111, "uqsub8">;

// Signed/Unsigned add/subtract -- for disassembly only

def SASX   : AAI<0b01100001, 0b11110011, "sasx">;
def SADD16 : AAI<0b01100001, 0b11110001, "sadd16">;
def SADD8  : AAI<0b01100001, 0b11111001, "sadd8">;
def SSAX   : AAI<0b01100001, 0b11110101, "ssax">;
def SSUB16 : AAI<0b01100001, 0b11110111, "ssub16">;
def SSUB8  : AAI<0b01100001, 0b11111111, "ssub8">;
def UASX   : AAI<0b01100101, 0b11110011, "uasx">;
def UADD16 : AAI<0b01100101, 0b11110001, "uadd16">;
def UADD8  : AAI<0b01100101, 0b11111001, "uadd8">;
def USAX   : AAI<0b01100101, 0b11110101, "usax">;
def USUB16 : AAI<0b01100101, 0b11110111, "usub16">;
def USUB8  : AAI<0b01100101, 0b11111111, "usub8">;

// Signed/Unsigned halving add/subtract -- for disassembly only

def SHASX   : AAI<0b01100011, 0b11110011, "shasx">;
def SHADD16 : AAI<0b01100011, 0b11110001, "shadd16">;
def SHADD8  : AAI<0b01100011, 0b11111001, "shadd8">;
def SHSAX   : AAI<0b01100011, 0b11110101, "shsax">;
def SHSUB16 : AAI<0b01100011, 0b11110111, "shsub16">;
def SHSUB8  : AAI<0b01100011, 0b11111111, "shsub8">;
def UHASX   : AAI<0b01100111, 0b11110011, "uhasx">;
def UHADD16 : AAI<0b01100111, 0b11110001, "uhadd16">;
def UHADD8  : AAI<0b01100111, 0b11111001, "uhadd8">;
def UHSAX   : AAI<0b01100111, 0b11110101, "uhsax">;
def UHSUB16 : AAI<0b01100111, 0b11110111, "uhsub16">;
def UHSUB8  : AAI<0b01100111, 0b11111111, "uhsub8">;

// Unsigned Sum of Absolute Differences [and Accumulate] -- for disassembly only

def USAD8  : AI<(outs GPR:$Rd), (ins GPR:$Rn, GPR:$Rm),
                MulFrm /* for convenience */, NoItinerary, "usad8",
                "\t$Rd, $Rn, $Rm", []>,
             Requires<[IsARM, HasV6]> {
  bits<4> Rd;
  bits<4> Rn;
  bits<4> Rm;
  let Inst{27-20} = 0b01111000;
  let Inst{15-12} = 0b1111;
  let Inst{7-4} = 0b0001;
  let Inst{19-16} = Rd;
  let Inst{11-8} = Rm;
  let Inst{3-0} = Rn;
}
def USADA8 : AI<(outs GPR:$Rd), (ins GPR:$Rn, GPR:$Rm, GPR:$Ra),
                MulFrm /* for convenience */, NoItinerary, "usada8",
                "\t$Rd, $Rn, $Rm, $Ra", []>,
             Requires<[IsARM, HasV6]> {
  bits<4> Rd;
  bits<4> Rn;
  bits<4> Rm;
  bits<4> Ra;
  let Inst{27-20} = 0b01111000;
  let Inst{7-4} = 0b0001;
  let Inst{19-16} = Rd;
  let Inst{15-12} = Ra;
  let Inst{11-8} = Rm;
  let Inst{3-0} = Rn;
}

// Signed/Unsigned saturate -- for disassembly only

def SSAT : AI<(outs GPR:$Rd), (ins i32imm:$sat_imm, GPR:$a, shift_imm:$sh),
              SatFrm, NoItinerary, "ssat", "\t$Rd, $sat_imm, $a$sh",
              [/* For disassembly only; pattern left blank */]> {
  bits<4> Rd;
  bits<5> sat_imm;
  bits<4> Rn;
  bits<8> sh;
  let Inst{27-21} = 0b0110101;
  let Inst{5-4} = 0b01;
  let Inst{20-16} = sat_imm;
  let Inst{15-12} = Rd;
  let Inst{11-7} = sh{7-3};
  let Inst{6} = sh{0};
  let Inst{3-0} = Rn;
}

def SSAT16 : AI<(outs GPR:$Rd), (ins i32imm:$sat_imm, GPR:$Rn), SatFrm,
                NoItinerary, "ssat16", "\t$Rd, $sat_imm, $Rn",
                [/* For disassembly only; pattern left blank */]> {
  bits<4> Rd;
  bits<4> sat_imm;
  bits<4> Rn;
  let Inst{27-20} = 0b01101010;
  let Inst{11-4} = 0b11110011;
  let Inst{15-12} = Rd;
  let Inst{19-16} = sat_imm;
  let Inst{3-0} = Rn;
}

def USAT : AI<(outs GPR:$Rd), (ins i32imm:$sat_imm, GPR:$a, shift_imm:$sh),
              SatFrm, NoItinerary, "usat", "\t$Rd, $sat_imm, $a$sh",
              [/* For disassembly only; pattern left blank */]> {
  bits<4> Rd;
  bits<5> sat_imm;
  bits<4> Rn;
  bits<8> sh;
  let Inst{27-21} = 0b0110111;
  let Inst{5-4} = 0b01;
  let Inst{15-12} = Rd;
  let Inst{11-7} = sh{7-3};
  let Inst{6} = sh{0};
  let Inst{20-16} = sat_imm;
  let Inst{3-0} = Rn;
}

def USAT16 : AI<(outs GPR:$Rd), (ins i32imm:$sat_imm, GPR:$a), SatFrm,
                NoItinerary, "usat16", "\t$Rd, $sat_imm, $a",
                [/* For disassembly only; pattern left blank */]> {
  bits<4> Rd;
  bits<4> sat_imm;
  bits<4> Rn;
  let Inst{27-20} = 0b01101110;
  let Inst{11-4} = 0b11110011;
  let Inst{15-12} = Rd;
  let Inst{19-16} = sat_imm;
  let Inst{3-0} = Rn;
}

def : ARMV6Pat<(int_arm_ssat GPR:$a, imm:$pos), (SSAT imm:$pos, GPR:$a, 0)>;
def : ARMV6Pat<(int_arm_usat GPR:$a, imm:$pos), (USAT imm:$pos, GPR:$a, 0)>;

//===----------------------------------------------------------------------===//
//  Bitwise Instructions.
//

defm AND   : AsI1_bin_irs<0b0000, "and",
                          IIC_iBITi, IIC_iBITr, IIC_iBITsr,
                          BinOpFrag<(and node:$LHS, node:$RHS)>, 1>;
defm ORR   : AsI1_bin_irs<0b1100, "orr",
                          IIC_iBITi, IIC_iBITr, IIC_iBITsr,
                          BinOpFrag<(or  node:$LHS, node:$RHS)>, 1>;
defm EOR   : AsI1_bin_irs<0b0001, "eor",
                          IIC_iBITi, IIC_iBITr, IIC_iBITsr,
                          BinOpFrag<(xor node:$LHS, node:$RHS)>, 1>;
defm BIC   : AsI1_bin_irs<0b1110, "bic",
                          IIC_iBITi, IIC_iBITr, IIC_iBITsr,
                          BinOpFrag<(and node:$LHS, (not node:$RHS))>>;

def BFC    : I<(outs GPR:$Rd), (ins GPR:$src, bf_inv_mask_imm:$imm),
               AddrMode1, Size4Bytes, IndexModeNone, DPFrm, IIC_iUNAsi,
               "bfc", "\t$Rd, $imm", "$src = $Rd",
               [(set GPR:$Rd, (and GPR:$src, bf_inv_mask_imm:$imm))]>,
               Requires<[IsARM, HasV6T2]> {
  bits<4> Rd;
  bits<10> imm;
  let Inst{27-21} = 0b0111110;
  let Inst{6-0}   = 0b0011111;
  let Inst{15-12} = Rd;
  let Inst{11-7}  = imm{4-0}; // lsb
  let Inst{20-16} = imm{9-5}; // width
}

// A8.6.18  BFI - Bitfield insert (Encoding A1)
def BFI    : I<(outs GPR:$Rd), (ins GPR:$src, GPR:$Rn, bf_inv_mask_imm:$imm),
               AddrMode1, Size4Bytes, IndexModeNone, DPFrm, IIC_iUNAsi,
               "bfi", "\t$Rd, $Rn, $imm", "$src = $Rd",
               [(set GPR:$Rd, (ARMbfi GPR:$src, GPR:$Rn,
                                bf_inv_mask_imm:$imm))]>,
               Requires<[IsARM, HasV6T2]> {
  bits<4> Rd;
  bits<4> Rn;
  bits<10> imm;
  let Inst{27-21} = 0b0111110;
  let Inst{6-4}   = 0b001; // Rn: Inst{3-0} != 15
  let Inst{15-12} = Rd;
  let Inst{11-7}  = imm{4-0}; // lsb
  let Inst{20-16} = imm{9-5}; // width
  let Inst{3-0}   = Rn;
}

def  MVNr  : AsI1<0b1111, (outs GPR:$Rd), (ins GPR:$Rm), DPFrm, IIC_iMVNr,
                  "mvn", "\t$Rd, $Rm",
                  [(set GPR:$Rd, (not GPR:$Rm))]>, UnaryDP {
  bits<4> Rd;
  bits<4> Rm;
  let Inst{25} = 0;
  let Inst{19-16} = 0b0000;
  let Inst{11-4} = 0b00000000;
  let Inst{15-12} = Rd;
  let Inst{3-0} = Rm;
}
def  MVNs  : AsI1<0b1111, (outs GPR:$Rd), (ins so_reg:$shift), DPSoRegFrm,
                  IIC_iMVNsr, "mvn", "\t$Rd, $shift",
                  [(set GPR:$Rd, (not so_reg:$shift))]>, UnaryDP {
  bits<4> Rd;
  bits<4> Rm;
  bits<12> shift;
  let Inst{25} = 0;
  let Inst{19-16} = 0b0000;
  let Inst{15-12} = Rd;
  let Inst{11-0} = shift;
}
let isReMaterializable = 1, isAsCheapAsAMove = 1 in
def  MVNi  : AsI1<0b1111, (outs GPR:$Rd), (ins so_imm:$imm), DPFrm,
                  IIC_iMVNi, "mvn", "\t$Rd, $imm",
                  [(set GPR:$Rd, so_imm_not:$imm)]>,UnaryDP {
  bits<4> Rd;
  bits<4> Rm;
  bits<12> imm;
  let Inst{25} = 1;
  let Inst{19-16} = 0b0000;
  let Inst{15-12} = Rd;
  let Inst{11-0} = imm;
}

def : ARMPat<(and   GPR:$src, so_imm_not:$imm),
             (BICri GPR:$src, so_imm_not:$imm)>;

//===----------------------------------------------------------------------===//
//  Multiply Instructions.
//
class AsMul1I32<bits<7> opcod, dag oops, dag iops, InstrItinClass itin,
             string opc, string asm, list<dag> pattern>
  : AsMul1I<opcod, oops, iops, itin, opc, asm, pattern> {
  bits<4> Rd;
  bits<4> Rm;
  bits<4> Rn;
  let Inst{19-16} = Rd;
  let Inst{11-8}  = Rm;
  let Inst{3-0}   = Rn;
}
class AsMul1I64<bits<7> opcod, dag oops, dag iops, InstrItinClass itin,
             string opc, string asm, list<dag> pattern>
  : AsMul1I<opcod, oops, iops, itin, opc, asm, pattern> {
  bits<4> RdLo;
  bits<4> RdHi;
  bits<4> Rm;
  bits<4> Rn;
  let Inst{19-16} = RdHi;
  let Inst{15-12} = RdLo;
  let Inst{11-8}  = Rm;
  let Inst{3-0}   = Rn;
}

let isCommutable = 1 in
def MUL  : AsMul1I32<0b0000000, (outs GPR:$Rd), (ins GPR:$Rn, GPR:$Rm),
                   IIC_iMUL32, "mul", "\t$Rd, $Rn, $Rm",
                   [(set GPR:$Rd, (mul GPR:$Rn, GPR:$Rm))]>;

def MLA  : AsMul1I32<0b0000001, (outs GPR:$Rd), (ins GPR:$Rn, GPR:$Rm, GPR:$Ra),
                    IIC_iMAC32, "mla", "\t$Rd, $Rn, $Rm, $Ra",
                   [(set GPR:$Rd, (add (mul GPR:$Rn, GPR:$Rm), GPR:$Ra))]> {
  bits<4> Ra;
  let Inst{15-12} = Ra;
}

def MLS  : AMul1I<0b0000011, (outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$c),
                   IIC_iMAC32, "mls", "\t$dst, $a, $b, $c",
                   [(set GPR:$dst, (sub GPR:$c, (mul GPR:$a, GPR:$b)))]>,
                   Requires<[IsARM, HasV6T2]> {
  bits<4> Rd;
  bits<4> Rm;
  bits<4> Rn;
  let Inst{19-16} = Rd;
  let Inst{11-8}  = Rm;
  let Inst{3-0}   = Rn;
}

// Extra precision multiplies with low / high results

let neverHasSideEffects = 1 in {
let isCommutable = 1 in {
def SMULL : AsMul1I64<0b0000110, (outs GPR:$RdLo, GPR:$RdHi),
                               (ins GPR:$Rn, GPR:$Rm), IIC_iMUL64,
                    "smull", "\t$RdLo, $RdHi, $Rn, $Rm", []>;

def UMULL : AsMul1I64<0b0000100, (outs GPR:$RdLo, GPR:$RdHi),
                               (ins GPR:$Rn, GPR:$Rm), IIC_iMUL64,
                    "umull", "\t$RdLo, $RdHi, $Rn, $Rm", []>;
}

// Multiply + accumulate
def SMLAL : AsMul1I64<0b0000111, (outs GPR:$RdLo, GPR:$RdHi),
                               (ins GPR:$Rn, GPR:$Rm), IIC_iMAC64,
                    "smlal", "\t$RdLo, $RdHi, $Rn, $Rm", []>;

def UMLAL : AsMul1I64<0b0000101, (outs GPR:$RdLo, GPR:$RdHi),
                               (ins GPR:$Rn, GPR:$Rm), IIC_iMAC64,
                    "umlal", "\t$RdLo, $RdHi, $Rn, $Rm", []>;

def UMAAL : AMul1I <0b0000010, (outs GPR:$RdLo, GPR:$RdHi),
                               (ins GPR:$Rn, GPR:$Rm), IIC_iMAC64,
                    "umaal", "\t$RdLo, $RdHi, $Rn, $Rm", []>,
                    Requires<[IsARM, HasV6]> {
  bits<4> RdLo;
  bits<4> RdHi;
  bits<4> Rm;
  bits<4> Rn;
  let Inst{19-16} = RdLo;
  let Inst{15-12} = RdHi;
  let Inst{11-8}  = Rm;
  let Inst{3-0}   = Rn;
}
} // neverHasSideEffects

// Most significant word multiply
def SMMUL : AMul2I <0b0111010, 0b0001, (outs GPR:$Rd), (ins GPR:$Rn, GPR:$Rm),
               IIC_iMUL32, "smmul", "\t$Rd, $Rn, $Rm",
               [(set GPR:$Rd, (mulhs GPR:$Rn, GPR:$Rm))]>,
            Requires<[IsARM, HasV6]> {
  let Inst{15-12} = 0b1111;
}

def SMMULR : AMul2I <0b0111010, 0b0011, (outs GPR:$Rd), (ins GPR:$Rn, GPR:$Rm),
               IIC_iMUL32, "smmulr", "\t$Rd, $Rn, $Rm",
               [/* For disassembly only; pattern left blank */]>,
            Requires<[IsARM, HasV6]> {
  let Inst{15-12} = 0b1111;
}

def SMMLA : AMul2Ia <0b0111010, 0b0001, (outs GPR:$Rd),
               (ins GPR:$Rn, GPR:$Rm, GPR:$Ra),
               IIC_iMAC32, "smmla", "\t$Rd, $Rn, $Rm, $Ra",
               [(set GPR:$Rd, (add (mulhs GPR:$Rn, GPR:$Rm), GPR:$Ra))]>,
            Requires<[IsARM, HasV6]>;

def SMMLAR : AMul2Ia <0b0111010, 0b0011, (outs GPR:$Rd),
               (ins GPR:$Rn, GPR:$Rm, GPR:$Ra),
               IIC_iMAC32, "smmlar", "\t$Rd, $Rn, $Rm, $Ra",
               [/* For disassembly only; pattern left blank */]>,
            Requires<[IsARM, HasV6]>;

def SMMLS : AMul2Ia <0b0111010, 0b1101, (outs GPR:$Rd),
               (ins GPR:$Rn, GPR:$Rm, GPR:$Ra),
               IIC_iMAC32, "smmls", "\t$Rd, $Rn, $Rm, $Ra",
               [(set GPR:$Rd, (sub GPR:$Ra, (mulhs GPR:$Rn, GPR:$Rm)))]>,
            Requires<[IsARM, HasV6]>;

def SMMLSR : AMul2Ia <0b0111010, 0b1111, (outs GPR:$Rd),
               (ins GPR:$Rn, GPR:$Rm, GPR:$Ra),
               IIC_iMAC32, "smmlsr", "\t$Rd, $Rn, $Rm, $Ra",
               [/* For disassembly only; pattern left blank */]>,
            Requires<[IsARM, HasV6]>;

multiclass AI_smul<string opc, PatFrag opnode> {
  def BB : AMulxyI<0b0001011, 0b00, (outs GPR:$Rd), (ins GPR:$Rn, GPR:$Rm),
              IIC_iMUL16, !strconcat(opc, "bb"), "\t$Rd, $Rn, $Rm",
              [(set GPR:$Rd, (opnode (sext_inreg GPR:$Rn, i16),
                                      (sext_inreg GPR:$Rm, i16)))]>,
           Requires<[IsARM, HasV5TE]>;

  def BT : AMulxyI<0b0001011, 0b10, (outs GPR:$Rd), (ins GPR:$Rn, GPR:$Rm),
              IIC_iMUL16, !strconcat(opc, "bt"), "\t$Rd, $Rn, $Rm",
              [(set GPR:$Rd, (opnode (sext_inreg GPR:$Rn, i16),
                                      (sra GPR:$Rm, (i32 16))))]>,
           Requires<[IsARM, HasV5TE]>;

  def TB : AMulxyI<0b0001011, 0b01, (outs GPR:$Rd), (ins GPR:$Rn, GPR:$Rm),
              IIC_iMUL16, !strconcat(opc, "tb"), "\t$Rd, $Rn, $Rm",
              [(set GPR:$Rd, (opnode (sra GPR:$Rn, (i32 16)),
                                      (sext_inreg GPR:$Rm, i16)))]>,
           Requires<[IsARM, HasV5TE]>;

  def TT : AMulxyI<0b0001011, 0b11, (outs GPR:$Rd), (ins GPR:$Rn, GPR:$Rm),
              IIC_iMUL16, !strconcat(opc, "tt"), "\t$Rd, $Rn, $Rm",
              [(set GPR:$Rd, (opnode (sra GPR:$Rn, (i32 16)),
                                      (sra GPR:$Rm, (i32 16))))]>,
            Requires<[IsARM, HasV5TE]>;

  def WB : AMulxyI<0b0001001, 0b01, (outs GPR:$Rd), (ins GPR:$Rn, GPR:$Rm),
              IIC_iMUL16, !strconcat(opc, "wb"), "\t$Rd, $Rn, $Rm",
              [(set GPR:$Rd, (sra (opnode GPR:$Rn,
                                    (sext_inreg GPR:$Rm, i16)), (i32 16)))]>,
           Requires<[IsARM, HasV5TE]>;

  def WT : AMulxyI<0b0001001, 0b11, (outs GPR:$Rd), (ins GPR:$Rn, GPR:$Rm),
              IIC_iMUL16, !strconcat(opc, "wt"), "\t$Rd, $Rn, $Rm",
              [(set GPR:$Rd, (sra (opnode GPR:$Rn,
                                    (sra GPR:$Rm, (i32 16))), (i32 16)))]>,
            Requires<[IsARM, HasV5TE]>;
}


multiclass AI_smla<string opc, PatFrag opnode> {
  def BB : AMulxyI<0b0001000, 0b00, (outs GPR:$Rd),
              (ins GPR:$Rn, GPR:$Rm, GPR:$Ra),
              IIC_iMAC16, !strconcat(opc, "bb"), "\t$Rd, $Rn, $Rm, $Ra",
              [(set GPR:$Rd, (add GPR:$Ra,
                               (opnode (sext_inreg GPR:$Rn, i16),
                                       (sext_inreg GPR:$Rm, i16))))]>,
           Requires<[IsARM, HasV5TE]>;

  def BT : AMulxyI<0b0001000, 0b10, (outs GPR:$Rd),
              (ins GPR:$Rn, GPR:$Rm, GPR:$Ra),
              IIC_iMAC16, !strconcat(opc, "bt"), "\t$Rd, $Rn, $Rm, $Ra",
              [(set GPR:$Rd, (add GPR:$Ra, (opnode (sext_inreg GPR:$Rn, i16),
                                                   (sra GPR:$Rm, (i32 16)))))]>,
           Requires<[IsARM, HasV5TE]>;

  def TB : AMulxyI<0b0001000, 0b01, (outs GPR:$Rd),
              (ins GPR:$Rn, GPR:$Rm, GPR:$Ra),
              IIC_iMAC16, !strconcat(opc, "tb"), "\t$Rd, $Rn, $Rm, $Ra",
              [(set GPR:$Rd, (add GPR:$Ra, (opnode (sra GPR:$Rn, (i32 16)),
                                                (sext_inreg GPR:$Rm, i16))))]>,
           Requires<[IsARM, HasV5TE]>;

  def TT : AMulxyI<0b0001000, 0b11, (outs GPR:$Rd),
              (ins GPR:$Rn, GPR:$Rm, GPR:$Ra),
              IIC_iMAC16, !strconcat(opc, "tt"), "\t$Rd, $Rn, $Rm, $Ra",
             [(set GPR:$Rd, (add GPR:$Ra, (opnode (sra GPR:$Rn, (i32 16)),
                                                   (sra GPR:$Rm, (i32 16)))))]>,
            Requires<[IsARM, HasV5TE]>;

  def WB : AMulxyI<0b0001001, 0b00, (outs GPR:$Rd),
              (ins GPR:$Rn, GPR:$Rm, GPR:$Ra),
              IIC_iMAC16, !strconcat(opc, "wb"), "\t$Rd, $Rn, $Rm, $Ra",
              [(set GPR:$Rd, (add GPR:$Ra, (sra (opnode GPR:$Rn,
                                      (sext_inreg GPR:$Rm, i16)), (i32 16))))]>,
           Requires<[IsARM, HasV5TE]>;

  def WT : AMulxyI<0b0001001, 0b10, (outs GPR:$Rd),
              (ins GPR:$Rn, GPR:$Rm, GPR:$Ra),
              IIC_iMAC16, !strconcat(opc, "wt"), "\t$Rd, $Rn, $Rm, $Ra",
              [(set GPR:$Rd, (add GPR:$Ra, (sra (opnode GPR:$Rn,
                                        (sra GPR:$Rm, (i32 16))), (i32 16))))]>,
            Requires<[IsARM, HasV5TE]>;
}

defm SMUL : AI_smul<"smul", BinOpFrag<(mul node:$LHS, node:$RHS)>>;
defm SMLA : AI_smla<"smla", BinOpFrag<(mul node:$LHS, node:$RHS)>>;

// Halfword multiply accumulate long: SMLAL<x><y> -- for disassembly only
def SMLALBB : AMulxyI64<0b0001010, 0b00, (outs GPR:$RdLo, GPR:$RdHi),
                      (ins GPR:$Rn, GPR:$Rm),
                      IIC_iMAC64, "smlalbb", "\t$RdLo, $RdHi, $Rn, $Rm",
                      [/* For disassembly only; pattern left blank */]>,
              Requires<[IsARM, HasV5TE]>;

def SMLALBT : AMulxyI64<0b0001010, 0b10, (outs GPR:$RdLo, GPR:$RdHi),
                      (ins GPR:$Rn, GPR:$Rm),
                      IIC_iMAC64, "smlalbt", "\t$RdLo, $RdHi, $Rn, $Rm",
                      [/* For disassembly only; pattern left blank */]>,
              Requires<[IsARM, HasV5TE]>;

def SMLALTB : AMulxyI64<0b0001010, 0b01, (outs GPR:$RdLo, GPR:$RdHi),
                      (ins GPR:$Rn, GPR:$Rm),
                      IIC_iMAC64, "smlaltb", "\t$RdLo, $RdHi, $Rn, $Rm",
                      [/* For disassembly only; pattern left blank */]>,
              Requires<[IsARM, HasV5TE]>;

def SMLALTT : AMulxyI64<0b0001010, 0b11, (outs GPR:$RdLo, GPR:$RdHi),
                      (ins GPR:$Rn, GPR:$Rm),
                      IIC_iMAC64, "smlaltt", "\t$RdLo, $RdHi, $Rn, $Rm",
                      [/* For disassembly only; pattern left blank */]>,
              Requires<[IsARM, HasV5TE]>;

// Helper class for AI_smld -- for disassembly only
class AMulDualIbase<bit long, bit sub, bit swap, dag oops, dag iops,
                    InstrItinClass itin, string opc, string asm>
  : AI<oops, iops, MulFrm, itin, opc, asm, []>, Requires<[IsARM, HasV6]> {
  bits<4> Rn;
  bits<4> Rm;
  let Inst{4}     = 1;
  let Inst{5}     = swap;
  let Inst{6}     = sub;
  let Inst{7}     = 0;
  let Inst{21-20} = 0b00;
  let Inst{22}    = long;
  let Inst{27-23} = 0b01110;
  let Inst{11-8}  = Rm;
  let Inst{3-0}   = Rn;
}
class AMulDualI<bit long, bit sub, bit swap, dag oops, dag iops,
                InstrItinClass itin, string opc, string asm>
  : AMulDualIbase<long, sub, swap, oops, iops, itin, opc, asm> {
  bits<4> Rd;
  let Inst{15-12} = 0b1111;
  let Inst{19-16} = Rd;
}
class AMulDualIa<bit long, bit sub, bit swap, dag oops, dag iops,
                InstrItinClass itin, string opc, string asm>
  : AMulDualIbase<long, sub, swap, oops, iops, itin, opc, asm> {
  bits<4> Ra;
  let Inst{15-12} = Ra;
}
class AMulDualI64<bit long, bit sub, bit swap, dag oops, dag iops,
                  InstrItinClass itin, string opc, string asm>
  : AMulDualIbase<long, sub, swap, oops, iops, itin, opc, asm> {
  bits<4> RdLo;
  bits<4> RdHi;
  let Inst{19-16} = RdHi;
  let Inst{15-12} = RdLo;
}

multiclass AI_smld<bit sub, string opc> {

  def D : AMulDualIa<0, sub, 0, (outs GPR:$Rd), (ins GPR:$Rn, GPR:$Rm, GPR:$Ra),
                  NoItinerary, !strconcat(opc, "d"), "\t$Rd, $Rn, $Rm, $Ra">;

  def DX: AMulDualIa<0, sub, 1, (outs GPR:$Rd), (ins GPR:$Rn, GPR:$Rm, GPR:$Ra),
                  NoItinerary, !strconcat(opc, "dx"), "\t$Rd, $Rn, $Rm, $Ra">;

  def LD: AMulDualI64<1, sub, 0, (outs GPR:$RdLo,GPR:$RdHi),
                  (ins GPR:$Rn, GPR:$Rm), NoItinerary,
                  !strconcat(opc, "ld"), "\t$RdLo, $RdHi, $Rn, $Rm">;

  def LDX : AMulDualI64<1, sub, 1, (outs GPR:$RdLo,GPR:$RdHi),
                  (ins GPR:$Rn, GPR:$Rm), NoItinerary,
                  !strconcat(opc, "ldx"),"\t$RdLo, $RdHi, $Rn, $Rm">;

}

defm SMLA : AI_smld<0, "smla">;
defm SMLS : AI_smld<1, "smls">;

multiclass AI_sdml<bit sub, string opc> {

  def D : AMulDualI<0, sub, 0, (outs GPR:$Rd), (ins GPR:$Rn, GPR:$Rm),
                    NoItinerary, !strconcat(opc, "d"), "\t$Rd, $Rn, $Rm">;
  def DX : AMulDualI<0, sub, 1, (outs GPR:$Rd), (ins GPR:$Rn, GPR:$Rm),
                    NoItinerary, !strconcat(opc, "dx"), "\t$Rd, $Rn, $Rm">;
}

defm SMUA : AI_sdml<0, "smua">;
defm SMUS : AI_sdml<1, "smus">;

//===----------------------------------------------------------------------===//
//  Misc. Arithmetic Instructions.
//

def CLZ  : AMiscA1I<0b000010110, 0b0001, (outs GPR:$Rd), (ins GPR:$Rm),
              IIC_iUNAr, "clz", "\t$Rd, $Rm",
              [(set GPR:$Rd, (ctlz GPR:$Rm))]>, Requires<[IsARM, HasV5T]>;

def RBIT : AMiscA1I<0b01101111, 0b0011, (outs GPR:$Rd), (ins GPR:$Rm),
              IIC_iUNAr, "rbit", "\t$Rd, $Rm",
              [(set GPR:$Rd, (ARMrbit GPR:$Rm))]>,
           Requires<[IsARM, HasV6T2]>;

def REV  : AMiscA1I<0b01101011, 0b0011, (outs GPR:$Rd), (ins GPR:$Rm),
              IIC_iUNAr, "rev", "\t$Rd, $Rm",
              [(set GPR:$Rd, (bswap GPR:$Rm))]>, Requires<[IsARM, HasV6]>;

def REV16 : AMiscA1I<0b01101011, 0b1011, (outs GPR:$Rd), (ins GPR:$Rm),
               IIC_iUNAr, "rev16", "\t$Rd, $Rm",
               [(set GPR:$Rd,
                   (or (and (srl GPR:$Rm, (i32 8)), 0xFF),
                       (or (and (shl GPR:$Rm, (i32 8)), 0xFF00),
                           (or (and (srl GPR:$Rm, (i32 8)), 0xFF0000),
                               (and (shl GPR:$Rm, (i32 8)), 0xFF000000)))))]>,
               Requires<[IsARM, HasV6]>;

def REVSH : AMiscA1I<0b01101111, 0b1011, (outs GPR:$Rd), (ins GPR:$Rm),
               IIC_iUNAr, "revsh", "\t$Rd, $Rm",
               [(set GPR:$Rd,
                  (sext_inreg
                    (or (srl (and GPR:$Rm, 0xFF00), (i32 8)),
                        (shl GPR:$Rm, (i32 8))), i16))]>,
               Requires<[IsARM, HasV6]>;

def lsl_shift_imm : SDNodeXForm<imm, [{
  unsigned Sh = ARM_AM::getSORegOpc(ARM_AM::lsl, N->getZExtValue());
  return CurDAG->getTargetConstant(Sh, MVT::i32);
}]>;

def lsl_amt : PatLeaf<(i32 imm), [{
  return (N->getZExtValue() < 32);
}], lsl_shift_imm>;

def PKHBT : APKHI<0b01101000, 0, (outs GPR:$Rd),
                              (ins GPR:$Rn, GPR:$Rm, shift_imm:$sh),
               IIC_iALUsi, "pkhbt", "\t$Rd, $Rn, $Rm$sh",
               [(set GPR:$Rd, (or (and GPR:$Rn, 0xFFFF),
                                  (and (shl GPR:$Rm, lsl_amt:$sh),
                                       0xFFFF0000)))]>,
               Requires<[IsARM, HasV6]>;

// Alternate cases for PKHBT where identities eliminate some nodes.
def : ARMV6Pat<(or (and GPR:$Rn, 0xFFFF), (and GPR:$Rm, 0xFFFF0000)),
               (PKHBT GPR:$Rn, GPR:$Rm, 0)>;
def : ARMV6Pat<(or (and GPR:$Rn, 0xFFFF), (shl GPR:$Rm, imm16_31:$sh)),
               (PKHBT GPR:$Rn, GPR:$Rm, (lsl_shift_imm imm16_31:$sh))>;

def asr_shift_imm : SDNodeXForm<imm, [{
  unsigned Sh = ARM_AM::getSORegOpc(ARM_AM::asr, N->getZExtValue());
  return CurDAG->getTargetConstant(Sh, MVT::i32);
}]>;

def asr_amt : PatLeaf<(i32 imm), [{
  return (N->getZExtValue() <= 32);
}], asr_shift_imm>;

// Note: Shifts of 1-15 bits will be transformed to srl instead of sra and
// will match the pattern below.
def PKHTB : APKHI<0b01101000, 1, (outs GPR:$Rd),
                              (ins GPR:$Rn, GPR:$Rm, shift_imm:$sh),
               IIC_iBITsi, "pkhtb", "\t$Rd, $Rn, $Rm$sh",
               [(set GPR:$Rd, (or (and GPR:$Rn, 0xFFFF0000),
                                  (and (sra GPR:$Rm, asr_amt:$sh),
                                       0xFFFF)))]>,
               Requires<[IsARM, HasV6]>;

// Alternate cases for PKHTB where identities eliminate some nodes.  Note that
// a shift amount of 0 is *not legal* here, it is PKHBT instead.
def : ARMV6Pat<(or (and GPR:$src1, 0xFFFF0000), (srl GPR:$src2, imm16_31:$sh)),
               (PKHTB GPR:$src1, GPR:$src2, (asr_shift_imm imm16_31:$sh))>;
def : ARMV6Pat<(or (and GPR:$src1, 0xFFFF0000),
                   (and (srl GPR:$src2, imm1_15:$sh), 0xFFFF)),
               (PKHTB GPR:$src1, GPR:$src2, (asr_shift_imm imm1_15:$sh))>;

//===----------------------------------------------------------------------===//
//  Comparison Instructions...
//

defm CMP  : AI1_cmp_irs<0b1010, "cmp",
                        IIC_iCMPi, IIC_iCMPr, IIC_iCMPsr,
                        BinOpFrag<(ARMcmp node:$LHS, node:$RHS)>>;

// FIXME: We have to be careful when using the CMN instruction and comparison
// with 0. One would expect these two pieces of code should give identical
// results:
//
//   rsbs r1, r1, 0
//   cmp  r0, r1
//   mov  r0, #0
//   it   ls
//   mov  r0, #1
//
// and:
//
//   cmn  r0, r1
//   mov  r0, #0
//   it   ls
//   mov  r0, #1
//
// However, the CMN gives the *opposite* result when r1 is 0. This is because
// the carry flag is set in the CMP case but not in the CMN case. In short, the
// CMP instruction doesn't perform a truncate of the (logical) NOT of 0 plus the
// value of r0 and the carry bit (because the "carry bit" parameter to
// AddWithCarry is defined as 1 in this case, the carry flag will always be set
// when r0 >= 0). The CMN instruction doesn't perform a NOT of 0 so there is
// never a "carry" when this AddWithCarry is performed (because the "carry bit"
// parameter to AddWithCarry is defined as 0).
//
// When x is 0 and unsigned:
//
//    x = 0
//   ~x = 0xFFFF FFFF
//   ~x + 1 = 0x1 0000 0000
//   (-x = 0) != (0x1 0000 0000 = ~x + 1)
//
// Therefore, we should disable CMN when comparing against zero, until we can
// limit when the CMN instruction is used (when we know that the RHS is not 0 or
// when it's a comparison which doesn't look at the 'carry' flag).
//
// (See the ARM docs for the "AddWithCarry" pseudo-code.)
//
// This is related to <rdar://problem/7569620>.
//
//defm CMN  : AI1_cmp_irs<0b1011, "cmn",
//                        BinOpFrag<(ARMcmp node:$LHS,(ineg node:$RHS))>>;

// Note that TST/TEQ don't set all the same flags that CMP does!
defm TST  : AI1_cmp_irs<0b1000, "tst",
                        IIC_iTSTi, IIC_iTSTr, IIC_iTSTsr,
                        BinOpFrag<(ARMcmpZ (and node:$LHS, node:$RHS), 0)>, 1>;
defm TEQ  : AI1_cmp_irs<0b1001, "teq",
                        IIC_iTSTi, IIC_iTSTr, IIC_iTSTsr,
                        BinOpFrag<(ARMcmpZ (xor node:$LHS, node:$RHS), 0)>, 1>;

defm CMPz  : AI1_cmp_irs<0b1010, "cmp",
                         IIC_iCMPi, IIC_iCMPr, IIC_iCMPsr,
                         BinOpFrag<(ARMcmpZ node:$LHS, node:$RHS)>>;
defm CMNz  : AI1_cmp_irs<0b1011, "cmn",
                         IIC_iCMPi, IIC_iCMPr, IIC_iCMPsr,
                         BinOpFrag<(ARMcmpZ node:$LHS,(ineg node:$RHS))>>;

//def : ARMPat<(ARMcmp GPR:$src, so_imm_neg:$imm),
//             (CMNri  GPR:$src, so_imm_neg:$imm)>;

def : ARMPat<(ARMcmpZ GPR:$src, so_imm_neg:$imm),
             (CMNzri  GPR:$src, so_imm_neg:$imm)>;

// Pseudo i64 compares for some floating point compares.
let usesCustomInserter = 1, isBranch = 1, isTerminator = 1,
    Defs = [CPSR] in {
def BCCi64 : PseudoInst<(outs),
    (ins i32imm:$cc, GPR:$lhs1, GPR:$lhs2, GPR:$rhs1, GPR:$rhs2, brtarget:$dst),
     IIC_Br, "",
    [(ARMBcci64 imm:$cc, GPR:$lhs1, GPR:$lhs2, GPR:$rhs1, GPR:$rhs2, bb:$dst)]>;

def BCCZi64 : PseudoInst<(outs),
     (ins i32imm:$cc, GPR:$lhs1, GPR:$lhs2, brtarget:$dst), IIC_Br, "",
    [(ARMBcci64 imm:$cc, GPR:$lhs1, GPR:$lhs2, 0, 0, bb:$dst)]>;
} // usesCustomInserter


// Conditional moves
// FIXME: should be able to write a pattern for ARMcmov, but can't use
// a two-value operand where a dag node expects two operands. :(
// FIXME: These should all be pseudo-instructions that get expanded to
//        the normal MOV instructions. That would fix the dependency on
//        special casing them in tblgen.
let neverHasSideEffects = 1 in {
def MOVCCr : AI1<0b1101, (outs GPR:$Rd), (ins GPR:$false, GPR:$Rm), DPFrm,
                IIC_iCMOVr, "mov", "\t$Rd, $Rm",
      [/*(set GPR:$Rd, (ARMcmov GPR:$false, GPR:$Rm, imm:$cc, CCR:$ccr))*/]>,
                RegConstraint<"$false = $Rd">, UnaryDP {
  bits<4> Rd;
  bits<4> Rm;
  let Inst{25} = 0;
  let Inst{20} = 0;
  let Inst{15-12} = Rd;
  let Inst{11-4} = 0b00000000;
  let Inst{3-0} = Rm;
}

def MOVCCs : AI1<0b1101, (outs GPR:$Rd),
                 (ins GPR:$false, so_reg:$shift), DPSoRegFrm, IIC_iCMOVsr,
                "mov", "\t$Rd, $shift",
   [/*(set GPR:$Rd, (ARMcmov GPR:$false, so_reg:$shift, imm:$cc, CCR:$ccr))*/]>,
                RegConstraint<"$false = $Rd">, UnaryDP {
  bits<4> Rd;
  bits<4> Rn;
  bits<12> shift;
  let Inst{25} = 0;
  let Inst{20} = 0;
  let Inst{19-16} = Rn;
  let Inst{15-12} = Rd;
  let Inst{11-0} = shift;
}

def MOVCCi16 : AI1<0b1000, (outs GPR:$Rd), (ins GPR:$false, i32imm:$imm),
                 DPFrm, IIC_iMOVi,
                 "movw", "\t$Rd, $imm",
                 []>,
                 RegConstraint<"$false = $Rd">, Requires<[IsARM, HasV6T2]>,
                 UnaryDP {
  bits<4> Rd;
  bits<16> imm;
  let Inst{25} = 1;
  let Inst{20} = 0;
  let Inst{19-16} = imm{15-12};
  let Inst{15-12} = Rd;
  let Inst{11-0}  = imm{11-0};
}

def MOVCCi : AI1<0b1101, (outs GPR:$Rd),
                         (ins GPR:$false, so_imm:$imm), DPFrm, IIC_iCMOVi,
                "mov", "\t$Rd, $imm",
   [/*(set GPR:$Rd, (ARMcmov GPR:$false, so_imm:$imm, imm:$cc, CCR:$ccr))*/]>,
                RegConstraint<"$false = $Rd">, UnaryDP {
  bits<4> Rd;
  bits<12> imm;
  let Inst{25} = 1;
  let Inst{20} = 0;
  let Inst{19-16} = 0b0000;
  let Inst{15-12} = Rd;
  let Inst{11-0} = imm;
}
} // neverHasSideEffects

//===----------------------------------------------------------------------===//
// Atomic operations intrinsics
//

def memb_opt : Operand<i32> {
  let PrintMethod = "printMemBOption";
}

// memory barriers protect the atomic sequences
let hasSideEffects = 1 in {
def DMB : AInoP<(outs), (ins memb_opt:$opt), MiscFrm, NoItinerary,
                "dmb", "\t$opt", [(ARMMemBarrier (i32 imm:$opt))]>,
                Requires<[IsARM, HasDB]> {
  bits<4> opt;
  let Inst{31-4} = 0xf57ff05;
  let Inst{3-0} = opt;
}

def DMB_MCR : AInoP<(outs), (ins GPR:$zero), MiscFrm, NoItinerary,
                       "mcr", "\tp15, 0, $zero, c7, c10, 5",
                       [(ARMMemBarrierMCR GPR:$zero)]>,
                       Requires<[IsARM, HasV6]> {
  // FIXME: add encoding
}
}

def DSB : AInoP<(outs), (ins memb_opt:$opt), MiscFrm, NoItinerary,
                "dsb", "\t$opt",
                [/* For disassembly only; pattern left blank */]>,
                Requires<[IsARM, HasDB]> {
  bits<4> opt;
  let Inst{31-4} = 0xf57ff04;
  let Inst{3-0} = opt;
}

// ISB has only full system option -- for disassembly only
def ISB : AInoP<(outs), (ins), MiscFrm, NoItinerary, "isb", "", []>,
                Requires<[IsARM, HasDB]> {
  let Inst{31-4} = 0xf57ff06;
  let Inst{3-0} = 0b1111;
}

let usesCustomInserter = 1 in {
  let Uses = [CPSR] in {
    def ATOMIC_LOAD_ADD_I8 : PseudoInst<
      (outs GPR:$dst), (ins GPR:$ptr, GPR:$incr), NoItinerary, "",
      [(set GPR:$dst, (atomic_load_add_8 GPR:$ptr, GPR:$incr))]>;
    def ATOMIC_LOAD_SUB_I8 : PseudoInst<
      (outs GPR:$dst), (ins GPR:$ptr, GPR:$incr), NoItinerary, "",
      [(set GPR:$dst, (atomic_load_sub_8 GPR:$ptr, GPR:$incr))]>;
    def ATOMIC_LOAD_AND_I8 : PseudoInst<
      (outs GPR:$dst), (ins GPR:$ptr, GPR:$incr), NoItinerary, "",
      [(set GPR:$dst, (atomic_load_and_8 GPR:$ptr, GPR:$incr))]>;
    def ATOMIC_LOAD_OR_I8 : PseudoInst<
      (outs GPR:$dst), (ins GPR:$ptr, GPR:$incr), NoItinerary, "",
      [(set GPR:$dst, (atomic_load_or_8 GPR:$ptr, GPR:$incr))]>;
    def ATOMIC_LOAD_XOR_I8 : PseudoInst<
      (outs GPR:$dst), (ins GPR:$ptr, GPR:$incr), NoItinerary, "",
      [(set GPR:$dst, (atomic_load_xor_8 GPR:$ptr, GPR:$incr))]>;
    def ATOMIC_LOAD_NAND_I8 : PseudoInst<
      (outs GPR:$dst), (ins GPR:$ptr, GPR:$incr), NoItinerary, "",
      [(set GPR:$dst, (atomic_load_nand_8 GPR:$ptr, GPR:$incr))]>;
    def ATOMIC_LOAD_ADD_I16 : PseudoInst<
      (outs GPR:$dst), (ins GPR:$ptr, GPR:$incr), NoItinerary, "",
      [(set GPR:$dst, (atomic_load_add_16 GPR:$ptr, GPR:$incr))]>;
    def ATOMIC_LOAD_SUB_I16 : PseudoInst<
      (outs GPR:$dst), (ins GPR:$ptr, GPR:$incr), NoItinerary, "",
      [(set GPR:$dst, (atomic_load_sub_16 GPR:$ptr, GPR:$incr))]>;
    def ATOMIC_LOAD_AND_I16 : PseudoInst<
      (outs GPR:$dst), (ins GPR:$ptr, GPR:$incr), NoItinerary, "",
      [(set GPR:$dst, (atomic_load_and_16 GPR:$ptr, GPR:$incr))]>;
    def ATOMIC_LOAD_OR_I16 : PseudoInst<
      (outs GPR:$dst), (ins GPR:$ptr, GPR:$incr), NoItinerary, "",
      [(set GPR:$dst, (atomic_load_or_16 GPR:$ptr, GPR:$incr))]>;
    def ATOMIC_LOAD_XOR_I16 : PseudoInst<
      (outs GPR:$dst), (ins GPR:$ptr, GPR:$incr), NoItinerary, "",
      [(set GPR:$dst, (atomic_load_xor_16 GPR:$ptr, GPR:$incr))]>;
    def ATOMIC_LOAD_NAND_I16 : PseudoInst<
      (outs GPR:$dst), (ins GPR:$ptr, GPR:$incr), NoItinerary, "",
      [(set GPR:$dst, (atomic_load_nand_16 GPR:$ptr, GPR:$incr))]>;
    def ATOMIC_LOAD_ADD_I32 : PseudoInst<
      (outs GPR:$dst), (ins GPR:$ptr, GPR:$incr), NoItinerary, "",
      [(set GPR:$dst, (atomic_load_add_32 GPR:$ptr, GPR:$incr))]>;
    def ATOMIC_LOAD_SUB_I32 : PseudoInst<
      (outs GPR:$dst), (ins GPR:$ptr, GPR:$incr), NoItinerary, "",
      [(set GPR:$dst, (atomic_load_sub_32 GPR:$ptr, GPR:$incr))]>;
    def ATOMIC_LOAD_AND_I32 : PseudoInst<
      (outs GPR:$dst), (ins GPR:$ptr, GPR:$incr), NoItinerary, "",
      [(set GPR:$dst, (atomic_load_and_32 GPR:$ptr, GPR:$incr))]>;
    def ATOMIC_LOAD_OR_I32 : PseudoInst<
      (outs GPR:$dst), (ins GPR:$ptr, GPR:$incr), NoItinerary, "",
      [(set GPR:$dst, (atomic_load_or_32 GPR:$ptr, GPR:$incr))]>;
    def ATOMIC_LOAD_XOR_I32 : PseudoInst<
      (outs GPR:$dst), (ins GPR:$ptr, GPR:$incr), NoItinerary, "",
      [(set GPR:$dst, (atomic_load_xor_32 GPR:$ptr, GPR:$incr))]>;
    def ATOMIC_LOAD_NAND_I32 : PseudoInst<
      (outs GPR:$dst), (ins GPR:$ptr, GPR:$incr), NoItinerary, "",
      [(set GPR:$dst, (atomic_load_nand_32 GPR:$ptr, GPR:$incr))]>;

    def ATOMIC_SWAP_I8 : PseudoInst<
      (outs GPR:$dst), (ins GPR:$ptr, GPR:$new), NoItinerary, "",
      [(set GPR:$dst, (atomic_swap_8 GPR:$ptr, GPR:$new))]>;
    def ATOMIC_SWAP_I16 : PseudoInst<
      (outs GPR:$dst), (ins GPR:$ptr, GPR:$new), NoItinerary, "",
      [(set GPR:$dst, (atomic_swap_16 GPR:$ptr, GPR:$new))]>;
    def ATOMIC_SWAP_I32 : PseudoInst<
      (outs GPR:$dst), (ins GPR:$ptr, GPR:$new), NoItinerary, "",
      [(set GPR:$dst, (atomic_swap_32 GPR:$ptr, GPR:$new))]>;

    def ATOMIC_CMP_SWAP_I8 : PseudoInst<
      (outs GPR:$dst), (ins GPR:$ptr, GPR:$old, GPR:$new), NoItinerary, "",
      [(set GPR:$dst, (atomic_cmp_swap_8 GPR:$ptr, GPR:$old, GPR:$new))]>;
    def ATOMIC_CMP_SWAP_I16 : PseudoInst<
      (outs GPR:$dst), (ins GPR:$ptr, GPR:$old, GPR:$new), NoItinerary, "",
      [(set GPR:$dst, (atomic_cmp_swap_16 GPR:$ptr, GPR:$old, GPR:$new))]>;
    def ATOMIC_CMP_SWAP_I32 : PseudoInst<
      (outs GPR:$dst), (ins GPR:$ptr, GPR:$old, GPR:$new), NoItinerary, "",
      [(set GPR:$dst, (atomic_cmp_swap_32 GPR:$ptr, GPR:$old, GPR:$new))]>;
}
}

let mayLoad = 1 in {
def LDREXB : AIldrex<0b10, (outs GPR:$Rt), (ins GPR:$Rn), NoItinerary,
                    "ldrexb", "\t$Rt, [$Rn]",
                    []>;
def LDREXH : AIldrex<0b11, (outs GPR:$Rt), (ins GPR:$Rn), NoItinerary,
                    "ldrexh", "\t$Rt, [$Rn]",
                    []>;
def LDREX  : AIldrex<0b00, (outs GPR:$Rt), (ins GPR:$Rn), NoItinerary,
                    "ldrex", "\t$Rt, [$Rn]",
                    []>;
def LDREXD : AIldrex<0b01, (outs GPR:$Rt, GPR:$Rt2), (ins GPR:$Rn),
                    NoItinerary,
                    "ldrexd", "\t$Rt, $Rt2, [$Rn]",
                    []>;
}

let mayStore = 1, Constraints = "@earlyclobber $Rd" in {
def STREXB : AIstrex<0b10, (outs GPR:$Rd), (ins GPR:$src, GPR:$Rn),
                    NoItinerary,
                    "strexb", "\t$Rd, $src, [$Rn]",
                    []>;
def STREXH : AIstrex<0b11, (outs GPR:$Rd), (ins GPR:$Rt, GPR:$Rn),
                    NoItinerary,
                    "strexh", "\t$Rd, $Rt, [$Rn]",
                    []>;
def STREX  : AIstrex<0b00, (outs GPR:$Rd), (ins GPR:$Rt, GPR:$Rn),
                    NoItinerary,
                    "strex", "\t$Rd, $Rt, [$Rn]",
                    []>;
def STREXD : AIstrex<0b01, (outs GPR:$Rd),
                    (ins GPR:$Rt, GPR:$Rt2, GPR:$Rn),
                    NoItinerary,
                    "strexd", "\t$Rd, $Rt, $Rt2, [$Rn]",
                    []>;
}

// Clear-Exclusive is for disassembly only.
def CLREX : AXI<(outs), (ins), MiscFrm, NoItinerary, "clrex",
                [/* For disassembly only; pattern left blank */]>,
            Requires<[IsARM, HasV7]>  {
  let Inst{31-0} = 0b11110101011111111111000000011111;
}

// SWP/SWPB are deprecated in V6/V7 and for disassembly only.
let mayLoad = 1 in {
def SWP  : AIswp<0, (outs GPR:$Rt), (ins GPR:$Rt2, GPR:$Rn), "swp",
             [/* For disassembly only; pattern left blank */]>;
def SWPB : AIswp<1, (outs GPR:$Rt), (ins GPR:$Rt2, GPR:$Rn), "swpb",
             [/* For disassembly only; pattern left blank */]>;
}

//===----------------------------------------------------------------------===//
// TLS Instructions
//

// __aeabi_read_tp preserves the registers r1-r3.
// FIXME: This needs to be a pseudo of some sort so that we can get the
// encoding right, complete with fixup for the aeabi_read_tp function.
let isCall = 1,
  Defs = [R0, R12, LR, CPSR] in {
  def TPsoft : ABXI<0b1011, (outs), (ins), IIC_Br,
               "bl\t__aeabi_read_tp",
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
//   A constant value is passed in $val, and we use the location as a scratch.
//
// These are pseudo-instructions and are lowered to individual MC-insts, so
// no encoding information is necessary.
let Defs =
  [ R0,  R1,  R2,  R3,  R4,  R5,  R6,  R7,  R8,  R9,  R10, R11, R12, LR,  D0,
    D1,  D2,  D3,  D4,  D5,  D6,  D7,  D8,  D9,  D10, D11, D12, D13, D14, D15,
    D16, D17, D18, D19, D20, D21, D22, D23, D24, D25, D26, D27, D28, D29, D30,
    D31 ], hasSideEffects = 1, isBarrier = 1 in {
  def Int_eh_sjlj_setjmp : XI<(outs), (ins GPR:$src, GPR:$val),
                               AddrModeNone, SizeSpecial, IndexModeNone,
                               Pseudo, NoItinerary, "", "",
                         [(set R0, (ARMeh_sjlj_setjmp GPR:$src, GPR:$val))]>,
                           Requires<[IsARM, HasVFP2]>;
}

let Defs =
  [ R0,  R1,  R2,  R3,  R4,  R5,  R6,  R7,  R8,  R9,  R10, R11, R12, LR ],
  hasSideEffects = 1, isBarrier = 1 in {
  def Int_eh_sjlj_setjmp_nofp : XI<(outs), (ins GPR:$src, GPR:$val),
                                   AddrModeNone, SizeSpecial, IndexModeNone,
                                   Pseudo, NoItinerary, "", "",
                         [(set R0, (ARMeh_sjlj_setjmp GPR:$src, GPR:$val))]>,
                                Requires<[IsARM, NoVFP]>;
}

// FIXME: Non-Darwin version(s)
let isBarrier = 1, hasSideEffects = 1, isTerminator = 1,
    Defs = [ R7, LR, SP ] in {
def Int_eh_sjlj_longjmp : XI<(outs), (ins GPR:$src, GPR:$scratch),
                             AddrModeNone, SizeSpecial, IndexModeNone,
                             Pseudo, NoItinerary, "", "",
                         [(ARMeh_sjlj_longjmp GPR:$src, GPR:$scratch)]>,
                                Requires<[IsARM, IsDarwin]>;
}

// eh.sjlj.dispatchsetup pseudo-instruction.
// This pseudo is used for ARM, Thumb1 and Thumb2. Any differences are
// handled when the pseudo is expanded (which happens before any passes
// that need the instruction size).
let isBarrier = 1, hasSideEffects = 1 in
def Int_eh_sjlj_dispatchsetup :
 PseudoInst<(outs), (ins GPR:$src), NoItinerary, "",
            [(ARMeh_sjlj_dispatchsetup GPR:$src)]>,
              Requires<[IsDarwin]>;

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//

// Large immediate handling.

// Two piece so_imms.
// FIXME: Remove this when we can do generalized remat.
let isReMaterializable = 1 in
def MOVi2pieces : PseudoInst<(outs GPR:$dst), (ins so_imm2part:$src),
                             IIC_iMOVix2, "",
                             [(set GPR:$dst, (so_imm2part:$src))]>,
                  Requires<[IsARM, NoV6T2]>;

def : ARMPat<(or GPR:$LHS, so_imm2part:$RHS),
             (ORRri (ORRri GPR:$LHS, (so_imm2part_1 imm:$RHS)),
                    (so_imm2part_2 imm:$RHS))>;
def : ARMPat<(xor GPR:$LHS, so_imm2part:$RHS),
             (EORri (EORri GPR:$LHS, (so_imm2part_1 imm:$RHS)),
                    (so_imm2part_2 imm:$RHS))>;
def : ARMPat<(add GPR:$LHS, so_imm2part:$RHS),
             (ADDri (ADDri GPR:$LHS, (so_imm2part_1 imm:$RHS)),
                    (so_imm2part_2 imm:$RHS))>;
def : ARMPat<(add GPR:$LHS, so_neg_imm2part:$RHS),
             (SUBri (SUBri GPR:$LHS, (so_neg_imm2part_1 imm:$RHS)),
                    (so_neg_imm2part_2 imm:$RHS))>;

// 32-bit immediate using movw + movt.
// This is a single pseudo instruction, the benefit is that it can be remat'd
// as a single unit instead of having to handle reg inputs.
// FIXME: Remove this when we can do generalized remat.
let isReMaterializable = 1 in
def MOVi32imm : PseudoInst<(outs GPR:$dst), (ins i32imm:$src), IIC_iMOVix2, "",
                           [(set GPR:$dst, (i32 imm:$src))]>,
                           Requires<[IsARM, HasV6T2]>;

// ConstantPool, GlobalAddress, and JumpTable
def : ARMPat<(ARMWrapper  tglobaladdr :$dst), (LEApcrel tglobaladdr :$dst)>,
            Requires<[IsARM, DontUseMovt]>;
def : ARMPat<(ARMWrapper  tconstpool  :$dst), (LEApcrel tconstpool  :$dst)>;
def : ARMPat<(ARMWrapper  tglobaladdr :$dst), (MOVi32imm tglobaladdr :$dst)>,
            Requires<[IsARM, UseMovt]>;
def : ARMPat<(ARMWrapperJT tjumptable:$dst, imm:$id),
             (LEApcrelJT tjumptable:$dst, imm:$id)>;

// TODO: add,sub,and, 3-instr forms?

// Tail calls
def : ARMPat<(ARMtcret tcGPR:$dst),
          (TCRETURNri tcGPR:$dst)>, Requires<[IsDarwin]>;

def : ARMPat<(ARMtcret (i32 tglobaladdr:$dst)),
          (TCRETURNdi texternalsym:$dst)>, Requires<[IsDarwin]>;

def : ARMPat<(ARMtcret (i32 texternalsym:$dst)),
          (TCRETURNdi texternalsym:$dst)>, Requires<[IsDarwin]>;

def : ARMPat<(ARMtcret tcGPR:$dst),
          (TCRETURNriND tcGPR:$dst)>, Requires<[IsNotDarwin]>;

def : ARMPat<(ARMtcret (i32 tglobaladdr:$dst)),
          (TCRETURNdiND texternalsym:$dst)>, Requires<[IsNotDarwin]>;

def : ARMPat<(ARMtcret (i32 texternalsym:$dst)),
          (TCRETURNdiND texternalsym:$dst)>, Requires<[IsNotDarwin]>;

// Direct calls
def : ARMPat<(ARMcall texternalsym:$func), (BL texternalsym:$func)>,
      Requires<[IsARM, IsNotDarwin]>;
def : ARMPat<(ARMcall texternalsym:$func), (BLr9 texternalsym:$func)>,
      Requires<[IsARM, IsDarwin]>;

// zextload i1 -> zextload i8
def : ARMPat<(zextloadi1 addrmode_imm12:$addr), (LDRBi12 addrmode_imm12:$addr)>;
def : ARMPat<(zextloadi1 ldst_so_reg:$addr),    (LDRBrs ldst_so_reg:$addr)>;

// extload -> zextload
def : ARMPat<(extloadi1 addrmode_imm12:$addr),  (LDRBi12 addrmode_imm12:$addr)>;
def : ARMPat<(extloadi1 ldst_so_reg:$addr),     (LDRBrs ldst_so_reg:$addr)>;
def : ARMPat<(extloadi8 addrmode_imm12:$addr),  (LDRBi12 addrmode_imm12:$addr)>;
def : ARMPat<(extloadi8 ldst_so_reg:$addr),     (LDRBrs ldst_so_reg:$addr)>;

def : ARMPat<(extloadi16 addrmode3:$addr),  (LDRH addrmode3:$addr)>;

def : ARMPat<(extloadi8  addrmodepc:$addr), (PICLDRB addrmodepc:$addr)>;
def : ARMPat<(extloadi16 addrmodepc:$addr), (PICLDRH addrmodepc:$addr)>;

// smul* and smla*
def : ARMV5TEPat<(mul (sra (shl GPR:$a, (i32 16)), (i32 16)),
                      (sra (shl GPR:$b, (i32 16)), (i32 16))),
                 (SMULBB GPR:$a, GPR:$b)>;
def : ARMV5TEPat<(mul sext_16_node:$a, sext_16_node:$b),
                 (SMULBB GPR:$a, GPR:$b)>;
def : ARMV5TEPat<(mul (sra (shl GPR:$a, (i32 16)), (i32 16)),
                      (sra GPR:$b, (i32 16))),
                 (SMULBT GPR:$a, GPR:$b)>;
def : ARMV5TEPat<(mul sext_16_node:$a, (sra GPR:$b, (i32 16))),
                 (SMULBT GPR:$a, GPR:$b)>;
def : ARMV5TEPat<(mul (sra GPR:$a, (i32 16)),
                      (sra (shl GPR:$b, (i32 16)), (i32 16))),
                 (SMULTB GPR:$a, GPR:$b)>;
def : ARMV5TEPat<(mul (sra GPR:$a, (i32 16)), sext_16_node:$b),
                (SMULTB GPR:$a, GPR:$b)>;
def : ARMV5TEPat<(sra (mul GPR:$a, (sra (shl GPR:$b, (i32 16)), (i32 16))),
                      (i32 16)),
                 (SMULWB GPR:$a, GPR:$b)>;
def : ARMV5TEPat<(sra (mul GPR:$a, sext_16_node:$b), (i32 16)),
                 (SMULWB GPR:$a, GPR:$b)>;

def : ARMV5TEPat<(add GPR:$acc,
                      (mul (sra (shl GPR:$a, (i32 16)), (i32 16)),
                           (sra (shl GPR:$b, (i32 16)), (i32 16)))),
                 (SMLABB GPR:$a, GPR:$b, GPR:$acc)>;
def : ARMV5TEPat<(add GPR:$acc,
                      (mul sext_16_node:$a, sext_16_node:$b)),
                 (SMLABB GPR:$a, GPR:$b, GPR:$acc)>;
def : ARMV5TEPat<(add GPR:$acc,
                      (mul (sra (shl GPR:$a, (i32 16)), (i32 16)),
                           (sra GPR:$b, (i32 16)))),
                 (SMLABT GPR:$a, GPR:$b, GPR:$acc)>;
def : ARMV5TEPat<(add GPR:$acc,
                      (mul sext_16_node:$a, (sra GPR:$b, (i32 16)))),
                 (SMLABT GPR:$a, GPR:$b, GPR:$acc)>;
def : ARMV5TEPat<(add GPR:$acc,
                      (mul (sra GPR:$a, (i32 16)),
                           (sra (shl GPR:$b, (i32 16)), (i32 16)))),
                 (SMLATB GPR:$a, GPR:$b, GPR:$acc)>;
def : ARMV5TEPat<(add GPR:$acc,
                      (mul (sra GPR:$a, (i32 16)), sext_16_node:$b)),
                 (SMLATB GPR:$a, GPR:$b, GPR:$acc)>;
def : ARMV5TEPat<(add GPR:$acc,
                      (sra (mul GPR:$a, (sra (shl GPR:$b, (i32 16)), (i32 16))),
                           (i32 16))),
                 (SMLAWB GPR:$a, GPR:$b, GPR:$acc)>;
def : ARMV5TEPat<(add GPR:$acc,
                      (sra (mul GPR:$a, sext_16_node:$b), (i32 16))),
                 (SMLAWB GPR:$a, GPR:$b, GPR:$acc)>;

//===----------------------------------------------------------------------===//
// Thumb Support
//

include "ARMInstrThumb.td"

//===----------------------------------------------------------------------===//
// Thumb2 Support
//

include "ARMInstrThumb2.td"

//===----------------------------------------------------------------------===//
// Floating Point Support
//

include "ARMInstrVFP.td"

//===----------------------------------------------------------------------===//
// Advanced SIMD (NEON) Support
//

include "ARMInstrNEON.td"

//===----------------------------------------------------------------------===//
// Coprocessor Instructions.  For disassembly only.
//

def CDP : ABI<0b1110, (outs), (ins nohash_imm:$cop, i32imm:$opc1,
            nohash_imm:$CRd, nohash_imm:$CRn, nohash_imm:$CRm, i32imm:$opc2),
            NoItinerary, "cdp", "\tp$cop, $opc1, cr$CRd, cr$CRn, cr$CRm, $opc2",
              [/* For disassembly only; pattern left blank */]> {
  let Inst{4} = 0;
}

def CDP2 : ABXI<0b1110, (outs), (ins nohash_imm:$cop, i32imm:$opc1,
               nohash_imm:$CRd, nohash_imm:$CRn, nohash_imm:$CRm, i32imm:$opc2),
               NoItinerary, "cdp2\tp$cop, $opc1, cr$CRd, cr$CRn, cr$CRm, $opc2",
               [/* For disassembly only; pattern left blank */]> {
  let Inst{31-28} = 0b1111;
  let Inst{4} = 0;
}

class ACI<dag oops, dag iops, string opc, string asm>
  : I<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, BrFrm, NoItinerary,
      opc, asm, "", [/* For disassembly only; pattern left blank */]> {
  let Inst{27-25} = 0b110;
}

multiclass LdStCop<bits<4> op31_28, bit load, string opc> {

  def _OFFSET : ACI<(outs),
      (ins nohash_imm:$cop, nohash_imm:$CRd, addrmode2:$addr),
      opc, "\tp$cop, cr$CRd, $addr"> {
    let Inst{31-28} = op31_28;
    let Inst{24} = 1; // P = 1
    let Inst{21} = 0; // W = 0
    let Inst{22} = 0; // D = 0
    let Inst{20} = load;
  }

  def _PRE : ACI<(outs),
      (ins nohash_imm:$cop, nohash_imm:$CRd, addrmode2:$addr),
      opc, "\tp$cop, cr$CRd, $addr!"> {
    let Inst{31-28} = op31_28;
    let Inst{24} = 1; // P = 1
    let Inst{21} = 1; // W = 1
    let Inst{22} = 0; // D = 0
    let Inst{20} = load;
  }

  def _POST : ACI<(outs),
      (ins nohash_imm:$cop, nohash_imm:$CRd, GPR:$base, am2offset:$offset),
      opc, "\tp$cop, cr$CRd, [$base], $offset"> {
    let Inst{31-28} = op31_28;
    let Inst{24} = 0; // P = 0
    let Inst{21} = 1; // W = 1
    let Inst{22} = 0; // D = 0
    let Inst{20} = load;
  }

  def _OPTION : ACI<(outs),
      (ins nohash_imm:$cop, nohash_imm:$CRd, GPR:$base, i32imm:$option),
      opc, "\tp$cop, cr$CRd, [$base], $option"> {
    let Inst{31-28} = op31_28;
    let Inst{24} = 0; // P = 0
    let Inst{23} = 1; // U = 1
    let Inst{21} = 0; // W = 0
    let Inst{22} = 0; // D = 0
    let Inst{20} = load;
  }

  def L_OFFSET : ACI<(outs),
      (ins nohash_imm:$cop, nohash_imm:$CRd, addrmode2:$addr),
      !strconcat(opc, "l"), "\tp$cop, cr$CRd, $addr"> {
    let Inst{31-28} = op31_28;
    let Inst{24} = 1; // P = 1
    let Inst{21} = 0; // W = 0
    let Inst{22} = 1; // D = 1
    let Inst{20} = load;
  }

  def L_PRE : ACI<(outs),
      (ins nohash_imm:$cop, nohash_imm:$CRd, addrmode2:$addr),
      !strconcat(opc, "l"), "\tp$cop, cr$CRd, $addr!"> {
    let Inst{31-28} = op31_28;
    let Inst{24} = 1; // P = 1
    let Inst{21} = 1; // W = 1
    let Inst{22} = 1; // D = 1
    let Inst{20} = load;
  }

  def L_POST : ACI<(outs),
      (ins nohash_imm:$cop, nohash_imm:$CRd, GPR:$base, am2offset:$offset),
      !strconcat(opc, "l"), "\tp$cop, cr$CRd, [$base], $offset"> {
    let Inst{31-28} = op31_28;
    let Inst{24} = 0; // P = 0
    let Inst{21} = 1; // W = 1
    let Inst{22} = 1; // D = 1
    let Inst{20} = load;
  }

  def L_OPTION : ACI<(outs),
      (ins nohash_imm:$cop, nohash_imm:$CRd, GPR:$base, nohash_imm:$option),
      !strconcat(opc, "l"), "\tp$cop, cr$CRd, [$base], $option"> {
    let Inst{31-28} = op31_28;
    let Inst{24} = 0; // P = 0
    let Inst{23} = 1; // U = 1
    let Inst{21} = 0; // W = 0
    let Inst{22} = 1; // D = 1
    let Inst{20} = load;
  }
}

defm LDC  : LdStCop<{?,?,?,?}, 1, "ldc">;
defm LDC2 : LdStCop<0b1111,    1, "ldc2">;
defm STC  : LdStCop<{?,?,?,?}, 0, "stc">;
defm STC2 : LdStCop<0b1111,    0, "stc2">;

def MCR : ABI<0b1110, (outs), (ins nohash_imm:$cop, i32imm:$opc1,
              GPR:$Rt, nohash_imm:$CRn, nohash_imm:$CRm, i32imm:$opc2),
              NoItinerary, "mcr", "\tp$cop, $opc1, $Rt, cr$CRn, cr$CRm, $opc2",
              [/* For disassembly only; pattern left blank */]> {
  let Inst{20} = 0;
  let Inst{4} = 1;
}

def MCR2 : ABXI<0b1110, (outs), (ins nohash_imm:$cop, i32imm:$opc1,
                GPR:$Rt, nohash_imm:$CRn, nohash_imm:$CRm, i32imm:$opc2),
                NoItinerary, "mcr2\tp$cop, $opc1, $Rt, cr$CRn, cr$CRm, $opc2",
                [/* For disassembly only; pattern left blank */]> {
  let Inst{31-28} = 0b1111;
  let Inst{20} = 0;
  let Inst{4} = 1;
}

def MRC : ABI<0b1110, (outs), (ins nohash_imm:$cop, i32imm:$opc1,
              GPR:$Rt, nohash_imm:$CRn, nohash_imm:$CRm, i32imm:$opc2),
              NoItinerary, "mrc", "\tp$cop, $opc1, $Rt, cr$CRn, cr$CRm, $opc2",
              [/* For disassembly only; pattern left blank */]> {
  let Inst{20} = 1;
  let Inst{4} = 1;
}

def MRC2 : ABXI<0b1110, (outs), (ins nohash_imm:$cop, i32imm:$opc1,
                GPR:$Rt, nohash_imm:$CRn, nohash_imm:$CRm, i32imm:$opc2),
                NoItinerary, "mrc2\tp$cop, $opc1, $Rt, cr$CRn, cr$CRm, $opc2",
                [/* For disassembly only; pattern left blank */]> {
  let Inst{31-28} = 0b1111;
  let Inst{20} = 1;
  let Inst{4} = 1;
}

def MCRR : ABI<0b1100, (outs), (ins nohash_imm:$cop, i32imm:$opc,
               GPR:$Rt, GPR:$Rt2, nohash_imm:$CRm),
               NoItinerary, "mcrr", "\tp$cop, $opc, $Rt, $Rt2, cr$CRm",
               [/* For disassembly only; pattern left blank */]> {
  let Inst{23-20} = 0b0100;
}

def MCRR2 : ABXI<0b1100, (outs), (ins nohash_imm:$cop, i32imm:$opc,
                 GPR:$Rt, GPR:$Rt2, nohash_imm:$CRm),
                 NoItinerary, "mcrr2\tp$cop, $opc, $Rt, $Rt2, cr$CRm",
                 [/* For disassembly only; pattern left blank */]> {
  let Inst{31-28} = 0b1111;
  let Inst{23-20} = 0b0100;
}

def MRRC : ABI<0b1100, (outs), (ins nohash_imm:$cop, i32imm:$opc,
               GPR:$Rt, GPR:$Rt2, nohash_imm:$CRm),
               NoItinerary, "mrrc", "\tp$cop, $opc, $Rt, $Rt2, cr$CRm",
               [/* For disassembly only; pattern left blank */]> {
  let Inst{23-20} = 0b0101;
}

def MRRC2 : ABXI<0b1100, (outs), (ins nohash_imm:$cop, i32imm:$opc,
                 GPR:$Rt, GPR:$Rt2, nohash_imm:$CRm),
                 NoItinerary, "mrrc2\tp$cop, $opc, $Rt, $Rt2, cr$CRm",
                 [/* For disassembly only; pattern left blank */]> {
  let Inst{31-28} = 0b1111;
  let Inst{23-20} = 0b0101;
}

//===----------------------------------------------------------------------===//
// Move between special register and ARM core register -- for disassembly only
//

def MRS : ABI<0b0001,(outs GPR:$dst),(ins), NoItinerary, "mrs", "\t$dst, cpsr",
              [/* For disassembly only; pattern left blank */]> {
  let Inst{23-20} = 0b0000;
  let Inst{7-4} = 0b0000;
}

def MRSsys : ABI<0b0001,(outs GPR:$dst),(ins), NoItinerary,"mrs","\t$dst, spsr",
              [/* For disassembly only; pattern left blank */]> {
  let Inst{23-20} = 0b0100;
  let Inst{7-4} = 0b0000;
}

def MSR : ABI<0b0001, (outs), (ins GPR:$src, msr_mask:$mask), NoItinerary,
              "msr", "\tcpsr$mask, $src",
              [/* For disassembly only; pattern left blank */]> {
  let Inst{23-20} = 0b0010;
  let Inst{7-4} = 0b0000;
}

def MSRi : ABI<0b0011, (outs), (ins so_imm:$a, msr_mask:$mask), NoItinerary,
              "msr", "\tcpsr$mask, $a",
              [/* For disassembly only; pattern left blank */]> {
  let Inst{23-20} = 0b0010;
  let Inst{7-4} = 0b0000;
}

def MSRsys : ABI<0b0001, (outs), (ins GPR:$src, msr_mask:$mask), NoItinerary,
              "msr", "\tspsr$mask, $src",
              [/* For disassembly only; pattern left blank */]> {
  let Inst{23-20} = 0b0110;
  let Inst{7-4} = 0b0000;
}

def MSRsysi : ABI<0b0011, (outs), (ins so_imm:$a, msr_mask:$mask), NoItinerary,
              "msr", "\tspsr$mask, $a",
              [/* For disassembly only; pattern left blank */]> {
  let Inst{23-20} = 0b0110;
  let Inst{7-4} = 0b0000;
}
