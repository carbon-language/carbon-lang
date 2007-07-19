//===- ARMInstrInfo.td - Target Description for ARM Target -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the "Instituto Nokia de Tecnologia" and
// is distributed under the University of Illinois Open Source
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
def SDT_ARMCallSeq : SDTypeProfile<0, 1, [ SDTCisVT<0, i32> ]>;

def SDT_ARMSaveCallPC : SDTypeProfile<0, 1, []>;

def SDT_ARMcall    : SDTypeProfile<0, -1, [SDTCisInt<0>]>;

def SDT_ARMCMov    : SDTypeProfile<1, 3,
                                   [SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>,
                                    SDTCisVT<3, i32>]>;

def SDT_ARMBrcond  : SDTypeProfile<0, 2,
                                   [SDTCisVT<0, OtherVT>, SDTCisVT<1, i32>]>;

def SDT_ARMBrJT    : SDTypeProfile<0, 3,
                                  [SDTCisPtrTy<0>, SDTCisVT<1, i32>,
                                   SDTCisVT<2, i32>]>;

def SDT_ARMCmp     : SDTypeProfile<0, 2, [SDTCisSameAs<0, 1>]>;

def SDT_ARMPICAdd  : SDTypeProfile<1, 2, [SDTCisSameAs<0, 1>,
                                          SDTCisPtrTy<1>, SDTCisVT<2, i32>]>;

def SDT_ARMThreadPointer : SDTypeProfile<1, 0, [SDTCisPtrTy<0>]>;

// Node definitions.
def ARMWrapper       : SDNode<"ARMISD::Wrapper",     SDTIntUnaryOp>;
def ARMWrapperJT     : SDNode<"ARMISD::WrapperJT",   SDTIntBinOp>;

def ARMcallseq_start : SDNode<"ISD::CALLSEQ_START", SDT_ARMCallSeq,
    		              [SDNPHasChain, SDNPOutFlag]>;
def ARMcallseq_end   : SDNode<"ISD::CALLSEQ_END",   SDT_ARMCallSeq,
    		              [SDNPHasChain, SDNPInFlag, SDNPOutFlag]>;

def ARMcall          : SDNode<"ARMISD::CALL", SDT_ARMcall,
                              [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;
def ARMcall_pred    : SDNode<"ARMISD::CALL_PRED", SDT_ARMcall,
                              [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;
def ARMcall_nolink   : SDNode<"ARMISD::CALL_NOLINK", SDT_ARMcall,
                              [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;

def ARMretflag       : SDNode<"ARMISD::RET_FLAG", SDTRet,
                              [SDNPHasChain, SDNPOptInFlag]>;

def ARMcmov          : SDNode<"ARMISD::CMOV", SDT_ARMCMov,
                              [SDNPInFlag]>;
def ARMcneg          : SDNode<"ARMISD::CNEG", SDT_ARMCMov,
                              [SDNPInFlag]>;

def ARMbrcond        : SDNode<"ARMISD::BRCOND", SDT_ARMBrcond,
                              [SDNPHasChain, SDNPInFlag, SDNPOutFlag]>;

def ARMbrjt          : SDNode<"ARMISD::BR_JT", SDT_ARMBrJT,
                              [SDNPHasChain]>;

def ARMcmp           : SDNode<"ARMISD::CMP", SDT_ARMCmp,
                              [SDNPOutFlag]>;

def ARMcmpNZ         : SDNode<"ARMISD::CMPNZ", SDT_ARMCmp,
                              [SDNPOutFlag]>;

def ARMpic_add       : SDNode<"ARMISD::PIC_ADD", SDT_ARMPICAdd>;

def ARMsrl_flag      : SDNode<"ARMISD::SRL_FLAG", SDTIntUnaryOp, [SDNPOutFlag]>;
def ARMsra_flag      : SDNode<"ARMISD::SRA_FLAG", SDTIntUnaryOp, [SDNPOutFlag]>;
def ARMrrx           : SDNode<"ARMISD::RRX"     , SDTIntUnaryOp, [SDNPInFlag ]>;

def ARMthread_pointer: SDNode<"ARMISD::THREAD_POINTER", SDT_ARMThreadPointer>;

//===----------------------------------------------------------------------===//
// ARM Instruction Predicate Definitions.
//
def HasV5T  : Predicate<"Subtarget->hasV5TOps()">;
def HasV5TE : Predicate<"Subtarget->hasV5TEOps()">;
def HasV6   : Predicate<"Subtarget->hasV6Ops()">;
def IsThumb : Predicate<"Subtarget->isThumb()">;
def IsARM   : Predicate<"!Subtarget->isThumb()">;

//===----------------------------------------------------------------------===//
// ARM Flag Definitions.

class RegConstraint<string C> {
  string Constraints = C;
}

//===----------------------------------------------------------------------===//
//  ARM specific transformation functions and pattern fragments.
//

// so_imm_XFORM - Return a so_imm value packed into the format described for
// so_imm def below.
def so_imm_XFORM : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(ARM_AM::getSOImmVal(N->getValue()),
                                   MVT::i32);
}]>;

// so_imm_neg_XFORM - Return a so_imm value packed into the format described for
// so_imm_neg def below.
def so_imm_neg_XFORM : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(ARM_AM::getSOImmVal(-(int)N->getValue()),
                                   MVT::i32);
}]>;

// so_imm_not_XFORM - Return a so_imm value packed into the format described for
// so_imm_not def below.
def so_imm_not_XFORM : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(ARM_AM::getSOImmVal(~(int)N->getValue()),
                                   MVT::i32);
}]>;

// rot_imm predicate - True if the 32-bit immediate is equal to 8, 16, or 24.
def rot_imm : PatLeaf<(i32 imm), [{
  int32_t v = (int32_t)N->getValue();
  return v == 8 || v == 16 || v == 24;
}]>;

/// imm1_15 predicate - True if the 32-bit immediate is in the range [1,15].
def imm1_15 : PatLeaf<(i32 imm), [{
  return (int32_t)N->getValue() >= 1 && (int32_t)N->getValue() < 16;
}]>;

/// imm16_31 predicate - True if the 32-bit immediate is in the range [16,31].
def imm16_31 : PatLeaf<(i32 imm), [{
  return (int32_t)N->getValue() >= 16 && (int32_t)N->getValue() < 32;
}]>;

def so_imm_neg : 
  PatLeaf<(imm), [{ return ARM_AM::getSOImmVal(-(int)N->getValue()) != -1; }],
          so_imm_neg_XFORM>;

def so_imm_not :
  PatLeaf<(imm), [{ return ARM_AM::getSOImmVal(~(int)N->getValue()) != -1; }],
          so_imm_not_XFORM>;

// sext_16_node predicate - True if the SDNode is sign-extended 16 or more bits.
def sext_16_node : PatLeaf<(i32 GPR:$a), [{
  return CurDAG->ComputeNumSignBits(SDOperand(N,0)) >= 17;
}]>;



//===----------------------------------------------------------------------===//
// Operand Definitions.
//

// Branch target.
def brtarget : Operand<OtherVT>;

// A list of registers separated by comma. Used by load/store multiple.
def reglist : Operand<i32> {
  let PrintMethod = "printRegisterList";
}

// An operand for the CONSTPOOL_ENTRY pseudo-instruction.
def cpinst_operand : Operand<i32> {
  let PrintMethod = "printCPInstOperand";
}

def jtblock_operand : Operand<i32> {
  let PrintMethod = "printJTBlockOperand";
}

// Local PC labels.
def pclabel : Operand<i32> {
  let PrintMethod = "printPCLabel";
}

// shifter_operand operands: so_reg and so_imm.
def so_reg : Operand<i32>,    // reg reg imm
            ComplexPattern<i32, 3, "SelectShifterOperandReg",
                            [shl,srl,sra,rotr]> {
  let PrintMethod = "printSORegOperand";
  let MIOperandInfo = (ops GPR, GPR, i32imm);
}

// so_imm - Match a 32-bit shifter_operand immediate operand, which is an
// 8-bit immediate rotated by an arbitrary number of bits.  so_imm values are
// represented in the imm field in the same 12-bit form that they are encoded
// into so_imm instructions: the 8-bit immediate is the least significant bits
// [bits 0-7], the 4-bit shift amount is the next 4 bits [bits 8-11].
def so_imm : Operand<i32>,
             PatLeaf<(imm),
                     [{ return ARM_AM::getSOImmVal(N->getValue()) != -1; }],
                     so_imm_XFORM> {
  let PrintMethod = "printSOImmOperand";
}

// Break so_imm's up into two pieces.  This handles immediates with up to 16
// bits set in them.  This uses so_imm2part to match and so_imm2part_[12] to
// get the first/second pieces.
def so_imm2part : Operand<i32>,
                  PatLeaf<(imm),
             [{ return ARM_AM::isSOImmTwoPartVal((unsigned)N->getValue()); }]> {
  let PrintMethod = "printSOImm2PartOperand";
}

def so_imm2part_1 : SDNodeXForm<imm, [{
  unsigned V = ARM_AM::getSOImmTwoPartFirst((unsigned)N->getValue());
  return CurDAG->getTargetConstant(ARM_AM::getSOImmVal(V), MVT::i32);
}]>;

def so_imm2part_2 : SDNodeXForm<imm, [{
  unsigned V = ARM_AM::getSOImmTwoPartSecond((unsigned)N->getValue());
  return CurDAG->getTargetConstant(ARM_AM::getSOImmVal(V), MVT::i32);
}]>;


// Define ARM specific addressing modes.

// addrmode2 := reg +/- reg shop imm
// addrmode2 := reg +/- imm12
//
def addrmode2 : Operand<i32>,
                ComplexPattern<i32, 3, "SelectAddrMode2", []> {
  let PrintMethod = "printAddrMode2Operand";
  let MIOperandInfo = (ops GPR:$base, GPR:$offsreg, i32imm:$offsimm);
}

def am2offset : Operand<i32>,
                ComplexPattern<i32, 2, "SelectAddrMode2Offset", []> {
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
                ComplexPattern<i32, 2, "SelectAddrMode3Offset", []> {
  let PrintMethod = "printAddrMode3OffsetOperand";
  let MIOperandInfo = (ops GPR, i32imm);
}

// addrmode4 := reg, <mode|W>
//
def addrmode4 : Operand<i32>,
                ComplexPattern<i32, 2, "", []> {
  let PrintMethod = "printAddrMode4Operand";
  let MIOperandInfo = (ops GPR, i32imm);
}

// addrmode5 := reg +/- imm8*4
//
def addrmode5 : Operand<i32>,
                ComplexPattern<i32, 2, "SelectAddrMode5", []> {
  let PrintMethod = "printAddrMode5Operand";
  let MIOperandInfo = (ops GPR, i32imm);
}

// addrmodepc := pc + reg
//
def addrmodepc : Operand<i32>,
                 ComplexPattern<i32, 2, "SelectAddrModePC", []> {
  let PrintMethod = "printAddrModePCOperand";
  let MIOperandInfo = (ops GPR, i32imm);
}

// ARM Predicate operand. Default to 14 = always (AL). Second part is CC
// register whose default is 0 (no register).
def pred : PredicateOperand<OtherVT, (ops i32imm, CCR),
                                     (ops (i32 14), (i32 zero_reg))> {
  let PrintMethod = "printPredicateOperand";
}

// Conditional code result for instructions whose 's' bit is set, e.g. subs.
//
def cc_out : OptionalDefOperand<OtherVT, (ops CCR), (ops (i32 zero_reg))> {
  let PrintMethod = "printSBitModifierOperand";
}

//===----------------------------------------------------------------------===//
// ARM Instruction flags.  These need to match ARMInstrInfo.h.
//

// Addressing mode.
class AddrMode<bits<4> val> {
  bits<4> Value = val;
}
def AddrModeNone : AddrMode<0>;
def AddrMode1    : AddrMode<1>;
def AddrMode2    : AddrMode<2>;
def AddrMode3    : AddrMode<3>;
def AddrMode4    : AddrMode<4>;
def AddrMode5    : AddrMode<5>;
def AddrModeT1   : AddrMode<6>;
def AddrModeT2   : AddrMode<7>;
def AddrModeT4   : AddrMode<8>;
def AddrModeTs   : AddrMode<9>;

// Instruction size.
class SizeFlagVal<bits<3> val> {
  bits<3> Value = val;
}
def SizeInvalid  : SizeFlagVal<0>;  // Unset.
def SizeSpecial  : SizeFlagVal<1>;  // Pseudo or special.
def Size8Bytes   : SizeFlagVal<2>;
def Size4Bytes   : SizeFlagVal<3>;
def Size2Bytes   : SizeFlagVal<4>;

// Load / store index mode.
class IndexMode<bits<2> val> {
  bits<2> Value = val;
}
def IndexModeNone : IndexMode<0>;
def IndexModePre  : IndexMode<1>;
def IndexModePost : IndexMode<2>;

//===----------------------------------------------------------------------===//
// ARM Instruction templates.
//

// ARMPat - Same as Pat<>, but requires that the compiler be in ARM mode.
class ARMPat<dag pattern, dag result> : Pat<pattern, result> {
  list<Predicate> Predicates = [IsARM];
}
class ARMV5TEPat<dag pattern, dag result> : Pat<pattern, result> {
  list<Predicate> Predicates = [IsARM, HasV5TE];
}
class ARMV6Pat<dag pattern, dag result> : Pat<pattern, result> {
  list<Predicate> Predicates = [IsARM, HasV6];
}

class InstARM<bits<4> opcod, AddrMode am, SizeFlagVal sz, IndexMode im,
              string cstr>
  : Instruction {
  let Namespace = "ARM";

  bits<4> Opcode = opcod;
  AddrMode AM = am;
  bits<4> AddrModeBits = AM.Value;
  
  SizeFlagVal SZ = sz;
  bits<3> SizeFlag = SZ.Value;

  IndexMode IM = im;
  bits<2> IndexModeBits = IM.Value;
  
  let Constraints = cstr;
}

class PseudoInst<dag oops, dag iops, string asm, list<dag> pattern>
  : InstARM<0, AddrModeNone, SizeSpecial, IndexModeNone, ""> {
  let OutOperandList = oops;
  let InOperandList = iops;
  let AsmString   = asm;
  let Pattern = pattern;
}

// Almost all ARM instructions are predicable.
class I<dag oops, dag iops, AddrMode am, SizeFlagVal sz, IndexMode im,
        string opc, string asm, string cstr, list<dag> pattern>
  // FIXME: Set all opcodes to 0 for now.
  : InstARM<0, am, sz, im, cstr> {
  let OutOperandList = oops;
  let InOperandList = !con(iops, (ops pred:$p));
  let AsmString   = !strconcat(opc, !strconcat("${p}", asm));
  let Pattern = pattern;
  list<Predicate> Predicates = [IsARM];
}

// Same as I except it can optionally modify CPSR. Note it's modeled as
// an input operand since by default it's a zero register. It will
// become an implicit def once it's "flipped".
class sI<dag oops, dag iops, AddrMode am, SizeFlagVal sz, IndexMode im,
        string opc, string asm, string cstr, list<dag> pattern>
  // FIXME: Set all opcodes to 0 for now.
  : InstARM<0, am, sz, im, cstr> {
  let OutOperandList = oops;
  let InOperandList = !con(iops, (ops pred:$p, cc_out:$s));
  let AsmString   = !strconcat(opc, !strconcat("${p}${s}", asm));
  let Pattern = pattern;
  list<Predicate> Predicates = [IsARM];
}

class AI<dag oops, dag iops, string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, opc, asm,"",pattern>;
class AsI<dag oops, dag iops, string opc, string asm, list<dag> pattern>
  : sI<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, opc,asm,"",pattern>;
class AI1<dag oops, dag iops, string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrMode1, Size4Bytes, IndexModeNone, opc, asm, "", pattern>;
class AsI1<dag oops, dag iops, string opc, string asm, list<dag> pattern>
  : sI<oops, iops, AddrMode1, Size4Bytes, IndexModeNone, opc, asm, "", pattern>;
class AI2<dag oops, dag iops, string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModeNone, opc, asm, "", pattern>;
class AI3<dag oops, dag iops, string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModeNone, opc, asm, "", pattern>;
class AI4<dag oops, dag iops, string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrMode4, Size4Bytes, IndexModeNone, opc, asm, "", pattern>;
class AI1x2<dag oops, dag iops, string opc, string asm, list<dag> pattern>
  : I<oops, iops, AddrMode1, Size8Bytes, IndexModeNone, opc, asm, "", pattern>;

// Pre-indexed ops
class AI2pr<dag oops, dag iops, string opc, string asm, string cstr,
            list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModePre, opc, asm, cstr, pattern>;
class AI3pr<dag oops, dag iops, string opc, string asm, string cstr,
            list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModePre, opc, asm, cstr, pattern>;

// Post-indexed ops
class AI2po<dag oops, dag iops, string opc, string asm, string cstr,
            list<dag> pattern>
  : I<oops, iops, AddrMode2, Size4Bytes, IndexModePost, opc, asm, cstr,pattern>;
class AI3po<dag oops, dag iops, string opc, string asm, string cstr,
            list<dag> pattern>
  : I<oops, iops, AddrMode3, Size4Bytes, IndexModePost, opc, asm, cstr,pattern>;


class BinOpFrag<dag res> : PatFrag<(ops node:$LHS, node:$RHS), res>;
class UnOpFrag <dag res> : PatFrag<(ops node:$Src), res>;


/// AI1_bin_irs - Defines a set of (op r, {so_imm|r|so_reg}) patterns for a
/// binop that produces a value.
multiclass AsI1_bin_irs<string opc, PatFrag opnode> {
  def ri : AsI1<(outs GPR:$dst), (ins GPR:$a, so_imm:$b),
               opc, " $dst, $a, $b",
               [(set GPR:$dst, (opnode GPR:$a, so_imm:$b))]>;
  def rr : AsI1<(outs GPR:$dst), (ins GPR:$a, GPR:$b),
               opc, " $dst, $a, $b",
               [(set GPR:$dst, (opnode GPR:$a, GPR:$b))]>;
  def rs : AsI1<(outs GPR:$dst), (ins GPR:$a, so_reg:$b),
               opc, " $dst, $a, $b",
               [(set GPR:$dst, (opnode GPR:$a, so_reg:$b))]>;
}

/// ASI1_bin_s_irs - Similar to AsI1_bin_irs except it sets the 's' bit so the
/// instruction modifies the CSPR register.
multiclass ASI1_bin_s_irs<string opc, PatFrag opnode> {
  def ri : AI1<(outs GPR:$dst), (ins GPR:$a, so_imm:$b),
               opc, "s $dst, $a, $b",
               [(set GPR:$dst, (opnode GPR:$a, so_imm:$b))]>, Imp<[], [CPSR]>;
  def rr : AI1<(outs GPR:$dst), (ins GPR:$a, GPR:$b),
               opc, "s $dst, $a, $b",
               [(set GPR:$dst, (opnode GPR:$a, GPR:$b))]>, Imp<[], [CPSR]>;
  def rs : AI1<(outs GPR:$dst), (ins GPR:$a, so_reg:$b),
               opc, "s $dst, $a, $b",
               [(set GPR:$dst, (opnode GPR:$a, so_reg:$b))]>, Imp<[], [CPSR]>;
}

/// AI1_cmp_irs - Defines a set of (op r, {so_imm|r|so_reg}) cmp / test
/// patterns. Similar to AsI1_bin_irs except the instruction does not produce
/// a explicit result, only implicitly set CPSR.
multiclass AI1_cmp_irs<string opc, PatFrag opnode> {
  def ri : AI1<(outs), (ins GPR:$a, so_imm:$b),
               opc, " $a, $b",
               [(opnode GPR:$a, so_imm:$b)]>, Imp<[], [CPSR]>;
  def rr : AI1<(outs), (ins GPR:$a, GPR:$b),
               opc, " $a, $b",
               [(opnode GPR:$a, GPR:$b)]>, Imp<[], [CPSR]>;
  def rs : AI1<(outs), (ins GPR:$a, so_reg:$b),
               opc, " $a, $b",
               [(opnode GPR:$a, so_reg:$b)]>, Imp<[], [CPSR]>;
}

/// AI_unary_rrot - A unary operation with two forms: one whose operand is a
/// register and one whose operand is a register rotated by 8/16/24.
multiclass AI_unary_rrot<string opc, PatFrag opnode> {
  def r     : AI<(outs GPR:$dst), (ins GPR:$Src),
                 opc, " $dst, $Src",
                 [(set GPR:$dst, (opnode GPR:$Src))]>, Requires<[IsARM, HasV6]>;
  def r_rot : AI<(outs GPR:$dst), (ins GPR:$Src, i32imm:$rot),
                 opc, " $dst, $Src, ror $rot",
                 [(set GPR:$dst, (opnode (rotr GPR:$Src, rot_imm:$rot)))]>,
              Requires<[IsARM, HasV6]>;
}

/// AI_bin_rrot - A binary operation with two forms: one whose operand is a
/// register and one whose operand is a register rotated by 8/16/24.
multiclass AI_bin_rrot<string opc, PatFrag opnode> {
  def rr     : AI<(outs GPR:$dst), (ins GPR:$LHS, GPR:$RHS),
                  opc, " $dst, $LHS, $RHS",
                  [(set GPR:$dst, (opnode GPR:$LHS, GPR:$RHS))]>,
                  Requires<[IsARM, HasV6]>;
  def rr_rot : AI<(outs GPR:$dst), (ins GPR:$LHS, GPR:$RHS, i32imm:$rot),
                  opc, " $dst, $LHS, $RHS, ror $rot",
                  [(set GPR:$dst, (opnode GPR:$LHS,
                                          (rotr GPR:$RHS, rot_imm:$rot)))]>,
                  Requires<[IsARM, HasV6]>;
}

// Special cases.
class XI<dag oops, dag iops, AddrMode am, SizeFlagVal sz, IndexMode im,
         string asm, string cstr, list<dag> pattern>
  // FIXME: Set all opcodes to 0 for now.
  : InstARM<0, am, sz, im, cstr> {
  let OutOperandList = oops;
  let InOperandList = iops;
  let AsmString   = asm;
  let Pattern = pattern;
  list<Predicate> Predicates = [IsARM];
}

class AXI<dag oops, dag iops, string asm, list<dag> pattern>
  : XI<oops, iops, AddrModeNone, Size4Bytes, IndexModeNone, asm, "", pattern>;
class AXI1<dag oops, dag iops, string asm, list<dag> pattern>
  : XI<oops, iops, AddrMode1, Size4Bytes, IndexModeNone, asm, "", pattern>;
class AXI2<dag oops, dag iops, string asm, list<dag> pattern>
  : XI<oops, iops, AddrMode2, Size4Bytes, IndexModeNone, asm, "", pattern>;
class AXI3<dag oops, dag iops, string asm, list<dag> pattern>
  : XI<oops, iops, AddrMode3, Size4Bytes, IndexModeNone, asm, "", pattern>;
class AXI4<dag oops, dag iops, string asm, list<dag> pattern>
  : XI<oops, iops, AddrMode4, Size4Bytes, IndexModeNone, asm, "", pattern>;

class AXIx2<dag oops, dag iops, string asm, list<dag> pattern>
  : XI<oops, iops, AddrModeNone, Size8Bytes, IndexModeNone, asm, "", pattern>;

// BR_JT instructions
class JTI<dag oops, dag iops, string asm, list<dag> pattern>
  : XI<oops, iops, AddrModeNone, SizeSpecial, IndexModeNone, asm, "", pattern>;
class JTI1<dag oops, dag iops, string asm, list<dag> pattern>
  : XI<oops, iops, AddrMode1, SizeSpecial, IndexModeNone, asm, "", pattern>;
class JTI2<dag oops, dag iops, string asm, list<dag> pattern>
  : XI<oops, iops, AddrMode2, SizeSpecial, IndexModeNone, asm, "", pattern>;

/// AsXI1_bin_c_irs - Same as AsI1_bin_irs but without the predicate operand and
/// setting carry bit. But it can optionally set CPSR.
multiclass AsXI1_bin_c_irs<string opc, PatFrag opnode> {
  def ri : AXI1<(outs GPR:$dst), (ins GPR:$a, so_imm:$b, cc_out:$s),
               !strconcat(opc, "${s} $dst, $a, $b"),
               [(set GPR:$dst, (opnode GPR:$a, so_imm:$b))]>, Imp<[CPSR], []>;
  def rr : AXI1<(outs GPR:$dst), (ins GPR:$a, GPR:$b, cc_out:$s),
               !strconcat(opc, "${s} $dst, $a, $b"),
               [(set GPR:$dst, (opnode GPR:$a, GPR:$b))]>, Imp<[CPSR], []>;
  def rs : AXI1<(outs GPR:$dst), (ins GPR:$a, so_reg:$b, cc_out:$s),
               !strconcat(opc, "${s} $dst, $a, $b"),
               [(set GPR:$dst, (opnode GPR:$a, so_reg:$b))]>, Imp<[CPSR], []>;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Miscellaneous Instructions.
//
def IMPLICIT_DEF_GPR : 
PseudoInst<(outs GPR:$rD), (ins pred:$p),
           "@ IMPLICIT_DEF_GPR $rD",
           [(set GPR:$rD, (undef))]>;


/// CONSTPOOL_ENTRY - This instruction represents a floating constant pool in
/// the function.  The first operand is the ID# for this instruction, the second
/// is the index into the MachineConstantPool that this is, the third is the
/// size in bytes of this constant pool entry.
let isNotDuplicable = 1 in
def CONSTPOOL_ENTRY :
PseudoInst<(outs), (ins cpinst_operand:$instid, cpinst_operand:$cpidx,
                        i32imm:$size),
           "${instid:label} ${cpidx:cpentry}", []>;

def ADJCALLSTACKUP :
PseudoInst<(outs), (ins i32imm:$amt, pred:$p),
           "@ ADJCALLSTACKUP $amt",
           [(ARMcallseq_end imm:$amt)]>, Imp<[SP],[SP]>;

def ADJCALLSTACKDOWN : 
PseudoInst<(outs), (ins i32imm:$amt, pred:$p),
           "@ ADJCALLSTACKDOWN $amt",
           [(ARMcallseq_start imm:$amt)]>, Imp<[SP],[SP]>;

def DWARF_LOC :
PseudoInst<(outs), (ins i32imm:$line, i32imm:$col, i32imm:$file),
           ".loc $file, $line, $col",
           [(dwarf_loc (i32 imm:$line), (i32 imm:$col), (i32 imm:$file))]>;

let isNotDuplicable = 1 in {
def PICADD : AXI1<(outs GPR:$dst), (ins GPR:$a, pclabel:$cp, pred:$p),
                   "$cp:\n\tadd$p $dst, pc, $a",
                   [(set GPR:$dst, (ARMpic_add GPR:$a, imm:$cp))]>;

let isLoad = 1, AddedComplexity = 10 in {
def PICLD   : AXI2<(outs GPR:$dst), (ins addrmodepc:$addr, pred:$p),
                  "${addr:label}:\n\tldr$p $dst, $addr",
                  [(set GPR:$dst, (load addrmodepc:$addr))]>;

def PICLDZH : AXI3<(outs GPR:$dst), (ins addrmodepc:$addr, pred:$p),
                  "${addr:label}:\n\tldr${p}h $dst, $addr",
                  [(set GPR:$dst, (zextloadi16 addrmodepc:$addr))]>;

def PICLDZB : AXI2<(outs GPR:$dst), (ins addrmodepc:$addr, pred:$p),
                  "${addr:label}:\n\tldr${p}b $dst, $addr",
                  [(set GPR:$dst, (zextloadi8 addrmodepc:$addr))]>;

def PICLDH  : AXI3<(outs GPR:$dst), (ins addrmodepc:$addr, pred:$p),
                  "${addr:label}:\n\tldr${p}h $dst, $addr",
                  [(set GPR:$dst, (extloadi16 addrmodepc:$addr))]>;

def PICLDB  : AXI2<(outs GPR:$dst), (ins addrmodepc:$addr, pred:$p),
                  "${addr:label}:\n\tldr${p}b $dst, $addr",
                  [(set GPR:$dst, (extloadi8 addrmodepc:$addr))]>;

def PICLDSH : AXI3<(outs GPR:$dst), (ins addrmodepc:$addr, pred:$p),
                  "${addr:label}:\n\tldr${p}sh $dst, $addr",
                  [(set GPR:$dst, (sextloadi16 addrmodepc:$addr))]>;

def PICLDSB : AXI3<(outs GPR:$dst), (ins addrmodepc:$addr, pred:$p),
                  "${addr:label}:\n\tldr${p}sb $dst, $addr",
                  [(set GPR:$dst, (sextloadi8 addrmodepc:$addr))]>;
}
let isStore = 1, AddedComplexity = 10 in {
def PICSTR  : AXI2<(outs), (ins GPR:$src, addrmodepc:$addr, pred:$p),
               "${addr:label}:\n\tstr$p $src, $addr",
               [(store GPR:$src, addrmodepc:$addr)]>;

def PICSTRH : AXI3<(outs), (ins GPR:$src, addrmodepc:$addr, pred:$p),
               "${addr:label}:\n\tstr${p}h $src, $addr",
               [(truncstorei16 GPR:$src, addrmodepc:$addr)]>;

def PICSTRB : AXI2<(outs), (ins GPR:$src, addrmodepc:$addr, pred:$p),
               "${addr:label}:\n\tstr${p}b $src, $addr",
               [(truncstorei8 GPR:$src, addrmodepc:$addr)]>;
}
}

//===----------------------------------------------------------------------===//
//  Control Flow Instructions.
//

let isReturn = 1, isTerminator = 1 in
  def BX_RET : AI<(outs), (ins), "bx", " lr", [(ARMretflag)]>;

// FIXME: remove when we have a way to marking a MI with these properties.
// FIXME: $dst1 should be a def. But the extra ops must be in the end of the
// operand list.
let isLoad = 1, isReturn = 1, isTerminator = 1 in
  def LDM_RET : AXI4<(outs),
                    (ins addrmode4:$addr, pred:$p, reglist:$dst1, variable_ops),
                    "ldm${p}${addr:submode} $addr, $dst1",
                    []>;

let isCall = 1, noResults = 1,
  Defs = [R0, R1, R2, R3, R12, LR,
          D0, D1, D2, D3, D4, D5, D6, D7, CPSR] in {
  def BL  : AXI<(outs), (ins i32imm:$func, variable_ops),
                "bl ${func:call}",
                [(ARMcall tglobaladdr:$func)]>;

  def BL_pred : AI<(outs), (ins i32imm:$func, variable_ops),
                    "bl", " ${func:call}",
                    [(ARMcall_pred tglobaladdr:$func)]>;

  // ARMv5T and above
  def BLX : AXI<(outs), (ins GPR:$func, variable_ops),
                "blx $func",
                [(ARMcall GPR:$func)]>, Requires<[IsARM, HasV5T]>;
  let Uses = [LR] in {
    // ARMv4T
    def BX : AXIx2<(outs), (ins GPR:$func, variable_ops),
                  "mov lr, pc\n\tbx $func",
                  [(ARMcall_nolink GPR:$func)]>;
  }
}

let isBranch = 1, isTerminator = 1, noResults = 1 in {
  // B is "predicable" since it can be xformed into a Bcc.
  let isBarrier = 1 in {
    let isPredicable = 1 in
    def B : AXI<(outs), (ins brtarget:$target), "b $target",
                [(br bb:$target)]>;

  let isNotDuplicable = 1 in {
  def BR_JTr : JTI<(outs), (ins GPR:$target, jtblock_operand:$jt, i32imm:$id),
                    "mov pc, $target \n$jt",
                    [(ARMbrjt GPR:$target, tjumptable:$jt, imm:$id)]>;
  def BR_JTm : JTI2<(outs), (ins addrmode2:$target, jtblock_operand:$jt, i32imm:$id),
                     "ldr pc, $target \n$jt",
                     [(ARMbrjt (i32 (load addrmode2:$target)), tjumptable:$jt,
                       imm:$id)]>;
  def BR_JTadd : JTI1<(outs), (ins GPR:$target, GPR:$idx, jtblock_operand:$jt,
                       i32imm:$id),
                       "add pc, $target, $idx \n$jt",
                       [(ARMbrjt (add GPR:$target, GPR:$idx), tjumptable:$jt,
                         imm:$id)]>;
  }
  }

  // FIXME: should be able to write a pattern for ARMBrcond, but can't use
  // a two-value operand where a dag node expects two operands. :( 
  def Bcc : AI<(outs), (ins brtarget:$target), "b", " $target",
                [/*(ARMbrcond bb:$target, imm:$cc, CCR:$ccr)*/]>;
}

//===----------------------------------------------------------------------===//
//  Load / store Instructions.
//

// Load
let isLoad = 1 in {
def LDR  : AI2<(outs GPR:$dst), (ins addrmode2:$addr),
               "ldr", " $dst, $addr",
               [(set GPR:$dst, (load addrmode2:$addr))]>;

// Special LDR for loads from non-pc-relative constpools.
let isReMaterializable = 1 in
def LDRcp : AI2<(outs GPR:$dst), (ins addrmode2:$addr),
                 "ldr", " $dst, $addr", []>;

// Loads with zero extension
def LDRH  : AI3<(outs GPR:$dst), (ins addrmode3:$addr),
                 "ldr", "h $dst, $addr",
                [(set GPR:$dst, (zextloadi16 addrmode3:$addr))]>;

def LDRB  : AI2<(outs GPR:$dst), (ins addrmode2:$addr),
                 "ldr", "b $dst, $addr",
                [(set GPR:$dst, (zextloadi8 addrmode2:$addr))]>;

// Loads with sign extension
def LDRSH : AI3<(outs GPR:$dst), (ins addrmode3:$addr),
                 "ldr", "sh $dst, $addr",
                [(set GPR:$dst, (sextloadi16 addrmode3:$addr))]>;

def LDRSB : AI3<(outs GPR:$dst), (ins addrmode3:$addr),
                 "ldr", "sb $dst, $addr",
                [(set GPR:$dst, (sextloadi8 addrmode3:$addr))]>;

// Load doubleword
def LDRD  : AI3<(outs GPR:$dst), (ins addrmode3:$addr),
                 "ldr", "d $dst, $addr",
                []>, Requires<[IsARM, HasV5T]>;

// Indexed loads
def LDR_PRE  : AI2pr<(outs GPR:$dst), (ins GPR:$base_wb, addrmode2:$addr),
                    "ldr", " $dst, $addr!", "$addr.base = $base_wb", []>;

def LDR_POST : AI2po<(outs GPR:$dst), (ins GPR:$base_wb, GPR:$base, am2offset:$offset),
                    "ldr", " $dst, [$base], $offset", "$base = $base_wb", []>;

def LDRH_PRE  : AI3pr<(outs GPR:$dst), (ins GPR:$base_wb, addrmode3:$addr),
                     "ldr", "h $dst, $addr!", "$addr.base = $base_wb", []>;

def LDRH_POST : AI3po<(outs GPR:$dst), (ins GPR:$base_wb, GPR:$base,am3offset:$offset),
                     "ldr", "h $dst, [$base], $offset", "$base = $base_wb", []>;

def LDRB_PRE  : AI2pr<(outs GPR:$dst), (ins GPR:$base_wb, addrmode2:$addr),
                     "ldr", "b $dst, $addr!", "$addr.base = $base_wb", []>;

def LDRB_POST : AI2po<(outs GPR:$dst), (ins GPR:$base_wb, GPR:$base,am2offset:$offset),
                     "ldr", "b $dst, [$base], $offset", "$base = $base_wb", []>;

def LDRSH_PRE : AI3pr<(outs GPR:$dst), (ins GPR:$base_wb, addrmode3:$addr),
                      "ldr", "sh $dst, $addr!", "$addr.base = $base_wb", []>;

def LDRSH_POST: AI3po<(outs GPR:$dst), (ins GPR:$base_wb, GPR:$base,am3offset:$offset),
                      "ldr", "sh $dst, [$base], $offset", "$base = $base_wb", []>;

def LDRSB_PRE : AI3pr<(outs GPR:$dst), (ins GPR:$base_wb, addrmode3:$addr),
                      "ldr", "sb $dst, $addr!", "$addr.base = $base_wb", []>;

def LDRSB_POST: AI3po<(outs GPR:$dst), (ins GPR:$base_wb, GPR:$base,am3offset:$offset),
                      "ldr", "sb $dst, [$base], $offset", "$base = $base_wb", []>;
} // isLoad

// Store
let isStore = 1 in {
def STR  : AI2<(outs), (ins GPR:$src, addrmode2:$addr),
               "str", " $src, $addr",
               [(store GPR:$src, addrmode2:$addr)]>;

// Stores with truncate
def STRH : AI3<(outs), (ins GPR:$src, addrmode3:$addr),
               "str", "h $src, $addr",
               [(truncstorei16 GPR:$src, addrmode3:$addr)]>;

def STRB : AI2<(outs), (ins GPR:$src, addrmode2:$addr),
               "str", "b $src, $addr",
               [(truncstorei8 GPR:$src, addrmode2:$addr)]>;

// Store doubleword
def STRD : AI3<(outs), (ins GPR:$src, addrmode3:$addr),
               "str", "d $src, $addr",
               []>, Requires<[IsARM, HasV5T]>;

// Indexed stores
def STR_PRE  : AI2pr<(outs GPR:$base_wb),
                     (ins GPR:$src, GPR:$base, am2offset:$offset),
                    "str", " $src, [$base, $offset]!", "$base = $base_wb",
                    [(set GPR:$base_wb,
                      (pre_store GPR:$src, GPR:$base, am2offset:$offset))]>;

def STR_POST : AI2po<(outs GPR:$base_wb),
                     (ins GPR:$src, GPR:$base,am2offset:$offset),
                    "str", " $src, [$base], $offset", "$base = $base_wb",
                    [(set GPR:$base_wb,
                      (post_store GPR:$src, GPR:$base, am2offset:$offset))]>;

def STRH_PRE : AI3pr<(outs GPR:$base_wb),
                     (ins GPR:$src, GPR:$base,am3offset:$offset),
                     "str", "h $src, [$base, $offset]!", "$base = $base_wb",
                    [(set GPR:$base_wb,
                      (pre_truncsti16 GPR:$src, GPR:$base,am3offset:$offset))]>;

def STRH_POST: AI3po<(outs GPR:$base_wb),
                     (ins GPR:$src, GPR:$base,am3offset:$offset),
                     "str", "h $src, [$base], $offset", "$base = $base_wb",
                    [(set GPR:$base_wb, (post_truncsti16 GPR:$src,
                                         GPR:$base, am3offset:$offset))]>;

def STRB_PRE : AI2pr<(outs GPR:$base_wb),
                     (ins GPR:$src, GPR:$base,am2offset:$offset),
                     "str", "b $src, [$base, $offset]!", "$base = $base_wb",
                    [(set GPR:$base_wb, (pre_truncsti8 GPR:$src,
                                         GPR:$base, am2offset:$offset))]>;

def STRB_POST: AI2po<(outs GPR:$base_wb),
                     (ins GPR:$src, GPR:$base,am2offset:$offset),
                     "str", "b $src, [$base], $offset", "$base = $base_wb",
                    [(set GPR:$base_wb, (post_truncsti8 GPR:$src,
                                         GPR:$base, am2offset:$offset))]>;
} // isStore

//===----------------------------------------------------------------------===//
//  Load / store multiple Instructions.
//

// FIXME: $dst1 should be a def.
let isLoad = 1 in
def LDM : AXI4<(outs),
               (ins addrmode4:$addr, pred:$p, reglist:$dst1, variable_ops),
               "ldm${p}${addr:submode} $addr, $dst1",
               []>;

let isStore = 1 in
def STM : AXI4<(outs),
               (ins addrmode4:$addr, pred:$p, reglist:$src1, variable_ops),
               "stm${p}${addr:submode} $addr, $src1",
               []>;

//===----------------------------------------------------------------------===//
//  Move Instructions.
//

def MOVr : AsI1<(outs GPR:$dst), (ins GPR:$src),
                 "mov", " $dst, $src", []>;
def MOVs : AsI1<(outs GPR:$dst), (ins so_reg:$src),
                 "mov", " $dst, $src", [(set GPR:$dst, so_reg:$src)]>;

let isReMaterializable = 1 in
def MOVi : AsI1<(outs GPR:$dst), (ins so_imm:$src),
                 "mov", " $dst, $src", [(set GPR:$dst, so_imm:$src)]>;

def MOVrx : AsI1<(outs GPR:$dst), (ins GPR:$src),
                 "mov", " $dst, $src, rrx",
                 [(set GPR:$dst, (ARMrrx GPR:$src))]>;

// These aren't really mov instructions, but we have to define them this way
// due to flag operands.

def MOVsrl_flag : AI1<(outs GPR:$dst), (ins GPR:$src),
                      "mov", "s $dst, $src, lsr #1",
                      [(set GPR:$dst, (ARMsrl_flag GPR:$src))]>, Imp<[], [CPSR]>;
def MOVsra_flag : AI1<(outs GPR:$dst), (ins GPR:$src),
                      "mov", "s $dst, $src, asr #1",
                      [(set GPR:$dst, (ARMsra_flag GPR:$src))]>, Imp<[], [CPSR]>;

//===----------------------------------------------------------------------===//
//  Extend Instructions.
//

// Sign extenders

defm SXTB  : AI_unary_rrot<"sxtb", UnOpFrag<(sext_inreg node:$Src, i8)>>;
defm SXTH  : AI_unary_rrot<"sxth", UnOpFrag<(sext_inreg node:$Src, i16)>>;

defm SXTAB : AI_bin_rrot<"sxtab",
                        BinOpFrag<(add node:$LHS, (sext_inreg node:$RHS, i8))>>;
defm SXTAH : AI_bin_rrot<"sxtah",
                        BinOpFrag<(add node:$LHS, (sext_inreg node:$RHS,i16))>>;

// TODO: SXT(A){B|H}16

// Zero extenders

let AddedComplexity = 16 in {
defm UXTB   : AI_unary_rrot<"uxtb"  , UnOpFrag<(and node:$Src, 0x000000FF)>>;
defm UXTH   : AI_unary_rrot<"uxth"  , UnOpFrag<(and node:$Src, 0x0000FFFF)>>;
defm UXTB16 : AI_unary_rrot<"uxtb16", UnOpFrag<(and node:$Src, 0x00FF00FF)>>;

def : ARMV6Pat<(and (shl GPR:$Src, 8), 0xFF00FF),
               (UXTB16r_rot GPR:$Src, 24)>;
def : ARMV6Pat<(and (srl GPR:$Src, 8), 0xFF00FF),
               (UXTB16r_rot GPR:$Src, 8)>;

defm UXTAB : AI_bin_rrot<"uxtab",
                        BinOpFrag<(add node:$LHS, (and node:$RHS, 0x00FF))>>;
defm UXTAH : AI_bin_rrot<"uxtah",
                        BinOpFrag<(add node:$LHS, (and node:$RHS, 0xFFFF))>>;
}

// This isn't safe in general, the add is two 16-bit units, not a 32-bit add.
//defm UXTAB16 : xxx<"uxtab16", 0xff00ff>;

// TODO: UXT(A){B|H}16

//===----------------------------------------------------------------------===//
//  Arithmetic Instructions.
//

defm ADD  : AsI1_bin_irs<"add", BinOpFrag<(add  node:$LHS, node:$RHS)>>;
defm SUB  : AsI1_bin_irs<"sub", BinOpFrag<(sub  node:$LHS, node:$RHS)>>;

// ADD and SUB with 's' bit set.
defm ADDS : ASI1_bin_s_irs<"add", BinOpFrag<(addc node:$LHS, node:$RHS)>>;
defm SUBS : ASI1_bin_s_irs<"sub", BinOpFrag<(subc node:$LHS, node:$RHS)>>;

// FIXME: Do not allow ADC / SBC to be predicated for now.
defm ADC  : AsXI1_bin_c_irs<"adc", BinOpFrag<(adde node:$LHS, node:$RHS)>>;
defm SBC  : AsXI1_bin_c_irs<"sbc", BinOpFrag<(sube node:$LHS, node:$RHS)>>;

// These don't define reg/reg forms, because they are handled above.
def RSBri : AsI1<(outs GPR:$dst), (ins GPR:$a, so_imm:$b),
                  "rsb", " $dst, $a, $b",
                  [(set GPR:$dst, (sub so_imm:$b, GPR:$a))]>;

def RSBrs : AsI1<(outs GPR:$dst), (ins GPR:$a, so_reg:$b),
                  "rsb", " $dst, $a, $b",
                  [(set GPR:$dst, (sub so_reg:$b, GPR:$a))]>;

// RSB with 's' bit set.
def RSBSri : AI1<(outs GPR:$dst), (ins GPR:$a, so_imm:$b),
                 "rsb", "s $dst, $a, $b",
                 [(set GPR:$dst, (subc so_imm:$b, GPR:$a))]>, Imp<[], [CPSR]>;
def RSBSrs : AI1<(outs GPR:$dst), (ins GPR:$a, so_reg:$b),
                 "rsb", "s $dst, $a, $b",
                 [(set GPR:$dst, (subc so_reg:$b, GPR:$a))]>, Imp<[], [CPSR]>;

// FIXME: Do not allow RSC to be predicated for now. But they can set CPSR.
def RSCri : AXI1<(outs GPR:$dst), (ins GPR:$a, so_imm:$b, cc_out:$s),
                 "rsc${s} $dst, $a, $b",
                 [(set GPR:$dst, (sube so_imm:$b, GPR:$a))]>, Imp<[CPSR], []>;
def RSCrs : AXI1<(outs GPR:$dst), (ins GPR:$a, so_reg:$b, cc_out:$s),
                 "rsc${s} $dst, $a, $b",
                 [(set GPR:$dst, (sube so_reg:$b, GPR:$a))]>, Imp<[CPSR], []>;

// (sub X, imm) gets canonicalized to (add X, -imm).  Match this form.
def : ARMPat<(add    GPR:$src, so_imm_neg:$imm),
             (SUBri  GPR:$src, so_imm_neg:$imm)>;

//def : ARMPat<(addc   GPR:$src, so_imm_neg:$imm),
//             (SUBSri GPR:$src, so_imm_neg:$imm)>;
//def : ARMPat<(adde   GPR:$src, so_imm_neg:$imm),
//             (SBCri  GPR:$src, so_imm_neg:$imm)>;

// Note: These are implemented in C++ code, because they have to generate
// ADD/SUBrs instructions, which use a complex pattern that a xform function
// cannot produce.
// (mul X, 2^n+1) -> (add (X << n), X)
// (mul X, 2^n-1) -> (rsb X, (X << n))


//===----------------------------------------------------------------------===//
//  Bitwise Instructions.
//

defm AND   : AsI1_bin_irs<"and", BinOpFrag<(and node:$LHS, node:$RHS)>>;
defm ORR   : AsI1_bin_irs<"orr", BinOpFrag<(or  node:$LHS, node:$RHS)>>;
defm EOR   : AsI1_bin_irs<"eor", BinOpFrag<(xor node:$LHS, node:$RHS)>>;
defm BIC   : AsI1_bin_irs<"bic", BinOpFrag<(and node:$LHS, (not node:$RHS))>>;

def  MVNr  : AsI<(outs GPR:$dst), (ins GPR:$src),
                 "mvn", " $dst, $src", [(set GPR:$dst, (not GPR:$src))]>;
def  MVNs  : AsI<(outs GPR:$dst), (ins so_reg:$src),
                 "mvn", " $dst, $src", [(set GPR:$dst, (not so_reg:$src))]>;
let isReMaterializable = 1 in
def  MVNi  : AsI<(outs GPR:$dst), (ins so_imm:$imm),
                 "mvn", " $dst, $imm", [(set GPR:$dst, so_imm_not:$imm)]>;

def : ARMPat<(and   GPR:$src, so_imm_not:$imm),
             (BICri GPR:$src, so_imm_not:$imm)>;

//===----------------------------------------------------------------------===//
//  Multiply Instructions.
//

def MUL  : AsI<(outs GPR:$dst), (ins GPR:$a, GPR:$b),
                "mul", " $dst, $a, $b",
                [(set GPR:$dst, (mul GPR:$a, GPR:$b))]>;

def MLA  : AsI<(outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$c),
                "mla", " $dst, $a, $b, $c",
                [(set GPR:$dst, (add (mul GPR:$a, GPR:$b), GPR:$c))]>;

// Extra precision multiplies with low / high results
def SMULL : AsI<(outs GPR:$ldst, GPR:$hdst), (ins GPR:$a, GPR:$b),
                "smull", " $ldst, $hdst, $a, $b", []>;

def UMULL : AsI<(outs GPR:$ldst, GPR:$hdst), (ins GPR:$a, GPR:$b),
                "umull", " $ldst, $hdst, $a, $b", []>;

// Multiply + accumulate
def SMLAL : AsI<(outs GPR:$ldst, GPR:$hdst), (ins GPR:$a, GPR:$b),
                "smlal", " $ldst, $hdst, $a, $b", []>;

def UMLAL : AsI<(outs GPR:$ldst, GPR:$hdst), (ins GPR:$a, GPR:$b),
                "umlal", " $ldst, $hdst, $a, $b", []>;

def UMAAL : AI<(outs GPR:$ldst, GPR:$hdst), (ins GPR:$a, GPR:$b),
               "umaal", " $ldst, $hdst, $a, $b", []>,
            Requires<[IsARM, HasV6]>;

// Most significant word multiply
def SMMUL : AI<(outs GPR:$dst), (ins GPR:$a, GPR:$b),
               "smmul", " $dst, $a, $b",
               [(set GPR:$dst, (mulhs GPR:$a, GPR:$b))]>,
            Requires<[IsARM, HasV6]>;

def SMMLA : AI<(outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$c),
               "smmla", " $dst, $a, $b, $c",
               [(set GPR:$dst, (add (mulhs GPR:$a, GPR:$b), GPR:$c))]>,
            Requires<[IsARM, HasV6]>;


def SMMLS : AI<(outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$c),
               "smmls", " $dst, $a, $b, $c",
               [(set GPR:$dst, (sub GPR:$c, (mulhs GPR:$a, GPR:$b)))]>,
               Requires<[IsARM, HasV6]>;

multiclass AI_smul<string opc, PatFrag opnode> {
  def BB : AI<(outs GPR:$dst), (ins GPR:$a, GPR:$b),
              !strconcat(opc, "bb"), " $dst, $a, $b",
              [(set GPR:$dst, (opnode (sext_inreg GPR:$a, i16),
                                      (sext_inreg GPR:$b, i16)))]>,
           Requires<[IsARM, HasV5TE]>;
  def BT : AI<(outs GPR:$dst), (ins GPR:$a, GPR:$b),
              !strconcat(opc, "bt"), " $dst, $a, $b",
              [(set GPR:$dst, (opnode (sext_inreg GPR:$a, i16),
                                      (sra GPR:$b, 16)))]>,
           Requires<[IsARM, HasV5TE]>;
  def TB : AI<(outs GPR:$dst), (ins GPR:$a, GPR:$b),
              !strconcat(opc, "tb"), " $dst, $a, $b",
              [(set GPR:$dst, (opnode (sra GPR:$a, 16),
                                      (sext_inreg GPR:$b, i16)))]>,
           Requires<[IsARM, HasV5TE]>;
  def TT : AI<(outs GPR:$dst), (ins GPR:$a, GPR:$b),
              !strconcat(opc, "tt"), " $dst, $a, $b",
              [(set GPR:$dst, (opnode (sra GPR:$a, 16),
                                      (sra GPR:$b, 16)))]>,
            Requires<[IsARM, HasV5TE]>;
  def WB : AI<(outs GPR:$dst), (ins GPR:$a, GPR:$b),
              !strconcat(opc, "wb"), " $dst, $a, $b",
              [(set GPR:$dst, (sra (opnode GPR:$a,
                                    (sext_inreg GPR:$b, i16)), 16))]>,
           Requires<[IsARM, HasV5TE]>;
  def WT : AI<(outs GPR:$dst), (ins GPR:$a, GPR:$b),
              !strconcat(opc, "wt"), " $dst, $a, $b",
              [(set GPR:$dst, (sra (opnode GPR:$a,
                                    (sra GPR:$b, 16)), 16))]>,
            Requires<[IsARM, HasV5TE]>;
}

multiclass AI_smla<string opc, PatFrag opnode> {
  def BB : AI<(outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$acc),
              !strconcat(opc, "bb"), " $dst, $a, $b, $acc",
              [(set GPR:$dst, (add GPR:$acc,
                               (opnode (sext_inreg GPR:$a, i16),
                                       (sext_inreg GPR:$b, i16))))]>,
           Requires<[IsARM, HasV5TE]>;
  def BT : AI<(outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$acc),
              !strconcat(opc, "bt"), " $dst, $a, $b, $acc",
              [(set GPR:$dst, (add GPR:$acc, (opnode (sext_inreg GPR:$a, i16),
                                                     (sra GPR:$b, 16))))]>,
           Requires<[IsARM, HasV5TE]>;
  def TB : AI<(outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$acc),
              !strconcat(opc, "tb"), " $dst, $a, $b, $acc",
              [(set GPR:$dst, (add GPR:$acc, (opnode (sra GPR:$a, 16),
                                                 (sext_inreg GPR:$b, i16))))]>,
           Requires<[IsARM, HasV5TE]>;
  def TT : AI<(outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$acc),
              !strconcat(opc, "tt"), " $dst, $a, $b, $acc",
              [(set GPR:$dst, (add GPR:$acc, (opnode (sra GPR:$a, 16),
                                                     (sra GPR:$b, 16))))]>,
            Requires<[IsARM, HasV5TE]>;

  def WB : AI<(outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$acc),
              !strconcat(opc, "wb"), " $dst, $a, $b, $acc",
              [(set GPR:$dst, (add GPR:$acc, (sra (opnode GPR:$a,
                                            (sext_inreg GPR:$b, i16)), 16)))]>,
           Requires<[IsARM, HasV5TE]>;
  def WT : AI<(outs GPR:$dst), (ins GPR:$a, GPR:$b, GPR:$acc),
              !strconcat(opc, "wt"), " $dst, $a, $b, $acc",
              [(set GPR:$dst, (add GPR:$acc, (sra (opnode GPR:$a,
                                                   (sra GPR:$b, 16)), 16)))]>,
            Requires<[IsARM, HasV5TE]>;
}

defm SMUL : AI_smul<"smul", BinOpFrag<(mul node:$LHS, node:$RHS)>>;
defm SMLA : AI_smla<"smla", BinOpFrag<(mul node:$LHS, node:$RHS)>>;

// TODO: Halfword multiple accumulate long: SMLAL<x><y>
// TODO: Dual halfword multiple: SMUAD, SMUSD, SMLAD, SMLSD, SMLALD, SMLSLD

//===----------------------------------------------------------------------===//
//  Misc. Arithmetic Instructions.
//

def CLZ  : AI<(outs GPR:$dst), (ins GPR:$src),
              "clz", " $dst, $src",
              [(set GPR:$dst, (ctlz GPR:$src))]>, Requires<[IsARM, HasV5T]>;

def REV  : AI<(outs GPR:$dst), (ins GPR:$src),
              "rev", " $dst, $src",
              [(set GPR:$dst, (bswap GPR:$src))]>, Requires<[IsARM, HasV6]>;

def REV16 : AI<(outs GPR:$dst), (ins GPR:$src),
               "rev16", " $dst, $src",
               [(set GPR:$dst,
                   (or (and (srl GPR:$src, 8), 0xFF),
                       (or (and (shl GPR:$src, 8), 0xFF00),
                           (or (and (srl GPR:$src, 8), 0xFF0000),
                               (and (shl GPR:$src, 8), 0xFF000000)))))]>,
               Requires<[IsARM, HasV6]>;

def REVSH : AI<(outs GPR:$dst), (ins GPR:$src),
               "revsh", " $dst, $src",
               [(set GPR:$dst,
                  (sext_inreg
                    (or (srl (and GPR:$src, 0xFF00), 8),
                        (shl GPR:$src, 8)), i16))]>,
               Requires<[IsARM, HasV6]>;

def PKHBT : AI<(outs GPR:$dst), (ins GPR:$src1, GPR:$src2, i32imm:$shamt),
               "pkhbt", " $dst, $src1, $src2, LSL $shamt",
               [(set GPR:$dst, (or (and GPR:$src1, 0xFFFF),
                                   (and (shl GPR:$src2, (i32 imm:$shamt)),
                                        0xFFFF0000)))]>,
               Requires<[IsARM, HasV6]>;

// Alternate cases for PKHBT where identities eliminate some nodes.
def : ARMV6Pat<(or (and GPR:$src1, 0xFFFF), (and GPR:$src2, 0xFFFF0000)),
               (PKHBT GPR:$src1, GPR:$src2, 0)>;
def : ARMV6Pat<(or (and GPR:$src1, 0xFFFF), (shl GPR:$src2, imm16_31:$shamt)),
               (PKHBT GPR:$src1, GPR:$src2, imm16_31:$shamt)>;


def PKHTB : AI<(outs GPR:$dst), (ins GPR:$src1, GPR:$src2, i32imm:$shamt),
               "pkhtb", " $dst, $src1, $src2, ASR $shamt",
               [(set GPR:$dst, (or (and GPR:$src1, 0xFFFF0000),
                                   (and (sra GPR:$src2, imm16_31:$shamt),
                                        0xFFFF)))]>, Requires<[IsARM, HasV6]>;

// Alternate cases for PKHTB where identities eliminate some nodes.  Note that
// a shift amount of 0 is *not legal* here, it is PKHBT instead.
def : ARMV6Pat<(or (and GPR:$src1, 0xFFFF0000), (srl GPR:$src2, 16)),
               (PKHTB GPR:$src1, GPR:$src2, 16)>;
def : ARMV6Pat<(or (and GPR:$src1, 0xFFFF0000),
                   (and (srl GPR:$src2, imm1_15:$shamt), 0xFFFF)),
               (PKHTB GPR:$src1, GPR:$src2, imm1_15:$shamt)>;


//===----------------------------------------------------------------------===//
//  Comparison Instructions...
//

defm CMP  : AI1_cmp_irs<"cmp", BinOpFrag<(ARMcmp node:$LHS, node:$RHS)>>;
defm CMN  : AI1_cmp_irs<"cmn", BinOpFrag<(ARMcmp node:$LHS,(ineg node:$RHS))>>;

// Note that TST/TEQ don't set all the same flags that CMP does!
defm TST  : AI1_cmp_irs<"tst", BinOpFrag<(ARMcmpNZ (and node:$LHS, node:$RHS), 0)>>;
defm TEQ  : AI1_cmp_irs<"teq", BinOpFrag<(ARMcmpNZ (xor node:$LHS, node:$RHS), 0)>>;

defm CMPnz : AI1_cmp_irs<"cmp", BinOpFrag<(ARMcmpNZ node:$LHS, node:$RHS)>>;
defm CMNnz : AI1_cmp_irs<"cmn", BinOpFrag<(ARMcmpNZ node:$LHS,(ineg node:$RHS))>>;

def : ARMPat<(ARMcmp GPR:$src, so_imm_neg:$imm),
             (CMNri  GPR:$src, so_imm_neg:$imm)>;

def : ARMPat<(ARMcmpNZ GPR:$src, so_imm_neg:$imm),
             (CMNri  GPR:$src, so_imm_neg:$imm)>;


// Conditional moves
// FIXME: should be able to write a pattern for ARMcmov, but can't use
// a two-value operand where a dag node expects two operands. :( 
def MOVCCr : AI<(outs GPR:$dst), (ins GPR:$false, GPR:$true),
                "mov", " $dst, $true",
      [/*(set GPR:$dst, (ARMcmov GPR:$false, GPR:$true, imm:$cc, CCR:$ccr))*/]>,
                RegConstraint<"$false = $dst">;

def MOVCCs : AI<(outs GPR:$dst), (ins GPR:$false, so_reg:$true),
                "mov", " $dst, $true",
   [/*(set GPR:$dst, (ARMcmov GPR:$false, so_reg:$true, imm:$cc, CCR:$ccr))*/]>,
                RegConstraint<"$false = $dst">;

def MOVCCi : AI<(outs GPR:$dst), (ins GPR:$false, so_imm:$true),
                "mov", " $dst, $true",
   [/*(set GPR:$dst, (ARMcmov GPR:$false, so_imm:$true, imm:$cc, CCR:$ccr))*/]>,
                RegConstraint<"$false = $dst">;


// LEApcrel - Load a pc-relative address into a register without offending the
// assembler.
def LEApcrel : AXI1<(outs GPR:$dst), (ins i32imm:$label, pred:$p),
                   !strconcat(!strconcat(".set PCRELV${:uid}, ($label-(",
                                         "${:private}PCRELL${:uid}+8))\n"),
                              !strconcat("${:private}PCRELL${:uid}:\n\t",
                                         "add$p $dst, pc, #PCRELV${:uid}")),
                   []>;

def LEApcrelJT : AXI1<(outs GPR:$dst), (ins i32imm:$label, i32imm:$id, pred:$p),
          !strconcat(!strconcat(".set PCRELV${:uid}, (${label}_${id:no_hash}-(",
                                         "${:private}PCRELL${:uid}+8))\n"),
                              !strconcat("${:private}PCRELL${:uid}:\n\t",
                                         "add$p $dst, pc, #PCRELV${:uid}")),
                   []>;

//===----------------------------------------------------------------------===//
// TLS Instructions
//

// __aeabi_read_tp preserves the registers r1-r3.
let isCall = 1,
  Defs = [R0, R12, LR, CPSR] in {
  def TPsoft : AXI<(outs), (ins),
               "bl __aeabi_read_tp",
               [(set R0, ARMthread_pointer)]>;
}

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//

// ConstantPool, GlobalAddress, and JumpTable
def : ARMPat<(ARMWrapper  tglobaladdr :$dst), (LEApcrel tglobaladdr :$dst)>;
def : ARMPat<(ARMWrapper  tconstpool  :$dst), (LEApcrel tconstpool  :$dst)>;
def : ARMPat<(ARMWrapperJT tjumptable:$dst, imm:$id),
             (LEApcrelJT tjumptable:$dst, imm:$id)>;

// Large immediate handling.

// Two piece so_imms.
let isReMaterializable = 1 in
def MOVi2pieces : AI1x2<(outs GPR:$dst), (ins so_imm2part:$src),
                         "mov", " $dst, $src",
                         [(set GPR:$dst, so_imm2part:$src)]>;

def : ARMPat<(or GPR:$LHS, so_imm2part:$RHS),
              (ORRri (ORRri GPR:$LHS, (so_imm2part_1 imm:$RHS)),
                     (so_imm2part_2 imm:$RHS))>;
def : ARMPat<(xor GPR:$LHS, so_imm2part:$RHS),
              (EORri (EORri GPR:$LHS, (so_imm2part_1 imm:$RHS)),
                     (so_imm2part_2 imm:$RHS))>;

// TODO: add,sub,and, 3-instr forms?


// Direct calls
def : ARMPat<(ARMcall texternalsym:$func), (BL texternalsym:$func)>;

// zextload i1 -> zextload i8
def : ARMPat<(zextloadi1 addrmode2:$addr),  (LDRB addrmode2:$addr)>;

// extload -> zextload
def : ARMPat<(extloadi1  addrmode2:$addr),  (LDRB addrmode2:$addr)>;
def : ARMPat<(extloadi8  addrmode2:$addr),  (LDRB addrmode2:$addr)>;
def : ARMPat<(extloadi16 addrmode3:$addr),  (LDRH addrmode3:$addr)>;

// truncstore i1 -> truncstore i8
def : ARMPat<(truncstorei1 GPR:$src, addrmode2:$dst), 
             (STRB GPR:$src, addrmode2:$dst)>;
def : ARMPat<(pre_truncsti1 GPR:$src, GPR:$base, am2offset:$offset), 
             (STRB_PRE GPR:$src, GPR:$base, am2offset:$offset)>;
def : ARMPat<(post_truncsti1 GPR:$src, GPR:$base, am2offset:$offset), 
             (STRB_POST GPR:$src, GPR:$base, am2offset:$offset)>;

// smul* and smla*
def : ARMV5TEPat<(mul (sra (shl GPR:$a, 16), 16), (sra (shl GPR:$b, 16), 16)),
                 (SMULBB GPR:$a, GPR:$b)>;
def : ARMV5TEPat<(mul sext_16_node:$a, sext_16_node:$b),
                 (SMULBB GPR:$a, GPR:$b)>;
def : ARMV5TEPat<(mul (sra (shl GPR:$a, 16), 16), (sra GPR:$b, 16)),
                 (SMULBT GPR:$a, GPR:$b)>;
def : ARMV5TEPat<(mul sext_16_node:$a, (sra GPR:$b, 16)),
                 (SMULBT GPR:$a, GPR:$b)>;
def : ARMV5TEPat<(mul (sra GPR:$a, 16), (sra (shl GPR:$b, 16), 16)),
                 (SMULTB GPR:$a, GPR:$b)>;
def : ARMV5TEPat<(mul (sra GPR:$a, 16), sext_16_node:$b),
                (SMULTB GPR:$a, GPR:$b)>;
def : ARMV5TEPat<(sra (mul GPR:$a, (sra (shl GPR:$b, 16), 16)), 16),
                 (SMULWB GPR:$a, GPR:$b)>;
def : ARMV5TEPat<(sra (mul GPR:$a, sext_16_node:$b), 16),
                 (SMULWB GPR:$a, GPR:$b)>;

def : ARMV5TEPat<(add GPR:$acc,
                      (mul (sra (shl GPR:$a, 16), 16),
                           (sra (shl GPR:$b, 16), 16))),
                 (SMLABB GPR:$a, GPR:$b, GPR:$acc)>;
def : ARMV5TEPat<(add GPR:$acc,
                      (mul sext_16_node:$a, sext_16_node:$b)),
                 (SMLABB GPR:$a, GPR:$b, GPR:$acc)>;
def : ARMV5TEPat<(add GPR:$acc,
                      (mul (sra (shl GPR:$a, 16), 16), (sra GPR:$b, 16))),
                 (SMLABT GPR:$a, GPR:$b, GPR:$acc)>;
def : ARMV5TEPat<(add GPR:$acc,
                      (mul sext_16_node:$a, (sra GPR:$b, 16))),
                 (SMLABT GPR:$a, GPR:$b, GPR:$acc)>;
def : ARMV5TEPat<(add GPR:$acc,
                      (mul (sra GPR:$a, 16), (sra (shl GPR:$b, 16), 16))),
                 (SMLATB GPR:$a, GPR:$b, GPR:$acc)>;
def : ARMV5TEPat<(add GPR:$acc,
                      (mul (sra GPR:$a, 16), sext_16_node:$b)),
                 (SMLATB GPR:$a, GPR:$b, GPR:$acc)>;
def : ARMV5TEPat<(add GPR:$acc,
                      (sra (mul GPR:$a, (sra (shl GPR:$b, 16), 16)), 16)),
                 (SMLAWB GPR:$a, GPR:$b, GPR:$acc)>;
def : ARMV5TEPat<(add GPR:$acc,
                      (sra (mul GPR:$a, sext_16_node:$b), 16)),
                 (SMLAWB GPR:$a, GPR:$b, GPR:$acc)>;

//===----------------------------------------------------------------------===//
// Thumb Support
//

include "ARMInstrThumb.td"

//===----------------------------------------------------------------------===//
// Floating Point Support
//

include "ARMInstrVFP.td"
