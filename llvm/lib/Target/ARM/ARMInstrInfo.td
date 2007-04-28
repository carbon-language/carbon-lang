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
  return TLI.ComputeNumSignBits(SDOperand(N,0)) >= 17;
}]>;



//===----------------------------------------------------------------------===//
// Operand Definitions.
//

// Branch target.
def brtarget : Operand<OtherVT>;

// Operand for printing out a condition code.
def CCOp : Operand<i32> {
  let PrintMethod = "printCCOperand";
}

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
              dag ops, string asmstr, string cstr>
  : Instruction {
  let Namespace = "ARM";

  bits<4> Opcode = opcod;
  AddrMode AM = am;
  bits<4> AddrModeBits = AM.Value;
  
  SizeFlagVal SZ = sz;
  bits<3> SizeFlag = SZ.Value;

  IndexMode IM = im;
  bits<2> IndexModeBits = IM.Value;
  
  dag OperandList = ops;
  let AsmString   = asmstr;
  let Constraints = cstr;
}

class PseudoInst<dag ops, string asm, list<dag> pattern>
  : InstARM<0, AddrModeNone, SizeSpecial, IndexModeNone, ops, asm, ""> {
  let Pattern = pattern;
}

class I<dag ops, AddrMode am, SizeFlagVal sz, IndexMode im,
        string asm, string cstr, list<dag> pattern>
  // FIXME: Set all opcodes to 0 for now.
  : InstARM<0, am, sz, im, ops, asm, cstr> {
  let Pattern = pattern;
  list<Predicate> Predicates = [IsARM];
}

class AI<dag ops, string asm, list<dag> pattern>
  : I<ops, AddrModeNone, Size4Bytes, IndexModeNone, asm, "", pattern>;
class AI1<dag ops, string asm, list<dag> pattern>
  : I<ops, AddrMode1, Size4Bytes, IndexModeNone, asm, "", pattern>;
class AI2<dag ops, string asm, list<dag> pattern>
  : I<ops, AddrMode2, Size4Bytes, IndexModeNone, asm, "", pattern>;
class AI3<dag ops, string asm, list<dag> pattern>
  : I<ops, AddrMode3, Size4Bytes, IndexModeNone, asm, "", pattern>;
class AI4<dag ops, string asm, list<dag> pattern>
  : I<ops, AddrMode4, Size4Bytes, IndexModeNone, asm, "", pattern>;
class AIx2<dag ops, string asm, list<dag> pattern>
  : I<ops, AddrModeNone, Size8Bytes, IndexModeNone, asm, "", pattern>;
class AI1x2<dag ops, string asm, list<dag> pattern>
  : I<ops, AddrMode1, Size8Bytes, IndexModeNone, asm, "", pattern>;

// Pre-indexed ops
class AI2pr<dag ops, string asm, string cstr, list<dag> pattern>
  : I<ops, AddrMode2, Size4Bytes, IndexModePre, asm, cstr, pattern>;
class AI3pr<dag ops, string asm, string cstr, list<dag> pattern>
  : I<ops, AddrMode3, Size4Bytes, IndexModePre, asm, cstr, pattern>;

// Post-indexed ops
class AI2po<dag ops, string asm, string cstr, list<dag> pattern>
  : I<ops, AddrMode2, Size4Bytes, IndexModePost, asm, cstr, pattern>;
class AI3po<dag ops, string asm, string cstr, list<dag> pattern>
  : I<ops, AddrMode3, Size4Bytes, IndexModePost, asm, cstr, pattern>;

// BR_JT instructions
class JTI<dag ops, string asm, list<dag> pattern>
  : I<ops, AddrModeNone, SizeSpecial, IndexModeNone, asm, "", pattern>;
class JTI1<dag ops, string asm, list<dag> pattern>
  : I<ops, AddrMode1, SizeSpecial, IndexModeNone, asm, "", pattern>;
class JTI2<dag ops, string asm, list<dag> pattern>
  : I<ops, AddrMode2, SizeSpecial, IndexModeNone, asm, "", pattern>;


class BinOpFrag<dag res> : PatFrag<(ops node:$LHS, node:$RHS), res>;
class UnOpFrag <dag res> : PatFrag<(ops node:$Src), res>;


/// AI1_bin_irs - Defines a set of (op r, {so_imm|r|so_reg}) patterns for a
/// binop that produces a value.
multiclass AI1_bin_irs<string opc, PatFrag opnode> {
  def ri : AI1<(ops GPR:$dst, GPR:$a, so_imm:$b),
               !strconcat(opc, " $dst, $a, $b"),
               [(set GPR:$dst, (opnode GPR:$a, so_imm:$b))]>;
  def rr : AI1<(ops GPR:$dst, GPR:$a, GPR:$b),
               !strconcat(opc, " $dst, $a, $b"),
               [(set GPR:$dst, (opnode GPR:$a, GPR:$b))]>;
  def rs : AI1<(ops GPR:$dst, GPR:$a, so_reg:$b),
               !strconcat(opc, " $dst, $a, $b"),
               [(set GPR:$dst, (opnode GPR:$a, so_reg:$b))]>;
}

/// AI1_bin0_irs - Defines a set of (op r, {so_imm|r|so_reg}) patterns.
/// Similar to AI1_bin_irs except the instruction does not produce a result.
multiclass AI1_bin0_irs<string opc, PatFrag opnode> {
  def ri : AI1<(ops GPR:$a, so_imm:$b),
               !strconcat(opc, " $a, $b"),
               [(opnode GPR:$a, so_imm:$b)]>;
  def rr : AI1<(ops GPR:$a, GPR:$b),
               !strconcat(opc, " $a, $b"),
               [(opnode GPR:$a, GPR:$b)]>;
  def rs : AI1<(ops GPR:$a, so_reg:$b),
               !strconcat(opc, " $a, $b"),
               [(opnode GPR:$a, so_reg:$b)]>;
}

/// AI1_bin_is - Defines a set of (op r, {so_imm|so_reg}) patterns for a binop.
multiclass AI1_bin_is<string opc, PatFrag opnode> {
  def ri : AI1<(ops GPR:$dst, GPR:$a, so_imm:$b),
               !strconcat(opc, " $dst, $a, $b"),
               [(set GPR:$dst, (opnode GPR:$a, so_imm:$b))]>;
  def rs : AI1<(ops GPR:$dst, GPR:$a, so_reg:$b),
               !strconcat(opc, " $dst, $a, $b"),
               [(set GPR:$dst, (opnode GPR:$a, so_reg:$b))]>;
}

/// AI1_unary_irs - Defines a set of (op {so_imm|r|so_reg}) patterns for unary
/// ops.
multiclass AI1_unary_irs<string opc, PatFrag opnode> {
  def i : AI1<(ops GPR:$dst, so_imm:$a),
              !strconcat(opc, " $dst, $a"),
              [(set GPR:$dst, (opnode so_imm:$a))]>;
  def r : AI1<(ops GPR:$dst, GPR:$a),
              !strconcat(opc, " $dst, $a"),
              [(set GPR:$dst, (opnode GPR:$a))]>;
  def s : AI1<(ops GPR:$dst, so_reg:$a),
              !strconcat(opc, " $dst, $a"),
              [(set GPR:$dst, (opnode so_reg:$a))]>;
}

/// AI_unary_rrot - A unary operation with two forms: one whose operand is a
/// register and one whose operand is a register rotated by 8/16/24.
multiclass AI_unary_rrot<string opc, PatFrag opnode> {
  def r     : AI<(ops GPR:$dst, GPR:$Src),
                 !strconcat(opc, " $dst, $Src"),
                 [(set GPR:$dst, (opnode GPR:$Src))]>, Requires<[IsARM, HasV6]>;
  def r_rot : AI<(ops GPR:$dst, GPR:$Src, i32imm:$rot),
                 !strconcat(opc, " $dst, $Src, ror $rot"),
                 [(set GPR:$dst, (opnode (rotr GPR:$Src, rot_imm:$rot)))]>,
              Requires<[IsARM, HasV6]>;
}

/// AI_bin_rrot - A binary operation with two forms: one whose operand is a
/// register and one whose operand is a register rotated by 8/16/24.
multiclass AI_bin_rrot<string opc, PatFrag opnode> {
  def rr     : AI<(ops GPR:$dst, GPR:$LHS, GPR:$RHS),
                  !strconcat(opc, " $dst, $LHS, $RHS"),
                  [(set GPR:$dst, (opnode GPR:$LHS, GPR:$RHS))]>,
                  Requires<[IsARM, HasV6]>;
  def rr_rot : AI<(ops GPR:$dst, GPR:$LHS, GPR:$RHS, i32imm:$rot),
                  !strconcat(opc, " $dst, $LHS, $RHS, ror $rot"),
                  [(set GPR:$dst, (opnode GPR:$LHS,
                                          (rotr GPR:$RHS, rot_imm:$rot)))]>,
                  Requires<[IsARM, HasV6]>;
}


//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Miscellaneous Instructions.
//
def IMPLICIT_DEF_GPR : 
PseudoInst<(ops GPR:$rD),
           "@ IMPLICIT_DEF_GPR $rD",
           [(set GPR:$rD, (undef))]>;


/// CONSTPOOL_ENTRY - This instruction represents a floating constant pool in
/// the function.  The first operand is the ID# for this instruction, the second
/// is the index into the MachineConstantPool that this is, the third is the
/// size in bytes of this constant pool entry.
def CONSTPOOL_ENTRY :
PseudoInst<(ops cpinst_operand:$instid, cpinst_operand:$cpidx, i32imm:$size),
           "${instid:label} ${cpidx:cpentry}", []>;

def ADJCALLSTACKUP :
PseudoInst<(ops i32imm:$amt),
           "@ ADJCALLSTACKUP $amt",
           [(ARMcallseq_end imm:$amt)]>, Imp<[SP],[SP]>;

def ADJCALLSTACKDOWN : 
PseudoInst<(ops i32imm:$amt),
           "@ ADJCALLSTACKDOWN $amt",
           [(ARMcallseq_start imm:$amt)]>, Imp<[SP],[SP]>;

def DWARF_LOC :
PseudoInst<(ops i32imm:$line, i32imm:$col, i32imm:$file),
           ".loc $file, $line, $col",
           [(dwarf_loc (i32 imm:$line), (i32 imm:$col), (i32 imm:$file))]>;

def PICADD : AI1<(ops GPR:$dst, GPR:$a, pclabel:$cp),
                  "$cp:\n\tadd $dst, pc, $a",
                  [(set GPR:$dst, (ARMpic_add GPR:$a, imm:$cp))]>;
let AddedComplexity = 10 in
def PICLD : AI2<(ops GPR:$dst, addrmodepc:$addr),
                  "${addr:label}:\n\tldr $dst, $addr",
                  [(set GPR:$dst, (load addrmodepc:$addr))]>;

//===----------------------------------------------------------------------===//
//  Control Flow Instructions.
//

let isReturn = 1, isTerminator = 1 in
  def BX_RET : AI<(ops), "bx lr", [(ARMretflag)]>;

// FIXME: remove when we have a way to marking a MI with these properties.
let isLoad = 1, isReturn = 1, isTerminator = 1 in
  def LDM_RET : AI4<(ops addrmode4:$addr, reglist:$dst1, variable_ops),
                    "ldm${addr:submode} $addr, $dst1",
                    []>;

let isCall = 1, noResults = 1, 
  Defs = [R0, R1, R2, R3, R12, LR,
          D0, D1, D2, D3, D4, D5, D6, D7] in {
  def BL  : AI<(ops i32imm:$func, variable_ops),
               "bl ${func:call}",
               [(ARMcall tglobaladdr:$func)]>;
  // ARMv5T and above
  def BLX : AI<(ops GPR:$dst, variable_ops),
               "blx $dst",
               [(ARMcall GPR:$dst)]>, Requires<[IsARM, HasV5T]>;
  let Uses = [LR] in {
    // ARMv4T
    def BX : AIx2<(ops GPR:$dst, variable_ops),
                  "mov lr, pc\n\tbx $dst",
                  [(ARMcall_nolink GPR:$dst)]>;
  }
}

let isBranch = 1, isTerminator = 1, isBarrier = 1 in {
  def B : AI<(ops brtarget:$dst), "b $dst",
             [(br bb:$dst)]>;

  def BR_JTr : JTI<(ops GPR:$dst, jtblock_operand:$jt, i32imm:$id),
                    "mov pc, $dst \n$jt",
                    [(ARMbrjt GPR:$dst, tjumptable:$jt, imm:$id)]>;
  def BR_JTm : JTI2<(ops addrmode2:$dst, jtblock_operand:$jt, i32imm:$id),
                     "ldr pc, $dst \n$jt",
                     [(ARMbrjt (i32 (load addrmode2:$dst)), tjumptable:$jt,
                       imm:$id)]>;
  def BR_JTadd : JTI1<(ops GPR:$dst, GPR:$idx, jtblock_operand:$jt, i32imm:$id),
                       "add pc, $dst, $idx \n$jt",
                       [(ARMbrjt (add GPR:$dst, GPR:$idx), tjumptable:$jt,
                         imm:$id)]>;
}

let isBranch = 1, isTerminator = 1, noResults = 1, isBarrier = 1 in
  def Bcc : AI<(ops brtarget:$dst, CCOp:$cc), "b$cc $dst",
                [(ARMbrcond bb:$dst, imm:$cc)]>;

//===----------------------------------------------------------------------===//
//  Load / store Instructions.
//

// Load
let isLoad = 1 in {
def LDR  : AI2<(ops GPR:$dst, addrmode2:$addr),
               "ldr $dst, $addr",
               [(set GPR:$dst, (load addrmode2:$addr))]>;

// Special LDR for loads from non-pc-relative constpools.
let isReMaterializable = 1 in
def LDRcp : AI2<(ops GPR:$dst, addrmode2:$addr),
                 "ldr $dst, $addr", []>;

// Loads with zero extension
def LDRH  : AI3<(ops GPR:$dst, addrmode3:$addr),
                 "ldrh $dst, $addr",
                [(set GPR:$dst, (zextloadi16 addrmode3:$addr))]>;

def LDRB  : AI2<(ops GPR:$dst, addrmode2:$addr),
                 "ldrb $dst, $addr",
                [(set GPR:$dst, (zextloadi8 addrmode2:$addr))]>;

// Loads with sign extension
def LDRSH : AI3<(ops GPR:$dst, addrmode3:$addr),
                 "ldrsh $dst, $addr",
                [(set GPR:$dst, (sextloadi16 addrmode3:$addr))]>;

def LDRSB : AI3<(ops GPR:$dst, addrmode3:$addr),
                 "ldrsb $dst, $addr",
                [(set GPR:$dst, (sextloadi8 addrmode3:$addr))]>;

// Load doubleword
def LDRD  : AI3<(ops GPR:$dst, addrmode3:$addr),
                 "ldrd $dst, $addr",
                []>, Requires<[IsARM, HasV5T]>;

// Indexed loads
def LDR_PRE  : AI2pr<(ops GPR:$dst, GPR:$base_wb, addrmode2:$addr),
                    "ldr $dst, $addr!", "$addr.base = $base_wb", []>;

def LDR_POST : AI2po<(ops GPR:$dst, GPR:$base_wb, GPR:$base, am2offset:$offset),
                    "ldr $dst, [$base], $offset", "$base = $base_wb", []>;

def LDRH_PRE  : AI3pr<(ops GPR:$dst, GPR:$base_wb, addrmode3:$addr),
                     "ldrh $dst, $addr!", "$addr.base = $base_wb", []>;

def LDRH_POST : AI3po<(ops GPR:$dst, GPR:$base_wb, GPR:$base,am3offset:$offset),
                     "ldrh $dst, [$base], $offset", "$base = $base_wb", []>;

def LDRB_PRE  : AI2pr<(ops GPR:$dst, GPR:$base_wb, addrmode2:$addr),
                     "ldrb $dst, $addr!", "$addr.base = $base_wb", []>;

def LDRB_POST : AI2po<(ops GPR:$dst, GPR:$base_wb, GPR:$base,am2offset:$offset),
                     "ldrb $dst, [$base], $offset", "$base = $base_wb", []>;

def LDRSH_PRE : AI3pr<(ops GPR:$dst, GPR:$base_wb, addrmode3:$addr),
                      "ldrsh $dst, $addr!", "$addr.base = $base_wb", []>;

def LDRSH_POST: AI3po<(ops GPR:$dst, GPR:$base_wb, GPR:$base,am3offset:$offset),
                      "ldrsh $dst, [$base], $offset", "$base = $base_wb", []>;

def LDRSB_PRE : AI3pr<(ops GPR:$dst, GPR:$base_wb, addrmode3:$addr),
                      "ldrsb $dst, $addr!", "$addr.base = $base_wb", []>;

def LDRSB_POST: AI3po<(ops GPR:$dst, GPR:$base_wb, GPR:$base,am3offset:$offset),
                      "ldrsb $dst, [$base], $offset", "$base = $base_wb", []>;
} // isLoad

// Store
let isStore = 1 in {
def STR  : AI2<(ops GPR:$src, addrmode2:$addr),
               "str $src, $addr",
               [(store GPR:$src, addrmode2:$addr)]>;

// Stores with truncate
def STRH : AI3<(ops GPR:$src, addrmode3:$addr),
               "strh $src, $addr",
               [(truncstorei16 GPR:$src, addrmode3:$addr)]>;

def STRB : AI2<(ops GPR:$src, addrmode2:$addr),
               "strb $src, $addr",
               [(truncstorei8 GPR:$src, addrmode2:$addr)]>;

// Store doubleword
def STRD : AI3<(ops GPR:$src, addrmode3:$addr),
               "strd $src, $addr",
               []>, Requires<[IsARM, HasV5T]>;

// Indexed stores
def STR_PRE  : AI2pr<(ops GPR:$base_wb, GPR:$src, GPR:$base, am2offset:$offset),
                    "str $src, [$base, $offset]!", "$base = $base_wb",
                    [(set GPR:$base_wb,
                      (pre_store GPR:$src, GPR:$base, am2offset:$offset))]>;

def STR_POST : AI2po<(ops  GPR:$base_wb, GPR:$src, GPR:$base,am2offset:$offset),
                    "str $src, [$base], $offset", "$base = $base_wb",
                    [(set GPR:$base_wb,
                      (post_store GPR:$src, GPR:$base, am2offset:$offset))]>;

def STRH_PRE : AI3pr<(ops  GPR:$base_wb, GPR:$src, GPR:$base,am3offset:$offset),
                     "strh $src, [$base, $offset]!", "$base = $base_wb",
                    [(set GPR:$base_wb,
                      (pre_truncsti16 GPR:$src, GPR:$base,am3offset:$offset))]>;

def STRH_POST: AI3po<(ops  GPR:$base_wb, GPR:$src, GPR:$base,am3offset:$offset),
                     "strh $src, [$base], $offset", "$base = $base_wb",
                    [(set GPR:$base_wb, (post_truncsti16 GPR:$src,
                                         GPR:$base, am3offset:$offset))]>;

def STRB_PRE : AI2pr<(ops  GPR:$base_wb, GPR:$src, GPR:$base,am2offset:$offset),
                     "strb $src, [$base, $offset]!", "$base = $base_wb",
                    [(set GPR:$base_wb, (pre_truncsti8 GPR:$src,
                                         GPR:$base, am2offset:$offset))]>;

def STRB_POST: AI2po<(ops  GPR:$base_wb, GPR:$src, GPR:$base,am2offset:$offset),
                     "strb $src, [$base], $offset", "$base = $base_wb",
                    [(set GPR:$base_wb, (post_truncsti8 GPR:$src,
                                         GPR:$base, am2offset:$offset))]>;
} // isStore

//===----------------------------------------------------------------------===//
//  Load / store multiple Instructions.
//

let isLoad = 1 in
def LDM : AI4<(ops addrmode4:$addr, reglist:$dst1, variable_ops),
              "ldm${addr:submode} $addr, $dst1",
              []>;

let isStore = 1 in
def STM : AI4<(ops addrmode4:$addr, reglist:$src1, variable_ops),
              "stm${addr:submode} $addr, $src1",
              []>;

//===----------------------------------------------------------------------===//
//  Move Instructions.
//

def MOVr : AI1<(ops GPR:$dst, GPR:$src),
                "mov $dst, $src", []>;
def MOVs : AI1<(ops GPR:$dst, so_reg:$src),
                "mov $dst, $src", [(set GPR:$dst, so_reg:$src)]>;

let isReMaterializable = 1 in
def MOVi : AI1<(ops GPR:$dst, so_imm:$src),
                "mov $dst, $src", [(set GPR:$dst, so_imm:$src)]>;

// These aren't really mov instructions, but we have to define them this way
// due to flag operands.

def MOVsrl_flag : AI1<(ops GPR:$dst, GPR:$src),
                      "movs $dst, $src, lsr #1",
                      [(set GPR:$dst, (ARMsrl_flag GPR:$src))]>;
def MOVsra_flag : AI1<(ops GPR:$dst, GPR:$src),
                      "movs $dst, $src, asr #1",
                      [(set GPR:$dst, (ARMsra_flag GPR:$src))]>;
def MOVrx       : AI1<(ops GPR:$dst, GPR:$src),
                      "mov $dst, $src, rrx",
                      [(set GPR:$dst, (ARMrrx GPR:$src))]>;


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

defm ADD  : AI1_bin_irs<"add" , BinOpFrag<(add  node:$LHS, node:$RHS)>>;
defm ADDS : AI1_bin_irs<"adds", BinOpFrag<(addc node:$LHS, node:$RHS)>>;
defm ADC  : AI1_bin_irs<"adc" , BinOpFrag<(adde node:$LHS, node:$RHS)>>;
defm SUB  : AI1_bin_irs<"sub" , BinOpFrag<(sub  node:$LHS, node:$RHS)>>;
defm SUBS : AI1_bin_irs<"subs", BinOpFrag<(subc node:$LHS, node:$RHS)>>;
defm SBC  : AI1_bin_irs<"sbc" , BinOpFrag<(sube node:$LHS, node:$RHS)>>;

// These don't define reg/reg forms, because they are handled above.
defm RSB  : AI1_bin_is <"rsb" , BinOpFrag<(sub  node:$RHS, node:$LHS)>>;
defm RSBS : AI1_bin_is <"rsbs", BinOpFrag<(subc node:$RHS, node:$LHS)>>;
defm RSC  : AI1_bin_is <"rsc" , BinOpFrag<(sube node:$RHS, node:$LHS)>>;

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

defm AND   : AI1_bin_irs<"and", BinOpFrag<(and node:$LHS, node:$RHS)>>;
defm ORR   : AI1_bin_irs<"orr", BinOpFrag<(or  node:$LHS, node:$RHS)>>;
defm EOR   : AI1_bin_irs<"eor", BinOpFrag<(xor node:$LHS, node:$RHS)>>;
defm BIC   : AI1_bin_irs<"bic", BinOpFrag<(and node:$LHS, (not node:$RHS))>>;

def  MVNr  : AI<(ops GPR:$dst, GPR:$src),
                "mvn $dst, $src", [(set GPR:$dst, (not GPR:$src))]>;
def  MVNs  : AI<(ops GPR:$dst, so_reg:$src),
                "mvn $dst, $src", [(set GPR:$dst, (not so_reg:$src))]>;
let isReMaterializable = 1 in
def  MVNi  : AI<(ops GPR:$dst, so_imm:$imm),
                "mvn $dst, $imm", [(set GPR:$dst, so_imm_not:$imm)]>;

def : ARMPat<(and   GPR:$src, so_imm_not:$imm),
             (BICri GPR:$src, so_imm_not:$imm)>;

//===----------------------------------------------------------------------===//
//  Multiply Instructions.
//

// AI_orr - Defines a (op r, r) pattern.
class AI_orr<string opc, SDNode opnode>
  : AI<(ops GPR:$dst, GPR:$a, GPR:$b),
       !strconcat(opc, " $dst, $a, $b"),
       [(set GPR:$dst, (opnode GPR:$a, GPR:$b))]>;

// AI_oorr - Defines a (op (op r, r), r) pattern.
class AI_oorr<string opc, SDNode opnode1, SDNode opnode2>
  : AI<(ops GPR:$dst, GPR:$a, GPR:$b, GPR:$c),
       !strconcat(opc, " $dst, $a, $b, $c"),
       [(set GPR:$dst, (opnode1 (opnode2 GPR:$a, GPR:$b), GPR:$c))]>;

def MUL  : AI_orr<"mul", mul>;
def MLA  : AI_oorr<"mla", add, mul>;

// Extra precision multiplies with low / high results
def SMULL : AI<(ops GPR:$ldst, GPR:$hdst, GPR:$a, GPR:$b),
               "smull $ldst, $hdst, $a, $b",
               []>;

def UMULL : AI<(ops GPR:$ldst, GPR:$hdst, GPR:$a, GPR:$b),
               "umull $ldst, $hdst, $a, $b",
               []>;

// Multiply + accumulate
def SMLAL : AI<(ops GPR:$ldst, GPR:$hdst, GPR:$a, GPR:$b),
               "smlal $ldst, $hdst, $a, $b",
               []>;

def UMLAL : AI<(ops GPR:$ldst, GPR:$hdst, GPR:$a, GPR:$b),
               "umlal $ldst, $hdst, $a, $b",
               []>;

def UMAAL : AI<(ops GPR:$ldst, GPR:$hdst, GPR:$a, GPR:$b),
               "umaal $ldst, $hdst, $a, $b",
               []>, Requires<[IsARM, HasV6]>;

// Most significant word multiply
def SMMUL : AI_orr<"smmul", mulhs>, Requires<[IsARM, HasV6]>;
def SMMLA : AI_oorr<"smmla", add, mulhs>, Requires<[IsARM, HasV6]>;


def SMMLS : AI<(ops GPR:$dst, GPR:$a, GPR:$b, GPR:$c),
               "smmls $dst, $a, $b, $c",
               [(set GPR:$dst, (sub GPR:$c, (mulhs GPR:$a, GPR:$b)))]>,
               Requires<[IsARM, HasV6]>;

multiclass AI_smul<string opc, PatFrag opnode> {
  def BB : AI<(ops GPR:$dst, GPR:$a, GPR:$b),
              !strconcat(opc, "bb $dst, $a, $b"),
              [(set GPR:$dst, (opnode (sext_inreg GPR:$a, i16),
                                      (sext_inreg GPR:$b, i16)))]>,
           Requires<[IsARM, HasV5TE]>;
  def BT : AI<(ops GPR:$dst, GPR:$a, GPR:$b),
              !strconcat(opc, "bt $dst, $a, $b"),
              [(set GPR:$dst, (opnode (sext_inreg GPR:$a, i16),
                                      (sra GPR:$b, 16)))]>,
           Requires<[IsARM, HasV5TE]>;
  def TB : AI<(ops GPR:$dst, GPR:$a, GPR:$b),
              !strconcat(opc, "tb $dst, $a, $b"),
              [(set GPR:$dst, (opnode (sra GPR:$a, 16),
                                      (sext_inreg GPR:$b, i16)))]>,
           Requires<[IsARM, HasV5TE]>;
  def TT : AI<(ops GPR:$dst, GPR:$a, GPR:$b),
              !strconcat(opc, "tt $dst, $a, $b"),
              [(set GPR:$dst, (opnode (sra GPR:$a, 16),
                                      (sra GPR:$b, 16)))]>,
            Requires<[IsARM, HasV5TE]>;
  def WB : AI<(ops GPR:$dst, GPR:$a, GPR:$b),
              !strconcat(opc, "wb $dst, $a, $b"),
              [(set GPR:$dst, (sra (opnode GPR:$a,
                                    (sext_inreg GPR:$b, i16)), 16))]>,
           Requires<[IsARM, HasV5TE]>;
  def WT : AI<(ops GPR:$dst, GPR:$a, GPR:$b),
              !strconcat(opc, "wt $dst, $a, $b"),
              [(set GPR:$dst, (sra (opnode GPR:$a,
                                    (sra GPR:$b, 16)), 16))]>,
            Requires<[IsARM, HasV5TE]>;
}

multiclass AI_smla<string opc, PatFrag opnode> {
  def BB : AI<(ops GPR:$dst, GPR:$a, GPR:$b, GPR:$acc),
              !strconcat(opc, "bb $dst, $a, $b, $acc"),
              [(set GPR:$dst, (add GPR:$acc,
                               (opnode (sext_inreg GPR:$a, i16),
                                       (sext_inreg GPR:$b, i16))))]>,
           Requires<[IsARM, HasV5TE]>;
  def BT : AI<(ops GPR:$dst, GPR:$a, GPR:$b, GPR:$acc),
              !strconcat(opc, "bt $dst, $a, $b, $acc"),
              [(set GPR:$dst, (add GPR:$acc, (opnode (sext_inreg GPR:$a, i16),
                                                     (sra GPR:$b, 16))))]>,
           Requires<[IsARM, HasV5TE]>;
  def TB : AI<(ops GPR:$dst, GPR:$a, GPR:$b, GPR:$acc),
              !strconcat(opc, "tb $dst, $a, $b, $acc"),
              [(set GPR:$dst, (add GPR:$acc, (opnode (sra GPR:$a, 16),
                                                 (sext_inreg GPR:$b, i16))))]>,
           Requires<[IsARM, HasV5TE]>;
  def TT : AI<(ops GPR:$dst, GPR:$a, GPR:$b, GPR:$acc),
              !strconcat(opc, "tt $dst, $a, $b, $acc"),
              [(set GPR:$dst, (add GPR:$acc, (opnode (sra GPR:$a, 16),
                                                     (sra GPR:$b, 16))))]>,
            Requires<[IsARM, HasV5TE]>;

  def WB : AI<(ops GPR:$dst, GPR:$a, GPR:$b, GPR:$acc),
              !strconcat(opc, "wb $dst, $a, $b, $acc"),
              [(set GPR:$dst, (add GPR:$acc, (sra (opnode GPR:$a,
                                            (sext_inreg GPR:$b, i16)), 16)))]>,
           Requires<[IsARM, HasV5TE]>;
  def WT : AI<(ops GPR:$dst, GPR:$a, GPR:$b, GPR:$acc),
              !strconcat(opc, "wt $dst, $a, $b, $acc"),
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

def CLZ  : AI<(ops GPR:$dst, GPR:$src),
              "clz $dst, $src",
              [(set GPR:$dst, (ctlz GPR:$src))]>, Requires<[IsARM, HasV5T]>;

def REV  : AI<(ops GPR:$dst, GPR:$src),
              "rev $dst, $src",
              [(set GPR:$dst, (bswap GPR:$src))]>, Requires<[IsARM, HasV6]>;

def REV16 : AI<(ops GPR:$dst, GPR:$src),
               "rev16 $dst, $src",
               [(set GPR:$dst,
                   (or (and (srl GPR:$src, 8), 0xFF),
                       (or (and (shl GPR:$src, 8), 0xFF00),
                           (or (and (srl GPR:$src, 8), 0xFF0000),
                               (and (shl GPR:$src, 8), 0xFF000000)))))]>,
               Requires<[IsARM, HasV6]>;

def REVSH : AI<(ops GPR:$dst, GPR:$src),
               "revsh $dst, $src",
               [(set GPR:$dst,
                  (sext_inreg
                    (or (srl (and GPR:$src, 0xFF00), 8),
                        (shl GPR:$src, 8)), i16))]>,
               Requires<[IsARM, HasV6]>;

def PKHBT : AI<(ops GPR:$dst, GPR:$src1, GPR:$src2, i32imm:$shamt),
               "pkhbt $dst, $src1, $src2, LSL $shamt",
               [(set GPR:$dst, (or (and GPR:$src1, 0xFFFF),
                                   (and (shl GPR:$src2, (i32 imm:$shamt)),
                                        0xFFFF0000)))]>,
               Requires<[IsARM, HasV6]>;

// Alternate cases for PKHBT where identities eliminate some nodes.
def : ARMV6Pat<(or (and GPR:$src1, 0xFFFF), (and GPR:$src2, 0xFFFF0000)),
               (PKHBT GPR:$src1, GPR:$src2, 0)>;
def : ARMV6Pat<(or (and GPR:$src1, 0xFFFF), (shl GPR:$src2, imm16_31:$shamt)),
               (PKHBT GPR:$src1, GPR:$src2, imm16_31:$shamt)>;


def PKHTB : AI<(ops  GPR:$dst, GPR:$src1, GPR:$src2, i32imm:$shamt),
               "pkhtb $dst, $src1, $src2, ASR $shamt",
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

defm CMP  : AI1_bin0_irs<"cmp", BinOpFrag<(ARMcmp node:$LHS, node:$RHS)>>;
defm CMN  : AI1_bin0_irs<"cmn", BinOpFrag<(ARMcmp node:$LHS,(ineg node:$RHS))>>;

def : ARMPat<(ARMcmp GPR:$src, so_imm_neg:$imm),
             (CMNri  GPR:$src, so_imm_neg:$imm)>;

// Note that TST/TEQ don't set all the same flags that CMP does!
defm TST  : AI1_bin0_irs<"tst", BinOpFrag<(ARMcmpNZ (and node:$LHS, node:$RHS), 0)>>;
defm TEQ  : AI1_bin0_irs<"teq", BinOpFrag<(ARMcmpNZ (xor node:$LHS, node:$RHS), 0)>>;

defm CMPnz : AI1_bin0_irs<"cmp", BinOpFrag<(ARMcmpNZ node:$LHS, node:$RHS)>>;
defm CMNnz : AI1_bin0_irs<"cmn", BinOpFrag<(ARMcmpNZ node:$LHS,(ineg node:$RHS))>>;

def : ARMPat<(ARMcmpNZ GPR:$src, so_imm_neg:$imm),
             (CMNri  GPR:$src, so_imm_neg:$imm)>;


// Conditional moves
def MOVCCr : AI<(ops GPR:$dst, GPR:$false, GPR:$true, CCOp:$cc),
                "mov$cc $dst, $true",
                [(set GPR:$dst, (ARMcmov GPR:$false, GPR:$true, imm:$cc))]>,
                RegConstraint<"$false = $dst">;

def MOVCCs : AI<(ops GPR:$dst, GPR:$false, so_reg:$true, CCOp:$cc),
                "mov$cc $dst, $true",
                [(set GPR:$dst, (ARMcmov GPR:$false, so_reg:$true,imm:$cc))]>,
                RegConstraint<"$false = $dst">;

def MOVCCi : AI<(ops GPR:$dst, GPR:$false, so_imm:$true, CCOp:$cc),
                "mov$cc $dst, $true",
                [(set GPR:$dst, (ARMcmov GPR:$false, so_imm:$true,imm:$cc))]>,
                RegConstraint<"$false = $dst">;


// LEApcrel - Load a pc-relative address into a register without offending the
// assembler.
def LEApcrel : AI1<(ops GPR:$dst, i32imm:$label),
                   !strconcat(!strconcat(".set PCRELV${:uid}, ($label-(",
                                         "${:private}PCRELL${:uid}+8))\n"),
                              !strconcat("${:private}PCRELL${:uid}:\n\t",
                                         "add $dst, pc, #PCRELV${:uid}")),
                   []>;

def LEApcrelJT : AI1<(ops GPR:$dst, i32imm:$label, i32imm:$id),
          !strconcat(!strconcat(".set PCRELV${:uid}, (${label}_${id:no_hash}-(",
                                         "${:private}PCRELL${:uid}+8))\n"),
                              !strconcat("${:private}PCRELL${:uid}:\n\t",
                                         "add $dst, pc, #PCRELV${:uid}")),
                   []>;
//===----------------------------------------------------------------------===//
// TLS Instructions
//

// __aeabi_read_tp preserves the registers r1-r3.
let isCall = 1,
  Defs = [R0, R12, LR] in {
  def TPsoft  : AI<(ops),
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
def MOVi2pieces : AI1x2<(ops GPR:$dst, so_imm2part:$src),
                         "mov $dst, $src",
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
