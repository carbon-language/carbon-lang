//===- SystemZInstrInfo.td - SystemZ Instruction defs ---------*- tblgen-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the SystemZ instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

include "SystemZInstrFormats.td"

//===----------------------------------------------------------------------===//
// Type Constraints.
//===----------------------------------------------------------------------===//
class SDTCisI8<int OpNum> : SDTCisVT<OpNum, i8>;
class SDTCisI16<int OpNum> : SDTCisVT<OpNum, i16>;
class SDTCisI32<int OpNum> : SDTCisVT<OpNum, i32>;
class SDTCisI64<int OpNum> : SDTCisVT<OpNum, i64>;

//===----------------------------------------------------------------------===//
// Type Profiles.
//===----------------------------------------------------------------------===//
def SDT_SystemZCall         : SDTypeProfile<0, -1, [SDTCisPtrTy<0>]>;
def SDT_SystemZCallSeqStart : SDCallSeqStart<[SDTCisI64<0>]>;
def SDT_SystemZCallSeqEnd   : SDCallSeqEnd<[SDTCisI64<0>, SDTCisI64<1>]>;
def SDT_CmpTest             : SDTypeProfile<0, 2, [SDTCisSameAs<0, 1>]>;
def SDT_BrCond              : SDTypeProfile<0, 2,
                                           [SDTCisVT<0, OtherVT>,
                                            SDTCisI8<1>]>;
def SDT_SelectCC            : SDTypeProfile<1, 3,
                                           [SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>,
                                            SDTCisI8<3>]>;
def SDT_Address             : SDTypeProfile<1, 1,
                                            [SDTCisSameAs<0, 1>, SDTCisPtrTy<0>]>;

//===----------------------------------------------------------------------===//
// SystemZ Specific Node Definitions.
//===----------------------------------------------------------------------===//
def SystemZretflag : SDNode<"SystemZISD::RET_FLAG", SDTNone,
                     [SDNPHasChain, SDNPOptInFlag]>;
def SystemZcall    : SDNode<"SystemZISD::CALL", SDT_SystemZCall,
                     [SDNPHasChain, SDNPOutFlag, SDNPOptInFlag]>;
def SystemZcallseq_start :
                 SDNode<"ISD::CALLSEQ_START", SDT_SystemZCallSeqStart,
                        [SDNPHasChain, SDNPOutFlag]>;
def SystemZcallseq_end :
                 SDNode<"ISD::CALLSEQ_END",   SDT_SystemZCallSeqEnd,
                        [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;
def SystemZcmp     : SDNode<"SystemZISD::CMP", SDT_CmpTest, [SDNPOutFlag]>;
def SystemZucmp    : SDNode<"SystemZISD::UCMP", SDT_CmpTest, [SDNPOutFlag]>;
def SystemZbrcond  : SDNode<"SystemZISD::BRCOND", SDT_BrCond,
                            [SDNPHasChain, SDNPInFlag]>;
def SystemZselect  : SDNode<"SystemZISD::SELECT", SDT_SelectCC, [SDNPInFlag]>;
def SystemZpcrelwrapper : SDNode<"SystemZISD::PCRelativeWrapper", SDT_Address, []>;

//===----------------------------------------------------------------------===//
// Instruction Pattern Stuff.
//===----------------------------------------------------------------------===//

// SystemZ specific condition code. These correspond to CondCode in
// SystemZ.h. They must be kept in synch.
def SYSTEMZ_COND_E  : PatLeaf<(i8 0)>;
def SYSTEMZ_COND_NE : PatLeaf<(i8 1)>;
def SYSTEMZ_COND_H  : PatLeaf<(i8 2)>;
def SYSTEMZ_COND_L  : PatLeaf<(i8 3)>;
def SYSTEMZ_COND_HE : PatLeaf<(i8 4)>;
def SYSTEMZ_COND_LE : PatLeaf<(i8 5)>;

def LL16 : SDNodeXForm<imm, [{
  // Transformation function: return low 16 bits.
  return getI16Imm(N->getZExtValue() & 0x000000000000FFFFULL);
}]>;

def LH16 : SDNodeXForm<imm, [{
  // Transformation function: return bits 16-31.
  return getI16Imm((N->getZExtValue() & 0x00000000FFFF0000ULL) >> 16);
}]>;

def HL16 : SDNodeXForm<imm, [{
  // Transformation function: return bits 32-47.
  return getI16Imm((N->getZExtValue() & 0x0000FFFF00000000ULL) >> 32);
}]>;

def HH16 : SDNodeXForm<imm, [{
  // Transformation function: return bits 48-63.
  return getI16Imm((N->getZExtValue() & 0xFFFF000000000000ULL) >> 48);
}]>;

def LO32 : SDNodeXForm<imm, [{
  // Transformation function: return low 32 bits.
  return getI32Imm(N->getZExtValue() & 0x00000000FFFFFFFFULL);
}]>;

def HI32 : SDNodeXForm<imm, [{
  // Transformation function: return bits 32-63.
  return getI32Imm(N->getZExtValue() >> 32);
}]>;

def i64ll16 : PatLeaf<(imm), [{  
  // i64ll16 predicate - true if the 64-bit immediate has only rightmost 16
  // bits set.
  return ((N->getZExtValue() & 0x000000000000FFFFULL) == N->getZExtValue());
}], LL16>;

def i64lh16 : PatLeaf<(imm), [{  
  // i64lh16 predicate - true if the 64-bit immediate has only bits 16-31 set.
  return ((N->getZExtValue() & 0x00000000FFFF0000ULL) == N->getZExtValue());
}], LH16>;

def i64hl16 : PatLeaf<(i64 imm), [{  
  // i64hl16 predicate - true if the 64-bit immediate has only bits 32-47 set.
  return ((N->getZExtValue() & 0x0000FFFF00000000ULL) == N->getZExtValue());
}], HL16>;

def i64hh16 : PatLeaf<(i64 imm), [{  
  // i64hh16 predicate - true if the 64-bit immediate has only bits 48-63 set.
  return ((N->getZExtValue() & 0xFFFF000000000000ULL) == N->getZExtValue());
}], HH16>;

def immSExt16 : PatLeaf<(imm), [{
  // immSExt16 predicate - true if the immediate fits in a 16-bit sign extended
  // field.
  if (N->getValueType(0) == MVT::i64) {
    uint64_t val = N->getZExtValue();
    return ((int64_t)val == (int16_t)val);
  } else if (N->getValueType(0) == MVT::i32) {
    uint32_t val = N->getZExtValue();
    return ((int32_t)val == (int16_t)val);
  }

  return false;
}]>;

def immSExt32 : PatLeaf<(i64 imm), [{
  // immSExt32 predicate - true if the immediate fits in a 32-bit sign extended
  // field.
  uint64_t val = N->getZExtValue();
  return ((int64_t)val == (int32_t)val);
}]>;

def i64lo32 : PatLeaf<(i64 imm), [{
  // i64lo32 predicate - true if the 64-bit immediate has only rightmost 32
  // bits set.
  return ((N->getZExtValue() & 0x00000000FFFFFFFFULL) == N->getZExtValue());
}], LO32>;

def i64hi32 : PatLeaf<(i64 imm), [{
  // i64hi32 predicate - true if the 64-bit immediate has only bits 32-63 set.
  return ((N->getZExtValue() & 0xFFFFFFFF00000000ULL) == N->getZExtValue());
}], HI32>;

def i32immSExt8  : PatLeaf<(i32 imm), [{
  // i32immSExt8 predicate - True if the 32-bit immediate fits in a 8-bit
  // sign extended field.
  return (int32_t)N->getZExtValue() == (int8_t)N->getZExtValue();
}]>;

def i32immSExt16 : PatLeaf<(i32 imm), [{
  // i32immSExt16 predicate - True if the 32-bit immediate fits in a 16-bit
  // sign extended field.
  return (int32_t)N->getZExtValue() == (int16_t)N->getZExtValue();
}]>;

def i64immSExt32 : PatLeaf<(i64 imm), [{
  // i64immSExt32 predicate - True if the 64-bit immediate fits in a 32-bit
  // sign extended field.
  return (int64_t)N->getZExtValue() == (int32_t)N->getZExtValue();
}]>;

def i64immZExt32 : PatLeaf<(i64 imm), [{
  // i64immZExt32 predicate - True if the 64-bit immediate fits in a 32-bit
  // zero extended field.
  return (uint64_t)N->getZExtValue() == (uint32_t)N->getZExtValue();
}]>;

// extloads
def extloadi32i8   : PatFrag<(ops node:$ptr), (i32 (extloadi8  node:$ptr))>;
def extloadi32i16  : PatFrag<(ops node:$ptr), (i32 (extloadi16 node:$ptr))>;
def extloadi64i8   : PatFrag<(ops node:$ptr), (i64 (extloadi8  node:$ptr))>;
def extloadi64i16  : PatFrag<(ops node:$ptr), (i64 (extloadi16 node:$ptr))>;
def extloadi64i32  : PatFrag<(ops node:$ptr), (i64 (extloadi32 node:$ptr))>;

def sextloadi32i8   : PatFrag<(ops node:$ptr), (i32 (sextloadi8  node:$ptr))>;
def sextloadi32i16  : PatFrag<(ops node:$ptr), (i32 (sextloadi16 node:$ptr))>;
def sextloadi64i8   : PatFrag<(ops node:$ptr), (i64 (sextloadi8  node:$ptr))>;
def sextloadi64i16  : PatFrag<(ops node:$ptr), (i64 (sextloadi16 node:$ptr))>;
def sextloadi64i32  : PatFrag<(ops node:$ptr), (i64 (sextloadi32 node:$ptr))>;

def zextloadi32i8   : PatFrag<(ops node:$ptr), (i32 (zextloadi8  node:$ptr))>;
def zextloadi32i16  : PatFrag<(ops node:$ptr), (i32 (zextloadi16 node:$ptr))>;
def zextloadi64i8   : PatFrag<(ops node:$ptr), (i64 (zextloadi8  node:$ptr))>;
def zextloadi64i16  : PatFrag<(ops node:$ptr), (i64 (zextloadi16 node:$ptr))>;
def zextloadi64i32  : PatFrag<(ops node:$ptr), (i64 (zextloadi32 node:$ptr))>;

// A couple of more descriptive operand definitions.
// 32-bits but only 8 bits are significant.
def i32i8imm  : Operand<i32>;
// 32-bits but only 16 bits are significant.
def i32i16imm : Operand<i32>;
// 64-bits but only 32 bits are significant.
def i64i32imm : Operand<i64>;
// Branch targets have OtherVT type.
def brtarget : Operand<OtherVT>;

// Unigned i12
def u12imm : Operand<i32> {
  let PrintMethod = "printU16ImmOperand";
}
// Signed i16
def s16imm : Operand<i32> {
  let PrintMethod = "printS16ImmOperand";
}
def s16imm64 : Operand<i64> {
  let PrintMethod = "printS16ImmOperand";
}
// Signed i20
def s20imm : Operand<i32> {
  let PrintMethod = "printS20ImmOperand";
}
def s20imm64 : Operand<i64> {
  let PrintMethod = "printS20ImmOperand";
}
// Signed i32
def s32imm : Operand<i32> {
  let PrintMethod = "printS32ImmOperand";
}
def s32imm64 : Operand<i64> {
  let PrintMethod = "printS32ImmOperand";
}

//===----------------------------------------------------------------------===//
// SystemZ Operand Definitions.
//===----------------------------------------------------------------------===//

// Address operands

// riaddr := reg + imm
def riaddr32 : Operand<i32>,
               ComplexPattern<i32, 2, "SelectAddrRI32", []> {
  let PrintMethod = "printRIAddrOperand";
  let MIOperandInfo = (ops ADDR32:$base, u12imm:$disp);
}

def riaddr : Operand<i64>,
             ComplexPattern<i64, 2, "SelectAddrRI", []> {
  let PrintMethod = "printRIAddrOperand";
  let MIOperandInfo = (ops ADDR64:$base, s20imm64:$disp);
}

//===----------------------------------------------------------------------===//

// rriaddr := reg + reg + imm
def rriaddr : Operand<i64>,
              ComplexPattern<i64, 3, "SelectAddrRRI", [], []> {
  let PrintMethod = "printRRIAddrOperand";
  let MIOperandInfo = (ops ADDR64:$base, s20imm64:$disp, ADDR64:$index);
}
def laaddr : Operand<i64>,
             ComplexPattern<i64, 3, "SelectLAAddr", [add, sub, or, frameindex], []> {
  let PrintMethod = "printRRIAddrOperand";
  let MIOperandInfo = (ops ADDR64:$base, s20imm64:$disp, ADDR64:$index);
}

//===----------------------------------------------------------------------===//
// Instruction list..

def ADJCALLSTACKDOWN : Pseudo<(outs), (ins i64imm:$amt),
                              "#ADJCALLSTACKDOWN",
                              [(SystemZcallseq_start timm:$amt)]>;
def ADJCALLSTACKUP   : Pseudo<(outs), (ins i64imm:$amt1, i64imm:$amt2),
                              "#ADJCALLSTACKUP",
                              [(SystemZcallseq_end timm:$amt1, timm:$amt2)]>;

let usesCustomDAGSchedInserter = 1 in {
  def Select32 : Pseudo<(outs GR32:$dst), (ins GR32:$src1, GR32:$src2, i8imm:$cc),
                        "# Select32 PSEUDO",
                        [(set GR32:$dst,
                              (SystemZselect GR32:$src1, GR32:$src2, imm:$cc))]>;
  def Select64 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, GR64:$src2, i8imm:$cc),
                        "# Select64 PSEUDO",
                        [(set GR64:$dst,
                              (SystemZselect GR64:$src1, GR64:$src2, imm:$cc))]>;
}


//===----------------------------------------------------------------------===//
//  Control Flow Instructions...
//

// FIXME: Provide proper encoding!
let isReturn = 1, isTerminator = 1, isBarrier = 1, hasCtrlDep = 1 in {
  def RET : Pseudo<(outs), (ins), "br\t%r14", [(SystemZretflag)]>;
}

let isBranch = 1, isTerminator = 1 in {
  let isBarrier = 1 in
    def JMP  : Pseudo<(outs), (ins brtarget:$dst), "j\t{$dst}", [(br bb:$dst)]>;

  let Uses = [PSW] in {
    def JE  : Pseudo<(outs), (ins brtarget:$dst),
                     "je\t$dst",
                     [(SystemZbrcond bb:$dst, SYSTEMZ_COND_E)]>;
    def JNE : Pseudo<(outs), (ins brtarget:$dst),
                     "jne\t$dst",
                     [(SystemZbrcond bb:$dst, SYSTEMZ_COND_NE)]>;
    def JH  : Pseudo<(outs), (ins brtarget:$dst),
                     "jh\t$dst",
                     [(SystemZbrcond bb:$dst, SYSTEMZ_COND_H)]>;
    def JL  : Pseudo<(outs), (ins brtarget:$dst),
                     "jl\t$dst",
                     [(SystemZbrcond bb:$dst, SYSTEMZ_COND_L)]>;
    def JHE : Pseudo<(outs), (ins brtarget:$dst),
                     "jhe\t$dst",
                     [(SystemZbrcond bb:$dst, SYSTEMZ_COND_HE)]>;
    def JLE : Pseudo<(outs), (ins brtarget:$dst),
                     "jle\t$dst",
                     [(SystemZbrcond bb:$dst, SYSTEMZ_COND_LE)]>;

  } // Uses = [PSW]
} // isBranch = 1

//===----------------------------------------------------------------------===//
//  Call Instructions...
//

let isCall = 1 in
  // All calls clobber the non-callee saved registers (except R14 which we
  // handle separately). Uses for argument registers are added manually.
  let Defs = [R0D, R1D, R2D, R3D, R4D, R5D] in {
    def CALLi     : Pseudo<(outs), (ins i64imm:$dst, variable_ops),
                           "brasl\t%r14, $dst", [(SystemZcall imm:$dst)]>;
    def CALLr     : Pseudo<(outs), (ins ADDR64:$dst, variable_ops),
                           "brasl\t%r14, $dst", [(SystemZcall ADDR64:$dst)]>;
  }

//===----------------------------------------------------------------------===//
//  Miscellaneous Instructions.
//

let isReMaterializable = 1 in
// FIXME: Provide imm12 variant
def LA64r  : Pseudo<(outs GR64:$dst), (ins laaddr:$src),
                    "lay\t{$dst, $src}",
                    [(set GR64:$dst, laaddr:$src)]>;
def LA64rm : Pseudo<(outs GR64:$dst), (ins i64imm:$src),
                    "larl\t{$dst, $src}",
                    [(set GR64:$dst,
                          (SystemZpcrelwrapper tglobaladdr:$src))]>;

let neverHasSideEffects = 1 in
def NOP : Pseudo<(outs), (ins), "# no-op", []>;

//===----------------------------------------------------------------------===//
// Move Instructions

// FIXME: Provide proper encoding!
let neverHasSideEffects = 1 in {
def MOV32rr : Pseudo<(outs GR32:$dst), (ins GR32:$src),
                     "lr\t{$dst, $src}",
                     []>;
def MOV64rr : Pseudo<(outs GR64:$dst), (ins GR64:$src),
                     "lgr\t{$dst, $src}",
                     []>;
def MOV128rr : Pseudo<(outs GR128:$dst), (ins GR128:$src),
                     "# MOV128 PSEUDO!"
                     "lgr\t{$dst:subreg_odd, $src:subreg_odd}\n"
                     "lgr\t{$dst:subreg_even, $src:subreg_even}",
                     []>;
def MOV64rrP : Pseudo<(outs GR64P:$dst), (ins GR64P:$src),
                     "# MOV64P PSEUDO!"
                     "lr\t{$dst:subreg_odd, $src:subreg_odd}\n"
                     "lr\t{$dst:subreg_even, $src:subreg_even}",
                     []>;
}

def MOVSX64rr32 : Pseudo<(outs GR64:$dst), (ins GR32:$src),
                         "lgfr\t{$dst, $src}",
                         [(set GR64:$dst, (sext GR32:$src))]>;
def MOVZX64rr32 : Pseudo<(outs GR64:$dst), (ins GR32:$src),
                         "llgfr\t{$dst, $src}",
                         [(set GR64:$dst, (zext GR32:$src))]>;

// FIXME: Provide proper encoding!
let isReMaterializable = 1, isAsCheapAsAMove = 1 in {
def MOV32ri16 : Pseudo<(outs GR32:$dst), (ins s16imm:$src),
                       "lhi\t{$dst, $src}",
                       [(set GR32:$dst, immSExt16:$src)]>;
def MOV64ri16 : Pseudo<(outs GR64:$dst), (ins s16imm64:$src),
                       "lghi\t{$dst, $src}",
                       [(set GR64:$dst, immSExt16:$src)]>;

def MOV64rill16 : Pseudo<(outs GR64:$dst), (ins i64imm:$src),
                         "llill\t{$dst, $src}",
                         [(set GR64:$dst, i64ll16:$src)]>;
def MOV64rilh16 : Pseudo<(outs GR64:$dst), (ins i64imm:$src),
                         "llilh\t{$dst, $src}",
                         [(set GR64:$dst, i64lh16:$src)]>;
def MOV64rihl16 : Pseudo<(outs GR64:$dst), (ins i64imm:$src),
                         "llihl\t{$dst, $src}",
                         [(set GR64:$dst, i64hl16:$src)]>;
def MOV64rihh16 : Pseudo<(outs GR64:$dst), (ins i64imm:$src),
                         "llihh\t{$dst, $src}",
                         [(set GR64:$dst, i64hh16:$src)]>;
// FIXME: these 3 instructions seem to require extimm facility
def MOV64ri32 : Pseudo<(outs GR64:$dst), (ins s32imm64:$src),
                       "lgfi\t{$dst, $src}",
                       [(set GR64:$dst, immSExt32:$src)]>;
def MOV64rilo32 : Pseudo<(outs GR64:$dst), (ins i64imm:$src),
                         "llilf\t{$dst, $src}",
                         [(set GR64:$dst, i64lo32:$src)]>;
def MOV64rihi32 : Pseudo<(outs GR64:$dst), (ins i64imm:$src),
                         "llihf\t{$dst, $src}",
                         [(set GR64:$dst, i64hi32:$src)]>;
}

let canFoldAsLoad = 1, isReMaterializable = 1, mayHaveSideEffects = 1 in {
def MOV32rm : Pseudo<(outs GR32:$dst), (ins rriaddr:$src),
                     "ly\t{$dst, $src}",
                     [(set GR32:$dst, (load rriaddr:$src))]>;
def MOV64rm : Pseudo<(outs GR64:$dst), (ins rriaddr:$src),
                     "lg\t{$dst, $src}",
                     [(set GR64:$dst, (load rriaddr:$src))]>;

}

def MOV32mr : Pseudo<(outs), (ins rriaddr:$dst, GR32:$src),
                     "sty\t{$src, $dst}",
                     [(store GR32:$src, rriaddr:$dst)]>;
def MOV64mr : Pseudo<(outs), (ins rriaddr:$dst, GR64:$src),
                     "stg\t{$src, $dst}",
                     [(store GR64:$src, rriaddr:$dst)]>;

// FIXME: displacements here are really 12 bit, not 20!
def MOV8mi    : Pseudo<(outs), (ins riaddr:$dst, i32i8imm:$src),
                       "mviy\t{$dst, $src}",
                       [(truncstorei8 (i32 i32immSExt8:$src), riaddr:$dst)]>;
def MOV16mi   : Pseudo<(outs), (ins riaddr:$dst, s16imm:$src),
                       "mvhhi\t{$dst, $src}",
                       [(truncstorei16 (i32 i32immSExt16:$src), riaddr:$dst)]>;
def MOV32mi16 : Pseudo<(outs), (ins riaddr:$dst, s32imm:$src),
                       "mvhi\t{$dst, $src}",
                       [(store (i32 immSExt16:$src), riaddr:$dst)]>;
def MOV64mi16 : Pseudo<(outs), (ins riaddr:$dst, s32imm64:$src),
                       "mvghi\t{$dst, $src}",
                       [(store (i64 immSExt16:$src), riaddr:$dst)]>;

// sexts
def MOVSX32rr8  : Pseudo<(outs GR32:$dst), (ins GR32:$src),
                         "lbr\t{$dst, $src}",
                         [(set GR32:$dst, (sext_inreg GR32:$src, i8))]>;
def MOVSX64rr8  : Pseudo<(outs GR64:$dst), (ins GR64:$src),
                         "lgbr\t{$dst, $src}",
                         [(set GR64:$dst, (sext_inreg GR64:$src, i8))]>;
def MOVSX32rr16 : Pseudo<(outs GR32:$dst), (ins GR32:$src),
                         "lhr\t{$dst, $src}",
                         [(set GR32:$dst, (sext_inreg GR32:$src, i16))]>;
def MOVSX64rr16 : Pseudo<(outs GR64:$dst), (ins GR64:$src),
                         "lghr\t{$dst, $src}",
                         [(set GR64:$dst, (sext_inreg GR64:$src, i16))]>;

// extloads
def MOVSX32rm8  : Pseudo<(outs GR32:$dst), (ins rriaddr:$src),
                         "lb\t{$dst, $src}",
                         [(set GR32:$dst, (sextloadi32i8 rriaddr:$src))]>;
def MOVSX32rm16 : Pseudo<(outs GR32:$dst), (ins rriaddr:$src),
                         "lhy\t{$dst, $src}",
                         [(set GR32:$dst, (sextloadi32i16 rriaddr:$src))]>;
def MOVSX64rm8  : Pseudo<(outs GR64:$dst), (ins rriaddr:$src),
                         "lgb\t{$dst, $src}",
                         [(set GR64:$dst, (sextloadi64i8 rriaddr:$src))]>;
def MOVSX64rm16 : Pseudo<(outs GR64:$dst), (ins rriaddr:$src),
                         "lgh\t{$dst, $src}",
                         [(set GR64:$dst, (sextloadi64i16 rriaddr:$src))]>;
def MOVSX64rm32 : Pseudo<(outs GR64:$dst), (ins rriaddr:$src),
                         "lgf\t{$dst, $src}",
                         [(set GR64:$dst, (sextloadi64i32 rriaddr:$src))]>;

def MOVZX32rm8  : Pseudo<(outs GR32:$dst), (ins rriaddr:$src),
                         "llc\t{$dst, $src}",
                         [(set GR32:$dst, (zextloadi32i8 rriaddr:$src))]>;
def MOVZX32rm16 : Pseudo<(outs GR32:$dst), (ins rriaddr:$src),
                         "llh\t{$dst, $src}",
                         [(set GR32:$dst, (zextloadi32i16 rriaddr:$src))]>;
def MOVZX64rm8  : Pseudo<(outs GR64:$dst), (ins rriaddr:$src),
                         "llgc\t{$dst, $src}",
                         [(set GR64:$dst, (zextloadi64i8 rriaddr:$src))]>;
def MOVZX64rm16 : Pseudo<(outs GR64:$dst), (ins rriaddr:$src),
                         "llgh\t{$dst, $src}",
                         [(set GR64:$dst, (zextloadi64i16 rriaddr:$src))]>;
def MOVZX64rm32 : Pseudo<(outs GR64:$dst), (ins rriaddr:$src),
                         "llgf\t{$dst, $src}",
                         [(set GR64:$dst, (zextloadi64i32 rriaddr:$src))]>;

// truncstores
// FIXME: Implement 12-bit displacement stuff someday
def MOV32m8r  : Pseudo<(outs), (ins rriaddr:$dst, GR32:$src),
                       "stcy\t{$src, $dst}",
                       [(truncstorei8 GR32:$src, rriaddr:$dst)]>;

def MOV32m16r : Pseudo<(outs), (ins rriaddr:$dst, GR32:$src),
                       "sthy\t{$src, $dst}",
                       [(truncstorei16 GR32:$src, rriaddr:$dst)]>;

def MOV64m8r  : Pseudo<(outs), (ins rriaddr:$dst, GR64:$src),
                       "stcy\t{$src, $dst}",
                       [(truncstorei8 GR64:$src, rriaddr:$dst)]>;

def MOV64m16r : Pseudo<(outs), (ins rriaddr:$dst, GR64:$src),
                       "sthy\t{$src, $dst}",
                       [(truncstorei16 GR64:$src, rriaddr:$dst)]>;

def MOV64m32r : Pseudo<(outs), (ins rriaddr:$dst, GR64:$src),
                       "sty\t{$src, $dst}",
                       [(truncstorei32 GR64:$src, rriaddr:$dst)]>;

// multiple regs moves
// FIXME: should we use multiple arg nodes?
def MOV32mrm  : Pseudo<(outs), (ins riaddr:$dst, GR32:$from, GR32:$to),
                       "stmy\t{$from, $to, $dst}",
                       []>;
def MOV64mrm  : Pseudo<(outs), (ins riaddr:$dst, GR64:$from, GR64:$to),
                       "stmg\t{$from, $to, $dst}",
                       []>;
def MOV32rmm  : Pseudo<(outs GR32:$from, GR32:$to), (ins riaddr:$dst),
                       "lmy\t{$from, $to, $dst}",
                       []>;
def MOV64rmm  : Pseudo<(outs GR64:$from, GR64:$to), (ins riaddr:$dst),
                       "lmg\t{$from, $to, $dst}",
                       []>;


//===----------------------------------------------------------------------===//
// Arithmetic Instructions

let isTwoAddress = 1 in {

let Defs = [PSW] in {

let isCommutable = 1 in { // X = ADD Y, Z  == X = ADD Z, Y
// FIXME: Provide proper encoding!
def ADD32rr : Pseudo<(outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                     "ar\t{$dst, $src2}",
                     [(set GR32:$dst, (add GR32:$src1, GR32:$src2)),
                      (implicit PSW)]>;
def ADD64rr : Pseudo<(outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                     "agr\t{$dst, $src2}",
                     [(set GR64:$dst, (add GR64:$src1, GR64:$src2)),
                      (implicit PSW)]>;
}

// FIXME: Provide proper encoding!
def ADD32ri16 : Pseudo<(outs GR32:$dst), (ins GR32:$src1, s16imm:$src2),
                       "ahi\t{$dst, $src2:}",
                       [(set GR32:$dst, (add GR32:$src1, immSExt16:$src2)),
                        (implicit PSW)]>;
def ADD32ri   : Pseudo<(outs GR32:$dst), (ins GR32:$src1, s32imm:$src2),
                       "afi\t{$dst, $src2}",
                       [(set GR32:$dst, (add GR32:$src1, imm:$src2)),
                        (implicit PSW)]>;
def ADD64ri16 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, s16imm64:$src2),
                       "aghi\t{$dst, $src2}",
                       [(set GR64:$dst, (add GR64:$src1, immSExt16:$src2)),
                        (implicit PSW)]>;
def ADD64ri32 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, s32imm64:$src2),
                       "agfi\t{$dst, $src2}",
                       [(set GR64:$dst, (add GR64:$src1, immSExt32:$src2)),
                        (implicit PSW)]>;

let isCommutable = 1 in { // X = AND Y, Z  == X = AND Z, Y
// FIXME: Provide proper encoding!
def AND32rr : Pseudo<(outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                     "nr\t{$dst, $src2}",
                     [(set GR32:$dst, (and GR32:$src1, GR32:$src2))]>;
def AND64rr : Pseudo<(outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                     "ngr\t{$dst, $src2}",
                     [(set GR64:$dst, (and GR64:$src1, GR64:$src2))]>;
}

// FIXME: Provide proper encoding!
def AND64rill16 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                         "nill\t{$dst, $src2}",
                         [(set GR64:$dst, (and GR64:$src1, i64ll16:$src2))]>;
def AND64rilh16 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                         "nilh\t{$dst, $src2}",
                         [(set GR64:$dst, (and GR64:$src1, i64lh16:$src2))]>;
def AND64rihl16 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                         "nihl\t{$dst, $src2}",
                         [(set GR64:$dst, (and GR64:$src1, i64hl16:$src2))]>;
def AND64rihh16 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                         "nihh\t{$dst, $src2}",
                         [(set GR64:$dst, (and GR64:$src1, i64hh16:$src2))]>;
// FIXME: these 2 instructions seem to require extimm facility
def AND64rilo32 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                         "nilf\t{$dst, $src2}",
                         [(set GR64:$dst, (and GR64:$src1, i64lo32:$src2))]>;
def AND64rihi32 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                         "nihf\t{$dst, $src2}",
                         [(set GR64:$dst, (and GR64:$src1, i64hi32:$src2))]>;

let isCommutable = 1 in { // X = OR Y, Z  == X = OR Z, Y
// FIXME: Provide proper encoding!
def OR32rr : Pseudo<(outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                    "or\t{$dst, $src2}",
                    [(set GR32:$dst, (or GR32:$src1, GR32:$src2))]>;
def OR64rr : Pseudo<(outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                    "ogr\t{$dst, $src2}",
                    [(set GR64:$dst, (or GR64:$src1, GR64:$src2))]>;
}

def OR32ri16  : Pseudo<(outs GR32:$dst), (ins GR32:$src1, i16imm:$src2),
                      "oill\t{$dst, $src2}",
                      [(set GR32:$dst, (or GR32:$src1, i64ll16:$src2))]>;
def OR32ri16h : Pseudo<(outs GR32:$dst), (ins GR32:$src1, i16imm:$src2),
                      "oilh\t{$dst, $src2}",
                      [(set GR32:$dst, (or GR32:$src1, i64lh16:$src2))]>;
def OR32ri : Pseudo<(outs GR32:$dst), (ins GR32:$src1, i32imm:$src2),
                    "oilf\t{$dst, $src2}",
                    [(set GR32:$dst, (or GR32:$src1, imm:$src2))]>;

def OR64rill16 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                        "oill\t{$dst, $src2}",
                        [(set GR64:$dst, (or GR64:$src1, i64ll16:$src2))]>;
def OR64rilh16 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                        "oilh\t{$dst, $src2}",
                        [(set GR64:$dst, (or GR64:$src1, i64lh16:$src2))]>;
def OR64rihl16 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                        "oihl\t{$dst, $src2}",
                        [(set GR64:$dst, (or GR64:$src1, i64hl16:$src2))]>;
def OR64rihh16 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                        "oihh\t{$dst, $src2}",
                        [(set GR64:$dst, (or GR64:$src1, i64hh16:$src2))]>;
// FIXME: these 2 instructions seem to require extimm facility
def OR64rilo32 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                        "oilf\t{$dst, $src2}",
                        [(set GR64:$dst, (or GR64:$src1, i64lo32:$src2))]>;
def OR64rihi32 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                        "oihf\t{$dst, $src2}",
                        [(set GR64:$dst, (or GR64:$src1, i64hi32:$src2))]>;

// FIXME: Provide proper encoding!
def SUB32rr : Pseudo<(outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                     "sr\t{$dst, $src2}",
                     [(set GR32:$dst, (sub GR32:$src1, GR32:$src2))]>;
def SUB64rr : Pseudo<(outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                     "sgr\t{$dst, $src2}",
                     [(set GR64:$dst, (sub GR64:$src1, GR64:$src2))]>;


let isCommutable = 1 in { // X = XOR Y, Z  == X = XOR Z, Y
// FIXME: Provide proper encoding!
def XOR32rr : Pseudo<(outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                     "xr\t{$dst, $src2}",
                     [(set GR32:$dst, (xor GR32:$src1, GR32:$src2))]>;
def XOR64rr : Pseudo<(outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                     "xgr\t{$dst, $src2}",
                     [(set GR64:$dst, (xor GR64:$src1, GR64:$src2))]>;
}

def XOR32ri : Pseudo<(outs GR32:$dst), (ins GR32:$src1, i32imm:$src2),
                     "xilf\t{$dst, $src2}",
                     [(set GR32:$dst, (xor GR32:$src1, imm:$src2))]>;

// FIXME: these 2 instructions seem to require extimm facility
def XOR64rilo32 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                         "xilf\t{$dst, $src2}",
                         [(set GR64:$dst, (xor GR64:$src1, i64lo32:$src2))]>;
def XOR64rihi32 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                         "xihf\t{$dst, $src2}",
                         [(set GR64:$dst, (xor GR64:$src1, i64hi32:$src2))]>;

} // Defs = [PSW]

let isCommutable = 1 in { // X = MUL Y, Z == X = MUL Z, Y
def MUL32rr : Pseudo<(outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                     "msr\t{$dst, $src2}",
                     [(set GR32:$dst, (mul GR32:$src1, GR32:$src2))]>;
def MUL64rr : Pseudo<(outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                     "msgr\t{$dst, $src2}",
                     [(set GR64:$dst, (mul GR64:$src1, GR64:$src2))]>;

def MUL64rrP   : Pseudo<(outs GR64P:$dst), (ins GR64P:$src1, GR32:$src2),
                        "mr\t{$dst, $src2}",
                        []>;
def UMUL64rrP  : Pseudo<(outs GR64P:$dst), (ins GR64P:$src1, GR32:$src2),
                         "mlr\t{$dst, $src2}",
                         []>;
def UMUL128rrP : Pseudo<(outs GR128:$dst), (ins GR128:$src1, GR64:$src2),
                        "mlgr\t{$dst, $src2}",
                        []>;
}


def MUL32ri16   : Pseudo<(outs GR32:$dst), (ins GR32:$src1, s16imm:$src2),
                         "mhi\t{$dst, $src2}",
                         [(set GR32:$dst, (mul GR32:$src1, i32immSExt16:$src2))]>;
def MUL32ri     : Pseudo<(outs GR32:$dst), (ins GR32:$src1, s32imm:$src2),
                         "msfi\t{$dst, $src2}",
                         [(set GR32:$dst, (mul GR32:$src1, imm:$src2))]>;
def MUL64ri16   : Pseudo<(outs GR64:$dst), (ins GR64:$src1, s16imm64:$src2),
                         "mghi\t{$dst, $src2}",
                         [(set GR64:$dst, (mul GR64:$src1, immSExt16:$src2))]>;
def MUL64ri32   : Pseudo<(outs GR64:$dst), (ins GR64:$src1, s32imm64:$src2),
                         "msgfi\t{$dst, $src2}",
                         [(set GR64:$dst, (mul GR64:$src1, i64immSExt32:$src2))]>;

def MUL32rm : Pseudo<(outs GR32:$dst), (ins GR32:$src1, rriaddr:$src2),
                     "msy\t{$dst, $src2}",
                     [(set GR32:$dst, (mul GR32:$src1, (load rriaddr:$src2)))]>;
def MUL64rm : Pseudo<(outs GR64:$dst), (ins GR64:$src1, rriaddr:$src2),
                     "msgy\t{$dst, $src2}",
                     [(set GR64:$dst, (mul GR64:$src1, (load rriaddr:$src2)))]>;

def MULSX64rr32 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, GR32:$src2),
                         "msgfr\t{$dst, $src2}",
                         [(set GR64:$dst, (mul GR64:$src1, (sext GR32:$src2)))]>;

def SDIVREM64rrP  : Pseudo<(outs GR64P:$dst), (ins GR64P:$src1, GR32:$src2),
                           "dr\t{$dst, $src2}",
                           []>;

def SDIVREM128rrP : Pseudo<(outs GR128:$dst), (ins GR128:$src1, GR64:$src2),
                           "dsgr\t{$dst, $src2}",
                           []>;

def UDIVREM64rrP  : Pseudo<(outs GR64P:$dst), (ins GR64P:$src1, GR32:$src2),
                           "dlr\t{$dst, $src2}",
                           []>;

def UDIVREM128rrP : Pseudo<(outs GR128:$dst), (ins GR128:$src1, GR64:$src2),
                           "dlgr\t{$dst, $src2}",
                           []>;

} // isTwoAddress = 1

//===----------------------------------------------------------------------===//
// Shifts

let isTwoAddress = 1 in
def SRL32rri : Pseudo<(outs GR32:$dst), (ins GR32:$src, riaddr32:$amt),
                      "srl\t{$src, $amt}",
                      [(set GR32:$dst, (srl GR32:$src, riaddr32:$amt))]>;
def SRL64rri : Pseudo<(outs GR64:$dst), (ins GR64:$src, riaddr:$amt),
                      "srlg\t{$dst, $src, $amt}",
                      [(set GR64:$dst, (srl GR64:$src, (i32 (trunc riaddr:$amt))))]>;
def SRLA64ri  : Pseudo<(outs GR64:$dst), (ins GR64:$src, i32imm:$amt),
                      "srlg\t{$dst, $src, $amt}",
                      [(set GR64:$dst, (srl GR64:$src, (i32 imm:$amt)))]>;

let isTwoAddress = 1 in
def SHL32rri : Pseudo<(outs GR32:$dst), (ins GR32:$src, riaddr32:$amt),
                      "sll\t{$src, $amt}",
                      [(set GR32:$dst, (shl GR32:$src, riaddr32:$amt))]>;
def SHL64rri : Pseudo<(outs GR64:$dst), (ins GR64:$src, riaddr:$amt),
                      "sllg\t{$dst, $src, $amt}",
                      [(set GR64:$dst, (shl GR64:$src, (i32 (trunc riaddr:$amt))))]>;
def SHL64ri  : Pseudo<(outs GR64:$dst), (ins GR64:$src, i32imm:$amt),
                      "sllg\t{$dst, $src, $amt}",
                      [(set GR64:$dst, (shl GR64:$src, (i32 imm:$amt)))]>;


let Defs = [PSW] in {
let isTwoAddress = 1 in
def SRA32rri : Pseudo<(outs GR32:$dst), (ins GR32:$src, riaddr32:$amt),
                      "sra\t{$src, $amt}",
                      [(set GR32:$dst, (sra GR32:$src, riaddr32:$amt)),
                       (implicit PSW)]>;
def SRA64rri : Pseudo<(outs GR64:$dst), (ins GR64:$src, riaddr:$amt),
                      "srag\t{$dst, $src, $amt}",
                      [(set GR64:$dst, (sra GR64:$src, (i32 (trunc riaddr:$amt)))),
                       (implicit PSW)]>;
def SRA64ri  : Pseudo<(outs GR64:$dst), (ins GR64:$src, i32imm:$amt),
                      "srag\t{$dst, $src, $amt}",
                      [(set GR64:$dst, (sra GR64:$src, (i32 imm:$amt))),
                       (implicit PSW)]>;
} // Defs = [PSW]

//===----------------------------------------------------------------------===//
// Test instructions (like AND but do not produce any result

// Integer comparisons
let Defs = [PSW] in {
def CMP32rr : Pseudo<(outs), (ins GR32:$src1, GR32:$src2),
                     "cr\t$src1, $src2",
                     [(SystemZcmp GR32:$src1, GR32:$src2), (implicit PSW)]>;
def CMP64rr : Pseudo<(outs), (ins GR64:$src1, GR64:$src2),
                     "cgr\t$src1, $src2",
                     [(SystemZcmp GR64:$src1, GR64:$src2), (implicit PSW)]>;

def CMP32ri   : Pseudo<(outs), (ins GR32:$src1, s32imm:$src2),
                       "cfi\t$src1, $src2",
                       [(SystemZcmp GR32:$src1, imm:$src2), (implicit PSW)]>;
def CMP64ri32 : Pseudo<(outs), (ins GR64:$src1, s32imm64:$src2),
                       "cgfi\t$src1, $src2",
                       [(SystemZcmp GR64:$src1, i64immSExt32:$src2),
                        (implicit PSW)]>;

def CMP32rm : Pseudo<(outs), (ins GR32:$src1, rriaddr:$src2),
                     "cy\t$src1, $src2",
                     [(SystemZcmp GR32:$src1, (load rriaddr:$src2)),
                      (implicit PSW)]>;
def CMP64rm : Pseudo<(outs), (ins GR64:$src1, rriaddr:$src2),
                     "cg\t$src1, $src2",
                     [(SystemZcmp GR64:$src1, (load rriaddr:$src2)),
                      (implicit PSW)]>;

def UCMP32rr : Pseudo<(outs), (ins GR32:$src1, GR32:$src2),
                      "clr\t$src1, $src2",
                      [(SystemZucmp GR32:$src1, GR32:$src2), (implicit PSW)]>;
def UCMP64rr : Pseudo<(outs), (ins GR64:$src1, GR64:$src2),
                      "clgr\t$src1, $src2",
                      [(SystemZucmp GR64:$src1, GR64:$src2), (implicit PSW)]>;

def UCMP32ri   : Pseudo<(outs), (ins GR32:$src1, i32imm:$src2),
                        "clfi\t$src1, $src2",
                        [(SystemZucmp GR32:$src1, imm:$src2), (implicit PSW)]>;
def UCMP64ri32 : Pseudo<(outs), (ins GR64:$src1, i64i32imm:$src2),
                        "clgfi\t$src1, $src2",
                        [(SystemZucmp GR64:$src1, i64immZExt32:$src2),
                         (implicit PSW)]>;

def UCMP32rm : Pseudo<(outs), (ins GR32:$src1, rriaddr:$src2),
                      "cly\t$src1, $src2",
                      [(SystemZucmp GR32:$src1, (load rriaddr:$src2)),
                       (implicit PSW)]>;
def UCMP64rm : Pseudo<(outs), (ins GR64:$src1, rriaddr:$src2),
                      "clg\t$src1, $src2",
                      [(SystemZucmp GR64:$src1, (load rriaddr:$src2)),
                       (implicit PSW)]>;

def CMPSX64rr32  : Pseudo<(outs), (ins GR64:$src1, GR32:$src2),
                          "cgfr\t$src1, $src2",
                          [(SystemZucmp GR64:$src1, (sext GR32:$src2)),
                           (implicit PSW)]>;
def UCMPZX64rr32 : Pseudo<(outs), (ins GR64:$src1, GR32:$src2),
                          "clgfr\t$src1, $src2",
                          [(SystemZucmp GR64:$src1, (zext GR32:$src2)),
                           (implicit PSW)]>;

def CMPSX64rm32   : Pseudo<(outs), (ins GR64:$src1, rriaddr:$src2),
                           "cgf\t$src1, $src2",
                           [(SystemZucmp GR64:$src1, (sextloadi64i32 rriaddr:$src2)),
                            (implicit PSW)]>;
def UCMPZX64rm32  : Pseudo<(outs), (ins GR64:$src1, rriaddr:$src2),
                           "clgf\t$src1, $src2",
                           [(SystemZucmp GR64:$src1, (zextloadi64i32 rriaddr:$src2)),
                            (implicit PSW)]>;

// FIXME: Add other crazy ucmp forms

} // Defs = [PSW]

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns.
//===----------------------------------------------------------------------===//

// anyext
def : Pat<(i64 (anyext GR32:$src)),
          (INSERT_SUBREG (i64 (IMPLICIT_DEF)), GR32:$src, subreg_32bit)>;

//===----------------------------------------------------------------------===//
// Peepholes.
//===----------------------------------------------------------------------===//

// FIXME: use add/sub tricks with 32678/-32768

// trunc patterns
def : Pat<(i32 (trunc GR64:$src)),
          (EXTRACT_SUBREG GR64:$src, subreg_32bit)>;

// sext_inreg patterns
def : Pat<(sext_inreg GR64:$src, i32),
          (MOVSX64rr32 (EXTRACT_SUBREG GR64:$src, subreg_32bit))>;

// extload patterns
def : Pat<(extloadi32i8  rriaddr:$src), (MOVZX32rm8  rriaddr:$src)>;
def : Pat<(extloadi32i16 rriaddr:$src), (MOVZX32rm16 rriaddr:$src)>;
def : Pat<(extloadi64i8  rriaddr:$src), (MOVZX64rm8  rriaddr:$src)>;
def : Pat<(extloadi64i16 rriaddr:$src), (MOVZX64rm16 rriaddr:$src)>;
def : Pat<(extloadi64i32 rriaddr:$src), (MOVZX64rm32 rriaddr:$src)>;

// calls
def : Pat<(SystemZcall (i64 tglobaladdr:$dst)),
          (CALLi tglobaladdr:$dst)>;
def : Pat<(SystemZcall (i64 texternalsym:$dst)),
          (CALLi texternalsym:$dst)>;

// muls
def : Pat<(mulhs GR32:$src1, GR32:$src2),
          (EXTRACT_SUBREG (MUL64rrP (INSERT_SUBREG (i64 (IMPLICIT_DEF)),
                                                   GR32:$src1, subreg_odd),
                                    GR32:$src2),
                          subreg_even)>;

def : Pat<(mulhu GR32:$src1, GR32:$src2),
          (EXTRACT_SUBREG (UMUL64rrP (INSERT_SUBREG (i64 (IMPLICIT_DEF)),
                                                    GR32:$src1, subreg_odd),
                                     GR32:$src2),
                          subreg_even)>;
def : Pat<(mulhu GR64:$src1, GR64:$src2),
          (EXTRACT_SUBREG (UMUL128rrP (INSERT_SUBREG (i128 (IMPLICIT_DEF)),
                                                     GR64:$src1, subreg_odd),
                                      GR64:$src2),
                          subreg_even)>;

// divs
// FIXME: Add memory versions
def : Pat<(sdiv GR32:$src1, GR32:$src2),
          (EXTRACT_SUBREG (SDIVREM64rrP (INSERT_SUBREG (i64 (IMPLICIT_DEF)),
                                                       GR32:$src1, subreg_odd),
                                         GR32:$src2),
                          subreg_odd)>;
def : Pat<(sdiv GR64:$src1, GR64:$src2),
          (EXTRACT_SUBREG (SDIVREM128rrP (INSERT_SUBREG (i128 (IMPLICIT_DEF)),
                                                        GR64:$src1, subreg_odd),
                                         GR64:$src2),
                          subreg_odd)>;
def : Pat<(udiv GR32:$src1, GR32:$src2),
          (EXTRACT_SUBREG (UDIVREM64rrP (INSERT_SUBREG (i64 (IMPLICIT_DEF)),
                                                       GR32:$src1, subreg_odd),
                                         GR32:$src2),
                          subreg_odd)>;
def : Pat<(udiv GR64:$src1, GR64:$src2),
          (EXTRACT_SUBREG (UDIVREM128rrP (INSERT_SUBREG (i128 (IMPLICIT_DEF)),
                                                        GR64:$src1, subreg_odd),
                                         GR64:$src2),
                          subreg_odd)>;

// rems
// FIXME: Add memory versions
def : Pat<(srem GR32:$src1, GR32:$src2),
          (EXTRACT_SUBREG (SDIVREM64rrP (INSERT_SUBREG (i64 (IMPLICIT_DEF)),
                                                       GR32:$src1, subreg_odd),
                                         GR32:$src2),
                          subreg_even)>;
def : Pat<(srem GR64:$src1, GR64:$src2),
          (EXTRACT_SUBREG (SDIVREM128rrP (INSERT_SUBREG (i128 (IMPLICIT_DEF)),
                                                        GR64:$src1, subreg_odd),
                                         GR64:$src2),
                          subreg_even)>;
def : Pat<(urem GR32:$src1, GR32:$src2),
          (EXTRACT_SUBREG (UDIVREM64rrP (INSERT_SUBREG (i64 (IMPLICIT_DEF)),
                                                       GR32:$src1, subreg_odd),
                                         GR32:$src2),
                          subreg_even)>;
def : Pat<(urem GR64:$src1, GR64:$src2),
          (EXTRACT_SUBREG (UDIVREM128rrP (INSERT_SUBREG (i128 (IMPLICIT_DEF)),
                                                        GR64:$src1, subreg_odd),
                                         GR64:$src2),
                          subreg_even)>;

def : Pat<(i32 imm:$src),
          (EXTRACT_SUBREG (MOV64ri32 (i64 imm:$src)), subreg_32bit)>;
