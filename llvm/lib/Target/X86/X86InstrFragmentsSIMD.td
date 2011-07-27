//======- X86InstrFragmentsSIMD.td - x86 ISA -------------*- tablegen -*-=====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file provides pattern fragments useful for SIMD instructions.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MMX Pattern Fragments
//===----------------------------------------------------------------------===//

def load_mmx : PatFrag<(ops node:$ptr), (x86mmx (load node:$ptr))>;
def bc_mmx  : PatFrag<(ops node:$in), (x86mmx  (bitconvert node:$in))>;

//===----------------------------------------------------------------------===//
// SSE specific DAG Nodes.
//===----------------------------------------------------------------------===//

def SDTX86FPShiftOp : SDTypeProfile<1, 2, [ SDTCisSameAs<0, 1>,
                                            SDTCisFP<0>, SDTCisInt<2> ]>;
def SDTX86VFCMP : SDTypeProfile<1, 3, [SDTCisInt<0>, SDTCisSameAs<1, 2>,
                                       SDTCisFP<1>, SDTCisVT<3, i8>]>;

def X86fmin    : SDNode<"X86ISD::FMIN",      SDTFPBinOp>;
def X86fmax    : SDNode<"X86ISD::FMAX",      SDTFPBinOp>;
def X86fand    : SDNode<"X86ISD::FAND",      SDTFPBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def X86for     : SDNode<"X86ISD::FOR",       SDTFPBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def X86fxor    : SDNode<"X86ISD::FXOR",      SDTFPBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def X86frsqrt  : SDNode<"X86ISD::FRSQRT",    SDTFPUnaryOp>;
def X86frcp    : SDNode<"X86ISD::FRCP",      SDTFPUnaryOp>;
def X86fsrl    : SDNode<"X86ISD::FSRL",      SDTX86FPShiftOp>;
def X86fgetsign: SDNode<"X86ISD::FGETSIGNx86",SDTFPToIntOp>;
def X86comi    : SDNode<"X86ISD::COMI",      SDTX86CmpTest>;
def X86ucomi   : SDNode<"X86ISD::UCOMI",     SDTX86CmpTest>;
def X86cmpss   : SDNode<"X86ISD::FSETCCss",    SDTX86Cmpss>;
def X86cmpsd   : SDNode<"X86ISD::FSETCCsd",    SDTX86Cmpsd>;
def X86pshufb  : SDNode<"X86ISD::PSHUFB",
                 SDTypeProfile<1, 2, [SDTCisVT<0, v16i8>, SDTCisSameAs<0,1>,
                                      SDTCisSameAs<0,2>]>>;
def X86andnp   : SDNode<"X86ISD::ANDNP",
                 SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                      SDTCisSameAs<0,2>]>>;
def X86psignb  : SDNode<"X86ISD::PSIGNB",
                 SDTypeProfile<1, 2, [SDTCisVT<0, v16i8>, SDTCisSameAs<0,1>,
                                      SDTCisSameAs<0,2>]>>;
def X86psignw  : SDNode<"X86ISD::PSIGNW",
                 SDTypeProfile<1, 2, [SDTCisVT<0, v8i16>, SDTCisSameAs<0,1>,
                                      SDTCisSameAs<0,2>]>>;
def X86psignd  : SDNode<"X86ISD::PSIGND",
                 SDTypeProfile<1, 2, [SDTCisVT<0, v4i32>, SDTCisSameAs<0,1>,
                                      SDTCisSameAs<0,2>]>>;
def X86pblendv : SDNode<"X86ISD::PBLENDVB",
                 SDTypeProfile<1, 3, [SDTCisVT<0, v16i8>, SDTCisSameAs<0,1>,
                                      SDTCisSameAs<0,2>, SDTCisSameAs<0,3>]>>;
def X86pextrb  : SDNode<"X86ISD::PEXTRB",
                 SDTypeProfile<1, 2, [SDTCisVT<0, i32>, SDTCisPtrTy<2>]>>;
def X86pextrw  : SDNode<"X86ISD::PEXTRW",
                 SDTypeProfile<1, 2, [SDTCisVT<0, i32>, SDTCisPtrTy<2>]>>;
def X86pinsrb  : SDNode<"X86ISD::PINSRB",
                 SDTypeProfile<1, 3, [SDTCisVT<0, v16i8>, SDTCisSameAs<0,1>,
                                      SDTCisVT<2, i32>, SDTCisPtrTy<3>]>>;
def X86pinsrw  : SDNode<"X86ISD::PINSRW",
                 SDTypeProfile<1, 3, [SDTCisVT<0, v8i16>, SDTCisSameAs<0,1>,
                                      SDTCisVT<2, i32>, SDTCisPtrTy<3>]>>;
def X86insrtps : SDNode<"X86ISD::INSERTPS",
                 SDTypeProfile<1, 3, [SDTCisVT<0, v4f32>, SDTCisSameAs<0,1>,
                                      SDTCisVT<2, v4f32>, SDTCisPtrTy<3>]>>;
def X86vzmovl  : SDNode<"X86ISD::VZEXT_MOVL",
                 SDTypeProfile<1, 1, [SDTCisSameAs<0,1>]>>;
def X86vzload  : SDNode<"X86ISD::VZEXT_LOAD", SDTLoad,
                        [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def X86vshl    : SDNode<"X86ISD::VSHL",      SDTIntShiftOp>;
def X86vshr    : SDNode<"X86ISD::VSRL",      SDTIntShiftOp>;
def X86cmpps   : SDNode<"X86ISD::CMPPS",     SDTX86VFCMP>;
def X86cmppd   : SDNode<"X86ISD::CMPPD",     SDTX86VFCMP>;
def X86pcmpeqb : SDNode<"X86ISD::PCMPEQB", SDTIntBinOp, [SDNPCommutative]>;
def X86pcmpeqw : SDNode<"X86ISD::PCMPEQW", SDTIntBinOp, [SDNPCommutative]>;
def X86pcmpeqd : SDNode<"X86ISD::PCMPEQD", SDTIntBinOp, [SDNPCommutative]>;
def X86pcmpeqq : SDNode<"X86ISD::PCMPEQQ", SDTIntBinOp, [SDNPCommutative]>;
def X86pcmpgtb : SDNode<"X86ISD::PCMPGTB", SDTIntBinOp>;
def X86pcmpgtw : SDNode<"X86ISD::PCMPGTW", SDTIntBinOp>;
def X86pcmpgtd : SDNode<"X86ISD::PCMPGTD", SDTIntBinOp>;
def X86pcmpgtq : SDNode<"X86ISD::PCMPGTQ", SDTIntBinOp>;

def SDTX86CmpPTest : SDTypeProfile<1, 2, [SDTCisVT<0, i32>,
                                          SDTCisVec<1>,
                                          SDTCisSameAs<2, 1>]>;
def X86ptest   : SDNode<"X86ISD::PTEST", SDTX86CmpPTest>;
def X86testp   : SDNode<"X86ISD::TESTP", SDTX86CmpPTest>;

// Specific shuffle nodes - At some point ISD::VECTOR_SHUFFLE will always get
// translated into one of the target nodes below during lowering.
// Note: this is a work in progress...
def SDTShuff1Op : SDTypeProfile<1, 1, [SDTCisVec<0>, SDTCisSameAs<0,1>]>;
def SDTShuff2Op : SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                SDTCisSameAs<0,2>]>;

def SDTShuff2OpI : SDTypeProfile<1, 2, [SDTCisVec<0>,
                                 SDTCisSameAs<0,1>, SDTCisInt<2>]>;
def SDTShuff3OpI : SDTypeProfile<1, 3, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                 SDTCisSameAs<0,2>, SDTCisInt<3>]>;

def X86PAlign : SDNode<"X86ISD::PALIGN", SDTShuff3OpI>;

def X86PShufd  : SDNode<"X86ISD::PSHUFD", SDTShuff2OpI>;
def X86PShufhw : SDNode<"X86ISD::PSHUFHW", SDTShuff2OpI>;
def X86PShuflw : SDNode<"X86ISD::PSHUFLW", SDTShuff2OpI>;

def X86Shufpd : SDNode<"X86ISD::SHUFPD", SDTShuff3OpI>;
def X86Shufps : SDNode<"X86ISD::SHUFPS", SDTShuff3OpI>;

def X86Movddup  : SDNode<"X86ISD::MOVDDUP", SDTShuff1Op>;
def X86Movshdup : SDNode<"X86ISD::MOVSHDUP", SDTShuff1Op>;
def X86Movsldup : SDNode<"X86ISD::MOVSLDUP", SDTShuff1Op>;

def X86Movsd : SDNode<"X86ISD::MOVSD", SDTShuff2Op>;
def X86Movss : SDNode<"X86ISD::MOVSS", SDTShuff2Op>;

def X86Movlhps : SDNode<"X86ISD::MOVLHPS", SDTShuff2Op>;
def X86Movlhpd : SDNode<"X86ISD::MOVLHPD", SDTShuff2Op>;
def X86Movhlps : SDNode<"X86ISD::MOVHLPS", SDTShuff2Op>;
def X86Movhlpd : SDNode<"X86ISD::MOVHLPD", SDTShuff2Op>;

def X86Movlps : SDNode<"X86ISD::MOVLPS", SDTShuff2Op>;
def X86Movlpd : SDNode<"X86ISD::MOVLPD", SDTShuff2Op>;

def X86Unpcklps  : SDNode<"X86ISD::UNPCKLPS", SDTShuff2Op>;
def X86Unpcklpd  : SDNode<"X86ISD::UNPCKLPD", SDTShuff2Op>;
def X86Unpcklpsy : SDNode<"X86ISD::VUNPCKLPSY", SDTShuff2Op>;
def X86Unpcklpdy : SDNode<"X86ISD::VUNPCKLPDY", SDTShuff2Op>;

def X86Unpckhps  : SDNode<"X86ISD::UNPCKHPS", SDTShuff2Op>;
def X86Unpckhpd  : SDNode<"X86ISD::UNPCKHPD", SDTShuff2Op>;
def X86Unpckhpsy : SDNode<"X86ISD::VUNPCKHPSY", SDTShuff2Op>;
def X86Unpckhpdy : SDNode<"X86ISD::VUNPCKHPDY", SDTShuff2Op>;

def X86Punpcklbw  : SDNode<"X86ISD::PUNPCKLBW", SDTShuff2Op>;
def X86Punpcklwd  : SDNode<"X86ISD::PUNPCKLWD", SDTShuff2Op>;
def X86Punpckldq  : SDNode<"X86ISD::PUNPCKLDQ", SDTShuff2Op>;
def X86Punpcklqdq : SDNode<"X86ISD::PUNPCKLQDQ", SDTShuff2Op>;

def X86Punpckhbw  : SDNode<"X86ISD::PUNPCKHBW", SDTShuff2Op>;
def X86Punpckhwd  : SDNode<"X86ISD::PUNPCKHWD", SDTShuff2Op>;
def X86Punpckhdq  : SDNode<"X86ISD::PUNPCKHDQ", SDTShuff2Op>;
def X86Punpckhqdq : SDNode<"X86ISD::PUNPCKHQDQ", SDTShuff2Op>;

def X86VPermil : SDNode<"X86ISD::VPERMIL", SDTShuff2OpI>;

//===----------------------------------------------------------------------===//
// SSE Complex Patterns
//===----------------------------------------------------------------------===//

// These are 'extloads' from a scalar to the low element of a vector, zeroing
// the top elements.  These are used for the SSE 'ss' and 'sd' instruction
// forms.
def sse_load_f32 : ComplexPattern<v4f32, 5, "SelectScalarSSELoad", [],
                                  [SDNPHasChain, SDNPMayLoad, SDNPMemOperand,
                                   SDNPWantRoot]>;
def sse_load_f64 : ComplexPattern<v2f64, 5, "SelectScalarSSELoad", [],
                                  [SDNPHasChain, SDNPMayLoad, SDNPMemOperand,
                                   SDNPWantRoot]>;

def ssmem : Operand<v4f32> {
  let PrintMethod = "printf32mem";
  let MIOperandInfo = (ops ptr_rc, i8imm, ptr_rc_nosp, i32imm, i8imm);
  let ParserMatchClass = X86MemAsmOperand;
  let OperandType = "OPERAND_MEMORY";
}
def sdmem : Operand<v2f64> {
  let PrintMethod = "printf64mem";
  let MIOperandInfo = (ops ptr_rc, i8imm, ptr_rc_nosp, i32imm, i8imm);
  let ParserMatchClass = X86MemAsmOperand;
  let OperandType = "OPERAND_MEMORY";
}

//===----------------------------------------------------------------------===//
// SSE pattern fragments
//===----------------------------------------------------------------------===//

// 128-bit load pattern fragments
def loadv4f32    : PatFrag<(ops node:$ptr), (v4f32 (load node:$ptr))>;
def loadv2f64    : PatFrag<(ops node:$ptr), (v2f64 (load node:$ptr))>;
def loadv4i32    : PatFrag<(ops node:$ptr), (v4i32 (load node:$ptr))>;
def loadv2i64    : PatFrag<(ops node:$ptr), (v2i64 (load node:$ptr))>;

// 256-bit load pattern fragments
def loadv8f32    : PatFrag<(ops node:$ptr), (v8f32 (load node:$ptr))>;
def loadv4f64    : PatFrag<(ops node:$ptr), (v4f64 (load node:$ptr))>;
def loadv8i32    : PatFrag<(ops node:$ptr), (v8i32 (load node:$ptr))>;
def loadv4i64    : PatFrag<(ops node:$ptr), (v4i64 (load node:$ptr))>;

// Like 'store', but always requires vector alignment.
def alignedstore : PatFrag<(ops node:$val, node:$ptr),
                           (store node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->getAlignment() >= 16;
}]>;

// Like 'load', but always requires vector alignment.
def alignedload : PatFrag<(ops node:$ptr), (load node:$ptr), [{
  return cast<LoadSDNode>(N)->getAlignment() >= 16;
}]>;

def alignedloadfsf32 : PatFrag<(ops node:$ptr),
                               (f32 (alignedload node:$ptr))>;
def alignedloadfsf64 : PatFrag<(ops node:$ptr),
                               (f64 (alignedload node:$ptr))>;

// 128-bit aligned load pattern fragments
def alignedloadv4f32 : PatFrag<(ops node:$ptr),
                               (v4f32 (alignedload node:$ptr))>;
def alignedloadv2f64 : PatFrag<(ops node:$ptr),
                               (v2f64 (alignedload node:$ptr))>;
def alignedloadv4i32 : PatFrag<(ops node:$ptr),
                               (v4i32 (alignedload node:$ptr))>;
def alignedloadv2i64 : PatFrag<(ops node:$ptr),
                               (v2i64 (alignedload node:$ptr))>;

// 256-bit aligned load pattern fragments
def alignedloadv8f32 : PatFrag<(ops node:$ptr),
                               (v8f32 (alignedload node:$ptr))>;
def alignedloadv4f64 : PatFrag<(ops node:$ptr),
                               (v4f64 (alignedload node:$ptr))>;
def alignedloadv8i32 : PatFrag<(ops node:$ptr),
                               (v8i32 (alignedload node:$ptr))>;
def alignedloadv4i64 : PatFrag<(ops node:$ptr),
                               (v4i64 (alignedload node:$ptr))>;

// Like 'load', but uses special alignment checks suitable for use in
// memory operands in most SSE instructions, which are required to
// be naturally aligned on some targets but not on others.  If the subtarget
// allows unaligned accesses, match any load, though this may require
// setting a feature bit in the processor (on startup, for example).
// Opteron 10h and later implement such a feature.
def memop : PatFrag<(ops node:$ptr), (load node:$ptr), [{
  return    Subtarget->hasVectorUAMem()
         || cast<LoadSDNode>(N)->getAlignment() >= 16;
}]>;

def memopfsf32 : PatFrag<(ops node:$ptr), (f32   (memop node:$ptr))>;
def memopfsf64 : PatFrag<(ops node:$ptr), (f64   (memop node:$ptr))>;

// 128-bit memop pattern fragments
def memopv4f32 : PatFrag<(ops node:$ptr), (v4f32 (memop node:$ptr))>;
def memopv2f64 : PatFrag<(ops node:$ptr), (v2f64 (memop node:$ptr))>;
def memopv4i32 : PatFrag<(ops node:$ptr), (v4i32 (memop node:$ptr))>;
def memopv2i64 : PatFrag<(ops node:$ptr), (v2i64 (memop node:$ptr))>;
def memopv8i16 : PatFrag<(ops node:$ptr), (v8i16 (memop node:$ptr))>;
def memopv16i8 : PatFrag<(ops node:$ptr), (v16i8 (memop node:$ptr))>;

// 256-bit memop pattern fragments
def memopv32i8 : PatFrag<(ops node:$ptr), (v32i8 (memop node:$ptr))>;
def memopv8f32 : PatFrag<(ops node:$ptr), (v8f32 (memop node:$ptr))>;
def memopv4f64 : PatFrag<(ops node:$ptr), (v4f64 (memop node:$ptr))>;
def memopv4i64 : PatFrag<(ops node:$ptr), (v4i64 (memop node:$ptr))>;
def memopv8i32 : PatFrag<(ops node:$ptr), (v8i32 (memop node:$ptr))>;

// SSSE3 uses MMX registers for some instructions. They aren't aligned on a
// 16-byte boundary.
// FIXME: 8 byte alignment for mmx reads is not required
def memop64 : PatFrag<(ops node:$ptr), (load node:$ptr), [{
  return cast<LoadSDNode>(N)->getAlignment() >= 8;
}]>;

def memopmmx  : PatFrag<(ops node:$ptr), (x86mmx  (memop64 node:$ptr))>;

// MOVNT Support
// Like 'store', but requires the non-temporal bit to be set
def nontemporalstore : PatFrag<(ops node:$val, node:$ptr),
                           (st node:$val, node:$ptr), [{
  if (StoreSDNode *ST = dyn_cast<StoreSDNode>(N))
    return ST->isNonTemporal();
  return false;
}]>;

def alignednontemporalstore : PatFrag<(ops node:$val, node:$ptr),
			           (st node:$val, node:$ptr), [{
  if (StoreSDNode *ST = dyn_cast<StoreSDNode>(N))
    return ST->isNonTemporal() && !ST->isTruncatingStore() &&
           ST->getAddressingMode() == ISD::UNINDEXED &&
           ST->getAlignment() >= 16;
  return false;
}]>;

def unalignednontemporalstore : PatFrag<(ops node:$val, node:$ptr),
			           (st node:$val, node:$ptr), [{
  if (StoreSDNode *ST = dyn_cast<StoreSDNode>(N))
    return ST->isNonTemporal() &&
           ST->getAlignment() < 16;
  return false;
}]>;

// 128-bit bitconvert pattern fragments
def bc_v4f32 : PatFrag<(ops node:$in), (v4f32 (bitconvert node:$in))>;
def bc_v2f64 : PatFrag<(ops node:$in), (v2f64 (bitconvert node:$in))>;
def bc_v16i8 : PatFrag<(ops node:$in), (v16i8 (bitconvert node:$in))>;
def bc_v8i16 : PatFrag<(ops node:$in), (v8i16 (bitconvert node:$in))>;
def bc_v4i32 : PatFrag<(ops node:$in), (v4i32 (bitconvert node:$in))>;
def bc_v2i64 : PatFrag<(ops node:$in), (v2i64 (bitconvert node:$in))>;

// 256-bit bitconvert pattern fragments
def bc_v8i32 : PatFrag<(ops node:$in), (v8i32 (bitconvert node:$in))>;
def bc_v4i64 : PatFrag<(ops node:$in), (v4i64 (bitconvert node:$in))>;

def vzmovl_v2i64 : PatFrag<(ops node:$src),
                           (bitconvert (v2i64 (X86vzmovl
                             (v2i64 (scalar_to_vector (loadi64 node:$src))))))>;
def vzmovl_v4i32 : PatFrag<(ops node:$src),
                           (bitconvert (v4i32 (X86vzmovl
                             (v4i32 (scalar_to_vector (loadi32 node:$src))))))>;

def vzload_v2i64 : PatFrag<(ops node:$src),
                           (bitconvert (v2i64 (X86vzload node:$src)))>;


def fp32imm0 : PatLeaf<(f32 fpimm), [{
  return N->isExactlyValue(+0.0);
}]>;

// BYTE_imm - Transform bit immediates into byte immediates.
def BYTE_imm  : SDNodeXForm<imm, [{
  // Transformation function: imm >> 3
  return getI32Imm(N->getZExtValue() >> 3);
}]>;

// SHUFFLE_get_shuf_imm xform function: convert vector_shuffle mask to PSHUF*,
// SHUFP* etc. imm.
def SHUFFLE_get_shuf_imm : SDNodeXForm<vector_shuffle, [{
  return getI8Imm(X86::getShuffleSHUFImmediate(N));
}]>;

// SHUFFLE_get_pshufhw_imm xform function: convert vector_shuffle mask to
// PSHUFHW imm.
def SHUFFLE_get_pshufhw_imm : SDNodeXForm<vector_shuffle, [{
  return getI8Imm(X86::getShufflePSHUFHWImmediate(N));
}]>;

// SHUFFLE_get_pshuflw_imm xform function: convert vector_shuffle mask to
// PSHUFLW imm.
def SHUFFLE_get_pshuflw_imm : SDNodeXForm<vector_shuffle, [{
  return getI8Imm(X86::getShufflePSHUFLWImmediate(N));
}]>;

// SHUFFLE_get_palign_imm xform function: convert vector_shuffle mask to
// a PALIGNR imm.
def SHUFFLE_get_palign_imm : SDNodeXForm<vector_shuffle, [{
  return getI8Imm(X86::getShufflePALIGNRImmediate(N));
}]>;

// EXTRACT_get_vextractf128_imm xform function: convert extract_subvector index
// to VEXTRACTF128 imm.
def EXTRACT_get_vextractf128_imm : SDNodeXForm<extract_subvector, [{
  return getI8Imm(X86::getExtractVEXTRACTF128Immediate(N));
}]>;

// INSERT_get_vinsertf128_imm xform function: convert insert_subvector index to
// VINSERTF128 imm.
def INSERT_get_vinsertf128_imm : SDNodeXForm<insert_subvector, [{
  return getI8Imm(X86::getInsertVINSERTF128Immediate(N));
}]>;

def splat_lo : PatFrag<(ops node:$lhs, node:$rhs),
                       (vector_shuffle node:$lhs, node:$rhs), [{
  ShuffleVectorSDNode *SVOp = cast<ShuffleVectorSDNode>(N);
  return SVOp->isSplat() && SVOp->getSplatIndex() == 0;
}]>;

def movddup : PatFrag<(ops node:$lhs, node:$rhs),
                      (vector_shuffle node:$lhs, node:$rhs), [{
  return X86::isMOVDDUPMask(cast<ShuffleVectorSDNode>(N));
}]>;

def movhlps : PatFrag<(ops node:$lhs, node:$rhs),
                      (vector_shuffle node:$lhs, node:$rhs), [{
  return X86::isMOVHLPSMask(cast<ShuffleVectorSDNode>(N));
}]>;

def movhlps_undef : PatFrag<(ops node:$lhs, node:$rhs),
                            (vector_shuffle node:$lhs, node:$rhs), [{
  return X86::isMOVHLPS_v_undef_Mask(cast<ShuffleVectorSDNode>(N));
}]>;

def movlhps : PatFrag<(ops node:$lhs, node:$rhs),
                      (vector_shuffle node:$lhs, node:$rhs), [{
  return X86::isMOVLHPSMask(cast<ShuffleVectorSDNode>(N));
}]>;

def movlp : PatFrag<(ops node:$lhs, node:$rhs),
                    (vector_shuffle node:$lhs, node:$rhs), [{
  return X86::isMOVLPMask(cast<ShuffleVectorSDNode>(N));
}]>;

def movl : PatFrag<(ops node:$lhs, node:$rhs),
                   (vector_shuffle node:$lhs, node:$rhs), [{
  return X86::isMOVLMask(cast<ShuffleVectorSDNode>(N));
}]>;

def unpckl : PatFrag<(ops node:$lhs, node:$rhs),
                     (vector_shuffle node:$lhs, node:$rhs), [{
  return X86::isUNPCKLMask(cast<ShuffleVectorSDNode>(N));
}]>;

def unpckh : PatFrag<(ops node:$lhs, node:$rhs),
                     (vector_shuffle node:$lhs, node:$rhs), [{
  return X86::isUNPCKHMask(cast<ShuffleVectorSDNode>(N));
}]>;

def pshufd : PatFrag<(ops node:$lhs, node:$rhs),
                     (vector_shuffle node:$lhs, node:$rhs), [{
  return X86::isPSHUFDMask(cast<ShuffleVectorSDNode>(N));
}], SHUFFLE_get_shuf_imm>;

def shufp : PatFrag<(ops node:$lhs, node:$rhs),
                    (vector_shuffle node:$lhs, node:$rhs), [{
  return X86::isSHUFPMask(cast<ShuffleVectorSDNode>(N));
}], SHUFFLE_get_shuf_imm>;

def pshufhw : PatFrag<(ops node:$lhs, node:$rhs),
                      (vector_shuffle node:$lhs, node:$rhs), [{
  return X86::isPSHUFHWMask(cast<ShuffleVectorSDNode>(N));
}], SHUFFLE_get_pshufhw_imm>;

def pshuflw : PatFrag<(ops node:$lhs, node:$rhs),
                      (vector_shuffle node:$lhs, node:$rhs), [{
  return X86::isPSHUFLWMask(cast<ShuffleVectorSDNode>(N));
}], SHUFFLE_get_pshuflw_imm>;

def palign : PatFrag<(ops node:$lhs, node:$rhs),
                     (vector_shuffle node:$lhs, node:$rhs), [{
  return X86::isPALIGNRMask(cast<ShuffleVectorSDNode>(N));
}], SHUFFLE_get_palign_imm>;

def vextractf128_extract : PatFrag<(ops node:$bigvec, node:$index),
                                   (extract_subvector node:$bigvec,
                                                      node:$index), [{
  return X86::isVEXTRACTF128Index(N);
}], EXTRACT_get_vextractf128_imm>;

def vinsertf128_insert : PatFrag<(ops node:$bigvec, node:$smallvec,
                                      node:$index),
                                 (insert_subvector node:$bigvec, node:$smallvec,
                                                   node:$index), [{
  return X86::isVINSERTF128Index(N);
}], INSERT_get_vinsertf128_imm>;

