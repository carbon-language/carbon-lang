//===- SPUNodes.td - Specialized SelectionDAG nodes used for CellSPU ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Type profiles and SelectionDAG nodes used by CellSPU
//
//===----------------------------------------------------------------------===//

// Type profile for a call sequence
def SDT_SPUCallSeq : SDTypeProfile<0, 1, [ SDTCisVT<0, i32> ]>;

// SPU_GenControl: Type profile for generating control words for insertions
def SPU_GenControl : SDTypeProfile<1, 1, []>;
def SPUvecinsmask : SDNode<"SPUISD::INSERT_MASK", SPU_GenControl, []>;

def callseq_start : SDNode<"ISD::CALLSEQ_START", SDT_SPUCallSeq,
                           [SDNPHasChain, SDNPOutFlag]>;
def callseq_end   : SDNode<"ISD::CALLSEQ_END",   SDT_SPUCallSeq,
                           [SDNPHasChain, SDNPOutFlag]>;
//===----------------------------------------------------------------------===//
// Operand constraints:
//===----------------------------------------------------------------------===//

def SDT_SPUCall   : SDTypeProfile<0, -1, [SDTCisInt<0>]>;
def SPUcall       : SDNode<"SPUISD::CALL", SDT_SPUCall,
                           [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;

// Operand type constraints for vector shuffle/permute operations
def SDT_SPUshuffle   : SDTypeProfile<1, 3, [
  SDTCisVT<3, v16i8>, SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>
]>;

// Unary, binary v16i8 operator type constraints:
def SPUv16i8_unop: SDTypeProfile<1, 1, [
  SDTCisVT<0, v16i8>, SDTCisSameAs<0, 1>]>;

def SPUv16i8_binop: SDTypeProfile<1, 2, [
  SDTCisVT<0, v16i8>, SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>]>;

// Binary v8i16 operator type constraints:
def SPUv8i16_unop: SDTypeProfile<1, 1, [
  SDTCisVT<0, v8i16>, SDTCisSameAs<0, 1>]>;

def SPUv8i16_binop: SDTypeProfile<1, 2, [
  SDTCisVT<0, v8i16>, SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>]>;

// Binary v4i32 operator type constraints:
def SPUv4i32_unop: SDTypeProfile<1, 1, [
  SDTCisVT<0, v4i32>, SDTCisSameAs<0, 1>]>;

def SPUv4i32_binop: SDTypeProfile<1, 2, [
  SDTCisVT<0, v4i32>, SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>]>;

// FSMBI type constraints: There are several variations for the various
// vector types (this avoids having to bit_convert all over the place.)
def SPUfsmbi_type_v16i8: SDTypeProfile<1, 1, [
  SDTCisVT<0, v16i8>, SDTCisVT<1, i32>]>;

def SPUfsmbi_type_v8i16: SDTypeProfile<1, 1, [
  SDTCisVT<0, v8i16>, SDTCisVT<1, i32>]>;

def SPUfsmbi_type_v4i32: SDTypeProfile<1, 1, [
  SDTCisVT<0, v4i32>, SDTCisVT<1, i32>]>;

// SELB type constraints:
def SPUselb_type_v16i8: SDTypeProfile<1, 3, [
  SDTCisVT<0, v16i8>, SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>,
  SDTCisSameAs<0, 3> ]>;

def SPUselb_type_v8i16: SDTypeProfile<1, 3, [
  SDTCisVT<0, v8i16>, SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>,
  SDTCisSameAs<0, 3> ]>;

def SPUselb_type_v4i32: SDTypeProfile<1, 3, [
  SDTCisVT<0, v4i32>, SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>,
  SDTCisSameAs<0, 3> ]>;

// SPU Vector shift pseudo-instruction type constraints
def SPUvecshift_type_v16i8: SDTypeProfile<1, 2, [
  SDTCisVT<0, v16i8>, SDTCisSameAs<0, 1>, SDTCisInt<2>]>;

def SPUvecshift_type_v8i16: SDTypeProfile<1, 2, [
  SDTCisVT<0, v8i16>, SDTCisSameAs<0, 1>, SDTCisInt<2>]>;

def SPUvecshift_type_v4i32: SDTypeProfile<1, 2, [
  SDTCisVT<0, v4i32>, SDTCisSameAs<0, 1>, SDTCisInt<2>]>;

//===----------------------------------------------------------------------===//
// Synthetic/pseudo-instructions
//===----------------------------------------------------------------------===//

// SPU CNTB:
def SPUcntb_v16i8: SDNode<"SPUISD::CNTB", SPUv16i8_unop, []>;
def SPUcntb_v8i16: SDNode<"SPUISD::CNTB", SPUv8i16_unop, []>;
def SPUcntb_v4i32: SDNode<"SPUISD::CNTB", SPUv4i32_unop, []>;

// SPU vector shuffle node, matched by the SPUISD::SHUFB enum (see
// SPUISelLowering.h):
def SPUshuffle: SDNode<"SPUISD::SHUFB", SDT_SPUshuffle, []>;

// SPU 16-bit multiply
def SPUmpy_v16i8: SDNode<"SPUISD::MPY", SPUv16i8_binop, []>;
def SPUmpy_v8i16: SDNode<"SPUISD::MPY", SPUv8i16_binop, []>;
def SPUmpy_v4i32: SDNode<"SPUISD::MPY", SPUv4i32_binop, []>;

// SPU multiply unsigned, used in instruction lowering for v4i32
// multiplies:
def SPUmpyu_v4i32: SDNode<"SPUISD::MPYU", SPUv4i32_binop, []>;
def SPUmpyu_i32: SDNode<"SPUISD::MPYU", SDTIntBinOp, []>;

// SPU 16-bit multiply high x low, shift result 16-bits
// Used to compute intermediate products for 32-bit multiplies
def SPUmpyh_v4i32: SDNode<"SPUISD::MPYH", SPUv4i32_binop, []>;
def SPUmpyh_i32: SDNode<"SPUISD::MPYH", SDTIntBinOp, []>;

// SPU 16-bit multiply high x high, 32-bit product
// Used to compute intermediate products for 16-bit multiplies
def SPUmpyhh_v8i16: SDNode<"SPUISD::MPYHH", SPUv8i16_binop, []>;

// Vector shifts (ISD::SHL,SRL,SRA are for _integers_ only):
def SPUvec_shl_v8i16: SDNode<"SPUISD::VEC_SHL", SPUvecshift_type_v8i16, []>;
def SPUvec_srl_v8i16: SDNode<"SPUISD::VEC_SRL", SPUvecshift_type_v8i16, []>;
def SPUvec_sra_v8i16: SDNode<"SPUISD::VEC_SRA", SPUvecshift_type_v8i16, []>;

def SPUvec_shl_v4i32: SDNode<"SPUISD::VEC_SHL", SPUvecshift_type_v4i32, []>;
def SPUvec_srl_v4i32: SDNode<"SPUISD::VEC_SRL", SPUvecshift_type_v4i32, []>;
def SPUvec_sra_v4i32: SDNode<"SPUISD::VEC_SRA", SPUvecshift_type_v4i32, []>;

def SPUvec_rotl_v8i16: SDNode<"SPUISD::VEC_ROTL", SPUvecshift_type_v8i16, []>;
def SPUvec_rotl_v4i32: SDNode<"SPUISD::VEC_ROTL", SPUvecshift_type_v4i32, []>;

def SPUvec_rotr_v8i16: SDNode<"SPUISD::VEC_ROTR", SPUvecshift_type_v8i16, []>;
def SPUvec_rotr_v4i32: SDNode<"SPUISD::VEC_ROTR", SPUvecshift_type_v4i32, []>;

def SPUrotbytes_right_zfill: SDNode<"SPUISD::ROTBYTES_RIGHT_Z",
                                    SPUvecshift_type_v16i8, []>;
def SPUrotbytes_right_sfill: SDNode<"SPUISD::ROTBYTES_RIGHT_S",
                                    SPUvecshift_type_v16i8, []>;
def SPUrotbytes_left: SDNode<"SPUISD::ROTBYTES_LEFT",
                             SPUvecshift_type_v16i8, []>;

def SPUrotbytes_left_chained : SDNode<"SPUISD::ROTBYTES_LEFT_CHAINED",
                                      SPUvecshift_type_v16i8, [SDNPHasChain]>;

// SPU form select mask for bytes, immediate
def SPUfsmbi_v16i8: SDNode<"SPUISD::FSMBI", SPUfsmbi_type_v16i8, []>;
def SPUfsmbi_v8i16: SDNode<"SPUISD::FSMBI", SPUfsmbi_type_v8i16, []>;
def SPUfsmbi_v4i32: SDNode<"SPUISD::FSMBI", SPUfsmbi_type_v4i32, []>;

// SPU select bits instruction
def SPUselb_v16i8: SDNode<"SPUISD::SELB", SPUselb_type_v16i8, []>;
def SPUselb_v8i16: SDNode<"SPUISD::SELB", SPUselb_type_v8i16, []>;
def SPUselb_v4i32: SDNode<"SPUISD::SELB", SPUselb_type_v4i32, []>;

// SPU single precision floating point constant load
def SPUFPconstant: SDNode<"SPUISD::SFPConstant", SDTFPUnaryOp, []>;

// SPU floating point interpolate
def SPUinterpolate : SDNode<"SPUISD::FPInterp", SDTFPBinOp, []>;

// SPU floating point reciprocal estimate (used for fdiv)
def SPUreciprocalEst: SDNode<"SPUISD::FPRecipEst", SDTFPUnaryOp, []>;

def SDT_vec_promote   : SDTypeProfile<1, 1, []>;
def SPUpromote_scalar: SDNode<"SPUISD::PROMOTE_SCALAR", SDT_vec_promote, []>;

def SPU_vec_demote   : SDTypeProfile<1, 1, []>;
def SPUextract_elt0: SDNode<"SPUISD::EXTRACT_ELT0", SPU_vec_demote, []>;
def SPU_vec_demote_chained : SDTypeProfile<1, 2, []>;
def SPUextract_elt0_chained: SDNode<"SPUISD::EXTRACT_ELT0_CHAINED",
                                    SPU_vec_demote_chained, [SDNPHasChain]>;
def SPUextract_i1_sext: SDNode<"SPUISD::EXTRACT_I1_SEXT", SPU_vec_demote, []>;
def SPUextract_i1_zext: SDNode<"SPUISD::EXTRACT_I1_ZEXT", SPU_vec_demote, []>;
def SPUextract_i8_sext: SDNode<"SPUISD::EXTRACT_I8_SEXT", SPU_vec_demote, []>;
def SPUextract_i8_zext: SDNode<"SPUISD::EXTRACT_I8_ZEXT", SPU_vec_demote, []>;

// Address high and low components, used for [r+r] type addressing
def SPUhi : SDNode<"SPUISD::Hi", SDTIntBinOp, []>;
def SPUlo : SDNode<"SPUISD::Lo", SDTIntBinOp, []>;

// PC-relative address
def SPUpcrel : SDNode<"SPUISD::PCRelAddr", SDTIntBinOp, []>;

// A-Form local store addresses
def SPUaform : SDNode<"SPUISD::AFormAddr", SDTIntBinOp, []>;

// D-Form "imm($reg)" addresses
def SPUdform : SDNode<"SPUISD::DFormAddr", SDTIntBinOp, []>;

// X-Form "$reg($reg)" addresses
def SPUxform : SDNode<"SPUISD::XFormAddr", SDTIntBinOp, []>;

// SPU 32-bit sign-extension to 64-bits
def SPUsext32_to_64: SDNode<"SPUISD::SEXT32TO64", SDTIntExtendOp, []>;

// Branches:

def SPUbrnz : SDNode<"SPUISD::BR_NOTZERO", SDTBrcond,  [SDNPHasChain]>;
def SPUbrz  : SDNode<"SPUISD::BR_ZERO",    SDTBrcond,  [SDNPHasChain]>;
/* def SPUbinz : SDNode<"SPUISD::BR_NOTZERO", SDTBrind,   [SDNPHasChain]>;
def SPUbiz  : SDNode<"SPUISD::BR_ZERO",    SPUBrind,   [SDNPHasChain]>; */

//===----------------------------------------------------------------------===//
// Constraints: (taken from PPCInstrInfo.td)
//===----------------------------------------------------------------------===//

class RegConstraint<string C> {
  string Constraints = C;
}

class NoEncode<string E> {
  string DisableEncoding = E;
}

//===----------------------------------------------------------------------===//
// Return (flag isn't quite what it means: the operations are flagged so that
// instruction scheduling doesn't disassociate them.)
//===----------------------------------------------------------------------===//

def retflag     : SDNode<"SPUISD::RET_FLAG", SDTNone,
                         [SDNPHasChain, SDNPOptInFlag]>;
