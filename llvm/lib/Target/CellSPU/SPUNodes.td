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
  SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>
]>;

// Unary, binary v16i8 operator type constraints:
def SPUv16i8_binop: SDTypeProfile<1, 2, [
  SDTCisVT<0, v16i8>, SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>]>;

// Binary v8i16 operator type constraints:
def SPUv8i16_binop: SDTypeProfile<1, 2, [
  SDTCisVT<0, v8i16>, SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>]>;

// Binary v4i32 operator type constraints:
def SPUv4i32_binop: SDTypeProfile<1, 2, [
  SDTCisVT<0, v4i32>, SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>]>;

// Trinary operators, e.g., addx, carry generate
def SPUIntTrinaryOp : SDTypeProfile<1, 3, [
  SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisSameAs<0, 3>, SDTCisInt<0>
]>;

// SELECT_MASK type constraints: There are several variations for the various
// vector types (this avoids having to bit_convert all over the place.)
def SPUselmask_type: SDTypeProfile<1, 1, [
  SDTCisInt<1>
]>;

// SELB type constraints:
def SPUselb_type: SDTypeProfile<1, 3, [
  SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>, SDTCisSameAs<0, 3> ]>;

// SPU Vector shift pseudo-instruction type constraints
def SPUvecshift_type: SDTypeProfile<1, 2, [
  SDTCisSameAs<0, 1>, SDTCisInt<2>]>;

//===----------------------------------------------------------------------===//
// Synthetic/pseudo-instructions
//===----------------------------------------------------------------------===//

/// Add extended, carry generate:
def SPUaddx : SDNode<"SPUISD::ADD_EXTENDED", SPUIntTrinaryOp, []>;
def SPUcarry_gen : SDNode<"SPUISD::CARRY_GENERATE", SDTIntBinOp, []>;

// Subtract extended, borrow generate
def SPUsubx : SDNode<"SPUISD::SUB_EXTENDED", SPUIntTrinaryOp, []>;
def SPUborrow_gen : SDNode<"SPUISD::BORROW_GENERATE", SDTIntBinOp, []>;

// SPU CNTB:
def SPUcntb : SDNode<"SPUISD::CNTB", SDTIntUnaryOp>;

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

// Shift left quadword by bits and bytes
def SPUshlquad_l_bits: SDNode<"SPUISD::SHLQUAD_L_BITS", SPUvecshift_type, []>;
def SPUshlquad_l_bytes: SDNode<"SPUISD::SHLQUAD_L_BYTES", SPUvecshift_type, []>;

// Vector shifts (ISD::SHL,SRL,SRA are for _integers_ only):
def SPUvec_shl: SDNode<"SPUISD::VEC_SHL", SPUvecshift_type, []>;
def SPUvec_srl: SDNode<"SPUISD::VEC_SRL", SPUvecshift_type, []>;
def SPUvec_sra: SDNode<"SPUISD::VEC_SRA", SPUvecshift_type, []>;

def SPUvec_rotl: SDNode<"SPUISD::VEC_ROTL", SPUvecshift_type, []>;
def SPUvec_rotr: SDNode<"SPUISD::VEC_ROTR", SPUvecshift_type, []>;

def SPUrotquad_rz_bytes: SDNode<"SPUISD::ROTQUAD_RZ_BYTES",
                                    SPUvecshift_type, []>;
def SPUrotquad_rz_bits: SDNode<"SPUISD::ROTQUAD_RZ_BITS",
                                    SPUvecshift_type, []>;

def SPUrotbytes_right_sfill: SDNode<"SPUISD::ROTBYTES_RIGHT_S",
                                    SPUvecshift_type, []>;

// Vector rotate left, bits shifted out of the left are rotated in on the right
def SPUrotbytes_left: SDNode<"SPUISD::ROTBYTES_LEFT",
                             SPUvecshift_type, []>;

// Same as above, but the node also has a chain associated (used in loads and
// stores)
def SPUrotbytes_left_chained : SDNode<"SPUISD::ROTBYTES_LEFT_CHAINED",
                                      SPUvecshift_type, [SDNPHasChain]>;

// Vector rotate left by bytes, but the count is given in bits and the SPU
// internally converts it to bytes (saves an instruction to mask off lower
// three bits)
def SPUrotbytes_left_bits : SDNode<"SPUISD::ROTBYTES_LEFT_BITS",
                                   SPUvecshift_type>;

// SPU form select mask for bytes, immediate
def SPUselmask: SDNode<"SPUISD::SELECT_MASK", SPUselmask_type, []>;

// SPU select bits instruction
def SPUselb: SDNode<"SPUISD::SELB", SPUselb_type, []>;

// SPU floating point interpolate
def SPUinterpolate : SDNode<"SPUISD::FPInterp", SDTFPBinOp, []>;

// SPU floating point reciprocal estimate (used for fdiv)
def SPUreciprocalEst: SDNode<"SPUISD::FPRecipEst", SDTFPUnaryOp, []>;

def SDTpromote_scalar: SDTypeProfile<1, 1, []>;
def SPUpromote_scalar: SDNode<"SPUISD::PROMOTE_SCALAR", SDTpromote_scalar, []>;

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

// Indirect [D-Form "imm($reg)" and X-Form "$reg($reg)"] addresses
def SPUindirect : SDNode<"SPUISD::IndirectAddr", SDTIntBinOp, []>;

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
