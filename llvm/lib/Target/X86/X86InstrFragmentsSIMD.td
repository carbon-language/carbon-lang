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

def load_mmx : PatFrag<(ops node:$ptr), (v1i64 (load node:$ptr))>;

def bc_v8i8  : PatFrag<(ops node:$in), (v8i8  (bitconvert node:$in))>;
def bc_v4i16 : PatFrag<(ops node:$in), (v4i16 (bitconvert node:$in))>;
def bc_v2i32 : PatFrag<(ops node:$in), (v2i32 (bitconvert node:$in))>;
def bc_v1i64 : PatFrag<(ops node:$in), (v1i64 (bitconvert node:$in))>;

//===----------------------------------------------------------------------===//
// MMX Masks
//===----------------------------------------------------------------------===//

// MMX_SHUFFLE_get_shuf_imm xform function: convert vector_shuffle mask to
// PSHUFW imm.
def MMX_SHUFFLE_get_shuf_imm : SDNodeXForm<vector_shuffle, [{
  return getI8Imm(X86::getShuffleSHUFImmediate(N));
}]>;

// Patterns for: vector_shuffle v1, v2, <2, 6, 3, 7, ...>
def mmx_unpckh : PatFrag<(ops node:$lhs, node:$rhs),
                         (vector_shuffle node:$lhs, node:$rhs), [{
  return X86::isUNPCKHMask(cast<ShuffleVectorSDNode>(N));
}]>;

// Patterns for: vector_shuffle v1, v2, <0, 4, 2, 5, ...>
def mmx_unpckl : PatFrag<(ops node:$lhs, node:$rhs),
                         (vector_shuffle node:$lhs, node:$rhs), [{
  return X86::isUNPCKLMask(cast<ShuffleVectorSDNode>(N));
}]>;

// Patterns for: vector_shuffle v1, <undef>, <0, 0, 1, 1, ...>
def mmx_unpckh_undef : PatFrag<(ops node:$lhs, node:$rhs),
                               (vector_shuffle node:$lhs, node:$rhs), [{
  return X86::isUNPCKH_v_undef_Mask(cast<ShuffleVectorSDNode>(N));
}]>;

// Patterns for: vector_shuffle v1, <undef>, <2, 2, 3, 3, ...>
def mmx_unpckl_undef : PatFrag<(ops node:$lhs, node:$rhs),
                               (vector_shuffle node:$lhs, node:$rhs), [{
  return X86::isUNPCKL_v_undef_Mask(cast<ShuffleVectorSDNode>(N));
}]>;

def mmx_pshufw : PatFrag<(ops node:$lhs, node:$rhs),
                         (vector_shuffle node:$lhs, node:$rhs), [{
  return X86::isPSHUFDMask(cast<ShuffleVectorSDNode>(N));
}], MMX_SHUFFLE_get_shuf_imm>;
