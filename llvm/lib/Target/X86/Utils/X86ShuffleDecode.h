//===-- X86ShuffleDecode.h - X86 shuffle decode logic -----------*-C++-*---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Define several functions to decode x86 specific shuffle semantics into a
// generic vector mask.
//
//===----------------------------------------------------------------------===//

#ifndef X86_SHUFFLE_DECODE_H
#define X86_SHUFFLE_DECODE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/ValueTypes.h"

//===----------------------------------------------------------------------===//
//  Vector Mask Decoding
//===----------------------------------------------------------------------===//

namespace llvm {
enum {
  SM_SentinelZero = ~0U
};

void DecodeINSERTPSMask(unsigned Imm, SmallVectorImpl<unsigned> &ShuffleMask);

// <3,1> or <6,7,2,3>
void DecodeMOVHLPSMask(unsigned NElts,
                       SmallVectorImpl<unsigned> &ShuffleMask);

// <0,2> or <0,1,4,5>
void DecodeMOVLHPSMask(unsigned NElts,
                       SmallVectorImpl<unsigned> &ShuffleMask);

void DecodePSHUFMask(unsigned NElts, unsigned Imm,
                     SmallVectorImpl<unsigned> &ShuffleMask);

void DecodePSHUFHWMask(unsigned Imm,
                       SmallVectorImpl<unsigned> &ShuffleMask);

void DecodePSHUFLWMask(unsigned Imm,
                       SmallVectorImpl<unsigned> &ShuffleMask);

void DecodePUNPCKLBWMask(unsigned NElts,
                         SmallVectorImpl<unsigned> &ShuffleMask);

void DecodePUNPCKLWDMask(unsigned NElts,
                         SmallVectorImpl<unsigned> &ShuffleMask);

void DecodePUNPCKLDQMask(unsigned NElts,
                         SmallVectorImpl<unsigned> &ShuffleMask);

void DecodePUNPCKLQDQMask(unsigned NElts,
                          SmallVectorImpl<unsigned> &ShuffleMask);

void DecodePUNPCKLMask(EVT VT,
                       SmallVectorImpl<unsigned> &ShuffleMask);

void DecodePUNPCKHMask(unsigned NElts,
                       SmallVectorImpl<unsigned> &ShuffleMask);

void DecodeSHUFPSMask(unsigned NElts, unsigned Imm,
                      SmallVectorImpl<unsigned> &ShuffleMask);

void DecodeUNPCKHPMask(unsigned NElts,
                       SmallVectorImpl<unsigned> &ShuffleMask);

void DecodeUNPCKLPSMask(unsigned NElts,
                        SmallVectorImpl<unsigned> &ShuffleMask);

void DecodeUNPCKLPDMask(unsigned NElts,
                        SmallVectorImpl<unsigned> &ShuffleMask);

/// DecodeUNPCKLPMask - This decodes the shuffle masks for unpcklps/unpcklpd
/// etc.  VT indicates the type of the vector allowing it to handle different
/// datatypes and vector widths.
void DecodeUNPCKLPMask(EVT VT,
                       SmallVectorImpl<unsigned> &ShuffleMask);


// DecodeVPERMILPSMask - Decodes VPERMILPS permutes for any 128-bit 32-bit
// elements. For 256-bit vectors, it's considered as two 128 lanes, the
// referenced elements can't cross lanes and the mask of the first lane must
// be the same of the second.
void DecodeVPERMILPSMask(unsigned NElts, unsigned Imm,
                        SmallVectorImpl<unsigned> &ShuffleMask);

// DecodeVPERMILPDMask - Decodes VPERMILPD permutes for any 128-bit 64-bit
// elements. For 256-bit vectors, it's considered as two 128 lanes, the
// referenced elements can't cross lanes but the mask of the first lane can
// be the different of the second (not like VPERMILPS).
void DecodeVPERMILPDMask(unsigned NElts, unsigned Imm,
                        SmallVectorImpl<unsigned> &ShuffleMask);

} // llvm namespace

#endif
