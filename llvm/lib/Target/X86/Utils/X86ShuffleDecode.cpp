//===-- X86ShuffleDecode.cpp - X86 shuffle decode logic -------------------===//
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

#include "X86ShuffleDecode.h"
#include "llvm/IR/Constants.h"
#include "llvm/CodeGen/MachineValueType.h"

//===----------------------------------------------------------------------===//
//  Vector Mask Decoding
//===----------------------------------------------------------------------===//

namespace llvm {

void DecodeINSERTPSMask(unsigned Imm, SmallVectorImpl<int> &ShuffleMask) {
  // Defaults the copying the dest value.
  ShuffleMask.push_back(0);
  ShuffleMask.push_back(1);
  ShuffleMask.push_back(2);
  ShuffleMask.push_back(3);

  // Decode the immediate.
  unsigned ZMask = Imm & 15;
  unsigned CountD = (Imm >> 4) & 3;
  unsigned CountS = (Imm >> 6) & 3;

  // CountS selects which input element to use.
  unsigned InVal = 4 + CountS;
  // CountD specifies which element of destination to update.
  ShuffleMask[CountD] = InVal;
  // ZMask zaps values, potentially overriding the CountD elt.
  if (ZMask & 1) ShuffleMask[0] = SM_SentinelZero;
  if (ZMask & 2) ShuffleMask[1] = SM_SentinelZero;
  if (ZMask & 4) ShuffleMask[2] = SM_SentinelZero;
  if (ZMask & 8) ShuffleMask[3] = SM_SentinelZero;
}

// <3,1> or <6,7,2,3>
void DecodeMOVHLPSMask(unsigned NElts, SmallVectorImpl<int> &ShuffleMask) {
  for (unsigned i = NElts / 2; i != NElts; ++i)
    ShuffleMask.push_back(NElts + i);

  for (unsigned i = NElts / 2; i != NElts; ++i)
    ShuffleMask.push_back(i);
}

// <0,2> or <0,1,4,5>
void DecodeMOVLHPSMask(unsigned NElts, SmallVectorImpl<int> &ShuffleMask) {
  for (unsigned i = 0; i != NElts / 2; ++i)
    ShuffleMask.push_back(i);

  for (unsigned i = 0; i != NElts / 2; ++i)
    ShuffleMask.push_back(NElts + i);
}

void DecodeMOVSLDUPMask(MVT VT, SmallVectorImpl<int> &ShuffleMask) {
  unsigned NumElts = VT.getVectorNumElements();
  for (int i = 0, e = NumElts / 2; i < e; ++i) {
    ShuffleMask.push_back(2 * i);
    ShuffleMask.push_back(2 * i);
  }
}

void DecodeMOVSHDUPMask(MVT VT, SmallVectorImpl<int> &ShuffleMask) {
  unsigned NumElts = VT.getVectorNumElements();
  for (int i = 0, e = NumElts / 2; i < e; ++i) {
    ShuffleMask.push_back(2 * i + 1);
    ShuffleMask.push_back(2 * i + 1);
  }
}

void DecodeMOVDDUPMask(MVT VT, SmallVectorImpl<int> &ShuffleMask) {
  unsigned VectorSizeInBits = VT.getSizeInBits();
  unsigned ScalarSizeInBits = VT.getScalarSizeInBits();
  unsigned NumElts = VT.getVectorNumElements();
  unsigned NumLanes = VectorSizeInBits / 128;
  unsigned NumLaneElts = NumElts / NumLanes;
  unsigned NumLaneSubElts = 64 / ScalarSizeInBits;

  for (unsigned l = 0; l < NumElts; l += NumLaneElts)
    for (unsigned i = 0; i < NumLaneElts; i += NumLaneSubElts)
      for (unsigned s = 0; s != NumLaneSubElts; s++)
        ShuffleMask.push_back(l + s);
}

void DecodePSLLDQMask(MVT VT, unsigned Imm, SmallVectorImpl<int> &ShuffleMask) {
  unsigned VectorSizeInBits = VT.getSizeInBits();
  unsigned NumElts = VectorSizeInBits / 8;
  unsigned NumLanes = VectorSizeInBits / 128;
  unsigned NumLaneElts = NumElts / NumLanes;

  for (unsigned l = 0; l < NumElts; l += NumLaneElts)
    for (unsigned i = 0; i < NumLaneElts; ++i) {
      int M = SM_SentinelZero;
      if (i >= Imm) M = i - Imm + l;
      ShuffleMask.push_back(M);
    }
}

void DecodePSRLDQMask(MVT VT, unsigned Imm, SmallVectorImpl<int> &ShuffleMask) {
  unsigned VectorSizeInBits = VT.getSizeInBits();
  unsigned NumElts = VectorSizeInBits / 8;
  unsigned NumLanes = VectorSizeInBits / 128;
  unsigned NumLaneElts = NumElts / NumLanes;

  for (unsigned l = 0; l < NumElts; l += NumLaneElts)
    for (unsigned i = 0; i < NumLaneElts; ++i) {
      unsigned Base = i + Imm;
      int M = Base + l;
      if (Base >= NumLaneElts) M = SM_SentinelZero;
      ShuffleMask.push_back(M);
    }
}

void DecodePALIGNRMask(MVT VT, unsigned Imm,
                       SmallVectorImpl<int> &ShuffleMask) {
  unsigned NumElts = VT.getVectorNumElements();
  unsigned Offset = Imm * (VT.getVectorElementType().getSizeInBits() / 8);

  unsigned NumLanes = VT.getSizeInBits() / 128;
  unsigned NumLaneElts = NumElts / NumLanes;

  for (unsigned l = 0; l != NumElts; l += NumLaneElts) {
    for (unsigned i = 0; i != NumLaneElts; ++i) {
      unsigned Base = i + Offset;
      // if i+offset is out of this lane then we actually need the other source
      if (Base >= NumLaneElts) Base += NumElts - NumLaneElts;
      ShuffleMask.push_back(Base + l);
    }
  }
}

/// DecodePSHUFMask - This decodes the shuffle masks for pshufd, and vpermilp*.
/// VT indicates the type of the vector allowing it to handle different
/// datatypes and vector widths.
void DecodePSHUFMask(MVT VT, unsigned Imm, SmallVectorImpl<int> &ShuffleMask) {
  unsigned NumElts = VT.getVectorNumElements();

  unsigned NumLanes = VT.getSizeInBits() / 128;
  unsigned NumLaneElts = NumElts / NumLanes;

  unsigned NewImm = Imm;
  for (unsigned l = 0; l != NumElts; l += NumLaneElts) {
    for (unsigned i = 0; i != NumLaneElts; ++i) {
      ShuffleMask.push_back(NewImm % NumLaneElts + l);
      NewImm /= NumLaneElts;
    }
    if (NumLaneElts == 4) NewImm = Imm; // reload imm
  }
}

void DecodePSHUFHWMask(MVT VT, unsigned Imm,
                       SmallVectorImpl<int> &ShuffleMask) {
  unsigned NumElts = VT.getVectorNumElements();

  for (unsigned l = 0; l != NumElts; l += 8) {
    unsigned NewImm = Imm;
    for (unsigned i = 0, e = 4; i != e; ++i) {
      ShuffleMask.push_back(l + i);
    }
    for (unsigned i = 4, e = 8; i != e; ++i) {
      ShuffleMask.push_back(l + 4 + (NewImm & 3));
      NewImm >>= 2;
    }
  }
}

void DecodePSHUFLWMask(MVT VT, unsigned Imm,
                       SmallVectorImpl<int> &ShuffleMask) {
  unsigned NumElts = VT.getVectorNumElements();

  for (unsigned l = 0; l != NumElts; l += 8) {
    unsigned NewImm = Imm;
    for (unsigned i = 0, e = 4; i != e; ++i) {
      ShuffleMask.push_back(l + (NewImm & 3));
      NewImm >>= 2;
    }
    for (unsigned i = 4, e = 8; i != e; ++i) {
      ShuffleMask.push_back(l + i);
    }
  }
}

/// DecodeSHUFPMask - This decodes the shuffle masks for shufp*. VT indicates
/// the type of the vector allowing it to handle different datatypes and vector
/// widths.
void DecodeSHUFPMask(MVT VT, unsigned Imm, SmallVectorImpl<int> &ShuffleMask) {
  unsigned NumElts = VT.getVectorNumElements();

  unsigned NumLanes = VT.getSizeInBits() / 128;
  unsigned NumLaneElts = NumElts / NumLanes;

  unsigned NewImm = Imm;
  for (unsigned l = 0; l != NumElts; l += NumLaneElts) {
    // each half of a lane comes from different source
    for (unsigned s = 0; s != NumElts * 2; s += NumElts) {
      for (unsigned i = 0; i != NumLaneElts / 2; ++i) {
        ShuffleMask.push_back(NewImm % NumLaneElts + s + l);
        NewImm /= NumLaneElts;
      }
    }
    if (NumLaneElts == 4) NewImm = Imm; // reload imm
  }
}

/// DecodeUNPCKHMask - This decodes the shuffle masks for unpckhps/unpckhpd
/// and punpckh*. VT indicates the type of the vector allowing it to handle
/// different datatypes and vector widths.
void DecodeUNPCKHMask(MVT VT, SmallVectorImpl<int> &ShuffleMask) {
  unsigned NumElts = VT.getVectorNumElements();

  // Handle 128 and 256-bit vector lengths. AVX defines UNPCK* to operate
  // independently on 128-bit lanes.
  unsigned NumLanes = VT.getSizeInBits() / 128;
  if (NumLanes == 0 ) NumLanes = 1;  // Handle MMX
  unsigned NumLaneElts = NumElts / NumLanes;

  for (unsigned l = 0; l != NumElts; l += NumLaneElts) {
    for (unsigned i = l + NumLaneElts / 2, e = l + NumLaneElts; i != e; ++i) {
      ShuffleMask.push_back(i);           // Reads from dest/src1
      ShuffleMask.push_back(i + NumElts); // Reads from src/src2
    }
  }
}

/// DecodeUNPCKLMask - This decodes the shuffle masks for unpcklps/unpcklpd
/// and punpckl*. VT indicates the type of the vector allowing it to handle
/// different datatypes and vector widths.
void DecodeUNPCKLMask(MVT VT, SmallVectorImpl<int> &ShuffleMask) {
  unsigned NumElts = VT.getVectorNumElements();

  // Handle 128 and 256-bit vector lengths. AVX defines UNPCK* to operate
  // independently on 128-bit lanes.
  unsigned NumLanes = VT.getSizeInBits() / 128;
  if (NumLanes == 0 ) NumLanes = 1;  // Handle MMX
  unsigned NumLaneElts = NumElts / NumLanes;

  for (unsigned l = 0; l != NumElts; l += NumLaneElts) {
    for (unsigned i = l, e = l + NumLaneElts / 2; i != e; ++i) {
      ShuffleMask.push_back(i);           // Reads from dest/src1
      ShuffleMask.push_back(i + NumElts); // Reads from src/src2
    }
  }
}

void DecodeVPERM2X128Mask(MVT VT, unsigned Imm,
                          SmallVectorImpl<int> &ShuffleMask) {
  if (Imm & 0x88)
    return; // Not a shuffle

  unsigned HalfSize = VT.getVectorNumElements() / 2;

  for (unsigned l = 0; l != 2; ++l) {
    unsigned HalfBegin = ((Imm >> (l * 4)) & 0x3) * HalfSize;
    for (unsigned i = HalfBegin, e = HalfBegin + HalfSize; i != e; ++i)
      ShuffleMask.push_back(i);
  }
}

void DecodePSHUFBMask(const Constant *C, SmallVectorImpl<int> &ShuffleMask) {
  Type *MaskTy = C->getType();
  // It is not an error for the PSHUFB mask to not be a vector of i8 because the
  // constant pool uniques constants by their bit representation.
  // e.g. the following take up the same space in the constant pool:
  //   i128 -170141183420855150465331762880109871104
  //
  //   <2 x i64> <i64 -9223372034707292160, i64 -9223372034707292160>
  //
  //   <4 x i32> <i32 -2147483648, i32 -2147483648,
  //              i32 -2147483648, i32 -2147483648>

  unsigned MaskTySize = MaskTy->getPrimitiveSizeInBits();

  if (MaskTySize != 128 && MaskTySize != 256) // FIXME: Add support for AVX-512.
    return;

  // This is a straightforward byte vector.
  if (MaskTy->isVectorTy() && MaskTy->getVectorElementType()->isIntegerTy(8)) {
    int NumElements = MaskTy->getVectorNumElements();
    ShuffleMask.reserve(NumElements);

    for (int i = 0; i < NumElements; ++i) {
      // For AVX vectors with 32 bytes the base of the shuffle is the 16-byte
      // lane of the vector we're inside.
      int Base = i < 16 ? 0 : 16;
      Constant *COp = C->getAggregateElement(i);
      if (!COp) {
        ShuffleMask.clear();
        return;
      } else if (isa<UndefValue>(COp)) {
        ShuffleMask.push_back(SM_SentinelUndef);
        continue;
      }
      uint64_t Element = cast<ConstantInt>(COp)->getZExtValue();
      // If the high bit (7) of the byte is set, the element is zeroed.
      if (Element & (1 << 7))
        ShuffleMask.push_back(SM_SentinelZero);
      else {
        // Only the least significant 4 bits of the byte are used.
        int Index = Base + (Element & 0xf);
        ShuffleMask.push_back(Index);
      }
    }
  }
  // TODO: Handle funny-looking vectors too.
}

void DecodePSHUFBMask(ArrayRef<uint64_t> RawMask,
                      SmallVectorImpl<int> &ShuffleMask) {
  for (int i = 0, e = RawMask.size(); i < e; ++i) {
    uint64_t M = RawMask[i];
    if (M == (uint64_t)SM_SentinelUndef) {
      ShuffleMask.push_back(M);
      continue;
    }
    // For AVX vectors with 32 bytes the base of the shuffle is the half of
    // the vector we're inside.
    int Base = i < 16 ? 0 : 16;
    // If the high bit (7) of the byte is set, the element is zeroed.
    if (M & (1 << 7))
      ShuffleMask.push_back(SM_SentinelZero);
    else {
      // Only the least significant 4 bits of the byte are used.
      int Index = Base + (M & 0xf);
      ShuffleMask.push_back(Index);
    }
  }
}

void DecodeBLENDMask(MVT VT, unsigned Imm, SmallVectorImpl<int> &ShuffleMask) {
  int ElementBits = VT.getScalarSizeInBits();
  int NumElements = VT.getVectorNumElements();
  for (int i = 0; i < NumElements; ++i) {
    // If there are more than 8 elements in the vector, then any immediate blend
    // mask applies to each 128-bit lane. There can never be more than
    // 8 elements in a 128-bit lane with an immediate blend.
    int Bit = NumElements > 8 ? i % (128 / ElementBits) : i;
    assert(Bit < 8 &&
           "Immediate blends only operate over 8 elements at a time!");
    ShuffleMask.push_back(((Imm >> Bit) & 1) ? NumElements + i : i);
  }
}

/// DecodeVPERMMask - this decodes the shuffle masks for VPERMQ/VPERMPD.
/// No VT provided since it only works on 256-bit, 4 element vectors.
void DecodeVPERMMask(unsigned Imm, SmallVectorImpl<int> &ShuffleMask) {
  for (unsigned i = 0; i != 4; ++i) {
    ShuffleMask.push_back((Imm >> (2 * i)) & 3);
  }
}

void DecodeVPERMILPMask(const Constant *C, SmallVectorImpl<int> &ShuffleMask) {
  Type *MaskTy = C->getType();
  assert(MaskTy->isVectorTy() && "Expected a vector constant mask!");
  assert(MaskTy->getVectorElementType()->isIntegerTy() &&
         "Expected integer constant mask elements!");
  int ElementBits = MaskTy->getScalarSizeInBits();
  int NumElements = MaskTy->getVectorNumElements();
  assert((NumElements == 2 || NumElements == 4 || NumElements == 8) &&
         "Unexpected number of vector elements.");
  ShuffleMask.reserve(NumElements);
  if (auto *CDS = dyn_cast<ConstantDataSequential>(C)) {
    assert((unsigned)NumElements == CDS->getNumElements() &&
           "Constant mask has a different number of elements!");

    for (int i = 0; i < NumElements; ++i) {
      int Base = (i * ElementBits / 128) * (128 / ElementBits);
      uint64_t Element = CDS->getElementAsInteger(i);
      // Only the least significant 2 bits of the integer are used.
      int Index = Base + (Element & 0x3);
      ShuffleMask.push_back(Index);
    }
  } else if (auto *CV = dyn_cast<ConstantVector>(C)) {
    assert((unsigned)NumElements == C->getNumOperands() &&
           "Constant mask has a different number of elements!");

    for (int i = 0; i < NumElements; ++i) {
      int Base = (i * ElementBits / 128) * (128 / ElementBits);
      Constant *COp = CV->getOperand(i);
      if (isa<UndefValue>(COp)) {
        ShuffleMask.push_back(SM_SentinelUndef);
        continue;
      }
      uint64_t Element = cast<ConstantInt>(COp)->getZExtValue();
      // Only the least significant 2 bits of the integer are used.
      int Index = Base + (Element & 0x3);
      ShuffleMask.push_back(Index);
    }
  }
}

void DecodeZeroExtendMask(MVT SrcVT, MVT DstVT, SmallVectorImpl<int> &Mask) {
  unsigned NumDstElts = DstVT.getVectorNumElements();
  unsigned SrcScalarBits = SrcVT.getScalarSizeInBits();
  unsigned DstScalarBits = DstVT.getScalarSizeInBits();
  unsigned Scale = DstScalarBits / SrcScalarBits;
  assert(SrcScalarBits < DstScalarBits &&
         "Expected zero extension mask to increase scalar size");
  assert(SrcVT.getVectorNumElements() >= NumDstElts &&
         "Too many zero extension lanes");

  for (unsigned i = 0; i != NumDstElts; i++) {
    Mask.push_back(i);
    for (unsigned j = 1; j != Scale; j++)
      Mask.push_back(SM_SentinelZero);
  }
}

void DecodeZeroMoveLowMask(MVT VT, SmallVectorImpl<int> &ShuffleMask) {
  unsigned NumElts = VT.getVectorNumElements();
  ShuffleMask.push_back(0);
  for (unsigned i = 1; i < NumElts; i++)
    ShuffleMask.push_back(SM_SentinelZero);
}

void DecodeScalarMoveMask(MVT VT, bool IsLoad, SmallVectorImpl<int> &Mask) {
  // First element comes from the first element of second source.
  // Remaining elements: Load zero extends / Move copies from first source.
  unsigned NumElts = VT.getVectorNumElements();
  Mask.push_back(NumElts);
  for (unsigned i = 1; i < NumElts; i++)
    Mask.push_back(IsLoad ? static_cast<int>(SM_SentinelZero) : i);
}
} // llvm namespace
