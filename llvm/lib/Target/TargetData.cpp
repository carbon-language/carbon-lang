//===-- TargetData.cpp - Data size & alignment routines --------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines target properties related to datatype size/offset/alignment
// information.
//
// This structure should be created once, filled in if the defaults are not
// correct and then passed around by const&.  None of the members functions
// require modification to the object.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetData.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Mutex.h"
#include "llvm/ADT/DenseMap.h"
#include <algorithm>
#include <cstdlib>
using namespace llvm;

// Handle the Pass registration stuff necessary to use TargetData's.

// Register the default SparcV9 implementation...
static RegisterPass<TargetData> X("targetdata", "Target Data Layout", false, 
                                  true);
char TargetData::ID = 0;

//===----------------------------------------------------------------------===//
// Support for StructLayout
//===----------------------------------------------------------------------===//

StructLayout::StructLayout(const StructType *ST, const TargetData &TD) {
  StructAlignment = 0;
  StructSize = 0;
  NumElements = ST->getNumElements();

  // Loop over each of the elements, placing them in memory.
  for (unsigned i = 0, e = NumElements; i != e; ++i) {
    const Type *Ty = ST->getElementType(i);
    unsigned TyAlign = ST->isPacked() ? 1 : TD.getABITypeAlignment(Ty);

    // Add padding if necessary to align the data element properly.
    if ((StructSize & (TyAlign-1)) != 0)
      StructSize = TargetData::RoundUpAlignment(StructSize, TyAlign);

    // Keep track of maximum alignment constraint.
    StructAlignment = std::max(TyAlign, StructAlignment);

    MemberOffsets[i] = StructSize;
    StructSize += TD.getTypeAllocSize(Ty); // Consume space for this data item
  }

  // Empty structures have alignment of 1 byte.
  if (StructAlignment == 0) StructAlignment = 1;

  // Add padding to the end of the struct so that it could be put in an array
  // and all array elements would be aligned correctly.
  if ((StructSize & (StructAlignment-1)) != 0)
    StructSize = TargetData::RoundUpAlignment(StructSize, StructAlignment);
}


/// getElementContainingOffset - Given a valid offset into the structure,
/// return the structure index that contains it.
unsigned StructLayout::getElementContainingOffset(uint64_t Offset) const {
  const uint64_t *SI =
    std::upper_bound(&MemberOffsets[0], &MemberOffsets[NumElements], Offset);
  assert(SI != &MemberOffsets[0] && "Offset not in structure type!");
  --SI;
  assert(*SI <= Offset && "upper_bound didn't work");
  assert((SI == &MemberOffsets[0] || *(SI-1) <= Offset) &&
         (SI+1 == &MemberOffsets[NumElements] || *(SI+1) > Offset) &&
         "Upper bound didn't work!");
  
  // Multiple fields can have the same offset if any of them are zero sized.
  // For example, in { i32, [0 x i32], i32 }, searching for offset 4 will stop
  // at the i32 element, because it is the last element at that offset.  This is
  // the right one to return, because anything after it will have a higher
  // offset, implying that this element is non-empty.
  return SI-&MemberOffsets[0];
}

//===----------------------------------------------------------------------===//
// TargetAlignElem, TargetAlign support
//===----------------------------------------------------------------------===//

TargetAlignElem
TargetAlignElem::get(AlignTypeEnum align_type, unsigned char abi_align,
                     unsigned char pref_align, uint32_t bit_width) {
  assert(abi_align <= pref_align && "Preferred alignment worse than ABI!");
  TargetAlignElem retval;
  retval.AlignType = align_type;
  retval.ABIAlign = abi_align;
  retval.PrefAlign = pref_align;
  retval.TypeBitWidth = bit_width;
  return retval;
}

bool
TargetAlignElem::operator==(const TargetAlignElem &rhs) const {
  return (AlignType == rhs.AlignType
          && ABIAlign == rhs.ABIAlign
          && PrefAlign == rhs.PrefAlign
          && TypeBitWidth == rhs.TypeBitWidth);
}

const TargetAlignElem TargetData::InvalidAlignmentElem =
                TargetAlignElem::get((AlignTypeEnum) -1, 0, 0, 0);

//===----------------------------------------------------------------------===//
//                       TargetData Class Implementation
//===----------------------------------------------------------------------===//

/// getInt - Get an integer ignoring errors.
static unsigned getInt(StringRef R) {
  unsigned Result = 0;
  R.getAsInteger(10, Result);
  return Result;
}

void TargetData::init(StringRef Desc) {
  LayoutMap = 0;
  LittleEndian = false;
  PointerMemSize = 8;
  PointerABIAlign = 8;
  PointerPrefAlign = PointerABIAlign;

  // Default alignments
  setAlignment(INTEGER_ALIGN,   1,  1, 1);   // i1
  setAlignment(INTEGER_ALIGN,   1,  1, 8);   // i8
  setAlignment(INTEGER_ALIGN,   2,  2, 16);  // i16
  setAlignment(INTEGER_ALIGN,   4,  4, 32);  // i32
  setAlignment(INTEGER_ALIGN,   4,  8, 64);  // i64
  setAlignment(FLOAT_ALIGN,     4,  4, 32);  // float
  setAlignment(FLOAT_ALIGN,     8,  8, 64);  // double
  setAlignment(VECTOR_ALIGN,    8,  8, 64);  // v2i32, v1i64, ...
  setAlignment(VECTOR_ALIGN,   16, 16, 128); // v16i8, v8i16, v4i32, ...
  setAlignment(AGGREGATE_ALIGN, 0,  8,  0);  // struct

  while (!Desc.empty()) {
    std::pair<StringRef, StringRef> Split = Desc.split('-');
    StringRef Token = Split.first;
    Desc = Split.second;
    
    if (Token.empty())
      continue;
    
    Split = Token.split(':');
    StringRef Specifier = Split.first;
    Token = Split.second;
    
    assert(!Specifier.empty() && "Can't be empty here");
    
    switch (Specifier[0]) {
    case 'E':
      LittleEndian = false;
      break;
    case 'e':
      LittleEndian = true;
      break;
    case 'p':
      Split = Token.split(':');
      PointerMemSize = getInt(Split.first) / 8;
      Split = Split.second.split(':');
      PointerABIAlign = getInt(Split.first) / 8;
      Split = Split.second.split(':');
      PointerPrefAlign = getInt(Split.first) / 8;
      if (PointerPrefAlign == 0)
        PointerPrefAlign = PointerABIAlign;
      break;
    case 'i':
    case 'v':
    case 'f':
    case 'a':
    case 's': {
      AlignTypeEnum AlignType;
      switch (Specifier[0]) {
      default:
      case 'i': AlignType = INTEGER_ALIGN; break;
      case 'v': AlignType = VECTOR_ALIGN; break;
      case 'f': AlignType = FLOAT_ALIGN; break;
      case 'a': AlignType = AGGREGATE_ALIGN; break;
      case 's': AlignType = STACK_ALIGN; break;
      }
      unsigned Size = getInt(Specifier.substr(1));
      Split = Token.split(':');
      unsigned char ABIAlign = getInt(Split.first) / 8;
      
      Split = Split.second.split(':');
      unsigned char PrefAlign = getInt(Split.first) / 8;
      if (PrefAlign == 0)
        PrefAlign = ABIAlign;
      setAlignment(AlignType, ABIAlign, PrefAlign, Size);
      break;
    }
    case 'n':  // Native integer types.
      Specifier = Specifier.substr(1);
      do {
        if (unsigned Width = getInt(Specifier))
          LegalIntWidths.push_back(Width);
        Split = Token.split(':');
        Specifier = Split.first;
        Token = Split.second;
      } while (!Specifier.empty() || !Token.empty());
      break;
        
    default:
      break;
    }
  }
}

/// Default ctor.
///
/// @note This has to exist, because this is a pass, but it should never be
/// used.
TargetData::TargetData() : ImmutablePass(&ID) {
  llvm_report_error("Bad TargetData ctor used.  "
                    "Tool did not specify a TargetData to use?");
}

TargetData::TargetData(const Module *M) 
  : ImmutablePass(&ID) {
  init(M->getDataLayout());
}

void
TargetData::setAlignment(AlignTypeEnum align_type, unsigned char abi_align,
                         unsigned char pref_align, uint32_t bit_width) {
  assert(abi_align <= pref_align && "Preferred alignment worse than ABI!");
  for (unsigned i = 0, e = Alignments.size(); i != e; ++i) {
    if (Alignments[i].AlignType == align_type &&
        Alignments[i].TypeBitWidth == bit_width) {
      // Update the abi, preferred alignments.
      Alignments[i].ABIAlign = abi_align;
      Alignments[i].PrefAlign = pref_align;
      return;
    }
  }
  
  Alignments.push_back(TargetAlignElem::get(align_type, abi_align,
                                            pref_align, bit_width));
}

/// getAlignmentInfo - Return the alignment (either ABI if ABIInfo = true or 
/// preferred if ABIInfo = false) the target wants for the specified datatype.
unsigned TargetData::getAlignmentInfo(AlignTypeEnum AlignType, 
                                      uint32_t BitWidth, bool ABIInfo,
                                      const Type *Ty) const {
  // Check to see if we have an exact match and remember the best match we see.
  int BestMatchIdx = -1;
  int LargestInt = -1;
  for (unsigned i = 0, e = Alignments.size(); i != e; ++i) {
    if (Alignments[i].AlignType == AlignType &&
        Alignments[i].TypeBitWidth == BitWidth)
      return ABIInfo ? Alignments[i].ABIAlign : Alignments[i].PrefAlign;
    
    // The best match so far depends on what we're looking for.
    if (AlignType == VECTOR_ALIGN && Alignments[i].AlignType == VECTOR_ALIGN) {
      // If this is a specification for a smaller vector type, we will fall back
      // to it.  This happens because <128 x double> can be implemented in terms
      // of 64 <2 x double>.
      if (Alignments[i].TypeBitWidth < BitWidth) {
        // Verify that we pick the biggest of the fallbacks.
        if (BestMatchIdx == -1 ||
            Alignments[BestMatchIdx].TypeBitWidth < Alignments[i].TypeBitWidth)
          BestMatchIdx = i;
      }
    } else if (AlignType == INTEGER_ALIGN && 
               Alignments[i].AlignType == INTEGER_ALIGN) {
      // The "best match" for integers is the smallest size that is larger than
      // the BitWidth requested.
      if (Alignments[i].TypeBitWidth > BitWidth && (BestMatchIdx == -1 || 
           Alignments[i].TypeBitWidth < Alignments[BestMatchIdx].TypeBitWidth))
        BestMatchIdx = i;
      // However, if there isn't one that's larger, then we must use the
      // largest one we have (see below)
      if (LargestInt == -1 || 
          Alignments[i].TypeBitWidth > Alignments[LargestInt].TypeBitWidth)
        LargestInt = i;
    }
  }

  // Okay, we didn't find an exact solution.  Fall back here depending on what
  // is being looked for.
  if (BestMatchIdx == -1) {
    // If we didn't find an integer alignment, fall back on most conservative.
    if (AlignType == INTEGER_ALIGN) {
      BestMatchIdx = LargestInt;
    } else {
      assert(AlignType == VECTOR_ALIGN && "Unknown alignment type!");

      // If we didn't find a vector size that is smaller or equal to this type,
      // then we will end up scalarizing this to its element type.  Just return
      // the alignment of the element.
      return getAlignment(cast<VectorType>(Ty)->getElementType(), ABIInfo);
    }
  }

  // Since we got a "best match" index, just return it.
  return ABIInfo ? Alignments[BestMatchIdx].ABIAlign
                 : Alignments[BestMatchIdx].PrefAlign;
}

namespace {

class StructLayoutMap : public AbstractTypeUser {
  typedef DenseMap<const StructType*, StructLayout*> LayoutInfoTy;
  LayoutInfoTy LayoutInfo;

  void RemoveEntry(LayoutInfoTy::iterator I, bool WasAbstract) {
    I->second->~StructLayout();
    free(I->second);
    if (WasAbstract)
      I->first->removeAbstractTypeUser(this);
    LayoutInfo.erase(I);
  }
  
  
  /// refineAbstractType - The callback method invoked when an abstract type is
  /// resolved to another type.  An object must override this method to update
  /// its internal state to reference NewType instead of OldType.
  ///
  virtual void refineAbstractType(const DerivedType *OldTy,
                                  const Type *) {
    LayoutInfoTy::iterator I = LayoutInfo.find(cast<const StructType>(OldTy));
    assert(I != LayoutInfo.end() && "Using type but not in map?");
    RemoveEntry(I, true);
  }

  /// typeBecameConcrete - The other case which AbstractTypeUsers must be aware
  /// of is when a type makes the transition from being abstract (where it has
  /// clients on its AbstractTypeUsers list) to concrete (where it does not).
  /// This method notifies ATU's when this occurs for a type.
  ///
  virtual void typeBecameConcrete(const DerivedType *AbsTy) {
    LayoutInfoTy::iterator I = LayoutInfo.find(cast<const StructType>(AbsTy));
    assert(I != LayoutInfo.end() && "Using type but not in map?");
    RemoveEntry(I, true);
  }

public:
  virtual ~StructLayoutMap() {
    // Remove any layouts.
    for (LayoutInfoTy::iterator
           I = LayoutInfo.begin(), E = LayoutInfo.end(); I != E; ++I) {
      const Type *Key = I->first;
      StructLayout *Value = I->second;

      if (Key->isAbstract())
        Key->removeAbstractTypeUser(this);

      Value->~StructLayout();
      free(Value);
    }
  }

  void InvalidateEntry(const StructType *Ty) {
    LayoutInfoTy::iterator I = LayoutInfo.find(Ty);
    if (I == LayoutInfo.end()) return;
    RemoveEntry(I, Ty->isAbstract());
  }

  StructLayout *&operator[](const StructType *STy) {
    return LayoutInfo[STy];
  }

  // for debugging...
  virtual void dump() const {}
};

} // end anonymous namespace

TargetData::~TargetData() {
  delete static_cast<StructLayoutMap*>(LayoutMap);
}

const StructLayout *TargetData::getStructLayout(const StructType *Ty) const {
  if (!LayoutMap)
    LayoutMap = new StructLayoutMap();
  
  StructLayoutMap *STM = static_cast<StructLayoutMap*>(LayoutMap);
  StructLayout *&SL = (*STM)[Ty];
  if (SL) return SL;

  // Otherwise, create the struct layout.  Because it is variable length, we 
  // malloc it, then use placement new.
  int NumElts = Ty->getNumElements();
  StructLayout *L =
    (StructLayout *)malloc(sizeof(StructLayout)+(NumElts-1) * sizeof(uint64_t));
  
  // Set SL before calling StructLayout's ctor.  The ctor could cause other
  // entries to be added to TheMap, invalidating our reference.
  SL = L;
  
  new (L) StructLayout(Ty, *this);

  if (Ty->isAbstract())
    Ty->addAbstractTypeUser(STM);

  return L;
}

/// InvalidateStructLayoutInfo - TargetData speculatively caches StructLayout
/// objects.  If a TargetData object is alive when types are being refined and
/// removed, this method must be called whenever a StructType is removed to
/// avoid a dangling pointer in this cache.
void TargetData::InvalidateStructLayoutInfo(const StructType *Ty) const {
  if (!LayoutMap) return;  // No cache.
  
  static_cast<StructLayoutMap*>(LayoutMap)->InvalidateEntry(Ty);
}

std::string TargetData::getStringRepresentation() const {
  std::string Result;
  raw_string_ostream OS(Result);
  
  OS << (LittleEndian ? "e" : "E")
     << "-p:" << PointerMemSize*8 << ':' << PointerABIAlign*8
     << ':' << PointerPrefAlign*8;
  for (unsigned i = 0, e = Alignments.size(); i != e; ++i) {
    const TargetAlignElem &AI = Alignments[i];
    OS << '-' << (char)AI.AlignType << AI.TypeBitWidth << ':'
       << AI.ABIAlign*8 << ':' << AI.PrefAlign*8;
  }
  
  if (!LegalIntWidths.empty()) {
    OS << "-n" << (unsigned)LegalIntWidths[0];
    
    for (unsigned i = 1, e = LegalIntWidths.size(); i != e; ++i)
      OS << ':' << (unsigned)LegalIntWidths[i];
  }
  return OS.str();
}


uint64_t TargetData::getTypeSizeInBits(const Type *Ty) const {
  assert(Ty->isSized() && "Cannot getTypeInfo() on a type that is unsized!");
  switch (Ty->getTypeID()) {
  case Type::LabelTyID:
  case Type::PointerTyID:
    return getPointerSizeInBits();
  case Type::ArrayTyID: {
    const ArrayType *ATy = cast<ArrayType>(Ty);
    return getTypeSizeInBits(ATy->getElementType())*ATy->getNumElements();
  }
  case Type::StructTyID:
    // Get the layout annotation... which is lazily created on demand.
    return getStructLayout(cast<StructType>(Ty))->getSizeInBits();
  case Type::IntegerTyID:
    return cast<IntegerType>(Ty)->getBitWidth();
  case Type::VoidTyID:
    return 8;
  case Type::FloatTyID:
    return 32;
  case Type::DoubleTyID:
    return 64;
  case Type::PPC_FP128TyID:
  case Type::FP128TyID:
    return 128;
  // In memory objects this is always aligned to a higher boundary, but
  // only 80 bits contain information.
  case Type::X86_FP80TyID:
    return 80;
  case Type::VectorTyID:
    return cast<VectorType>(Ty)->getBitWidth();
  default:
    llvm_unreachable("TargetData::getTypeSizeInBits(): Unsupported type");
    break;
  }
  return 0;
}

/// getTypeStoreSize - Return the maximum number of bytes that may be
/// overwritten by storing the specified type.  For example, returns 5
/// for i36 and 10 for x86_fp80.
uint64_t TargetData::getTypeStoreSize(const Type *Ty) const {
  // Arrays and vectors are allocated as sequences of elements.
  if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    if (ATy->getNumElements() == 0)
      return 0;
    const Type *ElementType = ATy->getElementType();
    return getTypeAllocSize(ElementType) * (ATy->getNumElements() - 1) +
           getTypeStoreSize(ElementType);
  }
  if (const VectorType *VTy = dyn_cast<VectorType>(Ty)) {
    const Type *ElementType = VTy->getElementType();
    return getTypeAllocSize(ElementType) * (VTy->getNumElements() - 1) +
           getTypeStoreSize(ElementType);
  }

  return (getTypeSizeInBits(Ty)+7)/8;
}

/// getTypeAllocSize - Return the offset in bytes between successive objects
/// of the specified type, including alignment padding.  This is the amount
/// that alloca reserves for this type.  For example, returns 12 or 16 for
/// x86_fp80, depending on alignment.
uint64_t TargetData::getTypeAllocSize(const Type* Ty) const {
  // Arrays and vectors are allocated as sequences of elements.
  // Note that this means that things like vectors-of-i1 are not bit-packed
  // in memory (except on a hypothetical bit-addressable machine). If
  // someone builds hardware with native vector-of-i1 stores and the idiom
  // of bitcasting vectors to integers in order to bitpack them for storage
  // isn't sufficient, TargetData may need new "size" concept.
  if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty))
    return getTypeAllocSize(ATy->getElementType()) * ATy->getNumElements();
  if (const VectorType *VTy = dyn_cast<VectorType>(Ty))
    return getTypeAllocSize(VTy->getElementType()) * VTy->getNumElements();

  // Round up to the next alignment boundary.
  return RoundUpAlignment(getTypeStoreSize(Ty), getABITypeAlignment(Ty));
}

/*!
  \param abi_or_pref Flag that determines which alignment is returned. true
  returns the ABI alignment, false returns the preferred alignment.
  \param Ty The underlying type for which alignment is determined.

  Get the ABI (\a abi_or_pref == true) or preferred alignment (\a abi_or_pref
  == false) for the requested type \a Ty.
 */
unsigned char TargetData::getAlignment(const Type *Ty, bool abi_or_pref) const {
  int AlignType = -1;

  assert(Ty->isSized() && "Cannot getTypeInfo() on a type that is unsized!");
  switch (Ty->getTypeID()) {
  // Early escape for the non-numeric types.
  case Type::LabelTyID:
  case Type::PointerTyID:
    return (abi_or_pref
            ? getPointerABIAlignment()
            : getPointerPrefAlignment());
  case Type::ArrayTyID:
    return getAlignment(cast<ArrayType>(Ty)->getElementType(), abi_or_pref);

  case Type::StructTyID: {
    // Packed structure types always have an ABI alignment of one.
    if (cast<StructType>(Ty)->isPacked() && abi_or_pref)
      return 1;

    // Get the layout annotation... which is lazily created on demand.
    const StructLayout *Layout = getStructLayout(cast<StructType>(Ty));
    unsigned Align = getAlignmentInfo(AGGREGATE_ALIGN, 0, abi_or_pref, Ty);
    return std::max(Align, (unsigned)Layout->getAlignment());
  }
  case Type::IntegerTyID:
  case Type::VoidTyID:
    AlignType = INTEGER_ALIGN;
    break;
  case Type::FloatTyID:
  case Type::DoubleTyID:
  // PPC_FP128TyID and FP128TyID have different data contents, but the
  // same size and alignment, so they look the same here.
  case Type::PPC_FP128TyID:
  case Type::FP128TyID:
  case Type::X86_FP80TyID:
    AlignType = FLOAT_ALIGN;
    break;
  case Type::VectorTyID:
    AlignType = VECTOR_ALIGN;
    break;
  default:
    llvm_unreachable("Bad type for getAlignment!!!");
    break;
  }

  return getAlignmentInfo((AlignTypeEnum)AlignType, getTypeSizeInBits(Ty),
                          abi_or_pref, Ty);
}

unsigned char TargetData::getABITypeAlignment(const Type *Ty) const {
  return getAlignment(Ty, true);
}

/// getABIIntegerTypeAlignment - Return the minimum ABI-required alignment for
/// an integer type of the specified bitwidth.
unsigned char TargetData::getABIIntegerTypeAlignment(unsigned BitWidth) const {
  return getAlignmentInfo(INTEGER_ALIGN, BitWidth, true, 0);
}


unsigned char TargetData::getCallFrameTypeAlignment(const Type *Ty) const {
  for (unsigned i = 0, e = Alignments.size(); i != e; ++i)
    if (Alignments[i].AlignType == STACK_ALIGN)
      return Alignments[i].ABIAlign;

  return getABITypeAlignment(Ty);
}

unsigned char TargetData::getPrefTypeAlignment(const Type *Ty) const {
  return getAlignment(Ty, false);
}

unsigned char TargetData::getPreferredTypeAlignmentShift(const Type *Ty) const {
  unsigned Align = (unsigned) getPrefTypeAlignment(Ty);
  assert(!(Align & (Align-1)) && "Alignment is not a power of two!");
  return Log2_32(Align);
}

/// getIntPtrType - Return an unsigned integer type that is the same size or
/// greater to the host pointer size.
const IntegerType *TargetData::getIntPtrType(LLVMContext &C) const {
  return IntegerType::get(C, getPointerSizeInBits());
}


uint64_t TargetData::getIndexedOffset(const Type *ptrTy, Value* const* Indices,
                                      unsigned NumIndices) const {
  const Type *Ty = ptrTy;
  assert(Ty->isPointerTy() && "Illegal argument for getIndexedOffset()");
  uint64_t Result = 0;

  generic_gep_type_iterator<Value* const*>
    TI = gep_type_begin(ptrTy, Indices, Indices+NumIndices);
  for (unsigned CurIDX = 0; CurIDX != NumIndices; ++CurIDX, ++TI) {
    if (const StructType *STy = dyn_cast<StructType>(*TI)) {
      assert(Indices[CurIDX]->getType() ==
             Type::getInt32Ty(ptrTy->getContext()) &&
             "Illegal struct idx");
      unsigned FieldNo = cast<ConstantInt>(Indices[CurIDX])->getZExtValue();

      // Get structure layout information...
      const StructLayout *Layout = getStructLayout(STy);

      // Add in the offset, as calculated by the structure layout info...
      Result += Layout->getElementOffset(FieldNo);

      // Update Ty to refer to current element
      Ty = STy->getElementType(FieldNo);
    } else {
      // Update Ty to refer to current element
      Ty = cast<SequentialType>(Ty)->getElementType();

      // Get the array index and the size of each array element.
      int64_t arrayIdx = cast<ConstantInt>(Indices[CurIDX])->getSExtValue();
      Result += arrayIdx * (int64_t)getTypeAllocSize(Ty);
    }
  }

  return Result;
}

/// getPreferredAlignment - Return the preferred alignment of the specified
/// global.  This includes an explicitly requested alignment (if the global
/// has one).
unsigned TargetData::getPreferredAlignment(const GlobalVariable *GV) const {
  const Type *ElemType = GV->getType()->getElementType();
  unsigned Alignment = getPrefTypeAlignment(ElemType);
  if (GV->getAlignment() > Alignment)
    Alignment = GV->getAlignment();

  if (GV->hasInitializer()) {
    if (Alignment < 16) {
      // If the global is not external, see if it is large.  If so, give it a
      // larger alignment.
      if (getTypeSizeInBits(ElemType) > 128)
        Alignment = 16;    // 16-byte alignment.
    }
  }
  return Alignment;
}

/// getPreferredAlignmentLog - Return the preferred alignment of the
/// specified global, returned in log form.  This includes an explicitly
/// requested alignment (if the global has one).
unsigned TargetData::getPreferredAlignmentLog(const GlobalVariable *GV) const {
  return Log2_32(getPreferredAlignment(GV));
}
