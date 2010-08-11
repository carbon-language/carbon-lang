//===-- llvm/Target/TargetData.h - Data size & alignment info ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines target properties related to datatype size/offset/alignment
// information.  It uses lazy annotations to cache information about how
// structure types are laid out and used.
//
// This structure should be created once, filled in if the defaults are not
// correct and then passed around by const&.  None of the members functions
// require modification to the object.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETDATA_H
#define LLVM_TARGET_TARGETDATA_H

#include "llvm/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

class Value;
class Type;
class IntegerType;
class StructType;
class StructLayout;
class GlobalVariable;
class LLVMContext;

/// Enum used to categorize the alignment types stored by TargetAlignElem
enum AlignTypeEnum {
  INTEGER_ALIGN = 'i',               ///< Integer type alignment
  VECTOR_ALIGN = 'v',                ///< Vector type alignment
  FLOAT_ALIGN = 'f',                 ///< Floating point type alignment
  AGGREGATE_ALIGN = 'a',             ///< Aggregate alignment
  STACK_ALIGN = 's'                  ///< Stack objects alignment
};
/// Target alignment element.
///
/// Stores the alignment data associated with a given alignment type (pointer,
/// integer, vector, float) and type bit width.
///
/// @note The unusual order of elements in the structure attempts to reduce
/// padding and make the structure slightly more cache friendly.
struct TargetAlignElem {
  AlignTypeEnum       AlignType : 8;  //< Alignment type (AlignTypeEnum)
  unsigned            ABIAlign;       //< ABI alignment for this type/bitw
  unsigned            PrefAlign;      //< Pref. alignment for this type/bitw
  uint32_t            TypeBitWidth;   //< Type bit width

  /// Initializer
  static TargetAlignElem get(AlignTypeEnum align_type, unsigned abi_align,
                             unsigned pref_align, uint32_t bit_width);
  /// Equality predicate
  bool operator==(const TargetAlignElem &rhs) const;
};

class TargetData : public ImmutablePass {
private:
  bool          LittleEndian;          ///< Defaults to false
  unsigned      PointerMemSize;        ///< Pointer size in bytes
  unsigned      PointerABIAlign;       ///< Pointer ABI alignment
  unsigned      PointerPrefAlign;      ///< Pointer preferred alignment

  SmallVector<unsigned char, 8> LegalIntWidths; ///< Legal Integers.
  
  /// Alignments- Where the primitive type alignment data is stored.
  ///
  /// @sa init().
  /// @note Could support multiple size pointer alignments, e.g., 32-bit
  /// pointers vs. 64-bit pointers by extending TargetAlignment, but for now,
  /// we don't.
  SmallVector<TargetAlignElem, 16> Alignments;
  
  /// InvalidAlignmentElem - This member is a signal that a requested alignment
  /// type and bit width were not found in the SmallVector.
  static const TargetAlignElem InvalidAlignmentElem;

  // The StructType -> StructLayout map.
  mutable void *LayoutMap;

  //! Set/initialize target alignments
  void setAlignment(AlignTypeEnum align_type, unsigned abi_align,
                    unsigned pref_align, uint32_t bit_width);
  unsigned getAlignmentInfo(AlignTypeEnum align_type, uint32_t bit_width,
                            bool ABIAlign, const Type *Ty) const;
  //! Internal helper method that returns requested alignment for type.
  unsigned getAlignment(const Type *Ty, bool abi_or_pref) const;

  /// Valid alignment predicate.
  ///
  /// Predicate that tests a TargetAlignElem reference returned by get() against
  /// InvalidAlignmentElem.
  bool validAlignment(const TargetAlignElem &align) const {
    return &align != &InvalidAlignmentElem;
  }

public:
  /// Default ctor.
  ///
  /// @note This has to exist, because this is a pass, but it should never be
  /// used.
  TargetData();
  
  /// Constructs a TargetData from a specification string. See init().
  explicit TargetData(StringRef TargetDescription)
    : ImmutablePass(ID) {
    init(TargetDescription);
  }

  /// Initialize target data from properties stored in the module.
  explicit TargetData(const Module *M);

  TargetData(const TargetData &TD) :
    ImmutablePass(ID),
    LittleEndian(TD.isLittleEndian()),
    PointerMemSize(TD.PointerMemSize),
    PointerABIAlign(TD.PointerABIAlign),
    PointerPrefAlign(TD.PointerPrefAlign),
    LegalIntWidths(TD.LegalIntWidths),
    Alignments(TD.Alignments),
    LayoutMap(0)
  { }

  ~TargetData();  // Not virtual, do not subclass this class

  //! Parse a target data layout string and initialize TargetData alignments.
  void init(StringRef TargetDescription);

  /// Target endianness...
  bool isLittleEndian() const { return LittleEndian; }
  bool isBigEndian() const { return !LittleEndian; }

  /// getStringRepresentation - Return the string representation of the
  /// TargetData.  This representation is in the same format accepted by the
  /// string constructor above.
  std::string getStringRepresentation() const;
  
  /// isLegalInteger - This function returns true if the specified type is
  /// known tobe a native integer type supported by the CPU.  For example,
  /// i64 is not native on most 32-bit CPUs and i37 is not native on any known
  /// one.  This returns false if the integer width is not legal.
  ///
  /// The width is specified in bits.
  ///
  bool isLegalInteger(unsigned Width) const {
    for (unsigned i = 0, e = (unsigned)LegalIntWidths.size(); i != e; ++i)
      if (LegalIntWidths[i] == Width)
        return true;
    return false;
  }
  
  bool isIllegalInteger(unsigned Width) const {
    return !isLegalInteger(Width);
  }
  
  /// Target pointer alignment
  unsigned getPointerABIAlignment() const { return PointerABIAlign; }
  /// Return target's alignment for stack-based pointers
  unsigned getPointerPrefAlignment() const { return PointerPrefAlign; }
  /// Target pointer size
  unsigned getPointerSize()         const { return PointerMemSize; }
  /// Target pointer size, in bits
  unsigned getPointerSizeInBits()   const { return 8*PointerMemSize; }

  /// Size examples:
  ///
  /// Type        SizeInBits  StoreSizeInBits  AllocSizeInBits[*]
  /// ----        ----------  ---------------  ---------------
  ///  i1            1           8                8
  ///  i8            8           8                8
  ///  i19          19          24               32
  ///  i32          32          32               32
  ///  i100        100         104              128
  ///  i128        128         128              128
  ///  Float        32          32               32
  ///  Double       64          64               64
  ///  X86_FP80     80          80               96
  ///
  /// [*] The alloc size depends on the alignment, and thus on the target.
  ///     These values are for x86-32 linux.

  /// getTypeSizeInBits - Return the number of bits necessary to hold the
  /// specified type.  For example, returns 36 for i36 and 80 for x86_fp80.
  uint64_t getTypeSizeInBits(const Type* Ty) const;

  /// getTypeStoreSize - Return the maximum number of bytes that may be
  /// overwritten by storing the specified type.  For example, returns 5
  /// for i36 and 10 for x86_fp80.
  uint64_t getTypeStoreSize(const Type *Ty) const {
    return (getTypeSizeInBits(Ty)+7)/8;
  }

  /// getTypeStoreSizeInBits - Return the maximum number of bits that may be
  /// overwritten by storing the specified type; always a multiple of 8.  For
  /// example, returns 40 for i36 and 80 for x86_fp80.
  uint64_t getTypeStoreSizeInBits(const Type *Ty) const {
    return 8*getTypeStoreSize(Ty);
  }

  /// getTypeAllocSize - Return the offset in bytes between successive objects
  /// of the specified type, including alignment padding.  This is the amount
  /// that alloca reserves for this type.  For example, returns 12 or 16 for
  /// x86_fp80, depending on alignment.
  uint64_t getTypeAllocSize(const Type* Ty) const {
    // Round up to the next alignment boundary.
    return RoundUpAlignment(getTypeStoreSize(Ty), getABITypeAlignment(Ty));
  }

  /// getTypeAllocSizeInBits - Return the offset in bits between successive
  /// objects of the specified type, including alignment padding; always a
  /// multiple of 8.  This is the amount that alloca reserves for this type.
  /// For example, returns 96 or 128 for x86_fp80, depending on alignment.
  uint64_t getTypeAllocSizeInBits(const Type* Ty) const {
    return 8*getTypeAllocSize(Ty);
  }

  /// getABITypeAlignment - Return the minimum ABI-required alignment for the
  /// specified type.
  unsigned getABITypeAlignment(const Type *Ty) const;
  
  /// getABIIntegerTypeAlignment - Return the minimum ABI-required alignment for
  /// an integer type of the specified bitwidth.
  unsigned getABIIntegerTypeAlignment(unsigned BitWidth) const;
  

  /// getCallFrameTypeAlignment - Return the minimum ABI-required alignment
  /// for the specified type when it is part of a call frame.
  unsigned getCallFrameTypeAlignment(const Type *Ty) const;


  /// getPrefTypeAlignment - Return the preferred stack/global alignment for
  /// the specified type.  This is always at least as good as the ABI alignment.
  unsigned getPrefTypeAlignment(const Type *Ty) const;

  /// getPreferredTypeAlignmentShift - Return the preferred alignment for the
  /// specified type, returned as log2 of the value (a shift amount).
  ///
  unsigned getPreferredTypeAlignmentShift(const Type *Ty) const;

  /// getIntPtrType - Return an unsigned integer type that is the same size or
  /// greater to the host pointer size.
  ///
  const IntegerType *getIntPtrType(LLVMContext &C) const;

  /// getIndexedOffset - return the offset from the beginning of the type for
  /// the specified indices.  This is used to implement getelementptr.
  ///
  uint64_t getIndexedOffset(const Type *Ty,
                            Value* const* Indices, unsigned NumIndices) const;

  /// getStructLayout - Return a StructLayout object, indicating the alignment
  /// of the struct, its size, and the offsets of its fields.  Note that this
  /// information is lazily cached.
  const StructLayout *getStructLayout(const StructType *Ty) const;

  /// InvalidateStructLayoutInfo - TargetData speculatively caches StructLayout
  /// objects.  If a TargetData object is alive when types are being refined and
  /// removed, this method must be called whenever a StructType is removed to
  /// avoid a dangling pointer in this cache.
  void InvalidateStructLayoutInfo(const StructType *Ty) const;

  /// getPreferredAlignment - Return the preferred alignment of the specified
  /// global.  This includes an explicitly requested alignment (if the global
  /// has one).
  unsigned getPreferredAlignment(const GlobalVariable *GV) const;

  /// getPreferredAlignmentLog - Return the preferred alignment of the
  /// specified global, returned in log form.  This includes an explicitly
  /// requested alignment (if the global has one).
  unsigned getPreferredAlignmentLog(const GlobalVariable *GV) const;

  /// RoundUpAlignment - Round the specified value up to the next alignment
  /// boundary specified by Alignment.  For example, 7 rounded up to an
  /// alignment boundary of 4 is 8.  8 rounded up to the alignment boundary of 4
  /// is 8 because it is already aligned.
  template <typename UIntTy>
  static UIntTy RoundUpAlignment(UIntTy Val, unsigned Alignment) {
    assert((Alignment & (Alignment-1)) == 0 && "Alignment must be power of 2!");
    return (Val + (Alignment-1)) & ~UIntTy(Alignment-1);
  }
  
  static char ID; // Pass identification, replacement for typeid
};

/// StructLayout - used to lazily calculate structure layout information for a
/// target machine, based on the TargetData structure.
///
class StructLayout {
  uint64_t StructSize;
  unsigned StructAlignment;
  unsigned NumElements;
  uint64_t MemberOffsets[1];  // variable sized array!
public:

  uint64_t getSizeInBytes() const {
    return StructSize;
  }

  uint64_t getSizeInBits() const {
    return 8*StructSize;
  }

  unsigned getAlignment() const {
    return StructAlignment;
  }

  /// getElementContainingOffset - Given a valid byte offset into the structure,
  /// return the structure index that contains it.
  ///
  unsigned getElementContainingOffset(uint64_t Offset) const;

  uint64_t getElementOffset(unsigned Idx) const {
    assert(Idx < NumElements && "Invalid element idx!");
    return MemberOffsets[Idx];
  }

  uint64_t getElementOffsetInBits(unsigned Idx) const {
    return getElementOffset(Idx)*8;
  }

private:
  friend class TargetData;   // Only TargetData can create this class
  StructLayout(const StructType *ST, const TargetData &TD);
};

} // End llvm namespace

#endif
