//===--- CGRecordLayoutBuilder.cpp - CGRecordLayout builder  ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Builder implementation for CGRecordLayout objects.
//
//===----------------------------------------------------------------------===//

#include "CGRecordLayout.h"
#include "CGCXXABI.h"
#include "CodeGenTypes.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;
using namespace CodeGen;

namespace {

class CGRecordLayoutBuilder {
public:
  /// FieldTypes - Holds the LLVM types that the struct is created from.
  /// 
  SmallVector<llvm::Type *, 16> FieldTypes;

  /// BaseSubobjectType - Holds the LLVM type for the non-virtual part
  /// of the struct. For example, consider:
  ///
  /// struct A { int i; };
  /// struct B { void *v; };
  /// struct C : virtual A, B { };
  ///
  /// The LLVM type of C will be
  /// %struct.C = type { i32 (...)**, %struct.A, i32, %struct.B }
  ///
  /// And the LLVM type of the non-virtual base struct will be
  /// %struct.C.base = type { i32 (...)**, %struct.A, i32 }
  ///
  /// This only gets initialized if the base subobject type is
  /// different from the complete-object type.
  llvm::StructType *BaseSubobjectType;

  /// FieldInfo - Holds a field and its corresponding LLVM field number.
  llvm::DenseMap<const FieldDecl *, unsigned> Fields;

  /// BitFieldInfo - Holds location and size information about a bit field.
  llvm::DenseMap<const FieldDecl *, CGBitFieldInfo> BitFields;

  llvm::DenseMap<const CXXRecordDecl *, unsigned> NonVirtualBases;
  llvm::DenseMap<const CXXRecordDecl *, unsigned> VirtualBases;

  /// IndirectPrimaryBases - Virtual base classes, direct or indirect, that are
  /// primary base classes for some other direct or indirect base class.
  CXXIndirectPrimaryBaseSet IndirectPrimaryBases;

  /// LaidOutVirtualBases - A set of all laid out virtual bases, used to avoid
  /// avoid laying out virtual bases more than once.
  llvm::SmallPtrSet<const CXXRecordDecl *, 4> LaidOutVirtualBases;
  
  /// IsZeroInitializable - Whether this struct can be C++
  /// zero-initialized with an LLVM zeroinitializer.
  bool IsZeroInitializable;
  bool IsZeroInitializableAsBase;

  /// Packed - Whether the resulting LLVM struct will be packed or not.
  bool Packed;

private:
  CodeGenTypes &Types;

  /// LastLaidOutBaseInfo - Contains the offset and non-virtual size of the
  /// last base laid out. Used so that we can replace the last laid out base
  /// type with an i8 array if needed.
  struct LastLaidOutBaseInfo {
    CharUnits Offset;
    CharUnits NonVirtualSize;

    bool isValid() const { return !NonVirtualSize.isZero(); }
    void invalidate() { NonVirtualSize = CharUnits::Zero(); }
  
  } LastLaidOutBase;

  /// Alignment - Contains the alignment of the RecordDecl.
  CharUnits Alignment;

  /// NextFieldOffset - Holds the next field offset.
  CharUnits NextFieldOffset;

  /// LayoutUnionField - Will layout a field in an union and return the type
  /// that the field will have.
  llvm::Type *LayoutUnionField(const FieldDecl *Field,
                               const ASTRecordLayout &Layout);
  
  /// LayoutUnion - Will layout a union RecordDecl.
  void LayoutUnion(const RecordDecl *D);

  /// Lay out a sequence of contiguous bitfields.
  bool LayoutBitfields(const ASTRecordLayout &Layout,
                       unsigned &FirstFieldNo,
                       RecordDecl::field_iterator &FI,
                       RecordDecl::field_iterator FE);

  /// LayoutFields - try to layout all fields in the record decl.
  /// Returns false if the operation failed because the struct is not packed.
  bool LayoutFields(const RecordDecl *D);

  /// Layout a single base, virtual or non-virtual
  bool LayoutBase(const CXXRecordDecl *base,
                  const CGRecordLayout &baseLayout,
                  CharUnits baseOffset);

  /// LayoutVirtualBase - layout a single virtual base.
  bool LayoutVirtualBase(const CXXRecordDecl *base,
                         CharUnits baseOffset);

  /// LayoutVirtualBases - layout the virtual bases of a record decl.
  bool LayoutVirtualBases(const CXXRecordDecl *RD,
                          const ASTRecordLayout &Layout);

  /// MSLayoutVirtualBases - layout the virtual bases of a record decl,
  /// like MSVC.
  bool MSLayoutVirtualBases(const CXXRecordDecl *RD,
                            const ASTRecordLayout &Layout);
  
  /// LayoutNonVirtualBase - layout a single non-virtual base.
  bool LayoutNonVirtualBase(const CXXRecordDecl *base,
                            CharUnits baseOffset);
  
  /// LayoutNonVirtualBases - layout the virtual bases of a record decl.
  bool LayoutNonVirtualBases(const CXXRecordDecl *RD, 
                             const ASTRecordLayout &Layout);

  /// ComputeNonVirtualBaseType - Compute the non-virtual base field types.
  bool ComputeNonVirtualBaseType(const CXXRecordDecl *RD);
  
  /// LayoutField - layout a single field. Returns false if the operation failed
  /// because the current struct is not packed.
  bool LayoutField(const FieldDecl *D, uint64_t FieldOffset);

  /// LayoutBitField - layout a single bit field.
  void LayoutBitField(const FieldDecl *D, uint64_t FieldOffset);

  /// AppendField - Appends a field with the given offset and type.
  void AppendField(CharUnits fieldOffset, llvm::Type *FieldTy);

  /// AppendPadding - Appends enough padding bytes so that the total
  /// struct size is a multiple of the field alignment.
  void AppendPadding(CharUnits fieldOffset, CharUnits fieldAlignment);

  /// ResizeLastBaseFieldIfNecessary - Fields and bases can be laid out in the
  /// tail padding of a previous base. If this happens, the type of the previous
  /// base needs to be changed to an array of i8. Returns true if the last
  /// laid out base was resized.
  bool ResizeLastBaseFieldIfNecessary(CharUnits offset);

  /// getByteArrayType - Returns a byte array type with the given number of
  /// elements.
  llvm::Type *getByteArrayType(CharUnits NumBytes);
  
  /// AppendBytes - Append a given number of bytes to the record.
  void AppendBytes(CharUnits numBytes);

  /// AppendTailPadding - Append enough tail padding so that the type will have
  /// the passed size.
  void AppendTailPadding(CharUnits RecordSize);

  CharUnits getTypeAlignment(llvm::Type *Ty) const;

  /// getAlignmentAsLLVMStruct - Returns the maximum alignment of all the
  /// LLVM element types.
  CharUnits getAlignmentAsLLVMStruct() const;

  /// CheckZeroInitializable - Check if the given type contains a pointer
  /// to data member.
  void CheckZeroInitializable(QualType T);

public:
  CGRecordLayoutBuilder(CodeGenTypes &Types)
    : BaseSubobjectType(0),
      IsZeroInitializable(true), IsZeroInitializableAsBase(true),
      Packed(false), Types(Types) { }

  /// Layout - Will layout a RecordDecl.
  void Layout(const RecordDecl *D);
};

}

void CGRecordLayoutBuilder::Layout(const RecordDecl *D) {
  const ASTRecordLayout &Layout = Types.getContext().getASTRecordLayout(D);
  Alignment = Layout.getAlignment();
  Packed = D->hasAttr<PackedAttr>() || Layout.getSize() % Alignment != 0;

  if (D->isUnion()) {
    LayoutUnion(D);
    return;
  }

  if (LayoutFields(D))
    return;

  // We weren't able to layout the struct. Try again with a packed struct
  Packed = true;
  LastLaidOutBase.invalidate();
  NextFieldOffset = CharUnits::Zero();
  FieldTypes.clear();
  Fields.clear();
  BitFields.clear();
  NonVirtualBases.clear();
  VirtualBases.clear();

  LayoutFields(D);
}

CGBitFieldInfo CGBitFieldInfo::MakeInfo(CodeGenTypes &Types,
                                        const FieldDecl *FD,
                                        uint64_t Offset, uint64_t Size,
                                        uint64_t StorageSize,
                                        uint64_t StorageAlignment) {
  llvm::Type *Ty = Types.ConvertTypeForMem(FD->getType());
  CharUnits TypeSizeInBytes =
    CharUnits::fromQuantity(Types.getDataLayout().getTypeAllocSize(Ty));
  uint64_t TypeSizeInBits = Types.getContext().toBits(TypeSizeInBytes);

  bool IsSigned = FD->getType()->isSignedIntegerOrEnumerationType();

  if (Size > TypeSizeInBits) {
    // We have a wide bit-field. The extra bits are only used for padding, so
    // if we have a bitfield of type T, with size N:
    //
    // T t : N;
    //
    // We can just assume that it's:
    //
    // T t : sizeof(T);
    //
    Size = TypeSizeInBits;
  }

  // Reverse the bit offsets for big endian machines. Because we represent
  // a bitfield as a single large integer load, we can imagine the bits
  // counting from the most-significant-bit instead of the
  // least-significant-bit.
  if (Types.getDataLayout().isBigEndian()) {
    Offset = StorageSize - (Offset + Size);
  }

  return CGBitFieldInfo(Offset, Size, IsSigned, StorageSize, StorageAlignment);
}

/// \brief Layout the range of bitfields from BFI to BFE as contiguous storage.
bool CGRecordLayoutBuilder::LayoutBitfields(const ASTRecordLayout &Layout,
                                            unsigned &FirstFieldNo,
                                            RecordDecl::field_iterator &FI,
                                            RecordDecl::field_iterator FE) {
  assert(FI != FE);
  uint64_t FirstFieldOffset = Layout.getFieldOffset(FirstFieldNo);
  uint64_t NextFieldOffsetInBits = Types.getContext().toBits(NextFieldOffset);

  unsigned CharAlign = Types.getTarget().getCharAlign();
  assert(FirstFieldOffset % CharAlign == 0 &&
         "First field offset is misaligned");
  CharUnits FirstFieldOffsetInBytes
    = Types.getContext().toCharUnitsFromBits(FirstFieldOffset);

  unsigned StorageAlignment
    = llvm::MinAlign(Alignment.getQuantity(),
                     FirstFieldOffsetInBytes.getQuantity());

  if (FirstFieldOffset < NextFieldOffsetInBits) {
    CharUnits FieldOffsetInCharUnits =
      Types.getContext().toCharUnitsFromBits(FirstFieldOffset);

    // Try to resize the last base field.
    if (!ResizeLastBaseFieldIfNecessary(FieldOffsetInCharUnits))
      llvm_unreachable("We must be able to resize the last base if we need to "
                       "pack bits into it.");

    NextFieldOffsetInBits = Types.getContext().toBits(NextFieldOffset);
    assert(FirstFieldOffset >= NextFieldOffsetInBits);
  }

  // Append padding if necessary.
  AppendPadding(Types.getContext().toCharUnitsFromBits(FirstFieldOffset),
                CharUnits::One());

  // Find the last bitfield in a contiguous run of bitfields.
  RecordDecl::field_iterator BFI = FI;
  unsigned LastFieldNo = FirstFieldNo;
  uint64_t NextContiguousFieldOffset = FirstFieldOffset;
  for (RecordDecl::field_iterator FJ = FI;
       (FJ != FE && (*FJ)->isBitField() &&
        NextContiguousFieldOffset == Layout.getFieldOffset(LastFieldNo) &&
        (*FJ)->getBitWidthValue(Types.getContext()) != 0); FI = FJ++) {
    NextContiguousFieldOffset += (*FJ)->getBitWidthValue(Types.getContext());
    ++LastFieldNo;

    // We must use packed structs for packed fields, and also unnamed bit
    // fields since they don't affect the struct alignment.
    if (!Packed && ((*FJ)->hasAttr<PackedAttr>() || !(*FJ)->getDeclName()))
      return false;
  }
  RecordDecl::field_iterator BFE = llvm::next(FI);
  --LastFieldNo;
  assert(LastFieldNo >= FirstFieldNo && "Empty run of contiguous bitfields");
  FieldDecl *LastFD = *FI;

  // Find the last bitfield's offset, add its size, and round it up to the
  // character alignment to compute the storage required.
  uint64_t LastFieldOffset = Layout.getFieldOffset(LastFieldNo);
  uint64_t LastFieldSize = LastFD->getBitWidthValue(Types.getContext());
  uint64_t TotalBits = (LastFieldOffset + LastFieldSize) - FirstFieldOffset;
  CharUnits StorageBytes = Types.getContext().toCharUnitsFromBits(
    llvm::RoundUpToAlignment(TotalBits, CharAlign));
  uint64_t StorageBits = Types.getContext().toBits(StorageBytes);

  // Grow the storage to encompass any known padding in the layout when doing
  // so will make the storage a power-of-two. There are two cases when we can
  // do this. The first is when we have a subsequent field and can widen up to
  // its offset. The second is when the data size of the AST record layout is
  // past the end of the current storage. The latter is true when there is tail
  // padding on a struct and no members of a super class can be packed into it.
  //
  // Note that we widen the storage as much as possible here to express the
  // maximum latitude the language provides, and rely on the backend to lower
  // these in conjunction with shifts and masks to narrower operations where
  // beneficial.
  uint64_t EndOffset = Types.getContext().toBits(Layout.getDataSize());
  if (BFE != FE)
    // If there are more fields to be laid out, the offset at the end of the
    // bitfield is the offset of the next field in the record.
    EndOffset = Layout.getFieldOffset(LastFieldNo + 1);
  assert(EndOffset >= (FirstFieldOffset + TotalBits) &&
         "End offset is not past the end of the known storage bits.");
  uint64_t SpaceBits = EndOffset - FirstFieldOffset;
  uint64_t LongBits = Types.getTarget().getLongWidth();
  uint64_t WidenedBits = (StorageBits / LongBits) * LongBits +
                         llvm::NextPowerOf2(StorageBits % LongBits - 1);
  assert(WidenedBits >= StorageBits && "Widening shrunk the bits!");
  if (WidenedBits <= SpaceBits) {
    StorageBits = WidenedBits;
    StorageBytes = Types.getContext().toCharUnitsFromBits(StorageBits);
    assert(StorageBits == (uint64_t)Types.getContext().toBits(StorageBytes));
  }

  unsigned FieldIndex = FieldTypes.size();
  AppendBytes(StorageBytes);

  // Now walk the bitfields associating them with this field of storage and
  // building up the bitfield specific info.
  unsigned FieldNo = FirstFieldNo;
  for (; BFI != BFE; ++BFI, ++FieldNo) {
    FieldDecl *FD = *BFI;
    uint64_t FieldOffset = Layout.getFieldOffset(FieldNo) - FirstFieldOffset;
    uint64_t FieldSize = FD->getBitWidthValue(Types.getContext());
    Fields[FD] = FieldIndex;
    BitFields[FD] = CGBitFieldInfo::MakeInfo(Types, FD, FieldOffset, FieldSize,
                                             StorageBits, StorageAlignment);
  }
  FirstFieldNo = LastFieldNo;
  return true;
}

bool CGRecordLayoutBuilder::LayoutField(const FieldDecl *D,
                                        uint64_t fieldOffset) {
  // If the field is packed, then we need a packed struct.
  if (!Packed && D->hasAttr<PackedAttr>())
    return false;

  assert(!D->isBitField() && "Bitfields should be laid out seperately.");

  CheckZeroInitializable(D->getType());

  assert(fieldOffset % Types.getTarget().getCharWidth() == 0
         && "field offset is not on a byte boundary!");
  CharUnits fieldOffsetInBytes
    = Types.getContext().toCharUnitsFromBits(fieldOffset);

  llvm::Type *Ty = Types.ConvertTypeForMem(D->getType());
  CharUnits typeAlignment = getTypeAlignment(Ty);

  // If the type alignment is larger then the struct alignment, we must use
  // a packed struct.
  if (typeAlignment > Alignment) {
    assert(!Packed && "Alignment is wrong even with packed struct!");
    return false;
  }

  if (!Packed) {
    if (const RecordType *RT = D->getType()->getAs<RecordType>()) {
      const RecordDecl *RD = cast<RecordDecl>(RT->getDecl());
      if (const MaxFieldAlignmentAttr *MFAA =
            RD->getAttr<MaxFieldAlignmentAttr>()) {
        if (MFAA->getAlignment() != Types.getContext().toBits(typeAlignment))
          return false;
      }
    }
  }

  // Round up the field offset to the alignment of the field type.
  CharUnits alignedNextFieldOffsetInBytes =
    NextFieldOffset.RoundUpToAlignment(typeAlignment);

  if (fieldOffsetInBytes < alignedNextFieldOffsetInBytes) {
    // Try to resize the last base field.
    if (ResizeLastBaseFieldIfNecessary(fieldOffsetInBytes)) {
      alignedNextFieldOffsetInBytes = 
        NextFieldOffset.RoundUpToAlignment(typeAlignment);
    }
  }

  if (fieldOffsetInBytes < alignedNextFieldOffsetInBytes) {
    assert(!Packed && "Could not place field even with packed struct!");
    return false;
  }

  AppendPadding(fieldOffsetInBytes, typeAlignment);

  // Now append the field.
  Fields[D] = FieldTypes.size();
  AppendField(fieldOffsetInBytes, Ty);

  LastLaidOutBase.invalidate();
  return true;
}

llvm::Type *
CGRecordLayoutBuilder::LayoutUnionField(const FieldDecl *Field,
                                        const ASTRecordLayout &Layout) {
  Fields[Field] = 0;
  if (Field->isBitField()) {
    uint64_t FieldSize = Field->getBitWidthValue(Types.getContext());

    // Ignore zero sized bit fields.
    if (FieldSize == 0)
      return 0;

    unsigned StorageBits = llvm::RoundUpToAlignment(
      FieldSize, Types.getTarget().getCharAlign());
    CharUnits NumBytesToAppend
      = Types.getContext().toCharUnitsFromBits(StorageBits);

    llvm::Type *FieldTy = llvm::Type::getInt8Ty(Types.getLLVMContext());
    if (NumBytesToAppend > CharUnits::One())
      FieldTy = llvm::ArrayType::get(FieldTy, NumBytesToAppend.getQuantity());

    // Add the bit field info.
    BitFields[Field] = CGBitFieldInfo::MakeInfo(Types, Field, 0, FieldSize,
                                                StorageBits,
                                                Alignment.getQuantity());
    return FieldTy;
  }

  // This is a regular union field.
  return Types.ConvertTypeForMem(Field->getType());
}

void CGRecordLayoutBuilder::LayoutUnion(const RecordDecl *D) {
  assert(D->isUnion() && "Can't call LayoutUnion on a non-union record!");

  const ASTRecordLayout &layout = Types.getContext().getASTRecordLayout(D);

  llvm::Type *unionType = 0;
  CharUnits unionSize = CharUnits::Zero();
  CharUnits unionAlign = CharUnits::Zero();

  bool hasOnlyZeroSizedBitFields = true;
  bool checkedFirstFieldZeroInit = false;

  unsigned fieldNo = 0;
  for (RecordDecl::field_iterator field = D->field_begin(),
       fieldEnd = D->field_end(); field != fieldEnd; ++field, ++fieldNo) {
    assert(layout.getFieldOffset(fieldNo) == 0 &&
          "Union field offset did not start at the beginning of record!");
    llvm::Type *fieldType = LayoutUnionField(*field, layout);

    if (!fieldType)
      continue;

    if (field->getDeclName() && !checkedFirstFieldZeroInit) {
      CheckZeroInitializable(field->getType());
      checkedFirstFieldZeroInit = true;
    }

    hasOnlyZeroSizedBitFields = false;

    CharUnits fieldAlign = CharUnits::fromQuantity(
                          Types.getDataLayout().getABITypeAlignment(fieldType));
    CharUnits fieldSize = CharUnits::fromQuantity(
                             Types.getDataLayout().getTypeAllocSize(fieldType));

    if (fieldAlign < unionAlign)
      continue;

    if (fieldAlign > unionAlign || fieldSize > unionSize) {
      unionType = fieldType;
      unionAlign = fieldAlign;
      unionSize = fieldSize;
    }
  }

  // Now add our field.
  if (unionType) {
    AppendField(CharUnits::Zero(), unionType);

    if (getTypeAlignment(unionType) > layout.getAlignment()) {
      // We need a packed struct.
      Packed = true;
      unionAlign = CharUnits::One();
    }
  }
  if (unionAlign.isZero()) {
    (void)hasOnlyZeroSizedBitFields;
    assert(hasOnlyZeroSizedBitFields &&
           "0-align record did not have all zero-sized bit-fields!");
    unionAlign = CharUnits::One();
  }

  // Append tail padding.
  CharUnits recordSize = layout.getSize();
  if (recordSize > unionSize)
    AppendPadding(recordSize, unionAlign);
}

bool CGRecordLayoutBuilder::LayoutBase(const CXXRecordDecl *base,
                                       const CGRecordLayout &baseLayout,
                                       CharUnits baseOffset) {
  ResizeLastBaseFieldIfNecessary(baseOffset);

  AppendPadding(baseOffset, CharUnits::One());

  const ASTRecordLayout &baseASTLayout
    = Types.getContext().getASTRecordLayout(base);

  LastLaidOutBase.Offset = NextFieldOffset;
  LastLaidOutBase.NonVirtualSize = baseASTLayout.getNonVirtualSize();

  llvm::StructType *subobjectType = baseLayout.getBaseSubobjectLLVMType();
  if (getTypeAlignment(subobjectType) > Alignment)
    return false;

  AppendField(baseOffset, subobjectType);
  return true;
}

bool CGRecordLayoutBuilder::LayoutNonVirtualBase(const CXXRecordDecl *base,
                                                 CharUnits baseOffset) {
  // Ignore empty bases.
  if (base->isEmpty()) return true;

  const CGRecordLayout &baseLayout = Types.getCGRecordLayout(base);
  if (IsZeroInitializableAsBase) {
    assert(IsZeroInitializable &&
           "class zero-initializable as base but not as complete object");

    IsZeroInitializable = IsZeroInitializableAsBase =
      baseLayout.isZeroInitializableAsBase();
  }

  if (!LayoutBase(base, baseLayout, baseOffset))
    return false;
  NonVirtualBases[base] = (FieldTypes.size() - 1);
  return true;
}

bool
CGRecordLayoutBuilder::LayoutVirtualBase(const CXXRecordDecl *base,
                                         CharUnits baseOffset) {
  // Ignore empty bases.
  if (base->isEmpty()) return true;

  const CGRecordLayout &baseLayout = Types.getCGRecordLayout(base);
  if (IsZeroInitializable)
    IsZeroInitializable = baseLayout.isZeroInitializableAsBase();

  if (!LayoutBase(base, baseLayout, baseOffset))
    return false;
  VirtualBases[base] = (FieldTypes.size() - 1);
  return true;
}

bool
CGRecordLayoutBuilder::MSLayoutVirtualBases(const CXXRecordDecl *RD,
                                          const ASTRecordLayout &Layout) {
  if (!RD->getNumVBases())
    return true;

  // The vbases list is uniqued and ordered by a depth-first
  // traversal, which is what we need here.
  for (CXXRecordDecl::base_class_const_iterator I = RD->vbases_begin(),
        E = RD->vbases_end(); I != E; ++I) {

    const CXXRecordDecl *BaseDecl = 
      cast<CXXRecordDecl>(I->getType()->castAs<RecordType>()->getDecl());

    CharUnits vbaseOffset = Layout.getVBaseClassOffset(BaseDecl);
    if (!LayoutVirtualBase(BaseDecl, vbaseOffset))
      return false;
  }
  return true;
}

/// LayoutVirtualBases - layout the non-virtual bases of a record decl.
bool
CGRecordLayoutBuilder::LayoutVirtualBases(const CXXRecordDecl *RD,
                                          const ASTRecordLayout &Layout) {
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    const CXXRecordDecl *BaseDecl = 
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    // We only want to lay out virtual bases that aren't indirect primary bases
    // of some other base.
    if (I->isVirtual() && !IndirectPrimaryBases.count(BaseDecl)) {
      // Only lay out the base once.
      if (!LaidOutVirtualBases.insert(BaseDecl))
        continue;

      CharUnits vbaseOffset = Layout.getVBaseClassOffset(BaseDecl);
      if (!LayoutVirtualBase(BaseDecl, vbaseOffset))
        return false;
    }

    if (!BaseDecl->getNumVBases()) {
      // This base isn't interesting since it doesn't have any virtual bases.
      continue;
    }
    
    if (!LayoutVirtualBases(BaseDecl, Layout))
      return false;
  }
  return true;
}

bool
CGRecordLayoutBuilder::LayoutNonVirtualBases(const CXXRecordDecl *RD,
                                             const ASTRecordLayout &Layout) {
  const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();

  // If we have a primary base, lay it out first.
  if (PrimaryBase) {
    if (!Layout.isPrimaryBaseVirtual()) {
      if (!LayoutNonVirtualBase(PrimaryBase, CharUnits::Zero()))
        return false;
    } else {
      if (!LayoutVirtualBase(PrimaryBase, CharUnits::Zero()))
        return false;
    }

  // Otherwise, add a vtable / vf-table if the layout says to do so.
  } else if (Layout.hasOwnVFPtr()) {
    llvm::Type *FunctionType =
      llvm::FunctionType::get(llvm::Type::getInt32Ty(Types.getLLVMContext()),
                              /*isVarArg=*/true);
    llvm::Type *VTableTy = FunctionType->getPointerTo();

    if (getTypeAlignment(VTableTy) > Alignment) {
      // FIXME: Should we allow this to happen in Sema?
      assert(!Packed && "Alignment is wrong even with packed struct!");
      return false;
    }

    assert(NextFieldOffset.isZero() &&
           "VTable pointer must come first!");
    AppendField(CharUnits::Zero(), VTableTy->getPointerTo());
  }

  // Layout the non-virtual bases.
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    if (I->isVirtual())
      continue;

    const CXXRecordDecl *BaseDecl = 
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    // We've already laid out the primary base.
    if (BaseDecl == PrimaryBase && !Layout.isPrimaryBaseVirtual())
      continue;

    if (!LayoutNonVirtualBase(BaseDecl, Layout.getBaseClassOffset(BaseDecl)))
      return false;
  }

  // Add a vb-table pointer if the layout insists.
    if (Layout.hasOwnVBPtr()) {
    CharUnits VBPtrOffset = Layout.getVBPtrOffset();
    llvm::Type *Vbptr = llvm::Type::getInt32PtrTy(Types.getLLVMContext());
    AppendPadding(VBPtrOffset, getTypeAlignment(Vbptr));
    AppendField(VBPtrOffset, Vbptr);
  }

  return true;
}

bool
CGRecordLayoutBuilder::ComputeNonVirtualBaseType(const CXXRecordDecl *RD) {
  const ASTRecordLayout &Layout = Types.getContext().getASTRecordLayout(RD);

  CharUnits NonVirtualSize  = Layout.getNonVirtualSize();
  CharUnits NonVirtualAlign = Layout.getNonVirtualAlign();
  CharUnits AlignedNonVirtualTypeSize =
    NonVirtualSize.RoundUpToAlignment(NonVirtualAlign);
  
  // First check if we can use the same fields as for the complete class.
  CharUnits RecordSize = Layout.getSize();
  if (AlignedNonVirtualTypeSize == RecordSize)
    return true;

  // Check if we need padding.
  CharUnits AlignedNextFieldOffset =
    NextFieldOffset.RoundUpToAlignment(getAlignmentAsLLVMStruct());

  if (AlignedNextFieldOffset > AlignedNonVirtualTypeSize) {
    assert(!Packed && "cannot layout even as packed struct");
    return false; // Needs packing.
  }

  bool needsPadding = (AlignedNonVirtualTypeSize != AlignedNextFieldOffset);
  if (needsPadding) {
    CharUnits NumBytes = AlignedNonVirtualTypeSize - AlignedNextFieldOffset;
    FieldTypes.push_back(getByteArrayType(NumBytes));
  }
  
  BaseSubobjectType = llvm::StructType::create(Types.getLLVMContext(),
                                               FieldTypes, "", Packed);
  Types.addRecordTypeName(RD, BaseSubobjectType, ".base");

  // Pull the padding back off.
  if (needsPadding)
    FieldTypes.pop_back();

  return true;
}

bool CGRecordLayoutBuilder::LayoutFields(const RecordDecl *D) {
  assert(!D->isUnion() && "Can't call LayoutFields on a union!");
  assert(!Alignment.isZero() && "Did not set alignment!");

  const ASTRecordLayout &Layout = Types.getContext().getASTRecordLayout(D);

  const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(D);
  if (RD)
    if (!LayoutNonVirtualBases(RD, Layout))
      return false;

  unsigned FieldNo = 0;
  
  for (RecordDecl::field_iterator FI = D->field_begin(), FE = D->field_end();
       FI != FE; ++FI, ++FieldNo) {
    FieldDecl *FD = *FI;

    // If this field is a bitfield, layout all of the consecutive
    // non-zero-length bitfields and the last zero-length bitfield; these will
    // all share storage.
    if (FD->isBitField()) {
      // If all we have is a zero-width bitfield, skip it.
      if (FD->getBitWidthValue(Types.getContext()) == 0)
        continue;

      // Layout this range of bitfields.
      if (!LayoutBitfields(Layout, FieldNo, FI, FE)) {
        assert(!Packed &&
               "Could not layout bitfields even with a packed LLVM struct!");
        return false;
      }
      assert(FI != FE && "Advanced past the last bitfield");
      continue;
    }

    if (!LayoutField(FD, Layout.getFieldOffset(FieldNo))) {
      assert(!Packed &&
             "Could not layout fields even with a packed LLVM struct!");
      return false;
    }
  }

  if (RD) {
    // We've laid out the non-virtual bases and the fields, now compute the
    // non-virtual base field types.
    if (!ComputeNonVirtualBaseType(RD)) {
      assert(!Packed && "Could not layout even with a packed LLVM struct!");
      return false;
    }

    // Lay out the virtual bases.  The MS ABI uses a different
    // algorithm here due to the lack of primary virtual bases.
    if (Types.getTarget().getCXXABI().hasPrimaryVBases()) {
      RD->getIndirectPrimaryBases(IndirectPrimaryBases);
      if (Layout.isPrimaryBaseVirtual())
        IndirectPrimaryBases.insert(Layout.getPrimaryBase());

      if (!LayoutVirtualBases(RD, Layout))
        return false;
    } else {
      if (!MSLayoutVirtualBases(RD, Layout))
        return false;
    }
  }
  
  // Append tail padding if necessary.
  AppendTailPadding(Layout.getSize());

  return true;
}

void CGRecordLayoutBuilder::AppendTailPadding(CharUnits RecordSize) {
  ResizeLastBaseFieldIfNecessary(RecordSize);

  assert(NextFieldOffset <= RecordSize && "Size mismatch!");

  CharUnits AlignedNextFieldOffset =
    NextFieldOffset.RoundUpToAlignment(getAlignmentAsLLVMStruct());

  if (AlignedNextFieldOffset == RecordSize) {
    // We don't need any padding.
    return;
  }

  CharUnits NumPadBytes = RecordSize - NextFieldOffset;
  AppendBytes(NumPadBytes);
}

void CGRecordLayoutBuilder::AppendField(CharUnits fieldOffset,
                                        llvm::Type *fieldType) {
  CharUnits fieldSize =
    CharUnits::fromQuantity(Types.getDataLayout().getTypeAllocSize(fieldType));

  FieldTypes.push_back(fieldType);

  NextFieldOffset = fieldOffset + fieldSize;
}

void CGRecordLayoutBuilder::AppendPadding(CharUnits fieldOffset,
                                          CharUnits fieldAlignment) {
  assert(NextFieldOffset <= fieldOffset &&
         "Incorrect field layout!");

  // Do nothing if we're already at the right offset.
  if (fieldOffset == NextFieldOffset) return;

  // If we're not emitting a packed LLVM type, try to avoid adding
  // unnecessary padding fields.
  if (!Packed) {
    // Round up the field offset to the alignment of the field type.
    CharUnits alignedNextFieldOffset =
      NextFieldOffset.RoundUpToAlignment(fieldAlignment);
    assert(alignedNextFieldOffset <= fieldOffset);

    // If that's the right offset, we're done.
    if (alignedNextFieldOffset == fieldOffset) return;
  }

  // Otherwise we need explicit padding.
  CharUnits padding = fieldOffset - NextFieldOffset;
  AppendBytes(padding);
}

bool CGRecordLayoutBuilder::ResizeLastBaseFieldIfNecessary(CharUnits offset) {
  // Check if we have a base to resize.
  if (!LastLaidOutBase.isValid())
    return false;

  // This offset does not overlap with the tail padding.
  if (offset >= NextFieldOffset)
    return false;

  // Restore the field offset and append an i8 array instead.
  FieldTypes.pop_back();
  NextFieldOffset = LastLaidOutBase.Offset;
  AppendBytes(LastLaidOutBase.NonVirtualSize);
  LastLaidOutBase.invalidate();

  return true;
}

llvm::Type *CGRecordLayoutBuilder::getByteArrayType(CharUnits numBytes) {
  assert(!numBytes.isZero() && "Empty byte arrays aren't allowed.");

  llvm::Type *Ty = llvm::Type::getInt8Ty(Types.getLLVMContext());
  if (numBytes > CharUnits::One())
    Ty = llvm::ArrayType::get(Ty, numBytes.getQuantity());

  return Ty;
}

void CGRecordLayoutBuilder::AppendBytes(CharUnits numBytes) {
  if (numBytes.isZero())
    return;

  // Append the padding field
  AppendField(NextFieldOffset, getByteArrayType(numBytes));
}

CharUnits CGRecordLayoutBuilder::getTypeAlignment(llvm::Type *Ty) const {
  if (Packed)
    return CharUnits::One();

  return CharUnits::fromQuantity(Types.getDataLayout().getABITypeAlignment(Ty));
}

CharUnits CGRecordLayoutBuilder::getAlignmentAsLLVMStruct() const {
  if (Packed)
    return CharUnits::One();

  CharUnits maxAlignment = CharUnits::One();
  for (size_t i = 0; i != FieldTypes.size(); ++i)
    maxAlignment = std::max(maxAlignment, getTypeAlignment(FieldTypes[i]));

  return maxAlignment;
}

/// Merge in whether a field of the given type is zero-initializable.
void CGRecordLayoutBuilder::CheckZeroInitializable(QualType T) {
  // This record already contains a member pointer.
  if (!IsZeroInitializableAsBase)
    return;

  // Can only have member pointers if we're compiling C++.
  if (!Types.getContext().getLangOpts().CPlusPlus)
    return;

  const Type *elementType = T->getBaseElementTypeUnsafe();

  if (const MemberPointerType *MPT = elementType->getAs<MemberPointerType>()) {
    if (!Types.getCXXABI().isZeroInitializable(MPT))
      IsZeroInitializable = IsZeroInitializableAsBase = false;
  } else if (const RecordType *RT = elementType->getAs<RecordType>()) {
    const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
    const CGRecordLayout &Layout = Types.getCGRecordLayout(RD);
    if (!Layout.isZeroInitializable())
      IsZeroInitializable = IsZeroInitializableAsBase = false;
  }
}

CGRecordLayout *CodeGenTypes::ComputeRecordLayout(const RecordDecl *D,
                                                  llvm::StructType *Ty) {
  CGRecordLayoutBuilder Builder(*this);

  Builder.Layout(D);

  Ty->setBody(Builder.FieldTypes, Builder.Packed);

  // If we're in C++, compute the base subobject type.
  llvm::StructType *BaseTy = 0;
  if (isa<CXXRecordDecl>(D) && !D->isUnion()) {
    BaseTy = Builder.BaseSubobjectType;
    if (!BaseTy) BaseTy = Ty;
  }

  CGRecordLayout *RL =
    new CGRecordLayout(Ty, BaseTy, Builder.IsZeroInitializable,
                       Builder.IsZeroInitializableAsBase);

  RL->NonVirtualBases.swap(Builder.NonVirtualBases);
  RL->CompleteObjectVirtualBases.swap(Builder.VirtualBases);

  // Add all the field numbers.
  RL->FieldInfo.swap(Builder.Fields);

  // Add bitfield info.
  RL->BitFields.swap(Builder.BitFields);

  // Dump the layout, if requested.
  if (getContext().getLangOpts().DumpRecordLayouts) {
    llvm::outs() << "\n*** Dumping IRgen Record Layout\n";
    llvm::outs() << "Record: ";
    D->dump(llvm::outs());
    llvm::outs() << "\nLayout: ";
    RL->print(llvm::outs());
  }

#ifndef NDEBUG
  // Verify that the computed LLVM struct size matches the AST layout size.
  const ASTRecordLayout &Layout = getContext().getASTRecordLayout(D);

  uint64_t TypeSizeInBits = getContext().toBits(Layout.getSize());
  assert(TypeSizeInBits == getDataLayout().getTypeAllocSizeInBits(Ty) &&
         "Type size mismatch!");

  if (BaseTy) {
    CharUnits NonVirtualSize  = Layout.getNonVirtualSize();
    CharUnits NonVirtualAlign = Layout.getNonVirtualAlign();
    CharUnits AlignedNonVirtualTypeSize = 
      NonVirtualSize.RoundUpToAlignment(NonVirtualAlign);

    uint64_t AlignedNonVirtualTypeSizeInBits = 
      getContext().toBits(AlignedNonVirtualTypeSize);

    assert(AlignedNonVirtualTypeSizeInBits == 
           getDataLayout().getTypeAllocSizeInBits(BaseTy) &&
           "Type size mismatch!");
  }
                                     
  // Verify that the LLVM and AST field offsets agree.
  llvm::StructType *ST =
    dyn_cast<llvm::StructType>(RL->getLLVMType());
  const llvm::StructLayout *SL = getDataLayout().getStructLayout(ST);

  const ASTRecordLayout &AST_RL = getContext().getASTRecordLayout(D);
  RecordDecl::field_iterator it = D->field_begin();
  for (unsigned i = 0, e = AST_RL.getFieldCount(); i != e; ++i, ++it) {
    const FieldDecl *FD = *it;

    // For non-bit-fields, just check that the LLVM struct offset matches the
    // AST offset.
    if (!FD->isBitField()) {
      unsigned FieldNo = RL->getLLVMFieldNo(FD);
      assert(AST_RL.getFieldOffset(i) == SL->getElementOffsetInBits(FieldNo) &&
             "Invalid field offset!");
      continue;
    }
    
    // Ignore unnamed bit-fields.
    if (!FD->getDeclName())
      continue;

    // Don't inspect zero-length bitfields.
    if (FD->getBitWidthValue(getContext()) == 0)
      continue;

    const CGBitFieldInfo &Info = RL->getBitFieldInfo(FD);
    llvm::Type *ElementTy = ST->getTypeAtIndex(RL->getLLVMFieldNo(FD));

    // Unions have overlapping elements dictating their layout, but for
    // non-unions we can verify that this section of the layout is the exact
    // expected size.
    if (D->isUnion()) {
      // For unions we verify that the start is zero and the size
      // is in-bounds. However, on BE systems, the offset may be non-zero, but
      // the size + offset should match the storage size in that case as it
      // "starts" at the back.
      if (getDataLayout().isBigEndian())
        assert(static_cast<unsigned>(Info.Offset + Info.Size) ==
               Info.StorageSize &&
               "Big endian union bitfield does not end at the back");
      else
        assert(Info.Offset == 0 &&
               "Little endian union bitfield with a non-zero offset");
      assert(Info.StorageSize <= SL->getSizeInBits() &&
             "Union not large enough for bitfield storage");
    } else {
      assert(Info.StorageSize ==
             getDataLayout().getTypeAllocSizeInBits(ElementTy) &&
             "Storage size does not match the element type size");
    }
    assert(Info.Size > 0 && "Empty bitfield!");
    assert(static_cast<unsigned>(Info.Offset) + Info.Size <= Info.StorageSize &&
           "Bitfield outside of its allocated storage");
  }
#endif

  return RL;
}

void CGRecordLayout::print(raw_ostream &OS) const {
  OS << "<CGRecordLayout\n";
  OS << "  LLVMType:" << *CompleteObjectType << "\n";
  if (BaseSubobjectType)
    OS << "  NonVirtualBaseLLVMType:" << *BaseSubobjectType << "\n"; 
  OS << "  IsZeroInitializable:" << IsZeroInitializable << "\n";
  OS << "  BitFields:[\n";

  // Print bit-field infos in declaration order.
  std::vector<std::pair<unsigned, const CGBitFieldInfo*> > BFIs;
  for (llvm::DenseMap<const FieldDecl*, CGBitFieldInfo>::const_iterator
         it = BitFields.begin(), ie = BitFields.end();
       it != ie; ++it) {
    const RecordDecl *RD = it->first->getParent();
    unsigned Index = 0;
    for (RecordDecl::field_iterator
           it2 = RD->field_begin(); *it2 != it->first; ++it2)
      ++Index;
    BFIs.push_back(std::make_pair(Index, &it->second));
  }
  llvm::array_pod_sort(BFIs.begin(), BFIs.end());
  for (unsigned i = 0, e = BFIs.size(); i != e; ++i) {
    OS.indent(4);
    BFIs[i].second->print(OS);
    OS << "\n";
  }

  OS << "]>\n";
}

void CGRecordLayout::dump() const {
  print(llvm::errs());
}

void CGBitFieldInfo::print(raw_ostream &OS) const {
  OS << "<CGBitFieldInfo"
     << " Offset:" << Offset
     << " Size:" << Size
     << " IsSigned:" << IsSigned
     << " StorageSize:" << StorageSize
     << " StorageAlignment:" << StorageAlignment << ">";
}

void CGBitFieldInfo::dump() const {
  print(llvm::errs());
}
