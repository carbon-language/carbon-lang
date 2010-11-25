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
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecordLayout.h"
#include "CodeGenTypes.h"
#include "CGCXXABI.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Type.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetData.h"
using namespace clang;
using namespace CodeGen;

namespace clang {
namespace CodeGen {

class CGRecordLayoutBuilder {
public:
  /// FieldTypes - Holds the LLVM types that the struct is created from.
  std::vector<const llvm::Type *> FieldTypes;

  /// NonVirtualBaseFieldTypes - Holds the LLVM types for the non-virtual part
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
  std::vector<const llvm::Type *> NonVirtualBaseFieldTypes;

  /// NonVirtualBaseTypeIsSameAsCompleteType - Whether the non-virtual part of
  /// the struct is equivalent to the complete struct.
  bool NonVirtualBaseTypeIsSameAsCompleteType;
  
  /// LLVMFieldInfo - Holds a field and its corresponding LLVM field number.
  typedef std::pair<const FieldDecl *, unsigned> LLVMFieldInfo;
  llvm::SmallVector<LLVMFieldInfo, 16> LLVMFields;

  /// LLVMBitFieldInfo - Holds location and size information about a bit field.
  typedef std::pair<const FieldDecl *, CGBitFieldInfo> LLVMBitFieldInfo;
  llvm::SmallVector<LLVMBitFieldInfo, 16> LLVMBitFields;

  typedef std::pair<const CXXRecordDecl *, unsigned> LLVMBaseInfo;
  llvm::SmallVector<LLVMBaseInfo, 16> LLVMNonVirtualBases;

  /// IndirectPrimaryBases - Virtual base classes, direct or indirect, that are
  /// primary base classes for some other direct or indirect base class.
  CXXIndirectPrimaryBaseSet IndirectPrimaryBases;

  /// IsZeroInitializable - Whether this struct can be C++
  /// zero-initialized with an LLVM zeroinitializer.
  bool IsZeroInitializable;

  /// Packed - Whether the resulting LLVM struct will be packed or not.
  bool Packed;

private:
  CodeGenTypes &Types;

  /// Alignment - Contains the alignment of the RecordDecl.
  //
  // FIXME: This is not needed and should be removed.
  unsigned Alignment;

  /// AlignmentAsLLVMStruct - Will contain the maximum alignment of all the
  /// LLVM types.
  unsigned AlignmentAsLLVMStruct;

  /// BitsAvailableInLastField - If a bit field spans only part of a LLVM field,
  /// this will have the number of bits still available in the field.
  char BitsAvailableInLastField;

  /// NextFieldOffsetInBytes - Holds the next field offset in bytes.
  uint64_t NextFieldOffsetInBytes;

  /// LayoutUnionField - Will layout a field in an union and return the type
  /// that the field will have.
  const llvm::Type *LayoutUnionField(const FieldDecl *Field,
                                     const ASTRecordLayout &Layout);
  
  /// LayoutUnion - Will layout a union RecordDecl.
  void LayoutUnion(const RecordDecl *D);

  /// LayoutField - try to layout all fields in the record decl.
  /// Returns false if the operation failed because the struct is not packed.
  bool LayoutFields(const RecordDecl *D);

  /// LayoutVirtualBase - layout a single virtual base.
  void LayoutVirtualBase(const CXXRecordDecl *BaseDecl, uint64_t BaseOffset);

  /// LayoutNonVirtualBase - layout a single non-virtual base.
  void LayoutNonVirtualBase(const CXXRecordDecl *BaseDecl,
                            uint64_t BaseOffset);
  
  /// LayoutNonVirtualBases - layout the non-virtual bases of a record decl.
  void LayoutNonVirtualBases(const CXXRecordDecl *RD, 
                             const ASTRecordLayout &Layout);

  /// ComputeNonVirtualBaseType - Compute the non-virtual base field types.
  void ComputeNonVirtualBaseType(const CXXRecordDecl *RD);
  
  /// LayoutField - layout a single field. Returns false if the operation failed
  /// because the current struct is not packed.
  bool LayoutField(const FieldDecl *D, uint64_t FieldOffset);

  /// LayoutBitField - layout a single bit field.
  void LayoutBitField(const FieldDecl *D, uint64_t FieldOffset);

  /// AppendField - Appends a field with the given offset and type.
  void AppendField(uint64_t FieldOffsetInBytes, const llvm::Type *FieldTy);

  /// AppendPadding - Appends enough padding bytes so that the total
  /// struct size is a multiple of the field alignment.
  void AppendPadding(uint64_t FieldOffsetInBytes, unsigned FieldAlignment);

  /// getByteArrayType - Returns a byte array type with the given number of
  /// elements.
  const llvm::Type *getByteArrayType(uint64_t NumBytes);
  
  /// AppendBytes - Append a given number of bytes to the record.
  void AppendBytes(uint64_t NumBytes);

  /// AppendTailPadding - Append enough tail padding so that the type will have
  /// the passed size.
  void AppendTailPadding(uint64_t RecordSize);

  unsigned getTypeAlignment(const llvm::Type *Ty) const;

  /// CheckZeroInitializable - Check if the given type contains a pointer
  /// to data member.
  void CheckZeroInitializable(QualType T);
  void CheckZeroInitializable(const CXXRecordDecl *RD);

public:
  CGRecordLayoutBuilder(CodeGenTypes &Types)
    : NonVirtualBaseTypeIsSameAsCompleteType(false), IsZeroInitializable(true),
      Packed(false), Types(Types), Alignment(0), AlignmentAsLLVMStruct(1),
      BitsAvailableInLastField(0), NextFieldOffsetInBytes(0) { }

  /// Layout - Will layout a RecordDecl.
  void Layout(const RecordDecl *D);
};

}
}

void CGRecordLayoutBuilder::Layout(const RecordDecl *D) {
  Alignment = Types.getContext().getASTRecordLayout(D).getAlignment() / 8;
  Packed = D->hasAttr<PackedAttr>();

  if (D->isUnion()) {
    LayoutUnion(D);
    return;
  }

  if (LayoutFields(D))
    return;

  // We weren't able to layout the struct. Try again with a packed struct
  Packed = true;
  AlignmentAsLLVMStruct = 1;
  NextFieldOffsetInBytes = 0;
  FieldTypes.clear();
  LLVMFields.clear();
  LLVMBitFields.clear();
  LLVMNonVirtualBases.clear();

  LayoutFields(D);
}

CGBitFieldInfo CGBitFieldInfo::MakeInfo(CodeGenTypes &Types,
                               const FieldDecl *FD,
                               uint64_t FieldOffset,
                               uint64_t FieldSize,
                               uint64_t ContainingTypeSizeInBits,
                               unsigned ContainingTypeAlign) {
  const llvm::Type *Ty = Types.ConvertTypeForMemRecursive(FD->getType());
  uint64_t TypeSizeInBytes = Types.getTargetData().getTypeAllocSize(Ty);
  uint64_t TypeSizeInBits = TypeSizeInBytes * 8;

  bool IsSigned = FD->getType()->isSignedIntegerType();

  if (FieldSize > TypeSizeInBits) {
    // We have a wide bit-field. The extra bits are only used for padding, so
    // if we have a bitfield of type T, with size N:
    //
    // T t : N;
    //
    // We can just assume that it's:
    //
    // T t : sizeof(T);
    //
    FieldSize = TypeSizeInBits;
  }

  // Compute the access components. The policy we use is to start by attempting
  // to access using the width of the bit-field type itself and to always access
  // at aligned indices of that type. If such an access would fail because it
  // extends past the bound of the type, then we reduce size to the next smaller
  // power of two and retry. The current algorithm assumes pow2 sized types,
  // although this is easy to fix.
  //
  // FIXME: This algorithm is wrong on big-endian systems, I think.
  assert(llvm::isPowerOf2_32(TypeSizeInBits) && "Unexpected type size!");
  CGBitFieldInfo::AccessInfo Components[3];
  unsigned NumComponents = 0;
  unsigned AccessedTargetBits = 0;       // The tumber of target bits accessed.
  unsigned AccessWidth = TypeSizeInBits; // The current access width to attempt.

  // Round down from the field offset to find the first access position that is
  // at an aligned offset of the initial access type.
  uint64_t AccessStart = FieldOffset - (FieldOffset % AccessWidth);

  // Adjust initial access size to fit within record.
  while (AccessWidth > 8 &&
         AccessStart + AccessWidth > ContainingTypeSizeInBits) {
    AccessWidth >>= 1;
    AccessStart = FieldOffset - (FieldOffset % AccessWidth);
  }

  while (AccessedTargetBits < FieldSize) {
    // Check that we can access using a type of this size, without reading off
    // the end of the structure. This can occur with packed structures and
    // -fno-bitfield-type-align, for example.
    if (AccessStart + AccessWidth > ContainingTypeSizeInBits) {
      // If so, reduce access size to the next smaller power-of-two and retry.
      AccessWidth >>= 1;
      assert(AccessWidth >= 8 && "Cannot access under byte size!");
      continue;
    }

    // Otherwise, add an access component.

    // First, compute the bits inside this access which are part of the
    // target. We are reading bits [AccessStart, AccessStart + AccessWidth); the
    // intersection with [FieldOffset, FieldOffset + FieldSize) gives the bits
    // in the target that we are reading.
    assert(FieldOffset < AccessStart + AccessWidth && "Invalid access start!");
    assert(AccessStart < FieldOffset + FieldSize && "Invalid access start!");
    uint64_t AccessBitsInFieldStart = std::max(AccessStart, FieldOffset);
    uint64_t AccessBitsInFieldSize =
      std::min(AccessWidth + AccessStart,
               FieldOffset + FieldSize) - AccessBitsInFieldStart;

    assert(NumComponents < 3 && "Unexpected number of components!");
    CGBitFieldInfo::AccessInfo &AI = Components[NumComponents++];
    AI.FieldIndex = 0;
    // FIXME: We still follow the old access pattern of only using the field
    // byte offset. We should switch this once we fix the struct layout to be
    // pretty.
    AI.FieldByteOffset = AccessStart / 8;
    AI.FieldBitStart = AccessBitsInFieldStart - AccessStart;
    AI.AccessWidth = AccessWidth;
    AI.AccessAlignment = llvm::MinAlign(ContainingTypeAlign, AccessStart) / 8;
    AI.TargetBitOffset = AccessedTargetBits;
    AI.TargetBitWidth = AccessBitsInFieldSize;

    AccessStart += AccessWidth;
    AccessedTargetBits += AI.TargetBitWidth;
  }

  assert(AccessedTargetBits == FieldSize && "Invalid bit-field access!");
  return CGBitFieldInfo(FieldSize, NumComponents, Components, IsSigned);
}

CGBitFieldInfo CGBitFieldInfo::MakeInfo(CodeGenTypes &Types,
                                        const FieldDecl *FD,
                                        uint64_t FieldOffset,
                                        uint64_t FieldSize) {
  const RecordDecl *RD = FD->getParent();
  const ASTRecordLayout &RL = Types.getContext().getASTRecordLayout(RD);
  uint64_t ContainingTypeSizeInBits = RL.getSize();
  unsigned ContainingTypeAlign = RL.getAlignment();

  return MakeInfo(Types, FD, FieldOffset, FieldSize, ContainingTypeSizeInBits,
                  ContainingTypeAlign);
}

void CGRecordLayoutBuilder::LayoutBitField(const FieldDecl *D,
                                           uint64_t FieldOffset) {
  uint64_t FieldSize =
    D->getBitWidth()->EvaluateAsInt(Types.getContext()).getZExtValue();

  if (FieldSize == 0)
    return;

  uint64_t NextFieldOffset = NextFieldOffsetInBytes * 8;
  unsigned NumBytesToAppend;

  if (FieldOffset < NextFieldOffset) {
    assert(BitsAvailableInLastField && "Bitfield size mismatch!");
    assert(NextFieldOffsetInBytes && "Must have laid out at least one byte!");

    // The bitfield begins in the previous bit-field.
    NumBytesToAppend =
      llvm::RoundUpToAlignment(FieldSize - BitsAvailableInLastField, 8) / 8;
  } else {
    assert(FieldOffset % 8 == 0 && "Field offset not aligned correctly");

    // Append padding if necessary.
    AppendBytes((FieldOffset - NextFieldOffset) / 8);

    NumBytesToAppend =
      llvm::RoundUpToAlignment(FieldSize, 8) / 8;

    assert(NumBytesToAppend && "No bytes to append!");
  }

  // Add the bit field info.
  LLVMBitFields.push_back(
    LLVMBitFieldInfo(D, CGBitFieldInfo::MakeInfo(Types, D, FieldOffset,
                                                 FieldSize)));

  AppendBytes(NumBytesToAppend);

  BitsAvailableInLastField =
    NextFieldOffsetInBytes * 8 - (FieldOffset + FieldSize);
}

bool CGRecordLayoutBuilder::LayoutField(const FieldDecl *D,
                                        uint64_t FieldOffset) {
  // If the field is packed, then we need a packed struct.
  if (!Packed && D->hasAttr<PackedAttr>())
    return false;

  if (D->isBitField()) {
    // We must use packed structs for unnamed bit fields since they
    // don't affect the struct alignment.
    if (!Packed && !D->getDeclName())
      return false;

    LayoutBitField(D, FieldOffset);
    return true;
  }

  CheckZeroInitializable(D->getType());

  assert(FieldOffset % 8 == 0 && "FieldOffset is not on a byte boundary!");
  uint64_t FieldOffsetInBytes = FieldOffset / 8;

  const llvm::Type *Ty = Types.ConvertTypeForMemRecursive(D->getType());
  unsigned TypeAlignment = getTypeAlignment(Ty);

  // If the type alignment is larger then the struct alignment, we must use
  // a packed struct.
  if (TypeAlignment > Alignment) {
    assert(!Packed && "Alignment is wrong even with packed struct!");
    return false;
  }

  if (const RecordType *RT = D->getType()->getAs<RecordType>()) {
    const RecordDecl *RD = cast<RecordDecl>(RT->getDecl());
    if (const MaxFieldAlignmentAttr *MFAA =
          RD->getAttr<MaxFieldAlignmentAttr>()) {
      if (MFAA->getAlignment() != TypeAlignment * 8 && !Packed)
        return false;
    }
  }

  // Round up the field offset to the alignment of the field type.
  uint64_t AlignedNextFieldOffsetInBytes =
    llvm::RoundUpToAlignment(NextFieldOffsetInBytes, TypeAlignment);

  if (FieldOffsetInBytes < AlignedNextFieldOffsetInBytes) {
    assert(!Packed && "Could not place field even with packed struct!");
    return false;
  }

  if (AlignedNextFieldOffsetInBytes < FieldOffsetInBytes) {
    // Even with alignment, the field offset is not at the right place,
    // insert padding.
    uint64_t PaddingInBytes = FieldOffsetInBytes - NextFieldOffsetInBytes;

    AppendBytes(PaddingInBytes);
  }

  // Now append the field.
  LLVMFields.push_back(LLVMFieldInfo(D, FieldTypes.size()));
  AppendField(FieldOffsetInBytes, Ty);

  return true;
}

const llvm::Type *
CGRecordLayoutBuilder::LayoutUnionField(const FieldDecl *Field,
                                        const ASTRecordLayout &Layout) {
  if (Field->isBitField()) {
    uint64_t FieldSize =
      Field->getBitWidth()->EvaluateAsInt(Types.getContext()).getZExtValue();

    // Ignore zero sized bit fields.
    if (FieldSize == 0)
      return 0;

    const llvm::Type *FieldTy = llvm::Type::getInt8Ty(Types.getLLVMContext());
    unsigned NumBytesToAppend =
      llvm::RoundUpToAlignment(FieldSize, 8) / 8;

    if (NumBytesToAppend > 1)
      FieldTy = llvm::ArrayType::get(FieldTy, NumBytesToAppend);

    // Add the bit field info.
    LLVMBitFields.push_back(
      LLVMBitFieldInfo(Field, CGBitFieldInfo::MakeInfo(Types, Field,
                                                       0, FieldSize)));
    return FieldTy;
  }

  // This is a regular union field.
  LLVMFields.push_back(LLVMFieldInfo(Field, 0));
  return Types.ConvertTypeForMemRecursive(Field->getType());
}

void CGRecordLayoutBuilder::LayoutUnion(const RecordDecl *D) {
  assert(D->isUnion() && "Can't call LayoutUnion on a non-union record!");

  const ASTRecordLayout &Layout = Types.getContext().getASTRecordLayout(D);

  const llvm::Type *Ty = 0;
  uint64_t Size = 0;
  unsigned Align = 0;

  bool HasOnlyZeroSizedBitFields = true;

  unsigned FieldNo = 0;
  for (RecordDecl::field_iterator Field = D->field_begin(),
       FieldEnd = D->field_end(); Field != FieldEnd; ++Field, ++FieldNo) {
    assert(Layout.getFieldOffset(FieldNo) == 0 &&
          "Union field offset did not start at the beginning of record!");
    const llvm::Type *FieldTy = LayoutUnionField(*Field, Layout);

    if (!FieldTy)
      continue;

    HasOnlyZeroSizedBitFields = false;

    unsigned FieldAlign = Types.getTargetData().getABITypeAlignment(FieldTy);
    uint64_t FieldSize = Types.getTargetData().getTypeAllocSize(FieldTy);

    if (FieldAlign < Align)
      continue;

    if (FieldAlign > Align || FieldSize > Size) {
      Ty = FieldTy;
      Align = FieldAlign;
      Size = FieldSize;
    }
  }

  // Now add our field.
  if (Ty) {
    AppendField(0, Ty);

    if (getTypeAlignment(Ty) > Layout.getAlignment() / 8) {
      // We need a packed struct.
      Packed = true;
      Align = 1;
    }
  }
  if (!Align) {
    assert(HasOnlyZeroSizedBitFields &&
           "0-align record did not have all zero-sized bit-fields!");
    Align = 1;
  }

  // Append tail padding.
  if (Layout.getSize() / 8 > Size)
    AppendPadding(Layout.getSize() / 8, Align);
}

void
CGRecordLayoutBuilder::LayoutVirtualBase(const CXXRecordDecl *BaseDecl,
                                         uint64_t BaseOffset) {
  // Ignore empty bases.
  if (BaseDecl->isEmpty())
    return;

  CheckZeroInitializable(BaseDecl);

  const ASTRecordLayout &Layout = 
    Types.getContext().getASTRecordLayout(BaseDecl);

  uint64_t NonVirtualSize = Layout.getNonVirtualSize();

  // FIXME: Actually use a better type than [sizeof(BaseDecl) x i8] when we can.
  AppendPadding(BaseOffset / 8, 1);
  
  // FIXME: Add the vbase field info.

  AppendBytes(NonVirtualSize / 8);

}

void CGRecordLayoutBuilder::LayoutNonVirtualBase(const CXXRecordDecl *BaseDecl,
                                                 uint64_t BaseOffset) {
  // Ignore empty bases.
  if (BaseDecl->isEmpty())
    return;

  CheckZeroInitializable(BaseDecl);
  
  const ASTRecordLayout &Layout = 
    Types.getContext().getASTRecordLayout(BaseDecl);

  uint64_t NonVirtualSize = Layout.getNonVirtualSize();

  // FIXME: Actually use a better type than [sizeof(BaseDecl) x i8] when we can.
  AppendPadding(BaseOffset / 8, 1);
  
  // Append the base field.
  LLVMNonVirtualBases.push_back(LLVMBaseInfo(BaseDecl, FieldTypes.size()));

  AppendBytes(NonVirtualSize / 8);
}

void
CGRecordLayoutBuilder::LayoutNonVirtualBases(const CXXRecordDecl *RD,
                                             const ASTRecordLayout &Layout) {
  const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();

  // Check if we need to add a vtable pointer.
  if (RD->isDynamicClass()) {
    if (!PrimaryBase) {
      const llvm::Type *FunctionType =
        llvm::FunctionType::get(llvm::Type::getInt32Ty(Types.getLLVMContext()),
                                /*isVarArg=*/true);
      const llvm::Type *VTableTy = FunctionType->getPointerTo();

      assert(NextFieldOffsetInBytes == 0 &&
             "VTable pointer must come first!");
      AppendField(NextFieldOffsetInBytes, VTableTy->getPointerTo());
    } else {
      if (!Layout.isPrimaryBaseVirtual())
        LayoutNonVirtualBase(PrimaryBase, 0);
      else
        LayoutVirtualBase(PrimaryBase, 0);
    }
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

    LayoutNonVirtualBase(BaseDecl, Layout.getBaseClassOffsetInBits(BaseDecl));
  }
}

void 
CGRecordLayoutBuilder::ComputeNonVirtualBaseType(const CXXRecordDecl *RD) {
  const ASTRecordLayout &Layout = Types.getContext().getASTRecordLayout(RD);

  uint64_t AlignedNonVirtualTypeSize =
    llvm::RoundUpToAlignment(Layout.getNonVirtualSize(),
                             Layout.getNonVirtualAlign()) / 8;
  
  
  // First check if we can use the same fields as for the complete class.
  if (AlignedNonVirtualTypeSize == Layout.getSize() / 8) {
    NonVirtualBaseTypeIsSameAsCompleteType = true;
    return;
  }

  NonVirtualBaseFieldTypes = FieldTypes;

  // Check if we need padding.
  uint64_t AlignedNextFieldOffset =
    llvm::RoundUpToAlignment(NextFieldOffsetInBytes, AlignmentAsLLVMStruct);

  assert(AlignedNextFieldOffset <= AlignedNonVirtualTypeSize && 
         "Size mismatch!");

  if (AlignedNonVirtualTypeSize == AlignedNextFieldOffset) {
    // We don't need any padding.
    return;
  }

  uint64_t NumBytes = AlignedNonVirtualTypeSize - AlignedNextFieldOffset;
  NonVirtualBaseFieldTypes.push_back(getByteArrayType(NumBytes));
}

bool CGRecordLayoutBuilder::LayoutFields(const RecordDecl *D) {
  assert(!D->isUnion() && "Can't call LayoutFields on a union!");
  assert(Alignment && "Did not set alignment!");

  const ASTRecordLayout &Layout = Types.getContext().getASTRecordLayout(D);

  const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(D);
  if (RD)
    LayoutNonVirtualBases(RD, Layout);

  unsigned FieldNo = 0;

  for (RecordDecl::field_iterator Field = D->field_begin(),
       FieldEnd = D->field_end(); Field != FieldEnd; ++Field, ++FieldNo) {
    if (!LayoutField(*Field, Layout.getFieldOffset(FieldNo))) {
      assert(!Packed &&
             "Could not layout fields even with a packed LLVM struct!");
      return false;
    }
  }

  // We've laid out the non-virtual bases and the fields, now compute the
  // non-virtual base field types.
  if (RD)
    ComputeNonVirtualBaseType(RD);
  
  if (RD) {
    RD->getIndirectPrimaryBases(IndirectPrimaryBases);
  }

  // FIXME: Lay out the virtual bases instead of just treating them as tail
  // padding.
  
  // Append tail padding if necessary.
  AppendTailPadding(Layout.getSize());

  return true;
}

void CGRecordLayoutBuilder::AppendTailPadding(uint64_t RecordSize) {
  assert(RecordSize % 8 == 0 && "Invalid record size!");

  uint64_t RecordSizeInBytes = RecordSize / 8;
  assert(NextFieldOffsetInBytes <= RecordSizeInBytes && "Size mismatch!");

  uint64_t AlignedNextFieldOffset =
    llvm::RoundUpToAlignment(NextFieldOffsetInBytes, AlignmentAsLLVMStruct);

  if (AlignedNextFieldOffset == RecordSizeInBytes) {
    // We don't need any padding.
    return;
  }

  unsigned NumPadBytes = RecordSizeInBytes - NextFieldOffsetInBytes;
  AppendBytes(NumPadBytes);
}

void CGRecordLayoutBuilder::AppendField(uint64_t FieldOffsetInBytes,
                                        const llvm::Type *FieldTy) {
  AlignmentAsLLVMStruct = std::max(AlignmentAsLLVMStruct,
                                   getTypeAlignment(FieldTy));

  uint64_t FieldSizeInBytes = Types.getTargetData().getTypeAllocSize(FieldTy);

  FieldTypes.push_back(FieldTy);

  NextFieldOffsetInBytes = FieldOffsetInBytes + FieldSizeInBytes;
  BitsAvailableInLastField = 0;
}

void CGRecordLayoutBuilder::AppendPadding(uint64_t FieldOffsetInBytes,
                                          unsigned FieldAlignment) {
  assert(NextFieldOffsetInBytes <= FieldOffsetInBytes &&
         "Incorrect field layout!");

  // Round up the field offset to the alignment of the field type.
  uint64_t AlignedNextFieldOffsetInBytes =
    llvm::RoundUpToAlignment(NextFieldOffsetInBytes, FieldAlignment);

  if (AlignedNextFieldOffsetInBytes < FieldOffsetInBytes) {
    // Even with alignment, the field offset is not at the right place,
    // insert padding.
    uint64_t PaddingInBytes = FieldOffsetInBytes - NextFieldOffsetInBytes;

    AppendBytes(PaddingInBytes);
  }
}

const llvm::Type *CGRecordLayoutBuilder::getByteArrayType(uint64_t NumBytes) {
  assert(NumBytes != 0 && "Empty byte array's aren't allowed.");

  const llvm::Type *Ty = llvm::Type::getInt8Ty(Types.getLLVMContext());
  if (NumBytes > 1)
    Ty = llvm::ArrayType::get(Ty, NumBytes);

  return Ty;
}

void CGRecordLayoutBuilder::AppendBytes(uint64_t NumBytes) {
  if (NumBytes == 0)
    return;

  // Append the padding field
  AppendField(NextFieldOffsetInBytes, getByteArrayType(NumBytes));
}

unsigned CGRecordLayoutBuilder::getTypeAlignment(const llvm::Type *Ty) const {
  if (Packed)
    return 1;

  return Types.getTargetData().getABITypeAlignment(Ty);
}

void CGRecordLayoutBuilder::CheckZeroInitializable(QualType T) {
  // This record already contains a member pointer.
  if (!IsZeroInitializable)
    return;

  // Can only have member pointers if we're compiling C++.
  if (!Types.getContext().getLangOptions().CPlusPlus)
    return;

  T = Types.getContext().getBaseElementType(T);

  if (const MemberPointerType *MPT = T->getAs<MemberPointerType>()) {
    if (!Types.getCXXABI().isZeroInitializable(MPT))
      IsZeroInitializable = false;
  } else if (const RecordType *RT = T->getAs<RecordType>()) {
    const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
    CheckZeroInitializable(RD);
  }
}

void CGRecordLayoutBuilder::CheckZeroInitializable(const CXXRecordDecl *RD) {
  // This record already contains a member pointer.
  if (!IsZeroInitializable)
    return;

  const CGRecordLayout &Layout = Types.getCGRecordLayout(RD);
  if (!Layout.isZeroInitializable())
    IsZeroInitializable = false;
}

CGRecordLayout *CodeGenTypes::ComputeRecordLayout(const RecordDecl *D) {
  CGRecordLayoutBuilder Builder(*this);

  Builder.Layout(D);

  const llvm::StructType *Ty = llvm::StructType::get(getLLVMContext(),
                                                     Builder.FieldTypes,
                                                     Builder.Packed);

  const llvm::StructType *BaseTy = 0;
  if (isa<CXXRecordDecl>(D)) {
    if (Builder.NonVirtualBaseTypeIsSameAsCompleteType)
      BaseTy = Ty;
    else if (!Builder.NonVirtualBaseFieldTypes.empty())
      BaseTy = llvm::StructType::get(getLLVMContext(), 
                                     Builder.NonVirtualBaseFieldTypes, 
                                     Builder.Packed);
  }

  CGRecordLayout *RL =
    new CGRecordLayout(Ty, BaseTy, Builder.IsZeroInitializable);

  // Add all the non-virtual base field numbers.
  RL->NonVirtualBaseFields.insert(Builder.LLVMNonVirtualBases.begin(),
                                  Builder.LLVMNonVirtualBases.end());

  // Add all the field numbers.
  RL->FieldInfo.insert(Builder.LLVMFields.begin(),
                       Builder.LLVMFields.end());

  // Add bitfield info.
  RL->BitFields.insert(Builder.LLVMBitFields.begin(),
                       Builder.LLVMBitFields.end());

  // Dump the layout, if requested.
  if (getContext().getLangOptions().DumpRecordLayouts) {
    llvm::errs() << "\n*** Dumping IRgen Record Layout\n";
    llvm::errs() << "Record: ";
    D->dump();
    llvm::errs() << "\nLayout: ";
    RL->dump();
  }

#ifndef NDEBUG
  // Verify that the computed LLVM struct size matches the AST layout size.
  const ASTRecordLayout &Layout = getContext().getASTRecordLayout(D);

  uint64_t TypeSizeInBits = Layout.getSize();
  assert(TypeSizeInBits == getTargetData().getTypeAllocSizeInBits(Ty) &&
         "Type size mismatch!");

  if (BaseTy) {
    uint64_t AlignedNonVirtualTypeSizeInBits =
      llvm::RoundUpToAlignment(Layout.getNonVirtualSize(),
                               Layout.getNonVirtualAlign());

    assert(AlignedNonVirtualTypeSizeInBits == 
           getTargetData().getTypeAllocSizeInBits(BaseTy) &&
           "Type size mismatch!");
  }
                                     
  // Verify that the LLVM and AST field offsets agree.
  const llvm::StructType *ST =
    dyn_cast<llvm::StructType>(RL->getLLVMType());
  const llvm::StructLayout *SL = getTargetData().getStructLayout(ST);

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

    const CGBitFieldInfo &Info = RL->getBitFieldInfo(FD);
    for (unsigned i = 0, e = Info.getNumComponents(); i != e; ++i) {
      const CGBitFieldInfo::AccessInfo &AI = Info.getComponent(i);

      // Verify that every component access is within the structure.
      uint64_t FieldOffset = SL->getElementOffsetInBits(AI.FieldIndex);
      uint64_t AccessBitOffset = FieldOffset + AI.FieldByteOffset * 8;
      assert(AccessBitOffset + AI.AccessWidth <= TypeSizeInBits &&
             "Invalid bit-field access (out of range)!");
    }
  }
#endif

  return RL;
}

void CGRecordLayout::print(llvm::raw_ostream &OS) const {
  OS << "<CGRecordLayout\n";
  OS << "  LLVMType:" << *LLVMType << "\n";
  if (NonVirtualBaseLLVMType)
    OS << "  NonVirtualBaseLLVMType:" << *NonVirtualBaseLLVMType << "\n"; 
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

void CGBitFieldInfo::print(llvm::raw_ostream &OS) const {
  OS << "<CGBitFieldInfo";
  OS << " Size:" << Size;
  OS << " IsSigned:" << IsSigned << "\n";

  OS.indent(4 + strlen("<CGBitFieldInfo"));
  OS << " NumComponents:" << getNumComponents();
  OS << " Components: [";
  if (getNumComponents()) {
    OS << "\n";
    for (unsigned i = 0, e = getNumComponents(); i != e; ++i) {
      const AccessInfo &AI = getComponent(i);
      OS.indent(8);
      OS << "<AccessInfo"
         << " FieldIndex:" << AI.FieldIndex
         << " FieldByteOffset:" << AI.FieldByteOffset
         << " FieldBitStart:" << AI.FieldBitStart
         << " AccessWidth:" << AI.AccessWidth << "\n";
      OS.indent(8 + strlen("<AccessInfo"));
      OS << " AccessAlignment:" << AI.AccessAlignment
         << " TargetBitOffset:" << AI.TargetBitOffset
         << " TargetBitWidth:" << AI.TargetBitWidth
         << ">\n";
    }
    OS.indent(4);
  }
  OS << "]>";
}

void CGBitFieldInfo::dump() const {
  print(llvm::errs());
}
