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
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecordLayout.h"
#include "CodeGenTypes.h"
#include "llvm/Type.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Target/TargetData.h"
using namespace clang;
using namespace CodeGen;

namespace clang {
namespace CodeGen {

class CGRecordLayoutBuilder {
public:
  /// FieldTypes - Holds the LLVM types that the struct is created from.
  std::vector<const llvm::Type *> FieldTypes;

  /// LLVMFieldInfo - Holds a field and its corresponding LLVM field number.
  typedef std::pair<const FieldDecl *, unsigned> LLVMFieldInfo;
  llvm::SmallVector<LLVMFieldInfo, 16> LLVMFields;

  /// LLVMBitFieldInfo - Holds location and size information about a bit field.
  typedef std::pair<const FieldDecl *, CGBitFieldInfo> LLVMBitFieldInfo;
  llvm::SmallVector<LLVMBitFieldInfo, 16> LLVMBitFields;

  /// ContainsPointerToDataMember - Whether one of the fields in this record
  /// layout is a pointer to data member, or a struct that contains pointer to
  /// data member.
  bool ContainsPointerToDataMember;

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

  /// LayoutUnion - Will layout a union RecordDecl.
  void LayoutUnion(const RecordDecl *D);

  /// LayoutField - try to layout all fields in the record decl.
  /// Returns false if the operation failed because the struct is not packed.
  bool LayoutFields(const RecordDecl *D);

  /// LayoutBases - layout the bases and vtable pointer of a record decl.
  void LayoutBases(const CXXRecordDecl *RD, const ASTRecordLayout &Layout);

  /// LayoutField - layout a single field. Returns false if the operation failed
  /// because the current struct is not packed.
  bool LayoutField(const FieldDecl *D, uint64_t FieldOffset);

  /// LayoutBitField - layout a single bit field.
  void LayoutBitField(const FieldDecl *D, uint64_t FieldOffset);

  /// AppendField - Appends a field with the given offset and type.
  void AppendField(uint64_t FieldOffsetInBytes, const llvm::Type *FieldTy);

  /// AppendPadding - Appends enough padding bytes so that the total struct
  /// size matches the alignment of the passed in type.
  void AppendPadding(uint64_t FieldOffsetInBytes, const llvm::Type *FieldTy);

  /// AppendPadding - Appends enough padding bytes so that the total
  /// struct size is a multiple of the field alignment.
  void AppendPadding(uint64_t FieldOffsetInBytes, unsigned FieldAlignment);

  /// AppendBytes - Append a given number of bytes to the record.
  void AppendBytes(uint64_t NumBytes);

  /// AppendTailPadding - Append enough tail padding so that the type will have
  /// the passed size.
  void AppendTailPadding(uint64_t RecordSize);

  unsigned getTypeAlignment(const llvm::Type *Ty) const;
  uint64_t getTypeSizeInBytes(const llvm::Type *Ty) const;

  /// CheckForPointerToDataMember - Check if the given type contains a pointer
  /// to data member.
  void CheckForPointerToDataMember(QualType T);

public:
  CGRecordLayoutBuilder(CodeGenTypes &Types)
    : ContainsPointerToDataMember(false), Packed(false), Types(Types),
      Alignment(0), AlignmentAsLLVMStruct(1),
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

  LayoutFields(D);
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

  const llvm::Type *Ty = Types.ConvertTypeForMemRecursive(D->getType());
  uint64_t TypeSizeInBits = getTypeSizeInBytes(Ty) * 8;

  bool IsSigned = D->getType()->isSignedIntegerType();
  LLVMBitFields.push_back(LLVMBitFieldInfo(
                            D, CGBitFieldInfo(FieldOffset / TypeSizeInBits,
                                              FieldOffset % TypeSizeInBits,
                                              FieldSize, IsSigned)));

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

  // Check if we have a pointer to data member in this field.
  CheckForPointerToDataMember(D->getType());

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
    if (const PragmaPackAttr *PPA = RD->getAttr<PragmaPackAttr>()) {
      if (PPA->getAlignment() != TypeAlignment * 8 && !Packed)
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

    if (Field->isBitField()) {
      uint64_t FieldSize =
        Field->getBitWidth()->EvaluateAsInt(Types.getContext()).getZExtValue();

      // Ignore zero sized bit fields.
      if (FieldSize == 0)
        continue;

      // Add the bit field info.
      bool IsSigned = Field->getType()->isSignedIntegerType();
      LLVMBitFields.push_back(LLVMBitFieldInfo(
                                *Field, CGBitFieldInfo(0, 0, FieldSize,
                                                       IsSigned)));
    } else {
      LLVMFields.push_back(LLVMFieldInfo(*Field, 0));
    }

    HasOnlyZeroSizedBitFields = false;

    const llvm::Type *FieldTy =
      Types.ConvertTypeForMemRecursive(Field->getType());
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

void CGRecordLayoutBuilder::LayoutBases(const CXXRecordDecl *RD,
                                        const ASTRecordLayout &Layout) {
  // Check if we need to add a vtable pointer.
  if (RD->isDynamicClass() && !Layout.getPrimaryBase()) {
    const llvm::Type *Int8PtrTy =
      llvm::Type::getInt8PtrTy(Types.getLLVMContext());

    assert(NextFieldOffsetInBytes == 0 &&
           "Vtable pointer must come first!");
    AppendField(NextFieldOffsetInBytes, Int8PtrTy->getPointerTo());
  }
}

bool CGRecordLayoutBuilder::LayoutFields(const RecordDecl *D) {
  assert(!D->isUnion() && "Can't call LayoutFields on a union!");
  assert(Alignment && "Did not set alignment!");

  const ASTRecordLayout &Layout = Types.getContext().getASTRecordLayout(D);

  if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(D))
    LayoutBases(RD, Layout);

  unsigned FieldNo = 0;

  for (RecordDecl::field_iterator Field = D->field_begin(),
       FieldEnd = D->field_end(); Field != FieldEnd; ++Field, ++FieldNo) {
    if (!LayoutField(*Field, Layout.getFieldOffset(FieldNo))) {
      assert(!Packed &&
             "Could not layout fields even with a packed LLVM struct!");
      return false;
    }
  }

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

  uint64_t FieldSizeInBytes = getTypeSizeInBytes(FieldTy);

  FieldTypes.push_back(FieldTy);

  NextFieldOffsetInBytes = FieldOffsetInBytes + FieldSizeInBytes;
  BitsAvailableInLastField = 0;
}

void
CGRecordLayoutBuilder::AppendPadding(uint64_t FieldOffsetInBytes,
                                     const llvm::Type *FieldTy) {
  AppendPadding(FieldOffsetInBytes, getTypeAlignment(FieldTy));
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

void CGRecordLayoutBuilder::AppendBytes(uint64_t NumBytes) {
  if (NumBytes == 0)
    return;

  const llvm::Type *Ty = llvm::Type::getInt8Ty(Types.getLLVMContext());
  if (NumBytes > 1)
    Ty = llvm::ArrayType::get(Ty, NumBytes);

  // Append the padding field
  AppendField(NextFieldOffsetInBytes, Ty);
}

unsigned CGRecordLayoutBuilder::getTypeAlignment(const llvm::Type *Ty) const {
  if (Packed)
    return 1;

  return Types.getTargetData().getABITypeAlignment(Ty);
}

uint64_t CGRecordLayoutBuilder::getTypeSizeInBytes(const llvm::Type *Ty) const {
  return Types.getTargetData().getTypeAllocSize(Ty);
}

void CGRecordLayoutBuilder::CheckForPointerToDataMember(QualType T) {
  // This record already contains a member pointer.
  if (ContainsPointerToDataMember)
    return;

  // Can only have member pointers if we're compiling C++.
  if (!Types.getContext().getLangOptions().CPlusPlus)
    return;

  T = Types.getContext().getBaseElementType(T);

  if (const MemberPointerType *MPT = T->getAs<MemberPointerType>()) {
    if (!MPT->getPointeeType()->isFunctionType()) {
      // We have a pointer to data member.
      ContainsPointerToDataMember = true;
    }
  } else if (const RecordType *RT = T->getAs<RecordType>()) {
    const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());

    // FIXME: It would be better if there was a way to explicitly compute the
    // record layout instead of converting to a type.
    Types.ConvertTagDeclType(RD);

    const CGRecordLayout &Layout = Types.getCGRecordLayout(RD);

    if (Layout.containsPointerToDataMember())
      ContainsPointerToDataMember = true;
  }
}

CGRecordLayout *CodeGenTypes::ComputeRecordLayout(const RecordDecl *D) {
  CGRecordLayoutBuilder Builder(*this);

  Builder.Layout(D);

  const llvm::Type *Ty = llvm::StructType::get(getLLVMContext(),
                                               Builder.FieldTypes,
                                               Builder.Packed);
  assert(getContext().getASTRecordLayout(D).getSize() / 8 ==
         getTargetData().getTypeAllocSize(Ty) &&
         "Type size mismatch!");

  CGRecordLayout *RL =
    new CGRecordLayout(Ty, Builder.ContainsPointerToDataMember);

  // Add all the field numbers.
  for (unsigned i = 0, e = Builder.LLVMFields.size(); i != e; ++i)
    RL->FieldInfo.insert(Builder.LLVMFields[i]);

  // Add bitfield info.
  for (unsigned i = 0, e = Builder.LLVMBitFields.size(); i != e; ++i)
    RL->BitFields.insert(Builder.LLVMBitFields[i]);

  return RL;
}
