//===--- RecordLayout.h - Layout information for a struct/union -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the RecordLayout interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_LAYOUTINFO_H
#define LLVM_CLANG_AST_LAYOUTINFO_H

#include "llvm/Support/DataTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/DeclCXX.h"

namespace clang {
  class ASTContext;
  class FieldDecl;
  class RecordDecl;
  class CXXRecordDecl;

/// ASTRecordLayout -
/// This class contains layout information for one RecordDecl,
/// which is a struct/union/class.  The decl represented must be a definition,
/// not a forward declaration.
/// This class is also used to contain layout information for one
/// ObjCInterfaceDecl. FIXME - Find appropriate name.
/// These objects are managed by ASTContext.
class ASTRecordLayout {
  /// Size - Size of record in bits.
  uint64_t Size;

  /// DataSize - Size of record in bits without tail padding.
  uint64_t DataSize;

  /// FieldOffsets - Array of field offsets in bits.
  uint64_t *FieldOffsets;

  // Alignment - Alignment of record in bits.
  unsigned Alignment;

  // FieldCount - Number of fields.
  unsigned FieldCount;

  /// CXXRecordLayoutInfo - Contains C++ specific layout information.
  struct CXXRecordLayoutInfo {
    /// NonVirtualSize - The non-virtual size (in bits) of an object, which is
    /// the size of the object without virtual bases.
    uint64_t NonVirtualSize;

    /// NonVirtualAlign - The non-virtual alignment (in bits) of an object,
    /// which is the alignment of the object without virtual bases.
    uint64_t NonVirtualAlign;

    /// SizeOfLargestEmptySubobject - The size of the largest empty subobject
    /// (either a base or a member). Will be zero if the class doesn't contain
    /// any empty subobjects.
    CharUnits SizeOfLargestEmptySubobject;
    
    /// PrimaryBase - The primary base info for this record.
    llvm::PointerIntPair<const CXXRecordDecl *, 1, bool> PrimaryBase;
    
    /// FIXME: This should really use a SmallPtrMap, once we have one in LLVM :)
    typedef llvm::DenseMap<const CXXRecordDecl *, CharUnits> BaseOffsetsMapTy;
    
    /// BaseOffsets - Contains a map from base classes to their offset.
    BaseOffsetsMapTy BaseOffsets;

    /// VBaseOffsets - Contains a map from vbase classes to their offset.
    BaseOffsetsMapTy VBaseOffsets;
  };

  /// CXXInfo - If the record layout is for a C++ record, this will have
  /// C++ specific information about the record.
  CXXRecordLayoutInfo *CXXInfo;

  friend class ASTContext;

  ASTRecordLayout(ASTContext &Ctx, uint64_t size, unsigned alignment,
                  unsigned datasize, const uint64_t *fieldoffsets,
                  unsigned fieldcount);

  // Constructor for C++ records.
  typedef CXXRecordLayoutInfo::BaseOffsetsMapTy BaseOffsetsMapTy;
  ASTRecordLayout(ASTContext &Ctx,
                  uint64_t size, unsigned alignment, uint64_t datasize,
                  const uint64_t *fieldoffsets, unsigned fieldcount,
                  uint64_t nonvirtualsize, unsigned nonvirtualalign,
                  CharUnits SizeOfLargestEmptySubobject,
                  const CXXRecordDecl *PrimaryBase,
                  bool IsPrimaryBaseVirtual,
                  const BaseOffsetsMapTy& BaseOffsets,
                  const BaseOffsetsMapTy& VBaseOffsets);

  ~ASTRecordLayout() {}

  void Destroy(ASTContext &Ctx);
  
  ASTRecordLayout(const ASTRecordLayout&);   // DO NOT IMPLEMENT
  void operator=(const ASTRecordLayout&); // DO NOT IMPLEMENT
public:

  /// getAlignment - Get the record alignment in bits.
  unsigned getAlignment() const { return Alignment; }

  /// getSize - Get the record size in bits.
  uint64_t getSize() const { return Size; }

  /// getFieldCount - Get the number of fields in the layout.
  unsigned getFieldCount() const { return FieldCount; }

  /// getFieldOffset - Get the offset of the given field index, in
  /// bits.
  uint64_t getFieldOffset(unsigned FieldNo) const {
    assert (FieldNo < FieldCount && "Invalid Field No");
    return FieldOffsets[FieldNo];
  }

  /// getDataSize() - Get the record data size, which is the record size
  /// without tail padding, in bits.
  uint64_t getDataSize() const {
    return DataSize;
  }

  /// getNonVirtualSize - Get the non-virtual size (in bits) of an object,
  /// which is the size of the object without virtual bases.
  uint64_t getNonVirtualSize() const {
    assert(CXXInfo && "Record layout does not have C++ specific info!");

    return CXXInfo->NonVirtualSize;
  }

  /// getNonVirtualSize - Get the non-virtual alignment (in bits) of an object,
  /// which is the alignment of the object without virtual bases.
  unsigned getNonVirtualAlign() const {
    assert(CXXInfo && "Record layout does not have C++ specific info!");

    return CXXInfo->NonVirtualAlign;
  }

  /// getPrimaryBase - Get the primary base for this record.
  const CXXRecordDecl *getPrimaryBase() const {
    assert(CXXInfo && "Record layout does not have C++ specific info!");

    return CXXInfo->PrimaryBase.getPointer();
  }

  /// isPrimaryBaseVirtual - Get whether the primary base for this record
  /// is virtual or not.
  bool isPrimaryBaseVirtual() const {
    assert(CXXInfo && "Record layout does not have C++ specific info!");

    return CXXInfo->PrimaryBase.getInt();
  }

  /// getBaseClassOffset - Get the offset, in chars, for the given base class.
  CharUnits getBaseClassOffset(const CXXRecordDecl *Base) const {
    assert(CXXInfo && "Record layout does not have C++ specific info!");
    assert(CXXInfo->BaseOffsets.count(Base) && "Did not find base!");

    return CXXInfo->BaseOffsets[Base];
  }

  /// getVBaseClassOffset - Get the offset, in chars, for the given base class.
  CharUnits getVBaseClassOffset(const CXXRecordDecl *VBase) const {
    assert(CXXInfo && "Record layout does not have C++ specific info!");
    assert(CXXInfo->VBaseOffsets.count(VBase) && "Did not find base!");

    return CXXInfo->VBaseOffsets[VBase];
  }

  /// getBaseClassOffsetInBits - Get the offset, in bits, for the given
  /// base class.
  uint64_t getBaseClassOffsetInBits(const CXXRecordDecl *Base) const {
    assert(CXXInfo && "Record layout does not have C++ specific info!");
    assert(CXXInfo->BaseOffsets.count(Base) && "Did not find base!");

    return getBaseClassOffset(Base).getQuantity() *
      Base->getASTContext().getCharWidth();
  }

  /// getVBaseClassOffsetInBits - Get the offset, in bits, for the given
  /// base class.
  uint64_t getVBaseClassOffsetInBits(const CXXRecordDecl *VBase) const {
    assert(CXXInfo && "Record layout does not have C++ specific info!");
    assert(CXXInfo->VBaseOffsets.count(VBase) && "Did not find base!");

    return getVBaseClassOffset(VBase).getQuantity() *
      VBase->getASTContext().getCharWidth();
  }

  CharUnits getSizeOfLargestEmptySubobject() const {
    assert(CXXInfo && "Record layout does not have C++ specific info!");
    return CXXInfo->SizeOfLargestEmptySubobject;
  }
};

}  // end namespace clang

#endif
