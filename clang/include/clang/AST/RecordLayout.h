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

#include "llvm/System/DataTypes.h"
#include "llvm/ADT/DenseMap.h"
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

public:
  /// PrimaryBaseInfo - Contains info about a primary base.
  struct PrimaryBaseInfo {
    PrimaryBaseInfo() {}

    PrimaryBaseInfo(const CXXRecordDecl *Base, bool IsVirtual)
      : Value(Base, Base && IsVirtual) {}

    /// Value - Points to the primary base. The single-bit value
    /// will be non-zero when the primary base is virtual.
    llvm::PointerIntPair<const CXXRecordDecl *, 1, bool> Value;
    
    /// getBase - Returns the primary base.
    const CXXRecordDecl *getBase() const { return Value.getPointer(); }
  
    /// isVirtual - Returns whether the primary base is virtual or not.
    bool isVirtual() const { return Value.getInt(); }

    friend bool operator==(const PrimaryBaseInfo &X, const PrimaryBaseInfo &Y) {
      return X.Value == Y.Value;
    }
  }; 
  
  /// primary_base_info_iterator - An iterator for iterating the primary base
  /// class chain.
  class primary_base_info_iterator {
    /// Current - The current base class info.
    PrimaryBaseInfo Current;
    
  public:
    primary_base_info_iterator() {}
    primary_base_info_iterator(PrimaryBaseInfo Info) : Current(Info) {}

    const PrimaryBaseInfo &operator*() const { return Current; }

    primary_base_info_iterator& operator++() {
      const CXXRecordDecl *RD = Current.getBase();
      Current = RD->getASTContext().getASTRecordLayout(RD).getPrimaryBaseInfo();
      return *this;
    }

    friend bool operator==(const primary_base_info_iterator &X,
                           const primary_base_info_iterator &Y) {
      return X.Current == Y.Current;
    }
    friend bool operator!=(const primary_base_info_iterator &X,
                           const primary_base_info_iterator &Y) {
      return !(X == Y);
    }
  };
    
private:
  /// CXXRecordLayoutInfo - Contains C++ specific layout information.
  struct CXXRecordLayoutInfo {
    /// NonVirtualSize - The non-virtual size (in bits) of an object, which is
    /// the size of the object without virtual bases.
    uint64_t NonVirtualSize;

    /// NonVirtualAlign - The non-virtual alignment (in bits) of an object,
    /// which is the alignment of the object without virtual bases.
    uint64_t NonVirtualAlign;

    /// PrimaryBase - The primary base info for this record.
    PrimaryBaseInfo PrimaryBase;
    
    /// FIXME: This should really use a SmallPtrMap, once we have one in LLVM :)
    typedef llvm::DenseMap<const CXXRecordDecl *, uint64_t> BaseOffsetsMapTy;
    
    /// BaseOffsets - Contains a map from base classes to their offset.
    BaseOffsetsMapTy BaseOffsets;

    /// VBaseOffsets - Contains a map from vbase classes to their offset.
    BaseOffsetsMapTy VBaseOffsets;
  };

  /// CXXInfo - If the record layout is for a C++ record, this will have
  /// C++ specific information about the record.
  CXXRecordLayoutInfo *CXXInfo;

  friend class ASTContext;
  friend class ASTRecordLayoutBuilder;

  ASTRecordLayout(ASTContext &Ctx, uint64_t size, unsigned alignment,
                  unsigned datasize, const uint64_t *fieldoffsets,
                  unsigned fieldcount);

  // Constructor for C++ records.
  typedef CXXRecordLayoutInfo::BaseOffsetsMapTy BaseOffsetsMapTy;
  ASTRecordLayout(ASTContext &Ctx,
                  uint64_t size, unsigned alignment, uint64_t datasize,
                  const uint64_t *fieldoffsets, unsigned fieldcount,
                  uint64_t nonvirtualsize, unsigned nonvirtualalign,
                  const PrimaryBaseInfo &PrimaryBase,
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

  /// getPrimaryBaseInfo - Get the primary base info.
  const PrimaryBaseInfo &getPrimaryBaseInfo() const {
    assert(CXXInfo && "Record layout does not have C++ specific info!");

    return CXXInfo->PrimaryBase;
  }

  // FIXME: Migrate off of this function and use getPrimaryBaseInfo directly.
  const CXXRecordDecl *getPrimaryBase() const {
    return getPrimaryBaseInfo().getBase();
  }

  // FIXME: Migrate off of this function and use getPrimaryBaseInfo directly.
  bool getPrimaryBaseWasVirtual() const {
    return getPrimaryBaseInfo().isVirtual();
  }

  /// getBaseClassOffset - Get the offset, in bits, for the given base class.
  uint64_t getBaseClassOffset(const CXXRecordDecl *Base) const {
    assert(CXXInfo && "Record layout does not have C++ specific info!");
    assert(CXXInfo->BaseOffsets.count(Base) && "Did not find base!");

    return CXXInfo->BaseOffsets[Base];
  }

  /// getVBaseClassOffset - Get the offset, in bits, for the given base class.
  uint64_t getVBaseClassOffset(const CXXRecordDecl *VBase) const {
    assert(CXXInfo && "Record layout does not have C++ specific info!");
    assert(CXXInfo->VBaseOffsets.count(VBase) && "Did not find base!");

    return CXXInfo->VBaseOffsets[VBase];
  }
  
  primary_base_info_iterator primary_base_begin() const {
    assert(CXXInfo && "Record layout does not have C++ specific info!");
  
    return primary_base_info_iterator(getPrimaryBaseInfo());
  }

  primary_base_info_iterator primary_base_end() const {
    assert(CXXInfo && "Record layout does not have C++ specific info!");
    
    return primary_base_info_iterator();
  }
};

}  // end namespace clang

#endif
