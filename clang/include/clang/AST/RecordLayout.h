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

  struct CXXRecordLayoutInfo {
    /// NonVirtualSize - The non-virtual size (in bits) of an object, which is 
    /// the size of the object without virtual bases.
    uint64_t NonVirtualSize;
    
    /// NonVirtualAlign - The non-virtual alignment (in bits) of an object,
    /// which is the alignment of the object without virtual bases.
    uint64_t NonVirtualAlign;
    
    /// PrimaryBase - The primary base for our vtable.
    const CXXRecordDecl *PrimaryBase;
    /// PrimaryBase - Wether or not the primary base was a virtual base.
    bool PrimaryBaseWasVirtual;

    /// BaseOffsets - Contains a map from base classes to their offset.
    /// FIXME: Does it make sense to store offsets for virtual base classes
    /// here?
    /// FIXME: This should really use a SmallPtrMap, once we have one in LLVM :)
    llvm::DenseMap<const CXXRecordDecl *, uint64_t> BaseOffsets;
  };
  
  /// CXXInfo - If the record layout is for a C++ record, this will have 
  /// C++ specific information about the record.
  CXXRecordLayoutInfo *CXXInfo;
  
  friend class ASTContext;
  friend class ASTRecordLayoutBuilder;

  ASTRecordLayout(uint64_t size, unsigned alignment, unsigned datasize,
                  const uint64_t *fieldoffsets, unsigned fieldcount) 
  : Size(size), DataSize(datasize), FieldOffsets(0), Alignment(alignment),
    FieldCount(fieldcount), CXXInfo(0) {
    if (FieldCount > 0)  {
      FieldOffsets = new uint64_t[FieldCount];
      for (unsigned i = 0; i < FieldCount; ++i)
        FieldOffsets[i] = fieldoffsets[i];
    }
  }
  
  // Constructor for C++ records.
  ASTRecordLayout(uint64_t size, unsigned alignment, uint64_t datasize,
                  const uint64_t *fieldoffsets, unsigned fieldcount,
                  uint64_t nonvirtualsize, unsigned nonvirtualalign,
                  const CXXRecordDecl *PB, bool PBVirtual,
                  const CXXRecordDecl **bases, const uint64_t *baseoffsets,
                  unsigned basecount)
  : Size(size), DataSize(datasize), FieldOffsets(0), Alignment(alignment),
  FieldCount(fieldcount), CXXInfo(new CXXRecordLayoutInfo) {
    if (FieldCount > 0)  {
      FieldOffsets = new uint64_t[FieldCount];
      for (unsigned i = 0; i < FieldCount; ++i)
        FieldOffsets[i] = fieldoffsets[i];
    }
    
    CXXInfo->PrimaryBase = PB;
    CXXInfo->PrimaryBaseWasVirtual = PBVirtual;
    CXXInfo->NonVirtualSize = nonvirtualsize;
    CXXInfo->NonVirtualAlign = nonvirtualalign;
    for (unsigned i = 0; i != basecount; ++i)
      CXXInfo->BaseOffsets[bases[i]] = baseoffsets[i];
  }
  
  ~ASTRecordLayout() {
    delete [] FieldOffsets;
    delete CXXInfo;
  }

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
  
  /// getPrimaryBase - Get the primary base.
  const CXXRecordDecl *getPrimaryBase() const {
    assert(CXXInfo && "Record layout does not have C++ specific info!");
    
    return CXXInfo->PrimaryBase;
  }
  /// getPrimaryBaseWasVirtual - Indicates if the primary base was virtual.
  bool getPrimaryBaseWasVirtual() const {
    assert(CXXInfo && "Record layout does not have C++ specific info!");
    
    return CXXInfo->PrimaryBaseWasVirtual;
  }

  /// getBaseClassOffset - Get the offset, in bits, for the given base class.
  uint64_t getBaseClassOffset(const CXXRecordDecl *Base) const {
    assert(CXXInfo && "Record layout does not have C++ specific info!");
    assert(CXXInfo->BaseOffsets.count(Base) && "Did not find base!");
    
    return CXXInfo->BaseOffsets[Base];
  }
};

}  // end namespace clang

#endif
