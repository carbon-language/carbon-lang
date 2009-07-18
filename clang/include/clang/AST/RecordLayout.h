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

/// ASTRecordLayout - 
/// This class contains layout information for one RecordDecl,
/// which is a struct/union/class.  The decl represented must be a definition,
/// not a forward declaration.  
/// This class is also used to contain layout information for one 
/// ObjCInterfaceDecl. FIXME - Find appropriate name.
/// These objects are managed by ASTContext.
class ASTRecordLayout {
  uint64_t Size;        // Size of record in bits.
  uint64_t DataSize;    // Size of record in bits without tail padding.
  uint64_t *FieldOffsets;
  unsigned Alignment;   // Alignment of record in bits.
  unsigned FieldCount;  // Number of fields
  friend class ASTContext;
  friend class ASTRecordLayoutBuilder;

  ASTRecordLayout(uint64_t size, unsigned alignment, unsigned datasize,
                  const uint64_t *fieldoffsets, unsigned fieldcount) 
  : Size(size), DataSize(datasize), FieldOffsets(0), Alignment(alignment),
    FieldCount(fieldcount) {
    if (FieldCount > 0)  {
      FieldOffsets = new uint64_t[FieldCount];
      for (unsigned i = 0; i < FieldCount; ++i)
        FieldOffsets[i] = fieldoffsets[i];
    }
  }
  ~ASTRecordLayout() {
    delete [] FieldOffsets;
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
};

}  // end namespace clang

#endif
