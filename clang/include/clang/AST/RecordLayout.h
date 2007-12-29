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
  class RecordDecl;

/// ASTRecordLayout - This class contains layout information for one RecordDecl,
/// which is a struct/union/class.  The decl represented must be a definition,
/// not a forward declaration.  These objects are managed by ASTContext.
class ASTRecordLayout {
  uint64_t Size;        // Size of record in bits.
  unsigned Alignment;   // Alignment of record in bits.
  uint64_t *FieldOffsets;
  friend class ASTContext;
  
  ASTRecordLayout() {}
  ~ASTRecordLayout() {
    delete [] FieldOffsets;
  }
  
  void SetLayout(uint64_t size, unsigned alignment, uint64_t *fieldOffsets) {
    Size = size; Alignment = alignment;
    FieldOffsets = fieldOffsets;
  }
  
  ASTRecordLayout(const ASTRecordLayout&);   // DO NOT IMPLEMENT
  void operator=(const ASTRecordLayout&); // DO NOT IMPLEMENT
public:
  
  unsigned getAlignment() const { return Alignment; }
  uint64_t getSize() const { return Size; }
  
  uint64_t getFieldOffset(unsigned FieldNo) const {
    return FieldOffsets[FieldNo];
  }
    
};

}  // end namespace clang

#endif
