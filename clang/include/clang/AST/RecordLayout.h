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

/// ASTRecordLayout - 
/// This class contains layout information for one RecordDecl,
/// which is a struct/union/class.  The decl represented must be a definition,
/// not a forward declaration.  
/// This class is also used to contain layout informaiton for one 
/// ObjCInterfaceDecl. FIXME - Find appropriate name.
/// These objects are managed by ASTContext.
class ASTRecordLayout {
  uint64_t Size;        // Size of record in bits.
  unsigned Alignment;   // Alignment of record in bits.
  uint64_t *FieldOffsets;
  friend class ASTContext;
  
  ASTRecordLayout() : Size(0), Alignment(8) {}
  ~ASTRecordLayout() {
    delete [] FieldOffsets;
  }

  /// Initialize record layout. N is the number of fields in this record.
  void InitializeLayout(unsigned N) {
    FieldOffsets = new uint64_t[N];
  }
  
  /// Finalize record layout. Adjust record size based on the alignment.
  void FinalizeLayout() {
    // Finally, round the size of the record up to the alignment of the
    // record itself.
    Size = (Size + (Alignment-1)) & ~(Alignment-1);
  }

  void SetAlignment(unsigned A) {  Alignment = A; }

  /// LayoutField - Field layout.
  void LayoutField(const FieldDecl *FD, unsigned FieldNo,
                   bool IsUnion, bool StructIsPacked,
                   ASTContext &Context);
  
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
