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
/// This class is also used to contain layout information for one 
/// ObjCInterfaceDecl. FIXME - Find appropriate name.
/// These objects are managed by ASTContext.
class ASTRecordLayout {
  uint64_t Size;        // Size of record in bits.
  uint64_t NextOffset;  // Next available offset
  uint64_t *FieldOffsets;
  unsigned Alignment;   // Alignment of record in bits.
  unsigned FieldCount;  // Number of fields
  friend class ASTContext;

  ASTRecordLayout(uint64_t S = 0, unsigned A = 8) 
    : Size(S), NextOffset(S), Alignment(A), FieldCount(0) {}
  ~ASTRecordLayout() {
    delete [] FieldOffsets;
  }

  /// Initialize record layout. N is the number of fields in this record.
  void InitializeLayout(unsigned N) {
    FieldCount = N;
    FieldOffsets = new uint64_t[N];
  }
  
  /// Finalize record layout. Adjust record size based on the alignment.
  void FinalizeLayout() {
    // Finally, round the size of the record up to the alignment of the
    // record itself.
    Size = (Size + (Alignment-1)) & ~(Alignment-1);
  }

  void SetFieldOffset(unsigned FieldNo, uint64_t Offset) {
    assert (FieldNo < FieldCount && "Invalid Field No");
    FieldOffsets[FieldNo] = Offset;
  }

  void SetAlignment(unsigned A) {  Alignment = A; }

  /// LayoutField - Field layout. StructPacking is the specified
  /// packing alignment (maximum alignment) in bits to use for the
  /// structure, or 0 if no packing alignment is specified.
  void LayoutField(const FieldDecl *FD, unsigned FieldNo,
                   bool IsUnion, unsigned StructPacking,
                   ASTContext &Context);
  
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
    
  /// getNextOffset - Get the next available (unused) offset in the
  /// structure, in bits.
  uint64_t getNextOffset() const {
    return NextOffset;
  }
};

}  // end namespace clang

#endif
