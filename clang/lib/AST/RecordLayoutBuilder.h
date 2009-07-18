//===- ASTRecordLayoutBuilder.h - Helper class for building record layouts ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_RECORDLAYOUTBUILDER_H
#define LLVM_CLANG_AST_RECORDLAYOUTBUILDER_H

#include "llvm/ADT/SmallVector.h"

namespace clang {
  class ASTContext;
  class ASTRecordLayout;
  class FieldDecl;
  class RecordDecl;
  
class ASTRecordLayoutBuilder {
  ASTContext &Ctx;

  uint64_t Size;
  uint64_t Alignment;
  llvm::SmallVector<uint64_t, 16> FieldOffsets;
  
  unsigned StructPacking;
  unsigned NextOffset;
  bool IsUnion;
  
  ASTRecordLayoutBuilder(ASTContext &Ctx);
  
  void Layout(const RecordDecl *D);
  void LayoutField(const FieldDecl *D);
  void FinishLayout();
  
  void UpdateAlignment(unsigned NewAlignment);

  ASTRecordLayoutBuilder(const ASTRecordLayoutBuilder&);   // DO NOT IMPLEMENT
  void operator=(const ASTRecordLayoutBuilder&); // DO NOT IMPLEMENT
public:
  static const ASTRecordLayout *ComputeLayout(ASTContext &Ctx, 
                                              const RecordDecl *RD);
  
};
  
} // end namespace clang

#endif

