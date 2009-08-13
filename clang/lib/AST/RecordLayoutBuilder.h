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
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/DataTypes.h"

namespace clang {
  class ASTContext;
  class ASTRecordLayout;
  class CXXRecordDecl;
  class FieldDecl;
  class ObjCImplementationDecl;
  class ObjCInterfaceDecl;
  class RecordDecl;
  
class ASTRecordLayoutBuilder {
  ASTContext &Ctx;

  uint64_t Size;
  unsigned Alignment;
  llvm::SmallVector<uint64_t, 16> FieldOffsets;
  
  bool Packed;
  unsigned MaxFieldAlignment;
  uint64_t NextOffset;
  bool IsUnion;
  
  uint64_t NonVirtualSize;
  unsigned NonVirtualAlignment;
  const CXXRecordDecl *PrimaryBase;
  bool PrimaryBaseWasVirtual;

  llvm::SmallVector<const CXXRecordDecl *, 4> Bases;
  llvm::SmallVector<uint64_t, 4> BaseOffsets;
  
  ASTRecordLayoutBuilder(ASTContext &Ctx);
  
  void Layout(const RecordDecl *D);
  void Layout(const CXXRecordDecl *D);
  void Layout(const ObjCInterfaceDecl *D,
              const ObjCImplementationDecl *Impl);

  void LayoutFields(const RecordDecl *D);
  void LayoutField(const FieldDecl *D);

  void SelectPrimaryBase(const CXXRecordDecl *RD,
                     llvm::SmallSet<const CXXRecordDecl*, 32> &IndirectPrimary);
  void SelectPrimaryVBase(const CXXRecordDecl *RD,
                          const CXXRecordDecl *&FirstPrimary,
                     llvm::SmallSet<const CXXRecordDecl*, 32> &IndirectPrimary);
  void SelectPrimaryForBase(const CXXRecordDecl *RD,
                     llvm::SmallSet<const CXXRecordDecl*, 32> &IndirectPrimary);
  void setPrimaryBase(const CXXRecordDecl *PB, bool Virtual) {
    PrimaryBase = PB;
    PrimaryBaseWasVirtual = Virtual;
  }
  bool IsNearlyEmpty(const CXXRecordDecl *RD);
  void LayoutVtable(const CXXRecordDecl *RD,
                     llvm::SmallSet<const CXXRecordDecl*, 32> &IndirectPrimary);
  void LayoutNonVirtualBases(const CXXRecordDecl *RD);
  void LayoutBaseNonVirtually(const CXXRecordDecl *RD);
  void LayoutVirtualBase(const CXXRecordDecl *RD);
  void LayoutVirtualBases(const CXXRecordDecl *RD,
                     llvm::SmallSet<const CXXRecordDecl*, 32> &IndirectPrimary);
  
  /// FinishLayout - Finalize record layout. Adjust record size based on the
  /// alignment.
  void FinishLayout();
  
  void UpdateAlignment(unsigned NewAlignment);

  ASTRecordLayoutBuilder(const ASTRecordLayoutBuilder&);   // DO NOT IMPLEMENT
  void operator=(const ASTRecordLayoutBuilder&); // DO NOT IMPLEMENT
public:
  static const ASTRecordLayout *ComputeLayout(ASTContext &Ctx, 
                                              const RecordDecl *RD);
  static const ASTRecordLayout *ComputeLayout(ASTContext &Ctx,
                                              const ObjCInterfaceDecl *D,
                                            const ObjCImplementationDecl *Impl);
};
  
} // end namespace clang

#endif

