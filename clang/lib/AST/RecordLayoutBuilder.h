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
#include <map>

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

  /// Packed - Whether the record is packed or not.
  bool Packed;
  
  /// MaxFieldAlignment - The maximum allowed field alignment. This is set by
  /// #pragma pack. 
  unsigned MaxFieldAlignment;
  
  /// DataSize - The data size of the record being laid out.
  uint64_t DataSize;
  
  bool IsUnion;

  uint64_t NonVirtualSize;
  unsigned NonVirtualAlignment;
  const CXXRecordDecl *PrimaryBase;
  bool PrimaryBaseWasVirtual;

  typedef llvm::SmallVector<std::pair<const CXXRecordDecl *, 
                                      uint64_t>, 4> BaseOffsetsTy;
  
  /// Bases - base classes and their offsets from the record.
  BaseOffsetsTy Bases;
  
  // VBases - virtual base classes and their offsets from the record.
  BaseOffsetsTy VBases;

  /// IndirectPrimaryBases - Virtual base classes, direct or indirect, that are
  /// primary base classes for some other direct or indirect base class.
  llvm::SmallSet<const CXXRecordDecl*, 32> IndirectPrimaryBases;
  
  /// EmptyClassOffsets - A map from offsets to empty record decls.
  typedef std::multimap<uint64_t, const CXXRecordDecl *> EmptyClassOffsetsTy;
  EmptyClassOffsetsTy EmptyClassOffsets;
  
  ASTRecordLayoutBuilder(ASTContext &Ctx);

  void Layout(const RecordDecl *D);
  void Layout(const CXXRecordDecl *D);
  void Layout(const ObjCInterfaceDecl *D,
              const ObjCImplementationDecl *Impl);

  void LayoutFields(const RecordDecl *D);
  void LayoutField(const FieldDecl *D);

  void SelectPrimaryBase(const CXXRecordDecl *RD);
  void SelectPrimaryVBase(const CXXRecordDecl *RD,
                          const CXXRecordDecl *&FirstPrimary);
  
  /// IdentifyPrimaryBases - Identify all virtual base classes, direct or 
  /// indirect, that are primary base classes for some other direct or indirect 
  /// base class.
  void IdentifyPrimaryBases(const CXXRecordDecl *RD);
  
  void setPrimaryBase(const CXXRecordDecl *PB, bool Virtual) {
    PrimaryBase = PB;
    PrimaryBaseWasVirtual = Virtual;
  }
  
  bool IsNearlyEmpty(const CXXRecordDecl *RD) const;
  
  /// LayoutBase - Will lay out a base and return the offset where it was 
  /// placed, in bits.
  uint64_t LayoutBase(const CXXRecordDecl *RD);
  
  void LayoutVtable(const CXXRecordDecl *RD);
  void LayoutNonVirtualBases(const CXXRecordDecl *RD);
  void LayoutBaseNonVirtually(const CXXRecordDecl *RD, bool IsVBase);
  void LayoutVirtualBase(const CXXRecordDecl *RD);
  void LayoutVirtualBases(const CXXRecordDecl *RD, const CXXRecordDecl *PB,
                          int64_t Offset,
                                 llvm::SmallSet<const CXXRecordDecl*, 32> &mark,
                     llvm::SmallSet<const CXXRecordDecl*, 32> &IndirectPrimary);

  /// canPlaceRecordAtOffset - Return whether a record (either a base class
  /// or a field) can be placed at the given offset. 
  /// Returns false if placing the record will result in two components 
  /// (direct or indirect) of the same type having the same offset.
  bool canPlaceRecordAtOffset(const CXXRecordDecl *RD, uint64_t Offset) const;

  /// canPlaceFieldAtOffset - Return whether a field can be placed at the given
  /// offset.
  bool canPlaceFieldAtOffset(const FieldDecl *FD, uint64_t Offset) const;

  /// UpdateEmptyClassOffsets - Called after a record (either a base class
  /// or a field) has been placed at the given offset. Will update the
  /// EmptyClassOffsets map if the class is empty or has any empty bases or
  /// fields.
  void UpdateEmptyClassOffsets(const CXXRecordDecl *RD, uint64_t Offset);

  /// UpdateEmptyClassOffsets - Called after a field has been placed at the 
  /// given offset.
  void UpdateEmptyClassOffsets(const FieldDecl *FD, uint64_t Offset);
  
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

