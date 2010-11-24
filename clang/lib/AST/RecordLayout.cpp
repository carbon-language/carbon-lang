//===-- RecordLayout.cpp - Layout information for a struct/union -*- C++ -*-==//
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

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"

using namespace clang;

void ASTRecordLayout::Destroy(ASTContext &Ctx) {
  if (FieldOffsets)
    Ctx.Deallocate(FieldOffsets);
  if (CXXInfo) {
    Ctx.Deallocate(CXXInfo);
    CXXInfo->~CXXRecordLayoutInfo();
  }
  this->~ASTRecordLayout();
  Ctx.Deallocate(this);
}

ASTRecordLayout::ASTRecordLayout(ASTContext &Ctx, uint64_t size, unsigned alignment,
                unsigned datasize, const uint64_t *fieldoffsets,
                unsigned fieldcount)
  : Size(size), DataSize(datasize), FieldOffsets(0), Alignment(alignment),
    FieldCount(fieldcount), CXXInfo(0) {
  if (FieldCount > 0)  {
    FieldOffsets = new (Ctx) uint64_t[FieldCount];
    memcpy(FieldOffsets, fieldoffsets, FieldCount * sizeof(*FieldOffsets));
  }
}

// Constructor for C++ records.
ASTRecordLayout::ASTRecordLayout(ASTContext &Ctx,
                                 uint64_t size, unsigned alignment,
                                 uint64_t datasize,
                                 const uint64_t *fieldoffsets,
                                 unsigned fieldcount,
                                 uint64_t nonvirtualsize,
                                 unsigned nonvirtualalign,
                                 CharUnits SizeOfLargestEmptySubobject,
                                 const CXXRecordDecl *PrimaryBase,
                                 bool IsPrimaryBaseVirtual,
                                 const BaseOffsetsMapTy& BaseOffsets,
                                 const BaseOffsetsMapTy& VBaseOffsets)
  : Size(size), DataSize(datasize), FieldOffsets(0), Alignment(alignment),
    FieldCount(fieldcount), CXXInfo(new (Ctx) CXXRecordLayoutInfo)
{
  if (FieldCount > 0)  {
    FieldOffsets = new (Ctx) uint64_t[FieldCount];
    memcpy(FieldOffsets, fieldoffsets, FieldCount * sizeof(*FieldOffsets));
  }

  CXXInfo->PrimaryBase.setPointer(PrimaryBase);
  CXXInfo->PrimaryBase.setInt(IsPrimaryBaseVirtual);
  CXXInfo->NonVirtualSize = nonvirtualsize;
  CXXInfo->NonVirtualAlign = nonvirtualalign;
  CXXInfo->SizeOfLargestEmptySubobject = SizeOfLargestEmptySubobject;
  CXXInfo->BaseOffsets = BaseOffsets;
  CXXInfo->VBaseOffsets = VBaseOffsets;

#ifndef NDEBUG
    if (const CXXRecordDecl *PrimaryBase = getPrimaryBase()) {
      if (isPrimaryBaseVirtual())
        assert(getVBaseClassOffset(PrimaryBase).isZero() &&
               "Primary virtual base must be at offset 0!");
      else
        assert(getBaseClassOffsetInBits(PrimaryBase) == 0 &&
               "Primary base must be at offset 0!");
    }
#endif        
}
