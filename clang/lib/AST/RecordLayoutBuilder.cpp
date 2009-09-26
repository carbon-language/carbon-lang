//=== ASTRecordLayoutBuilder.cpp - Helper class for building record layouts ==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RecordLayoutBuilder.h"

#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/TargetInfo.h"
#include <llvm/ADT/SmallSet.h>
#include <llvm/Support/MathExtras.h>

using namespace clang;

ASTRecordLayoutBuilder::ASTRecordLayoutBuilder(ASTContext &Ctx)
  : Ctx(Ctx), Size(0), Alignment(8), Packed(false), MaxFieldAlignment(0),
  DataSize(0), IsUnion(false), NonVirtualSize(0), NonVirtualAlignment(8),
  PrimaryBase(0), PrimaryBaseWasVirtual(false) {}

/// LayoutVtable - Lay out the vtable and set PrimaryBase.
void ASTRecordLayoutBuilder::LayoutVtable(const CXXRecordDecl *RD) {
  if (!RD->isDynamicClass()) {
    // There is no primary base in this case.
    return;
  }

  SelectPrimaryBase(RD);
  if (PrimaryBase == 0) {
    int AS = 0;
    UpdateAlignment(Ctx.Target.getPointerAlign(AS));
    Size += Ctx.Target.getPointerWidth(AS);
    DataSize = Size;
  }
}

void
ASTRecordLayoutBuilder::LayoutNonVirtualBases(const CXXRecordDecl *RD) {
  for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
       e = RD->bases_end(); i != e; ++i) {
    if (!i->isVirtual()) {
      const CXXRecordDecl *Base =
        cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
      // Skip the PrimaryBase here, as it is laid down first.
      if (Base != PrimaryBase || PrimaryBaseWasVirtual)
        LayoutBaseNonVirtually(Base, false);
    }
  }
}

// Helper routines related to the abi definition from:
//   http://www.codesourcery.com/public/cxx-abi/abi.html
//
/// IsNearlyEmpty - Indicates when a class has a vtable pointer, but
/// no other data.
bool ASTRecordLayoutBuilder::IsNearlyEmpty(const CXXRecordDecl *RD) const {
  // FIXME: Audit the corners
  if (!RD->isDynamicClass())
    return false;
  const ASTRecordLayout &BaseInfo = Ctx.getASTRecordLayout(RD);
  if (BaseInfo.getNonVirtualSize() == Ctx.Target.getPointerWidth(0))
    return true;
  return false;
}

void ASTRecordLayoutBuilder::IdentifyPrimaryBases(const CXXRecordDecl *RD) {
  const ASTRecordLayout &Layout = Ctx.getASTRecordLayout(RD);
  
  // If the record has a primary base class that is virtual, add it to the set
  // of primary bases.
  if (Layout.getPrimaryBaseWasVirtual())
    IndirectPrimaryBases.insert(Layout.getPrimaryBase());
  
  // Now traverse all bases and find primary bases for them.
  for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
       e = RD->bases_end(); i != e; ++i) {
    const CXXRecordDecl *Base =
      cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
    
    // Only bases with virtual bases participate in computing the
    // indirect primary virtual base classes.
    if (Base->getNumVBases())
      IdentifyPrimaryBases(Base);
  }
}

void
ASTRecordLayoutBuilder::SelectPrimaryVBase(const CXXRecordDecl *RD,
                                           const CXXRecordDecl *&FirstPrimary) {
  for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
         e = RD->bases_end(); i != e; ++i) {
    const CXXRecordDecl *Base =
      cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
    if (!i->isVirtual()) {
      SelectPrimaryVBase(Base, FirstPrimary);
      if (PrimaryBase)
        return;
      continue;
    }
    if (IsNearlyEmpty(Base)) {
      if (FirstPrimary==0)
        FirstPrimary = Base;
      if (!IndirectPrimaryBases.count(Base)) {
        setPrimaryBase(Base, true);
        return;
      }
    }
  }
}

/// SelectPrimaryBase - Selects the primary base for the given class and
/// record that with setPrimaryBase.  We also calculate the IndirectPrimaries.
void ASTRecordLayoutBuilder::SelectPrimaryBase(const CXXRecordDecl *RD) {
  // Compute all the primary virtual bases for all of our direct and
  // indirect bases, and record all their primary virtual base classes.
  for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
       e = RD->bases_end(); i != e; ++i) {
    const CXXRecordDecl *Base =
      cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
    IdentifyPrimaryBases(Base);
  }

  // If the record has a dynamic base class, attempt to choose a primary base 
  // class. It is the first (in direct base class order) non-virtual dynamic 
  // base class, if one exists.
  for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
       e = RD->bases_end(); i != e; ++i) {
    if (!i->isVirtual()) {
      const CXXRecordDecl *Base =
        cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
      if (Base->isDynamicClass()) {
        // We found it.
        setPrimaryBase(Base, false);
        return;
      }
    }
  }

  // Otherwise, it is the first nearly empty virtual base that is not an
  // indirect primary virtual base class, if one exists.

  // If we have no virtual bases at this point, bail out as the searching below
  // is expensive.
  if (RD->getNumVBases() == 0)
    return;
  
  // Then we can search for the first nearly empty virtual base itself.
  const CXXRecordDecl *FirstPrimary = 0;
  SelectPrimaryVBase(RD, FirstPrimary);

  // Otherwise if is the first nearly empty virtual base, if one exists,
  // otherwise there is no primary base class.
  if (!PrimaryBase)
    setPrimaryBase(FirstPrimary, true);
}

void ASTRecordLayoutBuilder::LayoutVirtualBase(const CXXRecordDecl *RD) {
  LayoutBaseNonVirtually(RD, true);
}

void ASTRecordLayoutBuilder::LayoutVirtualBases(const CXXRecordDecl *RD,
                                                const CXXRecordDecl *PB,
                                                int64_t Offset,
                                 llvm::SmallSet<const CXXRecordDecl*, 32> &mark,
                    llvm::SmallSet<const CXXRecordDecl*, 32> &IndirectPrimary) {
  for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
         e = RD->bases_end(); i != e; ++i) {
    const CXXRecordDecl *Base =
      cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
#if 0
    const ASTRecordLayout &L = Ctx.getASTRecordLayout(Base);
    const CXXRecordDecl *PB = L.getPrimaryBase();
    if (PB && L.getPrimaryBaseWasVirtual()
        && IndirectPrimary.count(PB)) {
      int64_t BaseOffset;
      // FIXME: calculate this.
      BaseOffset = (1<<63) | (1<<31);
      VBases.push_back(PB);
      VBaseOffsets.push_back(BaseOffset);
    }
#endif
    int64_t BaseOffset = Offset;;
    // FIXME: Calculate BaseOffset.
    if (i->isVirtual()) {
      if (Base == PB) {
        // Only lay things out once.
        if (mark.count(Base))
          continue;
        // Mark it so we don't lay it out twice.
        mark.insert(Base);
        assert (IndirectPrimary.count(Base) && "IndirectPrimary was wrong");
        VBases.push_back(std::make_pair(Base, Offset));
      } else if (IndirectPrimary.count(Base)) {
        // Someone else will eventually lay this out.
        ;
      } else {
        // Only lay things out once.
        if (mark.count(Base))
          continue;
        // Mark it so we don't lay it out twice.
        mark.insert(Base);
        LayoutVirtualBase(Base);
        BaseOffset = VBases.back().second;
      }
    }
    if (Base->getNumVBases()) {
      const ASTRecordLayout &L = Ctx.getASTRecordLayout(Base);
      const CXXRecordDecl *PB = L.getPrimaryBase();
      LayoutVirtualBases(Base, PB, BaseOffset, mark, IndirectPrimary);
    }
  }
}

bool ASTRecordLayoutBuilder::canPlaceRecordAtOffset(const CXXRecordDecl *RD, 
                                                    uint64_t Offset) const {
  // Look for an empty class with the same type at the same offset.
  for (EmptyClassOffsetsTy::const_iterator I = 
        EmptyClassOffsets.lower_bound(Offset), 
       E = EmptyClassOffsets.upper_bound(Offset); I != E; ++I) {
    
    if (I->second == RD)
      return false;
  }
  
  const ASTRecordLayout &Info = Ctx.getASTRecordLayout(RD);

  // Check bases.
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    if (I->isVirtual())
      continue;
    
    const CXXRecordDecl *Base =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    uint64_t BaseClassOffset = Info.getBaseClassOffset(Base);
    
    if (!canPlaceRecordAtOffset(Base, Offset + BaseClassOffset))
      return false;
  }
  
  // Check fields.
  unsigned FieldNo = 0;
  for (CXXRecordDecl::field_iterator I = RD->field_begin(), E = RD->field_end(); 
       I != E; ++I, ++FieldNo) {
    const FieldDecl *FD = *I;
    
    uint64_t FieldOffset = Info.getFieldOffset(FieldNo);
    
    if (!canPlaceFieldAtOffset(FD, Offset + FieldOffset))
      return false;
  }

  // FIXME: virtual bases.
  return true;
}

bool ASTRecordLayoutBuilder::canPlaceFieldAtOffset(const FieldDecl *FD, 
                                                   uint64_t Offset) const {
  QualType T = FD->getType();
  if (const RecordType *RT = T->getAs<RecordType>()) {
    if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(RT->getDecl()))
      return canPlaceRecordAtOffset(RD, Offset);
  }
  
  if (const ConstantArrayType *AT = Ctx.getAsConstantArrayType(T)) {
    QualType ElemTy = Ctx.getBaseElementType(AT);
    const RecordType *RT = ElemTy->getAs<RecordType>();
    if (!RT)
      return true;
    const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(RT->getDecl());
    if (!RD)
      return true;
    
    const ASTRecordLayout &Info = Ctx.getASTRecordLayout(RD);

    uint64_t NumElements = Ctx.getConstantArrayElementCount(AT);
    unsigned ElementOffset = Offset;
    for (uint64_t I = 0; I != NumElements; ++I) {
      if (!canPlaceRecordAtOffset(RD, ElementOffset))
        return false;
      
      ElementOffset += Info.getSize();
    }
  }
  
  return true;
}

void ASTRecordLayoutBuilder::UpdateEmptyClassOffsets(const CXXRecordDecl *RD,
                                                     uint64_t Offset) {
  if (RD->isEmpty())
    EmptyClassOffsets.insert(std::make_pair(Offset, RD));
  
  const ASTRecordLayout &Info = Ctx.getASTRecordLayout(RD);

  // Update bases.
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    if (I->isVirtual())
      continue;
    
    const CXXRecordDecl *Base =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());
    
    uint64_t BaseClassOffset = Info.getBaseClassOffset(Base);
    UpdateEmptyClassOffsets(Base, Offset + BaseClassOffset);
  }
  
  // Update fields.
  unsigned FieldNo = 0;
  for (CXXRecordDecl::field_iterator I = RD->field_begin(), E = RD->field_end(); 
       I != E; ++I, ++FieldNo) {
    const FieldDecl *FD = *I;
    
    uint64_t FieldOffset = Info.getFieldOffset(FieldNo);
    UpdateEmptyClassOffsets(FD, Offset + FieldOffset);
  }
  
  // FIXME: Update virtual bases.
}

void
ASTRecordLayoutBuilder::UpdateEmptyClassOffsets(const FieldDecl *FD, 
                                                uint64_t Offset) {
  QualType T = FD->getType();

  if (const RecordType *RT = T->getAs<RecordType>()) {
    if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(RT->getDecl())) {
      UpdateEmptyClassOffsets(RD, Offset);
      return;
    }
  }
  
  if (const ConstantArrayType *AT = Ctx.getAsConstantArrayType(T)) {
    QualType ElemTy = Ctx.getBaseElementType(AT);
    const RecordType *RT = ElemTy->getAs<RecordType>();
    if (!RT)
      return;
    const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(RT->getDecl());
    if (!RD)
      return;
    
    const ASTRecordLayout &Info = Ctx.getASTRecordLayout(RD);

    uint64_t NumElements = Ctx.getConstantArrayElementCount(AT);
    unsigned ElementOffset = Offset;

    for (uint64_t I = 0; I != NumElements; ++I) {
      UpdateEmptyClassOffsets(RD, ElementOffset);
      ElementOffset += Info.getSize();
    }
  }
}

uint64_t ASTRecordLayoutBuilder::LayoutBase(const CXXRecordDecl *RD) {
  const ASTRecordLayout &BaseInfo = Ctx.getASTRecordLayout(RD);

  // If we have an empty base class, try to place it at offset 0.
  if (RD->isEmpty() && canPlaceRecordAtOffset(RD, 0)) {
    // We were able to place the class at offset 0.
    UpdateEmptyClassOffsets(RD, 0);

    Size = std::max(Size, BaseInfo.getSize());

    return 0;
  }
  
  unsigned BaseAlign = BaseInfo.getNonVirtualAlign();
  
  // Round up the current record size to the base's alignment boundary.
  uint64_t Offset = llvm::RoundUpToAlignment(DataSize, BaseAlign);
  
  // Try to place the base.
  while (true) {
    if (canPlaceRecordAtOffset(RD, Offset))
      break;
    
    Offset += BaseAlign;
  }

  if (!RD->isEmpty()) {
    // Update the data size.
    DataSize = Offset + BaseInfo.getNonVirtualSize();

    Size = std::max(Size, DataSize);
  } else
    Size = std::max(Size, Offset + BaseInfo.getSize());

  // Remember max struct/class alignment.
  UpdateAlignment(BaseAlign);

  UpdateEmptyClassOffsets(RD, Offset);
  return Offset;
}

void ASTRecordLayoutBuilder::LayoutBaseNonVirtually(const CXXRecordDecl *RD,
  bool IsVirtualBase) {
  // Layout the base.
  unsigned Offset = LayoutBase(RD);

  // Add base class offsets.
  if (IsVirtualBase) 
    VBases.push_back(std::make_pair(RD, Offset));
  else
    Bases.push_back(std::make_pair(RD, Offset));

#if 0
  // And now add offsets for all our primary virtual bases as well, so
  // they all have offsets.
  const ASTRecordLayout *L = &BaseInfo;
  const CXXRecordDecl *PB = L->getPrimaryBase();
  while (PB) {
    if (L->getPrimaryBaseWasVirtual()) {
      VBases.push_back(PB);
      VBaseOffsets.push_back(Size);
    }
    PB = L->getPrimaryBase();
    if (PB)
      L = &Ctx.getASTRecordLayout(PB);
  }
#endif
}

void ASTRecordLayoutBuilder::Layout(const RecordDecl *D) {
  IsUnion = D->isUnion();

  Packed = D->hasAttr<PackedAttr>();

  // The #pragma pack attribute specifies the maximum field alignment.
  if (const PragmaPackAttr *PPA = D->getAttr<PragmaPackAttr>())
    MaxFieldAlignment = PPA->getAlignment();

  if (const AlignedAttr *AA = D->getAttr<AlignedAttr>())
    UpdateAlignment(AA->getAlignment());

  // If this is a C++ class, lay out the vtable and the non-virtual bases.
  const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(D);
  if (RD) {
    LayoutVtable(RD);
    // PrimaryBase goes first.
    if (PrimaryBase) {
      if (PrimaryBaseWasVirtual)
        IndirectPrimaryBases.insert(PrimaryBase);
      LayoutBaseNonVirtually(PrimaryBase, PrimaryBaseWasVirtual);
    }
    LayoutNonVirtualBases(RD);
  }

  LayoutFields(D);

  NonVirtualSize = Size;
  NonVirtualAlignment = Alignment;

  if (RD) {
    llvm::SmallSet<const CXXRecordDecl*, 32> mark;
    LayoutVirtualBases(RD, PrimaryBase, 0, mark, IndirectPrimaryBases);
  }

  // Finally, round the size of the total struct up to the alignment of the
  // struct itself.
  FinishLayout();
}

void ASTRecordLayoutBuilder::Layout(const ObjCInterfaceDecl *D,
                                    const ObjCImplementationDecl *Impl) {
  if (ObjCInterfaceDecl *SD = D->getSuperClass()) {
    const ASTRecordLayout &SL = Ctx.getASTObjCInterfaceLayout(SD);

    UpdateAlignment(SL.getAlignment());

    // We start laying out ivars not at the end of the superclass
    // structure, but at the next byte following the last field.
    Size = llvm::RoundUpToAlignment(SL.getDataSize(), 8);
    DataSize = Size;
  }

  Packed = D->hasAttr<PackedAttr>();

  // The #pragma pack attribute specifies the maximum field alignment.
  if (const PragmaPackAttr *PPA = D->getAttr<PragmaPackAttr>())
    MaxFieldAlignment = PPA->getAlignment();

  if (const AlignedAttr *AA = D->getAttr<AlignedAttr>())
    UpdateAlignment(AA->getAlignment());

  // Layout each ivar sequentially.
  llvm::SmallVector<ObjCIvarDecl*, 16> Ivars;
  Ctx.ShallowCollectObjCIvars(D, Ivars, Impl);
  for (unsigned i = 0, e = Ivars.size(); i != e; ++i)
    LayoutField(Ivars[i]);

  // Finally, round the size of the total struct up to the alignment of the
  // struct itself.
  FinishLayout();
}

void ASTRecordLayoutBuilder::LayoutFields(const RecordDecl *D) {
  // Layout each field, for now, just sequentially, respecting alignment.  In
  // the future, this will need to be tweakable by targets.
  for (RecordDecl::field_iterator Field = D->field_begin(),
       FieldEnd = D->field_end(); Field != FieldEnd; ++Field)
    LayoutField(*Field);
}

void ASTRecordLayoutBuilder::LayoutField(const FieldDecl *D) {
  bool FieldPacked = Packed;
  uint64_t FieldOffset = IsUnion ? 0 : DataSize;
  uint64_t FieldSize;
  unsigned FieldAlign;

  FieldPacked |= D->hasAttr<PackedAttr>();

  if (const Expr *BitWidthExpr = D->getBitWidth()) {
    // TODO: Need to check this algorithm on other targets!
    //       (tested on Linux-X86)
    FieldSize = BitWidthExpr->EvaluateAsInt(Ctx).getZExtValue();

    std::pair<uint64_t, unsigned> FieldInfo = Ctx.getTypeInfo(D->getType());
    uint64_t TypeSize = FieldInfo.first;

    FieldAlign = FieldInfo.second;

    if (FieldPacked)
      FieldAlign = 1;
    if (const AlignedAttr *AA = D->getAttr<AlignedAttr>())
      FieldAlign = std::max(FieldAlign, AA->getAlignment());
    // The maximum field alignment overrides the aligned attribute.
    if (MaxFieldAlignment)
      FieldAlign = std::min(FieldAlign, MaxFieldAlignment);

    // Check if we need to add padding to give the field the correct
    // alignment.
    if (FieldSize == 0 || (FieldOffset & (FieldAlign-1)) + FieldSize > TypeSize)
      FieldOffset = (FieldOffset + (FieldAlign-1)) & ~(FieldAlign-1);

    // Padding members don't affect overall alignment
    if (!D->getIdentifier())
      FieldAlign = 1;
  } else {
    if (D->getType()->isIncompleteArrayType()) {
      // This is a flexible array member; we can't directly
      // query getTypeInfo about these, so we figure it out here.
      // Flexible array members don't have any size, but they
      // have to be aligned appropriately for their element type.
      FieldSize = 0;
      const ArrayType* ATy = Ctx.getAsArrayType(D->getType());
      FieldAlign = Ctx.getTypeAlign(ATy->getElementType());
    } else if (const ReferenceType *RT = D->getType()->getAs<ReferenceType>()) {
      unsigned AS = RT->getPointeeType().getAddressSpace();
      FieldSize = Ctx.Target.getPointerWidth(AS);
      FieldAlign = Ctx.Target.getPointerAlign(AS);
    } else {
      std::pair<uint64_t, unsigned> FieldInfo = Ctx.getTypeInfo(D->getType());
      FieldSize = FieldInfo.first;
      FieldAlign = FieldInfo.second;
    }

    if (FieldPacked)
      FieldAlign = 8;
    if (const AlignedAttr *AA = D->getAttr<AlignedAttr>())
      FieldAlign = std::max(FieldAlign, AA->getAlignment());
    // The maximum field alignment overrides the aligned attribute.
    if (MaxFieldAlignment)
      FieldAlign = std::min(FieldAlign, MaxFieldAlignment);

    // Round up the current record size to the field's alignment boundary.
    FieldOffset = llvm::RoundUpToAlignment(FieldOffset, FieldAlign);
    
    if (!IsUnion) {
      while (true) {
        // Check if we can place the field at this offset.
        if (canPlaceFieldAtOffset(D, FieldOffset))
          break;
        
        // We can't try again.
        FieldOffset += FieldAlign;
      }
      
      UpdateEmptyClassOffsets(D, FieldOffset);
    }
  }

  // Place this field at the current location.
  FieldOffsets.push_back(FieldOffset);

  // Reserve space for this field.
  if (IsUnion)
    Size = std::max(Size, FieldSize);
  else
    Size = FieldOffset + FieldSize;

  // Update the data size.
  DataSize = Size;

  // Remember max struct/class alignment.
  UpdateAlignment(FieldAlign);
}

void ASTRecordLayoutBuilder::FinishLayout() {
  // In C++, records cannot be of size 0.
  if (Ctx.getLangOptions().CPlusPlus && Size == 0)
    Size = 8;
  // Finally, round the size of the record up to the alignment of the
  // record itself.
  Size = (Size + (Alignment-1)) & ~(Alignment-1);
}

void ASTRecordLayoutBuilder::UpdateAlignment(unsigned NewAlignment) {
  if (NewAlignment <= Alignment)
    return;

  assert(llvm::isPowerOf2_32(NewAlignment && "Alignment not a power of 2"));

  Alignment = NewAlignment;
}

const ASTRecordLayout *
ASTRecordLayoutBuilder::ComputeLayout(ASTContext &Ctx,
                                      const RecordDecl *D) {
  ASTRecordLayoutBuilder Builder(Ctx);

  Builder.Layout(D);

  if (!isa<CXXRecordDecl>(D))
    return new ASTRecordLayout(Builder.Size, Builder.Alignment, Builder.Size,
                               Builder.FieldOffsets.data(),
                               Builder.FieldOffsets.size());

  // FIXME: This is not always correct. See the part about bitfields at
  // http://www.codesourcery.com/public/cxx-abi/abi.html#POD for more info.
  // FIXME: IsPODForThePurposeOfLayout should be stored in the record layout.
  bool IsPODForThePurposeOfLayout = cast<CXXRecordDecl>(D)->isPOD();

  // FIXME: This should be done in FinalizeLayout.
  uint64_t DataSize =
    IsPODForThePurposeOfLayout ? Builder.Size : Builder.DataSize;
  uint64_t NonVirtualSize =
    IsPODForThePurposeOfLayout ? DataSize : Builder.NonVirtualSize;

  return new ASTRecordLayout(Builder.Size, Builder.Alignment, DataSize,
                             Builder.FieldOffsets.data(),
                             Builder.FieldOffsets.size(),
                             NonVirtualSize,
                             Builder.NonVirtualAlignment,
                             Builder.PrimaryBase,
                             Builder.PrimaryBaseWasVirtual,
                             Builder.Bases.data(),
                             Builder.Bases.size(),
                             Builder.VBases.data(),
                             Builder.VBases.size());
}

const ASTRecordLayout *
ASTRecordLayoutBuilder::ComputeLayout(ASTContext &Ctx,
                                      const ObjCInterfaceDecl *D,
                                      const ObjCImplementationDecl *Impl) {
  ASTRecordLayoutBuilder Builder(Ctx);

  Builder.Layout(D, Impl);

  return new ASTRecordLayout(Builder.Size, Builder.Alignment,
                             Builder.DataSize,
                             Builder.FieldOffsets.data(),
                             Builder.FieldOffsets.size());
}
