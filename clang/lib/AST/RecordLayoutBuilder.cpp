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
  NextOffset(0), IsUnion(false), NonVirtualSize(0), NonVirtualAlignment(8) {}

/// LayoutVtable - Lay out the vtable and set PrimaryBase.
void ASTRecordLayoutBuilder::LayoutVtable(const CXXRecordDecl *RD) {
  if (!RD->isDynamicClass()) {
    // There is no primary base in this case.
    setPrimaryBase(0, false);
    return;
  }

  SelectPrimaryBase(RD);
  if (PrimaryBase == 0) {
    int AS = 0;
    UpdateAlignment(Ctx.Target.getPointerAlign(AS));
    Size += Ctx.Target.getPointerWidth(AS);
    NextOffset = Size;
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
      if (Base != PrimaryBase)
        LayoutBaseNonVirtually(Base);
    }
  }
}

// Helper routines related to the abi definition from:
//   http://www.codesourcery.com/public/cxx-abi/abi.html
//
/// IsNearlyEmpty - Indicates when a class has a vtable pointer, but
/// no other data.
bool ASTRecordLayoutBuilder::IsNearlyEmpty(const CXXRecordDecl *RD) {
  // FIXME: Audit the corners
  if (!RD->isDynamicClass())
    return false;
  const ASTRecordLayout &BaseInfo = Ctx.getASTRecordLayout(RD);
  if (BaseInfo.getNonVirtualSize() == Ctx.Target.getPointerWidth(0))
    return true;
  return false;
}

void ASTRecordLayoutBuilder::SelectPrimaryForBase(const CXXRecordDecl *RD,
                    llvm::SmallSet<const CXXRecordDecl*, 32> &IndirectPrimary) {
  const ASTRecordLayout &Layout = Ctx.getASTRecordLayout(RD);
  const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();
  const bool PrimaryBaseWasVirtual = Layout.getPrimaryBaseWasVirtual();
  if (PrimaryBaseWasVirtual) {
    IndirectPrimary.insert(PrimaryBase);
  }
  for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
       e = RD->bases_end(); i != e; ++i) {
    const CXXRecordDecl *Base = 
      cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
    // Only bases with virtual bases participate in computing the
    // indirect primary virtual base classes.
    if (Base->getNumVBases() == 0)
      continue;
    SelectPrimaryForBase(Base, IndirectPrimary);
  }
}

void ASTRecordLayoutBuilder::SelectPrimaryVBase(const CXXRecordDecl *RD,
                                             const CXXRecordDecl *&FirstPrimary,
                    llvm::SmallSet<const CXXRecordDecl*, 32> &IndirectPrimary) {
  for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
         e = RD->bases_end(); i != e; ++i) {
    const CXXRecordDecl *Base = 
      cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
    if (!i->isVirtual()) {
      SelectPrimaryVBase(Base, FirstPrimary, IndirectPrimary);
      if (PrimaryBase)
        return;
      continue;
    }
    if (IsNearlyEmpty(Base)) {
      if (FirstPrimary==0)
        FirstPrimary = Base;
      if (!IndirectPrimary.count(Base)) {
        setPrimaryBase(Base, true);
        return;
      }
    }
  }
}

/// SelectPrimaryBase - Selects the primary base for the given class and
/// record that with setPrimaryBase.
void ASTRecordLayoutBuilder::SelectPrimaryBase(const CXXRecordDecl *RD) {
  // The primary base is the first non-virtual indirect or direct base class,
  // if one exists.
  for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
       e = RD->bases_end(); i != e; ++i) {
    if (!i->isVirtual()) {
      const CXXRecordDecl *Base = 
        cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
      if (Base->isDynamicClass()) {
        setPrimaryBase(Base, false);
        return;
      }
    }
  }

  setPrimaryBase(0, false);

  // Otherwise, it is the first nearly empty virtual base that is not an
  // indirect primary virtual base class, if one exists.

  // If we have no virtual bases at this point, bail out as the searching below
  // is expensive.
  if (RD->getNumVBases() == 0) {
    return;
  }

  // First, we compute all the primary virtual bases for all of our direct and
  // indirect bases, and record all their primary virtual base classes.
  const CXXRecordDecl *FirstPrimary = 0;
  llvm::SmallSet<const CXXRecordDecl*, 32> IndirectPrimary;
  for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
       e = RD->bases_end(); i != e; ++i) {
    const CXXRecordDecl *Base = 
      cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
    SelectPrimaryForBase(Base, IndirectPrimary);
  }

  // Then we can search for the first nearly empty virtual base itself.
  SelectPrimaryVBase(RD, FirstPrimary, IndirectPrimary);

  // Otherwise if is the first nearly empty virtual base, if one exists,
  // otherwise there is no primary base class.
  setPrimaryBase(FirstPrimary, true);
  return;
}

void ASTRecordLayoutBuilder::LayoutVirtualBase(const CXXRecordDecl *RD) {
  LayoutBaseNonVirtually(RD);
}

void ASTRecordLayoutBuilder::LayoutVirtualBases(const CXXRecordDecl *RD) {
  // FIXME: audit indirect virtual bases
  for (CXXRecordDecl::base_class_const_iterator i = RD->vbases_begin(),
         e = RD->vbases_end(); i != e; ++i) {
    const CXXRecordDecl *Base = 
      cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
    if (!PrimaryBaseWasVirtual || Base != PrimaryBase)
      LayoutVirtualBase(Base);
  }
}

void ASTRecordLayoutBuilder::LayoutBaseNonVirtually(const CXXRecordDecl *RD) {
  const ASTRecordLayout &BaseInfo = Ctx.getASTRecordLayout(RD);
    assert(BaseInfo.getDataSize() > 0 && 
           "FIXME: Handle empty classes.");
  
  unsigned BaseAlign = BaseInfo.getNonVirtualAlign();
  uint64_t BaseSize = BaseInfo.getNonVirtualSize();
  
  // Round up the current record size to the base's alignment boundary.
  Size = (Size + (BaseAlign-1)) & ~(BaseAlign-1);

  // Add base class offsets.
  Bases.push_back(RD);
  BaseOffsets.push_back(Size);

  // Reserve space for this base.
  Size += BaseSize;
  
  // Remember the next available offset.
  NextOffset = Size;
  
  // Remember max struct/class alignment.
  UpdateAlignment(BaseAlign);
}

void ASTRecordLayoutBuilder::Layout(const RecordDecl *D) {
  IsUnion = D->isUnion();

  Packed = D->hasAttr<PackedAttr>();

  // The #pragma pack attribute specifies the maximum field alignment.
  if (const PragmaPackAttr *PPA = D->getAttr<PragmaPackAttr>())
    MaxFieldAlignment = PPA->getAlignment();
  
  if (const AlignedAttr *AA = D->getAttr<AlignedAttr>())
    UpdateAlignment(AA->getAlignment());

  // If this is a C++ class, lay out the nonvirtual bases.
  const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(D);
  if (RD) {
    LayoutVtable(RD);
    // PrimaryBase goes first.
    if (PrimaryBase)
      LayoutBaseNonVirtually(PrimaryBase);
    LayoutNonVirtualBases(RD);
  }

  LayoutFields(D);
  
  NonVirtualSize = Size;
  NonVirtualAlignment = Alignment;

  if (RD)
    LayoutVirtualBases(RD);

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
    NextOffset = Size;
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
  uint64_t FieldOffset = IsUnion ? 0 : Size;
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
    FieldOffset = (FieldOffset + (FieldAlign-1)) & ~(FieldAlign-1);
  }
  
  // Place this field at the current location.
  FieldOffsets.push_back(FieldOffset);
  
  // Reserve space for this field.
  if (IsUnion)
    Size = std::max(Size, FieldSize);
  else
    Size = FieldOffset + FieldSize;
  
  // Remember the next available offset.
  NextOffset = Size;
  
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
  
  assert(Builder.Bases.size() == Builder.BaseOffsets.size() && 
         "Base offsets vector must be same size as bases vector!");

  // FIXME: This should be done in FinalizeLayout.
  uint64_t DataSize = 
    IsPODForThePurposeOfLayout ? Builder.Size : Builder.NextOffset;
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
                             Builder.BaseOffsets.data(),
                             Builder.Bases.size());
}

const ASTRecordLayout *
ASTRecordLayoutBuilder::ComputeLayout(ASTContext &Ctx,
                                      const ObjCInterfaceDecl *D,
                                      const ObjCImplementationDecl *Impl) {
  ASTRecordLayoutBuilder Builder(Ctx);
  
  Builder.Layout(D, Impl);
  
  return new ASTRecordLayout(Builder.Size, Builder.Alignment,
                             Builder.NextOffset,
                             Builder.FieldOffsets.data(), 
                             Builder.FieldOffsets.size());
}
