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
#include "clang/Basic/TargetInfo.h"
#include <llvm/ADT/SmallSet.h>
#include <llvm/Support/MathExtras.h>

using namespace clang;

ASTRecordLayoutBuilder::ASTRecordLayoutBuilder(ASTContext &Ctx)
  : Ctx(Ctx), Size(0), Alignment(8), Packed(false), UnfilledBitsInLastByte(0),
  MaxFieldAlignment(0), DataSize(0), IsUnion(false), NonVirtualSize(0), 
  NonVirtualAlignment(8) { }

/// LayoutVtable - Lay out the vtable and set PrimaryBase.
void ASTRecordLayoutBuilder::LayoutVtable(const CXXRecordDecl *RD) {
  if (!RD->isDynamicClass()) {
    // There is no primary base in this case.
    return;
  }

  SelectPrimaryBase(RD);
  if (!PrimaryBase.getBase()) {
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
      assert(!i->getType()->isDependentType() &&
             "Cannot layout class with dependent bases.");
      const CXXRecordDecl *Base =
        cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
      // Skip the PrimaryBase here, as it is laid down first.
      if (Base != PrimaryBase.getBase() || PrimaryBase.isVirtual())
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
  const ASTRecordLayout::PrimaryBaseInfo &BaseInfo = 
    Ctx.getASTRecordLayout(RD).getPrimaryBaseInfo();
  
  // If the record has a primary base class that is virtual, add it to the set
  // of primary bases.
  if (BaseInfo.isVirtual())
    IndirectPrimaryBases.insert(BaseInfo.getBase());
  
  // Now traverse all bases and find primary bases for them.
  for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
       e = RD->bases_end(); i != e; ++i) {
    assert(!i->getType()->isDependentType() &&
           "Cannot layout class with dependent bases.");
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
    assert(!i->getType()->isDependentType() &&
           "Cannot layout class with dependent bases.");
    const CXXRecordDecl *Base =
      cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
    if (!i->isVirtual()) {
      SelectPrimaryVBase(Base, FirstPrimary);
      if (PrimaryBase.getBase())
        return;
      continue;
    }
    if (IsNearlyEmpty(Base)) {
      if (FirstPrimary==0)
        FirstPrimary = Base;
      if (!IndirectPrimaryBases.count(Base)) {
        setPrimaryBase(Base, /*IsVirtual=*/true);
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
    assert(!i->getType()->isDependentType() &&
           "Cannot layout class with dependent bases.");
    const CXXRecordDecl *Base =
      cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
    IdentifyPrimaryBases(Base);
  }

  // If the record has a dynamic base class, attempt to choose a primary base 
  // class. It is the first (in direct base class order) non-virtual dynamic 
  // base class, if one exists.
  for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
       e = RD->bases_end(); i != e; ++i) {
    // Ignore virtual bases.
    if (i->isVirtual())
      continue;
    
    const CXXRecordDecl *Base =
      cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());

    if (Base->isDynamicClass()) {
      // We found it.
      PrimaryBase = ASTRecordLayout::PrimaryBaseInfo(Base, /*IsVirtual=*/false);
      return;
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
  if (!PrimaryBase.getBase())
    setPrimaryBase(FirstPrimary, /*IsVirtual=*/true);
}

void ASTRecordLayoutBuilder::LayoutVirtualBase(const CXXRecordDecl *RD) {
  LayoutBaseNonVirtually(RD, true);
}

uint64_t ASTRecordLayoutBuilder::getBaseOffset(const CXXRecordDecl *Base) {
  for (size_t i = 0; i < Bases.size(); ++i) {
    if (Bases[i].first == Base)
      return Bases[i].second;
  }
  for (size_t i = 0; i < VBases.size(); ++i) {
    if (VBases[i].first == Base)
      return VBases[i].second;
  }
  assert(0 && "missing base");
  return 0;
}


void ASTRecordLayoutBuilder::LayoutVirtualBases(const CXXRecordDecl *Class,
                                                const CXXRecordDecl *RD,
                                                const CXXRecordDecl *PB,
                                                uint64_t Offset,
                                 llvm::SmallSet<const CXXRecordDecl*, 32> &mark,
                    llvm::SmallSet<const CXXRecordDecl*, 32> &IndirectPrimary) {
  for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
         e = RD->bases_end(); i != e; ++i) {
    assert(!i->getType()->isDependentType() &&
           "Cannot layout class with dependent bases.");
    const CXXRecordDecl *Base =
      cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
    uint64_t BaseOffset = Offset;
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
    } else {
      if (RD == Class)
        BaseOffset = getBaseOffset(Base);
      else {
        const ASTRecordLayout &Layout = Ctx.getASTRecordLayout(RD);
        BaseOffset = Offset + Layout.getBaseClassOffset(Base);
      }
    }
    
    if (Base->getNumVBases()) {
      const ASTRecordLayout &Layout = Ctx.getASTRecordLayout(Base);
      const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBaseInfo().getBase();
      LayoutVirtualBases(Class, Base, PrimaryBase, BaseOffset, mark, 
                         IndirectPrimary);
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
    assert(!I->getType()->isDependentType() &&
           "Cannot layout class with dependent bases.");
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
    uint64_t ElementOffset = Offset;
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
    assert(!I->getType()->isDependentType() &&
           "Cannot layout class with dependent bases.");
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
    uint64_t ElementOffset = Offset;

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
  uint64_t Offset = LayoutBase(RD);

  // Add base class offsets.
  if (IsVirtualBase) 
    VBases.push_back(std::make_pair(RD, Offset));
  else
    Bases.push_back(std::make_pair(RD, Offset));
}

void ASTRecordLayoutBuilder::Layout(const RecordDecl *D) {
  IsUnion = D->isUnion();

  Packed = D->hasAttr<PackedAttr>();

  // The #pragma pack attribute specifies the maximum field alignment.
  if (const PragmaPackAttr *PPA = D->getAttr<PragmaPackAttr>())
    MaxFieldAlignment = PPA->getAlignment();

  if (const AlignedAttr *AA = D->getAttr<AlignedAttr>())
    UpdateAlignment(AA->getMaxAlignment());

  // If this is a C++ class, lay out the vtable and the non-virtual bases.
  const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(D);
  if (RD) {
    LayoutVtable(RD);
    // PrimaryBase goes first.
    if (PrimaryBase.getBase()) {
      if (PrimaryBase.isVirtual())
        IndirectPrimaryBases.insert(PrimaryBase.getBase());
      LayoutBaseNonVirtually(PrimaryBase.getBase(), PrimaryBase.isVirtual());
    }
    LayoutNonVirtualBases(RD);
  }

  LayoutFields(D);

  NonVirtualSize = Size;
  NonVirtualAlignment = Alignment;

  if (RD) {
    llvm::SmallSet<const CXXRecordDecl*, 32> mark;
    LayoutVirtualBases(RD, RD, PrimaryBase.getBase(), 
                       0, mark, IndirectPrimaryBases);
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
    UpdateAlignment(AA->getMaxAlignment());

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

void ASTRecordLayoutBuilder::LayoutBitField(const FieldDecl *D) {
  bool FieldPacked = Packed || D->hasAttr<PackedAttr>();
  uint64_t FieldOffset = IsUnion ? 0 : (DataSize - UnfilledBitsInLastByte);
  uint64_t FieldSize = D->getBitWidth()->EvaluateAsInt(Ctx).getZExtValue();
  
  std::pair<uint64_t, unsigned> FieldInfo = Ctx.getTypeInfo(D->getType());
  uint64_t TypeSize = FieldInfo.first;
  unsigned FieldAlign = FieldInfo.second;
  
  if (FieldPacked)
    FieldAlign = 1;
  if (const AlignedAttr *AA = D->getAttr<AlignedAttr>())
    FieldAlign = std::max(FieldAlign, AA->getMaxAlignment());

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
  
  // Place this field at the current location.
  FieldOffsets.push_back(FieldOffset);
  
  // Update DataSize to include the last byte containing (part of) the bitfield.
  if (IsUnion) {
    // FIXME: I think FieldSize should be TypeSize here.
    DataSize = std::max(DataSize, FieldSize);
  } else {
    uint64_t NewSizeInBits = FieldOffset + FieldSize;
    
    DataSize = llvm::RoundUpToAlignment(NewSizeInBits, 8);
    UnfilledBitsInLastByte = DataSize - NewSizeInBits;
  }
  
  // Update the size.
  Size = std::max(Size, DataSize);
  
  // Remember max struct/class alignment.
  UpdateAlignment(FieldAlign);
}

void ASTRecordLayoutBuilder::LayoutField(const FieldDecl *D) {
  if (D->isBitField()) {
    LayoutBitField(D);
    return;
  }

  // Reset the unfilled bits.
  UnfilledBitsInLastByte = 0;

  bool FieldPacked = Packed || D->hasAttr<PackedAttr>();
  uint64_t FieldOffset = IsUnion ? 0 : DataSize;
  uint64_t FieldSize;
  unsigned FieldAlign;
  
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
    FieldAlign = std::max(FieldAlign, AA->getMaxAlignment());

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
      
      // We couldn't place the field at the offset. Try again at a new offset.
      FieldOffset += FieldAlign;
    }
    
    UpdateEmptyClassOffsets(D, FieldOffset);
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
  Size = llvm::RoundUpToAlignment(Size, Alignment);
}

void ASTRecordLayoutBuilder::UpdateAlignment(unsigned NewAlignment) {
  if (NewAlignment <= Alignment)
    return;

  assert(llvm::isPowerOf2_32(NewAlignment && "Alignment not a power of 2"));

  Alignment = NewAlignment;
}

static const CXXMethodDecl *GetKeyFunction(const CXXRecordDecl *RD) {
  if (!RD->isDynamicClass())
    return 0;
  
  for (CXXRecordDecl::method_iterator I = RD->method_begin(), 
       E = RD->method_end(); I != E; ++I) {
    const CXXMethodDecl *MD = *I;
    
    if (!MD->isVirtual())
      continue;
    
    if (MD->isPure())
      continue;
    
    const FunctionDecl *fn;
    if (MD->getBody(fn) && !fn->isOutOfLine())
      continue;
    
    // We found it.
    return MD;
  }
  
  return 0;
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

  const CXXMethodDecl *KeyFunction = GetKeyFunction(cast<CXXRecordDecl>(D));
  
  return new ASTRecordLayout(Builder.Size, Builder.Alignment, DataSize,
                             Builder.FieldOffsets.data(),
                             Builder.FieldOffsets.size(),
                             NonVirtualSize,
                             Builder.NonVirtualAlignment,
                             Builder.PrimaryBase,
                             Builder.Bases.data(),
                             Builder.Bases.size(),
                             Builder.VBases.data(),
                             Builder.VBases.size(),
                             KeyFunction);
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
