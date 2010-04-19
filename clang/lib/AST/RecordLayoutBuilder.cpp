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
#include "llvm/Support/Format.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/MathExtras.h"

using namespace clang;

ASTRecordLayoutBuilder::ASTRecordLayoutBuilder(ASTContext &Context)
  : Context(Context), Size(0), Alignment(8), Packed(false), 
  UnfilledBitsInLastByte(0), MaxFieldAlignment(0), DataSize(0), IsUnion(false),
  NonVirtualSize(0), NonVirtualAlignment(8), FirstNearlyEmptyVBase(0) { }

/// IsNearlyEmpty - Indicates when a class has a vtable pointer, but
/// no other data.
bool ASTRecordLayoutBuilder::IsNearlyEmpty(const CXXRecordDecl *RD) const {
  // FIXME: Audit the corners
  if (!RD->isDynamicClass())
    return false;
  const ASTRecordLayout &BaseInfo = Context.getASTRecordLayout(RD);
  if (BaseInfo.getNonVirtualSize() == Context.Target.getPointerWidth(0))
    return true;
  return false;
}

void ASTRecordLayoutBuilder::IdentifyPrimaryBases(const CXXRecordDecl *RD) {
  const ASTRecordLayout::PrimaryBaseInfo &BaseInfo =
    Context.getASTRecordLayout(RD).getPrimaryBaseInfo();

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
ASTRecordLayoutBuilder::SelectPrimaryVBase(const CXXRecordDecl *RD) {
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
         E = RD->bases_end(); I != E; ++I) {
    assert(!I->getType()->isDependentType() &&
           "Cannot layout class with dependent bases.");

    const CXXRecordDecl *Base =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    // Check if this is a nearly empty virtual base.
    if (I->isVirtual() && IsNearlyEmpty(Base)) {
      // If it's not an indirect primary base, then we've found our primary
      // base.
      if (!IndirectPrimaryBases.count(Base)) {
        PrimaryBase = ASTRecordLayout::PrimaryBaseInfo(Base,
                                                       /*IsVirtual=*/true);
        return;
      }

      // Is this the first nearly empty virtual base?
      if (!FirstNearlyEmptyVBase)
        FirstNearlyEmptyVBase = Base;
    }

    SelectPrimaryVBase(Base);
    if (PrimaryBase.getBase())
      return;
  }
}

/// DeterminePrimaryBase - Determine the primary base of the given class.
void ASTRecordLayoutBuilder::DeterminePrimaryBase(const CXXRecordDecl *RD) {
  // If the class isn't dynamic, it won't have a primary base.
  if (!RD->isDynamicClass())
    return;

  // Compute all the primary virtual bases for all of our direct and
  // indirect bases, and record all their primary virtual base classes.
  for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
         e = RD->bases_end(); i != e; ++i) {
    assert(!i->getType()->isDependentType() &&
           "Cannot lay out class with dependent bases.");
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
  if (RD->getNumVBases() != 0) {
    SelectPrimaryVBase(RD);
    if (PrimaryBase.getBase())
      return;
  }

  // Otherwise, it is the first nearly empty virtual base that is not an
  // indirect primary virtual base class, if one exists.
  if (FirstNearlyEmptyVBase) {
    PrimaryBase = ASTRecordLayout::PrimaryBaseInfo(FirstNearlyEmptyVBase,
                                                   /*IsVirtual=*/true);
    return;
  }

  // Otherwise there is no primary base class.
  assert(!PrimaryBase.getBase() && "Should not get here with a primary base!");

  // Allocate the virtual table pointer at offset zero.
  assert(DataSize == 0 && "Vtable pointer must be at offset zero!");

  // Update the size.
  Size += Context.Target.getPointerWidth(0);
  DataSize = Size;

  // Update the alignment.
  UpdateAlignment(Context.Target.getPointerAlign(0));
}

void
ASTRecordLayoutBuilder::LayoutNonVirtualBases(const CXXRecordDecl *RD) {
  // First, determine the primary base class.
  DeterminePrimaryBase(RD);

  // If we have a primary base class, lay it out.
  if (const CXXRecordDecl *Base = PrimaryBase.getBase()) {
    if (PrimaryBase.isVirtual()) {
      // We have a virtual primary base, insert it as an indirect primary base.
      IndirectPrimaryBases.insert(Base);

      assert(!VisitedVirtualBases.count(Base) && "vbase already visited!");
      VisitedVirtualBases.insert(Base);
      
      LayoutVirtualBase(Base);
    } else
      LayoutNonVirtualBase(Base);
  }

  // Now lay out the non-virtual bases.
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
         E = RD->bases_end(); I != E; ++I) {

    // Ignore virtual bases.
    if (I->isVirtual())
      continue;

    const CXXRecordDecl *Base =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    // Skip the primary base.
    if (Base == PrimaryBase.getBase() && !PrimaryBase.isVirtual())
      continue;

    // Lay out the base.
    LayoutNonVirtualBase(Base);
  }
}

void ASTRecordLayoutBuilder::LayoutNonVirtualBase(const CXXRecordDecl *RD) {
  // Layout the base.
  uint64_t Offset = LayoutBase(RD);

  // Add its base class offset.
  if (!Bases.insert(std::make_pair(RD, Offset)).second)
    assert(false && "Added same base offset more than once!");
}

void
ASTRecordLayoutBuilder::AddPrimaryVirtualBaseOffsets(const CXXRecordDecl *RD, 
                                        uint64_t Offset,
                                        const CXXRecordDecl *MostDerivedClass) {
  // We already have the offset for the primary base of the most derived class.
  if (RD != MostDerivedClass) {
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
    const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();

    // If this is a primary virtual base and we haven't seen it before, add it.
    if (PrimaryBase && Layout.getPrimaryBaseWasVirtual() &&
        !VBases.count(PrimaryBase))
      VBases.insert(std::make_pair(PrimaryBase, Offset));
  }

  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    assert(!I->getType()->isDependentType() &&
           "Cannot layout class with dependent bases.");
    
    const CXXRecordDecl *BaseDecl =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    if (!BaseDecl->getNumVBases()) {
      // This base isn't interesting since it doesn't have any virtual bases.
      continue;
    }
    
    // Compute the offset of this base.
    uint64_t BaseOffset;
    
    if (I->isVirtual()) {
      // If we don't know this vbase yet, don't visit it. It will be visited
      // later.
      if (!VBases.count(BaseDecl)) {
        continue;
      }
      
      // Check if we've already visited this base.
      if (!VisitedVirtualBases.insert(BaseDecl))
        continue;

      // We want the vbase offset from the class we're currently laying out.
      BaseOffset = VBases[BaseDecl];
    } else if (RD == MostDerivedClass) {
      // We want the base offset from the class we're currently laying out.
      assert(Bases.count(BaseDecl) && "Did not find base!");
      BaseOffset = Bases[BaseDecl];
    } else {
      const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
      BaseOffset = Offset + Layout.getBaseClassOffset(BaseDecl);
    }

    AddPrimaryVirtualBaseOffsets(BaseDecl, BaseOffset, MostDerivedClass);
  }
}

void
ASTRecordLayoutBuilder::LayoutVirtualBases(const CXXRecordDecl *RD,
                                        const CXXRecordDecl *MostDerivedClass) {
  const CXXRecordDecl *PrimaryBase;
  bool PrimaryBaseIsVirtual;

  if (MostDerivedClass == RD) {
    PrimaryBase = this->PrimaryBase.getBase();
    PrimaryBaseIsVirtual = this->PrimaryBase.isVirtual();
  } else {
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
    PrimaryBase = Layout.getPrimaryBase();
    PrimaryBaseIsVirtual = Layout.getPrimaryBaseWasVirtual();
  }

  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
         E = RD->bases_end(); I != E; ++I) {
    assert(!I->getType()->isDependentType() &&
           "Cannot layout class with dependent bases.");

    const CXXRecordDecl *Base =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    if (I->isVirtual()) {
      if (PrimaryBase != Base || !PrimaryBaseIsVirtual) {
        bool IndirectPrimaryBase = IndirectPrimaryBases.count(Base);

        // Only lay out the virtual base if it's not an indirect primary base.
        if (!IndirectPrimaryBase) {
          // Only visit virtual bases once.
          if (!VisitedVirtualBases.insert(Base))
            continue;
          
          LayoutVirtualBase(Base);
        }
      }
    }

    if (!Base->getNumVBases()) {
      // This base isn't interesting since it doesn't have any virtual bases.
      continue;
    }

    LayoutVirtualBases(Base, MostDerivedClass);
  }
}

void ASTRecordLayoutBuilder::LayoutVirtualBase(const CXXRecordDecl *RD) {
  // Layout the base.
  uint64_t Offset = LayoutBase(RD);

  // Add its base class offset.
  if (!VBases.insert(std::make_pair(RD, Offset)).second)
    assert(false && "Added same vbase offset more than once!");
}

uint64_t ASTRecordLayoutBuilder::LayoutBase(const CXXRecordDecl *RD) {
  const ASTRecordLayout &BaseInfo = Context.getASTRecordLayout(RD);

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

bool ASTRecordLayoutBuilder::canPlaceRecordAtOffset(const CXXRecordDecl *RD,
                                                    uint64_t Offset) const {
  // Look for an empty class with the same type at the same offset.
  for (EmptyClassOffsetsTy::const_iterator I =
         EmptyClassOffsets.lower_bound(Offset),
         E = EmptyClassOffsets.upper_bound(Offset); I != E; ++I) {

    if (I->second == RD)
      return false;
  }

  const ASTRecordLayout &Info = Context.getASTRecordLayout(RD);

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

  if (const ConstantArrayType *AT = Context.getAsConstantArrayType(T)) {
    QualType ElemTy = Context.getBaseElementType(AT);
    const RecordType *RT = ElemTy->getAs<RecordType>();
    if (!RT)
      return true;
    const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(RT->getDecl());
    if (!RD)
      return true;

    const ASTRecordLayout &Info = Context.getASTRecordLayout(RD);

    uint64_t NumElements = Context.getConstantArrayElementCount(AT);
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

  const ASTRecordLayout &Info = Context.getASTRecordLayout(RD);

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

  if (const ConstantArrayType *AT = Context.getAsConstantArrayType(T)) {
    QualType ElemTy = Context.getBaseElementType(AT);
    const RecordType *RT = ElemTy->getAs<RecordType>();
    if (!RT)
      return;
    const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(RT->getDecl());
    if (!RD)
      return;

    const ASTRecordLayout &Info = Context.getASTRecordLayout(RD);

    uint64_t NumElements = Context.getConstantArrayElementCount(AT);
    uint64_t ElementOffset = Offset;

    for (uint64_t I = 0; I != NumElements; ++I) {
      UpdateEmptyClassOffsets(RD, ElementOffset);
      ElementOffset += Info.getSize();
    }
  }
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
  if (RD)
    LayoutNonVirtualBases(RD);

  LayoutFields(D);

  NonVirtualSize = Size;
  NonVirtualAlignment = Alignment;

  // If this is a C++ class, lay out its virtual bases and add its primary
  // virtual base offsets.
  if (RD) {
    LayoutVirtualBases(RD, RD);

    VisitedVirtualBases.clear();
    AddPrimaryVirtualBaseOffsets(RD, 0, RD);
  }

  // Finally, round the size of the total struct up to the alignment of the
  // struct itself.
  FinishLayout();
  
#ifndef NDEBUG
  if (RD) {
    // Check that we have base offsets for all bases.
    for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
         E = RD->bases_end(); I != E; ++I) {
      if (I->isVirtual())
        continue;
      
      const CXXRecordDecl *BaseDecl =
        cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());
      
      assert(Bases.count(BaseDecl) && "Did not find base offset!");
    }
    
    // And all virtual bases.
    for (CXXRecordDecl::base_class_const_iterator I = RD->vbases_begin(),
         E = RD->vbases_end(); I != E; ++I) {
      const CXXRecordDecl *BaseDecl =
        cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());
      
      assert(VBases.count(BaseDecl) && "Did not find base offset!");
    }
  }
#endif
}

// FIXME. Impl is no longer needed.
void ASTRecordLayoutBuilder::Layout(const ObjCInterfaceDecl *D,
                                    const ObjCImplementationDecl *Impl) {
  if (ObjCInterfaceDecl *SD = D->getSuperClass()) {
    const ASTRecordLayout &SL = Context.getASTObjCInterfaceLayout(SD);

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
  Context.ShallowCollectObjCIvars(D, Ivars);
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

void ASTRecordLayoutBuilder::LayoutWideBitField(uint64_t FieldSize, 
                                                uint64_t TypeSize) {
  assert(Context.getLangOptions().CPlusPlus &&
         "Can only have wide bit-fields in C++!");
  
  // Itanium C++ ABI 2.4:
  //   If sizeof(T)*8 < n, let T' be the largest integral POD type with 
  //   sizeof(T')*8 <= n.
  
  QualType IntegralPODTypes[] = {
    Context.UnsignedCharTy, Context.UnsignedShortTy, Context.UnsignedIntTy, 
    Context.UnsignedLongTy, Context.UnsignedLongLongTy
  };

  QualType Type;
  for (unsigned I = 0, E = llvm::array_lengthof(IntegralPODTypes);
       I != E; ++I) {
    uint64_t Size = Context.getTypeSize(IntegralPODTypes[I]);

    if (Size > FieldSize)
      break;

    Type = IntegralPODTypes[I];
  }
  assert(!Type.isNull() && "Did not find a type!");
  
  unsigned TypeAlign = Context.getTypeAlign(Type);

  // We're not going to use any of the unfilled bits in the last byte.
  UnfilledBitsInLastByte = 0;

  uint64_t FieldOffset;
  
  if (IsUnion) {
    DataSize = std::max(DataSize, FieldSize);
    FieldOffset = 0;
  } else {
    // The bitfield is allocated starting at the next offset aligned appropriately
    // for T', with length n bits. 
    FieldOffset = llvm::RoundUpToAlignment(DataSize, TypeAlign);
    
    uint64_t NewSizeInBits = FieldOffset + FieldSize;
    
    DataSize = llvm::RoundUpToAlignment(NewSizeInBits, 8);
    UnfilledBitsInLastByte = DataSize - NewSizeInBits;
  }

  // Place this field at the current location.
  FieldOffsets.push_back(FieldOffset);

  // Update the size.
  Size = std::max(Size, DataSize);
  
  // Remember max struct/class alignment.
  UpdateAlignment(TypeAlign);
}

void ASTRecordLayoutBuilder::LayoutBitField(const FieldDecl *D) {
  bool FieldPacked = Packed || D->hasAttr<PackedAttr>();
  uint64_t FieldOffset = IsUnion ? 0 : (DataSize - UnfilledBitsInLastByte);
  uint64_t FieldSize = D->getBitWidth()->EvaluateAsInt(Context).getZExtValue();

  std::pair<uint64_t, unsigned> FieldInfo = Context.getTypeInfo(D->getType());
  uint64_t TypeSize = FieldInfo.first;
  unsigned FieldAlign = FieldInfo.second;

  if (FieldSize > TypeSize) {
    LayoutWideBitField(FieldSize, TypeSize);
    return;
  }

  if (FieldPacked || !Context.Target.useBitFieldTypeAlignment())
    FieldAlign = 1;
  if (const AlignedAttr *AA = D->getAttr<AlignedAttr>())
    FieldAlign = std::max(FieldAlign, AA->getMaxAlignment());

  // The maximum field alignment overrides the aligned attribute.
  if (MaxFieldAlignment)
    FieldAlign = std::min(FieldAlign, MaxFieldAlignment);

  // Check if we need to add padding to give the field the correct alignment.
  if (FieldSize == 0 || (FieldOffset & (FieldAlign-1)) + FieldSize > TypeSize)
    FieldOffset = (FieldOffset + (FieldAlign-1)) & ~(FieldAlign-1);

  // Padding members don't affect overall alignment.
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
    const ArrayType* ATy = Context.getAsArrayType(D->getType());
    FieldAlign = Context.getTypeAlign(ATy->getElementType());
  } else if (const ReferenceType *RT = D->getType()->getAs<ReferenceType>()) {
    unsigned AS = RT->getPointeeType().getAddressSpace();
    FieldSize = Context.Target.getPointerWidth(AS);
    FieldAlign = Context.Target.getPointerAlign(AS);
  } else {
    std::pair<uint64_t, unsigned> FieldInfo = Context.getTypeInfo(D->getType());
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
  if (Context.getLangOptions().CPlusPlus && Size == 0)
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

const ASTRecordLayout *
ASTRecordLayoutBuilder::ComputeLayout(ASTContext &Ctx,
                                      const RecordDecl *D) {
  ASTRecordLayoutBuilder Builder(Ctx);

  Builder.Layout(D);

  if (!isa<CXXRecordDecl>(D))
    return new (Ctx) ASTRecordLayout(Ctx, Builder.Size, Builder.Alignment,
                                     Builder.Size,
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

  return new (Ctx) ASTRecordLayout(Ctx, Builder.Size, Builder.Alignment,
                                   DataSize, Builder.FieldOffsets.data(),
                                   Builder.FieldOffsets.size(),
                                   NonVirtualSize,
                                   Builder.NonVirtualAlignment,
                                   Builder.PrimaryBase,
                                   Builder.Bases, Builder.VBases);
}

const ASTRecordLayout *
ASTRecordLayoutBuilder::ComputeLayout(ASTContext &Ctx,
                                      const ObjCInterfaceDecl *D,
                                      const ObjCImplementationDecl *Impl) {
  ASTRecordLayoutBuilder Builder(Ctx);

  Builder.Layout(D, Impl);

  return new (Ctx) ASTRecordLayout(Ctx, Builder.Size, Builder.Alignment,
                                   Builder.DataSize,
                                   Builder.FieldOffsets.data(),
                                   Builder.FieldOffsets.size());
}

const CXXMethodDecl *
ASTRecordLayoutBuilder::ComputeKeyFunction(const CXXRecordDecl *RD) {
  assert(RD->isDynamicClass() && "Class does not have any virtual methods!");

  // If a class isn't polymorphic it doesn't have a key function.
  if (!RD->isPolymorphic())
    return 0;

  // A class inside an anonymous namespace doesn't have a key function.  (Or
  // at least, there's no point to assigning a key function to such a class;
  // this doesn't affect the ABI.)
  if (RD->isInAnonymousNamespace())
    return 0;

  for (CXXRecordDecl::method_iterator I = RD->method_begin(),
         E = RD->method_end(); I != E; ++I) {
    const CXXMethodDecl *MD = *I;

    if (!MD->isVirtual())
      continue;

    if (MD->isPure())
      continue;

    // Ignore implicit member functions, they are always marked as inline, but
    // they don't have a body until they're defined.
    if (MD->isImplicit())
      continue;

    if (MD->isInlineSpecified())
      continue;

    if (MD->hasInlineBody())
      continue;

    // We found it.
    return MD;
  }

  return 0;
}

static void PrintOffset(llvm::raw_ostream &OS,
                        uint64_t Offset, unsigned IndentLevel) {
  OS << llvm::format("%4d | ", Offset);
  OS.indent(IndentLevel * 2);
}

static void DumpCXXRecordLayout(llvm::raw_ostream &OS,
                                const CXXRecordDecl *RD, ASTContext &C,
                                uint64_t Offset,
                                unsigned IndentLevel,
                                const char* Description,
                                bool IncludeVirtualBases) {
  const ASTRecordLayout &Info = C.getASTRecordLayout(RD);

  PrintOffset(OS, Offset, IndentLevel);
  OS << C.getTypeDeclType(const_cast<CXXRecordDecl *>(RD)).getAsString();
  if (Description)
    OS << ' ' << Description;
  if (RD->isEmpty())
    OS << " (empty)";
  OS << '\n';

  IndentLevel++;

  const CXXRecordDecl *PrimaryBase = Info.getPrimaryBase();

  // Vtable pointer.
  if (RD->isDynamicClass() && !PrimaryBase) {
    PrintOffset(OS, Offset, IndentLevel);
    OS << '(' << RD << " vtable pointer)\n";
  }
  // Dump (non-virtual) bases
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
         E = RD->bases_end(); I != E; ++I) {
    assert(!I->getType()->isDependentType() &&
           "Cannot layout class with dependent bases.");
    if (I->isVirtual())
      continue;

    const CXXRecordDecl *Base =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    uint64_t BaseOffset = Offset + Info.getBaseClassOffset(Base) / 8;

    DumpCXXRecordLayout(OS, Base, C, BaseOffset, IndentLevel,
                        Base == PrimaryBase ? "(primary base)" : "(base)",
                        /*IncludeVirtualBases=*/false);
  }

  // Dump fields.
  uint64_t FieldNo = 0;
  for (CXXRecordDecl::field_iterator I = RD->field_begin(),
         E = RD->field_end(); I != E; ++I, ++FieldNo) {
    const FieldDecl *Field = *I;
    uint64_t FieldOffset = Offset + Info.getFieldOffset(FieldNo) / 8;

    if (const RecordType *RT = Field->getType()->getAs<RecordType>()) {
      if (const CXXRecordDecl *D = dyn_cast<CXXRecordDecl>(RT->getDecl())) {
        DumpCXXRecordLayout(OS, D, C, FieldOffset, IndentLevel,
                            Field->getNameAsCString(),
                            /*IncludeVirtualBases=*/true);
        continue;
      }
    }

    PrintOffset(OS, FieldOffset, IndentLevel);
    OS << Field->getType().getAsString() << ' ' << Field << '\n';
  }

  if (!IncludeVirtualBases)
    return;

  // Dump virtual bases.
  for (CXXRecordDecl::base_class_const_iterator I = RD->vbases_begin(),
         E = RD->vbases_end(); I != E; ++I) {
    assert(I->isVirtual() && "Found non-virtual class!");
    const CXXRecordDecl *VBase =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    uint64_t VBaseOffset = Offset + Info.getVBaseClassOffset(VBase) / 8;
    DumpCXXRecordLayout(OS, VBase, C, VBaseOffset, IndentLevel,
                        VBase == PrimaryBase ?
                        "(primary virtual base)" : "(virtual base)",
                        /*IncludeVirtualBases=*/false);
  }

  OS << "  sizeof=" << Info.getSize() / 8;
  OS << ", dsize=" << Info.getDataSize() / 8;
  OS << ", align=" << Info.getAlignment() / 8 << '\n';
  OS << "  nvsize=" << Info.getNonVirtualSize() / 8;
  OS << ", nvalign=" << Info.getNonVirtualAlign() / 8 << '\n';
  OS << '\n';
}

void ASTContext::DumpRecordLayout(const RecordDecl *RD,
                                  llvm::raw_ostream &OS) {
  const ASTRecordLayout &Info = getASTRecordLayout(RD);

  if (const CXXRecordDecl *CXXRD = dyn_cast<CXXRecordDecl>(RD))
    return DumpCXXRecordLayout(OS, CXXRD, *this, 0, 0, 0,
                               /*IncludeVirtualBases=*/true);

  OS << "Type: " << getTypeDeclType(RD).getAsString() << "\n";
  OS << "Record: ";
  RD->dump();
  OS << "\nLayout: ";
  OS << "<ASTRecordLayout\n";
  OS << "  Size:" << Info.getSize() << "\n";
  OS << "  DataSize:" << Info.getDataSize() << "\n";
  OS << "  Alignment:" << Info.getAlignment() << "\n";
  OS << "  FieldOffsets: [";
  for (unsigned i = 0, e = Info.getFieldCount(); i != e; ++i) {
    if (i) OS << ", ";
    OS << Info.getFieldOffset(i);
  }
  OS << "]>\n";
}
