//=== RecordLayoutBuilder.cpp - Helper class for building record layouts ---==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Support/Format.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/MathExtras.h"
#include <map>

using namespace clang;

namespace {

/// EmptySubobjectMap - Keeps track of which empty subobjects exist at different
/// offsets while laying out a C++ class.
class EmptySubobjectMap {
  ASTContext &Context;

  /// Class - The class whose empty entries we're keeping track of.
  const CXXRecordDecl *Class;

  /// EmptyClassOffsets - A map from offsets to empty record decls.
  typedef llvm::SmallVector<const CXXRecordDecl *, 1> ClassVectorTy;
  typedef llvm::DenseMap<uint64_t, ClassVectorTy> EmptyClassOffsetsMapTy;
  EmptyClassOffsetsMapTy EmptyClassOffsets;
  
  /// ComputeEmptySubobjectSizes - Compute the size of the largest base or
  /// member subobject that is empty.
  void ComputeEmptySubobjectSizes();

  struct BaseInfo {
    const CXXRecordDecl *Class;
    bool IsVirtual;

    const CXXRecordDecl *PrimaryVirtualBase;
    
    llvm::SmallVector<BaseInfo*, 4> Bases;
    const BaseInfo *Derived;
  };
  
  llvm::DenseMap<const CXXRecordDecl *, BaseInfo *> VirtualBaseInfo;
  llvm::DenseMap<const CXXRecordDecl *, BaseInfo *> NonVirtualBaseInfo;
  
  BaseInfo *ComputeBaseInfo(const CXXRecordDecl *RD, bool IsVirtual,
                            const BaseInfo *Derived);
  void ComputeBaseInfo();
  
  bool CanPlaceBaseSubobjectAtOffset(const BaseInfo *Info, uint64_t Offset);
  void UpdateEmptyBaseSubobjects(const BaseInfo *Info, uint64_t Offset);
  
public:
  /// This holds the size of the largest empty subobject (either a base
  /// or a member). Will be zero if the record being built doesn't contain
  /// any empty classes.
  uint64_t SizeOfLargestEmptySubobject;

  EmptySubobjectMap(ASTContext &Context, const CXXRecordDecl *Class)
    : Context(Context), Class(Class), SizeOfLargestEmptySubobject(0) {
      ComputeEmptySubobjectSizes();
      
      ComputeBaseInfo();
  }

  /// CanPlaceBaseAtOffset - Return whether the given base class can be placed
  /// at the given offset.
  /// Returns false if placing the record will result in two components
  /// (direct or indirect) of the same type having the same offset.
  bool CanPlaceBaseAtOffset(const CXXRecordDecl *RD, bool BaseIsVirtual,
                            uint64_t Offset);
};

void EmptySubobjectMap::ComputeEmptySubobjectSizes() {
  // Check the bases.
  for (CXXRecordDecl::base_class_const_iterator I = Class->bases_begin(),
       E = Class->bases_end(); I != E; ++I) {
    const CXXRecordDecl *BaseDecl =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    uint64_t EmptySize = 0;
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(BaseDecl);
    if (BaseDecl->isEmpty()) {
      // If the class decl is empty, get its size.
      EmptySize = Layout.getSize();
    } else {
      // Otherwise, we get the largest empty subobject for the decl.
      EmptySize = Layout.getSizeOfLargestEmptySubobject();
    }

    SizeOfLargestEmptySubobject = std::max(SizeOfLargestEmptySubobject,
                                           EmptySize);
  }

  // Check the fields.
  for (CXXRecordDecl::field_iterator I = Class->field_begin(),
       E = Class->field_end(); I != E; ++I) {
    const FieldDecl *FD = *I;

    const RecordType *RT =
      Context.getBaseElementType(FD->getType())->getAs<RecordType>();

    // We only care about record types.
    if (!RT)
      continue;

    uint64_t EmptySize = 0;
    const CXXRecordDecl *MemberDecl = cast<CXXRecordDecl>(RT->getDecl());
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(MemberDecl);
    if (MemberDecl->isEmpty()) {
      // If the class decl is empty, get its size.
      EmptySize = Layout.getSize();
    } else {
      // Otherwise, we get the largest empty subobject for the decl.
      EmptySize = Layout.getSizeOfLargestEmptySubobject();
    }

   SizeOfLargestEmptySubobject = std::max(SizeOfLargestEmptySubobject,
                                          EmptySize);
  }
}

EmptySubobjectMap::BaseInfo *
EmptySubobjectMap::ComputeBaseInfo(const CXXRecordDecl *RD, bool IsVirtual,
                                   const BaseInfo *Derived) {
  BaseInfo *Info;
  
  if (IsVirtual) {
    BaseInfo *&InfoSlot = VirtualBaseInfo[RD];
    if (InfoSlot) {
      assert(InfoSlot->Class == RD && "Wrong class for virtual base info!");
      return InfoSlot;
    }

    InfoSlot = new (Context) BaseInfo;
    Info = InfoSlot;
  } else {
    Info = new (Context) BaseInfo;
  }
  
  Info->Class = RD;
  Info->IsVirtual = IsVirtual;
  Info->Derived = Derived;
  Info->PrimaryVirtualBase = 0;
  
  if (RD->getNumVBases()) {
    // Check if this class has a primary virtual base.
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
    if (Layout.getPrimaryBaseWasVirtual()) {
      Info->PrimaryVirtualBase = Layout.getPrimaryBase();
      assert(Info->PrimaryVirtualBase && 
             "Didn't have a primary virtual base!");
    }
  }

  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    bool IsVirtual = I->isVirtual();
    
    const CXXRecordDecl *BaseDecl =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());
    
    Info->Bases.push_back(ComputeBaseInfo(BaseDecl, IsVirtual, Info));
  }
  
  return Info;
}

void EmptySubobjectMap::ComputeBaseInfo() {
  for (CXXRecordDecl::base_class_const_iterator I = Class->bases_begin(),
       E = Class->bases_end(); I != E; ++I) {
    bool IsVirtual = I->isVirtual();

    const CXXRecordDecl *BaseDecl =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());
    
    BaseInfo *Info = ComputeBaseInfo(BaseDecl, IsVirtual, /*Derived=*/0);
    if (IsVirtual) {
      // ComputeBaseInfo has already added this base for us.
      continue;
    }

    // Add the base info to the map of non-virtual bases.
    assert(!NonVirtualBaseInfo.count(BaseDecl) &&
           "Non-virtual base already exists!");
    NonVirtualBaseInfo.insert(std::make_pair(BaseDecl, Info));
  }
}

bool
EmptySubobjectMap::CanPlaceBaseSubobjectAtOffset(const BaseInfo *Info, 
                                                 uint64_t Offset) {
  // Traverse all non-virtual bases.
  for (unsigned I = 0, E = Info->Bases.size(); I != E; ++I) {
    BaseInfo* Base = Info->Bases[I];
    if (Base->IsVirtual)
      continue;

    const ASTRecordLayout &Layout = Context.getASTRecordLayout(Info->Class);
    uint64_t BaseOffset = Offset + Layout.getBaseClassOffset(Base->Class);

    if (!CanPlaceBaseSubobjectAtOffset(Base, BaseOffset))
      return false;
  }

  if (Info->PrimaryVirtualBase) {
    BaseInfo *PrimaryVirtualBaseInfo = 
      VirtualBaseInfo.lookup(Info->PrimaryVirtualBase);    
    assert(PrimaryVirtualBaseInfo && "Didn't find base info!");

    if (Info == PrimaryVirtualBaseInfo->Derived) {
      if (!CanPlaceBaseSubobjectAtOffset(PrimaryVirtualBaseInfo, Offset))
        return false;
    }
  }
  
  // FIXME: Member variables.
  return true;
}

void EmptySubobjectMap::UpdateEmptyBaseSubobjects(const BaseInfo *Info, 
                                                  uint64_t Offset) {
  if (Info->Class->isEmpty()) {
    // FIXME: Record that there is an empty class at this offset.
  }
  
  // Traverse all non-virtual bases.
  for (unsigned I = 0, E = Info->Bases.size(); I != E; ++I) {
    BaseInfo* Base = Info->Bases[I];
    if (Base->IsVirtual)
      continue;
    
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(Info->Class);
    uint64_t BaseOffset = Offset + Layout.getBaseClassOffset(Base->Class);
    
    UpdateEmptyBaseSubobjects(Base, BaseOffset);
  }

  if (Info->PrimaryVirtualBase) {
    BaseInfo *PrimaryVirtualBaseInfo = 
    VirtualBaseInfo.lookup(Info->PrimaryVirtualBase);    
    assert(PrimaryVirtualBaseInfo && "Didn't find base info!");
    
    if (Info == PrimaryVirtualBaseInfo->Derived)
      UpdateEmptyBaseSubobjects(PrimaryVirtualBaseInfo, Offset);
  }
  
  // FIXME: Member variables.
}

bool EmptySubobjectMap::CanPlaceBaseAtOffset(const CXXRecordDecl *RD,
                                             bool BaseIsVirtual,
                                             uint64_t Offset) {
  // If we know this class doesn't have any empty subobjects we don't need to
  // bother checking.
  if (!SizeOfLargestEmptySubobject)
    return true;

  BaseInfo *Info;
  
  if (BaseIsVirtual)
    Info = VirtualBaseInfo.lookup(RD);
  else
    Info = NonVirtualBaseInfo.lookup(RD);
  
  if (!CanPlaceBaseSubobjectAtOffset(Info, Offset))
    return false;
  
  UpdateEmptyBaseSubobjects(Info, Offset);
  return true;
}

class RecordLayoutBuilder {
  // FIXME: Remove this and make the appropriate fields public.
  friend class clang::ASTContext;

  ASTContext &Context;

  EmptySubobjectMap *EmptySubobjects;

  /// Size - The current size of the record layout.
  uint64_t Size;

  /// Alignment - The current alignment of the record layout.
  unsigned Alignment;

  llvm::SmallVector<uint64_t, 16> FieldOffsets;

  /// Packed - Whether the record is packed or not.
  unsigned Packed : 1;

  unsigned IsUnion : 1;

  unsigned IsMac68kAlign : 1;

  /// UnfilledBitsInLastByte - If the last field laid out was a bitfield,
  /// this contains the number of bits in the last byte that can be used for
  /// an adjacent bitfield if necessary.
  unsigned char UnfilledBitsInLastByte;

  /// MaxFieldAlignment - The maximum allowed field alignment. This is set by
  /// #pragma pack.
  unsigned MaxFieldAlignment;

  /// DataSize - The data size of the record being laid out.
  uint64_t DataSize;

  uint64_t NonVirtualSize;
  unsigned NonVirtualAlignment;

  /// PrimaryBase - the primary base class (if one exists) of the class
  /// we're laying out.
  const CXXRecordDecl *PrimaryBase;

  /// PrimaryBaseIsVirtual - Whether the primary base of the class we're laying
  /// out is virtual.
  bool PrimaryBaseIsVirtual;

  typedef llvm::DenseMap<const CXXRecordDecl *, uint64_t> BaseOffsetsMapTy;

  /// Bases - base classes and their offsets in the record.
  BaseOffsetsMapTy Bases;

  // VBases - virtual base classes and their offsets in the record.
  BaseOffsetsMapTy VBases;

  /// IndirectPrimaryBases - Virtual base classes, direct or indirect, that are
  /// primary base classes for some other direct or indirect base class.
  llvm::SmallSet<const CXXRecordDecl*, 32> IndirectPrimaryBases;

  /// FirstNearlyEmptyVBase - The first nearly empty virtual base class in
  /// inheritance graph order. Used for determining the primary base class.
  const CXXRecordDecl *FirstNearlyEmptyVBase;

  /// VisitedVirtualBases - A set of all the visited virtual bases, used to
  /// avoid visiting virtual bases more than once.
  llvm::SmallPtrSet<const CXXRecordDecl *, 4> VisitedVirtualBases;

  /// EmptyClassOffsets - A map from offsets to empty record decls.
  typedef std::multimap<uint64_t, const CXXRecordDecl *> EmptyClassOffsetsTy;
  EmptyClassOffsetsTy EmptyClassOffsets;

  RecordLayoutBuilder(ASTContext &Context, EmptySubobjectMap *EmptySubobjects)
    : Context(Context), EmptySubobjects(EmptySubobjects), Size(0), Alignment(8),
      Packed(false), IsUnion(false), IsMac68kAlign(false),
      UnfilledBitsInLastByte(0), MaxFieldAlignment(0), DataSize(0),
      NonVirtualSize(0), NonVirtualAlignment(8), PrimaryBase(0),
      PrimaryBaseIsVirtual(false), FirstNearlyEmptyVBase(0) { }

  void Layout(const RecordDecl *D);
  void Layout(const CXXRecordDecl *D);
  void Layout(const ObjCInterfaceDecl *D);

  void LayoutFields(const RecordDecl *D);
  void LayoutField(const FieldDecl *D);
  void LayoutWideBitField(uint64_t FieldSize, uint64_t TypeSize);
  void LayoutBitField(const FieldDecl *D);

  /// ComputeEmptySubobjectSizes - Compute the size of the largest base or
  /// member subobject that is empty.
  void ComputeEmptySubobjectSizes(const CXXRecordDecl *RD);

  /// DeterminePrimaryBase - Determine the primary base of the given class.
  void DeterminePrimaryBase(const CXXRecordDecl *RD);

  void SelectPrimaryVBase(const CXXRecordDecl *RD);

  /// IdentifyPrimaryBases - Identify all virtual base classes, direct or
  /// indirect, that are primary base classes for some other direct or indirect
  /// base class.
  void IdentifyPrimaryBases(const CXXRecordDecl *RD);

  bool IsNearlyEmpty(const CXXRecordDecl *RD) const;

  /// LayoutNonVirtualBases - Determines the primary base class (if any) and
  /// lays it out. Will then proceed to lay out all non-virtual base clasess.
  void LayoutNonVirtualBases(const CXXRecordDecl *RD);

  /// LayoutNonVirtualBase - Lays out a single non-virtual base.
  void LayoutNonVirtualBase(const CXXRecordDecl *Base);

  void AddPrimaryVirtualBaseOffsets(const CXXRecordDecl *RD, uint64_t Offset,
                                    const CXXRecordDecl *MostDerivedClass);

  /// LayoutVirtualBases - Lays out all the virtual bases.
  void LayoutVirtualBases(const CXXRecordDecl *RD,
                          const CXXRecordDecl *MostDerivedClass);

  /// LayoutVirtualBase - Lays out a single virtual base.
  void LayoutVirtualBase(const CXXRecordDecl *Base);

  /// LayoutBase - Will lay out a base and return the offset where it was
  /// placed, in bits.
  uint64_t LayoutBase(const CXXRecordDecl *Base, bool BaseIsVirtual);

  /// canPlaceRecordAtOffset - Return whether a record (either a base class
  /// or a field) can be placed at the given offset.
  /// Returns false if placing the record will result in two components
  /// (direct or indirect) of the same type having the same offset.
  bool canPlaceRecordAtOffset(const CXXRecordDecl *RD, uint64_t Offset,
                              bool CheckVBases) const;

  /// canPlaceFieldAtOffset - Return whether a field can be placed at the given
  /// offset.
  bool canPlaceFieldAtOffset(const FieldDecl *FD, uint64_t Offset) const;

  /// UpdateEmptyClassOffsets - Called after a record (either a base class
  /// or a field) has been placed at the given offset. Will update the
  /// EmptyClassOffsets map if the class is empty or has any empty bases or
  /// fields.
  void UpdateEmptyClassOffsets(const CXXRecordDecl *RD, uint64_t Offset,
                               bool UpdateVBases);

  /// UpdateEmptyClassOffsets - Called after a field has been placed at the
  /// given offset.
  void UpdateEmptyClassOffsets(const FieldDecl *FD, uint64_t Offset);

  /// InitializeLayout - Initialize record layout for the given record decl.
  void InitializeLayout(const Decl *D);

  /// FinishLayout - Finalize record layout. Adjust record size based on the
  /// alignment.
  void FinishLayout();

  void UpdateAlignment(unsigned NewAlignment);

  RecordLayoutBuilder(const RecordLayoutBuilder&);   // DO NOT IMPLEMENT
  void operator=(const RecordLayoutBuilder&); // DO NOT IMPLEMENT
public:
  static const CXXMethodDecl *ComputeKeyFunction(const CXXRecordDecl *RD);
};
} // end anonymous namespace

/// IsNearlyEmpty - Indicates when a class has a vtable pointer, but
/// no other data.
bool RecordLayoutBuilder::IsNearlyEmpty(const CXXRecordDecl *RD) const {
  // FIXME: Audit the corners
  if (!RD->isDynamicClass())
    return false;
  const ASTRecordLayout &BaseInfo = Context.getASTRecordLayout(RD);
  if (BaseInfo.getNonVirtualSize() == Context.Target.getPointerWidth(0))
    return true;
  return false;
}

void RecordLayoutBuilder::IdentifyPrimaryBases(const CXXRecordDecl *RD) {
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
RecordLayoutBuilder::SelectPrimaryVBase(const CXXRecordDecl *RD) {
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
        PrimaryBase = Base;
        PrimaryBaseIsVirtual = true;
        return;
      }

      // Is this the first nearly empty virtual base?
      if (!FirstNearlyEmptyVBase)
        FirstNearlyEmptyVBase = Base;
    }

    SelectPrimaryVBase(Base);
    if (PrimaryBase)
      return;
  }
}

/// DeterminePrimaryBase - Determine the primary base of the given class.
void RecordLayoutBuilder::DeterminePrimaryBase(const CXXRecordDecl *RD) {
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
      PrimaryBase = Base;
      PrimaryBaseIsVirtual = false;
      return;
    }
  }

  // Otherwise, it is the first nearly empty virtual base that is not an
  // indirect primary virtual base class, if one exists.
  if (RD->getNumVBases() != 0) {
    SelectPrimaryVBase(RD);
    if (PrimaryBase)
      return;
  }

  // Otherwise, it is the first nearly empty virtual base that is not an
  // indirect primary virtual base class, if one exists.
  if (FirstNearlyEmptyVBase) {
    PrimaryBase = FirstNearlyEmptyVBase;
    PrimaryBaseIsVirtual = true;
    return;
  }

  // Otherwise there is no primary base class.
  assert(!PrimaryBase && "Should not get here with a primary base!");

  // Allocate the virtual table pointer at offset zero.
  assert(DataSize == 0 && "Vtable pointer must be at offset zero!");

  // Update the size.
  Size += Context.Target.getPointerWidth(0);
  DataSize = Size;

  // Update the alignment.
  UpdateAlignment(Context.Target.getPointerAlign(0));
}

void
RecordLayoutBuilder::LayoutNonVirtualBases(const CXXRecordDecl *RD) {
  // First, determine the primary base class.
  DeterminePrimaryBase(RD);

  // If we have a primary base class, lay it out.
  if (PrimaryBase) {
    if (PrimaryBaseIsVirtual) {
      // We have a virtual primary base, insert it as an indirect primary base.
      IndirectPrimaryBases.insert(PrimaryBase);

      assert(!VisitedVirtualBases.count(PrimaryBase) &&
             "vbase already visited!");
      VisitedVirtualBases.insert(PrimaryBase);

      LayoutVirtualBase(PrimaryBase);
    } else
      LayoutNonVirtualBase(PrimaryBase);
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
    if (Base == PrimaryBase && !PrimaryBaseIsVirtual)
      continue;

    // Lay out the base.
    LayoutNonVirtualBase(Base);
  }
}

void RecordLayoutBuilder::LayoutNonVirtualBase(const CXXRecordDecl *Base) {
  // Layout the base.
  uint64_t Offset = LayoutBase(Base, /*BaseIsVirtual=*/false);

  // Add its base class offset.
  if (!Bases.insert(std::make_pair(Base, Offset)).second)
    assert(false && "Added same base offset more than once!");
}

void
RecordLayoutBuilder::AddPrimaryVirtualBaseOffsets(const CXXRecordDecl *RD,
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
RecordLayoutBuilder::LayoutVirtualBases(const CXXRecordDecl *RD,
                                        const CXXRecordDecl *MostDerivedClass) {
  const CXXRecordDecl *PrimaryBase;
  bool PrimaryBaseIsVirtual;

  if (MostDerivedClass == RD) {
    PrimaryBase = this->PrimaryBase;
    PrimaryBaseIsVirtual = this->PrimaryBaseIsVirtual;
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

void RecordLayoutBuilder::LayoutVirtualBase(const CXXRecordDecl *Base) {
  // Layout the base.
  uint64_t Offset = LayoutBase(Base, /*BaseIsVirtual=*/true);

  // Add its base class offset.
  if (!VBases.insert(std::make_pair(Base, Offset)).second)
    assert(false && "Added same vbase offset more than once!");
}

uint64_t RecordLayoutBuilder::LayoutBase(const CXXRecordDecl *Base,
                                         bool BaseIsVirtual) {
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(Base);

  // If we have an empty base class, try to place it at offset 0.
  if (Base->isEmpty() &&
      EmptySubobjects->CanPlaceBaseAtOffset(Base, BaseIsVirtual, 0) &&
      canPlaceRecordAtOffset(Base, 0, /*CheckVBases=*/false)) {
    // We were able to place the class at offset 0.
    UpdateEmptyClassOffsets(Base, 0, /*UpdateVBases=*/false);

    Size = std::max(Size, Layout.getSize());

    return 0;
  }

  unsigned BaseAlign = Layout.getNonVirtualAlign();

  // Round up the current record size to the base's alignment boundary.
  uint64_t Offset = llvm::RoundUpToAlignment(DataSize, BaseAlign);

  // Try to place the base.
  while (true) {
    if (EmptySubobjects->CanPlaceBaseAtOffset(Base, BaseIsVirtual, Offset) &&
        canPlaceRecordAtOffset(Base, Offset, /*CheckVBases=*/false))
      break;

    Offset += BaseAlign;
  }

  if (!Base->isEmpty()) {
    // Update the data size.
    DataSize = Offset + Layout.getNonVirtualSize();

    Size = std::max(Size, DataSize);
  } else
    Size = std::max(Size, Offset + Layout.getSize());

  // Remember max struct/class alignment.
  UpdateAlignment(BaseAlign);

  UpdateEmptyClassOffsets(Base, Offset, /*UpdateVBases=*/false);
  return Offset;
}

bool
RecordLayoutBuilder::canPlaceRecordAtOffset(const CXXRecordDecl *RD,
                                               uint64_t Offset,
                                               bool CheckVBases) const {
  // Look for an empty class with the same type at the same offset.
  for (EmptyClassOffsetsTy::const_iterator I =
         EmptyClassOffsets.lower_bound(Offset),
         E = EmptyClassOffsets.upper_bound(Offset); I != E; ++I) {

    if (I->second == RD)
      return false;
  }

  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);

  // Check bases.
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
         E = RD->bases_end(); I != E; ++I) {
    assert(!I->getType()->isDependentType() &&
           "Cannot layout class with dependent bases.");
    if (I->isVirtual())
      continue;

    const CXXRecordDecl *BaseDecl =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    uint64_t BaseOffset = Layout.getBaseClassOffset(BaseDecl);

    if (!canPlaceRecordAtOffset(BaseDecl, Offset + BaseOffset,
                                /*CheckVBases=*/false))
      return false;
  }

  // Check fields.
  unsigned FieldNo = 0;
  for (CXXRecordDecl::field_iterator I = RD->field_begin(), E = RD->field_end();
       I != E; ++I, ++FieldNo) {
    const FieldDecl *FD = *I;

    uint64_t FieldOffset = Layout.getFieldOffset(FieldNo);

    if (!canPlaceFieldAtOffset(FD, Offset + FieldOffset))
      return false;
  }

  if (CheckVBases) {
    // FIXME: virtual bases.
  }

  return true;
}

bool RecordLayoutBuilder::canPlaceFieldAtOffset(const FieldDecl *FD,
                                                   uint64_t Offset) const {
  QualType T = FD->getType();
  if (const RecordType *RT = T->getAs<RecordType>()) {
    if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(RT->getDecl()))
      return canPlaceRecordAtOffset(RD, Offset, /*CheckVBases=*/true);
  }

  if (const ConstantArrayType *AT = Context.getAsConstantArrayType(T)) {
    QualType ElemTy = Context.getBaseElementType(AT);
    const RecordType *RT = ElemTy->getAs<RecordType>();
    if (!RT)
      return true;
    const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(RT->getDecl());
    if (!RD)
      return true;

    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);

    uint64_t NumElements = Context.getConstantArrayElementCount(AT);
    uint64_t ElementOffset = Offset;
    for (uint64_t I = 0; I != NumElements; ++I) {
      if (!canPlaceRecordAtOffset(RD, ElementOffset, /*CheckVBases=*/true))
        return false;

      ElementOffset += Layout.getSize();
    }
  }

  return true;
}

void RecordLayoutBuilder::UpdateEmptyClassOffsets(const CXXRecordDecl *RD,
                                                     uint64_t Offset,
                                                     bool UpdateVBases) {
  if (RD->isEmpty())
    EmptyClassOffsets.insert(std::make_pair(Offset, RD));

  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);

  // Update bases.
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
         E = RD->bases_end(); I != E; ++I) {
    assert(!I->getType()->isDependentType() &&
           "Cannot layout class with dependent bases.");
    if (I->isVirtual())
      continue;

    const CXXRecordDecl *Base =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    uint64_t BaseClassOffset = Layout.getBaseClassOffset(Base);
    UpdateEmptyClassOffsets(Base, Offset + BaseClassOffset,
                            /*UpdateVBases=*/false);
  }

  // Update fields.
  unsigned FieldNo = 0;
  for (CXXRecordDecl::field_iterator I = RD->field_begin(), E = RD->field_end();
       I != E; ++I, ++FieldNo) {
    const FieldDecl *FD = *I;

    uint64_t FieldOffset = Layout.getFieldOffset(FieldNo);
    UpdateEmptyClassOffsets(FD, Offset + FieldOffset);
  }

  const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();

  if (UpdateVBases) {
    // FIXME: Update virtual bases.
  } else if (PrimaryBase && Layout.getPrimaryBaseWasVirtual()) {
    // We always want to update the offsets of a primary virtual base.
    assert(Layout.getVBaseClassOffset(PrimaryBase) == 0 &&
           "primary base class offset must always be 0!");
    UpdateEmptyClassOffsets(PrimaryBase, Offset, /*UpdateVBases=*/false);
  }
}

void
RecordLayoutBuilder::UpdateEmptyClassOffsets(const FieldDecl *FD,
                                                uint64_t Offset) {
  QualType T = FD->getType();

  if (const RecordType *RT = T->getAs<RecordType>()) {
    if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(RT->getDecl())) {
      UpdateEmptyClassOffsets(RD, Offset, /*UpdateVBases=*/true);
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
      UpdateEmptyClassOffsets(RD, ElementOffset, /*UpdateVBases=*/true);
      ElementOffset += Info.getSize();
    }
  }
}

void RecordLayoutBuilder::InitializeLayout(const Decl *D) {
  if (const RecordDecl *RD = dyn_cast<RecordDecl>(D))
    IsUnion = RD->isUnion();

  Packed = D->hasAttr<PackedAttr>();

  // mac68k alignment supersedes maximum field alignment and attribute aligned,
  // and forces all structures to have 2-byte alignment. The IBM docs on it
  // allude to additional (more complicated) semantics, especially with regard
  // to bit-fields, but gcc appears not to follow that.
  if (D->hasAttr<AlignMac68kAttr>()) {
    IsMac68kAlign = true;
    MaxFieldAlignment = 2 * 8;
    Alignment = 2 * 8;
  } else {
    if (const MaxFieldAlignmentAttr *MFAA = D->getAttr<MaxFieldAlignmentAttr>())
      MaxFieldAlignment = MFAA->getAlignment();

    if (const AlignedAttr *AA = D->getAttr<AlignedAttr>())
      UpdateAlignment(AA->getMaxAlignment());
  }
}

void RecordLayoutBuilder::Layout(const RecordDecl *D) {
  InitializeLayout(D);
  LayoutFields(D);

  // Finally, round the size of the total struct up to the alignment of the
  // struct itself.
  FinishLayout();
}

void RecordLayoutBuilder::Layout(const CXXRecordDecl *RD) {
  InitializeLayout(RD);

  // Lay out the vtable and the non-virtual bases.
  LayoutNonVirtualBases(RD);

  LayoutFields(RD);

  NonVirtualSize = Size;
  NonVirtualAlignment = Alignment;

  // Lay out the virtual bases and add the primary virtual base offsets.
  LayoutVirtualBases(RD, RD);

  VisitedVirtualBases.clear();
  AddPrimaryVirtualBaseOffsets(RD, 0, RD);

  // Finally, round the size of the total struct up to the alignment of the
  // struct itself.
  FinishLayout();

#ifndef NDEBUG
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
#endif
}

void RecordLayoutBuilder::Layout(const ObjCInterfaceDecl *D) {
  if (ObjCInterfaceDecl *SD = D->getSuperClass()) {
    const ASTRecordLayout &SL = Context.getASTObjCInterfaceLayout(SD);

    UpdateAlignment(SL.getAlignment());

    // We start laying out ivars not at the end of the superclass
    // structure, but at the next byte following the last field.
    Size = llvm::RoundUpToAlignment(SL.getDataSize(), 8);
    DataSize = Size;
  }

  InitializeLayout(D);

  // Layout each ivar sequentially.
  llvm::SmallVector<ObjCIvarDecl*, 16> Ivars;
  Context.ShallowCollectObjCIvars(D, Ivars);
  for (unsigned i = 0, e = Ivars.size(); i != e; ++i)
    LayoutField(Ivars[i]);

  // Finally, round the size of the total struct up to the alignment of the
  // struct itself.
  FinishLayout();
}

void RecordLayoutBuilder::LayoutFields(const RecordDecl *D) {
  // Layout each field, for now, just sequentially, respecting alignment.  In
  // the future, this will need to be tweakable by targets.
  for (RecordDecl::field_iterator Field = D->field_begin(),
         FieldEnd = D->field_end(); Field != FieldEnd; ++Field)
    LayoutField(*Field);
}

void RecordLayoutBuilder::LayoutWideBitField(uint64_t FieldSize,
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

void RecordLayoutBuilder::LayoutBitField(const FieldDecl *D) {
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

void RecordLayoutBuilder::LayoutField(const FieldDecl *D) {
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

void RecordLayoutBuilder::FinishLayout() {
  // In C++, records cannot be of size 0.
  if (Context.getLangOptions().CPlusPlus && Size == 0)
    Size = 8;
  // Finally, round the size of the record up to the alignment of the
  // record itself.
  Size = llvm::RoundUpToAlignment(Size, Alignment);
}

void RecordLayoutBuilder::UpdateAlignment(unsigned NewAlignment) {
  // The alignment is not modified when using 'mac68k' alignment.
  if (IsMac68kAlign)
    return;

  if (NewAlignment <= Alignment)
    return;

  assert(llvm::isPowerOf2_32(NewAlignment && "Alignment not a power of 2"));

  Alignment = NewAlignment;
}

const CXXMethodDecl *
RecordLayoutBuilder::ComputeKeyFunction(const CXXRecordDecl *RD) {
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

/// getASTRecordLayout - Get or compute information about the layout of the
/// specified record (struct/union/class), which indicates its size and field
/// position information.
const ASTRecordLayout &ASTContext::getASTRecordLayout(const RecordDecl *D) {
  D = D->getDefinition();
  assert(D && "Cannot get layout of forward declarations!");

  // Look up this layout, if already laid out, return what we have.
  // Note that we can't save a reference to the entry because this function
  // is recursive.
  const ASTRecordLayout *Entry = ASTRecordLayouts[D];
  if (Entry) return *Entry;

  const ASTRecordLayout *NewEntry;

  if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(D)) {
    EmptySubobjectMap EmptySubobjects(*this, RD);

    RecordLayoutBuilder Builder(*this, &EmptySubobjects);
    Builder.Layout(RD);

    // FIXME: This is not always correct. See the part about bitfields at
    // http://www.codesourcery.com/public/cxx-abi/abi.html#POD for more info.
    // FIXME: IsPODForThePurposeOfLayout should be stored in the record layout.
    bool IsPODForThePurposeOfLayout = cast<CXXRecordDecl>(D)->isPOD();

    // FIXME: This should be done in FinalizeLayout.
    uint64_t DataSize =
      IsPODForThePurposeOfLayout ? Builder.Size : Builder.DataSize;
    uint64_t NonVirtualSize =
      IsPODForThePurposeOfLayout ? DataSize : Builder.NonVirtualSize;

    NewEntry =
      new (*this) ASTRecordLayout(*this, Builder.Size, Builder.Alignment,
                                  DataSize, Builder.FieldOffsets.data(),
                                  Builder.FieldOffsets.size(),
                                  NonVirtualSize,
                                  Builder.NonVirtualAlignment,
                                  EmptySubobjects.SizeOfLargestEmptySubobject,
                                  Builder.PrimaryBase,
                                  Builder.PrimaryBaseIsVirtual,
                                  Builder.Bases, Builder.VBases);
  } else {
    RecordLayoutBuilder Builder(*this, /*EmptySubobjects=*/0);
    Builder.Layout(D);

    NewEntry =
      new (*this) ASTRecordLayout(*this, Builder.Size, Builder.Alignment,
                                  Builder.Size,
                                  Builder.FieldOffsets.data(),
                                  Builder.FieldOffsets.size());
  }

  ASTRecordLayouts[D] = NewEntry;

  if (getLangOptions().DumpRecordLayouts) {
    llvm::errs() << "\n*** Dumping AST Record Layout\n";
    DumpRecordLayout(D, llvm::errs());
  }

  return *NewEntry;
}

const CXXMethodDecl *ASTContext::getKeyFunction(const CXXRecordDecl *RD) {
  RD = cast<CXXRecordDecl>(RD->getDefinition());
  assert(RD && "Cannot get key function for forward declarations!");

  const CXXMethodDecl *&Entry = KeyFunctions[RD];
  if (!Entry)
    Entry = RecordLayoutBuilder::ComputeKeyFunction(RD);
  else
    assert(Entry == RecordLayoutBuilder::ComputeKeyFunction(RD) &&
           "Key function changed!");

  return Entry;
}

/// getInterfaceLayoutImpl - Get or compute information about the
/// layout of the given interface.
///
/// \param Impl - If given, also include the layout of the interface's
/// implementation. This may differ by including synthesized ivars.
const ASTRecordLayout &
ASTContext::getObjCLayout(const ObjCInterfaceDecl *D,
                          const ObjCImplementationDecl *Impl) {
  assert(!D->isForwardDecl() && "Invalid interface decl!");

  // Look up this layout, if already laid out, return what we have.
  ObjCContainerDecl *Key =
    Impl ? (ObjCContainerDecl*) Impl : (ObjCContainerDecl*) D;
  if (const ASTRecordLayout *Entry = ObjCLayouts[Key])
    return *Entry;

  // Add in synthesized ivar count if laying out an implementation.
  if (Impl) {
    unsigned SynthCount = CountNonClassIvars(D);
    // If there aren't any sythesized ivars then reuse the interface
    // entry. Note we can't cache this because we simply free all
    // entries later; however we shouldn't look up implementations
    // frequently.
    if (SynthCount == 0)
      return getObjCLayout(D, 0);
  }

  RecordLayoutBuilder Builder(*this, /*EmptySubobjects=*/0);
  Builder.Layout(D);

  const ASTRecordLayout *NewEntry =
    new (*this) ASTRecordLayout(*this, Builder.Size, Builder.Alignment,
                                Builder.DataSize,
                                Builder.FieldOffsets.data(),
                                Builder.FieldOffsets.size());

  ObjCLayouts[Key] = NewEntry;

  return *NewEntry;
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
