//===--- VTableBuilder.cpp - C++ vtable layout builder --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with generation of the layout of virtual tables.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/VTableBuilder.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdio>

using namespace clang;

#define DUMP_OVERRIDERS 0

namespace {

/// BaseOffset - Represents an offset from a derived class to a direct or
/// indirect base class.
struct BaseOffset {
  /// DerivedClass - The derived class.
  const CXXRecordDecl *DerivedClass;
  
  /// VirtualBase - If the path from the derived class to the base class
  /// involves virtual base classes, this holds the declaration of the last
  /// virtual base in this path (i.e. closest to the base class).
  const CXXRecordDecl *VirtualBase;

  /// NonVirtualOffset - The offset from the derived class to the base class.
  /// (Or the offset from the virtual base class to the base class, if the 
  /// path from the derived class to the base class involves a virtual base
  /// class.
  CharUnits NonVirtualOffset;
  
  BaseOffset() : DerivedClass(0), VirtualBase(0), 
    NonVirtualOffset(CharUnits::Zero()) { }
  BaseOffset(const CXXRecordDecl *DerivedClass,
             const CXXRecordDecl *VirtualBase, CharUnits NonVirtualOffset)
    : DerivedClass(DerivedClass), VirtualBase(VirtualBase), 
    NonVirtualOffset(NonVirtualOffset) { }

  bool isEmpty() const { return NonVirtualOffset.isZero() && !VirtualBase; }
};

/// FinalOverriders - Contains the final overrider member functions for all
/// member functions in the base subobjects of a class.
class FinalOverriders {
public:
  /// OverriderInfo - Information about a final overrider.
  struct OverriderInfo {
    /// Method - The method decl of the overrider.
    const CXXMethodDecl *Method;

    /// Offset - the base offset of the overrider's parent in the layout class.
    CharUnits Offset;
    
    OverriderInfo() : Method(0), Offset(CharUnits::Zero()) { }
  };

private:
  /// MostDerivedClass - The most derived class for which the final overriders
  /// are stored.
  const CXXRecordDecl *MostDerivedClass;
  
  /// MostDerivedClassOffset - If we're building final overriders for a 
  /// construction vtable, this holds the offset from the layout class to the
  /// most derived class.
  const CharUnits MostDerivedClassOffset;

  /// LayoutClass - The class we're using for layout information. Will be 
  /// different than the most derived class if the final overriders are for a
  /// construction vtable.  
  const CXXRecordDecl *LayoutClass;  

  ASTContext &Context;
  
  /// MostDerivedClassLayout - the AST record layout of the most derived class.
  const ASTRecordLayout &MostDerivedClassLayout;

  /// MethodBaseOffsetPairTy - Uniquely identifies a member function
  /// in a base subobject.
  typedef std::pair<const CXXMethodDecl *, CharUnits> MethodBaseOffsetPairTy;

  typedef llvm::DenseMap<MethodBaseOffsetPairTy,
                         OverriderInfo> OverridersMapTy;
  
  /// OverridersMap - The final overriders for all virtual member functions of 
  /// all the base subobjects of the most derived class.
  OverridersMapTy OverridersMap;
  
  /// SubobjectsToOffsetsMapTy - A mapping from a base subobject (represented
  /// as a record decl and a subobject number) and its offsets in the most
  /// derived class as well as the layout class.
  typedef llvm::DenseMap<std::pair<const CXXRecordDecl *, unsigned>, 
                         CharUnits> SubobjectOffsetMapTy;

  typedef llvm::DenseMap<const CXXRecordDecl *, unsigned> SubobjectCountMapTy;
  
  /// ComputeBaseOffsets - Compute the offsets for all base subobjects of the
  /// given base.
  void ComputeBaseOffsets(BaseSubobject Base, bool IsVirtual,
                          CharUnits OffsetInLayoutClass,
                          SubobjectOffsetMapTy &SubobjectOffsets,
                          SubobjectOffsetMapTy &SubobjectLayoutClassOffsets,
                          SubobjectCountMapTy &SubobjectCounts);

  typedef llvm::SmallPtrSet<const CXXRecordDecl *, 4> VisitedVirtualBasesSetTy;
  
  /// dump - dump the final overriders for a base subobject, and all its direct
  /// and indirect base subobjects.
  void dump(raw_ostream &Out, BaseSubobject Base,
            VisitedVirtualBasesSetTy& VisitedVirtualBases);
  
public:
  FinalOverriders(const CXXRecordDecl *MostDerivedClass,
                  CharUnits MostDerivedClassOffset,
                  const CXXRecordDecl *LayoutClass);

  /// getOverrider - Get the final overrider for the given method declaration in
  /// the subobject with the given base offset. 
  OverriderInfo getOverrider(const CXXMethodDecl *MD, 
                             CharUnits BaseOffset) const {
    assert(OverridersMap.count(std::make_pair(MD, BaseOffset)) && 
           "Did not find overrider!");
    
    return OverridersMap.lookup(std::make_pair(MD, BaseOffset));
  }
  
  /// dump - dump the final overriders.
  void dump() {
    VisitedVirtualBasesSetTy VisitedVirtualBases;
    dump(llvm::errs(), BaseSubobject(MostDerivedClass, CharUnits::Zero()), 
         VisitedVirtualBases);
  }
  
};

FinalOverriders::FinalOverriders(const CXXRecordDecl *MostDerivedClass,
                                 CharUnits MostDerivedClassOffset,
                                 const CXXRecordDecl *LayoutClass)
  : MostDerivedClass(MostDerivedClass), 
  MostDerivedClassOffset(MostDerivedClassOffset), LayoutClass(LayoutClass),
  Context(MostDerivedClass->getASTContext()),
  MostDerivedClassLayout(Context.getASTRecordLayout(MostDerivedClass)) {

  // Compute base offsets.
  SubobjectOffsetMapTy SubobjectOffsets;
  SubobjectOffsetMapTy SubobjectLayoutClassOffsets;
  SubobjectCountMapTy SubobjectCounts;
  ComputeBaseOffsets(BaseSubobject(MostDerivedClass, CharUnits::Zero()), 
                     /*IsVirtual=*/false,
                     MostDerivedClassOffset, 
                     SubobjectOffsets, SubobjectLayoutClassOffsets, 
                     SubobjectCounts);

  // Get the final overriders.
  CXXFinalOverriderMap FinalOverriders;
  MostDerivedClass->getFinalOverriders(FinalOverriders);

  for (CXXFinalOverriderMap::const_iterator I = FinalOverriders.begin(),
       E = FinalOverriders.end(); I != E; ++I) {
    const CXXMethodDecl *MD = I->first;
    const OverridingMethods& Methods = I->second;

    for (OverridingMethods::const_iterator I = Methods.begin(),
         E = Methods.end(); I != E; ++I) {
      unsigned SubobjectNumber = I->first;
      assert(SubobjectOffsets.count(std::make_pair(MD->getParent(), 
                                                   SubobjectNumber)) &&
             "Did not find subobject offset!");
      
      CharUnits BaseOffset = SubobjectOffsets[std::make_pair(MD->getParent(),
                                                            SubobjectNumber)];

      assert(I->second.size() == 1 && "Final overrider is not unique!");
      const UniqueVirtualMethod &Method = I->second.front();

      const CXXRecordDecl *OverriderRD = Method.Method->getParent();
      assert(SubobjectLayoutClassOffsets.count(
             std::make_pair(OverriderRD, Method.Subobject))
             && "Did not find subobject offset!");
      CharUnits OverriderOffset =
        SubobjectLayoutClassOffsets[std::make_pair(OverriderRD, 
                                                   Method.Subobject)];

      OverriderInfo& Overrider = OverridersMap[std::make_pair(MD, BaseOffset)];
      assert(!Overrider.Method && "Overrider should not exist yet!");
      
      Overrider.Offset = OverriderOffset;
      Overrider.Method = Method.Method;
    }
  }

#if DUMP_OVERRIDERS
  // And dump them (for now).
  dump();
#endif
}

static BaseOffset ComputeBaseOffset(ASTContext &Context, 
                                    const CXXRecordDecl *DerivedRD,
                                    const CXXBasePath &Path) {
  CharUnits NonVirtualOffset = CharUnits::Zero();

  unsigned NonVirtualStart = 0;
  const CXXRecordDecl *VirtualBase = 0;
  
  // First, look for the virtual base class.
  for (int I = Path.size(), E = 0; I != E; --I) {
    const CXXBasePathElement &Element = Path[I - 1];

    if (Element.Base->isVirtual()) {
      NonVirtualStart = I;
      QualType VBaseType = Element.Base->getType();
      VirtualBase = VBaseType->getAsCXXRecordDecl();
      break;
    }
  }
  
  // Now compute the non-virtual offset.
  for (unsigned I = NonVirtualStart, E = Path.size(); I != E; ++I) {
    const CXXBasePathElement &Element = Path[I];
    
    // Check the base class offset.
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(Element.Class);

    const CXXRecordDecl *Base = Element.Base->getType()->getAsCXXRecordDecl();

    NonVirtualOffset += Layout.getBaseClassOffset(Base);
  }
  
  // FIXME: This should probably use CharUnits or something. Maybe we should
  // even change the base offsets in ASTRecordLayout to be specified in 
  // CharUnits.
  return BaseOffset(DerivedRD, VirtualBase, NonVirtualOffset);
  
}

static BaseOffset ComputeBaseOffset(ASTContext &Context, 
                                    const CXXRecordDecl *BaseRD,
                                    const CXXRecordDecl *DerivedRD) {
  CXXBasePaths Paths(/*FindAmbiguities=*/false,
                     /*RecordPaths=*/true, /*DetectVirtual=*/false);

  if (!DerivedRD->isDerivedFrom(BaseRD, Paths))
    llvm_unreachable("Class must be derived from the passed in base class!");

  return ComputeBaseOffset(Context, DerivedRD, Paths.front());
}

static BaseOffset
ComputeReturnAdjustmentBaseOffset(ASTContext &Context, 
                                  const CXXMethodDecl *DerivedMD,
                                  const CXXMethodDecl *BaseMD) {
  const FunctionType *BaseFT = BaseMD->getType()->getAs<FunctionType>();
  const FunctionType *DerivedFT = DerivedMD->getType()->getAs<FunctionType>();
  
  // Canonicalize the return types.
  CanQualType CanDerivedReturnType =
      Context.getCanonicalType(DerivedFT->getReturnType());
  CanQualType CanBaseReturnType =
      Context.getCanonicalType(BaseFT->getReturnType());

  assert(CanDerivedReturnType->getTypeClass() == 
         CanBaseReturnType->getTypeClass() && 
         "Types must have same type class!");
  
  if (CanDerivedReturnType == CanBaseReturnType) {
    // No adjustment needed.
    return BaseOffset();
  }
  
  if (isa<ReferenceType>(CanDerivedReturnType)) {
    CanDerivedReturnType = 
      CanDerivedReturnType->getAs<ReferenceType>()->getPointeeType();
    CanBaseReturnType = 
      CanBaseReturnType->getAs<ReferenceType>()->getPointeeType();
  } else if (isa<PointerType>(CanDerivedReturnType)) {
    CanDerivedReturnType = 
      CanDerivedReturnType->getAs<PointerType>()->getPointeeType();
    CanBaseReturnType = 
      CanBaseReturnType->getAs<PointerType>()->getPointeeType();
  } else {
    llvm_unreachable("Unexpected return type!");
  }
  
  // We need to compare unqualified types here; consider
  //   const T *Base::foo();
  //   T *Derived::foo();
  if (CanDerivedReturnType.getUnqualifiedType() == 
      CanBaseReturnType.getUnqualifiedType()) {
    // No adjustment needed.
    return BaseOffset();
  }
  
  const CXXRecordDecl *DerivedRD = 
    cast<CXXRecordDecl>(cast<RecordType>(CanDerivedReturnType)->getDecl());
  
  const CXXRecordDecl *BaseRD = 
    cast<CXXRecordDecl>(cast<RecordType>(CanBaseReturnType)->getDecl());

  return ComputeBaseOffset(Context, BaseRD, DerivedRD);
}

void 
FinalOverriders::ComputeBaseOffsets(BaseSubobject Base, bool IsVirtual,
                              CharUnits OffsetInLayoutClass,
                              SubobjectOffsetMapTy &SubobjectOffsets,
                              SubobjectOffsetMapTy &SubobjectLayoutClassOffsets,
                              SubobjectCountMapTy &SubobjectCounts) {
  const CXXRecordDecl *RD = Base.getBase();
  
  unsigned SubobjectNumber = 0;
  if (!IsVirtual)
    SubobjectNumber = ++SubobjectCounts[RD];

  // Set up the subobject to offset mapping.
  assert(!SubobjectOffsets.count(std::make_pair(RD, SubobjectNumber))
         && "Subobject offset already exists!");
  assert(!SubobjectLayoutClassOffsets.count(std::make_pair(RD, SubobjectNumber)) 
         && "Subobject offset already exists!");

  SubobjectOffsets[std::make_pair(RD, SubobjectNumber)] = Base.getBaseOffset();
  SubobjectLayoutClassOffsets[std::make_pair(RD, SubobjectNumber)] =
    OffsetInLayoutClass;
  
  // Traverse our bases.
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    const CXXRecordDecl *BaseDecl = I->getType()->getAsCXXRecordDecl();

    CharUnits BaseOffset;
    CharUnits BaseOffsetInLayoutClass;
    if (I->isVirtual()) {
      // Check if we've visited this virtual base before.
      if (SubobjectOffsets.count(std::make_pair(BaseDecl, 0)))
        continue;

      const ASTRecordLayout &LayoutClassLayout =
        Context.getASTRecordLayout(LayoutClass);

      BaseOffset = MostDerivedClassLayout.getVBaseClassOffset(BaseDecl);
      BaseOffsetInLayoutClass = 
        LayoutClassLayout.getVBaseClassOffset(BaseDecl);
    } else {
      const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
      CharUnits Offset = Layout.getBaseClassOffset(BaseDecl);
    
      BaseOffset = Base.getBaseOffset() + Offset;
      BaseOffsetInLayoutClass = OffsetInLayoutClass + Offset;
    }

    ComputeBaseOffsets(BaseSubobject(BaseDecl, BaseOffset), 
                       I->isVirtual(), BaseOffsetInLayoutClass, 
                       SubobjectOffsets, SubobjectLayoutClassOffsets, 
                       SubobjectCounts);
  }
}

void FinalOverriders::dump(raw_ostream &Out, BaseSubobject Base,
                           VisitedVirtualBasesSetTy &VisitedVirtualBases) {
  const CXXRecordDecl *RD = Base.getBase();
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);

  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    const CXXRecordDecl *BaseDecl = I->getType()->getAsCXXRecordDecl();
    
    // Ignore bases that don't have any virtual member functions.
    if (!BaseDecl->isPolymorphic())
      continue;

    CharUnits BaseOffset;
    if (I->isVirtual()) {
      if (!VisitedVirtualBases.insert(BaseDecl)) {
        // We've visited this base before.
        continue;
      }
      
      BaseOffset = MostDerivedClassLayout.getVBaseClassOffset(BaseDecl);
    } else {
      BaseOffset = Layout.getBaseClassOffset(BaseDecl) + Base.getBaseOffset();
    }

    dump(Out, BaseSubobject(BaseDecl, BaseOffset), VisitedVirtualBases);
  }

  Out << "Final overriders for (";
  RD->printQualifiedName(Out);
  Out << ", ";
  Out << Base.getBaseOffset().getQuantity() << ")\n";

  // Now dump the overriders for this base subobject.
  for (CXXRecordDecl::method_iterator I = RD->method_begin(), 
       E = RD->method_end(); I != E; ++I) {
    const CXXMethodDecl *MD = *I;

    if (!MD->isVirtual())
      continue;
  
    OverriderInfo Overrider = getOverrider(MD, Base.getBaseOffset());

    Out << "  ";
    MD->printQualifiedName(Out);
    Out << " - (";
    Overrider.Method->printQualifiedName(Out);
    Out << ", " << Overrider.Offset.getQuantity() << ')';

    BaseOffset Offset;
    if (!Overrider.Method->isPure())
      Offset = ComputeReturnAdjustmentBaseOffset(Context, Overrider.Method, MD);

    if (!Offset.isEmpty()) {
      Out << " [ret-adj: ";
      if (Offset.VirtualBase) {
        Offset.VirtualBase->printQualifiedName(Out);
        Out << " vbase, ";
      }
             
      Out << Offset.NonVirtualOffset.getQuantity() << " nv]";
    }
    
    Out << "\n";
  }  
}

/// VCallOffsetMap - Keeps track of vcall offsets when building a vtable.
struct VCallOffsetMap {
  
  typedef std::pair<const CXXMethodDecl *, CharUnits> MethodAndOffsetPairTy;
  
  /// Offsets - Keeps track of methods and their offsets.
  // FIXME: This should be a real map and not a vector.
  SmallVector<MethodAndOffsetPairTy, 16> Offsets;

  /// MethodsCanShareVCallOffset - Returns whether two virtual member functions
  /// can share the same vcall offset.
  static bool MethodsCanShareVCallOffset(const CXXMethodDecl *LHS,
                                         const CXXMethodDecl *RHS);

public:
  /// AddVCallOffset - Adds a vcall offset to the map. Returns true if the
  /// add was successful, or false if there was already a member function with
  /// the same signature in the map.
  bool AddVCallOffset(const CXXMethodDecl *MD, CharUnits OffsetOffset);
  
  /// getVCallOffsetOffset - Returns the vcall offset offset (relative to the
  /// vtable address point) for the given virtual member function.
  CharUnits getVCallOffsetOffset(const CXXMethodDecl *MD);
  
  // empty - Return whether the offset map is empty or not.
  bool empty() const { return Offsets.empty(); }
};

static bool HasSameVirtualSignature(const CXXMethodDecl *LHS,
                                    const CXXMethodDecl *RHS) {
  const FunctionProtoType *LT =
    cast<FunctionProtoType>(LHS->getType().getCanonicalType());
  const FunctionProtoType *RT =
    cast<FunctionProtoType>(RHS->getType().getCanonicalType());

  // Fast-path matches in the canonical types.
  if (LT == RT) return true;

  // Force the signatures to match.  We can't rely on the overrides
  // list here because there isn't necessarily an inheritance
  // relationship between the two methods.
  if (LT->getTypeQuals() != RT->getTypeQuals() ||
      LT->getNumParams() != RT->getNumParams())
    return false;
  for (unsigned I = 0, E = LT->getNumParams(); I != E; ++I)
    if (LT->getParamType(I) != RT->getParamType(I))
      return false;
  return true;
}

bool VCallOffsetMap::MethodsCanShareVCallOffset(const CXXMethodDecl *LHS,
                                                const CXXMethodDecl *RHS) {
  assert(LHS->isVirtual() && "LHS must be virtual!");
  assert(RHS->isVirtual() && "LHS must be virtual!");
  
  // A destructor can share a vcall offset with another destructor.
  if (isa<CXXDestructorDecl>(LHS))
    return isa<CXXDestructorDecl>(RHS);

  // FIXME: We need to check more things here.
  
  // The methods must have the same name.
  DeclarationName LHSName = LHS->getDeclName();
  DeclarationName RHSName = RHS->getDeclName();
  if (LHSName != RHSName)
    return false;

  // And the same signatures.
  return HasSameVirtualSignature(LHS, RHS);
}

bool VCallOffsetMap::AddVCallOffset(const CXXMethodDecl *MD, 
                                    CharUnits OffsetOffset) {
  // Check if we can reuse an offset.
  for (unsigned I = 0, E = Offsets.size(); I != E; ++I) {
    if (MethodsCanShareVCallOffset(Offsets[I].first, MD))
      return false;
  }
  
  // Add the offset.
  Offsets.push_back(MethodAndOffsetPairTy(MD, OffsetOffset));
  return true;
}

CharUnits VCallOffsetMap::getVCallOffsetOffset(const CXXMethodDecl *MD) {
  // Look for an offset.
  for (unsigned I = 0, E = Offsets.size(); I != E; ++I) {
    if (MethodsCanShareVCallOffset(Offsets[I].first, MD))
      return Offsets[I].second;
  }
  
  llvm_unreachable("Should always find a vcall offset offset!");
}

/// VCallAndVBaseOffsetBuilder - Class for building vcall and vbase offsets.
class VCallAndVBaseOffsetBuilder {
public:
  typedef llvm::DenseMap<const CXXRecordDecl *, CharUnits> 
    VBaseOffsetOffsetsMapTy;

private:
  /// MostDerivedClass - The most derived class for which we're building vcall
  /// and vbase offsets.
  const CXXRecordDecl *MostDerivedClass;
  
  /// LayoutClass - The class we're using for layout information. Will be 
  /// different than the most derived class if we're building a construction
  /// vtable.
  const CXXRecordDecl *LayoutClass;
  
  /// Context - The ASTContext which we will use for layout information.
  ASTContext &Context;

  /// Components - vcall and vbase offset components
  typedef SmallVector<VTableComponent, 64> VTableComponentVectorTy;
  VTableComponentVectorTy Components;
  
  /// VisitedVirtualBases - Visited virtual bases.
  llvm::SmallPtrSet<const CXXRecordDecl *, 4> VisitedVirtualBases;
  
  /// VCallOffsets - Keeps track of vcall offsets.
  VCallOffsetMap VCallOffsets;


  /// VBaseOffsetOffsets - Contains the offsets of the virtual base offsets,
  /// relative to the address point.
  VBaseOffsetOffsetsMapTy VBaseOffsetOffsets;
  
  /// FinalOverriders - The final overriders of the most derived class.
  /// (Can be null when we're not building a vtable of the most derived class).
  const FinalOverriders *Overriders;

  /// AddVCallAndVBaseOffsets - Add vcall offsets and vbase offsets for the
  /// given base subobject.
  void AddVCallAndVBaseOffsets(BaseSubobject Base, bool BaseIsVirtual,
                               CharUnits RealBaseOffset);
  
  /// AddVCallOffsets - Add vcall offsets for the given base subobject.
  void AddVCallOffsets(BaseSubobject Base, CharUnits VBaseOffset);
  
  /// AddVBaseOffsets - Add vbase offsets for the given class.
  void AddVBaseOffsets(const CXXRecordDecl *Base, 
                       CharUnits OffsetInLayoutClass);
  
  /// getCurrentOffsetOffset - Get the current vcall or vbase offset offset in
  /// chars, relative to the vtable address point.
  CharUnits getCurrentOffsetOffset() const;
  
public:
  VCallAndVBaseOffsetBuilder(const CXXRecordDecl *MostDerivedClass,
                             const CXXRecordDecl *LayoutClass,
                             const FinalOverriders *Overriders,
                             BaseSubobject Base, bool BaseIsVirtual,
                             CharUnits OffsetInLayoutClass)
    : MostDerivedClass(MostDerivedClass), LayoutClass(LayoutClass), 
    Context(MostDerivedClass->getASTContext()), Overriders(Overriders) {
      
    // Add vcall and vbase offsets.
    AddVCallAndVBaseOffsets(Base, BaseIsVirtual, OffsetInLayoutClass);
  }
  
  /// Methods for iterating over the components.
  typedef VTableComponentVectorTy::const_reverse_iterator const_iterator;
  const_iterator components_begin() const { return Components.rbegin(); }
  const_iterator components_end() const { return Components.rend(); }
  
  const VCallOffsetMap &getVCallOffsets() const { return VCallOffsets; }
  const VBaseOffsetOffsetsMapTy &getVBaseOffsetOffsets() const {
    return VBaseOffsetOffsets;
  }
};
  
void 
VCallAndVBaseOffsetBuilder::AddVCallAndVBaseOffsets(BaseSubobject Base,
                                                    bool BaseIsVirtual,
                                                    CharUnits RealBaseOffset) {
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(Base.getBase());
  
  // Itanium C++ ABI 2.5.2:
  //   ..in classes sharing a virtual table with a primary base class, the vcall
  //   and vbase offsets added by the derived class all come before the vcall
  //   and vbase offsets required by the base class, so that the latter may be
  //   laid out as required by the base class without regard to additions from
  //   the derived class(es).

  // (Since we're emitting the vcall and vbase offsets in reverse order, we'll
  // emit them for the primary base first).
  if (const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase()) {
    bool PrimaryBaseIsVirtual = Layout.isPrimaryBaseVirtual();

    CharUnits PrimaryBaseOffset;
    
    // Get the base offset of the primary base.
    if (PrimaryBaseIsVirtual) {
      assert(Layout.getVBaseClassOffset(PrimaryBase).isZero() &&
             "Primary vbase should have a zero offset!");
      
      const ASTRecordLayout &MostDerivedClassLayout =
        Context.getASTRecordLayout(MostDerivedClass);
      
      PrimaryBaseOffset = 
        MostDerivedClassLayout.getVBaseClassOffset(PrimaryBase);
    } else {
      assert(Layout.getBaseClassOffset(PrimaryBase).isZero() &&
             "Primary base should have a zero offset!");

      PrimaryBaseOffset = Base.getBaseOffset();
    }

    AddVCallAndVBaseOffsets(
      BaseSubobject(PrimaryBase,PrimaryBaseOffset),
      PrimaryBaseIsVirtual, RealBaseOffset);
  }

  AddVBaseOffsets(Base.getBase(), RealBaseOffset);

  // We only want to add vcall offsets for virtual bases.
  if (BaseIsVirtual)
    AddVCallOffsets(Base, RealBaseOffset);
}

CharUnits VCallAndVBaseOffsetBuilder::getCurrentOffsetOffset() const {
  // OffsetIndex is the index of this vcall or vbase offset, relative to the 
  // vtable address point. (We subtract 3 to account for the information just
  // above the address point, the RTTI info, the offset to top, and the
  // vcall offset itself).
  int64_t OffsetIndex = -(int64_t)(3 + Components.size());
    
  CharUnits PointerWidth = 
    Context.toCharUnitsFromBits(Context.getTargetInfo().getPointerWidth(0));
  CharUnits OffsetOffset = PointerWidth * OffsetIndex;
  return OffsetOffset;
}

void VCallAndVBaseOffsetBuilder::AddVCallOffsets(BaseSubobject Base, 
                                                 CharUnits VBaseOffset) {
  const CXXRecordDecl *RD = Base.getBase();
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);

  const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();

  // Handle the primary base first.
  // We only want to add vcall offsets if the base is non-virtual; a virtual
  // primary base will have its vcall and vbase offsets emitted already.
  if (PrimaryBase && !Layout.isPrimaryBaseVirtual()) {
    // Get the base offset of the primary base.
    assert(Layout.getBaseClassOffset(PrimaryBase).isZero() &&
           "Primary base should have a zero offset!");

    AddVCallOffsets(BaseSubobject(PrimaryBase, Base.getBaseOffset()),
                    VBaseOffset);
  }
  
  // Add the vcall offsets.
  for (CXXRecordDecl::method_iterator I = RD->method_begin(),
       E = RD->method_end(); I != E; ++I) {
    const CXXMethodDecl *MD = *I;
    
    if (!MD->isVirtual())
      continue;

    CharUnits OffsetOffset = getCurrentOffsetOffset();
    
    // Don't add a vcall offset if we already have one for this member function
    // signature.
    if (!VCallOffsets.AddVCallOffset(MD, OffsetOffset))
      continue;

    CharUnits Offset = CharUnits::Zero();

    if (Overriders) {
      // Get the final overrider.
      FinalOverriders::OverriderInfo Overrider = 
        Overriders->getOverrider(MD, Base.getBaseOffset());
      
      /// The vcall offset is the offset from the virtual base to the object 
      /// where the function was overridden.
      Offset = Overrider.Offset - VBaseOffset;
    }
    
    Components.push_back(
      VTableComponent::MakeVCallOffset(Offset));
  }

  // And iterate over all non-virtual bases (ignoring the primary base).
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
  
    if (I->isVirtual())
      continue;

    const CXXRecordDecl *BaseDecl = I->getType()->getAsCXXRecordDecl();
    if (BaseDecl == PrimaryBase)
      continue;

    // Get the base offset of this base.
    CharUnits BaseOffset = Base.getBaseOffset() + 
      Layout.getBaseClassOffset(BaseDecl);
    
    AddVCallOffsets(BaseSubobject(BaseDecl, BaseOffset), 
                    VBaseOffset);
  }
}

void 
VCallAndVBaseOffsetBuilder::AddVBaseOffsets(const CXXRecordDecl *RD,
                                            CharUnits OffsetInLayoutClass) {
  const ASTRecordLayout &LayoutClassLayout = 
    Context.getASTRecordLayout(LayoutClass);

  // Add vbase offsets.
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    const CXXRecordDecl *BaseDecl = I->getType()->getAsCXXRecordDecl();

    // Check if this is a virtual base that we haven't visited before.
    if (I->isVirtual() && VisitedVirtualBases.insert(BaseDecl)) {
      CharUnits Offset = 
        LayoutClassLayout.getVBaseClassOffset(BaseDecl) - OffsetInLayoutClass;

      // Add the vbase offset offset.
      assert(!VBaseOffsetOffsets.count(BaseDecl) &&
             "vbase offset offset already exists!");

      CharUnits VBaseOffsetOffset = getCurrentOffsetOffset();
      VBaseOffsetOffsets.insert(
          std::make_pair(BaseDecl, VBaseOffsetOffset));

      Components.push_back(
          VTableComponent::MakeVBaseOffset(Offset));
    }

    // Check the base class looking for more vbase offsets.
    AddVBaseOffsets(BaseDecl, OffsetInLayoutClass);
  }
}

/// ItaniumVTableBuilder - Class for building vtable layout information.
class ItaniumVTableBuilder {
public:
  /// PrimaryBasesSetVectorTy - A set vector of direct and indirect 
  /// primary bases.
  typedef llvm::SmallSetVector<const CXXRecordDecl *, 8> 
    PrimaryBasesSetVectorTy;
  
  typedef llvm::DenseMap<const CXXRecordDecl *, CharUnits> 
    VBaseOffsetOffsetsMapTy;
  
  typedef llvm::DenseMap<BaseSubobject, uint64_t> 
    AddressPointsMapTy;

  typedef llvm::DenseMap<GlobalDecl, int64_t> MethodVTableIndicesTy;

private:
  /// VTables - Global vtable information.
  ItaniumVTableContext &VTables;
  
  /// MostDerivedClass - The most derived class for which we're building this
  /// vtable.
  const CXXRecordDecl *MostDerivedClass;

  /// MostDerivedClassOffset - If we're building a construction vtable, this
  /// holds the offset from the layout class to the most derived class.
  const CharUnits MostDerivedClassOffset;
  
  /// MostDerivedClassIsVirtual - Whether the most derived class is a virtual 
  /// base. (This only makes sense when building a construction vtable).
  bool MostDerivedClassIsVirtual;
  
  /// LayoutClass - The class we're using for layout information. Will be 
  /// different than the most derived class if we're building a construction
  /// vtable.
  const CXXRecordDecl *LayoutClass;
  
  /// Context - The ASTContext which we will use for layout information.
  ASTContext &Context;
  
  /// FinalOverriders - The final overriders of the most derived class.
  const FinalOverriders Overriders;

  /// VCallOffsetsForVBases - Keeps track of vcall offsets for the virtual
  /// bases in this vtable.
  llvm::DenseMap<const CXXRecordDecl *, VCallOffsetMap> VCallOffsetsForVBases;

  /// VBaseOffsetOffsets - Contains the offsets of the virtual base offsets for
  /// the most derived class.
  VBaseOffsetOffsetsMapTy VBaseOffsetOffsets;
  
  /// Components - The components of the vtable being built.
  SmallVector<VTableComponent, 64> Components;

  /// AddressPoints - Address points for the vtable being built.
  AddressPointsMapTy AddressPoints;

  /// MethodInfo - Contains information about a method in a vtable.
  /// (Used for computing 'this' pointer adjustment thunks.
  struct MethodInfo {
    /// BaseOffset - The base offset of this method.
    const CharUnits BaseOffset;
    
    /// BaseOffsetInLayoutClass - The base offset in the layout class of this
    /// method.
    const CharUnits BaseOffsetInLayoutClass;
    
    /// VTableIndex - The index in the vtable that this method has.
    /// (For destructors, this is the index of the complete destructor).
    const uint64_t VTableIndex;
    
    MethodInfo(CharUnits BaseOffset, CharUnits BaseOffsetInLayoutClass, 
               uint64_t VTableIndex)
      : BaseOffset(BaseOffset), 
      BaseOffsetInLayoutClass(BaseOffsetInLayoutClass),
      VTableIndex(VTableIndex) { }
    
    MethodInfo() 
      : BaseOffset(CharUnits::Zero()), 
      BaseOffsetInLayoutClass(CharUnits::Zero()), 
      VTableIndex(0) { }
  };
  
  typedef llvm::DenseMap<const CXXMethodDecl *, MethodInfo> MethodInfoMapTy;
  
  /// MethodInfoMap - The information for all methods in the vtable we're
  /// currently building.
  MethodInfoMapTy MethodInfoMap;

  /// MethodVTableIndices - Contains the index (relative to the vtable address
  /// point) where the function pointer for a virtual function is stored.
  MethodVTableIndicesTy MethodVTableIndices;

  typedef llvm::DenseMap<uint64_t, ThunkInfo> VTableThunksMapTy;
  
  /// VTableThunks - The thunks by vtable index in the vtable currently being 
  /// built.
  VTableThunksMapTy VTableThunks;

  typedef SmallVector<ThunkInfo, 1> ThunkInfoVectorTy;
  typedef llvm::DenseMap<const CXXMethodDecl *, ThunkInfoVectorTy> ThunksMapTy;
  
  /// Thunks - A map that contains all the thunks needed for all methods in the
  /// most derived class for which the vtable is currently being built.
  ThunksMapTy Thunks;
  
  /// AddThunk - Add a thunk for the given method.
  void AddThunk(const CXXMethodDecl *MD, const ThunkInfo &Thunk);
  
  /// ComputeThisAdjustments - Compute the 'this' pointer adjustments for the
  /// part of the vtable we're currently building.
  void ComputeThisAdjustments();
  
  typedef llvm::SmallPtrSet<const CXXRecordDecl *, 4> VisitedVirtualBasesSetTy;

  /// PrimaryVirtualBases - All known virtual bases who are a primary base of
  /// some other base.
  VisitedVirtualBasesSetTy PrimaryVirtualBases;

  /// ComputeReturnAdjustment - Compute the return adjustment given a return
  /// adjustment base offset.
  ReturnAdjustment ComputeReturnAdjustment(BaseOffset Offset);
  
  /// ComputeThisAdjustmentBaseOffset - Compute the base offset for adjusting
  /// the 'this' pointer from the base subobject to the derived subobject.
  BaseOffset ComputeThisAdjustmentBaseOffset(BaseSubobject Base,
                                             BaseSubobject Derived) const;

  /// ComputeThisAdjustment - Compute the 'this' pointer adjustment for the
  /// given virtual member function, its offset in the layout class and its
  /// final overrider.
  ThisAdjustment 
  ComputeThisAdjustment(const CXXMethodDecl *MD, 
                        CharUnits BaseOffsetInLayoutClass,
                        FinalOverriders::OverriderInfo Overrider);

  /// AddMethod - Add a single virtual member function to the vtable
  /// components vector.
  void AddMethod(const CXXMethodDecl *MD, ReturnAdjustment ReturnAdjustment);

  /// IsOverriderUsed - Returns whether the overrider will ever be used in this
  /// part of the vtable. 
  ///
  /// Itanium C++ ABI 2.5.2:
  ///
  ///   struct A { virtual void f(); };
  ///   struct B : virtual public A { int i; };
  ///   struct C : virtual public A { int j; };
  ///   struct D : public B, public C {};
  ///
  ///   When B and C are declared, A is a primary base in each case, so although
  ///   vcall offsets are allocated in the A-in-B and A-in-C vtables, no this
  ///   adjustment is required and no thunk is generated. However, inside D
  ///   objects, A is no longer a primary base of C, so if we allowed calls to
  ///   C::f() to use the copy of A's vtable in the C subobject, we would need
  ///   to adjust this from C* to B::A*, which would require a third-party 
  ///   thunk. Since we require that a call to C::f() first convert to A*, 
  ///   C-in-D's copy of A's vtable is never referenced, so this is not 
  ///   necessary.
  bool IsOverriderUsed(const CXXMethodDecl *Overrider,
                       CharUnits BaseOffsetInLayoutClass,
                       const CXXRecordDecl *FirstBaseInPrimaryBaseChain,
                       CharUnits FirstBaseOffsetInLayoutClass) const;

  
  /// AddMethods - Add the methods of this base subobject and all its
  /// primary bases to the vtable components vector.
  void AddMethods(BaseSubobject Base, CharUnits BaseOffsetInLayoutClass,
                  const CXXRecordDecl *FirstBaseInPrimaryBaseChain,
                  CharUnits FirstBaseOffsetInLayoutClass,
                  PrimaryBasesSetVectorTy &PrimaryBases);

  // LayoutVTable - Layout the vtable for the given base class, including its
  // secondary vtables and any vtables for virtual bases.
  void LayoutVTable();

  /// LayoutPrimaryAndSecondaryVTables - Layout the primary vtable for the
  /// given base subobject, as well as all its secondary vtables.
  ///
  /// \param BaseIsMorallyVirtual whether the base subobject is a virtual base
  /// or a direct or indirect base of a virtual base.
  ///
  /// \param BaseIsVirtualInLayoutClass - Whether the base subobject is virtual
  /// in the layout class. 
  void LayoutPrimaryAndSecondaryVTables(BaseSubobject Base,
                                        bool BaseIsMorallyVirtual,
                                        bool BaseIsVirtualInLayoutClass,
                                        CharUnits OffsetInLayoutClass);
  
  /// LayoutSecondaryVTables - Layout the secondary vtables for the given base
  /// subobject.
  ///
  /// \param BaseIsMorallyVirtual whether the base subobject is a virtual base
  /// or a direct or indirect base of a virtual base.
  void LayoutSecondaryVTables(BaseSubobject Base, bool BaseIsMorallyVirtual,
                              CharUnits OffsetInLayoutClass);

  /// DeterminePrimaryVirtualBases - Determine the primary virtual bases in this
  /// class hierarchy.
  void DeterminePrimaryVirtualBases(const CXXRecordDecl *RD, 
                                    CharUnits OffsetInLayoutClass,
                                    VisitedVirtualBasesSetTy &VBases);

  /// LayoutVTablesForVirtualBases - Layout vtables for all virtual bases of the
  /// given base (excluding any primary bases).
  void LayoutVTablesForVirtualBases(const CXXRecordDecl *RD, 
                                    VisitedVirtualBasesSetTy &VBases);

  /// isBuildingConstructionVTable - Return whether this vtable builder is
  /// building a construction vtable.
  bool isBuildingConstructorVTable() const { 
    return MostDerivedClass != LayoutClass;
  }

public:
  ItaniumVTableBuilder(ItaniumVTableContext &VTables,
                       const CXXRecordDecl *MostDerivedClass,
                       CharUnits MostDerivedClassOffset,
                       bool MostDerivedClassIsVirtual,
                       const CXXRecordDecl *LayoutClass)
      : VTables(VTables), MostDerivedClass(MostDerivedClass),
        MostDerivedClassOffset(MostDerivedClassOffset),
        MostDerivedClassIsVirtual(MostDerivedClassIsVirtual),
        LayoutClass(LayoutClass), Context(MostDerivedClass->getASTContext()),
        Overriders(MostDerivedClass, MostDerivedClassOffset, LayoutClass) {
    assert(!Context.getTargetInfo().getCXXABI().isMicrosoft());

    LayoutVTable();

    if (Context.getLangOpts().DumpVTableLayouts)
      dumpLayout(llvm::outs());
  }

  uint64_t getNumThunks() const {
    return Thunks.size();
  }

  ThunksMapTy::const_iterator thunks_begin() const {
    return Thunks.begin();
  }

  ThunksMapTy::const_iterator thunks_end() const {
    return Thunks.end();
  }

  const VBaseOffsetOffsetsMapTy &getVBaseOffsetOffsets() const {
    return VBaseOffsetOffsets;
  }

  const AddressPointsMapTy &getAddressPoints() const {
    return AddressPoints;
  }

  MethodVTableIndicesTy::const_iterator vtable_indices_begin() const {
    return MethodVTableIndices.begin();
  }

  MethodVTableIndicesTy::const_iterator vtable_indices_end() const {
    return MethodVTableIndices.end();
  }

  /// getNumVTableComponents - Return the number of components in the vtable
  /// currently built.
  uint64_t getNumVTableComponents() const {
    return Components.size();
  }

  const VTableComponent *vtable_component_begin() const {
    return Components.begin();
  }
  
  const VTableComponent *vtable_component_end() const {
    return Components.end();
  }
  
  AddressPointsMapTy::const_iterator address_points_begin() const {
    return AddressPoints.begin();
  }

  AddressPointsMapTy::const_iterator address_points_end() const {
    return AddressPoints.end();
  }

  VTableThunksMapTy::const_iterator vtable_thunks_begin() const {
    return VTableThunks.begin();
  }

  VTableThunksMapTy::const_iterator vtable_thunks_end() const {
    return VTableThunks.end();
  }

  /// dumpLayout - Dump the vtable layout.
  void dumpLayout(raw_ostream&);
};

void ItaniumVTableBuilder::AddThunk(const CXXMethodDecl *MD,
                                    const ThunkInfo &Thunk) {
  assert(!isBuildingConstructorVTable() && 
         "Can't add thunks for construction vtable");

  SmallVectorImpl<ThunkInfo> &ThunksVector = Thunks[MD];

  // Check if we have this thunk already.
  if (std::find(ThunksVector.begin(), ThunksVector.end(), Thunk) != 
      ThunksVector.end())
    return;
  
  ThunksVector.push_back(Thunk);
}

typedef llvm::SmallPtrSet<const CXXMethodDecl *, 8> OverriddenMethodsSetTy;

/// Visit all the methods overridden by the given method recursively,
/// in a depth-first pre-order. The Visitor's visitor method returns a bool
/// indicating whether to continue the recursion for the given overridden
/// method (i.e. returning false stops the iteration).
template <class VisitorTy>
static void
visitAllOverriddenMethods(const CXXMethodDecl *MD, VisitorTy &Visitor) {
  assert(MD->isVirtual() && "Method is not virtual!");

  for (CXXMethodDecl::method_iterator I = MD->begin_overridden_methods(),
       E = MD->end_overridden_methods(); I != E; ++I) {
    const CXXMethodDecl *OverriddenMD = *I;
    if (!Visitor.visit(OverriddenMD))
      continue;
    visitAllOverriddenMethods(OverriddenMD, Visitor);
  }
}

namespace {
  struct OverriddenMethodsCollector {
    OverriddenMethodsSetTy *Methods;

    bool visit(const CXXMethodDecl *MD) {
      // Don't recurse on this method if we've already collected it.
      return Methods->insert(MD);
    }
  };
}

/// ComputeAllOverriddenMethods - Given a method decl, will return a set of all
/// the overridden methods that the function decl overrides.
static void
ComputeAllOverriddenMethods(const CXXMethodDecl *MD,
                            OverriddenMethodsSetTy& OverriddenMethods) {
  OverriddenMethodsCollector Collector = { &OverriddenMethods };
  visitAllOverriddenMethods(MD, Collector);
}

void ItaniumVTableBuilder::ComputeThisAdjustments() {
  // Now go through the method info map and see if any of the methods need
  // 'this' pointer adjustments.
  for (MethodInfoMapTy::const_iterator I = MethodInfoMap.begin(),
       E = MethodInfoMap.end(); I != E; ++I) {
    const CXXMethodDecl *MD = I->first;
    const MethodInfo &MethodInfo = I->second;

    // Ignore adjustments for unused function pointers.
    uint64_t VTableIndex = MethodInfo.VTableIndex;
    if (Components[VTableIndex].getKind() == 
        VTableComponent::CK_UnusedFunctionPointer)
      continue;
    
    // Get the final overrider for this method.
    FinalOverriders::OverriderInfo Overrider =
      Overriders.getOverrider(MD, MethodInfo.BaseOffset);
    
    // Check if we need an adjustment at all.
    if (MethodInfo.BaseOffsetInLayoutClass == Overrider.Offset) {
      // When a return thunk is needed by a derived class that overrides a
      // virtual base, gcc uses a virtual 'this' adjustment as well. 
      // While the thunk itself might be needed by vtables in subclasses or
      // in construction vtables, there doesn't seem to be a reason for using
      // the thunk in this vtable. Still, we do so to match gcc.
      if (VTableThunks.lookup(VTableIndex).Return.isEmpty())
        continue;
    }

    ThisAdjustment ThisAdjustment =
      ComputeThisAdjustment(MD, MethodInfo.BaseOffsetInLayoutClass, Overrider);

    if (ThisAdjustment.isEmpty())
      continue;

    // Add it.
    VTableThunks[VTableIndex].This = ThisAdjustment;

    if (isa<CXXDestructorDecl>(MD)) {
      // Add an adjustment for the deleting destructor as well.
      VTableThunks[VTableIndex + 1].This = ThisAdjustment;
    }
  }

  /// Clear the method info map.
  MethodInfoMap.clear();
  
  if (isBuildingConstructorVTable()) {
    // We don't need to store thunk information for construction vtables.
    return;
  }

  for (VTableThunksMapTy::const_iterator I = VTableThunks.begin(),
       E = VTableThunks.end(); I != E; ++I) {
    const VTableComponent &Component = Components[I->first];
    const ThunkInfo &Thunk = I->second;
    const CXXMethodDecl *MD;
    
    switch (Component.getKind()) {
    default:
      llvm_unreachable("Unexpected vtable component kind!");
    case VTableComponent::CK_FunctionPointer:
      MD = Component.getFunctionDecl();
      break;
    case VTableComponent::CK_CompleteDtorPointer:
      MD = Component.getDestructorDecl();
      break;
    case VTableComponent::CK_DeletingDtorPointer:
      // We've already added the thunk when we saw the complete dtor pointer.
      continue;
    }

    if (MD->getParent() == MostDerivedClass)
      AddThunk(MD, Thunk);
  }
}

ReturnAdjustment
ItaniumVTableBuilder::ComputeReturnAdjustment(BaseOffset Offset) {
  ReturnAdjustment Adjustment;
  
  if (!Offset.isEmpty()) {
    if (Offset.VirtualBase) {
      // Get the virtual base offset offset.
      if (Offset.DerivedClass == MostDerivedClass) {
        // We can get the offset offset directly from our map.
        Adjustment.Virtual.Itanium.VBaseOffsetOffset =
          VBaseOffsetOffsets.lookup(Offset.VirtualBase).getQuantity();
      } else {
        Adjustment.Virtual.Itanium.VBaseOffsetOffset =
          VTables.getVirtualBaseOffsetOffset(Offset.DerivedClass,
                                             Offset.VirtualBase).getQuantity();
      }
    }

    Adjustment.NonVirtual = Offset.NonVirtualOffset.getQuantity();
  }
  
  return Adjustment;
}

BaseOffset ItaniumVTableBuilder::ComputeThisAdjustmentBaseOffset(
    BaseSubobject Base, BaseSubobject Derived) const {
  const CXXRecordDecl *BaseRD = Base.getBase();
  const CXXRecordDecl *DerivedRD = Derived.getBase();
  
  CXXBasePaths Paths(/*FindAmbiguities=*/true,
                     /*RecordPaths=*/true, /*DetectVirtual=*/true);

  if (!DerivedRD->isDerivedFrom(BaseRD, Paths))
    llvm_unreachable("Class must be derived from the passed in base class!");

  // We have to go through all the paths, and see which one leads us to the
  // right base subobject.
  for (CXXBasePaths::const_paths_iterator I = Paths.begin(), E = Paths.end();
       I != E; ++I) {
    BaseOffset Offset = ComputeBaseOffset(Context, DerivedRD, *I);
    
    CharUnits OffsetToBaseSubobject = Offset.NonVirtualOffset;
    
    if (Offset.VirtualBase) {
      // If we have a virtual base class, the non-virtual offset is relative
      // to the virtual base class offset.
      const ASTRecordLayout &LayoutClassLayout =
        Context.getASTRecordLayout(LayoutClass);
      
      /// Get the virtual base offset, relative to the most derived class 
      /// layout.
      OffsetToBaseSubobject += 
        LayoutClassLayout.getVBaseClassOffset(Offset.VirtualBase);
    } else {
      // Otherwise, the non-virtual offset is relative to the derived class 
      // offset.
      OffsetToBaseSubobject += Derived.getBaseOffset();
    }
    
    // Check if this path gives us the right base subobject.
    if (OffsetToBaseSubobject == Base.getBaseOffset()) {
      // Since we're going from the base class _to_ the derived class, we'll
      // invert the non-virtual offset here.
      Offset.NonVirtualOffset = -Offset.NonVirtualOffset;
      return Offset;
    }      
  }
  
  return BaseOffset();
}

ThisAdjustment ItaniumVTableBuilder::ComputeThisAdjustment(
    const CXXMethodDecl *MD, CharUnits BaseOffsetInLayoutClass,
    FinalOverriders::OverriderInfo Overrider) {
  // Ignore adjustments for pure virtual member functions.
  if (Overrider.Method->isPure())
    return ThisAdjustment();
  
  BaseSubobject OverriddenBaseSubobject(MD->getParent(), 
                                        BaseOffsetInLayoutClass);
  
  BaseSubobject OverriderBaseSubobject(Overrider.Method->getParent(),
                                       Overrider.Offset);
  
  // Compute the adjustment offset.
  BaseOffset Offset = ComputeThisAdjustmentBaseOffset(OverriddenBaseSubobject,
                                                      OverriderBaseSubobject);
  if (Offset.isEmpty())
    return ThisAdjustment();

  ThisAdjustment Adjustment;
  
  if (Offset.VirtualBase) {
    // Get the vcall offset map for this virtual base.
    VCallOffsetMap &VCallOffsets = VCallOffsetsForVBases[Offset.VirtualBase];

    if (VCallOffsets.empty()) {
      // We don't have vcall offsets for this virtual base, go ahead and
      // build them.
      VCallAndVBaseOffsetBuilder Builder(MostDerivedClass, MostDerivedClass,
                                         /*FinalOverriders=*/0,
                                         BaseSubobject(Offset.VirtualBase,
                                                       CharUnits::Zero()),
                                         /*BaseIsVirtual=*/true,
                                         /*OffsetInLayoutClass=*/
                                             CharUnits::Zero());
        
      VCallOffsets = Builder.getVCallOffsets();
    }
      
    Adjustment.Virtual.Itanium.VCallOffsetOffset =
      VCallOffsets.getVCallOffsetOffset(MD).getQuantity();
  }

  // Set the non-virtual part of the adjustment.
  Adjustment.NonVirtual = Offset.NonVirtualOffset.getQuantity();
  
  return Adjustment;
}

void ItaniumVTableBuilder::AddMethod(const CXXMethodDecl *MD,
                                     ReturnAdjustment ReturnAdjustment) {
  if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(MD)) {
    assert(ReturnAdjustment.isEmpty() && 
           "Destructor can't have return adjustment!");

    // Add both the complete destructor and the deleting destructor.
    Components.push_back(VTableComponent::MakeCompleteDtor(DD));
    Components.push_back(VTableComponent::MakeDeletingDtor(DD));
  } else {
    // Add the return adjustment if necessary.
    if (!ReturnAdjustment.isEmpty())
      VTableThunks[Components.size()].Return = ReturnAdjustment;

    // Add the function.
    Components.push_back(VTableComponent::MakeFunction(MD));
  }
}

/// OverridesIndirectMethodInBase - Return whether the given member function
/// overrides any methods in the set of given bases. 
/// Unlike OverridesMethodInBase, this checks "overriders of overriders".
/// For example, if we have:
///
/// struct A { virtual void f(); }
/// struct B : A { virtual void f(); }
/// struct C : B { virtual void f(); }
///
/// OverridesIndirectMethodInBase will return true if given C::f as the method 
/// and { A } as the set of bases.
static bool OverridesIndirectMethodInBases(
    const CXXMethodDecl *MD,
    ItaniumVTableBuilder::PrimaryBasesSetVectorTy &Bases) {
  if (Bases.count(MD->getParent()))
    return true;
  
  for (CXXMethodDecl::method_iterator I = MD->begin_overridden_methods(),
       E = MD->end_overridden_methods(); I != E; ++I) {
    const CXXMethodDecl *OverriddenMD = *I;
    
    // Check "indirect overriders".
    if (OverridesIndirectMethodInBases(OverriddenMD, Bases))
      return true;
  }
   
  return false;
}

bool ItaniumVTableBuilder::IsOverriderUsed(
    const CXXMethodDecl *Overrider, CharUnits BaseOffsetInLayoutClass,
    const CXXRecordDecl *FirstBaseInPrimaryBaseChain,
    CharUnits FirstBaseOffsetInLayoutClass) const {
  // If the base and the first base in the primary base chain have the same
  // offsets, then this overrider will be used.
  if (BaseOffsetInLayoutClass == FirstBaseOffsetInLayoutClass)
   return true;

  // We know now that Base (or a direct or indirect base of it) is a primary
  // base in part of the class hierarchy, but not a primary base in the most 
  // derived class.
  
  // If the overrider is the first base in the primary base chain, we know
  // that the overrider will be used.
  if (Overrider->getParent() == FirstBaseInPrimaryBaseChain)
    return true;

  ItaniumVTableBuilder::PrimaryBasesSetVectorTy PrimaryBases;

  const CXXRecordDecl *RD = FirstBaseInPrimaryBaseChain;
  PrimaryBases.insert(RD);

  // Now traverse the base chain, starting with the first base, until we find
  // the base that is no longer a primary base.
  while (true) {
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
    const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();
    
    if (!PrimaryBase)
      break;
    
    if (Layout.isPrimaryBaseVirtual()) {
      assert(Layout.getVBaseClassOffset(PrimaryBase).isZero() &&
             "Primary base should always be at offset 0!");

      const ASTRecordLayout &LayoutClassLayout =
        Context.getASTRecordLayout(LayoutClass);

      // Now check if this is the primary base that is not a primary base in the
      // most derived class.
      if (LayoutClassLayout.getVBaseClassOffset(PrimaryBase) !=
          FirstBaseOffsetInLayoutClass) {
        // We found it, stop walking the chain.
        break;
      }
    } else {
      assert(Layout.getBaseClassOffset(PrimaryBase).isZero() &&
             "Primary base should always be at offset 0!");
    }
    
    if (!PrimaryBases.insert(PrimaryBase))
      llvm_unreachable("Found a duplicate primary base!");

    RD = PrimaryBase;
  }
  
  // If the final overrider is an override of one of the primary bases,
  // then we know that it will be used.
  return OverridesIndirectMethodInBases(Overrider, PrimaryBases);
}

typedef llvm::SmallSetVector<const CXXRecordDecl *, 8> BasesSetVectorTy;

/// FindNearestOverriddenMethod - Given a method, returns the overridden method
/// from the nearest base. Returns null if no method was found.
/// The Bases are expected to be sorted in a base-to-derived order.
static const CXXMethodDecl *
FindNearestOverriddenMethod(const CXXMethodDecl *MD,
                            BasesSetVectorTy &Bases) {
  OverriddenMethodsSetTy OverriddenMethods;
  ComputeAllOverriddenMethods(MD, OverriddenMethods);
  
  for (int I = Bases.size(), E = 0; I != E; --I) {
    const CXXRecordDecl *PrimaryBase = Bases[I - 1];

    // Now check the overridden methods.
    for (OverriddenMethodsSetTy::const_iterator I = OverriddenMethods.begin(),
         E = OverriddenMethods.end(); I != E; ++I) {
      const CXXMethodDecl *OverriddenMD = *I;
      
      // We found our overridden method.
      if (OverriddenMD->getParent() == PrimaryBase)
        return OverriddenMD;
    }
  }
  
  return 0;
}

void ItaniumVTableBuilder::AddMethods(
    BaseSubobject Base, CharUnits BaseOffsetInLayoutClass,
    const CXXRecordDecl *FirstBaseInPrimaryBaseChain,
    CharUnits FirstBaseOffsetInLayoutClass,
    PrimaryBasesSetVectorTy &PrimaryBases) {
  // Itanium C++ ABI 2.5.2:
  //   The order of the virtual function pointers in a virtual table is the
  //   order of declaration of the corresponding member functions in the class.
  //
  //   There is an entry for any virtual function declared in a class,
  //   whether it is a new function or overrides a base class function,
  //   unless it overrides a function from the primary base, and conversion
  //   between their return types does not require an adjustment.

  const CXXRecordDecl *RD = Base.getBase();
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);

  if (const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase()) {
    CharUnits PrimaryBaseOffset;
    CharUnits PrimaryBaseOffsetInLayoutClass;
    if (Layout.isPrimaryBaseVirtual()) {
      assert(Layout.getVBaseClassOffset(PrimaryBase).isZero() &&
             "Primary vbase should have a zero offset!");
      
      const ASTRecordLayout &MostDerivedClassLayout =
        Context.getASTRecordLayout(MostDerivedClass);
      
      PrimaryBaseOffset = 
        MostDerivedClassLayout.getVBaseClassOffset(PrimaryBase);
      
      const ASTRecordLayout &LayoutClassLayout =
        Context.getASTRecordLayout(LayoutClass);

      PrimaryBaseOffsetInLayoutClass =
        LayoutClassLayout.getVBaseClassOffset(PrimaryBase);
    } else {
      assert(Layout.getBaseClassOffset(PrimaryBase).isZero() &&
             "Primary base should have a zero offset!");

      PrimaryBaseOffset = Base.getBaseOffset();
      PrimaryBaseOffsetInLayoutClass = BaseOffsetInLayoutClass;
    }

    AddMethods(BaseSubobject(PrimaryBase, PrimaryBaseOffset),
               PrimaryBaseOffsetInLayoutClass, FirstBaseInPrimaryBaseChain, 
               FirstBaseOffsetInLayoutClass, PrimaryBases);
    
    if (!PrimaryBases.insert(PrimaryBase))
      llvm_unreachable("Found a duplicate primary base!");
  }

  const CXXDestructorDecl *ImplicitVirtualDtor = 0;

  typedef llvm::SmallVector<const CXXMethodDecl *, 8> NewVirtualFunctionsTy;
  NewVirtualFunctionsTy NewVirtualFunctions;

  // Now go through all virtual member functions and add them.
  for (CXXRecordDecl::method_iterator I = RD->method_begin(),
       E = RD->method_end(); I != E; ++I) {
    const CXXMethodDecl *MD = *I;
  
    if (!MD->isVirtual())
      continue;

    // Get the final overrider.
    FinalOverriders::OverriderInfo Overrider = 
      Overriders.getOverrider(MD, Base.getBaseOffset());

    // Check if this virtual member function overrides a method in a primary
    // base. If this is the case, and the return type doesn't require adjustment
    // then we can just use the member function from the primary base.
    if (const CXXMethodDecl *OverriddenMD = 
          FindNearestOverriddenMethod(MD, PrimaryBases)) {
      if (ComputeReturnAdjustmentBaseOffset(Context, MD, 
                                            OverriddenMD).isEmpty()) {
        // Replace the method info of the overridden method with our own
        // method.
        assert(MethodInfoMap.count(OverriddenMD) && 
               "Did not find the overridden method!");
        MethodInfo &OverriddenMethodInfo = MethodInfoMap[OverriddenMD];
        
        MethodInfo MethodInfo(Base.getBaseOffset(), BaseOffsetInLayoutClass,
                              OverriddenMethodInfo.VTableIndex);

        assert(!MethodInfoMap.count(MD) &&
               "Should not have method info for this method yet!");
        
        MethodInfoMap.insert(std::make_pair(MD, MethodInfo));
        MethodInfoMap.erase(OverriddenMD);
        
        // If the overridden method exists in a virtual base class or a direct
        // or indirect base class of a virtual base class, we need to emit a
        // thunk if we ever have a class hierarchy where the base class is not
        // a primary base in the complete object.
        if (!isBuildingConstructorVTable() && OverriddenMD != MD) {
          // Compute the this adjustment.
          ThisAdjustment ThisAdjustment =
            ComputeThisAdjustment(OverriddenMD, BaseOffsetInLayoutClass,
                                  Overrider);

          if (ThisAdjustment.Virtual.Itanium.VCallOffsetOffset &&
              Overrider.Method->getParent() == MostDerivedClass) {

            // There's no return adjustment from OverriddenMD and MD,
            // but that doesn't mean there isn't one between MD and
            // the final overrider.
            BaseOffset ReturnAdjustmentOffset =
              ComputeReturnAdjustmentBaseOffset(Context, Overrider.Method, MD);
            ReturnAdjustment ReturnAdjustment = 
              ComputeReturnAdjustment(ReturnAdjustmentOffset);

            // This is a virtual thunk for the most derived class, add it.
            AddThunk(Overrider.Method, 
                     ThunkInfo(ThisAdjustment, ReturnAdjustment));
          }
        }

        continue;
      }
    }

    if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(MD)) {
      if (MD->isImplicit()) {
        // Itanium C++ ABI 2.5.2:
        //   If a class has an implicitly-defined virtual destructor,
        //   its entries come after the declared virtual function pointers.

        assert(!ImplicitVirtualDtor &&
               "Did already see an implicit virtual dtor!");
        ImplicitVirtualDtor = DD;
        continue;
      }
    }

    NewVirtualFunctions.push_back(MD);
  }

  if (ImplicitVirtualDtor)
    NewVirtualFunctions.push_back(ImplicitVirtualDtor);

  for (NewVirtualFunctionsTy::const_iterator I = NewVirtualFunctions.begin(),
       E = NewVirtualFunctions.end(); I != E; ++I) {
    const CXXMethodDecl *MD = *I;

    // Get the final overrider.
    FinalOverriders::OverriderInfo Overrider =
      Overriders.getOverrider(MD, Base.getBaseOffset());

    // Insert the method info for this method.
    MethodInfo MethodInfo(Base.getBaseOffset(), BaseOffsetInLayoutClass,
                          Components.size());

    assert(!MethodInfoMap.count(MD) &&
           "Should not have method info for this method yet!");
    MethodInfoMap.insert(std::make_pair(MD, MethodInfo));

    // Check if this overrider is going to be used.
    const CXXMethodDecl *OverriderMD = Overrider.Method;
    if (!IsOverriderUsed(OverriderMD, BaseOffsetInLayoutClass,
                         FirstBaseInPrimaryBaseChain, 
                         FirstBaseOffsetInLayoutClass)) {
      Components.push_back(VTableComponent::MakeUnusedFunction(OverriderMD));
      continue;
    }

    // Check if this overrider needs a return adjustment.
    // We don't want to do this for pure virtual member functions.
    BaseOffset ReturnAdjustmentOffset;
    if (!OverriderMD->isPure()) {
      ReturnAdjustmentOffset = 
        ComputeReturnAdjustmentBaseOffset(Context, OverriderMD, MD);
    }

    ReturnAdjustment ReturnAdjustment = 
      ComputeReturnAdjustment(ReturnAdjustmentOffset);
    
    AddMethod(Overrider.Method, ReturnAdjustment);
  }
}

void ItaniumVTableBuilder::LayoutVTable() {
  LayoutPrimaryAndSecondaryVTables(BaseSubobject(MostDerivedClass,
                                                 CharUnits::Zero()),
                                   /*BaseIsMorallyVirtual=*/false,
                                   MostDerivedClassIsVirtual,
                                   MostDerivedClassOffset);
  
  VisitedVirtualBasesSetTy VBases;
  
  // Determine the primary virtual bases.
  DeterminePrimaryVirtualBases(MostDerivedClass, MostDerivedClassOffset, 
                               VBases);
  VBases.clear();
  
  LayoutVTablesForVirtualBases(MostDerivedClass, VBases);

  // -fapple-kext adds an extra entry at end of vtbl.
  bool IsAppleKext = Context.getLangOpts().AppleKext;
  if (IsAppleKext)
    Components.push_back(VTableComponent::MakeVCallOffset(CharUnits::Zero()));
}

void ItaniumVTableBuilder::LayoutPrimaryAndSecondaryVTables(
    BaseSubobject Base, bool BaseIsMorallyVirtual,
    bool BaseIsVirtualInLayoutClass, CharUnits OffsetInLayoutClass) {
  assert(Base.getBase()->isDynamicClass() && "class does not have a vtable!");

  // Add vcall and vbase offsets for this vtable.
  VCallAndVBaseOffsetBuilder Builder(MostDerivedClass, LayoutClass, &Overriders,
                                     Base, BaseIsVirtualInLayoutClass, 
                                     OffsetInLayoutClass);
  Components.append(Builder.components_begin(), Builder.components_end());
  
  // Check if we need to add these vcall offsets.
  if (BaseIsVirtualInLayoutClass && !Builder.getVCallOffsets().empty()) {
    VCallOffsetMap &VCallOffsets = VCallOffsetsForVBases[Base.getBase()];
    
    if (VCallOffsets.empty())
      VCallOffsets = Builder.getVCallOffsets();
  }

  // If we're laying out the most derived class we want to keep track of the
  // virtual base class offset offsets.
  if (Base.getBase() == MostDerivedClass)
    VBaseOffsetOffsets = Builder.getVBaseOffsetOffsets();

  // Add the offset to top.
  CharUnits OffsetToTop = MostDerivedClassOffset - OffsetInLayoutClass;
  Components.push_back(VTableComponent::MakeOffsetToTop(OffsetToTop));

  // Next, add the RTTI.
  Components.push_back(VTableComponent::MakeRTTI(MostDerivedClass));

  uint64_t AddressPoint = Components.size();

  // Now go through all virtual member functions and add them.
  PrimaryBasesSetVectorTy PrimaryBases;
  AddMethods(Base, OffsetInLayoutClass,
             Base.getBase(), OffsetInLayoutClass, 
             PrimaryBases);

  const CXXRecordDecl *RD = Base.getBase();
  if (RD == MostDerivedClass) {
    assert(MethodVTableIndices.empty());
    for (MethodInfoMapTy::const_iterator I = MethodInfoMap.begin(),
         E = MethodInfoMap.end(); I != E; ++I) {
      const CXXMethodDecl *MD = I->first;
      const MethodInfo &MI = I->second;
      if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(MD)) {
        MethodVTableIndices[GlobalDecl(DD, Dtor_Complete)]
            = MI.VTableIndex - AddressPoint;
        MethodVTableIndices[GlobalDecl(DD, Dtor_Deleting)]
            = MI.VTableIndex + 1 - AddressPoint;
      } else {
        MethodVTableIndices[MD] = MI.VTableIndex - AddressPoint;
      }
    }
  }

  // Compute 'this' pointer adjustments.
  ComputeThisAdjustments();

  // Add all address points.
  while (true) {
    AddressPoints.insert(std::make_pair(
      BaseSubobject(RD, OffsetInLayoutClass),
      AddressPoint));

    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
    const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();
    
    if (!PrimaryBase)
      break;
    
    if (Layout.isPrimaryBaseVirtual()) {
      // Check if this virtual primary base is a primary base in the layout
      // class. If it's not, we don't want to add it.
      const ASTRecordLayout &LayoutClassLayout =
        Context.getASTRecordLayout(LayoutClass);

      if (LayoutClassLayout.getVBaseClassOffset(PrimaryBase) !=
          OffsetInLayoutClass) {
        // We don't want to add this class (or any of its primary bases).
        break;
      }
    }

    RD = PrimaryBase;
  }

  // Layout secondary vtables.
  LayoutSecondaryVTables(Base, BaseIsMorallyVirtual, OffsetInLayoutClass);
}

void
ItaniumVTableBuilder::LayoutSecondaryVTables(BaseSubobject Base,
                                             bool BaseIsMorallyVirtual,
                                             CharUnits OffsetInLayoutClass) {
  // Itanium C++ ABI 2.5.2:
  //   Following the primary virtual table of a derived class are secondary 
  //   virtual tables for each of its proper base classes, except any primary
  //   base(s) with which it shares its primary virtual table.

  const CXXRecordDecl *RD = Base.getBase();
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
  const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();
  
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    // Ignore virtual bases, we'll emit them later.
    if (I->isVirtual())
      continue;
    
    const CXXRecordDecl *BaseDecl = I->getType()->getAsCXXRecordDecl();

    // Ignore bases that don't have a vtable.
    if (!BaseDecl->isDynamicClass())
      continue;

    if (isBuildingConstructorVTable()) {
      // Itanium C++ ABI 2.6.4:
      //   Some of the base class subobjects may not need construction virtual
      //   tables, which will therefore not be present in the construction
      //   virtual table group, even though the subobject virtual tables are
      //   present in the main virtual table group for the complete object.
      if (!BaseIsMorallyVirtual && !BaseDecl->getNumVBases())
        continue;
    }

    // Get the base offset of this base.
    CharUnits RelativeBaseOffset = Layout.getBaseClassOffset(BaseDecl);
    CharUnits BaseOffset = Base.getBaseOffset() + RelativeBaseOffset;
    
    CharUnits BaseOffsetInLayoutClass = 
      OffsetInLayoutClass + RelativeBaseOffset;
    
    // Don't emit a secondary vtable for a primary base. We might however want 
    // to emit secondary vtables for other bases of this base.
    if (BaseDecl == PrimaryBase) {
      LayoutSecondaryVTables(BaseSubobject(BaseDecl, BaseOffset),
                             BaseIsMorallyVirtual, BaseOffsetInLayoutClass);
      continue;
    }

    // Layout the primary vtable (and any secondary vtables) for this base.
    LayoutPrimaryAndSecondaryVTables(
      BaseSubobject(BaseDecl, BaseOffset),
      BaseIsMorallyVirtual,
      /*BaseIsVirtualInLayoutClass=*/false,
      BaseOffsetInLayoutClass);
  }
}

void ItaniumVTableBuilder::DeterminePrimaryVirtualBases(
    const CXXRecordDecl *RD, CharUnits OffsetInLayoutClass,
    VisitedVirtualBasesSetTy &VBases) {
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
  
  // Check if this base has a primary base.
  if (const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase()) {

    // Check if it's virtual.
    if (Layout.isPrimaryBaseVirtual()) {
      bool IsPrimaryVirtualBase = true;

      if (isBuildingConstructorVTable()) {
        // Check if the base is actually a primary base in the class we use for
        // layout.
        const ASTRecordLayout &LayoutClassLayout =
          Context.getASTRecordLayout(LayoutClass);

        CharUnits PrimaryBaseOffsetInLayoutClass =
          LayoutClassLayout.getVBaseClassOffset(PrimaryBase);
        
        // We know that the base is not a primary base in the layout class if 
        // the base offsets are different.
        if (PrimaryBaseOffsetInLayoutClass != OffsetInLayoutClass)
          IsPrimaryVirtualBase = false;
      }
        
      if (IsPrimaryVirtualBase)
        PrimaryVirtualBases.insert(PrimaryBase);
    }
  }

  // Traverse bases, looking for more primary virtual bases.
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    const CXXRecordDecl *BaseDecl = I->getType()->getAsCXXRecordDecl();

    CharUnits BaseOffsetInLayoutClass;
    
    if (I->isVirtual()) {
      if (!VBases.insert(BaseDecl))
        continue;
      
      const ASTRecordLayout &LayoutClassLayout =
        Context.getASTRecordLayout(LayoutClass);

      BaseOffsetInLayoutClass = 
        LayoutClassLayout.getVBaseClassOffset(BaseDecl);
    } else {
      BaseOffsetInLayoutClass = 
        OffsetInLayoutClass + Layout.getBaseClassOffset(BaseDecl);
    }

    DeterminePrimaryVirtualBases(BaseDecl, BaseOffsetInLayoutClass, VBases);
  }
}

void ItaniumVTableBuilder::LayoutVTablesForVirtualBases(
    const CXXRecordDecl *RD, VisitedVirtualBasesSetTy &VBases) {
  // Itanium C++ ABI 2.5.2:
  //   Then come the virtual base virtual tables, also in inheritance graph
  //   order, and again excluding primary bases (which share virtual tables with
  //   the classes for which they are primary).
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    const CXXRecordDecl *BaseDecl = I->getType()->getAsCXXRecordDecl();

    // Check if this base needs a vtable. (If it's virtual, not a primary base
    // of some other class, and we haven't visited it before).
    if (I->isVirtual() && BaseDecl->isDynamicClass() && 
        !PrimaryVirtualBases.count(BaseDecl) && VBases.insert(BaseDecl)) {
      const ASTRecordLayout &MostDerivedClassLayout =
        Context.getASTRecordLayout(MostDerivedClass);
      CharUnits BaseOffset = 
        MostDerivedClassLayout.getVBaseClassOffset(BaseDecl);
      
      const ASTRecordLayout &LayoutClassLayout =
        Context.getASTRecordLayout(LayoutClass);
      CharUnits BaseOffsetInLayoutClass = 
        LayoutClassLayout.getVBaseClassOffset(BaseDecl);

      LayoutPrimaryAndSecondaryVTables(
        BaseSubobject(BaseDecl, BaseOffset),
        /*BaseIsMorallyVirtual=*/true,
        /*BaseIsVirtualInLayoutClass=*/true,
        BaseOffsetInLayoutClass);
    }
    
    // We only need to check the base for virtual base vtables if it actually
    // has virtual bases.
    if (BaseDecl->getNumVBases())
      LayoutVTablesForVirtualBases(BaseDecl, VBases);
  }
}

struct ItaniumThunkInfoComparator {
  bool operator() (const ThunkInfo &LHS, const ThunkInfo &RHS) {
    assert(LHS.Method == 0);
    assert(RHS.Method == 0);

    if (LHS.This != RHS.This)
      return LHS.This < RHS.This;

    if (LHS.Return != RHS.Return)
      return LHS.Return < RHS.Return;

    return false;
  }
};

/// dumpLayout - Dump the vtable layout.
void ItaniumVTableBuilder::dumpLayout(raw_ostream &Out) {
  // FIXME: write more tests that actually use the dumpLayout output to prevent
  // ItaniumVTableBuilder regressions.

  if (isBuildingConstructorVTable()) {
    Out << "Construction vtable for ('";
    MostDerivedClass->printQualifiedName(Out);
    Out << "', ";
    Out << MostDerivedClassOffset.getQuantity() << ") in '";
    LayoutClass->printQualifiedName(Out);
  } else {
    Out << "Vtable for '";
    MostDerivedClass->printQualifiedName(Out);
  }
  Out << "' (" << Components.size() << " entries).\n";

  // Iterate through the address points and insert them into a new map where
  // they are keyed by the index and not the base object.
  // Since an address point can be shared by multiple subobjects, we use an
  // STL multimap.
  std::multimap<uint64_t, BaseSubobject> AddressPointsByIndex;
  for (AddressPointsMapTy::const_iterator I = AddressPoints.begin(), 
       E = AddressPoints.end(); I != E; ++I) {
    const BaseSubobject& Base = I->first;
    uint64_t Index = I->second;
    
    AddressPointsByIndex.insert(std::make_pair(Index, Base));
  }
  
  for (unsigned I = 0, E = Components.size(); I != E; ++I) {
    uint64_t Index = I;

    Out << llvm::format("%4d | ", I);

    const VTableComponent &Component = Components[I];

    // Dump the component.
    switch (Component.getKind()) {

    case VTableComponent::CK_VCallOffset:
      Out << "vcall_offset ("
          << Component.getVCallOffset().getQuantity() 
          << ")";
      break;

    case VTableComponent::CK_VBaseOffset:
      Out << "vbase_offset ("
          << Component.getVBaseOffset().getQuantity()
          << ")";
      break;

    case VTableComponent::CK_OffsetToTop:
      Out << "offset_to_top ("
          << Component.getOffsetToTop().getQuantity()
          << ")";
      break;
    
    case VTableComponent::CK_RTTI:
      Component.getRTTIDecl()->printQualifiedName(Out);
      Out << " RTTI";
      break;
    
    case VTableComponent::CK_FunctionPointer: {
      const CXXMethodDecl *MD = Component.getFunctionDecl();

      std::string Str = 
        PredefinedExpr::ComputeName(PredefinedExpr::PrettyFunctionNoVirtual, 
                                    MD);
      Out << Str;
      if (MD->isPure())
        Out << " [pure]";

      if (MD->isDeleted())
        Out << " [deleted]";

      ThunkInfo Thunk = VTableThunks.lookup(I);
      if (!Thunk.isEmpty()) {
        // If this function pointer has a return adjustment, dump it.
        if (!Thunk.Return.isEmpty()) {
          Out << "\n       [return adjustment: ";
          Out << Thunk.Return.NonVirtual << " non-virtual";
          
          if (Thunk.Return.Virtual.Itanium.VBaseOffsetOffset) {
            Out << ", " << Thunk.Return.Virtual.Itanium.VBaseOffsetOffset;
            Out << " vbase offset offset";
          }

          Out << ']';
        }

        // If this function pointer has a 'this' pointer adjustment, dump it.
        if (!Thunk.This.isEmpty()) {
          Out << "\n       [this adjustment: ";
          Out << Thunk.This.NonVirtual << " non-virtual";
          
          if (Thunk.This.Virtual.Itanium.VCallOffsetOffset) {
            Out << ", " << Thunk.This.Virtual.Itanium.VCallOffsetOffset;
            Out << " vcall offset offset";
          }

          Out << ']';
        }          
      }

      break;
    }

    case VTableComponent::CK_CompleteDtorPointer: 
    case VTableComponent::CK_DeletingDtorPointer: {
      bool IsComplete = 
        Component.getKind() == VTableComponent::CK_CompleteDtorPointer;
      
      const CXXDestructorDecl *DD = Component.getDestructorDecl();
      
      DD->printQualifiedName(Out);
      if (IsComplete)
        Out << "() [complete]";
      else
        Out << "() [deleting]";

      if (DD->isPure())
        Out << " [pure]";

      ThunkInfo Thunk = VTableThunks.lookup(I);
      if (!Thunk.isEmpty()) {
        // If this destructor has a 'this' pointer adjustment, dump it.
        if (!Thunk.This.isEmpty()) {
          Out << "\n       [this adjustment: ";
          Out << Thunk.This.NonVirtual << " non-virtual";
          
          if (Thunk.This.Virtual.Itanium.VCallOffsetOffset) {
            Out << ", " << Thunk.This.Virtual.Itanium.VCallOffsetOffset;
            Out << " vcall offset offset";
          }
          
          Out << ']';
        }          
      }        

      break;
    }

    case VTableComponent::CK_UnusedFunctionPointer: {
      const CXXMethodDecl *MD = Component.getUnusedFunctionDecl();

      std::string Str = 
        PredefinedExpr::ComputeName(PredefinedExpr::PrettyFunctionNoVirtual, 
                                    MD);
      Out << "[unused] " << Str;
      if (MD->isPure())
        Out << " [pure]";
    }

    }

    Out << '\n';
    
    // Dump the next address point.
    uint64_t NextIndex = Index + 1;
    if (AddressPointsByIndex.count(NextIndex)) {
      if (AddressPointsByIndex.count(NextIndex) == 1) {
        const BaseSubobject &Base = 
          AddressPointsByIndex.find(NextIndex)->second;
        
        Out << "       -- (";
        Base.getBase()->printQualifiedName(Out);
        Out << ", " << Base.getBaseOffset().getQuantity();
        Out << ") vtable address --\n";
      } else {
        CharUnits BaseOffset =
          AddressPointsByIndex.lower_bound(NextIndex)->second.getBaseOffset();
        
        // We store the class names in a set to get a stable order.
        std::set<std::string> ClassNames;
        for (std::multimap<uint64_t, BaseSubobject>::const_iterator I =
             AddressPointsByIndex.lower_bound(NextIndex), E =
             AddressPointsByIndex.upper_bound(NextIndex); I != E; ++I) {
          assert(I->second.getBaseOffset() == BaseOffset &&
                 "Invalid base offset!");
          const CXXRecordDecl *RD = I->second.getBase();
          ClassNames.insert(RD->getQualifiedNameAsString());
        }
        
        for (std::set<std::string>::const_iterator I = ClassNames.begin(),
             E = ClassNames.end(); I != E; ++I) {
          Out << "       -- (" << *I;
          Out << ", " << BaseOffset.getQuantity() << ") vtable address --\n";
        }
      }
    }
  }

  Out << '\n';
  
  if (isBuildingConstructorVTable())
    return;
  
  if (MostDerivedClass->getNumVBases()) {
    // We store the virtual base class names and their offsets in a map to get
    // a stable order.

    std::map<std::string, CharUnits> ClassNamesAndOffsets;
    for (VBaseOffsetOffsetsMapTy::const_iterator I = VBaseOffsetOffsets.begin(),
         E = VBaseOffsetOffsets.end(); I != E; ++I) {
      std::string ClassName = I->first->getQualifiedNameAsString();
      CharUnits OffsetOffset = I->second;
      ClassNamesAndOffsets.insert(
          std::make_pair(ClassName, OffsetOffset));
    }
    
    Out << "Virtual base offset offsets for '";
    MostDerivedClass->printQualifiedName(Out);
    Out << "' (";
    Out << ClassNamesAndOffsets.size();
    Out << (ClassNamesAndOffsets.size() == 1 ? " entry" : " entries") << ").\n";

    for (std::map<std::string, CharUnits>::const_iterator I =
         ClassNamesAndOffsets.begin(), E = ClassNamesAndOffsets.end(); 
         I != E; ++I)
      Out << "   " << I->first << " | " << I->second.getQuantity() << '\n';

    Out << "\n";
  }
  
  if (!Thunks.empty()) {
    // We store the method names in a map to get a stable order.
    std::map<std::string, const CXXMethodDecl *> MethodNamesAndDecls;
    
    for (ThunksMapTy::const_iterator I = Thunks.begin(), E = Thunks.end();
         I != E; ++I) {
      const CXXMethodDecl *MD = I->first;
      std::string MethodName = 
        PredefinedExpr::ComputeName(PredefinedExpr::PrettyFunctionNoVirtual,
                                    MD);
      
      MethodNamesAndDecls.insert(std::make_pair(MethodName, MD));
    }

    for (std::map<std::string, const CXXMethodDecl *>::const_iterator I =
         MethodNamesAndDecls.begin(), E = MethodNamesAndDecls.end(); 
         I != E; ++I) {
      const std::string &MethodName = I->first;
      const CXXMethodDecl *MD = I->second;

      ThunkInfoVectorTy ThunksVector = Thunks[MD];
      std::sort(ThunksVector.begin(), ThunksVector.end(),
                ItaniumThunkInfoComparator());

      Out << "Thunks for '" << MethodName << "' (" << ThunksVector.size();
      Out << (ThunksVector.size() == 1 ? " entry" : " entries") << ").\n";
      
      for (unsigned I = 0, E = ThunksVector.size(); I != E; ++I) {
        const ThunkInfo &Thunk = ThunksVector[I];

        Out << llvm::format("%4d | ", I);
        
        // If this function pointer has a return pointer adjustment, dump it.
        if (!Thunk.Return.isEmpty()) {
          Out << "return adjustment: " << Thunk.Return.NonVirtual;
          Out << " non-virtual";
          if (Thunk.Return.Virtual.Itanium.VBaseOffsetOffset) {
            Out << ", " << Thunk.Return.Virtual.Itanium.VBaseOffsetOffset;
            Out << " vbase offset offset";
          }

          if (!Thunk.This.isEmpty())
            Out << "\n       ";
        }

        // If this function pointer has a 'this' pointer adjustment, dump it.
        if (!Thunk.This.isEmpty()) {
          Out << "this adjustment: ";
          Out << Thunk.This.NonVirtual << " non-virtual";
          
          if (Thunk.This.Virtual.Itanium.VCallOffsetOffset) {
            Out << ", " << Thunk.This.Virtual.Itanium.VCallOffsetOffset;
            Out << " vcall offset offset";
          }
        }
        
        Out << '\n';
      }
      
      Out << '\n';
    }
  }

  // Compute the vtable indices for all the member functions.
  // Store them in a map keyed by the index so we'll get a sorted table.
  std::map<uint64_t, std::string> IndicesMap;

  for (CXXRecordDecl::method_iterator i = MostDerivedClass->method_begin(),
       e = MostDerivedClass->method_end(); i != e; ++i) {
    const CXXMethodDecl *MD = *i;
    
    // We only want virtual member functions.
    if (!MD->isVirtual())
      continue;

    std::string MethodName =
      PredefinedExpr::ComputeName(PredefinedExpr::PrettyFunctionNoVirtual,
                                  MD);

    if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(MD)) {
      GlobalDecl GD(DD, Dtor_Complete);
      assert(MethodVTableIndices.count(GD));
      uint64_t VTableIndex = MethodVTableIndices[GD];
      IndicesMap[VTableIndex] = MethodName + " [complete]";
      IndicesMap[VTableIndex + 1] = MethodName + " [deleting]";
    } else {
      assert(MethodVTableIndices.count(MD));
      IndicesMap[MethodVTableIndices[MD]] = MethodName;
    }
  }

  // Print the vtable indices for all the member functions.
  if (!IndicesMap.empty()) {
    Out << "VTable indices for '";
    MostDerivedClass->printQualifiedName(Out);
    Out << "' (" << IndicesMap.size() << " entries).\n";

    for (std::map<uint64_t, std::string>::const_iterator I = IndicesMap.begin(),
         E = IndicesMap.end(); I != E; ++I) {
      uint64_t VTableIndex = I->first;
      const std::string &MethodName = I->second;

      Out << llvm::format("%4" PRIu64 " | ", VTableIndex) << MethodName
          << '\n';
    }
  }

  Out << '\n';
}

struct VTableThunksComparator {
  bool operator()(const VTableLayout::VTableThunkTy &LHS,
                  const VTableLayout::VTableThunkTy &RHS) {
    if (LHS.first == RHS.first) {
      assert(LHS.second == RHS.second &&
             "Different thunks should have unique indices!");
    }
    return LHS.first < RHS.first;
  }
};
}

VTableLayout::VTableLayout(uint64_t NumVTableComponents,
                           const VTableComponent *VTableComponents,
                           uint64_t NumVTableThunks,
                           const VTableThunkTy *VTableThunks,
                           const AddressPointsMapTy &AddressPoints,
                           bool IsMicrosoftABI)
  : NumVTableComponents(NumVTableComponents),
    VTableComponents(new VTableComponent[NumVTableComponents]),
    NumVTableThunks(NumVTableThunks),
    VTableThunks(new VTableThunkTy[NumVTableThunks]),
    AddressPoints(AddressPoints),
    IsMicrosoftABI(IsMicrosoftABI) {
  std::copy(VTableComponents, VTableComponents+NumVTableComponents,
            this->VTableComponents.get());
  std::copy(VTableThunks, VTableThunks+NumVTableThunks,
            this->VTableThunks.get());
  std::sort(this->VTableThunks.get(),
            this->VTableThunks.get() + NumVTableThunks,
            VTableThunksComparator());
}

VTableLayout::~VTableLayout() { }

ItaniumVTableContext::ItaniumVTableContext(ASTContext &Context)
    : VTableContextBase(/*MS=*/false) {}

ItaniumVTableContext::~ItaniumVTableContext() {
  llvm::DeleteContainerSeconds(VTableLayouts);
}

uint64_t ItaniumVTableContext::getMethodVTableIndex(GlobalDecl GD) {
  MethodVTableIndicesTy::iterator I = MethodVTableIndices.find(GD);
  if (I != MethodVTableIndices.end())
    return I->second;
  
  const CXXRecordDecl *RD = cast<CXXMethodDecl>(GD.getDecl())->getParent();

  computeVTableRelatedInformation(RD);

  I = MethodVTableIndices.find(GD);
  assert(I != MethodVTableIndices.end() && "Did not find index!");
  return I->second;
}

CharUnits
ItaniumVTableContext::getVirtualBaseOffsetOffset(const CXXRecordDecl *RD,
                                                 const CXXRecordDecl *VBase) {
  ClassPairTy ClassPair(RD, VBase);
  
  VirtualBaseClassOffsetOffsetsMapTy::iterator I = 
    VirtualBaseClassOffsetOffsets.find(ClassPair);
  if (I != VirtualBaseClassOffsetOffsets.end())
    return I->second;
  
  VCallAndVBaseOffsetBuilder Builder(RD, RD, /*FinalOverriders=*/0,
                                     BaseSubobject(RD, CharUnits::Zero()),
                                     /*BaseIsVirtual=*/false,
                                     /*OffsetInLayoutClass=*/CharUnits::Zero());

  for (VCallAndVBaseOffsetBuilder::VBaseOffsetOffsetsMapTy::const_iterator I =
       Builder.getVBaseOffsetOffsets().begin(), 
       E = Builder.getVBaseOffsetOffsets().end(); I != E; ++I) {
    // Insert all types.
    ClassPairTy ClassPair(RD, I->first);
    
    VirtualBaseClassOffsetOffsets.insert(
        std::make_pair(ClassPair, I->second));
  }
  
  I = VirtualBaseClassOffsetOffsets.find(ClassPair);
  assert(I != VirtualBaseClassOffsetOffsets.end() && "Did not find index!");
  
  return I->second;
}

static VTableLayout *CreateVTableLayout(const ItaniumVTableBuilder &Builder) {
  SmallVector<VTableLayout::VTableThunkTy, 1>
    VTableThunks(Builder.vtable_thunks_begin(), Builder.vtable_thunks_end());

  return new VTableLayout(Builder.getNumVTableComponents(),
                          Builder.vtable_component_begin(),
                          VTableThunks.size(),
                          VTableThunks.data(),
                          Builder.getAddressPoints(),
                          /*IsMicrosoftABI=*/false);
}

void
ItaniumVTableContext::computeVTableRelatedInformation(const CXXRecordDecl *RD) {
  const VTableLayout *&Entry = VTableLayouts[RD];

  // Check if we've computed this information before.
  if (Entry)
    return;

  ItaniumVTableBuilder Builder(*this, RD, CharUnits::Zero(),
                               /*MostDerivedClassIsVirtual=*/0, RD);
  Entry = CreateVTableLayout(Builder);

  MethodVTableIndices.insert(Builder.vtable_indices_begin(),
                             Builder.vtable_indices_end());

  // Add the known thunks.
  Thunks.insert(Builder.thunks_begin(), Builder.thunks_end());

  // If we don't have the vbase information for this class, insert it.
  // getVirtualBaseOffsetOffset will compute it separately without computing
  // the rest of the vtable related information.
  if (!RD->getNumVBases())
    return;
  
  const CXXRecordDecl *VBase =
    RD->vbases_begin()->getType()->getAsCXXRecordDecl();
  
  if (VirtualBaseClassOffsetOffsets.count(std::make_pair(RD, VBase)))
    return;

  for (ItaniumVTableBuilder::VBaseOffsetOffsetsMapTy::const_iterator
           I = Builder.getVBaseOffsetOffsets().begin(),
           E = Builder.getVBaseOffsetOffsets().end();
       I != E; ++I) {
    // Insert all types.
    ClassPairTy ClassPair(RD, I->first);
    
    VirtualBaseClassOffsetOffsets.insert(std::make_pair(ClassPair, I->second));
  }
}

VTableLayout *ItaniumVTableContext::createConstructionVTableLayout(
    const CXXRecordDecl *MostDerivedClass, CharUnits MostDerivedClassOffset,
    bool MostDerivedClassIsVirtual, const CXXRecordDecl *LayoutClass) {
  ItaniumVTableBuilder Builder(*this, MostDerivedClass, MostDerivedClassOffset,
                               MostDerivedClassIsVirtual, LayoutClass);
  return CreateVTableLayout(Builder);
}

namespace {

// Vtables in the Microsoft ABI are different from the Itanium ABI.
//
// The main differences are:
//  1. Separate vftable and vbtable.
//
//  2. Each subobject with a vfptr gets its own vftable rather than an address
//     point in a single vtable shared between all the subobjects.
//     Each vftable is represented by a separate section and virtual calls
//     must be done using the vftable which has a slot for the function to be
//     called.
//
//  3. Virtual method definitions expect their 'this' parameter to point to the
//     first vfptr whose table provides a compatible overridden method.  In many
//     cases, this permits the original vf-table entry to directly call
//     the method instead of passing through a thunk.
//
//     A compatible overridden method is one which does not have a non-trivial
//     covariant-return adjustment.
//
//     The first vfptr is the one with the lowest offset in the complete-object
//     layout of the defining class, and the method definition will subtract
//     that constant offset from the parameter value to get the real 'this'
//     value.  Therefore, if the offset isn't really constant (e.g. if a virtual
//     function defined in a virtual base is overridden in a more derived
//     virtual base and these bases have a reverse order in the complete
//     object), the vf-table may require a this-adjustment thunk.
//
//  4. vftables do not contain new entries for overrides that merely require
//     this-adjustment.  Together with #3, this keeps vf-tables smaller and
//     eliminates the need for this-adjustment thunks in many cases, at the cost
//     of often requiring redundant work to adjust the "this" pointer.
//
//  5. Instead of VTT and constructor vtables, vbtables and vtordisps are used.
//     Vtordisps are emitted into the class layout if a class has
//      a) a user-defined ctor/dtor
//     and
//      b) a method overriding a method in a virtual base.

class VFTableBuilder {
public:
  typedef MicrosoftVTableContext::MethodVFTableLocation MethodVFTableLocation;

  typedef llvm::DenseMap<GlobalDecl, MethodVFTableLocation>
    MethodVFTableLocationsTy;

private:
  /// VTables - Global vtable information.
  MicrosoftVTableContext &VTables;

  /// Context - The ASTContext which we will use for layout information.
  ASTContext &Context;

  /// MostDerivedClass - The most derived class for which we're building this
  /// vtable.
  const CXXRecordDecl *MostDerivedClass;

  const ASTRecordLayout &MostDerivedClassLayout;

  const VPtrInfo &WhichVFPtr;

  /// FinalOverriders - The final overriders of the most derived class.
  const FinalOverriders Overriders;

  /// Components - The components of the vftable being built.
  SmallVector<VTableComponent, 64> Components;

  MethodVFTableLocationsTy MethodVFTableLocations;

  /// MethodInfo - Contains information about a method in a vtable.
  /// (Used for computing 'this' pointer adjustment thunks.
  struct MethodInfo {
    /// VBTableIndex - The nonzero index in the vbtable that
    /// this method's base has, or zero.
    const uint64_t VBTableIndex;

    /// VFTableIndex - The index in the vftable that this method has.
    const uint64_t VFTableIndex;

    /// Shadowed - Indicates if this vftable slot is shadowed by
    /// a slot for a covariant-return override. If so, it shouldn't be printed
    /// or used for vcalls in the most derived class.
    bool Shadowed;

    MethodInfo(uint64_t VBTableIndex, uint64_t VFTableIndex)
        : VBTableIndex(VBTableIndex), VFTableIndex(VFTableIndex),
          Shadowed(false) {}

    MethodInfo() : VBTableIndex(0), VFTableIndex(0), Shadowed(false) {}
  };

  typedef llvm::DenseMap<const CXXMethodDecl *, MethodInfo> MethodInfoMapTy;

  /// MethodInfoMap - The information for all methods in the vftable we're
  /// currently building.
  MethodInfoMapTy MethodInfoMap;

  typedef llvm::DenseMap<uint64_t, ThunkInfo> VTableThunksMapTy;

  /// VTableThunks - The thunks by vftable index in the vftable currently being
  /// built.
  VTableThunksMapTy VTableThunks;

  typedef SmallVector<ThunkInfo, 1> ThunkInfoVectorTy;
  typedef llvm::DenseMap<const CXXMethodDecl *, ThunkInfoVectorTy> ThunksMapTy;

  /// Thunks - A map that contains all the thunks needed for all methods in the
  /// most derived class for which the vftable is currently being built.
  ThunksMapTy Thunks;

  /// AddThunk - Add a thunk for the given method.
  void AddThunk(const CXXMethodDecl *MD, const ThunkInfo &Thunk) {
    SmallVector<ThunkInfo, 1> &ThunksVector = Thunks[MD];

    // Check if we have this thunk already.
    if (std::find(ThunksVector.begin(), ThunksVector.end(), Thunk) !=
        ThunksVector.end())
      return;

    ThunksVector.push_back(Thunk);
  }

  /// ComputeThisOffset - Returns the 'this' argument offset for the given
  /// method in the given subobject, relative to the beginning of the
  /// MostDerivedClass.
  CharUnits ComputeThisOffset(const CXXMethodDecl *MD,
                              BaseSubobject Base,
                              FinalOverriders::OverriderInfo Overrider);

  void CalculateVtordispAdjustment(FinalOverriders::OverriderInfo Overrider,
                                   CharUnits ThisOffset, ThisAdjustment &TA);

  /// AddMethod - Add a single virtual member function to the vftable
  /// components vector.
  void AddMethod(const CXXMethodDecl *MD, ThunkInfo TI) {
    if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(MD)) {
      assert(TI.Return.isEmpty() &&
             "Destructor can't have return adjustment!");
      Components.push_back(VTableComponent::MakeDeletingDtor(DD));
    } else {
      if (!TI.isEmpty())
        VTableThunks[Components.size()] = TI;
      Components.push_back(VTableComponent::MakeFunction(MD));
    }
  }

  bool NeedsReturnAdjustingThunk(const CXXMethodDecl *MD);

  /// AddMethods - Add the methods of this base subobject and the relevant
  /// subbases to the vftable we're currently laying out.
  void AddMethods(BaseSubobject Base, unsigned BaseDepth,
                  const CXXRecordDecl *LastVBase,
                  BasesSetVectorTy &VisitedBases);

  void CheckBadVirtualInheritanceHierarchy() {
    // We fail at this-adjustment for virtual methods inherited from
    // non-virtual bases that overrides a method in a virtual base.
    if (Context.getLangOpts().DumpVTableLayouts)
      return;
    for (CXXRecordDecl::base_class_const_iterator BI =
        MostDerivedClass->bases_begin(), BE = MostDerivedClass->bases_end();
        BI != BE; ++BI) {
      const CXXRecordDecl *Base = BI->getType()->getAsCXXRecordDecl();
      if (BI->isVirtual())
        continue;
      for (CXXRecordDecl::method_iterator I = Base->method_begin(),
          E = Base->method_end(); I != E; ++I) {
        const CXXMethodDecl *Method = *I;
        if (!Method->isVirtual())
          continue;
        if (isa<CXXDestructorDecl>(Method))
          continue;
        OverriddenMethodsSetTy OverriddenMethods;
        ComputeAllOverriddenMethods(Method, OverriddenMethods);
        for (OverriddenMethodsSetTy::const_iterator I =
            OverriddenMethods.begin(),
            E = OverriddenMethods.end(); I != E; ++I) {
          const CXXMethodDecl *Overridden = *I;
          if (Base->isVirtuallyDerivedFrom(Overridden->getParent())) {
            ErrorUnsupported("classes with non-virtual base "
                "classes that override methods in virtual bases",
                BI->getLocStart());
            return;
          }
        }
      }
    }
  }

  void LayoutVFTable() {
    // FIXME: add support for RTTI when we have proper LLVM support for symbols
    // pointing to the middle of a section.

    CheckBadVirtualInheritanceHierarchy();

    BasesSetVectorTy VisitedBases;
    AddMethods(BaseSubobject(MostDerivedClass, CharUnits::Zero()), 0, 0,
               VisitedBases);

    assert(MethodVFTableLocations.empty());
    for (MethodInfoMapTy::const_iterator I = MethodInfoMap.begin(),
         E = MethodInfoMap.end(); I != E; ++I) {
      const CXXMethodDecl *MD = I->first;
      const MethodInfo &MI = I->second;
      // Skip the methods that the MostDerivedClass didn't override
      // and the entries shadowed by return adjusting thunks.
      if (MD->getParent() != MostDerivedClass || MI.Shadowed)
        continue;
      MethodVFTableLocation Loc(MI.VBTableIndex, WhichVFPtr.getVBaseWithVPtr(),
                                WhichVFPtr.NonVirtualOffset, MI.VFTableIndex);
      if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(MD)) {
        MethodVFTableLocations[GlobalDecl(DD, Dtor_Deleting)] = Loc;
      } else {
        MethodVFTableLocations[MD] = Loc;
      }
    }
  }

  void ErrorUnsupported(StringRef Feature, SourceLocation Location) {
    clang::DiagnosticsEngine &Diags = Context.getDiagnostics();
    unsigned DiagID = Diags.getCustomDiagID(
        DiagnosticsEngine::Error, "v-table layout for %0 is not supported yet");
    Diags.Report(Context.getFullLoc(Location), DiagID) << Feature;
  }

public:
  VFTableBuilder(MicrosoftVTableContext &VTables,
                 const CXXRecordDecl *MostDerivedClass, const VPtrInfo *Which)
      : VTables(VTables),
        Context(MostDerivedClass->getASTContext()),
        MostDerivedClass(MostDerivedClass),
        MostDerivedClassLayout(Context.getASTRecordLayout(MostDerivedClass)),
        WhichVFPtr(*Which),
        Overriders(MostDerivedClass, CharUnits(), MostDerivedClass) {
    LayoutVFTable();

    if (Context.getLangOpts().DumpVTableLayouts)
      dumpLayout(llvm::outs());
  }

  uint64_t getNumThunks() const { return Thunks.size(); }

  ThunksMapTy::const_iterator thunks_begin() const { return Thunks.begin(); }

  ThunksMapTy::const_iterator thunks_end() const { return Thunks.end(); }

  MethodVFTableLocationsTy::const_iterator vtable_indices_begin() const {
    return MethodVFTableLocations.begin();
  }

  MethodVFTableLocationsTy::const_iterator vtable_indices_end() const {
    return MethodVFTableLocations.end();
  }

  uint64_t getNumVTableComponents() const { return Components.size(); }

  const VTableComponent *vtable_component_begin() const {
    return Components.begin();
  }

  const VTableComponent *vtable_component_end() const {
    return Components.end();
  }

  VTableThunksMapTy::const_iterator vtable_thunks_begin() const {
    return VTableThunks.begin();
  }

  VTableThunksMapTy::const_iterator vtable_thunks_end() const {
    return VTableThunks.end();
  }

  void dumpLayout(raw_ostream &);
};

} // end namespace

/// InitialOverriddenDefinitionCollector - Finds the set of least derived bases
/// that define the given method.
struct InitialOverriddenDefinitionCollector {
  BasesSetVectorTy Bases;
  OverriddenMethodsSetTy VisitedOverriddenMethods;

  bool visit(const CXXMethodDecl *OverriddenMD) {
    if (OverriddenMD->size_overridden_methods() == 0)
      Bases.insert(OverriddenMD->getParent());
    // Don't recurse on this method if we've already collected it.
    return VisitedOverriddenMethods.insert(OverriddenMD);
  }
};

static bool BaseInSet(const CXXBaseSpecifier *Specifier,
                      CXXBasePath &Path, void *BasesSet) {
  BasesSetVectorTy *Bases = (BasesSetVectorTy *)BasesSet;
  return Bases->count(Specifier->getType()->getAsCXXRecordDecl());
}

CharUnits
VFTableBuilder::ComputeThisOffset(const CXXMethodDecl *MD,
                                  BaseSubobject Base,
                                  FinalOverriders::OverriderInfo Overrider) {
  InitialOverriddenDefinitionCollector Collector;
  visitAllOverriddenMethods(MD, Collector);

  CXXBasePaths Paths;
  Base.getBase()->lookupInBases(BaseInSet, &Collector.Bases, Paths);

  // This will hold the smallest this offset among overridees of MD.
  // This implies that an offset of a non-virtual base will dominate an offset
  // of a virtual base to potentially reduce the number of thunks required
  // in the derived classes that inherit this method.
  CharUnits Ret;
  bool First = true;

  for (CXXBasePaths::paths_iterator I = Paths.begin(), E = Paths.end();
       I != E; ++I) {
    const CXXBasePath &Path = (*I);
    CharUnits ThisOffset = Base.getBaseOffset();
    CharUnits LastVBaseOffset;

    // For each path from the overrider to the parents of the overridden methods,
    // traverse the path, calculating the this offset in the most derived class.
    for (int J = 0, F = Path.size(); J != F; ++J) {
      const CXXBasePathElement &Element = Path[J];
      QualType CurTy = Element.Base->getType();
      const CXXRecordDecl *PrevRD = Element.Class,
                          *CurRD = CurTy->getAsCXXRecordDecl();
      const ASTRecordLayout &Layout = Context.getASTRecordLayout(PrevRD);

      if (Element.Base->isVirtual()) {
        LastVBaseOffset = MostDerivedClassLayout.getVBaseClassOffset(CurRD);
        if (Overrider.Method->getParent() == PrevRD) {
          // This one's interesting. If the final overrider is in a vbase B of the
          // most derived class and it overrides a method of the B's own vbase A,
          // it uses A* as "this". In its prologue, it can cast A* to B* with
          // a static offset. This offset is used regardless of the actual
          // offset of A from B in the most derived class, requiring an
          // this-adjusting thunk in the vftable if A and B are laid out
          // differently in the most derived class.
          ThisOffset += Layout.getVBaseClassOffset(CurRD);
        } else {
          ThisOffset = LastVBaseOffset;
        }
      } else {
        ThisOffset += Layout.getBaseClassOffset(CurRD);
      }
    }

    if (isa<CXXDestructorDecl>(MD)) {
      if (LastVBaseOffset.isZero()) {
        // If a "Base" class has at least one non-virtual base with a virtual
        // destructor, the "Base" virtual destructor will take the address
        // of the "Base" subobject as the "this" argument.
        return Base.getBaseOffset();
      } else {
        // A virtual destructor of a virtual base takes the address of the
        // virtual base subobject as the "this" argument.
        return LastVBaseOffset;
      }
    }

    if (Ret > ThisOffset || First) {
      First = false;
      Ret = ThisOffset;
    }
  }

  assert(!First && "Method not found in the given subobject?");
  return Ret;
}

void VFTableBuilder::CalculateVtordispAdjustment(
    FinalOverriders::OverriderInfo Overrider, CharUnits ThisOffset,
    ThisAdjustment &TA) {
  const ASTRecordLayout::VBaseOffsetsMapTy &VBaseMap =
      MostDerivedClassLayout.getVBaseOffsetsMap();
  const ASTRecordLayout::VBaseOffsetsMapTy::const_iterator &VBaseMapEntry =
      VBaseMap.find(WhichVFPtr.getVBaseWithVPtr());
  assert(VBaseMapEntry != VBaseMap.end());

  // Check if we need a vtordisp adjustment at all.
  if (!VBaseMapEntry->second.hasVtorDisp())
    return;

  CharUnits VFPtrVBaseOffset = VBaseMapEntry->second.VBaseOffset;
  // The implicit vtordisp field is located right before the vbase.
  TA.Virtual.Microsoft.VtordispOffset =
      (VFPtrVBaseOffset - WhichVFPtr.FullOffsetInMDC).getQuantity() - 4;

  // If the final overrider is defined in either:
  // - the most derived class or its non-virtual base or
  // - the same vbase as the initial declaration,
  // a simple vtordisp thunk will suffice.
  const CXXRecordDecl *OverriderRD = Overrider.Method->getParent();
  if (OverriderRD == MostDerivedClass)
    return;

  const CXXRecordDecl *OverriderVBase =
      ComputeBaseOffset(Context, OverriderRD, MostDerivedClass).VirtualBase;
  if (!OverriderVBase || OverriderVBase == WhichVFPtr.getVBaseWithVPtr())
    return;

  // Otherwise, we need to do use the dynamic offset of the final overrider
  // in order to get "this" adjustment right.
  TA.Virtual.Microsoft.VBPtrOffset =
      (VFPtrVBaseOffset + WhichVFPtr.NonVirtualOffset -
       MostDerivedClassLayout.getVBPtrOffset()).getQuantity();
  TA.Virtual.Microsoft.VBOffsetOffset =
      Context.getTypeSizeInChars(Context.IntTy).getQuantity() *
      VTables.getVBTableIndex(MostDerivedClass, OverriderVBase);

  TA.NonVirtual = (ThisOffset - Overrider.Offset).getQuantity();
}

static void GroupNewVirtualOverloads(
    const CXXRecordDecl *RD,
    SmallVector<const CXXMethodDecl *, 10> &VirtualMethods) {
  // Put the virtual methods into VirtualMethods in the proper order:
  // 1) Group overloads by declaration name. New groups are added to the
  //    vftable in the order of their first declarations in this class
  //    (including overrides and non-virtual methods).
  // 2) In each group, new overloads appear in the reverse order of declaration.
  typedef SmallVector<const CXXMethodDecl *, 1> MethodGroup;
  SmallVector<MethodGroup, 10> Groups;
  typedef llvm::DenseMap<DeclarationName, unsigned> VisitedGroupIndicesTy;
  VisitedGroupIndicesTy VisitedGroupIndices;
  for (CXXRecordDecl::method_iterator I = RD->method_begin(),
       E = RD->method_end(); I != E; ++I) {
    const CXXMethodDecl *MD = *I;

    VisitedGroupIndicesTy::iterator J;
    bool Inserted;
    llvm::tie(J, Inserted) = VisitedGroupIndices.insert(
        std::make_pair(MD->getDeclName(), Groups.size()));
    if (Inserted)
      Groups.push_back(MethodGroup());
    if (I->isVirtual())
      Groups[J->second].push_back(MD);
  }

  for (unsigned I = 0, E = Groups.size(); I != E; ++I)
    VirtualMethods.append(Groups[I].rbegin(), Groups[I].rend());
}

/// We need a return adjusting thunk for this method if its return type is
/// not trivially convertible to the return type of any of its overridden
/// methods.
bool VFTableBuilder::NeedsReturnAdjustingThunk(const CXXMethodDecl *MD) {
  OverriddenMethodsSetTy OverriddenMethods;
  ComputeAllOverriddenMethods(MD, OverriddenMethods);
  for (OverriddenMethodsSetTy::iterator I = OverriddenMethods.begin(),
                                        E = OverriddenMethods.end();
       I != E; ++I) {
    const CXXMethodDecl *OverriddenMD = *I;
    BaseOffset Adjustment =
        ComputeReturnAdjustmentBaseOffset(Context, MD, OverriddenMD);
    if (!Adjustment.isEmpty())
      return true;
  }
  return false;
}

void VFTableBuilder::AddMethods(BaseSubobject Base, unsigned BaseDepth,
                                const CXXRecordDecl *LastVBase,
                                BasesSetVectorTy &VisitedBases) {
  const CXXRecordDecl *RD = Base.getBase();
  if (!RD->isPolymorphic())
    return;

  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);

  // See if this class expands a vftable of the base we look at, which is either
  // the one defined by the vfptr base path or the primary base of the current class.
  const CXXRecordDecl *NextBase = 0, *NextLastVBase = LastVBase;
  CharUnits NextBaseOffset;
  if (BaseDepth < WhichVFPtr.PathToBaseWithVPtr.size()) {
    NextBase = WhichVFPtr.PathToBaseWithVPtr[BaseDepth];
    if (Layout.getVBaseOffsetsMap().count(NextBase)) {
      NextLastVBase = NextBase;
      NextBaseOffset = MostDerivedClassLayout.getVBaseClassOffset(NextBase);
    } else {
      NextBaseOffset =
          Base.getBaseOffset() + Layout.getBaseClassOffset(NextBase);
    }
  } else if (const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase()) {
    assert(!Layout.isPrimaryBaseVirtual() &&
           "No primary virtual bases in this ABI");
    NextBase = PrimaryBase;
    NextBaseOffset = Base.getBaseOffset();
  }

  if (NextBase) {
    AddMethods(BaseSubobject(NextBase, NextBaseOffset), BaseDepth + 1,
               NextLastVBase, VisitedBases);
    if (!VisitedBases.insert(NextBase))
      llvm_unreachable("Found a duplicate primary base!");
  }

  SmallVector<const CXXMethodDecl*, 10> VirtualMethods;
  // Put virtual methods in the proper order.
  GroupNewVirtualOverloads(RD, VirtualMethods);

  // Now go through all virtual member functions and add them to the current
  // vftable. This is done by
  //  - replacing overridden methods in their existing slots, as long as they
  //    don't require return adjustment; calculating This adjustment if needed.
  //  - adding new slots for methods of the current base not present in any
  //    sub-bases;
  //  - adding new slots for methods that require Return adjustment.
  // We keep track of the methods visited in the sub-bases in MethodInfoMap.
  for (unsigned I = 0, E = VirtualMethods.size(); I != E; ++I) {
    const CXXMethodDecl *MD = VirtualMethods[I];

    FinalOverriders::OverriderInfo Overrider =
        Overriders.getOverrider(MD, Base.getBaseOffset());
    ThisAdjustment ThisAdjustmentOffset;
    bool ForceThunk = false;

    // Check if this virtual member function overrides
    // a method in one of the visited bases.
    if (const CXXMethodDecl *OverriddenMD =
            FindNearestOverriddenMethod(MD, VisitedBases)) {
      MethodInfoMapTy::iterator OverriddenMDIterator =
          MethodInfoMap.find(OverriddenMD);

      // If the overridden method went to a different vftable, skip it.
      if (OverriddenMDIterator == MethodInfoMap.end())
        continue;

      MethodInfo &OverriddenMethodInfo = OverriddenMDIterator->second;

      // Create a this-adjusting thunk if needed.
      CharUnits TI = ComputeThisOffset(MD, Base, Overrider);
      if (TI != WhichVFPtr.FullOffsetInMDC) {
        ThisAdjustmentOffset.NonVirtual =
            (TI - WhichVFPtr.FullOffsetInMDC).getQuantity();
      }

      if (WhichVFPtr.getVBaseWithVPtr())
        CalculateVtordispAdjustment(Overrider, TI, ThisAdjustmentOffset);

      if (!ThisAdjustmentOffset.isEmpty()) {
        VTableThunks[OverriddenMethodInfo.VFTableIndex].This =
            ThisAdjustmentOffset;
        AddThunk(MD, VTableThunks[OverriddenMethodInfo.VFTableIndex]);
      }

      if (!NeedsReturnAdjustingThunk(MD)) {
        // No return adjustment needed - just replace the overridden method info
        // with the current info.
        MethodInfo MI(OverriddenMethodInfo.VBTableIndex,
                      OverriddenMethodInfo.VFTableIndex);
        MethodInfoMap.erase(OverriddenMDIterator);

        assert(!MethodInfoMap.count(MD) &&
               "Should not have method info for this method yet!");
        MethodInfoMap.insert(std::make_pair(MD, MI));
        continue;
      }

      // In case we need a return adjustment, we'll add a new slot for
      // the overrider and put a return-adjusting thunk where the overridden
      // method was in the vftable.
      // For now, just mark the overriden method as shadowed by a new slot.
      OverriddenMethodInfo.Shadowed = true;
      ForceThunk = true;

      // Also apply this adjustment to the shadowed slots.
      if (!ThisAdjustmentOffset.isEmpty()) {
        // FIXME: this is O(N^2), can be O(N).
        const CXXMethodDecl *SubOverride = OverriddenMD;
        while ((SubOverride =
                    FindNearestOverriddenMethod(SubOverride, VisitedBases))) {
          MethodInfoMapTy::iterator SubOverrideIterator =
              MethodInfoMap.find(SubOverride);
          if (SubOverrideIterator == MethodInfoMap.end())
            break;
          MethodInfo &SubOverrideMI = SubOverrideIterator->second;
          assert(SubOverrideMI.Shadowed);
          VTableThunks[SubOverrideMI.VFTableIndex].This =
              ThisAdjustmentOffset;
          AddThunk(MD, VTableThunks[SubOverrideMI.VFTableIndex]);
        }
      }
    } else if (Base.getBaseOffset() != WhichVFPtr.FullOffsetInMDC ||
               MD->size_overridden_methods()) {
      // Skip methods that don't belong to the vftable of the current class,
      // e.g. each method that wasn't seen in any of the visited sub-bases
      // but overrides multiple methods of other sub-bases.
      continue;
    }

    // If we got here, MD is a method not seen in any of the sub-bases or
    // it requires return adjustment. Insert the method info for this method.
    unsigned VBIndex =
        LastVBase ? VTables.getVBTableIndex(MostDerivedClass, LastVBase) : 0;
    MethodInfo MI(VBIndex, Components.size());

    assert(!MethodInfoMap.count(MD) &&
           "Should not have method info for this method yet!");
    MethodInfoMap.insert(std::make_pair(MD, MI));

    const CXXMethodDecl *OverriderMD = Overrider.Method;

    // Check if this overrider needs a return adjustment.
    // We don't want to do this for pure virtual member functions.
    BaseOffset ReturnAdjustmentOffset;
    ReturnAdjustment ReturnAdjustment;
    if (!OverriderMD->isPure()) {
      ReturnAdjustmentOffset =
          ComputeReturnAdjustmentBaseOffset(Context, OverriderMD, MD);
    }
    if (!ReturnAdjustmentOffset.isEmpty()) {
      ForceThunk = true;
      ReturnAdjustment.NonVirtual =
          ReturnAdjustmentOffset.NonVirtualOffset.getQuantity();
      if (ReturnAdjustmentOffset.VirtualBase) {
        const ASTRecordLayout &DerivedLayout =
            Context.getASTRecordLayout(ReturnAdjustmentOffset.DerivedClass);
        ReturnAdjustment.Virtual.Microsoft.VBPtrOffset =
            DerivedLayout.getVBPtrOffset().getQuantity();
        ReturnAdjustment.Virtual.Microsoft.VBIndex =
            VTables.getVBTableIndex(ReturnAdjustmentOffset.DerivedClass,
                                    ReturnAdjustmentOffset.VirtualBase);
      }
    }

    AddMethod(OverriderMD, ThunkInfo(ThisAdjustmentOffset, ReturnAdjustment,
                                     ForceThunk ? MD : 0));
  }
}

static void PrintBasePath(const VPtrInfo::BasePath &Path, raw_ostream &Out) {
  for (VPtrInfo::BasePath::const_reverse_iterator I = Path.rbegin(),
       E = Path.rend(); I != E; ++I) {
    Out << "'";
    (*I)->printQualifiedName(Out);
    Out << "' in ";
  }
}

namespace {
struct MicrosoftThunkInfoStableSortComparator {
  bool operator() (const ThunkInfo &LHS, const ThunkInfo &RHS) {
    if (LHS.This != RHS.This)
      return LHS.This < RHS.This;

    if (LHS.Return != RHS.Return)
      return LHS.Return < RHS.Return;

    // Keep different thunks with the same adjustments in the order they
    // were put into the vector.
    return false;
  }
};
}

static void dumpMicrosoftThunkAdjustment(const ThunkInfo &TI, raw_ostream &Out,
                                         bool ContinueFirstLine) {
  const ReturnAdjustment &R = TI.Return;
  bool Multiline = false;
  const char *LinePrefix = "\n        ";
  if (!R.isEmpty()) {
    if (!ContinueFirstLine)
      Out << LinePrefix;
    Out << "[return adjustment: ";
    if (R.Virtual.Microsoft.VBPtrOffset)
      Out << "vbptr at offset " << R.Virtual.Microsoft.VBPtrOffset << ", ";
    if (R.Virtual.Microsoft.VBIndex)
      Out << "vbase #" << R.Virtual.Microsoft.VBIndex << ", ";
    Out << R.NonVirtual << " non-virtual]";
    Multiline = true;
  }

  const ThisAdjustment &T = TI.This;
  if (!T.isEmpty()) {
    if (Multiline || !ContinueFirstLine)
      Out << LinePrefix;
    Out << "[this adjustment: ";
    if (!TI.This.Virtual.isEmpty()) {
      assert(T.Virtual.Microsoft.VtordispOffset < 0);
      Out << "vtordisp at " << T.Virtual.Microsoft.VtordispOffset << ", ";
      if (T.Virtual.Microsoft.VBPtrOffset) {
        Out << "vbptr at " << T.Virtual.Microsoft.VBPtrOffset
            << " to the left, ";
        assert(T.Virtual.Microsoft.VBOffsetOffset > 0);
        Out << LinePrefix << " vboffset at "
            << T.Virtual.Microsoft.VBOffsetOffset << " in the vbtable, ";
      }
    }
    Out << T.NonVirtual << " non-virtual]";
  }
}

void VFTableBuilder::dumpLayout(raw_ostream &Out) {
  Out << "VFTable for ";
  PrintBasePath(WhichVFPtr.PathToBaseWithVPtr, Out);
  Out << "'";
  MostDerivedClass->printQualifiedName(Out);
  Out << "' (" << Components.size() << " entries).\n";

  for (unsigned I = 0, E = Components.size(); I != E; ++I) {
    Out << llvm::format("%4d | ", I);

    const VTableComponent &Component = Components[I];

    // Dump the component.
    switch (Component.getKind()) {
    case VTableComponent::CK_RTTI:
      Component.getRTTIDecl()->printQualifiedName(Out);
      Out << " RTTI";
      break;

    case VTableComponent::CK_FunctionPointer: {
      const CXXMethodDecl *MD = Component.getFunctionDecl();

      // FIXME: Figure out how to print the real thunk type, since they can
      // differ in the return type.
      std::string Str = PredefinedExpr::ComputeName(
          PredefinedExpr::PrettyFunctionNoVirtual, MD);
      Out << Str;
      if (MD->isPure())
        Out << " [pure]";

      if (MD->isDeleted()) {
        ErrorUnsupported("deleted methods", MD->getLocation());
        Out << " [deleted]";
      }

      ThunkInfo Thunk = VTableThunks.lookup(I);
      if (!Thunk.isEmpty())
        dumpMicrosoftThunkAdjustment(Thunk, Out, /*ContinueFirstLine=*/false);

      break;
    }

    case VTableComponent::CK_DeletingDtorPointer: {
      const CXXDestructorDecl *DD = Component.getDestructorDecl();

      DD->printQualifiedName(Out);
      Out << "() [scalar deleting]";

      if (DD->isPure())
        Out << " [pure]";

      ThunkInfo Thunk = VTableThunks.lookup(I);
      if (!Thunk.isEmpty()) {
        assert(Thunk.Return.isEmpty() &&
               "No return adjustment needed for destructors!");
        dumpMicrosoftThunkAdjustment(Thunk, Out, /*ContinueFirstLine=*/false);
      }

      break;
    }

    default:
      DiagnosticsEngine &Diags = Context.getDiagnostics();
      unsigned DiagID = Diags.getCustomDiagID(
          DiagnosticsEngine::Error,
          "Unexpected vftable component type %0 for component number %1");
      Diags.Report(MostDerivedClass->getLocation(), DiagID)
          << I << Component.getKind();
    }

    Out << '\n';
  }

  Out << '\n';

  if (!Thunks.empty()) {
    // We store the method names in a map to get a stable order.
    std::map<std::string, const CXXMethodDecl *> MethodNamesAndDecls;

    for (ThunksMapTy::const_iterator I = Thunks.begin(), E = Thunks.end();
         I != E; ++I) {
      const CXXMethodDecl *MD = I->first;
      std::string MethodName = PredefinedExpr::ComputeName(
          PredefinedExpr::PrettyFunctionNoVirtual, MD);

      MethodNamesAndDecls.insert(std::make_pair(MethodName, MD));
    }

    for (std::map<std::string, const CXXMethodDecl *>::const_iterator
             I = MethodNamesAndDecls.begin(),
             E = MethodNamesAndDecls.end();
         I != E; ++I) {
      const std::string &MethodName = I->first;
      const CXXMethodDecl *MD = I->second;

      ThunkInfoVectorTy ThunksVector = Thunks[MD];
      std::stable_sort(ThunksVector.begin(), ThunksVector.end(),
                       MicrosoftThunkInfoStableSortComparator());

      Out << "Thunks for '" << MethodName << "' (" << ThunksVector.size();
      Out << (ThunksVector.size() == 1 ? " entry" : " entries") << ").\n";

      for (unsigned I = 0, E = ThunksVector.size(); I != E; ++I) {
        const ThunkInfo &Thunk = ThunksVector[I];

        Out << llvm::format("%4d | ", I);
        dumpMicrosoftThunkAdjustment(Thunk, Out, /*ContinueFirstLine=*/true);
        Out << '\n';
      }

      Out << '\n';
    }
  }
}

static bool setsIntersect(const llvm::SmallPtrSet<const CXXRecordDecl *, 4> &A,
                          const llvm::ArrayRef<const CXXRecordDecl *> &B) {
  for (llvm::ArrayRef<const CXXRecordDecl *>::iterator I = B.begin(),
                                                       E = B.end();
       I != E; ++I) {
    if (A.count(*I))
      return true;
  }
  return false;
}

static bool rebucketPaths(VPtrInfoVector &Paths);

/// Produces MSVC-compatible vbtable data.  The symbols produced by this
/// algorithm match those produced by MSVC 2012 and newer, which is different
/// from MSVC 2010.
///
/// MSVC 2012 appears to minimize the vbtable names using the following
/// algorithm.  First, walk the class hierarchy in the usual order, depth first,
/// left to right, to find all of the subobjects which contain a vbptr field.
/// Visiting each class node yields a list of inheritance paths to vbptrs.  Each
/// record with a vbptr creates an initially empty path.
///
/// To combine paths from child nodes, the paths are compared to check for
/// ambiguity.  Paths are "ambiguous" if multiple paths have the same set of
/// components in the same order.  Each group of ambiguous paths is extended by
/// appending the class of the base from which it came.  If the current class
/// node produced an ambiguous path, its path is extended with the current class.
/// After extending paths, MSVC again checks for ambiguity, and extends any
/// ambiguous path which wasn't already extended.  Because each node yields an
/// unambiguous set of paths, MSVC doesn't need to extend any path more than once
/// to produce an unambiguous set of paths.
///
/// TODO: Presumably vftables use the same algorithm.
void MicrosoftVTableContext::computeVTablePaths(bool ForVBTables,
                                                const CXXRecordDecl *RD,
                                                VPtrInfoVector &Paths) {
  assert(Paths.empty());
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);

  // Base case: this subobject has its own vptr.
  if (ForVBTables ? Layout.hasOwnVBPtr() : Layout.hasOwnVFPtr())
    Paths.push_back(new VPtrInfo(RD));

  // Recursive case: get all the vbtables from our bases and remove anything
  // that shares a virtual base.
  llvm::SmallPtrSet<const CXXRecordDecl*, 4> VBasesSeen;
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
                                                E = RD->bases_end();
       I != E; ++I) {
    const CXXRecordDecl *Base = I->getType()->getAsCXXRecordDecl();
    if (I->isVirtual() && VBasesSeen.count(Base))
      continue;

    if (!Base->isDynamicClass())
      continue;

    const VPtrInfoVector &BasePaths =
        ForVBTables ? enumerateVBTables(Base) : getVFPtrOffsets(Base);

    for (VPtrInfoVector::const_iterator II = BasePaths.begin(),
                                        EE = BasePaths.end();
         II != EE; ++II) {
      VPtrInfo *BaseInfo = *II;

      // Don't include the path if it goes through a virtual base that we've
      // already included.
      if (setsIntersect(VBasesSeen, BaseInfo->ContainingVBases))
        continue;

      // Copy the path and adjust it as necessary.
      VPtrInfo *P = new VPtrInfo(*BaseInfo);

      // We mangle Base into the path if the path would've been ambiguous and it
      // wasn't already extended with Base.
      if (P->MangledPath.empty() || P->MangledPath.back() != Base)
        P->NextBaseToMangle = Base;

      // Keep track of the full path.
      // FIXME: Why do we need this?
      P->PathToBaseWithVPtr.insert(P->PathToBaseWithVPtr.begin(), Base);

      // Keep track of which derived class ultimately uses the vtable, and what
      // the full adjustment is from the MDC to this vtable.  The adjustment is
      // captured by an optional vbase and a non-virtual offset.
      if (Base == (ForVBTables ? Layout.getBaseSharingVBPtr()
                               : Layout.getPrimaryBase()))
        P->ReusingBase = RD;
      if (I->isVirtual())
        P->ContainingVBases.push_back(Base);
      else if (P->ContainingVBases.empty())
        P->NonVirtualOffset += Layout.getBaseClassOffset(Base);

      // Update the full offset in the MDC.
      P->FullOffsetInMDC = P->NonVirtualOffset;
      if (const CXXRecordDecl *VB = P->getVBaseWithVPtr())
        P->FullOffsetInMDC += Layout.getVBaseClassOffset(VB);

      Paths.push_back(P);
    }

    // After visiting any direct base, we've transitively visited all of its
    // morally virtual bases.
    for (CXXRecordDecl::base_class_const_iterator II = Base->vbases_begin(),
                                                  EE = Base->vbases_end();
         II != EE; ++II)
      VBasesSeen.insert(II->getType()->getAsCXXRecordDecl());
  }

  // Sort the paths into buckets, and if any of them are ambiguous, extend all
  // paths in ambiguous buckets.
  bool Changed = true;
  while (Changed)
    Changed = rebucketPaths(Paths);
}

static bool pathCompare(const VPtrInfo *LHS, const VPtrInfo *RHS) {
  return LHS->MangledPath < RHS->MangledPath;
}

static bool extendPath(VPtrInfo *P) {
  if (P->NextBaseToMangle) {
    P->MangledPath.push_back(P->NextBaseToMangle);
    P->NextBaseToMangle = 0;  // Prevent the path from being extended twice.
    return true;
  }
  return false;
}

static bool rebucketPaths(VPtrInfoVector &Paths) {
  // What we're essentially doing here is bucketing together ambiguous paths.
  // Any bucket with more than one path in it gets extended by NextBase, which
  // is usually the direct base of the inherited the vbptr.  This code uses a
  // sorted vector to implement a multiset to form the buckets.  Note that the
  // ordering is based on pointers, but it doesn't change our output order.  The
  // current algorithm is designed to match MSVC 2012's names.
  VPtrInfoVector PathsSorted(Paths);
  std::sort(PathsSorted.begin(), PathsSorted.end(), pathCompare);
  bool Changed = false;
  for (size_t I = 0, E = PathsSorted.size(); I != E;) {
    // Scan forward to find the end of the bucket.
    size_t BucketStart = I;
    do {
      ++I;
    } while (I != E && PathsSorted[BucketStart]->MangledPath ==
                           PathsSorted[I]->MangledPath);

    // If this bucket has multiple paths, extend them all.
    if (I - BucketStart > 1) {
      for (size_t II = BucketStart; II != I; ++II)
        Changed |= extendPath(PathsSorted[II]);
      assert(Changed && "no paths were extended to fix ambiguity");
    }
  }
  return Changed;
}

MicrosoftVTableContext::~MicrosoftVTableContext() {
  llvm::DeleteContainerSeconds(VFTableLayouts);
  llvm::DeleteContainerSeconds(VBaseInfo);
}

void MicrosoftVTableContext::computeVTableRelatedInformation(
    const CXXRecordDecl *RD) {
  assert(RD->isDynamicClass());

  // Check if we've computed this information before.
  if (VFPtrLocations.count(RD))
    return;

  const VTableLayout::AddressPointsMapTy EmptyAddressPointsMap;

  VPtrInfoVector *VFPtrs = new VPtrInfoVector();
  computeVTablePaths(/*ForVBTables=*/false, RD, *VFPtrs);
  VFPtrLocations[RD] = VFPtrs;

  MethodVFTableLocationsTy NewMethodLocations;
  for (VPtrInfoVector::iterator I = VFPtrs->begin(), E = VFPtrs->end();
       I != E; ++I) {
    VFTableBuilder Builder(*this, RD, *I);

    VFTableIdTy id(RD, (*I)->FullOffsetInMDC);
    assert(VFTableLayouts.count(id) == 0);
    SmallVector<VTableLayout::VTableThunkTy, 1> VTableThunks(
        Builder.vtable_thunks_begin(), Builder.vtable_thunks_end());
    VFTableLayouts[id] = new VTableLayout(
        Builder.getNumVTableComponents(), Builder.vtable_component_begin(),
        VTableThunks.size(), VTableThunks.data(), EmptyAddressPointsMap, true);
    NewMethodLocations.insert(Builder.vtable_indices_begin(),
                              Builder.vtable_indices_end());
    Thunks.insert(Builder.thunks_begin(), Builder.thunks_end());
  }

  MethodVFTableLocations.insert(NewMethodLocations.begin(),
                                NewMethodLocations.end());
  if (Context.getLangOpts().DumpVTableLayouts)
    dumpMethodLocations(RD, NewMethodLocations, llvm::outs());
}

void MicrosoftVTableContext::dumpMethodLocations(
    const CXXRecordDecl *RD, const MethodVFTableLocationsTy &NewMethods,
    raw_ostream &Out) {
  // Compute the vtable indices for all the member functions.
  // Store them in a map keyed by the location so we'll get a sorted table.
  std::map<MethodVFTableLocation, std::string> IndicesMap;
  bool HasNonzeroOffset = false;

  for (MethodVFTableLocationsTy::const_iterator I = NewMethods.begin(),
       E = NewMethods.end(); I != E; ++I) {
    const CXXMethodDecl *MD = cast<const CXXMethodDecl>(I->first.getDecl());
    assert(MD->isVirtual());

    std::string MethodName = PredefinedExpr::ComputeName(
        PredefinedExpr::PrettyFunctionNoVirtual, MD);

    if (isa<CXXDestructorDecl>(MD)) {
      IndicesMap[I->second] = MethodName + " [scalar deleting]";
    } else {
      IndicesMap[I->second] = MethodName;
    }

    if (!I->second.VFPtrOffset.isZero() || I->second.VBTableIndex != 0)
      HasNonzeroOffset = true;
  }

  // Print the vtable indices for all the member functions.
  if (!IndicesMap.empty()) {
    Out << "VFTable indices for ";
    Out << "'";
    RD->printQualifiedName(Out);
    Out << "' (" << IndicesMap.size() << " entries).\n";

    CharUnits LastVFPtrOffset = CharUnits::fromQuantity(-1);
    uint64_t LastVBIndex = 0;
    for (std::map<MethodVFTableLocation, std::string>::const_iterator
             I = IndicesMap.begin(),
             E = IndicesMap.end();
         I != E; ++I) {
      CharUnits VFPtrOffset = I->first.VFPtrOffset;
      uint64_t VBIndex = I->first.VBTableIndex;
      if (HasNonzeroOffset &&
          (VFPtrOffset != LastVFPtrOffset || VBIndex != LastVBIndex)) {
        assert(VBIndex > LastVBIndex || VFPtrOffset > LastVFPtrOffset);
        Out << " -- accessible via ";
        if (VBIndex)
          Out << "vbtable index " << VBIndex << ", ";
        Out << "vfptr at offset " << VFPtrOffset.getQuantity() << " --\n";
        LastVFPtrOffset = VFPtrOffset;
        LastVBIndex = VBIndex;
      }

      uint64_t VTableIndex = I->first.Index;
      const std::string &MethodName = I->second;
      Out << llvm::format("%4" PRIu64 " | ", VTableIndex) << MethodName << '\n';
    }
    Out << '\n';
  }
}

const VirtualBaseInfo *MicrosoftVTableContext::computeVBTableRelatedInformation(
    const CXXRecordDecl *RD) {
  VirtualBaseInfo *VBI;

  {
    // Get or create a VBI for RD.  Don't hold a reference to the DenseMap cell,
    // as it may be modified and rehashed under us.
    VirtualBaseInfo *&Entry = VBaseInfo[RD];
    if (Entry)
      return Entry;
    Entry = VBI = new VirtualBaseInfo();
  }

  computeVTablePaths(/*ForVBTables=*/true, RD, VBI->VBPtrPaths);

  // First, see if the Derived class shared the vbptr with a non-virtual base.
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
  if (const CXXRecordDecl *VBPtrBase = Layout.getBaseSharingVBPtr()) {
    // If the Derived class shares the vbptr with a non-virtual base, the shared
    // virtual bases come first so that the layout is the same.
    const VirtualBaseInfo *BaseInfo =
        computeVBTableRelatedInformation(VBPtrBase);
    VBI->VBTableIndices.insert(BaseInfo->VBTableIndices.begin(),
                               BaseInfo->VBTableIndices.end());
  }

  // New vbases are added to the end of the vbtable.
  // Skip the self entry and vbases visited in the non-virtual base, if any.
  unsigned VBTableIndex = 1 + VBI->VBTableIndices.size();
  for (CXXRecordDecl::base_class_const_iterator I = RD->vbases_begin(),
                                                E = RD->vbases_end();
       I != E; ++I) {
    const CXXRecordDecl *CurVBase = I->getType()->getAsCXXRecordDecl();
    if (!VBI->VBTableIndices.count(CurVBase))
      VBI->VBTableIndices[CurVBase] = VBTableIndex++;
  }

  return VBI;
}

unsigned MicrosoftVTableContext::getVBTableIndex(const CXXRecordDecl *Derived,
                                                 const CXXRecordDecl *VBase) {
  const VirtualBaseInfo *VBInfo = computeVBTableRelatedInformation(Derived);
  assert(VBInfo->VBTableIndices.count(VBase));
  return VBInfo->VBTableIndices.find(VBase)->second;
}

const VPtrInfoVector &
MicrosoftVTableContext::enumerateVBTables(const CXXRecordDecl *RD) {
  return computeVBTableRelatedInformation(RD)->VBPtrPaths;
}

const VPtrInfoVector &
MicrosoftVTableContext::getVFPtrOffsets(const CXXRecordDecl *RD) {
  computeVTableRelatedInformation(RD);

  assert(VFPtrLocations.count(RD) && "Couldn't find vfptr locations");
  return *VFPtrLocations[RD];
}

const VTableLayout &
MicrosoftVTableContext::getVFTableLayout(const CXXRecordDecl *RD,
                                         CharUnits VFPtrOffset) {
  computeVTableRelatedInformation(RD);

  VFTableIdTy id(RD, VFPtrOffset);
  assert(VFTableLayouts.count(id) && "Couldn't find a VFTable at this offset");
  return *VFTableLayouts[id];
}

const MicrosoftVTableContext::MethodVFTableLocation &
MicrosoftVTableContext::getMethodVFTableLocation(GlobalDecl GD) {
  assert(cast<CXXMethodDecl>(GD.getDecl())->isVirtual() &&
         "Only use this method for virtual methods or dtors");
  if (isa<CXXDestructorDecl>(GD.getDecl()))
    assert(GD.getDtorType() == Dtor_Deleting);

  MethodVFTableLocationsTy::iterator I = MethodVFTableLocations.find(GD);
  if (I != MethodVFTableLocations.end())
    return I->second;

  const CXXRecordDecl *RD = cast<CXXMethodDecl>(GD.getDecl())->getParent();

  computeVTableRelatedInformation(RD);

  I = MethodVFTableLocations.find(GD);
  assert(I != MethodVFTableLocations.end() && "Did not find index!");
  return I->second;
}
