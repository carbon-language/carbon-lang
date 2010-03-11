//===--- CGVtable.cpp - Emit LLVM Code for C++ vtables --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation of virtual tables.
//
//===----------------------------------------------------------------------===//

#include "CodeGenModule.h"
#include "CodeGenFunction.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/RecordLayout.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Format.h"
#include <cstdio>

using namespace clang;
using namespace CodeGen;

namespace {

/// BaseOffset - Represents an offset from a derived class to a direct or
/// indirect base class.
struct BaseOffset {
  /// DerivedClass - The derived class.
  const CXXRecordDecl *DerivedClass;
  
  /// VirtualBase - If the path from the derived class to the base class
  /// involves a virtual base class, this holds its declaration.
  const CXXRecordDecl *VirtualBase;

  /// NonVirtualOffset - The offset from the derived class to the base class.
  /// (Or the offset from the virtual base class to the base class, if the 
  /// path from the derived class to the base class involves a virtual base
  /// class.
  int64_t NonVirtualOffset;
  
  BaseOffset() : DerivedClass(0), VirtualBase(0), NonVirtualOffset(0) { }
  BaseOffset(const CXXRecordDecl *DerivedClass,
             const CXXRecordDecl *VirtualBase, int64_t NonVirtualOffset)
    : DerivedClass(DerivedClass), VirtualBase(VirtualBase), 
    NonVirtualOffset(NonVirtualOffset) { }

  bool isEmpty() const { return !NonVirtualOffset && !VirtualBase; }
};

/// FinalOverriders - Contains the final overrider member functions for all
/// member functions in the base subobjects of a class.
class FinalOverriders {
public:
  /// OverriderInfo - Information about a final overrider.
  struct OverriderInfo {
    /// Method - The method decl of the overrider.
    const CXXMethodDecl *Method;

    /// Offset - the base offset of the overrider in the layout class.
    uint64_t Offset;
    
    /// OldOffset - FIXME: Remove this.
    int64_t OldOffset;
    
    OverriderInfo() : Method(0), Offset(0), OldOffset(0) { }
  };

private:
  /// MostDerivedClass - The most derived class for which the final overriders
  /// are stored.
  const CXXRecordDecl *MostDerivedClass;
  
  /// MostDerivedClassOffset - If we're building final overriders for a 
  /// construction vtable, this holds the offset from the layout class to the
  /// most derived class.
  const uint64_t MostDerivedClassOffset;

  /// LayoutClass - The class we're using for layout information. Will be 
  /// different than the most derived class if the final overriders are for a
  /// construction vtable.  
  const CXXRecordDecl *LayoutClass;  

  ASTContext &Context;
  
  /// MostDerivedClassLayout - the AST record layout of the most derived class.
  const ASTRecordLayout &MostDerivedClassLayout;

  /// BaseSubobjectMethodPairTy - Uniquely identifies a member function
  /// in a base subobject.
  typedef std::pair<BaseSubobject, const CXXMethodDecl *>
    BaseSubobjectMethodPairTy;
  
  typedef llvm::DenseMap<BaseSubobjectMethodPairTy,
                         OverriderInfo> OverridersMapTy;
  
  /// OverridersMap - The final overriders for all virtual member functions of 
  /// all the base subobjects of the most derived class.
  OverridersMapTy OverridersMap;
  
  /// VisitedVirtualBases - A set of all the visited virtual bases, used to
  /// avoid visiting virtual bases more than once.
  llvm::SmallPtrSet<const CXXRecordDecl *, 4> VisitedVirtualBases;

  typedef llvm::DenseMap<BaseSubobjectMethodPairTy, BaseOffset>
    AdjustmentOffsetsMapTy;

  /// ReturnAdjustments - Holds return adjustments for all the overriders that 
  /// need to perform return value adjustments.
  AdjustmentOffsetsMapTy ReturnAdjustments;

  // FIXME: We might be able to get away with making this a SmallSet.
  typedef llvm::SmallSetVector<uint64_t, 2> OffsetSetVectorTy;
  
  /// SubobjectOffsetsMapTy - This map is used for keeping track of all the
  /// base subobject offsets that a single class declaration might refer to.
  ///
  /// For example, in:
  ///
  /// struct A { virtual void f(); };
  /// struct B1 : A { };
  /// struct B2 : A { };
  /// struct C : B1, B2 { virtual void f(); };
  ///
  /// when we determine that C::f() overrides A::f(), we need to update the
  /// overriders map for both A-in-B1 and A-in-B2 and the subobject offsets map
  /// will have the subobject offsets for both A copies.
  typedef llvm::DenseMap<const CXXRecordDecl *, OffsetSetVectorTy>
    SubobjectOffsetsMapTy;
  
  /// ComputeFinalOverriders - Compute the final overriders for a given base
  /// subobject (and all its direct and indirect bases).
  void ComputeFinalOverriders(BaseSubobject Base,
                              bool BaseSubobjectIsVisitedVBase,
                              uint64_t OffsetInLayoutClass,
                              SubobjectOffsetsMapTy &Offsets);
  
  /// AddOverriders - Add the final overriders for this base subobject to the
  /// map of final overriders.  
  void AddOverriders(BaseSubobject Base,uint64_t OffsetInLayoutClass,
                     SubobjectOffsetsMapTy &Offsets);

  /// PropagateOverrider - Propagate the NewMD overrider to all the functions 
  /// that OldMD overrides. For example, if we have:
  ///
  /// struct A { virtual void f(); };
  /// struct B : A { virtual void f(); };
  /// struct C : B { virtual void f(); };
  ///
  /// and we want to override B::f with C::f, we also need to override A::f with
  /// C::f.
  void PropagateOverrider(const CXXMethodDecl *OldMD,
                          BaseSubobject NewBase,
                          uint64_t OverriderOffsetInLayoutClass,
                          const CXXMethodDecl *NewMD,
                          SubobjectOffsetsMapTy &Offsets);
  
  static void MergeSubobjectOffsets(const SubobjectOffsetsMapTy &NewOffsets,
                                    SubobjectOffsetsMapTy &Offsets);

public:
  FinalOverriders(const CXXRecordDecl *MostDerivedClass,
                  uint64_t MostDerivedClassOffset,
                  const CXXRecordDecl *LayoutClass);

  /// getOverrider - Get the final overrider for the given method declaration in
  /// the given base subobject.
  OverriderInfo getOverrider(BaseSubobject Base,
                             const CXXMethodDecl *MD) const {
    assert(OverridersMap.count(std::make_pair(Base, MD)) && 
           "Did not find overrider!");
    
    return OverridersMap.lookup(std::make_pair(Base, MD));
  }
  
  /// getReturnAdjustmentOffset - Get the return adjustment offset for the
  /// method decl in the given base subobject. Returns an empty base offset if
  /// no adjustment is needed.
  BaseOffset getReturnAdjustmentOffset(BaseSubobject Base,
                                       const CXXMethodDecl *MD) const {
    return ReturnAdjustments.lookup(std::make_pair(Base, MD));
  }

  /// dump - dump the final overriders.
  void dump() {
    assert(VisitedVirtualBases.empty() &&
           "Visited virtual bases aren't empty!");
    dump(llvm::errs(), BaseSubobject(MostDerivedClass, 0)); 
    VisitedVirtualBases.clear();
  }
  
  /// dump - dump the final overriders for a base subobject, and all its direct
  /// and indirect base subobjects.
  void dump(llvm::raw_ostream &Out, BaseSubobject Base);
};

#define DUMP_OVERRIDERS 0

FinalOverriders::FinalOverriders(const CXXRecordDecl *MostDerivedClass,
                                 uint64_t MostDerivedClassOffset,
                                 const CXXRecordDecl *LayoutClass)
  : MostDerivedClass(MostDerivedClass), 
  MostDerivedClassOffset(MostDerivedClassOffset), LayoutClass(LayoutClass),
  Context(MostDerivedClass->getASTContext()),
  MostDerivedClassLayout(Context.getASTRecordLayout(MostDerivedClass)) {
    
  // Compute the final overriders.
  SubobjectOffsetsMapTy Offsets;
  ComputeFinalOverriders(BaseSubobject(MostDerivedClass, 0), 
                         /*BaseSubobjectIsVisitedVBase=*/false, 
                         MostDerivedClassOffset, Offsets);
  VisitedVirtualBases.clear();

#if DUMP_OVERRIDERS
  // And dump them (for now).
  dump();
    
  // Also dump the base offsets (for now).
  for (SubobjectOffsetsMapTy::const_iterator I = Offsets.begin(),
       E = Offsets.end(); I != E; ++I) {
    const OffsetSetVectorTy& OffsetSetVector = I->second;

    llvm::errs() << "Base offsets for ";
    llvm::errs() << I->first->getQualifiedNameAsString() << '\n';

    for (unsigned I = 0, E = OffsetSetVector.size(); I != E; ++I)
      llvm::errs() << "  " << I << " - " << OffsetSetVector[I] / 8 << '\n';
  }
#endif
}

void FinalOverriders::AddOverriders(BaseSubobject Base,
                                    uint64_t OffsetInLayoutClass,
                                    SubobjectOffsetsMapTy &Offsets) {
  const CXXRecordDecl *RD = Base.getBase();

  for (CXXRecordDecl::method_iterator I = RD->method_begin(), 
       E = RD->method_end(); I != E; ++I) {
    const CXXMethodDecl *MD = *I;
    
    if (!MD->isVirtual())
      continue;

    // First, propagate the overrider.
    PropagateOverrider(MD, Base, OffsetInLayoutClass, MD, Offsets);

    // Add the overrider as the final overrider of itself.
    OverriderInfo& Overrider = OverridersMap[std::make_pair(Base, MD)];
    assert(!Overrider.Method && "Overrider should not exist yet!");

    Overrider.OldOffset = Base.getBaseOffset();
    Overrider.Offset = OffsetInLayoutClass;
    Overrider.Method = MD;
  }
}

static BaseOffset ComputeBaseOffset(ASTContext &Context, 
                                    const CXXRecordDecl *DerivedRD,
                                    const CXXBasePath &Path) {
  int64_t NonVirtualOffset = 0;

  unsigned NonVirtualStart = 0;
  const CXXRecordDecl *VirtualBase = 0;
  
  // First, look for the virtual base class.
  for (unsigned I = 0, E = Path.size(); I != E; ++I) {
    const CXXBasePathElement &Element = Path[I];
    
    if (Element.Base->isVirtual()) {
      // FIXME: Can we break when we find the first virtual base?
      // (If we can't, can't we just iterate over the path in reverse order?)
      NonVirtualStart = I + 1;
      QualType VBaseType = Element.Base->getType();
      VirtualBase = 
        cast<CXXRecordDecl>(VBaseType->getAs<RecordType>()->getDecl());
    }
  }
  
  // Now compute the non-virtual offset.
  for (unsigned I = NonVirtualStart, E = Path.size(); I != E; ++I) {
    const CXXBasePathElement &Element = Path[I];
    
    // Check the base class offset.
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(Element.Class);

    const RecordType *BaseType = Element.Base->getType()->getAs<RecordType>();
    const CXXRecordDecl *Base = cast<CXXRecordDecl>(BaseType->getDecl());

    NonVirtualOffset += Layout.getBaseClassOffset(Base);
  }
  
  // FIXME: This should probably use CharUnits or something. Maybe we should
  // even change the base offsets in ASTRecordLayout to be specified in 
  // CharUnits.
  return BaseOffset(DerivedRD, VirtualBase, NonVirtualOffset / 8);
  
}

static BaseOffset ComputeBaseOffset(ASTContext &Context, 
                                    const CXXRecordDecl *BaseRD,
                                    const CXXRecordDecl *DerivedRD) {
  CXXBasePaths Paths(/*FindAmbiguities=*/false,
                     /*RecordPaths=*/true, /*DetectVirtual=*/false);
  
  if (!const_cast<CXXRecordDecl *>(DerivedRD)->
      isDerivedFrom(const_cast<CXXRecordDecl *>(BaseRD), Paths)) {
    assert(false && "Class must be derived from the passed in base class!");
    return BaseOffset();
  }

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
    Context.getCanonicalType(DerivedFT->getResultType());
  CanQualType CanBaseReturnType = 
    Context.getCanonicalType(BaseFT->getResultType());
  
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
    assert(false && "Unexpected return type!");
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

void FinalOverriders::PropagateOverrider(const CXXMethodDecl *OldMD,
                                         BaseSubobject NewBase,
                                         uint64_t OverriderOffsetInLayoutClass,
                                         const CXXMethodDecl *NewMD,
                                         SubobjectOffsetsMapTy &Offsets) {
  for (CXXMethodDecl::method_iterator I = OldMD->begin_overridden_methods(),
       E = OldMD->end_overridden_methods(); I != E; ++I) {
    const CXXMethodDecl *OverriddenMD = *I;
    const CXXRecordDecl *OverriddenRD = OverriddenMD->getParent();

    // We want to override OverriddenMD in all subobjects, for example:
    //
    /// struct A { virtual void f(); };
    /// struct B1 : A { };
    /// struct B2 : A { };
    /// struct C : B1, B2 { virtual void f(); };
    ///
    /// When overriding A::f with C::f we need to do so in both A subobjects.
    const OffsetSetVectorTy &OffsetVector = Offsets[OverriddenRD];
    
    // Go through all the subobjects.
    for (unsigned I = 0, E = OffsetVector.size(); I != E; ++I) {
      uint64_t Offset = OffsetVector[I];

      BaseSubobject OverriddenSubobject = BaseSubobject(OverriddenRD, Offset);
      BaseSubobjectMethodPairTy SubobjectAndMethod =
        std::make_pair(OverriddenSubobject, OverriddenMD);
      
      OverriderInfo &Overrider = OverridersMap[SubobjectAndMethod];

      assert(Overrider.Method && "Did not find existing overrider!");

      // Check if we need return adjustments or base adjustments.
      // (We don't want to do this for pure virtual member functions).
      if (!NewMD->isPure()) {
        // Get the return adjustment base offset.
        BaseOffset ReturnBaseOffset =
          ComputeReturnAdjustmentBaseOffset(Context, NewMD, OverriddenMD);

        if (!ReturnBaseOffset.isEmpty()) {
          // Store the return adjustment base offset.
          ReturnAdjustments[SubobjectAndMethod] = ReturnBaseOffset;
        }
      }

      // Set the new overrider.
      Overrider.Offset = OverriderOffsetInLayoutClass;
      Overrider.OldOffset = NewBase.getBaseOffset();
      Overrider.Method = NewMD;
      
      // And propagate it further.
      PropagateOverrider(OverriddenMD, NewBase, OverriderOffsetInLayoutClass,
                         NewMD, Offsets);
    }
  }
}

void 
FinalOverriders::MergeSubobjectOffsets(const SubobjectOffsetsMapTy &NewOffsets,
                                       SubobjectOffsetsMapTy &Offsets) {
  // Iterate over the new offsets.
  for (SubobjectOffsetsMapTy::const_iterator I = NewOffsets.begin(),
       E = NewOffsets.end(); I != E; ++I) {
    const CXXRecordDecl *NewRD = I->first;
    const OffsetSetVectorTy& NewOffsetVector = I->second;
    
    OffsetSetVectorTy &OffsetVector = Offsets[NewRD];
    
    // Merge the new offsets set vector into the old.
    OffsetVector.insert(NewOffsetVector.begin(), NewOffsetVector.end());
  }
}

void FinalOverriders::ComputeFinalOverriders(BaseSubobject Base,
                                             bool BaseSubobjectIsVisitedVBase,
                                             uint64_t OffsetInLayoutClass,
                                             SubobjectOffsetsMapTy &Offsets) {
  const CXXRecordDecl *RD = Base.getBase();
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
  
  SubobjectOffsetsMapTy NewOffsets;
  
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    const CXXRecordDecl *BaseDecl = 
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());
    
    // Ignore bases that don't have any virtual member functions.
    if (!BaseDecl->isPolymorphic())
      continue;
    
    bool IsVisitedVirtualBase = BaseSubobjectIsVisitedVBase;
    uint64_t BaseOffset;
    uint64_t BaseOffsetInLayoutClass;
    if (I->isVirtual()) {
      if (!VisitedVirtualBases.insert(BaseDecl))
        IsVisitedVirtualBase = true;
      BaseOffset = MostDerivedClassLayout.getVBaseClassOffset(BaseDecl);
      
      const ASTRecordLayout &LayoutClassLayout =
        Context.getASTRecordLayout(LayoutClass);
      BaseOffsetInLayoutClass = 
        LayoutClassLayout.getVBaseClassOffset(BaseDecl);
    } else {
      BaseOffset = Layout.getBaseClassOffset(BaseDecl) + Base.getBaseOffset();
      BaseOffsetInLayoutClass = Layout.getBaseClassOffset(BaseDecl) +
        OffsetInLayoutClass;
    }
    
    // Compute the final overriders for this base.
    // We always want to compute the final overriders, even if the base is a
    // visited virtual base. Consider:
    //
    // struct A {
    //   virtual void f();
    //   virtual void g();
    // };
    //  
    // struct B : virtual A {
    //   void f();
    // };
    //
    // struct C : virtual A {
    //   void g ();
    // };
    //
    // struct D : B, C { };
    //
    // Here, we still want to compute the overriders for A as a base of C, 
    // because otherwise we'll miss that C::g overrides A::f.
    ComputeFinalOverriders(BaseSubobject(BaseDecl, BaseOffset), 
                           IsVisitedVirtualBase, BaseOffsetInLayoutClass, 
                           NewOffsets);
  }

  /// Now add the overriders for this particular subobject.
  /// (We don't want to do this more than once for a virtual base).
  if (!BaseSubobjectIsVisitedVBase)
    AddOverriders(Base, OffsetInLayoutClass, NewOffsets);
  
  // And merge the newly discovered subobject offsets.
  MergeSubobjectOffsets(NewOffsets, Offsets);

  /// Finally, add the offset for our own subobject.
  Offsets[RD].insert(Base.getBaseOffset());
}

void FinalOverriders::dump(llvm::raw_ostream &Out, BaseSubobject Base) {
  const CXXRecordDecl *RD = Base.getBase();
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);

  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    const CXXRecordDecl *BaseDecl = 
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());
    
    // Ignore bases that don't have any virtual member functions.
    if (!BaseDecl->isPolymorphic())
      continue;

    uint64_t BaseOffset;
    if (I->isVirtual()) {
      if (!VisitedVirtualBases.insert(BaseDecl)) {
        // We've visited this base before.
        continue;
      }
      
      BaseOffset = MostDerivedClassLayout.getVBaseClassOffset(BaseDecl);
    } else {
      BaseOffset = Layout.getBaseClassOffset(BaseDecl) + 
        Base.getBaseOffset();
    }

    dump(Out, BaseSubobject(BaseDecl, BaseOffset));
  }

  Out << "Final overriders for (" << RD->getQualifiedNameAsString() << ", ";
  Out << Base.getBaseOffset() / 8 << ")\n";

  // Now dump the overriders for this base subobject.
  for (CXXRecordDecl::method_iterator I = RD->method_begin(), 
       E = RD->method_end(); I != E; ++I) {
    const CXXMethodDecl *MD = *I;

    if (!MD->isVirtual())
      continue;
  
    OverriderInfo Overrider = getOverrider(Base, MD);

    Out << "  " << MD->getQualifiedNameAsString() << " - (";
    Out << Overrider.Method->getQualifiedNameAsString();
    Out << ", " << Overrider.OldOffset / 8 << ", " << Overrider.Offset / 8 << ')';

    AdjustmentOffsetsMapTy::const_iterator AI =
      ReturnAdjustments.find(std::make_pair(Base, MD));
    if (AI != ReturnAdjustments.end()) {
      const BaseOffset &Offset = AI->second;

      Out << " [ret-adj: ";
      if (Offset.VirtualBase)
        Out << Offset.VirtualBase->getQualifiedNameAsString() << " vbase, ";
             
      Out << Offset.NonVirtualOffset << " nv]";
    }
    
    Out << "\n";
  }  
}

/// VtableComponent - Represents a single component in a vtable.
class VtableComponent {
public:
  enum Kind {
    CK_VCallOffset,
    CK_VBaseOffset,
    CK_OffsetToTop,
    CK_RTTI,
    CK_FunctionPointer,
    
    /// CK_CompleteDtorPointer - A pointer to the complete destructor.
    CK_CompleteDtorPointer,
    
    /// CK_DeletingDtorPointer - A pointer to the deleting destructor.
    CK_DeletingDtorPointer,
    
    /// CK_UnusedFunctionPointer - In some cases, a vtable function pointer
    /// will end up never being called. Such vtable function pointers are
    /// represented as a CK_UnusedFunctionPointer. 
    CK_UnusedFunctionPointer
  };

  static VtableComponent MakeVCallOffset(int64_t Offset) {
    return VtableComponent(CK_VCallOffset, Offset);
  }

  static VtableComponent MakeVBaseOffset(int64_t Offset) {
    return VtableComponent(CK_VBaseOffset, Offset);
  }

  static VtableComponent MakeOffsetToTop(int64_t Offset) {
    return VtableComponent(CK_OffsetToTop, Offset);
  }
  
  static VtableComponent MakeRTTI(const CXXRecordDecl *RD) {
    return VtableComponent(CK_RTTI, reinterpret_cast<uintptr_t>(RD));
  }

  static VtableComponent MakeFunction(const CXXMethodDecl *MD) {
    assert(!isa<CXXDestructorDecl>(MD) && 
           "Don't use MakeFunction with destructors!");

    return VtableComponent(CK_FunctionPointer, 
                           reinterpret_cast<uintptr_t>(MD));
  }
  
  static VtableComponent MakeCompleteDtor(const CXXDestructorDecl *DD) {
    return VtableComponent(CK_CompleteDtorPointer,
                           reinterpret_cast<uintptr_t>(DD));
  }

  static VtableComponent MakeDeletingDtor(const CXXDestructorDecl *DD) {
    return VtableComponent(CK_DeletingDtorPointer, 
                           reinterpret_cast<uintptr_t>(DD));
  }

  static VtableComponent MakeUnusedFunction(const CXXMethodDecl *MD) {
    assert(!isa<CXXDestructorDecl>(MD) && 
           "Don't use MakeUnusedFunction with destructors!");
    return VtableComponent(CK_UnusedFunctionPointer,
                           reinterpret_cast<uintptr_t>(MD));                           
  }

  /// getKind - Get the kind of this vtable component.
  Kind getKind() const {
    return (Kind)(Value & 0x7);
  }

  int64_t getVCallOffset() const {
    assert(getKind() == CK_VCallOffset && "Invalid component kind!");
    
    return getOffset();
  }

  int64_t getVBaseOffset() const {
    assert(getKind() == CK_VBaseOffset && "Invalid component kind!");
    
    return getOffset();
  }

  int64_t getOffsetToTop() const {
    assert(getKind() == CK_OffsetToTop && "Invalid component kind!");
    
    return getOffset();
  }
  
  const CXXRecordDecl *getRTTIDecl() const {
    assert(getKind() == CK_RTTI && "Invalid component kind!");
    
    return reinterpret_cast<CXXRecordDecl *>(getPointer());
  }
  
  const CXXMethodDecl *getFunctionDecl() const {
    assert(getKind() == CK_FunctionPointer);
    
    return reinterpret_cast<CXXMethodDecl *>(getPointer());
  }

  const CXXDestructorDecl *getDestructorDecl() const {
    assert((getKind() == CK_CompleteDtorPointer ||
            getKind() == CK_DeletingDtorPointer) && "Invalid component kind!");
    
    return reinterpret_cast<CXXDestructorDecl *>(getPointer());
  }

  const CXXMethodDecl *getUnusedFunctionDecl() const {
    assert(getKind() == CK_UnusedFunctionPointer);
    
    return reinterpret_cast<CXXMethodDecl *>(getPointer());
  }
  
private:
  VtableComponent(Kind ComponentKind, int64_t Offset) {
    assert((ComponentKind == CK_VCallOffset || 
            ComponentKind == CK_VBaseOffset ||
            ComponentKind == CK_OffsetToTop) && "Invalid component kind!");
    assert(Offset <= ((1LL << 56) - 1) && "Offset is too big!");
    
    Value = ((Offset << 3) | ComponentKind);
  }

  VtableComponent(Kind ComponentKind, uintptr_t Ptr) {
    assert((ComponentKind == CK_RTTI || 
            ComponentKind == CK_FunctionPointer ||
            ComponentKind == CK_CompleteDtorPointer ||
            ComponentKind == CK_DeletingDtorPointer ||
            ComponentKind == CK_UnusedFunctionPointer) &&
            "Invalid component kind!");
    
    assert((Ptr & 7) == 0 && "Pointer not sufficiently aligned!");
    
    Value = Ptr | ComponentKind;
  }
  
  int64_t getOffset() const {
    assert((getKind() == CK_VCallOffset || getKind() == CK_VBaseOffset ||
            getKind() == CK_OffsetToTop) && "Invalid component kind!");
    
    return Value >> 3;
  }

  uintptr_t getPointer() const {
    assert((getKind() == CK_RTTI || 
            getKind() == CK_FunctionPointer ||
            getKind() == CK_CompleteDtorPointer ||
            getKind() == CK_DeletingDtorPointer ||
            getKind() == CK_UnusedFunctionPointer) &&
           "Invalid component kind!");
    
    return static_cast<uintptr_t>(Value & ~7ULL);
  }
  
  /// The kind is stored in the lower 3 bits of the value. For offsets, we
  /// make use of the facts that classes can't be larger than 2^55 bytes,
  /// so we store the offset in the lower part of the 61 bytes that remain.
  /// (The reason that we're not simply using a PointerIntPair here is that we
  /// need the offsets to be 64-bit, even when on a 32-bit machine).
  int64_t Value;
};

/// VCallOffsetMap - Keeps track of vcall offsets when building a vtable.
struct VCallOffsetMap {
  
  typedef std::pair<const CXXMethodDecl *, int64_t> MethodAndOffsetPairTy;
  
  /// Offsets - Keeps track of methods and their offsets.
  // FIXME: This should be a real map and not a vector.
  llvm::SmallVector<MethodAndOffsetPairTy, 16> Offsets;

  /// MethodsCanShareVCallOffset - Returns whether two virtual member functions
  /// can share the same vcall offset.
  static bool MethodsCanShareVCallOffset(const CXXMethodDecl *LHS,
                                         const CXXMethodDecl *RHS);

public:
  /// AddVCallOffset - Adds a vcall offset to the map. Returns true if the
  /// add was successful, or false if there was already a member function with
  /// the same signature in the map.
  bool AddVCallOffset(const CXXMethodDecl *MD, int64_t OffsetOffset);
  
  /// getVCallOffsetOffset - Returns the vcall offset offset (relative to the
  /// vtable address point) for the given virtual member function.
  int64_t getVCallOffsetOffset(const CXXMethodDecl *MD);
  
  // empty - Return whether the offset map is empty or not.
  bool empty() const { return Offsets.empty(); }
};

static bool HasSameVirtualSignature(const CXXMethodDecl *LHS,
                                    const CXXMethodDecl *RHS) {
  ASTContext &C = LHS->getASTContext(); // TODO: thread this down
  CanQual<FunctionProtoType>
    LT = C.getCanonicalType(LHS->getType()).getAs<FunctionProtoType>(),
    RT = C.getCanonicalType(RHS->getType()).getAs<FunctionProtoType>();

  // Fast-path matches in the canonical types.
  if (LT == RT) return true;

  // Force the signatures to match.  We can't rely on the overrides
  // list here because there isn't necessarily an inheritance
  // relationship between the two methods.
  if (LT.getQualifiers() != RT.getQualifiers() ||
      LT->getNumArgs() != RT->getNumArgs())
    return false;
  for (unsigned I = 0, E = LT->getNumArgs(); I != E; ++I)
    if (LT->getArgType(I) != RT->getArgType(I))
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
                                    int64_t OffsetOffset) {
  // Check if we can reuse an offset.
  for (unsigned I = 0, E = Offsets.size(); I != E; ++I) {
    if (MethodsCanShareVCallOffset(Offsets[I].first, MD))
      return false;
  }
  
  // Add the offset.
  Offsets.push_back(MethodAndOffsetPairTy(MD, OffsetOffset));
  return true;
}

int64_t VCallOffsetMap::getVCallOffsetOffset(const CXXMethodDecl *MD) {
  // Look for an offset.
  for (unsigned I = 0, E = Offsets.size(); I != E; ++I) {
    if (MethodsCanShareVCallOffset(Offsets[I].first, MD))
      return Offsets[I].second;
  }
  
  assert(false && "Should always find a vcall offset offset!");
  return 0;
}

/// VCallAndVBaseOffsetBuilder - Class for building vcall and vbase offsets.
class VCallAndVBaseOffsetBuilder {
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
  typedef llvm::SmallVector<VtableComponent, 64> VtableComponentVectorTy;
  VtableComponentVectorTy Components;
  
  /// VisitedVirtualBases - Visited virtual bases.
  llvm::SmallPtrSet<const CXXRecordDecl *, 4> VisitedVirtualBases;
  
  /// VCallOffsets - Keeps track of vcall offsets.
  VCallOffsetMap VCallOffsets;

  typedef llvm::DenseMap<const CXXRecordDecl *, int64_t> 
    VBaseOffsetOffsetsMapTy;

  /// VBaseOffsetOffsets - Contains the offsets of the virtual base offsets,
  /// relative to the address point.
  VBaseOffsetOffsetsMapTy VBaseOffsetOffsets;
  
  /// FinalOverriders - The final overriders of the most derived class.
  /// (Can be null when we're not building a vtable of the most derived class).
  const FinalOverriders *Overriders;

  /// AddVCallAndVBaseOffsets - Add vcall offsets and vbase offsets for the
  /// given base subobject.
  void AddVCallAndVBaseOffsets(BaseSubobject Base, bool BaseIsVirtual,
                               uint64_t RealBaseOffset);
  
  /// AddVCallOffsets - Add vcall offsets for the given base subobject.
  void AddVCallOffsets(BaseSubobject Base, uint64_t VBaseOffset);
  
  /// AddVBaseOffsets - Add vbase offsets for the given class.
  void AddVBaseOffsets(const CXXRecordDecl *Base, uint64_t OffsetInLayoutClass);
  
  /// getCurrentOffsetOffset - Get the current vcall or vbase offset offset in
  /// bytes, relative to the vtable address point.
  int64_t getCurrentOffsetOffset() const;
  
public:
  VCallAndVBaseOffsetBuilder(const CXXRecordDecl *MostDerivedClass,
                             const CXXRecordDecl *LayoutClass,
                             const FinalOverriders *Overriders,
                             BaseSubobject Base, bool BaseIsVirtual,
                             uint64_t OffsetInLayoutClass)
    : MostDerivedClass(MostDerivedClass), LayoutClass(LayoutClass), 
    Context(MostDerivedClass->getASTContext()), Overriders(Overriders) {
      
    // Add vcall and vbase offsets.
    AddVCallAndVBaseOffsets(Base, BaseIsVirtual, OffsetInLayoutClass);
  }
  
  /// Methods for iterating over the components.
  typedef VtableComponentVectorTy::const_reverse_iterator const_iterator;
  const_iterator components_begin() const { return Components.rbegin(); }
  const_iterator components_end() const { return Components.rend(); }
  
  const VCallOffsetMap& getVCallOffsets() const { return VCallOffsets; }
  const VBaseOffsetOffsetsMapTy getVBaseOffsetOffsets() const {
    return VBaseOffsetOffsets;
  }
};
  
void 
VCallAndVBaseOffsetBuilder::AddVCallAndVBaseOffsets(BaseSubobject Base,
                                                    bool BaseIsVirtual,
                                                    uint64_t RealBaseOffset) {
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
    bool PrimaryBaseIsVirtual = Layout.getPrimaryBaseWasVirtual();

    uint64_t PrimaryBaseOffset;
    
    // Get the base offset of the primary base.
    if (PrimaryBaseIsVirtual) {
      assert(Layout.getVBaseClassOffset(PrimaryBase) == 0 &&
             "Primary vbase should have a zero offset!");
      
      const ASTRecordLayout &MostDerivedClassLayout =
        Context.getASTRecordLayout(MostDerivedClass);
      
      PrimaryBaseOffset = 
        MostDerivedClassLayout.getVBaseClassOffset(PrimaryBase);
    } else {
      assert(Layout.getBaseClassOffset(PrimaryBase) == 0 &&
             "Primary base should have a zero offset!");

      PrimaryBaseOffset = Base.getBaseOffset();
    }

    AddVCallAndVBaseOffsets(BaseSubobject(PrimaryBase, PrimaryBaseOffset),
                            PrimaryBaseIsVirtual, RealBaseOffset);
  }

  AddVBaseOffsets(Base.getBase(), RealBaseOffset);

  // We only want to add vcall offsets for virtual bases.
  if (BaseIsVirtual)
    AddVCallOffsets(Base, RealBaseOffset);
}

int64_t VCallAndVBaseOffsetBuilder::getCurrentOffsetOffset() const {
  // OffsetIndex is the index of this vcall or vbase offset, relative to the 
  // vtable address point. (We subtract 3 to account for the information just
  // above the address point, the RTTI info, the offset to top, and the
  // vcall offset itself).
  int64_t OffsetIndex = -(int64_t)(3 + Components.size());
    
  // FIXME: We shouldn't use / 8 here.
  int64_t OffsetOffset = OffsetIndex * 
    (int64_t)Context.Target.getPointerWidth(0) / 8;

  return OffsetOffset;
}

void VCallAndVBaseOffsetBuilder::AddVCallOffsets(BaseSubobject Base, 
                                                 uint64_t VBaseOffset) {
  const CXXRecordDecl *RD = Base.getBase();
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);

  const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();

  // Handle the primary base first.
  if (PrimaryBase) {
    uint64_t PrimaryBaseOffset;
    
    // Get the base offset of the primary base.
    if (Layout.getPrimaryBaseWasVirtual()) {
      assert(Layout.getVBaseClassOffset(PrimaryBase) == 0 &&
             "Primary vbase should have a zero offset!");
      
      const ASTRecordLayout &MostDerivedClassLayout =
        Context.getASTRecordLayout(MostDerivedClass);
      
      PrimaryBaseOffset = 
        MostDerivedClassLayout.getVBaseClassOffset(PrimaryBase);
    } else {
      assert(Layout.getBaseClassOffset(PrimaryBase) == 0 &&
             "Primary base should have a zero offset!");

      PrimaryBaseOffset = Base.getBaseOffset();
    }
    
    AddVCallOffsets(BaseSubobject(PrimaryBase, PrimaryBaseOffset),
                    VBaseOffset);
  }

  // Add the vcall offsets.
  for (CXXRecordDecl::method_iterator I = RD->method_begin(),
       E = RD->method_end(); I != E; ++I) {
    const CXXMethodDecl *MD = *I;
    
    if (!MD->isVirtual())
      continue;

    int64_t OffsetOffset = getCurrentOffsetOffset();
    
    // Don't add a vcall offset if we already have one for this member function
    // signature.
    if (!VCallOffsets.AddVCallOffset(MD, OffsetOffset))
      continue;

    int64_t Offset = 0;

    if (Overriders) {
      // Get the final overrider.
      FinalOverriders::OverriderInfo Overrider = 
        Overriders->getOverrider(Base, MD);
      
      /// The vcall offset is the offset from the virtual base to the object 
      /// where the function was overridden.
      // FIXME: We should not use / 8 here.
      Offset = (int64_t)(Overrider.Offset - VBaseOffset) / 8;
    }
    
    Components.push_back(VtableComponent::MakeVCallOffset(Offset));
  }

  // And iterate over all non-virtual bases (ignoring the primary base).
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
  
    if (I->isVirtual())
      continue;

    const CXXRecordDecl *BaseDecl =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());
    if (BaseDecl == PrimaryBase)
      continue;

    // Get the base offset of this base.
    uint64_t BaseOffset = Base.getBaseOffset() + 
      Layout.getBaseClassOffset(BaseDecl);
    
    AddVCallOffsets(BaseSubobject(BaseDecl, BaseOffset), VBaseOffset);
  }
}

void VCallAndVBaseOffsetBuilder::AddVBaseOffsets(const CXXRecordDecl *RD,
                                                 uint64_t OffsetInLayoutClass) {
  const ASTRecordLayout &LayoutClassLayout = 
    Context.getASTRecordLayout(LayoutClass);

  // Add vbase offsets.
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    const CXXRecordDecl *BaseDecl =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    // Check if this is a virtual base that we haven't visited before.
    if (I->isVirtual() && VisitedVirtualBases.insert(BaseDecl)) {
      // FIXME: We shouldn't use / 8 here.
      int64_t Offset = 
        (int64_t)(LayoutClassLayout.getVBaseClassOffset(BaseDecl) - 
                  OffsetInLayoutClass) / 8;

      // Add the vbase offset offset.
      assert(!VBaseOffsetOffsets.count(BaseDecl) &&
             "vbase offset offset already exists!");

      int64_t VBaseOffsetOffset = getCurrentOffsetOffset();
      VBaseOffsetOffsets.insert(std::make_pair(BaseDecl, VBaseOffsetOffset));

      Components.push_back(VtableComponent::MakeVBaseOffset(Offset));
    }

    // Check the base class looking for more vbase offsets.
    AddVBaseOffsets(BaseDecl, OffsetInLayoutClass);
  }
}

/// VtableBuilder - Class for building vtable layout information.
class VtableBuilder {
public:
  /// PrimaryBasesSetVectorTy - A set vector of direct and indirect 
  /// primary bases.
  typedef llvm::SmallSetVector<const CXXRecordDecl *, 8> 
    PrimaryBasesSetVectorTy;
  
private:
  /// VtableInfo - Global vtable information.
  CGVtableInfo &VtableInfo;
  
  /// MostDerivedClass - The most derived class for which we're building this
  /// vtable.
  const CXXRecordDecl *MostDerivedClass;

  /// MostDerivedClassOffset - If we're building a construction vtable, this
  /// holds the offset from the layout class to the most derived class.
  const uint64_t MostDerivedClassOffset;
  
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

  typedef llvm::DenseMap<const CXXRecordDecl *, int64_t> 
    VBaseOffsetOffsetsMapTy;
  
  /// VBaseOffsetOffsets - Contains the offsets of the virtual base offsets for
  /// the most derived class.
  VBaseOffsetOffsetsMapTy VBaseOffsetOffsets;
  
  /// Components - The components of the vtable being built.
  llvm::SmallVector<VtableComponent, 64> Components;

  /// AddressPoints - Address points for the vtable being built.
  CGVtableInfo::AddressPointsMapTy AddressPoints;

  /// ReturnAdjustment - A return adjustment.
  struct ReturnAdjustment {
    /// NonVirtual - The non-virtual adjustment from the derived object to its
    /// nearest virtual base.
    int64_t NonVirtual;
    
    /// VBaseOffsetOffset - The offset (in bytes), relative to the address point 
    /// of the virtual base class offset.
    int64_t VBaseOffsetOffset;
    
    ReturnAdjustment() : NonVirtual(0), VBaseOffsetOffset(0) { }
    
    bool isEmpty() const { return !NonVirtual && !VBaseOffsetOffset; }
  };
  
  /// ReturnAdjustments - The return adjustments needed in this vtable.
  llvm::SmallVector<std::pair<uint64_t, ReturnAdjustment>, 16> 
    ReturnAdjustments;

  /// MethodInfo - Contains information about a method in a vtable.
  /// (Used for computing 'this' pointer adjustment thunks.
  struct MethodInfo {
    /// BaseOffset - The base offset of this method.
    const uint64_t BaseOffset;
    
    /// BaseOffsetInLayoutClass - The base offset in the layout class of this
    /// method.
    const uint64_t BaseOffsetInLayoutClass;
    
    /// VtableIndex - The index in the vtable that this method has.
    /// (For destructors, this is the index of the complete destructor).
    const uint64_t VtableIndex;
    
    MethodInfo(uint64_t BaseOffset, uint64_t BaseOffsetInLayoutClass, 
               uint64_t VtableIndex)
      : BaseOffset(BaseOffset), 
      BaseOffsetInLayoutClass(BaseOffsetInLayoutClass),
      VtableIndex(VtableIndex) { }
    
    MethodInfo() : BaseOffset(0), BaseOffsetInLayoutClass(0), VtableIndex(0) { }
  };
  
  typedef llvm::DenseMap<const CXXMethodDecl *, MethodInfo> MethodInfoMapTy;
  
  /// MethodInfoMap - The information for all methods in the vtable we're
  /// currently building.
  MethodInfoMapTy MethodInfoMap;
  
  /// ThisAdjustment - A 'this' pointer adjustment thunk.
  struct ThisAdjustment {
    /// NonVirtual - The non-virtual adjustment from the derived object to its
    /// nearest virtual base.
    int64_t NonVirtual;

    /// VCallOffsetOffset - The offset (in bytes), relative to the address point,
    /// of the virtual call offset.
    int64_t VCallOffsetOffset;
    
    ThisAdjustment() : NonVirtual(0), VCallOffsetOffset(0) { }

    bool isEmpty() const { return !NonVirtual && !VCallOffsetOffset; }
  };
  
  /// ThisAdjustments - The 'this' pointer adjustments needed in this vtable.
  llvm::SmallVector<std::pair<uint64_t, ThisAdjustment>, 16> 
    ThisAdjustments;

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
  /// given virtual member function and the 'this' pointer adjustment base 
  /// offset.
  ThisAdjustment ComputeThisAdjustment(const CXXMethodDecl *MD,
                                       BaseOffset Offset);
  
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
                       uint64_t BaseOffsetInLayoutClass,
                       const CXXRecordDecl *FirstBaseInPrimaryBaseChain,
                       uint64_t FirstBaseOffsetInLayoutClass) const;

  
  /// AddMethods - Add the methods of this base subobject and all its
  /// primary bases to the vtable components vector.
  void AddMethods(BaseSubobject Base, uint64_t BaseOffsetInLayoutClass,                  
                  const CXXRecordDecl *FirstBaseInPrimaryBaseChain,
                  uint64_t FirstBaseOffsetInLayoutClass,
                  PrimaryBasesSetVectorTy &PrimaryBases);

  // LayoutVtable - Layout the vtable for the given base class, including its
  // secondary vtables and any vtables for virtual bases.
  void LayoutVtable();

  /// LayoutPrimaryAndSecondaryVtables - Layout the primary vtable for the
  /// given base subobject, as well as all its secondary vtables.
  void LayoutPrimaryAndSecondaryVtables(BaseSubobject Base,
                                        bool BaseIsVirtual,
                                        uint64_t OffsetInLayoutClass);
  
  /// LayoutSecondaryVtables - Layout the secondary vtables for the given base
  /// subobject.
  ///
  /// \param BaseIsMorallyVirtual whether the base subobject is a virtual base
  /// or a direct or indirect base of a virtual base.
  void LayoutSecondaryVtables(BaseSubobject Base, bool BaseIsMorallyVirtual,
                              uint64_t OffsetInLayoutClass);

  /// DeterminePrimaryVirtualBases - Determine the primary virtual bases in this
  /// class hierarchy.
  void DeterminePrimaryVirtualBases(const CXXRecordDecl *RD, 
                                    uint64_t OffsetInLayoutClass,
                                    VisitedVirtualBasesSetTy &VBases);

  /// LayoutVtablesForVirtualBases - Layout vtables for all virtual bases of the
  /// given base (excluding any primary bases).
  void LayoutVtablesForVirtualBases(const CXXRecordDecl *RD, 
                                    VisitedVirtualBasesSetTy &VBases);

  /// isBuildingConstructionVtable - Return whether this vtable builder is
  /// building a construction vtable.
  bool isBuildingConstructorVtable() const { 
    return MostDerivedClass != LayoutClass;
  }

public:
  VtableBuilder(CGVtableInfo &VtableInfo, const CXXRecordDecl *MostDerivedClass,
                uint64_t MostDerivedClassOffset, bool MostDerivedClassIsVirtual,
                const CXXRecordDecl *LayoutClass)
    : VtableInfo(VtableInfo), MostDerivedClass(MostDerivedClass),
    MostDerivedClassOffset(MostDerivedClassOffset), 
    MostDerivedClassIsVirtual(MostDerivedClassIsVirtual), 
    LayoutClass(LayoutClass), Context(MostDerivedClass->getASTContext()), 
    Overriders(MostDerivedClass, MostDerivedClassOffset, LayoutClass) {

    LayoutVtable();
  }

  /// dumpLayout - Dump the vtable layout.
  void dumpLayout(llvm::raw_ostream&);
};

/// OverridesMethodInBases - Checks whether whether this virtual member 
/// function overrides a member function in any of the given bases.
/// Returns the overridden member function, or null if none was found.
static const CXXMethodDecl * 
OverridesMethodInBases(const CXXMethodDecl *MD,
                       VtableBuilder::PrimaryBasesSetVectorTy &Bases) {
  for (CXXMethodDecl::method_iterator I = MD->begin_overridden_methods(),
       E = MD->end_overridden_methods(); I != E; ++I) {
    const CXXMethodDecl *OverriddenMD = *I;
    const CXXRecordDecl *OverriddenRD = OverriddenMD->getParent();
    assert(OverriddenMD->isCanonicalDecl() &&
           "Should have the canonical decl of the overridden RD!");
    
    if (Bases.count(OverriddenRD))
      return OverriddenMD;
  }
      
  return 0;
}

void VtableBuilder::ComputeThisAdjustments() {
  std::map<uint64_t, ThisAdjustment> SortedThisAdjustments;
  
  // Now go through the method info map and see if any of the methods need
  // 'this' pointer adjustments.
  for (MethodInfoMapTy::const_iterator I = MethodInfoMap.begin(),
       E = MethodInfoMap.end(); I != E; ++I) {
    const CXXMethodDecl *MD = I->first;
    const MethodInfo &MethodInfo = I->second;

    // Get the final overrider for this method.
    FinalOverriders::OverriderInfo Overrider =
      Overriders.getOverrider(BaseSubobject(MD->getParent(), 
                                            MethodInfo.BaseOffset), MD);
    
    // Check if we need an adjustment.
    if (Overrider.Offset == MethodInfo.BaseOffsetInLayoutClass)
      continue;
    
    uint64_t VtableIndex = MethodInfo.VtableIndex;

    // Ignore adjustments for pure virtual member functions.
    if (Overrider.Method->isPure())
      continue;

    // Ignore adjustments for unused function pointers.
    if (Components[VtableIndex].getKind() == 
        VtableComponent::CK_UnusedFunctionPointer)
      continue;

    BaseSubobject OverriddenBaseSubobject(MD->getParent(),
                                          MethodInfo.BaseOffsetInLayoutClass);
    
    BaseSubobject OverriderBaseSubobject(Overrider.Method->getParent(),
                                         Overrider.Offset);

    // Compute the adjustment offset.
    BaseOffset ThisAdjustmentOffset =
      ComputeThisAdjustmentBaseOffset(OverriddenBaseSubobject,
                                      OverriderBaseSubobject);

    // Then compute the adjustment itself.
    ThisAdjustment ThisAdjustment = ComputeThisAdjustment(Overrider.Method,
                                                          ThisAdjustmentOffset);

    // Add it.
    SortedThisAdjustments.insert(std::make_pair(VtableIndex, ThisAdjustment));
    
    if (isa<CXXDestructorDecl>(MD)) {
      // Add an adjustment for the deleting destructor as well.
      SortedThisAdjustments.insert(std::make_pair(VtableIndex + 1,
                                                  ThisAdjustment));
    }
  }

  /// Clear the method info map.
  MethodInfoMap.clear();
  
  // Add the sorted elements.
  ThisAdjustments.append(SortedThisAdjustments.begin(),
                         SortedThisAdjustments.end());
}

VtableBuilder::ReturnAdjustment 
VtableBuilder::ComputeReturnAdjustment(BaseOffset Offset) {
  ReturnAdjustment Adjustment;
  
  if (!Offset.isEmpty()) {
    if (Offset.VirtualBase) {
      // Get the virtual base offset offset.
      if (Offset.DerivedClass == MostDerivedClass) {
        // We can get the offset offset directly from our map.
        Adjustment.VBaseOffsetOffset = 
          VBaseOffsetOffsets.lookup(Offset.VirtualBase);
      } else {
        Adjustment.VBaseOffsetOffset = 
          VtableInfo.getVirtualBaseOffsetIndex(Offset.DerivedClass,
                                               Offset.VirtualBase);
      }

      // FIXME: Once the assert in getVirtualBaseOffsetIndex is back again,
      // we can get rid of this assert.
      assert(Adjustment.VBaseOffsetOffset != 0 && 
             "Invalid base offset offset!");
    }

    Adjustment.NonVirtual = Offset.NonVirtualOffset;
  }
  
  return Adjustment;
}

BaseOffset
VtableBuilder::ComputeThisAdjustmentBaseOffset(BaseSubobject Base,
                                               BaseSubobject Derived) const {
  const CXXRecordDecl *BaseRD = Base.getBase();
  const CXXRecordDecl *DerivedRD = Derived.getBase();
  
  CXXBasePaths Paths(/*FindAmbiguities=*/true,
                     /*RecordPaths=*/true, /*DetectVirtual=*/true);

  if (!const_cast<CXXRecordDecl *>(DerivedRD)->
      isDerivedFrom(const_cast<CXXRecordDecl *>(BaseRD), Paths)) {
    assert(false && "Class must be derived from the passed in base class!");
    return BaseOffset();
  }

  // We have to go through all the paths, and see which one leads us to the
  // right base subobject.
  for (CXXBasePaths::const_paths_iterator I = Paths.begin(), E = Paths.end();
       I != E; ++I) {
    BaseOffset Offset = ComputeBaseOffset(Context, DerivedRD, *I);
    
    // FIXME: Should not use * 8 here.
    uint64_t OffsetToBaseSubobject = Offset.NonVirtualOffset * 8;
    
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
  

VtableBuilder::ThisAdjustment
VtableBuilder::ComputeThisAdjustment(const CXXMethodDecl *MD,
                                     BaseOffset Offset) {
  ThisAdjustment Adjustment;
  
  if (!Offset.isEmpty()) {
    if (Offset.VirtualBase) {
      // Get the vcall offset map for this virtual base.
      VCallOffsetMap &VCallOffsets = VCallOffsetsForVBases[Offset.VirtualBase];
      
      if (VCallOffsets.empty()) {
        // We don't have vcall offsets for this virtual base, go ahead and
        // build them.
        VCallAndVBaseOffsetBuilder Builder(MostDerivedClass, MostDerivedClass,
                                           /*FinalOverriders=*/0,
                                           BaseSubobject(Offset.VirtualBase, 0),                                           
                                           /*BaseIsVirtual=*/true,
                                           /*OffsetInLayoutClass=*/0);
        
        VCallOffsets = Builder.getVCallOffsets();
      }
      
      Adjustment.VCallOffsetOffset = VCallOffsets.getVCallOffsetOffset(MD);
    }

    Adjustment.NonVirtual = Offset.NonVirtualOffset;
  }
  
  return Adjustment;
}

void 
VtableBuilder::AddMethod(const CXXMethodDecl *MD,
                         ReturnAdjustment ReturnAdjustment) {
  if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(MD)) {
    assert(ReturnAdjustment.isEmpty() && 
           "Destructor can't have return adjustment!");

    // Add both the complete destructor and the deleting destructor.
    Components.push_back(VtableComponent::MakeCompleteDtor(DD));
    Components.push_back(VtableComponent::MakeDeletingDtor(DD));
  } else {
    // Add the return adjustment if necessary.
    if (!ReturnAdjustment.isEmpty())
      ReturnAdjustments.push_back(std::make_pair(Components.size(),
                                                 ReturnAdjustment));

    // Add the function.
    Components.push_back(VtableComponent::MakeFunction(MD));
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
/// and { A } as the set of  bases.
static bool
OverridesIndirectMethodInBases(const CXXMethodDecl *MD,
                               VtableBuilder::PrimaryBasesSetVectorTy &Bases) {
  for (CXXMethodDecl::method_iterator I = MD->begin_overridden_methods(),
       E = MD->end_overridden_methods(); I != E; ++I) {
    const CXXMethodDecl *OverriddenMD = *I;
    const CXXRecordDecl *OverriddenRD = OverriddenMD->getParent();
    assert(OverriddenMD->isCanonicalDecl() &&
           "Should have the canonical decl of the overridden RD!");
    
    if (Bases.count(OverriddenRD))
      return true;
    
    // Check "indirect overriders".
    if (OverridesIndirectMethodInBases(OverriddenMD, Bases))
      return true;
  }
   
  return false;
}

bool 
VtableBuilder::IsOverriderUsed(const CXXMethodDecl *Overrider,
                               uint64_t BaseOffsetInLayoutClass,
                               const CXXRecordDecl *FirstBaseInPrimaryBaseChain,
                               uint64_t FirstBaseOffsetInLayoutClass) const {
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
  
  VtableBuilder::PrimaryBasesSetVectorTy PrimaryBases;

  const CXXRecordDecl *RD = FirstBaseInPrimaryBaseChain;
  PrimaryBases.insert(RD);

  // Now traverse the base chain, starting with the first base, until we find
  // the base that is no longer a primary base.
  while (true) {
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
    const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();
    
    if (!PrimaryBase)
      break;
    
    if (Layout.getPrimaryBaseWasVirtual()) {
      assert(Layout.getVBaseClassOffset(PrimaryBase) == 0 && 
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
      assert(Layout.getBaseClassOffset(PrimaryBase) == 0 && 
             "Primary base should always be at offset 0!");
    }
    
    if (!PrimaryBases.insert(PrimaryBase))
      assert(false && "Found a duplicate primary base!");

    RD = PrimaryBase;
  }
  
  // If the final overrider is an override of one of the primary bases,
  // then we know that it will be used.
  return OverridesIndirectMethodInBases(Overrider, PrimaryBases);
}

/// FindNearestOverriddenMethod - Given a method, returns the overridden method
/// from the nearest base. Returns null if no method was found.
static const CXXMethodDecl * 
FindNearestOverriddenMethod(const CXXMethodDecl *MD,
                            VtableBuilder::PrimaryBasesSetVectorTy &Bases) {
  for (int I = Bases.size(), E = 0; I != E; --I) {
    const CXXRecordDecl *PrimaryBase = Bases[I - 1];

    // Now check the overriden methods.
    for (CXXMethodDecl::method_iterator I = MD->begin_overridden_methods(),
         E = MD->end_overridden_methods(); I != E; ++I) {
      const CXXMethodDecl *OverriddenMD = *I;

      // We found our overridden method.
      if (OverriddenMD->getParent() == PrimaryBase)
        return OverriddenMD;
    }
  }
  
  return 0;
}  

void
VtableBuilder::AddMethods(BaseSubobject Base, uint64_t BaseOffsetInLayoutClass,                  
                          const CXXRecordDecl *FirstBaseInPrimaryBaseChain,
                          uint64_t FirstBaseOffsetInLayoutClass,
                          PrimaryBasesSetVectorTy &PrimaryBases) {
  const CXXRecordDecl *RD = Base.getBase();
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);

  if (const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase()) {
    uint64_t PrimaryBaseOffset;
    uint64_t PrimaryBaseOffsetInLayoutClass;
    if (Layout.getPrimaryBaseWasVirtual()) {
      assert(Layout.getVBaseClassOffset(PrimaryBase) == 0 &&
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
      assert(Layout.getBaseClassOffset(PrimaryBase) == 0 &&
             "Primary base should have a zero offset!");

      PrimaryBaseOffset = Base.getBaseOffset();
      PrimaryBaseOffsetInLayoutClass = BaseOffsetInLayoutClass;
    }

    AddMethods(BaseSubobject(PrimaryBase, PrimaryBaseOffset),
               PrimaryBaseOffsetInLayoutClass, FirstBaseInPrimaryBaseChain, 
               FirstBaseOffsetInLayoutClass, PrimaryBases);
    
    if (!PrimaryBases.insert(PrimaryBase))
      assert(false && "Found a duplicate primary base!");
  }

  // Now go through all virtual member functions and add them.
  for (CXXRecordDecl::method_iterator I = RD->method_begin(),
       E = RD->method_end(); I != E; ++I) {
    const CXXMethodDecl *MD = *I;
  
    if (!MD->isVirtual())
      continue;

    // Get the final overrider.
    FinalOverriders::OverriderInfo Overrider = 
      Overriders.getOverrider(Base, MD);

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
        
        MethodInfo MethodInfo(Base.getBaseOffset(), 
                              BaseOffsetInLayoutClass,
                              OverriddenMethodInfo.VtableIndex);

        assert(!MethodInfoMap.count(MD) &&
               "Should not have method info for this method yet!");
        
        MethodInfoMap.insert(std::make_pair(MD, MethodInfo));
        MethodInfoMap.erase(OverriddenMD);
        continue;
      }
    }

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
      Components.push_back(VtableComponent::MakeUnusedFunction(OverriderMD));
      continue;
    }
    
    // Check if this overrider needs a return adjustment.
    BaseOffset ReturnAdjustmentOffset = 
      Overriders.getReturnAdjustmentOffset(Base, MD);

    ReturnAdjustment ReturnAdjustment = 
      ComputeReturnAdjustment(ReturnAdjustmentOffset);
    
    AddMethod(Overrider.Method, ReturnAdjustment);
  }
}

void VtableBuilder::LayoutVtable() {
  LayoutPrimaryAndSecondaryVtables(BaseSubobject(MostDerivedClass, 0),
                                   MostDerivedClassIsVirtual,
                                   MostDerivedClassOffset);
  
  VisitedVirtualBasesSetTy VBases;
  
  // Determine the primary virtual bases.
  DeterminePrimaryVirtualBases(MostDerivedClass, MostDerivedClassOffset,
                               VBases);
  VBases.clear();
  
  LayoutVtablesForVirtualBases(MostDerivedClass, VBases);
}
  
void
VtableBuilder::LayoutPrimaryAndSecondaryVtables(BaseSubobject Base,
                                                bool BaseIsVirtual,
                                                uint64_t OffsetInLayoutClass) {
  assert(Base.getBase()->isDynamicClass() && "class does not have a vtable!");

  // Add vcall and vbase offsets for this vtable.
  VCallAndVBaseOffsetBuilder Builder(MostDerivedClass, LayoutClass, &Overriders,
                                     Base, BaseIsVirtual, OffsetInLayoutClass);
  Components.append(Builder.components_begin(), Builder.components_end());
  
  // Check if we need to add these vcall offsets.
  if (BaseIsVirtual && !Builder.getVCallOffsets().empty()) {
    VCallOffsetMap &VCallOffsets = VCallOffsetsForVBases[Base.getBase()];
    
    if (VCallOffsets.empty())
      VCallOffsets = Builder.getVCallOffsets();
  }

  // If we're laying out the most derived class we want to keep track of the
  // virtual base class offset offsets.
  if (Base.getBase() == MostDerivedClass)
    VBaseOffsetOffsets = Builder.getVBaseOffsetOffsets();

  // Add the offset to top.
  // FIXME: We should not use / 8 here.
  int64_t OffsetToTop = -(int64_t)(OffsetInLayoutClass -
                                   MostDerivedClassOffset) / 8;
  Components.push_back(VtableComponent::MakeOffsetToTop(OffsetToTop));
  
  // Next, add the RTTI.
  Components.push_back(VtableComponent::MakeRTTI(MostDerivedClass));
  
  uint64_t AddressPoint = Components.size();

  // Now go through all virtual member functions and add them.
  PrimaryBasesSetVectorTy PrimaryBases;
  AddMethods(Base, OffsetInLayoutClass, Base.getBase(), OffsetInLayoutClass, 
             PrimaryBases);

  // Compute 'this' pointer adjustments.
  ComputeThisAdjustments();

  // Record the address point.
  AddressPoints.insert(std::make_pair(BaseSubobject(Base.getBase(),
                                                    OffsetInLayoutClass),
                                      AddressPoint));
  
  // Record the address points for all primary bases.
  for (PrimaryBasesSetVectorTy::const_iterator I = PrimaryBases.begin(),
       E = PrimaryBases.end(); I != E; ++I) {
    const CXXRecordDecl *BaseDecl = *I;
    
    // We know that all the primary bases have the same offset as the base
    // subobject.
    BaseSubobject PrimaryBase(BaseDecl, OffsetInLayoutClass);
    AddressPoints.insert(std::make_pair(PrimaryBase, AddressPoint));
  }

  bool BaseIsMorallyVirtual = BaseIsVirtual;
  if (isBuildingConstructorVtable() && Base.getBase() == MostDerivedClass)
    BaseIsMorallyVirtual = false;
  
  // Layout secondary vtables.
  LayoutSecondaryVtables(Base, BaseIsMorallyVirtual, OffsetInLayoutClass);
}

void VtableBuilder::LayoutSecondaryVtables(BaseSubobject Base,
                                           bool BaseIsMorallyVirtual,
                                           uint64_t OffsetInLayoutClass) {
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
    
    const CXXRecordDecl *BaseDecl = 
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    // Ignore bases that don't have a vtable.
    if (!BaseDecl->isDynamicClass())
      continue;

    if (isBuildingConstructorVtable()) {
      // Itanium C++ ABI 2.6.4:
      //   Some of the base class subobjects may not need construction virtual
      //   tables, which will therefore not be present in the construction
      //   virtual table group, even though the subobject virtual tables are
      //   present in the main virtual table group for the complete object.
      if (!BaseIsMorallyVirtual && !BaseDecl->getNumVBases())
        continue;
    }

    // Get the base offset of this base.
    uint64_t RelativeBaseOffset = Layout.getBaseClassOffset(BaseDecl);
    uint64_t BaseOffset = Base.getBaseOffset() + RelativeBaseOffset;
    
    uint64_t BaseOffsetInLayoutClass = OffsetInLayoutClass + RelativeBaseOffset;
    
    // Don't emit a secondary vtable for a primary base. We might however want 
    // to emit secondary vtables for other bases of this base.
    if (BaseDecl == PrimaryBase) {
      LayoutSecondaryVtables(BaseSubobject(BaseDecl, BaseOffset),
                             BaseIsMorallyVirtual, BaseOffsetInLayoutClass);
      continue;
    }

    // Layout the primary vtable (and any secondary vtables) for this base.
    LayoutPrimaryAndSecondaryVtables(BaseSubobject(BaseDecl, BaseOffset),
                                     /*BaseIsVirtual=*/false,
                                     BaseOffsetInLayoutClass);
  }
}

void
VtableBuilder::DeterminePrimaryVirtualBases(const CXXRecordDecl *RD,
                                            uint64_t OffsetInLayoutClass,
                                            VisitedVirtualBasesSetTy &VBases) {
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
  
  // Check if this base has a primary base.
  if (const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase()) {

    // Check if it's virtual.
    if (Layout.getPrimaryBaseWasVirtual()) {
      bool IsPrimaryVirtualBase = true;

      if (isBuildingConstructorVtable()) {
        // Check if the base is actually a primary base in the class we use for
        // layout.
        const ASTRecordLayout &LayoutClassLayout =
          Context.getASTRecordLayout(LayoutClass);

        uint64_t PrimaryBaseOffsetInLayoutClass =
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
    const CXXRecordDecl *BaseDecl = 
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    uint64_t BaseOffsetInLayoutClass;
    
    if (I->isVirtual()) {
      if (!VBases.insert(BaseDecl))
        continue;
      
      const ASTRecordLayout &LayoutClassLayout =
        Context.getASTRecordLayout(LayoutClass);

      BaseOffsetInLayoutClass = LayoutClassLayout.getVBaseClassOffset(BaseDecl);
    } else {
      BaseOffsetInLayoutClass = 
        OffsetInLayoutClass + Layout.getBaseClassOffset(BaseDecl);
    }

    DeterminePrimaryVirtualBases(BaseDecl, BaseOffsetInLayoutClass, VBases);
  }
}

void
VtableBuilder::LayoutVtablesForVirtualBases(const CXXRecordDecl *RD, 
                                            VisitedVirtualBasesSetTy &VBases) {
  // Itanium C++ ABI 2.5.2:
  //   Then come the virtual base virtual tables, also in inheritance graph
  //   order, and again excluding primary bases (which share virtual tables with
  //   the classes for which they are primary).
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    const CXXRecordDecl *BaseDecl = 
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    // Check if this base needs a vtable. (If it's virtual, not a primary base
    // of some other class, and we haven't visited it before).
    if (I->isVirtual() && BaseDecl->isDynamicClass() && 
        !PrimaryVirtualBases.count(BaseDecl) && VBases.insert(BaseDecl)) {
      const ASTRecordLayout &MostDerivedClassLayout =
        Context.getASTRecordLayout(MostDerivedClass);
      uint64_t BaseOffset = 
        MostDerivedClassLayout.getVBaseClassOffset(BaseDecl);
      
      const ASTRecordLayout &LayoutClassLayout =
        Context.getASTRecordLayout(LayoutClass);
      uint64_t BaseOffsetInLayoutClass = 
        LayoutClassLayout.getVBaseClassOffset(BaseDecl);

      LayoutPrimaryAndSecondaryVtables(BaseSubobject(BaseDecl, BaseOffset),
                                       /*BaseIsVirtual=*/true,
                                       BaseOffsetInLayoutClass);
    }
    
    // We only need to check the base for virtual base vtables if it actually
    // has virtual bases.
    if (BaseDecl->getNumVBases())
      LayoutVtablesForVirtualBases(BaseDecl, VBases);
  }
}

/// dumpLayout - Dump the vtable layout.
void VtableBuilder::dumpLayout(llvm::raw_ostream& Out) {

  if (isBuildingConstructorVtable()) {
    Out << "Construction vtable for ('";
    Out << MostDerivedClass->getQualifiedNameAsString() << "', ";
    // FIXME: Don't use / 8 .
    Out << MostDerivedClassOffset / 8 << ") in '";
    Out << LayoutClass->getQualifiedNameAsString();
  } else {
    Out << "Vtable for '";
    Out << MostDerivedClass->getQualifiedNameAsString();
  }
  Out << "' (" << Components.size() << " entries).\n";

  // Iterate through the address points and insert them into a new map where
  // they are keyed by the index and not the base object.
  // Since an address point can be shared by multiple subobjects, we use an
  // STL multimap.
  std::multimap<uint64_t, BaseSubobject> AddressPointsByIndex;
  for (CGVtableInfo::AddressPointsMapTy::const_iterator I = 
       AddressPoints.begin(), E = AddressPoints.end(); I != E; ++I) {
    const BaseSubobject& Base = I->first;
    uint64_t Index = I->second;
    
    AddressPointsByIndex.insert(std::make_pair(Index, Base));
  }
  
  unsigned NextReturnAdjustmentIndex = 0;
  unsigned NextThisAdjustmentIndex = 0;
  for (unsigned I = 0, E = Components.size(); I != E; ++I) {
    uint64_t Index = I;

    Out << llvm::format("%4d | ", I);

    const VtableComponent &Component = Components[I];

    // Dump the component.
    switch (Component.getKind()) {

    case VtableComponent::CK_VCallOffset:
      Out << "vcall_offset (" << Component.getVCallOffset() << ")";
      break;

    case VtableComponent::CK_VBaseOffset:
      Out << "vbase_offset (" << Component.getVBaseOffset() << ")";
      break;

    case VtableComponent::CK_OffsetToTop:
      Out << "offset_to_top (" << Component.getOffsetToTop() << ")";
      break;
    
    case VtableComponent::CK_RTTI:
      Out << Component.getRTTIDecl()->getQualifiedNameAsString() << " RTTI";
      break;
    
    case VtableComponent::CK_FunctionPointer: {
      const CXXMethodDecl *MD = Component.getFunctionDecl();

      std::string Str = 
        PredefinedExpr::ComputeName(PredefinedExpr::PrettyFunctionNoVirtual, 
                                    MD);
      Out << Str;
      if (MD->isPure())
        Out << " [pure]";

      // If this function pointer has a return adjustment, dump it.
      if (NextReturnAdjustmentIndex < ReturnAdjustments.size() && 
          ReturnAdjustments[NextReturnAdjustmentIndex].first == I) {
        const ReturnAdjustment Adjustment = 
          ReturnAdjustments[NextReturnAdjustmentIndex].second;
        
        Out << "\n       [return adjustment: ";
        Out << Adjustment.NonVirtual << " non-virtual";
        
        if (Adjustment.VBaseOffsetOffset)
          Out << ", " << Adjustment.VBaseOffsetOffset << " vbase offset offset";

        Out << ']';

        NextReturnAdjustmentIndex++;
      }
      
      // If this function pointer has a 'this' pointer adjustment, dump it.
      if (NextThisAdjustmentIndex < ThisAdjustments.size() && 
          ThisAdjustments[NextThisAdjustmentIndex].first == I) {
        const ThisAdjustment Adjustment = 
          ThisAdjustments[NextThisAdjustmentIndex].second;
        
        Out << "\n       [this adjustment: ";
        Out << Adjustment.NonVirtual << " non-virtual";

        if (Adjustment.VCallOffsetOffset)
          Out << ", " << Adjustment.VCallOffsetOffset << " vcall offset offset";

        Out << ']';
        
        NextThisAdjustmentIndex++;
      }

      break;
    }

    case VtableComponent::CK_CompleteDtorPointer: 
    case VtableComponent::CK_DeletingDtorPointer: {
      bool IsComplete = 
        Component.getKind() == VtableComponent::CK_CompleteDtorPointer;
      
      const CXXDestructorDecl *DD = Component.getDestructorDecl();
      
      Out << DD->getQualifiedNameAsString();
      if (IsComplete)
        Out << "() [complete]";
      else
        Out << "() [deleting]";

      if (DD->isPure())
        Out << " [pure]";

      // If this destructor has a 'this' pointer adjustment, dump it.
      if (NextThisAdjustmentIndex < ThisAdjustments.size() && 
          ThisAdjustments[NextThisAdjustmentIndex].first == I) {
        const ThisAdjustment Adjustment = 
          ThisAdjustments[NextThisAdjustmentIndex].second;
        
        Out << "\n       [this adjustment: ";
        Out << Adjustment.NonVirtual << " non-virtual";

        if (Adjustment.VCallOffsetOffset)
          Out << ", " << Adjustment.VCallOffsetOffset << " vcall offset offset";

        Out << ']';
        
        NextThisAdjustmentIndex++;
      }


      break;
    }

    case VtableComponent::CK_UnusedFunctionPointer: {
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
        
        // FIXME: Instead of dividing by 8, we should be using CharUnits.
        Out << "       -- (" << Base.getBase()->getQualifiedNameAsString();
        Out << ", " << Base.getBaseOffset() / 8 << ") vtable address --\n";
      } else {
        uint64_t BaseOffset = 
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
          // FIXME: Instead of dividing by 8, we should be using CharUnits.
          Out << "       -- (" << *I;
          Out << ", " << BaseOffset / 8 << ") vtable address --\n";
        }
      }
    }
  }

  Out << '\n';
  
  if (!isBuildingConstructorVtable()) {
    Out << "Virtual base offset offsets for '";
    Out << MostDerivedClass->getQualifiedNameAsString() << "'.\n";
    
    // We store the virtual base class names and their offsets in a map to get
    // a stable order.
    std::map<std::string, int64_t> ClassNamesAndOffsets;

    for (VBaseOffsetOffsetsMapTy::const_iterator I = VBaseOffsetOffsets.begin(),
         E = VBaseOffsetOffsets.end(); I != E; ++I) {
      std::string ClassName = I->first->getQualifiedNameAsString();
      int64_t OffsetOffset = I->second;
      ClassNamesAndOffsets.insert(std::make_pair(ClassName, OffsetOffset));
    }

    for (std::map<std::string, int64_t>::const_iterator I =
         ClassNamesAndOffsets.begin(), E = ClassNamesAndOffsets.end(); 
         I != E; ++I)
      Out << "   " << I->first << " | " << I->second << '\n';

    Out << "\n";
  }
}
  
}

namespace {
class OldVtableBuilder {
public:
  /// Index_t - Vtable index type.
  typedef uint64_t Index_t;
  typedef std::vector<std::pair<GlobalDecl,
                                std::pair<GlobalDecl, ThunkAdjustment> > >
      SavedAdjustmentsVectorTy;
private:
  
  // VtableComponents - The components of the vtable being built.
  typedef llvm::SmallVector<llvm::Constant *, 64> VtableComponentsVectorTy;
  VtableComponentsVectorTy VtableComponents;
  
  const bool BuildVtable;

  llvm::Type *Ptr8Ty;
  
  /// MostDerivedClass - The most derived class that this vtable is being 
  /// built for.
  const CXXRecordDecl *MostDerivedClass;
  
  /// LayoutClass - The most derived class used for virtual base layout
  /// information.
  const CXXRecordDecl *LayoutClass;
  /// LayoutOffset - The offset for Class in LayoutClass.
  uint64_t LayoutOffset;
  /// BLayout - Layout for the most derived class that this vtable is being
  /// built for.
  const ASTRecordLayout &BLayout;
  llvm::SmallSet<const CXXRecordDecl *, 32> IndirectPrimary;
  llvm::SmallSet<const CXXRecordDecl *, 32> SeenVBase;
  llvm::Constant *rtti;
  llvm::LLVMContext &VMContext;
  CodeGenModule &CGM;  // Per-module state.
  
  llvm::DenseMap<const CXXMethodDecl *, Index_t> VCall;
  llvm::DenseMap<GlobalDecl, Index_t> VCallOffset;
  llvm::DenseMap<GlobalDecl, Index_t> VCallOffsetForVCall;
  // This is the offset to the nearest virtual base
  llvm::DenseMap<const CXXMethodDecl *, Index_t> NonVirtualOffset;
  llvm::DenseMap<const CXXRecordDecl *, Index_t> VBIndex;

  /// PureVirtualFunction - Points to __cxa_pure_virtual.
  llvm::Constant *PureVirtualFn;
  
  /// VtableMethods - A data structure for keeping track of methods in a vtable.
  /// Can add methods, override methods and iterate in vtable order.
  class VtableMethods {
    // MethodToIndexMap - Maps from a global decl to the index it has in the
    // Methods vector.
    llvm::DenseMap<GlobalDecl, uint64_t> MethodToIndexMap;

    /// Methods - The methods, in vtable order.
    typedef llvm::SmallVector<GlobalDecl, 16> MethodsVectorTy;
    MethodsVectorTy Methods;
    MethodsVectorTy OrigMethods;

  public:
    /// AddMethod - Add a method to the vtable methods.
    void AddMethod(GlobalDecl GD) {
      assert(!MethodToIndexMap.count(GD) && 
             "Method has already been added!");
      
      MethodToIndexMap[GD] = Methods.size();
      Methods.push_back(GD);
      OrigMethods.push_back(GD);
    }
    
    /// OverrideMethod - Replace a method with another.
    void OverrideMethod(GlobalDecl OverriddenGD, GlobalDecl GD) {
      llvm::DenseMap<GlobalDecl, uint64_t>::iterator i 
        = MethodToIndexMap.find(OverriddenGD);
      assert(i != MethodToIndexMap.end() && "Did not find entry!");

      // Get the index of the old decl.
      uint64_t Index = i->second;
      
      // Replace the old decl with the new decl.
      Methods[Index] = GD;

      // And add the new.
      MethodToIndexMap[GD] = Index;
    }

    /// getIndex - Gives the index of a passed in GlobalDecl. Returns false if
    /// the index couldn't be found.
    bool getIndex(GlobalDecl GD, uint64_t &Index) const {
      llvm::DenseMap<GlobalDecl, uint64_t>::const_iterator i 
        = MethodToIndexMap.find(GD);

      if (i == MethodToIndexMap.end())
        return false;
      
      Index = i->second;
      return true;
    }

    GlobalDecl getOrigMethod(uint64_t Index) const {
      return OrigMethods[Index];
    }

    MethodsVectorTy::size_type size() const {
      return Methods.size();
    }

    void clear() {
      MethodToIndexMap.clear();
      Methods.clear();
      OrigMethods.clear();
    }
    
    GlobalDecl operator[](uint64_t Index) const {
      return Methods[Index];
    }
  };
  
  /// Methods - The vtable methods we're currently building.
  VtableMethods Methods;
  
  /// ThisAdjustments - For a given index in the vtable, contains the 'this'
  /// pointer adjustment needed for a method.
  typedef llvm::DenseMap<uint64_t, ThunkAdjustment> ThisAdjustmentsMapTy;
  ThisAdjustmentsMapTy ThisAdjustments;

  SavedAdjustmentsVectorTy SavedAdjustments;

  /// BaseReturnTypes - Contains the base return types of methods who have been
  /// overridden with methods whose return types require adjustment. Used for
  /// generating covariant thunk information.
  typedef llvm::DenseMap<uint64_t, CanQualType> BaseReturnTypesMapTy;
  BaseReturnTypesMapTy BaseReturnTypes;
  
  std::vector<Index_t> VCalls;

  typedef std::pair<const CXXRecordDecl *, uint64_t> CtorVtable_t;
  // subAddressPoints - Used to hold the AddressPoints (offsets) into the built
  // vtable for use in computing the initializers for the VTT.
  llvm::DenseMap<CtorVtable_t, int64_t> &subAddressPoints;

  /// AddressPoints - Address points for this vtable.
  CGVtableInfo::AddressPointsMapTy& AddressPoints;
  
  typedef CXXRecordDecl::method_iterator method_iter;
  const uint32_t LLVMPointerWidth;
  Index_t extra;
  typedef std::vector<std::pair<const CXXRecordDecl *, int64_t> > Path_t;
  static llvm::DenseMap<CtorVtable_t, int64_t>&
  AllocAddressPoint(CodeGenModule &cgm, const CXXRecordDecl *l,
                    const CXXRecordDecl *c) {
    CGVtableInfo::AddrMap_t *&oref = cgm.getVtableInfo().AddressPoints[l];
    if (oref == 0)
      oref = new CGVtableInfo::AddrMap_t;

    llvm::DenseMap<CtorVtable_t, int64_t> *&ref = (*oref)[c];
    if (ref == 0)
      ref = new llvm::DenseMap<CtorVtable_t, int64_t>;
    return *ref;
  }
  
  bool DclIsSame(const FunctionDecl *New, const FunctionDecl *Old) {
    FunctionTemplateDecl *OldTemplate = Old->getDescribedFunctionTemplate();
    FunctionTemplateDecl *NewTemplate = New->getDescribedFunctionTemplate();

    // C++ [temp.fct]p2:
    //   A function template can be overloaded with other function templates
    //   and with normal (non-template) functions.
    if ((OldTemplate == 0) != (NewTemplate == 0))
      return false;

    // Is the function New an overload of the function Old?
    QualType OldQType = CGM.getContext().getCanonicalType(Old->getType());
    QualType NewQType = CGM.getContext().getCanonicalType(New->getType());

    // Compare the signatures (C++ 1.3.10) of the two functions to
    // determine whether they are overloads. If we find any mismatch
    // in the signature, they are overloads.

    // If either of these functions is a K&R-style function (no
    // prototype), then we consider them to have matching signatures.
    if (isa<FunctionNoProtoType>(OldQType.getTypePtr()) ||
        isa<FunctionNoProtoType>(NewQType.getTypePtr()))
      return true;

    FunctionProtoType* OldType = cast<FunctionProtoType>(OldQType);
    FunctionProtoType* NewType = cast<FunctionProtoType>(NewQType);

    // The signature of a function includes the types of its
    // parameters (C++ 1.3.10), which includes the presence or absence
    // of the ellipsis; see C++ DR 357).
    if (OldQType != NewQType &&
        (OldType->getNumArgs() != NewType->getNumArgs() ||
         OldType->isVariadic() != NewType->isVariadic() ||
         !std::equal(OldType->arg_type_begin(), OldType->arg_type_end(),
                     NewType->arg_type_begin())))
      return false;

#if 0
    // C++ [temp.over.link]p4:
    //   The signature of a function template consists of its function
    //   signature, its return type and its template parameter list. The names
    //   of the template parameters are significant only for establishing the
    //   relationship between the template parameters and the rest of the
    //   signature.
    //
    // We check the return type and template parameter lists for function
    // templates first; the remaining checks follow.
    if (NewTemplate &&
        (!TemplateParameterListsAreEqual(NewTemplate->getTemplateParameters(),
                                         OldTemplate->getTemplateParameters(),
                                         TPL_TemplateMatch) ||
         OldType->getResultType() != NewType->getResultType()))
      return false;
#endif

    // If the function is a class member, its signature includes the
    // cv-qualifiers (if any) on the function itself.
    //
    // As part of this, also check whether one of the member functions
    // is static, in which case they are not overloads (C++
    // 13.1p2). While not part of the definition of the signature,
    // this check is important to determine whether these functions
    // can be overloaded.
    const CXXMethodDecl* OldMethod = dyn_cast<CXXMethodDecl>(Old);
    const CXXMethodDecl* NewMethod = dyn_cast<CXXMethodDecl>(New);
    if (OldMethod && NewMethod &&
        !OldMethod->isStatic() && !NewMethod->isStatic() &&
        OldMethod->getTypeQualifiers() != NewMethod->getTypeQualifiers())
      return false;
  
    // The signatures match; this is not an overload.
    return true;
  }

  typedef llvm::DenseMap<const CXXMethodDecl *, const CXXMethodDecl*>
    ForwardUnique_t;
  ForwardUnique_t ForwardUnique;
  llvm::DenseMap<const CXXMethodDecl*, const CXXMethodDecl*> UniqueOverrider;

  void BuildUniqueOverrider(const CXXMethodDecl *U, const CXXMethodDecl *MD) {
    const CXXMethodDecl *PrevU = UniqueOverrider[MD];
    assert(U && "no unique overrider");
    if (PrevU == U)
      return;
    if (PrevU != U && PrevU != 0) {
      // If already set, note the two sets as the same
      if (0)
        printf("%s::%s same as %s::%s\n",
               PrevU->getParent()->getNameAsString().c_str(),
               PrevU->getNameAsString().c_str(),
               U->getParent()->getNameAsString().c_str(),
               U->getNameAsString().c_str());
      ForwardUnique[PrevU] = U;
      return;
    }

    // Not set, set it now
    if (0)
      printf("marking %s::%s %p override as %s::%s\n",
             MD->getParent()->getNameAsString().c_str(),
             MD->getNameAsString().c_str(),
             (void*)MD,
             U->getParent()->getNameAsString().c_str(),
             U->getNameAsString().c_str());
    UniqueOverrider[MD] = U;

    for (CXXMethodDecl::method_iterator mi = MD->begin_overridden_methods(),
           me = MD->end_overridden_methods(); mi != me; ++mi) {
      BuildUniqueOverrider(U, *mi);
    }
  }

  void BuildUniqueOverriders(const CXXRecordDecl *RD) {
    if (0) printf("walking %s\n", RD->getNameAsCString());
    for (CXXRecordDecl::method_iterator i = RD->method_begin(),
           e = RD->method_end(); i != e; ++i) {
      const CXXMethodDecl *MD = *i;
      if (!MD->isVirtual())
        continue;

      if (UniqueOverrider[MD] == 0) {
        // Only set this, if it hasn't been set yet.
        BuildUniqueOverrider(MD, MD);
        if (0)
          printf("top set is %s::%s %p\n",
                  MD->getParent()->getNameAsString().c_str(),
                  MD->getNameAsString().c_str(),
                  (void*)MD);
        ForwardUnique[MD] = MD;
      }
    }
    for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
           e = RD->bases_end(); i != e; ++i) {
      const CXXRecordDecl *Base =
        cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
      BuildUniqueOverriders(Base);
    }
  }

  static int DclCmp(const void *p1, const void *p2) {
    const CXXMethodDecl *MD1 = *(const CXXMethodDecl *const *)p1;
    const CXXMethodDecl *MD2 = *(const CXXMethodDecl *const *)p2;

    return (DeclarationName::compare(MD1->getDeclName(), MD2->getDeclName()));
  }
  
  void MergeForwarding() {
    typedef llvm::SmallVector<const CXXMethodDecl *, 100>  A_t;
    A_t A;
    for (ForwardUnique_t::iterator I = ForwardUnique.begin(),
           E = ForwardUnique.end(); I != E; ++I) {
      if (I->first == I->second)
        // Only add the roots of all trees
        A.push_back(I->first);
    }
    llvm::array_pod_sort(A.begin(), A.end(), DclCmp);
    for (A_t::iterator I = A.begin(),
           E = A.end(); I != E; ++I) {
      A_t::iterator J = I;
      while (++J != E  && DclCmp(I, J) == 0)
        if (DclIsSame(*I, *J)) {
          if (0) printf("connecting %s\n", (*I)->getNameAsString().c_str());
          ForwardUnique[*J] = *I;
        }
    }
  }

  const CXXMethodDecl *getUnique(const CXXMethodDecl *MD) {
    const CXXMethodDecl *U = UniqueOverrider[MD];
    assert(U && "unique overrider not found");
    while (ForwardUnique.count(U)) {
      const CXXMethodDecl *NU = ForwardUnique[U];
      if (NU == U) break;
      U = NU;
    }
    return U;
  }

  GlobalDecl getUnique(GlobalDecl GD) {
    const CXXMethodDecl *Unique = getUnique(cast<CXXMethodDecl>(GD.getDecl()));
    
    if (const CXXConstructorDecl *CD = dyn_cast<CXXConstructorDecl>(Unique))
      return GlobalDecl(CD, GD.getCtorType());
    
    if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(Unique))
      return GlobalDecl(DD, GD.getDtorType());
    
    return Unique;
  }

  /// getPureVirtualFn - Return the __cxa_pure_virtual function.
  llvm::Constant* getPureVirtualFn() {
    if (!PureVirtualFn) {
      const llvm::FunctionType *Ty = 
        llvm::FunctionType::get(llvm::Type::getVoidTy(VMContext), 
                                /*isVarArg=*/false);
      PureVirtualFn = wrap(CGM.CreateRuntimeFunction(Ty, "__cxa_pure_virtual"));
    }
    
    return PureVirtualFn;
  }
  
public:
  OldVtableBuilder(const CXXRecordDecl *MostDerivedClass,
                const CXXRecordDecl *l, uint64_t lo, CodeGenModule &cgm,
                bool build, CGVtableInfo::AddressPointsMapTy& AddressPoints)
    : BuildVtable(build), MostDerivedClass(MostDerivedClass), LayoutClass(l),
      LayoutOffset(lo), BLayout(cgm.getContext().getASTRecordLayout(l)),
      rtti(0), VMContext(cgm.getModule().getContext()),CGM(cgm),
      PureVirtualFn(0),
      subAddressPoints(AllocAddressPoint(cgm, l, MostDerivedClass)),
      AddressPoints(AddressPoints),
      LLVMPointerWidth(cgm.getContext().Target.getPointerWidth(0))
      {
    Ptr8Ty = llvm::PointerType::get(llvm::Type::getInt8Ty(VMContext), 0);
    if (BuildVtable) {
      QualType ClassType = CGM.getContext().getTagDeclType(MostDerivedClass);
      rtti = CGM.GetAddrOfRTTIDescriptor(ClassType);
    }
    BuildUniqueOverriders(MostDerivedClass);
    MergeForwarding();
  }

  // getVtableComponents - Returns a reference to the vtable components.
  const VtableComponentsVectorTy &getVtableComponents() const {
    return VtableComponents;
  }
  
  llvm::DenseMap<const CXXRecordDecl *, uint64_t> &getVBIndex()
    { return VBIndex; }

  SavedAdjustmentsVectorTy &getSavedAdjustments()
    { return SavedAdjustments; }

  llvm::Constant *wrap(Index_t i) {
    llvm::Constant *m;
    m = llvm::ConstantInt::get(llvm::Type::getInt64Ty(VMContext), i);
    return llvm::ConstantExpr::getIntToPtr(m, Ptr8Ty);
  }

  llvm::Constant *wrap(llvm::Constant *m) {
    return llvm::ConstantExpr::getBitCast(m, Ptr8Ty);
  }

//#define D1(x)
#define D1(X) do { if (getenv("CLANG_VTABLE_DEBUG")) { X; } } while (0)

  void GenerateVBaseOffsets(const CXXRecordDecl *RD, uint64_t Offset,
                            bool updateVBIndex, Index_t current_vbindex) {
    for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
           e = RD->bases_end(); i != e; ++i) {
      const CXXRecordDecl *Base =
        cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
      Index_t next_vbindex = current_vbindex;
      if (i->isVirtual() && !SeenVBase.count(Base)) {
        SeenVBase.insert(Base);
        if (updateVBIndex) {
          next_vbindex = (ssize_t)(-(VCalls.size()*LLVMPointerWidth/8)
                                   - 3*LLVMPointerWidth/8);
          VBIndex[Base] = next_vbindex;
        }
        int64_t BaseOffset = -(Offset/8) + BLayout.getVBaseClassOffset(Base)/8;
        VCalls.push_back((0?700:0) + BaseOffset);
        D1(printf("  vbase for %s at %d delta %d most derived %s\n",
                  Base->getNameAsCString(),
                  (int)-VCalls.size()-3, (int)BaseOffset,
                  MostDerivedClass->getNameAsCString()));
      }
      // We also record offsets for non-virtual bases to closest enclosing
      // virtual base.  We do this so that we don't have to search
      // for the nearst virtual base class when generating thunks.
      if (updateVBIndex && VBIndex.count(Base) == 0)
        VBIndex[Base] = next_vbindex;
      GenerateVBaseOffsets(Base, Offset, updateVBIndex, next_vbindex);
    }
  }

  void StartNewTable() {
    SeenVBase.clear();
  }

  Index_t getNVOffset_1(const CXXRecordDecl *D, const CXXRecordDecl *B,
    Index_t Offset = 0) {

    if (B == D)
      return Offset;

    const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(D);
    for (CXXRecordDecl::base_class_const_iterator i = D->bases_begin(),
           e = D->bases_end(); i != e; ++i) {
      const CXXRecordDecl *Base =
        cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
      int64_t BaseOffset = 0;
      if (!i->isVirtual())
        BaseOffset = Offset + Layout.getBaseClassOffset(Base);
      int64_t o = getNVOffset_1(Base, B, BaseOffset);
      if (o >= 0)
        return o;
    }

    return -1;
  }

  /// getNVOffset - Returns the non-virtual offset for the given (B) base of the
  /// derived class D.
  Index_t getNVOffset(QualType qB, QualType qD) {
    qD = qD->getPointeeType();
    qB = qB->getPointeeType();
    CXXRecordDecl *D = cast<CXXRecordDecl>(qD->getAs<RecordType>()->getDecl());
    CXXRecordDecl *B = cast<CXXRecordDecl>(qB->getAs<RecordType>()->getDecl());
    int64_t o = getNVOffset_1(D, B);
    if (o >= 0)
      return o;

    assert(false && "FIXME: non-virtual base not found");
    return 0;
  }

  /// getVbaseOffset - Returns the index into the vtable for the virtual base
  /// offset for the given (B) virtual base of the derived class D.
  Index_t getVbaseOffset(QualType qB, QualType qD) {
    qD = qD->getPointeeType();
    qB = qB->getPointeeType();
    CXXRecordDecl *D = cast<CXXRecordDecl>(qD->getAs<RecordType>()->getDecl());
    CXXRecordDecl *B = cast<CXXRecordDecl>(qB->getAs<RecordType>()->getDecl());
    if (D != MostDerivedClass)
      return CGM.getVtableInfo().getVirtualBaseOffsetIndex(D, B);
    llvm::DenseMap<const CXXRecordDecl *, Index_t>::iterator i;
    i = VBIndex.find(B);
    if (i != VBIndex.end())
      return i->second;

    assert(false && "FIXME: Base not found");
    return 0;
  }

  bool OverrideMethod(GlobalDecl GD, bool MorallyVirtual,
                      Index_t OverrideOffset, Index_t Offset,
                      int64_t CurrentVBaseOffset);

  /// AppendMethods - Append the current methods to the vtable.
  void AppendMethodsToVtable();
  
  llvm::Constant *WrapAddrOf(GlobalDecl GD) {
    const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());

    const llvm::Type *Ty = CGM.getTypes().GetFunctionTypeForVtable(MD);

    return wrap(CGM.GetAddrOfFunction(GD, Ty));
  }

  void OverrideMethods(Path_t *Path, bool MorallyVirtual, int64_t Offset,
                       int64_t CurrentVBaseOffset) {
    for (Path_t::reverse_iterator i = Path->rbegin(),
           e = Path->rend(); i != e; ++i) {
      const CXXRecordDecl *RD = i->first;
      int64_t OverrideOffset = i->second;
      for (method_iter mi = RD->method_begin(), me = RD->method_end(); mi != me;
           ++mi) {
        const CXXMethodDecl *MD = *mi;

        if (!MD->isVirtual())
          continue;

        if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(MD)) {
          // Override both the complete and the deleting destructor.
          GlobalDecl CompDtor(DD, Dtor_Complete);
          OverrideMethod(CompDtor, MorallyVirtual, OverrideOffset, Offset,
                         CurrentVBaseOffset);

          GlobalDecl DeletingDtor(DD, Dtor_Deleting);
          OverrideMethod(DeletingDtor, MorallyVirtual, OverrideOffset, Offset,
                         CurrentVBaseOffset);
        } else {
          OverrideMethod(MD, MorallyVirtual, OverrideOffset, Offset,
                         CurrentVBaseOffset);
        }
      }
    }
  }

  void AddMethod(const GlobalDecl GD, bool MorallyVirtual, Index_t Offset,
                 int64_t CurrentVBaseOffset) {
    // If we can find a previously allocated slot for this, reuse it.
    if (OverrideMethod(GD, MorallyVirtual, Offset, Offset,
                       CurrentVBaseOffset))
      return;

    D1(printf("  vfn for %s at %d\n",
              dyn_cast<CXXMethodDecl>(GD.getDecl())->getNameAsString().c_str(),
              (int)Methods.size()));

    // We didn't find an entry in the vtable that we could use, add a new
    // entry.
    Methods.AddMethod(GD);

    VCallOffset[GD] = Offset/8 - CurrentVBaseOffset/8;

    if (MorallyVirtual) {
      GlobalDecl UGD = getUnique(GD);
      const CXXMethodDecl *UMD = cast<CXXMethodDecl>(UGD.getDecl());
  
      assert(UMD && "final overrider not found");

      Index_t &idx = VCall[UMD];
      // Allocate the first one, after that, we reuse the previous one.
      if (idx == 0) {
        VCallOffsetForVCall[UGD] = Offset/8;
        NonVirtualOffset[UMD] = Offset/8 - CurrentVBaseOffset/8;
        idx = VCalls.size()+1;
        VCalls.push_back(Offset/8 - CurrentVBaseOffset/8);
        D1(printf("  vcall for %s at %d with delta %d\n",
                  dyn_cast<CXXMethodDecl>(GD.getDecl())->getNameAsString().c_str(),
                  (int)-VCalls.size()-3, (int)VCalls[idx-1]));
      }
    }
  }

  void AddMethods(const CXXRecordDecl *RD, bool MorallyVirtual,
                  Index_t Offset, int64_t CurrentVBaseOffset) {
    for (method_iter mi = RD->method_begin(), me = RD->method_end(); mi != me;
         ++mi) {
      const CXXMethodDecl *MD = *mi;
      if (!MD->isVirtual())
        continue;
      
      if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(MD)) {
        // For destructors, add both the complete and the deleting destructor
        // to the vtable.
        AddMethod(GlobalDecl(DD, Dtor_Complete), MorallyVirtual, Offset, 
                  CurrentVBaseOffset);
        AddMethod(GlobalDecl(DD, Dtor_Deleting), MorallyVirtual, Offset, 
                  CurrentVBaseOffset);
      } else
        AddMethod(MD, MorallyVirtual, Offset, CurrentVBaseOffset);
    }
  }

  void NonVirtualBases(const CXXRecordDecl *RD, const ASTRecordLayout &Layout,
                       const CXXRecordDecl *PrimaryBase,
                       bool PrimaryBaseWasVirtual, bool MorallyVirtual,
                       int64_t Offset, int64_t CurrentVBaseOffset,
                       Path_t *Path) {
    Path->push_back(std::make_pair(RD, Offset));
    for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
           e = RD->bases_end(); i != e; ++i) {
      if (i->isVirtual())
        continue;
      const CXXRecordDecl *Base =
        cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
      uint64_t o = Offset + Layout.getBaseClassOffset(Base);
      StartNewTable();
      GenerateVtableForBase(Base, o, MorallyVirtual, false,
                            true, Base == PrimaryBase && !PrimaryBaseWasVirtual,
                            CurrentVBaseOffset, Path);
    }
    Path->pop_back();
  }

// #define D(X) do { X; } while (0)
#define D(X)

  void insertVCalls(int InsertionPoint) {
    D1(printf("============= combining vbase/vcall\n"));
    D(VCalls.insert(VCalls.begin(), 673));
    D(VCalls.push_back(672));

    VtableComponents.insert(VtableComponents.begin() + InsertionPoint, 
                            VCalls.size(), 0);
    if (BuildVtable) {
      // The vcalls come first...
      for (std::vector<Index_t>::reverse_iterator i = VCalls.rbegin(),
             e = VCalls.rend();
           i != e; ++i)
        VtableComponents[InsertionPoint++] = wrap((0?600:0) + *i);
    }
    VCalls.clear();
    VCall.clear();
    VCallOffsetForVCall.clear();
    VCallOffset.clear();
    NonVirtualOffset.clear();
  }

  void AddAddressPoints(const CXXRecordDecl *RD, uint64_t Offset,
                       Index_t AddressPoint) {
    D1(printf("XXX address point for %s in %s layout %s at offset %d is %d\n",
              RD->getNameAsCString(), MostDerivedClass->getNameAsCString(),
              LayoutClass->getNameAsCString(), (int)Offset, (int)AddressPoint));
    subAddressPoints[std::make_pair(RD, Offset)] = AddressPoint;
    AddressPoints[BaseSubobject(RD, Offset)] = AddressPoint;

    // Now also add the address point for all our primary bases.
    while (1) {
      const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);
      RD = Layout.getPrimaryBase();
      const bool PrimaryBaseWasVirtual = Layout.getPrimaryBaseWasVirtual();
      // FIXME: Double check this.
      if (RD == 0)
        break;
      if (PrimaryBaseWasVirtual &&
          BLayout.getVBaseClassOffset(RD) != Offset)
        break;
      D1(printf("XXX address point for %s in %s layout %s at offset %d is %d\n",
                RD->getNameAsCString(), MostDerivedClass->getNameAsCString(),
                LayoutClass->getNameAsCString(), (int)Offset, (int)AddressPoint));
      subAddressPoints[std::make_pair(RD, Offset)] = AddressPoint;
      AddressPoints[BaseSubobject(RD, Offset)] = AddressPoint;
    }
  }


  void FinishGenerateVtable(const CXXRecordDecl *RD,
                            const ASTRecordLayout &Layout,
                            const CXXRecordDecl *PrimaryBase,
                            bool ForNPNVBases, bool WasPrimaryBase,
                            bool PrimaryBaseWasVirtual,
                            bool MorallyVirtual, int64_t Offset,
                            bool ForVirtualBase, int64_t CurrentVBaseOffset,
                            Path_t *Path) {
    bool alloc = false;
    if (Path == 0) {
      alloc = true;
      Path = new Path_t;
    }

    StartNewTable();
    extra = 0;
    Index_t AddressPoint = 0;
    int VCallInsertionPoint = 0;
    if (!ForNPNVBases || !WasPrimaryBase) {
      bool DeferVCalls = MorallyVirtual || ForVirtualBase;
      VCallInsertionPoint = VtableComponents.size();
      if (!DeferVCalls) {
        insertVCalls(VCallInsertionPoint);
      } else
        // FIXME: just for extra, or for all uses of VCalls.size post this?
        extra = -VCalls.size();

      // Add the offset to top.
      VtableComponents.push_back(BuildVtable ? wrap(-((Offset-LayoutOffset)/8)) : 0);
    
      // Add the RTTI information.
      VtableComponents.push_back(rtti);
    
      AddressPoint = VtableComponents.size();

      AppendMethodsToVtable();
    }

    // and then the non-virtual bases.
    NonVirtualBases(RD, Layout, PrimaryBase, PrimaryBaseWasVirtual,
                    MorallyVirtual, Offset, CurrentVBaseOffset, Path);

    if (ForVirtualBase) {
      // FIXME: We're adding to VCalls in callers, we need to do the overrides
      // in the inner part, so that we know the complete set of vcalls during
      // the build and don't have to insert into methods.  Saving out the
      // AddressPoint here, would need to be fixed, if we didn't do that.  Also
      // retroactively adding vcalls for overrides later wind up in the wrong
      // place, the vcall slot has to be alloted during the walk of the base
      // when the function is first introduces.
      AddressPoint += VCalls.size();
      insertVCalls(VCallInsertionPoint);
    }
    
    if (!ForNPNVBases || !WasPrimaryBase)
      AddAddressPoints(RD, Offset, AddressPoint);

    if (alloc) {
      delete Path;
    }
  }

  void Primaries(const CXXRecordDecl *RD, bool MorallyVirtual, int64_t Offset,
                 bool updateVBIndex, Index_t current_vbindex,
                 int64_t CurrentVBaseOffset) {
    if (!RD->isDynamicClass())
      return;

    const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);
    const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();
    const bool PrimaryBaseWasVirtual = Layout.getPrimaryBaseWasVirtual();

    // vtables are composed from the chain of primaries.
    if (PrimaryBase && !PrimaryBaseWasVirtual) {
      D1(printf(" doing primaries for %s most derived %s\n",
                RD->getNameAsCString(), MostDerivedClass->getNameAsCString()));
      Primaries(PrimaryBase, PrimaryBaseWasVirtual|MorallyVirtual, Offset,
                updateVBIndex, current_vbindex, CurrentVBaseOffset);
    }

    D1(printf(" doing vcall entries for %s most derived %s\n",
              RD->getNameAsCString(), MostDerivedClass->getNameAsCString()));

    // And add the virtuals for the class to the primary vtable.
    AddMethods(RD, MorallyVirtual, Offset, CurrentVBaseOffset);
  }

  void VBPrimaries(const CXXRecordDecl *RD, bool MorallyVirtual, int64_t Offset,
                   bool updateVBIndex, Index_t current_vbindex,
                   bool RDisVirtualBase, int64_t CurrentVBaseOffset,
                   bool bottom) {
    if (!RD->isDynamicClass())
      return;

    const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);
    const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();
    const bool PrimaryBaseWasVirtual = Layout.getPrimaryBaseWasVirtual();

    // vtables are composed from the chain of primaries.
    if (PrimaryBase) {
      int BaseCurrentVBaseOffset = CurrentVBaseOffset;
      if (PrimaryBaseWasVirtual) {
        IndirectPrimary.insert(PrimaryBase);
        BaseCurrentVBaseOffset = BLayout.getVBaseClassOffset(PrimaryBase);
      }

      D1(printf(" doing primaries for %s most derived %s\n",
                RD->getNameAsCString(), MostDerivedClass->getNameAsCString()));
      
      VBPrimaries(PrimaryBase, PrimaryBaseWasVirtual|MorallyVirtual, Offset,
                  updateVBIndex, current_vbindex, PrimaryBaseWasVirtual,
                  BaseCurrentVBaseOffset, false);
    }

    D1(printf(" doing vbase entries for %s most derived %s\n",
              RD->getNameAsCString(), MostDerivedClass->getNameAsCString()));
    GenerateVBaseOffsets(RD, Offset, updateVBIndex, current_vbindex);

    if (RDisVirtualBase || bottom) {
      Primaries(RD, MorallyVirtual, Offset, updateVBIndex, current_vbindex,
                CurrentVBaseOffset);
    }
  }

  void GenerateVtableForBase(const CXXRecordDecl *RD, int64_t Offset = 0,
                             bool MorallyVirtual = false, 
                             bool ForVirtualBase = false,
                             bool ForNPNVBases = false,
                             bool WasPrimaryBase = true,
                             int CurrentVBaseOffset = 0,
                             Path_t *Path = 0) {
    if (!RD->isDynamicClass())
      return;

    // Construction vtable don't need parts that have no virtual bases and
    // aren't morally virtual.
    if ((LayoutClass != MostDerivedClass) && 
        RD->getNumVBases() == 0 && !MorallyVirtual)
      return;

    const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);
    const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();
    const bool PrimaryBaseWasVirtual = Layout.getPrimaryBaseWasVirtual();

    extra = 0;
    D1(printf("building entries for base %s most derived %s\n",
              RD->getNameAsCString(), MostDerivedClass->getNameAsCString()));

    if (ForVirtualBase)
      extra = VCalls.size();

    if (!ForNPNVBases || !WasPrimaryBase) {
      VBPrimaries(RD, MorallyVirtual, Offset, !ForVirtualBase, 0,
                  ForVirtualBase, CurrentVBaseOffset, true);

      if (Path)
        OverrideMethods(Path, MorallyVirtual, Offset, CurrentVBaseOffset);
    }

    FinishGenerateVtable(RD, Layout, PrimaryBase, ForNPNVBases, WasPrimaryBase,
                         PrimaryBaseWasVirtual, MorallyVirtual, Offset,
                         ForVirtualBase, CurrentVBaseOffset, Path);
  }

  void GenerateVtableForVBases(const CXXRecordDecl *RD,
                               int64_t Offset = 0,
                               Path_t *Path = 0) {
    bool alloc = false;
    if (Path == 0) {
      alloc = true;
      Path = new Path_t;
    }
    // FIXME: We also need to override using all paths to a virtual base,
    // right now, we just process the first path
    Path->push_back(std::make_pair(RD, Offset));
    for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
           e = RD->bases_end(); i != e; ++i) {
      const CXXRecordDecl *Base =
        cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
      if (i->isVirtual() && !IndirectPrimary.count(Base)) {
        // Mark it so we don't output it twice.
        IndirectPrimary.insert(Base);
        StartNewTable();
        VCall.clear();
        int64_t BaseOffset = BLayout.getVBaseClassOffset(Base);
        int64_t CurrentVBaseOffset = BaseOffset;
        D1(printf("vtable %s virtual base %s\n",
                  MostDerivedClass->getNameAsCString(), Base->getNameAsCString()));
        GenerateVtableForBase(Base, BaseOffset, true, true, false,
                              true, CurrentVBaseOffset, Path);
      }
      int64_t BaseOffset;
      if (i->isVirtual())
        BaseOffset = BLayout.getVBaseClassOffset(Base);
      else {
        const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);
        BaseOffset = Offset + Layout.getBaseClassOffset(Base);
      }
        
      if (Base->getNumVBases()) {
        GenerateVtableForVBases(Base, BaseOffset, Path);
      }
    }
    Path->pop_back();
    if (alloc)
      delete Path;
  }
};
} // end anonymous namespace

bool OldVtableBuilder::OverrideMethod(GlobalDecl GD, bool MorallyVirtual,
                                   Index_t OverrideOffset, Index_t Offset,
                                   int64_t CurrentVBaseOffset) {
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());

  const bool isPure = MD->isPure();
  
  // FIXME: Should OverrideOffset's be Offset?

  for (CXXMethodDecl::method_iterator mi = MD->begin_overridden_methods(),
       e = MD->end_overridden_methods(); mi != e; ++mi) {
    GlobalDecl OGD;
    GlobalDecl OGD2;
    
    const CXXMethodDecl *OMD = *mi;
    if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(OMD))
      OGD = GlobalDecl(DD, GD.getDtorType());
    else
      OGD = OMD;

    // Check whether this is the method being overridden in this section of
    // the vtable.
    uint64_t Index;
    if (!Methods.getIndex(OGD, Index))
      continue;

    OGD2 = OGD;

    // Get the original method, which we should be computing thunks, etc,
    // against.
    OGD = Methods.getOrigMethod(Index);
    OMD = cast<CXXMethodDecl>(OGD.getDecl());

    QualType ReturnType = 
      MD->getType()->getAs<FunctionType>()->getResultType();
    QualType OverriddenReturnType = 
      OMD->getType()->getAs<FunctionType>()->getResultType();
    
    // Check if we need a return type adjustment.
    if (!ComputeReturnAdjustmentBaseOffset(CGM.getContext(), MD, 
                                           OMD).isEmpty()) {
      CanQualType &BaseReturnType = BaseReturnTypes[Index];

      // Insert the base return type.
      if (BaseReturnType.isNull())
        BaseReturnType =
          CGM.getContext().getCanonicalType(OverriddenReturnType);
    }

    Methods.OverrideMethod(OGD, GD);

    GlobalDecl UGD = getUnique(GD);
    const CXXMethodDecl *UMD = cast<CXXMethodDecl>(UGD.getDecl());
    assert(UGD.getDecl() && "unique overrider not found");
    assert(UGD == getUnique(OGD) && "unique overrider not unique");

    ThisAdjustments.erase(Index);
    if (MorallyVirtual || VCall.count(UMD)) {

      Index_t &idx = VCall[UMD];
      if (idx == 0) {
        VCallOffset[GD] = VCallOffset[OGD];
        // NonVirtualOffset[UMD] = CurrentVBaseOffset/8 - OverrideOffset/8;
        NonVirtualOffset[UMD] = VCallOffset[OGD];
        VCallOffsetForVCall[UMD] = OverrideOffset/8;
        idx = VCalls.size()+1;
        VCalls.push_back(OverrideOffset/8 - CurrentVBaseOffset/8);
        D1(printf("  vcall for %s at %d with delta %d most derived %s\n",
                  MD->getNameAsString().c_str(), (int)-idx-3,
                  (int)VCalls[idx-1], MostDerivedClass->getNameAsCString()));
      } else {
        VCallOffset[GD] = NonVirtualOffset[UMD];
        VCalls[idx-1] = -VCallOffsetForVCall[UGD] + OverrideOffset/8;
        D1(printf("  vcall patch for %s at %d with delta %d most derived %s\n",
                  MD->getNameAsString().c_str(), (int)-idx-3,
                  (int)VCalls[idx-1], MostDerivedClass->getNameAsCString()));
      }
      int64_t NonVirtualAdjustment = -VCallOffset[OGD];
      QualType DerivedType = MD->getThisType(CGM.getContext());
      QualType BaseType = cast<const CXXMethodDecl>(OGD.getDecl())->getThisType(CGM.getContext());
      int64_t NonVirtualAdjustment2 = -(getNVOffset(BaseType, DerivedType)/8);
      if (NonVirtualAdjustment2 != NonVirtualAdjustment) {
        NonVirtualAdjustment = NonVirtualAdjustment2;
      }
      int64_t VirtualAdjustment = 
        -((idx + extra + 2) * LLVMPointerWidth / 8);
      
      // Optimize out virtual adjustments of 0.
      if (VCalls[idx-1] == 0)
        VirtualAdjustment = 0;
      
      ThunkAdjustment ThisAdjustment(NonVirtualAdjustment,
                                      VirtualAdjustment);

      if (!isPure && !ThisAdjustment.isEmpty()) {
        ThisAdjustments[Index] = ThisAdjustment;
        SavedAdjustments.push_back(
            std::make_pair(GD, std::make_pair(OGD, ThisAdjustment)));
      }
      return true;
    }

    VCallOffset[GD] = VCallOffset[OGD2] - OverrideOffset/8;

    int64_t NonVirtualAdjustment = -VCallOffset[GD];
    QualType DerivedType = MD->getThisType(CGM.getContext());
    QualType BaseType = cast<const CXXMethodDecl>(OGD.getDecl())->getThisType(CGM.getContext());
    int64_t NonVirtualAdjustment2 = -(getNVOffset(BaseType, DerivedType)/8);
    if (NonVirtualAdjustment2 != NonVirtualAdjustment) {
      NonVirtualAdjustment = NonVirtualAdjustment2;
    }
      
    if (NonVirtualAdjustment) {
      ThunkAdjustment ThisAdjustment(NonVirtualAdjustment, 0);
      
      if (!isPure) {
        ThisAdjustments[Index] = ThisAdjustment;
        SavedAdjustments.push_back(
            std::make_pair(GD, std::make_pair(OGD, ThisAdjustment)));
      }
    }
    return true;
  }

  return false;
}

void OldVtableBuilder::AppendMethodsToVtable() {
  if (!BuildVtable) {
    VtableComponents.insert(VtableComponents.end(), Methods.size(), 
                            (llvm::Constant *)0);
    ThisAdjustments.clear();
    BaseReturnTypes.clear();
    Methods.clear();
    return;
  }

  // Reserve room in the vtable for our new methods.
  VtableComponents.reserve(VtableComponents.size() + Methods.size());

  for (unsigned i = 0, e = Methods.size(); i != e; ++i) {
    GlobalDecl GD = Methods[i];
    const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());
  
    // Get the 'this' pointer adjustment.
    ThunkAdjustment ThisAdjustment = ThisAdjustments.lookup(i);
  
    // Construct the return type adjustment.
    ThunkAdjustment ReturnAdjustment;

    QualType BaseReturnType = BaseReturnTypes.lookup(i);
    if (!BaseReturnType.isNull() && !MD->isPure()) {
      QualType DerivedType = 
        MD->getType()->getAs<FunctionType>()->getResultType();
      
      int64_t NonVirtualAdjustment = 
      getNVOffset(BaseReturnType, DerivedType) / 8;
      
      int64_t VirtualAdjustment = 
      getVbaseOffset(BaseReturnType, DerivedType);
      
      ReturnAdjustment = ThunkAdjustment(NonVirtualAdjustment, 
                                         VirtualAdjustment);
    }

    llvm::Constant *Method = 0;
    if (!ReturnAdjustment.isEmpty()) {
      // Build a covariant thunk.
      CovariantThunkAdjustment Adjustment(ThisAdjustment, ReturnAdjustment);
      Method = wrap(CGM.GetAddrOfCovariantThunk(GD, Adjustment));
    } else if (!ThisAdjustment.isEmpty()) {
      // Build a "regular" thunk.
      Method = wrap(CGM.GetAddrOfThunk(GD, ThisAdjustment));
    } else if (MD->isPure()) {
      // We have a pure virtual method.
      Method = getPureVirtualFn();
    } else {
      // We have a good old regular method.
      Method = WrapAddrOf(GD);
    }

    // Add the method to the vtable.
    VtableComponents.push_back(Method);
  }
  
  
  ThisAdjustments.clear();
  BaseReturnTypes.clear();
  
  Methods.clear();
}

void CGVtableInfo::ComputeMethodVtableIndices(const CXXRecordDecl *RD) {
  
  // Itanium C++ ABI 2.5.2:
  //   The order of the virtual function pointers in a virtual table is the 
  //   order of declaration of the corresponding member functions in the class.
  //
  //   There is an entry for any virtual function declared in a class, 
  //   whether it is a new function or overrides a base class function, 
  //   unless it overrides a function from the primary base, and conversion
  //   between their return types does not require an adjustment. 

  int64_t CurrentIndex = 0;
  
  const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);
  const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();
  
  if (PrimaryBase) {
    assert(PrimaryBase->isDefinition() && 
           "Should have the definition decl of the primary base!");

    // Since the record decl shares its vtable pointer with the primary base
    // we need to start counting at the end of the primary base's vtable.
    CurrentIndex = getNumVirtualFunctionPointers(PrimaryBase);
  }

  // Collect all the primary bases, so we can check whether methods override
  // a method from the base.
  VtableBuilder::PrimaryBasesSetVectorTy PrimaryBases;
  for (ASTRecordLayout::primary_base_info_iterator
       I = Layout.primary_base_begin(), E = Layout.primary_base_end();
       I != E; ++I)
    PrimaryBases.insert((*I).getBase());

  const CXXDestructorDecl *ImplicitVirtualDtor = 0;
  
  for (CXXRecordDecl::method_iterator i = RD->method_begin(),
       e = RD->method_end(); i != e; ++i) {
    const CXXMethodDecl *MD = *i;

    // We only want virtual methods.
    if (!MD->isVirtual())
      continue;

    // Check if this method overrides a method in the primary base.
    if (const CXXMethodDecl *OverriddenMD = 
          OverridesMethodInBases(MD, PrimaryBases)) {
      // Check if converting from the return type of the method to the 
      // return type of the overridden method requires conversion.
      if (ComputeReturnAdjustmentBaseOffset(CGM.getContext(), MD, 
                                            OverriddenMD).isEmpty()) {
        // This index is shared between the index in the vtable of the primary
        // base class.
        if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(MD)) {
          const CXXDestructorDecl *OverriddenDD = 
            cast<CXXDestructorDecl>(OverriddenMD);
          
          // Add both the complete and deleting entries.
          MethodVtableIndices[GlobalDecl(DD, Dtor_Complete)] = 
            getMethodVtableIndex(GlobalDecl(OverriddenDD, Dtor_Complete));
          MethodVtableIndices[GlobalDecl(DD, Dtor_Deleting)] = 
            getMethodVtableIndex(GlobalDecl(OverriddenDD, Dtor_Deleting));
        } else {
          MethodVtableIndices[MD] = getMethodVtableIndex(OverriddenMD);
        }
        
        // We don't need to add an entry for this method.
        continue;
      }
    }
    
    if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(MD)) {
      if (MD->isImplicit()) {
        assert(!ImplicitVirtualDtor && 
               "Did already see an implicit virtual dtor!");
        ImplicitVirtualDtor = DD;
        continue;
      } 

      // Add the complete dtor.
      MethodVtableIndices[GlobalDecl(DD, Dtor_Complete)] = CurrentIndex++;
      
      // Add the deleting dtor.
      MethodVtableIndices[GlobalDecl(DD, Dtor_Deleting)] = CurrentIndex++;
    } else {
      // Add the entry.
      MethodVtableIndices[MD] = CurrentIndex++;
    }
  }

  if (ImplicitVirtualDtor) {
    // Itanium C++ ABI 2.5.2:
    //   If a class has an implicitly-defined virtual destructor, 
    //   its entries come after the declared virtual function pointers.

    // Add the complete dtor.
    MethodVtableIndices[GlobalDecl(ImplicitVirtualDtor, Dtor_Complete)] = 
      CurrentIndex++;
    
    // Add the deleting dtor.
    MethodVtableIndices[GlobalDecl(ImplicitVirtualDtor, Dtor_Deleting)] = 
      CurrentIndex++;
  }
  
  NumVirtualFunctionPointers[RD] = CurrentIndex;
}

uint64_t CGVtableInfo::getNumVirtualFunctionPointers(const CXXRecordDecl *RD) {
  llvm::DenseMap<const CXXRecordDecl *, uint64_t>::iterator I = 
    NumVirtualFunctionPointers.find(RD);
  if (I != NumVirtualFunctionPointers.end())
    return I->second;

  ComputeMethodVtableIndices(RD);

  I = NumVirtualFunctionPointers.find(RD);
  assert(I != NumVirtualFunctionPointers.end() && "Did not find entry!");
  return I->second;
}
      
uint64_t CGVtableInfo::getMethodVtableIndex(GlobalDecl GD) {
  MethodVtableIndicesTy::iterator I = MethodVtableIndices.find(GD);
  if (I != MethodVtableIndices.end())
    return I->second;
  
  const CXXRecordDecl *RD = cast<CXXMethodDecl>(GD.getDecl())->getParent();

  ComputeMethodVtableIndices(RD);

  I = MethodVtableIndices.find(GD);
  assert(I != MethodVtableIndices.end() && "Did not find index!");
  return I->second;
}

CGVtableInfo::AdjustmentVectorTy*
CGVtableInfo::getAdjustments(GlobalDecl GD) {
  SavedAdjustmentsTy::iterator I = SavedAdjustments.find(GD);
  if (I != SavedAdjustments.end())
    return &I->second;

  const CXXRecordDecl *RD = cast<CXXRecordDecl>(GD.getDecl()->getDeclContext());
  if (!SavedAdjustmentRecords.insert(RD).second)
    return 0;

  AddressPointsMapTy AddressPoints;
  OldVtableBuilder b(RD, RD, 0, CGM, false, AddressPoints);
  D1(printf("vtable %s\n", RD->getNameAsCString()));
  b.GenerateVtableForBase(RD);
  b.GenerateVtableForVBases(RD);

  for (OldVtableBuilder::SavedAdjustmentsVectorTy::iterator
       i = b.getSavedAdjustments().begin(),
       e = b.getSavedAdjustments().end(); i != e; i++)
    SavedAdjustments[i->first].push_back(i->second);

  I = SavedAdjustments.find(GD);
  if (I != SavedAdjustments.end())
    return &I->second;

  return 0;
}

int64_t CGVtableInfo::getVirtualBaseOffsetIndex(const CXXRecordDecl *RD, 
                                                const CXXRecordDecl *VBase) {
  ClassPairTy ClassPair(RD, VBase);
  
  VirtualBaseClassIndiciesTy::iterator I = 
    VirtualBaseClassIndicies.find(ClassPair);
  if (I != VirtualBaseClassIndicies.end())
    return I->second;
  
  // FIXME: This seems expensive.  Can we do a partial job to get
  // just this data.
  AddressPointsMapTy AddressPoints;
  OldVtableBuilder b(RD, RD, 0, CGM, false, AddressPoints);
  D1(printf("vtable %s\n", RD->getNameAsCString()));
  b.GenerateVtableForBase(RD);
  b.GenerateVtableForVBases(RD);
  
  for (llvm::DenseMap<const CXXRecordDecl *, uint64_t>::iterator I =
       b.getVBIndex().begin(), E = b.getVBIndex().end(); I != E; ++I) {
    // Insert all types.
    ClassPairTy ClassPair(RD, I->first);
    
    VirtualBaseClassIndicies.insert(std::make_pair(ClassPair, I->second));
  }
  
  I = VirtualBaseClassIndicies.find(ClassPair);
  // FIXME: The assertion below assertion currently fails with the old vtable 
  /// layout code if there is a non-virtual thunk adjustment in a vtable.
  // Once the new layout is in place, this return should be removed.
  if (I == VirtualBaseClassIndicies.end())
    return 0;
  
  assert(I != VirtualBaseClassIndicies.end() && "Did not find index!");
  
  return I->second;
}

uint64_t CGVtableInfo::getVtableAddressPoint(const CXXRecordDecl *RD) {
  uint64_t AddressPoint = 
    (*(*(CGM.getVtableInfo().AddressPoints[RD]))[RD])[std::make_pair(RD, 0)];
  
  return AddressPoint;
}

llvm::GlobalVariable *
CGVtableInfo::GenerateVtable(llvm::GlobalVariable::LinkageTypes Linkage,
                             bool GenerateDefinition,
                             const CXXRecordDecl *LayoutClass,
                             const CXXRecordDecl *RD, uint64_t Offset,
                             bool IsVirtual,
                             AddressPointsMapTy& AddressPoints) {
  if (GenerateDefinition) {
    if (LayoutClass == RD) {
      assert(!IsVirtual && 
             "Can only have a virtual base in construction vtables!");
      assert(!Offset && 
             "Can only have a base offset in construction vtables!");
    }
    
    VtableBuilder Builder(*this, RD, Offset, 
                          /*MostDerivedClassIsVirtual=*/IsVirtual,
                          LayoutClass);

    if (CGM.getLangOptions().DumpVtableLayouts)
      Builder.dumpLayout(llvm::errs());
  }

  llvm::SmallString<256> OutName;
  if (LayoutClass != RD)
    CGM.getMangleContext().mangleCXXCtorVtable(LayoutClass, Offset / 8, 
                                               RD, OutName);
  else
    CGM.getMangleContext().mangleCXXVtable(RD, OutName);
  llvm::StringRef Name = OutName.str();

  llvm::GlobalVariable *GV = CGM.getModule().getGlobalVariable(Name);
  if (GV == 0 || CGM.getVtableInfo().AddressPoints[LayoutClass] == 0 || 
      GV->isDeclaration()) {
    OldVtableBuilder b(RD, LayoutClass, Offset, CGM, GenerateDefinition,
                       AddressPoints);

    D1(printf("vtable %s\n", RD->getNameAsCString()));
    // First comes the vtables for all the non-virtual bases...
    b.GenerateVtableForBase(RD, Offset);

    // then the vtables for all the virtual bases.
    b.GenerateVtableForVBases(RD, Offset);

    llvm::Constant *Init = 0;
    const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(CGM.getLLVMContext());
    llvm::ArrayType *ArrayType = 
      llvm::ArrayType::get(Int8PtrTy, b.getVtableComponents().size());

    if (GenerateDefinition)
      Init = llvm::ConstantArray::get(ArrayType, &b.getVtableComponents()[0], 
                                      b.getVtableComponents().size());

    llvm::GlobalVariable *OGV = GV;
    
    GV = new llvm::GlobalVariable(CGM.getModule(), ArrayType, 
                                  /*isConstant=*/true, Linkage, Init, Name);
    CGM.setGlobalVisibility(GV, RD);
  
    if (OGV) {
      GV->takeName(OGV);
      llvm::Constant *NewPtr = 
        llvm::ConstantExpr::getBitCast(GV, OGV->getType());
      OGV->replaceAllUsesWith(NewPtr);
      OGV->eraseFromParent();
    }
  }
  
  return GV;
}

void CGVtableInfo::GenerateClassData(llvm::GlobalVariable::LinkageTypes Linkage,
                                     const CXXRecordDecl *RD) {
  llvm::GlobalVariable *&Vtable = Vtables[RD];
  if (Vtable) {
    assert(Vtable->getInitializer() && "Vtable doesn't have a definition!");
    return;
  }
  
  AddressPointsMapTy AddressPoints;
  Vtable = GenerateVtable(Linkage, /*GenerateDefinition=*/true, RD, RD, 0,
                          /*IsVirtual=*/false,
                          AddressPoints);
  GenerateVTT(Linkage, /*GenerateDefinition=*/true, RD);

  for (CXXRecordDecl::method_iterator i = RD->method_begin(),
	 e = RD->method_end(); i != e; ++i) {
    if (!(*i)->isVirtual())
      continue;
    if(!(*i)->hasInlineBody() && !(*i)->isImplicit())
      continue;

    if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(*i)) {
      CGM.BuildThunksForVirtual(GlobalDecl(DD, Dtor_Complete));
      CGM.BuildThunksForVirtual(GlobalDecl(DD, Dtor_Deleting));
    } else {
      CGM.BuildThunksForVirtual(GlobalDecl(*i));
    }
  }
}

llvm::GlobalVariable *CGVtableInfo::getVtable(const CXXRecordDecl *RD) {
  llvm::GlobalVariable *Vtable = Vtables.lookup(RD);
  
  if (!Vtable) {
    AddressPointsMapTy AddressPoints;
    Vtable = GenerateVtable(llvm::GlobalValue::ExternalLinkage, 
                            /*GenerateDefinition=*/false, RD, RD, 0,
                            /*IsVirtual=*/false, AddressPoints);
  }

  return Vtable;
}

void CGVtableInfo::MaybeEmitVtable(GlobalDecl GD) {
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());
  const CXXRecordDecl *RD = MD->getParent();

  // If the class doesn't have a vtable we don't need to emit one.
  if (!RD->isDynamicClass())
    return;
  
  // Get the key function.
  const CXXMethodDecl *KeyFunction = CGM.getContext().getKeyFunction(RD);
  
  if (KeyFunction) {
    // We don't have the right key function.
    if (KeyFunction->getCanonicalDecl() != MD->getCanonicalDecl())
      return;
  }

  if (Vtables.count(RD))
    return;

  TemplateSpecializationKind kind = RD->getTemplateSpecializationKind();
  if (kind == TSK_ImplicitInstantiation)
    CGM.DeferredVtables.push_back(RD);
  else
    GenerateClassData(CGM.getVtableLinkage(RD), RD);
}
