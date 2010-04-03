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
#include <algorithm>
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
    
    OverriderInfo() : Method(0), Offset(0) { }
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
  void AddOverriders(BaseSubobject Base, uint64_t OffsetInLayoutClass,
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
    Out << ", " << ", " << Overrider.Offset / 8 << ')';

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

  static VtableComponent getFromOpaqueInteger(uint64_t I) {
    return VtableComponent(I);
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
  
  explicit VtableComponent(uint64_t Value)
    : Value(Value) { }

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
public:
  typedef llvm::DenseMap<const CXXRecordDecl *, int64_t> 
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
  typedef llvm::SmallVector<VtableComponent, 64> VtableComponentVectorTy;
  VtableComponentVectorTy Components;
  
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
  
  const VCallOffsetMap &getVCallOffsets() const { return VCallOffsets; }
  const VBaseOffsetOffsetsMapTy &getVBaseOffsetOffsets() const {
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
  
  typedef llvm::DenseMap<const CXXRecordDecl *, int64_t> 
    VBaseOffsetOffsetsMapTy;
  
  typedef llvm::DenseMap<BaseSubobject, uint64_t> 
    AddressPointsMapTy;

private:
  /// VTables - Global vtable information.
  CodeGenVTables &VTables;
  
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

  /// VBaseOffsetOffsets - Contains the offsets of the virtual base offsets for
  /// the most derived class.
  VBaseOffsetOffsetsMapTy VBaseOffsetOffsets;
  
  /// Components - The components of the vtable being built.
  llvm::SmallVector<VtableComponent, 64> Components;

  /// AddressPoints - Address points for the vtable being built.
  AddressPointsMapTy AddressPoints;

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
  
  typedef llvm::DenseMap<uint64_t, ThunkInfo> VtableThunksMapTy;
  
  /// VTableThunks - The thunks by vtable index in the vtable currently being 
  /// built.
  VtableThunksMapTy VTableThunks;

  typedef llvm::SmallVector<ThunkInfo, 1> ThunkInfoVectorTy;
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
                        uint64_t BaseOffsetInLayoutClass,
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
  VtableBuilder(CodeGenVTables &VTables, const CXXRecordDecl *MostDerivedClass,
                uint64_t MostDerivedClassOffset, bool MostDerivedClassIsVirtual,
                const CXXRecordDecl *LayoutClass)
    : VTables(VTables), MostDerivedClass(MostDerivedClass),
    MostDerivedClassOffset(MostDerivedClassOffset), 
    MostDerivedClassIsVirtual(MostDerivedClassIsVirtual), 
    LayoutClass(LayoutClass), Context(MostDerivedClass->getASTContext()), 
    Overriders(MostDerivedClass, MostDerivedClassOffset, LayoutClass) {

    LayoutVtable();
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

  /// getNumVTableComponents - Return the number of components in the vtable
  /// currently built.
  uint64_t getNumVTableComponents() const {
    return Components.size();
  }

  const uint64_t *vtable_components_data_begin() const {
    return reinterpret_cast<const uint64_t *>(Components.begin());
  }
  
  const uint64_t *vtable_components_data_end() const {
    return reinterpret_cast<const uint64_t *>(Components.end());
  }
  
  AddressPointsMapTy::const_iterator address_points_begin() const {
    return AddressPoints.begin();
  }

  AddressPointsMapTy::const_iterator address_points_end() const {
    return AddressPoints.end();
  }

  VtableThunksMapTy::const_iterator vtable_thunks_begin() const {
    return VTableThunks.begin();
  }

  VtableThunksMapTy::const_iterator vtable_thunks_end() const {
    return VTableThunks.end();
  }

  /// dumpLayout - Dump the vtable layout.
  void dumpLayout(llvm::raw_ostream&);
};

void VtableBuilder::AddThunk(const CXXMethodDecl *MD, const ThunkInfo &Thunk) {
  assert(!isBuildingConstructorVtable() && 
         "Can't add thunks for construction vtable");

  llvm::SmallVector<ThunkInfo, 1> &ThunksVector = Thunks[MD];
  
  // Check if we have this thunk already.
  if (std::find(ThunksVector.begin(), ThunksVector.end(), Thunk) != 
      ThunksVector.end())
    return;
  
  ThunksVector.push_back(Thunk);
}

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
  // Now go through the method info map and see if any of the methods need
  // 'this' pointer adjustments.
  for (MethodInfoMapTy::const_iterator I = MethodInfoMap.begin(),
       E = MethodInfoMap.end(); I != E; ++I) {
    const CXXMethodDecl *MD = I->first;
    const MethodInfo &MethodInfo = I->second;

    // Ignore adjustments for unused function pointers.
    uint64_t VtableIndex = MethodInfo.VtableIndex;
    if (Components[VtableIndex].getKind() == 
        VtableComponent::CK_UnusedFunctionPointer)
      continue;
    
    // Get the final overrider for this method.
    FinalOverriders::OverriderInfo Overrider =
      Overriders.getOverrider(BaseSubobject(MD->getParent(), 
                                            MethodInfo.BaseOffset), MD);
    
    // Check if we need an adjustment at all.
    if (MethodInfo.BaseOffsetInLayoutClass == Overrider.Offset) {
      // When a return thunk is needed by a derived class that overrides a
      // virtual base, gcc uses a virtual 'this' adjustment as well. 
      // While the thunk itself might be needed by vtables in subclasses or
      // in construction vtables, there doesn't seem to be a reason for using
      // the thunk in this vtable. Still, we do so to match gcc.
      if (VTableThunks.lookup(VtableIndex).Return.isEmpty())
        continue;
    }

    ThisAdjustment ThisAdjustment =
      ComputeThisAdjustment(MD, MethodInfo.BaseOffsetInLayoutClass, Overrider);

    if (ThisAdjustment.isEmpty())
      continue;

    // Add it.
    VTableThunks[VtableIndex].This = ThisAdjustment;

    if (isa<CXXDestructorDecl>(MD)) {
      // Add an adjustment for the deleting destructor as well.
      VTableThunks[VtableIndex + 1].This = ThisAdjustment;
    }
  }

  /// Clear the method info map.
  MethodInfoMap.clear();
  
  if (isBuildingConstructorVtable()) {
    // We don't need to store thunk information for construction vtables.
    return;
  }

  for (VtableThunksMapTy::const_iterator I = VTableThunks.begin(),
       E = VTableThunks.end(); I != E; ++I) {
    const VtableComponent &Component = Components[I->first];
    const ThunkInfo &Thunk = I->second;
    const CXXMethodDecl *MD;
    
    switch (Component.getKind()) {
    default:
      llvm_unreachable("Unexpected vtable component kind!");
    case VtableComponent::CK_FunctionPointer:
      MD = Component.getFunctionDecl();
      break;
    case VtableComponent::CK_CompleteDtorPointer:
      MD = Component.getDestructorDecl();
      break;
    case VtableComponent::CK_DeletingDtorPointer:
      // We've already added the thunk when we saw the complete dtor pointer.
      continue;
    }

    if (MD->getParent() == MostDerivedClass)
      AddThunk(MD, Thunk);
  }
}

ReturnAdjustment VtableBuilder::ComputeReturnAdjustment(BaseOffset Offset) {
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
          VTables.getVirtualBaseOffsetOffset(Offset.DerivedClass,
                                             Offset.VirtualBase);
      }

      // FIXME: Once the assert in getVirtualBaseOffsetOffset is back again,
      // we can get rid of this assert.
      assert(Adjustment.VBaseOffsetOffset != 0 && 
             "Invalid vbase offset offset!");
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
  
ThisAdjustment 
VtableBuilder::ComputeThisAdjustment(const CXXMethodDecl *MD, 
                                     uint64_t BaseOffsetInLayoutClass,
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
                                         BaseSubobject(Offset.VirtualBase, 0),                                           
                                         /*BaseIsVirtual=*/true,
                                         /*OffsetInLayoutClass=*/0);
        
      VCallOffsets = Builder.getVCallOffsets();
    }
      
    Adjustment.VCallOffsetOffset = VCallOffsets.getVCallOffsetOffset(MD);
  }

  // Set the non-virtual part of the adjustment.
  Adjustment.NonVirtual = Offset.NonVirtualOffset;
  
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
      VTableThunks[Components.size()].Return = ReturnAdjustment;

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
        
        // If the overridden method exists in a virtual base class or a direct
        // or indirect base class of a virtual base class, we need to emit a
        // thunk if we ever have a class hierarchy where the base class is not
        // a primary base in the complete object.
        if (!isBuildingConstructorVtable() && OverriddenMD != MD) {
          // Compute the this adjustment.
          ThisAdjustment ThisAdjustment =
            ComputeThisAdjustment(OverriddenMD, BaseOffsetInLayoutClass,
                                  Overrider);

          if (ThisAdjustment.VCallOffsetOffset &&
              Overrider.Method->getParent() == MostDerivedClass) {
            // This is a virtual thunk for the most derived class, add it.
            AddThunk(Overrider.Method, 
                     ThunkInfo(ThisAdjustment, ReturnAdjustment()));
          }
        }

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

  // Add all address points.
  const CXXRecordDecl *RD = Base.getBase();
  while (true) {
    AddressPoints.insert(std::make_pair(BaseSubobject(RD, OffsetInLayoutClass),
                                        AddressPoint));

    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
    const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();
    
    if (!PrimaryBase)
      break;
    
    if (Layout.getPrimaryBaseWasVirtual()) {
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
  for (AddressPointsMapTy::const_iterator I = AddressPoints.begin(), 
       E = AddressPoints.end(); I != E; ++I) {
    const BaseSubobject& Base = I->first;
    uint64_t Index = I->second;
    
    AddressPointsByIndex.insert(std::make_pair(Index, Base));
  }
  
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

      ThunkInfo Thunk = VTableThunks.lookup(I);
      if (!Thunk.isEmpty()) {
        // If this function pointer has a return adjustment, dump it.
        if (!Thunk.Return.isEmpty()) {
          Out << "\n       [return adjustment: ";
          Out << Thunk.Return.NonVirtual << " non-virtual";
          
          if (Thunk.Return.VBaseOffsetOffset) {
            Out << ", " << Thunk.Return.VBaseOffsetOffset;
            Out << " vbase offset offset";
          }

          Out << ']';
        }

        // If this function pointer has a 'this' pointer adjustment, dump it.
        if (!Thunk.This.isEmpty()) {
          Out << "\n       [this adjustment: ";
          Out << Thunk.This.NonVirtual << " non-virtual";
          
          if (Thunk.This.VCallOffsetOffset) {
            Out << ", " << Thunk.This.VCallOffsetOffset;
            Out << " vcall offset offset";
          }

          Out << ']';
        }          
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

      ThunkInfo Thunk = VTableThunks.lookup(I);
      if (!Thunk.isEmpty()) {
        // If this destructor has a 'this' pointer adjustment, dump it.
        if (!Thunk.This.isEmpty()) {
          Out << "\n       [this adjustment: ";
          Out << Thunk.This.NonVirtual << " non-virtual";
          
          if (Thunk.This.VCallOffsetOffset) {
            Out << ", " << Thunk.This.VCallOffsetOffset;
            Out << " vcall offset offset";
          }
          
          Out << ']';
        }          
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
  
  if (isBuildingConstructorVtable())
    return;
  
  if (MostDerivedClass->getNumVBases()) {
    // We store the virtual base class names and their offsets in a map to get
    // a stable order.

    std::map<std::string, int64_t> ClassNamesAndOffsets;
    for (VBaseOffsetOffsetsMapTy::const_iterator I = VBaseOffsetOffsets.begin(),
         E = VBaseOffsetOffsets.end(); I != E; ++I) {
      std::string ClassName = I->first->getQualifiedNameAsString();
      int64_t OffsetOffset = I->second;
      ClassNamesAndOffsets.insert(std::make_pair(ClassName, OffsetOffset));
    }
    
    Out << "Virtual base offset offsets for '";
    Out << MostDerivedClass->getQualifiedNameAsString() << "' (";
    Out << ClassNamesAndOffsets.size();
    Out << (ClassNamesAndOffsets.size() == 1 ? " entry" : " entries") << ").\n";

    for (std::map<std::string, int64_t>::const_iterator I =
         ClassNamesAndOffsets.begin(), E = ClassNamesAndOffsets.end(); 
         I != E; ++I)
      Out << "   " << I->first << " | " << I->second << '\n';

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
      std::sort(ThunksVector.begin(), ThunksVector.end());

      Out << "Thunks for '" << MethodName << "' (" << ThunksVector.size();
      Out << (ThunksVector.size() == 1 ? " entry" : " entries") << ").\n";
      
      for (unsigned I = 0, E = ThunksVector.size(); I != E; ++I) {
        const ThunkInfo &Thunk = ThunksVector[I];

        Out << llvm::format("%4d | ", I);
        
        // If this function pointer has a return pointer adjustment, dump it.
        if (!Thunk.Return.isEmpty()) {
          Out << "return adjustment: " << Thunk.This.NonVirtual;
          Out << " non-virtual";
          if (Thunk.Return.VBaseOffsetOffset) {
            Out << ", " << Thunk.Return.VBaseOffsetOffset;
            Out << " vbase offset offset";
          }

          if (!Thunk.This.isEmpty())
            Out << "\n       ";
        }

        // If this function pointer has a 'this' pointer adjustment, dump it.
        if (!Thunk.This.isEmpty()) {
          Out << "this adjustment: ";
          Out << Thunk.This.NonVirtual << " non-virtual";
          
          if (Thunk.This.VCallOffsetOffset) {
            Out << ", " << Thunk.This.VCallOffsetOffset;
            Out << " vcall offset offset";
          }
        }
        
        Out << '\n';
      }
      
      Out << '\n';

    }
  }
}
  
}

void CodeGenVTables::ComputeMethodVtableIndices(const CXXRecordDecl *RD) {
  
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

uint64_t CodeGenVTables::getNumVirtualFunctionPointers(const CXXRecordDecl *RD) {
  llvm::DenseMap<const CXXRecordDecl *, uint64_t>::iterator I = 
    NumVirtualFunctionPointers.find(RD);
  if (I != NumVirtualFunctionPointers.end())
    return I->second;

  ComputeMethodVtableIndices(RD);

  I = NumVirtualFunctionPointers.find(RD);
  assert(I != NumVirtualFunctionPointers.end() && "Did not find entry!");
  return I->second;
}
      
uint64_t CodeGenVTables::getMethodVtableIndex(GlobalDecl GD) {
  MethodVtableIndicesTy::iterator I = MethodVtableIndices.find(GD);
  if (I != MethodVtableIndices.end())
    return I->second;
  
  const CXXRecordDecl *RD = cast<CXXMethodDecl>(GD.getDecl())->getParent();

  ComputeMethodVtableIndices(RD);

  I = MethodVtableIndices.find(GD);
  assert(I != MethodVtableIndices.end() && "Did not find index!");
  return I->second;
}

int64_t CodeGenVTables::getVirtualBaseOffsetOffset(const CXXRecordDecl *RD, 
                                                   const CXXRecordDecl *VBase) {
  ClassPairTy ClassPair(RD, VBase);
  
  VirtualBaseClassOffsetOffsetsMapTy::iterator I = 
    VirtualBaseClassOffsetOffsets.find(ClassPair);
  if (I != VirtualBaseClassOffsetOffsets.end())
    return I->second;
  
  VCallAndVBaseOffsetBuilder Builder(RD, RD, /*FinalOverriders=*/0,
                                     BaseSubobject(RD, 0),                                           
                                     /*BaseIsVirtual=*/false,
                                     /*OffsetInLayoutClass=*/0);

  for (VCallAndVBaseOffsetBuilder::VBaseOffsetOffsetsMapTy::const_iterator I =
       Builder.getVBaseOffsetOffsets().begin(), 
       E = Builder.getVBaseOffsetOffsets().end(); I != E; ++I) {
    // Insert all types.
    ClassPairTy ClassPair(RD, I->first);
    
    VirtualBaseClassOffsetOffsets.insert(std::make_pair(ClassPair, I->second));
  }
  
  I = VirtualBaseClassOffsetOffsets.find(ClassPair);
  
  // FIXME: The assertion below assertion currently fails with the old vtable 
  /// layout code if there is a non-virtual thunk adjustment in a vtable.
  // Once the new layout is in place, this return should be removed.
  if (I == VirtualBaseClassOffsetOffsets.end())
    return 0;
  
  assert(I != VirtualBaseClassOffsetOffsets.end() && "Did not find index!");
  
  return I->second;
}

uint64_t
CodeGenVTables::getAddressPoint(BaseSubobject Base, const CXXRecordDecl *RD) {
  uint64_t AddressPoint = AddressPoints.lookup(std::make_pair(RD, Base));
  assert(AddressPoint && "Address point must not be zero!");

  return AddressPoint;
}

llvm::Constant *CodeGenModule::GetAddrOfThunk(GlobalDecl GD, 
                                              const ThunkInfo &Thunk) {
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());

  // Compute the mangled name.
  llvm::SmallString<256> Name;
  if (const CXXDestructorDecl* DD = dyn_cast<CXXDestructorDecl>(MD))
    getMangleContext().mangleCXXDtorThunk(DD, GD.getDtorType(), Thunk.This,
                                          Name);
  else
    getMangleContext().mangleThunk(MD, Thunk, Name);
  
  const llvm::Type *Ty = getTypes().GetFunctionTypeForVtable(MD);
  return GetOrCreateLLVMFunction(Name, Ty, GlobalDecl());
}

static llvm::Value *PerformTypeAdjustment(CodeGenFunction &CGF,
                                          llvm::Value *Ptr,
                                          int64_t NonVirtualAdjustment,
                                          int64_t VirtualAdjustment) {
  if (!NonVirtualAdjustment && !VirtualAdjustment)
    return Ptr;

  const llvm::Type *Int8PtrTy = 
    llvm::Type::getInt8PtrTy(CGF.getLLVMContext());
  
  llvm::Value *V = CGF.Builder.CreateBitCast(Ptr, Int8PtrTy);

  if (NonVirtualAdjustment) {
    // Do the non-virtual adjustment.
    V = CGF.Builder.CreateConstInBoundsGEP1_64(V, NonVirtualAdjustment);
  }

  if (VirtualAdjustment) {
    const llvm::Type *PtrDiffTy = 
      CGF.ConvertType(CGF.getContext().getPointerDiffType());

    // Do the virtual adjustment.
    llvm::Value *VTablePtrPtr = 
      CGF.Builder.CreateBitCast(V, Int8PtrTy->getPointerTo());
    
    llvm::Value *VTablePtr = CGF.Builder.CreateLoad(VTablePtrPtr);
  
    llvm::Value *OffsetPtr =
      CGF.Builder.CreateConstInBoundsGEP1_64(VTablePtr, VirtualAdjustment);
    
    OffsetPtr = CGF.Builder.CreateBitCast(OffsetPtr, PtrDiffTy->getPointerTo());
    
    // Load the adjustment offset from the vtable.
    llvm::Value *Offset = CGF.Builder.CreateLoad(OffsetPtr);
    
    // Adjust our pointer.
    V = CGF.Builder.CreateInBoundsGEP(V, Offset);
  }

  // Cast back to the original type.
  return CGF.Builder.CreateBitCast(V, Ptr->getType());
}

void CodeGenFunction::GenerateThunk(llvm::Function *Fn, GlobalDecl GD,
                                    const ThunkInfo &Thunk) {
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());
  const FunctionProtoType *FPT = MD->getType()->getAs<FunctionProtoType>();
  QualType ResultType = FPT->getResultType();
  QualType ThisType = MD->getThisType(getContext());

  FunctionArgList FunctionArgs;

  // FIXME: It would be nice if more of this code could be shared with 
  // CodeGenFunction::GenerateCode.

  // Create the implicit 'this' parameter declaration.
  CXXThisDecl = ImplicitParamDecl::Create(getContext(), 0,
                                          MD->getLocation(),
                                          &getContext().Idents.get("this"),
                                          ThisType);

  // Add the 'this' parameter.
  FunctionArgs.push_back(std::make_pair(CXXThisDecl, CXXThisDecl->getType()));

  // Add the rest of the parameters.
  for (FunctionDecl::param_const_iterator I = MD->param_begin(),
       E = MD->param_end(); I != E; ++I) {
    ParmVarDecl *Param = *I;
    
    FunctionArgs.push_back(std::make_pair(Param, Param->getType()));
  }
  
  StartFunction(GlobalDecl(), ResultType, Fn, FunctionArgs, SourceLocation());

  // Adjust the 'this' pointer if necessary.
  llvm::Value *AdjustedThisPtr = 
    PerformTypeAdjustment(*this, LoadCXXThis(), 
                          Thunk.This.NonVirtual, 
                          Thunk.This.VCallOffsetOffset);
  
  CallArgList CallArgs;
  
  // Add our adjusted 'this' pointer.
  CallArgs.push_back(std::make_pair(RValue::get(AdjustedThisPtr), ThisType));

  // Add the rest of the parameters.
  for (FunctionDecl::param_const_iterator I = MD->param_begin(),
       E = MD->param_end(); I != E; ++I) {
    ParmVarDecl *Param = *I;
    QualType ArgType = Param->getType();
    
    // FIXME: Declaring a DeclRefExpr on the stack is kinda icky.
    DeclRefExpr ArgExpr(Param, ArgType.getNonReferenceType(), SourceLocation());
    CallArgs.push_back(std::make_pair(EmitCallArg(&ArgExpr, ArgType), ArgType));
  }

  // Get our callee.
  const llvm::Type *Ty =
    CGM.getTypes().GetFunctionType(CGM.getTypes().getFunctionInfo(MD),
                                   FPT->isVariadic());
  llvm::Value *Callee = CGM.GetAddrOfFunction(GD, Ty);

  const CGFunctionInfo &FnInfo = 
    CGM.getTypes().getFunctionInfo(ResultType, CallArgs,
                                   FPT->getExtInfo());
  
  // Now emit our call.
  RValue RV = EmitCall(FnInfo, Callee, ReturnValueSlot(), CallArgs, MD);
  
  if (!Thunk.Return.isEmpty()) {
    // Emit the return adjustment.
    bool NullCheckValue = !ResultType->isReferenceType();
    
    llvm::BasicBlock *AdjustNull = 0;
    llvm::BasicBlock *AdjustNotNull = 0;
    llvm::BasicBlock *AdjustEnd = 0;
    
    llvm::Value *ReturnValue = RV.getScalarVal();

    if (NullCheckValue) {
      AdjustNull = createBasicBlock("adjust.null");
      AdjustNotNull = createBasicBlock("adjust.notnull");
      AdjustEnd = createBasicBlock("adjust.end");
    
      llvm::Value *IsNull = Builder.CreateIsNull(ReturnValue);
      Builder.CreateCondBr(IsNull, AdjustNull, AdjustNotNull);
      EmitBlock(AdjustNotNull);
    }
    
    ReturnValue = PerformTypeAdjustment(*this, ReturnValue, 
                                        Thunk.Return.NonVirtual, 
                                        Thunk.Return.VBaseOffsetOffset);
    
    if (NullCheckValue) {
      Builder.CreateBr(AdjustEnd);
      EmitBlock(AdjustNull);
      Builder.CreateBr(AdjustEnd);
      EmitBlock(AdjustEnd);
    
      llvm::PHINode *PHI = Builder.CreatePHI(ReturnValue->getType());
      PHI->reserveOperandSpace(2);
      PHI->addIncoming(ReturnValue, AdjustNotNull);
      PHI->addIncoming(llvm::Constant::getNullValue(ReturnValue->getType()), 
                       AdjustNull);
      ReturnValue = PHI;
    }
    
    RV = RValue::get(ReturnValue);
  }

  if (!ResultType->isVoidType())
    EmitReturnOfRValue(RV, ResultType);

  FinishFunction();

  // Destroy the 'this' declaration.
  CXXThisDecl->Destroy(getContext());
  
  // Set the right linkage.
  Fn->setLinkage(CGM.getFunctionLinkage(MD));
  
  // Set the right visibility.
  CGM.setGlobalVisibility(Fn, MD);
}

void CodeGenVTables::EmitThunk(GlobalDecl GD, const ThunkInfo &Thunk)
{
  llvm::Constant *Entry = CGM.GetAddrOfThunk(GD, Thunk);
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());
  
  // Strip off a bitcast if we got one back.
  if (llvm::ConstantExpr *CE = dyn_cast<llvm::ConstantExpr>(Entry)) {
    assert(CE->getOpcode() == llvm::Instruction::BitCast);
    Entry = CE->getOperand(0);
  }
  
  // There's already a declaration with the same name, check if it has the same
  // type or if we need to replace it.
  if (cast<llvm::GlobalValue>(Entry)->getType()->getElementType() != 
      CGM.getTypes().GetFunctionTypeForVtable(MD)) {
    llvm::GlobalValue *OldThunkFn = cast<llvm::GlobalValue>(Entry);
    
    // If the types mismatch then we have to rewrite the definition.
    assert(OldThunkFn->isDeclaration() &&
           "Shouldn't replace non-declaration");

    // Remove the name from the old thunk function and get a new thunk.
    OldThunkFn->setName(llvm::StringRef());
    Entry = CGM.GetAddrOfThunk(GD, Thunk);
    
    // If needed, replace the old thunk with a bitcast.
    if (!OldThunkFn->use_empty()) {
      llvm::Constant *NewPtrForOldDecl =
        llvm::ConstantExpr::getBitCast(Entry, OldThunkFn->getType());
      OldThunkFn->replaceAllUsesWith(NewPtrForOldDecl);
    }
    
    // Remove the old thunk.
    OldThunkFn->eraseFromParent();
  }

  // Actually generate the thunk body.
  llvm::Function *ThunkFn = cast<llvm::Function>(Entry);
  CodeGenFunction(CGM).GenerateThunk(ThunkFn, GD, Thunk);
}

void CodeGenVTables::EmitThunks(GlobalDecl GD)
{
  const CXXMethodDecl *MD = 
    cast<CXXMethodDecl>(GD.getDecl())->getCanonicalDecl();

  // We don't need to generate thunks for the base destructor.
  if (isa<CXXDestructorDecl>(MD) && GD.getDtorType() == Dtor_Base)
    return;

  const CXXRecordDecl *RD = MD->getParent();
  
  // Compute VTable related info for this class.
  ComputeVTableRelatedInformation(RD);
  
  ThunksMapTy::const_iterator I = Thunks.find(MD);
  if (I == Thunks.end()) {
    // We did not find a thunk for this method.
    return;
  }

  const ThunkInfoVectorTy &ThunkInfoVector = I->second;
  for (unsigned I = 0, E = ThunkInfoVector.size(); I != E; ++I)
    EmitThunk(GD, ThunkInfoVector[I]);
}

void CodeGenVTables::ComputeVTableRelatedInformation(const CXXRecordDecl *RD) {
  uint64_t *&LayoutData = VTableLayoutMap[RD];
  
  // Check if we've computed this information before.
  if (LayoutData)
    return;
      
  VtableBuilder Builder(*this, RD, 0, /*MostDerivedClassIsVirtual=*/0, RD);

  // Add the VTable layout.
  uint64_t NumVTableComponents = Builder.getNumVTableComponents();
  LayoutData = new uint64_t[NumVTableComponents + 1];

  // Store the number of components.
  LayoutData[0] = NumVTableComponents;

  // Store the components.
  std::copy(Builder.vtable_components_data_begin(),
            Builder.vtable_components_data_end(),
            &LayoutData[1]);

  // Add the known thunks.
  Thunks.insert(Builder.thunks_begin(), Builder.thunks_end());
  
  // Add the thunks needed in this vtable.
  assert(!VTableThunksMap.count(RD) && 
         "Thunks already exists for this vtable!");

  VTableThunksTy &VTableThunks = VTableThunksMap[RD];
  VTableThunks.append(Builder.vtable_thunks_begin(),
                      Builder.vtable_thunks_end());
  
  // Sort them.
  std::sort(VTableThunks.begin(), VTableThunks.end());
  
  // Add the address points.
  for (VtableBuilder::AddressPointsMapTy::const_iterator I =
       Builder.address_points_begin(), E = Builder.address_points_end();
       I != E; ++I) {
    
    uint64_t &AddressPoint = AddressPoints[std::make_pair(RD, I->first)];
    
    // Check if we already have the address points for this base.
    assert(!AddressPoint && "Address point already exists for this base!");
    
    AddressPoint = I->second;
  }
  
  // If we don't have the vbase information for this class, insert it.
  // getVirtualBaseOffsetOffset will compute it separately without computing
  // the rest of the vtable related information.
  if (!RD->getNumVBases())
    return;
  
  const RecordType *VBaseRT = 
    RD->vbases_begin()->getType()->getAs<RecordType>();
  const CXXRecordDecl *VBase = cast<CXXRecordDecl>(VBaseRT->getDecl());
  
  if (VirtualBaseClassOffsetOffsets.count(std::make_pair(RD, VBase)))
    return;
  
  for (VtableBuilder::VBaseOffsetOffsetsMapTy::const_iterator I =
       Builder.getVBaseOffsetOffsets().begin(), 
       E = Builder.getVBaseOffsetOffsets().end(); I != E; ++I) {
    // Insert all types.
    ClassPairTy ClassPair(RD, I->first);
    
    VirtualBaseClassOffsetOffsets.insert(std::make_pair(ClassPair, I->second));
  }
}

llvm::Constant *
CodeGenVTables::CreateVTableInitializer(const CXXRecordDecl *RD,
                                        const uint64_t *Components, 
                                        unsigned NumComponents,
                                        const VTableThunksTy &VTableThunks) {
  llvm::SmallVector<llvm::Constant *, 64> Inits;

  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(CGM.getLLVMContext());
  
  const llvm::Type *PtrDiffTy = 
    CGM.getTypes().ConvertType(CGM.getContext().getPointerDiffType());

  QualType ClassType = CGM.getContext().getTagDeclType(RD);
  llvm::Constant *RTTI = CGM.GetAddrOfRTTIDescriptor(ClassType);
  
  unsigned NextVTableThunkIndex = 0;
  
  llvm::Constant* PureVirtualFn = 0;

  for (unsigned I = 0; I != NumComponents; ++I) {
    VtableComponent Component = 
      VtableComponent::getFromOpaqueInteger(Components[I]);

    llvm::Constant *Init = 0;

    switch (Component.getKind()) {
    case VtableComponent::CK_VCallOffset:
      Init = llvm::ConstantInt::get(PtrDiffTy, Component.getVCallOffset());
      Init = llvm::ConstantExpr::getIntToPtr(Init, Int8PtrTy);
      break;
    case VtableComponent::CK_VBaseOffset:
      Init = llvm::ConstantInt::get(PtrDiffTy, Component.getVBaseOffset());
      Init = llvm::ConstantExpr::getIntToPtr(Init, Int8PtrTy);
      break;
    case VtableComponent::CK_OffsetToTop:
      Init = llvm::ConstantInt::get(PtrDiffTy, Component.getOffsetToTop());
      Init = llvm::ConstantExpr::getIntToPtr(Init, Int8PtrTy);
      break;
    case VtableComponent::CK_RTTI:
      Init = llvm::ConstantExpr::getBitCast(RTTI, Int8PtrTy);
      break;
    case VtableComponent::CK_FunctionPointer:
    case VtableComponent::CK_CompleteDtorPointer:
    case VtableComponent::CK_DeletingDtorPointer: {
      GlobalDecl GD;
      
      // Get the right global decl.
      switch (Component.getKind()) {
      default:
        llvm_unreachable("Unexpected vtable component kind");
      case VtableComponent::CK_FunctionPointer:
        GD = Component.getFunctionDecl();
        break;
      case VtableComponent::CK_CompleteDtorPointer:
        GD = GlobalDecl(Component.getDestructorDecl(), Dtor_Complete);
        break;
      case VtableComponent::CK_DeletingDtorPointer:
        GD = GlobalDecl(Component.getDestructorDecl(), Dtor_Deleting);
        break;
      }

      if (cast<CXXMethodDecl>(GD.getDecl())->isPure()) {
        // We have a pure virtual member function.
        if (!PureVirtualFn) {
          const llvm::FunctionType *Ty = 
            llvm::FunctionType::get(llvm::Type::getVoidTy(CGM.getLLVMContext()), 
                                    /*isVarArg=*/false);
          PureVirtualFn = 
            CGM.CreateRuntimeFunction(Ty, "__cxa_pure_virtual");
          PureVirtualFn = llvm::ConstantExpr::getBitCast(PureVirtualFn, 
                                                         Int8PtrTy);
        }
        
        Init = PureVirtualFn;
      } else {
        // Check if we should use a thunk.
        if (NextVTableThunkIndex < VTableThunks.size() &&
            VTableThunks[NextVTableThunkIndex].first == I) {
          const ThunkInfo &Thunk = VTableThunks[NextVTableThunkIndex].second;
        
          Init = CGM.GetAddrOfThunk(GD, Thunk);
        
          NextVTableThunkIndex++;
        } else {
          const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());
          const llvm::Type *Ty = CGM.getTypes().GetFunctionTypeForVtable(MD);
        
          Init = CGM.GetAddrOfFunction(GD, Ty);
        }

        Init = llvm::ConstantExpr::getBitCast(Init, Int8PtrTy);
      }
      break;
    }

    case VtableComponent::CK_UnusedFunctionPointer:
      Init = llvm::ConstantExpr::getNullValue(Int8PtrTy);
      break;
    };
    
    Inits.push_back(Init);
  }
  
  llvm::ArrayType *ArrayType = llvm::ArrayType::get(Int8PtrTy, NumComponents);
  return llvm::ConstantArray::get(ArrayType, Inits.data(), Inits.size());
}

/// GetGlobalVariable - Will return a global variable of the given type. 
/// If a variable with a different type already exists then a new variable
/// with the right type will be created.
/// FIXME: We should move this to CodeGenModule and rename it to something 
/// better and then use it in CGVTT and CGRTTI. 
static llvm::GlobalVariable *
GetGlobalVariable(llvm::Module &Module, llvm::StringRef Name,
                  const llvm::Type *Ty,
                  llvm::GlobalValue::LinkageTypes Linkage) {

  llvm::GlobalVariable *GV = Module.getNamedGlobal(Name);
  llvm::GlobalVariable *OldGV = 0;
  
  if (GV) {
    // Check if the variable has the right type.
    if (GV->getType()->getElementType() == Ty)
      return GV;

    assert(GV->isDeclaration() && "Declaration has wrong type!");
    
    OldGV = GV;
  }
  
  // Create a new variable.
  GV = new llvm::GlobalVariable(Module, Ty, /*isConstant=*/true,
                                Linkage, 0, Name);
  
  if (OldGV) {
    // Replace occurrences of the old variable if needed.
    GV->takeName(OldGV);
   
    if (!OldGV->use_empty()) {
      llvm::Constant *NewPtrForOldDecl =
        llvm::ConstantExpr::getBitCast(GV, OldGV->getType());
      OldGV->replaceAllUsesWith(NewPtrForOldDecl);
    }

    OldGV->eraseFromParent();
  }
  
  return GV;
}

llvm::GlobalVariable *CodeGenVTables::GetAddrOfVTable(const CXXRecordDecl *RD) {
  llvm::SmallString<256> OutName;
  CGM.getMangleContext().mangleCXXVtable(RD, OutName);
  llvm::StringRef Name = OutName.str();

  ComputeVTableRelatedInformation(RD);
  
  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(CGM.getLLVMContext());
  llvm::ArrayType *ArrayType = 
    llvm::ArrayType::get(Int8PtrTy, getNumVTableComponents(RD));

  return GetGlobalVariable(CGM.getModule(), Name, ArrayType, 
                           llvm::GlobalValue::ExternalLinkage);
}

void
CodeGenVTables::EmitVTableDefinition(llvm::GlobalVariable *VTable,
                                     llvm::GlobalVariable::LinkageTypes Linkage,
                                     const CXXRecordDecl *RD) {
  // Dump the vtable layout if necessary.
  if (CGM.getLangOptions().DumpVtableLayouts) {
    VtableBuilder Builder(*this, RD, 0, /*MostDerivedClassIsVirtual=*/0, RD);

    Builder.dumpLayout(llvm::errs());
  }

  assert(VTableThunksMap.count(RD) && 
         "No thunk status for this record decl!");
  
  const VTableThunksTy& Thunks = VTableThunksMap[RD];
  
  // Create and set the initializer.
  llvm::Constant *Init = 
    CreateVTableInitializer(RD, getVTableComponentsData(RD),
                            getNumVTableComponents(RD), Thunks);
  VTable->setInitializer(Init);
  
  // Set the correct linkage.
  VTable->setLinkage(Linkage);
}

llvm::GlobalVariable *
CodeGenVTables::GenerateConstructionVTable(const CXXRecordDecl *RD, 
                                      const BaseSubobject &Base, 
                                      bool BaseIsVirtual, 
                                      VTableAddressPointsMapTy& AddressPoints) {
  VtableBuilder Builder(*this, Base.getBase(), Base.getBaseOffset(), 
                        /*MostDerivedClassIsVirtual=*/BaseIsVirtual, RD);

  // Dump the vtable layout if necessary.
  if (CGM.getLangOptions().DumpVtableLayouts)
    Builder.dumpLayout(llvm::errs());

  // Add the address points.
  AddressPoints.insert(Builder.address_points_begin(),
                       Builder.address_points_end());

  // Get the mangled construction vtable name.
  llvm::SmallString<256> OutName;
  CGM.getMangleContext().mangleCXXCtorVtable(RD, Base.getBaseOffset() / 8, 
                                             Base.getBase(), OutName);
  llvm::StringRef Name = OutName.str();

  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(CGM.getLLVMContext());
  llvm::ArrayType *ArrayType = 
    llvm::ArrayType::get(Int8PtrTy, Builder.getNumVTableComponents());

  // Create the variable that will hold the construction vtable.
  llvm::GlobalVariable *VTable = 
    GetGlobalVariable(CGM.getModule(), Name, ArrayType, 
                      llvm::GlobalValue::InternalLinkage);

  // Add the thunks.
  VTableThunksTy VTableThunks;
  VTableThunks.append(Builder.vtable_thunks_begin(),
                      Builder.vtable_thunks_end());

  // Sort them.
  std::sort(VTableThunks.begin(), VTableThunks.end());

  // Create and set the initializer.
  llvm::Constant *Init = 
    CreateVTableInitializer(Base.getBase(), 
                            Builder.vtable_components_data_begin(), 
                            Builder.getNumVTableComponents(), VTableThunks);
  VTable->setInitializer(Init);
  
  return VTable;
}

void 
CodeGenVTables::GenerateClassData(llvm::GlobalVariable::LinkageTypes Linkage,
                                  const CXXRecordDecl *RD) {
  llvm::GlobalVariable *&VTable = Vtables[RD];
  if (VTable) {
    assert(VTable->getInitializer() && "Vtable doesn't have a definition!");
    return;
  }

  VTable = GetAddrOfVTable(RD);
  EmitVTableDefinition(VTable, Linkage, RD);

  GenerateVTT(Linkage, /*GenerateDefinition=*/true, RD);
}

void CodeGenVTables::EmitVTableRelatedData(GlobalDecl GD) {
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());
  const CXXRecordDecl *RD = MD->getParent();

  // If the class doesn't have a vtable we don't need to emit one.
  if (!RD->isDynamicClass())
    return;
  
  // Check if we need to emit thunks for this function.
  if (MD->isVirtual())
    EmitThunks(GD);

  // Get the key function.
  const CXXMethodDecl *KeyFunction = CGM.getContext().getKeyFunction(RD);
  
  TemplateSpecializationKind RDKind = RD->getTemplateSpecializationKind();
  TemplateSpecializationKind MDKind = MD->getTemplateSpecializationKind();

  if (KeyFunction) {
    // We don't have the right key function.
    if (KeyFunction->getCanonicalDecl() != MD->getCanonicalDecl())
      return;
  } else {
    // If we have no key funcion and this is a explicit instantiation declaration,
    // we will produce a vtable at the explicit instantiation. We don't need one
    // here.
    if (RDKind == clang::TSK_ExplicitInstantiationDeclaration)
      return;

    // If this is an explicit instantiation of a method, we don't need a vtable.
    // Since we have no key function, we will emit the vtable when we see
    // a use, and just defining a function is not an use.
    if (RDKind == TSK_ImplicitInstantiation &&
        MDKind == TSK_ExplicitInstantiationDefinition)
      return;
  }

  if (Vtables.count(RD))
    return;

  if (RDKind == TSK_ImplicitInstantiation)
    CGM.DeferredVtables.push_back(RD);
  else
    GenerateClassData(CGM.getVtableLinkage(RD), RD);
}
