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
#include "llvm/Support/Format.h"
#include <cstdio>

using namespace clang;
using namespace CodeGen;

/// TypeConversionRequiresAdjustment - Returns whether conversion from a 
/// derived type to a base type requires adjustment.
static bool
TypeConversionRequiresAdjustment(ASTContext &Ctx,
                                 const CXXRecordDecl *DerivedDecl,
                                 const CXXRecordDecl *BaseDecl) {
  CXXBasePaths Paths(/*FindAmbiguities=*/false,
                     /*RecordPaths=*/true, /*DetectVirtual=*/true);
  if (!const_cast<CXXRecordDecl *>(DerivedDecl)->
      isDerivedFrom(const_cast<CXXRecordDecl *>(BaseDecl), Paths)) {
    assert(false && "Class must be derived from the passed in base class!");
    return false;
  }
  
  // If we found a virtual base we always want to require adjustment.
  if (Paths.getDetectedVirtual())
    return true;
  
  const CXXBasePath &Path = Paths.front();
  
  for (size_t Start = 0, End = Path.size(); Start != End; ++Start) {
    const CXXBasePathElement &Element = Path[Start];
    
    // Check the base class offset.
    const ASTRecordLayout &Layout = Ctx.getASTRecordLayout(Element.Class);
    
    const RecordType *BaseType = Element.Base->getType()->getAs<RecordType>();
    const CXXRecordDecl *Base = cast<CXXRecordDecl>(BaseType->getDecl());
    
    if (Layout.getBaseClassOffset(Base) != 0) {
      // This requires an adjustment.
      return true;
    }
  }
  
  return false;
}

static bool 
TypeConversionRequiresAdjustment(ASTContext &Ctx,
                                 QualType DerivedType, QualType BaseType) {
  // Canonicalize the types.
  QualType CanDerivedType = Ctx.getCanonicalType(DerivedType);
  QualType CanBaseType = Ctx.getCanonicalType(BaseType);
  
  assert(CanDerivedType->getTypeClass() == CanBaseType->getTypeClass() && 
         "Types must have same type class!");
  
  if (CanDerivedType == CanBaseType) {
    // No adjustment needed.
    return false;
  }
  
  if (const ReferenceType *RT = CanDerivedType->getAs<ReferenceType>()) {
    CanDerivedType = RT->getPointeeType();
    CanBaseType = CanBaseType->getAs<ReferenceType>()->getPointeeType();
  } else if (const PointerType *PT = CanDerivedType->getAs<PointerType>()) {
    CanDerivedType = PT->getPointeeType();
    CanBaseType = CanBaseType->getAs<PointerType>()->getPointeeType();
  } else {
    assert(false && "Unexpected return type!");
  }
  
  if (CanDerivedType == CanBaseType) {
    // No adjustment needed.
    return false;
  }
  
  const CXXRecordDecl *DerivedDecl = 
    cast<CXXRecordDecl>(CanDerivedType->getAs<RecordType>()->getDecl());
  
  const CXXRecordDecl *BaseDecl = 
    cast<CXXRecordDecl>(CanBaseType->getAs<RecordType>()->getDecl());
  
  return TypeConversionRequiresAdjustment(Ctx, DerivedDecl, BaseDecl);
}

namespace {

/// FinalOverriders - Contains the final overrider member functions for all
/// member functions in the base subobjects of a class.
class FinalOverriders {
  /// MostDerivedClass - The most derived class for which the final overriders
  /// are stored.
  const CXXRecordDecl *MostDerivedClass;
  
  ASTContext &Context;
  
  /// MostDerivedClassLayout - the AST record layout of the most derived class.
  const ASTRecordLayout &MostDerivedClassLayout;

  typedef llvm::DenseMap<std::pair<BaseSubobject, const CXXMethodDecl *>,
                         const CXXMethodDecl *> OverridersMapTy;
  
  /// OverridersMap - The final overriders for all virtual member functions of 
  /// all the base subobjects of the most derived class.
  OverridersMapTy OverridersMap;
  
  typedef llvm::SmallVector<uint64_t, 1> OffsetVectorTy;
  
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
  typedef llvm::DenseMap<const CXXRecordDecl *, OffsetVectorTy>
    SubobjectOffsetsMapTy;
  
  /// ComputeFinalOverriders - Compute the final overriders for a given base
  /// subobject (and all its direct and indirect bases).
  void ComputeFinalOverriders(BaseSubobject Base,
                              SubobjectOffsetsMapTy &Offsets);
  
  /// AddOverriders - Add the final overriders for this base subobject to the
  /// map of final overriders.  
  void AddOverriders(BaseSubobject Base, SubobjectOffsetsMapTy &Offsets);

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
                          const CXXMethodDecl *NewMD,
                          SubobjectOffsetsMapTy &Offsets);
  
  static void MergeSubobjectOffsets(const SubobjectOffsetsMapTy &NewOffsets,
                                    SubobjectOffsetsMapTy &Offsets);

public:
  explicit FinalOverriders(const CXXRecordDecl *MostDerivedClass);
  
  /// getOverrider - Get the final overrider for the given method declaration in
  /// the given base subobject.
  const CXXMethodDecl *getOverrider(BaseSubobject Base,
                                    const CXXMethodDecl *MD) const {
    assert(OverridersMap.count(std::make_pair(Base, MD)) && 
           "Did not find overrider!");
    
    return OverridersMap.lookup(std::make_pair(Base, MD));
  }
  
  /// dump - dump the final overriders.
  void dump() const { 
      dump(llvm::errs(), BaseSubobject(MostDerivedClass, 0)); 
  }
  
  /// dump - dump the final overriders for a base subobject, and all its direct
  /// and indirect base subobjects.
  void dump(llvm::raw_ostream &Out, BaseSubobject Base) const;
};
  
FinalOverriders::FinalOverriders(const CXXRecordDecl *MostDerivedClass)
  : MostDerivedClass(MostDerivedClass), 
  Context(MostDerivedClass->getASTContext()),
  MostDerivedClassLayout(Context.getASTRecordLayout(MostDerivedClass)) {
    
  // Compute the final overriders.
  SubobjectOffsetsMapTy Offsets;
  ComputeFinalOverriders(BaseSubobject(MostDerivedClass, 0), Offsets);
    
  // And dump them (for now).
  dump();
}

void FinalOverriders::AddOverriders(BaseSubobject Base,
                                    SubobjectOffsetsMapTy &Offsets) {
  const CXXRecordDecl *RD = Base.getBase();

  for (CXXRecordDecl::method_iterator I = RD->method_begin(), 
       E = RD->method_end(); I != E; ++I) {
    const CXXMethodDecl *MD = *I;
    
    if (!MD->isVirtual())
      continue;

    // First, propagate the overrider.
    PropagateOverrider(MD, MD, Offsets);

    // Add the overrider as the final overrider of itself.
    const CXXMethodDecl *&Overrider = OverridersMap[std::make_pair(Base, MD)];
    assert(!Overrider && "Overrider should not exist yet!");

    Overrider = MD;
  }
}

void FinalOverriders::PropagateOverrider(const CXXMethodDecl *OldMD,
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
    const OffsetVectorTy &OffsetVector = Offsets[OverriddenRD];
    
    // Go through all the subobjects.
    for (unsigned I = 0, E = OffsetVector.size(); I != E; ++I) {
      uint64_t Offset = OffsetVector[I];

      const CXXMethodDecl *&Overrider = 
          OverridersMap[std::make_pair(BaseSubobject(OverriddenRD, Offset),
                                       OverriddenMD)];
      assert(Overrider && "Did not find existing overrider!");

      // Set the new overrider.
      Overrider = NewMD;
      
      // And propagate it further.
      PropagateOverrider(OverriddenMD, NewMD, Offsets);
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
    const OffsetVectorTy& NewOffsetsVector = I->second;
    
    OffsetVectorTy &OffsetsVector = Offsets[NewRD];
    if (OffsetsVector.empty()) {
      // There were no previous offsets in this vector, just insert all entries
      // from the new offsets vector.
      OffsetsVector.append(NewOffsetsVector.begin(), NewOffsetsVector.end());
      continue;
    }
    
    assert(false && "FIXME: Handle merging the subobject offsets!");
  }
}

void FinalOverriders::ComputeFinalOverriders(BaseSubobject Base,
                                             SubobjectOffsetsMapTy &Offsets) {
  const CXXRecordDecl *RD = Base.getBase();
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
  
  SubobjectOffsetsMapTy NewOffsets;
  
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    const CXXRecordDecl *BaseDecl = 
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());
    
    assert(!I->isVirtual() && "FIXME: Handle virtual bases!");

    uint64_t BaseOffset = Layout.getBaseClassOffset(BaseDecl) + 
      Base.getBaseOffset();

    // Compute the final overriders for this base.
    ComputeFinalOverriders(BaseSubobject(BaseDecl, BaseOffset), NewOffsets);
  }

  /// Now add the overriders for this particular subobject.
  AddOverriders(Base, NewOffsets);
  
  // And merge the newly discovered subobject offsets.
  MergeSubobjectOffsets(NewOffsets, Offsets);

  /// Finally, add the offset for our own subobject.
  Offsets[RD].push_back(Base.getBaseOffset());
}

void FinalOverriders::dump(llvm::raw_ostream &Out, BaseSubobject Base) const {
  
  const CXXRecordDecl *RD = Base.getBase();
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);

  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    assert(!I->isVirtual() && "FIXME: Handle virtual bases!");
    
    const CXXRecordDecl *BaseDecl = 
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());
    
    uint64_t BaseOffset = Layout.getBaseClassOffset(BaseDecl) + 
      Base.getBaseOffset();
    
    dump(Out, BaseSubobject(BaseDecl, BaseOffset));
  }

  Out << "Final overriders for (" << RD->getQualifiedNameAsString() << ", ";
  Out << Base.getBaseOffset() << ")\n";

  // Now dump the overriders for this base subobject.
  for (CXXRecordDecl::method_iterator I = RD->method_begin(), 
       E = RD->method_end(); I != E; ++I) {
    const CXXMethodDecl *MD = *I;

    if (!MD->isVirtual())
      continue;
  
    const CXXMethodDecl *Overrider = getOverrider(Base, MD);

    Out << "  " << MD->getQualifiedNameAsString() << " - ";
    Out << Overrider->getQualifiedNameAsString() << "\n";
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
    CK_DeletingDtorPointer
  };

  /// dump - Dump the contents of this component to the given stream.
  void dump(llvm::raw_ostream &Out);

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

  /// getKind - Get the kind of this vtable component.
  Kind getKind() const {
    return (Kind)(Value & 0x7);
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
            ComponentKind == CK_DeletingDtorPointer) &&
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
            getKind() == CK_DeletingDtorPointer) &&
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

/// VtableBuilder - Class for building vtable layout information.
class VtableBuilder {
  /// MostDerivedClass - The most derived class for which we're building this
  /// vtable.
  const CXXRecordDecl *MostDerivedClass;

  /// Context - The ASTContext which we will use for layout information.
  const ASTContext &Context;
  
  /// FinalOverriders - The final overriders of the most derived class.
  FinalOverriders Overriders;

  /// Components - The components of the vtable being built.
  llvm::SmallVector<VtableComponent, 64> Components;

  /// AddressPoints - Address points for the vtable being built.
  CGVtableInfo::AddressPointsMapTy AddressPoints;

  /// layoutSimpleVtable - A test function that will layout very simple vtables
  /// without any bases. Just used for testing for now.
  void layoutSimpleVtable(const CXXRecordDecl *RD);
  
public:
  VtableBuilder(const CXXRecordDecl *MostDerivedClass)
    : MostDerivedClass(MostDerivedClass), 
    Context(MostDerivedClass->getASTContext()), Overriders(MostDerivedClass) { 

    layoutSimpleVtable(MostDerivedClass);      
  }

  /// dumpLayout - Dump the vtable layout.
  void dumpLayout(llvm::raw_ostream&);
  
};

void VtableBuilder::layoutSimpleVtable(const CXXRecordDecl *RD) {
  assert(!RD->getNumBases() && 
         "We don't support layout for vtables with bases right now!");
  
  // First, add the offset to top.
  Components.push_back(VtableComponent::MakeOffsetToTop(0));
  
  // Next, add the RTTI.
  Components.push_back(VtableComponent::MakeRTTI(RD));
  
  // Record the address point.
  AddressPoints.insert(std::make_pair(BaseSubobject(RD, 0), Components.size()));

  // Now go through all virtual member functions and add them.
  for (CXXRecordDecl::method_iterator I = RD->method_begin(),
       E = RD->method_end(); I != E; ++I) {
    const CXXMethodDecl *MD = *I;
    
    if (!MD->isVirtual())
      continue;
    
    if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(MD)) {
      // Add both the complete destructor and the deleting destructor.
      Components.push_back(VtableComponent::MakeCompleteDtor(DD));
      Components.push_back(VtableComponent::MakeDeletingDtor(DD));
    } else {
      // Add the function.
      Components.push_back(VtableComponent::MakeFunction(MD));
    }
  }
}

/// dumpLayout - Dump the vtable layout.
void VtableBuilder::dumpLayout(llvm::raw_ostream& Out) {
  
  Out << "Vtable for '" << MostDerivedClass->getQualifiedNameAsString();
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
  
  for (unsigned I = 0, E = Components.size(); I != E; ++I) {
    if (AddressPointsByIndex.count(I)) {
      assert(AddressPointsByIndex.count(I) == 1 &&
             "FIXME: Handle dumping multiple base subobjects for a single "
             "address point!");
      
      const BaseSubobject &Base = AddressPointsByIndex.find(I)->second;
      Out << "       -- (" << Base.getBase()->getQualifiedNameAsString();
      
      // FIXME: Instead of dividing by 8, we should be using CharUnits.
      Out << ", " << Base.getBaseOffset() / 8 << ") vtable address --\n";
    }

    Out << llvm::format("%4d | ", I);

    const VtableComponent &Component = Components[I];

    // Dump the component.
    switch (Component.getKind()) {
    // FIXME: Remove this default case.
    default:
      assert(false && "Unhandled component kind!");
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
      break;
    }

    case VtableComponent::CK_CompleteDtorPointer: {
      const CXXDestructorDecl *DD = Component.getDestructorDecl();
      
      Out << DD->getQualifiedNameAsString() << "() [complete]";
      break;
    }

    case VtableComponent::CK_DeletingDtorPointer: {
      const CXXDestructorDecl *DD = Component.getDestructorDecl();
      
      Out << DD->getQualifiedNameAsString() << "() [deleting]";
      break;
    }

    }

    Out << '\n';
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
               PrevU->getParent()->getNameAsCString(),
               PrevU->getNameAsCString(),
               U->getParent()->getNameAsCString(),
               U->getNameAsCString());
      ForwardUnique[PrevU] = U;
      return;
    }

    // Not set, set it now
    if (0)
      printf("marking %s::%s %p override as %s::%s\n",
             MD->getParent()->getNameAsCString(),
             MD->getNameAsCString(),
             (void*)MD,
             U->getParent()->getNameAsCString(),
             U->getNameAsCString());
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
                  MD->getParent()->getNameAsCString(),
                  MD->getNameAsCString(),
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
    const CXXMethodDecl *MD1 = (const CXXMethodDecl *)p1;
    const CXXMethodDecl *MD2 = (const CXXMethodDecl *)p2;
    return (MD1->getIdentifier() - MD2->getIdentifier());
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
      while (++J != E  && DclCmp(*I, *J) == 0)
        if (DclIsSame(*I, *J)) {
          printf("connecting %s\n", (*I)->getNameAsCString());
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
#define D1(X) do { if (getenv("DEBUG")) { X; } } while (0)

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
              dyn_cast<CXXMethodDecl>(GD.getDecl())->getNameAsCString(),
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
                  dyn_cast<CXXMethodDecl>(GD.getDecl())->getNameAsCString(),
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
    if (TypeConversionRequiresAdjustment(CGM.getContext(), ReturnType, 
                                          OverriddenReturnType)) {
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
  llvm::SmallPtrSet<const CXXRecordDecl *, 5> PrimaryBases;
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

    bool ShouldAddEntryForMethod = true;
    
    // Check if this method overrides a method in the primary base.
    for (CXXMethodDecl::method_iterator i = MD->begin_overridden_methods(),
         e = MD->end_overridden_methods(); i != e; ++i) {
      const CXXMethodDecl *OverriddenMD = *i;
      const CXXRecordDecl *OverriddenRD = OverriddenMD->getParent();
      assert(OverriddenMD->isCanonicalDecl() &&
             "Should have the canonical decl of the overridden RD!");
      
      if (PrimaryBases.count(OverriddenRD)) {
        // Check if converting from the return type of the method to the 
        // return type of the overridden method requires conversion.
        QualType ReturnType = 
          MD->getType()->getAs<FunctionType>()->getResultType();
        QualType OverriddenReturnType =
          OverriddenMD->getType()->getAs<FunctionType>()->getResultType();
        
        if (!TypeConversionRequiresAdjustment(CGM.getContext(), 
                                            ReturnType, OverriddenReturnType)) {
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
          ShouldAddEntryForMethod = false;
          break;
        }        
      }
    }
    
    if (!ShouldAddEntryForMethod)
      continue;
    
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
    // If a class has an implicitly-defined virtual destructor, 
    // its entries come after the declared virtual function pointers.

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
                             AddressPointsMapTy& AddressPoints) {
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
  
  if (GenerateDefinition && CGM.getLangOptions().DumpVtableLayouts) {
    VtableBuilder Builder(RD);
    
    Builder.dumpLayout(llvm::errs());
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
                          AddressPoints);
  GenerateVTT(Linkage, /*GenerateDefinition=*/true, RD);  
}

llvm::GlobalVariable *CGVtableInfo::getVtable(const CXXRecordDecl *RD) {
  llvm::GlobalVariable *Vtable = Vtables.lookup(RD);
  
  if (!Vtable) {
    AddressPointsMapTy AddressPoints;
    Vtable = GenerateVtable(llvm::GlobalValue::ExternalLinkage, 
                            /*GenerateDefinition=*/false, RD, RD, 0,
                            AddressPoints);
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

  // Emit the data.
  GenerateClassData(CGM.getVtableLinkage(RD), RD);

  for (CXXRecordDecl::method_iterator i = RD->method_begin(),
       e = RD->method_end(); i != e; ++i) {
    if ((*i)->isVirtual() && ((*i)->hasInlineBody() || (*i)->isImplicit())) {
      if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(*i)) {
        CGM.BuildThunksForVirtual(GlobalDecl(DD, Dtor_Complete));
        CGM.BuildThunksForVirtual(GlobalDecl(DD, Dtor_Deleting));
      } else {
        CGM.BuildThunksForVirtual(GlobalDecl(*i));
      }
    }
  }
}

