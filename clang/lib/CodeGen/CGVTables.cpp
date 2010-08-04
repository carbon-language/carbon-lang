//===--- CGVTables.cpp - Emit LLVM Code for C++ vtables -------------------===//
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

  /// MethodBaseOffsetPairTy - Uniquely identifies a member function
  /// in a base subobject.
  typedef std::pair<const CXXMethodDecl *, uint64_t> MethodBaseOffsetPairTy;

  typedef llvm::DenseMap<MethodBaseOffsetPairTy,
                         OverriderInfo> OverridersMapTy;
  
  /// OverridersMap - The final overriders for all virtual member functions of 
  /// all the base subobjects of the most derived class.
  OverridersMapTy OverridersMap;
  
  /// SubobjectsToOffsetsMapTy - A mapping from a base subobject (represented
  /// as a record decl and a subobject number) and its offsets in the most
  /// derived class as well as the layout class.
  typedef llvm::DenseMap<std::pair<const CXXRecordDecl *, unsigned>, 
                         uint64_t> SubobjectOffsetMapTy;

  typedef llvm::DenseMap<const CXXRecordDecl *, unsigned> SubobjectCountMapTy;
  
  /// ComputeBaseOffsets - Compute the offsets for all base subobjects of the
  /// given base.
  void ComputeBaseOffsets(BaseSubobject Base, bool IsVirtual,
                          uint64_t OffsetInLayoutClass,
                          SubobjectOffsetMapTy &SubobjectOffsets,
                          SubobjectOffsetMapTy &SubobjectLayoutClassOffsets,
                          SubobjectCountMapTy &SubobjectCounts);

  typedef llvm::SmallPtrSet<const CXXRecordDecl *, 4> VisitedVirtualBasesSetTy;
  
  /// dump - dump the final overriders for a base subobject, and all its direct
  /// and indirect base subobjects.
  void dump(llvm::raw_ostream &Out, BaseSubobject Base,
            VisitedVirtualBasesSetTy& VisitedVirtualBases);
  
public:
  FinalOverriders(const CXXRecordDecl *MostDerivedClass,
                  uint64_t MostDerivedClassOffset,
                  const CXXRecordDecl *LayoutClass);

  /// getOverrider - Get the final overrider for the given method declaration in
  /// the subobject with the given base offset. 
  OverriderInfo getOverrider(const CXXMethodDecl *MD, 
                             uint64_t BaseOffset) const {
    assert(OverridersMap.count(std::make_pair(MD, BaseOffset)) && 
           "Did not find overrider!");
    
    return OverridersMap.lookup(std::make_pair(MD, BaseOffset));
  }
  
  /// dump - dump the final overriders.
  void dump() {
    VisitedVirtualBasesSetTy VisitedVirtualBases;
    dump(llvm::errs(), BaseSubobject(MostDerivedClass, 0), VisitedVirtualBases);
  }
  
};

#define DUMP_OVERRIDERS 0

FinalOverriders::FinalOverriders(const CXXRecordDecl *MostDerivedClass,
                                 uint64_t MostDerivedClassOffset,
                                 const CXXRecordDecl *LayoutClass)
  : MostDerivedClass(MostDerivedClass), 
  MostDerivedClassOffset(MostDerivedClassOffset), LayoutClass(LayoutClass),
  Context(MostDerivedClass->getASTContext()),
  MostDerivedClassLayout(Context.getASTRecordLayout(MostDerivedClass)) {

  // Compute base offsets.
  SubobjectOffsetMapTy SubobjectOffsets;
  SubobjectOffsetMapTy SubobjectLayoutClassOffsets;
  SubobjectCountMapTy SubobjectCounts;
  ComputeBaseOffsets(BaseSubobject(MostDerivedClass, 0), /*IsVirtual=*/false,
                     MostDerivedClassOffset, SubobjectOffsets, 
                     SubobjectLayoutClassOffsets, SubobjectCounts);

  // Get the the final overriders.
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
      
      uint64_t BaseOffset = SubobjectOffsets[std::make_pair(MD->getParent(),
                                                            SubobjectNumber)];

      assert(I->second.size() == 1 && "Final overrider is not unique!");
      const UniqueVirtualMethod &Method = I->second.front();

      const CXXRecordDecl *OverriderRD = Method.Method->getParent();
      assert(SubobjectLayoutClassOffsets.count(
             std::make_pair(OverriderRD, Method.Subobject))
             && "Did not find subobject offset!");
      uint64_t OverriderOffset =
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

void 
FinalOverriders::ComputeBaseOffsets(BaseSubobject Base, bool IsVirtual,
                              uint64_t OffsetInLayoutClass,
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

  SubobjectOffsets[std::make_pair(RD, SubobjectNumber)] =
    Base.getBaseOffset();
  SubobjectLayoutClassOffsets[std::make_pair(RD, SubobjectNumber)] =
    OffsetInLayoutClass;
  
  // Traverse our bases.
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    const CXXRecordDecl *BaseDecl = 
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    uint64_t BaseOffset;
    uint64_t BaseOffsetInLayoutClass;
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
      uint64_t Offset = Layout.getBaseClassOffset(BaseDecl);
    
      BaseOffset = Base.getBaseOffset() + Offset;
      BaseOffsetInLayoutClass = OffsetInLayoutClass + Offset;
    }

    ComputeBaseOffsets(BaseSubobject(BaseDecl, BaseOffset), I->isVirtual(),
                       BaseOffsetInLayoutClass, SubobjectOffsets,
                       SubobjectLayoutClassOffsets, SubobjectCounts);
  }
}

void FinalOverriders::dump(llvm::raw_ostream &Out, BaseSubobject Base,
                           VisitedVirtualBasesSetTy &VisitedVirtualBases) {
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

    dump(Out, BaseSubobject(BaseDecl, BaseOffset), VisitedVirtualBases);
  }

  Out << "Final overriders for (" << RD->getQualifiedNameAsString() << ", ";
  Out << Base.getBaseOffset() / 8 << ")\n";

  // Now dump the overriders for this base subobject.
  for (CXXRecordDecl::method_iterator I = RD->method_begin(), 
       E = RD->method_end(); I != E; ++I) {
    const CXXMethodDecl *MD = *I;

    if (!MD->isVirtual())
      continue;
  
    OverriderInfo Overrider = getOverrider(MD, Base.getBaseOffset());

    Out << "  " << MD->getQualifiedNameAsString() << " - (";
    Out << Overrider.Method->getQualifiedNameAsString();
    Out << ", " << ", " << Overrider.Offset / 8 << ')';

    BaseOffset Offset;
    if (!Overrider.Method->isPure())
      Offset = ComputeReturnAdjustmentBaseOffset(Context, Overrider.Method, MD);

    if (!Offset.isEmpty()) {
      Out << " [ret-adj: ";
      if (Offset.VirtualBase)
        Out << Offset.VirtualBase->getQualifiedNameAsString() << " vbase, ";
             
      Out << Offset.NonVirtualOffset << " nv]";
    }
    
    Out << "\n";
  }  
}

/// VTableComponent - Represents a single component in a vtable.
class VTableComponent {
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

  static VTableComponent MakeVCallOffset(int64_t Offset) {
    return VTableComponent(CK_VCallOffset, Offset);
  }

  static VTableComponent MakeVBaseOffset(int64_t Offset) {
    return VTableComponent(CK_VBaseOffset, Offset);
  }

  static VTableComponent MakeOffsetToTop(int64_t Offset) {
    return VTableComponent(CK_OffsetToTop, Offset);
  }
  
  static VTableComponent MakeRTTI(const CXXRecordDecl *RD) {
    return VTableComponent(CK_RTTI, reinterpret_cast<uintptr_t>(RD));
  }

  static VTableComponent MakeFunction(const CXXMethodDecl *MD) {
    assert(!isa<CXXDestructorDecl>(MD) && 
           "Don't use MakeFunction with destructors!");

    return VTableComponent(CK_FunctionPointer, 
                           reinterpret_cast<uintptr_t>(MD));
  }
  
  static VTableComponent MakeCompleteDtor(const CXXDestructorDecl *DD) {
    return VTableComponent(CK_CompleteDtorPointer,
                           reinterpret_cast<uintptr_t>(DD));
  }

  static VTableComponent MakeDeletingDtor(const CXXDestructorDecl *DD) {
    return VTableComponent(CK_DeletingDtorPointer, 
                           reinterpret_cast<uintptr_t>(DD));
  }

  static VTableComponent MakeUnusedFunction(const CXXMethodDecl *MD) {
    assert(!isa<CXXDestructorDecl>(MD) && 
           "Don't use MakeUnusedFunction with destructors!");
    return VTableComponent(CK_UnusedFunctionPointer,
                           reinterpret_cast<uintptr_t>(MD));                           
  }

  static VTableComponent getFromOpaqueInteger(uint64_t I) {
    return VTableComponent(I);
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
  VTableComponent(Kind ComponentKind, int64_t Offset) {
    assert((ComponentKind == CK_VCallOffset || 
            ComponentKind == CK_VBaseOffset ||
            ComponentKind == CK_OffsetToTop) && "Invalid component kind!");
    assert(Offset <= ((1LL << 56) - 1) && "Offset is too big!");
    
    Value = ((Offset << 3) | ComponentKind);
  }

  VTableComponent(Kind ComponentKind, uintptr_t Ptr) {
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
  
  explicit VTableComponent(uint64_t Value)
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
  typedef llvm::SmallVector<VTableComponent, 64> VTableComponentVectorTy;
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
  // We only want to add vcall offsets if the base is non-virtual; a virtual
  // primary base will have its vcall and vbase offsets emitted already.
  if (PrimaryBase && !Layout.getPrimaryBaseWasVirtual()) {
    // Get the base offset of the primary base.
    assert(Layout.getBaseClassOffset(PrimaryBase) == 0 &&
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

    int64_t OffsetOffset = getCurrentOffsetOffset();
    
    // Don't add a vcall offset if we already have one for this member function
    // signature.
    if (!VCallOffsets.AddVCallOffset(MD, OffsetOffset))
      continue;

    int64_t Offset = 0;

    if (Overriders) {
      // Get the final overrider.
      FinalOverriders::OverriderInfo Overrider = 
        Overriders->getOverrider(MD, Base.getBaseOffset());
      
      /// The vcall offset is the offset from the virtual base to the object 
      /// where the function was overridden.
      // FIXME: We should not use / 8 here.
      Offset = (int64_t)(Overrider.Offset - VBaseOffset) / 8;
    }
    
    Components.push_back(VTableComponent::MakeVCallOffset(Offset));
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

      Components.push_back(VTableComponent::MakeVBaseOffset(Offset));
    }

    // Check the base class looking for more vbase offsets.
    AddVBaseOffsets(BaseDecl, OffsetInLayoutClass);
  }
}

/// VTableBuilder - Class for building vtable layout information.
class VTableBuilder {
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
  llvm::SmallVector<VTableComponent, 64> Components;

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
    
    /// VTableIndex - The index in the vtable that this method has.
    /// (For destructors, this is the index of the complete destructor).
    const uint64_t VTableIndex;
    
    MethodInfo(uint64_t BaseOffset, uint64_t BaseOffsetInLayoutClass, 
               uint64_t VTableIndex)
      : BaseOffset(BaseOffset), 
      BaseOffsetInLayoutClass(BaseOffsetInLayoutClass),
      VTableIndex(VTableIndex) { }
    
    MethodInfo() : BaseOffset(0), BaseOffsetInLayoutClass(0), VTableIndex(0) { }
  };
  
  typedef llvm::DenseMap<const CXXMethodDecl *, MethodInfo> MethodInfoMapTy;
  
  /// MethodInfoMap - The information for all methods in the vtable we're
  /// currently building.
  MethodInfoMapTy MethodInfoMap;
  
  typedef llvm::DenseMap<uint64_t, ThunkInfo> VTableThunksMapTy;
  
  /// VTableThunks - The thunks by vtable index in the vtable currently being 
  /// built.
  VTableThunksMapTy VTableThunks;

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
                                        uint64_t OffsetInLayoutClass);
  
  /// LayoutSecondaryVTables - Layout the secondary vtables for the given base
  /// subobject.
  ///
  /// \param BaseIsMorallyVirtual whether the base subobject is a virtual base
  /// or a direct or indirect base of a virtual base.
  void LayoutSecondaryVTables(BaseSubobject Base, bool BaseIsMorallyVirtual,
                              uint64_t OffsetInLayoutClass);

  /// DeterminePrimaryVirtualBases - Determine the primary virtual bases in this
  /// class hierarchy.
  void DeterminePrimaryVirtualBases(const CXXRecordDecl *RD, 
                                    uint64_t OffsetInLayoutClass,
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
  VTableBuilder(CodeGenVTables &VTables, const CXXRecordDecl *MostDerivedClass,
                uint64_t MostDerivedClassOffset, bool MostDerivedClassIsVirtual,
                const CXXRecordDecl *LayoutClass)
    : VTables(VTables), MostDerivedClass(MostDerivedClass),
    MostDerivedClassOffset(MostDerivedClassOffset), 
    MostDerivedClassIsVirtual(MostDerivedClassIsVirtual), 
    LayoutClass(LayoutClass), Context(MostDerivedClass->getASTContext()), 
    Overriders(MostDerivedClass, MostDerivedClassOffset, LayoutClass) {

    LayoutVTable();
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

  VTableThunksMapTy::const_iterator vtable_thunks_begin() const {
    return VTableThunks.begin();
  }

  VTableThunksMapTy::const_iterator vtable_thunks_end() const {
    return VTableThunks.end();
  }

  /// dumpLayout - Dump the vtable layout.
  void dumpLayout(llvm::raw_ostream&);
};

void VTableBuilder::AddThunk(const CXXMethodDecl *MD, const ThunkInfo &Thunk) {
  assert(!isBuildingConstructorVTable() && 
         "Can't add thunks for construction vtable");

  llvm::SmallVector<ThunkInfo, 1> &ThunksVector = Thunks[MD];
  
  // Check if we have this thunk already.
  if (std::find(ThunksVector.begin(), ThunksVector.end(), Thunk) != 
      ThunksVector.end())
    return;
  
  ThunksVector.push_back(Thunk);
}

typedef llvm::SmallPtrSet<const CXXMethodDecl *, 8> OverriddenMethodsSetTy;

/// ComputeAllOverriddenMethods - Given a method decl, will return a set of all
/// the overridden methods that the function decl overrides.
static void 
ComputeAllOverriddenMethods(const CXXMethodDecl *MD,
                            OverriddenMethodsSetTy& OverriddenMethods) {
  assert(MD->isVirtual() && "Method is not virtual!");

  for (CXXMethodDecl::method_iterator I = MD->begin_overridden_methods(),
       E = MD->end_overridden_methods(); I != E; ++I) {
    const CXXMethodDecl *OverriddenMD = *I;
    
    OverriddenMethods.insert(OverriddenMD);
    
    ComputeAllOverriddenMethods(OverriddenMD, OverriddenMethods);
  }
}

void VTableBuilder::ComputeThisAdjustments() {
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

ReturnAdjustment VTableBuilder::ComputeReturnAdjustment(BaseOffset Offset) {
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
    }

    Adjustment.NonVirtual = Offset.NonVirtualOffset;
  }
  
  return Adjustment;
}

BaseOffset
VTableBuilder::ComputeThisAdjustmentBaseOffset(BaseSubobject Base,
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
VTableBuilder::ComputeThisAdjustment(const CXXMethodDecl *MD, 
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
VTableBuilder::AddMethod(const CXXMethodDecl *MD,
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
static bool
OverridesIndirectMethodInBases(const CXXMethodDecl *MD,
                               VTableBuilder::PrimaryBasesSetVectorTy &Bases) {
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

bool 
VTableBuilder::IsOverriderUsed(const CXXMethodDecl *Overrider,
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
  
  VTableBuilder::PrimaryBasesSetVectorTy PrimaryBases;

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
                            VTableBuilder::PrimaryBasesSetVectorTy &Bases) {
  OverriddenMethodsSetTy OverriddenMethods;
  ComputeAllOverriddenMethods(MD, OverriddenMethods);
  
  for (int I = Bases.size(), E = 0; I != E; --I) {
    const CXXRecordDecl *PrimaryBase = Bases[I - 1];

    // Now check the overriden methods.
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

void
VTableBuilder::AddMethods(BaseSubobject Base, uint64_t BaseOffsetInLayoutClass,                  
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
        
        MethodInfo MethodInfo(Base.getBaseOffset(), 
                              BaseOffsetInLayoutClass,
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

void VTableBuilder::LayoutVTable() {
  LayoutPrimaryAndSecondaryVTables(BaseSubobject(MostDerivedClass, 0),
                                   /*BaseIsMorallyVirtual=*/false,
                                   MostDerivedClassIsVirtual,
                                   MostDerivedClassOffset);
  
  VisitedVirtualBasesSetTy VBases;
  
  // Determine the primary virtual bases.
  DeterminePrimaryVirtualBases(MostDerivedClass, MostDerivedClassOffset,
                               VBases);
  VBases.clear();
  
  LayoutVTablesForVirtualBases(MostDerivedClass, VBases);
}
  
void
VTableBuilder::LayoutPrimaryAndSecondaryVTables(BaseSubobject Base,
                                                bool BaseIsMorallyVirtual,
                                                bool BaseIsVirtualInLayoutClass,
                                                uint64_t OffsetInLayoutClass) {
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
  // FIXME: We should not use / 8 here.
  int64_t OffsetToTop = -(int64_t)(OffsetInLayoutClass -
                                   MostDerivedClassOffset) / 8;
  Components.push_back(VTableComponent::MakeOffsetToTop(OffsetToTop));
  
  // Next, add the RTTI.
  Components.push_back(VTableComponent::MakeRTTI(MostDerivedClass));
  
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

  // Layout secondary vtables.
  LayoutSecondaryVTables(Base, BaseIsMorallyVirtual, OffsetInLayoutClass);
}

void VTableBuilder::LayoutSecondaryVTables(BaseSubobject Base,
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
    uint64_t RelativeBaseOffset = Layout.getBaseClassOffset(BaseDecl);
    uint64_t BaseOffset = Base.getBaseOffset() + RelativeBaseOffset;
    
    uint64_t BaseOffsetInLayoutClass = OffsetInLayoutClass + RelativeBaseOffset;
    
    // Don't emit a secondary vtable for a primary base. We might however want 
    // to emit secondary vtables for other bases of this base.
    if (BaseDecl == PrimaryBase) {
      LayoutSecondaryVTables(BaseSubobject(BaseDecl, BaseOffset),
                             BaseIsMorallyVirtual, BaseOffsetInLayoutClass);
      continue;
    }

    // Layout the primary vtable (and any secondary vtables) for this base.
    LayoutPrimaryAndSecondaryVTables(BaseSubobject(BaseDecl, BaseOffset),
                                     BaseIsMorallyVirtual,
                                     /*BaseIsVirtualInLayoutClass=*/false,
                                     BaseOffsetInLayoutClass);
  }
}

void
VTableBuilder::DeterminePrimaryVirtualBases(const CXXRecordDecl *RD,
                                            uint64_t OffsetInLayoutClass,
                                            VisitedVirtualBasesSetTy &VBases) {
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
  
  // Check if this base has a primary base.
  if (const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase()) {

    // Check if it's virtual.
    if (Layout.getPrimaryBaseWasVirtual()) {
      bool IsPrimaryVirtualBase = true;

      if (isBuildingConstructorVTable()) {
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
VTableBuilder::LayoutVTablesForVirtualBases(const CXXRecordDecl *RD, 
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

      LayoutPrimaryAndSecondaryVTables(BaseSubobject(BaseDecl, BaseOffset),
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

/// dumpLayout - Dump the vtable layout.
void VTableBuilder::dumpLayout(llvm::raw_ostream& Out) {

  if (isBuildingConstructorVTable()) {
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

    const VTableComponent &Component = Components[I];

    // Dump the component.
    switch (Component.getKind()) {

    case VTableComponent::CK_VCallOffset:
      Out << "vcall_offset (" << Component.getVCallOffset() << ")";
      break;

    case VTableComponent::CK_VBaseOffset:
      Out << "vbase_offset (" << Component.getVBaseOffset() << ")";
      break;

    case VTableComponent::CK_OffsetToTop:
      Out << "offset_to_top (" << Component.getOffsetToTop() << ")";
      break;
    
    case VTableComponent::CK_RTTI:
      Out << Component.getRTTIDecl()->getQualifiedNameAsString() << " RTTI";
      break;
    
    case VTableComponent::CK_FunctionPointer: {
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

    case VTableComponent::CK_CompleteDtorPointer: 
    case VTableComponent::CK_DeletingDtorPointer: {
      bool IsComplete = 
        Component.getKind() == VTableComponent::CK_CompleteDtorPointer;
      
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
  
  if (isBuildingConstructorVTable())
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

void CodeGenVTables::ComputeMethodVTableIndices(const CXXRecordDecl *RD) {
  
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
  VTableBuilder::PrimaryBasesSetVectorTy PrimaryBases;
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
          FindNearestOverriddenMethod(MD, PrimaryBases)) {
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
          MethodVTableIndices[GlobalDecl(DD, Dtor_Complete)] = 
            getMethodVTableIndex(GlobalDecl(OverriddenDD, Dtor_Complete));
          MethodVTableIndices[GlobalDecl(DD, Dtor_Deleting)] = 
            getMethodVTableIndex(GlobalDecl(OverriddenDD, Dtor_Deleting));
        } else {
          MethodVTableIndices[MD] = getMethodVTableIndex(OverriddenMD);
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
      MethodVTableIndices[GlobalDecl(DD, Dtor_Complete)] = CurrentIndex++;
      
      // Add the deleting dtor.
      MethodVTableIndices[GlobalDecl(DD, Dtor_Deleting)] = CurrentIndex++;
    } else {
      // Add the entry.
      MethodVTableIndices[MD] = CurrentIndex++;
    }
  }

  if (ImplicitVirtualDtor) {
    // Itanium C++ ABI 2.5.2:
    //   If a class has an implicitly-defined virtual destructor, 
    //   its entries come after the declared virtual function pointers.

    // Add the complete dtor.
    MethodVTableIndices[GlobalDecl(ImplicitVirtualDtor, Dtor_Complete)] = 
      CurrentIndex++;
    
    // Add the deleting dtor.
    MethodVTableIndices[GlobalDecl(ImplicitVirtualDtor, Dtor_Deleting)] = 
      CurrentIndex++;
  }
  
  NumVirtualFunctionPointers[RD] = CurrentIndex;
}

uint64_t CodeGenVTables::getNumVirtualFunctionPointers(const CXXRecordDecl *RD) {
  llvm::DenseMap<const CXXRecordDecl *, uint64_t>::iterator I = 
    NumVirtualFunctionPointers.find(RD);
  if (I != NumVirtualFunctionPointers.end())
    return I->second;

  ComputeMethodVTableIndices(RD);

  I = NumVirtualFunctionPointers.find(RD);
  assert(I != NumVirtualFunctionPointers.end() && "Did not find entry!");
  return I->second;
}
      
uint64_t CodeGenVTables::getMethodVTableIndex(GlobalDecl GD) {
  MethodVTableIndicesTy::iterator I = MethodVTableIndices.find(GD);
  if (I != MethodVTableIndices.end())
    return I->second;
  
  const CXXRecordDecl *RD = cast<CXXMethodDecl>(GD.getDecl())->getParent();

  ComputeMethodVTableIndices(RD);

  I = MethodVTableIndices.find(GD);
  assert(I != MethodVTableIndices.end() && "Did not find index!");
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
  assert(I != VirtualBaseClassOffsetOffsets.end() && "Did not find index!");
  
  return I->second;
}

uint64_t
CodeGenVTables::getAddressPoint(BaseSubobject Base, const CXXRecordDecl *RD) {
  assert(AddressPoints.count(std::make_pair(RD, Base)) &&
         "Did not find address point!");

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
  
  const llvm::Type *Ty = getTypes().GetFunctionTypeForVTable(MD);
  return GetOrCreateLLVMFunction(Name, Ty, GD);
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
    RValue Arg = EmitDelegateCallArg(Param);
    
    CallArgs.push_back(std::make_pair(Arg, ArgType));
  }

  // Get our callee.
  const llvm::Type *Ty =
    CGM.getTypes().GetFunctionType(CGM.getTypes().getFunctionInfo(MD),
                                   FPT->isVariadic());
  llvm::Value *Callee = CGM.GetAddrOfFunction(GD, Ty);

  const CGFunctionInfo &FnInfo = 
    CGM.getTypes().getFunctionInfo(ResultType, CallArgs,
                                   FPT->getExtInfo());
  
  // Determine whether we have a return value slot to use.
  ReturnValueSlot Slot;
  if (!ResultType->isVoidType() &&
      FnInfo.getReturnInfo().getKind() == ABIArgInfo::Indirect &&
      hasAggregateLLVMType(CurFnInfo->getReturnType()))
    Slot = ReturnValueSlot(ReturnValue, ResultType.isVolatileQualified());
  
  // Now emit our call.
  RValue RV = EmitCall(FnInfo, Callee, Slot, CallArgs, MD);
  
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

  if (!ResultType->isVoidType() && Slot.isNull())
    EmitReturnOfRValue(RV, ResultType);

  FinishFunction();

  // Set the right linkage.
  CGM.setFunctionLinkage(MD, Fn);
  
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
      CGM.getTypes().GetFunctionTypeForVTable(MD)) {
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
  ComputeVTableRelatedInformation(RD, false);
  
  ThunksMapTy::const_iterator I = Thunks.find(MD);
  if (I == Thunks.end()) {
    // We did not find a thunk for this method.
    return;
  }

  const ThunkInfoVectorTy &ThunkInfoVector = I->second;
  for (unsigned I = 0, E = ThunkInfoVector.size(); I != E; ++I)
    EmitThunk(GD, ThunkInfoVector[I]);
}

void CodeGenVTables::ComputeVTableRelatedInformation(const CXXRecordDecl *RD,
                                                     bool RequireVTable) {
  VTableLayoutData &Entry = VTableLayoutMap[RD];

  // We may need to generate a definition for this vtable.
  if (RequireVTable && !Entry.getInt()) {
    if (!isKeyFunctionInAnotherTU(CGM.getContext(), RD) &&
        RD->getTemplateSpecializationKind()
          != TSK_ExplicitInstantiationDeclaration)
      CGM.DeferredVTables.push_back(RD);

    Entry.setInt(true);
  }
  
  // Check if we've computed this information before.
  if (Entry.getPointer())
    return;

  VTableBuilder Builder(*this, RD, 0, /*MostDerivedClassIsVirtual=*/0, RD);

  // Add the VTable layout.
  uint64_t NumVTableComponents = Builder.getNumVTableComponents();
  uint64_t *LayoutData = new uint64_t[NumVTableComponents + 1];
  Entry.setPointer(LayoutData);

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
  for (VTableBuilder::AddressPointsMapTy::const_iterator I =
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
  
  for (VTableBuilder::VBaseOffsetOffsetsMapTy::const_iterator I =
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
    VTableComponent Component = 
      VTableComponent::getFromOpaqueInteger(Components[I]);

    llvm::Constant *Init = 0;

    switch (Component.getKind()) {
    case VTableComponent::CK_VCallOffset:
      Init = llvm::ConstantInt::get(PtrDiffTy, Component.getVCallOffset());
      Init = llvm::ConstantExpr::getIntToPtr(Init, Int8PtrTy);
      break;
    case VTableComponent::CK_VBaseOffset:
      Init = llvm::ConstantInt::get(PtrDiffTy, Component.getVBaseOffset());
      Init = llvm::ConstantExpr::getIntToPtr(Init, Int8PtrTy);
      break;
    case VTableComponent::CK_OffsetToTop:
      Init = llvm::ConstantInt::get(PtrDiffTy, Component.getOffsetToTop());
      Init = llvm::ConstantExpr::getIntToPtr(Init, Int8PtrTy);
      break;
    case VTableComponent::CK_RTTI:
      Init = llvm::ConstantExpr::getBitCast(RTTI, Int8PtrTy);
      break;
    case VTableComponent::CK_FunctionPointer:
    case VTableComponent::CK_CompleteDtorPointer:
    case VTableComponent::CK_DeletingDtorPointer: {
      GlobalDecl GD;
      
      // Get the right global decl.
      switch (Component.getKind()) {
      default:
        llvm_unreachable("Unexpected vtable component kind");
      case VTableComponent::CK_FunctionPointer:
        GD = Component.getFunctionDecl();
        break;
      case VTableComponent::CK_CompleteDtorPointer:
        GD = GlobalDecl(Component.getDestructorDecl(), Dtor_Complete);
        break;
      case VTableComponent::CK_DeletingDtorPointer:
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
          const llvm::Type *Ty = CGM.getTypes().GetFunctionTypeForVTable(MD);
        
          Init = CGM.GetAddrOfFunction(GD, Ty);
        }

        Init = llvm::ConstantExpr::getBitCast(Init, Int8PtrTy);
      }
      break;
    }

    case VTableComponent::CK_UnusedFunctionPointer:
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
  CGM.getMangleContext().mangleCXXVTable(RD, OutName);
  llvm::StringRef Name = OutName.str();

  ComputeVTableRelatedInformation(RD, true);
  
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
  if (CGM.getLangOptions().DumpVTableLayouts) {
    VTableBuilder Builder(*this, RD, 0, /*MostDerivedClassIsVirtual=*/0, RD);

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
  
  // Set the right visibility.
  CGM.setTypeVisibility(VTable, RD, /*ForRTTI*/ false);
}

llvm::GlobalVariable *
CodeGenVTables::GenerateConstructionVTable(const CXXRecordDecl *RD, 
                                      const BaseSubobject &Base, 
                                      bool BaseIsVirtual, 
                                      VTableAddressPointsMapTy& AddressPoints) {
  VTableBuilder Builder(*this, Base.getBase(), Base.getBaseOffset(), 
                        /*MostDerivedClassIsVirtual=*/BaseIsVirtual, RD);

  // Dump the vtable layout if necessary.
  if (CGM.getLangOptions().DumpVTableLayouts)
    Builder.dumpLayout(llvm::errs());

  // Add the address points.
  AddressPoints.insert(Builder.address_points_begin(),
                       Builder.address_points_end());

  // Get the mangled construction vtable name.
  llvm::SmallString<256> OutName;
  CGM.getMangleContext().mangleCXXCtorVTable(RD, Base.getBaseOffset() / 8, 
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
  llvm::GlobalVariable *&VTable = VTables[RD];
  if (VTable) {
    assert(VTable->getInitializer() && "VTable doesn't have a definition!");
    return;
  }

  VTable = GetAddrOfVTable(RD);
  EmitVTableDefinition(VTable, Linkage, RD);

  GenerateVTT(Linkage, /*GenerateDefinition=*/true, RD);

  // If this is the magic class __cxxabiv1::__fundamental_type_info,
  // we will emit the typeinfo for the fundamental types. This is the
  // same behaviour as GCC.
  const DeclContext *DC = RD->getDeclContext();
  if (RD->getIdentifier() &&
      RD->getIdentifier()->isStr("__fundamental_type_info") &&
      isa<NamespaceDecl>(DC) &&
      cast<NamespaceDecl>(DC)->getIdentifier() &&
      cast<NamespaceDecl>(DC)->getIdentifier()->isStr("__cxxabiv1") &&
      DC->getParent()->isTranslationUnit())
    CGM.EmitFundamentalRTTIDescriptors();
}
