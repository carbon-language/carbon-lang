//===--- MicrosoftVBTables.cpp - Virtual Base Table Emission --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class generates data about MSVC virtual base tables.
//
//===----------------------------------------------------------------------===//

#include "MicrosoftVBTables.h"
#include "CodeGenModule.h"
#include "CGCXXABI.h"

namespace clang {
namespace CodeGen {

/// Holds intermediate data about a path to a vbptr inside a base subobject.
struct VBTablePath {
  VBTablePath(const VBTableInfo &VBInfo)
    : VBInfo(VBInfo), NextBase(VBInfo.VBPtrSubobject.getBase()) { }

  /// All the data needed to build a vbtable, minus the GlobalVariable whose
  /// name we haven't computed yet.
  VBTableInfo VBInfo;

  /// Next base to use for disambiguation.  Can be null if we've already
  /// disambiguated this path once.
  const CXXRecordDecl *NextBase;

  /// Path is not really a full path like a CXXBasePath.  It holds the subset of
  /// records that need to be mangled into the vbtable symbol name in order to get
  /// a unique name.
  llvm::SmallVector<const CXXRecordDecl *, 1> Path;
};

VBTableBuilder::VBTableBuilder(CodeGenModule &CGM,
                               const CXXRecordDecl *MostDerived)
    : CGM(CGM), MostDerived(MostDerived),
      DerivedLayout(CGM.getContext().getASTRecordLayout(MostDerived)) {}

void VBTableBuilder::enumerateVBTables(VBTableVector &VBTables) {
  VBTablePathVector Paths;
  findUnambiguousPaths(MostDerived, BaseSubobject(MostDerived,
                                                  CharUnits::Zero()), Paths);
  for (VBTablePathVector::iterator I = Paths.begin(), E = Paths.end();
       I != E; ++I) {
    VBTablePath *P = *I;
    P->VBInfo.GV = getAddrOfVBTable(P->VBInfo.ReusingBase, P->Path);
    VBTables.push_back(P->VBInfo);
  }
}


void VBTableBuilder::findUnambiguousPaths(const CXXRecordDecl *ReusingBase,
                                          BaseSubobject CurSubobject,
                                          VBTablePathVector &Paths) {
  size_t PathsStart = Paths.size();
  bool ReuseVBPtrFromBase = true;
  const CXXRecordDecl *CurBase = CurSubobject.getBase();
  const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(CurBase);

  // If this base has a vbptr, then we've found a path.  These are not full
  // paths, so we don't use CXXBasePath.
  if (Layout.hasOwnVBPtr()) {
    ReuseVBPtrFromBase = false;
    VBTablePath *Info = new VBTablePath(
      VBTableInfo(ReusingBase, CurSubobject, /*GV=*/0));
    Paths.push_back(Info);
  }

  // Recurse onto any bases which themselves have virtual bases.
  for (CXXRecordDecl::base_class_const_iterator I = CurBase->bases_begin(),
       E = CurBase->bases_end(); I != E; ++I) {
    const CXXRecordDecl *Base = I->getType()->getAsCXXRecordDecl();
    if (!Base->getNumVBases())
      continue;  // Bases without virtual bases have no vbptrs.
    CharUnits NextOffset;
    const CXXRecordDecl *NextReusingBase = Base;
    if (I->isVirtual()) {
      if (!VBasesSeen.insert(Base))
        continue;  // Don't visit virtual bases twice.
      NextOffset = DerivedLayout.getVBaseClassOffset(Base);
    } else {
      NextOffset = (CurSubobject.getBaseOffset() +
                    Layout.getBaseClassOffset(Base));

      // If CurBase didn't have a vbptr, then ReusingBase will reuse the vbptr
      // from the first non-virtual base with vbases for its vbptr.
      if (ReuseVBPtrFromBase) {
        NextReusingBase = ReusingBase;
        ReuseVBPtrFromBase = false;
      }
    }

    size_t NumPaths = Paths.size();
    findUnambiguousPaths(NextReusingBase, BaseSubobject(Base, NextOffset),
                         Paths);

    // Tag paths through this base with the base itself.  We might use it to
    // disambiguate.
    for (size_t I = NumPaths, E = Paths.size(); I != E; ++I)
      Paths[I]->NextBase = Base;
  }

  bool AmbiguousPaths = rebucketPaths(Paths, PathsStart);
  if (AmbiguousPaths)
    rebucketPaths(Paths, PathsStart, /*SecondPass=*/true);

#ifndef NDEBUG
  // Check that the paths are in fact unique.
  for (size_t I = PathsStart + 1, E = Paths.size(); I != E; ++I) {
    assert(Paths[I]->Path != Paths[I - 1]->Path && "vbtable paths are not unique");
  }
#endif
}

static bool pathCompare(VBTablePath *LHS, VBTablePath *RHS) {
  return LHS->Path < RHS->Path;
}

void VBTableBuilder::extendPath(VBTablePath *P, bool SecondPass) {
  assert(P->NextBase || SecondPass);
  if (P->NextBase) {
    P->Path.push_back(P->NextBase);
    P->NextBase = 0;  // Prevent the path from being extended twice.
  }
}

bool VBTableBuilder::rebucketPaths(VBTablePathVector &Paths, size_t PathsStart,
                                   bool SecondPass) {
  // What we're essentially doing here is bucketing together ambiguous paths.
  // Any bucket with more than one path in it gets extended by NextBase, which
  // is usually the direct base of the inherited the vbptr.  This code uses a
  // sorted vector to implement a multiset to form the buckets.  Note that the
  // ordering is based on pointers, but it doesn't change our output order.  The
  // current algorithm is designed to match MSVC 2012's names.
  // TODO: Implement MSVC 2010 or earlier names to avoid extra vbtable cruft.
  VBTablePathVector PathsSorted(&Paths[PathsStart], &Paths.back() + 1);
  std::sort(PathsSorted.begin(), PathsSorted.end(), pathCompare);
  bool AmbiguousPaths = false;
  for (size_t I = 0, E = PathsSorted.size(); I != E;) {
    // Scan forward to find the end of the bucket.
    size_t BucketStart = I;
    do {
      ++I;
    } while (I != E && PathsSorted[BucketStart]->Path == PathsSorted[I]->Path);

    // If this bucket has multiple paths, extend them all.
    if (I - BucketStart > 1) {
      AmbiguousPaths = true;
      for (size_t II = BucketStart; II != I; ++II)
        extendPath(PathsSorted[II], SecondPass);
    }
  }
  return AmbiguousPaths;
}

llvm::GlobalVariable *
VBTableBuilder::getAddrOfVBTable(const CXXRecordDecl *ReusingBase,
                                 ArrayRef<const CXXRecordDecl *> BasePath) {
  // Caching at this layer is redundant with the caching in EnumerateVBTables().

  SmallString<256> OutName;
  llvm::raw_svector_ostream Out(OutName);
  MicrosoftMangleContext &Mangler =
      cast<MicrosoftMangleContext>(CGM.getCXXABI().getMangleContext());
  Mangler.mangleCXXVBTable(MostDerived, BasePath, Out);
  Out.flush();
  StringRef Name = OutName.str();

  llvm::ArrayType *VBTableType =
    llvm::ArrayType::get(CGM.IntTy, 1 + ReusingBase->getNumVBases());

  assert(!CGM.getModule().getNamedGlobal(Name) &&
         "vbtable with this name already exists: mangling bug?");
  llvm::GlobalVariable *VBTable =
    CGM.CreateOrReplaceCXXRuntimeVariable(Name, VBTableType,
                                          llvm::GlobalValue::ExternalLinkage);
  VBTable->setUnnamedAddr(true);
  return VBTable;
}

void VBTableInfo::EmitVBTableDefinition(
    CodeGenModule &CGM, const CXXRecordDecl *RD,
    llvm::GlobalVariable::LinkageTypes Linkage) const {
  assert(RD->getNumVBases() && ReusingBase->getNumVBases() &&
         "should only emit vbtables for classes with vbtables");

  const ASTRecordLayout &BaseLayout =
    CGM.getContext().getASTRecordLayout(VBPtrSubobject.getBase());
  const ASTRecordLayout &DerivedLayout =
    CGM.getContext().getASTRecordLayout(RD);

  SmallVector<llvm::Constant *, 4> Offsets(1 + ReusingBase->getNumVBases(), 0);

  // The offset from ReusingBase's vbptr to itself always leads.
  CharUnits VBPtrOffset = BaseLayout.getVBPtrOffset();
  Offsets[0] = llvm::ConstantInt::get(CGM.IntTy, -VBPtrOffset.getQuantity());

  MicrosoftVTableContext &Context = CGM.getMicrosoftVTableContext();
  for (CXXRecordDecl::base_class_const_iterator I = ReusingBase->vbases_begin(),
       E = ReusingBase->vbases_end(); I != E; ++I) {
    const CXXRecordDecl *VBase = I->getType()->getAsCXXRecordDecl();
    CharUnits Offset = DerivedLayout.getVBaseClassOffset(VBase);
    assert(!Offset.isNegative());
    // Make it relative to the subobject vbptr.
    Offset -= VBPtrSubobject.getBaseOffset() + VBPtrOffset;
    unsigned VBIndex = Context.getVBTableIndex(ReusingBase, VBase);
    assert(Offsets[VBIndex] == 0 && "The same vbindex seen twice?");
    Offsets[VBIndex] = llvm::ConstantInt::get(CGM.IntTy, Offset.getQuantity());
  }

  assert(Offsets.size() ==
         cast<llvm::ArrayType>(cast<llvm::PointerType>(GV->getType())
                               ->getElementType())->getNumElements());
  llvm::ArrayType *VBTableType =
    llvm::ArrayType::get(CGM.IntTy, Offsets.size());
  llvm::Constant *Init = llvm::ConstantArray::get(VBTableType, Offsets);
  GV->setInitializer(Init);

  // Set the correct linkage.
  GV->setLinkage(Linkage);

  // Set the right visibility.
  CGM.setTypeVisibility(GV, RD, CodeGenModule::TVK_ForVTable);
}

} // namespace CodeGen
} // namespace clang
