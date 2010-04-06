//===-- LLVMContextImpl.cpp - Implement LLVMContextImpl -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the opaque LLVMContextImpl.
//
//===----------------------------------------------------------------------===//

#include "LLVMContextImpl.h"
#include <algorithm>

LLVMContextImpl::LLVMContextImpl(LLVMContext &C)
  : TheTrueVal(0), TheFalseVal(0),
    VoidTy(C, Type::VoidTyID),
    LabelTy(C, Type::LabelTyID),
    FloatTy(C, Type::FloatTyID),
    DoubleTy(C, Type::DoubleTyID),
    MetadataTy(C, Type::MetadataTyID),
    X86_FP80Ty(C, Type::X86_FP80TyID),
    FP128Ty(C, Type::FP128TyID),
    PPC_FP128Ty(C, Type::PPC_FP128TyID),
    Int1Ty(C, 1),
    Int8Ty(C, 8),
    Int16Ty(C, 16),
    Int32Ty(C, 32),
    Int64Ty(C, 64),
    AlwaysOpaqueTy(new OpaqueType(C)) {
  InlineAsmDiagHandler = 0;
  InlineAsmDiagContext = 0;
      
  // Make sure the AlwaysOpaqueTy stays alive as long as the Context.
  AlwaysOpaqueTy->addRef();
  OpaqueTypes.insert(AlwaysOpaqueTy);
}

namespace {
struct DropReferences {
  // Takes the value_type of a ConstantUniqueMap's internal map, whose 'second'
  // is a Constant*.
  template<typename PairT>
  void operator()(const PairT &P) {
    P.second->dropAllReferences();
  }
};
}

LLVMContextImpl::~LLVMContextImpl() {
  std::for_each(ExprConstants.map_begin(), ExprConstants.map_end(),
                DropReferences());
  std::for_each(ArrayConstants.map_begin(), ArrayConstants.map_end(),
                DropReferences());
  std::for_each(StructConstants.map_begin(), StructConstants.map_end(),
                DropReferences());
  std::for_each(UnionConstants.map_begin(), UnionConstants.map_end(),
                DropReferences());
  std::for_each(VectorConstants.map_begin(), VectorConstants.map_end(),
                DropReferences());
  ExprConstants.freeConstants();
  ArrayConstants.freeConstants();
  StructConstants.freeConstants();
  UnionConstants.freeConstants();
  VectorConstants.freeConstants();
  AggZeroConstants.freeConstants();
  NullPtrConstants.freeConstants();
  UndefValueConstants.freeConstants();
  InlineAsms.freeConstants();
  for (IntMapTy::iterator I = IntConstants.begin(), E = IntConstants.end(); 
       I != E; ++I) {
    delete I->second;
  }
  for (FPMapTy::iterator I = FPConstants.begin(), E = FPConstants.end(); 
       I != E; ++I) {
    delete I->second;
  }
  AlwaysOpaqueTy->dropRef();
  for (OpaqueTypesTy::iterator I = OpaqueTypes.begin(), E = OpaqueTypes.end();
       I != E; ++I) {
    (*I)->AbstractTypeUsers.clear();
    delete *I;
  }
  // Destroy MDNodes.  ~MDNode can move and remove nodes between the MDNodeSet
  // and the NonUniquedMDNodes sets, so copy the values out first.
  SmallVector<MDNode*, 8> MDNodes;
  MDNodes.reserve(MDNodeSet.size() + NonUniquedMDNodes.size());
  for (FoldingSetIterator<MDNode> I = MDNodeSet.begin(), E = MDNodeSet.end();
       I != E; ++I) {
    MDNodes.push_back(&*I);
  }
  MDNodes.append(NonUniquedMDNodes.begin(), NonUniquedMDNodes.end());
  for (SmallVector<MDNode*, 8>::iterator I = MDNodes.begin(),
         E = MDNodes.end(); I != E; ++I) {
    (*I)->destroy();
  }
  assert(MDNodeSet.empty() && NonUniquedMDNodes.empty() &&
         "Destroying all MDNodes didn't empty the Context's sets.");
  // Destroy MDStrings.
  for (StringMap<MDString*>::iterator I = MDStringCache.begin(),
         E = MDStringCache.end(); I != E; ++I) {
    delete I->second;
  }
}
