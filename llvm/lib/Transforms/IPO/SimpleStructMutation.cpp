//===- SimpleStructMutation.cpp - Swap structure elements around ----------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This pass does a simple transformation that swaps all of the elements of the
// struct types in the program around.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/MutateStructTypes.h"
#include "llvm/Analysis/FindUsedTypes.h"
#include "llvm/Analysis/FindUnsafePointerTypes.h"
#include "llvm/Target/TargetData.h"
#include "llvm/DerivedTypes.h"
#include <algorithm>
using namespace llvm;

namespace {
  struct SimpleStructMutation : public MutateStructTypes {
    enum Transform { SwapElements, SortElements };
    
    virtual bool run(Module &M)  = 0;

    // getAnalysisUsage - This function needs the results of the
    // FindUsedTypes and FindUnsafePointerTypes analysis passes...
    //
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<TargetData>();
      AU.addRequired<FindUsedTypes>();
      AU.addRequired<FindUnsafePointerTypes>();
      MutateStructTypes::getAnalysisUsage(AU);
    }
    
  protected:
    TransformsType getTransforms(Module &M, enum Transform);
  };

  struct SwapStructElements : public SimpleStructMutation {
    virtual bool run(Module &M) {
      setTransforms(getTransforms(M, SwapElements));
      bool Changed = MutateStructTypes::run(M);
      clearTransforms();
      return Changed;
    }
  };

  struct SortStructElements : public SimpleStructMutation {
    virtual bool run(Module &M) {
      setTransforms(getTransforms(M, SortElements));
      bool Changed = MutateStructTypes::run(M);
      clearTransforms();
      return Changed;
    }
  };

  RegisterOpt<SwapStructElements> X("swapstructs",
                                    "Swap structure types around");
  RegisterOpt<SortStructElements> Y("sortstructs",
                                    "Sort structure elements by size");
}  // end anonymous namespace

Pass *createSwapElementsPass() { return new SwapStructElements(); }
Pass *createSortElementsPass() { return new SortStructElements(); }


// PruneTypes - Given a type Ty, make sure that neither it, or one of its
// subtypes, occur in TypesToModify.
//
static void PruneTypes(const Type *Ty,
                       std::set<const StructType*> &TypesToModify,
                       std::set<const Type*> &ProcessedTypes) {
  if (ProcessedTypes.count(Ty)) return;  // Already been checked
  ProcessedTypes.insert(Ty);

  // If the element is in TypesToModify, remove it now...
  if (const StructType *ST = dyn_cast<StructType>(Ty)) {
    TypesToModify.erase(ST);  // This doesn't fail if the element isn't present
    std::cerr << "Unable to swap type: " << ST << "\n";
  }

  // Remove all types that this type contains as well... do not remove types
  // that are referenced only through pointers, because we depend on the size of
  // the pointer, not on what the structure points to.
  //
  for (Type::subtype_iterator I = Ty->subtype_begin(), E = Ty->subtype_end();
       I != E; ++I) {
    if (!isa<PointerType>(*I))
      PruneTypes(*I, TypesToModify, ProcessedTypes);
  }
}

static bool FirstLess(const std::pair<unsigned, unsigned> &LHS,
                      const std::pair<unsigned, unsigned> &RHS) {
  return LHS.second < RHS.second;
}

static unsigned getIndex(const std::vector<std::pair<unsigned, unsigned> > &Vec,
                         unsigned Field) {
  for (unsigned i = 0; ; ++i)
    if (Vec[i].first == Field) return i;
}

static inline void GetTransformation(const TargetData &TD, const StructType *ST,
                                     std::vector<int> &Transform,
                                   enum SimpleStructMutation::Transform XForm) {
  unsigned NumElements = ST->getElementTypes().size();
  Transform.reserve(NumElements);

  switch (XForm) {
  case SimpleStructMutation::SwapElements:
    // The transformation to do is: just simply swap the elements
    for (unsigned i = 0; i < NumElements; ++i)
      Transform.push_back(NumElements-i-1);
    break;

  case SimpleStructMutation::SortElements: {
    std::vector<std::pair<unsigned, unsigned> > ElList;

    // Build mapping from index to size
    for (unsigned i = 0; i < NumElements; ++i)
      ElList.push_back(
              std::make_pair(i, TD.getTypeSize(ST->getElementTypes()[i])));

    sort(ElList.begin(), ElList.end(), ptr_fun(FirstLess));

    for (unsigned i = 0; i < NumElements; ++i)
      Transform.push_back(getIndex(ElList, i));

    break;
  }
  }
}


SimpleStructMutation::TransformsType
  SimpleStructMutation::getTransforms(Module &, enum Transform XForm) {
  // We need to know which types to modify, and which types we CAN'T modify
  // TODO: Do symbol tables as well

  // Get the results out of the analyzers...
  FindUsedTypes          &FUT = getAnalysis<FindUsedTypes>();
  const std::set<const Type *> &UsedTypes  = FUT.getTypes();

  FindUnsafePointerTypes &FUPT = getAnalysis<FindUnsafePointerTypes>();
  const std::set<PointerType*> &UnsafePTys = FUPT.getUnsafeTypes();


  // Combine the two sets, weeding out non structure types.  Closures in C++
  // sure would be nice.
  std::set<const StructType*> TypesToModify;
  for (std::set<const Type *>::const_iterator I = UsedTypes.begin(), 
         E = UsedTypes.end(); I != E; ++I)
    if (const StructType *ST = dyn_cast<StructType>(*I))
      TypesToModify.insert(ST);


  // Go through the Unsafe types and remove all types from TypesToModify that we
  // are not allowed to modify, because that would be unsafe.
  //
  std::set<const Type*> ProcessedTypes;
  for (std::set<PointerType*>::const_iterator I = UnsafePTys.begin(),
         E = UnsafePTys.end(); I != E; ++I) {
    //cerr << "Pruning type: " << *I << "\n";
    PruneTypes(*I, TypesToModify, ProcessedTypes);
  }


  // Build up a set of structure types that we are going to modify, and
  // information describing how to modify them.
  std::map<const StructType*, std::vector<int> > Transforms;
  TargetData &TD = getAnalysis<TargetData>();

  for (std::set<const StructType*>::iterator I = TypesToModify.begin(),
         E = TypesToModify.end(); I != E; ++I) {
    const StructType *ST = *I;

    std::vector<int> &Transform = Transforms[ST];  // Fill in the map directly
    GetTransformation(TD, ST, Transform, XForm);
  }
  
  return Transforms;
}

