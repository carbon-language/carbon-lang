//===- SimpleStructMutation.cpp - Swap structure elements around -*- C++ -*--=//
//
// This pass does a simple transformation that swaps all of the elements of the
// struct types in the program around.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/MutateStructTypes.h"
#include "llvm/Analysis/FindUsedTypes.h"
#include "llvm/Analysis/FindUnsafePointerTypes.h"
#include "llvm/Target/TargetData.h"
#include "llvm/DerivedTypes.h"
#include <algorithm>
#include <iostream>
using std::vector;
using std::set;
using std::pair;

namespace {
  struct SimpleStructMutation : public MutateStructTypes {
    enum Transform { SwapElements, SortElements };
    const TargetData &TD;
    SimpleStructMutation(const TargetData &td) : TD(td) {}
    
    virtual bool run(Module &M)  = 0;

    // getAnalysisUsage - This function needs the results of the
    // FindUsedTypes and FindUnsafePointerTypes analysis passes...
    //
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired(FindUsedTypes::ID);
      AU.addRequired(FindUnsafePointerTypes::ID);
      MutateStructTypes::getAnalysisUsage(AU);
    }
    
  protected:
    TransformsType getTransforms(Module &M, enum Transform);
  };

  struct SwapStructElements : public SimpleStructMutation {
    SwapStructElements(const TargetData &TD) : SimpleStructMutation(TD) {}

    virtual bool run(Module &M) {
      setTransforms(getTransforms(M, SwapElements));
      bool Changed = MutateStructTypes::run(M);
      clearTransforms();
      return Changed;
    }
  };

  struct SortStructElements : public SimpleStructMutation {
    SortStructElements(const TargetData &TD) : SimpleStructMutation(TD) {}

    virtual bool run(Module &M) {
      setTransforms(getTransforms(M, SortElements));
      bool Changed = MutateStructTypes::run(M);
      clearTransforms();
      return Changed;
    }
  };
}  // end anonymous namespace



// PruneTypes - Given a type Ty, make sure that neither it, or one of its
// subtypes, occur in TypesToModify.
//
static void PruneTypes(const Type *Ty, set<const StructType*> &TypesToModify,
                       set<const Type*> &ProcessedTypes) {
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

static bool FirstLess(const pair<unsigned, unsigned> &LHS,
                      const pair<unsigned, unsigned> &RHS) {
  return LHS.second < RHS.second;
}

static unsigned getIndex(const vector<pair<unsigned, unsigned> > &Vec,
                         unsigned Field) {
  for (unsigned i = 0; ; ++i)
    if (Vec[i].first == Field) return i;
}

static inline void GetTransformation(const TargetData &TD, const StructType *ST,
                                     vector<int> &Transform,
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
    vector<pair<unsigned, unsigned> > ElList;

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
  const set<const Type *> &UsedTypes  = FUT.getTypes();

  FindUnsafePointerTypes &FUPT = getAnalysis<FindUnsafePointerTypes>();
  const set<PointerType*> &UnsafePTys = FUPT.getUnsafeTypes();



  // Combine the two sets, weeding out non structure types.  Closures in C++
  // sure would be nice.
  set<const StructType*> TypesToModify;
  for (set<const Type *>::const_iterator I = UsedTypes.begin(), 
         E = UsedTypes.end(); I != E; ++I)
    if (const StructType *ST = dyn_cast<StructType>(*I))
      TypesToModify.insert(ST);


  // Go through the Unsafe types and remove all types from TypesToModify that we
  // are not allowed to modify, because that would be unsafe.
  //
  set<const Type*> ProcessedTypes;
  for (set<PointerType*>::const_iterator I = UnsafePTys.begin(),
         E = UnsafePTys.end(); I != E; ++I) {
    //cerr << "Pruning type: " << *I << "\n";
    PruneTypes(*I, TypesToModify, ProcessedTypes);
  }


  // Build up a set of structure types that we are going to modify, and
  // information describing how to modify them.
  std::map<const StructType*, vector<int> > Transforms;

  for (set<const StructType*>::iterator I = TypesToModify.begin(),
         E = TypesToModify.end(); I != E; ++I) {
    const StructType *ST = *I;

    vector<int> &Transform = Transforms[ST];  // Fill in the map directly
    GetTransformation(TD, ST, Transform, XForm);
  }
  
  return Transforms;
}


Pass *createSwapElementsPass(const TargetData &TD) {
  return new SwapStructElements(TD);
}
Pass *createSortElementsPass(const TargetData &TD) {
  return new SortStructElements(TD);
}

namespace {
  RegisterPass<SwapStructElements> X("swapstructs",
                                     "Swap structure types around",
                                     createSwapElementsPass);
  RegisterPass<SortStructElements> Y("sortstructs",
                                     "Sort structure elements by size",
                                     createSortElementsPass);
}
