//===- SwapStructContents.cpp - Swap structure elements around ---*- C++ -*--=//
//
// This pass does a simple transformation that swaps all of the elements of the
// struct types in the program around.
//
//===----------------------------------------------------------------------===//


#include "llvm/Transforms/SwapStructContents.h"
#include "llvm/Transforms/MutateStructTypes.h"
#include "llvm/Analysis/FindUsedTypes.h"
#include "llvm/Analysis/FindUnsafePointerTypes.h"
#include "TransformInternals.h"
#include <algorithm>

#include "llvm/Assembly/Writer.h"

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
    cerr << "Unable to swap type: " << ST << endl;
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

static inline void GetTransformation(const StructType *ST,
                                     vector<int> &Transform,
                                enum PrebuiltStructMutation::Transform XForm) {
  unsigned NumElements = ST->getElementTypes().size();
  Transform.reserve(NumElements);

  switch (XForm) {
  case PrebuiltStructMutation::SwapElements:
    // The transformation to do is: just simply swap the elements
    for (unsigned i = 0; i < NumElements; ++i)
      Transform.push_back(NumElements-i-1);
    break;

  case PrebuiltStructMutation::SortElements: {
    vector<pair<unsigned, unsigned> > ElList;

    // Build mapping from index to size
    for (unsigned i = 0; i < NumElements; ++i)
      ElList.push_back(make_pair(i, TD.getTypeSize(ST->getElementTypes()[i])));

    sort(ElList.begin(), ElList.end(), ptr_fun(FirstLess));

    for (unsigned i = 0; i < NumElements; ++i)
      Transform.push_back(getIndex(ElList, i));

    break;
  }
  }
}

// doPassInitialization - This does all of the work of the pass
//
PrebuiltStructMutation::TransformsType
  PrebuiltStructMutation::getTransforms(Module *M, enum Transform XForm) {
  // We need to know which types to modify, and which types we CAN'T modify
  FindUsedTypes          FUT/*(true)*/; // TODO: Do symbol tables as well
  FindUnsafePointerTypes FUPT;

  // Simutaneously find all of the types used, and all of the types that aren't
  // safe.
  //
  vector<Pass*> Analyses;
  Analyses.push_back(&FUT);
  Analyses.push_back(&FUPT);
  Pass::runAllPasses(M, Analyses);  // Do analyses


  // Get the results out of the analyzers...
  const set<PointerType*> &UnsafePTys = FUPT.getUnsafeTypes();
  const set<const Type *> &UsedTypes  = FUT.getTypes();


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
    cerr << "Pruning type: " << *I << endl;
    PruneTypes(*I, TypesToModify, ProcessedTypes);
  }


  // Build up a set of structure types that we are going to modify, and
  // information describing how to modify them.
  map<const StructType*, vector<int> > Transforms;

  for (set<const StructType*>::iterator I = TypesToModify.begin(),
         E = TypesToModify.end(); I != E; ++I) {
    const StructType *ST = *I;

    vector<int> &Transform = Transforms[ST];  // Fill in the map directly
    GetTransformation(ST, Transform, XForm);
  }
  
  return Transforms;
}

