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
#include "llvm/DerivedTypes.h"

// PruneTypes - Given a type Ty, make sure that neither it, or one of its
// subtypes, occur in TypesToModify.
//
static void PruneTypes(const Type *Ty, set<const StructType*> &TypesToModify,
                       set<const Type*> &ProcessedTypes) {
  if (ProcessedTypes.count(Ty)) return;  // Already been checked
  ProcessedTypes.insert(Ty);

  // If the element is in TypesToModify, remove it now...
  if (const StructType *ST = dyn_cast<StructType>(Ty))
    TypesToModify.erase(ST);  // This doesn't fail if the element isn't present

  // Remove all types that this type contains as well...
  //
  for (Type::subtype_iterator I = Ty->subtype_begin(), E = Ty->subtype_end();
       I != E; ++I)
    PruneTypes(*I, TypesToModify, ProcessedTypes);
}



// doPassInitialization - This does all of the work of the pass
//
bool SwapStructContents::doPassInitialization(Module *M) {
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


  // Combine the two sets, weeding out non structure types.  Closures should
  // would be nice.
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
         E = UnsafePTys.end(); I != E; ++I)
    PruneTypes(*I, TypesToModify, ProcessedTypes);


  // Build up a set of structure types that we are going to modify, and
  // information describing how to modify them.
  map<const StructType*, vector<int> > Transforms;

  for (set<const StructType*>::iterator I = TypesToModify.begin(),
         E = TypesToModify.end(); I != E; ++I) {
    const StructType *ST = *I;
    unsigned NumElements = ST->getElementTypes().size();

    vector<int> &Transform = Transforms[ST];  // Fill in the map directly
    Transform.reserve(NumElements);

    // The transformation to do is: just simply swap the elements
    for (unsigned i = 0; i < NumElements; ++i)
      Transform.push_back(NumElements-i-1);
  }
  
  // Create the Worker to do our stuff for us...
  StructMutator = new MutateStructTypes(Transforms);
  
  // Do initial work.
  return StructMutator->doPassInitialization(M);
}

