//===- AliasSetTracker.cpp - Alias Sets Tracker implementation-------------===//
//
// This file implements the AliasSetTracker and AliasSet classes.
// 
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/iMemory.h"
#include "llvm/iOther.h"
#include "llvm/iTerminators.h"

/// updateAccessTypes - Depending on what type of accesses are in this set,
/// decide whether the set contains just references, just modifications, or a
/// mix.
///
void AliasSet::updateAccessType() {
  if (!Calls.empty() || !Invokes.empty()) {
    AccessTy = ModRef;
  } else if (!Loads.empty()) {
    if (Stores.empty())
      AccessTy = Refs;
    else
      AccessTy = ModRef;
  } else {
    AccessTy = Mods;
  }
}

/// mergeSetIn - Merge the specified alias set into this alias set...
///
void AliasSet::mergeSetIn(const AliasSet &AS) {
  // Merge instruction sets...
    Loads.insert(  Loads.end(), AS.Loads.begin()  , AS.Loads.end());
   Stores.insert( Stores.end(), AS.Stores.begin() , AS.Stores.end());
    Calls.insert(  Calls.end(), AS.Calls.begin()  , AS.Calls.end());
  Invokes.insert(Invokes.end(), AS.Invokes.begin(), AS.Invokes.end());

  // Update the alias and access types of this set...
  if (AS.getAliasType() == MayAlias)
    AliasTy = MayAlias;
  updateAccessType();
}

/// pointerAliasesSet - Return true if the specified pointer "may" (or must)
/// alias one of the members in the set.
///
bool AliasSet::pointerAliasesSet(const Value *Ptr, AliasAnalysis &AA) const {
  if (!Calls.empty() || !Invokes.empty())
    return true;
  for (unsigned i = 0, e = Loads.size(); i != e; ++i)
    if (AA.alias(Ptr, Loads[i]->getOperand(0)))
      return true;
  for (unsigned i = 0, e = Stores.size(); i != e; ++i)
    if (AA.alias(Ptr, Stores[i]->getOperand(1)))
      return true;
  return false;
}

/// getSomePointer - This method may only be called when the AliasType of the
/// set is MustAlias.  This is used to return any old pointer (which must alias
/// all other pointers in the set) so that the caller can decide whether to turn
/// this set into a may alias set or not.
///
Value *AliasSet::getSomePointer() const {
  assert(getAliasType() == MustAlias &&
         "Cannot call getSomePointer on a 'MayAlias' set!");
  assert(Calls.empty() && Invokes.empty() && "Call/invokes mean may alias!");

  if (!Loads.empty())
    return Loads[0]->getOperand(0);
  assert(!Stores.empty() && "There are no instructions in this set!");
  return Stores[0]->getOperand(1);
}



/// findAliasSetForPointer - Given a pointer, find the one alias set to put the
/// instruction referring to the pointer into.  If there are multiple alias sets
/// that may alias the pointer, merge them together and return the unified set.
///
AliasSet *AliasSetTracker::findAliasSetForPointer(const Value *Ptr) {
  AliasSet *FoundSet = 0;
  for (unsigned i = 0; i != AliasSets.size(); ++i) {
    if (AliasSets[i].pointerAliasesSet(Ptr, AA)) {
      if (FoundSet == 0) {  // If this is the first alias set ptr can go into...
        FoundSet = &AliasSets[i];   // Remember it.
      } else {              // Otherwise, we must merge the sets...
        FoundSet->mergeSetIn(AliasSets[i]);     // Merge in contents...
        AliasSets.erase(AliasSets.begin()+i);   // Remove the set...
        --i;                                    // Don't skip the next set
      }
    }
  }

  return FoundSet;
}


void AliasSetTracker::add(LoadInst *LI) {
  Value *Pointer = LI->getOperand(0);

  // Check to see if the loaded pointer aliases any sets...
  AliasSet *AS = findAliasSetForPointer(Pointer);
  if (AS) {
    AS->Loads.push_back(LI);
    // Check to see if we need to change this into a MayAlias set now...
    if (AS->getAliasType() == AliasSet::MustAlias)
      if (AA.alias(AS->getSomePointer(), Pointer) != AliasAnalysis::MustAlias)
        AS->AliasTy = AliasSet::MayAlias;
    AS->updateAccessType();
  } else {
    // Otherwise create a new alias set to hold the load...
    AliasSets.push_back(AliasSet());
    AliasSets.back().Loads.push_back(LI);
    AliasSets.back().AccessTy = AliasSet::Refs;
  }
}

void AliasSetTracker::add(StoreInst *SI) {
  Value *Pointer = SI->getOperand(1);

  // Check to see if the loaded pointer aliases any sets...
  AliasSet *AS = findAliasSetForPointer(Pointer);
  if (AS) {
    AS->Stores.push_back(SI);
    // Check to see if we need to change this into a MayAlias set now...
    if (AS->getAliasType() == AliasSet::MustAlias)
      if (AA.alias(AS->getSomePointer(), Pointer) != AliasAnalysis::MustAlias)
        AS->AliasTy = AliasSet::MayAlias;
    AS->updateAccessType();
  } else {
    // Otherwise create a new alias set to hold the load...
    AliasSets.push_back(AliasSet());
    AliasSets.back().Stores.push_back(SI);
    AliasSets.back().AccessTy = AliasSet::Mods;
  }
}


void AliasSetTracker::mergeAllSets() {
  if (AliasSets.size() < 2) return;  // Noop

  // Merge all of the sets into set #0
  for (unsigned i = 1, e = AliasSets.size(); i != e; ++i)
    AliasSets[0].mergeSetIn(AliasSets[i]);

  // Delete extraneous sets...
  AliasSets.erase(AliasSets.begin()+1, AliasSets.end());
}

void AliasSetTracker::add(CallInst *CI) {
  if (!AliasSets.empty()) {
    mergeAllSets();
  } else {
    AliasSets.push_back(AliasSet());
  }
  AliasSets[0].AccessTy = AliasSet::ModRef;
  AliasSets[0].AliasTy = AliasSet::MayAlias;
  AliasSets[0].Calls.push_back(CI);
}

void AliasSetTracker::add(InvokeInst *II) {
  if (!AliasSets.empty()) {
    mergeAllSets();
  } else {
    AliasSets.push_back(AliasSet());
  }
  AliasSets[0].AccessTy = AliasSet::ModRef;
  AliasSets[0].AliasTy = AliasSet::MayAlias;
  AliasSets[0].Invokes.push_back(II);
}
