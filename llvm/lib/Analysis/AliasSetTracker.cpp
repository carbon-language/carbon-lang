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
#include "llvm/Pass.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/InstIterator.h"

/// mergeSetIn - Merge the specified alias set into this alias set...
///
void AliasSet::mergeSetIn(AliasSet &AS) {
  assert(!AS.Forward && "Alias set is already forwarding!");
  assert(!Forward && "This set is a forwarding set!!");

  // Update the alias and access types of this set...
  AccessTy |= AS.AccessTy;
  AliasTy  |= AS.AliasTy;

  if (CallSites.empty()) {            // Merge call sites...
    if (!AS.CallSites.empty())
      std::swap(CallSites, AS.CallSites);
  } else if (!AS.CallSites.empty()) {
    CallSites.insert(CallSites.end(), AS.CallSites.begin(), AS.CallSites.end());
    AS.CallSites.clear();
  }
  
  // FIXME: If AS's refcount is zero, nuke it now...
  assert(RefCount != 0);

  AS.Forward = this;  // Forward across AS now...
  RefCount++;         // AS is now pointing to us...

  // Merge the list of constituent pointers...
  PtrListTail->second.setTail(AS.PtrListHead);
  PtrListTail = AS.PtrListTail;
  AS.PtrListHead = AS.PtrListTail = 0;
}

void AliasSetTracker::removeAliasSet(AliasSet *AS) {
  AliasSets.erase(AS);
}

void AliasSet::removeFromTracker(AliasSetTracker &AST) {
  assert(RefCount == 0 && "Cannot remove non-dead alias set from tracker!");
  AST.removeAliasSet(this);
}

void AliasSet::addPointer(AliasSetTracker &AST, HashNodePair &Entry,
                          unsigned Size) {
  assert(!Entry.second.hasAliasSet() && "Entry already in set!");

  AliasAnalysis &AA = AST.getAliasAnalysis();

  if (isMustAlias())    // Check to see if we have to downgrade to _may_ alias
    if (HashNodePair *P = getSomePointer()) {
      AliasAnalysis::AliasResult Result =
        AA.alias(P->first, P->second.getSize(), Entry.first, Size);
      if (Result == AliasAnalysis::MayAlias)
        AliasTy = MayAlias;
      else                  // First entry of must alias must have maximum size!
        P->second.updateSize(Size);
      assert(Result != AliasAnalysis::NoAlias && "Cannot be part of must set!");
    }

  Entry.second.setAliasSet(this);
  Entry.second.updateSize(Size);

  // Add it to the end of the list...
  if (PtrListTail)
    PtrListTail->second.setTail(&Entry);
  else
    PtrListHead = &Entry;
  PtrListTail = &Entry;
  RefCount++;               // Entry points to alias set...
}

void AliasSet::addCallSite(CallSite CS) {
  CallSites.push_back(CS);
  AliasTy = MayAlias;         // FIXME: Too conservative?
}

/// aliasesPointer - Return true if the specified pointer "may" (or must)
/// alias one of the members in the set.
///
bool AliasSet::aliasesPointer(const Value *Ptr, unsigned Size,
                              AliasAnalysis &AA) const {
  if (AliasTy == MustAlias) {
    assert(CallSites.empty() && "Illegal must alias set!");

    // If this is a set of MustAliases, only check to see if the pointer aliases
    // SOME value in the set...
    HashNodePair *SomePtr = getSomePointer();
    assert(SomePtr && "Empty must-alias set??");
    return AA.alias(SomePtr->first, SomePtr->second.getSize(), Ptr, Size);
  }

  // If this is a may-alias set, we have to check all of the pointers in the set
  // to be sure it doesn't alias the set...
  for (iterator I = begin(), E = end(); I != E; ++I)
    if (AA.alias(Ptr, Size, I->first, I->second.getSize()))
      return true;

  // Check the call sites list and invoke list...
  if (!CallSites.empty())
    // FIXME: this is pessimistic!
    return true;

  return false;
}

bool AliasSet::aliasesCallSite(CallSite CS, AliasAnalysis &AA) const {
  // FIXME: Too conservative!
  return true;
}


/// findAliasSetForPointer - Given a pointer, find the one alias set to put the
/// instruction referring to the pointer into.  If there are multiple alias sets
/// that may alias the pointer, merge them together and return the unified set.
///
AliasSet *AliasSetTracker::findAliasSetForPointer(const Value *Ptr,
                                                  unsigned Size) {
  AliasSet *FoundSet = 0;
  for (iterator I = begin(), E = end(); I != E; ++I)
    if (!I->Forward && I->aliasesPointer(Ptr, Size, AA)) {
      if (FoundSet == 0) {  // If this is the first alias set ptr can go into...
        FoundSet = I;       // Remember it.
      } else {              // Otherwise, we must merge the sets...
        FoundSet->mergeSetIn(*I);     // Merge in contents...
      }
    }

  return FoundSet;
}

AliasSet *AliasSetTracker::findAliasSetForCallSite(CallSite CS) {
  AliasSet *FoundSet = 0;
  for (iterator I = begin(), E = end(); I != E; ++I)
    if (!I->Forward && I->aliasesCallSite(CS, AA)) {
      if (FoundSet == 0) {  // If this is the first alias set ptr can go into...
        FoundSet = I;       // Remember it.
      } else if (!I->Forward) {     // Otherwise, we must merge the sets...
        FoundSet->mergeSetIn(*I);     // Merge in contents...
      }
    }

  return FoundSet;
}




/// getAliasSetForPointer - Return the alias set that the specified pointer
/// lives in...
AliasSet &AliasSetTracker::getAliasSetForPointer(Value *Pointer, unsigned Size){
  AliasSet::HashNodePair &Entry = getEntryFor(Pointer);

  // Check to see if the pointer is already known...
  if (Entry.second.hasAliasSet() && Size <= Entry.second.getSize()) {
    // Return the set!
    return *Entry.second.getAliasSet(*this)->getForwardedTarget(*this);
  } else if (AliasSet *AS = findAliasSetForPointer(Pointer, Size)) {
    // Add it to the alias set it aliases...
    AS->addPointer(*this, Entry, Size);
    return *AS;
  } else {
    // Otherwise create a new alias set to hold the loaded pointer...
    AliasSets.push_back(AliasSet());
    AliasSets.back().addPointer(*this, Entry, Size);
    return AliasSets.back();
  }
}

void AliasSetTracker::add(LoadInst *LI) {
  addPointer(LI->getOperand(0),
             AA.getTargetData().getTypeSize(LI->getType()), AliasSet::Refs);
}

void AliasSetTracker::add(StoreInst *SI) {
  addPointer(SI->getOperand(1),
             AA.getTargetData().getTypeSize(SI->getOperand(0)->getType()),
             AliasSet::Mods);
}

void AliasSetTracker::add(CallSite CS) {
  AliasSet *AS = findAliasSetForCallSite(CS);
  if (!AS) {
    AliasSets.push_back(AliasSet());
    AS = &AliasSets.back();
  }
  AS->addCallSite(CS); 
}

void AliasSetTracker::add(Instruction *I) {
  // Dispatch to one of the other add methods...
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    add(LI);
  else if (StoreInst *SI = dyn_cast<StoreInst>(I))
    add(SI);
  else if (CallInst *CI = dyn_cast<CallInst>(I))
    add(CI);
  else if (InvokeInst *II = dyn_cast<InvokeInst>(I))
    add(II);
}

void AliasSetTracker::add(BasicBlock &BB) {
  for (BasicBlock::iterator I = BB.begin(), E = BB.end(); I != E; ++I)
    add(I);
}

void AliasSetTracker::add(const AliasSetTracker &AST) {
  assert(&AA == &AST.AA &&
         "Merging AliasSetTracker objects with different Alias Analyses!");

  // Loop over all of the alias sets in AST, adding the pointers contained
  // therein into the current alias sets.  This can cause alias sets to be
  // merged together in the current AST.
  for (const_iterator I = AST.begin(), E = AST.end(); I != E; ++I)
    if (!I->Forward) {   // Ignore forwarding alias sets
      AliasSet &AS = const_cast<AliasSet&>(*I);

      // If there are any call sites in the alias set, add them to this AST.
      for (unsigned i = 0, e = AS.CallSites.size(); i != e; ++i)
        add(AS.CallSites[i]);

      // Loop over all of the pointers in this alias set...
      AliasSet::iterator I = AS.begin(), E = AS.end();
      for (; I != E; ++I)
        addPointer(I->first, I->second.getSize(),
                   (AliasSet::AccessType)AS.AccessTy);
    }
}

//===----------------------------------------------------------------------===//
//               AliasSet/AliasSetTracker Printing Support
//===----------------------------------------------------------------------===//

void AliasSet::print(std::ostream &OS) const {
  OS << "  AliasSet[" << (void*)this << "," << RefCount << "] ";
  OS << (AliasTy == MustAlias ? "must" : "may ") << " alias, ";
  switch (AccessTy) {
  case NoModRef: OS << "No access "; break;
  case Refs    : OS << "Ref       "; break;
  case Mods    : OS << "Mod       "; break;
  case ModRef  : OS << "Mod/Ref   "; break;
  default: assert(0 && "Bad value for AccessTy!");
  }
  if (Forward)
    OS << " forwarding to " << (void*)Forward;


  if (begin() != end()) {
    OS << "Pointers: ";
    for (iterator I = begin(), E = end(); I != E; ++I) {
      if (I != begin()) OS << ", ";
      WriteAsOperand(OS << "(", I->first);
      OS << ", " << I->second.getSize() << ")";
    }
  }
  if (!CallSites.empty()) {
    OS << "\n    " << CallSites.size() << " Call Sites: ";
    for (unsigned i = 0, e = CallSites.size(); i != e; ++i) {
      if (i) OS << ", ";
      WriteAsOperand(OS, CallSites[i].getCalledValue());
    }      
  }
  OS << "\n";
}

void AliasSetTracker::print(std::ostream &OS) const {
  OS << "Alias Set Tracker: " << AliasSets.size() << " alias sets for "
     << PointerMap.size() << " pointer values.\n";
  for (const_iterator I = begin(), E = end(); I != E; ++I)
    I->print(OS);
  OS << "\n";
}

void AliasSet::dump() const { print (std::cerr); }
void AliasSetTracker::dump() const { print(std::cerr); }


//===----------------------------------------------------------------------===//
//                            AliasSetPrinter Pass
//===----------------------------------------------------------------------===//

namespace {
  class AliasSetPrinter : public FunctionPass {
    AliasSetTracker *Tracker;
  public:
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
      AU.addRequired<AliasAnalysis>();
    }

    virtual bool runOnFunction(Function &F) {
      Tracker = new AliasSetTracker(getAnalysis<AliasAnalysis>());

      for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I)
        Tracker->add(*I);
      return false;
    }

    /// print - Convert to human readable form
    virtual void print(std::ostream &OS) const {
      Tracker->print(OS);
    }

    virtual void releaseMemory() {
      delete Tracker;
    }
  };
  RegisterPass<AliasSetPrinter> X("print-alias-sets", "Alias Set Printer",
                                  PassInfo::Analysis | PassInfo::Optimization);
}
