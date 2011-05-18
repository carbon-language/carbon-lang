//===-- lib/Support/StandardPasses.cpp - Standard pass lists -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines utility functions for creating a "standard" set of
// optimization passes, so that compilers and tools which use optimization
// passes use the same set of standard passes.
//
// This allows the creation of multiple standard sets, and their later
// modification by plugins and front ends.
//
//===----------------------------------------------------------------------===//

#include "llvm/PassManager.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/DefaultPasses.h"
#include "llvm/Support/Mutex.h"

using namespace llvm::DefaultStandardPasses;
using namespace llvm;

namespace {

/// Entry in the standard passes list.
struct StandardPassEntry {
  /// Function called to create the pass
  PassInfo::NormalCtor_t createPass;
  /// Unique identifier for this pass
  unsigned char *passID;
  /// Flags specifying when this pass should be run
  unsigned flags;

  StandardPassEntry(PassInfo::NormalCtor_t constructor, unsigned char *ID,
      unsigned f) : createPass(constructor), passID(ID), flags(f) {};
};

/// Standard alias analysis passes
static llvm::SmallVector<StandardPassEntry, 4> AAPasses;
/// Standard function passes
static llvm::SmallVector<StandardPassEntry, 32> FunctionPasses;
/// Standard module passes
static llvm::SmallVector<StandardPassEntry, 32> ModulePasses;
/// Standard link-time optimization passes
static llvm::SmallVector<StandardPassEntry, 32> LTOPasses;

/// Entry in the unresolved standard pass list.  IF a pass is inserted in front
/// of a pass that is not yet registered in the standard pass list then it is
/// stored in a separate list and resolved later.
struct UnresolvedStandardPass : public StandardPassEntry {
  /// The set into which this is stored
  StandardPass::StandardSet set;
  /// The unique ID of the pass that should follow this one in the sequence
  unsigned char *next;
  UnresolvedStandardPass(PassInfo::NormalCtor_t constructor,
                         unsigned char *newPass,
                         unsigned char *oldPass,
                         StandardPass::StandardSet s,
                         unsigned f) :
    StandardPassEntry(constructor, newPass, f), set(s), next(oldPass) {}
};

/// The passes that can not be inserted into the correct lists yet because of
/// their place in the sequence.
static llvm::SmallVector<UnresolvedStandardPass, 16> UnresolvedPasses;

/// Returns a reference to the pass list for the corresponding set of
/// optimisations.
llvm::SmallVectorImpl<StandardPassEntry>&
PassList(StandardPass::StandardSet set) {
  switch (set) {
    case StandardPass::AliasAnalysis: return AAPasses;
    case StandardPass::Function: return FunctionPasses;
    case StandardPass::Module: return ModulePasses;
    case StandardPass::LTO: return LTOPasses; 
  }
  // We could use a map of standard pass lists to allow definition of new
  // default sets
  llvm_unreachable("Invalid standard optimization set requested");
}

static ManagedStatic<sys::SmartMutex<true> > Lock;

/// Registers the default set of standard passes.  This is called lazily when
/// an attempt is made to read or modify the standard pass list
void RegisterDefaultStandardPasses(void(*doRegister)(void)) {
  // Only initialize the standard passes once
  static volatile bool initialized = false;
  if (initialized) return;

  llvm::sys::SmartScopedLock<true> Guard(*Lock);
  if (initialized) return;
  if (doRegister) {
    assert("No passes registered before setting default passes" &&
            AAPasses.size() == 0 &&
            FunctionPasses.size() == 0 &&
            LTOPasses.size() == 0 &&
            ModulePasses.size() == 0);

    // We must set initialized to true before calling this function, because
    // the doRegister() function will probably call RegisterDefaultPasses(),
    // which will call this function, and we'd end up with infinite recursion
    // and breakage if we didn't.
    initialized = true;
    doRegister();
  }
}

} // Anonymous namespace

void (*StandardPass::RegisterDefaultPasses)(void);
Pass* (*StandardPass::CreateVerifierPass)(void);

void StandardPass::RegisterDefaultPass(PassInfo::NormalCtor_t constructor,
                                       unsigned char *newPass,
                                       unsigned char *oldPass,
                                       StandardPass::StandardSet set,
                                       unsigned flags) {
  // Make sure that the standard sets are already regstered
  RegisterDefaultStandardPasses(RegisterDefaultPasses);
  // Get the correct list to modify
  llvm::SmallVectorImpl<StandardPassEntry>& passList = PassList(set);

  // If there is no old pass specified, then we are adding a new final pass, so
  // just push it onto the end.
  if (!oldPass) {
    StandardPassEntry pass(constructor, newPass, flags);
    passList.push_back(pass);
    return;
  }

  // Find the correct place to insert the pass.  This is a linear search, but
  // this shouldn't be too slow since the SmallVector will store the values in
  // a contiguous block of memory.  Each entry is just three words of memory, so
  // in most cases we are only going to be looking in one or two cache lines.
  // The extra memory accesses from a more complex search structure would
  // offset any performance gain (unless someone decides to add an insanely
  // large set of standard passes to a set)
  for (SmallVectorImpl<StandardPassEntry>::iterator i=passList.begin(),
       e=passList.end() ; i!=e ; ++i) {
    if (i->passID == oldPass) {
      StandardPassEntry pass(constructor, newPass, flags);
      passList.insert(i, pass);
      // If we've added a new pass, then there may have gained the ability to
      // insert one of the previously unresolved ones.  If so, insert the new
      // one.
      for (SmallVectorImpl<UnresolvedStandardPass>::iterator
          u=UnresolvedPasses.begin(), eu=UnresolvedPasses.end() ; u!=eu ; ++u){
        if (u->next == newPass && u->set == set) {
          UnresolvedStandardPass p = *u;
          UnresolvedPasses.erase(u);
          RegisterDefaultPass(p.createPass, p.passID, p.next, p.set, p.flags);
        }
      }
      return;
    }
  }
  // If we get to here, then we didn't find the correct place to insert the new
  // pass
  UnresolvedStandardPass pass(constructor, newPass, oldPass, set, flags);
  UnresolvedPasses.push_back(pass);
}

void StandardPass::AddPassesFromSet(PassManagerBase *PM,
                                    StandardSet set,
                                    unsigned flags,
                                    bool VerifyEach,
                                    Pass *inliner) {
  RegisterDefaultStandardPasses(RegisterDefaultPasses);
  unsigned level = OptimizationLevel(flags);
  flags = RequiredFlags(flags);
  llvm::SmallVectorImpl<StandardPassEntry>& passList = PassList(set);

  // Add all of the passes from this set
  for (SmallVectorImpl<StandardPassEntry>::iterator i=passList.begin(),
       e=passList.end() ; i!=e ; ++i) {
    // Skip passes that don't have conditions that match the ones specified
    // here.  For a pass to match:
    // - Its minimum optimisation level must be less than or equal to the
    //   specified level.
    // - Its maximum optimisation level must be greater than or equal to the
    //   specified level
    // - All of its required flags must be set
    // - None of its disallowed flags may be set
    if ((level >= OptimizationLevel(i->flags)) &&
        ((level <= MaxOptimizationLevel(i->flags))
          || MaxOptimizationLevel(i->flags) == 0)  &&
        ((RequiredFlags(i->flags) & flags) == RequiredFlags(i->flags)) &&
        ((DisallowedFlags(i->flags) & flags) == 0)) {
      // This is quite an ugly way of allowing us to specify an inliner pass to
      // insert.  Ideally, we'd replace this with a general mechanism allowing
      // callers to replace arbitrary passes in the list.
      Pass *p = 0;
      if (&InlinerPlaceholderID == i->passID) {
          p = inliner;
      } else if (i->createPass)
        p = i->createPass();
      if (p) {
        PM->add(p);
        if (VerifyEach)
          PM->add(CreateVerifierPass());
      }
    }
  }
}

unsigned char DefaultStandardPasses::AggressiveDCEID;
unsigned char DefaultStandardPasses::ArgumentPromotionID;
unsigned char DefaultStandardPasses::BasicAliasAnalysisID;
unsigned char DefaultStandardPasses::CFGSimplificationID;
unsigned char DefaultStandardPasses::ConstantMergeID;
unsigned char DefaultStandardPasses::CorrelatedValuePropagationID;
unsigned char DefaultStandardPasses::DeadArgEliminationID;
unsigned char DefaultStandardPasses::DeadStoreEliminationID;
unsigned char DefaultStandardPasses::DeadTypeEliminationID;
unsigned char DefaultStandardPasses::EarlyCSEID;
unsigned char DefaultStandardPasses::FunctionAttrsID;
unsigned char DefaultStandardPasses::FunctionInliningID;
unsigned char DefaultStandardPasses::GVNID;
unsigned char DefaultStandardPasses::GlobalDCEID;
unsigned char DefaultStandardPasses::GlobalOptimizerID;
unsigned char DefaultStandardPasses::GlobalsModRefID;
unsigned char DefaultStandardPasses::IPSCCPID;
unsigned char DefaultStandardPasses::IndVarSimplifyID;
unsigned char DefaultStandardPasses::InlinerPlaceholderID;
unsigned char DefaultStandardPasses::InstructionCombiningID;
unsigned char DefaultStandardPasses::JumpThreadingID;
unsigned char DefaultStandardPasses::LICMID;
unsigned char DefaultStandardPasses::LoopDeletionID;
unsigned char DefaultStandardPasses::LoopIdiomID;
unsigned char DefaultStandardPasses::LoopRotateID;
unsigned char DefaultStandardPasses::LoopUnrollID;
unsigned char DefaultStandardPasses::LoopUnswitchID;
unsigned char DefaultStandardPasses::MemCpyOptID;
unsigned char DefaultStandardPasses::PruneEHID;
unsigned char DefaultStandardPasses::ReassociateID;
unsigned char DefaultStandardPasses::SCCPID;
unsigned char DefaultStandardPasses::ScalarReplAggregatesID;
unsigned char DefaultStandardPasses::SimplifyLibCallsID;
unsigned char DefaultStandardPasses::StripDeadPrototypesID;
unsigned char DefaultStandardPasses::TailCallEliminationID;
unsigned char DefaultStandardPasses::TypeBasedAliasAnalysisID;
