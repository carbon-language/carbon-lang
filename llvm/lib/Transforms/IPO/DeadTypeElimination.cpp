//===- CleanupGCCOutput.cpp - Cleanup GCC Output --------------------------===//
//
// This pass is used to cleanup the output of GCC.  GCC's output is
// unneccessarily gross for a couple of reasons. This pass does the following
// things to try to clean it up:
//
// * Eliminate names for GCC types that we know can't be needed by the user.
// * Eliminate names for types that are unused in the entire translation unit
// * Fix various problems that we might have in PHI nodes and casts
//
// Note:  This code produces dead declarations, it is a good idea to run DCE
//        after this pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/CleanupGCCOutput.h"
#include "llvm/Analysis/FindUsedTypes.h"
#include "llvm/Module.h"
#include "llvm/SymbolTable.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iPHINode.h"
#include "llvm/iMemory.h"
#include "llvm/iTerminators.h"
#include "llvm/iOther.h"
#include "llvm/Support/CFG.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "Support/StatisticReporter.h"
#include <algorithm>
#include <iostream>

static Statistic<> NumTypeSymtabEntriesKilled("cleangcc\t- Number of unused typenames removed from symtab");

using std::vector;

namespace {
  struct CleanupGCCOutput : public FunctionPass {
    const char *getPassName() const { return "Cleanup GCC Output"; }

    // doPassInitialization - For this pass, it removes global symbol table
    // entries for primitive types.  These are never used for linking in GCC and
    // they make the output uglier to look at, so we nuke them.
    //
    // Also, initialize instance variables.
    //
    bool doInitialization(Module &M);

    // FIXME:
    // FIXME: This FunctionPass should be a PASS!
    // FIXME:
    bool runOnFunction(Function &F) { return false; }
    
    // doPassFinalization - Strip out type names that are unused by the program
    bool doFinalization(Module &M);
    
    // getAnalysisUsage - This function needs FindUsedTypes to do its job...
    //
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired(FindUsedTypes::ID);
    }
  };
}

Pass *createCleanupGCCOutputPass() {
  return new CleanupGCCOutput();
}



// ShouldNukSymtabEntry - Return true if this module level symbol table entry
// should be eliminated.
//
static inline bool ShouldNukeSymtabEntry(const std::pair<std::string,Value*>&E){
  // Nuke all names for primitive types!
  if (cast<Type>(E.second)->isPrimitiveType()) return true;

  // Nuke all pointers to primitive types as well...
  if (const PointerType *PT = dyn_cast<PointerType>(E.second))
    if (PT->getElementType()->isPrimitiveType()) return true;

  return false;
}

// doInitialization - For this pass, it removes global symbol table
// entries for primitive types.  These are never used for linking in GCC and
// they make the output uglier to look at, so we nuke them.
//
bool CleanupGCCOutput::doInitialization(Module &M) {
  bool Changed = false;

  if (SymbolTable *ST = M.getSymbolTable()) {
    // Check the symbol table for superfluous type entries...
    //
    // Grab the 'type' plane of the module symbol...
    SymbolTable::iterator STI = ST->find(Type::TypeTy);
    if (STI != ST->end()) {
      // Loop over all entries in the type plane...
      SymbolTable::VarMap &Plane = STI->second;
      for (SymbolTable::VarMap::iterator PI = Plane.begin(); PI != Plane.end();)
        if (ShouldNukeSymtabEntry(*PI)) {    // Should we remove this entry?
#if MAP_IS_NOT_BRAINDEAD
          PI = Plane.erase(PI);     // STD C++ Map should support this!
#else
          Plane.erase(PI);          // Alas, GCC 2.95.3 doesn't  *SIGH*
          PI = Plane.begin();
#endif
          ++NumTypeSymtabEntriesKilled;
          Changed = true;
        } else {
          ++PI;
        }
    }
  }

  return Changed;
}


bool CleanupGCCOutput::doFinalization(Module &M) {
  bool Changed = false;

  if (SymbolTable *ST = M.getSymbolTable()) {
    const std::set<const Type *> &UsedTypes =
      getAnalysis<FindUsedTypes>().getTypes();

    // Check the symbol table for superfluous type entries that aren't used in
    // the program
    //
    // Grab the 'type' plane of the module symbol...
    SymbolTable::iterator STI = ST->find(Type::TypeTy);
    if (STI != ST->end()) {
      // Loop over all entries in the type plane...
      SymbolTable::VarMap &Plane = STI->second;
      for (SymbolTable::VarMap::iterator PI = Plane.begin(); PI != Plane.end();)
        if (!UsedTypes.count(cast<Type>(PI->second))) {
#if MAP_IS_NOT_BRAINDEAD
          PI = Plane.erase(PI);     // STD C++ Map should support this!
#else
          Plane.erase(PI);          // Alas, GCC 2.95.3 doesn't  *SIGH*
          PI = Plane.begin();       // N^2 algorithms are fun.  :(
#endif
          Changed = true;
        } else {
          ++PI;
        }
    }
  }
  return Changed;
}
