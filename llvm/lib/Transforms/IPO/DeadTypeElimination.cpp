//===- DeadTypeElimination.cpp - Eliminate unused types for symbol table --===//
//
// This pass is used to cleanup the output of GCC.  It eliminate names for types
// that are unused in the entire translation unit, using the FindUsedTypes pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/Analysis/FindUsedTypes.h"
#include "llvm/Module.h"
#include "llvm/SymbolTable.h"
#include "llvm/DerivedTypes.h"
#include "Support/StatisticReporter.h"

using std::vector;

namespace {
  struct DTE : public Pass {
    // doPassInitialization - For this pass, it removes global symbol table
    // entries for primitive types.  These are never used for linking in GCC and
    // they make the output uglier to look at, so we nuke them.
    //
    // Also, initialize instance variables.
    //
    bool run(Module &M);

    // getAnalysisUsage - This function needs FindUsedTypes to do its job...
    //
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired(FindUsedTypes::ID);
    }
  };
  RegisterPass<DTE> X("deadtypeelim", "Dead Type Elimination");
  Statistic<> NumKilled("deadtypeelim\t- Number of unused typenames removed from symtab");
}

Pass *createDeadTypeEliminationPass() {
  return new DTE();
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

// run - For this pass, it removes global symbol table entries for primitive
// types.  These are never used for linking in GCC and they make the output
// uglier to look at, so we nuke them.  Also eliminate types that are never used
// in the entire program as indicated by FindUsedTypes.
//
bool DTE::run(Module &M) {
  bool Changed = false;

  if (SymbolTable *ST = M.getSymbolTable()) {
    const std::set<const Type *> &UsedTypes =
      getAnalysis<FindUsedTypes>().getTypes();

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
          ++NumKilled;
          Changed = true;
        } else if (!UsedTypes.count(cast<Type>(PI->second))) {
#if MAP_IS_NOT_BRAINDEAD
          PI = Plane.erase(PI);     // STD C++ Map should support this!
#else
          Plane.erase(PI);          // Alas, GCC 2.95.3 doesn't  *SIGH*
          PI = Plane.begin();       // N^2 algorithms are fun.  :(
#endif
          ++NumKilled;
          Changed = true;
        } else {
          ++PI;
        }
    }
  }

  return Changed;
}
