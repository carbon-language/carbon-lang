//===- CleanupGCCOutput.cpp - Cleanup GCC Output ----------------------------=//
//
// This pass is used to cleanup the output of GCC.  GCC's output is
// unneccessarily gross for a couple of reasons. This pass does the following
// things to try to clean it up:
//
// Note:  This code produces dead declarations, it is a good idea to run DCE
//        after this pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/CleanupGCCOutput.h"
#include "llvm/SymbolTable.h"

static inline bool ShouldNukeSymtabEntry(const pair<string, Value*> &E) {
  // Nuke all names for primitive types!
  if (cast<Type>(E.second)->isPrimitiveType()) return true;

  // The only types that could contain .'s in the program are things generated
  // by GCC itself, including "complex.float" and friends.  Nuke them too.
  if (E.first.find('.') != string::npos) return true;

  return false;
}


// doPassInitialization - For this pass, it removes global symbol table
// entries for primitive types.  These are never used for linking in GCC and
// they make the output uglier to look at, so we nuke them.
//
bool CleanupGCCOutput::doPassInitialization(Module *M) {
  bool Changed = false;

  if (M->hasSymbolTable()) {
    // Grab the type plane of the module...
    SymbolTable *ST = M->getSymbolTable();
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
          Changed = true;
        } else {
          ++PI;
        }
    }
    
  }

  return Changed;
}


// doPerMethodWork - This method simplifies the specified method hopefully.
//
bool CleanupGCCOutput::doPerMethodWork(Method *M) {
  bool Changed = false;

  return Changed;
}
