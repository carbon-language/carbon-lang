//===-- Mangler.cpp - Self-contained c/asm llvm name mangler --------------===//
//
// Unified name mangler for CWriter and assembly backends.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <set>
#include <string>
#include "llvm/Value.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "Support/StringExtras.h"
#include "llvm/Support/Mangler.h"

/// makeNameProper - We don't want identifier names with ., space, or
/// - in them, so we mangle these characters into the strings "d_",
/// "s_", and "D_", respectively.
/// 
std::string Mangler::makeNameProper(std::string x) {
  std::string tmp;
  for (std::string::iterator sI = x.begin(), sEnd = x.end(); sI != sEnd; sI++)
    switch (*sI) {
    case '.': tmp += "d_"; break;
    case ' ': tmp += "s_"; break;
    case '-': tmp += "D_"; break;
    default:  tmp += *sI;
    }
  return tmp;
}

std::string Mangler::getValueName(const Value *V) {
  // Check to see whether we've already named V.
  ValueMap::iterator VI = Memo.find(V);
  if (VI != Memo.end()) {
    return VI->second; // Return the old name for V.
  }

  std::string name;
  if (V->hasName()) { // Print out the label if it exists...
    // Name mangling occurs as follows:
    // - If V is not a global, mangling always occurs.
    // - Otherwise, mangling occurs when any of the following are true:
    //   1) V has internal linkage
    //   2) V's name would collide if it is not mangled.
    //
    const GlobalValue* gv = dyn_cast<GlobalValue>(V);
    if(gv && !gv->hasInternalLinkage() && !MangledGlobals.count(gv)) {
      name = makeNameProper(gv->getName());
    } else {
      // Non-global, or global with internal linkage / colliding name
      // -> mangle.
      name = "l" + utostr(V->getType()->getUniqueID()) + "_" +
        makeNameProper(V->getName());      
    }
  } else {
    name = "ltmp_" + itostr(Count++) + "_"
      + utostr(V->getType()->getUniqueID());
  }
  Memo[V] = name;
  return name;
}

Mangler::Mangler(Module &_M) : M(_M)
{
  // Calculate which global values have names that will collide when we throw
  // away type information.
  std::set<std::string> FoundNames;
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (I->hasName())                      // If the global has a name...
      if (FoundNames.count(I->getName()))  // And the name is already used
        MangledGlobals.insert(I);          // Mangle the name
      else
        FoundNames.insert(I->getName());   // Otherwise, keep track of name

  for (Module::giterator I = M.gbegin(), E = M.gend(); I != E; ++I)
    if (I->hasName())                      // If the global has a name...
      if (FoundNames.count(I->getName()))  // And the name is already used
        MangledGlobals.insert(I);          // Mangle the name
      else
        FoundNames.insert(I->getName());   // Otherwise, keep track of name
}

