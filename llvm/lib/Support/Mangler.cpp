//===-- Mangler.cpp - Self-contained c/asm llvm name mangler --------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Unified name mangler for CWriter and assembly backends.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Mangler.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "Support/StringExtras.h"
using namespace llvm;

static char HexDigit(int V) {
  return V < 10 ? V+'0' : V+'A'-10;
}

static std::string MangleLetter(unsigned char C) {
  return std::string("_")+HexDigit(C >> 4) + HexDigit(C & 15) + "_";
}

/// makeNameProper - We don't want identifier names non-C-identifier characters
/// in them, so mangle them as appropriate.
/// 
std::string Mangler::makeNameProper(const std::string &X) {
  std::string Result;
  
  // Mangle the first letter specially, don't allow numbers...
  if ((X[0] < 'a' || X[0] > 'z') && (X[0] < 'A' || X[0] > 'Z') && X[0] != '_')
    Result += MangleLetter(X[0]);
  else
    Result += X[0];

  for (std::string::const_iterator I = X.begin()+1, E = X.end(); I != E; ++I)
    if ((*I < 'a' || *I > 'z') && (*I < 'A' || *I > 'Z') &&
        (*I < '0' || *I > '9') && *I != '_')
      Result += MangleLetter(*I);
    else
      Result += *I;
  return Result;
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
    if (gv && !gv->hasInternalLinkage() && !MangledGlobals.count(gv)) {
      name = makeNameProper(gv->getName());
      if (AddUnderscorePrefix) name = "_" + name;
    } else {
      // Non-global, or global with internal linkage / colliding name
      // -> mangle.
      name = "l" + utostr(V->getType()->getUniqueID()) + "_" +
        makeNameProper(V->getName());      
    }
  } else {
    name = "ltmp_" + utostr(Count++) + "_"
      + utostr(V->getType()->getUniqueID());
  }
  
  Memo[V] = name;
  return name;
}

Mangler::Mangler(Module &m, bool addUnderscorePrefix)
  : M(m), AddUnderscorePrefix(addUnderscorePrefix) {
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
