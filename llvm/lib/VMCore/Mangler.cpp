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
#include "llvm/ADT/StringExtras.h"
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
std::string Mangler::makeNameProper(const std::string &X, const char *Prefix) {
  std::string Result;

  // If X does not start with (char)1, add the prefix.
  std::string::const_iterator I = X.begin();
  if (*I != 1)
    Result = Prefix;
  else
    ++I;  // Skip over the marker.
  
  // Mangle the first letter specially, don't allow numbers...
  if (*I >= '0' && *I <= '9')
    Result += MangleLetter(*I++);

  for (std::string::const_iterator E = X.end(); I != E; ++I)
    if ((*I < 'a' || *I > 'z') && (*I < 'A' || *I > 'Z') &&
        (*I < '0' || *I > '9') && *I != '_' && *I != '$')
      Result += MangleLetter(*I);
    else
      Result += *I;
  return Result;
}

/// getTypeID - Return a unique ID for the specified LLVM type.
///
unsigned Mangler::getTypeID(const Type *Ty) {
  unsigned &E = TypeMap[Ty];
  if (E == 0) E = ++TypeCounter;
  return E;
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
    // - If V is an intrinsic function, do not change name at all
    // - If V is not a global, mangling always occurs.
    // - Otherwise, mangling occurs when any of the following are true:
    //   1) V has internal linkage
    //   2) V's name would collide if it is not mangled.
    //
    const GlobalValue* gv = dyn_cast<GlobalValue>(V);
    if (gv && isa<Function>(gv) && cast<Function>(gv)->getIntrinsicID()) {
      name = gv->getName(); // Is an intrinsic function
    } else if (gv && !gv->hasInternalLinkage() && !MangledGlobals.count(gv)) {
      name = makeNameProper(gv->getName(), Prefix);
    } else {
      // Non-global, or global with internal linkage / colliding name
      // -> mangle.
      unsigned TypeUniqueID = getTypeID(V->getType());
      name = "l" + utostr(TypeUniqueID) + "_" + makeNameProper(V->getName());
    }
  } else {
    name = "ltmp_" + utostr(Count++) + "_" + utostr(getTypeID(V->getType()));
  }

  Memo[V] = name;
  return name;
}

void Mangler::InsertName(GlobalValue *GV,
                         std::map<std::string, GlobalValue*> &Names) {
  if (!GV->hasName()) {   // We must mangle unnamed globals.
    MangledGlobals.insert(GV);
    return;
  }

  // Figure out if this is already used.
  GlobalValue *&ExistingValue = Names[GV->getName()];
  if (!ExistingValue) {
    ExistingValue = GV;
  } else {
    // If GV is external but the existing one is static, mangle the existing one
    if (GV->hasExternalLinkage() && !ExistingValue->hasExternalLinkage()) {
      MangledGlobals.insert(ExistingValue);
      ExistingValue = GV;
    } else {
      // Otherwise, mangle GV
      MangledGlobals.insert(GV);
    }
  }
}


Mangler::Mangler(Module &m, const char *prefix)
  : M(m), Prefix(prefix), TypeCounter(0), Count(0) {
  // Calculate which global values have names that will collide when we throw
  // away type information.
  std::map<std::string, GlobalValue*> Names;
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    InsertName(I, Names);
  for (Module::global_iterator I = M.global_begin(), E = M.global_end(); I != E; ++I)
    InsertName(I, Names);
}
