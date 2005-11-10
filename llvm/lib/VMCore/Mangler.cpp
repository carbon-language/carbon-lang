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
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
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
  
  if (!UseQuotes) {
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
  } else {
    bool NeedsQuotes = false;
    
    // If X does not start with (char)1, add the prefix.
    std::string::const_iterator I = X.begin();
    if (*I != 1)
      Result = Prefix;
    else
      ++I;  // Skip over the marker.
    
    // If the first character is a number, we need quotes.
    if (*I >= '0' && *I <= '9')
      NeedsQuotes = true;
    
    for (std::string::const_iterator E = X.end(); I != E; ++I)
      if (*I == '"')
        Result += "_QQ_";
      else {
        if ((*I < 'a' || *I > 'z') && (*I < 'A' || *I > 'Z') &&
            (*I < '0' || *I > '9') && *I != '_' && *I != '$' && *I != '.')
          NeedsQuotes = true;
        Result += *I;
      }
    if (NeedsQuotes)
      Result = '"' + Result + '"';
  }
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
  if (const GlobalValue *GV = dyn_cast<GlobalValue>(V))
    return getValueName(GV);
  
  std::string &Name = Memo[V];
  if (!Name.empty())
    return Name;       // Return the already-computed name for V.
  
  // Always mangle local names.
  Name = "ltmp_" + utostr(Count++) + "_" + utostr(getTypeID(V->getType()));
  return Name;
}


std::string Mangler::getValueName(const GlobalValue *GV) {
  // Check to see whether we've already named V.
  std::string &Name = Memo[GV];
  if (!Name.empty())
    return Name;       // Return the already-computed name for V.

  // Name mangling occurs as follows:
  // - If V is an intrinsic function, do not change name at all
  // - Otherwise, mangling occurs if global collides with existing name.
  if (isa<Function>(GV) && cast<Function>(GV)->getIntrinsicID()) {
    Name = GV->getName(); // Is an intrinsic function
  } else if (!MangledGlobals.count(GV)) {
    Name = makeNameProper(GV->getName(), Prefix);
  } else {
    unsigned TypeUniqueID = getTypeID(GV->getType());
    Name = "l" + utostr(TypeUniqueID) + "_" + makeNameProper(GV->getName());
  }

  return Name;
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


Mangler::Mangler(Module &M, const char *prefix)
  : Prefix(prefix), UseQuotes(false), Count(0), TypeCounter(0) {
  // Calculate which global values have names that will collide when we throw
  // away type information.
  std::map<std::string, GlobalValue*> Names;
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    InsertName(I, Names);
  for (Module::global_iterator I = M.global_begin(), E = M.global_end(); I != E; ++I)
    InsertName(I, Names);
}
