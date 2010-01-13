//===-- Mangler.cpp - Self-contained c/asm llvm name mangler --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Unified name mangler for CWriter and assembly backends.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Mangler.h"
#include "llvm/Function.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

static char HexDigit(int V) {
  return V < 10 ? V+'0' : V+'A'-10;
}

static std::string MangleLetter(unsigned char C) {
  char Result[] = { '_', HexDigit(C >> 4), HexDigit(C & 15), '_', 0 };
  return Result;
}

/// makeNameProper - We don't want identifier names non-C-identifier characters
/// in them, so mangle them as appropriate.
///
std::string Mangler::makeNameProper(StringRef X,
                                    ManglerPrefixTy PrefixTy) {
  assert(!X.empty() && "Cannot mangle empty strings");
  
  if (!UseQuotes) {
    std::string Result;

    // If X does not start with (char)1, add the prefix.
    bool NeedPrefix = true;
    StringRef::iterator I = X.begin();
    if (*I == 1) {
      NeedPrefix = false;
      ++I;  // Skip over the marker.
    }
    
    // Mangle the first letter specially, don't allow numbers unless the target
    // explicitly allows them.
    if (!SymbolsCanStartWithDigit && *I >= '0' && *I <= '9')
      Result += MangleLetter(*I++);

    for (StringRef::iterator E = X.end(); I != E; ++I) {
      if (!isCharAcceptable(*I))
        Result += MangleLetter(*I);
      else
        Result += *I;
    }

    if (NeedPrefix) {
      Result = Prefix + Result;

      if (PrefixTy == Mangler::Private)
        Result = PrivatePrefix + Result;
      else if (PrefixTy == Mangler::LinkerPrivate)
        Result = LinkerPrivatePrefix + Result;
    }

    return Result;
  }

  bool NeedPrefix = true;
  bool NeedQuotes = false;
  std::string Result;    
  StringRef::iterator I = X.begin();
  if (*I == 1) {
    NeedPrefix = false;
    ++I;  // Skip over the marker.
  }

  // If the first character is a number, we need quotes.
  if (*I >= '0' && *I <= '9')
    NeedQuotes = true;
    
  // Do an initial scan of the string, checking to see if we need quotes or
  // to escape a '"' or not.
  if (!NeedQuotes)
    for (StringRef::iterator E = X.end(); I != E; ++I)
      if (!isCharAcceptable(*I)) {
        NeedQuotes = true;
        break;
      }
    
  // In the common case, we don't need quotes.  Handle this quickly.
  if (!NeedQuotes) {
    if (!NeedPrefix)
      return X.substr(1);   // Strip off the \001.
    
    Result = Prefix + X.str();

    if (PrefixTy == Mangler::Private)
      Result = PrivatePrefix + Result;
    else if (PrefixTy == Mangler::LinkerPrivate)
      Result = LinkerPrivatePrefix + Result;

    return Result;
  }

  if (NeedPrefix)
    Result = X.substr(0, I-X.begin()).str();
    
  // Otherwise, construct the string the expensive way.
  for (StringRef::iterator E = X.end(); I != E; ++I) {
    if (*I == '"')
      Result += "_QQ_";
    else if (*I == '\n')
      Result += "_NL_";
    else
      Result += *I;
  }

  if (NeedPrefix) {
    Result = Prefix + Result;

    if (PrefixTy == Mangler::Private)
      Result = PrivatePrefix + Result;
    else if (PrefixTy == Mangler::LinkerPrivate)
      Result = LinkerPrivatePrefix + Result;
  }

  Result = '"' + Result + '"';
  return Result;
}

/// getMangledName - Returns the mangled name of V, an LLVM Value,
/// in the current module.  If 'Suffix' is specified, the name ends with the
/// specified suffix.  If 'ForcePrivate' is specified, the label is specified
/// to have a private label prefix.
///
std::string Mangler::getMangledName(const GlobalValue *GV, const char *Suffix,
                                    bool ForcePrivate) {
  assert((!isa<Function>(GV) || !cast<Function>(GV)->isIntrinsic()) &&
         "Intrinsic functions cannot be mangled by Mangler");

  ManglerPrefixTy PrefixTy =
    (GV->hasPrivateLinkage() || ForcePrivate) ? Mangler::Private :
      GV->hasLinkerPrivateLinkage() ? Mangler::LinkerPrivate : Mangler::Default;

  if (GV->hasName())
    return makeNameProper(GV->getNameStr() + Suffix, PrefixTy);
  
  // Get the ID for the global, assigning a new one if we haven't got one
  // already.
  unsigned &ID = AnonGlobalIDs[GV];
  if (ID == 0) ID = NextAnonGlobalID++;
  
  // Must mangle the global into a unique ID.
  return makeNameProper("__unnamed_" + utostr(ID) + Suffix, PrefixTy);
}


/// getNameWithPrefix - Fill OutName with the name of the appropriate prefix
/// and the specified global variable's name.  If the global variable doesn't
/// have a name, this fills in a unique name for the global.
void Mangler::getNameWithPrefix(SmallVectorImpl<char> &OutName,
                                const GlobalValue *GV,
                                bool isImplicitlyPrivate) {
   
  // If the global is anonymous or not led with \1, then add the appropriate
  // prefix.
  if (!GV->hasName() || GV->getName()[0] != '\1') {
    if (GV->hasPrivateLinkage() || isImplicitlyPrivate)
      OutName.append(PrivatePrefix, PrivatePrefix+strlen(PrivatePrefix));
    else if (GV->hasLinkerPrivateLinkage())
      OutName.append(LinkerPrivatePrefix,
                     LinkerPrivatePrefix+strlen(LinkerPrivatePrefix));;
    OutName.append(Prefix, Prefix+strlen(Prefix));
  }

  // If the global has a name, just append it now.
  if (GV->hasName()) {
    StringRef Name = GV->getName();
    
    // Strip off the prefix marker if present.
    if (Name[0] != '\1')
      OutName.append(Name.begin(), Name.end());
    else
      OutName.append(Name.begin()+1, Name.end());
    return;
  }
  
  // If the global variable doesn't have a name, return a unique name for the
  // global based on a numbering.
  
  // Get the ID for the global, assigning a new one if we haven't got one
  // already.
  unsigned &ID = AnonGlobalIDs[GV];
  if (ID == 0) ID = NextAnonGlobalID++;
  
  // Must mangle the global into a unique ID.
  raw_svector_ostream(OutName) << "__unnamed_" << ID;
}


Mangler::Mangler(Module &M, const char *prefix, const char *privatePrefix,
                 const char *linkerPrivatePrefix)
  : Prefix(prefix), PrivatePrefix(privatePrefix),
    LinkerPrivatePrefix(linkerPrivatePrefix), UseQuotes(false),
    SymbolsCanStartWithDigit(false), NextAnonGlobalID(1) {
  std::fill(AcceptableChars, array_endof(AcceptableChars), 0);

  // Letters and numbers are acceptable.
  for (unsigned char X = 'a'; X <= 'z'; ++X)
    markCharAcceptable(X);
  for (unsigned char X = 'A'; X <= 'Z'; ++X)
    markCharAcceptable(X);
  for (unsigned char X = '0'; X <= '9'; ++X)
    markCharAcceptable(X);
  
  // These chars are acceptable.
  markCharAcceptable('_');
  markCharAcceptable('$');
  markCharAcceptable('.');
  markCharAcceptable('@');
}
