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
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

static char HexDigit(int V) {
  return V < 10 ? V+'0' : V+'A'-10;
}

static void MangleLetter(SmallVectorImpl<char> &OutName, unsigned char C) {
  OutName.push_back('_');
  OutName.push_back(HexDigit(C >> 4));
  OutName.push_back(HexDigit(C & 15));
  OutName.push_back('_');
}

/// makeNameProper - We don't want identifier names non-C-identifier characters
/// in them, so mangle them as appropriate.
///
/// FIXME: This is deprecated, new code should use getNameWithPrefix and use
/// MCSymbol printing to handle quotes or not etc.
///
void Mangler::makeNameProper(SmallVectorImpl<char> &OutName,
                             const Twine &TheName,
                             ManglerPrefixTy PrefixTy) {
  SmallString<256> TmpData;
  TheName.toVector(TmpData);
  StringRef X = TmpData.str();
  assert(!X.empty() && "Cannot mangle empty strings");
  
  if (!UseQuotes) {
    // If X does not start with (char)1, add the prefix.
    StringRef::iterator I = X.begin();
    if (*I == 1) {
      ++I;  // Skip over the no-prefix marker.
    } else {
      if (PrefixTy == Mangler::Private)
        OutName.append(PrivatePrefix, PrivatePrefix+strlen(PrivatePrefix));
      else if (PrefixTy == Mangler::LinkerPrivate)
        OutName.append(LinkerPrivatePrefix,
                       LinkerPrivatePrefix+strlen(LinkerPrivatePrefix));
      OutName.append(Prefix, Prefix+strlen(Prefix));
    }
    
    // Mangle the first letter specially, don't allow numbers unless the target
    // explicitly allows them.
    if (!SymbolsCanStartWithDigit && *I >= '0' && *I <= '9')
      MangleLetter(OutName, *I++);

    for (StringRef::iterator E = X.end(); I != E; ++I) {
      if (!isCharAcceptable(*I))
        MangleLetter(OutName, *I);
      else
        OutName.push_back(*I);
    }
    return;
  }

  bool NeedPrefix = true;
  bool NeedQuotes = false;
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
    if (!NeedPrefix) {
      OutName.append(X.begin()+1, X.end());   // Strip off the \001.
      return;
    }

    if (PrefixTy == Mangler::Private)
      OutName.append(PrivatePrefix, PrivatePrefix+strlen(PrivatePrefix));
    else if (PrefixTy == Mangler::LinkerPrivate)
      OutName.append(LinkerPrivatePrefix,
                     LinkerPrivatePrefix+strlen(LinkerPrivatePrefix));
    
    if (Prefix[0] == 0)
      ; // Common noop, no prefix.
    else if (Prefix[1] == 0)
      OutName.push_back(Prefix[0]);  // Common, one character prefix.
    else
      OutName.append(Prefix, Prefix+strlen(Prefix)); // Arbitrary prefix.
    OutName.append(X.begin(), X.end());
    return;
  }

  // Add leading quote.
  OutName.push_back('"');
  
  // Add prefixes unless disabled.
  if (NeedPrefix) {
    if (PrefixTy == Mangler::Private)
      OutName.append(PrivatePrefix, PrivatePrefix+strlen(PrivatePrefix));
    else if (PrefixTy == Mangler::LinkerPrivate)
      OutName.append(LinkerPrivatePrefix,
                     LinkerPrivatePrefix+strlen(LinkerPrivatePrefix));
    OutName.append(Prefix, Prefix+strlen(Prefix));
  }
  
  // Add the piece that we already scanned through.
  OutName.append(X.begin(), I);
  
  // Otherwise, construct the string the expensive way.
  for (StringRef::iterator E = X.end(); I != E; ++I) {
    if (*I == '"') {
      const char *Quote = "_QQ_";
      OutName.append(Quote, Quote+4);
    } else if (*I == '\n') {
      const char *Newline = "_NL_";
      OutName.append(Newline, Newline+4);
    } else
      OutName.push_back(*I);
  }

  // Add trailing quote.
  OutName.push_back('"');
}

/// getMangledName - Returns the mangled name of V, an LLVM Value,
/// in the current module.  If 'Suffix' is specified, the name ends with the
/// specified suffix.  If 'ForcePrivate' is specified, the label is specified
/// to have a private label prefix.
///
/// FIXME: This is deprecated, new code should use getNameWithPrefix and use
/// MCSymbol printing to handle quotes or not etc.
///
std::string Mangler::getMangledName(const GlobalValue *GV, const char *Suffix,
                                    bool ForcePrivate) {
  assert((!isa<Function>(GV) || !cast<Function>(GV)->isIntrinsic()) &&
         "Intrinsic functions cannot be mangled by Mangler");

  ManglerPrefixTy PrefixTy =
    (GV->hasPrivateLinkage() || ForcePrivate) ? Mangler::Private :
      GV->hasLinkerPrivateLinkage() ? Mangler::LinkerPrivate : Mangler::Default;

  SmallString<128> Result;
  if (GV->hasName()) {
    makeNameProper(Result, GV->getNameStr() + Suffix, PrefixTy);
    return Result.str().str();
  }
  
  // Get the ID for the global, assigning a new one if we haven't got one
  // already.
  unsigned &ID = AnonGlobalIDs[GV];
  if (ID == 0) ID = NextAnonGlobalID++;
  
  // Must mangle the global into a unique ID.
  makeNameProper(Result, "__unnamed_" + utostr(ID) + Suffix, PrefixTy);
  return Result.str().str();
}

/// getNameWithPrefix - Fill OutName with the name of the appropriate prefix
/// and the specified name as the global variable name.  GVName must not be
/// empty.
void Mangler::getNameWithPrefix(SmallVectorImpl<char> &OutName,
                                const Twine &GVName, ManglerPrefixTy PrefixTy) {
  SmallString<256> TmpData;
  StringRef Name;
  if (GVName.isSingleStringRef())
    Name = GVName.getSingleStringRef();
  else {
    GVName.toVector(TmpData);
    Name = TmpData.str();
  }
  assert(!Name.empty() && "getNameWithPrefix requires non-empty name");
  
  // If the global name is not led with \1, add the appropriate prefixes.
  if (Name[0] != '\1') {
    if (PrefixTy == Mangler::Private)
      OutName.append(PrivatePrefix, PrivatePrefix+strlen(PrivatePrefix));
    else if (PrefixTy == Mangler::LinkerPrivate)
      OutName.append(LinkerPrivatePrefix,
                     LinkerPrivatePrefix+strlen(LinkerPrivatePrefix));
    
    if (Prefix[0] == 0)
      ; // Common noop, no prefix.
    else if (Prefix[1] == 0)
      OutName.push_back(Prefix[0]);  // Common, one character prefix.
    else
      OutName.append(Prefix, Prefix+strlen(Prefix)); // Arbitrary prefix.
  } else {
    Name = Name.substr(1);
  }
  
  OutName.append(Name.begin(), Name.end());
}


/// getNameWithPrefix - Fill OutName with the name of the appropriate prefix
/// and the specified global variable's name.  If the global variable doesn't
/// have a name, this fills in a unique name for the global.
void Mangler::getNameWithPrefix(SmallVectorImpl<char> &OutName,
                                const GlobalValue *GV,
                                bool isImplicitlyPrivate) {
  // If this global has a name, handle it simply.
  if (GV->hasName()) {
    ManglerPrefixTy PrefixTy = Mangler::Default;
    if (GV->hasPrivateLinkage() || isImplicitlyPrivate)
      PrefixTy = Mangler::Private;
    else if (GV->hasLinkerPrivateLinkage())
      PrefixTy = Mangler::LinkerPrivate;
    
    return getNameWithPrefix(OutName, GV->getName(), PrefixTy);
  }
  
  // If the global variable doesn't have a name, return a unique name for the
  // global based on a numbering.
  
  // Anonymous names always get prefixes.
  if (GV->hasPrivateLinkage() || isImplicitlyPrivate)
    OutName.append(PrivatePrefix, PrivatePrefix+strlen(PrivatePrefix));
  else if (GV->hasLinkerPrivateLinkage())
    OutName.append(LinkerPrivatePrefix,
                   LinkerPrivatePrefix+strlen(LinkerPrivatePrefix));;
  OutName.append(Prefix, Prefix+strlen(Prefix));
  
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
