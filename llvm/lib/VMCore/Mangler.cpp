//===-- Mangler.cpp - Self-contained c/asm llvm name mangler --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Unified name mangler for assembly backends.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Mangler.h"
#include "llvm/GlobalValue.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

/// getNameWithPrefix - Fill OutName with the name of the appropriate prefix
/// and the specified name as the global variable name.  GVName must not be
/// empty.
void Mangler::getNameWithPrefix(SmallVectorImpl<char> &OutName,
                                const Twine &GVName, ManglerPrefixTy PrefixTy) {
  SmallString<256> TmpData;
  StringRef Name = GVName.toStringRef(TmpData);
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

/// getNameWithPrefix - Fill OutName with the name of the appropriate prefix
/// and the specified global variable's name.  If the global variable doesn't
/// have a name, this fills in a unique name for the global.
std::string Mangler::getNameWithPrefix(const GlobalValue *GV,
                                       bool isImplicitlyPrivate) {
  SmallString<64> Buf;
  getNameWithPrefix(Buf, GV, isImplicitlyPrivate);
  return std::string(Buf.begin(), Buf.end());
}
  

Mangler::Mangler(Module &M, const char *prefix, const char *privatePrefix,
                 const char *linkerPrivatePrefix)
  : Prefix(prefix), PrivatePrefix(privatePrefix),
    LinkerPrivatePrefix(linkerPrivatePrefix), NextAnonGlobalID(1) {
}
