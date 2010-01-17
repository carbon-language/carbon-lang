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

#include "llvm/Target/Mangler.h"
#include "llvm/GlobalValue.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
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
    if (PrefixTy == Mangler::Private) {
      const char *Prefix = MAI.getPrivateGlobalPrefix();
      OutName.append(Prefix, Prefix+strlen(Prefix));
    } else if (PrefixTy == Mangler::LinkerPrivate) {
      const char *Prefix = MAI.getLinkerPrivateGlobalPrefix();
      OutName.append(Prefix, Prefix+strlen(Prefix));
    }

    const char *Prefix = MAI.getGlobalPrefix();
    if (Prefix[0] == 0)
      ; // Common noop, no prefix.
    else if (Prefix[1] == 0)
      OutName.push_back(Prefix[0]);  // Common, one character prefix.
    else
      OutName.append(Prefix, Prefix+strlen(Prefix)); // Arbitrary length prefix.
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
  ManglerPrefixTy PrefixTy = Mangler::Default;
  if (GV->hasPrivateLinkage() || isImplicitlyPrivate)
    PrefixTy = Mangler::Private;
  else if (GV->hasLinkerPrivateLinkage())
    PrefixTy = Mangler::LinkerPrivate;
  
  // If this global has a name, handle it simply.
  if (GV->hasName())
    return getNameWithPrefix(OutName, GV->getName(), PrefixTy);
  
  // Get the ID for the global, assigning a new one if we haven't got one
  // already.
  unsigned &ID = AnonGlobalIDs[GV];
  if (ID == 0) ID = NextAnonGlobalID++;
  
  // Must mangle the global into a unique ID.
  getNameWithPrefix(OutName, "__unnamed_" + Twine(ID), PrefixTy);
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
