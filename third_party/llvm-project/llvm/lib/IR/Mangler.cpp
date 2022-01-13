//===-- Mangler.cpp - Self-contained c/asm llvm name mangler --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unified name mangler for assembly backends.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Mangler.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
enum ManglerPrefixTy {
  Default,      ///< Emit default string before each symbol.
  Private,      ///< Emit "private" prefix before each symbol.
  LinkerPrivate ///< Emit "linker private" prefix before each symbol.
};
}

static void getNameWithPrefixImpl(raw_ostream &OS, const Twine &GVName,
                                  ManglerPrefixTy PrefixTy,
                                  const DataLayout &DL, char Prefix) {
  SmallString<256> TmpData;
  StringRef Name = GVName.toStringRef(TmpData);
  assert(!Name.empty() && "getNameWithPrefix requires non-empty name");

  // No need to do anything special if the global has the special "do not
  // mangle" flag in the name.
  if (Name[0] == '\1') {
    OS << Name.substr(1);
    return;
  }

  if (DL.doNotMangleLeadingQuestionMark() && Name[0] == '?')
    Prefix = '\0';

  if (PrefixTy == Private)
    OS << DL.getPrivateGlobalPrefix();
  else if (PrefixTy == LinkerPrivate)
    OS << DL.getLinkerPrivateGlobalPrefix();

  if (Prefix != '\0')
    OS << Prefix;

  // If this is a simple string that doesn't need escaping, just append it.
  OS << Name;
}

static void getNameWithPrefixImpl(raw_ostream &OS, const Twine &GVName,
                                  const DataLayout &DL,
                                  ManglerPrefixTy PrefixTy) {
  char Prefix = DL.getGlobalPrefix();
  return getNameWithPrefixImpl(OS, GVName, PrefixTy, DL, Prefix);
}

void Mangler::getNameWithPrefix(raw_ostream &OS, const Twine &GVName,
                                const DataLayout &DL) {
  return getNameWithPrefixImpl(OS, GVName, DL, Default);
}

void Mangler::getNameWithPrefix(SmallVectorImpl<char> &OutName,
                                const Twine &GVName, const DataLayout &DL) {
  raw_svector_ostream OS(OutName);
  char Prefix = DL.getGlobalPrefix();
  return getNameWithPrefixImpl(OS, GVName, Default, DL, Prefix);
}

static bool hasByteCountSuffix(CallingConv::ID CC) {
  switch (CC) {
  case CallingConv::X86_FastCall:
  case CallingConv::X86_StdCall:
  case CallingConv::X86_VectorCall:
    return true;
  default:
    return false;
  }
}

/// Microsoft fastcall and stdcall functions require a suffix on their name
/// indicating the number of words of arguments they take.
static void addByteCountSuffix(raw_ostream &OS, const Function *F,
                               const DataLayout &DL) {
  // Calculate arguments size total.
  unsigned ArgWords = 0;

  const unsigned PtrSize = DL.getPointerSize();

  for (const Argument &A : F->args()) {
    // 'Dereference' type in case of byval or inalloca parameter attribute.
    uint64_t AllocSize = A.hasPassPointeeByValueCopyAttr() ?
      A.getPassPointeeByValueCopySize(DL) :
      DL.getTypeAllocSize(A.getType());

    // Size should be aligned to pointer size.
    ArgWords += alignTo(AllocSize, PtrSize);
  }

  OS << '@' << ArgWords;
}

void Mangler::getNameWithPrefix(raw_ostream &OS, const GlobalValue *GV,
                                bool CannotUsePrivateLabel) const {
  ManglerPrefixTy PrefixTy = Default;
  if (GV->hasPrivateLinkage()) {
    if (CannotUsePrivateLabel)
      PrefixTy = LinkerPrivate;
    else
      PrefixTy = Private;
  }

  const DataLayout &DL = GV->getParent()->getDataLayout();
  if (!GV->hasName()) {
    // Get the ID for the global, assigning a new one if we haven't got one
    // already.
    unsigned &ID = AnonGlobalIDs[GV];
    if (ID == 0)
      ID = AnonGlobalIDs.size();

    // Must mangle the global into a unique ID.
    getNameWithPrefixImpl(OS, "__unnamed_" + Twine(ID), DL, PrefixTy);
    return;
  }

  StringRef Name = GV->getName();
  char Prefix = DL.getGlobalPrefix();

  // Mangle functions with Microsoft calling conventions specially.  Only do
  // this mangling for x86_64 vectorcall and 32-bit x86.
  const Function *MSFunc = dyn_cast<Function>(GV);

  // Don't add byte count suffixes when '\01' or '?' are in the first
  // character.
  if (Name.startswith("\01") ||
      (DL.doNotMangleLeadingQuestionMark() && Name.startswith("?")))
    MSFunc = nullptr;

  CallingConv::ID CC =
      MSFunc ? MSFunc->getCallingConv() : (unsigned)CallingConv::C;
  if (!DL.hasMicrosoftFastStdCallMangling() &&
      CC != CallingConv::X86_VectorCall)
    MSFunc = nullptr;
  if (MSFunc) {
    if (CC == CallingConv::X86_FastCall)
      Prefix = '@'; // fastcall functions have an @ prefix instead of _.
    else if (CC == CallingConv::X86_VectorCall)
      Prefix = '\0'; // vectorcall functions have no prefix.
  }

  getNameWithPrefixImpl(OS, Name, PrefixTy, DL, Prefix);

  if (!MSFunc)
    return;

  // If we are supposed to add a microsoft-style suffix for stdcall, fastcall,
  // or vectorcall, add it.  These functions have a suffix of @N where N is the
  // cumulative byte size of all of the parameters to the function in decimal.
  if (CC == CallingConv::X86_VectorCall)
    OS << '@'; // vectorcall functions use a double @ suffix.
  FunctionType *FT = MSFunc->getFunctionType();
  if (hasByteCountSuffix(CC) &&
      // "Pure" variadic functions do not receive @0 suffix.
      (!FT->isVarArg() || FT->getNumParams() == 0 ||
       (FT->getNumParams() == 1 && MSFunc->hasStructRetAttr())))
    addByteCountSuffix(OS, MSFunc, DL);
}

void Mangler::getNameWithPrefix(SmallVectorImpl<char> &OutName,
                                const GlobalValue *GV,
                                bool CannotUsePrivateLabel) const {
  raw_svector_ostream OS(OutName);
  getNameWithPrefix(OS, GV, CannotUsePrivateLabel);
}

// Check if the name needs quotes to be safe for the linker to interpret.
static bool canBeUnquotedInDirective(char C) {
  return isAlnum(C) || C == '_' || C == '$' || C == '.' || C == '@';
}

static bool canBeUnquotedInDirective(StringRef Name) {
  if (Name.empty())
    return false;

  // If any of the characters in the string is an unacceptable character, force
  // quotes.
  for (char C : Name) {
    if (!canBeUnquotedInDirective(C))
      return false;
  }

  return true;
}

void llvm::emitLinkerFlagsForGlobalCOFF(raw_ostream &OS, const GlobalValue *GV,
                                        const Triple &TT, Mangler &Mangler) {
  if (!GV->hasDLLExportStorageClass() || GV->isDeclaration())
    return;

  if (TT.isWindowsMSVCEnvironment())
    OS << " /EXPORT:";
  else
    OS << " -export:";

  bool NeedQuotes = GV->hasName() && !canBeUnquotedInDirective(GV->getName());
  if (NeedQuotes)
    OS << "\"";
  if (TT.isWindowsGNUEnvironment() || TT.isWindowsCygwinEnvironment()) {
    std::string Flag;
    raw_string_ostream FlagOS(Flag);
    Mangler.getNameWithPrefix(FlagOS, GV, false);
    FlagOS.flush();
    if (Flag[0] == GV->getParent()->getDataLayout().getGlobalPrefix())
      OS << Flag.substr(1);
    else
      OS << Flag;
  } else {
    Mangler.getNameWithPrefix(OS, GV, false);
  }
  if (NeedQuotes)
    OS << "\"";

  if (!GV->getValueType()->isFunctionTy()) {
    if (TT.isWindowsMSVCEnvironment())
      OS << ",DATA";
    else
      OS << ",data";
  }
}

void llvm::emitLinkerFlagsForUsedCOFF(raw_ostream &OS, const GlobalValue *GV,
                                      const Triple &T, Mangler &M) {
  if (!T.isWindowsMSVCEnvironment())
    return;

  OS << " /INCLUDE:";
  bool NeedQuotes = GV->hasName() && !canBeUnquotedInDirective(GV->getName());
  if (NeedQuotes)
    OS << "\"";
  M.getNameWithPrefix(OS, GV, false);
  if (NeedQuotes)
    OS << "\"";
}

