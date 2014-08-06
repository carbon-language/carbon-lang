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

#include "llvm/IR/Mangler.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

static void getNameWithPrefixx(raw_ostream &OS, const Twine &GVName,
                              Mangler::ManglerPrefixTy PrefixTy,
                              const DataLayout &DL, bool UseAt) {
  SmallString<256> TmpData;
  StringRef Name = GVName.toStringRef(TmpData);
  assert(!Name.empty() && "getNameWithPrefix requires non-empty name");

  // No need to do anything special if the global has the special "do not
  // mangle" flag in the name.
  if (Name[0] == '\1') {
    OS << Name.substr(1);
    return;
  }

  if (PrefixTy == Mangler::Private)
    OS << DL.getPrivateGlobalPrefix();
  else if (PrefixTy == Mangler::LinkerPrivate)
    OS << DL.getLinkerPrivateGlobalPrefix();

  if (UseAt) {
    OS << '@';
  } else {
    char Prefix = DL.getGlobalPrefix();
    if (Prefix != '\0')
      OS << Prefix;
  }

  // If this is a simple string that doesn't need escaping, just append it.
  OS << Name;
}

void Mangler::getNameWithPrefix(raw_ostream &OS, const Twine &GVName,
                                ManglerPrefixTy PrefixTy) const {
  return getNameWithPrefixx(OS, GVName, PrefixTy, *DL, false);
}

void Mangler::getNameWithPrefix(SmallVectorImpl<char> &OutName,
                                const Twine &GVName,
                                ManglerPrefixTy PrefixTy) const {
  raw_svector_ostream OS(OutName);
  return getNameWithPrefix(OS, GVName, PrefixTy);
}

/// AddFastCallStdCallSuffix - Microsoft fastcall and stdcall functions require
/// a suffix on their name indicating the number of words of arguments they
/// take.
static void AddFastCallStdCallSuffix(raw_ostream &OS, const Function *F,
                                     const DataLayout &TD) {
  // Calculate arguments size total.
  unsigned ArgWords = 0;
  for (Function::const_arg_iterator AI = F->arg_begin(), AE = F->arg_end();
       AI != AE; ++AI) {
    // Skip arguments in registers to handle typical fastcall lowering.
    if (F->getAttributes().hasAttribute(AI->getArgNo() + 1, Attribute::InReg))
      continue;
    Type *Ty = AI->getType();
    // 'Dereference' type in case of byval or inalloca parameter attribute.
    if (AI->hasByValOrInAllocaAttr())
      Ty = cast<PointerType>(Ty)->getElementType();
    // Size should be aligned to DWORD boundary
    ArgWords += ((TD.getTypeAllocSize(Ty) + 3)/4)*4;
  }

  OS << '@' << ArgWords;
}

void Mangler::getNameWithPrefix(raw_ostream &OS, const GlobalValue *GV,
                                bool CannotUsePrivateLabel) const {
  ManglerPrefixTy PrefixTy = Mangler::Default;
  if (GV->hasPrivateLinkage()) {
    if (CannotUsePrivateLabel)
      PrefixTy = Mangler::LinkerPrivate;
    else
      PrefixTy = Mangler::Private;
  }

  if (!GV->hasName()) {
    // Get the ID for the global, assigning a new one if we haven't got one
    // already.
    unsigned &ID = AnonGlobalIDs[GV];
    if (ID == 0)
      ID = NextAnonGlobalID++;

    // Must mangle the global into a unique ID.
    getNameWithPrefix(OS, "__unnamed_" + Twine(ID), PrefixTy);
    return;
  }

  StringRef Name = GV->getName();

  bool UseAt = false;
  const Function *MSFunc = nullptr;
  CallingConv::ID CC;
  if (Name[0] != '\1' && DL->hasMicrosoftFastStdCallMangling()) {
    if ((MSFunc = dyn_cast<Function>(GV))) {
      CC = MSFunc->getCallingConv();
      // fastcall functions need to start with @ instead of _.
      if (CC == CallingConv::X86_FastCall)
        UseAt = true;
    }
  }

  getNameWithPrefixx(OS, Name, PrefixTy, *DL, UseAt);

  if (!MSFunc)
    return;

  // If we are supposed to add a microsoft-style suffix for stdcall/fastcall,
  // add it.
  // fastcall and stdcall functions usually need @42 at the end to specify
  // the argument info.
  FunctionType *FT = MSFunc->getFunctionType();
  if ((CC == CallingConv::X86_FastCall || CC == CallingConv::X86_StdCall) &&
      // "Pure" variadic functions do not receive @0 suffix.
      (!FT->isVarArg() || FT->getNumParams() == 0 ||
       (FT->getNumParams() == 1 && MSFunc->hasStructRetAttr())))
    AddFastCallStdCallSuffix(OS, MSFunc, *DL);
}

void Mangler::getNameWithPrefix(SmallVectorImpl<char> &OutName,
                                const GlobalValue *GV,
                                bool CannotUsePrivateLabel) const {
  raw_svector_ostream OS(OutName);
  getNameWithPrefix(OS, GV, CannotUsePrivateLabel);
}
