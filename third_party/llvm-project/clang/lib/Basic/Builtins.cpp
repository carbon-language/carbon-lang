//===--- Builtins.cpp - Builtin function implementation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements various things for builtin functions.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Builtins.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/StringRef.h"
using namespace clang;

static const Builtin::Info BuiltinInfo[] = {
  { "not a builtin function", nullptr, nullptr, nullptr, ALL_LANGUAGES,nullptr},
#define BUILTIN(ID, TYPE, ATTRS)                                               \
  { #ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, nullptr },
#define LANGBUILTIN(ID, TYPE, ATTRS, LANGS)                                    \
  { #ID, TYPE, ATTRS, nullptr, LANGS, nullptr },
#define LIBBUILTIN(ID, TYPE, ATTRS, HEADER, LANGS)                             \
  { #ID, TYPE, ATTRS, HEADER, LANGS, nullptr },
#include "clang/Basic/Builtins.def"
};

const Builtin::Info &Builtin::Context::getRecord(unsigned ID) const {
  if (ID < Builtin::FirstTSBuiltin)
    return BuiltinInfo[ID];
  assert(((ID - Builtin::FirstTSBuiltin) <
          (TSRecords.size() + AuxTSRecords.size())) &&
         "Invalid builtin ID!");
  if (isAuxBuiltinID(ID))
    return AuxTSRecords[getAuxBuiltinID(ID) - Builtin::FirstTSBuiltin];
  return TSRecords[ID - Builtin::FirstTSBuiltin];
}

void Builtin::Context::InitializeTarget(const TargetInfo &Target,
                                        const TargetInfo *AuxTarget) {
  assert(TSRecords.empty() && "Already initialized target?");
  TSRecords = Target.getTargetBuiltins();
  if (AuxTarget)
    AuxTSRecords = AuxTarget->getTargetBuiltins();
}

bool Builtin::Context::isBuiltinFunc(llvm::StringRef FuncName) {
  for (unsigned i = Builtin::NotBuiltin + 1; i != Builtin::FirstTSBuiltin; ++i)
    if (FuncName.equals(BuiltinInfo[i].Name))
      return strchr(BuiltinInfo[i].Attributes, 'f') != nullptr;

  return false;
}

bool Builtin::Context::builtinIsSupported(const Builtin::Info &BuiltinInfo,
                                          const LangOptions &LangOpts) {
  bool BuiltinsUnsupported =
      (LangOpts.NoBuiltin || LangOpts.isNoBuiltinFunc(BuiltinInfo.Name)) &&
      strchr(BuiltinInfo.Attributes, 'f');
  bool CorBuiltinsUnsupported =
      !LangOpts.Coroutines && (BuiltinInfo.Langs & COR_LANG);
  bool MathBuiltinsUnsupported =
    LangOpts.NoMathBuiltin && BuiltinInfo.HeaderName &&
    llvm::StringRef(BuiltinInfo.HeaderName).equals("math.h");
  bool GnuModeUnsupported = !LangOpts.GNUMode && (BuiltinInfo.Langs & GNU_LANG);
  bool MSModeUnsupported =
      !LangOpts.MicrosoftExt && (BuiltinInfo.Langs & MS_LANG);
  bool ObjCUnsupported = !LangOpts.ObjC && BuiltinInfo.Langs == OBJC_LANG;
  bool OclC1Unsupported = (LangOpts.OpenCLVersion / 100) != 1 &&
                          (BuiltinInfo.Langs & ALL_OCLC_LANGUAGES ) ==  OCLC1X_LANG;
  bool OclC2Unsupported =
      (LangOpts.getOpenCLCompatibleVersion() != 200) &&
      (BuiltinInfo.Langs & ALL_OCLC_LANGUAGES) == OCLC20_LANG;
  bool OclCUnsupported = !LangOpts.OpenCL &&
                         (BuiltinInfo.Langs & ALL_OCLC_LANGUAGES);
  bool OpenMPUnsupported = !LangOpts.OpenMP && BuiltinInfo.Langs == OMP_LANG;
  bool CUDAUnsupported = !LangOpts.CUDA && BuiltinInfo.Langs == CUDA_LANG;
  bool CPlusPlusUnsupported =
      !LangOpts.CPlusPlus && BuiltinInfo.Langs == CXX_LANG;
  return !BuiltinsUnsupported && !CorBuiltinsUnsupported &&
         !MathBuiltinsUnsupported && !OclCUnsupported && !OclC1Unsupported &&
         !OclC2Unsupported && !OpenMPUnsupported && !GnuModeUnsupported &&
         !MSModeUnsupported && !ObjCUnsupported && !CPlusPlusUnsupported &&
         !CUDAUnsupported;
}

/// initializeBuiltins - Mark the identifiers for all the builtins with their
/// appropriate builtin ID # and mark any non-portable builtin identifiers as
/// such.
void Builtin::Context::initializeBuiltins(IdentifierTable &Table,
                                          const LangOptions& LangOpts) {
  // Step #1: mark all target-independent builtins with their ID's.
  for (unsigned i = Builtin::NotBuiltin+1; i != Builtin::FirstTSBuiltin; ++i)
    if (builtinIsSupported(BuiltinInfo[i], LangOpts)) {
      Table.get(BuiltinInfo[i].Name).setBuiltinID(i);
    }

  // Step #2: Register target-specific builtins.
  for (unsigned i = 0, e = TSRecords.size(); i != e; ++i)
    if (builtinIsSupported(TSRecords[i], LangOpts))
      Table.get(TSRecords[i].Name).setBuiltinID(i + Builtin::FirstTSBuiltin);

  // Step #3: Register target-specific builtins for AuxTarget.
  for (unsigned i = 0, e = AuxTSRecords.size(); i != e; ++i)
    Table.get(AuxTSRecords[i].Name)
        .setBuiltinID(i + Builtin::FirstTSBuiltin + TSRecords.size());
}

unsigned Builtin::Context::getRequiredVectorWidth(unsigned ID) const {
  const char *WidthPos = ::strchr(getRecord(ID).Attributes, 'V');
  if (!WidthPos)
    return 0;

  ++WidthPos;
  assert(*WidthPos == ':' &&
         "Vector width specifier must be followed by a ':'");
  ++WidthPos;

  char *EndPos;
  unsigned Width = ::strtol(WidthPos, &EndPos, 10);
  assert(*EndPos == ':' && "Vector width specific must end with a ':'");
  return Width;
}

bool Builtin::Context::isLike(unsigned ID, unsigned &FormatIdx,
                              bool &HasVAListArg, const char *Fmt) const {
  assert(Fmt && "Not passed a format string");
  assert(::strlen(Fmt) == 2 &&
         "Format string needs to be two characters long");
  assert(::toupper(Fmt[0]) == Fmt[1] &&
         "Format string is not in the form \"xX\"");

  const char *Like = ::strpbrk(getRecord(ID).Attributes, Fmt);
  if (!Like)
    return false;

  HasVAListArg = (*Like == Fmt[1]);

  ++Like;
  assert(*Like == ':' && "Format specifier must be followed by a ':'");
  ++Like;

  assert(::strchr(Like, ':') && "Format specifier must end with a ':'");
  FormatIdx = ::strtol(Like, nullptr, 10);
  return true;
}

bool Builtin::Context::isPrintfLike(unsigned ID, unsigned &FormatIdx,
                                    bool &HasVAListArg) {
  return isLike(ID, FormatIdx, HasVAListArg, "pP");
}

bool Builtin::Context::isScanfLike(unsigned ID, unsigned &FormatIdx,
                                   bool &HasVAListArg) {
  return isLike(ID, FormatIdx, HasVAListArg, "sS");
}

bool Builtin::Context::performsCallback(unsigned ID,
                                        SmallVectorImpl<int> &Encoding) const {
  const char *CalleePos = ::strchr(getRecord(ID).Attributes, 'C');
  if (!CalleePos)
    return false;

  ++CalleePos;
  assert(*CalleePos == '<' &&
         "Callback callee specifier must be followed by a '<'");
  ++CalleePos;

  char *EndPos;
  int CalleeIdx = ::strtol(CalleePos, &EndPos, 10);
  assert(CalleeIdx >= 0 && "Callee index is supposed to be positive!");
  Encoding.push_back(CalleeIdx);

  while (*EndPos == ',') {
    const char *PayloadPos = EndPos + 1;

    int PayloadIdx = ::strtol(PayloadPos, &EndPos, 10);
    Encoding.push_back(PayloadIdx);
  }

  assert(*EndPos == '>' && "Callback callee specifier must end with a '>'");
  return true;
}

bool Builtin::Context::canBeRedeclared(unsigned ID) const {
  return ID == Builtin::NotBuiltin ||
         ID == Builtin::BI__va_start ||
         (!hasReferenceArgsOrResult(ID) &&
          !hasCustomTypechecking(ID));
}
