//===--- LangStandards.cpp - Language Standard Definitions ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/LangStandard.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
using namespace clang;
using namespace clang::frontend;

#define LANGSTANDARD(id, name, desc, features) \
  static LangStandard Lang_##id = { name, desc, features };
#include "clang/Frontend/LangStandards.def"

const LangStandard &LangStandard::getLangStandardForKind(Kind K) {
  switch (K) {
  default:
    llvm::llvm_unreachable("Invalid language kind!");
  case lang_unspecified:
    llvm::llvm_report_error("getLangStandardForKind() on unspecified kind");
#define LANGSTANDARD(id, name, desc, features) \
    case lang_##id: return Lang_##id;
#include "clang/Frontend/LangStandards.def"
  }
}

const LangStandard *LangStandard::getLangStandardForName(llvm::StringRef Name) {
  Kind K = llvm::StringSwitch<Kind>(Name)
#define LANGSTANDARD(id, name, desc, features) \
    .Case(name, lang_##id)
#include "clang/Frontend/LangStandards.def"
    .Default(lang_unspecified);
  if (K == lang_unspecified)
    return 0;

  return &getLangStandardForKind(K);
}


