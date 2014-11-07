//===--- LangOptions.cpp - C Language Family Language Options ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the LangOptions class.
//
//===----------------------------------------------------------------------===//
#include "clang/Basic/LangOptions.h"

using namespace clang;

SanitizerOptions::SanitizerOptions()
    : Kind(0), SanitizeAddressFieldPadding(0) {}

bool SanitizerOptions::has(SanitizerKind K) const {
  unsigned Bit = static_cast<unsigned>(K);
  return Kind & (1 << Bit);
}

void SanitizerOptions::set(SanitizerKind K, bool Value) {
  unsigned Bit = static_cast<unsigned>(K);
  Kind = Value ? (Kind | (1 << Bit)) : (Kind & ~(1 << Bit));
}

void SanitizerOptions::clear() {
  SanitizerOptions Default;
  *this = std::move(Default);
}

LangOptions::LangOptions() {
#define LANGOPT(Name, Bits, Default, Description) Name = Default;
#define ENUM_LANGOPT(Name, Type, Bits, Default, Description) set##Name(Default);
#include "clang/Basic/LangOptions.def"
}

void LangOptions::resetNonModularOptions() {
#define LANGOPT(Name, Bits, Default, Description)
#define BENIGN_LANGOPT(Name, Bits, Default, Description) Name = Default;
#define BENIGN_ENUM_LANGOPT(Name, Type, Bits, Default, Description) \
  Name = Default;
#include "clang/Basic/LangOptions.def"

  // FIXME: This should not be reset; modules can be different with different
  // sanitizer options (this affects __has_feature(address_sanitizer) etc).
  Sanitize.clear();

  CurrentModule.clear();
  ImplementationOfModule.clear();
}

