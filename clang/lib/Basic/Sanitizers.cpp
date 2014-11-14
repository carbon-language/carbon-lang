//===--- Sanitizers.cpp - C Language Family Language Options ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the classes from Sanitizers.h
//
//===----------------------------------------------------------------------===//
#include "clang/Basic/Sanitizers.h"

using namespace clang;

SanitizerSet::SanitizerSet() : Kinds(0) {}

bool SanitizerSet::has(SanitizerKind K) const {
  unsigned Bit = static_cast<unsigned>(K);
  return Kinds & (1 << Bit);
}

void SanitizerSet::set(SanitizerKind K, bool Value) {
  unsigned Bit = static_cast<unsigned>(K);
  Kinds = Value ? (Kinds | (1 << Bit)) : (Kinds & ~(1 << Bit));
}

void SanitizerSet::clear() {
  Kinds = 0;
}

bool SanitizerSet::empty() const {
  return Kinds == 0;
}
