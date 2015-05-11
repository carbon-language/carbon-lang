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
#include "llvm/Support/MathExtras.h"

using namespace clang;

SanitizerSet::SanitizerSet() : Mask(0) {}

bool SanitizerSet::has(SanitizerMask K) const {
  assert(llvm::countPopulation(K) == 1);
  return Mask & K;
}

void SanitizerSet::set(SanitizerMask K, bool Value) {
  assert(llvm::countPopulation(K) == 1);
  Mask = Value ? (Mask | K) : (Mask & ~K);
}

void SanitizerSet::clear() {
  Mask = 0;
}

bool SanitizerSet::empty() const {
  return Mask == 0;
}
