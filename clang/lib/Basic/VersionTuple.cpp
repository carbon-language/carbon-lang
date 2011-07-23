//===- VersionTuple.cpp - Version Number Handling ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the VersionTuple class, which represents a version in
// the form major[.minor[.subminor]].
//
//===----------------------------------------------------------------------===//
#include "clang/Basic/VersionTuple.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

std::string VersionTuple::getAsString() const {
  std::string Result;
  {
    llvm::raw_string_ostream Out(Result);
    Out << *this;
  }
  return Result;
}

raw_ostream& clang::operator<<(raw_ostream &Out, 
                                     const VersionTuple &V) {
  Out << V.getMajor();
  if (llvm::Optional<unsigned> Minor = V.getMinor())
    Out << '.' << *Minor;
  if (llvm::Optional<unsigned> Subminor = V.getSubminor())
    Out << '.' << *Subminor;
  return Out;
}
