//===- YAML.cpp - YAMLIO utilities for object files -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines utility classes for handling the YAML representation of
// object files.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/YAML.h"

using namespace llvm;

void yaml::ScalarTraits<object::yaml::BinaryRef>::output(
    const object::yaml::BinaryRef &Val, void *, llvm::raw_ostream &Out) {
  ArrayRef<uint8_t> Data = Val.getBinary();
  for (ArrayRef<uint8_t>::iterator I = Data.begin(), E = Data.end(); I != E;
       ++I) {
    uint8_t Byte = *I;
    Out << hexdigit(Byte >> 4);
    Out << hexdigit(Byte & 0xf);
  }
}

StringRef yaml::ScalarTraits<object::yaml::BinaryRef>::input(
    StringRef Scalar, void *, object::yaml::BinaryRef &Val) {
  Val = object::yaml::BinaryRef(Scalar);
  return StringRef();
}
