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
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace object::yaml;

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

// Can't find this anywhere else in the codebase (clang has one, but it has
// some baggage). Deduplicate as required.
static bool isHexDigit(uint8_t C) {
  return ('0' <= C && C <= '9') ||
         ('A' <= C && C <= 'F') ||
         ('a' <= C && C <= 'f');
}

StringRef yaml::ScalarTraits<object::yaml::BinaryRef>::input(
    StringRef Scalar, void *, object::yaml::BinaryRef &Val) {
  if (Scalar.size() % 2 != 0)
    return "BinaryRef hex string must contain an even number of nybbles.";
  // TODO: Can we improve YAMLIO to permit a more accurate diagnostic here?
  // (e.g. a caret pointing to the offending character).
  for (unsigned I = 0, N = Scalar.size(); I != N; ++I)
    if (!isHexDigit(Scalar[I]))
      return "BinaryRef hex string must contain only hex digits.";
  Val = object::yaml::BinaryRef(Scalar);
  return StringRef();
}

void BinaryRef::writeAsBinary(raw_ostream &OS) const {
  if (isBinary) {
    OS.write((const char *)Data.data(), Data.size());
    return;
  }
  for (unsigned I = 0, N = Data.size(); I != N; I += 2) {
    uint8_t Byte;
    StringRef((const char *)&Data[I],  2).getAsInteger(16, Byte);
    OS.write(Byte);
  }
}
