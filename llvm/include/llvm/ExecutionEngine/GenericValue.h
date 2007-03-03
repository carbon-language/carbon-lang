//===-- GenericValue.h - Represent any type of LLVM value -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The GenericValue class is used to represent an LLVM value of arbitrary type.
//
//===----------------------------------------------------------------------===//


#ifndef GENERIC_VALUE_H
#define GENERIC_VALUE_H

#include "llvm/Support/DataTypes.h"

namespace llvm {

typedef uintptr_t PointerTy;
class APInt;
class Type;

union GenericValue {
  bool            Int1Val;
  unsigned char   Int8Val;
  unsigned short  Int16Val;
  unsigned int    Int32Val;
  uint64_t        Int64Val;
  APInt          *APIntVal;
  double          DoubleVal;
  float           FloatVal;
  struct { unsigned int first; unsigned int second; } UIntPairVal;
  PointerTy       PointerVal;
  unsigned char   Untyped[8];

  GenericValue() {}
  GenericValue(void *V) {
    PointerVal = (PointerTy)(intptr_t)V;
  }
};

inline GenericValue PTOGV(void *P) { return GenericValue(P); }
inline void* GVTOP(const GenericValue &GV) {
  return (void*)(intptr_t)GV.PointerVal;
}

} // End llvm namespace
#endif
