//===-- GenericValue.h - Represent any type of LLVM value -------*- C++ -*-===//
// 
// The GenericValue class is used to represent an LLVM value of arbitrary type.
//
//===----------------------------------------------------------------------===//


#ifndef GENERIC_VALUE_H
#define GENERIC_VALUE_H

#include "Support/DataTypes.h"

typedef uint64_t PointerTy;

union GenericValue {
  bool            BoolVal;
  unsigned char   UByteVal;
  signed   char   SByteVal;
  unsigned short  UShortVal;
  signed   short  ShortVal;
  unsigned int    UIntVal;
  signed   int    IntVal;
  uint64_t        ULongVal;
  int64_t         LongVal;
  double          DoubleVal;
  float           FloatVal;
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
#endif
