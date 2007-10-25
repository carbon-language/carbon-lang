//==- Serialization.h - Generic Object Serialization to Bitcode ---*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines traits for primitive types used for both object
// serialization and deserialization.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BITCODE_SERIALIZE
#define LLVM_BITCODE_SERIALIZE

#include "llvm/Bitcode/SerializationFwd.h"

namespace llvm {
  
#define SERIALIZE_INT_TRAIT(TYPE)\
template <> struct SerializeTrait<TYPE> {\
  static void Emit(Serializer& S, TYPE X);\
  static void Read(Deserializer& S, TYPE& X);\
  static TYPE ReadVal(Deserializer& S); };

SERIALIZE_INT_TRAIT(bool)
SERIALIZE_INT_TRAIT(unsigned char)
SERIALIZE_INT_TRAIT(unsigned short)
SERIALIZE_INT_TRAIT(unsigned int)
SERIALIZE_INT_TRAIT(unsigned long)

#undef SERIALIZE_INT_TRAIT
  
} // end namespace llvm

#endif
