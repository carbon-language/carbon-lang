//===- ValueTypes.cpp - Tablegen extended ValueType implementation --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The EVT type is used by tablegen as well as in LLVM. In order to handle
// extended types, the EVT type uses support functions that call into
// LLVM's type system code. These aren't accessible in tablegen, so this
// file provides simple replacements.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ValueTypes.h"
#include <map>
using namespace llvm;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wweak-vtables"

namespace llvm {

class Type {
public:
  virtual unsigned getSizeInBits() const = 0;
  virtual ~Type() {}
};

}

class ExtendedIntegerType : public Type {
  unsigned BitWidth;
public:
  explicit ExtendedIntegerType(unsigned bits)
    : BitWidth(bits) {}
  unsigned getSizeInBits() const {
    return getBitWidth();
  }
  unsigned getBitWidth() const {
    return BitWidth;
  }
};

class ExtendedVectorType : public Type {
  EVT ElementType;
  unsigned NumElements;
public:
  ExtendedVectorType(EVT elty, unsigned num)
    : ElementType(elty), NumElements(num) {}
  unsigned getSizeInBits() const {
    return getNumElements() * getElementType().getSizeInBits();
  }
  EVT getElementType() const {
    return ElementType;
  }
  unsigned getNumElements() const {
    return NumElements;
  }
};

#pragma clang diagnostic pop

static std::map<unsigned, const Type *>
  ExtendedIntegerTypeMap;
static std::map<std::pair<uintptr_t, uintptr_t>, const Type *>
  ExtendedVectorTypeMap;

bool EVT::isExtendedFloatingPoint() const {
  assert(isExtended() && "Type is not extended!");
  // Extended floating-point types are not supported yet.
  return false;
}

bool EVT::isExtendedInteger() const {
  assert(isExtended() && "Type is not extended!");
  return dynamic_cast<const ExtendedIntegerType *>(LLVMTy) != 0;
}

bool EVT::isExtendedVector() const {
  assert(isExtended() && "Type is not extended!");
  return dynamic_cast<const ExtendedVectorType *>(LLVMTy) != 0;
}

bool EVT::isExtended64BitVector() const {
  assert(isExtended() && "Type is not extended!");
  return isExtendedVector() && getSizeInBits() == 64;
}

bool EVT::isExtended128BitVector() const {
  assert(isExtended() && "Type is not extended!");
  return isExtendedVector() && getSizeInBits() == 128;
}

EVT EVT::getExtendedVectorElementType() const {
  assert(isExtendedVector() && "Type is not an extended vector!");
  return static_cast<const ExtendedVectorType *>(LLVMTy)->getElementType();
}

unsigned EVT::getExtendedVectorNumElements() const {
  assert(isExtendedVector() && "Type is not an extended vector!");
  return static_cast<const ExtendedVectorType *>(LLVMTy)->getNumElements();
}

unsigned EVT::getExtendedSizeInBits() const {
  assert(isExtended() && "Type is not extended!");
  return LLVMTy->getSizeInBits();
}
