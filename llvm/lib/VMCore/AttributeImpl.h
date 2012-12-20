//===-- AttributeImpl.h - Attribute Internals -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file defines various helper methods and classes used by
/// LLVMContextImpl for creating and managing attributes.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ATTRIBUTESIMPL_H
#define LLVM_ATTRIBUTESIMPL_H

#include "llvm/ADT/FoldingSet.h"
#include "llvm/Attributes.h"

namespace llvm {

class LLVMContext;

//===----------------------------------------------------------------------===//
/// \class
/// \brief This class represents a single, uniqued attribute. That attribute
/// could be a single enum, a tuple, or a string. It uses a discriminated union
/// to distinguish them.
class AttributeImpl : public FoldingSetNode {
  uint64_t Bits;                // FIXME: We will be expanding this.
public:
  AttributeImpl(uint64_t bits) : Bits(bits) {}

  bool hasAttribute(uint64_t A) const;

  bool hasAttributes() const;
  bool hasAttributes(const Attribute &A) const;

  uint64_t getAlignment() const;
  uint64_t getStackAlignment() const;

  uint64_t Raw() const { return Bits; } // FIXME: Remove.

  static uint64_t getAttrMask(uint64_t Val);

  void Profile(FoldingSetNodeID &ID) const {
    Profile(ID, Bits);
  }
  static void Profile(FoldingSetNodeID &ID, uint64_t Bits) {
    ID.AddInteger(Bits);
  }
};

//===----------------------------------------------------------------------===//
/// \class
/// \brief This class represents a set of attributes.
class AttributeSetImpl : public FoldingSetNode {
  // AttributesList is uniqued, these should not be publicly available.
  void operator=(const AttributeSetImpl &) LLVM_DELETED_FUNCTION;
  AttributeSetImpl(const AttributeSetImpl &) LLVM_DELETED_FUNCTION;
public:
  LLVMContext &Context;
  SmallVector<AttributeWithIndex, 4> Attrs;

  AttributeSetImpl(LLVMContext &C, ArrayRef<AttributeWithIndex> attrs)
    : Context(C), Attrs(attrs.begin(), attrs.end()) {}

  void Profile(FoldingSetNodeID &ID) const {
    Profile(ID, Attrs);
  }
  static void Profile(FoldingSetNodeID &ID, ArrayRef<AttributeWithIndex> Attrs){
    for (unsigned i = 0, e = Attrs.size(); i != e; ++i) {
      ID.AddInteger(Attrs[i].Attrs.Raw());
      ID.AddInteger(Attrs[i].Index);
    }
  }
};

} // end llvm namespace

#endif
