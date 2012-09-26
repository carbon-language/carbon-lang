//===-- AttributesImpl.h - Attributes Internals -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines various helper methods and classes used by LLVMContextImpl
// for creating and managing attributes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ATTRIBUTESIMPL_H
#define LLVM_ATTRIBUTESIMPL_H

#include "llvm/ADT/FoldingSet.h"

namespace llvm {

class AttributesImpl : public FoldingSetNode {
  uint64_t Bits;                // FIXME: We will be expanding this.

  void operator=(const AttributesImpl &) LLVM_DELETED_FUNCTION;
  AttributesImpl(const AttributesImpl &) LLVM_DELETED_FUNCTION;
public:
  AttributesImpl(uint64_t bits) : Bits(bits) {}

  void Profile(FoldingSetNodeID &ID) const {
    Profile(ID, Bits);
  }
  static void Profile(FoldingSetNodeID &ID, uint64_t Bits) {
    ID.AddInteger(Bits);
  }
};

} // end llvm namespace

#endif
