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

class Constant;
class LLVMContext;

//===----------------------------------------------------------------------===//
/// \class
/// \brief This class represents a single, uniqued attribute. That attribute
/// could be a single enum, a tuple, or a string.
class AttributeImpl : public FoldingSetNode {
  Constant *Data;
  SmallVector<Constant*, 0> Vals;
public:
  explicit AttributeImpl(LLVMContext &C, uint64_t data);
  explicit AttributeImpl(LLVMContext &C, Attribute::AttrKind data);
  AttributeImpl(LLVMContext &C, Attribute::AttrKind data,
                ArrayRef<Constant*> values);
  AttributeImpl(LLVMContext &C, StringRef data);

  ArrayRef<Constant*> getValues() const {
    return Vals;
  }

  bool contains(Attribute::AttrKind Kind) const;
  bool contains(StringRef Kind) const;

  bool hasAttribute(uint64_t A) const;

  bool hasAttributes() const;
  bool hasAttributes(const Attribute &A) const;

  uint64_t getAlignment() const;
  uint64_t getStackAlignment() const;

  bool operator==(Attribute::AttrKind Kind) const {
    return contains(Kind);
  }
  bool operator!=(Attribute::AttrKind Kind) const {
    return !contains(Kind);
  }

  bool operator==(StringRef Kind) const {
    return contains(Kind);
  }
  bool operator!=(StringRef Kind) const {
    return !contains(Kind);
  }

  uint64_t getBitMask() const;         // FIXME: Remove.

  static uint64_t getAttrMask(uint64_t Val);

  void Profile(FoldingSetNodeID &ID) const {
    Profile(ID, Data, Vals);
  }
  static void Profile(FoldingSetNodeID &ID, Constant *Data,
                      ArrayRef<Constant*> Vals) {
    ID.AddPointer(Data);
    for (ArrayRef<Constant*>::iterator I = Vals.begin(), E = Vals.end();
         I != E; ++I)
      ID.AddPointer(*I);
  }
};

//===----------------------------------------------------------------------===//
/// \class
/// \brief This class represents a set of attributes.
class AttributeSetImpl : public FoldingSetNode {
  // AttributesSet is uniqued, these should not be publicly available.
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
      ID.AddInteger(Attrs[i].Attrs.getBitMask());
      ID.AddInteger(Attrs[i].Index);
    }
  }
};

} // end llvm namespace

#endif
