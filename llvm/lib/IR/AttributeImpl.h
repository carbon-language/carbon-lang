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
#include "llvm/IR/Attributes.h"

namespace llvm {

class Constant;
class LLVMContext;

//===----------------------------------------------------------------------===//
/// \class
/// \brief This class represents a single, uniqued attribute. That attribute
/// could be a single enum, a tuple, or a string.
class AttributeImpl : public FoldingSetNode {
  LLVMContext &Context;
  Constant *Data;
  SmallVector<Constant*, 0> Vals;
public:
  explicit AttributeImpl(LLVMContext &C, uint64_t data);
  explicit AttributeImpl(LLVMContext &C, Attribute::AttrKind data);
  AttributeImpl(LLVMContext &C, Attribute::AttrKind data,
                ArrayRef<Constant*> values);
  AttributeImpl(LLVMContext &C, StringRef data);

  LLVMContext &getContext() { return Context; }

  ArrayRef<Constant*> getValues() const { return Vals; }

  bool hasAttribute(Attribute::AttrKind A) const;

  bool hasAttributes() const;

  uint64_t getAlignment() const;
  void setAlignment(unsigned Align);

  uint64_t getStackAlignment() const;
  void setStackAlignment(unsigned Align);

  bool operator==(Attribute::AttrKind Kind) const;
  bool operator!=(Attribute::AttrKind Kind) const;

  bool operator==(StringRef Kind) const;
  bool operator!=(StringRef Kind) const;

  uint64_t Raw() const;         // FIXME: Remove.

  static uint64_t getAttrMask(Attribute::AttrKind Val);

  void Profile(FoldingSetNodeID &ID) const {
    Profile(ID, Data, Vals);
  }
  static void Profile(FoldingSetNodeID &ID, Constant *Data,
                      ArrayRef<Constant*> Vals);
};

//===----------------------------------------------------------------------===//
/// \class
/// \brief This class represents a set of attributes.
class AttributeSetImpl : public FoldingSetNode {
  LLVMContext &Context;
  SmallVector<AttributeWithIndex, 4> AttrList;

  // AttributesSet is uniqued, these should not be publicly available.
  void operator=(const AttributeSetImpl &) LLVM_DELETED_FUNCTION;
  AttributeSetImpl(const AttributeSetImpl &) LLVM_DELETED_FUNCTION;
public:
  AttributeSetImpl(LLVMContext &C, ArrayRef<AttributeWithIndex> attrs)
    : Context(C), AttrList(attrs.begin(), attrs.end()) {}

  LLVMContext &getContext() { return Context; }
  ArrayRef<AttributeWithIndex> getAttributes() const { return AttrList; }
  unsigned getNumAttributes() const { return AttrList.size(); }

  void Profile(FoldingSetNodeID &ID) const {
    Profile(ID, AttrList);
  }
  static void Profile(FoldingSetNodeID &ID,
                      ArrayRef<AttributeWithIndex> AttrList){
    for (unsigned i = 0, e = AttrList.size(); i != e; ++i) {
      ID.AddInteger(AttrList[i].Index);
      ID.AddInteger(AttrList[i].Attrs.Raw());
    }
  }
};

} // end llvm namespace

#endif
