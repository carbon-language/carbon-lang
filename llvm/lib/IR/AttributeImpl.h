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
  uint64_t getStackAlignment() const;

  bool operator==(Attribute::AttrKind Kind) const;
  bool operator!=(Attribute::AttrKind Kind) const;

  bool operator==(StringRef Kind) const;
  bool operator!=(StringRef Kind) const;

  bool operator<(const AttributeImpl &AI) const;

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
/// \brief This class represents a group of attributes that apply to one
/// element: function, return type, or parameter.
class AttributeSetNode : public FoldingSetNode {
  SmallVector<Attribute, 4> AttrList;

  AttributeSetNode(ArrayRef<Attribute> Attrs)
    : AttrList(Attrs.begin(), Attrs.end()) {}
public:
  static AttributeSetNode *get(LLVMContext &C, ArrayRef<Attribute> Attrs);

  typedef SmallVectorImpl<Attribute>::iterator       iterator;
  typedef SmallVectorImpl<Attribute>::const_iterator const_iterator;

  iterator begin() { return AttrList.begin(); }
  iterator end()   { return AttrList.end(); }

  const_iterator begin() const { return AttrList.begin(); }
  const_iterator end() const   { return AttrList.end(); }

  void Profile(FoldingSetNodeID &ID) const {
    Profile(ID, AttrList);
  }
  static void Profile(FoldingSetNodeID &ID, ArrayRef<Attribute> AttrList) {
    for (unsigned I = 0, E = AttrList.size(); I != E; ++I)
      AttrList[I].Profile(ID);
  }
};

//===----------------------------------------------------------------------===//
/// \class
/// \brief This class represents a set of attributes that apply to the function,
/// return type, and parameters.
class AttributeSetImpl : public FoldingSetNode {
  friend class AttributeSet;

  LLVMContext &Context;
  SmallVector<AttributeWithIndex, 4> AttrList;

  SmallVector<std::pair<uint64_t, AttributeSetNode*>, 4> AttrNodes;

  // AttributesSet is uniqued, these should not be publicly available.
  void operator=(const AttributeSetImpl &) LLVM_DELETED_FUNCTION;
  AttributeSetImpl(const AttributeSetImpl &) LLVM_DELETED_FUNCTION;
public:
  AttributeSetImpl(LLVMContext &C, ArrayRef<AttributeWithIndex> attrs);

  /// \brief Get the context that created this AttributeSetImpl.
  LLVMContext &getContext() { return Context; }

  ArrayRef<AttributeWithIndex> getAttributes() const { return AttrList; }

  /// \brief Return the number of attributes this AttributeSet contains.
  unsigned getNumAttributes() const { return AttrNodes.size(); }

  /// \brief Get the index of the given "slot" in the AttrNodes list. This index
  /// is the index of the return, parameter, or function object that the
  /// attributes are applied to, not the index into the AttrNodes list where the
  /// attributes reside.
  unsigned getSlotIndex(unsigned Slot) const { return AttrNodes[Slot].first; }

  /// \brief Retrieve the attributes for the given "slot" in the AttrNode list.
  /// \p Slot is an index into the AttrNodes list, not the index of the return /
  /// parameter/ function which the attributes apply to.
  AttributeSet getSlotAttributes(unsigned Slot) const {
    // FIXME: This needs to use AttrNodes instead.
    return AttributeSet::get(Context, AttrList[Slot]);
  }

  void Profile(FoldingSetNodeID &ID) const {
    Profile(ID, AttrList);
  }
  static void Profile(FoldingSetNodeID &ID,
                      ArrayRef<AttributeWithIndex> AttrList) {
    for (unsigned i = 0, e = AttrList.size(); i != e; ++i) {
      ID.AddInteger(AttrList[i].Index);
      ID.AddInteger(AttrList[i].Attrs.Raw());
    }
  }

  static void Profile(FoldingSetNodeID &ID,
                      ArrayRef<std::pair<uint64_t, AttributeSetNode*> > Nodes) {
    for (unsigned i = 0, e = Nodes.size(); i != e; ++i) {
      ID.AddInteger(Nodes[i].first);
      ID.AddPointer(Nodes[i].second);
    }
  }
};

} // end llvm namespace

#endif
