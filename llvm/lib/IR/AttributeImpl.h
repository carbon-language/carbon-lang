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
#include <string>

namespace llvm {

class Constant;
class LLVMContext;

//===----------------------------------------------------------------------===//
/// \class
/// \brief This class represents a single, uniqued attribute. That attribute
/// could be a single enum, a tuple, or a string.
class AttributeImpl : public FoldingSetNode {
  LLVMContext &Context;
  Constant *Kind;
  SmallVector<Constant*, 0> Vals;

  // AttributesImpl is uniqued, these should not be publicly available.
  void operator=(const AttributeImpl &) LLVM_DELETED_FUNCTION;
  AttributeImpl(const AttributeImpl &) LLVM_DELETED_FUNCTION;
public:
  AttributeImpl(LLVMContext &C, Constant *Kind,
                ArrayRef<Constant*> Vals = ArrayRef<Constant*>())
    : Context(C), Kind(Kind), Vals(Vals.begin(), Vals.end()) {}

  LLVMContext &getContext() { return Context; }

  bool hasAttribute(Attribute::AttrKind A) const;

  Constant *getAttributeKind() const { return Kind; }
  ArrayRef<Constant*> getAttributeValues() const { return Vals; }

  uint64_t getAlignment() const;
  uint64_t getStackAlignment() const;

  /// \brief Equality and non-equality comparison operators.
  bool operator==(Attribute::AttrKind Kind) const;
  bool operator!=(Attribute::AttrKind Kind) const;

  bool operator==(StringRef Kind) const;
  bool operator!=(StringRef Kind) const;

  /// \brief Used when sorting the attributes.
  bool operator<(const AttributeImpl &AI) const;

  void Profile(FoldingSetNodeID &ID) const {
    Profile(ID, Kind, Vals);
  }
  static void Profile(FoldingSetNodeID &ID, Constant *Kind,
                      ArrayRef<Constant*> Vals) {
    ID.AddPointer(Kind);
    for (unsigned I = 0, E = Vals.size(); I != E; ++I)
      ID.AddPointer(Vals[I]);
  }

  // FIXME: Remove this!
  static uint64_t getAttrMask(Attribute::AttrKind Val);
};

//===----------------------------------------------------------------------===//
/// \class
/// \brief This class represents a group of attributes that apply to one
/// element: function, return type, or parameter.
class AttributeSetNode : public FoldingSetNode {
  SmallVector<Attribute, 4> AttrList;

  AttributeSetNode(ArrayRef<Attribute> Attrs)
    : AttrList(Attrs.begin(), Attrs.end()) {}

  // AttributesSetNode is uniqued, these should not be publicly available.
  void operator=(const AttributeSetNode &) LLVM_DELETED_FUNCTION;
  AttributeSetNode(const AttributeSetNode &) LLVM_DELETED_FUNCTION;
public:
  static AttributeSetNode *get(LLVMContext &C, ArrayRef<Attribute> Attrs);

  bool hasAttribute(Attribute::AttrKind Kind) const;
  bool hasAttributes() const { return !AttrList.empty(); }

  unsigned getAlignment() const;
  unsigned getStackAlignment() const;
  std::string getAsString() const;

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

  typedef std::pair<unsigned, AttributeSetNode*> IndexAttrPair;
  SmallVector<IndexAttrPair, 4> AttrNodes;

  // AttributesSet is uniqued, these should not be publicly available.
  void operator=(const AttributeSetImpl &) LLVM_DELETED_FUNCTION;
  AttributeSetImpl(const AttributeSetImpl &) LLVM_DELETED_FUNCTION;
public:
  AttributeSetImpl(LLVMContext &C,
                   ArrayRef<std::pair<unsigned, AttributeSetNode*> > attrs)
    : Context(C), AttrNodes(attrs.begin(), attrs.end()) {}

  /// \brief Get the context that created this AttributeSetImpl.
  LLVMContext &getContext() { return Context; }

  /// \brief Return the number of attributes this AttributeSet contains.
  unsigned getNumAttributes() const { return AttrNodes.size(); }

  /// \brief Get the index of the given "slot" in the AttrNodes list. This index
  /// is the index of the return, parameter, or function object that the
  /// attributes are applied to, not the index into the AttrNodes list where the
  /// attributes reside.
  uint64_t getSlotIndex(unsigned Slot) const {
    return AttrNodes[Slot].first;
  }

  /// \brief Retrieve the attributes for the given "slot" in the AttrNode list.
  /// \p Slot is an index into the AttrNodes list, not the index of the return /
  /// parameter/ function which the attributes apply to.
  AttributeSet getSlotAttributes(unsigned Slot) const {
    // FIXME: This needs to use AttrNodes instead.
    return AttributeSet::get(Context, AttrNodes[Slot]);
  }

  /// \brief Retrieve the attribute set node for the given "slot" in the
  /// AttrNode list.
  AttributeSetNode *getSlotNode(unsigned Slot) const {
    return AttrNodes[Slot].second;
  }

  typedef AttributeSetNode::iterator       iterator;
  typedef AttributeSetNode::const_iterator const_iterator;

  iterator begin(unsigned Idx)
    { return AttrNodes[Idx].second->begin(); }
  iterator end(unsigned Idx)
    { return AttrNodes[Idx].second->end(); }

  const_iterator begin(unsigned Idx) const
    { return AttrNodes[Idx].second->begin(); }
  const_iterator end(unsigned Idx) const
    { return AttrNodes[Idx].second->end(); }

  void Profile(FoldingSetNodeID &ID) const {
    Profile(ID, AttrNodes);
  }
  static void Profile(FoldingSetNodeID &ID,
                      ArrayRef<std::pair<unsigned, AttributeSetNode*> > Nodes) {
    for (unsigned i = 0, e = Nodes.size(); i != e; ++i) {
      ID.AddInteger(Nodes[i].first);
      ID.AddPointer(Nodes[i].second);
    }
  }

  // FIXME: This atrocity is temporary.
  uint64_t Raw(uint64_t Index) const;
};

} // end llvm namespace

#endif
