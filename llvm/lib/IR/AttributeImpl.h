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
/// \brief A set of classes that contain the kind and (optional) value of the
/// attribute object. There are three main categories: enum attribute entries,
/// represented by Attribute::AttrKind; alignment attribute entries; and string
/// attribute enties, which are for target-dependent attributes.
class AttributeEntry {
  unsigned char KindID;
protected:
  enum AttrEntryKind {
    EnumAttrEntry,
    AlignAttrEntry,
    StringAttrEntry
  };
public:
  AttributeEntry(AttrEntryKind Kind)
    : KindID(Kind) {}
  virtual ~AttributeEntry() {}

  unsigned getKindID() const { return KindID; }

  static inline bool classof(const AttributeEntry *) { return true; }
};

class EnumAttributeEntry : public AttributeEntry {
  Attribute::AttrKind Kind;
public:
  EnumAttributeEntry(Attribute::AttrKind Kind)
    : AttributeEntry(EnumAttrEntry), Kind(Kind) {}

  Attribute::AttrKind getEnumKind() const { return Kind; }

  static inline bool classof(const AttributeEntry *AE) {
    return AE->getKindID() == EnumAttrEntry;
  }
  static inline bool classof(const EnumAttributeEntry *) { return true; }
};

class AlignAttributeEntry : public AttributeEntry {
  Attribute::AttrKind Kind;
  unsigned Align;
public:
  AlignAttributeEntry(Attribute::AttrKind Kind, unsigned Align)
    : AttributeEntry(AlignAttrEntry), Kind(Kind), Align(Align) {}

  Attribute::AttrKind getEnumKind() const { return Kind; }
  unsigned getAlignment() const { return Align; }

  static inline bool classof(const AttributeEntry *AE) {
    return AE->getKindID() == AlignAttrEntry;
  }
  static inline bool classof(const AlignAttributeEntry *) { return true; }
};

class StringAttributeEntry : public AttributeEntry {
  std::string Kind;
  std::string Val;
public:
  StringAttributeEntry(StringRef Kind, StringRef Val = StringRef())
    : AttributeEntry(StringAttrEntry), Kind(Kind), Val(Val) {}

  StringRef getStringKind() const { return Kind; }
  StringRef getStringValue() const { return Val; }

  static inline bool classof(const AttributeEntry *AE) {
    return AE->getKindID() == StringAttrEntry;
  }
  static inline bool classof(const StringAttributeEntry *) { return true; }
};

//===----------------------------------------------------------------------===//
/// \class
/// \brief This class represents a single, uniqued attribute. That attribute
/// could be a single enum, a tuple, or a string.
class AttributeImpl : public FoldingSetNode {
  LLVMContext &Context;  ///< Global context for uniquing objects

  AttributeEntry *Entry; ///< Holds the kind and value of the attribute

  // AttributesImpl is uniqued, these should not be publicly available.
  void operator=(const AttributeImpl &) LLVM_DELETED_FUNCTION;
  AttributeImpl(const AttributeImpl &) LLVM_DELETED_FUNCTION;
public:
  AttributeImpl(LLVMContext &C, Attribute::AttrKind Kind);
  AttributeImpl(LLVMContext &C, Attribute::AttrKind Kind, unsigned Align);
  AttributeImpl(LLVMContext &C, StringRef Kind, StringRef Val = StringRef());
  ~AttributeImpl();

  LLVMContext &getContext() { return Context; }

  bool isEnumAttribute() const;
  bool isAlignAttribute() const;
  bool isStringAttribute() const;

  bool hasAttribute(Attribute::AttrKind A) const;
  bool hasAttribute(StringRef Kind) const;

  Attribute::AttrKind getKindAsEnum() const;
  uint64_t getValueAsInt() const;

  StringRef getKindAsString() const;
  StringRef getValueAsString() const;

  /// \brief Used when sorting the attributes.
  bool operator<(const AttributeImpl &AI) const;

  void Profile(FoldingSetNodeID &ID) const {
    if (isEnumAttribute())
      Profile(ID, getKindAsEnum(), 0);
    else if (isAlignAttribute())
      Profile(ID, getKindAsEnum(), getValueAsInt());
    else
      Profile(ID, getKindAsString(), getValueAsString());
  }
  static void Profile(FoldingSetNodeID &ID, Attribute::AttrKind Kind,
                      uint64_t Val) {
    ID.AddInteger(Kind);
    if (Val) ID.AddInteger(Val);
  }
  static void Profile(FoldingSetNodeID &ID, StringRef Kind, StringRef Values) {
    ID.AddString(Kind);
    if (!Values.empty()) ID.AddString(Values);
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
  bool hasAttribute(StringRef Kind) const;
  bool hasAttributes() const { return !AttrList.empty(); }

  Attribute getAttribute(Attribute::AttrKind Kind) const;
  Attribute getAttribute(StringRef Kind) const;

  unsigned getAlignment() const;
  unsigned getStackAlignment() const;
  std::string getAsString(bool TargetIndependent, bool InAttrGrp) const;

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
