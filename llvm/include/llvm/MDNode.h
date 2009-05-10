//===-- llvm/Metadata.h - Constant class subclass definitions ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// @file
/// This file contains the declarations for the subclasses of Constant, 
/// which represent the different flavors of constant values that live in LLVM.
/// Note that Constants are immutable (once created they never change) and are 
/// fully shared by structural equivalence.  This means that two structurally
/// equivalent constants will always have the same address.  Constant's are
/// created on demand as needed and never deleted: thus clients don't have to
/// worry about the lifetime of the objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MDNODE_H
#define LLVM_MDNODE_H

#include "llvm/Constant.h"
#include "llvm/Type.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ValueHandle.h"

namespace llvm {

//===----------------------------------------------------------------------===//
/// MDNode - a tuple of other values.
/// These contain a list of the Constants that represent the metadata. The
/// operand list is always empty, query the element list instead.
///
/// This class will attempt to keep track of values as they are modified. When
/// a value is replaced the element will be replaced with it, and when the
/// value is deleted the element is set to a null pointer. In order to preserve
/// structural equivalence while the elements mutate, the MDNode may call
/// replaceAllUsesWith on itself. Because of this, users of MDNode must use a
/// WeakVH or CallbackVH to hold the node pointer if there is a chance that one
/// of the elements held by the node may change.
///
class MDNode : public Constant, public FoldingSetNode {
  MDNode(const MDNode &);      // DO NOT IMPLEMENT

  friend class ElementVH;
  struct ElementVH : public CallbackVH {
    MDNode *OwningNode;

    ElementVH(Value *V, MDNode *Parent)
      : CallbackVH(V), OwningNode(Parent) {}

    ~ElementVH() {}

    /// deleted - Set this entry in the MDNode to 'null'. This will reallocate
    /// the MDNode.
    virtual void deleted() {
      OwningNode->replaceElement(this->operator Value*(), 0);
    }

    /// allUsesReplacedWith - Modify the MDNode by replacing this entry with
    /// new_value. This will reallocate the MDNode.
    virtual void allUsesReplacedWith(Value *new_value) {
      OwningNode->replaceElement(this->operator Value*(), new_value);
    }
  };

  void replaceElement(Value *From, Value *To);

  SmallVector<ElementVH, 4> Node;
  typedef SmallVectorImpl<ElementVH>::iterator elem_iterator;
protected:
  explicit MDNode(Value*const* Vals, unsigned NumVals);
public:
  typedef SmallVectorImpl<ElementVH>::const_iterator const_elem_iterator;

  /// get() - Static factory methods - Return objects of the specified value.
  ///
  static MDNode *get(Value*const* Vals, unsigned NumVals);

  Value *getElement(unsigned i) const {
    return Node[i];
  }

  unsigned getNumElements() const {
    return Node.size();
  }

  const_elem_iterator elem_begin() const {
    return Node.begin();
  }

  const_elem_iterator elem_end() const {
    return Node.end();
  }

  /// getType() specialization - Type is always an empty struct.
  ///
  inline const Type *getType() const {
    return Type::EmptyStructTy;
  }

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.  This always returns false because getNullValue will never
  /// produce metadata.
  virtual bool isNullValue() const {
    return false;
  }

  /// Profile - calculate a unique identifier for this MDNode to collapse
  /// duplicates
  void Profile(FoldingSetNodeID &ID) const;

  virtual void destroyConstant();
  virtual void replaceUsesOfWithOnConstant(Value *From, Value *To, Use *U) {
    assert(0 && "This should never be called because MDNodes have no ops");
    abort();
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const MDNode *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueID() == MDNodeVal;
  }
};

} // end llvm namespace

#endif
