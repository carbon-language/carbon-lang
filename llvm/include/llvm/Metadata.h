//===-- llvm/Metadata.h - Metadata definitions ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// @file
/// This file contains the declarations for metadata subclasses.
/// They represent the different flavors of metadata that live in LLVM.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MDNODE_H
#define LLVM_MDNODE_H

#include "llvm/User.h"
#include "llvm/Type.h"
#include "llvm/OperandTraits.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ValueHandle.h"

namespace llvm {
class Constant;
struct LLVMContext;
template<class ConstantClass, class TypeClass, class ValType>
struct ConstantCreator;

//===----------------------------------------------------------------------===//
// MetadataBase  - A base class for MDNode, MDString and NamedMDNode.
class MetadataBase : public User {
private:
  /// ReservedSpace - The number of operands actually allocated.  NumOperands is
  /// the number actually in use.
  unsigned ReservedSpace;

protected:
  MetadataBase(const Type *Ty, unsigned scid)
    : User(Ty, scid, NULL, 0), ReservedSpace(0) {}

  void resizeOperands(unsigned NumOps);
public:
  /// getType() specialization - Type is always MetadataTy.
  ///
  inline const Type *getType() const {
    return Type::MetadataTy;
  }

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.  This always returns false because getNullValue will never
  /// produce metadata.
  virtual bool isNullValue() const {
    return false;
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const MetadataBase *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueID() == MDStringVal || V->getValueID() == MDNodeVal
      || V->getValueID() == NamedMDNodeVal;
  }
};

//===----------------------------------------------------------------------===//
/// MDString - a single uniqued string.
/// These are used to efficiently contain a byte sequence for metadata.
/// MDString is always unnamd.
class MDString : public MetadataBase {
  MDString(const MDString &);            // DO NOT IMPLEMENT
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
  unsigned getNumOperands();             // DO NOT IMPLEMENT

  StringRef Str;
protected:
  explicit MDString(const char *begin, unsigned l)
    : MetadataBase(Type::MetadataTy, Value::MDStringVal), Str(begin, l) {}

public:
  // Do not allocate any space for operands.
  void *operator new(size_t s) {
    return User::operator new(s, 0);
  }
  static MDString *get(LLVMContext &Context, const StringRef &Str);
  
  StringRef getString() const { return Str; }

  unsigned length() const { return Str.size(); }

  /// begin() - Pointer to the first byte of the string.
  ///
  const char *begin() const { return Str.begin(); }

  /// end() - Pointer to one byte past the end of the string.
  ///
  const char *end() const { return Str.end(); }

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const MDString *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueID() == MDStringVal;
  }
};

//===----------------------------------------------------------------------===//
/// MDNode - a tuple of other values.
/// These contain a list of the values that represent the metadata. 
/// MDNode is always unnamed.
class MDNode : public MetadataBase {
  MDNode(const MDNode &);                // DO NOT IMPLEMENT
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
  // getNumOperands - Make this only available for private uses.
  unsigned getNumOperands() { return User::getNumOperands();  }

  SmallVector<WeakVH, 4> Node;
  
  friend struct ConstantCreator<MDNode, Type, std::vector<Value*> >;
protected:
  explicit MDNode(Value*const* Vals, unsigned NumVals);
public:
  // Do not allocate any space for operands.
  void *operator new(size_t s) {
    return User::operator new(s, 0);
  }
  // Constructors and destructors.
  static MDNode *get(LLVMContext &Context, 
                     Value* const* Vals, unsigned NumVals);

  /// dropAllReferences - Remove all uses and clear node vector.
  void dropAllReferences();

  /// ~MDNode - Destroy MDNode.
  ~MDNode();
  
  /// getElement - Return specified element.
  Value *getElement(unsigned i) const {
    assert (getNumElements() > i && "Invalid element number!");
    return Node[i];
  }

  /// getNumElements - Return number of MDNode elements.
  unsigned getNumElements() const {
    return Node.size();
  }

  // Element access
  typedef SmallVectorImpl<WeakVH>::const_iterator const_elem_iterator;
  typedef SmallVectorImpl<WeakVH>::iterator elem_iterator;
  /// elem_empty - Return true if MDNode is empty.
  bool elem_empty() const                { return Node.empty(); }
  const_elem_iterator elem_begin() const { return Node.begin(); }
  const_elem_iterator elem_end() const   { return Node.end();   }
  elem_iterator elem_begin()             { return Node.begin(); }
  elem_iterator elem_end()               { return Node.end();   }

  /// getType() specialization - Type is always MetadataTy.
  ///
  inline const Type *getType() const {
    return Type::MetadataTy;
  }

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.  This always returns false because getNullValue will never
  /// produce metadata.
  virtual bool isNullValue() const {
    return false;
  }

  virtual void replaceUsesOfWithOnConstant(Value *From, Value *To, Use *U) {
    llvm_unreachable("This should never be called because MDNodes have no ops");
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const MDNode *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueID() == MDNodeVal;
  }
};

//===----------------------------------------------------------------------===//
/// WeakMetadataVH - a weak value handle for metadata.
class WeakMetadataVH : public WeakVH {
public:
  WeakMetadataVH() : WeakVH() {}
  WeakMetadataVH(MetadataBase *M) : WeakVH(M) {}
  WeakMetadataVH(const WeakMetadataVH &RHS) : WeakVH(RHS) {}
  
  operator Value*() const {
    llvm_unreachable("WeakMetadataVH only handles Metadata");
  }

  operator MetadataBase*() const {
   return dyn_cast_or_null<MetadataBase>(getValPtr());
  }
};

//===----------------------------------------------------------------------===//
/// NamedMDNode - a tuple of other metadata. 
/// NamedMDNode is always named. All NamedMDNode element has a type of metadata.
template<typename ValueSubClass, typename ItemParentClass>
  class SymbolTableListTraits;

class NamedMDNode : public MetadataBase, public ilist_node<NamedMDNode> {
  friend class SymbolTableListTraits<NamedMDNode, Module>;
  friend struct LLVMContextImpl;

  NamedMDNode(const NamedMDNode &);      // DO NOT IMPLEMENT
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
  // getNumOperands - Make this only available for private uses.
  unsigned getNumOperands() { return User::getNumOperands();  }

  Module *Parent;
  SmallVector<WeakMetadataVH, 4> Node;
  typedef SmallVectorImpl<WeakMetadataVH>::iterator elem_iterator;

protected:
  explicit NamedMDNode(const Twine &N, MetadataBase*const* Vals, 
                       unsigned NumVals, Module *M = 0);
public:
  // Do not allocate any space for operands.
  void *operator new(size_t s) {
    return User::operator new(s, 0);
  }
  static NamedMDNode *Create(const Twine &N, MetadataBase*const*MDs, 
                             unsigned NumMDs, Module *M = 0) {
    return new NamedMDNode(N, MDs, NumMDs, M);
  }

  /// eraseFromParent - Drop all references and remove the node from parent
  /// module.
  void eraseFromParent();

  /// dropAllReferences - Remove all uses and clear node vector.
  void dropAllReferences();

  /// ~NamedMDNode - Destroy NamedMDNode.
  ~NamedMDNode();

  /// getParent - Get the module that holds this named metadata collection.
  inline Module *getParent() { return Parent; }
  inline const Module *getParent() const { return Parent; }
  void setParent(Module *M) { Parent = M; }

  /// getElement - Return specified element.
  MetadataBase *getElement(unsigned i) const {
    assert (getNumElements() > i && "Invalid element number!");
    return Node[i];
  }

  /// getNumElements - Return number of NamedMDNode elements.
  unsigned getNumElements() const {
    return Node.size();
  }

  /// addElement - Add metadata element.
  void addElement(MetadataBase *M) {
    resizeOperands(0);
    OperandList[NumOperands++] = M;
    Node.push_back(WeakMetadataVH(M));
  }

  typedef SmallVectorImpl<WeakMetadataVH>::const_iterator const_elem_iterator;
  bool elem_empty() const                { return Node.empty(); }
  const_elem_iterator elem_begin() const { return Node.begin(); }
  const_elem_iterator elem_end() const   { return Node.end();   }
  elem_iterator elem_begin()             { return Node.begin(); }
  elem_iterator elem_end()               { return Node.end();   }

  /// getType() specialization - Type is always MetadataTy.
  ///
  inline const Type *getType() const {
    return Type::MetadataTy;
  }

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.  This always returns false because getNullValue will never
  /// produce metadata.
  virtual bool isNullValue() const {
    return false;
  }

  virtual void replaceUsesOfWithOnConstant(Value *From, Value *To, Use *U) {
    llvm_unreachable(
                "This should never be called because NamedMDNodes have no ops");
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const NamedMDNode *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueID() == NamedMDNodeVal;
  }
};

} // end llvm namespace

#endif
