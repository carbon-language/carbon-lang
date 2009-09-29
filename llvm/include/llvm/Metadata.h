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
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ValueHandle.h"

namespace llvm {
class Constant;
class Instruction;
class LLVMContext;

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
  explicit MDString(LLVMContext &C, const char *begin, unsigned l)
    : MetadataBase(Type::getMetadataTy(C), Value::MDStringVal), Str(begin, l) {}

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
class MDNode : public MetadataBase, public FoldingSetNode {
  MDNode(const MDNode &);                // DO NOT IMPLEMENT
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
  // getNumOperands - Make this only available for private uses.
  unsigned getNumOperands() { return User::getNumOperands();  }

  friend class ElementVH;
  // Use CallbackVH to hold MDNOde elements.
  struct ElementVH : public CallbackVH {
    MDNode *Parent;
    ElementVH(Value *V, MDNode *P) : CallbackVH(V), Parent(P) {}
    ~ElementVH() {}

    virtual void deleted() {
      Parent->replaceElement(this->operator Value*(), 0);
    }

    virtual void allUsesReplacedWith(Value *NV) {
      Parent->replaceElement(this->operator Value*(), NV);
    }
  };
  // Replace each instance of F from the element list of this node with T.
  void replaceElement(Value *F, Value *T);

  SmallVector<ElementVH, 4> Node;

protected:
  explicit MDNode(LLVMContext &C, Value*const* Vals, unsigned NumVals);
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
  typedef SmallVectorImpl<ElementVH>::const_iterator const_elem_iterator;
  typedef SmallVectorImpl<ElementVH>::iterator elem_iterator;
  /// elem_empty - Return true if MDNode is empty.
  bool elem_empty() const                { return Node.empty(); }
  const_elem_iterator elem_begin() const { return Node.begin(); }
  const_elem_iterator elem_end() const   { return Node.end();   }
  elem_iterator elem_begin()             { return Node.begin(); }
  elem_iterator elem_end()               { return Node.end();   }

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.  This always returns false because getNullValue will never
  /// produce metadata.
  virtual bool isNullValue() const {
    return false;
  }

  /// Profile - calculate a unique identifier for this MDNode to collapse
  /// duplicates
  void Profile(FoldingSetNodeID &ID) const;

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
  friend class LLVMContextImpl;

  NamedMDNode(const NamedMDNode &);      // DO NOT IMPLEMENT
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
  // getNumOperands - Make this only available for private uses.
  unsigned getNumOperands() { return User::getNumOperands();  }

  Module *Parent;
  SmallVector<WeakMetadataVH, 4> Node;
  typedef SmallVectorImpl<WeakMetadataVH>::iterator elem_iterator;

protected:
  explicit NamedMDNode(LLVMContext &C, const Twine &N, MetadataBase*const* Vals, 
                       unsigned NumVals, Module *M = 0);
public:
  // Do not allocate any space for operands.
  void *operator new(size_t s) {
    return User::operator new(s, 0);
  }
  static NamedMDNode *Create(LLVMContext &C, const Twine &N, 
                             MetadataBase*const*MDs, 
                             unsigned NumMDs, Module *M = 0) {
    return new NamedMDNode(C, N, MDs, NumMDs, M);
  }

  static NamedMDNode *Create(const NamedMDNode *NMD, Module *M = 0);

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

//===----------------------------------------------------------------------===//
/// MetadataContext -
/// MetadataContext handles uniquing and assignment of IDs for custom metadata
/// types. Custom metadata handler names do not contain spaces. And the name
/// must start with an alphabet. The regular expression used to check name
/// is [a-zA-Z$._][a-zA-Z$._0-9]*
class MetadataContext {
public:
  typedef std::pair<unsigned, WeakVH> MDPairTy;
  typedef SmallVector<MDPairTy, 2> MDMapTy;
  typedef DenseMap<const Instruction *, MDMapTy> MDStoreTy;
  friend class BitcodeReader;
private:

  /// MetadataStore - Collection of metadata used in this context.
  MDStoreTy MetadataStore;

  /// MDHandlerNames - Map to hold metadata handler names.
  StringMap<unsigned> MDHandlerNames;

public:
  /// RegisterMDKind - Register a new metadata kind and return its ID.
  /// A metadata kind can be registered only once. 
  unsigned RegisterMDKind(const char *Name);

  /// getMDKind - Return metadata kind. If the requested metadata kind
  /// is not registered then return 0.
  unsigned getMDKind(const char *Name);

  /// validName - Return true if Name is a valid custom metadata handler name.
  bool validName(const char *Name);

  /// getMD - Get the metadata of given kind attached with an Instruction.
  /// If the metadata is not found then return 0.
  MDNode *getMD(unsigned Kind, const Instruction *Inst);

  /// getMDs - Get the metadata attached with an Instruction.
  const MDMapTy *getMDs(const Instruction *Inst);

  /// addMD - Attach the metadata of given kind with an Instruction.
  void addMD(unsigned Kind, MDNode *Node, Instruction *Inst);
  
  /// removeMD - Remove metadata of given kind attached with an instuction.
  void removeMD(unsigned Kind, Instruction *Inst);
  
  /// removeMDs - Remove all metadata attached with an instruction.
  void removeMDs(const Instruction *Inst);

  /// getHandlerNames - Get handler names. This is used by bitcode
  /// writer.
  const StringMap<unsigned> *getHandlerNames();

  /// ValueIsDeleted - This handler is used to update metadata store
  /// when a value is deleted.
  void ValueIsDeleted(const Value *V) {}
  void ValueIsDeleted(const Instruction *Inst) {
    removeMDs(Inst);
  }

  /// ValueIsCloned - This handler is used to update metadata store
  /// when In1 is cloned to create In2.
  void ValueIsCloned(const Instruction *In1, Instruction *In2);
};

} // end llvm namespace

#endif
