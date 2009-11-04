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

#ifndef LLVM_METADATA_H
#define LLVM_METADATA_H

#include "llvm/Value.h"
#include "llvm/Type.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/Support/ValueHandle.h"

namespace llvm {
class Constant;
class Instruction;
class LLVMContext;
class MetadataContextImpl;

//===----------------------------------------------------------------------===//
// MetadataBase  - A base class for MDNode, MDString and NamedMDNode.
class MetadataBase : public Value {
protected:
  MetadataBase(const Type *Ty, unsigned scid)
    : Value(Ty, scid) {}

public:

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

  StringRef Str;
protected:
  explicit MDString(LLVMContext &C, StringRef S)
    : MetadataBase(Type::getMetadataTy(C), Value::MDStringVal), Str(S) {}

public:
  static MDString *get(LLVMContext &Context, StringRef Str);
  
  StringRef getString() const { return Str; }

  unsigned getLength() const { return (unsigned)Str.size(); }

  typedef StringRef::iterator iterator;
  
  /// begin() - Pointer to the first byte of the string.
  ///
  iterator begin() const { return Str.begin(); }

  /// end() - Pointer to one byte past the end of the string.
  ///
  iterator end() const { return Str.end(); }

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

  friend class ElementVH;
  // Use CallbackVH to hold MDNOde elements.
  struct ElementVH : public CallbackVH {
    MDNode *Parent;
    ElementVH() {}
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

  ElementVH *Node;
  unsigned NodeSize;

protected:
  explicit MDNode(LLVMContext &C, Value *const *Vals, unsigned NumVals);
public:
  // Constructors and destructors.
  static MDNode *get(LLVMContext &Context, 
                     Value *const *Vals, unsigned NumVals);

  /// ~MDNode - Destroy MDNode.
  ~MDNode();
  
  /// getElement - Return specified element.
  Value *getElement(unsigned i) const {
    assert(i < getNumElements() && "Invalid element number!");
    return Node[i];
  }

  /// getNumElements - Return number of MDNode elements.
  unsigned getNumElements() const { return NodeSize; }

  /// Profile - calculate a unique identifier for this MDNode to collapse
  /// duplicates
  void Profile(FoldingSetNodeID &ID) const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const MDNode *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueID() == MDNodeVal;
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

  Module *Parent;
  SmallVector<TrackingVH<MetadataBase>, 4> Node;

  void setParent(Module *M) { Parent = M; }
protected:
  explicit NamedMDNode(LLVMContext &C, const Twine &N, MetadataBase*const *Vals, 
                       unsigned NumVals, Module *M = 0);
public:
  static NamedMDNode *Create(LLVMContext &C, const Twine &N, 
                             MetadataBase *const *MDs, 
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

  /// getElement - Return specified element.
  MetadataBase *getElement(unsigned i) const {
    assert(i < getNumElements() && "Invalid element number!");
    return Node[i];
  }

  /// getNumElements - Return number of NamedMDNode elements.
  unsigned getNumElements() const {
    return (unsigned)Node.size();
  }

  /// addElement - Add metadata element.
  void addElement(MetadataBase *M) {
    Node.push_back(TrackingVH<MetadataBase>(M));
  }

  typedef SmallVectorImpl<TrackingVH<MetadataBase> >::iterator elem_iterator;
  typedef SmallVectorImpl<TrackingVH<MetadataBase> >::const_iterator 
    const_elem_iterator;
  bool elem_empty() const                { return Node.empty(); }
  const_elem_iterator elem_begin() const { return Node.begin(); }
  const_elem_iterator elem_end() const   { return Node.end();   }
  elem_iterator elem_begin()             { return Node.begin(); }
  elem_iterator elem_end()               { return Node.end();   }

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
  // DO NOT IMPLEMENT
  MetadataContext(MetadataContext&);
  void operator=(MetadataContext&);

  MetadataContextImpl *const pImpl;
public:
  MetadataContext();
  ~MetadataContext();

  /// registerMDKind - Register a new metadata kind and return its ID.
  /// A metadata kind can be registered only once. 
  unsigned registerMDKind(StringRef Name);

  /// getMDKind - Return metadata kind. If the requested metadata kind
  /// is not registered then return 0.
  unsigned getMDKind(StringRef Name) const;

  /// isValidName - Return true if Name is a valid custom metadata handler name.
  static bool isValidName(StringRef Name);

  /// getMD - Get the metadata of given kind attached to an Instruction.
  /// If the metadata is not found then return 0.
  MDNode *getMD(unsigned Kind, const Instruction *Inst);

  /// getMDs - Get the metadata attached to an Instruction.
  void getMDs(const Instruction *Inst, 
        SmallVectorImpl<std::pair<unsigned, TrackingVH<MDNode> > > &MDs) const;

  /// addMD - Attach the metadata of given kind to an Instruction.
  void addMD(unsigned Kind, MDNode *Node, Instruction *Inst);
  
  /// removeMD - Remove metadata of given kind attached with an instuction.
  void removeMD(unsigned Kind, Instruction *Inst);
  
  /// removeAllMetadata - Remove all metadata attached with an instruction.
  void removeAllMetadata(Instruction *Inst);

  /// copyMD - If metadata is attached with Instruction In1 then attach
  /// the same metadata to In2.
  void copyMD(Instruction *In1, Instruction *In2);

  /// getHandlerNames - Populate client supplied smallvector using custome
  /// metadata name and ID.
  void getHandlerNames(SmallVectorImpl<std::pair<unsigned, StringRef> >&) const;

  /// ValueIsDeleted - This handler is used to update metadata store
  /// when a value is deleted.
  void ValueIsDeleted(const Value *) {}
  void ValueIsDeleted(Instruction *Inst);
  void ValueIsRAUWd(Value *V1, Value *V2);

  /// ValueIsCloned - This handler is used to update metadata store
  /// when In1 is cloned to create In2.
  void ValueIsCloned(const Instruction *In1, Instruction *In2);
};

} // end llvm namespace

#endif
