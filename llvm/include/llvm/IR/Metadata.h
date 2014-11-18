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

#ifndef LLVM_IR_METADATA_H
#define LLVM_IR_METADATA_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {
class LLVMContext;
class Module;
template<typename ValueSubClass, typename ItemParentClass>
  class SymbolTableListTraits;


enum LLVMConstants : uint32_t {
  DEBUG_METADATA_VERSION = 2  // Current debug info version number.
};

/// \brief Root of the metadata hierarchy.
///
/// This is a root class for typeless data in the IR.
///
/// TODO: Detach from the Value hierarchy.
class Metadata : public Value {
protected:
  Metadata(LLVMContext &Context, unsigned ID);

public:
  static bool classof(const Value *V) {
    return V->getValueID() == GenericMDNodeVal ||
           V->getValueID() == MDNodeFwdDeclVal ||
           V->getValueID() == MDStringVal;
  }
};

//===----------------------------------------------------------------------===//
/// \brief A single uniqued string.
///
/// These are used to efficiently contain a byte sequence for metadata.
/// MDString is always unnamed.
class MDString : public Metadata {
  friend class StringMapEntry<MDString>;

  virtual void anchor();
  MDString(const MDString &) LLVM_DELETED_FUNCTION;

  explicit MDString(LLVMContext &Context)
      : Metadata(Context, Value::MDStringVal) {}

  /// \brief Shadow Value::getName() to prevent its use.
  StringRef getName() const LLVM_DELETED_FUNCTION;

public:
  static MDString *get(LLVMContext &Context, StringRef Str);
  static MDString *get(LLVMContext &Context, const char *Str) {
    return get(Context, Str ? StringRef(Str) : StringRef());
  }

  StringRef getString() const;

  unsigned getLength() const { return (unsigned)getString().size(); }

  typedef StringRef::iterator iterator;

  /// \brief Pointer to the first byte of the string.
  iterator begin() const { return getString().begin(); }

  /// \brief Pointer to one byte past the end of the string.
  iterator end() const { return getString().end(); }

  /// \brief Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Value *V) {
    return V->getValueID() == MDStringVal;
  }
};

/// \brief A collection of metadata nodes that might be associated with a
/// memory access used by the alias-analysis infrastructure.
struct AAMDNodes {
  explicit AAMDNodes(MDNode *T = nullptr, MDNode *S = nullptr,
                     MDNode *N = nullptr)
      : TBAA(T), Scope(S), NoAlias(N) {}

  bool operator==(const AAMDNodes &A) const {
    return TBAA == A.TBAA && Scope == A.Scope && NoAlias == A.NoAlias;
  }

  bool operator!=(const AAMDNodes &A) const { return !(*this == A); }

  LLVM_EXPLICIT operator bool() const { return TBAA || Scope || NoAlias; }

  /// \brief The tag for type-based alias analysis.
  MDNode *TBAA;

  /// \brief The tag for alias scope specification (used with noalias).
  MDNode *Scope;

  /// \brief The tag specifying the noalias scope.
  MDNode *NoAlias;
};

// Specialize DenseMapInfo for AAMDNodes.
template<>
struct DenseMapInfo<AAMDNodes> {
  static inline AAMDNodes getEmptyKey() {
    return AAMDNodes(DenseMapInfo<MDNode *>::getEmptyKey(), 0, 0);
  }
  static inline AAMDNodes getTombstoneKey() {
    return AAMDNodes(DenseMapInfo<MDNode *>::getTombstoneKey(), 0, 0);
  }
  static unsigned getHashValue(const AAMDNodes &Val) {
    return DenseMapInfo<MDNode *>::getHashValue(Val.TBAA) ^
           DenseMapInfo<MDNode *>::getHashValue(Val.Scope) ^
           DenseMapInfo<MDNode *>::getHashValue(Val.NoAlias);
  }
  static bool isEqual(const AAMDNodes &LHS, const AAMDNodes &RHS) {
    return LHS == RHS;
  }
};

class MDNodeOperand;

//===----------------------------------------------------------------------===//
/// \brief Tuple of metadata.
class MDNode : public Metadata {
  MDNode(const MDNode &) LLVM_DELETED_FUNCTION;
  void operator=(const MDNode &) LLVM_DELETED_FUNCTION;
  friend class MDNodeOperand;
  friend class LLVMContextImpl;
  void *operator new(size_t) LLVM_DELETED_FUNCTION;

protected:
  void *operator new(size_t Size, unsigned NumOps);

  /// \brief Required by std, but never called.
  void operator delete(void *Mem);

  /// \brief Required by std, but never called.
  void operator delete(void *, unsigned) {
    llvm_unreachable("Constructor throws?");
  }

  /// \brief Required by std, but never called.
  void operator delete(void *, unsigned, bool) {
    llvm_unreachable("Constructor throws?");
  }

  /// \brief Subclass data enums.
  enum {
    /// FunctionLocalBit - This bit is set if this MDNode is function local.
    /// This is true when it (potentially transitively) contains a reference to
    /// something in a function, like an argument, basicblock, or instruction.
    FunctionLocalBit = 1 << 0,

    /// NotUniquedBit - This is set on MDNodes that are not uniqued because they
    /// have a null operand.
    NotUniquedBit    = 1 << 1
  };

  /// \brief FunctionLocal enums.
  enum FunctionLocalness {
    FL_Unknown = -1,
    FL_No = 0,
    FL_Yes = 1
  };

  /// \brief Replace each instance of the given operand with a new value.
  void replaceOperand(MDNodeOperand *Op, Value *NewVal);

  MDNode(LLVMContext &C, unsigned ID, ArrayRef<Value *> Vals,
         bool isFunctionLocal);
  ~MDNode() {}

  static MDNode *getMDNode(LLVMContext &C, ArrayRef<Value*> Vals,
                           FunctionLocalness FL, bool Insert = true);
public:
  static MDNode *get(LLVMContext &Context, ArrayRef<Value*> Vals);
  /// \brief Construct MDNode with an explicit function-localness.
  ///
  /// Don't analyze Vals; trust isFunctionLocal.
  static MDNode *getWhenValsUnresolved(LLVMContext &Context,
                                       ArrayRef<Value*> Vals,
                                       bool isFunctionLocal);

  static MDNode *getIfExists(LLVMContext &Context, ArrayRef<Value*> Vals);

  /// \brief Return a temporary MDNode
  ///
  /// For use in constructing cyclic MDNode structures. A temporary MDNode is
  /// not uniqued, may be RAUW'd, and must be manually deleted with
  /// deleteTemporary.
  static MDNode *getTemporary(LLVMContext &Context, ArrayRef<Value*> Vals);

  /// \brief Deallocate a node created by getTemporary.
  ///
  /// The node must not have any users.
  static void deleteTemporary(MDNode *N);

  /// \brief Replace a specific operand.
  void replaceOperandWith(unsigned i, Value *NewVal);

  /// \brief Return specified operand.
  Value *getOperand(unsigned i) const LLVM_READONLY;

  /// \brief Return number of MDNode operands.
  unsigned getNumOperands() const { return NumOperands; }

  /// \brief Return whether MDNode is local to a function.
  bool isFunctionLocal() const {
    return (getSubclassDataFromValue() & FunctionLocalBit) != 0;
  }

  /// \brief Return the first function-local operand's function.
  ///
  /// If this metadata is function-local and recursively has a function-local
  /// operand, return the first such operand's parent function.  Otherwise,
  /// return null. getFunction() should not be used for performance- critical
  /// code because it recursively visits all the MDNode's operands.
  const Function *getFunction() const;

  /// \brief Methods for support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const Value *V) {
    return V->getValueID() == GenericMDNodeVal ||
           V->getValueID() == MDNodeFwdDeclVal;
  }

  /// \brief Check whether MDNode is a vtable access.
  bool isTBAAVtableAccess() const;

  /// \brief Methods for metadata merging.
  static MDNode *concatenate(MDNode *A, MDNode *B);
  static MDNode *intersect(MDNode *A, MDNode *B);
  static MDNode *getMostGenericTBAA(MDNode *A, MDNode *B);
  static AAMDNodes getMostGenericAA(const AAMDNodes &A, const AAMDNodes &B);
  static MDNode *getMostGenericFPMath(MDNode *A, MDNode *B);
  static MDNode *getMostGenericRange(MDNode *A, MDNode *B);

protected:
  bool isNotUniqued() const {
    return (getSubclassDataFromValue() & NotUniquedBit) != 0;
  }
  void setIsNotUniqued();

  // Shadow Value::setValueSubclassData with a private forwarding method so that
  // any future subclasses cannot accidentally use it.
  void setValueSubclassData(unsigned short D) {
    Value::setValueSubclassData(D);
  }
};

/// \brief Generic metadata node.
///
/// Generic metadata nodes, with opt-out support for uniquing.
///
/// Although nodes are uniqued by default, \a GenericMDNode has no support for
/// RAUW.  If an operand change (due to RAUW or otherwise) causes a uniquing
/// collision, the uniquing bit is dropped.
///
/// TODO: Make uniquing opt-out (status: mandatory, sometimes dropped).
/// TODO: Drop support for RAUW.
class GenericMDNode : public MDNode {
  friend class MDNode;
  friend class LLVMContextImpl;

  unsigned Hash;

  GenericMDNode(LLVMContext &C, ArrayRef<Value *> Vals, bool isFunctionLocal)
      : MDNode(C, GenericMDNodeVal, Vals, isFunctionLocal), Hash(0) {}
  ~GenericMDNode();

  void dropAllReferences();

public:
  /// \brief Get the hash, if any.
  unsigned getHash() const { return Hash; }

  static bool classof(const Value *V) {
    return V->getValueID() == GenericMDNodeVal;
  }
};

/// \brief Forward declaration of metadata.
///
/// Forward declaration of metadata, in the form of a metadata node.  Unlike \a
/// GenericMDNode, this class has support for RAUW and is suitable for forward
/// references.
class MDNodeFwdDecl : public MDNode {
  friend class MDNode;

  MDNodeFwdDecl(LLVMContext &C, ArrayRef<Value *> Vals, bool isFunctionLocal)
      : MDNode(C, MDNodeFwdDeclVal, Vals, isFunctionLocal) {}
  ~MDNodeFwdDecl() {}

public:
  static bool classof(const Value *V) {
    return V->getValueID() == MDNodeFwdDeclVal;
  }
};

//===----------------------------------------------------------------------===//
/// \brief A tuple of MDNodes.
///
/// Despite its name, a NamedMDNode isn't itself an MDNode. NamedMDNodes belong
/// to modules, have names, and contain lists of MDNodes.
///
/// TODO: Inherit from Metadata.
class NamedMDNode : public ilist_node<NamedMDNode> {
  friend class SymbolTableListTraits<NamedMDNode, Module>;
  friend struct ilist_traits<NamedMDNode>;
  friend class LLVMContextImpl;
  friend class Module;
  NamedMDNode(const NamedMDNode &) LLVM_DELETED_FUNCTION;

  std::string Name;
  Module *Parent;
  void *Operands; // SmallVector<TrackingVH<MDNode>, 4>

  void setParent(Module *M) { Parent = M; }

  explicit NamedMDNode(const Twine &N);

  template<class T1, class T2>
  class op_iterator_impl :
      public std::iterator<std::bidirectional_iterator_tag, T2> {
    const NamedMDNode *Node;
    unsigned Idx;
    op_iterator_impl(const NamedMDNode *N, unsigned i) : Node(N), Idx(i) { }

    friend class NamedMDNode;

  public:
    op_iterator_impl() : Node(nullptr), Idx(0) { }

    bool operator==(const op_iterator_impl &o) const { return Idx == o.Idx; }
    bool operator!=(const op_iterator_impl &o) const { return Idx != o.Idx; }
    op_iterator_impl &operator++() {
      ++Idx;
      return *this;
    }
    op_iterator_impl operator++(int) {
      op_iterator_impl tmp(*this);
      operator++();
      return tmp;
    }
    op_iterator_impl &operator--() {
      --Idx;
      return *this;
    }
    op_iterator_impl operator--(int) {
      op_iterator_impl tmp(*this);
      operator--();
      return tmp;
    }

    T1 operator*() const { return Node->getOperand(Idx); }
  };

public:
  /// \brief Drop all references and remove the node from parent module.
  void eraseFromParent();

  /// \brief Remove all uses and clear node vector.
  void dropAllReferences();

  ~NamedMDNode();

  /// \brief Get the module that holds this named metadata collection.
  inline Module *getParent() { return Parent; }
  inline const Module *getParent() const { return Parent; }

  MDNode *getOperand(unsigned i) const;
  unsigned getNumOperands() const;
  void addOperand(MDNode *M);
  StringRef getName() const;
  void print(raw_ostream &ROS) const;
  void dump() const;

  // ---------------------------------------------------------------------------
  // Operand Iterator interface...
  //
  typedef op_iterator_impl<MDNode *, MDNode> op_iterator;
  op_iterator op_begin() { return op_iterator(this, 0); }
  op_iterator op_end()   { return op_iterator(this, getNumOperands()); }

  typedef op_iterator_impl<const MDNode *, MDNode> const_op_iterator;
  const_op_iterator op_begin() const { return const_op_iterator(this, 0); }
  const_op_iterator op_end()   const { return const_op_iterator(this, getNumOperands()); }

  inline iterator_range<op_iterator>  operands() {
    return iterator_range<op_iterator>(op_begin(), op_end());
  }
  inline iterator_range<const_op_iterator> operands() const {
    return iterator_range<const_op_iterator>(op_begin(), op_end());
  }
};

} // end llvm namespace

#endif
