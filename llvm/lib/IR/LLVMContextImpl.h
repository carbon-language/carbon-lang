//===-- LLVMContextImpl.h - The LLVMContextImpl opaque class ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file declares LLVMContextImpl, the opaque implementation 
//  of LLVMContext.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_IR_LLVMCONTEXTIMPL_H
#define LLVM_LIB_IR_LLVMCONTEXTIMPL_H

#include "AttributeImpl.h"
#include "ConstantsContext.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/ValueHandle.h"
#include <vector>

namespace llvm {

class ConstantInt;
class ConstantFP;
class DiagnosticInfoOptimizationRemark;
class DiagnosticInfoOptimizationRemarkMissed;
class DiagnosticInfoOptimizationRemarkAnalysis;
class GCStrategy;
class LLVMContext;
class Type;
class Value;

struct DenseMapAPIntKeyInfo {
  static inline APInt getEmptyKey() {
    APInt V(nullptr, 0);
    V.VAL = 0;
    return V;
  }
  static inline APInt getTombstoneKey() {
    APInt V(nullptr, 0);
    V.VAL = 1;
    return V;
  }
  static unsigned getHashValue(const APInt &Key) {
    return static_cast<unsigned>(hash_value(Key));
  }
  static bool isEqual(const APInt &LHS, const APInt &RHS) {
    return LHS.getBitWidth() == RHS.getBitWidth() && LHS == RHS;
  }
};

struct DenseMapAPFloatKeyInfo {
  static inline APFloat getEmptyKey() { return APFloat(APFloat::Bogus, 1); }
  static inline APFloat getTombstoneKey() { return APFloat(APFloat::Bogus, 2); }
  static unsigned getHashValue(const APFloat &Key) {
    return static_cast<unsigned>(hash_value(Key));
  }
  static bool isEqual(const APFloat &LHS, const APFloat &RHS) {
    return LHS.bitwiseIsEqual(RHS);
  }
};

struct AnonStructTypeKeyInfo {
  struct KeyTy {
    ArrayRef<Type*> ETypes;
    bool isPacked;
    KeyTy(const ArrayRef<Type*>& E, bool P) :
      ETypes(E), isPacked(P) {}
    KeyTy(const StructType *ST)
        : ETypes(ST->elements()), isPacked(ST->isPacked()) {}
    bool operator==(const KeyTy& that) const {
      if (isPacked != that.isPacked)
        return false;
      if (ETypes != that.ETypes)
        return false;
      return true;
    }
    bool operator!=(const KeyTy& that) const {
      return !this->operator==(that);
    }
  };
  static inline StructType* getEmptyKey() {
    return DenseMapInfo<StructType*>::getEmptyKey();
  }
  static inline StructType* getTombstoneKey() {
    return DenseMapInfo<StructType*>::getTombstoneKey();
  }
  static unsigned getHashValue(const KeyTy& Key) {
    return hash_combine(hash_combine_range(Key.ETypes.begin(),
                                           Key.ETypes.end()),
                        Key.isPacked);
  }
  static unsigned getHashValue(const StructType *ST) {
    return getHashValue(KeyTy(ST));
  }
  static bool isEqual(const KeyTy& LHS, const StructType *RHS) {
    if (RHS == getEmptyKey() || RHS == getTombstoneKey())
      return false;
    return LHS == KeyTy(RHS);
  }
  static bool isEqual(const StructType *LHS, const StructType *RHS) {
    return LHS == RHS;
  }
};

struct FunctionTypeKeyInfo {
  struct KeyTy {
    const Type *ReturnType;
    ArrayRef<Type*> Params;
    bool isVarArg;
    KeyTy(const Type* R, const ArrayRef<Type*>& P, bool V) :
      ReturnType(R), Params(P), isVarArg(V) {}
    KeyTy(const FunctionType *FT)
        : ReturnType(FT->getReturnType()), Params(FT->params()),
          isVarArg(FT->isVarArg()) {}
    bool operator==(const KeyTy& that) const {
      if (ReturnType != that.ReturnType)
        return false;
      if (isVarArg != that.isVarArg)
        return false;
      if (Params != that.Params)
        return false;
      return true;
    }
    bool operator!=(const KeyTy& that) const {
      return !this->operator==(that);
    }
  };
  static inline FunctionType* getEmptyKey() {
    return DenseMapInfo<FunctionType*>::getEmptyKey();
  }
  static inline FunctionType* getTombstoneKey() {
    return DenseMapInfo<FunctionType*>::getTombstoneKey();
  }
  static unsigned getHashValue(const KeyTy& Key) {
    return hash_combine(Key.ReturnType,
                        hash_combine_range(Key.Params.begin(),
                                           Key.Params.end()),
                        Key.isVarArg);
  }
  static unsigned getHashValue(const FunctionType *FT) {
    return getHashValue(KeyTy(FT));
  }
  static bool isEqual(const KeyTy& LHS, const FunctionType *RHS) {
    if (RHS == getEmptyKey() || RHS == getTombstoneKey())
      return false;
    return LHS == KeyTy(RHS);
  }
  static bool isEqual(const FunctionType *LHS, const FunctionType *RHS) {
    return LHS == RHS;
  }
};

/// \brief Structure for hashing arbitrary MDNode operands.
class MDNodeOpsKey {
  ArrayRef<Metadata *> RawOps;
  ArrayRef<MDOperand> Ops;

  unsigned Hash;

protected:
  MDNodeOpsKey(ArrayRef<Metadata *> Ops)
      : RawOps(Ops), Hash(calculateHash(Ops)) {}

  template <class NodeTy>
  MDNodeOpsKey(NodeTy *N, unsigned Offset = 0)
      : Ops(N->op_begin() + Offset, N->op_end()), Hash(N->getHash()) {}

  template <class NodeTy>
  bool compareOps(const NodeTy *RHS, unsigned Offset = 0) const {
    if (getHash() != RHS->getHash())
      return false;

    assert((RawOps.empty() || Ops.empty()) && "Two sets of operands?");
    return RawOps.empty() ? compareOps(Ops, RHS, Offset)
                          : compareOps(RawOps, RHS, Offset);
  }

  static unsigned calculateHash(MDNode *N, unsigned Offset = 0);

private:
  template <class T>
  static bool compareOps(ArrayRef<T> Ops, const MDNode *RHS, unsigned Offset) {
    if (Ops.size() != RHS->getNumOperands() - Offset)
      return false;
    return std::equal(Ops.begin(), Ops.end(), RHS->op_begin() + Offset);
  }

  static unsigned calculateHash(ArrayRef<Metadata *> Ops);

public:
  unsigned getHash() const { return Hash; }
};

/// \brief DenseMapInfo for MDTuple.
///
/// Note that we don't need the is-function-local bit, since that's implicit in
/// the operands.
struct MDTupleInfo {
  struct KeyTy : MDNodeOpsKey {
    KeyTy(ArrayRef<Metadata *> Ops) : MDNodeOpsKey(Ops) {}
    KeyTy(MDTuple *N) : MDNodeOpsKey(N) {}

    bool operator==(const MDTuple *RHS) const {
      if (RHS == getEmptyKey() || RHS == getTombstoneKey())
        return false;
      return compareOps(RHS);
    }

    static unsigned calculateHash(MDTuple *N) {
      return MDNodeOpsKey::calculateHash(N);
    }
  };
  static inline MDTuple *getEmptyKey() {
    return DenseMapInfo<MDTuple *>::getEmptyKey();
  }
  static inline MDTuple *getTombstoneKey() {
    return DenseMapInfo<MDTuple *>::getTombstoneKey();
  }
  static unsigned getHashValue(const KeyTy &Key) { return Key.getHash(); }
  static unsigned getHashValue(const MDTuple *U) { return U->getHash(); }
  static bool isEqual(const KeyTy &LHS, const MDTuple *RHS) {
    return LHS == RHS;
  }
  static bool isEqual(const MDTuple *LHS, const MDTuple *RHS) {
    return LHS == RHS;
  }
};

/// \brief DenseMapInfo for MDLocation.
struct MDLocationInfo {
  struct KeyTy {
    unsigned Line;
    unsigned Column;
    Metadata *Scope;
    Metadata *InlinedAt;

    KeyTy(unsigned Line, unsigned Column, Metadata *Scope, Metadata *InlinedAt)
        : Line(Line), Column(Column), Scope(Scope), InlinedAt(InlinedAt) {}

    KeyTy(const MDLocation *L)
        : Line(L->getLine()), Column(L->getColumn()), Scope(L->getScope()),
          InlinedAt(L->getInlinedAt()) {}

    bool operator==(const MDLocation *RHS) const {
      if (RHS == getEmptyKey() || RHS == getTombstoneKey())
        return false;
      return Line == RHS->getLine() && Column == RHS->getColumn() &&
             Scope == RHS->getScope() && InlinedAt == RHS->getInlinedAt();
    }
  };
  static inline MDLocation *getEmptyKey() {
    return DenseMapInfo<MDLocation *>::getEmptyKey();
  }
  static inline MDLocation *getTombstoneKey() {
    return DenseMapInfo<MDLocation *>::getTombstoneKey();
  }
  static unsigned getHashValue(const KeyTy &Key) {
    return hash_combine(Key.Line, Key.Column, Key.Scope, Key.InlinedAt);
  }
  static unsigned getHashValue(const MDLocation *U) {
    return getHashValue(KeyTy(U));
  }
  static bool isEqual(const KeyTy &LHS, const MDLocation *RHS) {
    return LHS == RHS;
  }
  static bool isEqual(const MDLocation *LHS, const MDLocation *RHS) {
    return LHS == RHS;
  }
};

/// \brief DenseMapInfo for GenericDebugNode.
struct GenericDebugNodeInfo {
  struct KeyTy : MDNodeOpsKey {
    unsigned Tag;
    StringRef Header;
    KeyTy(unsigned Tag, StringRef Header, ArrayRef<Metadata *> DwarfOps)
        : MDNodeOpsKey(DwarfOps), Tag(Tag), Header(Header) {}
    KeyTy(GenericDebugNode *N)
        : MDNodeOpsKey(N, 1), Tag(N->getTag()), Header(N->getHeader()) {}

    bool operator==(const GenericDebugNode *RHS) const {
      if (RHS == getEmptyKey() || RHS == getTombstoneKey())
        return false;
      return Tag == RHS->getTag() && Header == RHS->getHeader() &&
             compareOps(RHS, 1);
    }

    static unsigned calculateHash(GenericDebugNode *N) {
      return MDNodeOpsKey::calculateHash(N, 1);
    }
  };
  static inline GenericDebugNode *getEmptyKey() {
    return DenseMapInfo<GenericDebugNode *>::getEmptyKey();
  }
  static inline GenericDebugNode *getTombstoneKey() {
    return DenseMapInfo<GenericDebugNode *>::getTombstoneKey();
  }
  static unsigned getHashValue(const KeyTy &Key) {
    return hash_combine(Key.getHash(), Key.Tag, Key.Header);
  }
  static unsigned getHashValue(const GenericDebugNode *U) {
    return hash_combine(U->getHash(), U->getTag(), U->getHeader());
  }
  static bool isEqual(const KeyTy &LHS, const GenericDebugNode *RHS) {
    return LHS == RHS;
  }
  static bool isEqual(const GenericDebugNode *LHS,
                      const GenericDebugNode *RHS) {
    return LHS == RHS;
  }
};

class LLVMContextImpl {
public:
  /// OwnedModules - The set of modules instantiated in this context, and which
  /// will be automatically deleted if this context is deleted.
  SmallPtrSet<Module*, 4> OwnedModules;
  
  LLVMContext::InlineAsmDiagHandlerTy InlineAsmDiagHandler;
  void *InlineAsmDiagContext;

  LLVMContext::DiagnosticHandlerTy DiagnosticHandler;
  void *DiagnosticContext;
  bool RespectDiagnosticFilters;

  LLVMContext::YieldCallbackTy YieldCallback;
  void *YieldOpaqueHandle;

  typedef DenseMap<APInt, ConstantInt *, DenseMapAPIntKeyInfo> IntMapTy;
  IntMapTy IntConstants;

  typedef DenseMap<APFloat, ConstantFP *, DenseMapAPFloatKeyInfo> FPMapTy;
  FPMapTy FPConstants;

  FoldingSet<AttributeImpl> AttrsSet;
  FoldingSet<AttributeSetImpl> AttrsLists;
  FoldingSet<AttributeSetNode> AttrsSetNodes;

  StringMap<MDString> MDStringCache;
  DenseMap<Value *, ValueAsMetadata *> ValuesAsMetadata;
  DenseMap<Metadata *, MetadataAsValue *> MetadataAsValues;

  DenseSet<MDTuple *, MDTupleInfo> MDTuples;
  DenseSet<MDLocation *, MDLocationInfo> MDLocations;
  DenseSet<GenericDebugNode *, GenericDebugNodeInfo> GenericDebugNodes;

  // MDNodes may be uniqued or not uniqued.  When they're not uniqued, they
  // aren't in the MDNodeSet, but they're still shared between objects, so no
  // one object can destroy them.  This set allows us to at least destroy them
  // on Context destruction.
  SmallPtrSet<MDNode *, 1> DistinctMDNodes;

  DenseMap<Type*, ConstantAggregateZero*> CAZConstants;

  typedef ConstantUniqueMap<ConstantArray> ArrayConstantsTy;
  ArrayConstantsTy ArrayConstants;
  
  typedef ConstantUniqueMap<ConstantStruct> StructConstantsTy;
  StructConstantsTy StructConstants;
  
  typedef ConstantUniqueMap<ConstantVector> VectorConstantsTy;
  VectorConstantsTy VectorConstants;
  
  DenseMap<PointerType*, ConstantPointerNull*> CPNConstants;

  DenseMap<Type*, UndefValue*> UVConstants;
  
  StringMap<ConstantDataSequential*> CDSConstants;

  DenseMap<std::pair<const Function *, const BasicBlock *>, BlockAddress *>
    BlockAddresses;
  ConstantUniqueMap<ConstantExpr> ExprConstants;

  ConstantUniqueMap<InlineAsm> InlineAsms;

  ConstantInt *TheTrueVal;
  ConstantInt *TheFalseVal;

  // Basic type instances.
  Type VoidTy, LabelTy, HalfTy, FloatTy, DoubleTy, MetadataTy;
  Type X86_FP80Ty, FP128Ty, PPC_FP128Ty, X86_MMXTy;
  IntegerType Int1Ty, Int8Ty, Int16Ty, Int32Ty, Int64Ty;

  
  /// TypeAllocator - All dynamically allocated types are allocated from this.
  /// They live forever until the context is torn down.
  BumpPtrAllocator TypeAllocator;
  
  DenseMap<unsigned, IntegerType*> IntegerTypes;

  typedef DenseSet<FunctionType *, FunctionTypeKeyInfo> FunctionTypeSet;
  FunctionTypeSet FunctionTypes;
  typedef DenseSet<StructType *, AnonStructTypeKeyInfo> StructTypeSet;
  StructTypeSet AnonStructTypes;
  StringMap<StructType*> NamedStructTypes;
  unsigned NamedStructTypesUniqueID;
    
  DenseMap<std::pair<Type *, uint64_t>, ArrayType*> ArrayTypes;
  DenseMap<std::pair<Type *, unsigned>, VectorType*> VectorTypes;
  DenseMap<Type*, PointerType*> PointerTypes;  // Pointers in AddrSpace = 0
  DenseMap<std::pair<Type*, unsigned>, PointerType*> ASPointerTypes;


  /// ValueHandles - This map keeps track of all of the value handles that are
  /// watching a Value*.  The Value::HasValueHandle bit is used to know
  /// whether or not a value has an entry in this map.
  typedef DenseMap<Value*, ValueHandleBase*> ValueHandlesTy;
  ValueHandlesTy ValueHandles;
  
  /// CustomMDKindNames - Map to hold the metadata string to ID mapping.
  StringMap<unsigned> CustomMDKindNames;

  typedef std::pair<unsigned, TrackingMDNodeRef> MDPairTy;
  typedef SmallVector<MDPairTy, 2> MDMapTy;

  /// MetadataStore - Collection of per-instruction metadata used in this
  /// context.
  DenseMap<const Instruction *, MDMapTy> MetadataStore;
  
  /// DiscriminatorTable - This table maps file:line locations to an
  /// integer representing the next DWARF path discriminator to assign to
  /// instructions in different blocks at the same location.
  DenseMap<std::pair<const char *, unsigned>, unsigned> DiscriminatorTable;

  /// IntrinsicIDCache - Cache of intrinsic name (string) to numeric ID mappings
  /// requested in this context
  typedef DenseMap<const Function*, unsigned> IntrinsicIDCacheTy;
  IntrinsicIDCacheTy IntrinsicIDCache;

  /// \brief Mapping from a function to its prefix data, which is stored as the
  /// operand of an unparented ReturnInst so that the prefix data has a Use.
  typedef DenseMap<const Function *, ReturnInst *> PrefixDataMapTy;
  PrefixDataMapTy PrefixDataMap;

  /// \brief Mapping from a function to its prologue data, which is stored as
  /// the operand of an unparented ReturnInst so that the prologue data has a
  /// Use.
  typedef DenseMap<const Function *, ReturnInst *> PrologueDataMapTy;
  PrologueDataMapTy PrologueDataMap;

  int getOrAddScopeRecordIdxEntry(MDNode *N, int ExistingIdx);
  int getOrAddScopeInlinedAtIdxEntry(MDNode *Scope, MDNode *IA,int ExistingIdx);

  LLVMContextImpl(LLVMContext &C);
  ~LLVMContextImpl();

  /// Destroy the ConstantArrays if they are not used.
  void dropTriviallyDeadConstantArrays();
};

}

#endif
