//===-- Metadata.cpp - Implement Metadata classes -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Metadata classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Metadata.h"
#include "LLVMContextImpl.h"
#include "SymbolTableListTraitsImpl.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LeakDetector.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"

using namespace llvm;

Metadata::Metadata(LLVMContext &Context, unsigned ID)
    : Value(Type::getMetadataTy(Context), ID) {}

//===----------------------------------------------------------------------===//
// MDString implementation.
//

void MDString::anchor() { }

MDString *MDString::get(LLVMContext &Context, StringRef Str) {
  auto &Store = Context.pImpl->MDStringCache;
  auto I = Store.find(Str);
  if (I != Store.end())
    return &I->second;

  auto *Entry =
      StringMapEntry<MDString>::Create(Str, Store.getAllocator(), Context);
  bool WasInserted = Store.insert(Entry);
  (void)WasInserted;
  assert(WasInserted && "Expected entry to be inserted");
  return &Entry->second;
}

StringRef MDString::getString() const {
  return StringMapEntry<MDString>::GetStringMapEntryFromValue(*this).first();
}

//===----------------------------------------------------------------------===//
// MDNodeOperand implementation.
//

// Use CallbackVH to hold MDNode operands.
namespace llvm {
class MDNodeOperand : public CallbackVH {
  MDNode *getParent() {
    MDNodeOperand *Cur = this;

    while (Cur->getValPtrInt() != 1)
      ++Cur;

    assert(Cur->getValPtrInt() == 1 &&
           "Couldn't find the end of the operand list!");
    return reinterpret_cast<MDNode *>(Cur + 1);
  }

public:
  MDNodeOperand() {}
  virtual ~MDNodeOperand();

  void set(Value *V) {
    unsigned IsLast = this->getValPtrInt();
    this->setValPtr(V);
    this->setAsLastOperand(IsLast);
  }

  /// \brief Accessor method to mark the operand as the first in the list.
  void setAsLastOperand(unsigned I) { this->setValPtrInt(I); }

  void deleted() override;
  void allUsesReplacedWith(Value *NV) override;
};
} // end namespace llvm.

// Provide out-of-line definition to prevent weak vtable.
MDNodeOperand::~MDNodeOperand() {}

void MDNodeOperand::deleted() {
  getParent()->replaceOperand(this, nullptr);
}

void MDNodeOperand::allUsesReplacedWith(Value *NV) {
  getParent()->replaceOperand(this, NV);
}

//===----------------------------------------------------------------------===//
// MDNode implementation.
//

/// \brief Get the MDNodeOperand's coallocated on the end of the MDNode.
static MDNodeOperand *getOperandPtr(MDNode *N, unsigned Op) {
  // Use <= instead of < to permit a one-past-the-end address.
  assert(Op <= N->getNumOperands() && "Invalid operand number");
  return reinterpret_cast<MDNodeOperand *>(N) - N->getNumOperands() + Op;
}

void MDNode::replaceOperandWith(unsigned i, Value *Val) {
  MDNodeOperand *Op = getOperandPtr(this, i);
  replaceOperand(Op, Val);
}

void *MDNode::operator new(size_t Size, unsigned NumOps) {
  void *Ptr = ::operator new(Size + NumOps * sizeof(MDNodeOperand));
  MDNodeOperand *Op = static_cast<MDNodeOperand *>(Ptr);
  if (NumOps) {
    MDNodeOperand *Last = Op + NumOps;
    for (; Op != Last; ++Op)
      new (Op) MDNodeOperand();
    (Op - 1)->setAsLastOperand(1);
  }
  return Op;
}

void MDNode::operator delete(void *Mem) {
  MDNode *N = static_cast<MDNode *>(Mem);
  MDNodeOperand *Op = static_cast<MDNodeOperand *>(Mem);
  for (unsigned I = 0, E = N->NumOperands; I != E; ++I)
    (--Op)->~MDNodeOperand();
  ::operator delete(Op);
}

MDNode::MDNode(LLVMContext &C, unsigned ID, ArrayRef<Value *> Vals,
               bool isFunctionLocal)
    : Metadata(C, ID) {
  NumOperands = Vals.size();

  if (isFunctionLocal)
    setValueSubclassData(getSubclassDataFromValue() | FunctionLocalBit);

  // Initialize the operand list.
  unsigned i = 0;
  for (MDNodeOperand *Op = getOperandPtr(this, 0), *E = Op + NumOperands;
       Op != E; ++Op, ++i)
    Op->set(Vals[i]);
}

GenericMDNode::~GenericMDNode() {
  LLVMContextImpl *pImpl = getType()->getContext().pImpl;
  if (isNotUniqued()) {
    pImpl->NonUniquedMDNodes.erase(this);
  } else {
    pImpl->MDNodeSet.erase(this);
  }
}

void GenericMDNode::dropAllReferences() {
  for (MDNodeOperand *Op = getOperandPtr(this, 0), *E = Op + NumOperands;
       Op != E; ++Op)
    Op->set(nullptr);
}

static const Function *getFunctionForValue(Value *V) {
  if (!V) return nullptr;
  if (Instruction *I = dyn_cast<Instruction>(V)) {
    BasicBlock *BB = I->getParent();
    return BB ? BB->getParent() : nullptr;
  }
  if (Argument *A = dyn_cast<Argument>(V))
    return A->getParent();
  if (BasicBlock *BB = dyn_cast<BasicBlock>(V))
    return BB->getParent();
  if (MDNode *MD = dyn_cast<MDNode>(V))
    return MD->getFunction();
  return nullptr;
}

#ifndef NDEBUG
static const Function *assertLocalFunction(const MDNode *N) {
  if (!N->isFunctionLocal()) return nullptr;

  // FIXME: This does not handle cyclic function local metadata.
  const Function *F = nullptr, *NewF = nullptr;
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    if (Value *V = N->getOperand(i)) {
      if (MDNode *MD = dyn_cast<MDNode>(V))
        NewF = assertLocalFunction(MD);
      else
        NewF = getFunctionForValue(V);
    }
    if (!F)
      F = NewF;
    else
      assert((NewF == nullptr || F == NewF) &&
             "inconsistent function-local metadata");
  }
  return F;
}
#endif

// getFunction - If this metadata is function-local and recursively has a
// function-local operand, return the first such operand's parent function.
// Otherwise, return null. getFunction() should not be used for performance-
// critical code because it recursively visits all the MDNode's operands.  
const Function *MDNode::getFunction() const {
#ifndef NDEBUG
  return assertLocalFunction(this);
#else
  if (!isFunctionLocal()) return nullptr;
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    if (const Function *F = getFunctionForValue(getOperand(i)))
      return F;
  return nullptr;
#endif
}

/// \brief Check if the Value  would require a function-local MDNode.
static bool isFunctionLocalValue(Value *V) {
  return isa<Instruction>(V) || isa<Argument>(V) || isa<BasicBlock>(V) ||
         (isa<MDNode>(V) && cast<MDNode>(V)->isFunctionLocal());
}

MDNode *MDNode::getMDNode(LLVMContext &Context, ArrayRef<Value*> Vals,
                          FunctionLocalness FL, bool Insert) {
  auto &Store = Context.pImpl->MDNodeSet;

  GenericMDNodeInfo::KeyTy Key(Vals);
  auto I = Store.find_as(Key);
  if (I != Store.end())
    return *I;
  if (!Insert)
    return nullptr;

  bool isFunctionLocal = false;
  switch (FL) {
  case FL_Unknown:
    for (Value *V : Vals) {
      if (!V) continue;
      if (isFunctionLocalValue(V)) {
        isFunctionLocal = true;
        break;
      }
    }
    break;
  case FL_No:
    isFunctionLocal = false;
    break;
  case FL_Yes:
    isFunctionLocal = true;
    break;
  }

  // Coallocate space for the node and Operands together, then placement new.
  GenericMDNode *N =
      new (Vals.size()) GenericMDNode(Context, Vals, isFunctionLocal);

  N->Hash = Key.Hash;
  Store.insert(N);
  return N;
}

MDNode *MDNode::get(LLVMContext &Context, ArrayRef<Value*> Vals) {
  return getMDNode(Context, Vals, FL_Unknown);
}

MDNode *MDNode::getWhenValsUnresolved(LLVMContext &Context,
                                      ArrayRef<Value*> Vals,
                                      bool isFunctionLocal) {
  return getMDNode(Context, Vals, isFunctionLocal ? FL_Yes : FL_No);
}

MDNode *MDNode::getIfExists(LLVMContext &Context, ArrayRef<Value*> Vals) {
  return getMDNode(Context, Vals, FL_Unknown, false);
}

MDNode *MDNode::getTemporary(LLVMContext &Context, ArrayRef<Value*> Vals) {
  MDNode *N = new (Vals.size()) MDNodeFwdDecl(Context, Vals, FL_No);
  N->setValueSubclassData(N->getSubclassDataFromValue() | NotUniquedBit);
  LeakDetector::addGarbageObject(N);
  return N;
}

void MDNode::deleteTemporary(MDNode *N) {
  assert(N->use_empty() && "Temporary MDNode has uses!");
  assert(isa<MDNodeFwdDecl>(N) && "Expected forward declaration");
  assert((N->getSubclassDataFromValue() & NotUniquedBit) &&
         "Temporary MDNode does not have NotUniquedBit set!");
  LeakDetector::removeGarbageObject(N);
  delete cast<MDNodeFwdDecl>(N);
}

/// \brief Return specified operand.
Value *MDNode::getOperand(unsigned i) const {
  assert(i < getNumOperands() && "Invalid operand number");
  return *getOperandPtr(const_cast<MDNode*>(this), i);
}

void MDNode::setIsNotUniqued() {
  setValueSubclassData(getSubclassDataFromValue() | NotUniquedBit);
  LLVMContextImpl *pImpl = getType()->getContext().pImpl;
  auto *G = cast<GenericMDNode>(this);
  G->Hash = 0;
  pImpl->NonUniquedMDNodes.insert(G);
}

// Replace value from this node's operand list.
void MDNode::replaceOperand(MDNodeOperand *Op, Value *To) {
  Value *From = *Op;

  // If is possible that someone did GV->RAUW(inst), replacing a global variable
  // with an instruction or some other function-local object.  If this is a
  // non-function-local MDNode, it can't point to a function-local object.
  // Handle this case by implicitly dropping the MDNode reference to null.
  // Likewise if the MDNode is function-local but for a different function.
  if (To && isFunctionLocalValue(To)) {
    if (!isFunctionLocal())
      To = nullptr;
    else {
      const Function *F = getFunction();
      const Function *FV = getFunctionForValue(To);
      // Metadata can be function-local without having an associated function.
      // So only consider functions to have changed if non-null.
      if (F && FV && F != FV)
        To = nullptr;
    }
  }
  
  if (From == To)
    return;

  // If this node is already not being uniqued (because one of the operands
  // already went to null), then there is nothing else to do here.
  if (isNotUniqued()) {
    Op->set(To);
    return;
  }

  auto &Store = getContext().pImpl->MDNodeSet;
  auto *N = cast<GenericMDNode>(this);

  // Remove "this" from the context map.
  Store.erase(N);

  // Update the operand.
  Op->set(To);

  // If we are dropping an argument to null, we choose to not unique the MDNode
  // anymore.  This commonly occurs during destruction, and uniquing these
  // brings little reuse.  Also, this means we don't need to include
  // isFunctionLocal bits in the hash for MDNodes.
  if (!To) {
    setIsNotUniqued();
    return;
  }

  // Now that the node is out of the table, get ready to reinsert it.  First,
  // check to see if another node with the same operands already exists in the
  // set.  If so, then this node is redundant.
  SmallVector<Value *, 8> Vals;
  GenericMDNodeInfo::KeyTy Key(N, Vals);
  auto I = Store.find_as(Key);
  if (I != Store.end()) {
    N->replaceAllUsesWith(*I);
    delete N;
    return;
  }

  N->Hash = Key.Hash;
  Store.insert(N);

  // If this MDValue was previously function-local but no longer is, clear
  // its function-local flag.
  if (isFunctionLocal() && !isFunctionLocalValue(To)) {
    bool isStillFunctionLocal = false;
    for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
      Value *V = getOperand(i);
      if (!V) continue;
      if (isFunctionLocalValue(V)) {
        isStillFunctionLocal = true;
        break;
      }
    }
    if (!isStillFunctionLocal)
      setValueSubclassData(getSubclassDataFromValue() & ~FunctionLocalBit);
  }
}

MDNode *MDNode::concatenate(MDNode *A, MDNode *B) {
  if (!A)
    return B;
  if (!B)
    return A;

  SmallVector<Value *, 4> Vals(A->getNumOperands() +
                               B->getNumOperands());

  unsigned j = 0;
  for (unsigned i = 0, ie = A->getNumOperands(); i != ie; ++i)
    Vals[j++] = A->getOperand(i);
  for (unsigned i = 0, ie = B->getNumOperands(); i != ie; ++i)
    Vals[j++] = B->getOperand(i);

  return MDNode::get(A->getContext(), Vals);
}

MDNode *MDNode::intersect(MDNode *A, MDNode *B) {
  if (!A || !B)
    return nullptr;

  SmallVector<Value *, 4> Vals;
  for (unsigned i = 0, ie = A->getNumOperands(); i != ie; ++i) {
    Value *V = A->getOperand(i);
    for (unsigned j = 0, je = B->getNumOperands(); j != je; ++j)
      if (V == B->getOperand(j)) {
        Vals.push_back(V);
        break;
      }
  }

  return MDNode::get(A->getContext(), Vals);
}

MDNode *MDNode::getMostGenericFPMath(MDNode *A, MDNode *B) {
  if (!A || !B)
    return nullptr;

  APFloat AVal = cast<ConstantFP>(A->getOperand(0))->getValueAPF();
  APFloat BVal = cast<ConstantFP>(B->getOperand(0))->getValueAPF();
  if (AVal.compare(BVal) == APFloat::cmpLessThan)
    return A;
  return B;
}

static bool isContiguous(const ConstantRange &A, const ConstantRange &B) {
  return A.getUpper() == B.getLower() || A.getLower() == B.getUpper();
}

static bool canBeMerged(const ConstantRange &A, const ConstantRange &B) {
  return !A.intersectWith(B).isEmptySet() || isContiguous(A, B);
}

static bool tryMergeRange(SmallVectorImpl<Value *> &EndPoints, ConstantInt *Low,
                          ConstantInt *High) {
  ConstantRange NewRange(Low->getValue(), High->getValue());
  unsigned Size = EndPoints.size();
  APInt LB = cast<ConstantInt>(EndPoints[Size - 2])->getValue();
  APInt LE = cast<ConstantInt>(EndPoints[Size - 1])->getValue();
  ConstantRange LastRange(LB, LE);
  if (canBeMerged(NewRange, LastRange)) {
    ConstantRange Union = LastRange.unionWith(NewRange);
    Type *Ty = High->getType();
    EndPoints[Size - 2] = ConstantInt::get(Ty, Union.getLower());
    EndPoints[Size - 1] = ConstantInt::get(Ty, Union.getUpper());
    return true;
  }
  return false;
}

static void addRange(SmallVectorImpl<Value *> &EndPoints, ConstantInt *Low,
                     ConstantInt *High) {
  if (!EndPoints.empty())
    if (tryMergeRange(EndPoints, Low, High))
      return;

  EndPoints.push_back(Low);
  EndPoints.push_back(High);
}

MDNode *MDNode::getMostGenericRange(MDNode *A, MDNode *B) {
  // Given two ranges, we want to compute the union of the ranges. This
  // is slightly complitade by having to combine the intervals and merge
  // the ones that overlap.

  if (!A || !B)
    return nullptr;

  if (A == B)
    return A;

  // First, walk both lists in older of the lower boundary of each interval.
  // At each step, try to merge the new interval to the last one we adedd.
  SmallVector<Value*, 4> EndPoints;
  int AI = 0;
  int BI = 0;
  int AN = A->getNumOperands() / 2;
  int BN = B->getNumOperands() / 2;
  while (AI < AN && BI < BN) {
    ConstantInt *ALow = cast<ConstantInt>(A->getOperand(2 * AI));
    ConstantInt *BLow = cast<ConstantInt>(B->getOperand(2 * BI));

    if (ALow->getValue().slt(BLow->getValue())) {
      addRange(EndPoints, ALow, cast<ConstantInt>(A->getOperand(2 * AI + 1)));
      ++AI;
    } else {
      addRange(EndPoints, BLow, cast<ConstantInt>(B->getOperand(2 * BI + 1)));
      ++BI;
    }
  }
  while (AI < AN) {
    addRange(EndPoints, cast<ConstantInt>(A->getOperand(2 * AI)),
             cast<ConstantInt>(A->getOperand(2 * AI + 1)));
    ++AI;
  }
  while (BI < BN) {
    addRange(EndPoints, cast<ConstantInt>(B->getOperand(2 * BI)),
             cast<ConstantInt>(B->getOperand(2 * BI + 1)));
    ++BI;
  }

  // If we have more than 2 ranges (4 endpoints) we have to try to merge
  // the last and first ones.
  unsigned Size = EndPoints.size();
  if (Size > 4) {
    ConstantInt *FB = cast<ConstantInt>(EndPoints[0]);
    ConstantInt *FE = cast<ConstantInt>(EndPoints[1]);
    if (tryMergeRange(EndPoints, FB, FE)) {
      for (unsigned i = 0; i < Size - 2; ++i) {
        EndPoints[i] = EndPoints[i + 2];
      }
      EndPoints.resize(Size - 2);
    }
  }

  // If in the end we have a single range, it is possible that it is now the
  // full range. Just drop the metadata in that case.
  if (EndPoints.size() == 2) {
    ConstantRange Range(cast<ConstantInt>(EndPoints[0])->getValue(),
                        cast<ConstantInt>(EndPoints[1])->getValue());
    if (Range.isFullSet())
      return nullptr;
  }

  return MDNode::get(A->getContext(), EndPoints);
}

//===----------------------------------------------------------------------===//
// NamedMDNode implementation.
//

static SmallVector<TrackingVH<MDNode>, 4> &getNMDOps(void *Operands) {
  return *(SmallVector<TrackingVH<MDNode>, 4> *)Operands;
}

NamedMDNode::NamedMDNode(const Twine &N)
    : Name(N.str()), Parent(nullptr),
      Operands(new SmallVector<TrackingVH<MDNode>, 4>()) {}

NamedMDNode::~NamedMDNode() {
  dropAllReferences();
  delete &getNMDOps(Operands);
}

unsigned NamedMDNode::getNumOperands() const {
  return (unsigned)getNMDOps(Operands).size();
}

MDNode *NamedMDNode::getOperand(unsigned i) const {
  assert(i < getNumOperands() && "Invalid Operand number!");
  return &*getNMDOps(Operands)[i];
}

void NamedMDNode::addOperand(MDNode *M) {
  assert(!M->isFunctionLocal() &&
         "NamedMDNode operands must not be function-local!");
  getNMDOps(Operands).push_back(TrackingVH<MDNode>(M));
}

void NamedMDNode::eraseFromParent() {
  getParent()->eraseNamedMetadata(this);
}

void NamedMDNode::dropAllReferences() {
  getNMDOps(Operands).clear();
}

StringRef NamedMDNode::getName() const {
  return StringRef(Name);
}

//===----------------------------------------------------------------------===//
// Instruction Metadata method implementations.
//

void Instruction::setMetadata(StringRef Kind, MDNode *Node) {
  if (!Node && !hasMetadata())
    return;
  setMetadata(getContext().getMDKindID(Kind), Node);
}

MDNode *Instruction::getMetadataImpl(StringRef Kind) const {
  return getMetadataImpl(getContext().getMDKindID(Kind));
}

void Instruction::dropUnknownMetadata(ArrayRef<unsigned> KnownIDs) {
  SmallSet<unsigned, 5> KnownSet;
  KnownSet.insert(KnownIDs.begin(), KnownIDs.end());

  // Drop debug if needed
  if (KnownSet.erase(LLVMContext::MD_dbg))
    DbgLoc = DebugLoc();

  if (!hasMetadataHashEntry())
    return; // Nothing to remove!

  DenseMap<const Instruction *, LLVMContextImpl::MDMapTy> &MetadataStore =
      getContext().pImpl->MetadataStore;

  if (KnownSet.empty()) {
    // Just drop our entry at the store.
    MetadataStore.erase(this);
    setHasMetadataHashEntry(false);
    return;
  }

  LLVMContextImpl::MDMapTy &Info = MetadataStore[this];
  unsigned I;
  unsigned E;
  // Walk the array and drop any metadata we don't know.
  for (I = 0, E = Info.size(); I != E;) {
    if (KnownSet.count(Info[I].first)) {
      ++I;
      continue;
    }

    Info[I] = Info.back();
    Info.pop_back();
    --E;
  }
  assert(E == Info.size());

  if (E == 0) {
    // Drop our entry at the store.
    MetadataStore.erase(this);
    setHasMetadataHashEntry(false);
  }
}

/// setMetadata - Set the metadata of of the specified kind to the specified
/// node.  This updates/replaces metadata if already present, or removes it if
/// Node is null.
void Instruction::setMetadata(unsigned KindID, MDNode *Node) {
  if (!Node && !hasMetadata())
    return;

  // Handle 'dbg' as a special case since it is not stored in the hash table.
  if (KindID == LLVMContext::MD_dbg) {
    DbgLoc = DebugLoc::getFromDILocation(Node);
    return;
  }
  
  // Handle the case when we're adding/updating metadata on an instruction.
  if (Node) {
    LLVMContextImpl::MDMapTy &Info = getContext().pImpl->MetadataStore[this];
    assert(!Info.empty() == hasMetadataHashEntry() &&
           "HasMetadata bit is wonked");
    if (Info.empty()) {
      setHasMetadataHashEntry(true);
    } else {
      // Handle replacement of an existing value.
      for (auto &P : Info)
        if (P.first == KindID) {
          P.second = Node;
          return;
        }
    }

    // No replacement, just add it to the list.
    Info.push_back(std::make_pair(KindID, Node));
    return;
  }

  // Otherwise, we're removing metadata from an instruction.
  assert((hasMetadataHashEntry() ==
          (getContext().pImpl->MetadataStore.count(this) > 0)) &&
         "HasMetadata bit out of date!");
  if (!hasMetadataHashEntry())
    return;  // Nothing to remove!
  LLVMContextImpl::MDMapTy &Info = getContext().pImpl->MetadataStore[this];

  // Common case is removing the only entry.
  if (Info.size() == 1 && Info[0].first == KindID) {
    getContext().pImpl->MetadataStore.erase(this);
    setHasMetadataHashEntry(false);
    return;
  }

  // Handle removal of an existing value.
  for (unsigned i = 0, e = Info.size(); i != e; ++i)
    if (Info[i].first == KindID) {
      Info[i] = Info.back();
      Info.pop_back();
      assert(!Info.empty() && "Removing last entry should be handled above");
      return;
    }
  // Otherwise, removing an entry that doesn't exist on the instruction.
}

void Instruction::setAAMetadata(const AAMDNodes &N) {
  setMetadata(LLVMContext::MD_tbaa, N.TBAA);
  setMetadata(LLVMContext::MD_alias_scope, N.Scope);
  setMetadata(LLVMContext::MD_noalias, N.NoAlias);
}

MDNode *Instruction::getMetadataImpl(unsigned KindID) const {
  // Handle 'dbg' as a special case since it is not stored in the hash table.
  if (KindID == LLVMContext::MD_dbg)
    return DbgLoc.getAsMDNode(getContext());
  
  if (!hasMetadataHashEntry()) return nullptr;
  
  LLVMContextImpl::MDMapTy &Info = getContext().pImpl->MetadataStore[this];
  assert(!Info.empty() && "bit out of sync with hash table");

  for (const auto &I : Info)
    if (I.first == KindID)
      return I.second;
  return nullptr;
}

void Instruction::getAllMetadataImpl(
    SmallVectorImpl<std::pair<unsigned, MDNode *>> &Result) const {
  Result.clear();
  
  // Handle 'dbg' as a special case since it is not stored in the hash table.
  if (!DbgLoc.isUnknown()) {
    Result.push_back(std::make_pair((unsigned)LLVMContext::MD_dbg,
                                    DbgLoc.getAsMDNode(getContext())));
    if (!hasMetadataHashEntry()) return;
  }
  
  assert(hasMetadataHashEntry() &&
         getContext().pImpl->MetadataStore.count(this) &&
         "Shouldn't have called this");
  const LLVMContextImpl::MDMapTy &Info =
    getContext().pImpl->MetadataStore.find(this)->second;
  assert(!Info.empty() && "Shouldn't have called this");

  Result.append(Info.begin(), Info.end());

  // Sort the resulting array so it is stable.
  if (Result.size() > 1)
    array_pod_sort(Result.begin(), Result.end());
}

void Instruction::getAllMetadataOtherThanDebugLocImpl(
    SmallVectorImpl<std::pair<unsigned, MDNode *>> &Result) const {
  Result.clear();
  assert(hasMetadataHashEntry() &&
         getContext().pImpl->MetadataStore.count(this) &&
         "Shouldn't have called this");
  const LLVMContextImpl::MDMapTy &Info =
    getContext().pImpl->MetadataStore.find(this)->second;
  assert(!Info.empty() && "Shouldn't have called this");
  Result.append(Info.begin(), Info.end());

  // Sort the resulting array so it is stable.
  if (Result.size() > 1)
    array_pod_sort(Result.begin(), Result.end());
}

/// clearMetadataHashEntries - Clear all hashtable-based metadata from
/// this instruction.
void Instruction::clearMetadataHashEntries() {
  assert(hasMetadataHashEntry() && "Caller should check");
  getContext().pImpl->MetadataStore.erase(this);
  setHasMetadataHashEntry(false);
}

