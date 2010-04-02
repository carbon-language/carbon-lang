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

#include "llvm/Metadata.h"
#include "LLVMContextImpl.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Instruction.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/SmallString.h"
#include "SymbolTableListTraitsImpl.h"
#include "llvm/Support/ValueHandle.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// MDString implementation.
//

MDString::MDString(LLVMContext &C, StringRef S)
  : Value(Type::getMetadataTy(C), Value::MDStringVal), Str(S) {}

MDString *MDString::get(LLVMContext &Context, StringRef Str) {
  LLVMContextImpl *pImpl = Context.pImpl;
  StringMapEntry<MDString *> &Entry =
    pImpl->MDStringCache.GetOrCreateValue(Str);
  MDString *&S = Entry.getValue();
  if (!S) S = new MDString(Context, Entry.getKey());
  return S;
}

//===----------------------------------------------------------------------===//
// MDNodeOperand implementation.
//

// Use CallbackVH to hold MDNode operands.
namespace llvm {
class MDNodeOperand : public CallbackVH {
  MDNode *Parent;
public:
  MDNodeOperand(Value *V, MDNode *P) : CallbackVH(V), Parent(P) {}
  ~MDNodeOperand() {}

  void set(Value *V) {
    setValPtr(V);
  }

  virtual void deleted();
  virtual void allUsesReplacedWith(Value *NV);
};
} // end namespace llvm.


void MDNodeOperand::deleted() {
  Parent->replaceOperand(this, 0);
}

void MDNodeOperand::allUsesReplacedWith(Value *NV) {
  Parent->replaceOperand(this, NV);
}



//===----------------------------------------------------------------------===//
// MDNode implementation.
//

/// getOperandPtr - Helper function to get the MDNodeOperand's coallocated on
/// the end of the MDNode.
static MDNodeOperand *getOperandPtr(MDNode *N, unsigned Op) {
  assert(Op < N->getNumOperands() && "Invalid operand number");
  return reinterpret_cast<MDNodeOperand*>(N+1)+Op;
}

MDNode::MDNode(LLVMContext &C, Value *const *Vals, unsigned NumVals,
               bool isFunctionLocal)
: Value(Type::getMetadataTy(C), Value::MDNodeVal) {
  NumOperands = NumVals;

  if (isFunctionLocal)
    setValueSubclassData(getSubclassDataFromValue() | FunctionLocalBit);

  // Initialize the operand list, which is co-allocated on the end of the node.
  for (MDNodeOperand *Op = getOperandPtr(this, 0), *E = Op+NumOperands;
       Op != E; ++Op, ++Vals)
    new (Op) MDNodeOperand(*Vals, this);
}


/// ~MDNode - Destroy MDNode.
MDNode::~MDNode() {
  assert((getSubclassDataFromValue() & DestroyFlag) != 0 &&
         "Not being destroyed through destroy()?");
  LLVMContextImpl *pImpl = getType()->getContext().pImpl;
  if (isNotUniqued()) {
    pImpl->NonUniquedMDNodes.erase(this);
  } else {
    pImpl->MDNodeSet.RemoveNode(this);
  }

  // Destroy the operands.
  for (MDNodeOperand *Op = getOperandPtr(this, 0), *E = Op+NumOperands;
       Op != E; ++Op)
    Op->~MDNodeOperand();
}

static const Function *getFunctionForValue(Value *V) {
  assert(!isa<MDNode>(V) && "does not iterate over metadata operands");
  if (!V) return NULL;
  if (Instruction *I = dyn_cast<Instruction>(V))
    return I->getParent()->getParent();
  if (BasicBlock *BB = dyn_cast<BasicBlock>(V))
    return BB->getParent();
  if (Argument *A = dyn_cast<Argument>(V))
    return A->getParent();
  return NULL;
}

#ifndef NDEBUG
static const Function *assertLocalFunction(const MDNode *N) {
  if (!N->isFunctionLocal()) return 0;

  const Function *F = 0, *NewF = 0;
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    if (Value *V = N->getOperand(i)) {
      if (MDNode *MD = dyn_cast<MDNode>(V))
        NewF = assertLocalFunction(MD);
      else
        NewF = getFunctionForValue(V);
    }
    if (F == 0)
      F = NewF;
    else 
      assert((NewF == 0 || F == NewF) &&"inconsistent function-local metadata");
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
#endif
  if (!isFunctionLocal()) return NULL;

  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    if (Value *V = getOperand(i)) {
      if (MDNode *MD = dyn_cast<MDNode>(V)) {
        if (const Function *F = MD->getFunction())
          return F;
      } else {
        return getFunctionForValue(V);
      }
    }
  }
  return NULL;
}

// destroy - Delete this node.  Only when there are no uses.
void MDNode::destroy() {
  setValueSubclassData(getSubclassDataFromValue() | DestroyFlag);
  // Placement delete, the free the memory.
  this->~MDNode();
  free(this);
}

MDNode *MDNode::getMDNode(LLVMContext &Context, Value *const *Vals,
                          unsigned NumVals, FunctionLocalness FL,
                          bool Insert) {
  LLVMContextImpl *pImpl = Context.pImpl;
  bool isFunctionLocal = false;
  switch (FL) {
  case FL_Unknown:
    for (unsigned i = 0; i != NumVals; ++i) {
      Value *V = Vals[i];
      if (!V) continue;
      if (isa<Instruction>(V) || isa<Argument>(V) || isa<BasicBlock>(V) ||
          (isa<MDNode>(V) && cast<MDNode>(V)->isFunctionLocal())) {
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

  FoldingSetNodeID ID;
  for (unsigned i = 0; i != NumVals; ++i)
    ID.AddPointer(Vals[i]);
  ID.AddBoolean(isFunctionLocal);

  void *InsertPoint;
  MDNode *N = NULL;
  
  if ((N = pImpl->MDNodeSet.FindNodeOrInsertPos(ID, InsertPoint)))
    return N;
    
  if (!Insert)
    return NULL;
    
  // Coallocate space for the node and Operands together, then placement new.
  void *Ptr = malloc(sizeof(MDNode)+NumVals*sizeof(MDNodeOperand));
  N = new (Ptr) MDNode(Context, Vals, NumVals, isFunctionLocal);

  // InsertPoint will have been set by the FindNodeOrInsertPos call.
  pImpl->MDNodeSet.InsertNode(N, InsertPoint);

  return N;
}

MDNode *MDNode::get(LLVMContext &Context, Value*const* Vals, unsigned NumVals) {
  return getMDNode(Context, Vals, NumVals, FL_Unknown);
}

MDNode *MDNode::getWhenValsUnresolved(LLVMContext &Context, Value *const *Vals,
                                      unsigned NumVals, bool isFunctionLocal) {
  return getMDNode(Context, Vals, NumVals, isFunctionLocal ? FL_Yes : FL_No);
}

MDNode *MDNode::getIfExists(LLVMContext &Context, Value *const *Vals,
                            unsigned NumVals) {
  return getMDNode(Context, Vals, NumVals, FL_Unknown, false);
}

/// getOperand - Return specified operand.
Value *MDNode::getOperand(unsigned i) const {
  return *getOperandPtr(const_cast<MDNode*>(this), i);
}

void MDNode::Profile(FoldingSetNodeID &ID) const {
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    ID.AddPointer(getOperand(i));
  ID.AddBoolean(isFunctionLocal());
}

void MDNode::setIsNotUniqued() {
  setValueSubclassData(getSubclassDataFromValue() | NotUniquedBit);
  LLVMContextImpl *pImpl = getType()->getContext().pImpl;
  pImpl->NonUniquedMDNodes.insert(this);
}

// Replace value from this node's operand list.
void MDNode::replaceOperand(MDNodeOperand *Op, Value *To) {
  Value *From = *Op;

  if (From == To)
    return;

  // Update the operand.
  Op->set(To);

  // If this node is already not being uniqued (because one of the operands
  // already went to null), then there is nothing else to do here.
  if (isNotUniqued()) return;

  LLVMContextImpl *pImpl = getType()->getContext().pImpl;

  // Remove "this" from the context map.  FoldingSet doesn't have to reprofile
  // this node to remove it, so we don't care what state the operands are in.
  pImpl->MDNodeSet.RemoveNode(this);

  // If we are dropping an argument to null, we choose to not unique the MDNode
  // anymore.  This commonly occurs during destruction, and uniquing these
  // brings little reuse.
  if (To == 0) {
    setIsNotUniqued();
    return;
  }

  // Now that the node is out of the folding set, get ready to reinsert it.
  // First, check to see if another node with the same operands already exists
  // in the set.  If it doesn't exist, this returns the position to insert it.
  FoldingSetNodeID ID;
  Profile(ID);
  void *InsertPoint;
  MDNode *N = pImpl->MDNodeSet.FindNodeOrInsertPos(ID, InsertPoint);

  if (N) {
    N->replaceAllUsesWith(this);
    N->destroy();
    N = pImpl->MDNodeSet.FindNodeOrInsertPos(ID, InsertPoint);
    assert(N == 0 && "shouldn't be in the map now!"); (void)N;
  }

  // InsertPoint will have been set by the FindNodeOrInsertPos call.
  pImpl->MDNodeSet.InsertNode(this, InsertPoint);
}

//===----------------------------------------------------------------------===//
// NamedMDNode implementation.
//

namespace llvm {
// SymbolTableListTraits specialization for MDSymbolTable.
void ilist_traits<NamedMDNode>
::addNodeToList(NamedMDNode *N) {
  assert(N->getParent() == 0 && "Value already in a container!!");
  Module *Owner = getListOwner();
  N->setParent(Owner);
  MDSymbolTable &ST = Owner->getMDSymbolTable();
  ST.insert(N->getName(), N);
}

void ilist_traits<NamedMDNode>::removeNodeFromList(NamedMDNode *N) {
  N->setParent(0);
  Module *Owner = getListOwner();
  MDSymbolTable &ST = Owner->getMDSymbolTable();
  ST.remove(N->getName());
}
}

static SmallVector<WeakVH, 4> &getNMDOps(void *Operands) {
  return *(SmallVector<WeakVH, 4>*)Operands;
}

NamedMDNode::NamedMDNode(LLVMContext &C, const Twine &N,
                         MDNode *const *MDs,
                         unsigned NumMDs, Module *ParentModule)
  : Value(Type::getMetadataTy(C), Value::NamedMDNodeVal), Parent(0) {
  setName(N);
  Operands = new SmallVector<WeakVH, 4>();

  SmallVector<WeakVH, 4> &Node = getNMDOps(Operands);
  for (unsigned i = 0; i != NumMDs; ++i)
    Node.push_back(WeakVH(MDs[i]));

  if (ParentModule)
    ParentModule->getNamedMDList().push_back(this);
}

NamedMDNode *NamedMDNode::Create(const NamedMDNode *NMD, Module *M) {
  assert(NMD && "Invalid source NamedMDNode!");
  SmallVector<MDNode *, 4> Elems;
  Elems.reserve(NMD->getNumOperands());

  for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i)
    Elems.push_back(NMD->getOperand(i));
  return new NamedMDNode(NMD->getContext(), NMD->getName().data(),
                         Elems.data(), Elems.size(), M);
}

NamedMDNode::~NamedMDNode() {
  dropAllReferences();
  delete &getNMDOps(Operands);
}

/// getNumOperands - Return number of NamedMDNode operands.
unsigned NamedMDNode::getNumOperands() const {
  return (unsigned)getNMDOps(Operands).size();
}

/// getOperand - Return specified operand.
MDNode *NamedMDNode::getOperand(unsigned i) const {
  assert(i < getNumOperands() && "Invalid Operand number!");
  return dyn_cast_or_null<MDNode>(getNMDOps(Operands)[i]);
}

/// addOperand - Add metadata Operand.
void NamedMDNode::addOperand(MDNode *M) {
  getNMDOps(Operands).push_back(WeakVH(M));
}

/// eraseFromParent - Drop all references and remove the node from parent
/// module.
void NamedMDNode::eraseFromParent() {
  getParent()->getNamedMDList().erase(this);
}

/// dropAllReferences - Remove all uses and clear node vector.
void NamedMDNode::dropAllReferences() {
  getNMDOps(Operands).clear();
}

/// setName - Set the name of this named metadata.
void NamedMDNode::setName(const Twine &NewName) {
  assert (!NewName.isTriviallyEmpty() && "Invalid named metadata name!");

  SmallString<256> NameData;
  StringRef NameRef = NewName.toStringRef(NameData);

  // Name isn't changing?
  if (getName() == NameRef)
    return;

  Name = NameRef.str();
  if (Parent)
    Parent->getMDSymbolTable().insert(NameRef, this);
}

/// getName - Return a constant reference to this named metadata's name.
StringRef NamedMDNode::getName() const {
  return StringRef(Name);
}

//===----------------------------------------------------------------------===//
// Instruction Metadata method implementations.
//

void Instruction::setMetadata(const char *Kind, MDNode *Node) {
  if (Node == 0 && !hasMetadata()) return;
  setMetadata(getContext().getMDKindID(Kind), Node);
}

MDNode *Instruction::getMetadataImpl(const char *Kind) const {
  return getMetadataImpl(getContext().getMDKindID(Kind));
}

void Instruction::setDbgMetadata(MDNode *Node) {
  DbgLoc = DebugLoc::getFromDILocation(Node);
}

/// setMetadata - Set the metadata of of the specified kind to the specified
/// node.  This updates/replaces metadata if already present, or removes it if
/// Node is null.
void Instruction::setMetadata(unsigned KindID, MDNode *Node) {
  if (Node == 0 && !hasMetadata()) return;

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
      for (unsigned i = 0, e = Info.size(); i != e; ++i)
        if (Info[i].first == KindID) {
          Info[i].second = Node;
          return;
        }
    }

    // No replacement, just add it to the list.
    Info.push_back(std::make_pair(KindID, Node));
    return;
  }

  // Otherwise, we're removing metadata from an instruction.
  assert(hasMetadataHashEntry() &&
         getContext().pImpl->MetadataStore.count(this) &&
         "HasMetadata bit out of date!");
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

MDNode *Instruction::getMetadataImpl(unsigned KindID) const {
  // Handle 'dbg' as a special case since it is not stored in the hash table.
  if (KindID == LLVMContext::MD_dbg)
    return DbgLoc.getAsMDNode(getContext());
  
  if (!hasMetadataHashEntry()) return 0;
  
  LLVMContextImpl::MDMapTy &Info = getContext().pImpl->MetadataStore[this];
  assert(!Info.empty() && "bit out of sync with hash table");

  for (LLVMContextImpl::MDMapTy::iterator I = Info.begin(), E = Info.end();
       I != E; ++I)
    if (I->first == KindID)
      return I->second;
  return 0;
}

void Instruction::getAllMetadataImpl(SmallVectorImpl<std::pair<unsigned,
                                       MDNode*> > &Result) const {
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

void Instruction::
getAllMetadataOtherThanDebugLocImpl(SmallVectorImpl<std::pair<unsigned,
                                    MDNode*> > &Result) const {
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


/// removeAllMetadata - Remove all metadata from this instruction.
void Instruction::removeAllMetadata() {
  assert(hasMetadata() && "Caller should check");
  DbgLoc = DebugLoc();
  if (hasMetadataHashEntry()) {
    getContext().pImpl->MetadataStore.erase(this);
    setHasMetadataHashEntry(false);
  }
}

