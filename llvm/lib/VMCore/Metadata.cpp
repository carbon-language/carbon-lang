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
#include "SymbolTableListTraitsImpl.h"
#include "llvm/Support/ValueHandle.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// MDString implementation.
//

MDString::MDString(LLVMContext &C, StringRef S)
  : MetadataBase(Type::getMetadataTy(C), Value::MDStringVal), Str(S) {}

MDString *MDString::get(LLVMContext &Context, StringRef Str) {
  LLVMContextImpl *pImpl = Context.pImpl;
  StringMapEntry<MDString *> &Entry =
    pImpl->MDStringCache.GetOrCreateValue(Str);
  MDString *&S = Entry.getValue();
  if (!S) S = new MDString(Context, Entry.getKey());
  return S;
}

MDString *MDString::get(LLVMContext &Context, const char *Str) {
  LLVMContextImpl *pImpl = Context.pImpl;
  StringMapEntry<MDString *> &Entry =
    pImpl->MDStringCache.GetOrCreateValue(Str ? StringRef(Str) : StringRef());
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
: MetadataBase(Type::getMetadataTy(C), Value::MDNodeVal) {
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
  if (!isNotUniqued()) {
    LLVMContextImpl *pImpl = getType()->getContext().pImpl;
    pImpl->MDNodeSet.RemoveNode(this);
  }

  // Destroy the operands.
  for (MDNodeOperand *Op = getOperandPtr(this, 0), *E = Op+NumOperands;
       Op != E; ++Op)
    Op->~MDNodeOperand();
}

// destroy - Delete this node.  Only when there are no uses.
void MDNode::destroy() {
  setValueSubclassData(getSubclassDataFromValue() | DestroyFlag);
  // Placement delete, the free the memory.
  this->~MDNode();
  free(this);
}

MDNode *MDNode::getMDNode(LLVMContext &Context, Value *const *Vals,
                          unsigned NumVals, FunctionLocalness FL) {
  LLVMContextImpl *pImpl = Context.pImpl;
  FoldingSetNodeID ID;
  for (unsigned i = 0; i != NumVals; ++i)
    ID.AddPointer(Vals[i]);

  void *InsertPoint;
  MDNode *N = pImpl->MDNodeSet.FindNodeOrInsertPos(ID, InsertPoint);
  if (!N) {
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

    // Coallocate space for the node and Operands together, then placement new.
    void *Ptr = malloc(sizeof(MDNode)+NumVals*sizeof(MDNodeOperand));
    N = new (Ptr) MDNode(Context, Vals, NumVals, isFunctionLocal);

    // InsertPoint will have been set by the FindNodeOrInsertPos call.
    pImpl->MDNodeSet.InsertNode(N, InsertPoint);
  }
  return N;
}

MDNode *MDNode::get(LLVMContext &Context, Value*const* Vals, unsigned NumVals) {
  return getMDNode(Context, Vals, NumVals, FL_Unknown);
}

MDNode *MDNode::getWhenValsUnresolved(LLVMContext &Context, Value*const* Vals,
                                      unsigned NumVals, bool isFunctionLocal) {
  return getMDNode(Context, Vals, NumVals, isFunctionLocal ? FL_Yes : FL_No);
}

/// getOperand - Return specified operand.
Value *MDNode::getOperand(unsigned i) const {
  return *getOperandPtr(const_cast<MDNode*>(this), i);
}

void MDNode::Profile(FoldingSetNodeID &ID) const {
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    ID.AddPointer(getOperand(i));
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
static SmallVector<WeakVH, 4> &getNMDOps(void *Operands) {
  return *(SmallVector<WeakVH, 4>*)Operands;
}

NamedMDNode::NamedMDNode(LLVMContext &C, StringRef N,
                         MDNode *const *MDs,
                         unsigned NumMDs, Module *ParentModule)
  : Value(Type::getMetadataTy(C), Value::NamedMDNodeVal), Parent(0) {
  setName(N);
  Operands = new SmallVector<WeakVH, 4>();

  SmallVector<WeakVH, 4> &Node = getNMDOps(Operands);
  for (unsigned i = 0; i != NumMDs; ++i)
    Node.push_back(WeakVH(MDs[i]));

  if (ParentModule) {
    ParentModule->getNamedMDList().push_back(this);
    ParentModule->addMDNodeName(N, this);
  }
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
  getParent()->getMDSymbolTable().remove(getName());
  getParent()->getNamedMDList().erase(this);
}

/// dropAllReferences - Remove all uses and clear node vector.
void NamedMDNode::dropAllReferences() {
  getNMDOps(Operands).clear();
}

/// setName - Set the name of this named metadata.
void NamedMDNode::setName(StringRef N) {
  if (!N.empty())
    Name = N.str();
}

/// getName - Return a constant reference to this named metadata's name.
StringRef NamedMDNode::getName() const {
  return StringRef(Name);
}

//===----------------------------------------------------------------------===//
// LLVMContext MDKind naming implementation.
//

#ifndef NDEBUG
/// isValidName - Return true if Name is a valid custom metadata handler name.
static bool isValidName(StringRef MDName) {
  if (MDName.empty())
    return false;

  if (!isalpha(MDName[0]))
    return false;

  for (StringRef::iterator I = MDName.begin() + 1, E = MDName.end(); I != E;
       ++I) {
    if (!isalnum(*I) && *I != '_' && *I != '-' && *I != '.')
        return false;
  }
  return true;
}
#endif

/// getMDKindID - Return a unique non-zero ID for the specified metadata kind.
unsigned LLVMContext::getMDKindID(StringRef Name) const {
  assert(isValidName(Name) && "Invalid MDNode name");

  unsigned &Entry = pImpl->CustomMDKindNames[Name];

  // If this is new, assign it its ID.
  if (Entry == 0) Entry = pImpl->CustomMDKindNames.size();
  return Entry;
}

/// getHandlerNames - Populate client supplied smallvector using custome
/// metadata name and ID.
void LLVMContext::getMDKindNames(SmallVectorImpl<StringRef> &Names) const {
  Names.resize(pImpl->CustomMDKindNames.size()+1);
  Names[0] = "";
  for (StringMap<unsigned>::const_iterator I = pImpl->CustomMDKindNames.begin(),
       E = pImpl->CustomMDKindNames.end(); I != E; ++I)
    // MD Handlers are numbered from 1.
    Names[I->second] = I->first();
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

/// setMetadata - Set the metadata of of the specified kind to the specified
/// node.  This updates/replaces metadata if already present, or removes it if
/// Node is null.
void Instruction::setMetadata(unsigned KindID, MDNode *Node) {
  if (Node == 0 && !hasMetadata()) return;

  // Handle the case when we're adding/updating metadata on an instruction.
  if (Node) {
    LLVMContextImpl::MDMapTy &Info = getContext().pImpl->MetadataStore[this];
    assert(!Info.empty() == hasMetadata() && "HasMetadata bit is wonked");
    if (Info.empty()) {
      setHasMetadata(true);
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
  assert(hasMetadata() && getContext().pImpl->MetadataStore.count(this) &&
         "HasMetadata bit out of date!");
  LLVMContextImpl::MDMapTy &Info = getContext().pImpl->MetadataStore[this];

  // Common case is removing the only entry.
  if (Info.size() == 1 && Info[0].first == KindID) {
    getContext().pImpl->MetadataStore.erase(this);
    setHasMetadata(false);
    return;
  }

  // Handle replacement of an existing value.
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
  LLVMContextImpl::MDMapTy &Info = getContext().pImpl->MetadataStore[this];
  assert(hasMetadata() && !Info.empty() && "Shouldn't have called this");

  for (LLVMContextImpl::MDMapTy::iterator I = Info.begin(), E = Info.end();
       I != E; ++I)
    if (I->first == KindID)
      return I->second;
  return 0;
}

void Instruction::getAllMetadataImpl(SmallVectorImpl<std::pair<unsigned,
                                       MDNode*> > &Result)const {
  assert(hasMetadata() && getContext().pImpl->MetadataStore.count(this) &&
         "Shouldn't have called this");
  const LLVMContextImpl::MDMapTy &Info =
    getContext().pImpl->MetadataStore.find(this)->second;
  assert(!Info.empty() && "Shouldn't have called this");

  Result.clear();
  Result.append(Info.begin(), Info.end());

  // Sort the resulting array so it is stable.
  if (Result.size() > 1)
    array_pod_sort(Result.begin(), Result.end());
}

/// removeAllMetadata - Remove all metadata from this instruction.
void Instruction::removeAllMetadata() {
  assert(hasMetadata() && "Caller should check");
  getContext().pImpl->MetadataStore.erase(this);
  setHasMetadata(false);
}

