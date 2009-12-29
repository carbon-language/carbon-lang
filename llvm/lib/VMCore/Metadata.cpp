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
// MDNodeElement implementation.
//

// Use CallbackVH to hold MDNode elements.
namespace llvm {
class MDNodeElement : public CallbackVH {
  MDNode *Parent;
public:
  MDNodeElement() {}
  MDNodeElement(Value *V, MDNode *P) : CallbackVH(V), Parent(P) {}
  ~MDNodeElement() {}
  
  void set(Value *V, MDNode *P) {
    setValPtr(V);
    Parent = P;
  }
  
  virtual void deleted();
  virtual void allUsesReplacedWith(Value *NV);
};
} // end namespace llvm.


void MDNodeElement::deleted() {
  Parent->replaceElement(this, 0);
}

void MDNodeElement::allUsesReplacedWith(Value *NV) {
  Parent->replaceElement(this, NV);
}



//===----------------------------------------------------------------------===//
// MDNode implementation.
//

/// ~MDNode - Destroy MDNode.
MDNode::~MDNode() {
  LLVMContextImpl *pImpl = getType()->getContext().pImpl;
  pImpl->MDNodeSet.RemoveNode(this);
  delete [] Operands;
  Operands = NULL;
}

MDNode::MDNode(LLVMContext &C, Value *const *Vals, unsigned NumVals,
               bool isFunctionLocal)
  : MetadataBase(Type::getMetadataTy(C), Value::MDNodeVal) {
  NumOperands = NumVals;
  Operands = new MDNodeElement[NumOperands];
    
  for (unsigned i = 0; i != NumVals; ++i) 
    Operands[i].set(Vals[i], this);
    
  if (isFunctionLocal)
    setValueSubclassData(getSubclassDataFromValue() | FunctionLocalBit);
}

MDNode *MDNode::get(LLVMContext &Context, Value*const* Vals, unsigned NumVals,
                    bool isFunctionLocal) {
  LLVMContextImpl *pImpl = Context.pImpl;
  FoldingSetNodeID ID;
  for (unsigned i = 0; i != NumVals; ++i)
    ID.AddPointer(Vals[i]);

  void *InsertPoint;
  MDNode *N = pImpl->MDNodeSet.FindNodeOrInsertPos(ID, InsertPoint);
  if (!N) {
    // InsertPoint will have been set by the FindNodeOrInsertPos call.
    N = new MDNode(Context, Vals, NumVals, isFunctionLocal);
    pImpl->MDNodeSet.InsertNode(N, InsertPoint);
  }
  return N;
}

void MDNode::Profile(FoldingSetNodeID &ID) const {
  for (unsigned i = 0, e = getNumElements(); i != e; ++i)
    ID.AddPointer(getElement(i));
  // HASH TABLE COLLISIONS?
  // DO NOT REINSERT AFTER AN OPERAND DROPS TO NULL!
}


/// getElement - Return specified element.
Value *MDNode::getElement(unsigned i) const {
  assert(i < getNumElements() && "Invalid element number!");
  return Operands[i];
}



// Replace value from this node's element list.
void MDNode::replaceElement(MDNodeElement *Op, Value *To) {
  Value *From = *Op;
  
  if (From == To)
    return;

  LLVMContextImpl *pImpl = getType()->getContext().pImpl;

  // Remove "this" from the context map.  FoldingSet doesn't have to reprofile
  // this node to remove it, so we don't care what state the operands are in.
  pImpl->MDNodeSet.RemoveNode(this);

  // Update the operand.
  Op->set(To, this);

  // Insert updated "this" into the context's folding node set.
  // If a node with same element list already exist then before inserting 
  // updated "this" into the folding node set, replace all uses of existing 
  // node with updated "this" node.
  FoldingSetNodeID ID;
  Profile(ID);
  void *InsertPoint;
  MDNode *N = pImpl->MDNodeSet.FindNodeOrInsertPos(ID, InsertPoint);

  if (N) {
    N->replaceAllUsesWith(this);
    delete N;
    N = pImpl->MDNodeSet.FindNodeOrInsertPos(ID, InsertPoint);
    assert(N == 0 && "shouldn't be in the map now!"); (void)N;
  }

  // InsertPoint will have been set by the FindNodeOrInsertPos call.
  pImpl->MDNodeSet.InsertNode(this, InsertPoint);
}

//===----------------------------------------------------------------------===//
// NamedMDNode implementation.
//
static SmallVector<TrackingVH<MetadataBase>, 4> &getNMDOps(void *Operands) {
  return *(SmallVector<TrackingVH<MetadataBase>, 4>*)Operands;
}

NamedMDNode::NamedMDNode(LLVMContext &C, const Twine &N,
                         MetadataBase *const *MDs, 
                         unsigned NumMDs, Module *ParentModule)
  : MetadataBase(Type::getMetadataTy(C), Value::NamedMDNodeVal), Parent(0) {
  setName(N);
    
  Operands = new SmallVector<TrackingVH<MetadataBase>, 4>();
    
  SmallVector<TrackingVH<MetadataBase>, 4> &Node = getNMDOps(Operands);
  for (unsigned i = 0; i != NumMDs; ++i)
    Node.push_back(TrackingVH<MetadataBase>(MDs[i]));

  if (ParentModule)
    ParentModule->getNamedMDList().push_back(this);
}

NamedMDNode *NamedMDNode::Create(const NamedMDNode *NMD, Module *M) {
  assert(NMD && "Invalid source NamedMDNode!");
  SmallVector<MetadataBase *, 4> Elems;
  Elems.reserve(NMD->getNumElements());
  
  for (unsigned i = 0, e = NMD->getNumElements(); i != e; ++i)
    Elems.push_back(NMD->getElement(i));
  return new NamedMDNode(NMD->getContext(), NMD->getName().data(),
                         Elems.data(), Elems.size(), M);
}

NamedMDNode::~NamedMDNode() {
  dropAllReferences();
  delete &getNMDOps(Operands);
}

/// getNumElements - Return number of NamedMDNode elements.
unsigned NamedMDNode::getNumElements() const {
  return (unsigned)getNMDOps(Operands).size();
}

/// getElement - Return specified element.
MetadataBase *NamedMDNode::getElement(unsigned i) const {
  assert(i < getNumElements() && "Invalid element number!");
  return getNMDOps(Operands)[i];
}

/// addElement - Add metadata element.
void NamedMDNode::addElement(MetadataBase *M) {
  getNMDOps(Operands).push_back(TrackingVH<MetadataBase>(M));
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


//===----------------------------------------------------------------------===//
// MetadataContext implementation.
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

