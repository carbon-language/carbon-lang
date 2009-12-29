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
// MetadataContextImpl implementation.
//
namespace llvm {
class MetadataContextImpl {
public:
  typedef std::pair<unsigned, TrackingVH<MDNode> > MDPairTy;
  typedef SmallVector<MDPairTy, 2> MDMapTy;
  typedef DenseMap<const Instruction *, MDMapTy> MDStoreTy;
  friend class BitcodeReader;
private:

  /// MetadataStore - Collection of metadata used in this context.
  MDStoreTy MetadataStore;

  /// MDHandlerNames - Map to hold metadata handler names.
  StringMap<unsigned> MDHandlerNames;

public:
  // Name <-> ID mapping methods.
  unsigned getMDKindID(StringRef Name);
  void getMDKindNames(SmallVectorImpl<StringRef> &) const;
  
  
  // Instruction metadata methods.
  MDNode *getMetadata(const Instruction *Inst, unsigned Kind);
  void getAllMetadata(const Instruction *Inst,
                      SmallVectorImpl<std::pair<unsigned, MDNode*> > &MDs)const;

  void setMetadata(Instruction *Inst, unsigned Kind, MDNode *Node);

  /// removeAllMetadata - Remove all metadata attached with an instruction.
  void removeAllMetadata(Instruction *Inst);
  
  
  
  /// copyMD - If metadata is attached with Instruction In1 then attach
  /// the same metadata to In2.
  void copyMD(Instruction *In1, Instruction *In2);
  

  /// ValueIsDeleted - This handler is used to update metadata store
  /// when a value is deleted.
  void ValueIsDeleted(const Value *) {}
  void ValueIsDeleted(Instruction *Inst) {
    removeAllMetadata(Inst);
  }
  void ValueIsRAUWd(Value *V1, Value *V2);

  /// ValueIsCloned - This handler is used to update metadata store
  /// when In1 is cloned to create In2.
  void ValueIsCloned(const Instruction *In1, Instruction *In2);
};
}

/// getMDKindID - Return a unique non-zero ID for the specified metadata kind.
unsigned MetadataContextImpl::getMDKindID(StringRef Name) {
  unsigned &Entry = MDHandlerNames[Name];

  // If this is new, assign it its ID.
  if (Entry == 0) Entry = MDHandlerNames.size();
  return Entry;
}

/// getHandlerNames - Populate client supplied smallvector using custome
/// metadata name and ID.
void MetadataContextImpl::
getMDKindNames(SmallVectorImpl<StringRef> &Names) const {
  Names.resize(MDHandlerNames.size()+1);
  Names[0] = "";
  for (StringMap<unsigned>::const_iterator I = MDHandlerNames.begin(),
       E = MDHandlerNames.end(); I != E; ++I) 
    // MD Handlers are numbered from 1.
    Names[I->second] = I->first();
}


/// getMetadata - Get the metadata of given kind attached to an Instruction.
/// If the metadata is not found then return 0.
MDNode *MetadataContextImpl::
getMetadata(const Instruction *Inst, unsigned MDKind) {
  MDMapTy &Info = MetadataStore[Inst];
  assert(Inst->hasMetadata() && !Info.empty() && "Shouldn't have called this");
  
  for (MDMapTy::iterator I = Info.begin(), E = Info.end(); I != E; ++I)
    if (I->first == MDKind)
      return I->second;
  return 0;
}

/// getAllMetadata - Get all of the metadata attached to an Instruction.
void MetadataContextImpl::
getAllMetadata(const Instruction *Inst,
               SmallVectorImpl<std::pair<unsigned, MDNode*> > &Result) const {
  assert(Inst->hasMetadata() && MetadataStore.find(Inst) != MetadataStore.end()
         && "Shouldn't have called this");
  const MDMapTy &Info = MetadataStore.find(Inst)->second;
  assert(!Info.empty() && "Shouldn't have called this");

  Result.clear();
  Result.append(Info.begin(), Info.end());
  
  // Sort the resulting array so it is stable.
  if (Result.size() > 1)
    array_pod_sort(Result.begin(), Result.end());
}


void MetadataContextImpl::setMetadata(Instruction *Inst, unsigned Kind,
                                      MDNode *Node) {
  // Handle the case when we're adding/updating metadata on an instruction.
  if (Node) {
    MDMapTy &Info = MetadataStore[Inst];
    assert(!Info.empty() == Inst->hasMetadata() && "HasMetadata bit is wonked");
    if (Info.empty()) {
      Inst->setHasMetadata(true);
    } else {
      // Handle replacement of an existing value.
      for (unsigned i = 0, e = Info.size(); i != e; ++i)
        if (Info[i].first == Kind) {
          Info[i].second = Node;
          return;
        }
    }
    
    // No replacement, just add it to the list.
    Info.push_back(std::make_pair(Kind, Node));
    return;
  }
  
  // Otherwise, we're removing metadata from an instruction.
  assert(Inst->hasMetadata() && MetadataStore.count(Inst) &&
         "HasMetadata bit out of date!");
  MDMapTy &Info = MetadataStore[Inst];

  // Common case is removing the only entry.
  if (Info.size() == 1 && Info[0].first == Kind) {
    MetadataStore.erase(Inst);
    Inst->setHasMetadata(false);
    return;
  }
  
  // Handle replacement of an existing value.
  for (unsigned i = 0, e = Info.size(); i != e; ++i)
    if (Info[i].first == Kind) {
      Info[i] = Info.back();
      Info.pop_back();
      assert(!Info.empty() && "Removing last entry should be handled above");
      return;
    }
  // Otherwise, removing an entry that doesn't exist on the instruction.
}

/// removeAllMetadata - Remove all metadata attached with an instruction.
void MetadataContextImpl::removeAllMetadata(Instruction *Inst) {
  MetadataStore.erase(Inst);
  Inst->setHasMetadata(false);
}


/// copyMD - If metadata is attached with Instruction In1 then attach
/// the same metadata to In2.
void MetadataContextImpl::copyMD(Instruction *In1, Instruction *In2) {
  assert(In1 && In2 && "Invalid instruction!");
  MDMapTy &In1Info = MetadataStore[In1];
  if (In1Info.empty())
    return;
  
  for (MDMapTy::iterator I = In1Info.begin(), E = In1Info.end(); I != E; ++I)
    In2->setMetadata(I->first, I->second);
}

/// ValueIsCloned - This handler is used to update metadata store
/// when In1 is cloned to create In2.
void MetadataContextImpl::ValueIsCloned(const Instruction *In1, 
                                        Instruction *In2) {
  // Find Metadata handles for In1.
  MDStoreTy::iterator I = MetadataStore.find(In1);
  assert(I != MetadataStore.end() && "Invalid custom metadata info!");

  // FIXME: Give all metadata handlers a chance to adjust.
  MDMapTy &In1Info = I->second;
  for (MDMapTy::iterator I = In1Info.begin(), E = In1Info.end(); I != E; ++I)
    In2->setMetadata(I->first, I->second);
}

/// ValueIsRAUWd - This handler is used when V1's all uses are replaced by
/// V2.
void MetadataContextImpl::ValueIsRAUWd(Value *V1, Value *V2) {
  Instruction *I1 = dyn_cast<Instruction>(V1);
  Instruction *I2 = dyn_cast<Instruction>(V2);
  if (!I1 || !I2)
    return;

  // FIXME: Give custom handlers a chance to override this.
  ValueIsCloned(I1, I2);
}

//===----------------------------------------------------------------------===//
// MetadataContext implementation.
//
MetadataContext::MetadataContext() : pImpl(new MetadataContextImpl()) { }
MetadataContext::~MetadataContext() { delete pImpl; }

/// isValidName - Return true if Name is a valid custom metadata handler name.
bool MetadataContext::isValidName(StringRef MDName) {
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

/// getMDKindID - Return a unique non-zero ID for the specified metadata kind.
unsigned MetadataContext::getMDKindID(StringRef Name) const {
  return pImpl->getMDKindID(Name);
}

/// copyMD - If metadata is attached with Instruction In1 then attach
/// the same metadata to In2.
void MetadataContext::copyMD(Instruction *In1, Instruction *In2) {
  pImpl->copyMD(In1, In2);
}

/// getHandlerNames - Populate client supplied smallvector using custome
/// metadata name and ID.
void MetadataContext::getMDKindNames(SmallVectorImpl<StringRef> &N) const {
  pImpl->getMDKindNames(N);
}

/// ValueIsDeleted - This handler is used to update metadata store
/// when a value is deleted.
void MetadataContext::ValueIsDeleted(Instruction *Inst) {
  pImpl->ValueIsDeleted(Inst);
}
void MetadataContext::ValueIsRAUWd(Value *V1, Value *V2) {
  pImpl->ValueIsRAUWd(V1, V2);
}

/// ValueIsCloned - This handler is used to update metadata store
/// when In1 is cloned to create In2.
void MetadataContext::ValueIsCloned(const Instruction *In1, Instruction *In2) {
  pImpl->ValueIsCloned(In1, In2);
}

//===----------------------------------------------------------------------===//
// Instruction Metadata method implementations.
//

void Instruction::setMetadata(const char *Kind, MDNode *Node) {
  if (Node == 0 && !hasMetadata()) return;
  setMetadata(getContext().getMetadata().getMDKindID(Kind), Node);
}

MDNode *Instruction::getMetadataImpl(const char *Kind) const {
  return getMetadataImpl(getContext().getMetadata().getMDKindID(Kind));
}

/// setMetadata - Set the metadata of of the specified kind to the specified
/// node.  This updates/replaces metadata if already present, or removes it if
/// Node is null.
void Instruction::setMetadata(unsigned KindID, MDNode *Node) {
  if (Node == 0 && !hasMetadata()) return;
  
  getContext().getMetadata().pImpl->setMetadata(this, KindID, Node);
}

MDNode *Instruction::getMetadataImpl(unsigned KindID) const {
  return getContext().getMetadata().pImpl->getMetadata(this, KindID);
}

void Instruction::getAllMetadataImpl(SmallVectorImpl<std::pair<unsigned,
                                       MDNode*> > &Result)const {
  getContext().getMetadata().pImpl->getAllMetadata(this, Result);
}

