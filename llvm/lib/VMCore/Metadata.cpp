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

#include "LLVMContextImpl.h"
#include "llvm/Metadata.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Instruction.h"
#include "SymbolTableListTraitsImpl.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// MetadataBase implementation.
//

//===----------------------------------------------------------------------===//
// MDString implementation.
//
MDString *MDString::get(LLVMContext &Context, const StringRef &Str) {
  LLVMContextImpl *pImpl = Context.pImpl;
  StringMapEntry<MDString *> &Entry = 
    pImpl->MDStringCache.GetOrCreateValue(Str);
  MDString *&S = Entry.getValue();
  if (S) return S;
  
  return S = new MDString(Context, Entry.getKeyData(), Entry.getKeyLength());
}

//===----------------------------------------------------------------------===//
// MDNode implementation.
//
MDNode::MDNode(LLVMContext &C, Value *const *Vals, unsigned NumVals)
  : MetadataBase(Type::getMetadataTy(C), Value::MDNodeVal) {
  NodeSize = NumVals;
  Node = new ElementVH[NodeSize];
  ElementVH *Ptr = Node;
  for (unsigned i = 0; i != NumVals; ++i) 
    *Ptr++ = ElementVH(Vals[i], this);
}

void MDNode::Profile(FoldingSetNodeID &ID) const {
  for (unsigned i = 0, e = getNumElements(); i != e; ++i)
    ID.AddPointer(getElement(i));
}

MDNode *MDNode::get(LLVMContext &Context, Value*const* Vals, unsigned NumVals) {
  LLVMContextImpl *pImpl = Context.pImpl;
  FoldingSetNodeID ID;
  for (unsigned i = 0; i != NumVals; ++i)
    ID.AddPointer(Vals[i]);

  void *InsertPoint;
  MDNode *N;
  {
    N = pImpl->MDNodeSet.FindNodeOrInsertPos(ID, InsertPoint);
  }  
  if (N) return N;
  
  N = pImpl->MDNodeSet.FindNodeOrInsertPos(ID, InsertPoint);
  if (!N) {
    // InsertPoint will have been set by the FindNodeOrInsertPos call.
    N = new MDNode(Context, Vals, NumVals);
    pImpl->MDNodeSet.InsertNode(N, InsertPoint);
  }

  return N;
}

/// ~MDNode - Destroy MDNode.
MDNode::~MDNode() {
  {
    LLVMContextImpl *pImpl = getType()->getContext().pImpl;
    pImpl->MDNodeSet.RemoveNode(this);
  }
  delete [] Node;
  Node = NULL;
}

// Replace value from this node's element list.
void MDNode::replaceElement(Value *From, Value *To) {
  if (From == To || !getType())
    return;
  LLVMContext &Context = getType()->getContext();
  LLVMContextImpl *pImpl = Context.pImpl;

  // Find value. This is a linear search, do something if it consumes 
  // lot of time. It is possible that to have multiple instances of
  // From in this MDNode's element list.
  SmallVector<unsigned, 4> Indexes;
  unsigned Index = 0;
  for (unsigned i = 0, e = getNumElements(); i != e; ++i, ++Index) {
    Value *V = getElement(i);
    if (V && V == From) 
      Indexes.push_back(Index);
  }

  if (Indexes.empty())
    return;

  // Remove "this" from the context map. 
  pImpl->MDNodeSet.RemoveNode(this);

  // Replace From element(s) in place.
  for (SmallVector<unsigned, 4>::iterator I = Indexes.begin(), E = Indexes.end(); 
       I != E; ++I) {
    unsigned Index = *I;
    Node[Index] = ElementVH(To, this);
  }

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
    N = 0;
  }

  N = pImpl->MDNodeSet.FindNodeOrInsertPos(ID, InsertPoint);
  if (!N) {
    // InsertPoint will have been set by the FindNodeOrInsertPos call.
    N = this;
    pImpl->MDNodeSet.InsertNode(N, InsertPoint);
  }
}

//===----------------------------------------------------------------------===//
// NamedMDNode implementation.
//
NamedMDNode::NamedMDNode(LLVMContext &C, const Twine &N,
                         MetadataBase *const *MDs, 
                         unsigned NumMDs, Module *ParentModule)
  : MetadataBase(Type::getMetadataTy(C), Value::NamedMDNodeVal), Parent(0) {
  setName(N);

  for (unsigned i = 0; i != NumMDs; ++i)
    Node.push_back(WeakMetadataVH(MDs[i]));

  if (ParentModule)
    ParentModule->getNamedMDList().push_back(this);
}

NamedMDNode *NamedMDNode::Create(const NamedMDNode *NMD, Module *M) {
  assert(NMD && "Invalid source NamedMDNode!");
  SmallVector<MetadataBase *, 4> Elems;
  for (unsigned i = 0, e = NMD->getNumElements(); i != e; ++i)
    Elems.push_back(NMD->getElement(i));
  return new NamedMDNode(NMD->getContext(), NMD->getName().data(),
                         Elems.data(), Elems.size(), M);
}

/// eraseFromParent - Drop all references and remove the node from parent
/// module.
void NamedMDNode::eraseFromParent() {
  getParent()->getNamedMDList().erase(this);
}

/// dropAllReferences - Remove all uses and clear node vector.
void NamedMDNode::dropAllReferences() {
  Node.clear();
}

NamedMDNode::~NamedMDNode() {
  dropAllReferences();
}

//===----------------------------------------------------------------------===//
// MetadataContext implementation.
//

/// registerMDKind - Register a new metadata kind and return its ID.
/// A metadata kind can be registered only once. 
unsigned MetadataContext::registerMDKind(const StringRef Name) {
  assert(isValidName(Name) && "Invalid custome metadata name!");
  unsigned Count = MDHandlerNames.size();
  assert(MDHandlerNames.count(Name) == 0 && "Already registered MDKind!");
  return MDHandlerNames[Name] = Count + 1;
}

/// isValidName - Return true if Name is a valid custom metadata handler name.
bool MetadataContext::isValidName(const StringRef MDName) {
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

/// getMDKind - Return metadata kind. If the requested metadata kind
/// is not registered then return 0.
unsigned MetadataContext::getMDKind(const StringRef Name) const {
  StringMap<unsigned>::const_iterator I = MDHandlerNames.find(Name);
  if (I == MDHandlerNames.end()) {
    assert(isValidName(Name) && "Invalid custome metadata name!");
    return 0;
  }

  return I->getValue();
}

/// addMD - Attach the metadata of given kind to an Instruction.
void MetadataContext::addMD(unsigned MDKind, MDNode *Node, Instruction *Inst) {
  assert(Node && "Invalid null MDNode");
  Inst->HasMetadata = true;
  MDMapTy &Info = MetadataStore[Inst];
  if (Info.empty()) {
    Info.push_back(std::make_pair(MDKind, Node));
    MetadataStore.insert(std::make_pair(Inst, Info));
    return;
  }

  // If there is an entry for this MDKind then replace it.
  for (unsigned i = 0, e = Info.size(); i != e; ++i) {
    MDPairTy &P = Info[i];
    if (P.first == MDKind) {
      Info[i] = std::make_pair(MDKind, Node);
      return;
    }
  }

  // Otherwise add a new entry.
  Info.push_back(std::make_pair(MDKind, Node));
}

/// removeMD - Remove metadata of given kind attached with an instuction.
void MetadataContext::removeMD(unsigned Kind, Instruction *Inst) {
  MDStoreTy::iterator I = MetadataStore.find(Inst);
  if (I == MetadataStore.end())
    return;

  MDMapTy &Info = I->second;
  for (MDMapTy::iterator MI = Info.begin(), ME = Info.end(); MI != ME; ++MI) {
    MDPairTy &P = *MI;
    if (P.first == Kind) {
      Info.erase(MI);
      return;
    }
  }
}
  
/// removeAllMetadata - Remove all metadata attached with an instruction.
void MetadataContext::removeAllMetadata(Instruction *Inst) {
  MetadataStore.erase(Inst);
  Inst->HasMetadata = false;
}

/// copyMD - If metadata is attached with Instruction In1 then attach
/// the same metadata to In2.
void MetadataContext::copyMD(Instruction *In1, Instruction *In2) {
  assert(In1 && In2 && "Invalid instruction!");
  MDMapTy &In1Info = MetadataStore[In1];
  if (In1Info.empty())
    return;

  for (MDMapTy::iterator I = In1Info.begin(), E = In1Info.end(); I != E; ++I)
    if (MDNode *MD = dyn_cast_or_null<MDNode>(I->second))
      addMD(I->first, MD, In2);
}

/// getMD - Get the metadata of given kind attached to an Instruction.
/// If the metadata is not found then return 0.
MDNode *MetadataContext::getMD(unsigned MDKind, const Instruction *Inst) {
  MDMapTy &Info = MetadataStore[Inst];
  if (Info.empty())
    return NULL;

  for (MDMapTy::iterator I = Info.begin(), E = Info.end(); I != E; ++I)
    if (I->first == MDKind)
      return dyn_cast_or_null<MDNode>(I->second);
  return NULL;
}

/// getMDs - Get the metadata attached to an Instruction.
const MetadataContext::MDMapTy *
MetadataContext::getMDs(const Instruction *Inst) {
  MDStoreTy::iterator I = MetadataStore.find(Inst);
  if (I == MetadataStore.end())
    return NULL;
  
  return &I->second;
}

/// getHandlerNames - Get handler names. This is used by bitcode
/// writer.
const StringMap<unsigned> *MetadataContext::getHandlerNames() {
  return &MDHandlerNames;
}

/// ValueIsCloned - This handler is used to update metadata store
/// when In1 is cloned to create In2.
void MetadataContext::ValueIsCloned(const Instruction *In1, Instruction *In2) {
  // Find Metadata handles for In1.
  MDStoreTy::iterator I = MetadataStore.find(In1);
  assert(I != MetadataStore.end() && "Invalid custom metadata info!");

  // FIXME : Give all metadata handlers a chance to adjust.

  MDMapTy &In1Info = I->second;
  MDMapTy In2Info;
  for (MDMapTy::iterator I = In1Info.begin(), E = In1Info.end(); I != E; ++I)
    if (MDNode *MD = dyn_cast_or_null<MDNode>(I->second))
      addMD(I->first, MD, In2);
}

/// ValueIsRAUWd - This handler is used when V1's all uses are replaced by
/// V2.
void MetadataContext::ValueIsRAUWd(Value *V1, Value *V2) {
  Instruction *I1 = dyn_cast<Instruction>(V1);
  Instruction *I2 = dyn_cast<Instruction>(V2);
  if (!I1 || !I2)
    return;

  // FIXME : Give custom handlers a chance to override this.
  ValueIsCloned(I1, I2);
}

