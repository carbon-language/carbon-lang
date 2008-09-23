//===-- Attributes.cpp - Implement ParamAttrsList -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ParamAttrsList class and ParamAttr utilities.
//
//===----------------------------------------------------------------------===//

#include "llvm/Attributes.h"
#include "llvm/Type.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/Support/Streams.h"
#include "llvm/Support/ManagedStatic.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// ParamAttr Function Definitions
//===----------------------------------------------------------------------===//

std::string ParamAttr::getAsString(Attributes Attrs) {
  std::string Result;
  if (Attrs & ParamAttr::ZExt)
    Result += "zeroext ";
  if (Attrs & ParamAttr::SExt)
    Result += "signext ";
  if (Attrs & ParamAttr::NoReturn)
    Result += "noreturn ";
  if (Attrs & ParamAttr::NoUnwind)
    Result += "nounwind ";
  if (Attrs & ParamAttr::InReg)
    Result += "inreg ";
  if (Attrs & ParamAttr::NoAlias)
    Result += "noalias ";
  if (Attrs & ParamAttr::StructRet)
    Result += "sret ";  
  if (Attrs & ParamAttr::ByVal)
    Result += "byval ";
  if (Attrs & ParamAttr::Nest)
    Result += "nest ";
  if (Attrs & ParamAttr::ReadNone)
    Result += "readnone ";
  if (Attrs & ParamAttr::ReadOnly)
    Result += "readonly ";
  if (Attrs & ParamAttr::Alignment) {
    Result += "align ";
    Result += utostr((Attrs & ParamAttr::Alignment) >> 16);
    Result += " ";
  }
  // Trim the trailing space.
  Result.erase(Result.end()-1);
  return Result;
}

Attributes ParamAttr::typeIncompatible(const Type *Ty) {
  Attributes Incompatible = None;
  
  if (!Ty->isInteger())
    // Attributes that only apply to integers.
    Incompatible |= SExt | ZExt;
  
  if (!isa<PointerType>(Ty))
    // Attributes that only apply to pointers.
    Incompatible |= ByVal | Nest | NoAlias | StructRet;
  
  return Incompatible;
}

//===----------------------------------------------------------------------===//
// ParamAttributeListImpl Definition
//===----------------------------------------------------------------------===//

namespace llvm {
class ParamAttributeListImpl : public FoldingSetNode {
  unsigned RefCount;
  
  // ParamAttrsList is uniqued, these should not be publicly available.
  void operator=(const ParamAttributeListImpl &); // Do not implement
  ParamAttributeListImpl(const ParamAttributeListImpl &); // Do not implement
  ~ParamAttributeListImpl();                        // Private implementation
public:
  SmallVector<ParamAttrsWithIndex, 4> Attrs;
  
  ParamAttributeListImpl(const ParamAttrsWithIndex *Attr, unsigned NumAttrs)
    : Attrs(Attr, Attr+NumAttrs) {
    RefCount = 0;
  }
  
  void AddRef() { ++RefCount; }
  void DropRef() { if (--RefCount == 0) delete this; }
  
  void Profile(FoldingSetNodeID &ID) const {
    Profile(ID, &Attrs[0], Attrs.size());
  }
  static void Profile(FoldingSetNodeID &ID, const ParamAttrsWithIndex *Attr,
                      unsigned NumAttrs) {
    for (unsigned i = 0; i != NumAttrs; ++i)
      ID.AddInteger(uint64_t(Attr[i].Attrs) << 32 | unsigned(Attr[i].Index));
  }
};
}

static ManagedStatic<FoldingSet<ParamAttributeListImpl> > ParamAttrsLists;

ParamAttributeListImpl::~ParamAttributeListImpl() {
  ParamAttrsLists->RemoveNode(this);
}


PAListPtr PAListPtr::get(const ParamAttrsWithIndex *Attrs, unsigned NumAttrs) {
  // If there are no attributes then return a null ParamAttrsList pointer.
  if (NumAttrs == 0)
    return PAListPtr();
  
#ifndef NDEBUG
  for (unsigned i = 0; i != NumAttrs; ++i) {
    assert(Attrs[i].Attrs != ParamAttr::None && 
           "Pointless parameter attribute!");
    assert((!i || Attrs[i-1].Index < Attrs[i].Index) &&
           "Misordered ParamAttrsList!");
  }
#endif
  
  // Otherwise, build a key to look up the existing attributes.
  FoldingSetNodeID ID;
  ParamAttributeListImpl::Profile(ID, Attrs, NumAttrs);
  void *InsertPos;
  ParamAttributeListImpl *PAL =
    ParamAttrsLists->FindNodeOrInsertPos(ID, InsertPos);
  
  // If we didn't find any existing attributes of the same shape then
  // create a new one and insert it.
  if (!PAL) {
    PAL = new ParamAttributeListImpl(Attrs, NumAttrs);
    ParamAttrsLists->InsertNode(PAL, InsertPos);
  }
  
  // Return the ParamAttrsList that we found or created.
  return PAListPtr(PAL);
}


//===----------------------------------------------------------------------===//
// PAListPtr Method Implementations
//===----------------------------------------------------------------------===//

PAListPtr::PAListPtr(ParamAttributeListImpl *LI) : PAList(LI) {
  if (LI) LI->AddRef();
}

PAListPtr::PAListPtr(const PAListPtr &P) : PAList(P.PAList) {
  if (PAList) PAList->AddRef();  
}

const PAListPtr &PAListPtr::operator=(const PAListPtr &RHS) {
  if (PAList == RHS.PAList) return *this;
  if (PAList) PAList->DropRef();
  PAList = RHS.PAList;
  if (PAList) PAList->AddRef();
  return *this;
}

PAListPtr::~PAListPtr() {
  if (PAList) PAList->DropRef();
}

/// getNumSlots - Return the number of slots used in this attribute list. 
/// This is the number of arguments that have an attribute set on them
/// (including the function itself).
unsigned PAListPtr::getNumSlots() const {
  return PAList ? PAList->Attrs.size() : 0;
}

/// getSlot - Return the ParamAttrsWithIndex at the specified slot.  This
/// holds a parameter number plus a set of attributes.
const ParamAttrsWithIndex &PAListPtr::getSlot(unsigned Slot) const {
  assert(PAList && Slot < PAList->Attrs.size() && "Slot # out of range!");
  return PAList->Attrs[Slot];
}


/// getParamAttrs - The parameter attributes for the specified parameter are
/// returned.  Parameters for the result are denoted with Idx = 0.
/// Function notes are denoted with idx = ~0.
Attributes PAListPtr::getParamAttrs(unsigned Idx) const {
  if (PAList == 0) return ParamAttr::None;
  
  const SmallVector<ParamAttrsWithIndex, 4> &Attrs = PAList->Attrs;
  for (unsigned i = 0, e = Attrs.size(); i != e && Attrs[i].Index <= Idx; ++i)
    if (Attrs[i].Index == Idx)
      return Attrs[i].Attrs;
  return ParamAttr::None;
}

/// hasAttrSomewhere - Return true if the specified attribute is set for at
/// least one parameter or for the return value.
bool PAListPtr::hasAttrSomewhere(Attributes Attr) const {
  if (PAList == 0) return false;
  
  const SmallVector<ParamAttrsWithIndex, 4> &Attrs = PAList->Attrs;
  for (unsigned i = 0, e = Attrs.size(); i != e; ++i)
    if (Attrs[i].Attrs & Attr)
      return true;
  return false;
}


PAListPtr PAListPtr::addAttr(unsigned Idx, Attributes Attrs) const {
  Attributes OldAttrs = getParamAttrs(Idx);
#ifndef NDEBUG
  // FIXME it is not obvious how this should work for alignment.
  // For now, say we can't change a known alignment.
  Attributes OldAlign = OldAttrs & ParamAttr::Alignment;
  Attributes NewAlign = Attrs & ParamAttr::Alignment;
  assert((!OldAlign || !NewAlign || OldAlign == NewAlign) &&
         "Attempt to change alignment!");
#endif
  
  Attributes NewAttrs = OldAttrs | Attrs;
  if (NewAttrs == OldAttrs)
    return *this;
  
  SmallVector<ParamAttrsWithIndex, 8> NewAttrList;
  if (PAList == 0)
    NewAttrList.push_back(ParamAttrsWithIndex::get(Idx, Attrs));
  else {
    const SmallVector<ParamAttrsWithIndex, 4> &OldAttrList = PAList->Attrs;
    unsigned i = 0, e = OldAttrList.size();
    // Copy attributes for arguments before this one.
    for (; i != e && OldAttrList[i].Index < Idx; ++i)
      NewAttrList.push_back(OldAttrList[i]);

    // If there are attributes already at this index, merge them in.
    if (i != e && OldAttrList[i].Index == Idx) {
      Attrs |= OldAttrList[i].Attrs;
      ++i;
    }
    
    NewAttrList.push_back(ParamAttrsWithIndex::get(Idx, Attrs));
    
    // Copy attributes for arguments after this one.
    NewAttrList.insert(NewAttrList.end(), 
                       OldAttrList.begin()+i, OldAttrList.end());
  }
  
  return get(&NewAttrList[0], NewAttrList.size());
}

PAListPtr PAListPtr::removeAttr(unsigned Idx, Attributes Attrs) const {
#ifndef NDEBUG
  // FIXME it is not obvious how this should work for alignment.
  // For now, say we can't pass in alignment, which no current use does.
  assert(!(Attrs & ParamAttr::Alignment) && "Attempt to exclude alignment!");
#endif
  if (PAList == 0) return PAListPtr();
  
  Attributes OldAttrs = getParamAttrs(Idx);
  Attributes NewAttrs = OldAttrs & ~Attrs;
  if (NewAttrs == OldAttrs)
    return *this;

  SmallVector<ParamAttrsWithIndex, 8> NewAttrList;
  const SmallVector<ParamAttrsWithIndex, 4> &OldAttrList = PAList->Attrs;
  unsigned i = 0, e = OldAttrList.size();
  
  // Copy attributes for arguments before this one.
  for (; i != e && OldAttrList[i].Index < Idx; ++i)
    NewAttrList.push_back(OldAttrList[i]);
  
  // If there are attributes already at this index, merge them in.
  assert(OldAttrList[i].Index == Idx && "Attribute isn't set?");
  Attrs = OldAttrList[i].Attrs & ~Attrs;
  ++i;
  if (Attrs)  // If any attributes left for this parameter, add them.
    NewAttrList.push_back(ParamAttrsWithIndex::get(Idx, Attrs));
  
  // Copy attributes for arguments after this one.
  NewAttrList.insert(NewAttrList.end(), 
                     OldAttrList.begin()+i, OldAttrList.end());
  
  return get(&NewAttrList[0], NewAttrList.size());
}

void PAListPtr::dump() const {
  cerr << "PAL[ ";
  for (unsigned i = 0; i < getNumSlots(); ++i) {
    const ParamAttrsWithIndex &PAWI = getSlot(i);
    cerr << "{" << PAWI.Index << "," << PAWI.Attrs << "} ";
  }
  
  cerr << "]\n";
}
