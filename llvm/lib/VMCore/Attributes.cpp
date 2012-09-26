//===-- Attributes.cpp - Implement AttributesList -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the AttributesList class and Attribute utilities.
//
//===----------------------------------------------------------------------===//

#include "llvm/Attributes.h"
#include "AttributesImpl.h"
#include "LLVMContextImpl.h"
#include "llvm/Type.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/Support/Atomic.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// Attribute Function Definitions
//===----------------------------------------------------------------------===//

std::string Attributes::getAsString() const {
  std::string Result;
  if (hasZExtAttr())
    Result += "zeroext ";
  if (hasSExtAttr())
    Result += "signext ";
  if (hasNoReturnAttr())
    Result += "noreturn ";
  if (hasNoUnwindAttr())
    Result += "nounwind ";
  if (hasUWTableAttr())
    Result += "uwtable ";
  if (hasReturnsTwiceAttr())
    Result += "returns_twice ";
  if (hasInRegAttr())
    Result += "inreg ";
  if (hasNoAliasAttr())
    Result += "noalias ";
  if (hasNoCaptureAttr())
    Result += "nocapture ";
  if (hasStructRetAttr())
    Result += "sret ";
  if (hasByValAttr())
    Result += "byval ";
  if (hasNestAttr())
    Result += "nest ";
  if (hasReadNoneAttr())
    Result += "readnone ";
  if (hasReadOnlyAttr())
    Result += "readonly ";
  if (hasOptimizeForSizeAttr())
    Result += "optsize ";
  if (hasNoInlineAttr())
    Result += "noinline ";
  if (hasInlineHintAttr())
    Result += "inlinehint ";
  if (hasAlwaysInlineAttr())
    Result += "alwaysinline ";
  if (hasStackProtectAttr())
    Result += "ssp ";
  if (hasStackProtectReqAttr())
    Result += "sspreq ";
  if (hasNoRedZoneAttr())
    Result += "noredzone ";
  if (hasNoImplicitFloatAttr())
    Result += "noimplicitfloat ";
  if (hasNakedAttr())
    Result += "naked ";
  if (hasNonLazyBindAttr())
    Result += "nonlazybind ";
  if (hasAddressSafetyAttr())
    Result += "address_safety ";
  if (hasStackAlignmentAttr()) {
    Result += "alignstack(";
    Result += utostr(getStackAlignment());
    Result += ") ";
  }
  if (hasAlignmentAttr()) {
    Result += "align ";
    Result += utostr(getAlignment());
    Result += " ";
  }
  // Trim the trailing space.
  assert(!Result.empty() && "Unknown attribute!");
  Result.erase(Result.end()-1);
  return Result;
}

Attributes Attributes::typeIncompatible(Type *Ty) {
  Attributes Incompatible = Attribute::None;
  
  if (!Ty->isIntegerTy())
    // Attributes that only apply to integers.
    Incompatible |= Attribute::SExt | Attribute::ZExt;
  
  if (!Ty->isPointerTy())
    // Attributes that only apply to pointers.
    Incompatible |= Attribute::ByVal | Attribute::Nest | Attribute::NoAlias |
      Attribute::StructRet | Attribute::NoCapture;
  
  return Incompatible;
}

//===----------------------------------------------------------------------===//
// AttributeImpl Definition
//===----------------------------------------------------------------------===//

Attributes::Attributes(AttributesImpl *A) : Bits(0) {}

Attributes Attributes::get(LLVMContext &Context, Attributes::Builder &B) {
  // If there are no attributes, return an empty Attributes class.
  if (B.Bits == 0)
    return Attributes();

  // Otherwise, build a key to look up the existing attributes.
  LLVMContextImpl *pImpl = Context.pImpl;
  FoldingSetNodeID ID;
  ID.AddInteger(B.Bits);

  void *InsertPoint;
  AttributesImpl *PA = pImpl->AttrsSet.FindNodeOrInsertPos(ID, InsertPoint);

  if (!PA) {
    // If we didn't find any existing attributes of the same shape then create a
    // new one and insert it.
    PA = new AttributesImpl(B.Bits);
    pImpl->AttrsSet.InsertNode(PA, InsertPoint);
  }

  // Return the AttributesList that we found or created.
  return Attributes(PA);
}

//===----------------------------------------------------------------------===//
// AttributeListImpl Definition
//===----------------------------------------------------------------------===//

namespace llvm {
  class AttributeListImpl;
}

static ManagedStatic<FoldingSet<AttributeListImpl> > AttributesLists;

namespace llvm {
static ManagedStatic<sys::SmartMutex<true> > ALMutex;

class AttributeListImpl : public FoldingSetNode {
  sys::cas_flag RefCount;
  
  // AttributesList is uniqued, these should not be publicly available.
  void operator=(const AttributeListImpl &) LLVM_DELETED_FUNCTION;
  AttributeListImpl(const AttributeListImpl &) LLVM_DELETED_FUNCTION;
  ~AttributeListImpl();                        // Private implementation
public:
  SmallVector<AttributeWithIndex, 4> Attrs;
  
  AttributeListImpl(ArrayRef<AttributeWithIndex> attrs)
    : Attrs(attrs.begin(), attrs.end()) {
    RefCount = 0;
  }
  
  void AddRef() {
    sys::SmartScopedLock<true> Lock(*ALMutex);
    ++RefCount;
  }
  void DropRef() {
    sys::SmartScopedLock<true> Lock(*ALMutex);
    if (!AttributesLists.isConstructed())
      return;
    sys::cas_flag new_val = --RefCount;
    if (new_val == 0)
      delete this;
  }
  
  void Profile(FoldingSetNodeID &ID) const {
    Profile(ID, Attrs);
  }
  static void Profile(FoldingSetNodeID &ID, ArrayRef<AttributeWithIndex> Attrs){
    for (unsigned i = 0, e = Attrs.size(); i != e; ++i) {
      ID.AddInteger(Attrs[i].Attrs.Raw());
      ID.AddInteger(Attrs[i].Index);
    }
  }
};
}

AttributeListImpl::~AttributeListImpl() {
  // NOTE: Lock must be acquired by caller.
  AttributesLists->RemoveNode(this);
}


AttrListPtr AttrListPtr::get(ArrayRef<AttributeWithIndex> Attrs) {
  // If there are no attributes then return a null AttributesList pointer.
  if (Attrs.empty())
    return AttrListPtr();
  
#ifndef NDEBUG
  for (unsigned i = 0, e = Attrs.size(); i != e; ++i) {
    assert(Attrs[i].Attrs.hasAttributes() && 
           "Pointless attribute!");
    assert((!i || Attrs[i-1].Index < Attrs[i].Index) &&
           "Misordered AttributesList!");
  }
#endif
  
  // Otherwise, build a key to look up the existing attributes.
  FoldingSetNodeID ID;
  AttributeListImpl::Profile(ID, Attrs);
  void *InsertPos;
  
  sys::SmartScopedLock<true> Lock(*ALMutex);
  
  AttributeListImpl *PAL =
    AttributesLists->FindNodeOrInsertPos(ID, InsertPos);
  
  // If we didn't find any existing attributes of the same shape then
  // create a new one and insert it.
  if (!PAL) {
    PAL = new AttributeListImpl(Attrs);
    AttributesLists->InsertNode(PAL, InsertPos);
  }
  
  // Return the AttributesList that we found or created.
  return AttrListPtr(PAL);
}


//===----------------------------------------------------------------------===//
// AttrListPtr Method Implementations
//===----------------------------------------------------------------------===//

AttrListPtr::AttrListPtr(AttributeListImpl *LI) : AttrList(LI) {
  if (LI) LI->AddRef();
}

AttrListPtr::AttrListPtr(const AttrListPtr &P) : AttrList(P.AttrList) {
  if (AttrList) AttrList->AddRef();  
}

const AttrListPtr &AttrListPtr::operator=(const AttrListPtr &RHS) {
  sys::SmartScopedLock<true> Lock(*ALMutex);
  if (AttrList == RHS.AttrList) return *this;
  if (AttrList) AttrList->DropRef();
  AttrList = RHS.AttrList;
  if (AttrList) AttrList->AddRef();
  return *this;
}

AttrListPtr::~AttrListPtr() {
  if (AttrList) AttrList->DropRef();
}

/// getNumSlots - Return the number of slots used in this attribute list. 
/// This is the number of arguments that have an attribute set on them
/// (including the function itself).
unsigned AttrListPtr::getNumSlots() const {
  return AttrList ? AttrList->Attrs.size() : 0;
}

/// getSlot - Return the AttributeWithIndex at the specified slot.  This
/// holds a number plus a set of attributes.
const AttributeWithIndex &AttrListPtr::getSlot(unsigned Slot) const {
  assert(AttrList && Slot < AttrList->Attrs.size() && "Slot # out of range!");
  return AttrList->Attrs[Slot];
}


/// getAttributes - The attributes for the specified index are
/// returned.  Attributes for the result are denoted with Idx = 0.
/// Function notes are denoted with idx = ~0.
Attributes AttrListPtr::getAttributes(unsigned Idx) const {
  if (AttrList == 0) return Attributes();
  
  const SmallVector<AttributeWithIndex, 4> &Attrs = AttrList->Attrs;
  for (unsigned i = 0, e = Attrs.size(); i != e && Attrs[i].Index <= Idx; ++i)
    if (Attrs[i].Index == Idx)
      return Attrs[i].Attrs;

  return Attributes();
}

/// hasAttrSomewhere - Return true if the specified attribute is set for at
/// least one parameter or for the return value.
bool AttrListPtr::hasAttrSomewhere(Attributes Attr) const {
  if (AttrList == 0) return false;
  
  const SmallVector<AttributeWithIndex, 4> &Attrs = AttrList->Attrs;
  for (unsigned i = 0, e = Attrs.size(); i != e; ++i)
    if (Attrs[i].Attrs.hasAttributes(Attr))
      return true;
  return false;
}


AttrListPtr AttrListPtr::addAttr(unsigned Idx, Attributes Attrs) const {
  Attributes OldAttrs = getAttributes(Idx);
#ifndef NDEBUG
  // FIXME it is not obvious how this should work for alignment.
  // For now, say we can't change a known alignment.
  unsigned OldAlign = OldAttrs.getAlignment();
  unsigned NewAlign = Attrs.getAlignment();
  assert((!OldAlign || !NewAlign || OldAlign == NewAlign) &&
         "Attempt to change alignment!");
#endif
  
  Attributes NewAttrs = OldAttrs | Attrs;
  if (NewAttrs == OldAttrs)
    return *this;
  
  SmallVector<AttributeWithIndex, 8> NewAttrList;
  if (AttrList == 0)
    NewAttrList.push_back(AttributeWithIndex::get(Idx, Attrs));
  else {
    const SmallVector<AttributeWithIndex, 4> &OldAttrList = AttrList->Attrs;
    unsigned i = 0, e = OldAttrList.size();
    // Copy attributes for arguments before this one.
    for (; i != e && OldAttrList[i].Index < Idx; ++i)
      NewAttrList.push_back(OldAttrList[i]);

    // If there are attributes already at this index, merge them in.
    if (i != e && OldAttrList[i].Index == Idx) {
      Attrs |= OldAttrList[i].Attrs;
      ++i;
    }
    
    NewAttrList.push_back(AttributeWithIndex::get(Idx, Attrs));
    
    // Copy attributes for arguments after this one.
    NewAttrList.insert(NewAttrList.end(), 
                       OldAttrList.begin()+i, OldAttrList.end());
  }
  
  return get(NewAttrList);
}

AttrListPtr AttrListPtr::removeAttr(unsigned Idx, Attributes Attrs) const {
#ifndef NDEBUG
  // FIXME it is not obvious how this should work for alignment.
  // For now, say we can't pass in alignment, which no current use does.
  assert(!Attrs.hasAlignmentAttr() && "Attempt to exclude alignment!");
#endif
  if (AttrList == 0) return AttrListPtr();
  
  Attributes OldAttrs = getAttributes(Idx);
  Attributes NewAttrs = OldAttrs & ~Attrs;
  if (NewAttrs == OldAttrs)
    return *this;

  SmallVector<AttributeWithIndex, 8> NewAttrList;
  const SmallVector<AttributeWithIndex, 4> &OldAttrList = AttrList->Attrs;
  unsigned i = 0, e = OldAttrList.size();
  
  // Copy attributes for arguments before this one.
  for (; i != e && OldAttrList[i].Index < Idx; ++i)
    NewAttrList.push_back(OldAttrList[i]);
  
  // If there are attributes already at this index, merge them in.
  assert(OldAttrList[i].Index == Idx && "Attribute isn't set?");
  Attrs = OldAttrList[i].Attrs & ~Attrs;
  ++i;
  if (Attrs)  // If any attributes left for this parameter, add them.
    NewAttrList.push_back(AttributeWithIndex::get(Idx, Attrs));
  
  // Copy attributes for arguments after this one.
  NewAttrList.insert(NewAttrList.end(), 
                     OldAttrList.begin()+i, OldAttrList.end());
  
  return get(NewAttrList);
}

void AttrListPtr::dump() const {
  dbgs() << "PAL[ ";
  for (unsigned i = 0; i < getNumSlots(); ++i) {
    const AttributeWithIndex &PAWI = getSlot(i);
    dbgs() << "{" << PAWI.Index << "," << PAWI.Attrs << "} ";
  }
  
  dbgs() << "]\n";
}
