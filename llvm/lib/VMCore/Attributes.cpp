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
// Attributes Implementation
//===----------------------------------------------------------------------===//

Attributes::Attributes(uint64_t Val) : Attrs(Val) {}

Attributes::Attributes(Attribute::AttrConst Val) : Attrs(Val.v) {}

Attributes::Attributes(AttributesImpl *A) : Attrs(A->Bits) {}

Attributes::Attributes(const Attributes &A) : Attrs(A.Attrs) {}

// FIXME: This is temporary until we have implemented the uniquified version of
// AttributesImpl.
Attributes Attributes::get(Attributes::Builder &B) {
  return Attributes(B.Bits);
}

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

bool Attributes::hasAttributes(const Attributes &A) const {
  return Attrs.hasAttributes(A);
}
bool Attributes::hasAddressSafetyAttr() const {
  return Attrs.hasAttribute(Attribute::AddressSafety_i);
}
bool Attributes::hasAlignmentAttr() const {
  return Attrs.hasAttribute(Attribute::Alignment_i);
}
bool Attributes::hasAlwaysInlineAttr() const {
  return Attrs.hasAttribute(Attribute::AlwaysInline_i);
}
bool Attributes::hasByValAttr() const {
  return Attrs.hasAttribute(Attribute::ByVal_i);
}
bool Attributes::hasInlineHintAttr() const {
  return Attrs.hasAttribute(Attribute::InlineHint_i);
}
bool Attributes::hasInRegAttr() const {
  return Attrs.hasAttribute(Attribute::InReg_i);
}
bool Attributes::hasNakedAttr() const {
  return Attrs.hasAttribute(Attribute::Naked_i);
}
bool Attributes::hasNestAttr() const {
  return Attrs.hasAttribute(Attribute::Nest_i);
}
bool Attributes::hasNoAliasAttr() const {
  return Attrs.hasAttribute(Attribute::NoAlias_i);
}
bool Attributes::hasNoCaptureAttr() const {
  return Attrs.hasAttribute(Attribute::NoCapture_i);
}
bool Attributes::hasNoImplicitFloatAttr() const {
  return Attrs.hasAttribute(Attribute::NoImplicitFloat_i);
}
bool Attributes::hasNoInlineAttr() const {
  return Attrs.hasAttribute(Attribute::NoInline_i);
}
bool Attributes::hasNonLazyBindAttr() const {
  return Attrs.hasAttribute(Attribute::NonLazyBind_i);
}
bool Attributes::hasNoRedZoneAttr() const {
  return Attrs.hasAttribute(Attribute::NoRedZone_i);
}
bool Attributes::hasNoReturnAttr() const {
  return Attrs.hasAttribute(Attribute::NoReturn_i);
}
bool Attributes::hasNoUnwindAttr() const {
  return Attrs.hasAttribute(Attribute::NoUnwind_i);
}
bool Attributes::hasOptimizeForSizeAttr() const {
  return Attrs.hasAttribute(Attribute::OptimizeForSize_i);
}
bool Attributes::hasReadNoneAttr() const {
  return Attrs.hasAttribute(Attribute::ReadNone_i);
}
bool Attributes::hasReadOnlyAttr() const {
  return Attrs.hasAttribute(Attribute::ReadOnly_i);
}
bool Attributes::hasReturnsTwiceAttr() const {
  return Attrs.hasAttribute(Attribute::ReturnsTwice_i);
}
bool Attributes::hasSExtAttr() const {
  return Attrs.hasAttribute(Attribute::SExt_i);
}
bool Attributes::hasStackAlignmentAttr() const {
  return Attrs.hasAttribute(Attribute::StackAlignment_i);
}
bool Attributes::hasStackProtectAttr() const {
  return Attrs.hasAttribute(Attribute::StackProtect_i);
}
bool Attributes::hasStackProtectReqAttr() const {
  return Attrs.hasAttribute(Attribute::StackProtectReq_i);
}
bool Attributes::hasStructRetAttr() const {
  return Attrs.hasAttribute(Attribute::StructRet_i);
}
bool Attributes::hasUWTableAttr() const {
  return Attrs.hasAttribute(Attribute::UWTable_i);
}
bool Attributes::hasZExtAttr() const {
  return Attrs.hasAttribute(Attribute::ZExt_i);
}

/// This returns the alignment field of an attribute as a byte alignment value.
unsigned Attributes::getAlignment() const {
  if (!hasAlignmentAttr())
    return 0;
  return 1U << ((Attrs.getAlignment() >> 16) - 1);
}

/// This returns the stack alignment field of an attribute as a byte alignment
/// value.
unsigned Attributes::getStackAlignment() const {
  if (!hasStackAlignmentAttr())
    return 0;
  return 1U << ((Attrs.getStackAlignment() >> 26) - 1);
}

bool Attributes::isEmptyOrSingleton() const {
  return Attrs.isEmptyOrSingleton();
}

Attributes Attributes::operator | (const Attributes &A) const {
  return Attributes(Raw() | A.Raw());
}
Attributes Attributes::operator & (const Attributes &A) const {
  return Attributes(Raw() & A.Raw());
}
Attributes Attributes::operator ^ (const Attributes &A) const {
  return Attributes(Raw() ^ A.Raw());
}
Attributes &Attributes::operator |= (const Attributes &A) {
  Attrs.Bits |= A.Raw();
  return *this;
}
Attributes &Attributes::operator &= (const Attributes &A) {
  Attrs.Bits &= A.Raw();
  return *this;
}
Attributes Attributes::operator ~ () const {
  return Attributes(~Raw());
}

uint64_t Attributes::Raw() const {
  return Attrs.Bits;
}

Attributes Attributes::typeIncompatible(Type *Ty) {
  Attributes::Builder Incompatible;
  
  if (!Ty->isIntegerTy()) {
    // Attributes that only apply to integers.
    Incompatible.addSExtAttr();
    Incompatible.addZExtAttr();
  }
  
  if (!Ty->isPointerTy()) {
    // Attributes that only apply to pointers.
    Incompatible.addByValAttr();
    Incompatible.addNestAttr();
    Incompatible.addNoAliasAttr();
    Incompatible.addNoCaptureAttr();
    Incompatible.addStructRetAttr();
  }
  
  return Attributes(Incompatible.Bits); // FIXME: Use Attributes::get().
}

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

//===----------------------------------------------------------------------===//
// Attributes::Builder Implementation
//===----------------------------------------------------------------------===//

void Attributes::Builder::addAddressSafetyAttr() {
  Bits |= Attribute::AddressSafety_i;
}
void Attributes::Builder::addAlwaysInlineAttr() {
  Bits |= Attribute::AlwaysInline_i;
}
void Attributes::Builder::addByValAttr() {
  Bits |= Attribute::ByVal_i;
}
void Attributes::Builder::addInlineHintAttr() {
  Bits |= Attribute::InlineHint_i;
}
void Attributes::Builder::addInRegAttr() {
  Bits |= Attribute::InReg_i;
}
void Attributes::Builder::addNakedAttr() {
  Bits |= Attribute::Naked_i;
}
void Attributes::Builder::addNestAttr() {
  Bits |= Attribute::Nest_i;
}
void Attributes::Builder::addNoAliasAttr() {
  Bits |= Attribute::NoAlias_i;
}
void Attributes::Builder::addNoCaptureAttr() {
  Bits |= Attribute::NoCapture_i;
}
void Attributes::Builder::addNoImplicitFloatAttr() {
  Bits |= Attribute::NoImplicitFloat_i;
}
void Attributes::Builder::addNoInlineAttr() {
  Bits |= Attribute::NoInline_i;
}
void Attributes::Builder::addNonLazyBindAttr() {
  Bits |= Attribute::NonLazyBind_i;
}
void Attributes::Builder::addNoRedZoneAttr() {
  Bits |= Attribute::NoRedZone_i;
}
void Attributes::Builder::addNoReturnAttr() {
  Bits |= Attribute::NoReturn_i;
}
void Attributes::Builder::addNoUnwindAttr() {
  Bits |= Attribute::NoUnwind_i;
}
void Attributes::Builder::addOptimizeForSizeAttr() {
  Bits |= Attribute::OptimizeForSize_i;
}
void Attributes::Builder::addReadNoneAttr() {
  Bits |= Attribute::ReadNone_i;
}
void Attributes::Builder::addReadOnlyAttr() {
  Bits |= Attribute::ReadOnly_i;
}
void Attributes::Builder::addReturnsTwiceAttr() {
  Bits |= Attribute::ReturnsTwice_i;
}
void Attributes::Builder::addSExtAttr() {
  Bits |= Attribute::SExt_i;
}
void Attributes::Builder::addStackProtectAttr() {
  Bits |= Attribute::StackProtect_i;
}
void Attributes::Builder::addStackProtectReqAttr() {
  Bits |= Attribute::StackProtectReq_i;
}
void Attributes::Builder::addStructRetAttr() {
  Bits |= Attribute::StructRet_i;
}
void Attributes::Builder::addUWTableAttr() {
  Bits |= Attribute::UWTable_i;
}
void Attributes::Builder::addZExtAttr() {
  Bits |= Attribute::ZExt_i;
}

void Attributes::Builder::addAlignmentAttr(unsigned Align) {
  if (Align == 0) return;
  assert(isPowerOf2_32(Align) && "Alignment must be a power of two.");
  assert(Align <= 0x40000000 && "Alignment too large.");
  Bits |= (Log2_32(Align) + 1) << 16;
}
void Attributes::Builder::addStackAlignmentAttr(unsigned Align) {
  // Default alignment, allow the target to define how to align it.
  if (Align == 0) return;
  assert(isPowerOf2_32(Align) && "Alignment must be a power of two.");
  assert(Align <= 0x100 && "Alignment too large.");
  Bits |= (Log2_32(Align) + 1) << 26;
}

void Attributes::Builder::removeAttributes(const Attributes &A) {
  Bits &= ~A.Raw();
}

void Attributes::Builder::removeAddressSafetyAttr() {
  Bits &= ~Attribute::AddressSafety_i;
}
void Attributes::Builder::removeAlwaysInlineAttr() {
  Bits &= ~Attribute::AlwaysInline_i;
}
void Attributes::Builder::removeByValAttr() {
  Bits &= ~Attribute::ByVal_i;
}
void Attributes::Builder::removeInlineHintAttr() {
  Bits &= ~Attribute::InlineHint_i;
}
void Attributes::Builder::removeInRegAttr() {
  Bits &= ~Attribute::InReg_i;
}
void Attributes::Builder::removeNakedAttr() {
  Bits &= ~Attribute::Naked_i;
}
void Attributes::Builder::removeNestAttr() {
  Bits &= ~Attribute::Nest_i;
}
void Attributes::Builder::removeNoAliasAttr() {
  Bits &= ~Attribute::NoAlias_i;
}
void Attributes::Builder::removeNoCaptureAttr() {
  Bits &= ~Attribute::NoCapture_i;
}
void Attributes::Builder::removeNoImplicitFloatAttr() {
  Bits &= ~Attribute::NoImplicitFloat_i;
}
void Attributes::Builder::removeNoInlineAttr() {
  Bits &= ~Attribute::NoInline_i;
}
void Attributes::Builder::removeNonLazyBindAttr() {
  Bits &= ~Attribute::NonLazyBind_i;
}
void Attributes::Builder::removeNoRedZoneAttr() {
  Bits &= ~Attribute::NoRedZone_i;
}
void Attributes::Builder::removeNoReturnAttr() {
  Bits &= ~Attribute::NoReturn_i;
}
void Attributes::Builder::removeNoUnwindAttr() {
  Bits &= ~Attribute::NoUnwind_i;
}
void Attributes::Builder::removeOptimizeForSizeAttr() {
  Bits &= ~Attribute::OptimizeForSize_i;
}
void Attributes::Builder::removeReadNoneAttr() {
  Bits &= ~Attribute::ReadNone_i;
}
void Attributes::Builder::removeReadOnlyAttr() {
  Bits &= ~Attribute::ReadOnly_i;
}
void Attributes::Builder::removeReturnsTwiceAttr() {
  Bits &= ~Attribute::ReturnsTwice_i;
}
void Attributes::Builder::removeSExtAttr() {
  Bits &= ~Attribute::SExt_i;
}
void Attributes::Builder::removeStackProtectAttr() {
  Bits &= ~Attribute::StackProtect_i;
}
void Attributes::Builder::removeStackProtectReqAttr() {
  Bits &= ~Attribute::StackProtectReq_i;
}
void Attributes::Builder::removeStructRetAttr() {
  Bits &= ~Attribute::StructRet_i;
}
void Attributes::Builder::removeUWTableAttr() {
  Bits &= ~Attribute::UWTable_i;
}
void Attributes::Builder::removeZExtAttr() {
  Bits &= ~Attribute::ZExt_i;
}

void Attributes::Builder::removeAlignmentAttr() {
  Bits &= ~Attribute::Alignment_i;
}
void Attributes::Builder::removeStackAlignmentAttr() {
  Bits &= ~Attribute::StackAlignment_i;
}

bool Attributes::Builder::hasAttributes() const {
  return Bits != 0;
}
bool Attributes::Builder::hasAttributes(const Attributes &A) const {
  return Bits & A.Raw();
}
bool Attributes::Builder::hasAlignmentAttr() const {
  return Bits & Attribute::Alignment_i;
}

uint64_t Attributes::Builder::getAlignment() const {
  if (!hasAlignmentAttr())
    return 0;
  return 1U << (((Bits & Attribute::Alignment_i) >> 16) - 1);
}

//===----------------------------------------------------------------------===//
// AttributeImpl Definition
//===----------------------------------------------------------------------===//

bool AttributesImpl::hasAttribute(uint64_t A) const {
  return (Bits & A) != 0;
}

bool AttributesImpl::hasAttributes() const {
  return Bits != 0;
}

bool AttributesImpl::hasAttributes(const Attributes &A) const {
  return Bits & A.Raw();        // FIXME: Raw() won't work here in the future.
}

uint64_t AttributesImpl::getAlignment() const {
  return Bits & Attribute::Alignment_i;
}

uint64_t AttributesImpl::getStackAlignment() const {
  return Bits & Attribute::StackAlignment_i;
}

bool AttributesImpl::isEmptyOrSingleton() const {
  return (Bits & (Bits - 1)) == 0;
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

unsigned AttrListPtr::getNumAttrs() const {
  return AttrList ? AttrList->Attrs.size() : 0;
}

Attributes &AttrListPtr::getAttributesAtIndex(unsigned i) const {
  assert(AttrList && "Trying to get an attribute from an empty list!");
  assert(i < AttrList->Attrs.size() && "Index out of range!");
  return AttrList->Attrs[i].Attrs;
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
