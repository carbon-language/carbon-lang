//===-- Attributes.cpp - Implement AttributesList -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// \file
// \brief This file implements the Attribute, AttributeImpl, AttrBuilder,
// AttributeSetImpl, and AttributeSet classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Attributes.h"
#include "AttributeImpl.h"
#include "LLVMContextImpl.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Atomic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
using namespace llvm;

//===----------------------------------------------------------------------===//
// Attribute Construction Methods
//===----------------------------------------------------------------------===//

Attribute Attribute::get(LLVMContext &Context, Constant *Kind, Constant *Val) {
  LLVMContextImpl *pImpl = Context.pImpl;
  FoldingSetNodeID ID;
  ID.AddPointer(Kind);
  if (Val) ID.AddPointer(Val);

  void *InsertPoint;
  AttributeImpl *PA = pImpl->AttrsSet.FindNodeOrInsertPos(ID, InsertPoint);

  if (!PA) {
    // If we didn't find any existing attributes of the same shape then create a
    // new one and insert it.
    PA = new AttributeImpl(Context, Kind, Val);
    pImpl->AttrsSet.InsertNode(PA, InsertPoint);
  }

  // Return the AttributesList that we found or created.
  return Attribute(PA);
}

Attribute Attribute::get(LLVMContext &Context, AttrKind Kind, Constant *Val) {
  ConstantInt *KindVal = ConstantInt::get(Type::getInt64Ty(Context), Kind);
  return get(Context, KindVal, Val);
}

Attribute Attribute::getWithAlignment(LLVMContext &Context, uint64_t Align) {
  assert(isPowerOf2_32(Align) && "Alignment must be a power of two.");
  assert(Align <= 0x40000000 && "Alignment too large.");
  return get(Context, Alignment,
             ConstantInt::get(Type::getInt64Ty(Context), Align));
}

Attribute Attribute::getWithStackAlignment(LLVMContext &Context,
                                           uint64_t Align) {
  assert(isPowerOf2_32(Align) && "Alignment must be a power of two.");
  assert(Align <= 0x100 && "Alignment too large.");
  return get(Context, StackAlignment,
             ConstantInt::get(Type::getInt64Ty(Context), Align));
}

//===----------------------------------------------------------------------===//
// Attribute Accessor Methods
//===----------------------------------------------------------------------===//

bool Attribute::hasAttribute(AttrKind Val) const {
  return pImpl && pImpl->hasAttribute(Val);
}

Constant *Attribute::getAttributeKind() const {
  return pImpl ? pImpl->getAttributeKind() : 0;
}

Constant *Attribute::getAttributeValues() const {
  return pImpl ? pImpl->getAttributeValues() : 0;
}

/// This returns the alignment field of an attribute as a byte alignment value.
unsigned Attribute::getAlignment() const {
  assert(hasAttribute(Attribute::Alignment) &&
         "Trying to get alignment from non-alignment attribute!");
  return pImpl->getAlignment();
}

/// This returns the stack alignment field of an attribute as a byte alignment
/// value.
unsigned Attribute::getStackAlignment() const {
  assert(hasAttribute(Attribute::StackAlignment) &&
         "Trying to get alignment from non-alignment attribute!");
  return pImpl->getStackAlignment();
}

std::string Attribute::getAsString() const {
  if (!pImpl) return "";

  if (hasAttribute(Attribute::AddressSafety))
    return "address_safety";
  if (hasAttribute(Attribute::AlwaysInline))
    return "alwaysinline";
  if (hasAttribute(Attribute::ByVal))
    return "byval";
  if (hasAttribute(Attribute::InlineHint))
    return "inlinehint";
  if (hasAttribute(Attribute::InReg))
    return "inreg";
  if (hasAttribute(Attribute::MinSize))
    return "minsize";
  if (hasAttribute(Attribute::Naked))
    return "naked";
  if (hasAttribute(Attribute::Nest))
    return "nest";
  if (hasAttribute(Attribute::NoAlias))
    return "noalias";
  if (hasAttribute(Attribute::NoCapture))
    return "nocapture";
  if (hasAttribute(Attribute::NoDuplicate))
    return "noduplicate";
  if (hasAttribute(Attribute::NoImplicitFloat))
    return "noimplicitfloat";
  if (hasAttribute(Attribute::NoInline))
    return "noinline";
  if (hasAttribute(Attribute::NonLazyBind))
    return "nonlazybind";
  if (hasAttribute(Attribute::NoRedZone))
    return "noredzone";
  if (hasAttribute(Attribute::NoReturn))
    return "noreturn";
  if (hasAttribute(Attribute::NoUnwind))
    return "nounwind";
  if (hasAttribute(Attribute::OptimizeForSize))
    return "optsize";
  if (hasAttribute(Attribute::ReadNone))
    return "readnone";
  if (hasAttribute(Attribute::ReadOnly))
    return "readonly";
  if (hasAttribute(Attribute::ReturnsTwice))
    return "returns_twice";
  if (hasAttribute(Attribute::SExt))
    return "signext";
  if (hasAttribute(Attribute::StackProtect))
    return "ssp";
  if (hasAttribute(Attribute::StackProtectReq))
    return "sspreq";
  if (hasAttribute(Attribute::StackProtectStrong))
    return "sspstrong";
  if (hasAttribute(Attribute::StructRet))
    return "sret";
  if (hasAttribute(Attribute::UWTable))
    return "uwtable";
  if (hasAttribute(Attribute::ZExt))
    return "zeroext";

  // FIXME: These should be output like this:
  //
  //   align=4
  //   alignstack=8
  //
  if (hasAttribute(Attribute::StackAlignment)) {
    std::string Result;
    Result += "alignstack(";
    Result += utostr(getStackAlignment());
    Result += ")";
    return Result;
  }
  if (hasAttribute(Attribute::Alignment)) {
    std::string Result;
    Result += "align ";
    Result += utostr(getAlignment());
    return Result;
  }

  // Convert target-dependent attributes to strings of the form:
  //
  //   "kind"
  //   "kind" = "value"
  //   "kind" = ( "value1" "value2" "value3" )
  //
  if (ConstantDataArray *CDA =
      dyn_cast<ConstantDataArray>(pImpl->getAttributeKind())) {
    std::string Result;
    Result += '\"' + CDA->getAsString().str() + '"';

    Constant *Vals = pImpl->getAttributeValues();
    if (!Vals) return Result;

    // FIXME: This should support more than just ConstantDataArrays. Also,
    // support a vector of attribute values.

    Result += " = ";
    Result += '\"' + cast<ConstantDataArray>(Vals)->getAsString().str() + '"';

    return Result;
  }

  llvm_unreachable("Unknown attribute");
}

bool Attribute::operator==(AttrKind K) const {
  return (pImpl && *pImpl == K) || (!pImpl && K == None);
}
bool Attribute::operator!=(AttrKind K) const {
  return !(*this == K);
}

bool Attribute::operator<(Attribute A) const {
  if (!pImpl && !A.pImpl) return false;
  if (!pImpl) return true;
  if (!A.pImpl) return false;
  return *pImpl < *A.pImpl;
}

//===----------------------------------------------------------------------===//
// AttributeImpl Definition
//===----------------------------------------------------------------------===//

bool AttributeImpl::hasAttribute(Attribute::AttrKind A) const {
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Kind))
    return CI->getZExtValue() == A;
  return false;
}

uint64_t AttributeImpl::getAlignment() const {
  assert(hasAttribute(Attribute::Alignment) &&
         "Trying to retrieve the alignment from a non-alignment attr!");
  return cast<ConstantInt>(Values)->getZExtValue();
}

uint64_t AttributeImpl::getStackAlignment() const {
  assert(hasAttribute(Attribute::StackAlignment) &&
         "Trying to retrieve the stack alignment from a non-alignment attr!");
  return cast<ConstantInt>(Values)->getZExtValue();
}

bool AttributeImpl::operator==(Attribute::AttrKind kind) const {
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Kind))
    return CI->getZExtValue() == kind;
  return false;
}
bool AttributeImpl::operator!=(Attribute::AttrKind kind) const {
  return !(*this == kind);
}

bool AttributeImpl::operator==(StringRef kind) const {
  if (ConstantDataArray *CDA = dyn_cast<ConstantDataArray>(Kind))
    if (CDA->isString())
      return CDA->getAsString() == kind;
  return false;
}

bool AttributeImpl::operator!=(StringRef kind) const {
  return !(*this == kind);
}

bool AttributeImpl::operator<(const AttributeImpl &AI) const {
  // This sorts the attributes with Attribute::AttrKinds coming first (sorted
  // relative to their enum value) and then strings.

  if (!Kind && !AI.Kind) return false;
  if (!Kind && AI.Kind) return true;
  if (Kind && !AI.Kind) return false;

  ConstantInt *ThisCI = dyn_cast<ConstantInt>(Kind);
  ConstantInt *ThatCI = dyn_cast<ConstantInt>(AI.Kind);

  ConstantDataArray *ThisCDA = dyn_cast<ConstantDataArray>(Kind);
  ConstantDataArray *ThatCDA = dyn_cast<ConstantDataArray>(AI.Kind);

  if (ThisCI && ThatCI)
    return ThisCI->getZExtValue() < ThatCI->getZExtValue();

  if (ThisCI && ThatCDA)
    return true;

  if (ThisCDA && ThatCI)
    return false;

  return ThisCDA->getAsString() < ThatCDA->getAsString();
}

uint64_t AttributeImpl::getAttrMask(Attribute::AttrKind Val) {
  // FIXME: Remove this.
  switch (Val) {
  case Attribute::EndAttrKinds:
  case Attribute::AttrKindEmptyKey:
  case Attribute::AttrKindTombstoneKey:
    llvm_unreachable("Synthetic enumerators which should never get here");

  case Attribute::None:            return 0;
  case Attribute::ZExt:            return 1 << 0;
  case Attribute::SExt:            return 1 << 1;
  case Attribute::NoReturn:        return 1 << 2;
  case Attribute::InReg:           return 1 << 3;
  case Attribute::StructRet:       return 1 << 4;
  case Attribute::NoUnwind:        return 1 << 5;
  case Attribute::NoAlias:         return 1 << 6;
  case Attribute::ByVal:           return 1 << 7;
  case Attribute::Nest:            return 1 << 8;
  case Attribute::ReadNone:        return 1 << 9;
  case Attribute::ReadOnly:        return 1 << 10;
  case Attribute::NoInline:        return 1 << 11;
  case Attribute::AlwaysInline:    return 1 << 12;
  case Attribute::OptimizeForSize: return 1 << 13;
  case Attribute::StackProtect:    return 1 << 14;
  case Attribute::StackProtectReq: return 1 << 15;
  case Attribute::Alignment:       return 31 << 16;
  case Attribute::NoCapture:       return 1 << 21;
  case Attribute::NoRedZone:       return 1 << 22;
  case Attribute::NoImplicitFloat: return 1 << 23;
  case Attribute::Naked:           return 1 << 24;
  case Attribute::InlineHint:      return 1 << 25;
  case Attribute::StackAlignment:  return 7 << 26;
  case Attribute::ReturnsTwice:    return 1 << 29;
  case Attribute::UWTable:         return 1 << 30;
  case Attribute::NonLazyBind:     return 1U << 31;
  case Attribute::AddressSafety:   return 1ULL << 32;
  case Attribute::MinSize:         return 1ULL << 33;
  case Attribute::NoDuplicate:     return 1ULL << 34;
  case Attribute::StackProtectStrong: return 1ULL << 35;
  }
  llvm_unreachable("Unsupported attribute type");
}

//===----------------------------------------------------------------------===//
// AttributeSetNode Definition
//===----------------------------------------------------------------------===//

AttributeSetNode *AttributeSetNode::get(LLVMContext &C,
                                        ArrayRef<Attribute> Attrs) {
  if (Attrs.empty())
    return 0;

  // Otherwise, build a key to look up the existing attributes.
  LLVMContextImpl *pImpl = C.pImpl;
  FoldingSetNodeID ID;

  SmallVector<Attribute, 8> SortedAttrs(Attrs.begin(), Attrs.end());
  std::sort(SortedAttrs.begin(), SortedAttrs.end());

  for (SmallVectorImpl<Attribute>::iterator I = SortedAttrs.begin(),
         E = SortedAttrs.end(); I != E; ++I)
    I->Profile(ID);

  void *InsertPoint;
  AttributeSetNode *PA =
    pImpl->AttrsSetNodes.FindNodeOrInsertPos(ID, InsertPoint);

  // If we didn't find any existing attributes of the same shape then create a
  // new one and insert it.
  if (!PA) {
    PA = new AttributeSetNode(SortedAttrs);
    pImpl->AttrsSetNodes.InsertNode(PA, InsertPoint);
  }

  // Return the AttributesListNode that we found or created.
  return PA;
}

bool AttributeSetNode::hasAttribute(Attribute::AttrKind Kind) const {
  for (SmallVectorImpl<Attribute>::const_iterator I = AttrList.begin(),
         E = AttrList.end(); I != E; ++I)
    if (I->hasAttribute(Kind))
      return true;
  return false;
}

unsigned AttributeSetNode::getAlignment() const {
  for (SmallVectorImpl<Attribute>::const_iterator I = AttrList.begin(),
         E = AttrList.end(); I != E; ++I)
    if (I->hasAttribute(Attribute::Alignment))
      return I->getAlignment();
  return 0;
}

unsigned AttributeSetNode::getStackAlignment() const {
  for (SmallVectorImpl<Attribute>::const_iterator I = AttrList.begin(),
         E = AttrList.end(); I != E; ++I)
    if (I->hasAttribute(Attribute::StackAlignment))
      return I->getStackAlignment();
  return 0;
}

std::string AttributeSetNode::getAsString() const {
  std::string Str = "";
  for (SmallVectorImpl<Attribute>::const_iterator I = AttrList.begin(),
         E = AttrList.end(); I != E; ) {
    Str += I->getAsString();
    if (++I != E) Str += " ";
  }
  return Str;
}

//===----------------------------------------------------------------------===//
// AttributeSetImpl Definition
//===----------------------------------------------------------------------===//

uint64_t AttributeSetImpl::Raw(uint64_t Index) const {
  for (unsigned I = 0, E = getNumAttributes(); I != E; ++I) {
    if (getSlotIndex(I) != Index) continue;
    const AttributeSetNode *ASN = AttrNodes[I].second;
    uint64_t Mask = 0;

    for (AttributeSetNode::const_iterator II = ASN->begin(),
           IE = ASN->end(); II != IE; ++II) {
      Attribute Attr = *II;
      ConstantInt *Kind = cast<ConstantInt>(Attr.getAttributeKind());
      Attribute::AttrKind KindVal = Attribute::AttrKind(Kind->getZExtValue());

      if (KindVal == Attribute::Alignment)
        Mask |= (Log2_32(ASN->getAlignment()) + 1) << 16;
      else if (KindVal == Attribute::StackAlignment)
        Mask |= (Log2_32(ASN->getStackAlignment()) + 1) << 26;
      else
        Mask |= AttributeImpl::getAttrMask(KindVal);
    }

    return Mask;
  }

  return 0;
}

//===----------------------------------------------------------------------===//
// AttributeSet Construction and Mutation Methods
//===----------------------------------------------------------------------===//

AttributeSet
AttributeSet::getImpl(LLVMContext &C,
                      ArrayRef<std::pair<unsigned, AttributeSetNode*> > Attrs) {
  LLVMContextImpl *pImpl = C.pImpl;
  FoldingSetNodeID ID;
  AttributeSetImpl::Profile(ID, Attrs);

  void *InsertPoint;
  AttributeSetImpl *PA = pImpl->AttrsLists.FindNodeOrInsertPos(ID, InsertPoint);

  // If we didn't find any existing attributes of the same shape then
  // create a new one and insert it.
  if (!PA) {
    PA = new AttributeSetImpl(C, Attrs);
    pImpl->AttrsLists.InsertNode(PA, InsertPoint);
  }

  // Return the AttributesList that we found or created.
  return AttributeSet(PA);
}

AttributeSet AttributeSet::get(LLVMContext &C,
                               ArrayRef<std::pair<unsigned, Attribute> > Attrs){
  // If there are no attributes then return a null AttributesList pointer.
  if (Attrs.empty())
    return AttributeSet();

#ifndef NDEBUG
  for (unsigned i = 0, e = Attrs.size(); i != e; ++i) {
    assert((!i || Attrs[i-1].first <= Attrs[i].first) &&
           "Misordered Attributes list!");
    assert(Attrs[i].second != Attribute::None &&
           "Pointless attribute!");
  }
#endif

  // Create a vector if (unsigned, AttributeSetNode*) pairs from the attributes
  // list.
  SmallVector<std::pair<unsigned, AttributeSetNode*>, 8> AttrPairVec;
  for (ArrayRef<std::pair<unsigned, Attribute> >::iterator I = Attrs.begin(),
         E = Attrs.end(); I != E; ) {
    unsigned Index = I->first;
    SmallVector<Attribute, 4> AttrVec;
    while (I != E && I->first == Index) {
      AttrVec.push_back(I->second);
      ++I;
    }

    AttrPairVec.push_back(std::make_pair(Index,
                                         AttributeSetNode::get(C, AttrVec)));
  }

  return getImpl(C, AttrPairVec);
}

AttributeSet AttributeSet::get(LLVMContext &C,
                               ArrayRef<std::pair<unsigned,
                                                  AttributeSetNode*> > Attrs) {
  // If there are no attributes then return a null AttributesList pointer.
  if (Attrs.empty())
    return AttributeSet();

  return getImpl(C, Attrs);
}

AttributeSet AttributeSet::get(LLVMContext &C, unsigned Idx, AttrBuilder &B) {
  if (!B.hasAttributes())
    return AttributeSet();

  SmallVector<std::pair<unsigned, Attribute>, 8> Attrs;
  for (AttrBuilder::iterator I = B.begin(), E = B.end(); I != E; ++I) {
    Attribute::AttrKind Kind = *I;
    if (Kind == Attribute::Alignment)
      Attrs.push_back(std::make_pair(Idx, Attribute::
                                     getWithAlignment(C, B.getAlignment())));
    else if (Kind == Attribute::StackAlignment)
      Attrs.push_back(std::make_pair(Idx, Attribute::
                              getWithStackAlignment(C, B.getStackAlignment())));
    else
      Attrs.push_back(std::make_pair(Idx, Attribute::get(C, Kind)));
  }

  return get(C, Attrs);
}

AttributeSet AttributeSet::get(LLVMContext &C, unsigned Idx,
                               ArrayRef<Attribute::AttrKind> Kind) {
  SmallVector<std::pair<unsigned, Attribute>, 8> Attrs;
  for (ArrayRef<Attribute::AttrKind>::iterator I = Kind.begin(),
         E = Kind.end(); I != E; ++I)
    Attrs.push_back(std::make_pair(Idx, Attribute::get(C, *I)));
  return get(C, Attrs);
}

AttributeSet AttributeSet::get(LLVMContext &C, ArrayRef<AttributeSet> Attrs) {
  if (Attrs.empty()) return AttributeSet();

  SmallVector<std::pair<unsigned, AttributeSetNode*>, 8> AttrNodeVec;
  for (unsigned I = 0, E = Attrs.size(); I != E; ++I) {
    AttributeSet AS = Attrs[I];
    if (!AS.pImpl) continue;
    AttrNodeVec.append(AS.pImpl->AttrNodes.begin(), AS.pImpl->AttrNodes.end());
  }

  return getImpl(C, AttrNodeVec);
}

AttributeSet AttributeSet::addAttribute(LLVMContext &C, unsigned Idx,
                                        Attribute::AttrKind Attr) const {
  return addAttributes(C, Idx, AttributeSet::get(C, Idx, Attr));
}

AttributeSet AttributeSet::addAttributes(LLVMContext &C, unsigned Idx,
                                         AttributeSet Attrs) const {
  if (!pImpl) return Attrs;
  if (!Attrs.pImpl) return *this;

#ifndef NDEBUG
  // FIXME it is not obvious how this should work for alignment. For now, say
  // we can't change a known alignment.
  unsigned OldAlign = getParamAlignment(Idx);
  unsigned NewAlign = Attrs.getParamAlignment(Idx);
  assert((!OldAlign || !NewAlign || OldAlign == NewAlign) &&
         "Attempt to change alignment!");
#endif

  // Add the attribute slots before the one we're trying to add.
  SmallVector<AttributeSet, 4> AttrSet;
  uint64_t NumAttrs = pImpl->getNumAttributes();
  AttributeSet AS;
  uint64_t LastIndex = 0;
  for (unsigned I = 0, E = NumAttrs; I != E; ++I) {
    if (getSlotIndex(I) >= Idx) {
      if (getSlotIndex(I) == Idx) AS = getSlotAttributes(LastIndex++);
      break;
    }
    LastIndex = I + 1;
    AttrSet.push_back(getSlotAttributes(I));
  }

  // Now add the attribute into the correct slot. There may already be an
  // AttributeSet there.
  AttrBuilder B(AS, Idx);

  for (unsigned I = 0, E = Attrs.pImpl->getNumAttributes(); I != E; ++I)
    if (Attrs.getSlotIndex(I) == Idx) {
      for (AttributeSetImpl::const_iterator II = Attrs.pImpl->begin(I),
             IE = Attrs.pImpl->end(I); II != IE; ++II)
        B.addAttribute(*II);
      break;
    }

  AttrSet.push_back(AttributeSet::get(C, Idx, B));

  // Add the remaining attribute slots.
  for (unsigned I = LastIndex, E = NumAttrs; I < E; ++I)
    AttrSet.push_back(getSlotAttributes(I));

  return get(C, AttrSet);
}

AttributeSet AttributeSet::removeAttribute(LLVMContext &C, unsigned Idx,
                                           Attribute::AttrKind Attr) const {
  return removeAttributes(C, Idx, AttributeSet::get(C, Idx, Attr));
}

AttributeSet AttributeSet::removeAttributes(LLVMContext &C, unsigned Idx,
                                            AttributeSet Attrs) const {
  if (!pImpl) return AttributeSet();
  if (!Attrs.pImpl) return *this;

#ifndef NDEBUG
  // FIXME it is not obvious how this should work for alignment.
  // For now, say we can't pass in alignment, which no current use does.
  assert(!Attrs.hasAttribute(Idx, Attribute::Alignment) &&
         "Attempt to change alignment!");
#endif

  // Add the attribute slots before the one we're trying to add.
  SmallVector<AttributeSet, 4> AttrSet;
  uint64_t NumAttrs = pImpl->getNumAttributes();
  AttributeSet AS;
  uint64_t LastIndex = 0;
  for (unsigned I = 0, E = NumAttrs; I != E; ++I) {
    if (getSlotIndex(I) >= Idx) {
      if (getSlotIndex(I) == Idx) AS = getSlotAttributes(LastIndex++);
      break;
    }
    LastIndex = I + 1;
    AttrSet.push_back(getSlotAttributes(I));
  }

  // Now remove the attribute from the correct slot. There may already be an
  // AttributeSet there.
  AttrBuilder B(AS, Idx);

  for (unsigned I = 0, E = Attrs.pImpl->getNumAttributes(); I != E; ++I)
    if (Attrs.getSlotIndex(I) == Idx) {
      B.removeAttributes(Attrs.pImpl->getSlotAttributes(I), Idx);
      break;
    }

  AttrSet.push_back(AttributeSet::get(C, Idx, B));

  // Add the remaining attribute slots.
  for (unsigned I = LastIndex, E = NumAttrs; I < E; ++I)
    AttrSet.push_back(getSlotAttributes(I));

  return get(C, AttrSet);
}

//===----------------------------------------------------------------------===//
// AttributeSet Accessor Methods
//===----------------------------------------------------------------------===//

AttributeSet AttributeSet::getParamAttributes(unsigned Idx) const {
  return pImpl && hasAttributes(Idx) ?
    AttributeSet::get(pImpl->getContext(),
                      ArrayRef<std::pair<unsigned, AttributeSetNode*> >(
                        std::make_pair(Idx, getAttributes(Idx)))) :
    AttributeSet();
}

AttributeSet AttributeSet::getRetAttributes() const {
  return pImpl && hasAttributes(ReturnIndex) ?
    AttributeSet::get(pImpl->getContext(),
                      ArrayRef<std::pair<unsigned, AttributeSetNode*> >(
                        std::make_pair(ReturnIndex,
                                       getAttributes(ReturnIndex)))) :
    AttributeSet();
}

AttributeSet AttributeSet::getFnAttributes() const {
  return pImpl && hasAttributes(FunctionIndex) ?
    AttributeSet::get(pImpl->getContext(),
                      ArrayRef<std::pair<unsigned, AttributeSetNode*> >(
                        std::make_pair(FunctionIndex,
                                       getAttributes(FunctionIndex)))) :
    AttributeSet();
}

bool AttributeSet::hasAttribute(unsigned Index, Attribute::AttrKind Kind) const{
  AttributeSetNode *ASN = getAttributes(Index);
  return ASN ? ASN->hasAttribute(Kind) : false;
}

bool AttributeSet::hasAttributes(unsigned Index) const {
  AttributeSetNode *ASN = getAttributes(Index);
  return ASN ? ASN->hasAttributes() : false;
}

/// \brief Return true if the specified attribute is set for at least one
/// parameter or for the return value.
bool AttributeSet::hasAttrSomewhere(Attribute::AttrKind Attr) const {
  if (pImpl == 0) return false;

  for (unsigned I = 0, E = pImpl->getNumAttributes(); I != E; ++I)
    for (AttributeSetImpl::const_iterator II = pImpl->begin(I),
           IE = pImpl->end(I); II != IE; ++II)
      if (II->hasAttribute(Attr))
        return true;

  return false;
}

unsigned AttributeSet::getParamAlignment(unsigned Index) const {
  AttributeSetNode *ASN = getAttributes(Index);
  return ASN ? ASN->getAlignment() : 0;
}

unsigned AttributeSet::getStackAlignment(unsigned Index) const {
  AttributeSetNode *ASN = getAttributes(Index);
  return ASN ? ASN->getStackAlignment() : 0;
}

std::string AttributeSet::getAsString(unsigned Index) const {
  AttributeSetNode *ASN = getAttributes(Index);
  return ASN ? ASN->getAsString() : std::string("");
}

/// \brief The attributes for the specified index are returned.
AttributeSetNode *AttributeSet::getAttributes(unsigned Idx) const {
  if (!pImpl) return 0;

  // Loop through to find the attribute node we want.
  for (unsigned I = 0, E = pImpl->getNumAttributes(); I != E; ++I)
    if (pImpl->getSlotIndex(I) == Idx)
      return pImpl->getSlotNode(I);

  return 0;
}

AttributeSet::iterator AttributeSet::begin(unsigned Idx) const {
  if (!pImpl)
    return ArrayRef<Attribute>().begin();
  return pImpl->begin(Idx);
}

AttributeSet::iterator AttributeSet::end(unsigned Idx) const {
  if (!pImpl)
    return ArrayRef<Attribute>().end();
  return pImpl->end(Idx);
}

//===----------------------------------------------------------------------===//
// AttributeSet Introspection Methods
//===----------------------------------------------------------------------===//

/// \brief Return the number of slots used in this attribute list.  This is the
/// number of arguments that have an attribute set on them (including the
/// function itself).
unsigned AttributeSet::getNumSlots() const {
  return pImpl ? pImpl->getNumAttributes() : 0;
}

uint64_t AttributeSet::getSlotIndex(unsigned Slot) const {
  assert(pImpl && Slot < pImpl->getNumAttributes() &&
         "Slot # out of range!");
  return pImpl->getSlotIndex(Slot);
}

AttributeSet AttributeSet::getSlotAttributes(unsigned Slot) const {
  assert(pImpl && Slot < pImpl->getNumAttributes() &&
         "Slot # out of range!");
  return pImpl->getSlotAttributes(Slot);
}

uint64_t AttributeSet::Raw(unsigned Index) const {
  // FIXME: Remove this.
  return pImpl ? pImpl->Raw(Index) : 0;
}

void AttributeSet::dump() const {
  dbgs() << "PAL[\n";

  for (unsigned i = 0, e = getNumSlots(); i < e; ++i) {
    uint64_t Index = getSlotIndex(i);
    dbgs() << "  { ";
    if (Index == ~0U)
      dbgs() << "~0U";
    else
      dbgs() << Index;
    dbgs() << " => " << getAsString(Index) << " }\n";
  }

  dbgs() << "]\n";
}

//===----------------------------------------------------------------------===//
// AttrBuilder Method Implementations
//===----------------------------------------------------------------------===//

AttrBuilder::AttrBuilder(AttributeSet AS, unsigned Idx)
  : Alignment(0), StackAlignment(0) {
  AttributeSetImpl *pImpl = AS.pImpl;
  if (!pImpl) return;

  for (unsigned I = 0, E = pImpl->getNumAttributes(); I != E; ++I) {
    if (pImpl->getSlotIndex(I) != Idx) continue;

    for (AttributeSetImpl::const_iterator II = pImpl->begin(I),
           IE = pImpl->end(I); II != IE; ++II)
      addAttribute(*II);

    break;
  }
}

void AttrBuilder::clear() {
  Attrs.clear();
  Alignment = StackAlignment = 0;
}

AttrBuilder &AttrBuilder::addAttribute(Attribute::AttrKind Val) {
  assert(Val != Attribute::Alignment && Val != Attribute::StackAlignment &&
         "Adding alignment attribute without adding alignment value!");
  Attrs.insert(Val);
  return *this;
}

AttrBuilder &AttrBuilder::addAttribute(Attribute Attr) {
  ConstantInt *Kind = cast<ConstantInt>(Attr.getAttributeKind());
  Attribute::AttrKind KindVal = Attribute::AttrKind(Kind->getZExtValue());
  Attrs.insert(KindVal);

  if (KindVal == Attribute::Alignment)
    Alignment = Attr.getAlignment();
  else if (KindVal == Attribute::StackAlignment)
    StackAlignment = Attr.getStackAlignment();
  return *this;
}

AttrBuilder &AttrBuilder::removeAttribute(Attribute::AttrKind Val) {
  Attrs.erase(Val);

  if (Val == Attribute::Alignment)
    Alignment = 0;
  else if (Val == Attribute::StackAlignment)
    StackAlignment = 0;

  return *this;
}

AttrBuilder &AttrBuilder::removeAttributes(AttributeSet A, uint64_t Index) {
  unsigned Idx = ~0U;
  for (unsigned I = 0, E = A.getNumSlots(); I != E; ++I)
    if (A.getSlotIndex(I) == Index) {
      Idx = I;
      break;
    }

  assert(Idx != ~0U && "Couldn't find index in AttributeSet!");

  for (AttributeSet::iterator I = A.begin(Idx), E = A.end(Idx); I != E; ++I) {
    ConstantInt *CI = cast<ConstantInt>(I->getAttributeKind());
    Attribute::AttrKind Kind = Attribute::AttrKind(CI->getZExtValue());
    Attrs.erase(Kind);

    if (Kind == Attribute::Alignment)
      Alignment = 0;
    else if (Kind == Attribute::StackAlignment)
      StackAlignment = 0;
  }

  return *this;
}

AttrBuilder &AttrBuilder::addAlignmentAttr(unsigned Align) {
  if (Align == 0) return *this;

  assert(isPowerOf2_32(Align) && "Alignment must be a power of two.");
  assert(Align <= 0x40000000 && "Alignment too large.");

  Attrs.insert(Attribute::Alignment);
  Alignment = Align;
  return *this;
}

AttrBuilder &AttrBuilder::addStackAlignmentAttr(unsigned Align) {
  // Default alignment, allow the target to define how to align it.
  if (Align == 0) return *this;

  assert(isPowerOf2_32(Align) && "Alignment must be a power of two.");
  assert(Align <= 0x100 && "Alignment too large.");

  Attrs.insert(Attribute::StackAlignment);
  StackAlignment = Align;
  return *this;
}

bool AttrBuilder::contains(Attribute::AttrKind A) const {
  return Attrs.count(A);
}

bool AttrBuilder::hasAttributes() const {
  return !Attrs.empty();
}

bool AttrBuilder::hasAttributes(AttributeSet A, uint64_t Index) const {
  unsigned Idx = ~0U;
  for (unsigned I = 0, E = A.getNumSlots(); I != E; ++I)
    if (A.getSlotIndex(I) == Index) {
      Idx = I;
      break;
    }

  assert(Idx != ~0U && "Couldn't find the index!");

  for (AttributeSet::iterator I = A.begin(Idx), E = A.end(Idx);
       I != E; ++I) {
    Attribute Attr = *I;
    // FIXME: Support StringRefs.
    ConstantInt *Kind = cast<ConstantInt>(Attr.getAttributeKind());
    Attribute::AttrKind KindVal = Attribute::AttrKind(Kind->getZExtValue());

    if (Attrs.count(KindVal))
      return true;
  }

  return false;
}

bool AttrBuilder::hasAlignmentAttr() const {
  return Alignment != 0;
}

bool AttrBuilder::operator==(const AttrBuilder &B) {
  SmallVector<Attribute::AttrKind, 8> This(Attrs.begin(), Attrs.end());
  SmallVector<Attribute::AttrKind, 8> That(B.Attrs.begin(), B.Attrs.end());
  return This == That;
}

AttrBuilder &AttrBuilder::addRawValue(uint64_t Val) {
  if (!Val) return *this;

  for (Attribute::AttrKind I = Attribute::None; I != Attribute::EndAttrKinds;
       I = Attribute::AttrKind(I + 1)) {
    if (uint64_t A = (Val & AttributeImpl::getAttrMask(I))) {
      Attrs.insert(I);
 
      if (I == Attribute::Alignment)
        Alignment = 1ULL << ((A >> 16) - 1);
      else if (I == Attribute::StackAlignment)
        StackAlignment = 1ULL << ((A >> 26)-1);
    }
  }
 
  return *this;
}

//===----------------------------------------------------------------------===//
// AttributeFuncs Function Defintions
//===----------------------------------------------------------------------===//

/// \brief Which attributes cannot be applied to a type.
AttributeSet AttributeFuncs::typeIncompatible(Type *Ty, uint64_t Index) {
  AttrBuilder Incompatible;

  if (!Ty->isIntegerTy())
    // Attribute that only apply to integers.
    Incompatible.addAttribute(Attribute::SExt)
      .addAttribute(Attribute::ZExt);

  if (!Ty->isPointerTy())
    // Attribute that only apply to pointers.
    Incompatible.addAttribute(Attribute::ByVal)
      .addAttribute(Attribute::Nest)
      .addAttribute(Attribute::NoAlias)
      .addAttribute(Attribute::NoCapture)
      .addAttribute(Attribute::StructRet);

  return AttributeSet::get(Ty->getContext(), Index, Incompatible);
}

/// \brief This returns an integer containing an encoding of all the LLVM
/// attributes found in the given attribute bitset.  Any change to this encoding
/// is a breaking change to bitcode compatibility.
/// N.B. This should be used only by the bitcode reader!
uint64_t AttributeFuncs::encodeLLVMAttributesForBitcode(AttributeSet Attrs,
                                                        unsigned Index) {
  // FIXME: It doesn't make sense to store the alignment information as an
  // expanded out value, we should store it as a log2 value.  However, we can't
  // just change that here without breaking bitcode compatibility.  If this ever
  // becomes a problem in practice, we should introduce new tag numbers in the
  // bitcode file and have those tags use a more efficiently encoded alignment
  // field.

  // Store the alignment in the bitcode as a 16-bit raw value instead of a 5-bit
  // log2 encoded value. Shift the bits above the alignment up by 11 bits.
  uint64_t EncodedAttrs = Attrs.Raw(Index) & 0xffff;
  if (Attrs.hasAttribute(Index, Attribute::Alignment))
    EncodedAttrs |= Attrs.getParamAlignment(Index) << 16;
  EncodedAttrs |= (Attrs.Raw(Index) & (0xffffULL << 21)) << 11;
  return EncodedAttrs;
}

/// \brief This fills an AttrBuilder object with the LLVM attributes that have
/// been decoded from the given integer. This function must stay in sync with
/// 'encodeLLVMAttributesForBitcode'.
/// N.B. This should be used only by the bitcode reader!
void AttributeFuncs::decodeLLVMAttributesForBitcode(LLVMContext &C,
                                                    AttrBuilder &B,
                                                    uint64_t EncodedAttrs) {
  // The alignment is stored as a 16-bit raw value from bits 31--16.  We shift
  // the bits above 31 down by 11 bits.
  unsigned Alignment = (EncodedAttrs & (0xffffULL << 16)) >> 16;
  assert((!Alignment || isPowerOf2_32(Alignment)) &&
         "Alignment must be a power of two.");

  if (Alignment)
    B.addAlignmentAttr(Alignment);
  B.addRawValue(((EncodedAttrs & (0xffffULL << 32)) >> 11) |
                (EncodedAttrs & 0xffff));
}
