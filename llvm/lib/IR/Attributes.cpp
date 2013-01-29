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

Attribute Attribute::get(LLVMContext &Context, AttrKind Kind) {
  AttrBuilder B;
  return Attribute::get(Context, B.addAttribute(Kind));
}

Attribute Attribute::get(LLVMContext &Context, AttrBuilder &B) {
  // If there are no attributes, return an empty Attribute class.
  if (!B.hasAttributes())
    return Attribute();

  // Otherwise, build a key to look up the existing attributes.
  LLVMContextImpl *pImpl = Context.pImpl;
  FoldingSetNodeID ID;
  ConstantInt *CI = ConstantInt::get(Type::getInt64Ty(Context), B.Raw());
  ID.AddPointer(CI);

  void *InsertPoint;
  AttributeImpl *PA = pImpl->AttrsSet.FindNodeOrInsertPos(ID, InsertPoint);

  if (!PA) {
    // If we didn't find any existing attributes of the same shape then create a
    // new one and insert it.
    PA = new AttributeImpl(Context, CI);
    pImpl->AttrsSet.InsertNode(PA, InsertPoint);
  }

  // Return the AttributesList that we found or created.
  return Attribute(PA);
}

Attribute Attribute::getWithAlignment(LLVMContext &Context, uint64_t Align) {
  AttrBuilder B;
  return get(Context, B.addAlignmentAttr(Align));
}

Attribute Attribute::getWithStackAlignment(LLVMContext &Context,
                                           uint64_t Align) {
  AttrBuilder B;
  return get(Context, B.addStackAlignmentAttr(Align));
}

//===----------------------------------------------------------------------===//
// Attribute Accessor Methods
//===----------------------------------------------------------------------===//

bool Attribute::hasAttribute(AttrKind Val) const {
  return pImpl && pImpl->hasAttribute(Val);
}

bool Attribute::hasAttributes() const {
  return pImpl && pImpl->hasAttributes();
}

/// This returns the alignment field of an attribute as a byte alignment value.
unsigned Attribute::getAlignment() const {
  if (!hasAttribute(Attribute::Alignment))
    return 0;
  return pImpl->getAlignment();
}

/// This returns the stack alignment field of an attribute as a byte alignment
/// value.
unsigned Attribute::getStackAlignment() const {
  if (!hasAttribute(Attribute::StackAlignment))
    return 0;
  return pImpl->getStackAlignment();
}

std::string Attribute::getAsString() const {
  std::string Result;
  if (hasAttribute(Attribute::ZExt))
    Result += "zeroext ";
  if (hasAttribute(Attribute::SExt))
    Result += "signext ";
  if (hasAttribute(Attribute::NoReturn))
    Result += "noreturn ";
  if (hasAttribute(Attribute::NoUnwind))
    Result += "nounwind ";
  if (hasAttribute(Attribute::UWTable))
    Result += "uwtable ";
  if (hasAttribute(Attribute::ReturnsTwice))
    Result += "returns_twice ";
  if (hasAttribute(Attribute::InReg))
    Result += "inreg ";
  if (hasAttribute(Attribute::NoAlias))
    Result += "noalias ";
  if (hasAttribute(Attribute::NoCapture))
    Result += "nocapture ";
  if (hasAttribute(Attribute::StructRet))
    Result += "sret ";
  if (hasAttribute(Attribute::ByVal))
    Result += "byval ";
  if (hasAttribute(Attribute::Nest))
    Result += "nest ";
  if (hasAttribute(Attribute::ReadNone))
    Result += "readnone ";
  if (hasAttribute(Attribute::ReadOnly))
    Result += "readonly ";
  if (hasAttribute(Attribute::OptimizeForSize))
    Result += "optsize ";
  if (hasAttribute(Attribute::NoInline))
    Result += "noinline ";
  if (hasAttribute(Attribute::InlineHint))
    Result += "inlinehint ";
  if (hasAttribute(Attribute::AlwaysInline))
    Result += "alwaysinline ";
  if (hasAttribute(Attribute::StackProtect))
    Result += "ssp ";
  if (hasAttribute(Attribute::StackProtectReq))
    Result += "sspreq ";
  if (hasAttribute(Attribute::StackProtectStrong))
    Result += "sspstrong ";
  if (hasAttribute(Attribute::NoRedZone))
    Result += "noredzone ";
  if (hasAttribute(Attribute::NoImplicitFloat))
    Result += "noimplicitfloat ";
  if (hasAttribute(Attribute::Naked))
    Result += "naked ";
  if (hasAttribute(Attribute::NonLazyBind))
    Result += "nonlazybind ";
  if (hasAttribute(Attribute::AddressSafety))
    Result += "address_safety ";
  if (hasAttribute(Attribute::MinSize))
    Result += "minsize ";
  if (hasAttribute(Attribute::StackAlignment)) {
    Result += "alignstack(";
    Result += utostr(getStackAlignment());
    Result += ") ";
  }
  if (hasAttribute(Attribute::Alignment)) {
    Result += "align ";
    Result += utostr(getAlignment());
    Result += " ";
  }
  if (hasAttribute(Attribute::NoDuplicate))
    Result += "noduplicate ";
  // Trim the trailing space.
  assert(!Result.empty() && "Unknown attribute!");
  Result.erase(Result.end()-1);
  return Result;
}

bool Attribute::operator==(AttrKind K) const {
  return pImpl && *pImpl == K;
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

uint64_t Attribute::Raw() const {
  return pImpl ? pImpl->Raw() : 0;
}

//===----------------------------------------------------------------------===//
// AttributeImpl Definition
//===----------------------------------------------------------------------===//

AttributeImpl::AttributeImpl(LLVMContext &C, Attribute::AttrKind data)
  : Context(C) {
  Data = ConstantInt::get(Type::getInt64Ty(C), data);
}
AttributeImpl::AttributeImpl(LLVMContext &C, Attribute::AttrKind data,
                             ArrayRef<Constant*> values)
  : Context(C) {
  Data = ConstantInt::get(Type::getInt64Ty(C), data);
  Vals.reserve(values.size());
  Vals.append(values.begin(), values.end());
}
AttributeImpl::AttributeImpl(LLVMContext &C, StringRef data)
  : Context(C) {
  Data = ConstantDataArray::getString(C, data);
}

bool AttributeImpl::hasAttribute(Attribute::AttrKind A) const {
  return (Raw() & getAttrMask(A)) != 0;
}

bool AttributeImpl::hasAttributes() const {
  return Raw() != 0;
}

uint64_t AttributeImpl::getAlignment() const {
  uint64_t Mask = Raw() & getAttrMask(Attribute::Alignment);
  return 1ULL << ((Mask >> 16) - 1);
}

uint64_t AttributeImpl::getStackAlignment() const {
  uint64_t Mask = Raw() & getAttrMask(Attribute::StackAlignment);
  return 1ULL << ((Mask >> 26) - 1);
}

bool AttributeImpl::operator==(Attribute::AttrKind Kind) const {
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Data))
    return CI->getZExtValue() == Kind;
  return false;
}
bool AttributeImpl::operator!=(Attribute::AttrKind Kind) const {
  return !(*this == Kind);
}

bool AttributeImpl::operator==(StringRef Kind) const {
  if (ConstantDataArray *CDA = dyn_cast<ConstantDataArray>(Data))
    if (CDA->isString())
      return CDA->getAsString() == Kind;
  return false;
}

bool AttributeImpl::operator!=(StringRef Kind) const {
  return !(*this == Kind);
}

bool AttributeImpl::operator<(const AttributeImpl &AI) const {
  if (!Data && !AI.Data) return false;
  if (!Data && AI.Data) return true;
  if (Data && !AI.Data) return false;

  ConstantInt *ThisCI = dyn_cast<ConstantInt>(Data);
  ConstantInt *ThatCI = dyn_cast<ConstantInt>(AI.Data);

  ConstantDataArray *ThisCDA = dyn_cast<ConstantDataArray>(Data);
  ConstantDataArray *ThatCDA = dyn_cast<ConstantDataArray>(AI.Data);

  if (ThisCI && ThatCI)
    return ThisCI->getZExtValue() < ThatCI->getZExtValue();

  if (ThisCI && ThatCDA)
    return true;

  if (ThisCDA && ThatCI)
    return false;

  return ThisCDA->getAsString() < ThatCDA->getAsString();
}

uint64_t AttributeImpl::Raw() const {
  // FIXME: Remove this.
  return cast<ConstantInt>(Data)->getZExtValue();
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

//===----------------------------------------------------------------------===//
// AttributeSetImpl Definition
//===----------------------------------------------------------------------===//

uint64_t AttributeSetImpl::Raw(uint64_t Index) const {
  for (unsigned I = 0, E = getNumAttributes(); I != E; ++I) {
    if (getSlotIndex(I) != Index) continue;
    const AttributeSetNode *ASN = AttrNodes[I].second;
    AttrBuilder B;

    for (AttributeSetNode::const_iterator II = ASN->begin(),
           IE = ASN->end(); II != IE; ++II)
      B.addAttributes(*II);
    return B.Raw();
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
    assert(Attrs[i].second.hasAttributes() &&
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
    while (I->first == Index && I != E) {
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
        B.addAttributes(*II);
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

  // Now add the attribute into the correct slot. There may already be an
  // AttributeSet there.
  AttrBuilder B(AS, Idx);

  for (unsigned I = 0, E = Attrs.pImpl->getNumAttributes(); I != E; ++I)
    if (Attrs.getSlotIndex(I) == Idx) {
      for (AttributeSetImpl::const_iterator II = Attrs.pImpl->begin(I),
             IE = Attrs.pImpl->end(I); II != IE; ++II)
        B.removeAttributes(*II);
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
                      ArrayRef<std::pair<unsigned, Attribute> >(
                        std::make_pair(Idx, getAttributes(Idx)))) :
    AttributeSet();
}

AttributeSet AttributeSet::getRetAttributes() const {
  return pImpl && hasAttributes(ReturnIndex) ?
    AttributeSet::get(pImpl->getContext(),
                      ArrayRef<std::pair<unsigned, Attribute> >(
                        std::make_pair(ReturnIndex,
                                       getAttributes(ReturnIndex)))) :
    AttributeSet();
}

AttributeSet AttributeSet::getFnAttributes() const {
  return pImpl && hasAttributes(FunctionIndex) ?
    AttributeSet::get(pImpl->getContext(),
                      ArrayRef<std::pair<unsigned, Attribute> >(
                        std::make_pair(FunctionIndex,
                                       getAttributes(FunctionIndex)))) :
    AttributeSet();
}

bool AttributeSet::hasAttribute(unsigned Index, Attribute::AttrKind Kind) const{
  return getAttributes(Index).hasAttribute(Kind);
}

bool AttributeSet::hasAttributes(unsigned Index) const {
  return getAttributes(Index).hasAttributes();
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

unsigned AttributeSet::getParamAlignment(unsigned Idx) const {
  return getAttributes(Idx).getAlignment();
}

unsigned AttributeSet::getStackAlignment(unsigned Index) const {
  return getAttributes(Index).getStackAlignment();
}

std::string AttributeSet::getAsString(unsigned Index) const {
  return getAttributes(Index).getAsString();
}

/// \brief The attributes for the specified index are returned.
///
/// FIXME: This shouldn't return 'Attribute'.
Attribute AttributeSet::getAttributes(unsigned Idx) const {
  if (pImpl == 0) return Attribute();

  // Loop through to find the attribute we want.
  for (unsigned I = 0, E = pImpl->getNumAttributes(); I != E; ++I) {
    if (pImpl->getSlotIndex(I) != Idx) continue;

    AttrBuilder B;
    for (AttributeSetImpl::const_iterator II = pImpl->begin(I),
           IE = pImpl->end(I); II != IE; ++II)
      B.addAttributes(*II);
    return Attribute::get(pImpl->getContext(), B);
  }

  return Attribute();
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

  AttrBuilder B;

  for (unsigned I = 0, E = pImpl->getNumAttributes(); I != E; ++I) {
    if (pImpl->getSlotIndex(I) != Idx) continue;

    for (AttributeSetNode::const_iterator II = pImpl->begin(I),
           IE = pImpl->end(I); II != IE; ++II)
      B.addAttributes(*II);

    break;
  }

  if (!B.hasAttributes()) return;

  uint64_t Mask = B.Raw();

  for (Attribute::AttrKind I = Attribute::None; I != Attribute::EndAttrKinds;
       I = Attribute::AttrKind(I + 1)) {
    if (uint64_t A = (Mask & AttributeImpl::getAttrMask(I))) {
      Attrs.insert(I);

      if (I == Attribute::Alignment)
        Alignment = 1ULL << ((A >> 16) - 1);
      else if (I == Attribute::StackAlignment)
        StackAlignment = 1ULL << ((A >> 26)-1);
    }
  }
}

void AttrBuilder::clear() {
  Attrs.clear();
  Alignment = StackAlignment = 0;
}

AttrBuilder &AttrBuilder::addAttribute(Attribute::AttrKind Val) {
  Attrs.insert(Val);
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

AttrBuilder &AttrBuilder::addAttributes(Attribute Attr) {
  uint64_t Mask = Attr.Raw();

  for (Attribute::AttrKind I = Attribute::None; I != Attribute::EndAttrKinds;
       I = Attribute::AttrKind(I + 1))
    if ((Mask & AttributeImpl::getAttrMask(I)) != 0)
      Attrs.insert(I);

  if (Attr.getAlignment())
    Alignment = Attr.getAlignment();
  if (Attr.getStackAlignment())
    StackAlignment = Attr.getStackAlignment();
  return *this;
}

AttrBuilder &AttrBuilder::removeAttributes(Attribute A) {
  uint64_t Mask = A.Raw();

  for (Attribute::AttrKind I = Attribute::None; I != Attribute::EndAttrKinds;
       I = Attribute::AttrKind(I + 1)) {
    if (Mask & AttributeImpl::getAttrMask(I)) {
      Attrs.erase(I);

      if (I == Attribute::Alignment)
        Alignment = 0;
      else if (I == Attribute::StackAlignment)
        StackAlignment = 0;
    }
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

bool AttrBuilder::hasAttributes(const Attribute &A) const {
  return Raw() & A.Raw();
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

uint64_t AttrBuilder::Raw() const {
  uint64_t Mask = 0;

  for (DenseSet<Attribute::AttrKind>::const_iterator I = Attrs.begin(),
         E = Attrs.end(); I != E; ++I) {
    Attribute::AttrKind Kind = *I;

    if (Kind == Attribute::Alignment)
      Mask |= (Log2_32(Alignment) + 1) << 16;
    else if (Kind == Attribute::StackAlignment)
      Mask |= (Log2_32(StackAlignment) + 1) << 26;
    else
      Mask |= AttributeImpl::getAttrMask(Kind);
  }

  return Mask;
}

//===----------------------------------------------------------------------===//
// AttributeFuncs Function Defintions
//===----------------------------------------------------------------------===//

Attribute AttributeFuncs::typeIncompatible(Type *Ty) {
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

  return Attribute::get(Ty->getContext(), Incompatible);
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

  B.addRawValue(EncodedAttrs & 0xffff);
  if (Alignment)
    B.addAlignmentAttr(Alignment);
  B.addRawValue((EncodedAttrs & (0xffffULL << 32)) >> 11);
}

