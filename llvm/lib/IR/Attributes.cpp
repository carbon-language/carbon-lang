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

Attribute Attribute::get(LLVMContext &Context, Attribute::AttrKind Kind,
                         uint64_t Val) {
  LLVMContextImpl *pImpl = Context.pImpl;
  FoldingSetNodeID ID;
  ID.AddInteger(Kind);
  if (Val) ID.AddInteger(Val);

  void *InsertPoint;
  AttributeImpl *PA = pImpl->AttrsSet.FindNodeOrInsertPos(ID, InsertPoint);

  if (!PA) {
    // If we didn't find any existing attributes of the same shape then create a
    // new one and insert it.
    PA = !Val ?
      new AttributeImpl(Context, Kind) :
      new AttributeImpl(Context, Kind, Val);
    pImpl->AttrsSet.InsertNode(PA, InsertPoint);
  }

  // Return the Attribute that we found or created.
  return Attribute(PA);
}

Attribute Attribute::get(LLVMContext &Context, StringRef Kind, StringRef Val) {
  LLVMContextImpl *pImpl = Context.pImpl;
  FoldingSetNodeID ID;
  ID.AddString(Kind);
  if (!Val.empty()) ID.AddString(Val);

  void *InsertPoint;
  AttributeImpl *PA = pImpl->AttrsSet.FindNodeOrInsertPos(ID, InsertPoint);

  if (!PA) {
    // If we didn't find any existing attributes of the same shape then create a
    // new one and insert it.
    PA = new AttributeImpl(Context, Kind, Val);
    pImpl->AttrsSet.InsertNode(PA, InsertPoint);
  }

  // Return the Attribute that we found or created.
  return Attribute(PA);
}

Attribute Attribute::getWithAlignment(LLVMContext &Context, uint64_t Align) {
  assert(isPowerOf2_32(Align) && "Alignment must be a power of two.");
  assert(Align <= 0x40000000 && "Alignment too large.");
  return get(Context, Alignment, Align);
}

Attribute Attribute::getWithStackAlignment(LLVMContext &Context,
                                           uint64_t Align) {
  assert(isPowerOf2_32(Align) && "Alignment must be a power of two.");
  assert(Align <= 0x100 && "Alignment too large.");
  return get(Context, StackAlignment, Align);
}

//===----------------------------------------------------------------------===//
// Attribute Accessor Methods
//===----------------------------------------------------------------------===//

bool Attribute::isEnumAttribute() const {
  return pImpl && pImpl->isEnumAttribute();
}

bool Attribute::isAlignAttribute() const {
  return pImpl && pImpl->isAlignAttribute();
}

bool Attribute::isStringAttribute() const {
  return pImpl && pImpl->isStringAttribute();
}

Attribute::AttrKind Attribute::getKindAsEnum() const {
  assert((isEnumAttribute() || isAlignAttribute()) &&
         "Invalid attribute type to get the kind as an enum!");
  return pImpl ? pImpl->getKindAsEnum() : None;
}

uint64_t Attribute::getValueAsInt() const {
  assert(isAlignAttribute() &&
         "Expected the attribute to be an alignment attribute!");
  return pImpl ? pImpl->getValueAsInt() : 0;
}

StringRef Attribute::getKindAsString() const {
  assert(isStringAttribute() &&
         "Invalid attribute type to get the kind as a string!");
  return pImpl ? pImpl->getKindAsString() : StringRef();
}

StringRef Attribute::getValueAsString() const {
  assert(isStringAttribute() &&
         "Invalid attribute type to get the value as a string!");
  return pImpl ? pImpl->getValueAsString() : StringRef();
}

bool Attribute::hasAttribute(AttrKind Kind) const {
  return (pImpl && pImpl->hasAttribute(Kind)) || (!pImpl && Kind == None);
}

bool Attribute::hasAttribute(StringRef Kind) const {
  if (!isStringAttribute()) return false;
  return pImpl && pImpl->hasAttribute(Kind);
}

/// This returns the alignment field of an attribute as a byte alignment value.
unsigned Attribute::getAlignment() const {
  assert(hasAttribute(Attribute::Alignment) &&
         "Trying to get alignment from non-alignment attribute!");
  return pImpl->getValueAsInt();
}

/// This returns the stack alignment field of an attribute as a byte alignment
/// value.
unsigned Attribute::getStackAlignment() const {
  assert(hasAttribute(Attribute::StackAlignment) &&
         "Trying to get alignment from non-alignment attribute!");
  return pImpl->getValueAsInt();
}

std::string Attribute::getAsString(bool InAttrGrp) const {
  if (!pImpl) return "";

  if (hasAttribute(Attribute::SanitizeAddress))
    return "sanitize_address";
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
  if (hasAttribute(Attribute::NoBuiltin))
    return "nobuiltin";
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
  if (hasAttribute(Attribute::Returned))
    return "returned";
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
  if (hasAttribute(Attribute::SanitizeThread))
    return "sanitize_thread";
  if (hasAttribute(Attribute::SanitizeMemory))
    return "sanitize_memory";
  if (hasAttribute(Attribute::UWTable))
    return "uwtable";
  if (hasAttribute(Attribute::ZExt))
    return "zeroext";

  // FIXME: These should be output like this:
  //
  //   align=4
  //   alignstack=8
  //
  if (hasAttribute(Attribute::Alignment)) {
    std::string Result;
    Result += "align";
    Result += (InAttrGrp) ? "=" : " ";
    Result += utostr(getValueAsInt());
    return Result;
  }

  if (hasAttribute(Attribute::StackAlignment)) {
    std::string Result;
    Result += "alignstack";
    if (InAttrGrp) {
      Result += "=";
      Result += utostr(getValueAsInt());
    } else {
      Result += "(";
      Result += utostr(getValueAsInt());
      Result += ")";
    }
    return Result;
  }

  // Convert target-dependent attributes to strings of the form:
  //
  //   "kind"
  //   "kind" = "value"
  //
  if (isStringAttribute()) {
    std::string Result;
    Result += '\"' + getKindAsString().str() + '"';

    StringRef Val = pImpl->getValueAsString();
    if (Val.empty()) return Result;

    Result += "=\"" + Val.str() + '"';
    return Result;
  }

  llvm_unreachable("Unknown attribute");
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

AttributeImpl::AttributeImpl(LLVMContext &C, Attribute::AttrKind Kind)
  : Context(C), Entry(new EnumAttributeEntry(Kind)) {}

AttributeImpl::AttributeImpl(LLVMContext &C, Attribute::AttrKind Kind,
                             unsigned Align)
  : Context(C) {
  assert((Kind == Attribute::Alignment || Kind == Attribute::StackAlignment) &&
         "Wrong kind for alignment attribute!");
  Entry = new AlignAttributeEntry(Kind, Align);
}

AttributeImpl::AttributeImpl(LLVMContext &C, StringRef Kind, StringRef Val)
  : Context(C), Entry(new StringAttributeEntry(Kind, Val)) {}

AttributeImpl::~AttributeImpl() {
  delete Entry;
}

bool AttributeImpl::isEnumAttribute() const {
  return isa<EnumAttributeEntry>(Entry);
}

bool AttributeImpl::isAlignAttribute() const {
  return isa<AlignAttributeEntry>(Entry);
}

bool AttributeImpl::isStringAttribute() const {
  return isa<StringAttributeEntry>(Entry);
}

bool AttributeImpl::hasAttribute(Attribute::AttrKind A) const {
  if (isStringAttribute()) return false;
  return getKindAsEnum() == A;
}

bool AttributeImpl::hasAttribute(StringRef Kind) const {
  if (!isStringAttribute()) return false;
  return getKindAsString() == Kind;
}

Attribute::AttrKind AttributeImpl::getKindAsEnum() const {
  if (EnumAttributeEntry *E = dyn_cast<EnumAttributeEntry>(Entry))
    return E->getEnumKind();
  return cast<AlignAttributeEntry>(Entry)->getEnumKind();
}

uint64_t AttributeImpl::getValueAsInt() const {
  return cast<AlignAttributeEntry>(Entry)->getAlignment();
}

StringRef AttributeImpl::getKindAsString() const {
  return cast<StringAttributeEntry>(Entry)->getStringKind();
}

StringRef AttributeImpl::getValueAsString() const {
  return cast<StringAttributeEntry>(Entry)->getStringValue();
}

bool AttributeImpl::operator<(const AttributeImpl &AI) const {
  // This sorts the attributes with Attribute::AttrKinds coming first (sorted
  // relative to their enum value) and then strings.
  if (isEnumAttribute()) {
    if (AI.isEnumAttribute()) return getKindAsEnum() < AI.getKindAsEnum();
    if (AI.isAlignAttribute()) return true;
    if (AI.isStringAttribute()) return true;
  }

  if (isAlignAttribute()) {
    if (AI.isEnumAttribute()) return false;
    if (AI.isAlignAttribute()) return getValueAsInt() < AI.getValueAsInt();
    if (AI.isStringAttribute()) return true;
  }

  if (AI.isEnumAttribute()) return false;
  if (AI.isAlignAttribute()) return false;
  if (getKindAsString() == AI.getKindAsString())
    return getValueAsString() < AI.getValueAsString();
  return getKindAsString() < AI.getKindAsString();
}

uint64_t AttributeImpl::getAttrMask(Attribute::AttrKind Val) {
  // FIXME: Remove this.
  switch (Val) {
  case Attribute::EndAttrKinds:
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
  case Attribute::SanitizeAddress: return 1ULL << 32;
  case Attribute::MinSize:         return 1ULL << 33;
  case Attribute::NoDuplicate:     return 1ULL << 34;
  case Attribute::StackProtectStrong: return 1ULL << 35;
  case Attribute::SanitizeThread:  return 1ULL << 36;
  case Attribute::SanitizeMemory:  return 1ULL << 37;
  case Attribute::NoBuiltin:       return 1ULL << 38;
  case Attribute::Returned:        return 1ULL << 39;
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
  array_pod_sort(SortedAttrs.begin(), SortedAttrs.end());

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

bool AttributeSetNode::hasAttribute(StringRef Kind) const {
  for (SmallVectorImpl<Attribute>::const_iterator I = AttrList.begin(),
         E = AttrList.end(); I != E; ++I)
    if (I->hasAttribute(Kind))
      return true;
  return false;
}

Attribute AttributeSetNode::getAttribute(Attribute::AttrKind Kind) const {
  for (SmallVectorImpl<Attribute>::const_iterator I = AttrList.begin(),
         E = AttrList.end(); I != E; ++I)
    if (I->hasAttribute(Kind))
      return *I;
  return Attribute();
}

Attribute AttributeSetNode::getAttribute(StringRef Kind) const {
  for (SmallVectorImpl<Attribute>::const_iterator I = AttrList.begin(),
         E = AttrList.end(); I != E; ++I)
    if (I->hasAttribute(Kind))
      return *I;
  return Attribute();
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

std::string AttributeSetNode::getAsString(bool TargetIndependent,
                                          bool InAttrGrp) const {
  std::string Str;
  for (SmallVectorImpl<Attribute>::const_iterator I = AttrList.begin(),
         E = AttrList.end(); I != E; ++I) {
    if (TargetIndependent || !I->isStringAttribute()) {
      if (I != AttrList.begin())
        Str += ' ';
      Str += I->getAsString(InAttrGrp);
    }
  }
  return Str;
}

//===----------------------------------------------------------------------===//
// AttributeSetImpl Definition
//===----------------------------------------------------------------------===//

uint64_t AttributeSetImpl::Raw(unsigned Index) const {
  for (unsigned I = 0, E = getNumAttributes(); I != E; ++I) {
    if (getSlotIndex(I) != Index) continue;
    const AttributeSetNode *ASN = AttrNodes[I].second;
    uint64_t Mask = 0;

    for (AttributeSetNode::const_iterator II = ASN->begin(),
           IE = ASN->end(); II != IE; ++II) {
      Attribute Attr = *II;

      // This cannot handle string attributes.
      if (Attr.isStringAttribute()) continue;

      Attribute::AttrKind Kind = Attr.getKindAsEnum();

      if (Kind == Attribute::Alignment)
        Mask |= (Log2_32(ASN->getAlignment()) + 1) << 16;
      else if (Kind == Attribute::StackAlignment)
        Mask |= (Log2_32(ASN->getStackAlignment()) + 1) << 26;
      else
        Mask |= AttributeImpl::getAttrMask(Kind);
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
    assert(!Attrs[i].second.hasAttribute(Attribute::None) &&
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

AttributeSet AttributeSet::get(LLVMContext &C, unsigned Index, AttrBuilder &B) {
  if (!B.hasAttributes())
    return AttributeSet();

  // Add target-independent attributes.
  SmallVector<std::pair<unsigned, Attribute>, 8> Attrs;
  for (Attribute::AttrKind Kind = Attribute::None;
       Kind != Attribute::EndAttrKinds; Kind = Attribute::AttrKind(Kind + 1)) {
    if (!B.contains(Kind))
      continue;

    if (Kind == Attribute::Alignment)
      Attrs.push_back(std::make_pair(Index, Attribute::
                                     getWithAlignment(C, B.getAlignment())));
    else if (Kind == Attribute::StackAlignment)
      Attrs.push_back(std::make_pair(Index, Attribute::
                              getWithStackAlignment(C, B.getStackAlignment())));
    else
      Attrs.push_back(std::make_pair(Index, Attribute::get(C, Kind)));
  }

  // Add target-dependent (string) attributes.
  for (AttrBuilder::td_iterator I = B.td_begin(), E = B.td_end();
       I != E; ++I)
    Attrs.push_back(std::make_pair(Index, Attribute::get(C, I->first,I->second)));

  return get(C, Attrs);
}

AttributeSet AttributeSet::get(LLVMContext &C, unsigned Index,
                               ArrayRef<Attribute::AttrKind> Kind) {
  SmallVector<std::pair<unsigned, Attribute>, 8> Attrs;
  for (ArrayRef<Attribute::AttrKind>::iterator I = Kind.begin(),
         E = Kind.end(); I != E; ++I)
    Attrs.push_back(std::make_pair(Index, Attribute::get(C, *I)));
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

AttributeSet AttributeSet::addAttribute(LLVMContext &C, unsigned Index,
                                        Attribute::AttrKind Attr) const {
  if (hasAttribute(Index, Attr)) return *this;
  return addAttributes(C, Index, AttributeSet::get(C, Index, Attr));
}

AttributeSet AttributeSet::addAttribute(LLVMContext &C, unsigned Index,
                                        StringRef Kind) const {
  llvm::AttrBuilder B;
  B.addAttribute(Kind);
  return addAttributes(C, Index, AttributeSet::get(C, Index, B));
}

AttributeSet AttributeSet::addAttributes(LLVMContext &C, unsigned Index,
                                         AttributeSet Attrs) const {
  if (!pImpl) return Attrs;
  if (!Attrs.pImpl) return *this;

#ifndef NDEBUG
  // FIXME it is not obvious how this should work for alignment. For now, say
  // we can't change a known alignment.
  unsigned OldAlign = getParamAlignment(Index);
  unsigned NewAlign = Attrs.getParamAlignment(Index);
  assert((!OldAlign || !NewAlign || OldAlign == NewAlign) &&
         "Attempt to change alignment!");
#endif

  // Add the attribute slots before the one we're trying to add.
  SmallVector<AttributeSet, 4> AttrSet;
  uint64_t NumAttrs = pImpl->getNumAttributes();
  AttributeSet AS;
  uint64_t LastIndex = 0;
  for (unsigned I = 0, E = NumAttrs; I != E; ++I) {
    if (getSlotIndex(I) >= Index) {
      if (getSlotIndex(I) == Index) AS = getSlotAttributes(LastIndex++);
      break;
    }
    LastIndex = I + 1;
    AttrSet.push_back(getSlotAttributes(I));
  }

  // Now add the attribute into the correct slot. There may already be an
  // AttributeSet there.
  AttrBuilder B(AS, Index);

  for (unsigned I = 0, E = Attrs.pImpl->getNumAttributes(); I != E; ++I)
    if (Attrs.getSlotIndex(I) == Index) {
      for (AttributeSetImpl::const_iterator II = Attrs.pImpl->begin(I),
             IE = Attrs.pImpl->end(I); II != IE; ++II)
        B.addAttribute(*II);
      break;
    }

  AttrSet.push_back(AttributeSet::get(C, Index, B));

  // Add the remaining attribute slots.
  for (unsigned I = LastIndex, E = NumAttrs; I < E; ++I)
    AttrSet.push_back(getSlotAttributes(I));

  return get(C, AttrSet);
}

AttributeSet AttributeSet::removeAttribute(LLVMContext &C, unsigned Index,
                                           Attribute::AttrKind Attr) const {
  if (!hasAttribute(Index, Attr)) return *this;
  return removeAttributes(C, Index, AttributeSet::get(C, Index, Attr));
}

AttributeSet AttributeSet::removeAttributes(LLVMContext &C, unsigned Index,
                                            AttributeSet Attrs) const {
  if (!pImpl) return AttributeSet();
  if (!Attrs.pImpl) return *this;

#ifndef NDEBUG
  // FIXME it is not obvious how this should work for alignment.
  // For now, say we can't pass in alignment, which no current use does.
  assert(!Attrs.hasAttribute(Index, Attribute::Alignment) &&
         "Attempt to change alignment!");
#endif

  // Add the attribute slots before the one we're trying to add.
  SmallVector<AttributeSet, 4> AttrSet;
  uint64_t NumAttrs = pImpl->getNumAttributes();
  AttributeSet AS;
  uint64_t LastIndex = 0;
  for (unsigned I = 0, E = NumAttrs; I != E; ++I) {
    if (getSlotIndex(I) >= Index) {
      if (getSlotIndex(I) == Index) AS = getSlotAttributes(LastIndex++);
      break;
    }
    LastIndex = I + 1;
    AttrSet.push_back(getSlotAttributes(I));
  }

  // Now remove the attribute from the correct slot. There may already be an
  // AttributeSet there.
  AttrBuilder B(AS, Index);

  for (unsigned I = 0, E = Attrs.pImpl->getNumAttributes(); I != E; ++I)
    if (Attrs.getSlotIndex(I) == Index) {
      B.removeAttributes(Attrs.pImpl->getSlotAttributes(I), Index);
      break;
    }

  AttrSet.push_back(AttributeSet::get(C, Index, B));

  // Add the remaining attribute slots.
  for (unsigned I = LastIndex, E = NumAttrs; I < E; ++I)
    AttrSet.push_back(getSlotAttributes(I));

  return get(C, AttrSet);
}

//===----------------------------------------------------------------------===//
// AttributeSet Accessor Methods
//===----------------------------------------------------------------------===//

LLVMContext &AttributeSet::getContext() const {
  return pImpl->getContext();
}

AttributeSet AttributeSet::getParamAttributes(unsigned Index) const {
  return pImpl && hasAttributes(Index) ?
    AttributeSet::get(pImpl->getContext(),
                      ArrayRef<std::pair<unsigned, AttributeSetNode*> >(
                        std::make_pair(Index, getAttributes(Index)))) :
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

bool AttributeSet::hasAttribute(unsigned Index, StringRef Kind) const {
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

Attribute AttributeSet::getAttribute(unsigned Index,
                                     Attribute::AttrKind Kind) const {
  AttributeSetNode *ASN = getAttributes(Index);
  return ASN ? ASN->getAttribute(Kind) : Attribute();
}

Attribute AttributeSet::getAttribute(unsigned Index,
                                     StringRef Kind) const {
  AttributeSetNode *ASN = getAttributes(Index);
  return ASN ? ASN->getAttribute(Kind) : Attribute();
}

unsigned AttributeSet::getParamAlignment(unsigned Index) const {
  AttributeSetNode *ASN = getAttributes(Index);
  return ASN ? ASN->getAlignment() : 0;
}

unsigned AttributeSet::getStackAlignment(unsigned Index) const {
  AttributeSetNode *ASN = getAttributes(Index);
  return ASN ? ASN->getStackAlignment() : 0;
}

std::string AttributeSet::getAsString(unsigned Index, bool TargetIndependent,
                                      bool InAttrGrp) const {
  AttributeSetNode *ASN = getAttributes(Index);
  return ASN ? ASN->getAsString(TargetIndependent, InAttrGrp) : std::string("");
}

/// \brief The attributes for the specified index are returned.
AttributeSetNode *AttributeSet::getAttributes(unsigned Index) const {
  if (!pImpl) return 0;

  // Loop through to find the attribute node we want.
  for (unsigned I = 0, E = pImpl->getNumAttributes(); I != E; ++I)
    if (pImpl->getSlotIndex(I) == Index)
      return pImpl->getSlotNode(I);

  return 0;
}

AttributeSet::iterator AttributeSet::begin(unsigned Slot) const {
  if (!pImpl)
    return ArrayRef<Attribute>().begin();
  return pImpl->begin(Slot);
}

AttributeSet::iterator AttributeSet::end(unsigned Slot) const {
  if (!pImpl)
    return ArrayRef<Attribute>().end();
  return pImpl->end(Slot);
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

unsigned AttributeSet::getSlotIndex(unsigned Slot) const {
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

AttrBuilder::AttrBuilder(AttributeSet AS, unsigned Index)
  : Attrs(0), Alignment(0), StackAlignment(0) {
  AttributeSetImpl *pImpl = AS.pImpl;
  if (!pImpl) return;

  for (unsigned I = 0, E = pImpl->getNumAttributes(); I != E; ++I) {
    if (pImpl->getSlotIndex(I) != Index) continue;

    for (AttributeSetImpl::const_iterator II = pImpl->begin(I),
           IE = pImpl->end(I); II != IE; ++II)
      addAttribute(*II);

    break;
  }
}

void AttrBuilder::clear() {
  Attrs.reset();
  Alignment = StackAlignment = 0;
}

AttrBuilder &AttrBuilder::addAttribute(Attribute::AttrKind Val) {
  assert((unsigned)Val < Attribute::EndAttrKinds && "Attribute out of range!");
  assert(Val != Attribute::Alignment && Val != Attribute::StackAlignment &&
         "Adding alignment attribute without adding alignment value!");
  Attrs[Val] = true;
  return *this;
}

AttrBuilder &AttrBuilder::addAttribute(Attribute Attr) {
  if (Attr.isStringAttribute()) {
    addAttribute(Attr.getKindAsString(), Attr.getValueAsString());
    return *this;
  }

  Attribute::AttrKind Kind = Attr.getKindAsEnum();
  Attrs[Kind] = true;

  if (Kind == Attribute::Alignment)
    Alignment = Attr.getAlignment();
  else if (Kind == Attribute::StackAlignment)
    StackAlignment = Attr.getStackAlignment();
  return *this;
}

AttrBuilder &AttrBuilder::addAttribute(StringRef A, StringRef V) {
  TargetDepAttrs[A] = V;
  return *this;
}

AttrBuilder &AttrBuilder::removeAttribute(Attribute::AttrKind Val) {
  assert((unsigned)Val < Attribute::EndAttrKinds && "Attribute out of range!");
  Attrs[Val] = false;

  if (Val == Attribute::Alignment)
    Alignment = 0;
  else if (Val == Attribute::StackAlignment)
    StackAlignment = 0;

  return *this;
}

AttrBuilder &AttrBuilder::removeAttributes(AttributeSet A, uint64_t Index) {
  unsigned Slot = ~0U;
  for (unsigned I = 0, E = A.getNumSlots(); I != E; ++I)
    if (A.getSlotIndex(I) == Index) {
      Slot = I;
      break;
    }

  assert(Slot != ~0U && "Couldn't find index in AttributeSet!");

  for (AttributeSet::iterator I = A.begin(Slot), E = A.end(Slot); I != E; ++I) {
    Attribute Attr = *I;
    if (Attr.isEnumAttribute() || Attr.isAlignAttribute()) {
      Attribute::AttrKind Kind = I->getKindAsEnum();
      Attrs[Kind] = false;

      if (Kind == Attribute::Alignment)
        Alignment = 0;
      else if (Kind == Attribute::StackAlignment)
        StackAlignment = 0;
    } else {
      assert(Attr.isStringAttribute() && "Invalid attribute type!");
      std::map<std::string, std::string>::iterator
        Iter = TargetDepAttrs.find(Attr.getKindAsString());
      if (Iter != TargetDepAttrs.end())
        TargetDepAttrs.erase(Iter);
    }
  }

  return *this;
}

AttrBuilder &AttrBuilder::removeAttribute(StringRef A) {
  std::map<std::string, std::string>::iterator I = TargetDepAttrs.find(A);
  if (I != TargetDepAttrs.end())
    TargetDepAttrs.erase(I);
  return *this;
}

AttrBuilder &AttrBuilder::addAlignmentAttr(unsigned Align) {
  if (Align == 0) return *this;

  assert(isPowerOf2_32(Align) && "Alignment must be a power of two.");
  assert(Align <= 0x40000000 && "Alignment too large.");

  Attrs[Attribute::Alignment] = true;
  Alignment = Align;
  return *this;
}

AttrBuilder &AttrBuilder::addStackAlignmentAttr(unsigned Align) {
  // Default alignment, allow the target to define how to align it.
  if (Align == 0) return *this;

  assert(isPowerOf2_32(Align) && "Alignment must be a power of two.");
  assert(Align <= 0x100 && "Alignment too large.");

  Attrs[Attribute::StackAlignment] = true;
  StackAlignment = Align;
  return *this;
}

AttrBuilder &AttrBuilder::merge(const AttrBuilder &B) {
  // FIXME: What if both have alignments, but they don't match?!
  if (!Alignment)
    Alignment = B.Alignment;

  if (!StackAlignment)
    StackAlignment = B.StackAlignment;

  Attrs |= B.Attrs;

  for (td_const_iterator I = B.TargetDepAttrs.begin(),
         E = B.TargetDepAttrs.end(); I != E; ++I)
    TargetDepAttrs[I->first] = I->second;

  return *this;
}

bool AttrBuilder::contains(StringRef A) const {
  return TargetDepAttrs.find(A) != TargetDepAttrs.end();
}

bool AttrBuilder::hasAttributes() const {
  return !Attrs.none() || !TargetDepAttrs.empty();
}

bool AttrBuilder::hasAttributes(AttributeSet A, uint64_t Index) const {
  unsigned Slot = ~0U;
  for (unsigned I = 0, E = A.getNumSlots(); I != E; ++I)
    if (A.getSlotIndex(I) == Index) {
      Slot = I;
      break;
    }

  assert(Slot != ~0U && "Couldn't find the index!");

  for (AttributeSet::iterator I = A.begin(Slot), E = A.end(Slot);
       I != E; ++I) {
    Attribute Attr = *I;
    if (Attr.isEnumAttribute() || Attr.isAlignAttribute()) {
      if (Attrs[I->getKindAsEnum()])
        return true;
    } else {
      assert(Attr.isStringAttribute() && "Invalid attribute kind!");
      return TargetDepAttrs.find(Attr.getKindAsString())!=TargetDepAttrs.end();
    }
  }

  return false;
}

bool AttrBuilder::hasAlignmentAttr() const {
  return Alignment != 0;
}

bool AttrBuilder::operator==(const AttrBuilder &B) {
  if (Attrs != B.Attrs)
    return false;

  for (td_const_iterator I = TargetDepAttrs.begin(),
         E = TargetDepAttrs.end(); I != E; ++I)
    if (B.TargetDepAttrs.find(I->first) == B.TargetDepAttrs.end())
      return false;

  return Alignment == B.Alignment && StackAlignment == B.StackAlignment;
}

AttrBuilder &AttrBuilder::addRawValue(uint64_t Val) {
  // FIXME: Remove this in 4.0.
  if (!Val) return *this;

  for (Attribute::AttrKind I = Attribute::None; I != Attribute::EndAttrKinds;
       I = Attribute::AttrKind(I + 1)) {
    if (uint64_t A = (Val & AttributeImpl::getAttrMask(I))) {
      Attrs[I] = true;
 
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
