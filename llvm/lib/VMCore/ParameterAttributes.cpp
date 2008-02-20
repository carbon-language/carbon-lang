//===-- ParameterAttributes.cpp - Implement ParameterAttrs ----------------===//
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

#include "llvm/ParameterAttributes.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Support/ManagedStatic.h"
#include <sstream>

using namespace llvm;

static ManagedStatic<FoldingSet<ParamAttrsList> > ParamAttrsLists;

ParamAttrsList::ParamAttrsList(const ParamAttrsVector &attrVec) 
  : attrs(attrVec), refCount(0) {
}

ParamAttrsList::~ParamAttrsList() {
  ParamAttrsLists->RemoveNode(this);
}

ParameterAttributes
ParamAttrsList::getParamAttrs(uint16_t Index) const {
  unsigned limit = attrs.size();
  for (unsigned i = 0; i < limit && attrs[i].index <= Index; ++i)
    if (attrs[i].index == Index)
      return attrs[i].attrs;
  return ParamAttr::None;
}

bool ParamAttrsList::hasAttrSomewhere(ParameterAttributes attr) const {
  for (unsigned i = 0, e = attrs.size(); i < e; ++i)
    if (attrs[i].attrs & attr)
      return true;
  return false;
}

std::string 
ParamAttrsList::getParamAttrsText(ParameterAttributes Attrs) {
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
    std::stringstream s;
    s << ((Attrs & ParamAttr::Alignment) >> 16);
    Result += "align ";
    Result += s.str();
    Result += " ";
  }
  return Result;
}

void ParamAttrsList::Profile(FoldingSetNodeID &ID,
                             const ParamAttrsVector &Attrs) {
  for (unsigned i = 0; i < Attrs.size(); ++i)
    ID.AddInteger(uint64_t(Attrs[i].attrs) << 16 | unsigned(Attrs[i].index));
}

const ParamAttrsList *
ParamAttrsList::get(const ParamAttrsVector &attrVec) {
  // If there are no attributes then return a null ParamAttrsList pointer.
  if (attrVec.empty())
    return 0;

#ifndef NDEBUG
  for (unsigned i = 0, e = attrVec.size(); i < e; ++i) {
    assert(attrVec[i].attrs != ParamAttr::None
           && "Pointless parameter attribute!");
    assert((!i || attrVec[i-1].index < attrVec[i].index)
           && "Misordered ParamAttrsList!");
  }
#endif

  // Otherwise, build a key to look up the existing attributes.
  FoldingSetNodeID ID;
  ParamAttrsList::Profile(ID, attrVec);
  void *InsertPos;
  ParamAttrsList *PAL = ParamAttrsLists->FindNodeOrInsertPos(ID, InsertPos);

  // If we didn't find any existing attributes of the same shape then
  // create a new one and insert it.
  if (!PAL) {
    PAL = new ParamAttrsList(attrVec);
    ParamAttrsLists->InsertNode(PAL, InsertPos);
  }

  // Return the ParamAttrsList that we found or created.
  return PAL;
}

const ParamAttrsList *
ParamAttrsList::getModified(const ParamAttrsList *PAL,
                            const ParamAttrsVector &modVec) {
  if (modVec.empty())
    return PAL;

#ifndef NDEBUG
  for (unsigned i = 0, e = modVec.size(); i < e; ++i)
    assert((!i || modVec[i-1].index < modVec[i].index)
           && "Misordered ParamAttrsList!");
#endif

  if (!PAL) {
    // Strip any instances of ParamAttr::None from modVec before calling 'get'.
    ParamAttrsVector newVec;
    newVec.reserve(modVec.size());
    for (unsigned i = 0, e = modVec.size(); i < e; ++i)
      if (modVec[i].attrs != ParamAttr::None)
        newVec.push_back(modVec[i]);
    return get(newVec);
  }

  const ParamAttrsVector &oldVec = PAL->attrs;

  ParamAttrsVector newVec;
  unsigned oldI = 0;
  unsigned modI = 0;
  unsigned oldE = oldVec.size();
  unsigned modE = modVec.size();

  while (oldI < oldE && modI < modE) {
    uint16_t oldIndex = oldVec[oldI].index;
    uint16_t modIndex = modVec[modI].index;

    if (oldIndex < modIndex) {
      newVec.push_back(oldVec[oldI]);
      ++oldI;
    } else if (modIndex < oldIndex) {
      if (modVec[modI].attrs != ParamAttr::None)
        newVec.push_back(modVec[modI]);
      ++modI;
    } else {
      // Same index - overwrite or delete existing attributes.
      if (modVec[modI].attrs != ParamAttr::None)
        newVec.push_back(modVec[modI]);
      ++oldI;
      ++modI;
    }
  }

  for (; oldI < oldE; ++oldI)
    newVec.push_back(oldVec[oldI]);
  for (; modI < modE; ++modI)
    if (modVec[modI].attrs != ParamAttr::None)
      newVec.push_back(modVec[modI]);

  return get(newVec);
}

const ParamAttrsList *
ParamAttrsList::includeAttrs(const ParamAttrsList *PAL,
                             uint16_t idx, ParameterAttributes attrs) {
  ParameterAttributes OldAttrs = PAL ? PAL->getParamAttrs(idx) : 
                                       ParamAttr::None;
#ifndef NDEBUG
  // FIXME it is not obvious how this should work for alignment.
  // For now, say we can't change a known alignment.
  ParameterAttributes OldAlign = OldAttrs & ParamAttr::Alignment;
  ParameterAttributes NewAlign = attrs & ParamAttr::Alignment;
  assert((!OldAlign || !NewAlign || OldAlign == NewAlign) &&
         "Attempt to change alignment!");
#endif

  ParameterAttributes NewAttrs = OldAttrs | attrs;
  if (NewAttrs == OldAttrs)
    return PAL;

  ParamAttrsVector modVec(1);
  modVec[0] = ParamAttrsWithIndex::get(idx, NewAttrs);
  return getModified(PAL, modVec);
}

const ParamAttrsList *
ParamAttrsList::excludeAttrs(const ParamAttrsList *PAL,
                             uint16_t idx, ParameterAttributes attrs) {
#ifndef NDEBUG
  // FIXME it is not obvious how this should work for alignment.
  // For now, say we can't pass in alignment, which no current use does.
  assert(!(attrs & ParamAttr::Alignment) && "Attempt to exclude alignment!");
#endif
  ParameterAttributes OldAttrs = PAL ? PAL->getParamAttrs(idx) : 
                                       ParamAttr::None;
  ParameterAttributes NewAttrs = OldAttrs & ~attrs;
  if (NewAttrs == OldAttrs)
    return PAL;

  ParamAttrsVector modVec(1);
  modVec[0] = ParamAttrsWithIndex::get(idx, NewAttrs);
  return getModified(PAL, modVec);
}

ParameterAttributes ParamAttr::typeIncompatible (const Type *Ty) {
  ParameterAttributes Incompatible = None;

  if (!Ty->isInteger())
    // Attributes that only apply to integers.
    Incompatible |= SExt | ZExt;

  if (!isa<PointerType>(Ty))
    // Attributes that only apply to pointers.
    Incompatible |= ByVal | Nest | NoAlias | StructRet;

  return Incompatible;
}
