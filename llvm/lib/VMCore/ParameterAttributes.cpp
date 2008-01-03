//===-- ParameterAttributes.cpp - Implement ParameterAttrs ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ParamAttrsList class.
//
//===----------------------------------------------------------------------===//

#include "llvm/ParameterAttributes.h"
#include "llvm/Support/ManagedStatic.h"

using namespace llvm;

static ManagedStatic<FoldingSet<ParamAttrsList> > ParamAttrsLists;

ParamAttrsList::ParamAttrsList(const ParamAttrsVector &attrVec) 
  : attrs(attrVec), refCount(0) {
}

ParamAttrsList::~ParamAttrsList() {
  ParamAttrsLists->RemoveNode(this);
}

uint16_t
ParamAttrsList::getParamAttrs(uint16_t Index) const {
  unsigned limit = attrs.size();
  for (unsigned i = 0; i < limit && attrs[i].index <= Index; ++i)
    if (attrs[i].index == Index)
      return attrs[i].attrs;
  return ParamAttr::None;
}

std::string 
ParamAttrsList::getParamAttrsText(uint16_t Attrs) {
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
  return Result;
}

/// onlyInformative - Returns whether only informative attributes are set.
static inline bool onlyInformative(uint16_t attrs) {
  return !(attrs & ~ParamAttr::Informative);
}

bool
ParamAttrsList::areCompatible(const ParamAttrsList *A, const ParamAttrsList *B){
  if (A == B)
    return true;
  unsigned ASize = A ? A->size() : 0;
  unsigned BSize = B ? B->size() : 0;
  unsigned AIndex = 0;
  unsigned BIndex = 0;

  while (AIndex < ASize && BIndex < BSize) {
    uint16_t AIdx = A->getParamIndex(AIndex);
    uint16_t BIdx = B->getParamIndex(BIndex);
    uint16_t AAttrs = A->getParamAttrsAtIndex(AIndex);
    uint16_t BAttrs = B->getParamAttrsAtIndex(AIndex);

    if (AIdx < BIdx) {
      if (!onlyInformative(AAttrs))
        return false;
      ++AIndex;
    } else if (BIdx < AIdx) {
      if (!onlyInformative(BAttrs))
        return false;
      ++BIndex;
    } else {
      if (!onlyInformative(AAttrs ^ BAttrs))
        return false;
      ++AIndex;
      ++BIndex;
    }
  }
  for (; AIndex < ASize; ++AIndex)
    if (!onlyInformative(A->getParamAttrsAtIndex(AIndex)))
      return false;
  for (; BIndex < BSize; ++BIndex)
    if (!onlyInformative(B->getParamAttrsAtIndex(AIndex)))
      return false;
  return true;
}

void ParamAttrsList::Profile(FoldingSetNodeID &ID,
                             const ParamAttrsVector &Attrs) {
  for (unsigned i = 0; i < Attrs.size(); ++i)
    ID.AddInteger(unsigned(Attrs[i].attrs) << 16 | unsigned(Attrs[i].index));
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
                             uint16_t idx, uint16_t attrs) {
  uint16_t OldAttrs = PAL ? PAL->getParamAttrs(idx) : 0;
  uint16_t NewAttrs = OldAttrs | attrs;
  if (NewAttrs == OldAttrs)
    return PAL;

  ParamAttrsVector modVec;
  modVec.push_back(ParamAttrsWithIndex::get(idx, NewAttrs));
  return getModified(PAL, modVec);
}

const ParamAttrsList *
ParamAttrsList::excludeAttrs(const ParamAttrsList *PAL,
                             uint16_t idx, uint16_t attrs) {
  uint16_t OldAttrs = PAL ? PAL->getParamAttrs(idx) : 0;
  uint16_t NewAttrs = OldAttrs & ~attrs;
  if (NewAttrs == OldAttrs)
    return PAL;

  ParamAttrsVector modVec;
  modVec.push_back(ParamAttrsWithIndex::get(idx, NewAttrs));
  return getModified(PAL, modVec);
}

