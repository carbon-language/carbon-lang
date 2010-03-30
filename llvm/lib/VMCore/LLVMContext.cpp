//===-- LLVMContext.cpp - Implement LLVMContext -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements LLVMContext, as a wrapper around the opaque
//  class LLVMContextImpl.
//
//===----------------------------------------------------------------------===//

#include "llvm/LLVMContext.h"
#include "llvm/Metadata.h"
#include "llvm/Constants.h"
#include "llvm/Instruction.h"
#include "llvm/Support/ManagedStatic.h"
#include "LLVMContextImpl.h"
using namespace llvm;

static ManagedStatic<LLVMContext> GlobalContext;

LLVMContext& llvm::getGlobalContext() {
  return *GlobalContext;
}

LLVMContext::LLVMContext() : pImpl(new LLVMContextImpl(*this)) {
  // Create the first metadata kind, which is always 'dbg'.
  unsigned DbgID = getMDKindID("dbg");
  assert(DbgID == MD_dbg && "dbg kind id drifted"); (void)DbgID;
}
LLVMContext::~LLVMContext() { delete pImpl; }


#ifndef NDEBUG
/// isValidName - Return true if Name is a valid custom metadata handler name.
static bool isValidName(StringRef MDName) {
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
#endif

/// getMDKindID - Return a unique non-zero ID for the specified metadata kind.
unsigned LLVMContext::getMDKindID(StringRef Name) const {
  assert(isValidName(Name) && "Invalid MDNode name");
  
  unsigned &Entry = pImpl->CustomMDKindNames[Name];
  
  // If this is new, assign it its ID.
  if (Entry == 0) Entry = pImpl->CustomMDKindNames.size();
  return Entry;
}

/// getHandlerNames - Populate client supplied smallvector using custome
/// metadata name and ID.
void LLVMContext::getMDKindNames(SmallVectorImpl<StringRef> &Names) const {
  Names.resize(pImpl->CustomMDKindNames.size()+1);
  Names[0] = "";
  for (StringMap<unsigned>::const_iterator I = pImpl->CustomMDKindNames.begin(),
       E = pImpl->CustomMDKindNames.end(); I != E; ++I)
    // MD Handlers are numbered from 1.
    Names[I->second] = I->first();
}


