//===-- Target.cpp --------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the common infrastructure (including C bindings) for 
// libLLVMTarget.a, which implements target information.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Target.h"
#include "llvm-c/Initialization.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include <cstring>

using namespace llvm;

inline TargetLibraryInfo *unwrap(LLVMTargetLibraryInfoRef P) {
  return reinterpret_cast<TargetLibraryInfo*>(P);
}

inline LLVMTargetLibraryInfoRef wrap(const TargetLibraryInfo *P) {
  TargetLibraryInfo *X = const_cast<TargetLibraryInfo*>(P);
  return reinterpret_cast<LLVMTargetLibraryInfoRef>(X);
}

void llvm::initializeTarget(PassRegistry &Registry) {
  initializeDataLayoutPassPass(Registry);
  initializeTargetLibraryInfoPass(Registry);
}

void LLVMInitializeTarget(LLVMPassRegistryRef R) {
  initializeTarget(*unwrap(R));
}

LLVMTargetDataRef LLVMCreateTargetData(const char *StringRep) {
  return wrap(new DataLayout(StringRep));
}

void LLVMAddTargetData(LLVMTargetDataRef TD, LLVMPassManagerRef PM) {
  // The DataLayoutPass must now be in sync with the module. Unfortunatelly we
  // cannot enforce that from the C api.
  unwrap(PM)->add(new DataLayoutPass(*unwrap(TD)));
}

void LLVMAddTargetLibraryInfo(LLVMTargetLibraryInfoRef TLI,
                              LLVMPassManagerRef PM) {
  unwrap(PM)->add(new TargetLibraryInfo(*unwrap(TLI)));
}

char *LLVMCopyStringRepOfTargetData(LLVMTargetDataRef TD) {
  std::string StringRep = unwrap(TD)->getStringRepresentation();
  return strdup(StringRep.c_str());
}

LLVMByteOrdering LLVMByteOrder(LLVMTargetDataRef TD) {
  return unwrap(TD)->isLittleEndian() ? LLVMLittleEndian : LLVMBigEndian;
}

unsigned LLVMPointerSize(LLVMTargetDataRef TD) {
  return unwrap(TD)->getPointerSize(0);
}

unsigned LLVMPointerSizeForAS(LLVMTargetDataRef TD, unsigned AS) {
  return unwrap(TD)->getPointerSize(AS);
}

LLVMTypeRef LLVMIntPtrType(LLVMTargetDataRef TD) {
  return wrap(unwrap(TD)->getIntPtrType(getGlobalContext()));
}

LLVMTypeRef LLVMIntPtrTypeForAS(LLVMTargetDataRef TD, unsigned AS) {
  return wrap(unwrap(TD)->getIntPtrType(getGlobalContext(), AS));
}

LLVMTypeRef LLVMIntPtrTypeInContext(LLVMContextRef C, LLVMTargetDataRef TD) {
  return wrap(unwrap(TD)->getIntPtrType(*unwrap(C)));
}

LLVMTypeRef LLVMIntPtrTypeForASInContext(LLVMContextRef C, LLVMTargetDataRef TD, unsigned AS) {
  return wrap(unwrap(TD)->getIntPtrType(*unwrap(C), AS));
}

unsigned long long LLVMSizeOfTypeInBits(LLVMTargetDataRef TD, LLVMTypeRef Ty) {
  return unwrap(TD)->getTypeSizeInBits(unwrap(Ty));
}

unsigned long long LLVMStoreSizeOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty) {
  return unwrap(TD)->getTypeStoreSize(unwrap(Ty));
}

unsigned long long LLVMABISizeOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty) {
  return unwrap(TD)->getTypeAllocSize(unwrap(Ty));
}

unsigned LLVMABIAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty) {
  return unwrap(TD)->getABITypeAlignment(unwrap(Ty));
}

unsigned LLVMCallFrameAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty) {
  return unwrap(TD)->getABITypeAlignment(unwrap(Ty));
}

unsigned LLVMPreferredAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty) {
  return unwrap(TD)->getPrefTypeAlignment(unwrap(Ty));
}

unsigned LLVMPreferredAlignmentOfGlobal(LLVMTargetDataRef TD,
                                        LLVMValueRef GlobalVar) {
  return unwrap(TD)->getPreferredAlignment(unwrap<GlobalVariable>(GlobalVar));
}

unsigned LLVMElementAtOffset(LLVMTargetDataRef TD, LLVMTypeRef StructTy,
                             unsigned long long Offset) {
  StructType *STy = unwrap<StructType>(StructTy);
  return unwrap(TD)->getStructLayout(STy)->getElementContainingOffset(Offset);
}

unsigned long long LLVMOffsetOfElement(LLVMTargetDataRef TD, LLVMTypeRef StructTy,
                                       unsigned Element) {
  StructType *STy = unwrap<StructType>(StructTy);
  return unwrap(TD)->getStructLayout(STy)->getElementOffset(Element);
}

void LLVMDisposeTargetData(LLVMTargetDataRef TD) {
  delete unwrap(TD);
}
