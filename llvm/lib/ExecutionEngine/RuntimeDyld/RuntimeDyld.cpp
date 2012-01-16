//===-- RuntimeDyld.cpp - Run-time dynamic linker for MC-JIT ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the MC-JIT runtime dynamic linker.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "dyld"
#include "RuntimeDyldImpl.h"
#include "llvm/Support/Path.h"
using namespace llvm;
using namespace llvm::object;

// Empty out-of-line virtual destructor as the key function.
RTDyldMemoryManager::~RTDyldMemoryManager() {}
RuntimeDyldImpl::~RuntimeDyldImpl() {}

namespace llvm {

void RuntimeDyldImpl::extractFunction(StringRef Name, uint8_t *StartAddress,
                                      uint8_t *EndAddress) {
  // Allocate memory for the function via the memory manager.
  uintptr_t Size = EndAddress - StartAddress + 1;
  uintptr_t AllocSize = Size;
  uint8_t *Mem = MemMgr->startFunctionBody(Name.data(), AllocSize);
  assert(Size >= (uint64_t)(EndAddress - StartAddress + 1) &&
         "Memory manager failed to allocate enough memory!");
  // Copy the function payload into the memory block.
  memcpy(Mem, StartAddress, Size);
  MemMgr->endFunctionBody(Name.data(), Mem, Mem + Size);
  // Remember where we put it.
  Functions[Name] = sys::MemoryBlock(Mem, Size);
  // Default the assigned address for this symbol to wherever this
  // allocated it.
  SymbolTable[Name] = Mem;
  DEBUG(dbgs() << "    allocated to [" << Mem << ", " << Mem + Size << "]\n");
}

// Resolve the relocations for all symbols we currently know about.
void RuntimeDyldImpl::resolveRelocations() {
  // Just iterate over the symbols in our symbol table and assign their
  // addresses.
  StringMap<uint8_t*>::iterator i = SymbolTable.begin();
  StringMap<uint8_t*>::iterator e = SymbolTable.end();
  for (;i != e; ++i)
    reassignSymbolAddress(i->getKey(), i->getValue());
}

//===----------------------------------------------------------------------===//
// RuntimeDyld class implementation
RuntimeDyld::RuntimeDyld(RTDyldMemoryManager *mm) {
  Dyld = 0;
  MM = mm;
}

RuntimeDyld::~RuntimeDyld() {
  delete Dyld;
}

bool RuntimeDyld::loadObject(MemoryBuffer *InputBuffer) {
  if (!Dyld) {
    sys::LLVMFileType type = sys::IdentifyFileType(
            InputBuffer->getBufferStart(),
            static_cast<unsigned>(InputBuffer->getBufferSize()));
    switch (type) {
      case sys::ELF_Relocatable_FileType:
      case sys::ELF_Executable_FileType:
      case sys::ELF_SharedObject_FileType:
      case sys::ELF_Core_FileType:
        Dyld = new RuntimeDyldELF(MM);
        break;
      case sys::Mach_O_Object_FileType:
      case sys::Mach_O_Executable_FileType:
      case sys::Mach_O_FixedVirtualMemorySharedLib_FileType:
      case sys::Mach_O_Core_FileType:
      case sys::Mach_O_PreloadExecutable_FileType:
      case sys::Mach_O_DynamicallyLinkedSharedLib_FileType:
      case sys::Mach_O_DynamicLinker_FileType:
      case sys::Mach_O_Bundle_FileType:
      case sys::Mach_O_DynamicallyLinkedSharedLibStub_FileType:
      case sys::Mach_O_DSYMCompanion_FileType:
        Dyld = new RuntimeDyldMachO(MM);
        break;
      case sys::Unknown_FileType:
      case sys::Bitcode_FileType:
      case sys::Archive_FileType:
      case sys::COFF_FileType:
        report_fatal_error("Incompatible object format!");
    }
  } else {
    if (!Dyld->isCompatibleFormat(InputBuffer))
      report_fatal_error("Incompatible object format!");
  }

  return Dyld->loadObject(InputBuffer);
}

void *RuntimeDyld::getSymbolAddress(StringRef Name) {
  return Dyld->getSymbolAddress(Name);
}

void RuntimeDyld::resolveRelocations() {
  Dyld->resolveRelocations();
}

void RuntimeDyld::reassignSymbolAddress(StringRef Name, uint8_t *Addr) {
  Dyld->reassignSymbolAddress(Name, Addr);
}

StringRef RuntimeDyld::getErrorString() {
  return Dyld->getErrorString();
}

} // end namespace llvm
