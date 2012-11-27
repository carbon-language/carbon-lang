//===- SectionMemoryManager.cpp - Memory manager for MCJIT/RtDyld *- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the section-based memory manager used by the MCJIT
// execution engine and RuntimeDyld
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/MathExtras.h"
#include "SectionMemoryManager.h"

#ifdef __linux__
  // These includes used by SectionMemoryManager::getPointerToNamedFunction()
  // for Glibc trickery. See comments in this function for more information.
  #ifdef HAVE_SYS_STAT_H
    #include <sys/stat.h>
  #endif
  #include <fcntl.h>
  #include <unistd.h>
#endif

namespace llvm {

uint8_t *SectionMemoryManager::allocateDataSection(uintptr_t Size,
                                                    unsigned Alignment,
                                                    unsigned SectionID,
                                                    bool IsReadOnly) {
  if (IsReadOnly)
    return allocateSection(RODataMem, Size, Alignment);
  return allocateSection(RWDataMem, Size, Alignment);
}

uint8_t *SectionMemoryManager::allocateCodeSection(uintptr_t Size,
                                                   unsigned Alignment,
                                                   unsigned SectionID) {
  return allocateSection(CodeMem, Size, Alignment);
}

uint8_t *SectionMemoryManager::allocateSection(MemoryGroup &MemGroup,
                                               uintptr_t Size,
                                               unsigned Alignment) {
  if (!Alignment)
    Alignment = 16;

  assert(!(Alignment & (Alignment - 1)) && "Alignment must be a power of two.");

  uintptr_t RequiredSize = Alignment * ((Size + Alignment - 1)/Alignment + 1);
  uintptr_t Addr = 0;

  // Look in the list of free memory regions and use a block there if one
  // is available.
  for (int i = 0, e = MemGroup.FreeMem.size(); i != e; ++i) {
    sys::MemoryBlock &MB = MemGroup.FreeMem[i];
    if (MB.size() >= RequiredSize) {
      Addr = (uintptr_t)MB.base();
      uintptr_t EndOfBlock = Addr + MB.size();
      // Align the address.
      Addr = (Addr + Alignment - 1) & ~(uintptr_t)(Alignment - 1);
      // Store cutted free memory block.
      MemGroup.FreeMem[i] = sys::MemoryBlock((void*)(Addr + Size),
                                             EndOfBlock - Addr - Size);
      return (uint8_t*)Addr;
    }
  }

  // No pre-allocated free block was large enough. Allocate a new memory region.
  // Note that all sections get allocated as read-write.  The permissions will
  // be updated later based on memory group.
  //
  // FIXME: It would be useful to define a default allocation size (or add
  // it as a constructor parameter) to minimize the number of allocations.
  // 
  // FIXME: Initialize the Near member for each memory group to avoid
  // interleaving.
  error_code ec;
  sys::MemoryBlock MB = sys::Memory::allocateMappedMemory(RequiredSize,
                                                          &MemGroup.Near,
                                                          sys::Memory::MF_READ |
                                                            sys::Memory::MF_WRITE,
                                                          ec);
  if (ec) {
    // FIXME: Add error propogation to the interface.
    return NULL;
  }

  // Save this address as the basis for our next request
  MemGroup.Near = MB;

  MemGroup.AllocatedMem.push_back(MB);
  Addr = (uintptr_t)MB.base();
  uintptr_t EndOfBlock = Addr + MB.size();

  // Align the address.
  Addr = (Addr + Alignment - 1) & ~(uintptr_t)(Alignment - 1);

  // The allocateMappedMemory may allocate much more memory than we need. In
  // this case, we store the unused memory as a free memory block.
  unsigned FreeSize = EndOfBlock-Addr-Size;
  if (FreeSize > 16)
    MemGroup.FreeMem.push_back(sys::MemoryBlock((void*)(Addr + Size), FreeSize));

  // Return aligned address
  return (uint8_t*)Addr;
}

bool SectionMemoryManager::applyPermissions(std::string *ErrMsg)
{
  // FIXME: Should in-progress permissions be reverted if an error occurs?
  error_code ec;

  // Make code memory executable.
  ec = applyMemoryGroupPermissions(CodeMem,
                                   sys::Memory::MF_READ | sys::Memory::MF_EXEC);
  if (ec) {
    if (ErrMsg) {
      *ErrMsg = ec.message();
    }
    return true;
  }

  // Make read-only data memory read-only.
  ec = applyMemoryGroupPermissions(RODataMem,
                                   sys::Memory::MF_READ | sys::Memory::MF_EXEC);
  if (ec) {
    if (ErrMsg) {
      *ErrMsg = ec.message();
    }
    return true;
  }

  // Read-write data memory already has the correct permissions

  return false;
}

error_code SectionMemoryManager::applyMemoryGroupPermissions(MemoryGroup &MemGroup,
                                                             unsigned Permissions) {

  for (int i = 0, e = MemGroup.AllocatedMem.size(); i != e; ++i) {
      error_code ec;
      ec = sys::Memory::protectMappedMemory(MemGroup.AllocatedMem[i],
                                            Permissions);
      if (ec) {
        return ec;
      }
  }

  return error_code::success();
}

void SectionMemoryManager::invalidateInstructionCache() {
  for (int i = 0, e = CodeMem.AllocatedMem.size(); i != e; ++i)
    sys::Memory::InvalidateInstructionCache(CodeMem.AllocatedMem[i].base(),
                                            CodeMem.AllocatedMem[i].size());
}

static int jit_noop() {
  return 0;
}

void *SectionMemoryManager::getPointerToNamedFunction(const std::string &Name,
                                                       bool AbortOnFailure) {
#if defined(__linux__)
  //===--------------------------------------------------------------------===//
  // Function stubs that are invoked instead of certain library calls
  //
  // Force the following functions to be linked in to anything that uses the
  // JIT. This is a hack designed to work around the all-too-clever Glibc
  // strategy of making these functions work differently when inlined vs. when
  // not inlined, and hiding their real definitions in a separate archive file
  // that the dynamic linker can't see. For more info, search for
  // 'libc_nonshared.a' on Google, or read http://llvm.org/PR274.
  if (Name == "stat") return (void*)(intptr_t)&stat;
  if (Name == "fstat") return (void*)(intptr_t)&fstat;
  if (Name == "lstat") return (void*)(intptr_t)&lstat;
  if (Name == "stat64") return (void*)(intptr_t)&stat64;
  if (Name == "fstat64") return (void*)(intptr_t)&fstat64;
  if (Name == "lstat64") return (void*)(intptr_t)&lstat64;
  if (Name == "atexit") return (void*)(intptr_t)&atexit;
  if (Name == "mknod") return (void*)(intptr_t)&mknod;
#endif // __linux__

  // We should not invoke parent's ctors/dtors from generated main()!
  // On Mingw and Cygwin, the symbol __main is resolved to
  // callee's(eg. tools/lli) one, to invoke wrong duplicated ctors
  // (and register wrong callee's dtors with atexit(3)).
  // We expect ExecutionEngine::runStaticConstructorsDestructors()
  // is called before ExecutionEngine::runFunctionAsMain() is called.
  if (Name == "__main") return (void*)(intptr_t)&jit_noop;

  const char *NameStr = Name.c_str();
  void *Ptr = sys::DynamicLibrary::SearchForAddressOfSymbol(NameStr);
  if (Ptr) return Ptr;

  // If it wasn't found and if it starts with an underscore ('_') character,
  // try again without the underscore.
  if (NameStr[0] == '_') {
    Ptr = sys::DynamicLibrary::SearchForAddressOfSymbol(NameStr+1);
    if (Ptr) return Ptr;
  }

  if (AbortOnFailure)
    report_fatal_error("Program used external function '" + Name +
                      "' which could not be resolved!");
  return 0;
}

SectionMemoryManager::~SectionMemoryManager() {
  for (unsigned i = 0, e = CodeMem.AllocatedMem.size(); i != e; ++i)
    sys::Memory::releaseMappedMemory(CodeMem.AllocatedMem[i]);
  for (unsigned i = 0, e = RWDataMem.AllocatedMem.size(); i != e; ++i)
    sys::Memory::releaseMappedMemory(RWDataMem.AllocatedMem[i]);
  for (unsigned i = 0, e = RODataMem.AllocatedMem.size(); i != e; ++i)
    sys::Memory::releaseMappedMemory(RODataMem.AllocatedMem[i]);
}

} // namespace llvm

