//===-- RTDyldMemoryManager.cpp - Memory manager for MC-JIT -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the runtime dynamic memory manager base class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdlib>

#ifdef __linux__
  // These includes used by RTDyldMemoryManager::getPointerToNamedFunction()
  // for Glibc trickery. See comments in this function for more information.
  #ifdef HAVE_SYS_STAT_H
    #include <sys/stat.h>
  #endif
  #include <fcntl.h>
  #include <unistd.h>
#endif

namespace llvm {

RTDyldMemoryManager::~RTDyldMemoryManager() {}

// Determine whether we can register EH tables.
#if (defined(__GNUC__) && !defined(__ARM_EABI__) && \
     !defined(__USING_SJLJ_EXCEPTIONS__))
#define HAVE_EHTABLE_SUPPORT 1
#else
#define HAVE_EHTABLE_SUPPORT 0
#endif

#if HAVE_EHTABLE_SUPPORT
extern "C" void __register_frame(void*);

static const char *processFDE(const char *Entry) {
  const char *P = Entry;
  uint32_t Length = *((const uint32_t *)P);
  P += 4;
  uint32_t Offset = *((const uint32_t *)P);
  if (Offset != 0)
    __register_frame(const_cast<char *>(Entry));
  return P + Length;
}
#endif

void RTDyldMemoryManager::registerEHFrames(StringRef SectionData) {
#if HAVE_EHTABLE_SUPPORT
  const char *P = SectionData.data();
  const char *End = SectionData.data() + SectionData.size();
  do  {
    P = processFDE(P);
  } while(P != End);
#endif
}

static int jit_noop() {
  return 0;
}

void *RTDyldMemoryManager::getPointerToNamedFunction(const std::string &Name,
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

} // namespace llvm
