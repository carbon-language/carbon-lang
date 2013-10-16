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
#if (defined(__GNUC__) && !defined(__ARM_EABI__) && !defined(__ia64__) && \
     !defined(__USING_SJLJ_EXCEPTIONS__))
#define HAVE_EHTABLE_SUPPORT 1
#else
#define HAVE_EHTABLE_SUPPORT 0
#endif

#if HAVE_EHTABLE_SUPPORT
extern "C" void __register_frame(void*);
extern "C" void __deregister_frame(void*);
#else
// The building compiler does not have __(de)register_frame but
// it may be found at runtime in a dynamically-loaded library.
// For example, this happens when building LLVM with Visual C++
// but using the MingW runtime.
void __register_frame(void *p) {
  static bool Searched = false;
  static void *rf = 0;

  if (!Searched) {
    Searched = true;
    rf = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(
                                                      "__register_frame");
  }
  if (rf)
    ((void (*)(void *))rf)(p);
}

void __deregister_frame(void *p) {
  static bool Searched = false;
  static void *df = 0;

  if (!Searched) {
    Searched = true;
    df = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(
                                                      "__deregister_frame");
  }
  if (df)
    ((void (*)(void *))df)(p);
}
#endif

#ifdef __APPLE__

static const char *processFDE(const char *Entry, bool isDeregister) {
  const char *P = Entry;
  uint32_t Length = *((const uint32_t *)P);
  P += 4;
  uint32_t Offset = *((const uint32_t *)P);
  if (Offset != 0)
    if (isDeregister)
      __deregister_frame(const_cast<char *>(Entry));
    else
      __register_frame(const_cast<char *>(Entry));
  return P + Length;
}

// This implementation handles frame registration for local targets.
// Memory managers for remote targets should re-implement this function
// and use the LoadAddr parameter.
void RTDyldMemoryManager::registerEHFrames(uint8_t *Addr,
                                           uint64_t LoadAddr,
                                           size_t Size) {
  // On OS X OS X __register_frame takes a single FDE as an argument.
  // See http://lists.cs.uiuc.edu/pipermail/llvmdev/2013-April/061768.html
  const char *P = (const char *)Addr;
  const char *End = P + Size;
  do  {
    P = processFDE(P, false);
  } while(P != End);
}

void RTDyldMemoryManager::deregisterEHFrames(uint8_t *Addr,
                                           uint64_t LoadAddr,
                                           size_t Size) {
  const char *P = (const char *)Addr;
  const char *End = P + Size;
  do  {
    P = processFDE(P, true);
  } while(P != End);
}

#else

void RTDyldMemoryManager::registerEHFrames(uint8_t *Addr,
                                           uint64_t LoadAddr,
                                           size_t Size) {
  // On Linux __register_frame takes a single argument: 
  // a pointer to the start of the .eh_frame section.

  // How can it find the end? Because crtendS.o is linked 
  // in and it has an .eh_frame section with four zero chars.
  // FIXME: make sure EH frame is followed by four zero bytes.
  // This should be done in the linker RuntimeDyldELF::getEHFrameSection(),
  // return pointer to .eh_frame properly appended by four zero bytes.
  // If the linker can not fixed, do it here.
  __register_frame(Addr);
}

void RTDyldMemoryManager::deregisterEHFrames(uint8_t *Addr,
                                           uint64_t LoadAddr,
                                           size_t Size) {
  __deregister_frame(Addr);
}

#endif

static int jit_noop() {
  return 0;
}

uint64_t RTDyldMemoryManager::getSymbolAddress(const std::string &Name) {
  // This implementation assumes that the host program is the target.
  // Clients generating code for a remote target should implement their own
  // memory manager.
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
  if (Name == "stat") return (uint64_t)&stat;
  if (Name == "fstat") return (uint64_t)&fstat;
  if (Name == "lstat") return (uint64_t)&lstat;
  if (Name == "stat64") return (uint64_t)&stat64;
  if (Name == "fstat64") return (uint64_t)&fstat64;
  if (Name == "lstat64") return (uint64_t)&lstat64;
  if (Name == "atexit") return (uint64_t)&atexit;
  if (Name == "mknod") return (uint64_t)&mknod;
#endif // __linux__

  // We should not invoke parent's ctors/dtors from generated main()!
  // On Mingw and Cygwin, the symbol __main is resolved to
  // callee's(eg. tools/lli) one, to invoke wrong duplicated ctors
  // (and register wrong callee's dtors with atexit(3)).
  // We expect ExecutionEngine::runStaticConstructorsDestructors()
  // is called before ExecutionEngine::runFunctionAsMain() is called.
  if (Name == "__main") return (uint64_t)&jit_noop;

  const char *NameStr = Name.c_str();
  void *Ptr = sys::DynamicLibrary::SearchForAddressOfSymbol(NameStr);
  if (Ptr)
    return (uint64_t)Ptr;

  // If it wasn't found and if it starts with an underscore ('_') character,
  // try again without the underscore.
  if (NameStr[0] == '_') {
    Ptr = sys::DynamicLibrary::SearchForAddressOfSymbol(NameStr+1);
    if (Ptr)
      return (uint64_t)Ptr;
  }
  return 0;
}

void *RTDyldMemoryManager::getPointerToNamedFunction(const std::string &Name,
                                                     bool AbortOnFailure) {
  uint64_t Addr = getSymbolAddress(Name);

  if (!Addr && AbortOnFailure)
    report_fatal_error("Program used external function '" + Name +
                       "' which could not be resolved!");
  return (void*)Addr;
}

} // namespace llvm
