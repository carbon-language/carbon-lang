//===-- sanitizer_symbolizer.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Symbolizer is used by sanitizers to map instruction address to a location in
// source code at run-time. Symbolizer either uses __sanitizer_symbolize_*
// defined in the program, or (if they are missing) tries to find and
// launch "llvm-symbolizer" commandline tool in a separate process and
// communicate with it.
//
// Generally we should try to avoid calling system library functions during
// symbolization (and use their replacements from sanitizer_libc.h instead).
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_SYMBOLIZER_H
#define SANITIZER_SYMBOLIZER_H

#include "sanitizer_allocator_internal.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_libc.h"

namespace __sanitizer {

struct AddressInfo {
  uptr address;

  char *module;
  uptr module_offset;

  static const uptr kUnknown = ~(uptr)0;
  char *function;
  uptr function_offset;

  char *file;
  int line;
  int column;

  AddressInfo() {
    internal_memset(this, 0, sizeof(AddressInfo));
    function_offset = kUnknown;
  }

  // Deletes all strings and resets all fields.
  void Clear() {
    InternalFree(module);
    InternalFree(function);
    InternalFree(file);
    internal_memset(this, 0, sizeof(AddressInfo));
    function_offset = kUnknown;
  }

  void FillAddressAndModuleInfo(uptr addr, const char *mod_name,
                                uptr mod_offset) {
    address = addr;
    module = internal_strdup(mod_name);
    module_offset = mod_offset;
  }
};

struct DataInfo {
  uptr address;
  char *module;
  uptr module_offset;
  char *name;
  uptr start;
  uptr size;
};

class Symbolizer {
 public:
  /// Initialize and return platform-specific implementation of symbolizer
  /// (if it wasn't already initialized).
  static Symbolizer *GetOrInit();
  // Fills at most "max_frames" elements of "frames" with descriptions
  // for a given address (in all inlined functions). Returns the number
  // of descriptions actually filled.
  virtual uptr SymbolizePC(uptr address, AddressInfo *frames, uptr max_frames) {
    return 0;
  }
  virtual bool SymbolizeData(uptr address, DataInfo *info) {
    return false;
  }
  virtual bool GetModuleNameAndOffsetForPC(uptr pc, const char **module_name,
                                           uptr *module_address) {
    return false;
  }
  virtual bool CanReturnFileLineInfo() {
    return false;
  }
  // Release internal caches (if any).
  virtual void Flush() {}
  // Attempts to demangle the provided C++ mangled name.
  virtual const char *Demangle(const char *name) {
    return name;
  }
  virtual void PrepareForSandboxing() {}

  // Allow user to install hooks that would be called before/after Symbolizer
  // does the actual file/line info fetching. Specific sanitizers may need this
  // to distinguish system library calls made in user code from calls made
  // during in-process symbolization.
  typedef void (*StartSymbolizationHook)();
  typedef void (*EndSymbolizationHook)();
  // May be called at most once.
  void AddHooks(StartSymbolizationHook start_hook,
                EndSymbolizationHook end_hook);

 private:
  /// Platform-specific function for creating a Symbolizer object.
  static Symbolizer *PlatformInit();
  /// Initialize the symbolizer in a disabled state.  Not thread safe.
  static Symbolizer *Disable();

  static Symbolizer *symbolizer_;
  static StaticSpinMutex init_mu_;

 protected:
  Symbolizer();

  static LowLevelAllocator symbolizer_allocator_;

  StartSymbolizationHook start_hook_;
  EndSymbolizationHook end_hook_;
  class SymbolizerScope {
   public:
    explicit SymbolizerScope(const Symbolizer *sym);
    ~SymbolizerScope();
   private:
    const Symbolizer *sym_;
  };
};

}  // namespace __sanitizer

#endif  // SANITIZER_SYMBOLIZER_H
