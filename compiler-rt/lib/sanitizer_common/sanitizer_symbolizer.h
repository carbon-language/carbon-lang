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

#include "sanitizer_common.h"
#include "sanitizer_mutex.h"

namespace __sanitizer {

struct AddressInfo {
  // Owns all the string members. Storage for them is
  // (de)allocated using sanitizer internal allocator.
  uptr address;

  char *module;
  uptr module_offset;

  static const uptr kUnknown = ~(uptr)0;
  char *function;
  uptr function_offset;

  char *file;
  int line;
  int column;

  AddressInfo();
  // Deletes all strings and resets all fields.
  void Clear();
  void FillAddressAndModuleInfo(uptr addr, const char *mod_name,
                                uptr mod_offset);
};

// Linked list of symbolized frames (each frame is described by AddressInfo).
struct SymbolizedStack {
  SymbolizedStack *next;
  AddressInfo info;
  static SymbolizedStack *New(uptr addr);
  // Deletes current, and all subsequent frames in the linked list.
  // The object cannot be accessed after the call to this function.
  void ClearAll();

 private:
  SymbolizedStack();
};

// For now, DataInfo is used to describe global variable.
struct DataInfo {
  // Owns all the string members. Storage for them is
  // (de)allocated using sanitizer internal allocator.
  char *module;
  uptr module_offset;
  char *name;
  uptr start;
  uptr size;

  DataInfo();
  void Clear();
};

class Symbolizer {
 public:
  /// Initialize and return platform-specific implementation of symbolizer
  /// (if it wasn't already initialized).
  static Symbolizer *GetOrInit();
  // Returns a list of symbolized frames for a given address (containing
  // all inlined functions, if necessary).
  virtual SymbolizedStack *SymbolizePC(uptr address) {
    return SymbolizedStack::New(address);
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

class ExternalSymbolizerInterface {
 public:
  // Can't declare pure virtual functions in sanitizer runtimes:
  // __cxa_pure_virtual might be unavailable.
  virtual char *SendCommand(bool is_data, const char *module_name,
                            uptr module_offset) {
    UNIMPLEMENTED();
  }
};

// SymbolizerProcess encapsulates communication between the tool and
// external symbolizer program, running in a different subprocess.
// SymbolizerProcess may not be used from two threads simultaneously.
class SymbolizerProcess : public ExternalSymbolizerInterface {
 public:
  explicit SymbolizerProcess(const char *path);
  char *SendCommand(bool is_data, const char *module_name,
                    uptr module_offset) override;

 private:
  bool Restart();
  char *SendCommandImpl(bool is_data, const char *module_name,
                        uptr module_offset);
  bool ReadFromSymbolizer(char *buffer, uptr max_length);
  bool WriteToSymbolizer(const char *buffer, uptr length);
  bool StartSymbolizerSubprocess();

  virtual bool RenderInputCommand(char *buffer, uptr max_length, bool is_data,
                                  const char *module_name,
                                  uptr module_offset) const {
    UNIMPLEMENTED();
  }

  virtual bool ReachedEndOfOutput(const char *buffer, uptr length) const {
    UNIMPLEMENTED();
  }

  virtual void ExecuteWithDefaultArgs(const char *path_to_binary) const {
    UNIMPLEMENTED();
  }

  const char *path_;
  int input_fd_;
  int output_fd_;

  static const uptr kBufferSize = 16 * 1024;
  char buffer_[kBufferSize];

  static const uptr kMaxTimesRestarted = 5;
  static const int kSymbolizerStartupTimeMillis = 10;
  uptr times_restarted_;
  bool failed_to_start_;
  bool reported_invalid_path_;
};

}  // namespace __sanitizer

#endif  // SANITIZER_SYMBOLIZER_H
