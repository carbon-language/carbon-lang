//===-- sanitizer_symbolizer_internal.h -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Header for internal classes and functions to be used by implementations of
// symbolizers.
//
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_SYMBOLIZER_INTERNAL_H
#define SANITIZER_SYMBOLIZER_INTERNAL_H

#include "sanitizer_symbolizer.h"

namespace __sanitizer {

// Parsing helpers, 'str' is searched for delimiter(s) and a string or uptr
// is extracted. When extracting a string, a newly allocated (using
// InternalAlloc) and null-terminataed buffer is returned. They return a pointer
// to the next characted after the found delimiter.
const char *ExtractToken(const char *str, const char *delims, char **result);
const char *ExtractInt(const char *str, const char *delims, int *result);
const char *ExtractUptr(const char *str, const char *delims, uptr *result);

class SymbolizerTool {
 public:
  // Can't declare pure virtual functions in sanitizer runtimes:
  // __cxa_pure_virtual might be unavailable.

  // The |stack| parameter is inout. It is pre-filled with the address,
  // module base and module offset values and is to be used to construct
  // other stack frames.
  virtual bool SymbolizePC(uptr addr, SymbolizedStack *stack) {
    UNIMPLEMENTED();
  }

  // The |info| parameter is inout. It is pre-filled with the module base
  // and module offset values.
  virtual bool SymbolizeData(uptr addr, DataInfo *info) {
    UNIMPLEMENTED();
  }

  virtual void Flush() {}

  // Return nullptr to fallback to the default __cxxabiv1 demangler.
  virtual const char *Demangle(const char *name) {
    return nullptr;
  }
};

// SymbolizerProcess encapsulates communication between the tool and
// external symbolizer program, running in a different subprocess.
// SymbolizerProcess may not be used from two threads simultaneously.
class SymbolizerProcess {
 public:
  explicit SymbolizerProcess(const char *path);
  const char *SendCommand(const char *command);

 private:
  bool Restart();
  const char *SendCommandImpl(const char *command);
  bool ReadFromSymbolizer(char *buffer, uptr max_length);
  bool WriteToSymbolizer(const char *buffer, uptr length);
  bool StartSymbolizerSubprocess();

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

#endif  // SANITIZER_SYMBOLIZER_INTERNAL_H
