//===--- InputInfo.h - Input Source & Type Information ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_DRIVER_INPUTINFO_H_
#define CLANG_LIB_DRIVER_INPUTINFO_H_

#include <cassert>

namespace clang {
namespace driver {
  class PipedJob;

/// InputInfo - Wrapper for information about an input source.
class InputInfo {
  union {
    const char *Filename;
    PipedJob *Pipe;
  } Data;
  bool IsPipe;
  types::ID Type;
  const char *BaseInput;

public:
  InputInfo() {}
  InputInfo(const char *Filename, types::ID _Type, const char *_BaseInput)
    : IsPipe(false), Type(_Type), BaseInput(_BaseInput) {
    Data.Filename = Filename;
  }
  InputInfo(PipedJob *Pipe, types::ID _Type, const char *_BaseInput)
    : IsPipe(true), Type(_Type), BaseInput(_BaseInput) {
    Data.Pipe = Pipe;
  }

  bool isPipe() const { return IsPipe; }
  types::ID getType() const { return Type; }
  const char *getBaseInput() const { return BaseInput; }

  const char *getInputFilename() const {
    assert(!isPipe() && "Invalid accessor.");
    return Data.Filename;
  }
  PipedJob &getPipe() const {
    assert(isPipe() && "Invalid accessor.");
    return *Data.Pipe;
  }
};

} // end namespace driver
} // end namespace clang

#endif
