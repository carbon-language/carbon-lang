//===-- ubsan_diag.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Diagnostics emission for Clang's undefined behavior sanitizer.
//
//===----------------------------------------------------------------------===//
#ifndef UBSAN_DIAG_H
#define UBSAN_DIAG_H

#include "ubsan_value.h"

namespace __ubsan {

/// \brief Representation of an in-flight diagnostic.
///
/// Temporary \c Diag instances are created by the handler routines to
/// accumulate arguments for a diagnostic. The destructor emits the diagnostic
/// message.
class Diag {
  /// The source location at which the problem occurred.
  const SourceLocation &Loc;

  /// The message which will be emitted, with %0, %1, ... placeholders for
  /// arguments.
  const char *Message;

  /// Kinds of arguments, corresponding to members of \c Arg's union.
  enum ArgKind {
    AK_String, ///< A string argument, displayed as-is.
    AK_UInt,   ///< An unsigned integer argument.
    AK_SInt,   ///< A signed integer argument.
    AK_Pointer ///< A pointer argument, displayed in hexadecimal.
  };

  /// An individual diagnostic message argument.
  struct Arg {
    Arg() {}
    Arg(const char *String) : Kind(AK_String), String(String) {}
    Arg(UIntMax UInt) : Kind(AK_UInt), UInt(UInt) {}
    Arg(SIntMax SInt) : Kind(AK_SInt), SInt(SInt) {}
    Arg(const void *Pointer) : Kind(AK_Pointer), Pointer(Pointer) {}

    ArgKind Kind;
    union {
      const char *String;
      UIntMax UInt;
      SIntMax SInt;
      const void *Pointer;
    };
  };

  static const unsigned MaxArgs = 5;

  /// The arguments which have been added to this diagnostic so far.
  Arg Args[MaxArgs];
  unsigned NumArgs;

  Diag &AddArg(Arg A) {
    CHECK(NumArgs != MaxArgs);
    Args[NumArgs++] = A;
    return *this;
  }

  /// \c Diag objects are not copyable.
  Diag(const Diag &); // NOT IMPLEMENTED
  Diag &operator=(const Diag &);

public:
  Diag(const SourceLocation &Loc, const char *Message)
    : Loc(Loc), Message(Message), NumArgs(0) {}
  ~Diag();

  Diag &operator<<(const char *Str) { return AddArg(Str); }
  Diag &operator<<(unsigned long long V) { return AddArg(UIntMax(V)); }
  Diag &operator<<(const void *V) { return AddArg(V); }
  Diag &operator<<(const TypeDescriptor &V);
  Diag &operator<<(const Value &V);
};

} // namespace __ubsan

#endif // UBSAN_DIAG_H
