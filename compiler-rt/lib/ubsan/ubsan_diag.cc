//===-- ubsan_diag.cc -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Diagnostic reporting for the UBSan runtime.
//
//===----------------------------------------------------------------------===//

#include "ubsan_diag.h"
#include <stdio.h>
#include <unistd.h>
#include <limits.h>

using namespace __ubsan;

Diag &Diag::operator<<(const TypeDescriptor &V) {
  return AddArg(V.getTypeName());
}

Diag &Diag::operator<<(const Value &V) {
  if (V.getType().isSignedIntegerTy())
    AddArg(V.getSIntValue());
  else if (V.getType().isUnsignedIntegerTy())
    AddArg(V.getUIntValue());
  else if (V.getType().isFloatTy())
    AddArg(V.getFloatValue());
  else
    AddArg("<unknown>");
  return *this;
}

/// Hexadecimal printing for numbers too large for fprintf to handle directly.
static void PrintHex(UIntMax Val) {
#if HAVE_INT128_T
  fprintf(stderr, "0x%08x%08x%08x%08x",
          (unsigned int)(Val >> 96),
          (unsigned int)(Val >> 64),
          (unsigned int)(Val >> 32),
          (unsigned int)(Val));
#else
  UNREACHABLE("long long smaller than 64 bits?");
#endif
}

Diag::~Diag() {
  // FIXME: This is non-portable.
  bool UseAnsiColor = isatty(STDERR_FILENO);
  if (UseAnsiColor)
    fprintf(stderr, "\033[1m");
  if (Loc.isInvalid())
    fprintf(stderr, "<unknown>:");
  else {
    fprintf(stderr, "%s:%d:", Loc.getFilename(), Loc.getLine());
    if (Loc.getColumn())
      fprintf(stderr, "%d:", Loc.getColumn());
  }
  if (UseAnsiColor)
    fprintf(stderr, "\033[31m");
  fprintf(stderr, " fatal error: ");
  if (UseAnsiColor)
    fprintf(stderr, "\033[0;1m");
  for (const char *Msg = Message; *Msg; ++Msg) {
    if (*Msg != '%')
      fputc((unsigned char)*Msg, stderr);
    else {
      const Arg &A = Args[*++Msg - '0'];
      switch (A.Kind) {
      case AK_String:
        fprintf(stderr, "%s", A.String);
        break;
      case AK_SInt:
        // 'long long' is guaranteed to be at least 64 bits wide.
        if (A.SInt >= INT64_MIN && A.SInt <= INT64_MAX)
          fprintf(stderr, "%lld", (long long)A.SInt);
        else
          PrintHex(A.SInt);
        break;
      case AK_UInt:
        if (A.UInt <= UINT64_MAX)
          fprintf(stderr, "%llu", (unsigned long long)A.UInt);
        else
          PrintHex(A.UInt);
        break;
      case AK_Float:
        fprintf(stderr, "%Lg", (long double)A.Float);
        break;
      case AK_Pointer:
        fprintf(stderr, "0x%zx", (uptr)A.Pointer);
        break;
      }
    }
  }
  fputc('\n', stderr);
  if (UseAnsiColor)
    fprintf(stderr, "\033[0m");
  fflush(stderr);
}
