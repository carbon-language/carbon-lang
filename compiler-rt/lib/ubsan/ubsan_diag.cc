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
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include <stdio.h>

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
  Printf("0x%08x%08x%08x%08x",
          (unsigned int)(Val >> 96),
          (unsigned int)(Val >> 64),
          (unsigned int)(Val >> 32),
          (unsigned int)(Val));
#else
  UNREACHABLE("long long smaller than 64 bits?");
#endif
}

Diag::~Diag() {
  bool UseAnsiColor = PrintsToTty();
  if (UseAnsiColor)
    RawWrite("\033[1m");
  if (Loc.isInvalid())
    RawWrite("<unknown>:");
  else {
    Printf("%s:%d:", Loc.getFilename(), Loc.getLine());
    if (Loc.getColumn())
      Printf("%d:", Loc.getColumn());
  }
  if (UseAnsiColor)
    RawWrite("\033[31m");
  RawWrite(" runtime error: ");
  if (UseAnsiColor)
    RawWrite("\033[0;1m");
  for (const char *Msg = Message; *Msg; ++Msg) {
    if (*Msg != '%') {
      char Buffer[64];
      unsigned I;
      for (I = 0; Msg[I] && Msg[I] != '%' && I != 63; ++I)
        Buffer[I] = Msg[I];
      Buffer[I] = '\0';
      RawWrite(Buffer);
      Msg += I - 1;
    } else {
      const Arg &A = Args[*++Msg - '0'];
      switch (A.Kind) {
      case AK_String:
        Printf("%s", A.String);
        break;
      case AK_SInt:
        // 'long long' is guaranteed to be at least 64 bits wide.
        if (A.SInt >= INT64_MIN && A.SInt <= INT64_MAX)
          Printf("%lld", (long long)A.SInt);
        else
          PrintHex(A.SInt);
        break;
      case AK_UInt:
        if (A.UInt <= UINT64_MAX)
          Printf("%llu", (unsigned long long)A.UInt);
        else
          PrintHex(A.UInt);
        break;
      case AK_Float: {
        // FIXME: Support floating-point formatting in sanitizer_common's
        //        printf, and stop using snprintf here.
        char Buffer[32];
        snprintf(Buffer, sizeof(Buffer), "%Lg", (long double)A.Float);
        Printf("%s", Buffer);
        break;
      }
      case AK_Pointer:
        Printf("0x%zx", (uptr)A.Pointer);
        break;
      }
    }
  }
  RawWrite("\n");
  if (UseAnsiColor)
    Printf("\033[0m");
}
