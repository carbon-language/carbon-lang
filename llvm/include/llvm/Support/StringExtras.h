//===-- StringExtras.h - Useful string functions -----------------*- C++ -*--=//
//
// This file contains some functions that are useful when dealing with strings.
// No library is required when using these functinons.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_STRING_EXTRAS_H
#define LLVM_TOOLS_STRING_EXTRAS_H

#include <string>
#include <stdio.h>
#include "llvm/Support/DataTypes.h"

static inline string utostr(uint64_t X, bool isNeg = false) {
  char Buffer[40];
  char *BufPtr = Buffer+39;

  *BufPtr = 0;                  // Null terminate buffer...
  if (X == 0) *--BufPtr = '0';  // Handle special case...

  while (X) {
    *--BufPtr = '0' + (X % 10);
    X /= 10;
  }

  if (isNeg) *--BufPtr = '-';   // Add negative sign...

  return string(BufPtr);
}

static inline string itostr(int64_t X) {
  if (X < 0) 
    return utostr((uint64_t)-X, true);
  else
    return utostr((uint64_t)X);
}


static inline string utostr(unsigned X, bool isNeg = false) {
  char Buffer[20];
  char *BufPtr = Buffer+19;

  *BufPtr = 0;                  // Null terminate buffer...
  if (X == 0) *--BufPtr = '0';  // Handle special case...

  while (X) {
    *--BufPtr = '0' + (X % 10);
    X /= 10;
  }

  if (isNeg) *--BufPtr = '-';   // Add negative sign...

  return string(BufPtr);
}

static inline string itostr(int X) {
  if (X < 0) 
    return utostr((unsigned)-X, true);
  else
    return utostr((unsigned)X);
}

static inline string ftostr(double V) {
  char Buffer[200];
  snprintf(Buffer, 200, "%f", V);
  return Buffer;
}

#endif
