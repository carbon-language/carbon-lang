//===-- Support/StringExtras.h - Useful string functions --------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains some functions that are useful when dealing with strings.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_STRINGEXTRAS_H
#define SUPPORT_STRINGEXTRAS_H

#include "Support/DataTypes.h"
#include <string>
#include <stdio.h>

namespace llvm {

static inline std::string utohexstr(uint64_t X) {
  char Buffer[40];
  char *BufPtr = Buffer+39;

  *BufPtr = 0;                  // Null terminate buffer...
  if (X == 0) *--BufPtr = '0';  // Handle special case...

  while (X) {
    unsigned char Mod = (unsigned char)X & 15;
    if (Mod < 10)
      *--BufPtr = '0' + Mod;
    else
      *--BufPtr = 'A' + Mod-10;
    X >>= 4;
  }
  return std::string(BufPtr);
}

static inline std::string utostr(unsigned long long X, bool isNeg = false) {
  char Buffer[40];
  char *BufPtr = Buffer+39;

  *BufPtr = 0;                  // Null terminate buffer...
  if (X == 0) *--BufPtr = '0';  // Handle special case...

  while (X) {
    *--BufPtr = '0' + char(X % 10);
    X /= 10;
  }

  if (isNeg) *--BufPtr = '-';   // Add negative sign...

  return std::string(BufPtr);
}

static inline std::string itostr(int64_t X) {
  if (X < 0) 
    return utostr(static_cast<uint64_t>(-X), true);
  else
    return utostr(static_cast<uint64_t>(X));
}


static inline std::string utostr(unsigned long X, bool isNeg = false) {
  return utostr(static_cast<unsigned long long>(X), isNeg);
}

static inline std::string utostr(unsigned X, bool isNeg = false) {
  char Buffer[20];
  char *BufPtr = Buffer+19;

  *BufPtr = 0;                  // Null terminate buffer...
  if (X == 0) *--BufPtr = '0';  // Handle special case...

  while (X) {
    *--BufPtr = '0' + char(X % 10);
    X /= 10;
  }

  if (isNeg) *--BufPtr = '-';   // Add negative sign...

  return std::string(BufPtr);
}

static inline std::string itostr(int X) {
  if (X < 0) 
    return utostr(static_cast<unsigned>(-X), true);
  else
    return utostr(static_cast<unsigned>(X));
}

static inline std::string ftostr(double V) {
  char Buffer[200];
  sprintf(Buffer, "%20.6e", V);
  return Buffer;
}


/// getToken - This function extracts one token from source, ignoring any
/// leading characters that appear in the Delimiters string, and ending the
/// token at any of the characters that appear in the Delimiters string.  If
/// there are no tokens in the source string, an empty string is returned.
/// The Source source string is updated in place to remove the returned string
/// and any delimiter prefix from it.
std::string getToken(std::string &Source,
                     const char *Delimiters = " \t\n\v\f\r");

} // End llvm namespace

#endif
