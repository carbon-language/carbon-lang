//===-- lld/Support/NumParse.h - Number parsing -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Parses string in various formats to decimal.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_SUPPORT_NUM_PARSE_H
#define LLD_SUPPORT_NUM_PARSE_H

#include "lld/Core/LLVM.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <system_error>
#include <vector>

namespace lld {

/// \brief Convert a string in decimal to decimal.
llvm::ErrorOr<uint64_t> parseDecimal(StringRef str);

/// \brief Convert a string in octal to decimal.
llvm::ErrorOr<uint64_t> parseOctal(StringRef str);

/// \brief Convert a string in Binary to decimal.
llvm::ErrorOr<uint64_t> parseBinary(StringRef str);

/// \brief Convert a string in Hexadecimal to decimal.
llvm::ErrorOr<uint64_t> parseHex(StringRef str);

/// \brief Parse a number represested in a string as
//  Hexadecimal, Octal, Binary or Decimal to decimal
llvm::ErrorOr<uint64_t> parseNum(StringRef str, bool parseExtensions = true);
}

#endif // LLD_SUPPORT_NUM_PARSE_H
