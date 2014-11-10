//===-- lld/Support/NumParse.cpp - Number parsing ---------------*- C++ -*-===//
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

#include "lld/Support/NumParse.h"

using namespace llvm;

namespace lld {
/// \brief Convert a string in decimal to decimal.
llvm::ErrorOr<uint64_t> parseDecimal(StringRef str) {
  uint64_t res = 0;
  for (auto &c : str) {
    res *= 10;
    if (c < '0' || c > '9')
      return llvm::ErrorOr<uint64_t>(std::make_error_code(std::errc::io_error));
    res += c - '0';
  }
  return res;
}

/// \brief Convert a string in octal to decimal.
llvm::ErrorOr<uint64_t> parseOctal(StringRef str) {
  uint64_t res = 0;
  for (auto &c : str) {
    res <<= 3;
    if (c < '0' || c > '7')
      return llvm::ErrorOr<uint64_t>(std::make_error_code(std::errc::io_error));
    res += c - '0';
  }
  return res;
}

/// \brief Convert a string in Binary to decimal.
llvm::ErrorOr<uint64_t> parseBinary(StringRef str) {
  uint64_t res = 0;
  for (auto &c : str) {
    res <<= 1;
    if (c != '0' && c != '1')
      return llvm::ErrorOr<uint64_t>(std::make_error_code(std::errc::io_error));
    res += c - '0';
  }
  return res;
}

/// \brief Convert a string in Hexadecimal to decimal.
llvm::ErrorOr<uint64_t> parseHex(StringRef str) {
  uint64_t res = 0;
  for (auto &c : str) {
    res <<= 4;
    if (c >= '0' && c <= '9')
      res += c - '0';
    else if (c >= 'a' && c <= 'f')
      res += c - 'a' + 10;
    else if (c >= 'A' && c <= 'F')
      res += c - 'A' + 10;
    else
      return llvm::ErrorOr<uint64_t>(std::make_error_code(std::errc::io_error));
  }
  return res;
}

/// \brief Parse a number represested in a string as
//  Hexadecimal, Octal, Binary or Decimal to decimal
llvm::ErrorOr<uint64_t> parseNum(StringRef str, bool parseExtensions) {
  unsigned multiplier = 1;
  enum NumKind { decimal, hex, octal, binary };
  NumKind kind = llvm::StringSwitch<NumKind>(str)
                     .StartsWith("0x", hex)
                     .StartsWith("0X", hex)
                     .StartsWith("0", octal)
                     .Default(decimal);

  if (parseExtensions) {
    // Parse scale
    if (str.endswith("K")) {
      multiplier = 1 << 10;
      str = str.drop_back();
    } else if (str.endswith("M")) {
      multiplier = 1 << 20;
      str = str.drop_back();
    }

    // Parse type
    if (str.endswith_lower("o")) {
      kind = octal;
      str = str.drop_back();
    } else if (str.endswith_lower("h")) {
      kind = hex;
      str = str.drop_back();
    } else if (str.endswith_lower("d")) {
      kind = decimal;
      str = str.drop_back();
    } else if (str.endswith_lower("b")) {
      kind = binary;
      str = str.drop_back();
    }
  }

  llvm::ErrorOr<uint64_t> res(0);
  switch (kind) {
  case hex:
    if (str.startswith_lower("0x"))
      str = str.drop_front(2);
    res = parseHex(str);
    break;
  case octal:
    res = parseOctal(str);
    break;
  case decimal:
    res = parseDecimal(str);
    break;
  case binary:
    res = parseBinary(str);
    break;
  }
  if (res.getError())
    return res;

  *res = *res * multiplier;
  return res;
}
}
