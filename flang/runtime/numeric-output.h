//===-- runtime/numeric-output.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_NUMERIC_OUTPUT_H_
#define FORTRAN_RUNTIME_NUMERIC_OUTPUT_H_

// Output data editing templates implementing the FORMAT data editing
// descriptors E, EN, ES, EX, D, F, and G for REAL data (and COMPLEX
// components, I and G for INTEGER, and B/O/Z for both.
// See subclauses in 13.7.2.3 of Fortran 2018 for the
// detailed specifications of these descriptors.
// Drives the same binary-to-decimal formatting templates used
// by the f18 compiler.

#include "format.h"
#include "flang/common/unsigned-const-division.h"
#include "flang/decimal/decimal.h"

namespace Fortran::runtime::io {

class IoStatementState;

// Utility subroutines
static bool EmitRepeated(IoStatementState &io, char ch, int n) {
  while (n-- > 0) {
    if (!io.Emit(&ch, 1)) {
      return false;
    }
  }
  return true;
}

static bool EmitField(
    IoStatementState &io, const char *p, std::size_t length, int width) {
  if (width <= 0) {
    width = static_cast<int>(length);
  }
  if (length > static_cast<std::size_t>(width)) {
    return EmitRepeated(io, '*', width);
  } else {
    return EmitRepeated(io, ' ', static_cast<int>(width - length)) &&
        io.Emit(p, length);
  }
}

// I, B, O, Z, and (for INTEGER) G output editing.
// edit is const here so that a repeated edit descriptor may safely serve
// multiple array elements
static bool EditIntegerOutput(
    IoStatementState &io, const DataEdit &edit, std::int64_t n) {
  char buffer[66], *end = &buffer[sizeof buffer], *p = end;
  std::uint64_t un{static_cast<std::uint64_t>(n < 0 ? -n : n)};
  int signChars{0};
  switch (edit.descriptor) {
  case 'G':
  case 'I':
    if (n < 0 || (edit.modes.editingFlags & signPlus)) {
      signChars = 1;  // '-' or '+'
    }
    while (un > 0) {
      auto quotient{common::DivideUnsignedBy<std::uint64_t, 10>(un)};
      *--p = '0' + un - 10 * quotient;
      un = quotient;
    }
    break;
  case 'B':
    for (; un > 0; un >>= 1) {
      *--p = '0' + (un & 1);
    }
    break;
  case 'O':
    for (; un > 0; un >>= 3) {
      *--p = '0' + (un & 7);
    }
    break;
  case 'Z':
    for (; un > 0; un >>= 4) {
      int digit = un & 0xf;
      *--p = digit >= 10 ? 'A' + (digit - 10) : '0' + digit;
    }
    break;
  default:
    io.Crash(
        "Data edit descriptor '%c' may not be used with an INTEGER data item",
        edit.descriptor);
    return false;
  }

  int digits = end - p;
  int leadingZeroes{0};
  int editWidth{edit.width.value_or(0)};
  if (edit.digits && digits <= *edit.digits) {  // Iw.m
    if (*edit.digits == 0 && n == 0) {
      // Iw.0 with zero value: output field must be blank.  For I0.0
      // and a zero value, emit one blank character.
      signChars = 0;  // in case of SP
      editWidth = std::max(1, editWidth);
    } else {
      leadingZeroes = *edit.digits - digits;
    }
  } else if (n == 0) {
    leadingZeroes = 1;
  }
  int total{signChars + leadingZeroes + digits};
  if (edit.width > 0 && total > editWidth) {
    return EmitRepeated(io, '*', editWidth);
  }
  if (total < editWidth) {
    EmitRepeated(io, '*', editWidth - total);
    return false;
  }
  if (signChars) {
    if (!io.Emit(n < 0 ? "-" : "+", 1)) {
      return false;
    }
  }
  return EmitRepeated(io, '0', leadingZeroes) && io.Emit(p, digits);
}

// Encapsulates the state of a REAL output conversion.
template<typename FLOAT = double, int decimalPrecision = 15,
    int binaryPrecision = 53, std::size_t bufferSize = 1024>
class RealOutputEditing {
public:
  RealOutputEditing(IoStatementState &io, FLOAT x) : io_{io}, x_{x} {}
  bool Edit(const DataEdit &edit);

private:
  // The DataEdit arguments here are const references or copies so that
  // the original DataEdit can safely serve multiple array elements if
  // it has a repeat count.
  bool EditEorDOutput(const DataEdit &);
  bool EditFOutput(const DataEdit &);
  DataEdit EditForGOutput(DataEdit);  // returns an E or F edit
  bool EditEXOutput(const DataEdit &);

  bool IsZero() const { return x_ == 0; }
  const char *FormatExponent(int, const DataEdit &edit, int &length);

  static enum decimal::FortranRounding SetRounding(
      common::RoundingMode rounding) {
    switch (rounding) {
    case common::RoundingMode::TiesToEven: break;
    case common::RoundingMode::Up: return decimal::RoundUp;
    case common::RoundingMode::Down: return decimal::RoundDown;
    case common::RoundingMode::ToZero: return decimal::RoundToZero;
    case common::RoundingMode::TiesAwayFromZero:
      return decimal::RoundCompatible;
    }
    return decimal::RoundNearest;  // arranged thus to dodge bogus G++ warning
  }

  static bool IsDecimalNumber(const char *p) {
    if (!p) {
      return false;
    }
    if (*p == '-' || *p == '+') {
      ++p;
    }
    return *p >= '0' && *p <= '9';
  }

  decimal::ConversionToDecimalResult Convert(
      int significantDigits, const DataEdit &, int flags = 0);

  IoStatementState &io_;
  FLOAT x_;
  char buffer_[bufferSize];
  int trailingBlanks_{0};  // created when G editing maps to F
  char exponent_[16];
};

template<typename FLOAT, int decimalPrecision, int binaryPrecision,
    std::size_t bufferSize>
decimal::ConversionToDecimalResult RealOutputEditing<FLOAT, decimalPrecision,
    binaryPrecision, bufferSize>::Convert(int significantDigits,
    const DataEdit &edit, int flags) {
  if (edit.modes.editingFlags & signPlus) {
    flags |= decimal::AlwaysSign;
  }
  auto converted{decimal::ConvertToDecimal<binaryPrecision>(buffer_, bufferSize,
      static_cast<enum decimal::DecimalConversionFlags>(flags),
      significantDigits, SetRounding(edit.modes.roundingMode),
      decimal::BinaryFloatingPointNumber<binaryPrecision>(x_))};
  if (!converted.str) {  // overflow
    io_.Crash("RealOutputEditing::Convert : buffer size %zd was insufficient",
        bufferSize);
  }
  return converted;
}

// 13.7.2.3.3 in F'2018
template<typename FLOAT, int decimalPrecision, int binaryPrecision,
    std::size_t bufferSize>
bool RealOutputEditing<FLOAT, decimalPrecision, binaryPrecision,
    bufferSize>::EditEorDOutput(const DataEdit &edit) {
  int editDigits{edit.digits.value_or(0)};  // 'd' field
  int editWidth{edit.width.value_or(0)};  // 'w' field
  int significantDigits{editDigits};
  int flags{0};
  if (editWidth == 0) {  // "the processor selects the field width"
    if (edit.digits.has_value()) {  // E0.d
      editWidth = editDigits + 6;  // -.666E+ee
    } else {  // E0
      flags |= decimal::Minimize;
      significantDigits =
          bufferSize - 5;  // sign, NUL, + 3 extra for EN scaling
    }
  }
  bool isEN{edit.variation == 'N'};
  bool isES{edit.variation == 'S'};
  int scale{isEN || isES ? 1 : edit.modes.scale};  // 'kP' value
  int zeroesAfterPoint{0};
  if (scale < 0) {
    zeroesAfterPoint = -scale;
    significantDigits = std::max(0, significantDigits - zeroesAfterPoint);
  } else if (scale > 0) {
    ++significantDigits;
    scale = std::min(scale, significantDigits + 1);
  }
  // In EN editing, multiple attempts may be necessary, so it's in a loop.
  while (true) {
    decimal::ConversionToDecimalResult converted{
        Convert(significantDigits, edit, flags)};
    if (converted.length > 0 && !IsDecimalNumber(converted.str)) {  // Inf, NaN
      return EmitField(io_, converted.str, converted.length, editWidth);
    }
    if (!IsZero()) {
      converted.decimalExponent -= scale;
    }
    if (isEN && scale < 3 && (converted.decimalExponent % 3) != 0) {
      // EN mode: boost the scale and significant digits, try again; need
      // an effective exponent field that's a multiple of three.
      ++scale;
      ++significantDigits;
      continue;
    }
    // Format the exponent (see table 13.1 for all the cases)
    int expoLength{0};
    const char *exponent{
        FormatExponent(converted.decimalExponent, edit, expoLength)};
    int signLength{*converted.str == '-' || *converted.str == '+' ? 1 : 0};
    int convertedDigits{static_cast<int>(converted.length) - signLength};
    int zeroesBeforePoint{std::max(0, scale - convertedDigits)};
    int digitsBeforePoint{std::max(0, scale - zeroesBeforePoint)};
    int digitsAfterPoint{convertedDigits - digitsBeforePoint};
    int trailingZeroes{flags & decimal::Minimize
            ? 0
            : std::max(0,
                  significantDigits - (convertedDigits + zeroesBeforePoint))};
    int totalLength{signLength + digitsBeforePoint + zeroesBeforePoint +
        1 /*'.'*/ + zeroesAfterPoint + digitsAfterPoint + trailingZeroes +
        expoLength};
    int width{editWidth > 0 ? editWidth : totalLength};
    if (totalLength > width) {
      return EmitRepeated(io_, '*', width);
    }
    if (totalLength < width && digitsBeforePoint == 0 &&
        zeroesBeforePoint == 0) {
      zeroesBeforePoint = 1;
      ++totalLength;
    }
    return EmitRepeated(io_, ' ', width - totalLength) &&
        io_.Emit(converted.str, signLength + digitsBeforePoint) &&
        EmitRepeated(io_, '0', zeroesBeforePoint) &&
        io_.Emit(edit.modes.editingFlags & decimalComma ? "," : ".", 1) &&
        EmitRepeated(io_, '0', zeroesAfterPoint) &&
        io_.Emit(
            converted.str + signLength + digitsBeforePoint, digitsAfterPoint) &&
        EmitRepeated(io_, '0', trailingZeroes) &&
        io_.Emit(exponent, expoLength);
  }
}

// Formats the exponent (see table 13.1 for all the cases)
template<typename FLOAT, int decimalPrecision, int binaryPrecision,
    std::size_t bufferSize>
const char *RealOutputEditing<FLOAT, decimalPrecision, binaryPrecision,
    bufferSize>::FormatExponent(int expo, const DataEdit &edit, int &length) {
  char *eEnd{&exponent_[sizeof exponent_]};
  char *exponent{eEnd};
  for (unsigned e{static_cast<unsigned>(std::abs(expo))}; e > 0;) {
    unsigned quotient{common::DivideUnsignedBy<unsigned, 10>(e)};
    *--exponent = '0' + e - 10 * quotient;
    e = quotient;
  }
  if (edit.expoDigits) {
    if (int ed{*edit.expoDigits}) {  // Ew.dEe with e > 0
      while (exponent > exponent_ + 2 /*E+*/ && exponent + ed > eEnd) {
        *--exponent = '0';
      }
    } else if (exponent == eEnd) {
      *--exponent = '0';  // Ew.dE0 with zero-valued exponent
    }
  } else {  // ensure at least two exponent digits
    while (exponent + 2 > eEnd) {
      *--exponent = '0';
    }
  }
  *--exponent = expo < 0 ? '-' : '+';
  if (edit.expoDigits || exponent + 3 == eEnd) {
    *--exponent = edit.descriptor == 'D' ? 'D' : 'E';  // not 'G'
  }
  length = eEnd - exponent;
  return exponent;
}

// 13.7.2.3.2 in F'2018
template<typename FLOAT, int decimalPrecision, int binaryPrecision,
    std::size_t bufferSize>
bool RealOutputEditing<FLOAT, decimalPrecision, binaryPrecision,
    bufferSize>::EditFOutput(const DataEdit &edit) {
  int fracDigits{edit.digits.value_or(0)};  // 'd' field
  int extraDigits{0};
  int editWidth{edit.width.value_or(0)};  // 'w' field
  int flags{0};
  if (editWidth == 0) {  // "the processor selects the field width"
    if (!edit.digits.has_value()) {  // F0
      flags |= decimal::Minimize;
      fracDigits = bufferSize - 2;  // sign & NUL
    }
  }
  // Multiple conversions may be needed to get the right number of
  // effective rounded fractional digits.
  while (true) {
    decimal::ConversionToDecimalResult converted{
        Convert(extraDigits + fracDigits, edit, flags)};
    if (converted.length > 0 && !IsDecimalNumber(converted.str)) {  // Inf, NaN
      return EmitField(io_, converted.str, converted.length, editWidth);
    }
    int scale{IsZero() ? -1 : edit.modes.scale};
    int expo{converted.decimalExponent - scale};
    if (expo > extraDigits) {
      extraDigits = expo;
      if (flags & decimal::Minimize) {
        fracDigits = bufferSize - extraDigits - 2;  // sign & NUL
      }
      continue;  // try again
    }
    int signLength{*converted.str == '-' || *converted.str == '+' ? 1 : 0};
    int convertedDigits{static_cast<int>(converted.length) - signLength};
    int digitsBeforePoint{std::max(0, std::min(expo, convertedDigits))};
    int zeroesBeforePoint{std::max(0, expo - digitsBeforePoint)};
    int zeroesAfterPoint{std::max(0, -expo)};
    int digitsAfterPoint{convertedDigits - digitsBeforePoint};
    int trailingZeroes{flags & decimal::Minimize
            ? 0
            : std::max(0, fracDigits - (zeroesAfterPoint + digitsAfterPoint))};
    if (digitsBeforePoint + zeroesBeforePoint + zeroesAfterPoint +
            digitsAfterPoint + trailingZeroes ==
        0) {
      ++zeroesBeforePoint;  // "." -> "0."
    }
    int totalLength{signLength + digitsBeforePoint + zeroesBeforePoint +
        1 /*'.'*/ + zeroesAfterPoint + digitsAfterPoint + trailingZeroes};
    int width{editWidth > 0 ? editWidth : totalLength};
    if (totalLength > width) {
      return EmitRepeated(io_, '*', width);
    }
    if (totalLength < width && digitsBeforePoint + zeroesBeforePoint == 0) {
      zeroesBeforePoint = 1;
      ++totalLength;
    }
    return EmitRepeated(io_, ' ', width - totalLength) &&
        io_.Emit(converted.str, signLength + digitsBeforePoint) &&
        EmitRepeated(io_, '0', zeroesBeforePoint) &&
        io_.Emit(edit.modes.editingFlags & decimalComma ? "," : ".", 1) &&
        EmitRepeated(io_, '0', zeroesAfterPoint) &&
        io_.Emit(
            converted.str + signLength + digitsBeforePoint, digitsAfterPoint) &&
        EmitRepeated(io_, '0', trailingZeroes) &&
        EmitRepeated(io_, ' ', trailingBlanks_);
  }
}

// 13.7.5.2.3 in F'2018
template<typename FLOAT, int decimalPrecision, int binaryPrecision,
    std::size_t bufferSize>
DataEdit RealOutputEditing<FLOAT, decimalPrecision, binaryPrecision,
    bufferSize>::EditForGOutput(DataEdit edit) {
  edit.descriptor = 'E';
  if (!edit.width.has_value() ||
      (*edit.width > 0 && edit.digits.value_or(-1) == 0)) {
    return edit;  // Gw.0 -> Ew.0 for w > 0
  }
  decimal::ConversionToDecimalResult converted{Convert(1, edit)};
  if (!IsDecimalNumber(converted.str)) {  // Inf, NaN
    return edit;
  }
  int expo{IsZero() ? 1 : converted.decimalExponent};  // 's'
  int significantDigits{edit.digits.value_or(decimalPrecision)};  // 'd'
  if (expo < 0 || expo > significantDigits) {
    return edit;  // Ew.d
  }
  edit.descriptor = 'F';
  edit.modes.scale = 0;  // kP is ignored for G when no exponent field
  trailingBlanks_ = 0;
  int editWidth{edit.width.value_or(0)};
  if (editWidth > 0) {
    int expoDigits{edit.expoDigits.value_or(0)};
    trailingBlanks_ = expoDigits > 0 ? expoDigits + 2 : 4;  // 'n'
    *edit.width = std::max(0, editWidth - trailingBlanks_);
  }
  if (edit.digits.has_value()) {
    *edit.digits = std::max(0, *edit.digits - expo);
  }
  return edit;
}

// 13.7.5.2.6 in F'2018
template<typename FLOAT, int decimalPrecision, int binaryPrecision,
    std::size_t bufferSize>
bool RealOutputEditing<FLOAT, decimalPrecision, binaryPrecision,
    bufferSize>::EditEXOutput(const DataEdit &) {
  io_.Crash("EX output editing is not yet implemented");  // TODO
}

template<typename FLOAT, int decimalPrecision, int binaryPrecision,
    std::size_t bufferSize>
bool RealOutputEditing<FLOAT, decimalPrecision, binaryPrecision,
    bufferSize>::Edit(const DataEdit &edit) {
  switch (edit.descriptor) {
  case 'D': return EditEorDOutput(edit);
  case 'E':
    if (edit.variation == 'X') {
      return EditEXOutput(edit);
    } else {
      return EditEorDOutput(edit);
    }
  case 'F': return EditFOutput(edit);
  case 'B':
  case 'O':
  case 'Z':
    return EditIntegerOutput(io_, edit, decimal::BinaryFloatingPointNumber<binaryPrecision>{x_}.raw);
  case 'G': return Edit(EditForGOutput(edit));
  default:
    io_.Crash("Data edit descriptor '%c' may not be used with a REAL data item",
        edit.descriptor);
    return false;
  }
  return false;
}
}
#endif  // FORTRAN_RUNTIME_NUMERIC_OUTPUT_H_
