//===- Format.h - Efficient printf-style formatting for streams -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the format() function, which can be used with other
// LLVM subsystems to provide printf-style formatting.  This gives all the power
// and risk of printf.  This can be used like this (with raw_ostreams as an
// example):
//
//    OS << "mynumber: " << format("%4.5f", 1234.412) << '\n';
//
// Or if you prefer:
//
//  OS << format("mynumber: %4.5f\n", 1234.412);
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_FORMAT_H
#define LLVM_SUPPORT_FORMAT_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"
#include <cassert>
#include <cstdio>
#ifdef _MSC_VER
// FIXME: This define is wrong:
//  - _snprintf does not guarantee that trailing null is always added - if
//    there is no space for null, it does not report any error.
//  - According to C++ standard, snprintf should be visible in the 'std' 
//    namespace - this define makes this impossible.
#define snprintf _snprintf
#endif

namespace llvm {

/// This is a helper class used for handling formatted output.  It is the
/// abstract base class of a templated derived class.
class format_object_base {
protected:
  const char *Fmt;
  ~format_object_base() {} // Disallow polymorphic deletion.
  virtual void home(); // Out of line virtual method.

  /// Call snprintf() for this object, on the given buffer and size.
  virtual int snprint(char *Buffer, unsigned BufferSize) const = 0;

public:
  format_object_base(const char *fmt) : Fmt(fmt) {}

  /// Format the object into the specified buffer.  On success, this returns
  /// the length of the formatted string.  If the buffer is too small, this
  /// returns a length to retry with, which will be larger than BufferSize.
  unsigned print(char *Buffer, unsigned BufferSize) const {
    assert(BufferSize && "Invalid buffer size!");

    // Print the string, leaving room for the terminating null.
    int N = snprint(Buffer, BufferSize);

    // VC++ and old GlibC return negative on overflow, just double the size.
    if (N < 0)
      return BufferSize * 2;

    // Other implementations yield number of bytes needed, not including the
    // final '\0'.
    if (unsigned(N) >= BufferSize)
      return N + 1;

    // Otherwise N is the length of output (not including the final '\0').
    return N;
  }
};

/// These are templated helper classes used by the format function that
/// capture the object to be formated and the format string. When actually
/// printed, this synthesizes the string into a temporary buffer provided and
/// returns whether or not it is big enough.

template <typename T>
class format_object1 final : public format_object_base {
  T Val;
public:
  format_object1(const char *fmt, const T &val)
    : format_object_base(fmt), Val(val) {
  }

  int snprint(char *Buffer, unsigned BufferSize) const override {
    return snprintf(Buffer, BufferSize, Fmt, Val);
  }
};

template <typename T1, typename T2>
class format_object2 final : public format_object_base {
  T1 Val1;
  T2 Val2;
public:
  format_object2(const char *fmt, const T1 &val1, const T2 &val2)
  : format_object_base(fmt), Val1(val1), Val2(val2) {
  }

  int snprint(char *Buffer, unsigned BufferSize) const override {
    return snprintf(Buffer, BufferSize, Fmt, Val1, Val2);
  }
};

template <typename T1, typename T2, typename T3>
class format_object3 final : public format_object_base {
  T1 Val1;
  T2 Val2;
  T3 Val3;
public:
  format_object3(const char *fmt, const T1 &val1, const T2 &val2,const T3 &val3)
    : format_object_base(fmt), Val1(val1), Val2(val2), Val3(val3) {
  }

  int snprint(char *Buffer, unsigned BufferSize) const override {
    return snprintf(Buffer, BufferSize, Fmt, Val1, Val2, Val3);
  }
};

template <typename T1, typename T2, typename T3, typename T4>
class format_object4 final : public format_object_base {
  T1 Val1;
  T2 Val2;
  T3 Val3;
  T4 Val4;
public:
  format_object4(const char *fmt, const T1 &val1, const T2 &val2,
                 const T3 &val3, const T4 &val4)
    : format_object_base(fmt), Val1(val1), Val2(val2), Val3(val3), Val4(val4) {
  }

  int snprint(char *Buffer, unsigned BufferSize) const override {
    return snprintf(Buffer, BufferSize, Fmt, Val1, Val2, Val3, Val4);
  }
};

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class format_object5 final : public format_object_base {
  T1 Val1;
  T2 Val2;
  T3 Val3;
  T4 Val4;
  T5 Val5;
public:
  format_object5(const char *fmt, const T1 &val1, const T2 &val2,
                 const T3 &val3, const T4 &val4, const T5 &val5)
    : format_object_base(fmt), Val1(val1), Val2(val2), Val3(val3), Val4(val4),
      Val5(val5) {
  }

  int snprint(char *Buffer, unsigned BufferSize) const override {
    return snprintf(Buffer, BufferSize, Fmt, Val1, Val2, Val3, Val4, Val5);
  }
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
          typename T6>
class format_object6 final : public format_object_base {
  T1 Val1;
  T2 Val2;
  T3 Val3;
  T4 Val4;
  T5 Val5;
  T6 Val6;
public:
  format_object6(const char *Fmt, const T1 &Val1, const T2 &Val2,
                 const T3 &Val3, const T4 &Val4, const T5 &Val5, const T6 &Val6)
    : format_object_base(Fmt), Val1(Val1), Val2(Val2), Val3(Val3), Val4(Val4),
      Val5(Val5), Val6(Val6) { }

  int snprint(char *Buffer, unsigned BufferSize) const override {
    return snprintf(Buffer, BufferSize, Fmt, Val1, Val2, Val3, Val4, Val5, Val6);
  }
};

/// These are helper functions used to produce formatted output.  They use
/// template type deduction to construct the appropriate instance of the
/// format_object class to simplify their construction.
///
/// This is typically used like:
/// \code
///   OS << format("%0.4f", myfloat) << '\n';
/// \endcode

template <typename T>
inline format_object1<T> format(const char *Fmt, const T &Val) {
  return format_object1<T>(Fmt, Val);
}

template <typename T1, typename T2>
inline format_object2<T1, T2> format(const char *Fmt, const T1 &Val1,
                                     const T2 &Val2) {
  return format_object2<T1, T2>(Fmt, Val1, Val2);
}

template <typename T1, typename T2, typename T3>
  inline format_object3<T1, T2, T3> format(const char *Fmt, const T1 &Val1,
                                           const T2 &Val2, const T3 &Val3) {
  return format_object3<T1, T2, T3>(Fmt, Val1, Val2, Val3);
}

template <typename T1, typename T2, typename T3, typename T4>
inline format_object4<T1, T2, T3, T4> format(const char *Fmt, const T1 &Val1,
                                             const T2 &Val2, const T3 &Val3,
                                             const T4 &Val4) {
  return format_object4<T1, T2, T3, T4>(Fmt, Val1, Val2, Val3, Val4);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
inline format_object5<T1, T2, T3, T4, T5> format(const char *Fmt,const T1 &Val1,
                                             const T2 &Val2, const T3 &Val3,
                                             const T4 &Val4, const T5 &Val5) {
  return format_object5<T1, T2, T3, T4, T5>(Fmt, Val1, Val2, Val3, Val4, Val5);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
          typename T6>
inline format_object6<T1, T2, T3, T4, T5, T6>
format(const char *Fmt, const T1 &Val1, const T2 &Val2, const T3 &Val3,
       const T4 &Val4, const T5 &Val5, const T6 &Val6) {
  return format_object6<T1, T2, T3, T4, T5, T6>(Fmt, Val1, Val2, Val3, Val4,
                                                Val5, Val6);
}

/// This is a helper class used for left_justify() and right_justify().
class FormattedString {
  StringRef Str;
  unsigned Width;
  bool RightJustify;
  friend class raw_ostream;
public:
    FormattedString(StringRef S, unsigned W, bool R)
      : Str(S), Width(W), RightJustify(R) { }
};

/// left_justify - append spaces after string so total output is
/// \p Width characters.  If \p Str is larger that \p Width, full string
/// is written with no padding.
inline FormattedString left_justify(StringRef Str, unsigned Width) {
  return FormattedString(Str, Width, false);
}

/// right_justify - add spaces before string so total output is
/// \p Width characters.  If \p Str is larger that \p Width, full string
/// is written with no padding.
inline FormattedString right_justify(StringRef Str, unsigned Width) {
  return FormattedString(Str, Width, true);
}

/// This is a helper class used for format_hex() and format_decimal().
class FormattedNumber {
  uint64_t HexValue;
  int64_t DecValue;
  unsigned Width;
  bool Hex;
  bool Upper;
  friend class raw_ostream;
public:
    FormattedNumber(uint64_t HV, int64_t DV, unsigned W, bool H, bool U)
      : HexValue(HV), DecValue(DV), Width(W), Hex(H), Upper(U) { }
};

/// format_hex - Output \p N as a fixed width hexadecimal. If number will not
/// fit in width, full number is still printed.  Examples:
///   OS << format_hex(255, 4)        => 0xff
///   OS << format_hex(255, 4, true)  => 0xFF
///   OS << format_hex(255, 6)        => 0x00ff
///   OS << format_hex(255, 2)        => 0xff
inline FormattedNumber format_hex(uint64_t N, unsigned Width, bool Upper=false) {
  assert(Width <= 18 && "hex width must be <= 18");
  return FormattedNumber(N, 0, Width, true, Upper);
}

/// format_decimal - Output \p N as a right justified, fixed-width decimal. If 
/// number will not fit in width, full number is still printed.  Examples:
///   OS << format_decimal(0, 5)     => "    0"
///   OS << format_decimal(255, 5)   => "  255"
///   OS << format_decimal(-1, 3)    => " -1"
///   OS << format_decimal(12345, 3) => "12345"
inline FormattedNumber format_decimal(int64_t N, unsigned Width) {
  return FormattedNumber(0, N, Width, false, false);
}


} // end namespace llvm

#endif
