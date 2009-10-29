//===--- StringSwitch.h - Switch-on-literal-string Construct --------------===/
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===/
//
//  This file implements the StringSwitch template, which mimics a switch()
//  statements whose cases are string literals.
//
//===----------------------------------------------------------------------===/
#ifndef LLVM_ADT_STRINGSWITCH_H
#define LLVM_ADT_STRINGSWITCH_H

#include "llvm/ADT/StringRef.h"
#include <cassert>
#include <cstring>

namespace llvm {
  
/// \brief A switch()-like statement whose cases are string literals.
///
/// The StringSwitch class is a simple form of a switch() statement that
/// determines whether the given string matches one of the given string
/// literals. The template type parameter \p T is the type of the value that
/// will be returned from the string-switch expression. For example,
/// the following code switches on the name of a color in \c argv[i]:
///
/// \code
/// Color color = StringSwitch<Color>(argv[i])
///   .Case("red", Red)
///   .Case("orange", Orange)
///   .Case("yellow", Yellow)
///   .Case("green", Green)
///   .Case("blue", Blue)
///   .Case("indigo", Indigo)
///   .Case("violet", Violet)
///   .Default(UnknownColor);
/// \endcode
template<typename T>
class StringSwitch {
  /// \brief The string we are matching.
  StringRef Str;
  
  /// \brief The result of this switch statement, once known.
  T Result;
  
  /// \brief Set true when the result of this switch is already known; in this
  /// case, Result is valid.
  bool ResultKnown;
  
public:
  explicit StringSwitch(StringRef Str) 
  : Str(Str), ResultKnown(false) { }
  
  template<unsigned N>
  StringSwitch& Case(const char (&S)[N], const T& Value) {
    if (!ResultKnown && N-1 == Str.size() && 
        (std::memcmp(S, Str.data(), N-1) == 0)) {
      Result = Value;
      ResultKnown = true;
    }
    
    return *this;
  }
  
  T Default(const T& Value) {
    if (ResultKnown)
      return Result;
    
    return Value;
  }
  
  operator T() {
    assert(ResultKnown && "Fell off the end of a string-switch");
    return Result;
  }
};

} // end namespace llvm

#endif // LLVM_ADT_STRINGSWITCH_H