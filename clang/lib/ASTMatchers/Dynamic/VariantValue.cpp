//===--- VariantValue.cpp - Polymorphic value type -*- C++ -*-===/
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Polymorphic value type.
///
//===----------------------------------------------------------------------===//

#include "clang/ASTMatchers/Dynamic/VariantValue.h"

#include "clang/Basic/LLVM.h"

namespace clang {
namespace ast_matchers {
namespace dynamic {

VariantValue::VariantValue(const VariantValue &Other) : Type(VT_Nothing) {
  *this = Other;
}

VariantValue::VariantValue(unsigned Unsigned) : Type(VT_Nothing) {
  setUnsigned(Unsigned);
}

VariantValue::VariantValue(const std::string &String) : Type(VT_Nothing) {
  setString(String);
}

VariantValue::VariantValue(const DynTypedMatcher &Matcher) : Type(VT_Nothing) {
  setMatcher(Matcher);
}

VariantValue::~VariantValue() { reset(); }

VariantValue &VariantValue::operator=(const VariantValue &Other) {
  if (this == &Other) return *this;
  reset();
  switch (Other.Type) {
  case VT_Unsigned:
    setUnsigned(Other.getUnsigned());
    break;
  case VT_String:
    setString(Other.getString());
    break;
  case VT_Matcher:
    setMatcher(Other.getMatcher());
    break;
  case VT_Nothing:
    Type = VT_Nothing;
    break;
  }
  return *this;
}

void VariantValue::reset() {
  switch (Type) {
  case VT_String:
    delete Value.String;
    break;
  case VT_Matcher:
    delete Value.Matcher;
    break;
  // Cases that do nothing.
  case VT_Unsigned:
  case VT_Nothing:
    break;
  }
  Type = VT_Nothing;
}

bool VariantValue::isUnsigned() const {
  return Type == VT_Unsigned;
}

unsigned VariantValue::getUnsigned() const {
  assert(isUnsigned());
  return Value.Unsigned;
}

void VariantValue::setUnsigned(unsigned NewValue) {
  reset();
  Type = VT_Unsigned;
  Value.Unsigned = NewValue;
}

bool VariantValue::isString() const {
  return Type == VT_String;
}

const std::string &VariantValue::getString() const {
  assert(isString());
  return *Value.String;
}

void VariantValue::setString(const std::string &NewValue) {
  reset();
  Type = VT_String;
  Value.String = new std::string(NewValue);
}

bool VariantValue::isMatcher() const {
  return Type == VT_Matcher;
}

const DynTypedMatcher &VariantValue::getMatcher() const {
  assert(isMatcher());
  return *Value.Matcher;
}

void VariantValue::setMatcher(const DynTypedMatcher &NewValue) {
  reset();
  Type = VT_Matcher;
  Value.Matcher = NewValue.clone();
}

void VariantValue::takeMatcher(DynTypedMatcher *NewValue) {
  reset();
  Type = VT_Matcher;
  Value.Matcher = NewValue;
}

std::string VariantValue::getTypeAsString() const {
  switch (Type) {
  case VT_String: return "String";
  case VT_Matcher:
    return (Twine("Matcher<") + getMatcher().getSupportedKind().asStringRef() +
            ">").str();
  case VT_Unsigned: return "Unsigned";
  case VT_Nothing: return "Nothing";
  }
  llvm_unreachable("Invalid Type");
}

} // end namespace dynamic
} // end namespace ast_matchers
} // end namespace clang
