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

MatcherList::MatcherList() : List() {}

MatcherList::MatcherList(const DynTypedMatcher &Matcher)
    : List(1, Matcher.clone()) {}

MatcherList::MatcherList(const MatcherList& Other) {
  *this = Other;
}

MatcherList::~MatcherList() {
  reset();
}

MatcherList &MatcherList::operator=(const MatcherList &Other) {
  if (this == &Other) return *this;
  reset();
  for (size_t i = 0, e = Other.List.size(); i != e; ++i) {
    List.push_back(Other.List[i]->clone());
  }
  return *this;
}

void MatcherList::add(const DynTypedMatcher &Matcher) {
  List.push_back(Matcher.clone());
}

void MatcherList::reset() {
  for (size_t i = 0, e = List.size(); i != e; ++i) {
    delete List[i];
  }
  List.resize(0);
}

std::string MatcherList::getTypeAsString() const {
  std::string Inner;
  for (size_t I = 0, E = List.size(); I != E; ++I) {
    if (I != 0) Inner += "|";
    Inner += List[I]->getSupportedKind().asStringRef();
  }
  return (Twine("Matcher<") + Inner + ">").str();
}

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
  setMatchers(MatcherList(Matcher));
}

VariantValue::VariantValue(const MatcherList &Matchers) : Type(VT_Nothing) {
  setMatchers(Matchers);
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
  case VT_Matchers:
    setMatchers(Other.getMatchers());
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
  case VT_Matchers:
    delete Value.Matchers;
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

bool VariantValue::isMatchers() const {
  return Type == VT_Matchers;
}

const MatcherList &VariantValue::getMatchers() const {
  assert(isMatchers());
  return *Value.Matchers;
}

void VariantValue::setMatchers(const MatcherList &NewValue) {
  reset();
  Type = VT_Matchers;
  Value.Matchers = new MatcherList(NewValue);
}

std::string VariantValue::getTypeAsString() const {
  switch (Type) {
  case VT_String: return "String";
  case VT_Matchers: return getMatchers().getTypeAsString();
  case VT_Unsigned: return "Unsigned";
  case VT_Nothing: return "Nothing";
  }
  llvm_unreachable("Invalid Type");
}

} // end namespace dynamic
} // end namespace ast_matchers
} // end namespace clang
