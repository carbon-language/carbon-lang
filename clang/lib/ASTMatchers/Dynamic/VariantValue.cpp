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
#include "llvm/ADT/STLExtras.h"

namespace clang {
namespace ast_matchers {
namespace dynamic {

VariantMatcher::VariantMatcher() : List() {}

VariantMatcher::VariantMatcher(const VariantMatcher& Other) {
  *this = Other;
}

VariantMatcher VariantMatcher::SingleMatcher(const DynTypedMatcher &Matcher) {
  VariantMatcher Out;
  Out.List.push_back(Matcher.clone());
  return Out;
}

VariantMatcher
VariantMatcher::PolymorphicMatcher(ArrayRef<const DynTypedMatcher *> Matchers) {
  VariantMatcher Out;
  for (size_t i = 0, e = Matchers.size(); i != e; ++i) {
    Out.List.push_back(Matchers[i]->clone());
  }
  return Out;
}

VariantMatcher::~VariantMatcher() {
  reset();
}

VariantMatcher &VariantMatcher::operator=(const VariantMatcher &Other) {
  if (this == &Other) return *this;
  reset();
  for (size_t i = 0, e = Other.List.size(); i != e; ++i) {
    List.push_back(Other.List[i]->clone());
  }
  return *this;
}

bool VariantMatcher::getSingleMatcher(const DynTypedMatcher *&Out) const {
  if (List.size() != 1) return false;
  Out = List[0];
  return true;
}

void VariantMatcher::reset() {
  llvm::DeleteContainerPointers(List);
}

std::string VariantMatcher::getTypeAsString() const {
  std::string Inner;
  for (size_t I = 0, E = List.size(); I != E; ++I) {
    if (I != 0) Inner += "|";
    Inner += List[I]->getSupportedKind().asStringRef();
  }
  return (Twine("Matcher<") + Inner + ">").str();
}

const DynTypedMatcher *VariantMatcher::getTypedMatcher(
    bool (*CanConstructCallback)(const DynTypedMatcher &)) const {
  const DynTypedMatcher *Out = NULL;
  for (size_t i = 0, e = List.size(); i != e; ++i) {
    if (CanConstructCallback(*List[i])) {
      if (Out) return NULL;
      Out = List[i];
    }
  }
  return Out;
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

VariantValue::VariantValue(const VariantMatcher &Matcher) : Type(VT_Nothing) {
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

const VariantMatcher &VariantValue::getMatcher() const {
  assert(isMatcher());
  return *Value.Matcher;
}

void VariantValue::setMatcher(const VariantMatcher &NewValue) {
  reset();
  Type = VT_Matcher;
  Value.Matcher = new VariantMatcher(NewValue);
}

std::string VariantValue::getTypeAsString() const {
  switch (Type) {
  case VT_String: return "String";
  case VT_Matcher: return getMatcher().getTypeAsString();
  case VT_Unsigned: return "Unsigned";
  case VT_Nothing: return "Nothing";
  }
  llvm_unreachable("Invalid Type");
}

} // end namespace dynamic
} // end namespace ast_matchers
} // end namespace clang
