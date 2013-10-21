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

VariantMatcher::MatcherOps::~MatcherOps() {}
VariantMatcher::Payload::~Payload() {}

class VariantMatcher::SinglePayload : public VariantMatcher::Payload {
public:
  SinglePayload(const DynTypedMatcher &Matcher) : Matcher(Matcher) {}

  virtual llvm::Optional<DynTypedMatcher> getSingleMatcher() const {
    return Matcher;
  }

  virtual std::string getTypeAsString() const {
    return (Twine("Matcher<") + Matcher.getSupportedKind().asStringRef() + ">")
        .str();
  }

  virtual void makeTypedMatcher(MatcherOps &Ops) const {
    if (Ops.canConstructFrom(Matcher))
      Ops.constructFrom(Matcher);
  }

private:
  const DynTypedMatcher Matcher;
};

class VariantMatcher::PolymorphicPayload : public VariantMatcher::Payload {
public:
  PolymorphicPayload(ArrayRef<DynTypedMatcher> MatchersIn)
      : Matchers(MatchersIn) {}

  virtual ~PolymorphicPayload() {}

  virtual llvm::Optional<DynTypedMatcher> getSingleMatcher() const {
    if (Matchers.size() != 1)
      return llvm::Optional<DynTypedMatcher>();
    return Matchers[0];
  }

  virtual std::string getTypeAsString() const {
    std::string Inner;
    for (size_t i = 0, e = Matchers.size(); i != e; ++i) {
      if (i != 0)
        Inner += "|";
      Inner += Matchers[i].getSupportedKind().asStringRef();
    }
    return (Twine("Matcher<") + Inner + ">").str();
  }

  virtual void makeTypedMatcher(MatcherOps &Ops) const {
    const DynTypedMatcher *Found = NULL;
    for (size_t i = 0, e = Matchers.size(); i != e; ++i) {
      if (Ops.canConstructFrom(Matchers[i])) {
        if (Found)
          return;
        Found = &Matchers[i];
      }
    }
    if (Found)
      Ops.constructFrom(*Found);
  }

  const std::vector<DynTypedMatcher> Matchers;
};

class VariantMatcher::VariadicOpPayload : public VariantMatcher::Payload {
public:
  VariadicOpPayload(ast_matchers::internal::VariadicOperatorFunction Func,
                    ArrayRef<VariantMatcher> Args)
      : Func(Func), Args(Args) {}

  virtual llvm::Optional<DynTypedMatcher> getSingleMatcher() const {
    return llvm::Optional<DynTypedMatcher>();
  }

  virtual std::string getTypeAsString() const {
    std::string Inner;
    for (size_t i = 0, e = Args.size(); i != e; ++i) {
      if (i != 0)
        Inner += "&";
      Inner += Args[i].getTypeAsString();
    }
    return Inner;
  }

  virtual void makeTypedMatcher(MatcherOps &Ops) const {
    Ops.constructVariadicOperator(Func, Args);
  }

private:
  const ast_matchers::internal::VariadicOperatorFunction Func;
  const std::vector<VariantMatcher> Args;
};

VariantMatcher::VariantMatcher() {}

VariantMatcher VariantMatcher::SingleMatcher(const DynTypedMatcher &Matcher) {
  return VariantMatcher(new SinglePayload(Matcher));
}

VariantMatcher
VariantMatcher::PolymorphicMatcher(ArrayRef<DynTypedMatcher> Matchers) {
  return VariantMatcher(new PolymorphicPayload(Matchers));
}

VariantMatcher VariantMatcher::VariadicOperatorMatcher(
    ast_matchers::internal::VariadicOperatorFunction Func,
    ArrayRef<VariantMatcher> Args) {
  return VariantMatcher(new VariadicOpPayload(Func, Args));
}

llvm::Optional<DynTypedMatcher> VariantMatcher::getSingleMatcher() const {
  return Value ? Value->getSingleMatcher() : llvm::Optional<DynTypedMatcher>();
}

void VariantMatcher::reset() { Value.reset(); }

std::string VariantMatcher::getTypeAsString() const {
  if (Value) return Value->getTypeAsString();
  return "<Nothing>";
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
