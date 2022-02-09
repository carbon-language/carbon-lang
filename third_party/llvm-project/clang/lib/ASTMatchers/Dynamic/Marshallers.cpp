//===--- Marshallers.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Marshallers.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Regex.h"
#include <string>

static llvm::Optional<std::string>
getBestGuess(llvm::StringRef Search, llvm::ArrayRef<llvm::StringRef> Allowed,
             llvm::StringRef DropPrefix = "", unsigned MaxEditDistance = 3) {
  if (MaxEditDistance != ~0U)
    ++MaxEditDistance;
  llvm::StringRef Res;
  for (const llvm::StringRef &Item : Allowed) {
    if (Item.equals_insensitive(Search)) {
      assert(!Item.equals(Search) && "This should be handled earlier on.");
      MaxEditDistance = 1;
      Res = Item;
      continue;
    }
    unsigned Distance = Item.edit_distance(Search);
    if (Distance < MaxEditDistance) {
      MaxEditDistance = Distance;
      Res = Item;
    }
  }
  if (!Res.empty())
    return Res.str();
  if (!DropPrefix.empty()) {
    --MaxEditDistance; // Treat dropping the prefix as 1 edit
    for (const llvm::StringRef &Item : Allowed) {
      auto NoPrefix = Item;
      if (!NoPrefix.consume_front(DropPrefix))
        continue;
      if (NoPrefix.equals_insensitive(Search)) {
        if (NoPrefix.equals(Search))
          return Item.str();
        MaxEditDistance = 1;
        Res = Item;
        continue;
      }
      unsigned Distance = NoPrefix.edit_distance(Search);
      if (Distance < MaxEditDistance) {
        MaxEditDistance = Distance;
        Res = Item;
      }
    }
    if (!Res.empty())
      return Res.str();
  }
  return llvm::None;
}

llvm::Optional<std::string>
clang::ast_matchers::dynamic::internal::ArgTypeTraits<
    clang::attr::Kind>::getBestGuess(const VariantValue &Value) {
  static constexpr llvm::StringRef Allowed[] = {
#define ATTR(X) "attr::" #X,
#include "clang/Basic/AttrList.inc"
  };
  if (Value.isString())
    return ::getBestGuess(Value.getString(), llvm::makeArrayRef(Allowed),
                          "attr::");
  return llvm::None;
}

llvm::Optional<std::string>
clang::ast_matchers::dynamic::internal::ArgTypeTraits<
    clang::CastKind>::getBestGuess(const VariantValue &Value) {
  static constexpr llvm::StringRef Allowed[] = {
#define CAST_OPERATION(Name) "CK_" #Name,
#include "clang/AST/OperationKinds.def"
  };
  if (Value.isString())
    return ::getBestGuess(Value.getString(), llvm::makeArrayRef(Allowed),
                          "CK_");
  return llvm::None;
}

llvm::Optional<std::string>
clang::ast_matchers::dynamic::internal::ArgTypeTraits<
    clang::OpenMPClauseKind>::getBestGuess(const VariantValue &Value) {
  static constexpr llvm::StringRef Allowed[] = {
#define GEN_CLANG_CLAUSE_CLASS
#define CLAUSE_CLASS(Enum, Str, Class) #Enum,
#include "llvm/Frontend/OpenMP/OMP.inc"
  };
  if (Value.isString())
    return ::getBestGuess(Value.getString(), llvm::makeArrayRef(Allowed),
                          "OMPC_");
  return llvm::None;
}

llvm::Optional<std::string>
clang::ast_matchers::dynamic::internal::ArgTypeTraits<
    clang::UnaryExprOrTypeTrait>::getBestGuess(const VariantValue &Value) {
  static constexpr llvm::StringRef Allowed[] = {
#define UNARY_EXPR_OR_TYPE_TRAIT(Spelling, Name, Key) "UETT_" #Name,
#define CXX11_UNARY_EXPR_OR_TYPE_TRAIT(Spelling, Name, Key) "UETT_" #Name,
#include "clang/Basic/TokenKinds.def"
  };
  if (Value.isString())
    return ::getBestGuess(Value.getString(), llvm::makeArrayRef(Allowed),
                          "UETT_");
  return llvm::None;
}

static constexpr std::pair<llvm::StringRef, llvm::Regex::RegexFlags>
    RegexMap[] = {
        {"NoFlags", llvm::Regex::RegexFlags::NoFlags},
        {"IgnoreCase", llvm::Regex::RegexFlags::IgnoreCase},
        {"Newline", llvm::Regex::RegexFlags::Newline},
        {"BasicRegex", llvm::Regex::RegexFlags::BasicRegex},
};

static llvm::Optional<llvm::Regex::RegexFlags>
getRegexFlag(llvm::StringRef Flag) {
  for (const auto &StringFlag : RegexMap) {
    if (Flag == StringFlag.first)
      return StringFlag.second;
  }
  return llvm::None;
}

static llvm::Optional<llvm::StringRef>
getCloseRegexMatch(llvm::StringRef Flag) {
  for (const auto &StringFlag : RegexMap) {
    if (Flag.edit_distance(StringFlag.first) < 3)
      return StringFlag.first;
  }
  return llvm::None;
}

llvm::Optional<llvm::Regex::RegexFlags>
clang::ast_matchers::dynamic::internal::ArgTypeTraits<
    llvm::Regex::RegexFlags>::getFlags(llvm::StringRef Flags) {
  llvm::Optional<llvm::Regex::RegexFlags> Flag;
  SmallVector<StringRef, 4> Split;
  Flags.split(Split, '|', -1, false);
  for (StringRef OrFlag : Split) {
    if (llvm::Optional<llvm::Regex::RegexFlags> NextFlag =
            getRegexFlag(OrFlag.trim()))
      Flag = Flag.getValueOr(llvm::Regex::NoFlags) | *NextFlag;
    else
      return None;
  }
  return Flag;
}

llvm::Optional<std::string>
clang::ast_matchers::dynamic::internal::ArgTypeTraits<
    llvm::Regex::RegexFlags>::getBestGuess(const VariantValue &Value) {
  if (!Value.isString())
    return llvm::None;
  SmallVector<StringRef, 4> Split;
  llvm::StringRef(Value.getString()).split(Split, '|', -1, false);
  for (llvm::StringRef &Flag : Split) {
    if (llvm::Optional<llvm::StringRef> BestGuess =
            getCloseRegexMatch(Flag.trim()))
      Flag = *BestGuess;
    else
      return None;
  }
  if (Split.empty())
    return None;
  return llvm::join(Split, " | ");
}
