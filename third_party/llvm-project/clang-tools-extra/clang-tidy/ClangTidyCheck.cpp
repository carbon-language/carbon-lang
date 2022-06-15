//===--- ClangTidyCheck.cpp - clang-tidy ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangTidyCheck.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/YAMLParser.h"

namespace clang {
namespace tidy {

ClangTidyCheck::ClangTidyCheck(StringRef CheckName, ClangTidyContext *Context)
    : CheckName(CheckName), Context(Context),
      Options(CheckName, Context->getOptions().CheckOptions, Context) {
  assert(Context != nullptr);
  assert(!CheckName.empty());
}

DiagnosticBuilder ClangTidyCheck::diag(SourceLocation Loc, StringRef Message,
                                       DiagnosticIDs::Level Level) {
  return Context->diag(CheckName, Loc, Message, Level);
}

DiagnosticBuilder ClangTidyCheck::diag(StringRef Message,
                                       DiagnosticIDs::Level Level) {
  return Context->diag(CheckName, Message, Level);
}

DiagnosticBuilder
ClangTidyCheck::configurationDiag(StringRef Description,
                                  DiagnosticIDs::Level Level) const {
  return Context->configurationDiag(Description, Level);
}

void ClangTidyCheck::run(const ast_matchers::MatchFinder::MatchResult &Result) {
  // For historical reasons, checks don't implement the MatchFinder run()
  // callback directly. We keep the run()/check() distinction to avoid interface
  // churn, and to allow us to add cross-cutting logic in the future.
  check(Result);
}

ClangTidyCheck::OptionsView::OptionsView(
    StringRef CheckName, const ClangTidyOptions::OptionMap &CheckOptions,
    ClangTidyContext *Context)
    : NamePrefix((CheckName + ".").str()), CheckOptions(CheckOptions),
      Context(Context) {}

llvm::Optional<StringRef>
ClangTidyCheck::OptionsView::get(StringRef LocalName) const {
  const auto &Iter = CheckOptions.find((NamePrefix + LocalName).str());
  if (Iter != CheckOptions.end())
    return StringRef(Iter->getValue().Value);
  return None;
}

static ClangTidyOptions::OptionMap::const_iterator
findPriorityOption(const ClangTidyOptions::OptionMap &Options, StringRef NamePrefix,
          StringRef LocalName) {
  auto IterLocal = Options.find((NamePrefix + LocalName).str());
  auto IterGlobal = Options.find(LocalName);
  if (IterLocal == Options.end())
    return IterGlobal;
  if (IterGlobal == Options.end())
    return IterLocal;
  if (IterLocal->getValue().Priority >= IterGlobal->getValue().Priority)
    return IterLocal;
  return IterGlobal;
}

llvm::Optional<StringRef>
ClangTidyCheck::OptionsView::getLocalOrGlobal(StringRef LocalName) const {
  auto Iter = findPriorityOption(CheckOptions, NamePrefix, LocalName);
  if (Iter != CheckOptions.end())
    return StringRef(Iter->getValue().Value);
  return None;
}

static Optional<bool> getAsBool(StringRef Value,
                                const llvm::Twine &LookupName) {

  if (llvm::Optional<bool> Parsed = llvm::yaml::parseBool(Value))
    return *Parsed;
  // To maintain backwards compatability, we support parsing numbers as
  // booleans, even though its not supported in YAML.
  long long Number;
  if (!Value.getAsInteger(10, Number))
    return Number != 0;
  return None;
}

template <>
llvm::Optional<bool>
ClangTidyCheck::OptionsView::get<bool>(StringRef LocalName) const {
  if (llvm::Optional<StringRef> ValueOr = get(LocalName)) {
    if (auto Result = getAsBool(*ValueOr, NamePrefix + LocalName))
      return Result;
    diagnoseBadBooleanOption(NamePrefix + LocalName, *ValueOr);
  }
  return None;
}

template <>
llvm::Optional<bool>
ClangTidyCheck::OptionsView::getLocalOrGlobal<bool>(StringRef LocalName) const {
  auto Iter = findPriorityOption(CheckOptions, NamePrefix, LocalName);
  if (Iter != CheckOptions.end()) {
    if (auto Result = getAsBool(Iter->getValue().Value, Iter->getKey()))
      return Result;
    diagnoseBadBooleanOption(Iter->getKey(), Iter->getValue().Value);
  }
  return None;
}

void ClangTidyCheck::OptionsView::store(ClangTidyOptions::OptionMap &Options,
                                        StringRef LocalName,
                                        StringRef Value) const {
  Options[(NamePrefix + LocalName).str()] = Value;
}

void ClangTidyCheck::OptionsView::storeInt(ClangTidyOptions::OptionMap &Options,
                                           StringRef LocalName,
                                           int64_t Value) const {
  store(Options, LocalName, llvm::itostr(Value));
}

template <>
void ClangTidyCheck::OptionsView::store<bool>(
    ClangTidyOptions::OptionMap &Options, StringRef LocalName,
    bool Value) const {
  store(Options, LocalName, Value ? StringRef("true") : StringRef("false"));
}

llvm::Optional<int64_t> ClangTidyCheck::OptionsView::getEnumInt(
    StringRef LocalName, ArrayRef<NameAndValue> Mapping, bool CheckGlobal,
    bool IgnoreCase) const {
  auto Iter = CheckGlobal
                  ? findPriorityOption(CheckOptions, NamePrefix, LocalName)
                  : CheckOptions.find((NamePrefix + LocalName).str());
  if (Iter == CheckOptions.end())
    return None;

  StringRef Value = Iter->getValue().Value;
  StringRef Closest;
  unsigned EditDistance = 3;
  for (const auto &NameAndEnum : Mapping) {
    if (IgnoreCase) {
      if (Value.equals_insensitive(NameAndEnum.second))
        return NameAndEnum.first;
    } else if (Value.equals(NameAndEnum.second)) {
      return NameAndEnum.first;
    } else if (Value.equals_insensitive(NameAndEnum.second)) {
      Closest = NameAndEnum.second;
      EditDistance = 0;
      continue;
    }
    unsigned Distance =
        Value.edit_distance(NameAndEnum.second, true, EditDistance);
    if (Distance < EditDistance) {
      EditDistance = Distance;
      Closest = NameAndEnum.second;
    }
  }
  if (EditDistance < 3)
    diagnoseBadEnumOption(Iter->getKey(), Iter->getValue().Value, Closest);
  else
    diagnoseBadEnumOption(Iter->getKey(), Iter->getValue().Value);
  return None;
}

static constexpr llvm::StringLiteral ConfigWarning(
    "invalid configuration value '%0' for option '%1'%select{|; expected a "
    "bool|; expected an integer|; did you mean '%3'?}2");

void ClangTidyCheck::OptionsView::diagnoseBadBooleanOption(
    const Twine &Lookup, StringRef Unparsed) const {
  SmallString<64> Buffer;
  Context->configurationDiag(ConfigWarning)
      << Unparsed << Lookup.toStringRef(Buffer) << 1;
}

void ClangTidyCheck::OptionsView::diagnoseBadIntegerOption(
    const Twine &Lookup, StringRef Unparsed) const {
  SmallString<64> Buffer;
  Context->configurationDiag(ConfigWarning)
      << Unparsed << Lookup.toStringRef(Buffer) << 2;
}

void ClangTidyCheck::OptionsView::diagnoseBadEnumOption(
    const Twine &Lookup, StringRef Unparsed, StringRef Suggestion) const {
  SmallString<64> Buffer;
  auto Diag = Context->configurationDiag(ConfigWarning)
              << Unparsed << Lookup.toStringRef(Buffer);
  if (Suggestion.empty())
    Diag << 0;
  else
    Diag << 3 << Suggestion;
}

StringRef ClangTidyCheck::OptionsView::get(StringRef LocalName,
                                           StringRef Default) const {
  return get(LocalName).getValueOr(Default);
}

StringRef
ClangTidyCheck::OptionsView::getLocalOrGlobal(StringRef LocalName,
                                              StringRef Default) const {
  return getLocalOrGlobal(LocalName).getValueOr(Default);
}
} // namespace tidy
} // namespace clang
