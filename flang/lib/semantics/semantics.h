// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FORTRAN_SEMANTICS_SEMANTICS_H_
#define FORTRAN_SEMANTICS_SEMANTICS_H_

#include "scope.h"
#include "../common/Fortran-features.h"
#include "symbol.h"
#include "../evaluate/common.h"
#include "../evaluate/intrinsics.h"
#include "../parser/message.h"
#include <iosfwd>
#include <string>
#include <vector>

namespace Fortran::common {
class IntrinsicTypeDefaultKinds;
}

namespace Fortran::parser {
struct Name;
struct Program;
class CookedSource;
struct AssociateConstruct;
struct BlockConstruct;
struct CaseConstruct;
struct DoConstruct;
struct ChangeTeamConstruct;
struct CriticalConstruct;
struct ForallConstruct;
struct IfConstruct;
struct SelectRankConstruct;
struct SelectTypeConstruct;
struct WhereConstruct;
}

namespace Fortran::semantics {

class Symbol;

using ConstructNode = std::variant<const parser::AssociateConstruct *,
    const parser::BlockConstruct *, const parser::CaseConstruct *,
    const parser::ChangeTeamConstruct *, const parser::CriticalConstruct *,
    const parser::DoConstruct *, const parser::ForallConstruct *,
    const parser::IfConstruct *, const parser::SelectRankConstruct *,
    const parser::SelectTypeConstruct *, const parser::WhereConstruct *>;
using ConstructStack = std::vector<ConstructNode>;

class SemanticsContext {
public:
  SemanticsContext(const common::IntrinsicTypeDefaultKinds &,
      const common::LanguageFeatureControl &, parser::AllSources &);
  ~SemanticsContext();

  const common::IntrinsicTypeDefaultKinds &defaultKinds() const {
    return defaultKinds_;
  }
  const common::LanguageFeatureControl &languageFeatures() const {
    return languageFeatures_;
  };
  int GetDefaultKind(TypeCategory) const;
  int doublePrecisionKind() const {
    return defaultKinds_.doublePrecisionKind();
  }
  int quadPrecisionKind() const { return defaultKinds_.quadPrecisionKind(); }
  bool IsEnabled(common::LanguageFeature) const;
  bool ShouldWarn(common::LanguageFeature) const;
  const std::optional<parser::CharBlock> &location() const { return location_; }
  const std::vector<std::string> &searchDirectories() const {
    return searchDirectories_;
  }
  const std::string &moduleDirectory() const { return moduleDirectory_; }
  const std::string &moduleFileSuffix() const { return moduleFileSuffix_; }
  bool warnOnNonstandardUsage() const { return warnOnNonstandardUsage_; }
  bool warningsAreErrors() const { return warningsAreErrors_; }
  const evaluate::IntrinsicProcTable &intrinsics() const { return intrinsics_; }
  Scope &globalScope() { return globalScope_; }
  parser::Messages &messages() { return messages_; }
  evaluate::FoldingContext &foldingContext() { return foldingContext_; }
  parser::AllSources &allSources() { return allSources_; }

  SemanticsContext &set_location(
      const std::optional<parser::CharBlock> &location) {
    location_ = location;
    return *this;
  }
  SemanticsContext &set_searchDirectories(const std::vector<std::string> &x) {
    searchDirectories_ = x;
    return *this;
  }
  SemanticsContext &set_moduleDirectory(const std::string &x) {
    moduleDirectory_ = x;
    return *this;
  }
  SemanticsContext &set_moduleFileSuffix(const std::string &x) {
    moduleFileSuffix_ = x;
    return *this;
  }
  SemanticsContext &set_warnOnNonstandardUsage(bool x) {
    warnOnNonstandardUsage_ = x;
    return *this;
  }
  SemanticsContext &set_warningsAreErrors(bool x) {
    warningsAreErrors_ = x;
    return *this;
  }

  const DeclTypeSpec &MakeNumericType(TypeCategory, int kind = 0);
  const DeclTypeSpec &MakeLogicalType(int kind = 0);

  bool AnyFatalError() const;

  // Test or set the Error flag on a Symbol
  bool HasError(const Symbol &);
  bool HasError(const Symbol *);
  bool HasError(const parser::Name &);
  void SetError(Symbol &, bool = true);

  template<typename... A> parser::Message &Say(A &&... args) {
    CHECK(location_);
    return messages_.Say(*location_, std::forward<A>(args)...);
  }
  template<typename... A>
  parser::Message &Say(parser::CharBlock at, A &&... args) {
    return messages_.Say(at, std::forward<A>(args)...);
  }
  parser::Message &Say(parser::Message &&msg) {
    return messages_.Say(std::move(msg));
  }
  template<typename... A>
  void SayWithDecl(const Symbol &symbol, const parser::CharBlock &at,
      parser::MessageFixedText &&msg, A &&... args) {
    auto &message{Say(at, std::move(msg), args...)};
    evaluate::AttachDeclaration(&message, &symbol);
  }

  const Scope &FindScope(parser::CharBlock) const;
  Scope &FindScope(parser::CharBlock);

  const ConstructStack &constructStack() const { return constructStack_; }
  template<typename N> void PushConstruct(const N &node) {
    constructStack_.emplace_back(&node);
  }
  void PopConstruct();

private:
  const common::IntrinsicTypeDefaultKinds &defaultKinds_;
  const common::LanguageFeatureControl languageFeatures_;
  parser::AllSources &allSources_;
  std::optional<parser::CharBlock> location_;
  std::vector<std::string> searchDirectories_;
  std::string moduleDirectory_{"."s};
  std::string moduleFileSuffix_{".mod"};
  bool warnOnNonstandardUsage_{false};
  bool warningsAreErrors_{false};
  const evaluate::IntrinsicProcTable intrinsics_;
  Scope globalScope_;
  parser::Messages messages_;
  evaluate::FoldingContext foldingContext_;

  bool CheckError(bool);
  ConstructStack constructStack_;
};

class Semantics {
public:
  explicit Semantics(SemanticsContext &context, parser::Program &program,
      parser::CookedSource &cooked)
    : context_{context}, program_{program}, cooked_{cooked} {
    context.globalScope().AddSourceRange(parser::CharBlock{cooked.data()});
  }

  SemanticsContext &context() const { return context_; }
  bool Perform();
  const Scope &FindScope(const parser::CharBlock &where) const {
    return context_.FindScope(where);
  }
  bool AnyFatalError() const { return context_.AnyFatalError(); }
  void EmitMessages(std::ostream &) const;
  void DumpSymbols(std::ostream &);
  void DumpSymbolsSources(std::ostream &) const;

private:
  SemanticsContext &context_;
  parser::Program &program_;
  const parser::CookedSource &cooked_;
};

// Base class for semantics checkers.
struct BaseChecker {
  template<typename N> void Enter(const N &) {}
  template<typename N> void Leave(const N &) {}
};
}
#endif
