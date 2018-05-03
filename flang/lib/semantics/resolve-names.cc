// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

#include "resolve-names.h"
#include "attr.h"
#include "rewrite-parse-tree.h"
#include "scope.h"
#include "symbol.h"
#include "type.h"
#include "../parser/indirection.h"
#include "../parser/parse-tree-visitor.h"
#include "../parser/parse-tree.h"
#include <list>
#include <memory>
#include <ostream>
#include <set>
#include <stack>

namespace Fortran::semantics {

using namespace parser::literals;

class MessageHandler;

// ImplicitRules maps initial character of identifier to the DeclTypeSpec*
// representing the implicit type; nullptr if none.
class ImplicitRules {
public:
  ImplicitRules(MessageHandler &messages);
  bool isImplicitNoneType() const { return isImplicitNoneType_; }
  bool isImplicitNoneExternal() const { return isImplicitNoneExternal_; }
  void set_isImplicitNoneType(bool x) { isImplicitNoneType_ = x; }
  void set_isImplicitNoneExternal(bool x) { isImplicitNoneExternal_ = x; }
  // Get the implicit type for identifiers starting with ch. May be null.
  const DeclTypeSpec *GetType(char ch) const;
  // Record the implicit type for this range of characters.
  void SetType(const DeclTypeSpec &type, parser::Location lo, parser::Location,
      bool isDefault = false);
  // Apply the default implicit rules (if no IMPLICIT NONE).
  void AddDefaultRules();

private:
  static char Incr(char ch);

  MessageHandler &messages_;
  bool isImplicitNoneType_{false};
  bool isImplicitNoneExternal_{false};
  // map initial character of identifier to nullptr or its default type
  std::map<char, const DeclTypeSpec> map_;
  friend std::ostream &operator<<(std::ostream &, const ImplicitRules &);
  friend void ShowImplicitRule(std::ostream &, const ImplicitRules &, char);
};

// Provide Post methods to collect attributes into a member variable.
class AttrsVisitor {
public:
  void BeginAttrs();
  Attrs EndAttrs();
  void Post(const parser::LanguageBindingSpec &);
  bool Pre(const parser::AccessSpec &);
  bool Pre(const parser::IntentSpec &);

// Simple case: encountering CLASSNAME causes ATTRNAME to be set.
#define HANDLE_ATTR_CLASS(CLASSNAME, ATTRNAME) \
  bool Pre(const parser::CLASSNAME &) { \
    attrs_->set(Attr::ATTRNAME); \
    return false; \
  }
  HANDLE_ATTR_CLASS(PrefixSpec::Elemental, ELEMENTAL)
  HANDLE_ATTR_CLASS(PrefixSpec::Impure, IMPURE)
  HANDLE_ATTR_CLASS(PrefixSpec::Module, MODULE)
  HANDLE_ATTR_CLASS(PrefixSpec::Non_Recursive, NON_RECURSIVE)
  HANDLE_ATTR_CLASS(PrefixSpec::Pure, PURE)
  HANDLE_ATTR_CLASS(PrefixSpec::Recursive, RECURSIVE)
  HANDLE_ATTR_CLASS(TypeAttrSpec::BindC, BIND_C)
  HANDLE_ATTR_CLASS(Abstract, ABSTRACT)
  HANDLE_ATTR_CLASS(Allocatable, ALLOCATABLE)
  HANDLE_ATTR_CLASS(Asynchronous, ASYNCHRONOUS)
  HANDLE_ATTR_CLASS(Contiguous, CONTIGUOUS)
  HANDLE_ATTR_CLASS(External, EXTERNAL)
  HANDLE_ATTR_CLASS(Intrinsic, INTRINSIC)
  HANDLE_ATTR_CLASS(NoPass, NOPASS)
  HANDLE_ATTR_CLASS(Optional, OPTIONAL)
  HANDLE_ATTR_CLASS(Parameter, PARAMETER)
  HANDLE_ATTR_CLASS(Pass, PASS)
  HANDLE_ATTR_CLASS(Pointer, POINTER)
  HANDLE_ATTR_CLASS(Protected, PROTECTED)
  HANDLE_ATTR_CLASS(Save, SAVE)
  HANDLE_ATTR_CLASS(Target, TARGET)
  HANDLE_ATTR_CLASS(Value, VALUE)
  HANDLE_ATTR_CLASS(Volatile, VOLATILE)
#undef HANDLE_ATTR_CLASS

protected:
  std::optional<Attrs> attrs_;
  std::string langBindingName_{""};

  Attr AccessSpecToAttr(const parser::AccessSpec &x) {
    switch (x.v) {
    case parser::AccessSpec::Kind::Public: return Attr::PUBLIC;
    case parser::AccessSpec::Kind::Private: return Attr::PRIVATE;
    }
    // unnecessary but g++ warns "control reaches end of non-void function"
    parser::die("unreachable");
  }
};

// Find and create types from declaration-type-spec nodes.
class DeclTypeSpecVisitor : public AttrsVisitor {
public:
  using AttrsVisitor::Post;
  using AttrsVisitor::Pre;
  void BeginDeclTypeSpec();
  void EndDeclTypeSpec();
  bool Pre(const parser::IntegerTypeSpec &);
  bool Pre(const parser::IntrinsicTypeSpec::Logical &);
  bool Pre(const parser::IntrinsicTypeSpec::Real &);
  bool Pre(const parser::IntrinsicTypeSpec::Complex &);
  bool Pre(const parser::DeclarationTypeSpec::ClassStar &);
  bool Pre(const parser::DeclarationTypeSpec::TypeStar &);
  void Post(const parser::DeclarationTypeSpec::Type &);
  void Post(const parser::DeclarationTypeSpec::Class &);
  bool Pre(const parser::DeclarationTypeSpec::Record &);
  bool Pre(const parser::DerivedTypeSpec &);
  void Post(const parser::TypeParamSpec &);
  bool Pre(const parser::TypeParamValue &);

protected:
  std::unique_ptr<DeclTypeSpec> declTypeSpec_;
  std::unique_ptr<DerivedTypeSpec> derivedTypeSpec_;
  std::unique_ptr<ParamValue> typeParamValue_;

private:
  bool expectDeclTypeSpec_{false};  // should only see decl-type-spec when true
  void MakeIntrinsic(const IntrinsicTypeSpec &intrinsicTypeSpec);
  void SetDeclTypeSpec(const DeclTypeSpec &declTypeSpec);
  static KindParamValue GetKindParamValue(
      const std::optional<parser::KindSelector> &kind);
};

// Track statement source locations and save messages.
class MessageHandler {
public:
  using Message = parser::Message;
  using MessageFixedText = parser::MessageFixedText;
  using MessageFormattedText = parser::MessageFormattedText;

  MessageHandler(parser::Messages &messages) : messages_{messages} {}

  template<typename T> bool Pre(const parser::Statement<T> &x) {
    currStmtSource_ = &x.source;
    return true;
  }
  template<typename T> void Post(const parser::Statement<T> &) {
    currStmtSource_ = nullptr;
  }

  const SourceName *currStmtSource() { return currStmtSource_; }

  // Add a message to the messages to be emitted.
  void Say(Message &&);
  // Emit a message associated with the current statement source.
  void Say(MessageFixedText &&);
  // Emit a message about a SourceName or parser::Name
  void Say(const SourceName &, MessageFixedText &&);
  void Say(const parser::Name &, MessageFixedText &&);
  // Emit a formatted message associated with a source location.
  void Say(const SourceName &, MessageFixedText &&, const std::string &);
  void Say(const SourceName &, MessageFixedText &&, const SourceName &,
      const SourceName &);

private:
  // Where messages are emitted:
  parser::Messages &messages_;
  // Source location of current statement; null if not in a statement
  const SourceName *currStmtSource_{nullptr};
};

// Visit ImplicitStmt and related parse tree nodes and updates implicit rules.
class ImplicitRulesVisitor : public DeclTypeSpecVisitor, public MessageHandler {
public:
  using DeclTypeSpecVisitor::Post;
  using DeclTypeSpecVisitor::Pre;
  using MessageHandler::Post;
  using MessageHandler::Pre;
  using ImplicitNoneNameSpec = parser::ImplicitStmt::ImplicitNoneNameSpec;

  ImplicitRulesVisitor(parser::Messages &messages) : MessageHandler(messages) {}

  void Post(const parser::ParameterStmt &);
  bool Pre(const parser::ImplicitStmt &);
  bool Pre(const parser::LetterSpec &);
  bool Pre(const parser::ImplicitSpec &);
  void Post(const parser::ImplicitSpec &);

  ImplicitRules &implicitRules() { return implicitRules_.top(); }
  const ImplicitRules &implicitRules() const { return implicitRules_.top(); }
  bool isImplicitNoneType() const {
    return implicitRules().isImplicitNoneType();
  }
  bool isImplicitNoneExternal() const {
    return implicitRules().isImplicitNoneExternal();
  }

protected:
  void PushScope();
  void PopScope();
  void CopyImplicitRules();  // copy from parent into this scope

private:
  // implicit rules in effect for current scope
  std::stack<ImplicitRules, std::list<ImplicitRules>> implicitRules_;
  // previous occurence of these kinds of statements:
  const SourceName *prevImplicit_{nullptr};
  const SourceName *prevImplicitNone_{nullptr};
  const SourceName *prevImplicitNoneType_{nullptr};
  const SourceName *prevParameterStmt_{nullptr};

  bool HandleImplicitNone(const std::list<ImplicitNoneNameSpec> &nameSpecs);
};

// Track array specifications. They can occur in AttrSpec, EntityDecl,
// ObjectDecl, DimensionStmt, CommonBlockObject, or BasedPointerStmt.
// 1. INTEGER, DIMENSION(10) :: x
// 2. INTEGER :: x(10)
// 3. ALLOCATABLE :: x(:)
// 4. DIMENSION :: x(10)
// 5. TODO: COMMON x(10)
// 6. TODO: BasedPointerStmt
class ArraySpecVisitor {
public:
  const ArraySpec &arraySpec() {
    return !arraySpec_.empty() ? arraySpec_ : attrArraySpec_;
  }

  void BeginArraySpec() { CHECK(attrArraySpec_.empty()); }
  void EndArraySpec() { attrArraySpec_.clear(); }
  void ClearArraySpec() { arraySpec_.clear(); }

  bool Pre(const parser::ArraySpec &x) {
    CHECK(arraySpec_.empty());
    return true;
  }
  void Post(const parser::AttrSpec &) {
    if (!arraySpec_.empty()) {
      // Example: integer, dimension(<1>) :: x(<2>)
      // This saves <1> in attrArraySpec_ so we can process <2> into arraySpec_
      CHECK(attrArraySpec_.empty());
      attrArraySpec_.splice(attrArraySpec_.cbegin(), arraySpec_);
      CHECK(arraySpec_.empty());
    }
  }

  bool Pre(const parser::DeferredShapeSpecList &x) {
    for (int i = 0; i < x.v; ++i) {
      arraySpec_.push_back(ShapeSpec::MakeDeferred());
    }
    return false;
  }

  bool Pre(const parser::AssumedShapeSpec &x) {
    const auto &lb = x.v;
    arraySpec_.push_back(
        lb ? ShapeSpec::MakeAssumed(GetBound(*lb)) : ShapeSpec::MakeAssumed());
    return false;
  }

  bool Pre(const parser::ExplicitShapeSpec &x) {
    const auto &lb = std::get<std::optional<parser::SpecificationExpr>>(x.t);
    const auto &ub = GetBound(std::get<parser::SpecificationExpr>(x.t));
    arraySpec_.push_back(lb ? ShapeSpec::MakeExplicit(GetBound(*lb), ub)
                            : ShapeSpec::MakeExplicit(ub));
    return false;
  }

  bool Pre(const parser::AssumedImpliedSpec &x) {
    const auto &lb = x.v;
    arraySpec_.push_back(
        lb ? ShapeSpec::MakeImplied(GetBound(*lb)) : ShapeSpec::MakeImplied());
    return false;
  }

  bool Pre(const parser::AssumedRankSpec &) {
    arraySpec_.push_back(ShapeSpec::MakeAssumedRank());
    return false;
  }

private:
  // arraySpec_ is populated by any ArraySpec
  ArraySpec arraySpec_;
  // When an ArraySpec is under an AttrSpec, it is moved into attrArraySpec_
  ArraySpec attrArraySpec_;

  Bound GetBound(const parser::SpecificationExpr &x) {
    return Bound(IntExpr(x.v));
  }
};

// Manage a stack of Scopes
class ScopeHandler : public ImplicitRulesVisitor {
public:
  ScopeHandler(parser::Messages &messages) : ImplicitRulesVisitor(messages) {
    PushScope(Scope::globalScope);
  }
  Scope &CurrScope() { return *scopes_.top(); }
  void PushScope(Scope &scope) {
    scopes_.push(&scope);
    ImplicitRulesVisitor::PushScope();
  }
  void PopScope() {
    ApplyImplicitRules();
    scopes_.pop();
    ImplicitRulesVisitor::PopScope();
  }

  // Helpers to make a Symbol in the current scope
  template<typename D>
  Symbol &MakeSymbol(const SourceName &name, const Attrs &attrs, D &&details) {
    const auto &it = CurrScope().find(name);
    auto &symbol = it->second;
    if (it == CurrScope().end()) {
      const auto pair = CurrScope().try_emplace(name, attrs, details);
      CHECK(pair.second);  // name was not found, so must be able to add
      return pair.first->second;
    }
    symbol.add_occurrence(name);
    if (symbol.has<UnknownDetails>()) {
      // update the existing symbol
      symbol.attrs() |= attrs;
      symbol.set_details(details);
      return symbol;
    } else if (std::is_same<UnknownDetails, D>::value) {
      symbol.attrs() |= attrs;
      return symbol;
    } else {
      Say(name, "'%s' is already declared in this scoping unit"_err_en_US);
      Say(symbol.name(), "Previous declaration of '%s'"_en_US);
      // replace the old symbols with a new one with correct details
      CurrScope().erase(symbol.name());
      return MakeSymbol(name, attrs, details);
    }
  }
  template<typename D>
  Symbol &MakeSymbol(
      const parser::Name &name, const Attrs &attrs, D &&details) {
    return MakeSymbol(name.source, attrs, std::move(details));
  }
  template<typename D>
  Symbol &MakeSymbol(const parser::Name &name, D &&details) {
    return MakeSymbol(name, Attrs(), details);
  }
  Symbol &MakeSymbol(const SourceName &name, Attrs attrs = Attrs{}) {
    return MakeSymbol(name, attrs, UnknownDetails());
  }

private:
  // Stack of containing scopes; memory referenced is owned by parent scopes
  std::stack<Scope *, std::list<Scope *>> scopes_;

  // On leaving a scope, add implicit types if appropriate.
  void ApplyImplicitRules();
};

class ModuleVisitor : public ScopeHandler {
public:
  using ImplicitRulesVisitor::Post;
  using ImplicitRulesVisitor::Pre;

  ModuleVisitor(parser::Messages &messages) : ScopeHandler(messages) {}

  bool Pre(const parser::ModuleStmt &);
  void Post(const parser::EndModuleStmt &);
  bool Pre(const parser::AccessStmt &);

  bool Pre(const parser::Only &x) {
    std::visit(
        parser::visitors{
            [&](const parser::Indirection<parser::GenericSpec> &generic) {
              std::visit(
                  parser::visitors{
                      [&](const parser::Name &name) { AddUse(name); },
                      [](const auto &) { parser::die("TODO: GenericSpec"); },
                  },
                  generic->u);
            },
            [&](const parser::Name &name) { AddUse(name); },
            [&](const parser::Rename &rename) {
              std::visit(
                  parser::visitors{
                      [&](const parser::Rename::Names &names) {
                        AddUse(names);
                      },
                      [&](const parser::Rename::Operators &ops) {
                        parser::die("TODO: Rename::Operators");
                      },
                  },
                  rename.u);
            },
        },
        x.u);
    return false;
  }

  bool Pre(const parser::Rename::Names &x) {
    AddUse(x);
    return false;
  }

  // Set useModuleScope_ to the Scope of the module being used.
  bool Pre(const parser::UseStmt &x) {
    // x.nature = UseStmt::ModuleNature::Intrinsic or Non_Intrinsic
    const auto it = Scope::globalScope.find(x.moduleName.source);
    if (it == Scope::globalScope.end()) {
      Say(x.moduleName, "Module '%s' not found"_err_en_US);
      return false;
    }
    const auto *details = it->second.detailsIf<ModuleDetails>();
    if (!details) {
      Say(x.moduleName, "'%s' is not a module"_err_en_US);
      return false;
    }
    useModuleScope_ = details->scope();
    CHECK(useModuleScope_);
    return true;
  }

  void Post(const parser::UseStmt &x) {
    if (const auto *list = std::get_if<std::list<parser::Rename>>(&x.u)) {
      // Not a use-only: collect the names that were used in renames,
      // then add a use for each public name that was not renamed.
      std::set<SourceName> useNames;
      for (const auto &rename : *list) {
        std::visit(
            parser::visitors{
                [&](const parser::Rename::Names &names) {
                  useNames.insert(std::get<1>(names.t).source);
                },
                [&](const parser::Rename::Operators &ops) {
                  CHECK(!"TODO: Rename::Operators");
                },
            },
            rename.u);
      }
      const SourceName &moduleName{x.moduleName.source};
      for (const auto &pair : *useModuleScope_) {
        const Symbol &symbol{pair.second};
        if (symbol.attrs().test(Attr::PUBLIC) &&
            !symbol.detailsIf<ModuleDetails>()) {
          const SourceName &name{symbol.name()};
          if (useNames.count(name) == 0) {
            AddUse(moduleName, name, name);
          }
        }
      }
    }
    useModuleScope_ = nullptr;
  }

private:
  // The default access spec for this module.
  Attr defaultAccess_{Attr::PUBLIC};
  // The location of the last AccessStmt without access-ids, if any.
  const SourceName *prevAccessStmt_{nullptr};
  // The scope of the module during a UseStmt
  const Scope *useModuleScope_{nullptr};
  void SetAccess(const parser::Name &, Attr);
  void ApplyDefaultAccess();

  void AddUse(const parser::Rename::Names &names) {
    const SourceName &useName{std::get<0>(names.t).source};
    const SourceName &localName{std::get<1>(names.t).source};
    AddUse(useName, useName, localName);
  }
  void AddUse(const parser::Name &useName) {
    AddUse(useName.source, useName.source, useName.source);
  }
  // Record a use from useModuleScope_ of useName as localName. location is
  // where it occurred (either the module or the rename) for error reporting.
  void AddUse(const SourceName &location, const SourceName &localName,
      const SourceName &useName) {
    if (!useModuleScope_) {
      return;  // error occurred finding module
    }
    const auto it = useModuleScope_->find(useName);
    if (it == useModuleScope_->end()) {
      Say(useName, "'%s' not found in module '%s'"_err_en_US, useName,
          useModuleScope_->name());
      return;
    }
    const Symbol &useSymbol{it->second};
    if (useSymbol.attrs().test(Attr::PRIVATE)) {
      Say(useName, "'%s' is PRIVATE in '%s'"_err_en_US, useName,
          useModuleScope_->name());
      return;
    }
    Symbol &localSymbol{MakeSymbol(localName, useSymbol.attrs())};
    localSymbol.attrs() &= ~Attrs{Attr::PUBLIC, Attr::PRIVATE};
    if (auto *details = localSymbol.detailsIf<UseDetails>()) {
      // check for importing the same symbol again:
      if (localSymbol.GetUltimate() != useSymbol.GetUltimate()) {
        localSymbol.set_details(
            UseErrorDetails{details->location(), *useModuleScope_});
      }
    } else if (auto *details = localSymbol.detailsIf<UseErrorDetails>()) {
      details->add_occurrence(location, *useModuleScope_);
    } else if (localSymbol.has<UnknownDetails>()) {
      localSymbol.set_details(UseDetails{location, useSymbol});
    } else {
      CHECK(!"can't happen");
    }
  }
};

// Walk the parse tree and resolve names to symbols.
class ResolveNamesVisitor : public ArraySpecVisitor, public ModuleVisitor {
public:
  using ArraySpecVisitor::Post;
  using ArraySpecVisitor::Pre;
  using ModuleVisitor::Post;
  using ModuleVisitor::Pre;

  ResolveNamesVisitor(parser::Messages &messages) : ModuleVisitor(messages) {}

  // Default action for a parse tree node is to visit children.
  template<typename T> bool Pre(const T &) { return true; }
  template<typename T> void Post(const T &) {}

  bool Pre(const parser::TypeDeclarationStmt &);
  void Post(const parser::TypeDeclarationStmt &);
  void Post(const parser::EntityDecl &);
  void Post(const parser::ObjectDecl &);
  bool Pre(const parser::PrefixSpec &);
  bool Pre(const parser::AsynchronousStmt &);
  bool Pre(const parser::ContiguousStmt &);
  bool Pre(const parser::ExternalStmt &);
  bool Pre(const parser::IntrinsicStmt &);
  bool Pre(const parser::OptionalStmt &);
  bool Pre(const parser::ProtectedStmt &);
  bool Pre(const parser::ValueStmt &);
  bool Pre(const parser::VolatileStmt &);
  void Post(const parser::SpecificationPart &);
  bool Pre(const parser::Suffix &);
  bool Pre(const parser::StmtFunctionStmt &);
  void Post(const parser::StmtFunctionStmt &);
  bool Pre(const parser::SubroutineStmt &);
  void Post(const parser::SubroutineStmt &);
  void Post(const parser::EndSubroutineStmt &);
  bool Pre(const parser::FunctionStmt &);
  void Post(const parser::FunctionStmt &);
  void Post(const parser::EndFunctionStmt &);
  bool Pre(const parser::MainProgram &);
  void Post(const parser::EndProgramStmt &);
  void Post(const parser::Program &);

  bool Pre(const parser::AllocatableStmt &) {
    objectDeclAttr_ = Attr::ALLOCATABLE;
    return true;
  }
  void Post(const parser::AllocatableStmt &) { objectDeclAttr_ = std::nullopt; }
  bool Pre(const parser::TargetStmt &x) {
    objectDeclAttr_ = Attr::TARGET;
    return true;
  }
  void Post(const parser::TargetStmt &) { objectDeclAttr_ = std::nullopt; }
  void Post(const parser::DimensionStmt::Declaration &);

  void Post(const parser::Expr &x) { CheckImplicitSymbol(GetVariableName(x)); }
  void Post(const parser::Variable &x) {
    CheckImplicitSymbol(GetVariableName(x));
  }

  void Post(const parser::ProcedureDesignator &x) {
    if (const auto *name = std::get_if<parser::Name>(&x.u)) {
      Symbol &symbol{MakeSymbol(name->source)};
      if (symbol.has<UnknownDetails>()) {
        if (isImplicitNoneExternal() && !symbol.attrs().test(Attr::EXTERNAL)) {
          Say(*name,
              "'%s' is an external procedure without the EXTERNAL"
              " attribute in a scope with IMPLICIT NONE(EXTERNAL)"_err_en_US);
        }
        symbol.attrs().set(Attr::EXTERNAL);
        symbol.set_details(SubprogramDetails{});
      } else if (!symbol.has<SubprogramDetails>()) {
        auto *details = symbol.detailsIf<EntityDetails>();
        if (!details || !details->isArray()) {
          Say(*name,
              "Use of '%s' as a procedure conflicts with its declaration"_err_en_US);
          Say(symbol.name(), "Declaration of '%s'"_en_US);
        }
      }
    }
  }

private:
  // Function result name from parser::Suffix, if any.
  const parser::Name *funcResultName_{nullptr};
  // The attribute corresponding to the statement containing an ObjectDecl
  std::optional<Attr> objectDeclAttr_;
  // Set when we see a stmt function that is really an array element assignment
  bool badStmtFuncFound_{false};

  // Create a subprogram symbol in the current scope and push a new scope.
  Symbol &PushSubprogramScope(const parser::Name &);

  // Handle a statement that sets an attribute on a list of names.
  bool HandleAttributeStmt(Attr, const std::list<parser::Name> &);

  void DeclareEntity(const parser::Name &, Attrs);
  const parser::Name *GetVariableName(const parser::DataRef &);
  const parser::Name *GetVariableName(const parser::Designator &);
  const parser::Name *GetVariableName(const parser::Expr &);
  const parser::Name *GetVariableName(const parser::Variable &);
  void CheckImplicitSymbol(const parser::Name *);
};

// ImplicitRules implementation

ImplicitRules::ImplicitRules(MessageHandler &messages) : messages_{messages} {}

const DeclTypeSpec *ImplicitRules::GetType(char ch) const {
  auto it = map_.find(ch);
  return it != map_.end() ? &it->second : nullptr;
}

// isDefault is set when we are applying the default rules, so it is not
// an error if the type is already set.
void ImplicitRules::SetType(const DeclTypeSpec &type, parser::Location lo,
    parser::Location hi, bool isDefault) {
  for (char ch = *lo; ch; ch = ImplicitRules::Incr(ch)) {
    auto res = map_.emplace(ch, type);
    if (!res.second && !isDefault) {
      messages_.Say(lo,
          "More than one implicit type specified for '%s'"_err_en_US,
          std::string(1, ch));
    }
    if (ch == *hi) {
      break;
    }
  }
}

void ImplicitRules::AddDefaultRules() {
  SetType(DeclTypeSpec::MakeIntrinsic(IntegerTypeSpec::Make()), "i", "n", true);
  SetType(DeclTypeSpec::MakeIntrinsic(RealTypeSpec::Make()), "a", "z", true);
}

// Return the next char after ch in a way that works for ASCII or EBCDIC.
// Return '\0' for the char after 'z'.
char ImplicitRules::Incr(char ch) {
  switch (ch) {
  case 'i': return 'j';
  case 'r': return 's';
  case 'z': return '\0';
  default: return ch + 1;
  }
}

std::ostream &operator<<(std::ostream &o, const ImplicitRules &implicitRules) {
  o << "ImplicitRules:\n";
  for (char ch = 'a'; ch; ch = ImplicitRules::Incr(ch)) {
    ShowImplicitRule(o, implicitRules, ch);
  }
  ShowImplicitRule(o, implicitRules, '_');
  ShowImplicitRule(o, implicitRules, '$');
  ShowImplicitRule(o, implicitRules, '@');
  return o;
}
void ShowImplicitRule(
    std::ostream &o, const ImplicitRules &implicitRules, char ch) {
  auto it = implicitRules.map_.find(ch);
  if (it != implicitRules.map_.end()) {
    o << "  " << ch << ": " << it->second << '\n';
  }
}

// AttrsVisitor implementation

void AttrsVisitor::BeginAttrs() {
  CHECK(!attrs_);
  attrs_ = std::make_optional<Attrs>();
}
Attrs AttrsVisitor::EndAttrs() {
  CHECK(attrs_);
  Attrs result{*attrs_};
  attrs_.reset();
  return result;
}
void AttrsVisitor::Post(const parser::LanguageBindingSpec &x) {
  attrs_->set(Attr::BIND_C);
  if (x.v) {
    // TODO: set langBindingName_ from ScalarDefaultCharConstantExpr
  }
}
bool AttrsVisitor::Pre(const parser::AccessSpec &x) {
  attrs_->set(AccessSpecToAttr(x));
  return false;
}
bool AttrsVisitor::Pre(const parser::IntentSpec &x) {
  switch (x.v) {
  case parser::IntentSpec::Intent::In: attrs_->set(Attr::INTENT_IN); break;
  case parser::IntentSpec::Intent::Out: attrs_->set(Attr::INTENT_OUT); break;
  case parser::IntentSpec::Intent::InOut:
    attrs_->set(Attr::INTENT_IN);
    attrs_->set(Attr::INTENT_OUT);
    break;
  }
  return false;
}

// DeclTypeSpecVisitor implementation

void DeclTypeSpecVisitor::BeginDeclTypeSpec() {
  CHECK(!expectDeclTypeSpec_);
  expectDeclTypeSpec_ = true;
}
void DeclTypeSpecVisitor::EndDeclTypeSpec() {
  CHECK(expectDeclTypeSpec_);
  expectDeclTypeSpec_ = false;
  declTypeSpec_.reset();
}

bool DeclTypeSpecVisitor::Pre(const parser::DeclarationTypeSpec::ClassStar &x) {
  SetDeclTypeSpec(DeclTypeSpec::MakeClassStar());
  return false;
}
bool DeclTypeSpecVisitor::Pre(const parser::DeclarationTypeSpec::TypeStar &x) {
  SetDeclTypeSpec(DeclTypeSpec::MakeTypeStar());
  return false;
}
bool DeclTypeSpecVisitor::Pre(const parser::DerivedTypeSpec &x) {
  CHECK(!derivedTypeSpec_);
  derivedTypeSpec_ =
      std::make_unique<DerivedTypeSpec>(std::get<parser::Name>(x.t).ToString());
  return true;
}
void DeclTypeSpecVisitor::Post(const parser::TypeParamSpec &x) {
  if (const auto &keyword = std::get<std::optional<parser::Keyword>>(x.t)) {
    derivedTypeSpec_->AddParamValue(keyword->v.ToString(), *typeParamValue_);
  } else {
    derivedTypeSpec_->AddParamValue(*typeParamValue_);
  }
  typeParamValue_.reset();
}
bool DeclTypeSpecVisitor::Pre(const parser::TypeParamValue &x) {
  typeParamValue_ = std::make_unique<ParamValue>(std::visit(
      parser::visitors{
          [&](const parser::ScalarIntExpr &x) { return Bound{IntExpr{x}}; },
          [&](const parser::Star &x) { return Bound::ASSUMED; },
          [&](const parser::TypeParamValue::Deferred &x) {
            return Bound::DEFERRED;
          },
      },
      x.u));
  return false;
}

void DeclTypeSpecVisitor::Post(const parser::DeclarationTypeSpec::Type &) {
  SetDeclTypeSpec(
      DeclTypeSpec::MakeTypeDerivedType(std::move(derivedTypeSpec_)));
}
void DeclTypeSpecVisitor::Post(const parser::DeclarationTypeSpec::Class &) {
  SetDeclTypeSpec(
      DeclTypeSpec::MakeClassDerivedType(std::move(derivedTypeSpec_)));
}
bool DeclTypeSpecVisitor::Pre(const parser::DeclarationTypeSpec::Record &x) {
  // TODO
  return true;
}
bool DeclTypeSpecVisitor::Pre(const parser::IntegerTypeSpec &x) {
  MakeIntrinsic(IntegerTypeSpec::Make(GetKindParamValue(x.v)));
  return false;
}
bool DeclTypeSpecVisitor::Pre(const parser::IntrinsicTypeSpec::Logical &x) {
  MakeIntrinsic(LogicalTypeSpec::Make(GetKindParamValue(x.kind)));
  return false;
}
bool DeclTypeSpecVisitor::Pre(const parser::IntrinsicTypeSpec::Real &x) {
  MakeIntrinsic(RealTypeSpec::Make(GetKindParamValue(x.kind)));
  return false;
}
bool DeclTypeSpecVisitor::Pre(const parser::IntrinsicTypeSpec::Complex &x) {
  MakeIntrinsic(ComplexTypeSpec::Make(GetKindParamValue(x.kind)));
  return false;
}
void DeclTypeSpecVisitor::MakeIntrinsic(
    const IntrinsicTypeSpec &intrinsicTypeSpec) {
  SetDeclTypeSpec(DeclTypeSpec::MakeIntrinsic(intrinsicTypeSpec));
}
// Check that we're expecting to see a DeclTypeSpec (and haven't seen one yet)
// and save it in declTypeSpec_.
void DeclTypeSpecVisitor::SetDeclTypeSpec(const DeclTypeSpec &declTypeSpec) {
  CHECK(expectDeclTypeSpec_);
  CHECK(!declTypeSpec_);
  declTypeSpec_ = std::make_unique<DeclTypeSpec>(declTypeSpec);
}

KindParamValue DeclTypeSpecVisitor::GetKindParamValue(
    const std::optional<parser::KindSelector> &kind) {
  if (!kind) {
    return KindParamValue();
  } else if (const auto *expr =
                 std::get_if<parser::ScalarIntConstantExpr>(&kind->u)) {
    const auto &lit =
        std::get<parser::LiteralConstant>(expr->thing.thing.thing->u);
    const auto &intlit = std::get<parser::IntLiteralConstant>(lit.u);
    return KindParamValue(std::get<std::uint64_t>(intlit.t));
  } else {
    CHECK(!"TODO: translate star-size to kind");
    return {};  // silence compiler warning
  }
}

// MessageHandler implementation

void MessageHandler::Say(Message &&msg) { messages_.Put(std::move(msg)); }
void MessageHandler::Say(MessageFixedText &&msg) {
  CHECK(currStmtSource_);
  Say(Message{*currStmtSource_, std::move(msg)});
}
void MessageHandler::Say(const SourceName &name, MessageFixedText &&msg) {
  Say(name, std::move(msg), name.ToString());
}
void MessageHandler::Say(const parser::Name &name, MessageFixedText &&msg) {
  Say(name.source, std::move(msg), name.ToString());
}
void MessageHandler::Say(const SourceName &location, MessageFixedText &&msg,
    const std::string &arg1) {
  Say(Message{location, MessageFormattedText{msg, arg1.c_str()}});
}
void MessageHandler::Say(const SourceName &location, MessageFixedText &&msg,
    const SourceName &arg1, const SourceName &arg2) {
  Say(Message{location,
      MessageFormattedText{
          msg, arg1.ToString().c_str(), arg2.ToString().c_str()}});
}

// ImplicitRulesVisitor implementation

void ImplicitRulesVisitor::Post(const parser::ParameterStmt &x) {
  prevParameterStmt_ = currStmtSource();
}

bool ImplicitRulesVisitor::Pre(const parser::ImplicitStmt &x) {
  bool res = std::visit(
      parser::visitors{
          [&](const std::list<ImplicitNoneNameSpec> &x) {
            return HandleImplicitNone(x);
          },
          [&](const std::list<parser::ImplicitSpec> &x) {
            if (prevImplicitNoneType_) {
              Say("IMPLICIT statement after IMPLICIT NONE or "
                  "IMPLICIT NONE(TYPE) statement"_err_en_US);
              return false;
            }
            return true;
          },
      },
      x.u);
  prevImplicit_ = currStmtSource();
  return res;
}

bool ImplicitRulesVisitor::Pre(const parser::LetterSpec &x) {
  auto loLoc = std::get<parser::Location>(x.t);
  auto hiLoc = loLoc;
  if (auto hiLocOpt = std::get<std::optional<parser::Location>>(x.t)) {
    hiLoc = *hiLocOpt;
    if (*hiLoc < *loLoc) {
      Say(hiLoc, "'%s' does not follow '%s' alphabetically"_err_en_US,
          std::string(hiLoc, 1), std::string(loLoc, 1));
      return false;
    }
  }
  implicitRules().SetType(*declTypeSpec_.get(), loLoc, hiLoc);
  return false;
}

bool ImplicitRulesVisitor::Pre(const parser::ImplicitSpec &) {
  BeginDeclTypeSpec();
  return true;
}

void ImplicitRulesVisitor::Post(const parser::ImplicitSpec &) {
  EndDeclTypeSpec();
}

void ImplicitRulesVisitor::PushScope() {
  implicitRules_.push(ImplicitRules(*this));
  prevImplicit_ = nullptr;
  prevImplicitNone_ = nullptr;
  prevImplicitNoneType_ = nullptr;
  prevParameterStmt_ = nullptr;
}

void ImplicitRulesVisitor::CopyImplicitRules() {
  implicitRules_.pop();
  implicitRules_.push(ImplicitRules(implicitRules_.top()));
}

void ImplicitRulesVisitor::PopScope() { implicitRules_.pop(); }

// TODO: for all of these errors, reference previous statement too
bool ImplicitRulesVisitor::HandleImplicitNone(
    const std::list<ImplicitNoneNameSpec> &nameSpecs) {
  if (prevImplicitNone_ != nullptr) {
    Say("More than one IMPLICIT NONE statement"_err_en_US);
    return false;
  }
  if (prevParameterStmt_ != nullptr) {
    Say("IMPLICIT NONE statement after PARAMETER statement"_err_en_US);
    return false;
  }
  prevImplicitNone_ = currStmtSource();
  if (nameSpecs.empty()) {
    prevImplicitNoneType_ = currStmtSource();
    implicitRules().set_isImplicitNoneType(true);
    if (prevImplicit_) {
      Say("IMPLICIT NONE statement after IMPLICIT statement"_err_en_US);
      return false;
    }
  } else {
    int sawType{0};
    int sawExternal{0};
    for (const auto noneSpec : nameSpecs) {
      switch (noneSpec) {
      case ImplicitNoneNameSpec::External:
        implicitRules().set_isImplicitNoneExternal(true);
        ++sawExternal;
        break;
      case ImplicitNoneNameSpec::Type:
        prevImplicitNoneType_ = currStmtSource();
        implicitRules().set_isImplicitNoneType(true);
        if (prevImplicit_) {
          Say("IMPLICIT NONE(TYPE) after IMPLICIT statement"_err_en_US);
          return false;
        }
        ++sawType;
        break;
      }
    }
    if (sawType > 1) {
      Say("TYPE specified more than once in IMPLICIT NONE statement"_err_en_US);
      return false;
    }
    if (sawExternal > 1) {
      Say("EXTERNAL specified more than once in IMPLICIT NONE statement"_err_en_US);
      return false;
    }
  }
  return true;
}

// ScopeHandler implementation

void ScopeHandler::ApplyImplicitRules() {
  if (!isImplicitNoneType()) {
    implicitRules().AddDefaultRules();
    for (auto &pair : CurrScope()) {
      Symbol &symbol = pair.second;
      if (symbol.has<UnknownDetails>()) {
        symbol.set_details(EntityDetails());
      }
      if (auto *details = symbol.detailsIf<EntityDetails>()) {
        if (!details->type()) {
          const auto &name = pair.first;
          if (const auto *type = implicitRules().GetType(name.begin()[0])) {
            details->set_type(*type);
          } else {
            Say(name, "No explicit type declared for '%s'"_err_en_US);
          }
        }
      }
    }
  }
}

// ModuleVisitor implementation

bool ModuleVisitor::Pre(const parser::ModuleStmt &stmt) {
  const auto &name = stmt.v;
  auto &symbol = MakeSymbol(name, ModuleDetails{});
  ModuleDetails &details{symbol.details<ModuleDetails>()};
  Scope &modScope = CurrScope().MakeScope(Scope::Kind::Module, &symbol);
  details.set_scope(&modScope);
  PushScope(modScope);
  MakeSymbol(name, ModuleDetails{details});
  return false;
}

void ModuleVisitor::Post(const parser::EndModuleStmt &) {
  ApplyDefaultAccess();
  PopScope();
}

void ModuleVisitor::ApplyDefaultAccess() {
  for (auto &pair : CurrScope()) {
    Symbol &symbol = pair.second;
    if (!symbol.attrs().HasAny({Attr::PUBLIC, Attr::PRIVATE})) {
      symbol.attrs().set(defaultAccess_);
    }
  }
}

// ResolveNamesVisitor implementation

void ResolveNamesVisitor::Post(const parser::EntityDecl &x) {
  // TODO: may be under StructureStmt
  const auto &name{std::get<parser::ObjectName>(x.t)};
  // TODO: CoarraySpec, CharLength, Initialization
  DeclareEntity(name, attrs_ ? *attrs_ : Attrs());
}

bool ResolveNamesVisitor::Pre(const parser::TypeDeclarationStmt &x) {
  BeginDeclTypeSpec();
  BeginAttrs();
  BeginArraySpec();
  return true;
}

void ResolveNamesVisitor::Post(const parser::TypeDeclarationStmt &x) {
  EndDeclTypeSpec();
  EndAttrs();
  EndArraySpec();
}

bool ResolveNamesVisitor::Pre(const parser::PrefixSpec &x) {
  return true;  // TODO
}

bool ResolveNamesVisitor::Pre(const parser::AsynchronousStmt &x) {
  return HandleAttributeStmt(Attr::ASYNCHRONOUS, x.v);
}
bool ResolveNamesVisitor::Pre(const parser::ContiguousStmt &x) {
  return HandleAttributeStmt(Attr::CONTIGUOUS, x.v);
}
bool ResolveNamesVisitor::Pre(const parser::ExternalStmt &x) {
  return HandleAttributeStmt(Attr::EXTERNAL, x.v);
}
bool ResolveNamesVisitor::Pre(const parser::IntrinsicStmt &x) {
  return HandleAttributeStmt(Attr::INTRINSIC, x.v);
}
bool ResolveNamesVisitor::Pre(const parser::OptionalStmt &x) {
  return HandleAttributeStmt(Attr::OPTIONAL, x.v);
}
bool ResolveNamesVisitor::Pre(const parser::ProtectedStmt &x) {
  return HandleAttributeStmt(Attr::PROTECTED, x.v);
}
bool ResolveNamesVisitor::Pre(const parser::ValueStmt &x) {
  return HandleAttributeStmt(Attr::VALUE, x.v);
}
bool ResolveNamesVisitor::Pre(const parser::VolatileStmt &x) {
  return HandleAttributeStmt(Attr::VOLATILE, x.v);
}
bool ResolveNamesVisitor::HandleAttributeStmt(
    Attr attr, const std::list<parser::Name> &names) {
  for (const auto &name : names) {
    const auto pair = CurrScope().try_emplace(name.source, Attrs{attr});
    if (!pair.second) {
      // symbol was already there: set attribute on it
      Symbol &symbol{pair.first->second};
      if (attr != Attr::ASYNCHRONOUS && attr != Attr::VOLATILE &&
          symbol.has<UseDetails>()) {
        Say(*currStmtSource(),
            "Cannot change %s attribute on use-associated '%s'"_err_en_US,
            EnumToString(attr), name.source);
      }
      symbol.attrs().set(attr);
    }
  }
  return false;
}

void ResolveNamesVisitor::Post(const parser::ObjectDecl &x) {
  CHECK(objectDeclAttr_.has_value());
  const auto &name = std::get<parser::ObjectName>(x.t);
  DeclareEntity(name, Attrs{*objectDeclAttr_});
}

void ResolveNamesVisitor::Post(const parser::DimensionStmt::Declaration &x) {
  const auto &name = std::get<parser::Name>(x.t);
  DeclareEntity(name, Attrs{});
}

void ResolveNamesVisitor::DeclareEntity(const parser::Name &name, Attrs attrs) {
  Symbol &symbol{MakeSymbol(name.source, attrs)};
  // TODO: check attribute consistency
  if (symbol.has<UnknownDetails>()) {
    symbol.set_details(EntityDetails());
  }
  if (EntityDetails *details = symbol.detailsIf<EntityDetails>()) {
    if (declTypeSpec_) {
      if (details->type().has_value()) {
        Say(name, "The type of '%s' has already been declared"_err_en_US);
      } else {
        details->set_type(*declTypeSpec_);
      }
    }
    if (!arraySpec().empty()) {
      if (!details->shape().empty()) {
        Say(name,
            "The dimensions of '%s' have already been declared"_err_en_US);
      } else {
        details->set_shape(arraySpec());
      }
      ClearArraySpec();
    }
  } else if (UseDetails *details = symbol.detailsIf<UseDetails>()) {
    Say(name.source,
        "'%s' is use-associated from module '%s' and cannot be re-declared"_err_en_US,
        name.source, details->module().name());
  } else {
    Say(name, "'%s' is already declared in this scoping unit"_err_en_US);
    Say(symbol.name(), "Previous declaration of '%s'"_en_US);
  }
}

bool ModuleVisitor::Pre(const parser::AccessStmt &x) {
  Attr accessAttr = AccessSpecToAttr(std::get<parser::AccessSpec>(x.t));
  if (CurrScope().kind() != Scope::Kind::Module) {
    Say(*currStmtSource(),
        "%s statement may only appear in the specification part of a module"_err_en_US,
        EnumToString(accessAttr));
    return false;
  }
  const auto &accessIds = std::get<std::list<parser::AccessId>>(x.t);
  if (accessIds.empty()) {
    if (prevAccessStmt_) {
      Say("The default accessibility of this module has already been declared"_err_en_US);
      Say(*prevAccessStmt_, "Previous declaration"_en_US);
    }
    prevAccessStmt_ = currStmtSource();
    defaultAccess_ = accessAttr;
  } else {
    for (const auto &accessId : accessIds) {
      std::visit(
          parser::visitors{
              [=](const parser::Name &y) { SetAccess(y, accessAttr); },
              [=](const parser::Indirection<parser::GenericSpec> &y) {
                std::visit(
                    parser::visitors{
                        [=](const parser::Name &z) {
                          SetAccess(z, accessAttr);
                        },
                        [](const auto &) { parser::die("TODO: GenericSpec"); },
                    },
                    y->u);
              },
          },
          accessId.u);
    }
  }
  return false;
}

// Set the access specification for this name.
void ModuleVisitor::SetAccess(const parser::Name &name, Attr attr) {
  Symbol &symbol{MakeSymbol(name.source)};
  Attrs &attrs{symbol.attrs()};
  if (attrs.HasAny({Attr::PUBLIC, Attr::PRIVATE})) {
    // PUBLIC/PRIVATE already set: make it a fatal error if it changed
    Attr prev = attrs.test(Attr::PUBLIC) ? Attr::PUBLIC : Attr::PRIVATE;
    Say(name.source,
        attr == prev
            ? "The accessibility of '%s' has already been specified as %s"_en_US
            : "The accessibility of '%s' has already been specified as %s"_err_en_US,
        name.source, EnumToString(prev));
  } else {
    attrs.set(attr);
  }
}

void ResolveNamesVisitor::Post(const parser::SpecificationPart &s) {
  badStmtFuncFound_ = false;
  if (isImplicitNoneType()) {
    // Check that every name referenced has an explicit type
    for (const auto &pair : CurrScope()) {
      const auto &name = pair.first;
      const auto &symbol = pair.second;
      if (symbol.has<UnknownDetails>()) {
        Say(name, "No explicit type declared for '%s'"_err_en_US);
      } else if (const auto *details = symbol.detailsIf<EntityDetails>()) {
        if (!details->type()) {
          Say(name, "No explicit type declared for '%s'"_err_en_US);
        }
      }
    }
  }
}

void ResolveNamesVisitor::Post(const parser::EndSubroutineStmt &subp) {
  PopScope();
}

void ResolveNamesVisitor::Post(const parser::EndFunctionStmt &subp) {
  PopScope();
}

bool ResolveNamesVisitor::Pre(const parser::Suffix &suffix) {
  funcResultName_ = &suffix.resultName.value();
  return true;
}

bool ResolveNamesVisitor::Pre(const parser::StmtFunctionStmt &x) {
  const auto &name = std::get<parser::Name>(x.t);
  std::optional<SourceName> occurrence;
  std::optional<DeclTypeSpec> resultType;
  // Look up name: provides return type or tells us if it's an array
  auto it = CurrScope().find(name.source);
  if (it != CurrScope().end()) {
    Symbol &symbol{it->second};
    if (auto *details = symbol.detailsIf<EntityDetails>()) {
      if (details->isArray()) {
        // not a stmt-func at all but an array; do nothing
        symbol.add_occurrence(name.source);
        badStmtFuncFound_ = true;
        return true;
      }
      // TODO: check that attrs are compatible with stmt func
      resultType = details->type();
      occurrence = symbol.name();
      CurrScope().erase(symbol.name());
    }
  }
  if (badStmtFuncFound_) {
    Say(name, "'%s' has not been declared as an array"_err_en_US);
    return true;
  }
  BeginAttrs();  // no attrs to collect, but PushSubprogramScope expects this
  auto &symbol = PushSubprogramScope(name);
  CopyImplicitRules();
  if (occurrence) {
    symbol.add_occurrence(*occurrence);
  }
  auto &details = symbol.details<SubprogramDetails>();
  for (const auto &dummyName : std::get<std::list<parser::Name>>(x.t)) {
    EntityDetails dummyDetails{true};
    auto it = CurrScope().parent().find(dummyName.source);
    if (it != CurrScope().parent().end()) {
      if (auto *d = it->second.detailsIf<EntityDetails>()) {
        if (d->type()) {
          dummyDetails.set_type(*d->type());
        }
      }
    }
    details.add_dummyArg(MakeSymbol(dummyName, std::move(dummyDetails)));
  }
  CurrScope().erase(name.source);  // added by PushSubprogramScope
  EntityDetails resultDetails;
  if (resultType) {
    resultDetails.set_type(*resultType);
  }
  details.set_result(MakeSymbol(name, resultDetails));
  return true;
}

void ResolveNamesVisitor::Post(const parser::StmtFunctionStmt &x) {
  if (badStmtFuncFound_) {
    return;  // This wasn't really a stmt function so no scope was created
  }
  PopScope();
}

bool ResolveNamesVisitor::Pre(const parser::SubroutineStmt &stmt) {
  BeginAttrs();
  return true;
}
bool ResolveNamesVisitor::Pre(const parser::FunctionStmt &stmt) {
  BeginAttrs();
  BeginDeclTypeSpec();
  CHECK(!funcResultName_);
  return true;
}

void ResolveNamesVisitor::Post(const parser::SubroutineStmt &stmt) {
  const auto &subrName = std::get<parser::Name>(stmt.t);
  auto &symbol = PushSubprogramScope(subrName);
  auto &details = symbol.details<SubprogramDetails>();
  for (const auto &dummyArg : std::get<std::list<parser::DummyArg>>(stmt.t)) {
    const parser::Name *dummyName = std::get_if<parser::Name>(&dummyArg.u);
    CHECK(dummyName != nullptr && "TODO: alternate return indicator");
    Symbol &dummy{MakeSymbol(*dummyName, EntityDetails(true))};
    details.add_dummyArg(dummy);
  }
}

void ResolveNamesVisitor::Post(const parser::FunctionStmt &stmt) {
  const auto &funcName = std::get<parser::Name>(stmt.t);
  auto &symbol = PushSubprogramScope(funcName);
  auto &details = symbol.details<SubprogramDetails>();
  for (const auto &dummyName : std::get<std::list<parser::Name>>(stmt.t)) {
    Symbol &dummy{MakeSymbol(dummyName, EntityDetails(true))};
    details.add_dummyArg(dummy);
  }
  // add function result to function scope
  EntityDetails funcResultDetails;
  if (declTypeSpec_) {
    funcResultDetails.set_type(*declTypeSpec_);
  }
  EndDeclTypeSpec();

  const parser::Name *funcResultName;
  if (funcResultName_ && funcResultName_->source != funcName.source) {
    funcResultName = funcResultName_;
    funcResultName_ = nullptr;
  } else {
    CurrScope().erase(funcName.source);  // was added by PushSubprogramScope
    funcResultName = &funcName;
  }
  details.set_result(MakeSymbol(*funcResultName, funcResultDetails));
}

Symbol &ResolveNamesVisitor::PushSubprogramScope(const parser::Name &name) {
  auto &symbol = MakeSymbol(name, EndAttrs(), SubprogramDetails());
  Scope &subpScope = CurrScope().MakeScope(Scope::Kind::Subprogram, &symbol);
  PushScope(subpScope);
  auto &details = symbol.details<SubprogramDetails>();
  // can't reuse this name inside subprogram:
  MakeSymbol(name, SubprogramDetails(details));
  return symbol;
}

bool ResolveNamesVisitor::Pre(const parser::MainProgram &x) {
  Scope &scope = CurrScope().MakeScope(Scope::Kind::MainProgram);
  PushScope(scope);
  using stmtType = std::optional<parser::Statement<parser::ProgramStmt>>;
  if (const stmtType &stmt = std::get<stmtType>(x.t)) {
    const parser::Name &name{stmt->statement.v};
    MakeSymbol(name, MainProgramDetails());
  }
  return true;
}

void ResolveNamesVisitor::Post(const parser::EndProgramStmt &) { PopScope(); }

const parser::Name *ResolveNamesVisitor::GetVariableName(
    const parser::DataRef &x) {
  return std::get_if<parser::Name>(&x.u);
}
const parser::Name *ResolveNamesVisitor::GetVariableName(
    const parser::Designator &x) {
  return std::visit(
      parser::visitors{
          [&](const parser::ObjectName &x) { return &x; },
          [&](const parser::DataRef &x) { return GetVariableName(x); },
          [&](const auto &) {
            return static_cast<const parser::Name *>(nullptr);
          },
      },
      x.u);
}
const parser::Name *ResolveNamesVisitor::GetVariableName(
    const parser::Expr &x) {
  if (const auto *designator =
          std::get_if<parser::Indirection<parser::Designator>>(&x.u)) {
    return GetVariableName(**designator);
  } else {
    return nullptr;
  }
}
const parser::Name *ResolveNamesVisitor::GetVariableName(
    const parser::Variable &x) {
  if (const auto *designator =
          std::get_if<parser::Indirection<parser::Designator>>(&x.u)) {
    return GetVariableName(**designator);
  } else {
    return nullptr;
  }
}

// If implicit types are allowed, ensure name is in the symbol table
void ResolveNamesVisitor::CheckImplicitSymbol(const parser::Name *name) {
  if (name) {
    const auto &it = CurrScope().find(name->source);
    if (const auto *details = it->second.detailsIf<UseErrorDetails>()) {
      Say(*name, "Reference to '%s' is ambiguous"_err_en_US);
      for (const auto &pair : details->occurrences()) {
        const SourceName &location{*pair.first};
        const SourceName &moduleName{pair.second->name()};
        Say(location, "'%s' was use-associated from module '%s'"_en_US,
            name->source, moduleName);
      }
    } else if (!isImplicitNoneType()) {
      CurrScope().try_emplace(name->source);
    } else if (it == CurrScope().end() || it->second.has<UnknownDetails>()) {
      Say(*name, "No explicit type declared for '%s'"_err_en_US);
    }
  }
}

void ResolveNamesVisitor::Post(const parser::Program &) {
  // ensure that all temps were deallocated
  CHECK(!attrs_);
  CHECK(!declTypeSpec_);
}

void ResolveNames(
    parser::Program &program, const parser::CookedSource &cookedSource) {
  parser::Messages messages;
  ResolveNamesVisitor visitor{messages};
  parser::Walk(static_cast<const parser::Program &>(program), visitor);
  if (!messages.empty()) {
    messages.Emit(std::cerr, cookedSource);
    return;
  }
  RewriteParseTree(program);
}

static void PutIndent(std::ostream &os, int indent) {
  for (int i = 0; i < indent; ++i) {
    os << "  ";
  }
}

static void DumpSymbols(std::ostream &os, const Scope &scope, int indent = 0) {
  PutIndent(os, indent);
  os << Scope::EnumToString(scope.kind()) << " scope:";
  if (const auto *symbol = scope.symbol()) {
    os << ' ' << symbol->name().ToString();
  }
  os << '\n';
  ++indent;
  for (const auto &symbol : scope) {
    PutIndent(os, indent);
    os << symbol.second << "\n";
  }
  for (const auto &child : scope.children()) {
    DumpSymbols(os, child, indent);
  }
  --indent;
}

void DumpSymbols(std::ostream &os) { DumpSymbols(os, Scope::globalScope); }

}  // namespace Fortran::semantics
