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

#include "resolve-names.h"
#include "assignment.h"
#include "attr.h"
#include "expression.h"
#include "mod-file.h"
#include "program-tree.h"
#include "resolve-names-utils.h"
#include "rewrite-parse-tree.h"
#include "scope.h"
#include "semantics.h"
#include "symbol.h"
#include "tools.h"
#include "type.h"
#include "../common/Fortran.h"
#include "../common/default-kinds.h"
#include "../common/indirection.h"
#include "../common/restorer.h"
#include "../evaluate/characteristics.h"
#include "../evaluate/common.h"
#include "../evaluate/fold.h"
#include "../evaluate/intrinsics.h"
#include "../evaluate/tools.h"
#include "../evaluate/type.h"
#include "../parser/parse-tree-visitor.h"
#include "../parser/parse-tree.h"
#include "../parser/tools.h"
#include <list>
#include <map>
#include <ostream>
#include <set>
#include <stack>

namespace Fortran::semantics {

using namespace parser::literals;

template<typename T> using Indirection = common::Indirection<T>;
using Message = parser::Message;
using Messages = parser::Messages;
using MessageFixedText = parser::MessageFixedText;
using MessageFormattedText = parser::MessageFormattedText;

class ResolveNamesVisitor;

// ImplicitRules maps initial character of identifier to the DeclTypeSpec
// representing the implicit type; std::nullopt if none.
// It also records the presence of IMPLICIT NONE statements.
// When inheritFromParent is set, defaults come from the parent rules.
class ImplicitRules {
public:
  ImplicitRules(SemanticsContext &context, ImplicitRules *parent)
    : parent_{parent}, context_{context} {
    inheritFromParent_ = parent != nullptr;
  }
  bool isImplicitNoneType() const;
  bool isImplicitNoneExternal() const;
  void set_isImplicitNoneType(bool x) { isImplicitNoneType_ = x; }
  void set_isImplicitNoneExternal(bool x) { isImplicitNoneExternal_ = x; }
  void set_inheritFromParent(bool x) { inheritFromParent_ = x; }
  // Get the implicit type for identifiers starting with ch. May be null.
  const DeclTypeSpec *GetType(char ch) const;
  // Record the implicit type for this range of characters.
  void SetType(const DeclTypeSpec &type, parser::Location lo, parser::Location,
      bool isDefault = false);

private:
  static char Incr(char ch);

  ImplicitRules *parent_;
  SemanticsContext &context_;
  bool inheritFromParent_;  // look in parent if not specified here
  std::optional<bool> isImplicitNoneType_;
  std::optional<bool> isImplicitNoneExternal_;
  // map initial character of identifier to nullptr or its default type
  std::map<char, const DeclTypeSpec *> map_;

  friend std::ostream &operator<<(std::ostream &, const ImplicitRules &);
  friend void ShowImplicitRule(std::ostream &, const ImplicitRules &, char);
};

// Track statement source locations and save messages.
class MessageHandler {
public:
  Messages &messages() { return context_->messages(); };
  void set_context(SemanticsContext &context) { context_ = &context; }
  const SourceName *currStmtSource() { return context_->location(); }
  void set_currStmtSource(const SourceName *source) {
    context_->set_location(source);
  }

  // Emit a message associated with the current statement source.
  Message &Say(MessageFixedText &&);
  Message &Say(MessageFormattedText &&);
  // Emit a message about a SourceName
  Message &Say(const SourceName &, MessageFixedText &&);
  // Emit a formatted message associated with a source location.
  template<typename... A>
  Message &Say(const SourceName &source, MessageFixedText &&msg, A &&... args) {
    return context_->Say(source, std::move(msg), std::forward<A>(args)...);
  }

private:
  SemanticsContext *context_{nullptr};
};

// Inheritance graph for the parse tree visitation classes that follow:
//   BaseVisitor
//   + AttrsVisitor
//   | + DeclTypeSpecVisitor
//   |   + ImplicitRulesVisitor
//   |     + ScopeHandler -----------+--+
//   |       + ModuleVisitor ========|==+
//   |       + InterfaceVisitor      |  |
//   |       +-+ SubprogramVisitor ==|==+
//   + ArraySpecVisitor              |  |
//     + DeclarationVisitor <--------+  |
//       + ConstructVisitor             |
//         + ResolveNamesVisitor <------+

class BaseVisitor {
public:
  template<typename T> void Walk(const T &);
  void set_this(ResolveNamesVisitor *x) { this_ = x; }

  MessageHandler &messageHandler() { return messageHandler_; }
  const SourceName *currStmtSource() { return context_->location(); }
  SemanticsContext &context() const { return *context_; }
  void set_context(SemanticsContext &);
  evaluate::FoldingContext &GetFoldingContext() const {
    return context_->foldingContext();
  }

  // Make a placeholder symbol for a Name that otherwise wouldn't have one.
  // It is not in any scope and always has MiscDetails.
  void MakePlaceholder(const parser::Name &, MiscDetails::Kind);

  template<typename T> common::IfNoLvalue<T, T> FoldExpr(T &&expr) {
    return evaluate::Fold(GetFoldingContext(), std::move(expr));
  }

  template<typename T> MaybeExpr EvaluateExpr(const T &expr) {
    return FoldExpr(AnalyzeExpr(*context_, expr));
  }

  template<typename T>
  MaybeExpr EvaluateConvertedExpr(
      const Symbol &symbol, const T &expr, parser::CharBlock source) {
    if (auto maybeExpr{AnalyzeExpr(*context_, expr)}) {
      if (auto converted{
              evaluate::ConvertToType(symbol, std::move(*maybeExpr))}) {
        return FoldExpr(std::move(*converted));
      } else {
        Say(source,
            "Initialization expression could not be converted to declared type of symbol '%s'"_err_en_US,
            symbol.name());
      }
    }
    return std::nullopt;
  }

  template<typename T> MaybeIntExpr EvaluateIntExpr(const T &expr) {
    if (MaybeExpr maybeExpr{EvaluateExpr(expr)}) {
      if (auto *intExpr{evaluate::UnwrapExpr<SomeIntExpr>(*maybeExpr)}) {
        return std::move(*intExpr);
      }
    }
    return std::nullopt;
  }

  template<typename T>
  MaybeSubscriptIntExpr EvaluateSubscriptIntExpr(const T &expr) {
    if (MaybeIntExpr maybeIntExpr{EvaluateIntExpr(expr)}) {
      return FoldExpr(evaluate::ConvertToType<evaluate::SubscriptInteger>(
          std::move(*maybeIntExpr)));
    } else {
      return std::nullopt;
    }
  }

  template<typename... A> Message &Say(A &&... args) {
    return messageHandler_.Say(std::forward<A>(args)...);
  }
  template<typename... A>
  Message &Say(
      const parser::Name &name, MessageFixedText &&text, const A &... args) {
    return messageHandler_.Say(name.source, std::move(text), args...);
  }

private:
  ResolveNamesVisitor *this_{nullptr};
  SemanticsContext *context_{nullptr};
  MessageHandler messageHandler_;
};

// Provide Post methods to collect attributes into a member variable.
class AttrsVisitor : public virtual BaseVisitor {
public:
  bool BeginAttrs();  // always returns true
  Attrs GetAttrs();
  Attrs EndAttrs();
  bool SetPassNameOn(Symbol &);
  bool SetBindNameOn(Symbol &);
  void Post(const parser::LanguageBindingSpec &);
  bool Pre(const parser::AccessSpec &);
  bool Pre(const parser::IntentSpec &);
  bool Pre(const parser::Pass &);

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
  HANDLE_ATTR_CLASS(BindAttr::Deferred, DEFERRED)
  HANDLE_ATTR_CLASS(BindAttr::Non_Overridable, NON_OVERRIDABLE)
  HANDLE_ATTR_CLASS(Abstract, ABSTRACT)
  HANDLE_ATTR_CLASS(Allocatable, ALLOCATABLE)
  HANDLE_ATTR_CLASS(Asynchronous, ASYNCHRONOUS)
  HANDLE_ATTR_CLASS(Contiguous, CONTIGUOUS)
  HANDLE_ATTR_CLASS(External, EXTERNAL)
  HANDLE_ATTR_CLASS(Intrinsic, INTRINSIC)
  HANDLE_ATTR_CLASS(NoPass, NOPASS)
  HANDLE_ATTR_CLASS(Optional, OPTIONAL)
  HANDLE_ATTR_CLASS(Parameter, PARAMETER)
  HANDLE_ATTR_CLASS(Pointer, POINTER)
  HANDLE_ATTR_CLASS(Protected, PROTECTED)
  HANDLE_ATTR_CLASS(Save, SAVE)
  HANDLE_ATTR_CLASS(Target, TARGET)
  HANDLE_ATTR_CLASS(Value, VALUE)
  HANDLE_ATTR_CLASS(Volatile, VOLATILE)
#undef HANDLE_ATTR_CLASS

protected:
  std::optional<Attrs> attrs_;

  Attr AccessSpecToAttr(const parser::AccessSpec &x) {
    switch (x.v) {
    case parser::AccessSpec::Kind::Public: return Attr::PUBLIC;
    case parser::AccessSpec::Kind::Private: return Attr::PRIVATE;
    }
    common::die("unreachable");  // suppress g++ warning
  }
  Attr IntentSpecToAttr(const parser::IntentSpec &x) {
    switch (x.v) {
    case parser::IntentSpec::Intent::In: return Attr::INTENT_IN;
    case parser::IntentSpec::Intent::Out: return Attr::INTENT_OUT;
    case parser::IntentSpec::Intent::InOut: return Attr::INTENT_INOUT;
    }
    common::die("unreachable");  // suppress g++ warning
  }

private:
  MaybeExpr bindName_;  // from BIND(C, NAME="...")
  const SourceName *passName_{nullptr};  // from PASS(...)
};

// Find and create types from declaration-type-spec nodes.
class DeclTypeSpecVisitor : public AttrsVisitor {
public:
  explicit DeclTypeSpecVisitor() {}
  using AttrsVisitor::Post;
  using AttrsVisitor::Pre;
  void Post(const parser::IntrinsicTypeSpec::DoublePrecision &);
  void Post(const parser::IntrinsicTypeSpec::DoubleComplex &);
  void Post(const parser::DeclarationTypeSpec::ClassStar &);
  void Post(const parser::DeclarationTypeSpec::TypeStar &);
  bool Pre(const parser::TypeGuardStmt &);
  void Post(const parser::TypeGuardStmt &);
  void Post(const parser::TypeSpec &);

protected:
  struct State {
    bool expectDeclTypeSpec{false};  // should see decl-type-spec only when true
    const DeclTypeSpec *declTypeSpec{nullptr};
    struct {
      DerivedTypeSpec *type{nullptr};
      DeclTypeSpec::Category category{DeclTypeSpec::TypeDerived};
    } derived;
  };

  // Walk the parse tree of a type spec and return the DeclTypeSpec for it.
  template<typename T> const DeclTypeSpec *ProcessTypeSpec(const T &x) {
    auto save{common::ScopedSet(state_, State{})};
    BeginDeclTypeSpec();
    Walk(x);
    const auto *type{GetDeclTypeSpec()};
    EndDeclTypeSpec();
    return type;
  }

  const DeclTypeSpec *GetDeclTypeSpec();
  void BeginDeclTypeSpec();
  void EndDeclTypeSpec();
  void SetDeclTypeSpec(const DeclTypeSpec &);
  void SetDeclTypeSpecCategory(DeclTypeSpec::Category);
  DeclTypeSpec::Category GetDeclTypeSpecCategory() const {
    return state_.derived.category;
  }
  KindExpr GetKindParamExpr(
      TypeCategory, const std::optional<parser::KindSelector> &);

private:
  State state_;

  void MakeNumericType(TypeCategory, int kind);
};

// Visit ImplicitStmt and related parse tree nodes and updates implicit rules.
class ImplicitRulesVisitor : public DeclTypeSpecVisitor {
public:
  using DeclTypeSpecVisitor::Post;
  using DeclTypeSpecVisitor::Pre;
  using ImplicitNoneNameSpec = parser::ImplicitStmt::ImplicitNoneNameSpec;

  void Post(const parser::ParameterStmt &);
  bool Pre(const parser::ImplicitStmt &);
  bool Pre(const parser::LetterSpec &);
  bool Pre(const parser::ImplicitSpec &);
  void Post(const parser::ImplicitSpec &);

  ImplicitRules &implicitRules() { return *implicitRules_; }
  const ImplicitRules &implicitRules() const { return *implicitRules_; }
  bool isImplicitNoneType() const {
    return implicitRules().isImplicitNoneType();
  }
  bool isImplicitNoneExternal() const {
    return implicitRules().isImplicitNoneExternal();
  }

protected:
  void BeginScope(const Scope &);
  void SetScope(const Scope &);

private:
  // scope -> implicit rules for that scope
  std::map<const Scope *, ImplicitRules> implicitRulesMap_;
  // implicit rules in effect for current scope
  ImplicitRules *implicitRules_{nullptr};
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
// 5. COMMON x(10)
// 6. TODO: BasedPointerStmt
class ArraySpecVisitor : public virtual BaseVisitor {
public:
  void Post(const parser::ArraySpec &);
  void Post(const parser::ComponentArraySpec &);
  void Post(const parser::CoarraySpec &);
  void Post(const parser::AttrSpec &) { PostAttrSpec(); }
  void Post(const parser::ComponentAttrSpec &) { PostAttrSpec(); }

protected:
  const ArraySpec &arraySpec();
  const ArraySpec &coarraySpec();
  void BeginArraySpec();
  void EndArraySpec();
  void ClearArraySpec() { arraySpec_.clear(); }
  void ClearCoarraySpec() { coarraySpec_.clear(); }

private:
  // arraySpec_/coarraySpec_ are populated from any ArraySpec/CoarraySpec
  ArraySpec arraySpec_;
  ArraySpec coarraySpec_;
  // When an ArraySpec is under an AttrSpec or ComponentAttrSpec, it is moved
  // into attrArraySpec_
  ArraySpec attrArraySpec_;
  ArraySpec attrCoarraySpec_;

  void PostAttrSpec();
};

// Manage a stack of Scopes
class ScopeHandler : public ImplicitRulesVisitor {
public:
  using ImplicitRulesVisitor::Post;
  using ImplicitRulesVisitor::Pre;

  Scope &currScope() { return *currScope_; }
  // The enclosing scope, skipping blocks and derived types.
  Scope &InclusiveScope();
  // The global scope, containing program units.
  Scope &GlobalScope();

  // Create a new scope and push it on the scope stack.
  void PushScope(Scope::Kind kind, Symbol *symbol);
  void PushScope(Scope &scope);
  void PopScope();
  void SetScope(Scope &);

  template<typename T> bool Pre(const parser::Statement<T> &x) {
    messageHandler().set_currStmtSource(&x.source);
    currScope_->AddSourceRange(x.source);
    return true;
  }
  template<typename T> void Post(const parser::Statement<T> &) {
    messageHandler().set_currStmtSource(nullptr);
  }

  // Special messages: already declared; referencing symbol's declaration;
  // about a type; two names & locations
  void SayAlreadyDeclared(const SourceName &, Symbol &);
  void SayAlreadyDeclared(const parser::Name &, Symbol &);
  void SayWithReason(
      const parser::Name &, Symbol &, MessageFixedText &&, MessageFixedText &&);
  void SayWithDecl(const parser::Name &, Symbol &, MessageFixedText &&);
  void SayLocalMustBeVariable(const parser::Name &, Symbol &);
  void SayDerivedType(const SourceName &, MessageFixedText &&, const Scope &);
  void Say2(const SourceName &, MessageFixedText &&, const SourceName &,
      MessageFixedText &&);
  void Say2(
      const SourceName &, MessageFixedText &&, Symbol &, MessageFixedText &&);
  void Say2(
      const parser::Name &, MessageFixedText &&, Symbol &, MessageFixedText &&);

  // Search for symbol by name in current and containing scopes
  Symbol *FindSymbol(const parser::Name &);
  Symbol *FindSymbol(const Scope &, const parser::Name &);
  // Search for name only in scope, not in enclosing scopes.
  Symbol *FindInScope(const Scope &, const parser::Name &);
  Symbol *FindInScope(const Scope &, const SourceName &);
  // Search for name in a derived type scope and its parents.
  Symbol *FindInTypeOrParents(const Scope &, SourceName);
  Symbol *FindInTypeOrParents(const Scope &, const parser::Name &);
  Symbol *FindInTypeOrParents(const parser::Name &);
  void EraseSymbol(const parser::Name &);
  void EraseSymbol(const Symbol &symbol) { currScope().erase(symbol.name()); }
  // Make a new symbol with the name and attrs of an existing one
  Symbol &CopySymbol(const SourceName &, const Symbol &);

  // Make symbols in the current or named scope
  Symbol &MakeSymbol(Scope &, const SourceName &, Attrs);
  Symbol &MakeSymbol(const SourceName &, Attrs = Attrs{});
  Symbol &MakeSymbol(const parser::Name &, Attrs = Attrs{});

  template<typename D>
  common::IfNoLvalue<Symbol &, D> MakeSymbol(
      const parser::Name &name, D &&details) {
    return MakeSymbol(name, Attrs{}, std::move(details));
  }

  template<typename D>
  common::IfNoLvalue<Symbol &, D> MakeSymbol(
      const parser::Name &name, const Attrs &attrs, D &&details) {
    return Resolve(name, MakeSymbol(name.source, attrs, std::move(details)));
  }

  template<typename D>
  common::IfNoLvalue<Symbol &, D> MakeSymbol(
      const SourceName &name, const Attrs &attrs, D &&details) {
    // Note: don't use FindSymbol here. If this is a derived type scope,
    // we want to detect whether the name is already declared as a component.
    auto *symbol{FindInScope(currScope(), name)};
    if (!symbol) {
      symbol = &MakeSymbol(name, attrs);
      symbol->set_details(std::move(details));
      return *symbol;
    }
    if constexpr (std::is_same_v<DerivedTypeDetails, D>) {
      if (auto *d{symbol->detailsIf<GenericDetails>()}) {
        // derived type with same name as a generic
        auto *derivedType{d->derivedType()};
        if (!derivedType) {
          derivedType =
              &currScope().MakeSymbol(name, attrs, std::move(details));
          d->set_derivedType(*derivedType);
        } else {
          SayAlreadyDeclared(name, *derivedType);
        }
        return *derivedType;
      }
    }
    if (symbol->CanReplaceDetails(details)) {
      // update the existing symbol
      symbol->attrs() |= attrs;
      symbol->set_details(std::move(details));
      return *symbol;
    } else if constexpr (std::is_same_v<UnknownDetails, D>) {
      symbol->attrs() |= attrs;
      return *symbol;
    } else {
      SayAlreadyDeclared(name, *symbol);
      // replace the old symbol with a new one with correct details
      EraseSymbol(*symbol);
      return MakeSymbol(name, attrs, std::move(details));
    }
  }

  void MakeExternal(Symbol &);

protected:
  // Apply the implicit type rules to this symbol.
  void ApplyImplicitRules(Symbol &);
  const DeclTypeSpec *GetImplicitType(Symbol &);
  bool ConvertToObjectEntity(Symbol &);
  bool ConvertToProcEntity(Symbol &);

  const DeclTypeSpec &MakeNumericType(
      TypeCategory, const std::optional<parser::KindSelector> &);
  const DeclTypeSpec &MakeLogicalType(
      const std::optional<parser::KindSelector> &);

private:
  Scope *currScope_{nullptr};
};

class ModuleVisitor : public virtual ScopeHandler {
public:
  bool Pre(const parser::AccessStmt &);
  bool Pre(const parser::Only &);
  bool Pre(const parser::Rename::Names &);
  bool Pre(const parser::Rename::Operators &);
  bool Pre(const parser::UseStmt &);
  void Post(const parser::UseStmt &);

  void BeginModule(const parser::Name &, bool isSubmodule);
  bool BeginSubmodule(const parser::Name &, const parser::ParentIdentifier &);
  void ApplyDefaultAccess();

private:
  // The default access spec for this module.
  Attr defaultAccess_{Attr::PUBLIC};
  // The location of the last AccessStmt without access-ids, if any.
  const SourceName *prevAccessStmt_{nullptr};
  // The scope of the module during a UseStmt
  const Scope *useModuleScope_{nullptr};

  Symbol &SetAccess(const SourceName &, Attr);
  void AddUse(const parser::Rename::Names &);
  void AddUse(const parser::Rename::Operators &);
  Symbol *AddUse(const SourceName &);
  // A rename in a USE statement: local => use
  struct SymbolRename {
    Symbol *local{nullptr};
    Symbol *use{nullptr};
  };
  // Record a use from useModuleScope_ of use Name/Symbol as local Name/Symbol
  SymbolRename AddUse(const SourceName &localName, const SourceName &useName);
  void AddUse(const SourceName &, Symbol &localSymbol, const Symbol &useSymbol);
  Scope *FindModule(const parser::Name &, Scope *ancestor = nullptr);
};

class InterfaceVisitor : public virtual ScopeHandler {
public:
  bool Pre(const parser::InterfaceStmt &);
  void Post(const parser::EndInterfaceStmt &);
  bool Pre(const parser::GenericSpec &);
  bool Pre(const parser::ProcedureStmt &);
  bool Pre(const parser::GenericStmt &);
  void Post(const parser::GenericStmt &);

  bool inInterfaceBlock() const;
  bool isGeneric() const;
  bool isAbstract() const;

protected:
  GenericDetails &GetGenericDetails();
  // Add to generic the symbol for the subprogram with the same name
  void CheckGenericProcedures(Symbol &);
  void CheckSpecificsAreDistinguishable(const Symbol &, const SymbolVector &);

private:
  // A new GenericInfo is pushed for each interface block and generic stmt
  struct GenericInfo {
    GenericInfo(bool isInterface, bool isAbstract = false)
      : isInterface{isInterface}, isAbstract{isAbstract} {}
    bool isInterface;  // in interface block
    bool isAbstract;  // in abstract interface block
    Symbol *symbol{nullptr};  // the generic symbol being defined
  };
  std::stack<GenericInfo> genericInfo_;
  const GenericInfo &GetGenericInfo() const { return genericInfo_.top(); }
  void SetGenericSymbol(Symbol &symbol) { genericInfo_.top().symbol = &symbol; }

  using ProcedureKind = parser::ProcedureStmt::Kind;
  // mapping of generic to its specific proc names and kinds
  std::multimap<Symbol *, std::pair<const parser::Name *, ProcedureKind>>
      specificProcs_;

  void AddSpecificProcs(const std::list<parser::Name> &, ProcedureKind);
  void ResolveSpecificsInGeneric(Symbol &generic);
  void SayNotDistinguishable(const Symbol &, const Symbol &, const Symbol &);
};

class SubprogramVisitor : public virtual ScopeHandler, public InterfaceVisitor {
public:
  bool HandleStmtFunction(const parser::StmtFunctionStmt &);
  void Post(const parser::StmtFunctionStmt &);
  bool Pre(const parser::SubroutineStmt &);
  void Post(const parser::SubroutineStmt &);
  bool Pre(const parser::FunctionStmt &);
  void Post(const parser::FunctionStmt &);
  bool Pre(const parser::InterfaceBody::Subroutine &);
  void Post(const parser::InterfaceBody::Subroutine &);
  bool Pre(const parser::InterfaceBody::Function &);
  void Post(const parser::InterfaceBody::Function &);
  bool Pre(const parser::Suffix &);
  bool Pre(const parser::PrefixSpec &);
  void Post(const parser::ImplicitPart &);

  bool BeginSubprogram(
      const parser::Name &, Symbol::Flag, bool hasModulePrefix = false);
  void EndSubprogram();

protected:
  // Set when we see a stmt function that is really an array element assignment
  bool badStmtFuncFound_{false};

private:
  // Info about the current function: parse tree of the type in the PrefixSpec;
  // name and symbol of the function result from the Suffix; source location.
  struct {
    const parser::DeclarationTypeSpec *parsedType{nullptr};
    const parser::Name *resultName{nullptr};
    Symbol *resultSymbol{nullptr};
    const SourceName *source{nullptr};
  } funcInfo_;

  // Create a subprogram symbol in the current scope and push a new scope.
  Symbol &PushSubprogramScope(const parser::Name &, Symbol::Flag);
  Symbol *GetSpecificFromGeneric(const parser::Name &);
  SubprogramDetails &PostSubprogramStmt(const parser::Name &);
};

class DeclarationVisitor : public ArraySpecVisitor,
                           public virtual ScopeHandler {
public:
  using ArraySpecVisitor::Post;
  using ScopeHandler::Post;
  using ScopeHandler::Pre;

  bool Pre(const parser::Initialization &);
  void Post(const parser::EntityDecl &);
  void Post(const parser::ObjectDecl &);
  void Post(const parser::PointerDecl &);
  bool Pre(const parser::BindStmt &) { return BeginAttrs(); }
  void Post(const parser::BindStmt &) { EndAttrs(); }
  bool Pre(const parser::BindEntity &);
  bool Pre(const parser::NamedConstantDef &);
  bool Pre(const parser::NamedConstant &);
  bool Pre(const parser::AsynchronousStmt &);
  bool Pre(const parser::ContiguousStmt &);
  bool Pre(const parser::ExternalStmt &);
  bool Pre(const parser::IntentStmt &);
  bool Pre(const parser::IntrinsicStmt &);
  bool Pre(const parser::OptionalStmt &);
  bool Pre(const parser::ProtectedStmt &);
  bool Pre(const parser::ValueStmt &);
  bool Pre(const parser::VolatileStmt &);
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
  void Post(const parser::CodimensionDecl &);
  bool Pre(const parser::TypeDeclarationStmt &) { return BeginDecl(); }
  void Post(const parser::TypeDeclarationStmt &) { EndDecl(); }
  void Post(const parser::IntegerTypeSpec &);
  void Post(const parser::IntrinsicTypeSpec::Real &);
  void Post(const parser::IntrinsicTypeSpec::Complex &);
  void Post(const parser::IntrinsicTypeSpec::Logical &);
  void Post(const parser::IntrinsicTypeSpec::Character &);
  void Post(const parser::CharSelector::LengthAndKind &);
  void Post(const parser::CharLength &);
  void Post(const parser::LengthSelector &);
  bool Pre(const parser::KindParam &);
  bool Pre(const parser::DeclarationTypeSpec::Type &);
  bool Pre(const parser::DeclarationTypeSpec::Class &);
  bool Pre(const parser::DeclarationTypeSpec::Record &);
  void Post(const parser::DerivedTypeSpec &);
  bool Pre(const parser::DerivedTypeDef &);
  bool Pre(const parser::DerivedTypeStmt &x);
  void Post(const parser::DerivedTypeStmt &x);
  bool Pre(const parser::TypeParamDefStmt &x) { return BeginDecl(); }
  void Post(const parser::TypeParamDefStmt &);
  bool Pre(const parser::TypeAttrSpec::Extends &x);
  bool Pre(const parser::PrivateStmt &x);
  bool Pre(const parser::SequenceStmt &x);
  bool Pre(const parser::ComponentDefStmt &) { return BeginDecl(); }
  void Post(const parser::ComponentDefStmt &) { EndDecl(); }
  void Post(const parser::ComponentDecl &);
  bool Pre(const parser::ProcedureDeclarationStmt &);
  void Post(const parser::ProcedureDeclarationStmt &);
  bool Pre(const parser::ProcComponentDefStmt &);
  void Post(const parser::ProcComponentDefStmt &);
  bool Pre(const parser::ProcPointerInit &);
  void Post(const parser::ProcInterface &);
  void Post(const parser::ProcDecl &);
  bool Pre(const parser::TypeBoundProcedurePart &);
  void Post(const parser::TypeBoundProcedurePart &);
  void Post(const parser::ContainsStmt &);
  bool Pre(const parser::TypeBoundProcBinding &) { return BeginAttrs(); }
  void Post(const parser::TypeBoundProcBinding &) { EndAttrs(); }
  void Post(const parser::TypeBoundProcedureStmt::WithoutInterface &);
  void Post(const parser::TypeBoundProcedureStmt::WithInterface &);
  void Post(const parser::FinalProcedureStmt &);
  bool Pre(const parser::TypeBoundGenericStmt &);
  bool Pre(const parser::AllocateStmt &);
  void Post(const parser::AllocateStmt &);
  bool Pre(const parser::StructureConstructor &);
  bool Pre(const parser::NamelistStmt::Group &);
  bool Pre(const parser::IoControlSpec &);
  bool Pre(const parser::CommonStmt::Block &);
  void Post(const parser::CommonStmt::Block &);
  bool Pre(const parser::CommonBlockObject &);
  void Post(const parser::CommonBlockObject &);
  bool Pre(const parser::EquivalenceStmt &);
  bool Pre(const parser::SaveStmt &);

protected:
  bool BeginDecl();
  void EndDecl();
  Symbol &DeclareObjectEntity(const parser::Name &, Attrs);
  // Make sure that there's an entity in an enclosing scope called Name
  Symbol &FindOrDeclareEnclosingEntity(const parser::Name &);
  // Declare a LOCAL/LOCAL_INIT entity. If there isn't a type specified
  // it comes from the entity in the containing scope, or implicit rules.
  // Return pointer to the new symbol, or nullptr on error.
  Symbol *DeclareLocalEntity(const parser::Name &);
  // Declare a statement entity (e.g., an implied DO loop index).
  // If there isn't a type specified, implicit rules apply.
  // Return pointer to the new symbol, or nullptr on error.
  Symbol *DeclareStatementEntity(
      const parser::Name &, const std::optional<parser::IntegerTypeSpec> &);
  bool CheckUseError(const parser::Name &);
  void CheckAccessibility(const SourceName &, bool, Symbol &);
  bool CheckAccessibleComponent(const SourceName &, const Symbol &);
  void CheckCommonBlocks();
  void CheckSaveStmts();
  void CheckEquivalenceSets();
  bool CheckNotInBlock(const char *);
  bool NameIsKnownOrIntrinsic(const parser::Name &);

  // Each of these returns a pointer to a resolved Name (i.e. with symbol)
  // or nullptr in case of error.
  const parser::Name *ResolveStructureComponent(
      const parser::StructureComponent &);
  const parser::Name *ResolveDesignator(const parser::Designator &);
  const parser::Name *ResolveDataRef(const parser::DataRef &);
  const parser::Name *ResolveVariable(const parser::Variable &);
  const parser::Name *ResolveName(const parser::Name &);
  bool PassesSharedLocalityChecks(const parser::Name &name, Symbol &symbol);
  Symbol *NoteInterfaceName(const parser::Name &);
  void CheckExplicitInterface(Symbol &);

private:
  // The attribute corresponding to the statement containing an ObjectDecl
  std::optional<Attr> objectDeclAttr_;
  // Info about current character type while walking DeclTypeSpec.
  // Also captures any "*length" specifier on an individual declaration.
  struct {
    std::optional<ParamValue> length;
    std::optional<KindExpr> kind;
  } charInfo_;
  // Info about current derived type while walking DerivedTypeDef
  struct {
    const parser::Name *extends{nullptr};  // EXTENDS(name)
    bool privateComps{false};  // components are private by default
    bool privateBindings{false};  // bindings are private by default
    bool sawContains{false};  // currently processing bindings
    bool sequence{false};  // is a sequence type
    const Symbol *type{nullptr};  // derived type being defined
  } derivedTypeInfo_;
  // Collect equivalence sets and process at end of specification part
  std::vector<const std::list<parser::EquivalenceObject> *> equivalenceSets_;
  // Info about common blocks in the current scope
  struct {
    Symbol *curr{nullptr};  // common block currently being processed
    std::set<SourceName> names;  // names in any common block of scope
  } commonBlockInfo_;
  // Info about about SAVE statements and attributes in current scope
  struct {
    const SourceName *saveAll{nullptr};  // "SAVE" without entity list
    std::set<SourceName> entities;  // names of entities with save attr
    std::set<SourceName> commons;  // names of common blocks with save attr
  } saveInfo_;
  // In a ProcedureDeclarationStmt or ProcComponentDefStmt, this is
  // the interface name, if any.
  const parser::Name *interfaceName_{nullptr};
  // Map type-bound generic to binding names of its specific bindings
  std::multimap<Symbol *, const parser::Name *> genericBindings_;

  bool HandleAttributeStmt(Attr, const std::list<parser::Name> &);
  Symbol &HandleAttributeStmt(Attr, const parser::Name &);
  Symbol &DeclareUnknownEntity(const parser::Name &, Attrs);
  Symbol &DeclareProcEntity(const parser::Name &, Attrs, const ProcInterface &);
  void SetType(const parser::Name &, const DeclTypeSpec &);
  const Symbol *ResolveDerivedType(const parser::Name &);
  bool CanBeTypeBoundProc(const Symbol &);
  Symbol *MakeTypeSymbol(const SourceName &, Details &&);
  Symbol *MakeTypeSymbol(const parser::Name &, Details &&);
  bool OkToAddComponent(const parser::Name &, const Symbol * = nullptr);
  ParamValue GetParamValue(const parser::TypeParamValue &);
  Symbol &MakeCommonBlockSymbol(const parser::Name &);
  void CheckCommonBlockDerivedType(const SourceName &, const Symbol &);
  std::optional<MessageFixedText> CheckSaveAttr(const Symbol &);
  Attrs HandleSaveName(const SourceName &, Attrs);
  void AddSaveName(std::set<SourceName> &, const SourceName &);
  void SetSaveAttr(Symbol &);
  bool HandleUnrestrictedSpecificIntrinsicFunction(const parser::Name &);
  const parser::Name *FindComponent(const parser::Name *, const parser::Name &);
  void CheckInitialDataTarget(const Symbol &, const SomeExpr &, SourceName);
  void Initialization(const parser::Name &, const parser::Initialization &,
      bool inComponentDecl);
  bool PassesLocalityChecks(const parser::Name &name, Symbol &symbol);

  // Declare an object or procedure entity.
  // T is one of: EntityDetails, ObjectEntityDetails, ProcEntityDetails
  template<typename T>
  Symbol &DeclareEntity(const parser::Name &name, Attrs attrs) {
    Symbol &symbol{MakeSymbol(name, attrs)};
    if (symbol.has<T>()) {
      // OK
    } else if (symbol.has<UnknownDetails>()) {
      symbol.set_details(T{});
    } else if (auto *details{symbol.detailsIf<EntityDetails>()}) {
      symbol.set_details(T{std::move(*details)});
    } else if (std::is_same_v<EntityDetails, T> &&
        (symbol.has<ObjectEntityDetails>() ||
            symbol.has<ProcEntityDetails>())) {
      // OK
    } else if (auto *details{symbol.detailsIf<UseDetails>()}) {
      Say(name.source,
          "'%s' is use-associated from module '%s' and cannot be re-declared"_err_en_US,
          name.source, details->module().name());
    } else if (auto *details{symbol.detailsIf<SubprogramNameDetails>()}) {
      if (details->kind() == SubprogramKind::Module) {
        Say2(name,
            "Declaration of '%s' conflicts with its use as module procedure"_err_en_US,
            symbol, "Module procedure definition"_en_US);
      } else if (details->kind() == SubprogramKind::Internal) {
        Say2(name,
            "Declaration of '%s' conflicts with its use as internal procedure"_err_en_US,
            symbol, "Internal procedure definition"_en_US);
      } else {
        CHECK(!"unexpected kind");
      }
    } else if (std::is_same_v<ObjectEntityDetails, T> &&
        symbol.has<ProcEntityDetails>()) {
      SayWithDecl(
          name, symbol, "'%s' is already declared as a procedure"_err_en_US);
    } else if (std::is_same_v<ProcEntityDetails, T> &&
        symbol.has<ObjectEntityDetails>()) {
      SayWithDecl(
          name, symbol, "'%s' is already declared as an object"_err_en_US);
    } else {
      SayAlreadyDeclared(name, symbol);
    }
    return symbol;
  }
};

// Resolve construct entities and statement entities.
// Check that construct names don't conflict with other names.
class ConstructVisitor : public DeclarationVisitor {
public:
  bool Pre(const parser::ConcurrentHeader &);
  bool Pre(const parser::LocalitySpec::Local &);
  bool Pre(const parser::LocalitySpec::LocalInit &);
  bool Pre(const parser::LocalitySpec::Shared &);
  bool Pre(const parser::AcSpec &);
  bool Pre(const parser::AcImpliedDo &);
  bool Pre(const parser::DataImpliedDo &);
  bool Pre(const parser::DataStmtObject &);
  bool Pre(const parser::DoConstruct &);
  void Post(const parser::DoConstruct &);
  bool Pre(const parser::ForallConstruct &);
  void Post(const parser::ForallConstruct &);
  bool Pre(const parser::ForallStmt &);
  void Post(const parser::ForallStmt &);
  bool Pre(const parser::BlockStmt &);
  bool Pre(const parser::EndBlockStmt &);
  void Post(const parser::Selector &);
  bool Pre(const parser::AssociateStmt &);
  void Post(const parser::EndAssociateStmt &);
  void Post(const parser::Association &);
  void Post(const parser::SelectTypeStmt &);
  bool Pre(const parser::SelectTypeConstruct &);
  void Post(const parser::SelectTypeConstruct &);
  bool Pre(const parser::SelectTypeConstruct::TypeCase &);
  void Post(const parser::SelectTypeConstruct::TypeCase &);
  void Post(const parser::TypeGuardStmt::Guard &);
  bool Pre(const parser::ChangeTeamStmt &);
  void Post(const parser::EndChangeTeamStmt &);
  void Post(const parser::CoarrayAssociation &);

  // Definitions of construct names
  bool Pre(const parser::WhereConstructStmt &x) { return CheckDef(x.t); }
  bool Pre(const parser::ForallConstructStmt &x) { return CheckDef(x.t); }
  bool Pre(const parser::CriticalStmt &x) { return CheckDef(x.t); }
  bool Pre(const parser::LabelDoStmt &x) {
    return false;  // error recovery
  }
  bool Pre(const parser::NonLabelDoStmt &x) { return CheckDef(x.t); }
  bool Pre(const parser::IfThenStmt &x) { return CheckDef(x.t); }
  bool Pre(const parser::SelectCaseStmt &x) { return CheckDef(x.t); }
  bool Pre(const parser::SelectRankConstruct &);
  void Post(const parser::SelectRankConstruct &);
  bool Pre(const parser::SelectRankStmt &x) {
    return CheckDef(std::get<0>(x.t));
  }
  bool Pre(const parser::SelectTypeStmt &x) {
    return CheckDef(std::get<0>(x.t));
  }

  // References to construct names
  void Post(const parser::MaskedElsewhereStmt &x) { CheckRef(x.t); }
  void Post(const parser::ElsewhereStmt &x) { CheckRef(x.v); }
  void Post(const parser::EndWhereStmt &x) { CheckRef(x.v); }
  void Post(const parser::EndForallStmt &x) { CheckRef(x.v); }
  void Post(const parser::EndCriticalStmt &x) { CheckRef(x.v); }
  void Post(const parser::EndDoStmt &x) { CheckRef(x.v); }
  void Post(const parser::ElseIfStmt &x) { CheckRef(x.t); }
  void Post(const parser::ElseStmt &x) { CheckRef(x.v); }
  void Post(const parser::EndIfStmt &x) { CheckRef(x.v); }
  void Post(const parser::CaseStmt &x) { CheckRef(x.t); }
  void Post(const parser::EndSelectStmt &x) { CheckRef(x.v); }
  void Post(const parser::SelectRankCaseStmt &x) { CheckRef(x.t); }
  void Post(const parser::TypeGuardStmt &x) { CheckRef(x.t); }
  void Post(const parser::CycleStmt &x) { CheckRef(x.v); }
  void Post(const parser::ExitStmt &x) { CheckRef(x.v); }

private:
  // R1105 selector -> expr | variable
  // expr is set in either case unless there were errors
  struct Selector {
    Selector() {}
    Selector(const parser::CharBlock &source, MaybeExpr &&expr)
      : source{source}, expr{std::move(expr)} {}
    operator bool() const { return expr.has_value(); }
    parser::CharBlock source;
    MaybeExpr expr;
  };
  // association -> [associate-name =>] selector
  struct Association {
    const parser::Name *name{nullptr};
    Selector selector;
  };
  std::vector<Association> associationStack_;

  template<typename T> bool CheckDef(const T &t) {
    return CheckDef(std::get<std::optional<parser::Name>>(t));
  }
  template<typename T> void CheckRef(const T &t) {
    CheckRef(std::get<std::optional<parser::Name>>(t));
  }
  bool CheckDef(const std::optional<parser::Name> &);
  void CheckRef(const std::optional<parser::Name> &);
  const DeclTypeSpec &ToDeclTypeSpec(evaluate::DynamicType &&);
  const DeclTypeSpec &ToDeclTypeSpec(
      evaluate::DynamicType &&, MaybeSubscriptIntExpr &&length);
  Symbol *MakeAssocEntity();
  void SetTypeFromAssociation(Symbol &);
  void SetAttrsFromAssociation(Symbol &);
  Selector ResolveSelector(const parser::Selector &);
  void ResolveControlExpressions(const parser::ConcurrentControl &control);
  void ResolveIndexName(const parser::ConcurrentControl &control);
  Association &GetCurrentAssociation();
  void PushAssociation();
  void PopAssociation();
};

// Walk the parse tree and resolve names to symbols.
class ResolveNamesVisitor : public virtual ScopeHandler,
                            public ModuleVisitor,
                            public SubprogramVisitor,
                            public ConstructVisitor {
public:
  using ArraySpecVisitor::Post;
  using ConstructVisitor::Post;
  using ConstructVisitor::Pre;
  using DeclarationVisitor::Post;
  using DeclarationVisitor::Pre;
  using ImplicitRulesVisitor::Post;
  using ImplicitRulesVisitor::Pre;
  using InterfaceVisitor::Post;
  using InterfaceVisitor::Pre;
  using ModuleVisitor::Post;
  using ModuleVisitor::Pre;
  using ScopeHandler::Post;
  using ScopeHandler::Pre;
  using SubprogramVisitor::Post;
  using SubprogramVisitor::Pre;

  ResolveNamesVisitor(SemanticsContext &context) {
    set_context(context);
    set_this(this);
    PushScope(context.globalScope());
  }

  // Default action for a parse tree node is to visit children.
  template<typename T> bool Pre(const T &) { return true; }
  template<typename T> void Post(const T &) {}

  void Post(const parser::SpecificationPart &);
  void Post(const parser::Program &);
  bool Pre(const parser::ImplicitStmt &);
  void Post(const parser::PointerObject &);
  void Post(const parser::AllocateObject &);
  bool Pre(const parser::PointerAssignmentStmt &);
  void Post(const parser::Designator &);
  template<typename A, typename B>
  void Post(const parser::LoopBounds<A, B> &x) {
    ResolveName(*parser::Unwrap<parser::Name>(x.name));
  }
  void Post(const parser::ProcComponentRef &);
  bool Pre(const parser::FunctionReference &);
  bool Pre(const parser::CallStmt &);
  bool Pre(const parser::ImportStmt &);
  void Post(const parser::TypeGuardStmt &);
  bool Pre(const parser::StmtFunctionStmt &);
  bool Pre(const parser::DefinedOpName &);
  bool Pre(const parser::ProgramUnit &);

  // These nodes should never be reached: they are handled in ProgramUnit
  bool Pre(const parser::MainProgram &) { DIE("unreachable"); }
  bool Pre(const parser::FunctionSubprogram &) { DIE("unreachable"); }
  bool Pre(const parser::SubroutineSubprogram &) { DIE("unreachable"); }
  bool Pre(const parser::SeparateModuleSubprogram &) { DIE("unreachable"); }
  bool Pre(const parser::Module &) { DIE("unreachable"); }
  bool Pre(const parser::Submodule &) { DIE("unreachable"); }
  bool Pre(const parser::BlockData &) { DIE("unreachable"); }

  void NoteExecutablePartCall(Symbol::Flag, const parser::Call &);

private:
  // Kind of procedure we are expecting to see in a ProcedureDesignator
  std::optional<Symbol::Flag> expectedProcFlag_;
  const SourceName *prevImportStmt_{nullptr};

  void CheckImports();
  void CheckImport(const SourceName &, const SourceName &);
  void HandleCall(Symbol::Flag, const parser::Call &);
  void HandleProcedureName(Symbol::Flag, const parser::Name &);
  bool SetProcFlag(const parser::Name &, Symbol &, Symbol::Flag);
  void ResolveSpecificationParts(ProgramTree &);
  void AddSubpNames(const ProgramTree &);
  bool BeginScope(const ProgramTree &);
  void FinishSpecificationParts(const ProgramTree &);
  void FinishDerivedType(Scope &);
  void SetPassArg(const Symbol &, const Symbol *, WithPassArg &);
  void ResolveExecutionParts(const ProgramTree &);
};

// ImplicitRules implementation

bool ImplicitRules::isImplicitNoneType() const {
  if (isImplicitNoneType_.has_value()) {
    return isImplicitNoneType_.value();
  } else if (inheritFromParent_) {
    return parent_->isImplicitNoneType();
  } else {
    return false;  // default if not specified
  }
}

bool ImplicitRules::isImplicitNoneExternal() const {
  if (isImplicitNoneExternal_.has_value()) {
    return isImplicitNoneExternal_.value();
  } else if (inheritFromParent_) {
    return parent_->isImplicitNoneExternal();
  } else {
    return false;  // default if not specified
  }
}

const DeclTypeSpec *ImplicitRules::GetType(char ch) const {
  if (auto it{map_.find(ch)}; it != map_.end()) {
    return it->second;
  } else if (inheritFromParent_) {
    return parent_->GetType(ch);
  } else if (ch >= 'i' && ch <= 'n') {
    return &context_.MakeNumericType(TypeCategory::Integer);
  } else if (ch >= 'a' && ch <= 'z') {
    return &context_.MakeNumericType(TypeCategory::Real);
  } else {
    return nullptr;
  }
}

// isDefault is set when we are applying the default rules, so it is not
// an error if the type is already set.
void ImplicitRules::SetType(const DeclTypeSpec &type, parser::Location lo,
    parser::Location hi, bool isDefault) {
  for (char ch = *lo; ch; ch = ImplicitRules::Incr(ch)) {
    auto res{map_.emplace(ch, &type)};
    if (!res.second && !isDefault) {
      context_.Say(parser::CharBlock{lo},
          "More than one implicit type specified for '%c'"_err_en_US, ch);
    }
    if (ch == *hi) {
      break;
    }
  }
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
  auto it{implicitRules.map_.find(ch)};
  if (it != implicitRules.map_.end()) {
    o << "  " << ch << ": " << *it->second << '\n';
  }
}

template<typename T> void BaseVisitor::Walk(const T &x) {
  parser::Walk(x, *this_);
}

void BaseVisitor::set_context(SemanticsContext &context) {
  context_ = &context;
  messageHandler_.set_context(context);
}

void BaseVisitor::MakePlaceholder(
    const parser::Name &name, MiscDetails::Kind kind) {
  if (!name.symbol) {
    name.symbol = &context_->globalScope().MakeSymbol(
        name.source, Attrs{}, MiscDetails{kind});
  }
}

// AttrsVisitor implementation

bool AttrsVisitor::BeginAttrs() {
  CHECK(!attrs_);
  attrs_ = std::make_optional<Attrs>();
  return true;
}
Attrs AttrsVisitor::GetAttrs() {
  CHECK(attrs_);
  return *attrs_;
}
Attrs AttrsVisitor::EndAttrs() {
  Attrs result{GetAttrs()};
  attrs_.reset();
  passName_ = nullptr;
  bindName_.reset();
  return result;
}

bool AttrsVisitor::SetPassNameOn(Symbol &symbol) {
  if (!passName_) {
    return false;
  }
  std::visit(
      common::visitors{
          [&](ProcEntityDetails &x) { x.set_passName(*passName_); },
          [&](ProcBindingDetails &x) { x.set_passName(*passName_); },
          [](auto &) { common::die("unexpected pass name"); },
      },
      symbol.details());
  return true;
}

bool AttrsVisitor::SetBindNameOn(Symbol &symbol) {
  if (!bindName_) {
    return false;
  }
  std::visit(
      common::visitors{
          [&](EntityDetails &x) { x.set_bindName(std::move(bindName_)); },
          [&](ObjectEntityDetails &x) { x.set_bindName(std::move(bindName_)); },
          [&](ProcEntityDetails &x) { x.set_bindName(std::move(bindName_)); },
          [&](SubprogramDetails &x) { x.set_bindName(std::move(bindName_)); },
          [&](CommonBlockDetails &x) { x.set_bindName(std::move(bindName_)); },
          [](auto &) { common::die("unexpected bind name"); },
      },
      symbol.details());
  return true;
}

void AttrsVisitor::Post(const parser::LanguageBindingSpec &x) {
  CHECK(attrs_);
  attrs_->set(Attr::BIND_C);
  if (x.v) {
    bindName_ = EvaluateExpr(*x.v);
  }
}
bool AttrsVisitor::Pre(const parser::AccessSpec &x) {
  attrs_->set(AccessSpecToAttr(x));
  return false;
}
bool AttrsVisitor::Pre(const parser::IntentSpec &x) {
  CHECK(attrs_);
  attrs_->set(IntentSpecToAttr(x));
  return false;
}
bool AttrsVisitor::Pre(const parser::Pass &x) {
  if (x.v) {
    passName_ = &x.v->source;
    MakePlaceholder(*x.v, MiscDetails::Kind::PassName);
  } else {
    attrs_->set(Attr::PASS);
  }
  return false;
}

// DeclTypeSpecVisitor implementation

const DeclTypeSpec *DeclTypeSpecVisitor::GetDeclTypeSpec() {
  return state_.declTypeSpec;
}

void DeclTypeSpecVisitor::BeginDeclTypeSpec() {
  CHECK(!state_.expectDeclTypeSpec);
  CHECK(!state_.declTypeSpec);
  state_.expectDeclTypeSpec = true;
}
void DeclTypeSpecVisitor::EndDeclTypeSpec() {
  CHECK(state_.expectDeclTypeSpec);
  state_ = {};
}

void DeclTypeSpecVisitor::SetDeclTypeSpecCategory(
    DeclTypeSpec::Category category) {
  CHECK(state_.expectDeclTypeSpec);
  state_.derived.category = category;
}

bool DeclTypeSpecVisitor::Pre(const parser::TypeGuardStmt &) {
  BeginDeclTypeSpec();
  return true;
}
void DeclTypeSpecVisitor::Post(const parser::TypeGuardStmt &) {
  EndDeclTypeSpec();
}

void DeclTypeSpecVisitor::Post(const parser::TypeSpec &typeSpec) {
  // Record the resolved DeclTypeSpec in the parse tree for use by
  // expression semantics if the DeclTypeSpec is a valid TypeSpec.
  // The grammar ensures that it's an intrinsic or derived type spec,
  // not TYPE(*) or CLASS(*) or CLASS(T).
  if (const DeclTypeSpec * spec{state_.declTypeSpec}) {
    switch (spec->category()) {
    case DeclTypeSpec::Numeric:
    case DeclTypeSpec::Logical:
    case DeclTypeSpec::Character: typeSpec.declTypeSpec = spec; break;
    case DeclTypeSpec::TypeDerived:
      if (const DerivedTypeSpec * derived{spec->AsDerived()}) {
        if (derived->typeSymbol().attrs().test(Attr::ABSTRACT)) {
          Say("ABSTRACT derived type may not be used here"_err_en_US);
        }
        typeSpec.declTypeSpec = spec;
      }
      break;
    default: CRASH_NO_CASE;
    }
  }
}

void DeclTypeSpecVisitor::Post(
    const parser::IntrinsicTypeSpec::DoublePrecision &) {
  MakeNumericType(TypeCategory::Real, context().doublePrecisionKind());
}
void DeclTypeSpecVisitor::Post(
    const parser::IntrinsicTypeSpec::DoubleComplex &) {
  MakeNumericType(TypeCategory::Complex, context().doublePrecisionKind());
}
void DeclTypeSpecVisitor::MakeNumericType(TypeCategory category, int kind) {
  SetDeclTypeSpec(context().MakeNumericType(category, kind));
}

void DeclTypeSpecVisitor::Post(const parser::DeclarationTypeSpec::ClassStar &) {
  SetDeclTypeSpec(context().globalScope().MakeClassStarType());
}
void DeclTypeSpecVisitor::Post(const parser::DeclarationTypeSpec::TypeStar &) {
  SetDeclTypeSpec(context().globalScope().MakeTypeStarType());
}

// Check that we're expecting to see a DeclTypeSpec (and haven't seen one yet)
// and save it in state_.declTypeSpec.
void DeclTypeSpecVisitor::SetDeclTypeSpec(const DeclTypeSpec &declTypeSpec) {
  CHECK(state_.expectDeclTypeSpec);
  CHECK(!state_.declTypeSpec);
  state_.declTypeSpec = &declTypeSpec;
}

KindExpr DeclTypeSpecVisitor::GetKindParamExpr(
    TypeCategory category, const std::optional<parser::KindSelector> &kind) {
  return AnalyzeKindSelector(context(), category, kind);
}

// MessageHandler implementation

Message &MessageHandler::Say(MessageFixedText &&msg) {
  return context_->Say(*currStmtSource(), std::move(msg));
}
Message &MessageHandler::Say(MessageFormattedText &&msg) {
  return context_->Say(*currStmtSource(), std::move(msg));
}
Message &MessageHandler::Say(const SourceName &name, MessageFixedText &&msg) {
  return Say(name, std::move(msg), name);
}

// ImplicitRulesVisitor implementation

void ImplicitRulesVisitor::Post(const parser::ParameterStmt &x) {
  prevParameterStmt_ = currStmtSource();
}

bool ImplicitRulesVisitor::Pre(const parser::ImplicitStmt &x) {
  bool result{std::visit(
      common::visitors{
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
      x.u)};
  prevImplicit_ = currStmtSource();
  return result;
}

bool ImplicitRulesVisitor::Pre(const parser::LetterSpec &x) {
  auto loLoc{std::get<parser::Location>(x.t)};
  auto hiLoc{loLoc};
  if (auto hiLocOpt{std::get<std::optional<parser::Location>>(x.t)}) {
    hiLoc = *hiLocOpt;
    if (*hiLoc < *loLoc) {
      Say(hiLoc, "'%s' does not follow '%s' alphabetically"_err_en_US,
          std::string(hiLoc, 1), std::string(loLoc, 1));
      return false;
    }
  }
  implicitRules().SetType(*GetDeclTypeSpec(), loLoc, hiLoc);
  return false;
}

bool ImplicitRulesVisitor::Pre(const parser::ImplicitSpec &) {
  BeginDeclTypeSpec();
  return true;
}

void ImplicitRulesVisitor::Post(const parser::ImplicitSpec &) {
  EndDeclTypeSpec();
}

void ImplicitRulesVisitor::SetScope(const Scope &scope) {
  implicitRules_ = &implicitRulesMap_.at(&scope);
  prevImplicit_ = nullptr;
  prevImplicitNone_ = nullptr;
  prevImplicitNoneType_ = nullptr;
  prevParameterStmt_ = nullptr;
}
void ImplicitRulesVisitor::BeginScope(const Scope &scope) {
  // find or create implicit rules for this scope
  implicitRulesMap_.try_emplace(&scope, context(), implicitRules_);
  SetScope(scope);
}

// TODO: for all of these errors, reference previous statement too
bool ImplicitRulesVisitor::HandleImplicitNone(
    const std::list<ImplicitNoneNameSpec> &nameSpecs) {
  if (prevImplicitNone_ != nullptr) {
    Say("More than one IMPLICIT NONE statement"_err_en_US);
    Say(*prevImplicitNone_, "Previous IMPLICIT NONE statement"_en_US);
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

// ArraySpecVisitor implementation

void ArraySpecVisitor::Post(const parser::ArraySpec &x) {
  CHECK(arraySpec_.empty());
  arraySpec_ = AnalyzeArraySpec(context(), x);
}
void ArraySpecVisitor::Post(const parser::ComponentArraySpec &x) {
  CHECK(arraySpec_.empty());
  arraySpec_ = AnalyzeArraySpec(context(), x);
}
void ArraySpecVisitor::Post(const parser::CoarraySpec &x) {
  CHECK(coarraySpec_.empty());
  coarraySpec_ = AnalyzeCoarraySpec(context(), x);
}

const ArraySpec &ArraySpecVisitor::arraySpec() {
  return !arraySpec_.empty() ? arraySpec_ : attrArraySpec_;
}
const ArraySpec &ArraySpecVisitor::coarraySpec() {
  return !coarraySpec_.empty() ? coarraySpec_ : attrCoarraySpec_;
}
void ArraySpecVisitor::BeginArraySpec() {
  CHECK(arraySpec_.empty());
  CHECK(coarraySpec_.empty());
  CHECK(attrArraySpec_.empty());
  CHECK(attrCoarraySpec_.empty());
}
void ArraySpecVisitor::EndArraySpec() {
  CHECK(arraySpec_.empty());
  CHECK(coarraySpec_.empty());
  attrArraySpec_.clear();
  attrCoarraySpec_.clear();
}
void ArraySpecVisitor::PostAttrSpec() {
  // Save dimension/codimension from attrs so we can process array/coarray-spec
  // on the entity-decl
  if (!arraySpec_.empty()) {
    CHECK(attrArraySpec_.empty());
    attrArraySpec_.splice(attrArraySpec_.cbegin(), arraySpec_);
  }
  if (!coarraySpec_.empty()) {
    CHECK(attrCoarraySpec_.empty());
    attrCoarraySpec_.splice(attrCoarraySpec_.cbegin(), coarraySpec_);
  }
}

// ScopeHandler implementation

void ScopeHandler::SayAlreadyDeclared(const parser::Name &name, Symbol &prev) {
  SayAlreadyDeclared(name.source, prev);
}
void ScopeHandler::SayAlreadyDeclared(const SourceName &name, Symbol &prev) {
  auto &msg{
      Say(name, "'%s' is already declared in this scoping unit"_err_en_US)};
  if (const auto *details{prev.detailsIf<UseDetails>()}) {
    msg.Attach(details->location(),
        "It is use-associated with '%s' in module '%s'"_err_en_US,
        details->symbol().name(), details->module().name());
  } else {
    msg.Attach(prev.name(), "Previous declaration of '%s'"_en_US, prev.name());
  }
  context().SetError(prev);
}

void ScopeHandler::SayWithReason(const parser::Name &name, Symbol &symbol,
    MessageFixedText &&msg1, MessageFixedText &&msg2) {
  Say2(name, std::move(msg1), symbol, std::move(msg2));
  context().SetError(symbol, msg1.isFatal());
}

void ScopeHandler::SayWithDecl(
    const parser::Name &name, Symbol &symbol, MessageFixedText &&msg) {
  SayWithReason(name, symbol, std::move(msg),
      symbol.test(Symbol::Flag::Implicit) ? "Implicit declaration of '%s'"_en_US
                                          : "Declaration of '%s'"_en_US);
}

void ScopeHandler::SayLocalMustBeVariable(
    const parser::Name &name, Symbol &symbol) {
  SayWithDecl(name, symbol,
      "The name '%s' must be a variable to appear"
      " in a locality-spec"_err_en_US);
}

void ScopeHandler::SayDerivedType(
    const SourceName &name, MessageFixedText &&msg, const Scope &type) {
  const Symbol *typeSymbol{type.GetSymbol()};
  CHECK(typeSymbol != nullptr);
  Say(name, std::move(msg), name, typeSymbol->name())
      .Attach(typeSymbol->name(), "Declaration of derived type '%s'"_en_US,
          typeSymbol->name());
}
void ScopeHandler::Say2(const SourceName &name1, MessageFixedText &&msg1,
    const SourceName &name2, MessageFixedText &&msg2) {
  Say(name1, std::move(msg1)).Attach(name2, std::move(msg2), name2);
}
void ScopeHandler::Say2(const SourceName &name, MessageFixedText &&msg1,
    Symbol &symbol, MessageFixedText &&msg2) {
  Say2(name, std::move(msg1), symbol.name(), std::move(msg2));
  context().SetError(symbol, msg1.isFatal());
}
void ScopeHandler::Say2(const parser::Name &name, MessageFixedText &&msg1,
    Symbol &symbol, MessageFixedText &&msg2) {
  Say2(name.source, std::move(msg1), symbol.name(), std::move(msg2));
  context().SetError(symbol, msg1.isFatal());
}

Scope &ScopeHandler::InclusiveScope() {
  for (auto *scope{&currScope()};; scope = &scope->parent()) {
    if (scope->kind() != Scope::Kind::Block && !scope->IsDerivedType()) {
      return *scope;
    }
  }
  common::die("inclusive scope not found");
}
Scope &ScopeHandler::GlobalScope() {
  for (auto *scope = currScope_; scope; scope = &scope->parent()) {
    if (scope->IsGlobal()) {
      return *scope;
    }
  }
  common::die("global scope not found");
}
void ScopeHandler::PushScope(Scope::Kind kind, Symbol *symbol) {
  PushScope(currScope().MakeScope(kind, symbol));
}
void ScopeHandler::PushScope(Scope &scope) {
  currScope_ = &scope;
  auto kind{currScope_->kind()};
  if (kind != Scope::Kind::Block) {
    ImplicitRulesVisitor::BeginScope(scope);
  }
  if (!currScope_->IsDerivedType()) {
    if (auto *symbol{scope.symbol()}) {
      // Create a dummy symbol so we can't create another one with the same
      // name. It might already be there if we previously pushed the scope.
      if (!FindInScope(scope, symbol->name())) {
        auto &newSymbol{MakeSymbol(symbol->name())};
        if (kind == Scope::Kind::Subprogram) {
          // Allow for recursive references.  If this symbol is a function
          // without an explicit RESULT(), this new symbol will be discarded
          // and replaced with an object of the same name.
          newSymbol.set_details(HostAssocDetails{*symbol});
        } else {
          newSymbol.set_details(MiscDetails{MiscDetails::Kind::ScopeName});
        }
      }
    }
  }
}
void ScopeHandler::PopScope() {
  // Entities that are not yet classified as objects or procedures are now
  // assumed to be objects.
  // TODO: Statement functions
  for (auto &pair : currScope()) {
    ConvertToObjectEntity(*pair.second);
  }
  SetScope(currScope_->parent());
}
void ScopeHandler::SetScope(Scope &scope) {
  currScope_ = &scope;
  ImplicitRulesVisitor::SetScope(InclusiveScope());
}

Symbol *ScopeHandler::FindSymbol(const parser::Name &name) {
  return FindSymbol(currScope(), name);
}
Symbol *ScopeHandler::FindSymbol(const Scope &scope, const parser::Name &name) {
  // Scope::FindSymbol() skips over innermost derived type scopes.
  // Ensure that "bare" type parameter names are not overlooked.
  if (Symbol * symbol{FindInTypeOrParents(scope, name.source)}) {
    if (symbol->has<TypeParamDetails>()) {
      return Resolve(name, symbol);
    }
  }
  return Resolve(name, scope.FindSymbol(name.source));
}

Symbol &ScopeHandler::MakeSymbol(
    Scope &scope, const SourceName &name, Attrs attrs) {
  auto *symbol{FindInScope(scope, name)};
  if (symbol) {
    symbol->attrs() |= attrs;
  } else {
    const auto pair{scope.try_emplace(name, attrs, UnknownDetails{})};
    CHECK(pair.second);  // name was not found, so must be able to add
    symbol = pair.first->second;
  }
  return *symbol;
}
Symbol &ScopeHandler::MakeSymbol(const SourceName &name, Attrs attrs) {
  return MakeSymbol(currScope(), name, attrs);
}
Symbol &ScopeHandler::MakeSymbol(const parser::Name &name, Attrs attrs) {
  return Resolve(name, MakeSymbol(name.source, attrs));
}
Symbol &ScopeHandler::CopySymbol(const SourceName &name, const Symbol &symbol) {
  CHECK(!FindInScope(currScope(), name));
  return MakeSymbol(currScope(), name, symbol.attrs());
}

// Look for name only in scope, not in enclosing scopes.
Symbol *ScopeHandler::FindInScope(
    const Scope &scope, const parser::Name &name) {
  return Resolve(name, FindInScope(scope, name.source));
}
Symbol *ScopeHandler::FindInScope(const Scope &scope, const SourceName &name) {
  if (auto it{scope.find(name)}; it != scope.end()) {
    return it->second;
  } else {
    return nullptr;
  }
}

// Find a component or type parameter by name in a derived type or its parents.
Symbol *ScopeHandler::FindInTypeOrParents(const Scope &scope, SourceName name) {
  if (scope.IsDerivedType()) {
    if (Symbol * symbol{FindInScope(scope, name)}) {
      return symbol;
    }
    if (const Scope * parent{scope.GetDerivedTypeParent()}) {
      return FindInTypeOrParents(*parent, name);
    }
  }
  return nullptr;
}
Symbol *ScopeHandler::FindInTypeOrParents(
    const Scope &scope, const parser::Name &name) {
  return Resolve(name, FindInTypeOrParents(scope, name.source));
}
Symbol *ScopeHandler::FindInTypeOrParents(const parser::Name &name) {
  return FindInTypeOrParents(currScope(), name);
}

void ScopeHandler::EraseSymbol(const parser::Name &name) {
  currScope().erase(name.source);
  name.symbol = nullptr;
}

static bool NeedsType(const Symbol &symbol) {
  return symbol.GetType() == nullptr &&
      std::visit(
          common::visitors{
              [](const EntityDetails &) { return true; },
              [](const ObjectEntityDetails &) { return true; },
              [](const AssocEntityDetails &) { return true; },
              [&](const ProcEntityDetails &p) {
                return symbol.test(Symbol::Flag::Function) &&
                    !symbol.attrs().test(Attr::INTRINSIC) &&
                    p.interface().type() == nullptr &&
                    p.interface().symbol() == nullptr;
              },
              [](const auto &) { return false; },
          },
          symbol.details());
}
void ScopeHandler::ApplyImplicitRules(Symbol &symbol) {
  if (NeedsType(symbol)) {
    if (isImplicitNoneType()) {
      if (symbol.has<ProcEntityDetails>() &&
          !symbol.attrs().test(Attr::EXTERNAL) &&
          context().intrinsics().IsIntrinsic(symbol.name().ToString())) {
        // type will be determined in expression semantics
        symbol.attrs().set(Attr::INTRINSIC);
      } else {
        Say(symbol.name(), "No explicit type declared for '%s'"_err_en_US);
      }
    } else if (const auto *type{GetImplicitType(symbol)}) {
      symbol.SetType(*type);
    }
  }
}
const DeclTypeSpec *ScopeHandler::GetImplicitType(Symbol &symbol) {
  auto &name{symbol.name()};
  const auto *type{implicitRules().GetType(name.begin()[0])};
  if (type) {
    symbol.set(Symbol::Flag::Implicit);
  } else {
    Say(name, "No explicit type declared for '%s'"_err_en_US);
  }
  return type;
}

// Convert symbol to be a ObjectEntity or return false if it can't be.
bool ScopeHandler::ConvertToObjectEntity(Symbol &symbol) {
  if (symbol.has<ObjectEntityDetails>()) {
    // nothing to do
  } else if (symbol.has<UnknownDetails>()) {
    symbol.set_details(ObjectEntityDetails{});
  } else if (auto *details{symbol.detailsIf<EntityDetails>()}) {
    symbol.set_details(ObjectEntityDetails{std::move(*details)});
  } else if (auto *useDetails{symbol.detailsIf<UseDetails>()}) {
    return useDetails->symbol().has<ObjectEntityDetails>();
  } else {
    return false;
  }
  return true;
}
// Convert symbol to be a ProcEntity or return false if it can't be.
bool ScopeHandler::ConvertToProcEntity(Symbol &symbol) {
  if (symbol.has<ProcEntityDetails>()) {
    // nothing to do
  } else if (symbol.has<UnknownDetails>()) {
    symbol.set_details(ProcEntityDetails{});
  } else if (auto *details{symbol.detailsIf<EntityDetails>()}) {
    symbol.set_details(ProcEntityDetails{std::move(*details)});
    if (symbol.GetType() && !symbol.test(Symbol::Flag::Implicit)) {
      CHECK(!symbol.test(Symbol::Flag::Subroutine));
      symbol.set(Symbol::Flag::Function);
    }
  } else {
    return false;
  }
  return true;
}

const DeclTypeSpec &ScopeHandler::MakeNumericType(
    TypeCategory category, const std::optional<parser::KindSelector> &kind) {
  KindExpr value{GetKindParamExpr(category, kind)};
  if (auto known{evaluate::ToInt64(value)}) {
    return context().MakeNumericType(category, static_cast<int>(*known));
  } else {
    return currScope_->MakeNumericType(category, std::move(value));
  }
}

const DeclTypeSpec &ScopeHandler::MakeLogicalType(
    const std::optional<parser::KindSelector> &kind) {
  KindExpr value{GetKindParamExpr(TypeCategory::Logical, kind)};
  if (auto known{evaluate::ToInt64(value)}) {
    return context().MakeLogicalType(static_cast<int>(*known));
  } else {
    return currScope_->MakeLogicalType(std::move(value));
  }
}

void ScopeHandler::MakeExternal(Symbol &symbol) {
  if (!symbol.attrs().test(Attr::EXTERNAL)) {
    symbol.attrs().set(Attr::EXTERNAL);
    if (symbol.attrs().test(Attr::INTRINSIC)) {  // C840
      Say(symbol.name(),
          "Symbol '%s' cannot have both EXTERNAL and INTRINSIC attributes"_err_en_US,
          symbol.name());
    }
  }
}

// ModuleVisitor implementation

bool ModuleVisitor::Pre(const parser::Only &x) {
  std::visit(
      common::visitors{
          [&](const Indirection<parser::GenericSpec> &generic) {
            auto info{GenericSpecInfo{generic.value()}};
            info.Resolve(AddUse(info.symbolName()));
          },
          [&](const parser::Name &name) { Resolve(name, AddUse(name.source)); },
          [&](const parser::Rename &rename) {
            std::visit(
                common::visitors{
                    [&](const parser::Rename::Names &names) { AddUse(names); },
                    [&](const parser::Rename::Operators &ops) { AddUse(ops); },
                },
                rename.u);
          },
      },
      x.u);
  return false;
}

bool ModuleVisitor::Pre(const parser::Rename::Names &x) {
  AddUse(x);
  return false;
}
bool ModuleVisitor::Pre(const parser::Rename::Operators &x) {
  AddUse(x);
  return false;
}

// Set useModuleScope_ to the Scope of the module being used.
bool ModuleVisitor::Pre(const parser::UseStmt &x) {
  useModuleScope_ = FindModule(x.moduleName);
  return useModuleScope_ != nullptr;
}
void ModuleVisitor::Post(const parser::UseStmt &x) {
  if (const auto *list{std::get_if<std::list<parser::Rename>>(&x.u)}) {
    // Not a use-only: collect the names that were used in renames,
    // then add a use for each public name that was not renamed.
    std::set<SourceName> useNames;
    for (const auto &rename : *list) {
      std::visit(
          common::visitors{
              [&](const parser::Rename::Names &names) {
                useNames.insert(std::get<1>(names.t).source);
              },
              [&](const parser::Rename::Operators &ops) {
                useNames.insert(std::get<1>(ops.t).v.source);
              },
          },
          rename.u);
    }
    for (const auto &[name, symbol] : *useModuleScope_) {
      if (symbol->attrs().test(Attr::PUBLIC) &&
          !symbol->detailsIf<MiscDetails>()) {
        if (useNames.count(name) == 0) {
          auto *localSymbol{FindInScope(currScope(), name)};
          if (!localSymbol) {
            localSymbol = &CopySymbol(name, *symbol);
          }
          AddUse(x.moduleName.source, *localSymbol, *symbol);
        }
      }
    }
  }
  useModuleScope_ = nullptr;
}

void ModuleVisitor::AddUse(const parser::Rename::Names &names) {
  const auto &localName{std::get<0>(names.t)};
  const auto &useName{std::get<1>(names.t)};
  SymbolRename rename{AddUse(localName.source, useName.source)};
  Resolve(useName, rename.use);
  Resolve(localName, rename.local);
}

void ModuleVisitor::AddUse(const parser::Rename::Operators &ops) {
  const parser::DefinedOpName &local{std::get<0>(ops.t)};
  const parser::DefinedOpName &use{std::get<1>(ops.t)};
  GenericSpecInfo localInfo{local};
  GenericSpecInfo useInfo{use};
  if (IsIntrinsicOperator(context(), local.v.source)) {
    Say(local.v,
        "Intrinsic operator '%s' may not be used as a defined operator"_err_en_US);
  } else if (IsLogicalConstant(context(), local.v.source)) {
    Say(local.v,
        "Logical constant '%s' may not be used as a defined operator"_err_en_US);
  } else {
    SymbolRename rename{AddUse(localInfo.symbolName(), useInfo.symbolName())};
    useInfo.Resolve(rename.use);
    localInfo.Resolve(rename.local);
  }
}

Symbol *ModuleVisitor::AddUse(const SourceName &useName) {
  return AddUse(useName, useName).use;
}

ModuleVisitor::SymbolRename ModuleVisitor::AddUse(
    const SourceName &localName, const SourceName &useName) {
  if (!useModuleScope_) {
    return {};  // error occurred finding module
  }
  auto *useSymbol{FindInScope(*useModuleScope_, useName)};
  if (!useSymbol) {
    Say(useName,
        IsDefinedOperator(useName)
            ? "Operator '%s' not found in module '%s'"_err_en_US
            : "'%s' not found in module '%s'"_err_en_US,
        useName, useModuleScope_->name());
    return {};
  }
  if (useSymbol->attrs().test(Attr::PRIVATE)) {
    Say(useName,
        IsDefinedOperator(useName)
            ? "Operator '%s' is PRIVATE in '%s'"_err_en_US
            : "'%s' is PRIVATE in '%s'"_err_en_US,
        useName, useModuleScope_->name());
    return {};
  }
  auto &localSymbol{MakeSymbol(localName)};
  AddUse(useName, localSymbol, *useSymbol);
  return {&localSymbol, useSymbol};
}

// symbol must be either a Use or a Generic formed by merging two uses.
// Convert it to a UseError with this additional location.
static void ConvertToUseError(
    Symbol &symbol, const SourceName &location, const Scope &module) {
  const auto *useDetails{symbol.detailsIf<UseDetails>()};
  if (!useDetails) {
    auto &genericDetails{symbol.get<GenericDetails>()};
    useDetails = &genericDetails.useDetails().value();
  }
  symbol.set_details(
      UseErrorDetails{*useDetails}.add_occurrence(location, module));
}

void ModuleVisitor::AddUse(
    const SourceName &location, Symbol &localSymbol, const Symbol &useSymbol) {
  localSymbol.attrs() = useSymbol.attrs();
  localSymbol.attrs() &= ~Attrs{Attr::PUBLIC, Attr::PRIVATE};
  localSymbol.flags() = useSymbol.flags();
  if (auto *useDetails{localSymbol.detailsIf<UseDetails>()}) {
    const Symbol &ultimate{localSymbol.GetUltimate()};
    if (ultimate == useSymbol.GetUltimate()) {
      // use-associating the same symbol again -- ok
    } else if (ultimate.has<GenericDetails>() &&
        useSymbol.has<GenericDetails>()) {
      // use-associating generics with the same names: merge them into a
      // new generic in this scope
      auto genericDetails{ultimate.get<GenericDetails>()};
      genericDetails.set_useDetails(*useDetails);
      genericDetails.AddSpecificProcsFrom(useSymbol);
      EraseSymbol(localSymbol);
      MakeSymbol(
          localSymbol.name(), ultimate.attrs(), std::move(genericDetails));
    } else {
      ConvertToUseError(localSymbol, location, *useModuleScope_);
    }
  } else {
    auto *genericDetails{localSymbol.detailsIf<GenericDetails>()};
    if (genericDetails && genericDetails->useDetails().has_value()) {
      // localSymbol came from merging two use-associated generics
      if (useSymbol.has<GenericDetails>()) {
        genericDetails->AddSpecificProcsFrom(useSymbol);
      } else {
        ConvertToUseError(localSymbol, location, *useModuleScope_);
      }
    } else if (auto *details{localSymbol.detailsIf<UseErrorDetails>()}) {
      details->add_occurrence(location, *useModuleScope_);
    } else if (!localSymbol.has<UnknownDetails>()) {
      Say(location,
          "Cannot use-associate '%s'; it is already declared in this scope"_err_en_US,
          localSymbol.name())
          .Attach(localSymbol.name(), "Previous declaration of '%s'"_en_US,
              localSymbol.name());
    } else {
      localSymbol.set_details(UseDetails{location, useSymbol});
    }
  }
}

bool ModuleVisitor::BeginSubmodule(
    const parser::Name &name, const parser::ParentIdentifier &parentId) {
  auto &ancestorName{std::get<parser::Name>(parentId.t)};
  auto &parentName{std::get<std::optional<parser::Name>>(parentId.t)};
  Scope *ancestor{FindModule(ancestorName)};
  if (!ancestor) {
    return false;
  }
  Scope *parentScope{parentName ? FindModule(*parentName, ancestor) : ancestor};
  if (!parentScope) {
    return false;
  }
  PushScope(*parentScope);  // submodule is hosted in parent
  BeginModule(name, true);
  if (!ancestor->AddSubmodule(name.source, currScope())) {
    Say(name, "Module '%s' already has a submodule named '%s'"_err_en_US,
        ancestorName.source, name.source);
  }
  return true;
}

void ModuleVisitor::BeginModule(const parser::Name &name, bool isSubmodule) {
  auto &symbol{MakeSymbol(name, ModuleDetails{isSubmodule})};
  auto &details{symbol.get<ModuleDetails>()};
  PushScope(Scope::Kind::Module, &symbol);
  details.set_scope(&currScope());
  defaultAccess_ = Attr::PUBLIC;
  prevAccessStmt_ = nullptr;
}

// Find a module or submodule by name and return its scope.
// If ancestor is present, look for a submodule of that ancestor module.
// May have to read a .mod file to find it.
// If an error occurs, report it and return nullptr.
Scope *ModuleVisitor::FindModule(const parser::Name &name, Scope *ancestor) {
  ModFileReader reader{context()};
  auto *scope{reader.Read(name.source, ancestor)};
  if (!scope) {
    return nullptr;
  }
  if (scope->kind() != Scope::Kind::Module) {
    Say(name, "'%s' is not a module"_err_en_US);
    return nullptr;
  }
  Resolve(name, scope->symbol());
  return scope;
}

void ModuleVisitor::ApplyDefaultAccess() {
  for (auto &pair : currScope()) {
    Symbol &symbol = *pair.second;
    if (!symbol.attrs().HasAny({Attr::PUBLIC, Attr::PRIVATE})) {
      symbol.attrs().set(defaultAccess_);
    }
  }
}

// InterfaceVistor implementation

bool InterfaceVisitor::Pre(const parser::InterfaceStmt &x) {
  bool isAbstract{std::holds_alternative<parser::Abstract>(x.u)};
  genericInfo_.emplace(/*isInterface*/ true, isAbstract);
  return true;
}

void InterfaceVisitor::Post(const parser::EndInterfaceStmt &) {
  genericInfo_.pop();
}

// Create a symbol in genericSymbol_ for this GenericSpec.
bool InterfaceVisitor::Pre(const parser::GenericSpec &x) {
  auto info{GenericSpecInfo{x}};
  const SourceName &symbolName{info.symbolName()};
  if (IsLogicalConstant(context(), symbolName)) {
    Say(symbolName,
        "Logical constant '%s' may not be used as a defined operator"_err_en_US);
    return false;
  }
  Symbol *symbol{currScope().FindSymbol(symbolName)};
  if (symbol) {
    if (symbol->has<DerivedTypeDetails>()) {
      // A generic and derived type with same name: create a generic symbol
      // and save derived type in it.
      CHECK(symbol->scope()->symbol() == symbol);
      GenericDetails details;
      details.set_derivedType(*symbol);
      EraseSymbol(*symbol);
      symbol = &MakeSymbol(symbolName);
      symbol->set_details(details);
      // preserve access attributes
      symbol->attrs() |=
          details.derivedType()->attrs() & Attrs{Attr::PUBLIC, Attr::PRIVATE};
    } else if (symbol->has<UnknownDetails>()) {
      // okay
    } else if (!symbol->IsSubprogram()) {
      SayAlreadyDeclared(symbolName, *symbol);
      EraseSymbol(*symbol);
      symbol = nullptr;
    } else if (symbol->has<UseDetails>()) {
      // copy the USEd symbol into this scope so we can modify it
      const Symbol &ultimate{symbol->GetUltimate()};
      EraseSymbol(*symbol);
      symbol = &CopySymbol(symbolName, ultimate);
      if (const auto *details{ultimate.detailsIf<GenericDetails>()}) {
        symbol->set_details(GenericDetails{details->specificProcs()});
      } else if (const auto *details{ultimate.detailsIf<SubprogramDetails>()}) {
        symbol->set_details(SubprogramDetails{*details});
      } else {
        DIE("unexpected kind of symbol");
      }
    }
  }
  if (!symbol || symbol->has<UnknownDetails>()) {
    symbol = &MakeSymbol(symbolName);
    symbol->set_details(GenericDetails{});
  }
  if (symbol->has<GenericDetails>()) {
    // okay
  } else if (symbol->has<SubprogramDetails>() ||
      symbol->has<SubprogramNameDetails>()) {
    GenericDetails genericDetails;
    genericDetails.set_specific(*symbol);
    EraseSymbol(*symbol);
    symbol = &MakeSymbol(symbolName);
    symbol->set_details(genericDetails);
  } else {
    common::die("unexpected kind of symbol");
  }
  info.Resolve(symbol);
  SetGenericSymbol(*symbol);
  return false;
}

bool InterfaceVisitor::Pre(const parser::ProcedureStmt &x) {
  if (!isGeneric()) {
    Say("A PROCEDURE statement is only allowed in a generic interface block"_err_en_US);
    return false;
  }
  auto kind{std::get<parser::ProcedureStmt::Kind>(x.t)};
  const auto &names{std::get<std::list<parser::Name>>(x.t)};
  AddSpecificProcs(names, kind);
  return false;
}

bool InterfaceVisitor::Pre(const parser::GenericStmt &x) {
  genericInfo_.emplace(/*isInterface*/ false);
  return true;
}
void InterfaceVisitor::Post(const parser::GenericStmt &x) {
  if (auto &accessSpec{std::get<std::optional<parser::AccessSpec>>(x.t)}) {
    GetGenericInfo().symbol->attrs().set(AccessSpecToAttr(*accessSpec));
  }
  const auto &names{std::get<std::list<parser::Name>>(x.t)};
  AddSpecificProcs(names, ProcedureKind::Procedure);
  genericInfo_.pop();
}

bool InterfaceVisitor::inInterfaceBlock() const {
  return !genericInfo_.empty() && GetGenericInfo().isInterface;
}
bool InterfaceVisitor::isGeneric() const {
  return !genericInfo_.empty() && GetGenericInfo().symbol != nullptr;
}
bool InterfaceVisitor::isAbstract() const {
  return !genericInfo_.empty() && GetGenericInfo().isAbstract;
}
GenericDetails &InterfaceVisitor::GetGenericDetails() {
  return GetGenericInfo().symbol->get<GenericDetails>();
}

void InterfaceVisitor::AddSpecificProcs(
    const std::list<parser::Name> &names, ProcedureKind kind) {
  for (const auto &name : names) {
    specificProcs_.emplace(
        GetGenericInfo().symbol, std::make_pair(&name, kind));
  }
}

// By now we should have seen all specific procedures referenced by name in
// this generic interface. Resolve those names to symbols.
void InterfaceVisitor::ResolveSpecificsInGeneric(Symbol &generic) {
  auto &details{generic.get<GenericDetails>()};
  std::set<SourceName> namesSeen;  // to check for duplicate names
  for (const auto *symbol : details.specificProcs()) {
    namesSeen.insert(symbol->name());
  }
  auto range{specificProcs_.equal_range(&generic)};
  for (auto it{range.first}; it != range.second; ++it) {
    auto *name{it->second.first};
    auto kind{it->second.second};
    const auto *symbol{FindSymbol(*name)};
    if (!symbol) {
      Say(*name, "Procedure '%s' not found"_err_en_US);
      continue;
    }
    symbol = &symbol->GetUltimate();
    if (symbol == &generic) {
      if (auto *specific{generic.get<GenericDetails>().specific()}) {
        symbol = specific;
      }
    }
    if (!symbol->has<SubprogramDetails>() &&
        !symbol->has<SubprogramNameDetails>()) {
      Say(*name, "'%s' is not a subprogram"_err_en_US);
      continue;
    }
    if (kind == ProcedureKind::ModuleProcedure) {
      if (const auto *nd{symbol->detailsIf<SubprogramNameDetails>()}) {
        if (nd->kind() != SubprogramKind::Module) {
          Say(*name, "'%s' is not a module procedure"_err_en_US);
        }
      } else {
        // USE-associated procedure
        const auto *sd{symbol->detailsIf<SubprogramDetails>()};
        CHECK(sd != nullptr);
        if (symbol->owner().kind() != Scope::Kind::Module ||
            sd->isInterface()) {
          Say(*name, "'%s' is not a module procedure"_err_en_US);
        }
      }
    }
    if (!namesSeen.insert(name->source).second) {
      Say(*name,
          IsDefinedOperator(generic.name())
              ? "Procedure '%s' is already specified in generic operator '%s'"_err_en_US
              : "Procedure '%s' is already specified in generic '%s'"_err_en_US,
          name->source, generic.name());
      continue;
    }
    details.add_specificProc(*symbol);
  }
  specificProcs_.erase(range.first, range.second);
}

// Check that the specific procedures are all functions or all subroutines.
// If there is a derived type with the same name they must be functions.
// Set the corresponding flag on generic.
void InterfaceVisitor::CheckGenericProcedures(Symbol &generic) {
  ResolveSpecificsInGeneric(generic);
  auto &details{generic.get<GenericDetails>()};
  if (auto *proc{details.CheckSpecific()}) {
    SayAlreadyDeclared(generic.name(), *proc);
  }
  auto &specifics{details.specificProcs()};
  if (specifics.empty()) {
    if (details.derivedType()) {
      generic.set(Symbol::Flag::Function);
    }
    return;
  }
  auto &firstSpecific{*specifics.front()};
  bool isFunction{firstSpecific.test(Symbol::Flag::Function)};
  for (const auto *specific : specifics) {
    if (isFunction != specific->test(Symbol::Flag::Function)) {  // C1514
      auto &msg{Say(generic.name(),
          "Generic interface '%s' has both a function and a subroutine"_err_en_US)};
      if (isFunction) {
        msg.Attach(firstSpecific.name(), "Function declaration"_en_US);
        msg.Attach(specific->name(), "Subroutine declaration"_en_US);
      } else {
        msg.Attach(firstSpecific.name(), "Subroutine declaration"_en_US);
        msg.Attach(specific->name(), "Function declaration"_en_US);
      }
    }
  }
  if (!isFunction && details.derivedType()) {
    SayDerivedType(generic.name(),
        "Generic interface '%s' may only contain functions due to derived type"
        " with same name"_err_en_US,
        *details.derivedType()->scope());
  }
  generic.set(isFunction ? Symbol::Flag::Function : Symbol::Flag::Subroutine);
}

void InterfaceVisitor::SayNotDistinguishable(
    const Symbol &generic, const Symbol &proc1, const Symbol &proc2) {
  auto &&text{IsDefinedOperator(generic.name())
          ? "Generic operator '%s' may not have specific procedures '%s'"
            " and '%s' as their interfaces are not distinguishable"_err_en_US
          : "Generic '%s' may not have specific procedures '%s'"
            " and '%s' as their interfaces are not distinguishable"_err_en_US};
  auto &msg{context().Say(generic.name(), std::move(text), generic.name(),
      proc1.name(), proc2.name())};
  if (!proc1.IsFromModFile()) {
    msg.Attach(proc1.name(), "Declaration of '%s'"_en_US, proc1.name());
  }
  if (!proc2.IsFromModFile()) {
    msg.Attach(proc2.name(), "Declaration of '%s'"_en_US, proc2.name());
  }
}

static GenericKind GetGenericKind(const Symbol &generic) {
  return std::visit(
      common::visitors{
          [&](const GenericDetails &x) { return x.kind(); },
          [&](const GenericBindingDetails &x) { return x.kind(); },
          [](auto &) -> GenericKind { DIE("not a generic"); },
      },
      generic.details());
}

static bool IsOperatorOrAssignment(const Symbol &generic) {
  auto kind{GetGenericKind(generic)};
  return kind == GenericKind::DefinedOp || kind == GenericKind::Assignment ||
      (kind >= GenericKind::OpPower && kind <= GenericKind::OpNEQV);
}

// Check that the specifics of this generic are distinguishable from each other
void InterfaceVisitor::CheckSpecificsAreDistinguishable(
    const Symbol &generic, const SymbolVector &specifics) {
  auto count{specifics.size()};
  if (specifics.size() < 2) {
    return;
  }
  auto distinguishable{IsOperatorOrAssignment(generic)
          ? evaluate::characteristics::DistinguishableOpOrAssign
          : evaluate::characteristics::Distinguishable};
  using evaluate::characteristics::Procedure;
  std::vector<Procedure> procs;
  for (const Symbol *symbol : specifics) {
    auto proc{Procedure::Characterize(*symbol, context().intrinsics())};
    if (!proc) {
      return;
    }
    procs.emplace_back(*proc);
  }
  for (std::size_t i1{0}; i1 < count - 1; ++i1) {
    auto &proc1{procs[i1]};
    for (std::size_t i2{i1 + 1}; i2 < count; ++i2) {
      auto &proc2{procs[i2]};
      if (!distinguishable(proc1, proc2)) {
        SayNotDistinguishable(generic, *specifics[i1], *specifics[i2]);
      }
    }
  }
}

// SubprogramVisitor implementation

void SubprogramVisitor::Post(const parser::StmtFunctionStmt &x) {
  if (badStmtFuncFound_) {
    return;  // This wasn't really a stmt function so no scope was created
  }
  PopScope();
}
// Return false if it is actually an assignment statement.
bool SubprogramVisitor::HandleStmtFunction(const parser::StmtFunctionStmt &x) {
  const auto &name{std::get<parser::Name>(x.t)};
  const DeclTypeSpec *resultType{nullptr};
  // Look up name: provides return type or tells us if it's an array
  if (auto *symbol{FindSymbol(name)}) {
    auto *details{symbol->detailsIf<EntityDetails>()};
    if (!details) {
      badStmtFuncFound_ = true;
      return false;
    }
    // TODO: check that attrs are compatible with stmt func
    resultType = details->type();
    EraseSymbol(name);
  }
  if (badStmtFuncFound_) {
    Say(name, "'%s' has not been declared as an array"_err_en_US);
    return true;
  }
  auto &symbol{PushSubprogramScope(name, Symbol::Flag::Function)};
  auto &details{symbol.get<SubprogramDetails>()};
  for (const auto &dummyName : std::get<std::list<parser::Name>>(x.t)) {
    EntityDetails dummyDetails{true};
    if (auto *dummySymbol{FindInScope(currScope().parent(), dummyName)}) {
      if (auto *d{dummySymbol->detailsIf<EntityDetails>()}) {
        if (d->type()) {
          dummyDetails.set_type(*d->type());
        }
      }
    }
    details.add_dummyArg(MakeSymbol(dummyName, std::move(dummyDetails)));
  }
  EraseSymbol(name);  // added by PushSubprogramScope
  EntityDetails resultDetails;
  if (resultType) {
    resultDetails.set_type(*resultType);
  }
  details.set_result(MakeSymbol(name, std::move(resultDetails)));
  return true;
}

bool SubprogramVisitor::Pre(const parser::Suffix &suffix) {
  if (suffix.resultName) {
    funcInfo_.resultName = &suffix.resultName.value();
  }
  return true;
}

bool SubprogramVisitor::Pre(const parser::PrefixSpec &x) {
  // Save this to process after UseStmt and ImplicitPart
  if (const auto *parsedType{std::get_if<parser::DeclarationTypeSpec>(&x.u)}) {
    funcInfo_.parsedType = parsedType;
    funcInfo_.source = currStmtSource();
    return false;
  } else {
    return true;
  }
}

void SubprogramVisitor::Post(const parser::ImplicitPart &) {
  // If the function has a type in the prefix, process it now
  if (funcInfo_.parsedType) {
    messageHandler().set_currStmtSource(funcInfo_.source);
    if (const auto *type{ProcessTypeSpec(*funcInfo_.parsedType)}) {
      funcInfo_.resultSymbol->SetType(*type);
    }
  }
  funcInfo_ = {};
}

bool SubprogramVisitor::Pre(const parser::InterfaceBody::Subroutine &x) {
  const auto &name{std::get<parser::Name>(
      std::get<parser::Statement<parser::SubroutineStmt>>(x.t).statement.t)};
  return BeginSubprogram(name, Symbol::Flag::Subroutine);
}
void SubprogramVisitor::Post(const parser::InterfaceBody::Subroutine &) {
  EndSubprogram();
}
bool SubprogramVisitor::Pre(const parser::InterfaceBody::Function &x) {
  const auto &name{std::get<parser::Name>(
      std::get<parser::Statement<parser::FunctionStmt>>(x.t).statement.t)};
  return BeginSubprogram(name, Symbol::Flag::Function);
}
void SubprogramVisitor::Post(const parser::InterfaceBody::Function &) {
  EndSubprogram();
}

bool SubprogramVisitor::Pre(const parser::SubroutineStmt &stmt) {
  return BeginAttrs();
}
bool SubprogramVisitor::Pre(const parser::FunctionStmt &stmt) {
  return BeginAttrs();
}

void SubprogramVisitor::Post(const parser::SubroutineStmt &stmt) {
  const auto &name{std::get<parser::Name>(stmt.t)};
  auto &details{PostSubprogramStmt(name)};
  for (const auto &dummyArg : std::get<std::list<parser::DummyArg>>(stmt.t)) {
    if (const auto *dummyName{std::get_if<parser::Name>(&dummyArg.u)}) {
      Symbol &dummy{MakeSymbol(*dummyName, EntityDetails(true))};
      details.add_dummyArg(dummy);
    } else {
      details.add_alternateReturn();
    }
  }
}

void SubprogramVisitor::Post(const parser::FunctionStmt &stmt) {
  const auto &name{std::get<parser::Name>(stmt.t)};
  auto &details{PostSubprogramStmt(name)};
  for (const auto &dummyName : std::get<std::list<parser::Name>>(stmt.t)) {
    Symbol &dummy{MakeSymbol(dummyName, EntityDetails(true))};
    details.add_dummyArg(dummy);
  }
  const parser::Name *funcResultName;
  if (funcInfo_.resultName && funcInfo_.resultName->source != name.source) {
    funcResultName = funcInfo_.resultName;
  } else {
    EraseSymbol(name);  // was added by PushSubprogramScope
    funcResultName = &name;
  }
  // add function result to function scope
  EntityDetails funcResultDetails;
  funcResultDetails.set_funcResult(true);
  funcInfo_.resultSymbol =
      &MakeSymbol(*funcResultName, std::move(funcResultDetails));
  details.set_result(*funcInfo_.resultSymbol);
  name.symbol = currScope().symbol();  // must not be function result symbol
}

SubprogramDetails &SubprogramVisitor::PostSubprogramStmt(
    const parser::Name &name) {
  Symbol &symbol{*currScope().symbol()};
  CHECK(name.source == symbol.name());
  SetBindNameOn(symbol);
  symbol.attrs() |= EndAttrs();
  if (symbol.attrs().test(Attr::MODULE)) {
    symbol.attrs().set(Attr::EXTERNAL, false);
  }
  return symbol.get<SubprogramDetails>();
}

bool SubprogramVisitor::BeginSubprogram(
    const parser::Name &name, Symbol::Flag subpFlag, bool hasModulePrefix) {
  if (hasModulePrefix && !inInterfaceBlock()) {
    auto *symbol{FindSymbol(name)};
    if (!symbol || !symbol->IsSeparateModuleProc()) {
      Say(name, "'%s' was not declared a separate module procedure"_err_en_US);
      return false;
    }
    if (symbol->owner() == currScope()) {
      // separate module procedure declared and defined in same module
      PushScope(*symbol->scope());
    } else {
      PushSubprogramScope(name, subpFlag);
    }
  } else {
    PushSubprogramScope(name, subpFlag);
  }
  return true;
}
void SubprogramVisitor::EndSubprogram() { PopScope(); }

Symbol &SubprogramVisitor::PushSubprogramScope(
    const parser::Name &name, Symbol::Flag subpFlag) {
  auto *symbol{GetSpecificFromGeneric(name)};
  if (!symbol) {
    if (auto *prev{FindSymbol(name)}) {
      if (prev->attrs().test(Attr::EXTERNAL) &&
          prev->has<ProcEntityDetails>()) {
        // this subprogram was previously called, now being declared
        if (!prev->test(subpFlag)) {
          Say2(name,
              subpFlag == Symbol::Flag::Function
                  ? "'%s' was previously called as a subroutine"_err_en_US
                  : "'%s' was previously called as a function"_err_en_US,
              *prev, "Previous call of '%s'"_en_US);
        }
        EraseSymbol(name);
      }
    }
    symbol = &MakeSymbol(name, SubprogramDetails{});
    symbol->set(subpFlag);
  }
  PushScope(Scope::Kind::Subprogram, symbol);
  auto &details{symbol->get<SubprogramDetails>()};
  if (inInterfaceBlock()) {
    details.set_isInterface();
    if (!isAbstract()) {
      MakeExternal(*symbol);
    }
    if (isGeneric()) {
      GetGenericDetails().add_specificProc(*symbol);
    }
    implicitRules().set_inheritFromParent(false);
  }
  FindSymbol(name)->set(subpFlag);
  return *symbol;
}

// If name is a generic, return specific subprogram with the same name.
Symbol *SubprogramVisitor::GetSpecificFromGeneric(const parser::Name &name) {
  if (auto *symbol{FindSymbol(name)}) {
    if (auto *details{symbol->detailsIf<GenericDetails>()}) {
      // found generic, want subprogram
      auto *specific{details->specific()};
      if (isGeneric()) {
        if (specific) {
          SayAlreadyDeclared(name, *specific);
        } else {
          EraseSymbol(name);
          specific = &MakeSymbol(name, Attrs{}, SubprogramDetails{});
          GetGenericDetails().set_specific(*specific);
        }
      }
      if (specific) {
        if (!specific->has<SubprogramDetails>()) {
          specific->set_details(SubprogramDetails{});
        }
        return specific;
      }
    }
  }
  return nullptr;
}

// DeclarationVisitor implementation

bool DeclarationVisitor::BeginDecl() {
  BeginDeclTypeSpec();
  BeginArraySpec();
  return BeginAttrs();
}
void DeclarationVisitor::EndDecl() {
  EndDeclTypeSpec();
  EndArraySpec();
  EndAttrs();
}

bool DeclarationVisitor::CheckUseError(const parser::Name &name) {
  const auto *details{name.symbol->detailsIf<UseErrorDetails>()};
  if (!details) {
    return false;
  }
  Message &msg{Say(name, "Reference to '%s' is ambiguous"_err_en_US)};
  for (const auto &[location, module] : details->occurrences()) {
    msg.Attach(location, "'%s' was use-associated from module '%s'"_en_US,
        name.source, module->name());
  }
  return true;
}

// Report error if accessibility of symbol doesn't match isPrivate.
void DeclarationVisitor::CheckAccessibility(
    const SourceName &name, bool isPrivate, Symbol &symbol) {
  if (symbol.attrs().test(Attr::PRIVATE) != isPrivate) {
    Say2(name,
        "'%s' does not have the same accessibility as its previous declaration"_err_en_US,
        symbol, "Previous declaration of '%s'"_en_US);
  }
}

// Check that component is accessible from current scope.
bool DeclarationVisitor::CheckAccessibleComponent(
    const SourceName &name, const Symbol &symbol) {
  if (!symbol.attrs().test(Attr::PRIVATE)) {
    return true;
  }
  // component must be in a module/submodule because of PRIVATE:
  const Scope *moduleScope{&symbol.owner()};
  CHECK(moduleScope->IsDerivedType());
  while (
      moduleScope->kind() != Scope::Kind::Module && !moduleScope->IsGlobal()) {
    moduleScope = &moduleScope->parent();
  }
  if (moduleScope->kind() == Scope::Kind::Module) {
    for (auto *scope{&currScope()}; !scope->IsGlobal();
         scope = &scope->parent()) {
      if (scope == moduleScope) {
        return true;
      }
    }
    Say(name,
        "PRIVATE component '%s' is only accessible within module '%s'"_err_en_US,
        name.ToString(), moduleScope->name());
  } else {
    Say(name,
        "PRIVATE component '%s' is only accessible within its module"_err_en_US,
        name.ToString());
  }
  return false;
}

void DeclarationVisitor::Post(const parser::DimensionStmt::Declaration &x) {
  const auto &name{std::get<parser::Name>(x.t)};
  DeclareObjectEntity(name, Attrs{});
}
void DeclarationVisitor::Post(const parser::CodimensionDecl &x) {
  const auto &name{std::get<parser::Name>(x.t)};
  DeclareObjectEntity(name, Attrs{});
}

bool DeclarationVisitor::Pre(const parser::Initialization &) {
  // Defer inspection of initializers to Initialization() so that the
  // symbol being initialized will be available within the initialization
  // expression.
  return false;
}

void DeclarationVisitor::Post(const parser::EntityDecl &x) {
  // TODO: may be under StructureStmt
  const auto &name{std::get<parser::ObjectName>(x.t)};
  Attrs attrs{attrs_ ? HandleSaveName(name.source, *attrs_) : Attrs{}};
  Symbol &symbol{DeclareUnknownEntity(name, attrs)};
  if (auto &init{std::get<std::optional<parser::Initialization>>(x.t)}) {
    if (ConvertToObjectEntity(symbol)) {
      Initialization(name, *init, false);
    }
  } else if (attrs.test(Attr::PARAMETER)) {
    Say(name, "Missing initialization for parameter '%s'"_err_en_US);
  }
}

void DeclarationVisitor::Post(const parser::PointerDecl &x) {
  const auto &name{std::get<parser::Name>(x.t)};
  DeclareUnknownEntity(name, Attrs{Attr::POINTER});
}

bool DeclarationVisitor::Pre(const parser::BindEntity &x) {
  auto kind{std::get<parser::BindEntity::Kind>(x.t)};
  auto &name{std::get<parser::Name>(x.t)};
  Symbol *symbol;
  if (kind == parser::BindEntity::Kind::Object) {
    symbol = &HandleAttributeStmt(Attr::BIND_C, name);
  } else {
    symbol = &MakeCommonBlockSymbol(name);
    symbol->attrs().set(Attr::BIND_C);
  }
  SetBindNameOn(*symbol);
  return false;
}
bool DeclarationVisitor::Pre(const parser::NamedConstantDef &x) {
  auto &name{std::get<parser::NamedConstant>(x.t).v};
  auto &symbol{HandleAttributeStmt(Attr::PARAMETER, name)};
  if (!ConvertToObjectEntity(symbol)) {
    SayWithDecl(
        name, symbol, "PARAMETER attribute not allowed on '%s'"_err_en_US);
    return false;
  }
  const auto &expr{std::get<parser::ConstantExpr>(x.t)};
  ApplyImplicitRules(symbol);
  Walk(expr);
  if (auto converted{
          EvaluateConvertedExpr(symbol, expr, expr.thing.value().source)}) {
    symbol.get<ObjectEntityDetails>().set_init(std::move(*converted));
  }
  return false;
}
bool DeclarationVisitor::Pre(const parser::NamedConstant &x) {
  const parser::Name &name{x.v};
  if (!FindSymbol(name)) {
    Say(name, "Named constant '%s' not found"_err_en_US);
  } else {
    CheckUseError(name);
  }
  return false;
}
bool DeclarationVisitor::Pre(const parser::AsynchronousStmt &x) {
  return HandleAttributeStmt(Attr::ASYNCHRONOUS, x.v);
}
bool DeclarationVisitor::Pre(const parser::ContiguousStmt &x) {
  return HandleAttributeStmt(Attr::CONTIGUOUS, x.v);
}
bool DeclarationVisitor::Pre(const parser::ExternalStmt &x) {
  HandleAttributeStmt(Attr::EXTERNAL, x.v);
  for (const auto &name : x.v) {
    auto *symbol{FindSymbol(name)};
    if (!ConvertToProcEntity(*symbol)) {
      SayWithDecl(
          name, *symbol, "EXTERNAL attribute not allowed on '%s'"_err_en_US);
    }
  }
  return false;
}
bool DeclarationVisitor::Pre(const parser::IntentStmt &x) {
  auto &intentSpec{std::get<parser::IntentSpec>(x.t)};
  auto &names{std::get<std::list<parser::Name>>(x.t)};
  return CheckNotInBlock("INTENT") &&
      HandleAttributeStmt(IntentSpecToAttr(intentSpec), names);
}
bool DeclarationVisitor::Pre(const parser::IntrinsicStmt &x) {
  HandleAttributeStmt(Attr::INTRINSIC, x.v);
  for (const auto &name : x.v) {
    auto *symbol{FindSymbol(name)};
    if (!ConvertToProcEntity(*symbol)) {
      SayWithDecl(
          name, *symbol, "INTRINSIC attribute not allowed on '%s'"_err_en_US);
    } else if (symbol->attrs().test(Attr::EXTERNAL)) {  // C840
      Say(symbol->name(),
          "Symbol '%s' cannot have both EXTERNAL and INTRINSIC attributes"_err_en_US,
          symbol->name());
    }
  }
  return false;
}
bool DeclarationVisitor::Pre(const parser::OptionalStmt &x) {
  return CheckNotInBlock("OPTIONAL") &&
      HandleAttributeStmt(Attr::OPTIONAL, x.v);
}
bool DeclarationVisitor::Pre(const parser::ProtectedStmt &x) {
  return HandleAttributeStmt(Attr::PROTECTED, x.v);
}
bool DeclarationVisitor::Pre(const parser::ValueStmt &x) {
  return CheckNotInBlock("VALUE") && HandleAttributeStmt(Attr::VALUE, x.v);
}
bool DeclarationVisitor::Pre(const parser::VolatileStmt &x) {
  return HandleAttributeStmt(Attr::VOLATILE, x.v);
}
// Handle a statement that sets an attribute on a list of names.
bool DeclarationVisitor::HandleAttributeStmt(
    Attr attr, const std::list<parser::Name> &names) {
  for (const auto &name : names) {
    HandleAttributeStmt(attr, name);
  }
  return false;
}
Symbol &DeclarationVisitor::HandleAttributeStmt(
    Attr attr, const parser::Name &name) {
  if (attr == Attr::INTRINSIC &&
      !context().intrinsics().IsIntrinsic(name.source.ToString())) {
    Say(name.source, "'%s' is not a known intrinsic procedure"_err_en_US);
  }
  auto *symbol{FindInScope(currScope(), name)};
  if (symbol) {
    // symbol was already there: set attribute on it
    if (attr == Attr::ASYNCHRONOUS || attr == Attr::VOLATILE) {
      // TODO: if in a BLOCK, attribute should only be set while in the block
    } else if (symbol->has<UseDetails>()) {
      Say(*currStmtSource(),
          "Cannot change %s attribute on use-associated '%s'"_err_en_US,
          EnumToString(attr), name.source);
    }
  } else {
    symbol = &MakeSymbol(name, EntityDetails{});
  }
  symbol->attrs().set(attr);
  symbol->attrs() = HandleSaveName(name.source, symbol->attrs());
  return *symbol;
}

bool DeclarationVisitor::CheckNotInBlock(const char *stmt) {
  if (currScope().kind() == Scope::Kind::Block) {
    Say(MessageFormattedText{
        "%s statement is not allowed in a BLOCK construct"_err_en_US, stmt});
    return false;
  } else {
    return true;
  }
}

void DeclarationVisitor::Post(const parser::ObjectDecl &x) {
  CHECK(objectDeclAttr_.has_value());
  const auto &name{std::get<parser::ObjectName>(x.t)};
  DeclareObjectEntity(name, Attrs{*objectDeclAttr_});
}

// Declare an entity not yet known to be an object or proc.
Symbol &DeclarationVisitor::DeclareUnknownEntity(
    const parser::Name &name, Attrs attrs) {
  if (!arraySpec().empty() || !coarraySpec().empty()) {
    return DeclareObjectEntity(name, attrs);
  } else {
    Symbol &symbol{DeclareEntity<EntityDetails>(name, attrs)};
    if (auto *type{GetDeclTypeSpec()}) {
      SetType(name, *type);
    }
    charInfo_.length.reset();
    SetBindNameOn(symbol);
    if (symbol.attrs().test(Attr::EXTERNAL)) {
      ConvertToProcEntity(symbol);
    }
    return symbol;
  }
}

Symbol &DeclarationVisitor::DeclareProcEntity(
    const parser::Name &name, Attrs attrs, const ProcInterface &interface) {
  Symbol &symbol{DeclareEntity<ProcEntityDetails>(name, attrs)};
  if (auto *details{symbol.detailsIf<ProcEntityDetails>()}) {
    if (interface.type()) {
      symbol.set(Symbol::Flag::Function);
    } else if (interface.symbol()) {
      if (interface.symbol()->test(Symbol::Flag::Function)) {
        symbol.set(Symbol::Flag::Function);
      } else if (interface.symbol()->test(Symbol::Flag::Subroutine)) {
        symbol.set(Symbol::Flag::Subroutine);
      }
    }
    details->set_interface(interface);
    SetBindNameOn(symbol);
    SetPassNameOn(symbol);
  }
  return symbol;
}

Symbol &DeclarationVisitor::DeclareObjectEntity(
    const parser::Name &name, Attrs attrs) {
  Symbol &symbol{DeclareEntity<ObjectEntityDetails>(name, attrs)};
  if (auto *details{symbol.detailsIf<ObjectEntityDetails>()}) {
    if (auto *type{GetDeclTypeSpec()}) {
      SetType(name, *type);
    }
    if (!arraySpec().empty()) {
      if (details->IsArray()) {
        Say(name,
            "The dimensions of '%s' have already been declared"_err_en_US);
        context().SetError(symbol);
      } else {
        details->set_shape(arraySpec());
      }
      ClearArraySpec();
    }
    if (!coarraySpec().empty()) {
      if (details->IsCoarray()) {
        Say(name,
            "The codimensions of '%s' have already been declared"_err_en_US);
        context().SetError(symbol);
      } else {
        details->set_coshape(coarraySpec());
      }
      ClearCoarraySpec();
    }
    SetBindNameOn(symbol);
  }
  charInfo_.length.reset();
  return symbol;
}

void DeclarationVisitor::Post(const parser::IntegerTypeSpec &x) {
  SetDeclTypeSpec(MakeNumericType(TypeCategory::Integer, x.v));
}
void DeclarationVisitor::Post(const parser::IntrinsicTypeSpec::Real &x) {
  SetDeclTypeSpec(MakeNumericType(TypeCategory::Real, x.kind));
}
void DeclarationVisitor::Post(const parser::IntrinsicTypeSpec::Complex &x) {
  SetDeclTypeSpec(MakeNumericType(TypeCategory::Complex, x.kind));
}
void DeclarationVisitor::Post(const parser::IntrinsicTypeSpec::Logical &x) {
  SetDeclTypeSpec(MakeLogicalType(x.kind));
}
void DeclarationVisitor::Post(const parser::IntrinsicTypeSpec::Character &x) {
  if (!charInfo_.length) {
    charInfo_.length = ParamValue{1};
  }
  if (!charInfo_.kind.has_value()) {
    charInfo_.kind =
        KindExpr{context().GetDefaultKind(TypeCategory::Character)};
  }
  SetDeclTypeSpec(currScope().MakeCharacterType(
      std::move(*charInfo_.length), std::move(*charInfo_.kind)));
  charInfo_ = {};
}
void DeclarationVisitor::Post(const parser::CharSelector::LengthAndKind &x) {
  charInfo_.kind = EvaluateSubscriptIntExpr(x.kind);
  if (x.length) {
    charInfo_.length = GetParamValue(*x.length);
  }
}
void DeclarationVisitor::Post(const parser::CharLength &x) {
  if (const auto *length{std::get_if<std::int64_t>(&x.u)}) {
    charInfo_.length = ParamValue{*length};
  } else {
    charInfo_.length = GetParamValue(std::get<parser::TypeParamValue>(x.u));
  }
}
void DeclarationVisitor::Post(const parser::LengthSelector &x) {
  if (const auto *param{std::get_if<parser::TypeParamValue>(&x.u)}) {
    charInfo_.length = GetParamValue(*param);
  }
}

bool DeclarationVisitor::Pre(const parser::KindParam &x) {
  if (const auto *kind{std::get_if<
          parser::Scalar<parser::Integer<parser::Constant<parser::Name>>>>(
          &x.u)}) {
    const parser::Name &name{kind->thing.thing.thing};
    if (!FindSymbol(name)) {
      Say(name, "Parameter '%s' not found"_err_en_US);
    }
  }
  return false;
}

bool DeclarationVisitor::Pre(const parser::DeclarationTypeSpec::Type &x) {
  CHECK(GetDeclTypeSpecCategory() == DeclTypeSpec::Category::TypeDerived);
  return true;
}

bool DeclarationVisitor::Pre(const parser::DeclarationTypeSpec::Class &x) {
  SetDeclTypeSpecCategory(DeclTypeSpec::Category::ClassDerived);
  return true;
}

bool DeclarationVisitor::Pre(const parser::DeclarationTypeSpec::Record &) {
  // TODO
  return true;
}

void DeclarationVisitor::Post(const parser::DerivedTypeSpec &x) {
  const auto &typeName{std::get<parser::Name>(x.t)};
  const Symbol *typeSymbol{ResolveDerivedType(typeName)};
  if (typeSymbol == nullptr) {
    return;
  }

  // This DerivedTypeSpec is created initially as a search key.
  // If it turns out to have the same name and actual parameter
  // value expressions as some other DerivedTypeSpec in the current
  // scope, then we'll use that extant spec; otherwise, when this
  // spec is distinct from all derived types previously instantiated
  // in the current scope, this spec will be moved to that collection.
  DerivedTypeSpec spec{*typeSymbol};

  // The expressions in a derived type specifier whose values define
  // non-defaulted type parameters are evaluated in the enclosing scope.
  // Default initialization expressions for the derived type's parameters
  // may reference other parameters so long as the declaration precedes the
  // use in the expression (10.1.12).  This is not necessarily the same
  // order as "type parameter order" (7.5.3.2).
  // Parameters of the most deeply nested "base class" come first when the
  // derived type is an extension.
  auto parameterNames{OrderParameterNames(*typeSymbol)};
  auto parameterDecls{OrderParameterDeclarations(*typeSymbol)};
  auto nextNameIter{parameterNames.begin()};
  bool seenAnyName{false};
  for (const auto &typeParamSpec :
      std::get<std::list<parser::TypeParamSpec>>(x.t)) {
    const auto &optKeyword{
        std::get<std::optional<parser::Keyword>>(typeParamSpec.t)};
    SourceName name;
    if (optKeyword.has_value()) {
      seenAnyName = true;
      name = optKeyword->v.source;
      auto it{std::find_if(parameterDecls.begin(), parameterDecls.end(),
          [&](const Symbol *symbol) { return symbol->name() == name; })};
      if (it == parameterDecls.end()) {
        Say(name,
            "'%s' is not the name of a parameter for this type"_err_en_US);
      } else {
        Resolve(optKeyword->v, const_cast<Symbol *>(*it));
      }
    } else if (seenAnyName) {
      Say(typeName.source, "Type parameter value must have a name"_err_en_US);
      continue;
    } else if (nextNameIter != parameterNames.end()) {
      name = *nextNameIter++;
    } else {
      Say(typeName.source,
          "Too many type parameters given for derived type '%s'"_err_en_US);
      break;
    }
    if (spec.FindParameter(name)) {
      Say(typeName.source,
          "Multiple values given for type parameter '%s'"_err_en_US, name);
    } else {
      const auto &value{std::get<parser::TypeParamValue>(typeParamSpec.t)};
      ParamValue param{GetParamValue(value)};  // folded
      if (!param.isExplicit() || param.GetExplicit().has_value()) {
        spec.AddParamValue(name, std::move(param));
      }
    }
  }

  // Ensure that any type parameter without an explicit value has a
  // default initialization in the derived type's definition.
  const Scope *typeScope{typeSymbol->scope()};
  CHECK(typeScope != nullptr);
  for (const SourceName &name : parameterNames) {
    if (!spec.FindParameter(name)) {
      auto it{std::find_if(parameterDecls.begin(), parameterDecls.end(),
          [&](const Symbol *symbol) { return symbol->name() == name; })};
      if (it != parameterDecls.end()) {
        const auto *details{(*it)->detailsIf<TypeParamDetails>()};
        if (details == nullptr || !details->init().has_value()) {
          Say(typeName.source,
              "Type parameter '%s' lacks a value and has no default"_err_en_US,
              name);
        }
      }
    }
  }

  auto category{GetDeclTypeSpecCategory()};
  ProcessParameterExpressions(spec, context().foldingContext());
  if (const DeclTypeSpec *
      extant{currScope().FindInstantiatedDerivedType(spec, category)}) {
    // This derived type and parameter expressions (if any) are already present
    // in this scope.
    SetDeclTypeSpec(*extant);
  } else {
    DeclTypeSpec &type{currScope().MakeDerivedType(std::move(spec), category)};
    if (parameterNames.empty() || currScope().IsParameterizedDerivedType()) {
      // The derived type being instantiated is not a parameterized derived
      // type, or the instantiation is within the definition of a parameterized
      // derived type; don't instantiate a new scope.
      type.derivedTypeSpec().set_scope(*typeScope);
    } else {
      // This is a parameterized derived type and this spec is not in the
      // context of a parameterized derived type definition, so we need to
      // clone its contents, specialize them with the actual type parameter
      // values, and check constraints.
      auto save{GetFoldingContext().messages().SetLocation(*currStmtSource())};
      InstantiateDerivedType(type.derivedTypeSpec(), currScope(), context());
    }
    SetDeclTypeSpec(type);
  }
  // Capture the DerivedTypeSpec in the parse tree for use in building
  // structure constructor expressions.
  x.derivedTypeSpec = &GetDeclTypeSpec()->derivedTypeSpec();
}

// The descendents of DerivedTypeDef in the parse tree are visited directly
// in this Pre() routine so that recursive use of the derived type can be
// supported in the components.
bool DeclarationVisitor::Pre(const parser::DerivedTypeDef &x) {
  auto &stmt{std::get<parser::Statement<parser::DerivedTypeStmt>>(x.t)};
  Walk(stmt);
  Walk(std::get<std::list<parser::Statement<parser::TypeParamDefStmt>>>(x.t));
  auto &scope{currScope()};
  CHECK(scope.symbol() != nullptr);
  CHECK(scope.symbol()->scope() == &scope);
  auto &details{scope.symbol()->get<DerivedTypeDetails>()};
  std::set<SourceName> paramNames;
  for (auto &paramName : std::get<std::list<parser::Name>>(stmt.statement.t)) {
    details.add_paramName(paramName.source);
    auto *symbol{FindInScope(scope, paramName)};
    if (!symbol) {
      Say(paramName,
          "No definition found for type parameter '%s'"_err_en_US);  // C742
    } else if (!symbol->has<TypeParamDetails>()) {
      Say2(paramName, "'%s' is not defined as a type parameter"_err_en_US,
          *symbol, "Definition of '%s'"_en_US);  // C741
    }
    if (!paramNames.insert(paramName.source).second) {
      Say(paramName,
          "Duplicate type parameter name: '%s'"_err_en_US);  // C731
    }
  }
  for (const auto &[name, symbol] : currScope()) {
    if (symbol->has<TypeParamDetails>() && !paramNames.count(name)) {
      SayDerivedType(name,
          "'%s' is not a type parameter of this derived type"_err_en_US,
          currScope());  // C742
    }
  }
  Walk(std::get<std::list<parser::Statement<parser::PrivateOrSequence>>>(x.t));
  if (derivedTypeInfo_.sequence) {
    details.set_sequence(true);
    if (derivedTypeInfo_.extends) {
      Say(stmt.source,
          "A sequence type may not have the EXTENDS attribute"_err_en_US);  // C735
    }
    if (!details.paramNames().empty()) {
      Say(stmt.source,
          "A sequence type may not have type parameters"_err_en_US);  // C740
    }
  }
  Walk(std::get<std::list<parser::Statement<parser::ComponentDefStmt>>>(x.t));
  Walk(std::get<std::optional<parser::TypeBoundProcedurePart>>(x.t));
  Walk(std::get<parser::Statement<parser::EndTypeStmt>>(x.t));
  derivedTypeInfo_ = {};
  PopScope();
  return false;
}
bool DeclarationVisitor::Pre(const parser::DerivedTypeStmt &x) {
  return BeginAttrs();
}
void DeclarationVisitor::Post(const parser::DerivedTypeStmt &x) {
  auto &name{std::get<parser::Name>(x.t)};
  // Resolve the EXTENDS() clause before creating the derived
  // type's symbol to foil attempts to recursively extend a type.
  auto *extendsName{derivedTypeInfo_.extends};
  const Symbol *extendsType{nullptr};
  if (extendsName != nullptr) {
    if (extendsName->source == name.source) {
      Say(extendsName->source,
          "Derived type '%s' cannot extend itself"_err_en_US);
    } else {
      extendsType = ResolveDerivedType(*extendsName);
    }
  }
  auto &symbol{MakeSymbol(name, GetAttrs(), DerivedTypeDetails{})};
  derivedTypeInfo_.type = &symbol;
  PushScope(Scope::Kind::DerivedType, &symbol);
  if (extendsType != nullptr) {
    // Declare the "parent component"; private if the type is
    // Any symbol stored in the EXTENDS() clause is temporarily
    // hidden so that a new symbol can be created for the parent
    // component without producing spurious errors about already
    // existing.
    auto restorer{common::ScopedSet(extendsName->symbol, nullptr)};
    if (OkToAddComponent(*extendsName, extendsType)) {
      auto &comp{DeclareEntity<ObjectEntityDetails>(*extendsName, Attrs{})};
      comp.attrs().set(Attr::PRIVATE, extendsType->attrs().test(Attr::PRIVATE));
      comp.set(Symbol::Flag::ParentComp);
      DeclTypeSpec &type{currScope().MakeDerivedType(*extendsType)};
      type.derivedTypeSpec().set_scope(*extendsType->scope());
      comp.SetType(type);
      DerivedTypeDetails &details{symbol.get<DerivedTypeDetails>()};
      details.add_component(comp);
    }
  }
  EndAttrs();
}

void DeclarationVisitor::Post(const parser::TypeParamDefStmt &x) {
  auto *type{GetDeclTypeSpec()};
  auto attr{std::get<common::TypeParamAttr>(x.t)};
  for (auto &decl : std::get<std::list<parser::TypeParamDecl>>(x.t)) {
    auto &name{std::get<parser::Name>(decl.t)};
    if (Symbol * symbol{MakeTypeSymbol(name, TypeParamDetails{attr})}) {
      SetType(name, *type);
      if (auto &init{
              std::get<std::optional<parser::ScalarIntConstantExpr>>(decl.t)}) {
        if (auto maybeExpr{EvaluateConvertedExpr(
                *symbol, *init, init->thing.thing.thing.value().source)}) {
          auto *intExpr{std::get_if<SomeIntExpr>(&maybeExpr->u)};
          CHECK(intExpr != nullptr);
          symbol->get<TypeParamDetails>().set_init(std::move(*intExpr));
        }
      }
    }
  }
  EndDecl();
}
bool DeclarationVisitor::Pre(const parser::TypeAttrSpec::Extends &x) {
  derivedTypeInfo_.extends = &x.v;
  return false;
}

bool DeclarationVisitor::Pre(const parser::PrivateStmt &x) {
  if (!currScope().parent().IsModule()) {
    Say("PRIVATE is only allowed in a derived type that is"
        " in a module"_err_en_US);  // C766
  } else if (derivedTypeInfo_.sawContains) {
    derivedTypeInfo_.privateBindings = true;
  } else if (!derivedTypeInfo_.privateComps) {
    derivedTypeInfo_.privateComps = true;
  } else {
    Say("PRIVATE may not appear more than once in"
        " derived type components"_en_US);  // C738
  }
  return false;
}
bool DeclarationVisitor::Pre(const parser::SequenceStmt &x) {
  derivedTypeInfo_.sequence = true;
  return false;
}
void DeclarationVisitor::Post(const parser::ComponentDecl &x) {
  const auto &name{std::get<parser::Name>(x.t)};
  auto attrs{GetAttrs()};
  if (derivedTypeInfo_.privateComps &&
      !attrs.HasAny({Attr::PUBLIC, Attr::PRIVATE})) {
    attrs.set(Attr::PRIVATE);
  }
  if (!attrs.HasAny({Attr::POINTER, Attr::ALLOCATABLE})) {
    if (const auto *declType{GetDeclTypeSpec()}) {
      if (const auto *derived{declType->AsDerived()}) {
        if (derivedTypeInfo_.type == &derived->typeSymbol()) {  // C737
          Say("Recursive use of the derived type requires "
              "POINTER or ALLOCATABLE"_err_en_US);
        }
      }
    }
  }
  if (OkToAddComponent(name)) {
    auto &symbol{DeclareObjectEntity(name, attrs)};
    if (symbol.has<ObjectEntityDetails>()) {
      if (auto &init{std::get<std::optional<parser::Initialization>>(x.t)}) {
        Initialization(name, *init, true);
      }
    }
    currScope().symbol()->get<DerivedTypeDetails>().add_component(symbol);
  }
  ClearArraySpec();
  ClearCoarraySpec();
}
bool DeclarationVisitor::Pre(const parser::ProcedureDeclarationStmt &) {
  CHECK(!interfaceName_);
  return BeginDecl();
}
void DeclarationVisitor::Post(const parser::ProcedureDeclarationStmt &) {
  interfaceName_ = nullptr;
  EndDecl();
}
bool DeclarationVisitor::Pre(const parser::ProcComponentDefStmt &) {
  CHECK(!interfaceName_);
  return true;
}
void DeclarationVisitor::Post(const parser::ProcComponentDefStmt &) {
  interfaceName_ = nullptr;
}
bool DeclarationVisitor::Pre(const parser::ProcPointerInit &x) {
  if (auto *name{std::get_if<parser::Name>(&x.u)}) {
    return !NameIsKnownOrIntrinsic(*name);
  }
  return true;
}
void DeclarationVisitor::Post(const parser::ProcInterface &x) {
  if (auto *name{std::get_if<parser::Name>(&x.u)}) {
    interfaceName_ = name;
    NoteInterfaceName(*name);
  }
}

void DeclarationVisitor::Post(const parser::ProcDecl &x) {
  const auto &name{std::get<parser::Name>(x.t)};
  ProcInterface interface;
  if (interfaceName_) {
    interface.set_symbol(*interfaceName_->symbol);
  } else if (auto *type{GetDeclTypeSpec()}) {
    interface.set_type(*type);
  }
  auto attrs{HandleSaveName(name.source, GetAttrs())};
  DerivedTypeDetails *dtDetails{nullptr};
  if (Symbol * symbol{currScope().symbol()}) {
    dtDetails = symbol->detailsIf<DerivedTypeDetails>();
  }
  if (dtDetails == nullptr) {
    attrs.set(Attr::EXTERNAL);
  }
  Symbol &symbol{DeclareProcEntity(name, attrs, interface)};
  if (dtDetails != nullptr) {
    dtDetails->add_component(symbol);
  }
}

bool DeclarationVisitor::Pre(const parser::TypeBoundProcedurePart &x) {
  derivedTypeInfo_.sawContains = true;
  return true;
}

// Resolve binding names from type-bound generics, saved in genericBindings_.
void DeclarationVisitor::Post(const parser::TypeBoundProcedurePart &) {
  // track specifics seen for the current generic to detect duplicates:
  const Symbol *currGeneric{nullptr};
  std::set<SourceName> specifics;
  for (const auto [generic, bindingName] : genericBindings_) {
    if (generic != currGeneric) {
      currGeneric = generic;
      specifics.clear();
    }
    auto [it, inserted]{specifics.insert(bindingName->source)};
    if (!inserted) {
      Say(*bindingName,  // C773
          "Binding name '%s' was already specified for generic '%s'"_err_en_US,
          bindingName->source, generic->name())
          .Attach(*it, "Previous specification of '%s'"_en_US, *it);
      continue;
    }
    auto *symbol{FindInTypeOrParents(*bindingName)};
    if (!symbol) {
      Say(*bindingName,  // C772
          "Binding name '%s' not found in this derived type"_err_en_US);
    } else if (!symbol->has<ProcBindingDetails>()) {
      SayWithDecl(*bindingName, *symbol,  // C772
          "'%s' is not the name of a specific binding of this type"_err_en_US);
    } else {
      auto &details{generic->get<GenericBindingDetails>()};
      details.add_specificProc(*symbol);
    }
  }
  genericBindings_.clear();
}

void DeclarationVisitor::Post(const parser::ContainsStmt &) {
  if (derivedTypeInfo_.sequence) {
    Say("A sequence type may not have a CONTAINS statement"_err_en_US);  // C740
  }
}

void DeclarationVisitor::Post(
    const parser::TypeBoundProcedureStmt::WithoutInterface &x) {
  if (GetAttrs().test(Attr::DEFERRED)) {  // C783
    Say("DEFERRED is only allowed when an interface-name is provided"_err_en_US);
  }
  for (auto &declaration : x.declarations) {
    auto &bindingName{std::get<parser::Name>(declaration.t)};
    auto &optName{std::get<std::optional<parser::Name>>(declaration.t)};
    auto &procedureName{optName ? *optName : bindingName};
    auto *procedure{FindSymbol(procedureName)};
    if (!procedure) {
      Say(procedureName, "Procedure '%s' not found"_err_en_US);
      continue;
    }
    procedure = &procedure->GetUltimate();  // may come from USE
    if (!CanBeTypeBoundProc(*procedure)) {
      SayWithDecl(procedureName, *procedure,
          "'%s' is not a module procedure or external procedure"
          " with explicit interface"_err_en_US);
      continue;
    }
    if (auto *s{MakeTypeSymbol(bindingName, ProcBindingDetails{*procedure})}) {
      SetPassNameOn(*s);
    }
  }
}

void DeclarationVisitor::Post(
    const parser::TypeBoundProcedureStmt::WithInterface &x) {
  if (!GetAttrs().test(Attr::DEFERRED)) {  // C783
    Say("DEFERRED is required when an interface-name is provided"_err_en_US);
  }
  if (Symbol * interface{NoteInterfaceName(x.interfaceName)}) {
    for (auto &bindingName : x.bindingNames) {
      if (auto *s{
              MakeTypeSymbol(bindingName, ProcBindingDetails{*interface})}) {
        SetPassNameOn(*s);
      }
    }
  }
}

void DeclarationVisitor::Post(const parser::FinalProcedureStmt &x) {
  for (auto &name : x.v) {
    MakeTypeSymbol(name, FinalProcDetails{});
  }
}

bool DeclarationVisitor::Pre(const parser::TypeBoundGenericStmt &x) {
  const auto &accessSpec{std::get<std::optional<parser::AccessSpec>>(x.t)};
  const auto &genericSpec{std::get<Indirection<parser::GenericSpec>>(x.t)};
  const auto &bindingNames{std::get<std::list<parser::Name>>(x.t)};
  auto info{GenericSpecInfo{genericSpec.value()}};
  const SourceName &symbolName{info.symbolName()};
  bool isPrivate{accessSpec ? accessSpec->v == parser::AccessSpec::Kind::Private
                            : derivedTypeInfo_.privateBindings};
  auto *genericSymbol{FindInScope(currScope(), symbolName)};
  if (genericSymbol) {
    if (!genericSymbol->has<GenericBindingDetails>()) {
      genericSymbol = nullptr;  // MakeTypeSymbol will report the error below
    }
  } else if (auto *inheritedSymbol{
                 FindInTypeOrParents(currScope(), symbolName)}) {
    // look in parent types:
    if (inheritedSymbol->has<GenericBindingDetails>()) {
      CheckAccessibility(symbolName, isPrivate, *inheritedSymbol);  // C771
    }
  }
  if (genericSymbol) {
    CheckAccessibility(symbolName, isPrivate, *genericSymbol);  // C771
  } else {
    genericSymbol = MakeTypeSymbol(symbolName, GenericBindingDetails{});
    if (!genericSymbol) {
      return false;
    }
    if (isPrivate) {
      genericSymbol->attrs().set(Attr::PRIVATE);
    }
  }
  for (const parser::Name &bindingName : bindingNames) {
    genericBindings_.emplace(genericSymbol, &bindingName);
  }
  info.Resolve(genericSymbol);
  return false;
}

bool DeclarationVisitor::Pre(const parser::AllocateStmt &) {
  BeginDeclTypeSpec();
  return true;
}
void DeclarationVisitor::Post(const parser::AllocateStmt &) {
  EndDeclTypeSpec();
}

bool DeclarationVisitor::Pre(const parser::StructureConstructor &x) {
  auto &parsedType{std::get<parser::DerivedTypeSpec>(x.t)};
  const DeclTypeSpec *type{ProcessTypeSpec(parsedType)};
  if (type == nullptr) {
    return false;
  }
  const DerivedTypeSpec *spec{type->AsDerived()};
  const Scope *typeScope{spec ? spec->scope() : nullptr};
  if (typeScope == nullptr) {
    return false;
  }

  // N.B C7102 is implicitly enforced by having inaccessible types not
  // being found in resolution.
  // More constraints are enforced in expression.cc so that they
  // can apply to structure constructors that have been converted
  // from misparsed function references.
  for (const auto &component :
      std::get<std::list<parser::ComponentSpec>>(x.t)) {
    // Visit the component spec expression, but not the keyword, since
    // we need to resolve its symbol in the scope of the derived type.
    Walk(std::get<parser::ComponentDataSource>(component.t));
    if (const auto &kw{std::get<std::optional<parser::Keyword>>(component.t)}) {
      if (Symbol * symbol{FindInTypeOrParents(*typeScope, kw->v)}) {
        if (kw->v.symbol == nullptr) {
          kw->v.symbol = symbol;
        }
        CheckAccessibleComponent(kw->v.source, *symbol);
      }
    }
  }
  return false;
}

bool DeclarationVisitor::Pre(const parser::NamelistStmt::Group &x) {
  if (!CheckNotInBlock("NAMELIST")) {
    return false;
  }

  NamelistDetails details;
  for (const auto &name : std::get<std::list<parser::Name>>(x.t)) {
    auto *symbol{FindSymbol(name)};
    if (!symbol) {
      symbol = &MakeSymbol(name, ObjectEntityDetails{});
      ApplyImplicitRules(*symbol);
    } else if (!ConvertToObjectEntity(*symbol)) {
      SayWithDecl(name, *symbol, "'%s' is not a variable"_err_en_US);
    }
    details.add_object(*symbol);
  }

  const auto &groupName{std::get<parser::Name>(x.t)};
  auto *groupSymbol{FindInScope(currScope(), groupName)};
  if (!groupSymbol) {
    groupSymbol = &MakeSymbol(groupName, std::move(details));
  } else if (groupSymbol->has<NamelistDetails>()) {
    groupSymbol->get<NamelistDetails>().add_objects(details.objects());
  } else {
    SayAlreadyDeclared(groupName, *groupSymbol);
  }
  return false;
}

bool DeclarationVisitor::Pre(const parser::IoControlSpec &x) {
  if (const auto *name{std::get_if<parser::Name>(&x.u)}) {
    auto *symbol{FindSymbol(*name)};
    if (!symbol) {
      Say(*name, "Namelist group '%s' not found"_err_en_US);
    } else if (!symbol->GetUltimate().has<NamelistDetails>()) {
      SayWithDecl(
          *name, *symbol, "'%s' is not the name of a namelist group"_err_en_US);
    }
  }
  return true;
}

bool DeclarationVisitor::Pre(const parser::CommonStmt::Block &x) {
  CheckNotInBlock("COMMON");
  const auto &optName{std::get<std::optional<parser::Name>>(x.t)};
  parser::Name blankCommon;
  blankCommon.source = SourceName{currStmtSource()->begin(), std::size_t{0}};
  CHECK(!commonBlockInfo_.curr);
  commonBlockInfo_.curr =
      &MakeCommonBlockSymbol(optName ? *optName : blankCommon);
  return true;
}

void DeclarationVisitor::Post(const parser::CommonStmt::Block &) {
  commonBlockInfo_.curr = nullptr;
}

bool DeclarationVisitor::Pre(const parser::CommonBlockObject &x) {
  BeginArraySpec();
  return true;
}

void DeclarationVisitor::Post(const parser::CommonBlockObject &x) {
  CHECK(commonBlockInfo_.curr);
  const auto &name{std::get<parser::Name>(x.t)};
  auto &symbol{DeclareObjectEntity(name, Attrs{})};
  ClearArraySpec();
  ClearCoarraySpec();
  auto *details{symbol.detailsIf<ObjectEntityDetails>()};
  if (!details) {
    return;  // error was reported
  }
  commonBlockInfo_.curr->get<CommonBlockDetails>().add_object(symbol);
  if (!IsAllocatableOrPointer(symbol) && !IsExplicit(details->shape())) {
    Say(name,
        "The shape of common block object '%s' must be explicit"_err_en_US);
    return;
  }
  auto pair{commonBlockInfo_.names.insert(name.source)};
  if (!pair.second) {
    const SourceName &prev{*pair.first};
    Say2(name.source, "'%s' is already in a COMMON block"_err_en_US, prev,
        "Previous occurrence of '%s' in a COMMON block"_en_US);
    return;
  }
  details->set_commonBlock(*commonBlockInfo_.curr);
}

bool DeclarationVisitor::Pre(const parser::EquivalenceStmt &x) {
  // save equivalence sets to be processed after specification part
  for (const std::list<parser::EquivalenceObject> &set : x.v) {
    equivalenceSets_.push_back(&set);
  }
  return false;  // don't implicitly declare names yet
}

void DeclarationVisitor::CheckEquivalenceSets() {
  EquivalenceSets equivSets{context()};
  for (const auto *set : equivalenceSets_) {
    const auto &source{set->front().v.value().source};
    if (set->size() <= 1) {  // R871
      Say(source, "Equivalence set must have more than one object"_err_en_US);
    }
    for (const parser::EquivalenceObject &object : *set) {
      const auto &designator{object.v.value()};
      // The designator was not resolved when it was encountered so do it now.
      // AnalyzeExpr causes array sections to be changed to substrings as needed
      Walk(designator);
      if (AnalyzeExpr(context(), designator)) {
        equivSets.AddToSet(designator);
      }
    }
    equivSets.FinishSet(source);
  }
  for (auto &set : equivSets.sets()) {
    if (!set.empty()) {
      currScope().add_equivalenceSet(std::move(set));
    }
  }
  equivalenceSets_.clear();
}

bool DeclarationVisitor::Pre(const parser::SaveStmt &x) {
  if (x.v.empty()) {
    saveInfo_.saveAll = currStmtSource();
  } else {
    for (const parser::SavedEntity &y : x.v) {
      auto kind{std::get<parser::SavedEntity::Kind>(y.t)};
      const auto &name{std::get<parser::Name>(y.t)};
      if (kind == parser::SavedEntity::Kind::Common) {
        MakeCommonBlockSymbol(name);
        AddSaveName(saveInfo_.commons, name.source);
      } else {
        HandleAttributeStmt(Attr::SAVE, name);
      }
    }
  }
  return false;
}

void DeclarationVisitor::CheckSaveStmts() {
  for (const SourceName &name : saveInfo_.entities) {
    auto *symbol{FindInScope(currScope(), name)};
    if (!symbol) {
      // error was reported
    } else if (saveInfo_.saveAll) {
      // C889 - note that pgi, ifort, xlf do not enforce this constraint
      Say2(name,
          "Explicit SAVE of '%s' is redundant due to global SAVE statement"_err_en_US,
          *saveInfo_.saveAll, "Global SAVE statement"_en_US);
    } else if (auto msg{CheckSaveAttr(*symbol)}) {
      Say(name, std::move(*msg));
    } else {
      SetSaveAttr(*symbol);
    }
  }
  for (const SourceName &name : saveInfo_.commons) {
    if (auto *symbol{currScope().FindCommonBlock(name)}) {
      auto &objects{symbol->get<CommonBlockDetails>().objects()};
      if (objects.empty()) {
        Say(name,
            "'%s' appears as a COMMON block in a SAVE statement but not in"
            " a COMMON statement"_err_en_US);
      } else {
        for (Symbol *object : symbol->get<CommonBlockDetails>().objects()) {
          SetSaveAttr(*object);
        }
      }
    }
  }
  if (saveInfo_.saveAll) {
    // Apply SAVE attribute to applicable symbols
    for (auto pair : currScope()) {
      auto &symbol{*pair.second};
      if (!CheckSaveAttr(symbol)) {
        SetSaveAttr(symbol);
      }
    }
  }
  saveInfo_ = {};
}

// If SAVE attribute can't be set on symbol, return error message.
std::optional<MessageFixedText> DeclarationVisitor::CheckSaveAttr(
    const Symbol &symbol) {
  if (symbol.IsDummy()) {
    return "SAVE attribute may not be applied to dummy argument '%s'"_err_en_US;
  } else if (symbol.IsFuncResult()) {
    return "SAVE attribute may not be applied to function result '%s'"_err_en_US;
  } else if (symbol.has<ProcEntityDetails>() &&
      !symbol.attrs().test(Attr::POINTER)) {
    return "Procedure '%s' with SAVE attribute must also have POINTER attribute"_err_en_US;
  } else {
    return std::nullopt;
  }
}

// Instead of setting SAVE attribute, record the name in saveInfo_.entities.
Attrs DeclarationVisitor::HandleSaveName(const SourceName &name, Attrs attrs) {
  if (attrs.test(Attr::SAVE)) {
    attrs.set(Attr::SAVE, false);
    AddSaveName(saveInfo_.entities, name);
  }
  return attrs;
}

// Record a name in a set of those to be saved.
void DeclarationVisitor::AddSaveName(
    std::set<SourceName> &set, const SourceName &name) {
  auto pair{set.insert(name)};
  if (!pair.second) {
    Say2(name, "SAVE attribute was already specified on '%s'"_err_en_US,
        *pair.first, "Previous specification of SAVE attribute"_en_US);
  }
}

// Set the SAVE attribute on symbol unless it is implicitly saved anyway.
void DeclarationVisitor::SetSaveAttr(Symbol &symbol) {
  auto scopeKind{symbol.owner().kind()};
  if (scopeKind == Scope::Kind::MainProgram ||
      scopeKind == Scope::Kind::Module) {
    return;
  }
  if (const auto *details{symbol.detailsIf<ObjectEntityDetails>()}) {
    if (details->init()) {
      return;
    }
  }
  symbol.attrs().set(Attr::SAVE);
}

// Check types of common block objects, now that they are known.
void DeclarationVisitor::CheckCommonBlocks() {
  // check for empty common blocks
  for (const auto pair : currScope().commonBlocks()) {
    const auto &symbol{*pair.second};
    if (symbol.get<CommonBlockDetails>().objects().empty() &&
        symbol.attrs().test(Attr::BIND_C)) {
      Say(symbol.name(),
          "'%s' appears as a COMMON block in a BIND statement but not in"
          " a COMMON statement"_err_en_US);
    }
  }
  // check objects in common blocks
  for (const auto &name : commonBlockInfo_.names) {
    const auto *symbol{currScope().FindSymbol(name)};
    if (symbol == nullptr) {
      continue;
    }
    const auto &attrs{symbol->attrs()};
    if (attrs.test(Attr::ALLOCATABLE)) {
      Say(name,
          "ALLOCATABLE object '%s' may not appear in a COMMON block"_err_en_US);
    } else if (attrs.test(Attr::BIND_C)) {
      Say(name,
          "Variable '%s' with BIND attribute may not appear in a COMMON block"_err_en_US);
    } else if (symbol->IsDummy()) {
      Say(name,
          "Dummy argument '%s' may not appear in a COMMON block"_err_en_US);
    } else if (symbol->IsFuncResult()) {
      Say(name,
          "Function result '%s' may not appear in a COMMON block"_err_en_US);
    } else if (const DeclTypeSpec * type{symbol->GetType()}) {
      if (type->category() == DeclTypeSpec::ClassStar) {
        Say(name,
            "Unlimited polymorphic pointer '%s' may not appear in a COMMON block"_err_en_US);
      } else if (const auto *derived{type->AsDerived()}) {
        auto &typeSymbol{derived->typeSymbol()};
        if (!typeSymbol.attrs().test(Attr::BIND_C) &&
            !typeSymbol.get<DerivedTypeDetails>().sequence()) {
          Say(name,
              "Derived type '%s' in COMMON block must have the BIND or"
              " SEQUENCE attribute"_err_en_US);
        }
        CheckCommonBlockDerivedType(name, typeSymbol);
      }
    }
  }
  commonBlockInfo_ = {};
}

Symbol &DeclarationVisitor::MakeCommonBlockSymbol(const parser::Name &name) {
  return Resolve(name, currScope().MakeCommonBlock(name.source));
}

bool DeclarationVisitor::NameIsKnownOrIntrinsic(const parser::Name &name) {
  if (Symbol * symbol{FindSymbol(name)}) {
    Resolve(name, *symbol);
    return true;
  } else {
    return HandleUnrestrictedSpecificIntrinsicFunction(name);
  }
}

// Check if this derived type can be in a COMMON block.
void DeclarationVisitor::CheckCommonBlockDerivedType(
    const SourceName &name, const Symbol &typeSymbol) {
  if (const auto *scope{typeSymbol.scope()}) {
    for (const auto &pair : *scope) {
      const Symbol &component{*pair.second};
      if (component.attrs().test(Attr::ALLOCATABLE)) {
        Say2(name,
            "Derived type variable '%s' may not appear in a COMMON block"
            " due to ALLOCATABLE component"_err_en_US,
            component.name(), "Component with ALLOCATABLE attribute"_en_US);
        return;
      }
      if (const auto *details{component.detailsIf<ObjectEntityDetails>()}) {
        if (details->init()) {
          Say2(name,
              "Derived type variable '%s' may not appear in a COMMON block"
              " due to component with default initialization"_err_en_US,
              component.name(), "Component with default initialization"_en_US);
          return;
        }
        if (const auto *type{details->type()}) {
          if (const auto *derived{type->AsDerived()}) {
            CheckCommonBlockDerivedType(name, derived->typeSymbol());
          }
        }
      }
    }
  }
}

bool DeclarationVisitor::HandleUnrestrictedSpecificIntrinsicFunction(
    const parser::Name &name) {
  if (context()
          .intrinsics()
          .IsUnrestrictedSpecificIntrinsicFunction(name.source.ToString())
          .has_value()) {
    // Unrestricted specific intrinsic function names (e.g., "cos")
    // are acceptable as procedure interfaces.
    Symbol &symbol{
        MakeSymbol(InclusiveScope(), name.source, Attrs{Attr::INTRINSIC})};
    symbol.set_details(ProcEntityDetails{});
    Resolve(name, symbol);
    return true;
  } else {
    return false;
  }
}

// Checks for all locality-specs: LOCAL, LOCAL_INIT, and SHARED
bool DeclarationVisitor::PassesSharedLocalityChecks(
    const parser::Name &name, Symbol &symbol) {
  if (!IsVariableName(symbol)) {
    SayLocalMustBeVariable(name, symbol);  // C1124
    return false;
  }
  if (symbol.owner() == currScope()) {  // C1125 and C1126
    SayAlreadyDeclared(name, symbol);
    return false;
  }
  return true;
}

// Checks for locality-specs LOCAL and LOCAL_INIT
bool DeclarationVisitor::PassesLocalityChecks(
    const parser::Name &name, Symbol &symbol) {
  if (IsAllocatable(symbol)) {  // C1128
    SayWithDecl(name, symbol,
        "ALLOCATABLE variable '%s' not allowed in a locality-spec"_err_en_US);
    return false;
  }
  if (IsOptional(symbol)) {  // C1128
    SayWithDecl(name, symbol,
        "OPTIONAL argument '%s' not allowed in a locality-spec"_err_en_US);
    return false;
  }
  if (IsIntentIn(symbol)) {  // C1128
    SayWithDecl(name, symbol,
        "INTENT IN argument '%s' not allowed in a locality-spec"_err_en_US);
    return false;
  }
  if (IsFinalizable(symbol)) {  // C1128
    SayWithDecl(name, symbol,
        "Finalizable variable '%s' not allowed in a locality-spec"_err_en_US);
    return false;
  }
  if (IsCoarray(symbol)) {  // C1128
    SayWithDecl(
        name, symbol, "Coarray '%s' not allowed in a locality-spec"_err_en_US);
    return false;
  }
  if (const DeclTypeSpec * type{symbol.GetType()}) {
    if (type->IsPolymorphic() && symbol.IsDummy() &&
        !IsPointer(symbol)) {  // C1128
      SayWithDecl(name, symbol,
          "Nonpointer polymorphic argument '%s' not allowed in a "
          "locality-spec"_err_en_US);
      return false;
    }
  }
  if (IsAssumedSizeArray(symbol)) {  // C1128
    SayWithDecl(name, symbol,
        "Assumed size array '%s' not allowed in a locality-spec"_err_en_US);
    return false;
  }
  if (std::optional<MessageFixedText> msg{
          WhyNotModifiable(symbol, currScope())}) {
    SayWithReason(name, symbol,
        "'%s' may not appear in a locality-spec because it is not "
        "definable"_err_en_US,
        std::move(*msg));
    return false;
  }
  return PassesSharedLocalityChecks(name, symbol);
}

Symbol &DeclarationVisitor::FindOrDeclareEnclosingEntity(
    const parser::Name &name) {
  Symbol *prev{FindSymbol(name)};
  if (prev == nullptr) {
    // Declare the name as an object in the enclosing scope so that
    // the name can't be repurposed there later as something else.
    prev = &MakeSymbol(InclusiveScope(), name.source, Attrs{});
    ConvertToObjectEntity(*prev);
    ApplyImplicitRules(*prev);
  }
  return *prev;
}

Symbol *DeclarationVisitor::DeclareLocalEntity(const parser::Name &name) {
  Symbol &prev{FindOrDeclareEnclosingEntity(name)};
  if (!PassesLocalityChecks(name, prev)) {
    return nullptr;
  }
  name.symbol = nullptr;
  Symbol &symbol{DeclareEntity<ObjectEntityDetails>(name, {})};
  if (auto *type{prev.GetType()}) {
    symbol.SetType(*type);
    symbol.set(Symbol::Flag::Implicit, prev.test(Symbol::Flag::Implicit));
  }
  return &symbol;
}

Symbol *DeclarationVisitor::DeclareStatementEntity(const parser::Name &name,
    const std::optional<parser::IntegerTypeSpec> &type) {
  const DeclTypeSpec *declTypeSpec{nullptr};
  if (auto *prev{FindSymbol(name)}) {
    if (prev->owner() == currScope()) {
      SayAlreadyDeclared(name, *prev);
      return nullptr;
    }
    name.symbol = nullptr;
    declTypeSpec = prev->GetType();
  }
  Symbol &symbol{DeclareEntity<ObjectEntityDetails>(name, {})};
  if (!symbol.has<ObjectEntityDetails>()) {
    return nullptr;  // error was reported in DeclareEntity
  }
  if (type.has_value()) {
    declTypeSpec = ProcessTypeSpec(*type);
  }
  if (declTypeSpec != nullptr) {
    // Subtlety: Don't let a "*length" specifier (if any is pending) affect the
    // declaration of this implied DO loop control variable.
    auto save{common::ScopedSet(charInfo_.length, std::optional<ParamValue>{})};
    SetType(name, *declTypeSpec);
  } else {
    ApplyImplicitRules(symbol);
  }
  return Resolve(name, &symbol);
}

// Set the type of an entity or report an error.
void DeclarationVisitor::SetType(
    const parser::Name &name, const DeclTypeSpec &type) {
  CHECK(name.symbol);
  auto &symbol{*name.symbol};
  if (charInfo_.length.has_value()) {  // Declaration has "*length" (R723)
    auto length{std::move(*charInfo_.length)};
    charInfo_.length.reset();
    if (type.category() == DeclTypeSpec::Character) {
      auto kind{type.characterTypeSpec().kind()};
      // Recurse with correct type.
      SetType(name,
          currScope().MakeCharacterType(std::move(length), std::move(kind)));
      return;
    } else {
      Say(name,
          "A length specifier cannot be used to declare the non-character entity '%s'"_err_en_US);
    }
  }
  auto *prevType{symbol.GetType()};
  if (!prevType) {
    symbol.SetType(type);
  } else if (symbol.has<UseDetails>()) {
    // error recovery case, redeclaration of use-associated name
  } else if (!symbol.test(Symbol::Flag::Implicit)) {
    SayWithDecl(
        name, symbol, "The type of '%s' has already been declared"_err_en_US);
  } else if (type != *prevType) {
    SayWithDecl(name, symbol,
        "The type of '%s' has already been implicitly declared"_err_en_US);
  } else {
    symbol.set(Symbol::Flag::Implicit, false);
  }
}

// Find the Symbol for this derived type.
const Symbol *DeclarationVisitor::ResolveDerivedType(const parser::Name &name) {
  const Symbol *symbol{FindSymbol(name)};
  if (!symbol) {
    Say(name, "Derived type '%s' not found"_err_en_US);
    return nullptr;
  }
  if (CheckUseError(name)) {
    return nullptr;
  }
  symbol = &symbol->GetUltimate();
  if (auto *details{symbol->detailsIf<GenericDetails>()}) {
    if (details->derivedType()) {
      symbol = details->derivedType();
    }
  }
  if (!symbol->has<DerivedTypeDetails>()) {
    Say(name, "'%s' is not a derived type"_err_en_US);
    return nullptr;
  }
  return symbol;
}

// Check this symbol suitable as a type-bound procedure - C769
bool DeclarationVisitor::CanBeTypeBoundProc(const Symbol &symbol) {
  if (symbol.has<SubprogramNameDetails>()) {
    return symbol.owner().kind() == Scope::Kind::Module;
  } else if (auto *details{symbol.detailsIf<SubprogramDetails>()}) {
    return symbol.owner().kind() == Scope::Kind::Module ||
        details->isInterface();
  } else {
    return false;
  }
}

Symbol *DeclarationVisitor::NoteInterfaceName(const parser::Name &name) {
  // The symbol is checked later by CheckExplicitInterface() to ensure
  // that it defines an explicit interface.  The name can be a forward
  // reference.
  if (!NameIsKnownOrIntrinsic(name)) {
    Resolve(name, MakeSymbol(InclusiveScope(), name.source, Attrs{}));
  }
  return name.symbol;
}

void DeclarationVisitor::CheckExplicitInterface(Symbol &symbol) {
  if (const Symbol * interface{FindInterface(symbol)}) {
    const Symbol *subp{FindSubprogram(*interface)};
    if (subp == nullptr || !subp->HasExplicitInterface()) {
      Say(symbol.name(),
          "The interface of '%s' ('%s') is not an abstract interface or a "
          "procedure with an explicit interface"_err_en_US,
          symbol.name(), interface->name());
      context().SetError(symbol);
    }
  }
}

// Create a symbol for a type parameter, component, or procedure binding in
// the current derived type scope. Return false on error.
Symbol *DeclarationVisitor::MakeTypeSymbol(
    const parser::Name &name, Details &&details) {
  return Resolve(name, MakeTypeSymbol(name.source, std::move(details)));
}
Symbol *DeclarationVisitor::MakeTypeSymbol(
    const SourceName &name, Details &&details) {
  Scope &derivedType{currScope()};
  CHECK(derivedType.IsDerivedType());
  if (auto *symbol{FindInScope(derivedType, name)}) {
    Say2(name,
        "Type parameter, component, or procedure binding '%s'"
        " already defined in this type"_err_en_US,
        *symbol, "Previous definition of '%s'"_en_US);
    return nullptr;
  } else {
    auto attrs{GetAttrs()};
    // Apply binding-private-stmt if present and this is a procedure binding
    if (derivedTypeInfo_.privateBindings &&
        !attrs.HasAny({Attr::PUBLIC, Attr::PRIVATE}) &&
        std::holds_alternative<ProcBindingDetails>(details)) {
      attrs.set(Attr::PRIVATE);
    }
    Symbol &result{MakeSymbol(name, attrs, std::move(details))};
    if (result.has<TypeParamDetails>()) {
      derivedType.symbol()->get<DerivedTypeDetails>().add_paramDecl(result);
    }
    return &result;
  }
}

// Return true if it is ok to declare this component in the current scope.
// Otherwise, emit an error and return false.
bool DeclarationVisitor::OkToAddComponent(
    const parser::Name &name, const Symbol *extends) {
  for (const Scope *scope{&currScope()}; scope != nullptr;) {
    CHECK(scope->IsDerivedType());
    if (auto *prev{FindInScope(*scope, name)}) {
      auto msg{""_en_US};
      if (extends != nullptr) {
        msg = "Type cannot be extended as it has a component named"
              " '%s'"_err_en_US;
      } else if (prev->test(Symbol::Flag::ParentComp)) {
        msg = "'%s' is a parent type of this type and so cannot be"
              " a component"_err_en_US;
      } else if (scope != &currScope()) {
        msg = "Component '%s' is already declared in a parent of this"
              " derived type"_err_en_US;
      } else {
        msg = "Component '%s' is already declared in this"
              " derived type"_err_en_US;
      }
      Say2(name, std::move(msg), *prev, "Previous declaration of '%s'"_en_US);
      return false;
    }
    if (scope == &currScope() && extends != nullptr) {
      // The parent component has not yet been added to the scope.
      scope = extends->scope();
    } else {
      scope = scope->GetDerivedTypeParent();
    }
  }
  return true;
}

ParamValue DeclarationVisitor::GetParamValue(const parser::TypeParamValue &x) {
  return std::visit(
      common::visitors{
          [=](const parser::ScalarIntExpr &x) {
            return ParamValue{EvaluateIntExpr(x)};
          },
          [](const parser::Star &) { return ParamValue::Assumed(); },
          [](const parser::TypeParamValue::Deferred &) {
            return ParamValue::Deferred();
          },
      },
      x.u);
}

// ConstructVisitor implementation

void ConstructVisitor::ResolveIndexName(
    const parser::ConcurrentControl &control) {
  const parser::Name &name{std::get<parser::Name>(control.t)};
  auto *prev{FindSymbol(name)};
  if (prev) {
    if (prev->owner().kind() == Scope::Kind::Forall ||
        prev->owner() == currScope()) {
      SayAlreadyDeclared(name, *prev);
      return;
    }
    name.symbol = nullptr;
  }
  auto &symbol{DeclareObjectEntity(name, {})};

  if (symbol.GetType()) {
    // type came from explicit type-spec
  } else if (!prev) {
    ApplyImplicitRules(symbol);
  } else if (!prev->has<ObjectEntityDetails>() && !prev->has<EntityDetails>()) {
    Say2(name, "Index name '%s' conflicts with existing identifier"_err_en_US,
        *prev, "Previous declaration of '%s'"_en_US);
    return;
  } else {
    if (const auto *type{prev->GetType()}) {
      symbol.SetType(*type);
    }
    if (prev->IsObjectArray()) {
      SayWithDecl(name, *prev, "Index variable '%s' is not scalar"_err_en_US);
      return;
    }
  }
  EvaluateExpr(parser::Scalar{parser::Integer{common::Clone(name)}});
}

void ConstructVisitor::ResolveControlExpressions(
    const parser::ConcurrentControl &control) {
  Walk(std::get<1>(control.t));  // Initial expression
  Walk(std::get<2>(control.t));  // Final expression
  Walk(std::get<3>(control.t));  // Step expression
}

// We need to make sure that all of the index-names get declared before the
// expressions in the loop control are evaluated so that references to the
// index-names in the expressions are correctly detected.
bool ConstructVisitor::Pre(const parser::ConcurrentHeader &header) {
  BeginDeclTypeSpec();

  // Process the type spec, if present
  const auto &typeSpec{
      std::get<std::optional<parser::IntegerTypeSpec>>(header.t)};
  if (typeSpec.has_value()) {
    SetDeclTypeSpec(MakeNumericType(TypeCategory::Integer, typeSpec->v));
  }

  // Process the index-name nodes in the ConcurrentControl nodes
  const auto &controls{
      std::get<std::list<parser::ConcurrentControl>>(header.t)};
  for (const auto &control : controls) {
    ResolveIndexName(control);
  }

  // Process the expressions in ConcurrentControls
  for (const auto &control : controls) {
    ResolveControlExpressions(control);
  }

  // Resolve the names in the scalar-mask-expr
  Walk(std::get<std::optional<parser::ScalarLogicalExpr>>(header.t));

  EndDeclTypeSpec();
  return false;
}

bool ConstructVisitor::Pre(const parser::LocalitySpec::Local &x) {
  for (auto &name : x.v) {
    if (auto *symbol{DeclareLocalEntity(name)}) {
      symbol->set(Symbol::Flag::LocalityLocal);
    }
  }
  return false;
}

bool ConstructVisitor::Pre(const parser::LocalitySpec::LocalInit &x) {
  for (auto &name : x.v) {
    if (auto *symbol{DeclareLocalEntity(name)}) {
      symbol->set(Symbol::Flag::LocalityLocalInit);
    }
  }
  return false;
}

bool ConstructVisitor::Pre(const parser::LocalitySpec::Shared &x) {
  for (const auto &name : x.v) {
    if (!FindSymbol(name)) {
      Say(name, "Variable '%s' with SHARED locality implicitly declared"_en_US);
    }
    Symbol &prev{FindOrDeclareEnclosingEntity(name)};
    if (PassesSharedLocalityChecks(name, prev)) {
      auto &symbol{MakeSymbol(name, HostAssocDetails{prev})};
      symbol.set(Symbol::Flag::LocalityShared);
      name.symbol = &symbol;  // override resolution to parent
    }
  }
  return false;
}

bool ConstructVisitor::Pre(const parser::AcSpec &x) {
  ProcessTypeSpec(x.type);
  PushScope(Scope::Kind::ImpliedDos, nullptr);
  Walk(x.values);
  PopScope();
  return false;
}

bool ConstructVisitor::Pre(const parser::AcImpliedDo &x) {
  auto &values{std::get<std::list<parser::AcValue>>(x.t)};
  auto &control{std::get<parser::AcImpliedDoControl>(x.t)};
  auto &type{std::get<std::optional<parser::IntegerTypeSpec>>(control.t)};
  auto &bounds{std::get<parser::AcImpliedDoControl::Bounds>(control.t)};
  DeclareStatementEntity(bounds.name.thing.thing, type);
  Walk(bounds);
  Walk(values);
  return false;
}

bool ConstructVisitor::Pre(const parser::DataImpliedDo &x) {
  auto &objects{std::get<std::list<parser::DataIDoObject>>(x.t)};
  auto &type{std::get<std::optional<parser::IntegerTypeSpec>>(x.t)};
  auto &bounds{std::get<parser::DataImpliedDo::Bounds>(x.t)};
  DeclareStatementEntity(bounds.name.thing.thing, type);
  Walk(bounds);
  Walk(objects);
  return false;
}

bool ConstructVisitor::Pre(const parser::DataStmtObject &x) {
  std::visit(
      common::visitors{
          [&](const common::Indirection<parser::Variable> &y) {
            Walk(y.value());
          },
          [&](const parser::DataImpliedDo &y) {
            PushScope(Scope::Kind::ImpliedDos, nullptr);
            Walk(y);
            PopScope();
          },
      },
      x.u);
  return false;
}

bool ConstructVisitor::Pre(const parser::DoConstruct &x) {
  if (x.IsDoConcurrent()) {
    PushScope(Scope::Kind::Block, nullptr);
  }
  return true;
}
void ConstructVisitor::Post(const parser::DoConstruct &x) {
  if (x.IsDoConcurrent()) {
    PopScope();
  }
}

bool ConstructVisitor::Pre(const parser::ForallConstruct &) {
  PushScope(Scope::Kind::Forall, nullptr);
  return true;
}
void ConstructVisitor::Post(const parser::ForallConstruct &) { PopScope(); }
bool ConstructVisitor::Pre(const parser::ForallStmt &) {
  PushScope(Scope::Kind::Forall, nullptr);
  return true;
}
void ConstructVisitor::Post(const parser::ForallStmt &) { PopScope(); }

bool ConstructVisitor::Pre(const parser::BlockStmt &x) {
  CheckDef(x.v);
  PushScope(Scope::Kind::Block, nullptr);
  return false;
}
bool ConstructVisitor::Pre(const parser::EndBlockStmt &x) {
  PopScope();
  CheckRef(x.v);
  return false;
}

void ConstructVisitor::Post(const parser::Selector &x) {
  GetCurrentAssociation().selector = ResolveSelector(x);
}

bool ConstructVisitor::Pre(const parser::AssociateStmt &x) {
  CheckDef(x.t);
  PushScope(Scope::Kind::Block, nullptr);
  PushAssociation();
  return true;
}
void ConstructVisitor::Post(const parser::EndAssociateStmt &x) {
  PopAssociation();
  PopScope();
  CheckRef(x.v);
}

void ConstructVisitor::Post(const parser::Association &x) {
  const auto &name{std::get<parser::Name>(x.t)};
  GetCurrentAssociation().name = &name;
  if (auto *symbol{MakeAssocEntity()}) {
    SetTypeFromAssociation(*symbol);
    SetAttrsFromAssociation(*symbol);
  }
  GetCurrentAssociation() = {};  // clean for further parser::Association.
}

bool ConstructVisitor::Pre(const parser::ChangeTeamStmt &x) {
  CheckDef(x.t);
  PushScope(Scope::Kind::Block, nullptr);
  PushAssociation();
  return true;
}

void ConstructVisitor::Post(const parser::CoarrayAssociation &x) {
  const auto &decl{std::get<parser::CodimensionDecl>(x.t)};
  const auto &name{std::get<parser::Name>(decl.t)};
  if (auto *symbol{FindInScope(currScope(), name)}) {
    const auto &selector{std::get<parser::Selector>(x.t)};
    if (auto sel{ResolveSelector(selector)}) {
      const Symbol *whole{UnwrapWholeSymbolDataRef(sel.expr)};
      if (!whole || whole->Corank() == 0) {
        Say(sel.source,  // C1116
            "Selector in coarray association must name a coarray"_err_en_US);
      } else if (auto dynType{sel.expr->GetType()}) {
        if (!symbol->GetType()) {
          symbol->SetType(ToDeclTypeSpec(std::move(*dynType)));
        }
      }
    }
  }
}

void ConstructVisitor::Post(const parser::EndChangeTeamStmt &x) {
  PopAssociation();
  PopScope();
  CheckRef(x.t);
}

bool ConstructVisitor::Pre(const parser::SelectTypeConstruct &) {
  PushAssociation();
  return true;
}

void ConstructVisitor::Post(const parser::SelectTypeConstruct &) {
  PopAssociation();
}

void ConstructVisitor::Post(const parser::SelectTypeStmt &x) {
  auto &association{GetCurrentAssociation()};
  if (const std::optional<parser::Name> &name{std::get<1>(x.t)}) {
    // This isn't a name in the current scope, it is in each TypeGuardStmt
    MakePlaceholder(*name, MiscDetails::Kind::SelectTypeAssociateName);
    association.name = &*name;
  } else {
    if (const Symbol *
        whole{UnwrapWholeSymbolDataRef(association.selector.expr)}) {
      ConvertToObjectEntity(const_cast<Symbol &>(*whole));
      if (!IsVariableName(*whole)) {
        Say(association.selector.source,  // C901
            "Selector is not a variable"_err_en_US);
        association = {};
      }
    } else {
      Say(association.selector.source,  // C1157
          "Selector is not a named variable: 'associate-name =>' is required"_err_en_US);
      association = {};
    }
  }
}

bool ConstructVisitor::Pre(const parser::SelectTypeConstruct::TypeCase &) {
  PushScope(Scope::Kind::Block, nullptr);
  return true;
}
void ConstructVisitor::Post(const parser::SelectTypeConstruct::TypeCase &) {
  PopScope();
}

void ConstructVisitor::Post(const parser::TypeGuardStmt::Guard &x) {
  if (auto *symbol{MakeAssocEntity()}) {
    if (std::holds_alternative<parser::Default>(x.u)) {
      SetTypeFromAssociation(*symbol);
    } else if (const auto *type{GetDeclTypeSpec()}) {
      symbol->SetType(*type);
    }
    SetAttrsFromAssociation(*symbol);
  }
}

bool ConstructVisitor::Pre(const parser::SelectRankConstruct &) {
  PushAssociation();
  return true;
}

void ConstructVisitor::Post(const parser::SelectRankConstruct &) {
  PopAssociation();
}

bool ConstructVisitor::CheckDef(const std::optional<parser::Name> &x) {
  if (x) {
    MakeSymbol(*x, MiscDetails{MiscDetails::Kind::ConstructName});
  }
  return true;
}

void ConstructVisitor::CheckRef(const std::optional<parser::Name> &x) {
  if (x) {
    // Just add an occurrence of this name; checking is done in ValidateLabels
    FindSymbol(*x);
  }
}

// Make a symbol representing an associating entity from current association.
Symbol *ConstructVisitor::MakeAssocEntity() {
  Symbol *symbol{nullptr};
  auto &association{GetCurrentAssociation()};
  if (association.name) {
    symbol = &MakeSymbol(*association.name, UnknownDetails{});
    if (symbol->has<AssocEntityDetails>() && symbol->owner() == currScope()) {
      Say(*association.name,  // C1104
          "The associate name '%s' is already used in this associate statement"_err_en_US);
      return nullptr;
    }
  } else if (const Symbol *
      whole{UnwrapWholeSymbolDataRef(association.selector.expr)}) {
    symbol = &MakeSymbol(whole->name());
  } else {
    return nullptr;
  }
  if (auto &expr{association.selector.expr}) {
    symbol->set_details(AssocEntityDetails{common::Clone(*expr)});
  } else {
    symbol->set_details(AssocEntityDetails{});
  }
  return symbol;
}

// Set the type of symbol based on the current association selector.
void ConstructVisitor::SetTypeFromAssociation(Symbol &symbol) {
  auto &details{symbol.get<AssocEntityDetails>()};
  const MaybeExpr *pexpr{&details.expr()};
  if (!pexpr->has_value()) {
    pexpr = &GetCurrentAssociation().selector.expr;
  }
  if (pexpr->has_value()) {
    const SomeExpr &expr{**pexpr};
    if (std::optional<evaluate::DynamicType> type{expr.GetType()}) {
      if (const auto *charExpr{
              evaluate::UnwrapExpr<evaluate::Expr<evaluate::SomeCharacter>>(
                  expr)}) {
        symbol.SetType(ToDeclTypeSpec(std::move(*type),
            FoldExpr(
                std::visit([](const auto &kindChar) { return kindChar.LEN(); },
                    charExpr->u))));
      } else {
        symbol.SetType(ToDeclTypeSpec(std::move(*type)));
      }
    } else {
      // BOZ literals, procedure designators, &c. are not acceptable
      Say(symbol.name(), "Associate name '%s' must have a type"_err_en_US);
    }
  }
}

// If current selector is a variable, set some of its attributes on symbol.
void ConstructVisitor::SetAttrsFromAssociation(Symbol &symbol) {
  Attrs attrs{evaluate::GetAttrs(GetCurrentAssociation().selector.expr)};
  symbol.attrs() |= attrs &
      Attrs{Attr::TARGET, Attr::ASYNCHRONOUS, Attr::VOLATILE, Attr::CONTIGUOUS};
  if (attrs.test(Attr::POINTER)) {
    symbol.attrs().set(Attr::TARGET);
  }
}

ConstructVisitor::Selector ConstructVisitor::ResolveSelector(
    const parser::Selector &x) {
  return std::visit(
      common::visitors{
          [&](const parser::Expr &expr) {
            return Selector{expr.source, EvaluateExpr(expr)};
          },
          [&](const parser::Variable &var) {
            return Selector{var.GetSource(), EvaluateExpr(var)};
          },
      },
      x.u);
}

ConstructVisitor::Association &ConstructVisitor::GetCurrentAssociation() {
  CHECK(!associationStack_.empty());
  return associationStack_.back();
}

void ConstructVisitor::PushAssociation() {
  associationStack_.emplace_back(Association{});
}

void ConstructVisitor::PopAssociation() {
  CHECK(!associationStack_.empty());
  associationStack_.pop_back();
}

const DeclTypeSpec &ConstructVisitor::ToDeclTypeSpec(
    evaluate::DynamicType &&type) {
  switch (type.category()) {
  case common::TypeCategory::Integer:
  case common::TypeCategory::Real:
  case common::TypeCategory::Complex:
    return context().MakeNumericType(type.category(), type.kind());
  case common::TypeCategory::Logical:
    return context().MakeLogicalType(type.kind());
  case common::TypeCategory::Derived:
    if (type.IsAssumedType()) {
      return currScope().MakeTypeStarType();
    } else if (type.IsUnlimitedPolymorphic()) {
      return currScope().MakeClassStarType();
    } else {
      return currScope().MakeDerivedType(type.IsPolymorphic()
              ? DeclTypeSpec::ClassDerived
              : DeclTypeSpec::TypeDerived,
          DerivedTypeSpec{type.GetDerivedTypeSpec()});
    }
  case common::TypeCategory::Character:
  default: CRASH_NO_CASE;
  }
}

const DeclTypeSpec &ConstructVisitor::ToDeclTypeSpec(
    evaluate::DynamicType &&type, MaybeSubscriptIntExpr &&length) {
  CHECK(type.category() == common::TypeCategory::Character);
  if (length.has_value()) {
    return currScope().MakeCharacterType(
        ParamValue{SomeIntExpr{*std::move(length)}}, KindExpr{type.kind()});
  } else {
    return currScope().MakeCharacterType(
        ParamValue::Deferred(), KindExpr{type.kind()});
  }
}

// ResolveNamesVisitor implementation

bool ResolveNamesVisitor::Pre(const parser::FunctionReference &x) {
  HandleCall(Symbol::Flag::Function, x.v);
  return false;
}
bool ResolveNamesVisitor::Pre(const parser::CallStmt &x) {
  HandleCall(Symbol::Flag::Subroutine, x.v);
  return false;
}

bool ResolveNamesVisitor::Pre(const parser::ImportStmt &x) {
  auto &scope{currScope()};
  // Check C896 and C899: where IMPORT statements are allowed
  switch (scope.kind()) {
  case Scope::Kind::Module:
    if (scope.IsModule()) {
      Say("IMPORT is not allowed in a module scoping unit"_err_en_US);
      return false;
    } else if (x.kind == common::ImportKind::None) {
      Say("IMPORT,NONE is not allowed in a submodule scoping unit"_err_en_US);
      return false;
    }
    break;
  case Scope::Kind::MainProgram:
    Say("IMPORT is not allowed in a main program scoping unit"_err_en_US);
    return false;
  case Scope::Kind::Subprogram:
    if (scope.parent().IsGlobal()) {
      Say("IMPORT is not allowed in an external subprogram scoping unit"_err_en_US);
      return false;
    }
    break;
  default:;
  }
  if (auto error{scope.SetImportKind(x.kind)}) {
    Say(std::move(*error));
  }
  for (auto &name : x.names) {
    if (FindSymbol(scope.parent(), name)) {
      scope.add_importName(name.source);
    } else {
      Say(name, "'%s' not found in host scope"_err_en_US);
    }
  }
  prevImportStmt_ = currStmtSource();
  return false;
}

const parser::Name *DeclarationVisitor::ResolveStructureComponent(
    const parser::StructureComponent &x) {
  return FindComponent(ResolveDataRef(x.base), x.component);
}

const parser::Name *DeclarationVisitor::ResolveDesignator(
    const parser::Designator &x) {
  return std::visit(
      common::visitors{
          [&](const parser::DataRef &x) { return ResolveDataRef(x); },
          [&](const parser::Substring &x) {
            return ResolveDataRef(std::get<parser::DataRef>(x.t));
          },
      },
      x.u);
}

const parser::Name *DeclarationVisitor::ResolveDataRef(
    const parser::DataRef &x) {
  return std::visit(
      common::visitors{
          [=](const parser::Name &y) { return ResolveName(y); },
          [=](const Indirection<parser::StructureComponent> &y) {
            return ResolveStructureComponent(y.value());
          },
          [=](const auto &y) { return ResolveDataRef(y.value().base); },
      },
      x.u);
}

const parser::Name *DeclarationVisitor::ResolveVariable(
    const parser::Variable &x) {
  return std::visit(
      common::visitors{
          [&](const common::Indirection<parser::Designator> &y) {
            return ResolveDesignator(y.value());
          },
          [&](const common::Indirection<parser::FunctionReference> &y) {
            const auto &proc{
                std::get<parser::ProcedureDesignator>(y.value().v.t)};
            return std::visit(
                common::visitors{
                    [&](const parser::Name &z) { return &z; },
                    [&](const parser::ProcComponentRef &z) {
                      return ResolveStructureComponent(z.v.thing);
                    },
                },
                proc.u);
          },
      },
      x.u);
}

// If implicit types are allowed, ensure name is in the symbol table.
// Otherwise, report an error if it hasn't been declared.
const parser::Name *DeclarationVisitor::ResolveName(const parser::Name &name) {
  if (Symbol * symbol{FindSymbol(name)}) {
    if (CheckUseError(name)) {
      return nullptr;  // reported an error
    }
    if (symbol->IsDummy()) {
      ConvertToObjectEntity(*symbol);
      ApplyImplicitRules(*symbol);
    }
    return &name;
  }
  if (isImplicitNoneType()) {
    Say(name, "No explicit type declared for '%s'"_err_en_US);
    return nullptr;
  }
  // Create the symbol then ensure it is accessible
  MakeSymbol(InclusiveScope(), name.source, Attrs{});
  auto *symbol{FindSymbol(name)};
  if (!symbol) {
    Say(name,
        "'%s' from host scoping unit is not accessible due to IMPORT"_err_en_US);
    return nullptr;
  }
  ConvertToObjectEntity(*symbol);
  ApplyImplicitRules(*symbol);
  return &name;
}

// base is a part-ref of a derived type; find the named component in its type.
// Also handles intrinsic type parameter inquiries (%kind, %len) and
// COMPLEX component references (%re, %im).
const parser::Name *DeclarationVisitor::FindComponent(
    const parser::Name *base, const parser::Name &component) {
  if (!base || !base->symbol) {
    return nullptr;
  }
  auto &symbol{*base->symbol};
  if (!symbol.has<AssocEntityDetails>() && !ConvertToObjectEntity(symbol)) {
    SayWithDecl(*base, symbol,
        "'%s' is an invalid base for a component reference"_err_en_US);
    return nullptr;
  }
  auto *type{symbol.GetType()};
  if (!type) {
    return nullptr;  // should have already reported error
  }
  if (const IntrinsicTypeSpec * intrinsic{type->AsIntrinsic()}) {
    auto name{component.ToString()};
    auto category{intrinsic->category()};
    MiscDetails::Kind miscKind{MiscDetails::Kind::None};
    if (name == "kind") {
      miscKind = MiscDetails::Kind::KindParamInquiry;
    } else if (category == TypeCategory::Character) {
      if (name == "len") {
        miscKind = MiscDetails::Kind::LenParamInquiry;
      }
    } else if (category == TypeCategory::Complex) {
      if (name == "re") {
        miscKind = MiscDetails::Kind::ComplexPartRe;
      } else if (name == "im") {
        miscKind = MiscDetails::Kind::ComplexPartIm;
      }
    }
    if (miscKind != MiscDetails::Kind::None) {
      MakePlaceholder(component, miscKind);
      return nullptr;
    }
  } else if (const DerivedTypeSpec * derived{type->AsDerived()}) {
    if (const Scope * scope{derived->scope()}) {
      if (Resolve(component, FindInTypeOrParents(*scope, component.source))) {
        if (CheckAccessibleComponent(component.source, *component.symbol)) {
          return &component;
        }
      } else {
        SayDerivedType(component.source,
            "Component '%s' not found in derived type '%s'"_err_en_US, *scope);
      }
    }
    return nullptr;
  }
  if (symbol.test(Symbol::Flag::Implicit)) {
    Say(*base,
        "'%s' is not an object of derived type; it is implicitly typed"_err_en_US);
  } else {
    SayWithDecl(
        *base, symbol, "'%s' is not an object of derived type"_err_en_US);
  }
  return nullptr;
}

// C764, C765
void DeclarationVisitor::CheckInitialDataTarget(
    const Symbol &pointer, const SomeExpr &expr, SourceName source) {
  if (!evaluate::IsInitialDataTarget(expr)) {
    Say(source,
        "Pointer '%s' cannot be initialized with a reference to a designator with non-constant subscripts"_err_en_US,
        pointer.name());
    return;
  }
  if (pointer.Rank() != expr.Rank()) {
    Say(source,
        "Pointer '%s' of rank %d cannot be initialized with a target of different rank (%d)"_err_en_US,
        pointer.name(), pointer.Rank(), expr.Rank());
    return;
  }
  if (auto base{evaluate::GetBaseObject(expr)}) {
    if (const Symbol * baseSym{base->symbol()}) {
      const Symbol &ultimate{baseSym->GetUltimate()};
      if (IsAllocatable(ultimate)) {
        Say(source,
            "Pointer '%s' cannot be initialized with a reference to an allocatable '%s'"_err_en_US,
            pointer.name(), ultimate.name());
        return;
      }
      if (ultimate.Corank() > 0) {
        Say(source,
            "Pointer '%s' cannot be initialized with a reference to a coarray '%s'"_err_en_US,
            pointer.name(), ultimate.name());
        return;
      }
      if (!ultimate.attrs().test(Attr::TARGET)) {
        Say(source,
            "Pointer '%s' cannot be initialized with a reference to an object '%s' that lacks the TARGET attribute"_err_en_US,
            pointer.name(), ultimate.name());
        return;
      }
      if (!ultimate.attrs().test(Attr::SAVE)) {
        Say(source,
            "Pointer '%s' cannot be initialized with a reference to an object '%s' that lacks the SAVE attribute"_err_en_US,
            pointer.name(), ultimate.name());
        return;
      }
    }
  }
  // TODO: check type compatibility
  // TODO: check non-deferred type parameter values
  // TODO: check contiguity if pointer is CONTIGUOUS
}

void DeclarationVisitor::Initialization(const parser::Name &name,
    const parser::Initialization &init, bool inComponentDecl) {
  if (name.symbol == nullptr) {
    return;
  }
  // Traversal of the initializer was deferred to here so that the
  // symbol being declared can be available for e.g.
  //   real, parameter :: x = tiny(x)
  Walk(init.u);
  Symbol &ultimate{name.symbol->GetUltimate()};
  if (auto *details{ultimate.detailsIf<ObjectEntityDetails>()}) {
    // TODO: check C762 - all bounds and type parameters of component
    // are colons or constant expressions if component is initialized
    bool isPointer{false};
    std::visit(
        common::visitors{
            [&](const parser::ConstantExpr &expr) {
              if (inComponentDecl) {
                // Can't convert to type of component, which might not yet
                // be known; that's done later during instantiation.
                if (MaybeExpr value{EvaluateExpr(expr)}) {
                  details->set_init(std::move(*value));
                }
              } else {
                if (MaybeExpr folded{EvaluateConvertedExpr(
                        ultimate, expr, expr.thing.value().source)}) {
                  details->set_init(std::move(*folded));
                }
              }
            },
            [&](const parser::NullInit &) {
              isPointer = true;
              details->set_init(SomeExpr{evaluate::NullPointer{}});
            },
            [&](const parser::InitialDataTarget &initExpr) {
              isPointer = true;
              if (MaybeExpr expr{EvaluateExpr(initExpr)}) {
                CheckInitialDataTarget(
                    ultimate, *expr, initExpr.value().source);
                details->set_init(std::move(*expr));
              }
            },
            [&](const std::list<common::Indirection<parser::DataStmtValue>>
                    &list) {
              if (inComponentDecl) {
                Say(name,
                    "Component '%s' initialized with DATA statement values"_err_en_US);
              } else {
                // TODO - DATA statements and DATA-like initialization extension
              }
            },
        },
        init.u);
    if (isPointer) {
      if (!IsPointer(ultimate)) {
        Say(name,
            "Non-pointer component '%s' initialized with pointer target"_err_en_US);
      }
    } else {
      if (IsPointer(ultimate)) {
        Say(name,
            "Object pointer component '%s' initialized with non-pointer expression"_err_en_US);
      } else if (IsAllocatable(ultimate)) {
        Say(name, "Allocatable component '%s' cannot be initialized"_err_en_US);
      }
    }
  }
}

void ResolveNamesVisitor::HandleCall(
    Symbol::Flag procFlag, const parser::Call &call) {
  std::visit(
      common::visitors{
          [&](const parser::Name &x) { HandleProcedureName(procFlag, x); },
          [&](const parser::ProcComponentRef &x) { Walk(x); },
      },
      std::get<parser::ProcedureDesignator>(call.t).u);
  Walk(std::get<std::list<parser::ActualArgSpec>>(call.t));
}

void ResolveNamesVisitor::HandleProcedureName(
    Symbol::Flag flag, const parser::Name &name) {
  CHECK(flag == Symbol::Flag::Function || flag == Symbol::Flag::Subroutine);
  auto *symbol{FindSymbol(name)};
  if (symbol == nullptr) {
    if (context().intrinsics().IsIntrinsic(name.source.ToString())) {
      symbol =
          &MakeSymbol(InclusiveScope(), name.source, Attrs{Attr::INTRINSIC});
    } else {
      symbol = &MakeSymbol(context().globalScope(), name.source, Attrs{});
    }
    Resolve(name, *symbol);
    if (symbol->has<ModuleDetails>()) {
      SayWithDecl(name, *symbol,
          "Use of '%s' as a procedure conflicts with its declaration"_err_en_US);
      return;
    }
    if (!symbol->attrs().test(Attr::INTRINSIC)) {
      if (isImplicitNoneExternal() && !symbol->attrs().test(Attr::EXTERNAL)) {
        Say(name,
            "'%s' is an external procedure without the EXTERNAL"
            " attribute in a scope with IMPLICIT NONE(EXTERNAL)"_err_en_US);
        return;
      }
      MakeExternal(*symbol);
    }
    if (!symbol->has<ProcEntityDetails>()) {
      ConvertToProcEntity(*symbol);
    }
    SetProcFlag(name, *symbol, flag);
  } else if (symbol->has<UnknownDetails>()) {
    CHECK(!"unexpected UnknownDetails");
  } else if (CheckUseError(name)) {
    // error was reported
  } else {
    symbol = &Resolve(name, symbol)->GetUltimate();
    ConvertToProcEntity(*symbol);
    if (!SetProcFlag(name, *symbol, flag)) {
      return;  // reported error
    }
    if (IsProcedure(*symbol) || symbol->has<DerivedTypeDetails>() ||
        symbol->has<ObjectEntityDetails>()) {
      // these are all valid as procedure-designators
    } else if (symbol->test(Symbol::Flag::Implicit)) {
      Say(name,
          "Use of '%s' as a procedure conflicts with its implicit definition"_err_en_US);
    } else {
      SayWithDecl(name, *symbol,
          "Use of '%s' as a procedure conflicts with its declaration"_err_en_US);
    }
  }
}

// Variant of HandleProcedureName() for use while skimming the executable
// part of a subprogram to catch calls that might be part of the subprogram's
// interface, and to mark as procedures any symbols that might otherwise be
// miscategorized as objects.
void ResolveNamesVisitor::NoteExecutablePartCall(
    Symbol::Flag flag, const parser::Call &call) {
  auto &designator{std::get<parser::ProcedureDesignator>(call.t)};
  if (const auto *name{std::get_if<parser::Name>(&designator.u)}) {
    if (Symbol * symbol{FindSymbol(*name)}) {
      Symbol::Flag other{flag == Symbol::Flag::Subroutine
              ? Symbol::Flag::Function
              : Symbol::Flag::Subroutine};
      if (!symbol->test(other)) {
        ConvertToProcEntity(*symbol);
        if (symbol->has<ProcEntityDetails>()) {
          symbol->set(flag);
          if (symbol->IsDummy()) {
            symbol->attrs().set(Attr::EXTERNAL);
          }
          ApplyImplicitRules(*symbol);
        }
      }
    }
  }
}

// Check and set the Function or Subroutine flag on symbol; false on error.
bool ResolveNamesVisitor::SetProcFlag(
    const parser::Name &name, Symbol &symbol, Symbol::Flag flag) {
  if (symbol.test(Symbol::Flag::Function) && flag == Symbol::Flag::Subroutine) {
    SayWithDecl(
        name, symbol, "Cannot call function '%s' like a subroutine"_err_en_US);
    return false;
  } else if (symbol.test(Symbol::Flag::Subroutine) &&
      flag == Symbol::Flag::Function) {
    SayWithDecl(
        name, symbol, "Cannot call subroutine '%s' like a function"_err_en_US);
    return false;
  } else if (symbol.has<ProcEntityDetails>()) {
    symbol.set(flag);  // in case it hasn't been set yet
    if (flag == Symbol::Flag::Function) {
      ApplyImplicitRules(symbol);
    }
  } else if (symbol.GetType() != nullptr && flag == Symbol::Flag::Subroutine) {
    SayWithDecl(
        name, symbol, "Cannot call function '%s' like a subroutine"_err_en_US);
  }
  return true;
}

bool ModuleVisitor::Pre(const parser::AccessStmt &x) {
  Attr accessAttr{AccessSpecToAttr(std::get<parser::AccessSpec>(x.t))};
  if (currScope().kind() != Scope::Kind::Module) {
    Say(*currStmtSource(),
        "%s statement may only appear in the specification part of a module"_err_en_US,
        EnumToString(accessAttr));
    return false;
  }
  const auto &accessIds{std::get<std::list<parser::AccessId>>(x.t)};
  if (accessIds.empty()) {
    if (prevAccessStmt_) {
      Say("The default accessibility of this module has already been declared"_err_en_US)
          .Attach(*prevAccessStmt_, "Previous declaration"_en_US);
    }
    prevAccessStmt_ = currStmtSource();
    defaultAccess_ = accessAttr;
  } else {
    for (const auto &accessId : accessIds) {
      std::visit(
          common::visitors{
              [=](const parser::Name &y) {
                Resolve(y, SetAccess(y.source, accessAttr));
              },
              [=](const Indirection<parser::GenericSpec> &y) {
                auto info{GenericSpecInfo{y.value()}};
                info.Resolve(&SetAccess(info.symbolName(), accessAttr));
              },
          },
          accessId.u);
    }
  }
  return false;
}

// Set the access specification for this name.
Symbol &ModuleVisitor::SetAccess(const SourceName &name, Attr attr) {
  Symbol &symbol{MakeSymbol(name)};
  Attrs &attrs{symbol.attrs()};
  if (attrs.HasAny({Attr::PUBLIC, Attr::PRIVATE})) {
    // PUBLIC/PRIVATE already set: make it a fatal error if it changed
    Attr prev = attrs.test(Attr::PUBLIC) ? Attr::PUBLIC : Attr::PRIVATE;
    auto msg{IsDefinedOperator(name)
            ? "The accessibility of operator '%s' has already been specified as %s"_en_US
            : "The accessibility of '%s' has already been specified as %s"_en_US};
    Say(name, WithIsFatal(msg, attr != prev), name, EnumToString(prev));
  } else {
    attrs.set(attr);
  }
  return symbol;
}

static bool NeedsExplicitType(const Symbol &symbol) {
  if (symbol.has<UnknownDetails>()) {
    return true;
  } else if (const auto *details{symbol.detailsIf<EntityDetails>()}) {
    return !details->type();
  } else if (const auto *details{symbol.detailsIf<ObjectEntityDetails>()}) {
    return !details->type();
  } else if (const auto *details{symbol.detailsIf<ProcEntityDetails>()}) {
    return details->interface().symbol() == nullptr &&
        details->interface().type() == nullptr;
  } else {
    return false;
  }
}

void ResolveNamesVisitor::Post(const parser::SpecificationPart &) {
  badStmtFuncFound_ = false;
  CheckImports();
  bool inModule{currScope().kind() == Scope::Kind::Module};
  for (auto &pair : currScope()) {
    auto &symbol{*pair.second};
    if (NeedsExplicitType(symbol)) {
      ApplyImplicitRules(symbol);
    }
    if (symbol.has<GenericDetails>()) {
      CheckGenericProcedures(symbol);
    }
    if (inModule && symbol.attrs().test(Attr::EXTERNAL) &&
        !symbol.test(Symbol::Flag::Function)) {
      // in a module, external proc without return type is subroutine
      symbol.set(Symbol::Flag::Subroutine);
    }
  }
  CheckSaveStmts();
  CheckCommonBlocks();
  CheckEquivalenceSets();
}

void ResolveNamesVisitor::CheckImports() {
  auto &scope{currScope()};
  switch (scope.GetImportKind()) {
  case common::ImportKind::None: break;
  case common::ImportKind::All:
    // C8102: all entities in host must not be hidden
    for (const auto &pair : scope.parent()) {
      auto &name{pair.first};
      if (name != scope.name()) {
        CheckImport(*prevImportStmt_, name);
      }
    }
    break;
  case common::ImportKind::Default:
  case common::ImportKind::Only:
    // C8102: entities named in IMPORT must not be hidden
    for (auto &name : scope.importNames()) {
      CheckImport(name, name);
    }
    break;
  }
}

void ResolveNamesVisitor::CheckImport(
    const SourceName &location, const SourceName &name) {
  if (auto *symbol{FindInScope(currScope(), name)}) {
    Say(location, "'%s' from host is not accessible"_err_en_US, name)
        .Attach(symbol->name(), "'%s' is hidden by this entity"_en_US,
            symbol->name());
  }
}

bool ResolveNamesVisitor::Pre(const parser::ImplicitStmt &x) {
  return CheckNotInBlock("IMPLICIT") && ImplicitRulesVisitor::Pre(x);
}

void ResolveNamesVisitor::Post(const parser::PointerObject &x) {
  std::visit(
      common::visitors{
          [&](const parser::Name &x) { ResolveName(x); },
          [&](const parser::StructureComponent &x) {
            ResolveStructureComponent(x);
          },
      },
      x.u);
}
void ResolveNamesVisitor::Post(const parser::AllocateObject &x) {
  std::visit(
      common::visitors{
          [&](const parser::Name &x) { ResolveName(x); },
          [&](const parser::StructureComponent &x) {
            ResolveStructureComponent(x);
          },
      },
      x.u);
}

bool ResolveNamesVisitor::Pre(const parser::PointerAssignmentStmt &x) {
  const auto &dataRef{std::get<parser::DataRef>(x.t)};
  const auto &bounds{std::get<parser::PointerAssignmentStmt::Bounds>(x.t)};
  const auto &expr{std::get<parser::Expr>(x.t)};
  ResolveDataRef(dataRef);
  Walk(bounds);
  // Resolve unrestricted specific intrinsic procedures as in "p => cos".
  if (const parser::Name * name{parser::Unwrap<parser::Name>(expr)}) {
    if (NameIsKnownOrIntrinsic(*name)) {
      return false;
    }
  }
  Walk(expr);
  return false;
}
void ResolveNamesVisitor::Post(const parser::Designator &x) {
  ResolveDesignator(x);
}

void ResolveNamesVisitor::Post(const parser::ProcComponentRef &x) {
  ResolveStructureComponent(x.v.thing);
}
void ResolveNamesVisitor::Post(const parser::TypeGuardStmt &x) {
  DeclTypeSpecVisitor::Post(x);
  ConstructVisitor::Post(x);
}
bool ResolveNamesVisitor::Pre(const parser::StmtFunctionStmt &x) {
  if (!HandleStmtFunction(x)) {
    // This is an array element assignment: resolve names of indices
    const auto &names{std::get<std::list<parser::Name>>(x.t)};
    for (auto &name : names) {
      ResolveName(name);
    }
  }
  return true;
}

bool ResolveNamesVisitor::Pre(const parser::DefinedOpName &x) {
  const parser::Name &name{x.v};
  if (FindSymbol(name)) {
    // OK
  } else if (IsLogicalConstant(context(), name.source)) {
    Say(name,
        "Logical constant '%s' may not be used as a defined operator"_err_en_US);
  } else {
    Say(name, "Defined operator '%s' not found"_err_en_US);
  }
  return false;
}

bool ResolveNamesVisitor::Pre(const parser::ProgramUnit &x) {
  auto root{ProgramTree::Build(x)};
  SetScope(context().globalScope());
  ResolveSpecificationParts(root);
  FinishSpecificationParts(root);
  ResolveExecutionParts(root);
  return false;
}

// References to procedures need to record that their symbols are known
// to be procedures, so that they don't get converted to objects by default.
class ExecutionPartSkimmer {
public:
  explicit ExecutionPartSkimmer(ResolveNamesVisitor &resolver)
    : resolver_{resolver} {}

  void Walk(const parser::ExecutionPart *exec) {
    if (exec != nullptr) {
      parser::Walk(*exec, *this);
    }
  }

  template<typename A> bool Pre(const A &) { return true; }
  template<typename A> void Post(const A &) {}
  void Post(const parser::FunctionReference &fr) {
    resolver_.NoteExecutablePartCall(Symbol::Flag::Function, fr.v);
  }
  void Post(const parser::CallStmt &cs) {
    resolver_.NoteExecutablePartCall(Symbol::Flag::Subroutine, cs.v);
  }

private:
  ResolveNamesVisitor &resolver_;
};

// Build the scope tree and resolve names in the specification parts of this
// node and its children
void ResolveNamesVisitor::ResolveSpecificationParts(ProgramTree &node) {
  if (!BeginScope(node)) {
    return;  // an error prevented scope from being created
  }
  Scope &scope{currScope()};
  node.set_scope(scope);
  AddSubpNames(node);
  std::visit([&](const auto *x) { Walk(*x); }, node.stmt());
  Walk(node.spec());
  if (node.IsModule()) {
    ApplyDefaultAccess();
  }
  for (auto &child : node.children()) {
    ResolveSpecificationParts(child);
  }
  ExecutionPartSkimmer{*this}.Walk(node.exec());
  PopScope();
  // Ensure that every object entity has a type.
  for (auto &pair : *node.scope()) {
    ApplyImplicitRules(*pair.second);
  }
}

// Add SubprogramNameDetails symbols for contained subprograms
void ResolveNamesVisitor::AddSubpNames(const ProgramTree &node) {
  auto kind{
      node.IsModule() ? SubprogramKind::Module : SubprogramKind::Internal};
  for (const auto &child : node.children()) {
    auto &symbol{MakeSymbol(child.name(), SubprogramNameDetails{kind})};
    symbol.set(child.GetSubpFlag());
  }
}

// Push a new scope for this node or return false on error.
bool ResolveNamesVisitor::BeginScope(const ProgramTree &node) {
  switch (node.GetKind()) {
  case ProgramTree::Kind::Program:
    PushScope(Scope::Kind::MainProgram,
        &MakeSymbol(node.name(), MainProgramDetails{}));
    return true;
  case ProgramTree::Kind::Function:
  case ProgramTree::Kind::Subroutine:
    return BeginSubprogram(
        node.name(), node.GetSubpFlag(), node.HasModulePrefix());
  case ProgramTree::Kind::MpSubprogram:
    return BeginSubprogram(
        node.name(), Symbol::Flag::Subroutine, /*hasModulePrefix*/ true);
  case ProgramTree::Kind::Module: BeginModule(node.name(), false); return true;
  case ProgramTree::Kind::Submodule:
    return BeginSubmodule(node.name(), node.GetParentId());
  default: CRASH_NO_CASE;
  }
}

// Perform checks that need to happen after all of the specification parts
// but before any of the execution parts.
void ResolveNamesVisitor::FinishSpecificationParts(const ProgramTree &node) {
  if (!node.scope()) {
    return;  // error occurred creating scope
  }
  SetScope(*node.scope());
  for (auto &pair : currScope()) {
    Symbol &symbol{*pair.second};
    if (const auto *details{symbol.detailsIf<GenericDetails>()}) {
      CheckSpecificsAreDistinguishable(symbol, details->specificProcs());
    } else if (symbol.has<ProcEntityDetails>()) {
      CheckExplicitInterface(symbol);
    }
  }
  for (Scope &childScope : currScope().children()) {
    if (childScope.IsDerivedType() && childScope.symbol()) {
      FinishDerivedType(childScope);
    }
  }
  for (const auto &child : node.children()) {
    FinishSpecificationParts(child);
  }
}

static int FindIndexOfName(
    const SourceName &name, std::vector<Symbol *> symbols) {
  for (std::size_t i{0}; i < symbols.size(); ++i) {
    if (symbols[i] && symbols[i]->name() == name) {
      return i;
    }
  }
  return -1;
}

// Perform checks on procedure bindings of this type
void ResolveNamesVisitor::FinishDerivedType(Scope &scope) {
  CHECK(scope.IsDerivedType());
  for (auto &pair : scope) {
    Symbol &comp{*pair.second};
    std::visit(
        common::visitors{
            [&](ProcEntityDetails &x) {
              SetPassArg(comp, x.interface().symbol(), x);
              CheckExplicitInterface(comp);
            },
            [&](ProcBindingDetails &x) {
              SetPassArg(comp, &x.symbol(), x);
              CheckExplicitInterface(comp);
            },
            [](auto &) {},
        },
        comp.details());
  }
  for (auto &pair : scope) {
    Symbol &comp{*pair.second};
    if (const auto *details{comp.detailsIf<GenericBindingDetails>()}) {
      CheckSpecificsAreDistinguishable(comp, details->specificProcs());
    }
  }
}

// Check C760, constraints on the passed-object dummy argument
// If they all pass, set the passIndex in details.
void ResolveNamesVisitor::SetPassArg(
    const Symbol &proc, const Symbol *interface, WithPassArg &details) {
  if (proc.attrs().test(Attr::NOPASS)) {
    return;
  }
  const auto &name{proc.name()};
  if (!interface) {
    Say(name,
        "Procedure component '%s' must have NOPASS attribute or explicit interface"_err_en_US,
        name);
    return;
  }
  const SourceName *passName{details.passName()};
  const auto &dummyArgs{interface->get<SubprogramDetails>().dummyArgs()};
  if (!passName && dummyArgs.empty()) {
    Say(name,
        proc.has<ProcEntityDetails>()
            ? "Procedure component '%s' with no dummy arguments"
              " must have NOPASS attribute"_err_en_US
            : "Procedure binding '%s' with no dummy arguments"
              " must have NOPASS attribute"_err_en_US,
        name);
    return;
  }
  int passArgIndex{0};
  if (!passName) {
    passName = &dummyArgs[0]->name();
  } else {
    passArgIndex = FindIndexOfName(*passName, dummyArgs);
    if (passArgIndex < 0) {
      Say(*passName,
          "'%s' is not a dummy argument of procedure interface '%s'"_err_en_US,
          *passName, interface->name());
      return;
    }
  }
  const Symbol &passArg{*dummyArgs[passArgIndex]};
  std::optional<MessageFixedText> msg;
  if (!passArg.has<ObjectEntityDetails>()) {
    msg = "Passed-object dummy argument '%s' of procedure '%s'"
          " must be a data object"_err_en_US;
  } else if (passArg.attrs().test(Attr::POINTER)) {
    msg = "Passed-object dummy argument '%s' of procedure '%s'"
          " may not have the POINTER attribute"_err_en_US;
  } else if (passArg.attrs().test(Attr::ALLOCATABLE)) {
    msg = "Passed-object dummy argument '%s' of procedure '%s'"
          " may not have the ALLOCATABLE attribute"_err_en_US;
  } else if (passArg.attrs().test(Attr::VALUE)) {
    msg = "Passed-object dummy argument '%s' of procedure '%s'"
          " may not have the VALUE attribute"_err_en_US;
  } else if (passArg.Rank() > 0) {
    msg = "Passed-object dummy argument '%s' of procedure '%s'"
          " must be scalar"_err_en_US;
  }
  if (msg) {
    Say(name, std::move(*msg), *passName, name);
    return;
  }
  const DeclTypeSpec *type{passArg.GetType()};
  if (!type) {
    return;  // an error already occurred
  }
  const Symbol &typeSymbol{*proc.owner().GetSymbol()};
  const DerivedTypeSpec *derived{type->AsDerived()};
  if (!derived || derived->typeSymbol() != typeSymbol) {
    Say(name,
        "Passed-object dummy argument '%s' of procedure '%s'"
        " must be of type '%s' but is '%s'"_err_en_US,
        *passName, name, typeSymbol.name(), type->AsFortran());
    return;
  }
  if (IsExtensibleType(derived) != type->IsPolymorphic()) {
    Say(name,
        type->IsPolymorphic()
            ? "Passed-object dummy argument '%s' of procedure '%s'"
              " must not be polymorphic because '%s' is not extensible"_err_en_US
            : "Passed-object dummy argument '%s' of procedure '%s'"
              " must polymorphic because '%s' is extensible"_err_en_US,
        *passName, name, typeSymbol.name());
    return;
  }
  for (const auto &[paramName, paramValue] : derived->parameters()) {
    if (paramValue.isLen() && !paramValue.isAssumed()) {
      Say(name,
          "Passed-object dummy argument '%s' of procedure '%s'"
          " has non-assumed length parameter '%s'"_err_en_US,
          *passName, name, paramName);
    }
  }
  details.set_passIndex(passArgIndex);
}

// Resolve names in the execution part of this node and its children
void ResolveNamesVisitor::ResolveExecutionParts(const ProgramTree &node) {
  if (!node.scope()) {
    return;  // error occurred creating scope
  }
  SetScope(*node.scope());
  if (const auto *exec{node.exec()}) {
    Walk(*exec);
  }
  PopScope();  // converts unclassified entities into objects
  for (const auto &child : node.children()) {
    ResolveExecutionParts(child);
  }
}

void ResolveNamesVisitor::Post(const parser::Program &) {
  // ensure that all temps were deallocated
  CHECK(!attrs_);
  CHECK(!GetDeclTypeSpec());
}

bool ResolveNames(SemanticsContext &context, const parser::Program &program) {
  ResolveNamesVisitor{context}.Walk(program);
  return !context.AnyFatalError();
}

}
