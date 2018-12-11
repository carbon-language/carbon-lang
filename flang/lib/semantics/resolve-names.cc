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
#include "default-kinds.h"
#include "expression.h"
#include "mod-file.h"
#include "rewrite-parse-tree.h"
#include "scope.h"
#include "semantics.h"
#include "symbol.h"
#include "type.h"
#include "../common/fortran.h"
#include "../common/indirection.h"
#include "../evaluate/common.h"
#include "../evaluate/fold.h"
#include "../evaluate/tools.h"
#include "../parser/parse-tree-visitor.h"
#include "../parser/parse-tree.h"
#include <list>
#include <memory>
#include <ostream>
#include <set>

namespace Fortran::semantics {

using namespace parser::literals;

using Message = parser::Message;
using Messages = parser::Messages;
using MessageFixedText = parser::MessageFixedText;
using MessageFormattedText = parser::MessageFormattedText;

class ResolveNamesVisitor;

static const parser::Name *GetGenericSpecName(const parser::GenericSpec &);

// ImplicitRules maps initial character of identifier to the DeclTypeSpec
// representing the implicit type; std::nullopt if none.
// It also records the presence of IMPLICIT NONE statements.
// When inheritFromParent is set, defaults come from the parent rules.
class ImplicitRules {
public:
  ImplicitRules() : inheritFromParent_{false} {}
  ImplicitRules(std::unique_ptr<ImplicitRules> &&parent)
    : inheritFromParent_{parent.get() != nullptr} {
    parent_.swap(parent);
  }
  void set_context(SemanticsContext &context) { context_ = &context; }
  std::unique_ptr<ImplicitRules> &&parent() { return std::move(parent_); }
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

  std::unique_ptr<ImplicitRules> parent_;
  std::optional<bool> isImplicitNoneType_;
  std::optional<bool> isImplicitNoneExternal_;
  bool inheritFromParent_;  // look in parent if not specified here
  SemanticsContext *context_{nullptr};
  // map initial character of identifier to nullptr or its default type
  std::map<char, const DeclTypeSpec *> map_;

  friend std::ostream &operator<<(std::ostream &, const ImplicitRules &);
  friend void ShowImplicitRule(std::ostream &, const ImplicitRules &, char);
};

// Track statement source locations and save messages.
class MessageHandler {
public:
  void set_messages(Messages &messages) { messages_ = &messages; }
  const SourceName *currStmtSource() { return currStmtSource_; }
  void set_currStmtSource(const SourceName *);

  // Emit a message
  Message &Say(Message &&);
  // Emit a message associated with the current statement source.
  Message &Say(MessageFixedText &&);
  // Emit a message about a SourceName
  Message &Say(const SourceName &, MessageFixedText &&);
  // Emit a formatted message associated with a source location.
  Message &Say(const SourceName &, MessageFixedText &&, const SourceName &);
  Message &Say(const SourceName &, MessageFixedText &&, const SourceName &,
      const SourceName &);

private:
  // Where messages are emitted:
  Messages *messages_{nullptr};
  // Source location of current statement; null if not in a statement
  const SourceName *currStmtSource_{nullptr};
};

class BaseVisitor {
public:
  template<typename T> void Walk(const T &);
  void set_this(ResolveNamesVisitor *x) { this_ = x; }

  MessageHandler &messageHandler() { return messageHandler_; }
  const SourceName *currStmtSource();
  SemanticsContext &context() const { return *context_; }
  void set_context(SemanticsContext &);

  template<typename T> MaybeExpr EvaluateExpr(const T &expr) {
    if (auto maybeExpr{AnalyzeExpr(*context_, expr)}) {
      return evaluate::Fold(context_->foldingContext(), std::move(*maybeExpr));
    } else {
      return std::nullopt;
    }
  }

  template<typename... A> Message &Say(const parser::Name &name, A... args) {
    return messageHandler_.Say(name.source, std::forward<A>(args)...);
  }
  template<typename... A> Message &Say(A... args) {
    return messageHandler_.Say(std::forward<A>(args)...);
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
};

// Find and create types from declaration-type-spec nodes.
class DeclTypeSpecVisitor : public AttrsVisitor {
public:
  explicit DeclTypeSpecVisitor() {}
  using AttrsVisitor::Post;
  using AttrsVisitor::Pre;
  void Post(const parser::IntegerTypeSpec &);
  void Post(const parser::IntrinsicTypeSpec::Logical &);
  void Post(const parser::IntrinsicTypeSpec::Real &);
  void Post(const parser::IntrinsicTypeSpec::Complex &);
  void Post(const parser::IntrinsicTypeSpec::DoublePrecision &);
  void Post(const parser::IntrinsicTypeSpec::DoubleComplex &);
  void Post(const parser::IntrinsicTypeSpec::Character &);
  void Post(const parser::DeclarationTypeSpec::ClassStar &);
  void Post(const parser::DeclarationTypeSpec::TypeStar &);
  void Post(const parser::TypeParamSpec &);
  bool Pre(const parser::TypeGuardStmt &);
  void Post(const parser::TypeGuardStmt &);

protected:
  DeclTypeSpec *GetDeclTypeSpec();
  void BeginDeclTypeSpec();
  void EndDeclTypeSpec();
  const parser::Name *derivedTypeName() const { return derivedTypeName_; }
  void SetDeclTypeSpec(const parser::Name &, DeclTypeSpec &);

private:
  bool expectDeclTypeSpec_{false};  // should only see decl-type-spec when true
  DeclTypeSpec *declTypeSpec_{nullptr};
  const parser::Name *derivedTypeName_{nullptr};

  void SetDeclTypeSpec(DeclTypeSpec &declTypeSpec);
  void MakeIntrinsic(TypeCategory, const std::optional<parser::KindSelector> &);
  void MakeIntrinsic(TypeCategory, int kind);
  int GetKindParamValue(
      TypeCategory, const std::optional<parser::KindSelector> &);
  ParamValue GetParamValue(const parser::TypeParamValue &);
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
  void PushScope();
  void PopScope();
  void ClearScopes() { implicitRules_.reset(); }

private:
  // implicit rules in effect for current scope
  std::unique_ptr<ImplicitRules> implicitRules_;
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
class ArraySpecVisitor : public virtual BaseVisitor {
public:
  bool Pre(const parser::ArraySpec &);
  void Post(const parser::AttrSpec &) { PostAttrSpec(); }
  void Post(const parser::ComponentAttrSpec &) { PostAttrSpec(); }
  void Post(const parser::DeferredShapeSpecList &);
  void Post(const parser::AssumedShapeSpec &);
  void Post(const parser::ExplicitShapeSpec &);
  void Post(const parser::AssumedImpliedSpec &);
  void Post(const parser::AssumedRankSpec &);

protected:
  const ArraySpec &arraySpec();
  void BeginArraySpec();
  void EndArraySpec();
  void ClearArraySpec() { arraySpec_.clear(); }

private:
  // arraySpec_ is populated by any ArraySpec
  ArraySpec arraySpec_;
  // When an ArraySpec is under an AttrSpec or ComponentAttrSpec, it is moved
  // into attrArraySpec_
  ArraySpec attrArraySpec_;

  void PostAttrSpec();
  Bound GetBound(const parser::SpecificationExpr &);
};

// Manage a stack of Scopes
class ScopeHandler : public ImplicitRulesVisitor {
public:
  Scope &currScope() { return *currScope_; }
  // The enclosing scope, skipping blocks and derived types.
  Scope &InclusiveScope();
  // The global scope, containing program units.
  Scope &GlobalScope();

  // Create a new scope and push it on the scope stack.
  void PushScope(Scope::Kind kind, Symbol *symbol);
  void PushScope(Scope &scope);
  void PopScope();
  void ClearScopes() {
    PopScope();  // trigger ConvertToObjectEntity calls
    currScope_ = &context().globalScope();
    ImplicitRulesVisitor::ClearScopes();
  }

  template<typename T> bool Pre(const parser::Statement<T> &x) {
    messageHandler().set_currStmtSource(&x.source);
    currScope_->AddSourceRange(x.source);
    return true;
  }
  template<typename T> void Post(const parser::Statement<T> &) {
    messageHandler().set_currStmtSource(nullptr);
  }

  // Special messages: already declared; about a type; two names & locations
  void SayAlreadyDeclared(const parser::Name &, const Symbol &);
  void SayDerivedType(const SourceName &, MessageFixedText &&, const Scope &);
  void Say2(const parser::Name &, MessageFixedText &&, const Symbol &,
      MessageFixedText &&);

  // Search for symbol by name in current and containing scopes
  Symbol *FindSymbol(const parser::Name &);
  Symbol *FindSymbol(const Scope &, const parser::Name &);
  // Search for name only in scope, not in enclosing scopes.
  Symbol *FindInScope(const Scope &, const parser::Name &);
  Symbol *FindInScope(const Scope &, const SourceName &);
  void EraseSymbol(const parser::Name &);
  // Record that name resolved to symbol
  Symbol *Resolve(const parser::Name &, Symbol *);
  Symbol &Resolve(const parser::Name &, Symbol &);
  // Make a new symbol with the name and attrs of an existing one
  Symbol &CopySymbol(const Symbol &);

  // Make symbols in the current or named scope
  Symbol &MakeSymbol(Scope &, const SourceName &, Attrs);
  Symbol &MakeSymbol(const parser::Name &, Attrs = Attrs{});

  template<typename D>
  Symbol &MakeSymbol(const parser::Name &name, D &&details) {
    return MakeSymbol(name, Attrs{}, std::move(details));
  }

  template<typename D>
  Symbol &MakeSymbol(
      const parser::Name &name, const Attrs &attrs, D &&details) {
    // Note: don't use FindSymbol here. If this is a derived type scope,
    // we want to detect if the name is already declared as a component.
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
              &currScope().MakeSymbol(name.source, attrs, std::move(details));
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
      // replace the old symbols with a new one with correct details
      EraseSymbol(name);
      return MakeSymbol(name, attrs, details);
    }
  }

protected:
  // When subpNamesOnly_ is set we are only collecting procedure names.
  // Create symbols with SubprogramNameDetails of the given kind.
  std::optional<SubprogramKind> subpNamesOnly_;

  // Apply the implicit type rules to this symbol.
  void ApplyImplicitRules(Symbol &);
  const DeclTypeSpec *GetImplicitType(Symbol &);
  bool ConvertToObjectEntity(Symbol &);
  bool ConvertToProcEntity(Symbol &);

  // Walk the ModuleSubprogramPart or InternalSubprogramPart collecting names.
  template<typename T>
  void WalkSubprogramPart(const std::optional<T> &subpPart) {
    if (subpPart) {
      if (std::is_same_v<T, parser::ModuleSubprogramPart>) {
        subpNamesOnly_ = SubprogramKind::Module;
      } else if (std::is_same_v<T, parser::InternalSubprogramPart>) {
        subpNamesOnly_ = SubprogramKind::Internal;
      } else {
        static_assert("unexpected type");
      }
      Walk(*subpPart);
      subpNamesOnly_ = std::nullopt;
    }
  }

private:
  Scope *currScope_{nullptr};
};

class ModuleVisitor : public virtual ScopeHandler {
public:
  bool Pre(const parser::Module &);
  void Post(const parser::Module &);
  bool Pre(const parser::Submodule &);
  void Post(const parser::Submodule &);
  bool Pre(const parser::AccessStmt &);
  bool Pre(const parser::Only &);
  bool Pre(const parser::Rename::Names &);
  bool Pre(const parser::UseStmt &);
  void Post(const parser::UseStmt &);

private:
  // The default access spec for this module.
  Attr defaultAccess_{Attr::PUBLIC};
  // The location of the last AccessStmt without access-ids, if any.
  const SourceName *prevAccessStmt_{nullptr};
  // The scope of the module during a UseStmt
  const Scope *useModuleScope_{nullptr};

  void SetAccess(const parser::Name &, Attr);
  void ApplyDefaultAccess();
  void AddUse(const parser::Rename::Names &);
  void AddUse(const parser::Name &);
  // Record a use from useModuleScope_ of useName as localName. location is
  // where it occurred (either the module or the rename) for error reporting.
  void AddUse(const SourceName &, const parser::Name &, const parser::Name &);
  void AddUse(const SourceName &, const Symbol &, Symbol &);
  Symbol &BeginModule(const parser::Name &, bool isSubmodule,
      const std::optional<parser::ModuleSubprogramPart> &);
  Scope *FindModule(const parser::Name &, Scope *ancestor = nullptr);
};

class InterfaceVisitor : public virtual ScopeHandler {
public:
  bool Pre(const parser::InterfaceStmt &);
  void Post(const parser::InterfaceStmt &);
  void Post(const parser::EndInterfaceStmt &);
  bool Pre(const parser::GenericSpec &);
  bool Pre(const parser::TypeBoundGenericStmt &);
  void Post(const parser::TypeBoundGenericStmt &);
  bool Pre(const parser::ProcedureStmt &);
  void Post(const parser::GenericStmt &);

  bool inInterfaceBlock() const { return inInterfaceBlock_; }
  bool isGeneric() const { return genericName_ != nullptr; }
  bool isAbstract() const { return isAbstract_; }

protected:
  GenericDetails &GetGenericDetails();
  // Add to generic the symbol for the subprogram with the same name
  void CheckGenericProcedures(Symbol &);

private:
  bool inInterfaceBlock_{false};  // set when in interface block
  bool isAbstract_{false};  // set when in abstract interface block
  const parser::Name *genericName_{nullptr};  // set in generic interface block

  void ResolveSpecificsInGeneric(Symbol &generic);
};

class SubprogramVisitor : public virtual ScopeHandler, public InterfaceVisitor {
public:
  bool HandleStmtFunction(const parser::StmtFunctionStmt &);
  void Post(const parser::StmtFunctionStmt &);
  bool Pre(const parser::SubroutineStmt &);
  void Post(const parser::SubroutineStmt &);
  bool Pre(const parser::FunctionStmt &);
  void Post(const parser::FunctionStmt &);
  bool Pre(const parser::SubroutineSubprogram &);
  void Post(const parser::SubroutineSubprogram &);
  bool Pre(const parser::FunctionSubprogram &);
  void Post(const parser::FunctionSubprogram &);
  bool Pre(const parser::InterfaceBody::Subroutine &);
  void Post(const parser::InterfaceBody::Subroutine &);
  bool Pre(const parser::InterfaceBody::Function &);
  void Post(const parser::InterfaceBody::Function &);
  bool Pre(const parser::SeparateModuleSubprogram &);
  void Post(const parser::SeparateModuleSubprogram &);
  bool Pre(const parser::Suffix &);

protected:
  // Set when we see a stmt function that is really an array element assignment
  bool badStmtFuncFound_{false};

private:
  // Function result name from parser::Suffix, if any.
  const parser::Name *funcResultName_{nullptr};

  bool BeginSubprogram(const parser::Name &, Symbol::Flag, bool hasModulePrefix,
      const std::optional<parser::InternalSubprogramPart> &);
  void EndSubprogram();
  // Create a subprogram symbol in the current scope and push a new scope.
  Symbol &PushSubprogramScope(const parser::Name &, Symbol::Flag);
  Symbol *GetSpecificFromGeneric(const parser::Name &);
  SubprogramDetails &PostSubprogramStmt(const parser::Name &);
};

class DeclarationVisitor : public ArraySpecVisitor,
                           public virtual ScopeHandler {
public:
  using ArraySpecVisitor::Post;
  using ArraySpecVisitor::Pre;

  void Post(const parser::EntityDecl &);
  void Post(const parser::ObjectDecl &);
  void Post(const parser::PointerDecl &);

  bool Pre(const parser::BindStmt &) { return BeginAttrs(); }
  void Post(const parser::BindStmt &) { EndAttrs(); }
  bool Pre(const parser::BindEntity &);
  void Post(const parser::NamedConstantDef &);
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
  bool Pre(const parser::TypeDeclarationStmt &) { return BeginDecl(); }
  void Post(const parser::TypeDeclarationStmt &) { EndDecl(); }
  void Post(const parser::DeclarationTypeSpec::Class &);
  bool Pre(const parser::DeclarationTypeSpec::Record &);
  bool Pre(const parser::DerivedTypeSpec &);
  void Post(const parser::DerivedTypeSpec &);
  void Post(const parser::DerivedTypeDef &x);
  bool Pre(const parser::DerivedTypeStmt &x);
  void Post(const parser::DerivedTypeStmt &x);
  bool Pre(const parser::TypeParamDefStmt &x) { return BeginDecl(); }
  void Post(const parser::TypeParamDefStmt &);
  bool Pre(const parser::TypeAttrSpec::Extends &x);
  bool Pre(const parser::PrivateStmt &x);
  bool Pre(const parser::SequenceStmt &x);
  bool Pre(const parser::ComponentDefStmt &) { return BeginDecl(); }
  void Post(const parser::ComponentDefStmt &) { EndDecl(); }
  void Post(const parser::ComponentDecl &x);
  bool Pre(const parser::ProcedureDeclarationStmt &);
  void Post(const parser::ProcedureDeclarationStmt &);
  bool Pre(const parser::ProcComponentDefStmt &);
  void Post(const parser::ProcComponentDefStmt &);
  void Post(const parser::ProcInterface &x);
  void Post(const parser::ProcDecl &x);
  bool Pre(const parser::TypeBoundProcedurePart &);
  bool Pre(const parser::TypeBoundProcBinding &) { return BeginAttrs(); }
  void Post(const parser::TypeBoundProcBinding &) { EndAttrs(); }
  void Post(const parser::TypeBoundProcedureStmt::WithoutInterface &);
  void Post(const parser::TypeBoundProcedureStmt::WithInterface &);
  void Post(const parser::FinalProcedureStmt &);
  bool Pre(const parser::AllocateStmt &);
  void Post(const parser::AllocateStmt &);
  bool Pre(const parser::StructureConstructor &);
  void Post(const parser::StructureConstructor &);

protected:
  bool BeginDecl();
  void EndDecl();
  // Declare a construct or statement entity. If there isn't a type specified
  // it comes from the entity in the containing scope, or implicit rules.
  // Return pointer to the new symbol, or nullptr on error.
  Symbol *DeclareConstructEntity(const parser::Name &);
  bool CheckUseError(const parser::Name &);

private:
  // The attribute corresponding to the statement containing an ObjectDecl
  std::optional<Attr> objectDeclAttr_;
  // Info about current derived type while walking DerivedTypeStmt
  struct {
    const parser::Name *extends{nullptr};  // EXTENDS(name)
    bool privateComps{false};  // components are private by default
    bool privateBindings{false};  // bindings are private by default
    bool sawContains{false};  // currently processing bindings
    bool sequence{false};  // is a sequence type
  } derivedTypeInfo_;
  // In a ProcedureDeclarationStmt or ProcComponentDefStmt, this is
  // the interface name, if any.
  const parser::Name *interfaceName_{nullptr};

  bool HandleAttributeStmt(Attr, const std::list<parser::Name> &);
  Symbol &HandleAttributeStmt(Attr, const parser::Name &);
  Symbol &DeclareUnknownEntity(const parser::Name &, Attrs);
  Symbol &DeclareObjectEntity(const parser::Name &, Attrs);
  Symbol &DeclareProcEntity(const parser::Name &, Attrs, const ProcInterface &);
  void SetType(const parser::Name &, const DeclTypeSpec &);
  const Symbol *ResolveDerivedType(const parser::Name * = nullptr);
  bool CanBeTypeBoundProc(const Symbol &);
  Symbol *FindExplicitInterface(const parser::Name &);
  bool MakeTypeSymbol(const parser::Name &, Details &&);
  bool OkToAddComponent(const parser::Name &, bool isParentComp = false);

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
      symbol.set_details(T{details});
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
  void Post(const parser::ConcurrentHeader &);
  bool Pre(const parser::LocalitySpec::Local &);
  bool Pre(const parser::LocalitySpec::LocalInit &);
  bool Pre(const parser::LocalitySpec::Shared &);
  bool Pre(const parser::DataImpliedDo &);
  bool Pre(const parser::DataStmt &);
  void Post(const parser::DataStmt &);
  bool Pre(const parser::DoConstruct &);
  void Post(const parser::DoConstruct &);
  void Post(const parser::ConcurrentControl &);
  bool Pre(const parser::ForallConstruct &);
  void Post(const parser::ForallConstruct &);
  bool Pre(const parser::ForallStmt &);
  void Post(const parser::ForallStmt &);
  bool Pre(const parser::BlockStmt &);
  bool Pre(const parser::EndBlockStmt &);

  // Definitions of construct names
  bool Pre(const parser::WhereConstructStmt &x) { return CheckDef(x.t); }
  bool Pre(const parser::ForallConstructStmt &x) { return CheckDef(x.t); }
  bool Pre(const parser::AssociateStmt &x) { return CheckDef(x.t); }
  bool Pre(const parser::ChangeTeamStmt &x) { return CheckDef(x.t); }
  bool Pre(const parser::CriticalStmt &x) { return CheckDef(x.t); }
  bool Pre(const parser::LabelDoStmt &x) {
    CHECK(false);
    return false;
  }
  bool Pre(const parser::NonLabelDoStmt &x) { return CheckDef(x.t); }
  bool Pre(const parser::IfThenStmt &x) { return CheckDef(x.t); }
  bool Pre(const parser::SelectCaseStmt &x) { return CheckDef(x.t); }
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
  void Post(const parser::EndAssociateStmt &x) { CheckRef(x.v); }
  void Post(const parser::EndChangeTeamStmt &x) { CheckRef(x.t); }
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
  template<typename T> bool CheckDef(const T &t) {
    return CheckDef(std::get<std::optional<parser::Name>>(t));
  }
  template<typename T> void CheckRef(const T &t) {
    CheckRef(std::get<std::optional<parser::Name>>(t));
  }
  bool CheckDef(const std::optional<parser::Name> &);
  void CheckRef(const std::optional<parser::Name> &);
  void CheckIntegerType(const Symbol &);
};

// Walk the parse tree and resolve names to symbols.
class ResolveNamesVisitor : public virtual ScopeHandler,
                            public ModuleVisitor,
                            public SubprogramVisitor,
                            public ConstructVisitor {
public:
  using ArraySpecVisitor::Post;
  using ArraySpecVisitor::Pre;
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

  bool Pre(const parser::CommonBlockObject &);
  void Post(const parser::CommonBlockObject &);
  bool Pre(const parser::PrefixSpec &);
  void Post(const parser::SpecificationPart &);
  bool Pre(const parser::MainProgram &);
  void Post(const parser::EndProgramStmt &);
  void Post(const parser::Program &);
  bool Pre(const parser::ImplicitStmt &);
  void Post(const parser::PointerObject &);
  void Post(const parser::AllocateObject &);
  void Post(const parser::PointerAssignmentStmt &);
  void Post(const parser::Designator &);
  template<typename T> void Post(const parser::LoopBounds<T> &);
  void Post(const parser::ProcComponentRef &);
  void Post(const parser::ProcedureDesignator &);
  bool Pre(const parser::FunctionReference &);
  void Post(const parser::FunctionReference &);
  bool Pre(const parser::CallStmt &);
  void Post(const parser::CallStmt &);
  bool Pre(const parser::ImportStmt &);
  void Post(const parser::TypeGuardStmt &);
  bool Pre(const parser::StmtFunctionStmt &);

private:
  // Kind of procedure we are expecting to see in a ProcedureDesignator
  std::optional<Symbol::Flag> expectedProcFlag_;
  const SourceName *prevImportStmt_{nullptr};

  // Each of these returns a pointer to a resolved Name (i.e. with symbol)
  // or nullptr in case of error.
  const parser::Name *ResolveStructureComponent(
      const parser::StructureComponent &);
  const parser::Name *ResolveArrayElement(const parser::ArrayElement &);
  const parser::Name *ResolveCoindexedNamedObject(
      const parser::CoindexedNamedObject &);
  const parser::Name *ResolveDataRef(const parser::DataRef &);
  const parser::Name *ResolveName(const parser::Name &);
  const parser::Name *FindComponent(const parser::Name *, const parser::Name &);

  Symbol *FindComponent(const Scope &, const parser::Name &);
  bool CheckAccessibleComponent(const parser::Name &);
  void CheckImports();
  void CheckImport(const SourceName &, const SourceName &);
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
  } else if (inheritFromParent_ && parent_->context_) {
    return parent_->GetType(ch);
  } else if (ch >= 'i' && ch <= 'n') {
    return &context_->MakeIntrinsicTypeSpec(TypeCategory::Integer);
  } else if (ch >= 'a' && ch <= 'z') {
    return &context_->MakeIntrinsicTypeSpec(TypeCategory::Real);
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
      context_->Say(lo,
          "More than one implicit type specified for '%s'"_err_en_US,
          std::string(1, ch).c_str());
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

const SourceName *BaseVisitor::currStmtSource() {
  return messageHandler_.currStmtSource();
}

void BaseVisitor::set_context(SemanticsContext &context) {
  context_ = &context;
  messageHandler_.set_messages(context.messages());
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
  CHECK(attrs_);
  Attrs result{*attrs_};
  attrs_.reset();
  return result;
}
void AttrsVisitor::Post(const parser::LanguageBindingSpec &x) {
  CHECK(attrs_);
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
  CHECK(attrs_);
  attrs_->set(IntentSpecToAttr(x));
  return false;
}

// DeclTypeSpecVisitor implementation

DeclTypeSpec *DeclTypeSpecVisitor::GetDeclTypeSpec() { return declTypeSpec_; }

void DeclTypeSpecVisitor::BeginDeclTypeSpec() {
  CHECK(!expectDeclTypeSpec_);
  CHECK(!declTypeSpec_);
  expectDeclTypeSpec_ = true;
}
void DeclTypeSpecVisitor::EndDeclTypeSpec() {
  CHECK(expectDeclTypeSpec_);
  expectDeclTypeSpec_ = false;
  declTypeSpec_ = nullptr;
  derivedTypeName_ = nullptr;
}

void DeclTypeSpecVisitor::Post(const parser::TypeParamSpec &x) {
  DerivedTypeSpec &derivedTypeSpec{declTypeSpec_->derivedTypeSpec()};
  const auto &value{std::get<parser::TypeParamValue>(x.t)};
  if (const auto &keyword{std::get<std::optional<parser::Keyword>>(x.t)}) {
    derivedTypeSpec.AddParamValue(keyword->v.source, GetParamValue(value));
  } else {
    derivedTypeSpec.AddParamValue(GetParamValue(value));
  }
}

ParamValue DeclTypeSpecVisitor::GetParamValue(const parser::TypeParamValue &x) {
  return std::visit(
      common::visitors{
          [=](const parser::ScalarIntExpr &x) {
            return ParamValue{EvaluateExpr(x)};
          },
          [](const parser::Star &) { return ParamValue::Assumed(); },
          [](const parser::TypeParamValue::Deferred &) {
            return ParamValue::Deferred();
          },
      },
      x.u);
}

bool DeclTypeSpecVisitor::Pre(const parser::TypeGuardStmt &) {
  BeginDeclTypeSpec();
  return true;
}
void DeclTypeSpecVisitor::Post(const parser::TypeGuardStmt &) {
  // TODO: TypeGuardStmt
  EndDeclTypeSpec();
  declTypeSpec_ = nullptr;
  derivedTypeName_ = nullptr;
}

void DeclTypeSpecVisitor::Post(const parser::IntegerTypeSpec &x) {
  MakeIntrinsic(TypeCategory::Integer, x.v);
}
void DeclTypeSpecVisitor::Post(const parser::IntrinsicTypeSpec::Character &x) {
  CHECK(!"TODO: character");
}
void DeclTypeSpecVisitor::Post(const parser::IntrinsicTypeSpec::Logical &x) {
  MakeIntrinsic(TypeCategory::Logical, x.kind);
}
void DeclTypeSpecVisitor::Post(const parser::IntrinsicTypeSpec::Real &x) {
  MakeIntrinsic(TypeCategory::Real, x.kind);
}
void DeclTypeSpecVisitor::Post(const parser::IntrinsicTypeSpec::Complex &x) {
  MakeIntrinsic(TypeCategory::Complex, x.kind);
}
void DeclTypeSpecVisitor::Post(
    const parser::IntrinsicTypeSpec::DoublePrecision &) {
  MakeIntrinsic(
      TypeCategory::Real, context().defaultKinds().doublePrecisionKind());
}
void DeclTypeSpecVisitor::Post(
    const parser::IntrinsicTypeSpec::DoubleComplex &) {
  MakeIntrinsic(
      TypeCategory::Complex, context().defaultKinds().doublePrecisionKind());
}
void DeclTypeSpecVisitor::MakeIntrinsic(
    TypeCategory category, const std::optional<parser::KindSelector> &kind) {
  MakeIntrinsic(category, GetKindParamValue(category, kind));
}
void DeclTypeSpecVisitor::MakeIntrinsic(TypeCategory category, int kind) {
  SetDeclTypeSpec(context().MakeIntrinsicTypeSpec(category, kind));
}

void DeclTypeSpecVisitor::Post(const parser::DeclarationTypeSpec::ClassStar &) {
  SetDeclTypeSpec(
      context().globalScope().MakeDeclTypeSpec(DeclTypeSpec::ClassStar));
}
void DeclTypeSpecVisitor::Post(const parser::DeclarationTypeSpec::TypeStar &) {
  SetDeclTypeSpec(
      context().globalScope().MakeDeclTypeSpec(DeclTypeSpec::TypeStar));
}

// Check that we're expecting to see a DeclTypeSpec (and haven't seen one yet)
// and save it in declTypeSpec_.
void DeclTypeSpecVisitor::SetDeclTypeSpec(DeclTypeSpec &declTypeSpec) {
  CHECK(expectDeclTypeSpec_);
  CHECK(!declTypeSpec_);
  declTypeSpec_ = &declTypeSpec;
}
// Set both the derived type name and corresponding DeclTypeSpec.
void DeclTypeSpecVisitor::SetDeclTypeSpec(
    const parser::Name &name, DeclTypeSpec &declTypeSpec) {
  derivedTypeName_ = &name;
  SetDeclTypeSpec(declTypeSpec);
}

int DeclTypeSpecVisitor::GetKindParamValue(
    TypeCategory category, const std::optional<parser::KindSelector> &kind) {
  if (!kind) {
    return 0;
  }
  // TODO: check that we get a valid kind
  return std::visit(
      common::visitors{
          [&](const parser::ScalarIntConstantExpr &x) -> int {
            if (auto maybeExpr{EvaluateExpr(x)}) {
              return evaluate::ToInt64(*maybeExpr).value();
            } else {
              return 0;
            }
          },
          [&](const parser::KindSelector::StarSize &x) -> int {
            std::uint64_t size{x.v};
            if (category == TypeCategory::Complex) {
              size /= 2;
            }
            return size;
          },
      },
      kind->u);
}

// MessageHandler implementation

void MessageHandler::set_currStmtSource(const SourceName *source) {
  currStmtSource_ = source;
}
Message &MessageHandler::Say(MessageFixedText &&msg) {
  CHECK(currStmtSource_);
  return messages_->Say(*currStmtSource_, std::move(msg));
}
Message &MessageHandler::Say(const SourceName &name, MessageFixedText &&msg) {
  return Say(name, std::move(msg), name);
}
Message &MessageHandler::Say(const SourceName &location, MessageFixedText &&msg,
    const SourceName &arg1) {
  return messages_->Say(location, std::move(msg), arg1.ToString().c_str());
}
Message &MessageHandler::Say(const SourceName &location, MessageFixedText &&msg,
    const SourceName &arg1, const SourceName &arg2) {
  return messages_->Say(location, std::move(msg), arg1.ToString().c_str(),
      arg2.ToString().c_str());
}

// ImplicitRulesVisitor implementation

void ImplicitRulesVisitor::Post(const parser::ParameterStmt &x) {
  prevParameterStmt_ = currStmtSource();
}

bool ImplicitRulesVisitor::Pre(const parser::ImplicitStmt &x) {
  bool res = std::visit(
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
      x.u);
  prevImplicit_ = currStmtSource();
  return res;
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

void ImplicitRulesVisitor::PushScope() {
  implicitRules_ = std::make_unique<ImplicitRules>(std::move(implicitRules_));
  implicitRules_->set_context(context());
  prevImplicit_ = nullptr;
  prevImplicitNone_ = nullptr;
  prevImplicitNoneType_ = nullptr;
  prevParameterStmt_ = nullptr;
}

void ImplicitRulesVisitor::PopScope() {
  implicitRules_ = std::move(implicitRules_->parent());
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

bool ArraySpecVisitor::Pre(const parser::ArraySpec &x) {
  CHECK(arraySpec_.empty());
  return true;
}

void ArraySpecVisitor::Post(const parser::DeferredShapeSpecList &x) {
  for (int i = 0; i < x.v; ++i) {
    arraySpec_.push_back(ShapeSpec::MakeDeferred());
  }
}

void ArraySpecVisitor::Post(const parser::AssumedShapeSpec &x) {
  const auto &lb{x.v};
  arraySpec_.push_back(
      lb ? ShapeSpec::MakeAssumed(GetBound(*lb)) : ShapeSpec::MakeAssumed());
}

void ArraySpecVisitor::Post(const parser::ExplicitShapeSpec &x) {
  auto &&ub{GetBound(std::get<parser::SpecificationExpr>(x.t))};
  if (const auto &lb{std::get<std::optional<parser::SpecificationExpr>>(x.t)}) {
    arraySpec_.push_back(ShapeSpec::MakeExplicit(GetBound(*lb), std::move(ub)));
  } else {
    arraySpec_.push_back(ShapeSpec::MakeExplicit(Bound{1}, std::move(ub)));
  }
}

void ArraySpecVisitor::Post(const parser::AssumedImpliedSpec &x) {
  const auto &lb{x.v};
  arraySpec_.push_back(
      lb ? ShapeSpec::MakeImplied(GetBound(*lb)) : ShapeSpec::MakeImplied());
}

void ArraySpecVisitor::Post(const parser::AssumedRankSpec &) {
  arraySpec_.push_back(ShapeSpec::MakeAssumedRank());
}

const ArraySpec &ArraySpecVisitor::arraySpec() {
  return !arraySpec_.empty() ? arraySpec_ : attrArraySpec_;
}
void ArraySpecVisitor::BeginArraySpec() {
  CHECK(arraySpec_.empty());
  CHECK(attrArraySpec_.empty());
}
void ArraySpecVisitor::EndArraySpec() {
  CHECK(arraySpec_.empty());
  attrArraySpec_.clear();
}
void ArraySpecVisitor::PostAttrSpec() {
  if (!arraySpec_.empty()) {
    // Example: integer, dimension(<1>) :: x(<2>)
    // This saves <1> in attrArraySpec_ so we can process <2> into arraySpec_
    CHECK(attrArraySpec_.empty());
    attrArraySpec_.splice(attrArraySpec_.cbegin(), arraySpec_);
    CHECK(arraySpec_.empty());
  }
}

Bound ArraySpecVisitor::GetBound(const parser::SpecificationExpr &x) {
  return Bound{EvaluateExpr(x.v)};
}

// ScopeHandler implementation

void ScopeHandler::SayAlreadyDeclared(
    const parser::Name &name, const Symbol &prev) {
  Say2(name, "'%s' is already declared in this scoping unit"_err_en_US, prev,
      "Previous declaration of '%s'"_en_US);
}
void ScopeHandler::SayDerivedType(
    const SourceName &name, MessageFixedText &&msg, const Scope &type) {
  Say(name, std::move(msg), name, type.name())
      .Attach(type.name(), "Declaration of derived type '%s'"_en_US,
          type.name().ToString().c_str());
}
void ScopeHandler::Say2(const parser::Name &name, MessageFixedText &&msg1,
    const Symbol &symbol, MessageFixedText &&msg2) {
  Say(name.source, std::move(msg1))
      .Attach(symbol.name(), msg2, symbol.name().ToString().c_str());
}

Scope &ScopeHandler::InclusiveScope() {
  for (auto *scope{&currScope()};; scope = &scope->parent()) {
    if (scope->kind() != Scope::Kind::Block &&
        scope->kind() != Scope::Kind::DerivedType) {
      return *scope;
    }
  }
  common::die("inclusive scope not found");
}
Scope &ScopeHandler::GlobalScope() {
  for (auto *scope = currScope_; scope; scope = &scope->parent()) {
    if (scope->kind() == Scope::Kind::Global) {
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
    ImplicitRulesVisitor::PushScope();
  }
  if (kind != Scope::Kind::DerivedType) {
    if (auto *symbol{scope.symbol()}) {
      // Create a dummy symbol so we can't create another one with the same name
      // It might already be there if we previously pushed the scope.
      if (!FindInScope(scope, symbol->name())) {
        auto &newSymbol{CopySymbol(*symbol)};
        if (kind == Scope::Kind::Subprogram) {
          newSymbol.set_details(symbol->get<SubprogramDetails>());
        } else {
          newSymbol.set_details(MiscDetails{MiscDetails::Kind::ScopeName});
        }
      }
    }
  }
}
void ScopeHandler::PopScope() {
  for (auto &pair : currScope()) {
    auto &symbol{*pair.second};
    ConvertToObjectEntity(symbol);  // if not a proc by now, it is an object
  }
  if (currScope_->kind() != Scope::Kind::Block) {
    ImplicitRulesVisitor::PopScope();
  }
  currScope_ = &currScope_->parent();
}

Symbol *ScopeHandler::FindSymbol(const parser::Name &name) {
  return FindSymbol(currScope(), name);
}
Symbol *ScopeHandler::FindSymbol(const Scope &scope, const parser::Name &name) {
  return Resolve(name, scope.FindSymbol(name.source));
}

Symbol &ScopeHandler::Resolve(const parser::Name &name, Symbol &symbol) {
  return *Resolve(name, &symbol);
}
Symbol *ScopeHandler::Resolve(const parser::Name &name, Symbol *symbol) {
  if (symbol && !name.symbol) {
    name.symbol = symbol;
  }
  return symbol;
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
Symbol &ScopeHandler::MakeSymbol(const parser::Name &name, Attrs attrs) {
  return Resolve(name, MakeSymbol(currScope(), name.source, attrs));
}
Symbol &ScopeHandler::CopySymbol(const Symbol &symbol) {
  CHECK(!FindInScope(currScope(), symbol.name()));
  return MakeSymbol(currScope(), symbol.name(), symbol.attrs());
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

void ScopeHandler::EraseSymbol(const parser::Name &name) {
  currScope().erase(name.source);
  name.symbol = nullptr;
}

static bool NeedsType(const Symbol &symbol) {
  if (symbol.GetType()) {
    return false;
  }
  if (auto *details{symbol.detailsIf<ProcEntityDetails>()}) {
    if (details->interface().symbol()) {
      return false;  // the interface determines the type
    }
    if (!symbol.test(Symbol::Flag::Function)) {
      return false;  // not known to be a function
    }
  }
  return true;
}
void ScopeHandler::ApplyImplicitRules(Symbol &symbol) {
  ConvertToObjectEntity(symbol);
  if (NeedsType(symbol)) {
    if (isImplicitNoneType()) {
      Say(symbol.name(), "No explicit type declared for '%s'"_err_en_US);
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
    symbol.set_details(ObjectEntityDetails{*details});
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
    symbol.set_details(ProcEntityDetails{*details});
  } else {
    return false;
  }
  if (symbol.GetType()) {
    symbol.set(Symbol::Flag::Function);
  }
  return true;
}

// ModuleVisitor implementation

bool ModuleVisitor::Pre(const parser::Only &x) {
  std::visit(
      common::visitors{
          [&](const common::Indirection<parser::GenericSpec> &generic) {
            std::visit(
                common::visitors{
                    [&](const parser::Name &name) { AddUse(name); },
                    [](const auto &) { common::die("TODO: GenericSpec"); },
                },
                generic->u);
          },
          [&](const parser::Name &name) { AddUse(name); },
          [&](const parser::Rename &rename) {
            std::visit(
                common::visitors{
                    [&](const parser::Rename::Names &names) { AddUse(names); },
                    [&](const parser::Rename::Operators &ops) {
                      common::die("TODO: Rename::Operators");
                    },
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
                CHECK(!"TODO: Rename::Operators");
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
            localSymbol = &CopySymbol(*symbol);
          }
          AddUse(x.moduleName.source, *symbol, *localSymbol);
        }
      }
    }
  }
  useModuleScope_ = nullptr;
}

void ModuleVisitor::AddUse(const parser::Rename::Names &names) {
  const auto &useName{std::get<0>(names.t)};
  const auto &localName{std::get<1>(names.t)};
  AddUse(useName.source, useName, localName);
}
void ModuleVisitor::AddUse(const parser::Name &useName) {
  AddUse(useName.source, useName, useName);
}

void ModuleVisitor::AddUse(const SourceName &location,
    const parser::Name &localName, const parser::Name &useName) {
  if (!useModuleScope_) {
    return;  // error occurred finding module
  }
  auto *useSymbol{FindInScope(*useModuleScope_, useName)};
  if (!useSymbol) {
    Say(useName, "'%s' not found in module '%s'"_err_en_US, useName.source,
        useModuleScope_->name());
    return;
  }
  if (useSymbol->attrs().test(Attr::PRIVATE)) {
    Say(useName, "'%s' is PRIVATE in '%s'"_err_en_US, useName.source,
        useModuleScope_->name());
    return;
  }
  AddUse(location, *useSymbol, MakeSymbol(localName));
}

void ModuleVisitor::AddUse(
    const SourceName &location, const Symbol &useSymbol, Symbol &localSymbol) {
  localSymbol.attrs() = useSymbol.attrs();
  localSymbol.attrs() &= ~Attrs{Attr::PUBLIC, Attr::PRIVATE};
  localSymbol.flags() = useSymbol.flags();
  if (auto *details{localSymbol.detailsIf<UseDetails>()}) {
    // check for use-associating the same symbol again:
    if (localSymbol.GetUltimate() != useSymbol.GetUltimate()) {
      localSymbol.set_details(
          UseErrorDetails{*details}.add_occurrence(location, *useModuleScope_));
    }
  } else if (auto *details{localSymbol.detailsIf<UseErrorDetails>()}) {
    details->add_occurrence(location, *useModuleScope_);
  } else if (!localSymbol.has<UnknownDetails>()) {
    Say(location,
        "Cannot use-associate '%s'; it is already declared in this scope"_err_en_US,
        localSymbol.name())
        .Attach(localSymbol.name(), "Previous declaration of '%s'"_en_US,
            localSymbol.name().ToString().c_str());
  } else {
    localSymbol.set_details(UseDetails{location, useSymbol});
  }
}

bool ModuleVisitor::Pre(const parser::Submodule &x) {
  auto &stmt{std::get<parser::Statement<parser::SubmoduleStmt>>(x.t)};
  auto &name{std::get<parser::Name>(stmt.statement.t)};
  auto &subpPart{std::get<std::optional<parser::ModuleSubprogramPart>>(x.t)};
  auto &parentId{std::get<parser::ParentIdentifier>(stmt.statement.t)};
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
  BeginModule(name, true, subpPart);
  if (!ancestor->AddSubmodule(name.source, currScope())) {
    Say(name, "Module '%s' already has a submodule named '%s'"_err_en_US,
        ancestorName.source, name.source);
  }
  return true;
}
void ModuleVisitor::Post(const parser::Submodule &) { ClearScopes(); }

bool ModuleVisitor::Pre(const parser::Module &x) {
  // Make a symbol and push a scope for this module
  const auto &name{
      std::get<parser::Statement<parser::ModuleStmt>>(x.t).statement.v};
  auto &subpPart{std::get<std::optional<parser::ModuleSubprogramPart>>(x.t)};
  BeginModule(name, false, subpPart);
  return true;
}

void ModuleVisitor::Post(const parser::Module &) {
  ApplyDefaultAccess();
  PopScope();
  prevAccessStmt_ = nullptr;
}

Symbol &ModuleVisitor::BeginModule(const parser::Name &name, bool isSubmodule,
    const std::optional<parser::ModuleSubprogramPart> &subpPart) {
  auto &symbol{MakeSymbol(name, ModuleDetails{isSubmodule})};
  auto &details{symbol.get<ModuleDetails>()};
  PushScope(Scope::Kind::Module, &symbol);
  details.set_scope(&currScope());
  WalkSubprogramPart(subpPart);
  return symbol;
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
  inInterfaceBlock_ = true;
  isAbstract_ = std::holds_alternative<parser::Abstract>(x.u);
  return true;
}
void InterfaceVisitor::Post(const parser::InterfaceStmt &) {}

void InterfaceVisitor::Post(const parser::EndInterfaceStmt &) {
  if (genericName_) {
    if (const auto *proc{GetGenericDetails().CheckSpecific()}) {
      SayAlreadyDeclared(*genericName_, *proc);
    }
    genericName_ = nullptr;
  }
  inInterfaceBlock_ = false;
  isAbstract_ = false;
}

// Create a symbol for the name name in this GenericSpec, if any.
bool InterfaceVisitor::Pre(const parser::GenericSpec &x) {
  genericName_ = GetGenericSpecName(x);
  if (!genericName_) {
    return false;
  }
  auto *genericSymbol{FindSymbol(*genericName_)};
  if (genericSymbol) {
    if (genericSymbol->has<DerivedTypeDetails>()) {
      // A generic and derived type with same name: create a generic symbol
      // and save derived type in it.
      CHECK(genericSymbol->scope()->symbol() == genericSymbol);
      GenericDetails details;
      details.set_derivedType(*genericSymbol);
      EraseSymbol(*genericName_);
      genericSymbol = &MakeSymbol(*genericName_);
      genericSymbol->set_details(details);
    } else if (!genericSymbol->IsSubprogram()) {
      SayAlreadyDeclared(*genericName_, *genericSymbol);
      EraseSymbol(*genericName_);
      genericSymbol = nullptr;
    } else if (genericSymbol->has<UseDetails>()) {
      // copy the USEd symbol into this scope so we can modify it
      const Symbol &ultimate{genericSymbol->GetUltimate()};
      EraseSymbol(*genericName_);
      genericSymbol = &CopySymbol(ultimate);
      genericName_->symbol = genericSymbol;
      if (const auto *details{ultimate.detailsIf<GenericDetails>()}) {
        genericSymbol->set_details(GenericDetails{details->specificProcs()});
      } else if (const auto *details{ultimate.detailsIf<SubprogramDetails>()}) {
        genericSymbol->set_details(SubprogramDetails{*details});
      } else {
        common::die("unexpected kind of symbol");
      }
    }
  }
  if (!genericSymbol) {
    genericSymbol = &MakeSymbol(*genericName_);
    genericSymbol->set_details(GenericDetails{});
  }
  if (genericSymbol->has<GenericDetails>()) {
    // okay
  } else if (genericSymbol->has<SubprogramDetails>() ||
      genericSymbol->has<SubprogramNameDetails>()) {
    GenericDetails genericDetails;
    genericDetails.set_specific(*genericSymbol);
    EraseSymbol(*genericName_);
    genericSymbol = &MakeSymbol(*genericName_, genericDetails);
  } else {
    common::die("unexpected kind of symbol");
  }
  CHECK(genericName_->symbol == genericSymbol);
  return false;
}

bool InterfaceVisitor::Pre(const parser::TypeBoundGenericStmt &) {
  return true;
}
void InterfaceVisitor::Post(const parser::TypeBoundGenericStmt &) {
  // TODO: TypeBoundGenericStmt
}

bool InterfaceVisitor::Pre(const parser::ProcedureStmt &x) {
  if (!isGeneric()) {
    Say("A PROCEDURE statement is only allowed in a generic interface block"_err_en_US);
    return false;
  }
  bool expectModuleProc = std::get<parser::ProcedureStmt::Kind>(x.t) ==
      parser::ProcedureStmt::Kind::ModuleProcedure;
  for (const auto &name : std::get<std::list<parser::Name>>(x.t)) {
    GetGenericDetails().add_specificProcName(name.source, expectModuleProc);
  }
  return false;
}

void InterfaceVisitor::Post(const parser::GenericStmt &x) {
  if (auto &accessSpec{std::get<std::optional<parser::AccessSpec>>(x.t)}) {
    genericName_->symbol->attrs().set(AccessSpecToAttr(*accessSpec));
  }
  for (const auto &name : std::get<std::list<parser::Name>>(x.t)) {
    GetGenericDetails().add_specificProcName(name.source, false);
  }
}

GenericDetails &InterfaceVisitor::GetGenericDetails() {
  CHECK(genericName_);
  CHECK(genericName_->symbol);
  return genericName_->symbol->get<GenericDetails>();
}

// By now we should have seen all specific procedures referenced by name in
// this generic interface. Resolve those names to symbols.
void InterfaceVisitor::ResolveSpecificsInGeneric(Symbol &generic) {
  auto &details{generic.get<GenericDetails>()};
  std::set<SourceName> namesSeen;  // to check for duplicate names
  for (const auto *symbol : details.specificProcs()) {
    namesSeen.insert(symbol->name());
  }
  for (const auto &[name, expectModuleProc] : details.specificProcNames()) {
    const auto *symbol{currScope().FindSymbol(name)};
    if (!symbol) {
      Say(name, "Procedure '%s' not found"_err_en_US);
      continue;
    }
    if (symbol == &generic) {
      if (auto *specific{generic.get<GenericDetails>().specific()}) {
        symbol = specific;
      }
    }
    if (!symbol->has<SubprogramDetails>() &&
        !symbol->has<SubprogramNameDetails>()) {
      Say(name, "'%s' is not a subprogram"_err_en_US);
      continue;
    }
    if (expectModuleProc) {
      const auto *d{symbol->detailsIf<SubprogramNameDetails>()};
      if (!d || d->kind() != SubprogramKind::Module) {
        Say(name, "'%s' is not a module procedure"_err_en_US);
      }
    }
    if (!namesSeen.insert(name).second) {
      Say(name, "Procedure '%s' is already specified in generic '%s'"_err_en_US,
          name, generic.name());
      continue;
    }
    details.add_specificProc(symbol);
  }
  details.ClearSpecificProcNames();
}

// Check that the specific procedures are all functions or all subroutines.
// If there is a derived type with the same name they must be functions.
// Set the corresponding flag on generic.
void InterfaceVisitor::CheckGenericProcedures(Symbol &generic) {
  ResolveSpecificsInGeneric(generic);
  auto &details{generic.get<GenericDetails>()};
  auto &specifics{details.specificProcs()};
  if (specifics.empty()) {
    if (details.derivedType()) {
      generic.set(Symbol::Flag::Function);
    }
    return;
  }
  auto &firstSpecific{*specifics.front()};
  bool isFunction{firstSpecific.test(Symbol::Flag::Function)};
  for (auto *specific : specifics) {
    if (isFunction != specific->test(Symbol::Flag::Function)) {
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
    details.add_dummyArg(MakeSymbol(dummyName, dummyDetails));
  }
  EraseSymbol(name);  // added by PushSubprogramScope
  EntityDetails resultDetails;
  if (resultType) {
    resultDetails.set_type(*resultType);
  }
  details.set_result(MakeSymbol(name, resultDetails));
  return true;
}

bool SubprogramVisitor::Pre(const parser::Suffix &suffix) {
  if (suffix.resultName) {
    funcResultName_ = &suffix.resultName.value();
  }
  return true;
}

bool HasModulePrefix(const std::list<parser::PrefixSpec> &prefixes) {
  for (const auto &prefix : prefixes) {
    if (std::holds_alternative<parser::PrefixSpec::Module>(prefix.u)) {
      return true;
    }
  }
  return false;
}
bool SubprogramVisitor::Pre(const parser::SubroutineSubprogram &x) {
  const auto &stmt{
      std::get<parser::Statement<parser::SubroutineStmt>>(x.t).statement};
  bool hasModulePrefix{
      HasModulePrefix(std::get<std::list<parser::PrefixSpec>>(stmt.t))};
  const auto &name{std::get<parser::Name>(stmt.t)};
  const auto &subpPart{
      std::get<std::optional<parser::InternalSubprogramPart>>(x.t)};
  return BeginSubprogram(
      name, Symbol::Flag::Subroutine, hasModulePrefix, subpPart);
}
void SubprogramVisitor::Post(const parser::SubroutineSubprogram &) {
  EndSubprogram();
}

bool SubprogramVisitor::Pre(const parser::FunctionSubprogram &x) {
  const auto &stmt{
      std::get<parser::Statement<parser::FunctionStmt>>(x.t).statement};
  bool hasModulePrefix{
      HasModulePrefix(std::get<std::list<parser::PrefixSpec>>(stmt.t))};
  const auto &name{std::get<parser::Name>(stmt.t)};
  const auto &subpPart{
      std::get<std::optional<parser::InternalSubprogramPart>>(x.t)};
  return BeginSubprogram(
      name, Symbol::Flag::Function, hasModulePrefix, subpPart);
}
void SubprogramVisitor::Post(const parser::FunctionSubprogram &) {
  EndSubprogram();
}

bool SubprogramVisitor::Pre(const parser::InterfaceBody::Subroutine &x) {
  const auto &name{std::get<parser::Name>(
      std::get<parser::Statement<parser::SubroutineStmt>>(x.t).statement.t)};
  return BeginSubprogram(
      name, Symbol::Flag::Subroutine, /*hasModulePrefix*/ false, std::nullopt);
}
void SubprogramVisitor::Post(const parser::InterfaceBody::Subroutine &) {
  EndSubprogram();
}
bool SubprogramVisitor::Pre(const parser::InterfaceBody::Function &x) {
  const auto &name{std::get<parser::Name>(
      std::get<parser::Statement<parser::FunctionStmt>>(x.t).statement.t)};
  return BeginSubprogram(
      name, Symbol::Flag::Function, /*hasModulePrefix*/ false, std::nullopt);
}
void SubprogramVisitor::Post(const parser::InterfaceBody::Function &) {
  EndSubprogram();
}

bool SubprogramVisitor::Pre(const parser::SubroutineStmt &stmt) {
  return BeginAttrs();
}
bool SubprogramVisitor::Pre(const parser::FunctionStmt &stmt) {
  if (!subpNamesOnly_) {
    BeginDeclTypeSpec();
    CHECK(!funcResultName_);
  }
  return BeginAttrs();
}

void SubprogramVisitor::Post(const parser::SubroutineStmt &stmt) {
  const auto &name{std::get<parser::Name>(stmt.t)};
  auto &details{PostSubprogramStmt(name)};
  for (const auto &dummyArg : std::get<std::list<parser::DummyArg>>(stmt.t)) {
    const parser::Name *dummyName = std::get_if<parser::Name>(&dummyArg.u);
    CHECK(dummyName != nullptr && "TODO: alternate return indicator");
    Symbol &dummy{MakeSymbol(*dummyName, EntityDetails(true))};
    details.add_dummyArg(dummy);
  }
}

void SubprogramVisitor::Post(const parser::FunctionStmt &stmt) {
  const auto &name{std::get<parser::Name>(stmt.t)};
  auto &details{PostSubprogramStmt(name)};
  for (const auto &dummyName : std::get<std::list<parser::Name>>(stmt.t)) {
    Symbol &dummy{MakeSymbol(dummyName, EntityDetails(true))};
    details.add_dummyArg(dummy);
  }
  // add function result to function scope
  EntityDetails funcResultDetails;
  if (auto *type{GetDeclTypeSpec()}) {
    funcResultDetails.set_type(*type);
  }
  EndDeclTypeSpec();

  const parser::Name *funcResultName;
  if (funcResultName_ && funcResultName_->source != name.source) {
    funcResultName = funcResultName_;
  } else {
    EraseSymbol(name);  // was added by PushSubprogramScope
    funcResultName = &name;
  }
  details.set_result(MakeSymbol(*funcResultName, funcResultDetails));
  funcResultName_ = nullptr;
}

SubprogramDetails &SubprogramVisitor::PostSubprogramStmt(
    const parser::Name &name) {
  Symbol &symbol{*currScope().symbol()};
  CHECK(name.source == symbol.name());
  symbol.attrs() |= EndAttrs();
  if (symbol.attrs().test(Attr::MODULE)) {
    symbol.attrs().set(Attr::EXTERNAL, false);
  }
  return symbol.get<SubprogramDetails>();
}

bool SubprogramVisitor::BeginSubprogram(const parser::Name &name,
    Symbol::Flag subpFlag, bool hasModulePrefix,
    const std::optional<parser::InternalSubprogramPart> &subpPart) {
  if (subpNamesOnly_) {
    auto &symbol{MakeSymbol(name, SubprogramNameDetails{*subpNamesOnly_})};
    symbol.set(subpFlag);
    return false;
  }
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
  WalkSubprogramPart(subpPart);
  return true;
}
void SubprogramVisitor::EndSubprogram() {
  if (!subpNamesOnly_) {
    PopScope();
  }
}

bool SubprogramVisitor::Pre(const parser::SeparateModuleSubprogram &x) {
  if (subpNamesOnly_) {
    return false;
  }
  const auto &name{
      std::get<parser::Statement<parser::MpSubprogramStmt>>(x.t).statement.v};
  const auto &subpPart{
      std::get<std::optional<parser::InternalSubprogramPart>>(x.t)};
  return BeginSubprogram(
      name, Symbol::Flag::Subroutine, /*hasModulePrefix*/ true, subpPart);
}

void SubprogramVisitor::Post(const parser::SeparateModuleSubprogram &) {
  EndSubprogram();
}

Symbol &SubprogramVisitor::PushSubprogramScope(
    const parser::Name &name, Symbol::Flag subpFlag) {
  auto *symbol{GetSpecificFromGeneric(name)};
  if (!symbol) {
    symbol = &MakeSymbol(name, SubprogramDetails{});
    symbol->set(subpFlag);
  }
  PushScope(Scope::Kind::Subprogram, symbol);
  auto &details{symbol->get<SubprogramDetails>()};
  if (inInterfaceBlock()) {
    details.set_isInterface();
    if (!isAbstract()) {
      symbol->attrs().set(Attr::EXTERNAL);
    }
    if (isGeneric()) {
      GetGenericDetails().add_specificProc(symbol);
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
        name.ToString().data(), module->name().ToString().data());
  }
  return true;
}

void DeclarationVisitor::Post(const parser::DimensionStmt::Declaration &x) {
  const auto &name{std::get<parser::Name>(x.t)};
  DeclareObjectEntity(name, Attrs{});
}

void DeclarationVisitor::Post(const parser::EntityDecl &x) {
  // TODO: may be under StructureStmt
  const auto &name{std::get<parser::ObjectName>(x.t)};
  // TODO: CoarraySpec, CharLength, Initialization
  Attrs attrs{attrs_ ? *attrs_ : Attrs{}};
  Symbol &symbol{DeclareUnknownEntity(name, attrs)};
  if (auto &init{std::get<std::optional<parser::Initialization>>(x.t)}) {
    if (ConvertToObjectEntity(symbol)) {
      if (auto *expr{std::get_if<parser::ConstantExpr>(&init->u)}) {
        symbol.get<ObjectEntityDetails>().set_init(EvaluateExpr(*expr));
      }
    }
  }
}

void DeclarationVisitor::Post(const parser::PointerDecl &x) {
  const auto &name{std::get<parser::Name>(x.t)};
  DeclareUnknownEntity(name, Attrs{Attr::POINTER});
}

bool DeclarationVisitor::Pre(const parser::BindEntity &x) {
  auto &name{std::get<parser::Name>(x.t)};
  if (std::get<parser::BindEntity::Kind>(x.t) ==
      parser::BindEntity::Kind::Object) {
    HandleAttributeStmt(Attr::BIND_C, name);
  } else {
    // TODO: name is common block
  }
  return false;
}
void DeclarationVisitor::Post(const parser::NamedConstantDef &x) {
  auto &name{std::get<parser::NamedConstant>(x.t).v};
  auto &symbol{HandleAttributeStmt(Attr::PARAMETER, name)};
  if (!ConvertToObjectEntity(symbol)) {
    Say2(name, "PARAMETER attribute not allowed on '%s'"_err_en_US, symbol,
        "Declaration of '%s'"_en_US);
    return;
  }
  const auto &expr{std::get<parser::ConstantExpr>(x.t)};
  symbol.get<ObjectEntityDetails>().set_init(EvaluateExpr(expr));
  ApplyImplicitRules(symbol);
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
      Say2(name, "EXTERNAL attribute not allowed on '%s'"_err_en_US, *symbol,
          "Declaration of '%s'"_en_US);
    }
  }
  return false;
}
bool DeclarationVisitor::Pre(const parser::IntentStmt &x) {
  auto &intentSpec{std::get<parser::IntentSpec>(x.t)};
  auto &names{std::get<std::list<parser::Name>>(x.t)};
  return HandleAttributeStmt(IntentSpecToAttr(intentSpec), names);
}
bool DeclarationVisitor::Pre(const parser::IntrinsicStmt &x) {
  return HandleAttributeStmt(Attr::INTRINSIC, x.v);
}
bool DeclarationVisitor::Pre(const parser::OptionalStmt &x) {
  return HandleAttributeStmt(Attr::OPTIONAL, x.v);
}
bool DeclarationVisitor::Pre(const parser::ProtectedStmt &x) {
  return HandleAttributeStmt(Attr::PROTECTED, x.v);
}
bool DeclarationVisitor::Pre(const parser::ValueStmt &x) {
  return HandleAttributeStmt(Attr::VALUE, x.v);
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
  if (auto *symbol{FindSymbol(name)}) {
    // symbol was already there: set attribute on it
    if (attr == Attr::ASYNCHRONOUS || attr == Attr::VOLATILE) {
      // TODO: if in a BLOCK, attribute should only be set while in the block
    } else if (symbol->has<UseDetails>()) {
      Say(*currStmtSource(),
          "Cannot change %s attribute on use-associated '%s'"_err_en_US,
          EnumToString(attr), name.source);
    }
    symbol->attrs().set(attr);
    return *symbol;
  } else {
    return MakeSymbol(name, Attrs{attr});
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
  if (!arraySpec().empty()) {
    return DeclareObjectEntity(name, attrs);
  } else {
    Symbol &symbol{DeclareEntity<EntityDetails>(name, attrs)};
    if (auto *type{GetDeclTypeSpec()}) {
      SetType(name, *type);
    }
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
      symbol.set(interface.symbol()->test(Symbol::Flag::Function)
              ? Symbol::Flag::Function
              : Symbol::Flag::Subroutine);
    }
    details->set_interface(interface);
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
      if (!details->shape().empty()) {
        Say(name,
            "The dimensions of '%s' have already been declared"_err_en_US);
      } else {
        details->set_shape(arraySpec());
      }
      ClearArraySpec();
    }
  }
  return symbol;
}

void DeclarationVisitor::Post(const parser::DeclarationTypeSpec::Class &x) {
  // created by default with TypeDerived; change to ClassDerived
  GetDeclTypeSpec()->set_category(DeclTypeSpec::ClassDerived);
}

bool DeclarationVisitor::Pre(const parser::DeclarationTypeSpec::Record &) {
  return true;  // TODO
}

bool DeclarationVisitor::Pre(const parser::DerivedTypeSpec &x) {
  const auto &name{std::get<parser::Name>(x.t)};
  SetDeclTypeSpec(name,
      currScope().MakeDeclTypeSpec(DeclTypeSpec::TypeDerived, name.source));
  return true;
}
void DeclarationVisitor::Post(const parser::DerivedTypeSpec &x) {
  if (const auto *symbol{ResolveDerivedType()}) {
    GetDeclTypeSpec()->derivedTypeSpec().set_scope(*symbol->scope());
  }
}

void DeclarationVisitor::Post(const parser::DerivedTypeDef &x) {
  std::set<SourceName> paramNames;
  auto &scope{currScope()};
  auto &details{scope.symbol()->get<DerivedTypeDetails>()};
  auto &stmt{std::get<parser::Statement<parser::DerivedTypeStmt>>(x.t)};
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
    if (derivedTypeInfo_.sawContains) {
      Say(stmt.source,
          "A sequence type may not have a CONTAINS statement"_err_en_US);  // C740
    }
  }
  derivedTypeInfo_ = {};
  PopScope();
}
bool DeclarationVisitor::Pre(const parser::DerivedTypeStmt &x) {
  return BeginAttrs();
}
void DeclarationVisitor::Post(const parser::DerivedTypeStmt &x) {
  auto &name{std::get<parser::Name>(x.t)};
  auto &symbol{MakeSymbol(name, GetAttrs(), DerivedTypeDetails{})};
  PushScope(Scope::Kind::DerivedType, &symbol);
  if (auto *extendsName{derivedTypeInfo_.extends}) {
    if (auto *extends{ResolveDerivedType(extendsName)}) {
      symbol.get<DerivedTypeDetails>().set_extends(extends);
      // Declare the "parent component"; private if the type is
      if (OkToAddComponent(*extendsName, true)) {
        auto &comp{DeclareEntity<ObjectEntityDetails>(*extendsName, Attrs{})};
        comp.attrs().set(Attr::PRIVATE, extends->attrs().test(Attr::PRIVATE));
        comp.set(Symbol::Flag::ParentComp);
        auto &type{currScope().MakeDeclTypeSpec(
            DeclTypeSpec::TypeDerived, extendsName->source)};
        type.derivedTypeSpec().set_scope(currScope());
        comp.SetType(type);
      }
    }
  }
  EndAttrs();
}
void DeclarationVisitor::Post(const parser::TypeParamDefStmt &x) {
  auto *type{GetDeclTypeSpec()};
  auto attr{std::get<common::TypeParamAttr>(x.t)};
  for (auto &decl : std::get<std::list<parser::TypeParamDecl>>(x.t)) {
    auto &name{std::get<parser::Name>(decl.t)};
    auto details{TypeParamDetails{attr}};
    if (auto &init{
            std::get<std::optional<parser::ScalarIntConstantExpr>>(decl.t)}) {
      details.set_init(EvaluateExpr(*init));
    }
    if (MakeTypeSymbol(name, std::move(details))) {
      SetType(name, *type);
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
  if (OkToAddComponent(name)) {
    auto &symbol{DeclareObjectEntity(name, attrs)};
    if (auto *details{symbol.detailsIf<ObjectEntityDetails>()}) {
      if (auto &init{std::get<std::optional<parser::Initialization>>(x.t)}) {
        if (auto *initExpr{std::get_if<parser::ConstantExpr>(&init->u)}) {
          details->set_init(EvaluateExpr(*initExpr));
        }
      }
    }
  }
  ClearArraySpec();
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
void DeclarationVisitor::Post(const parser::ProcInterface &x) {
  if (auto *name{std::get_if<parser::Name>(&x.u)}) {
    interfaceName_ = name;
  }
}

void DeclarationVisitor::Post(const parser::ProcDecl &x) {
  ProcInterface interface;
  if (interfaceName_) {
    if (auto *symbol{FindExplicitInterface(*interfaceName_)}) {
      interface.set_symbol(*symbol);
    }
  } else if (auto *type{GetDeclTypeSpec()}) {
    interface.set_type(*type);
  }
  auto attrs{GetAttrs()};
  if (currScope().kind() != Scope::Kind::DerivedType) {
    attrs.set(Attr::EXTERNAL);
  }
  const auto &name{std::get<parser::Name>(x.t)};
  DeclareProcEntity(name, attrs, interface);
}

bool DeclarationVisitor::Pre(const parser::TypeBoundProcedurePart &x) {
  derivedTypeInfo_.sawContains = true;
  return true;
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
      Say2(procedureName,
          "'%s' is not a module procedure or external procedure"
          " with explicit interface"_err_en_US,
          *procedure, "Declaration of '%s'"_en_US);
      continue;
    }
    MakeTypeSymbol(bindingName, ProcBindingDetails{*procedure});
  }
}

void DeclarationVisitor::Post(
    const parser::TypeBoundProcedureStmt::WithInterface &x) {
  if (!GetAttrs().test(Attr::DEFERRED)) {  // C783
    Say("DEFERRED is required when an interface-name is provided"_err_en_US);
  }
  Symbol *interface{FindExplicitInterface(x.interfaceName)};
  if (!interface) {
    return;
  }
  for (auto &bindingName : x.bindingNames) {
    MakeTypeSymbol(bindingName, ProcBindingDetails{*interface});
  }
}

void DeclarationVisitor::Post(const parser::FinalProcedureStmt &x) {
  for (auto &name : x.v) {
    MakeTypeSymbol(name, FinalProcDetails{});
  }
}

bool DeclarationVisitor::Pre(const parser::AllocateStmt &) {
  BeginDeclTypeSpec();
  return true;
}
void DeclarationVisitor::Post(const parser::AllocateStmt &) {
  ResolveDerivedType();
  EndDeclTypeSpec();
}

bool DeclarationVisitor::Pre(const parser::StructureConstructor &) {
  BeginDeclTypeSpec();
  return true;
}
void DeclarationVisitor::Post(const parser::StructureConstructor &) {
  ResolveDerivedType();
  EndDeclTypeSpec();
}

Symbol *DeclarationVisitor::DeclareConstructEntity(const parser::Name &name) {
  auto *prev{FindSymbol(name)};
  if (prev) {
    if (prev->owner().kind() == Scope::Kind::Forall ||
        prev->owner() == currScope()) {
      SayAlreadyDeclared(name, *prev);
      return nullptr;
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
    return nullptr;
  } else if (auto *type{prev->GetType()}) {
    symbol.SetType(*type);
  }
  return &symbol;
}

// Set the type of an entity or report an error.
void DeclarationVisitor::SetType(
    const parser::Name &name, const DeclTypeSpec &type) {
  CHECK(name.symbol);
  auto &symbol{*name.symbol};
  auto *prevType{symbol.GetType()};
  if (!prevType) {
    symbol.SetType(type);
  } else if (!symbol.test(Symbol::Flag::Implicit)) {
    Say2(name, "The type of '%s' has already been declared"_err_en_US, symbol,
        "Declaration of '%s'"_en_US);
  } else if (type != *prevType) {
    Say2(name,
        "The type of '%s' has already been implicitly declared"_err_en_US,
        symbol, "Declaration of '%s'"_en_US);
  } else {
    symbol.set(Symbol::Flag::Implicit, false);
  }
}

// Find the Symbol for this derived type; derivedTypeName if not specified.
const Symbol *DeclarationVisitor::ResolveDerivedType(const parser::Name *name) {
  if (name == nullptr) {
    name = derivedTypeName();
    if (name == nullptr) {
      return nullptr;
    }
  }
  const auto *symbol{FindSymbol(*name)};
  if (!symbol) {
    Say(*name, "Derived type '%s' not found"_err_en_US);
    return nullptr;
  }
  if (CheckUseError(*name)) {
    return nullptr;
  }
  if (auto *details{symbol->detailsIf<UseDetails>()}) {
    symbol = &details->symbol();
  }
  if (auto *details{symbol->detailsIf<GenericDetails>()}) {
    if (details->derivedType()) {
      symbol = details->derivedType();
    }
  }
  if (!symbol->has<DerivedTypeDetails>()) {
    Say(*name, "'%s' is not a derived type"_err_en_US);
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

Symbol *DeclarationVisitor::FindExplicitInterface(const parser::Name &name) {
  auto *symbol{FindSymbol(name)};
  if (!symbol) {
    Say(name, "Explicit interface '%s' not found"_err_en_US);
  } else if (!symbol->HasExplicitInterface()) {
    Say2(name,
        "'%s' is not an abstract interface or a procedure with an"
        " explicit interface"_err_en_US,
        *symbol, "Declaration of '%s'"_en_US);
    symbol = nullptr;
  }
  return symbol;
}

// Create a symbol for a type parameter, component, or procedure binding in
// the current derived type scope. Return false on error.
bool DeclarationVisitor::MakeTypeSymbol(
    const parser::Name &name, Details &&details) {
  Scope &derivedType{currScope()};
  CHECK(derivedType.kind() == Scope::Kind::DerivedType);
  if (auto *symbol{FindInScope(derivedType, name)}) {
    Say2(name,
        "Type parameter, component, or procedure binding '%s'"
        " already defined in this type"_err_en_US,
        *symbol, "Previous definition of '%s'"_en_US);
    return false;
  } else {
    auto attrs{GetAttrs()};
    // Apply binding-private-stmt if present and this is a procedure binding
    if (derivedTypeInfo_.privateBindings &&
        !attrs.HasAny({Attr::PUBLIC, Attr::PRIVATE}) &&
        std::holds_alternative<ProcBindingDetails>(details)) {
      attrs.set(Attr::PRIVATE);
    }
    MakeSymbol(name, attrs, details);
    return true;
  }
}

// Return true if it is ok to declare this component in the current scope.
// Otherwise, emit an error and return false.
bool DeclarationVisitor::OkToAddComponent(
    const parser::Name &name, bool isParentComp) {
  const Scope *scope{&currScope()};
  for (bool inParent{false};; inParent = true) {
    CHECK(scope->kind() == Scope::Kind::DerivedType);
    if (auto *prev{FindInScope(*scope, name)}) {
      auto msg{""_en_US};
      if (isParentComp) {
        msg = "Type cannot be extended as it has a component named"
              " '%s'"_err_en_US;
      } else if (prev->test(Symbol::Flag::ParentComp)) {
        msg = "'%s' is a parent type of this type and so cannot be"
              " a component"_err_en_US;
      } else if (inParent) {
        msg = "Component '%s' is already declared in a parent of this"
              " derived type"_err_en_US;
      } else {
        msg = "Component '%s' is already declared in this"
              " derived type"_err_en_US;
      }
      Say2(name, std::move(msg), *prev, "Previous declaration of '%s'"_en_US);
      return false;
    }
    auto *extends{scope->symbol()->get<DerivedTypeDetails>().extends()};
    if (!extends) {
      return true;
    }
    scope = extends->scope();
  }
}

// ConstructVisitor implementation

bool ConstructVisitor::Pre(const parser::ConcurrentHeader &) {
  BeginDeclTypeSpec();
  return true;
}
void ConstructVisitor::Post(const parser::ConcurrentHeader &) {
  EndDeclTypeSpec();
}

bool ConstructVisitor::Pre(const parser::LocalitySpec::Local &x) {
  for (auto &name : x.v) {
    if (auto *symbol{DeclareConstructEntity(name)}) {
      symbol->set(Symbol::Flag::LocalityLocal);
    }
  }
  return false;
}
bool ConstructVisitor::Pre(const parser::LocalitySpec::LocalInit &x) {
  for (auto &name : x.v) {
    if (auto *symbol{DeclareConstructEntity(name)}) {
      symbol->set(Symbol::Flag::LocalityLocalInit);
    }
  }
  return false;
}
bool ConstructVisitor::Pre(const parser::LocalitySpec::Shared &x) {
  for (auto &name : x.v) {
    if (auto *prev{FindSymbol(name)}) {
      if (prev->owner() == currScope()) {
        SayAlreadyDeclared(name, *prev);
      }
      auto &symbol{MakeSymbol(name, HostAssocDetails{*prev})};
      symbol.set(Symbol::Flag::LocalityShared);
    } else {
      Say(name, "Variable '%s' not found"_err_en_US);
    }
  }
  return false;
}

bool ConstructVisitor::Pre(const parser::DataImpliedDo &x) {
  auto &objects{std::get<std::list<parser::DataIDoObject>>(x.t)};
  auto &type{std::get<std::optional<parser::IntegerTypeSpec>>(x.t)};
  auto &bounds{
      std::get<parser::LoopBounds<parser::ScalarIntConstantExpr>>(x.t)};
  if (type) {
    BeginDeclTypeSpec();
    DeclTypeSpecVisitor::Post(*type);
  }
  if (auto *symbol{DeclareConstructEntity(bounds.name.thing.thing)}) {
    CheckIntegerType(*symbol);
  }
  if (type) {
    EndDeclTypeSpec();
  }
  Walk(bounds);
  Walk(objects);
  return false;
}

bool ConstructVisitor::Pre(const parser::DataStmt &) {
  PushScope(Scope::Kind::Block, nullptr);
  return true;
}
void ConstructVisitor::Post(const parser::DataStmt &) { PopScope(); }

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

void ConstructVisitor::Post(const parser::ConcurrentControl &x) {
  auto &name{std::get<parser::Name>(x.t)};
  if (auto *symbol{DeclareConstructEntity(name)}) {
    CheckIntegerType(*symbol);
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

void ConstructVisitor::CheckIntegerType(const Symbol &symbol) {
  if (auto *type{symbol.GetType()}) {
    if (type->category() != DeclTypeSpec::Intrinsic ||
        type->intrinsicTypeSpec().category() != TypeCategory::Integer) {
      Say(symbol.name(), "Variable '%s' is not scalar integer"_err_en_US);
    }
  }
}

// ResolveNamesVisitor implementation

bool ResolveNamesVisitor::Pre(const parser::CommonBlockObject &x) {
  BeginArraySpec();
  return true;
}
void ResolveNamesVisitor::Post(const parser::CommonBlockObject &x) {
  ClearArraySpec();
  // TODO: CommonBlockObject
}

bool ResolveNamesVisitor::Pre(const parser::PrefixSpec &x) {
  return true;  // TODO
}

bool ResolveNamesVisitor::Pre(const parser::FunctionReference &) {
  expectedProcFlag_ = Symbol::Flag::Function;
  return true;
}
void ResolveNamesVisitor::Post(const parser::FunctionReference &) {
  expectedProcFlag_ = std::nullopt;
}
bool ResolveNamesVisitor::Pre(const parser::CallStmt &) {
  expectedProcFlag_ = Symbol::Flag::Subroutine;
  return true;
}
void ResolveNamesVisitor::Post(const parser::CallStmt &) {
  expectedProcFlag_ = std::nullopt;
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
    if (scope.parent().kind() == Scope::Kind::Global) {
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

const parser::Name *ResolveNamesVisitor::ResolveStructureComponent(
    const parser::StructureComponent &x) {
  return FindComponent(ResolveDataRef(x.base), x.component);
}

const parser::Name *ResolveNamesVisitor::ResolveArrayElement(
    const parser::ArrayElement &x) {
  // TODO: need to resolve these
  // for (auto &subscript : x.subscripts) {
  //  ResolveSectionSubscript(subscript);
  //}
  return ResolveDataRef(x.base);
}

const parser::Name *ResolveNamesVisitor::ResolveCoindexedNamedObject(
    const parser::CoindexedNamedObject &x) {
  return nullptr;  // TODO
}

const parser::Name *ResolveNamesVisitor::ResolveDataRef(
    const parser::DataRef &x) {
  return std::visit(
      common::visitors{
          [=](const parser::Name &y) { return ResolveName(y); },
          [=](const common::Indirection<parser::StructureComponent> &y) {
            return ResolveStructureComponent(*y);
          },
          [=](const common::Indirection<parser::ArrayElement> &y) {
            return ResolveArrayElement(*y);
          },
          [=](const common::Indirection<parser::CoindexedNamedObject> &y) {
            return ResolveCoindexedNamedObject(*y);
          },
      },
      x.u);
}

// If implicit types are allowed, ensure name is in the symbol table.
// Otherwise, report an error if it hasn't been declared.
const parser::Name *ResolveNamesVisitor::ResolveName(const parser::Name &name) {
  if (FindSymbol(name)) {
    if (CheckUseError(name)) {
      return nullptr;  // reported an error
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
  ApplyImplicitRules(*symbol);
  return &name;
}

// base is a part-ref of a derived type; find the named component in its type.
const parser::Name *ResolveNamesVisitor::FindComponent(
    const parser::Name *base, const parser::Name &component) {
  if (!base || !base->symbol) {
    return nullptr;
  }
  auto &symbol{*base->symbol};
  if (!ConvertToObjectEntity(symbol)) {
    Say2(*base, "'%s' is an invalid base for a component reference"_err_en_US,
        symbol, "Declaration of '%s'"_en_US);
    return nullptr;
  }
  auto *type{symbol.GetType()};
  if (!type) {
    return nullptr;  // should have already reported error
  }
  if (type->category() == DeclTypeSpec::Intrinsic &&
      type->intrinsicTypeSpec().category() == TypeCategory::Complex) {
    auto name{component.ToString()};
    if (name == "re" || name == "im") {
      return nullptr;  // complex-part-designator, not structure-component
    }
  }
  if (type->category() != DeclTypeSpec::TypeDerived) {
    if (symbol.test(Symbol::Flag::Implicit)) {
      Say(*base,
          "'%s' is not an object of derived type; it is implicitly typed"_err_en_US);
    } else {
      Say2(*base, "'%s' is not an object of derived type"_err_en_US, symbol,
          "Declaration of '%s'"_en_US);
    }
    return nullptr;
  }
  const Scope *scope{type->derivedTypeSpec().scope()};
  if (!scope) {
    return nullptr;  // previously failed to resolve type
  }
  auto *result{FindComponent(*scope, component)};
  if (!result) {
    SayDerivedType(component.source,
        "Component '%s' not found in derived type '%s'"_err_en_US, *scope);
    return nullptr;
  } else if (!CheckAccessibleComponent(component)) {
    return nullptr;
  } else {
    return &component;
  }
}

// Check that component is accessible from current scope.
bool ResolveNamesVisitor::CheckAccessibleComponent(
    const parser::Name &component) {
  CHECK(component.symbol);
  auto &symbol{*component.symbol};
  if (!symbol.attrs().test(Attr::PRIVATE)) {
    return true;
  }
  CHECK(symbol.owner().kind() == Scope::Kind::DerivedType);
  // component must be in a module/submodule because of PRIVATE:
  const Scope &moduleScope{symbol.owner().parent()};
  CHECK(moduleScope.kind() == Scope::Kind::Module);
  for (auto *scope{&currScope()}; scope->kind() != Scope::Kind::Global;
       scope = &scope->parent()) {
    if (scope == &moduleScope) {
      return true;
    }
  }
  Say(component,
      "PRIVATE component '%s' is only accessible within module '%s'"_err_en_US,
      component.ToString(), moduleScope.name());
  return false;
}

// Look in this type's scope and then its parents for component.
Symbol *ResolveNamesVisitor::FindComponent(
    const Scope &type, const parser::Name &component) {
  CHECK(type.kind() == Scope::Kind::DerivedType);
  if (auto *symbol{FindInScope(type, component)}) {
    return symbol;
  }
  auto &details{type.symbol()->get<DerivedTypeDetails>()};
  if (auto *extends{details.extends()}) {
    return FindComponent(*extends->scope(), component);
  } else {
    return nullptr;
  }
}

void ResolveNamesVisitor::Post(const parser::ProcedureDesignator &x) {
  if (const auto *name{std::get_if<parser::Name>(&x.u)}) {
    auto *symbol{FindSymbol(*name)};
    if (symbol == nullptr) {
      symbol = &MakeSymbol(*name);
      if (isImplicitNoneExternal() && !symbol->attrs().test(Attr::EXTERNAL)) {
        Say(*name,
            "'%s' is an external procedure without the EXTERNAL"
            " attribute in a scope with IMPLICIT NONE(EXTERNAL)"_err_en_US);
      }
      symbol->attrs().set(Attr::EXTERNAL);
      symbol->set_details(ProcEntityDetails{});
      if (const auto type{GetImplicitType(*symbol)}) {
        symbol->get<ProcEntityDetails>().interface().set_type(*type);
      }
      CHECK(expectedProcFlag_);
      symbol->set(*expectedProcFlag_);
    } else if (symbol->has<UnknownDetails>()) {
      CHECK(!"unexpected UnknownDetails");
    } else if (CheckUseError(*name)) {
      // error was reported
    } else {
      symbol = Resolve(*name, &symbol->GetUltimate());
      ConvertToProcEntity(*symbol);
      if (symbol->test(Symbol::Flag::Function) &&
          expectedProcFlag_ == Symbol::Flag::Subroutine) {
        Say2(*name, "Cannot call function '%s' like a subroutine"_err_en_US,
            *symbol, "Declaration of '%s'"_en_US);
      } else if (symbol->test(Symbol::Flag::Subroutine) &&
          expectedProcFlag_ == Symbol::Flag::Function) {
        Say2(*name, "Cannot call subroutine '%s' like a function"_err_en_US,
            *symbol, "Declaration of '%s'"_en_US);
      } else if (symbol->has<ProcEntityDetails>()) {
        symbol->set(*expectedProcFlag_);  // in case it hasn't been set yet
        if (expectedProcFlag_ == Symbol::Flag::Function) {
          ApplyImplicitRules(*symbol);
        }
      } else if (symbol->has<SubprogramDetails>()) {
        // OK
      } else if (symbol->has<SubprogramNameDetails>()) {
        // OK
      } else if (symbol->has<GenericDetails>()) {
        // OK
      } else if (symbol->has<DerivedTypeDetails>()) {
        // OK: type constructor
      } else if (symbol->has<ObjectEntityDetails>()) {
        // OK: array mis-parsed as a call
      } else if (symbol->test(Symbol::Flag::Implicit)) {
        Say(*name,
            "Use of '%s' as a procedure conflicts with its implicit definition"_err_en_US);
      } else {
        Say2(*name,
            "Use of '%s' as a procedure conflicts with its declaration"_err_en_US,
            *symbol, "Declaration of '%s'"_en_US);
      }
    }
  }
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
              [=](const parser::Name &y) { SetAccess(y, accessAttr); },
              [=](const common::Indirection<parser::GenericSpec> &y) {
                std::visit(
                    common::visitors{
                        [=](const parser::Name &z) {
                          SetAccess(z, accessAttr);
                        },
                        [](const auto &) { common::die("TODO: GenericSpec"); },
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
  Symbol &symbol{MakeSymbol(name)};
  Attrs &attrs{symbol.attrs()};
  if (attrs.HasAny({Attr::PUBLIC, Attr::PRIVATE})) {
    // PUBLIC/PRIVATE already set: make it a fatal error if it changed
    Attr prev = attrs.test(Attr::PUBLIC) ? Attr::PUBLIC : Attr::PRIVATE;
    Say(name,
        attr == prev
            ? "The accessibility of '%s' has already been specified as %s"_en_US
            : "The accessibility of '%s' has already been specified as %s"_err_en_US,
        name.source, EnumToString(prev));
  } else {
    attrs.set(attr);
  }
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
            symbol->name().ToString().c_str());
  }
}

bool ResolveNamesVisitor::Pre(const parser::MainProgram &x) {
  using stmtType = std::optional<parser::Statement<parser::ProgramStmt>>;
  Symbol *symbol{nullptr};
  if (auto &stmt{std::get<stmtType>(x.t)}) {
    symbol = &MakeSymbol(stmt->statement.v, MainProgramDetails{});
  }
  PushScope(Scope::Kind::MainProgram, symbol);
  auto &subpPart{std::get<std::optional<parser::InternalSubprogramPart>>(x.t)};
  WalkSubprogramPart(subpPart);
  return true;
}

void ResolveNamesVisitor::Post(const parser::EndProgramStmt &) { PopScope(); }

bool ResolveNamesVisitor::Pre(const parser::ImplicitStmt &x) {
  if (currScope().kind() == Scope::Kind::Block) {
    Say("IMPLICIT statement is not allowed in BLOCK construct"_err_en_US);
    return false;
  }
  return ImplicitRulesVisitor::Pre(x);
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
void ResolveNamesVisitor::Post(const parser::PointerAssignmentStmt &x) {
  ResolveDataRef(std::get<parser::DataRef>(x.t));
}
void ResolveNamesVisitor::Post(const parser::Designator &x) {
  std::visit(
      common::visitors{
          [&](const parser::ObjectName &x) { ResolveName(x); },
          [&](const parser::DataRef &x) { ResolveDataRef(x); },
          [&](const parser::Substring &x) {
            ResolveDataRef(std::get<parser::DataRef>(x.t));
            // TODO: SubstringRange
          },
      },
      x.u);
}

template<typename T>
void ResolveNamesVisitor::Post(const parser::LoopBounds<T> &x) {
  ResolveName(x.name.thing.thing);
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

void ResolveNamesVisitor::Post(const parser::Program &) {
  // ensure that all temps were deallocated
  CHECK(!attrs_);
  CHECK(!GetDeclTypeSpec());
}

void ResolveNames(SemanticsContext &context, const parser::Program &program) {
  ResolveNamesVisitor{context}.Walk(program);
}

// Get the Name out of a GenericSpec, or nullptr if none.
static const parser::Name *GetGenericSpecName(const parser::GenericSpec &x) {
  const auto *op{std::get_if<parser::DefinedOperator>(&x.u)};
  if (!op) {
    return std::get_if<parser::Name>(&x.u);
  } else if (const auto *opName{std::get_if<parser::DefinedOpName>(&op->u)}) {
    return &opName->v;
  } else {
    return nullptr;
  }
}
}
