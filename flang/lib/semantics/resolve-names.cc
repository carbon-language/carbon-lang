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
#include "mod-file.h"
#include "rewrite-parse-tree.h"
#include "scope.h"
#include "symbol.h"
#include "type.h"
#include "../common/indirection.h"
#include "../parser/parse-tree-visitor.h"
#include "../parser/parse-tree.h"
#include <list>
#include <memory>
#include <ostream>
#include <set>
#include <stack>
#include <vector>

namespace Fortran::semantics {

using namespace parser::literals;

class MessageHandler;

static GenericSpec MapGenericSpec(const parser::GenericSpec &);

// ImplicitRules maps initial character of identifier to the DeclTypeSpec
// representing the implicit type; std::nullopt if none.
// It also records the presence of IMPLICIT NONE statements.
// When inheritFromParent is set, defaults come from the parent rules.
class ImplicitRules {
public:
  ImplicitRules(MessageHandler &messages)
    : messages_{messages}, inheritFromParent_{false} {}
  ImplicitRules(std::unique_ptr<ImplicitRules> &&parent)
    : messages_{parent->messages_}, inheritFromParent_{true} {
    parent_.swap(parent);
  }
  std::unique_ptr<ImplicitRules> &&parent() { return std::move(parent_); }
  bool isImplicitNoneType() const;
  bool isImplicitNoneExternal() const;
  void set_isImplicitNoneType(bool x) { isImplicitNoneType_ = x; }
  void set_isImplicitNoneExternal(bool x) { isImplicitNoneExternal_ = x; }
  void set_inheritFromParent(bool x) { inheritFromParent_ = x; }
  // Get the implicit type for identifiers starting with ch. May be null.
  std::optional<const DeclTypeSpec> GetType(char ch) const;
  // Record the implicit type for this range of characters.
  void SetType(const DeclTypeSpec &type, parser::Location lo, parser::Location,
      bool isDefault = false);

private:
  static char Incr(char ch);

  std::unique_ptr<ImplicitRules> parent_;
  MessageHandler &messages_;
  std::optional<bool> isImplicitNoneType_;
  std::optional<bool> isImplicitNoneExternal_;
  bool inheritFromParent_;  // look in parent if not specified here
  // map initial character of identifier to nullptr or its default type
  std::map<char, const DeclTypeSpec> map_;

  friend std::ostream &operator<<(std::ostream &, const ImplicitRules &);
  friend void ShowImplicitRule(std::ostream &, const ImplicitRules &, char);
};

// Provide Post methods to collect attributes into a member variable.
class AttrsVisitor {
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
  using AttrsVisitor::Post;
  using AttrsVisitor::Pre;
  bool Pre(const parser::IntegerTypeSpec &);
  bool Pre(const parser::IntrinsicTypeSpec::Logical &);
  bool Pre(const parser::IntrinsicTypeSpec::Real &);
  bool Pre(const parser::IntrinsicTypeSpec::Complex &);
  bool Pre(const parser::IntrinsicTypeSpec::DoublePrecision &);
  bool Pre(const parser::DeclarationTypeSpec::ClassStar &);
  bool Pre(const parser::DeclarationTypeSpec::TypeStar &);
  bool Pre(const parser::DeclarationTypeSpec::Record &);
  void Post(const parser::TypeParamSpec &);
  bool Pre(const parser::TypeParamValue &);
  void Post(const parser::StructureConstructor &);
  bool Pre(const parser::AllocateStmt &);
  void Post(const parser::AllocateStmt &);
  bool Pre(const parser::TypeGuardStmt &);
  void Post(const parser::TypeGuardStmt &);

protected:
  std::unique_ptr<DeclTypeSpec> &GetDeclTypeSpec();
  void BeginDeclTypeSpec();
  void EndDeclTypeSpec();
  void BeginDerivedTypeSpec(DerivedTypeSpec &);
  void SetDerivedDeclTypeSpec(DeclTypeSpec::Category);

private:
  bool expectDeclTypeSpec_{false};  // should only see decl-type-spec when true
  std::unique_ptr<DeclTypeSpec> declTypeSpec_;
  DerivedTypeSpec *derivedTypeSpec_{nullptr};
  std::unique_ptr<ParamValue> typeParamValue_;

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

  const parser::Messages &messages() const { return messages_; }

  template<typename T> bool Pre(const parser::Statement<T> &x) {
    currStmtSource_ = &x.source;
    return true;
  }
  template<typename T> void Post(const parser::Statement<T> &) {
    currStmtSource_ = nullptr;
  }

  const SourceName *currStmtSource() { return currStmtSource_; }

  // Add a message to the messages to be emitted.
  Message &Say(Message &&);
  // Emit a message associated with the current statement source.
  Message &Say(MessageFixedText &&);
  // Emit a message about a SourceName or parser::Name
  Message &Say(const SourceName &, MessageFixedText &&);
  Message &Say(const parser::Name &, MessageFixedText &&);
  // Emit a formatted message associated with a source location.
  Message &Say(const SourceName &, MessageFixedText &&, const std::string &);
  Message &Say(const SourceName &, MessageFixedText &&, const SourceName &,
      const SourceName &);
  void SayAlreadyDeclared(const SourceName &, const Symbol &);
  // Emit a message and attached message with two names and locations.
  void Say2(const SourceName &, MessageFixedText &&, const SourceName &,
      MessageFixedText &&);
  void Annex(parser::Messages &&);

private:
  // Where messages are emitted:
  parser::Messages messages_;
  // Source location of current statement; null if not in a statement
  const SourceName *currStmtSource_{nullptr};
};

// Visit ImplicitStmt and related parse tree nodes and updates implicit rules.
class ImplicitRulesVisitor : public DeclTypeSpecVisitor,
                             public virtual MessageHandler {
public:
  using DeclTypeSpecVisitor::Post;
  using DeclTypeSpecVisitor::Pre;
  using MessageHandler::Post;
  using MessageHandler::Pre;
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

private:
  // implicit rules in effect for current scope
  std::unique_ptr<ImplicitRules> implicitRules_{
      std::make_unique<ImplicitRules>(*this)};
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
  bool Pre(const parser::ArraySpec &);
  void Post(const parser::AttrSpec &) { PostAttrSpec(); }
  void Post(const parser::ComponentAttrSpec &) { PostAttrSpec(); }
  bool Pre(const parser::DeferredShapeSpecList &);
  bool Pre(const parser::AssumedShapeSpec &);
  bool Pre(const parser::ExplicitShapeSpec &);
  bool Pre(const parser::AssumedImpliedSpec &);
  bool Pre(const parser::AssumedRankSpec &);

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
class ScopeHandler : public virtual ImplicitRulesVisitor {
public:
  void set_rootScope(Scope &scope) { PushScope(scope); }
  Scope &currScope() { return *currScope_; }
  // The enclosing scope, skipping blocks and derived types.
  Scope &InclusiveScope();

  // Create a new scope and push it on the scope stack.
  void PushScope(Scope::Kind kind, Symbol *symbol);
  void PushScope(Scope &scope);
  void PopScope();

  Symbol *FindSymbol(const SourceName &name);
  void EraseSymbol(const SourceName &name);

  // Helpers to make a Symbol in the current scope
  template<typename D>
  Symbol &MakeSymbol(
      const SourceName &name, const Attrs &attrs, const D &details) {
    // Note: don't use FindSymbol here. If this is a derived type scope,
    // we want to detect if the name is already declared as a component.
    const auto &it{currScope().find(name)};
    if (it == currScope().end()) {
      const auto pair{currScope().try_emplace(name, attrs, details)};
      CHECK(pair.second);  // name was not found, so must be able to add
      return *pair.first->second;
    }
    auto &symbol{*it->second};
    symbol.add_occurrence(name);
    if constexpr (std::is_same_v<DerivedTypeDetails, D>) {
      if (auto *d{symbol.detailsIf<GenericDetails>()}) {
        // derived type with same name as a generic
        auto *derivedType{d->derivedType()};
        if (!derivedType) {
          derivedType = &currScope().MakeSymbol(name, attrs, details);
          d->set_derivedType(*derivedType);
        } else {
          Say2(name, "'%s' is already declared in this scoping unit"_err_en_US,
              derivedType->name(), "Previous declaration of '%s'"_en_US);
        }
        return *derivedType;
      }
    }
    if (symbol.CanReplaceDetails(details)) {
      // update the existing symbol
      symbol.attrs() |= attrs;
      symbol.set_details(details);
      return symbol;
    } else if constexpr (std::is_same_v<UnknownDetails, D>) {
      symbol.attrs() |= attrs;
      return symbol;
    } else {
      SayAlreadyDeclared(name, symbol);
      // replace the old symbols with a new one with correct details
      EraseSymbol(symbol.name());
      return MakeSymbol(name, attrs, details);
    }
  }
  template<typename D>
  Symbol &MakeSymbol(
      const parser::Name &name, const Attrs &attrs, const D &details) {
    return MakeSymbol(name.source, attrs, details);
  }
  template<typename D>
  Symbol &MakeSymbol(const parser::Name &name, const D &details) {
    return MakeSymbol(name, Attrs(), details);
  }
  template<typename D>
  Symbol &MakeSymbol(const SourceName &name, const D &details) {
    return MakeSymbol(name, Attrs(), details);
  }
  Symbol &MakeSymbol(const SourceName &name, Attrs attrs = Attrs{}) {
    return MakeSymbol(name, attrs, UnknownDetails{});
  }

protected:
  // When subpNamesOnly_ is set we are only collecting procedure names.
  // Create symbols with SubprogramNameDetails of the given kind.
  std::optional<SubprogramKind> subpNamesOnly_;

  // Apply the implicit type rules to this symbol.
  void ApplyImplicitRules(const SourceName &, Symbol &);
  std::optional<const DeclTypeSpec> GetImplicitType(Symbol &);

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

  void add_searchDirectory(const std::string &dir) {
    searchDirectories_.push_back(dir);
  }

private:
  // The default access spec for this module.
  Attr defaultAccess_{Attr::PUBLIC};
  // The location of the last AccessStmt without access-ids, if any.
  const SourceName *prevAccessStmt_{nullptr};
  // The scope of the module during a UseStmt
  const Scope *useModuleScope_{nullptr};
  // Directories to search for .mod files
  std::vector<std::string> searchDirectories_;

  void SetAccess(const parser::Name &, Attr);
  void ApplyDefaultAccess();
  void AddUse(const parser::Rename::Names &);
  void AddUse(const parser::Name &);
  // Record a use from useModuleScope_ of useName as localName. location is
  // where it occurred (either the module or the rename) for error reporting.
  void AddUse(const SourceName &location, const SourceName &localName,
      const SourceName &useName);
  Symbol &BeginModule(const SourceName &, bool isSubmodule,
      const std::optional<parser::ModuleSubprogramPart> &);
  Scope *FindModule(const SourceName &, Scope *ancestor = nullptr);
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
  bool isGeneric() const { return genericSymbol_ != nullptr; }
  bool isAbstract() const { return isAbstract_; }

protected:
  // Add name or symbol to the generic we are currently processing
  void AddToGeneric(const parser::Name &name, bool expectModuleProc = false);
  void AddToGeneric(const Symbol &symbol);
  // Add to generic the symbol for the subprogram with the same name
  void SetSpecificInGeneric(Symbol &symbol);
  void CheckGenericProcedures(Symbol &);

private:
  bool inInterfaceBlock_{false};  // set when in interface block
  bool isAbstract_{false};  // set when in abstract interface block
  Symbol *genericSymbol_{nullptr};  // set when in generic interface block

  void ResolveSpecificsInGeneric(Symbol &generic);
};

class SubprogramVisitor : public InterfaceVisitor {
public:
  bool Pre(const parser::StmtFunctionStmt &);
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
  bool Pre(const parser::Suffix &);

protected:
  // Set when we see a stmt function that is really an array element assignment
  bool badStmtFuncFound_{false};

private:
  // Function result name from parser::Suffix, if any.
  const parser::Name *funcResultName_{nullptr};

  bool BeginSubprogram(const parser::Name &, Symbol::Flag,
      const std::optional<parser::InternalSubprogramPart> &);
  void EndSubprogram();
  // Create a subprogram symbol in the current scope and push a new scope.
  Symbol &PushSubprogramScope(const parser::Name &, Symbol::Flag);
  Symbol *GetSpecificFromGeneric(const SourceName &);
};

class DeclarationVisitor : public ArraySpecVisitor,
                           public virtual ScopeHandler {
public:
  using ArraySpecVisitor::Post;
  using ArraySpecVisitor::Pre;

  void Post(const parser::EntityDecl &);
  void Post(const parser::ObjectDecl &);
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
  void Post(const parser::DeclarationTypeSpec::Type &);
  void Post(const parser::DeclarationTypeSpec::Class &);
  bool Pre(const parser::DerivedTypeSpec &);
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

protected:
  bool BeginDecl();
  void EndDecl();
  bool CheckUseError(const SourceName &, const Symbol &);

private:
  // The attribute corresponding to the statement containing an ObjectDecl
  std::optional<Attr> objectDeclAttr_;
  // Info about current derived type while walking DerivedTypeStmt
  struct {
    const SourceName *extends{nullptr};  // EXTENDS(name)
    bool privateComps{false};  // components are private by default
    bool privateBindings{false};  // bindings are private by default
    bool sawContains{false};  // currently processing bindings
    bool sequence{false};  // is a sequence type
  } derivedTypeInfo_;
  // In a ProcedureDeclarationStmt or ProcComponentDefStmt, this is
  // the interface name, if any.
  const SourceName *interfaceName_{nullptr};

  // Handle a statement that sets an attribute on a list of names.
  bool HandleAttributeStmt(Attr, const std::list<parser::Name> &);
  void DeclareObjectEntity(const parser::Name &, Attrs);
  void DeclareProcEntity(const parser::Name &, Attrs, const ProcInterface &);
  bool ConvertToProcEntity(Symbol &);

  // Set the type of an entity or report an error.
  void SetType(
      const SourceName &name, Symbol &symbol, const DeclTypeSpec &type);
  const Symbol *ResolveDerivedType(const SourceName &);
  bool CanBeTypeBoundProc(const Symbol &);
  Symbol *FindExplicitInterface(const SourceName &);
  Symbol &MakeTypeSymbol(const SourceName &, const Details &);

  // Declare an object or procedure entity.
  // T is one of: EntityDetails, ObjectEntityDetails, ProcEntityDetails
  template<typename T>
  Symbol &DeclareEntity(const parser::Name &name, Attrs attrs) {
    Symbol &symbol{MakeSymbol(name.source, attrs)};
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
        Say2(name.source,
            "Declaration of '%s' conflicts with its use as module procedure"_err_en_US,
            symbol.name(), "Module procedure definition"_en_US);
      } else if (details->kind() == SubprogramKind::Internal) {
        Say2(name.source,
            "Declaration of '%s' conflicts with its use as internal procedure"_err_en_US,
            symbol.name(), "Internal procedure definition"_en_US);
      } else {
        CHECK(!"unexpected kind");
      }
    } else {
      SayAlreadyDeclared(name.source, symbol);
    }
    return symbol;
  }
};

// Walk the parse tree and resolve names to symbols.
class ResolveNamesVisitor : public ModuleVisitor,
                            public SubprogramVisitor,
                            public DeclarationVisitor {
public:
  using ArraySpecVisitor::Post;
  using ArraySpecVisitor::Pre;
  using DeclarationVisitor::Post;
  using DeclarationVisitor::Pre;
  using ImplicitRulesVisitor::Post;
  using ImplicitRulesVisitor::Pre;
  using InterfaceVisitor::Post;
  using InterfaceVisitor::Pre;
  using ModuleVisitor::Post;
  using ModuleVisitor::Pre;
  using SubprogramVisitor::Post;
  using SubprogramVisitor::Pre;

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
  bool Pre(const parser::BlockStmt &);
  bool Pre(const parser::EndBlockStmt &);
  bool Pre(const parser::ImplicitStmt &);

  void Post(const parser::Expr &x) { CheckImplicitSymbol(GetVariableName(x)); }
  void Post(const parser::Variable &x) {
    CheckImplicitSymbol(GetVariableName(x));
  }
  template<typename T> void Post(const parser::LoopBounds<T> &x) {
    CheckImplicitSymbol(&x.name.thing.thing);
  }
  bool Pre(const parser::StructureComponent &);
  void Post(const parser::ProcedureDesignator &);
  bool Pre(const parser::FunctionReference &);
  void Post(const parser::FunctionReference &);
  bool Pre(const parser::CallStmt &);
  void Post(const parser::CallStmt &);
  bool Pre(const parser::ImportStmt &);

private:
  // Kind of procedure we are expecting to see in a ProcedureDesignator
  std::optional<Symbol::Flag> expectedProcFlag_;
  const SourceName *prevImportStmt_{nullptr};

  const parser::Name *GetVariableName(const parser::DataRef &);
  const parser::Name *GetVariableName(const parser::Designator &);
  const parser::Name *GetVariableName(const parser::Expr &);
  const parser::Name *GetVariableName(const parser::Variable &);
  const Symbol *CheckImplicitSymbol(const parser::Name *);
  const Symbol *ResolveStructureComponent(const parser::StructureComponent &);
  const Symbol *ResolveArrayElement(const parser::ArrayElement &);
  const Symbol *ResolveCoindexedNamedObject(
      const parser::CoindexedNamedObject &);
  const Symbol *ResolveDataRef(const parser::DataRef &);
  const Symbol *FindComponent(const Symbol &base, const SourceName &component);
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

std::optional<const DeclTypeSpec> ImplicitRules::GetType(char ch) const {
  if (auto it{map_.find(ch)}; it != map_.end()) {
    return it->second;
  } else if (inheritFromParent_) {
    return parent_->GetType(ch);
  } else if (ch >= 'i' && ch <= 'n') {
    return DeclTypeSpec{IntegerTypeSpec::Make()};
  } else if (ch >= 'a' && ch <= 'z') {
    return DeclTypeSpec{RealTypeSpec::Make()};
  } else {
    return std::nullopt;
  }
}

// isDefault is set when we are applying the default rules, so it is not
// an error if the type is already set.
void ImplicitRules::SetType(const DeclTypeSpec &type, parser::Location lo,
    parser::Location hi, bool isDefault) {
  for (char ch = *lo; ch; ch = ImplicitRules::Incr(ch)) {
    auto res{map_.emplace(ch, type)};
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
    o << "  " << ch << ": " << it->second << '\n';
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

std::unique_ptr<DeclTypeSpec> &DeclTypeSpecVisitor::GetDeclTypeSpec() {
  return declTypeSpec_;
}
void DeclTypeSpecVisitor::BeginDeclTypeSpec() {
  CHECK(!expectDeclTypeSpec_);
  CHECK(!derivedTypeSpec_);
  expectDeclTypeSpec_ = true;
}
void DeclTypeSpecVisitor::EndDeclTypeSpec() {
  CHECK(expectDeclTypeSpec_);
  expectDeclTypeSpec_ = false;
  declTypeSpec_.reset();
  derivedTypeSpec_ = nullptr;
}

bool DeclTypeSpecVisitor::Pre(const parser::DeclarationTypeSpec::ClassStar &x) {
  SetDeclTypeSpec(DeclTypeSpec{DeclTypeSpec::ClassStar});
  return false;
}
bool DeclTypeSpecVisitor::Pre(const parser::DeclarationTypeSpec::TypeStar &x) {
  SetDeclTypeSpec(DeclTypeSpec{DeclTypeSpec::TypeStar});
  return false;
}
void DeclTypeSpecVisitor::Post(const parser::TypeParamSpec &x) {
  typeParamValue_.reset();
}
bool DeclTypeSpecVisitor::Pre(const parser::TypeParamValue &x) {
  typeParamValue_ = std::make_unique<ParamValue>(std::visit(
      common::visitors{
          // TODO: create IntExpr from ScalarIntExpr
          [&](const parser::ScalarIntExpr &x) { return Bound{IntExpr{}}; },
          [&](const parser::Star &x) { return Bound::ASSUMED; },
          [&](const parser::TypeParamValue::Deferred &x) {
            return Bound::DEFERRED;
          },
      },
      x.u));
  return false;
}

bool DeclTypeSpecVisitor::Pre(const parser::DeclarationTypeSpec::Record &x) {
  // TODO
  return true;
}

void DeclTypeSpecVisitor::Post(const parser::StructureConstructor &) {
  // TODO: StructureConstructor
  derivedTypeSpec_ = nullptr;
}
bool DeclTypeSpecVisitor::Pre(const parser::AllocateStmt &) {
  BeginDeclTypeSpec();
  return true;
}
void DeclTypeSpecVisitor::Post(const parser::AllocateStmt &) {
  // TODO: AllocateStmt
  EndDeclTypeSpec();
  derivedTypeSpec_ = nullptr;
}
bool DeclTypeSpecVisitor::Pre(const parser::TypeGuardStmt &) {
  BeginDeclTypeSpec();
  return true;
}
void DeclTypeSpecVisitor::Post(const parser::TypeGuardStmt &) {
  // TODO: TypeGuardStmt
  EndDeclTypeSpec();
  derivedTypeSpec_ = nullptr;
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
bool DeclTypeSpecVisitor::Pre(
    const parser::IntrinsicTypeSpec::DoublePrecision &) {
  CHECK(!"TODO: double precision");
  return false;
}
void DeclTypeSpecVisitor::MakeIntrinsic(
    const IntrinsicTypeSpec &intrinsicTypeSpec) {
  SetDeclTypeSpec(DeclTypeSpec{intrinsicTypeSpec});
}

// Set declTypeSpec_ based on derivedTypeSpec_
void DeclTypeSpecVisitor::SetDerivedDeclTypeSpec(
    DeclTypeSpec::Category category) {
  SetDeclTypeSpec(DeclTypeSpec{category, *derivedTypeSpec_});
}

void DeclTypeSpecVisitor::BeginDerivedTypeSpec(
    DerivedTypeSpec &derivedTypeSpec) {
  CHECK(!derivedTypeSpec_);
  derivedTypeSpec_ = &derivedTypeSpec;
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
  if (kind) {
    if (auto *intExpr{std::get_if<parser::ScalarIntConstantExpr>(&kind->u)}) {
      const parser::Expr &expr{*intExpr->thing.thing.thing};
      if (auto *lit{std::get_if<parser::LiteralConstant>(&expr.u)}) {
        if (auto *intLit{std::get_if<parser::IntLiteralConstant>(&lit->u)}) {
          return KindParamValue{
              IntConst::Make(std::get<std::uint64_t>(intLit->t))};
        }
      }
      CHECK(!"TODO: constant evaluation");
    } else {
      CHECK(!"TODO: translate star-size to kind");
    }
  }
  return KindParamValue{};
}

// MessageHandler implementation

MessageHandler::Message &MessageHandler::Say(MessageFixedText &&msg) {
  CHECK(currStmtSource_);
  return messages_.Say(*currStmtSource_, std::move(msg));
}
MessageHandler::Message &MessageHandler::Say(
    const SourceName &name, MessageFixedText &&msg) {
  return Say(name, std::move(msg), name.ToString());
}
MessageHandler::Message &MessageHandler::Say(
    const parser::Name &name, MessageFixedText &&msg) {
  return messages_.Say(name.source, std::move(msg), name.ToString().c_str());
}
MessageHandler::Message &MessageHandler::Say(const SourceName &location,
    MessageFixedText &&msg, const std::string &arg1) {
  return messages_.Say(location, std::move(msg), arg1.c_str());
}
MessageHandler::Message &MessageHandler::Say(const SourceName &location,
    MessageFixedText &&msg, const SourceName &arg1, const SourceName &arg2) {
  return messages_.Say(location, std::move(msg), arg1.ToString().c_str(),
      arg2.ToString().c_str());
}
void MessageHandler::SayAlreadyDeclared(
    const SourceName &name, const Symbol &prev) {
  Say2(name, "'%s' is already declared in this scoping unit"_err_en_US,
      prev.name(), "Previous declaration of '%s'"_en_US);
}
void MessageHandler::Say2(const SourceName &name1, MessageFixedText &&msg1,
    const SourceName &name2, MessageFixedText &&msg2) {
  Say(name1, std::move(msg1)).Attach(name2, msg2, name2.ToString().c_str());
}
void MessageHandler::Annex(parser::Messages &&msgs) {
  messages_.Annex(std::move(msgs));
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

bool ArraySpecVisitor::Pre(const parser::DeferredShapeSpecList &x) {
  for (int i = 0; i < x.v; ++i) {
    arraySpec_.push_back(ShapeSpec::MakeDeferred());
  }
  return false;
}

bool ArraySpecVisitor::Pre(const parser::AssumedShapeSpec &x) {
  const auto &lb{x.v};
  arraySpec_.push_back(
      lb ? ShapeSpec::MakeAssumed(GetBound(*lb)) : ShapeSpec::MakeAssumed());
  return false;
}

bool ArraySpecVisitor::Pre(const parser::ExplicitShapeSpec &x) {
  const auto &lb{std::get<std::optional<parser::SpecificationExpr>>(x.t)};
  const auto &ub{GetBound(std::get<parser::SpecificationExpr>(x.t))};
  arraySpec_.push_back(lb ? ShapeSpec::MakeExplicit(GetBound(*lb), ub)
                          : ShapeSpec::MakeExplicit(ub));
  return false;
}

bool ArraySpecVisitor::Pre(const parser::AssumedImpliedSpec &x) {
  const auto &lb{x.v};
  arraySpec_.push_back(
      lb ? ShapeSpec::MakeImplied(GetBound(*lb)) : ShapeSpec::MakeImplied());
  return false;
}

bool ArraySpecVisitor::Pre(const parser::AssumedRankSpec &) {
  arraySpec_.push_back(ShapeSpec::MakeAssumedRank());
  return false;
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
  return Bound(IntExpr{});  // TODO: convert x.v to IntExpr
}

// ScopeHandler implementation

Scope &ScopeHandler::InclusiveScope() {
  for (auto *scope{&currScope()};; scope = &scope->parent()) {
    if (scope->kind() != Scope::Kind::Block &&
        scope->kind() != Scope::Kind::DerivedType) {
      return *scope;
    }
  }
}
void ScopeHandler::PushScope(Scope::Kind kind, Symbol *symbol) {
  PushScope(currScope().MakeScope(kind, symbol));
}
void ScopeHandler::PushScope(Scope &scope) {
  currScope_ = &scope;
  if (currScope_->kind() != Scope::Kind::Block) {
    ImplicitRulesVisitor::PushScope();
  }
}
void ScopeHandler::PopScope() {
  if (currScope_->kind() != Scope::Kind::Block) {
    ImplicitRulesVisitor::PopScope();
  }
  currScope_ = &currScope_->parent();
}

Symbol *ScopeHandler::FindSymbol(const SourceName &name) {
  return currScope().FindSymbol(name);
}
void ScopeHandler::EraseSymbol(const SourceName &name) {
  currScope().erase(name);
}

void ScopeHandler::ApplyImplicitRules(const SourceName &name, Symbol &symbol) {
  if (symbol.has<UnknownDetails>()) {
    symbol.set_details(ObjectEntityDetails{});
  } else if (symbol.has<EntityDetails>()) {
    symbol.set_details(ObjectEntityDetails{symbol.get<EntityDetails>()});
  }
  if (auto *details{symbol.detailsIf<ObjectEntityDetails>()}) {
    if (!details->type()) {
      if (const auto type{GetImplicitType(symbol)}) {
        details->set_type(*type);
      }
    }
  }
}
std::optional<const DeclTypeSpec> ScopeHandler::GetImplicitType(
    Symbol &symbol) {
  auto &name{symbol.name()};
  const auto type{implicitRules().GetType(name.begin()[0])};
  if (type) {
    symbol.set(Symbol::Flag::Implicit);
  } else {
    Say(name, "No explicit type declared for '%s'"_err_en_US);
  }
  return type;
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
  useModuleScope_ = FindModule(x.moduleName.source);
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
    const SourceName &moduleName{x.moduleName.source};
    for (const auto &pair : *useModuleScope_) {
      const Symbol &symbol{*pair.second};
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

void ModuleVisitor::AddUse(const parser::Rename::Names &names) {
  const SourceName &useName{std::get<0>(names.t).source};
  const SourceName &localName{std::get<1>(names.t).source};
  AddUse(useName, useName, localName);
}
void ModuleVisitor::AddUse(const parser::Name &useName) {
  AddUse(useName.source, useName.source, useName.source);
}
void ModuleVisitor::AddUse(const SourceName &location,
    const SourceName &localName, const SourceName &useName) {
  if (!useModuleScope_) {
    return;  // error occurred finding module
  }
  const auto it{useModuleScope_->find(useName)};
  if (it == useModuleScope_->end()) {
    Say(useName, "'%s' not found in module '%s'"_err_en_US, useName,
        useModuleScope_->name());
    return;
  }
  const Symbol &useSymbol{*it->second};
  if (useSymbol.attrs().test(Attr::PRIVATE)) {
    Say(useName, "'%s' is PRIVATE in '%s'"_err_en_US, useName,
        useModuleScope_->name());
    return;
  }
  Symbol &localSymbol{MakeSymbol(localName, useSymbol.attrs())};
  localSymbol.attrs() &= ~Attrs{Attr::PUBLIC, Attr::PRIVATE};
  localSymbol.flags() |= useSymbol.flags();
  if (auto *details{localSymbol.detailsIf<UseDetails>()}) {
    // check for use-associating the same symbol again:
    if (localSymbol.GetUltimate() != useSymbol.GetUltimate()) {
      localSymbol.set_details(
          UseErrorDetails{*details}.add_occurrence(location, *useModuleScope_));
    }
  } else if (auto *details{localSymbol.detailsIf<UseErrorDetails>()}) {
    details->add_occurrence(location, *useModuleScope_);
  } else {
    CHECK(localSymbol.has<UnknownDetails>());
    localSymbol.set_details(UseDetails{location, useSymbol});
  }
}

bool ModuleVisitor::Pre(const parser::Submodule &x) {
  auto &stmt{std::get<parser::Statement<parser::SubmoduleStmt>>(x.t)};
  auto &name{std::get<parser::Name>(stmt.statement.t).source};
  auto &subpPart{std::get<std::optional<parser::ModuleSubprogramPart>>(x.t)};
  auto &parentId{std::get<parser::ParentIdentifier>(stmt.statement.t)};
  auto &ancestorName{std::get<parser::Name>(parentId.t).source};
  auto &parentName{std::get<std::optional<parser::Name>>(parentId.t)};
  Scope *ancestor{FindModule(ancestorName)};
  if (!ancestor) {
    return false;
  }
  Scope *parentScope{
      parentName ? FindModule(parentName->source, ancestor) : ancestor};
  if (!parentScope) {
    return false;
  }
  PushScope(*parentScope);  // submodule is hosted in parent
  auto &symbol{BeginModule(name, true, subpPart)};
  if (!ancestor->AddSubmodule(name, currScope())) {
    Say(name, "Module '%s' already has a submodule named '%s'"_err_en_US,
        ancestorName, name);
  }
  MakeSymbol(name, symbol.get<ModuleDetails>());
  return true;
}
void ModuleVisitor::Post(const parser::Submodule &) {
  PopScope();  // submodule's scope
  PopScope();  // parent's scope
}

bool ModuleVisitor::Pre(const parser::Module &x) {
  // Make a symbol and push a scope for this module
  const auto &name{
      std::get<parser::Statement<parser::ModuleStmt>>(x.t).statement.v.source};
  auto &subpPart{std::get<std::optional<parser::ModuleSubprogramPart>>(x.t)};
  auto &symbol{BeginModule(name, false, subpPart)};
  MakeSymbol(name, symbol.details());
  return true;
}

void ModuleVisitor::Post(const parser::Module &) {
  ApplyDefaultAccess();
  PopScope();
  prevAccessStmt_ = nullptr;
}

Symbol &ModuleVisitor::BeginModule(const SourceName &name, bool isSubmodule,
    const std::optional<parser::ModuleSubprogramPart> &subpPart) {
  auto &symbol{MakeSymbol(name, ModuleDetails{isSubmodule})};
  auto &details{symbol.get<ModuleDetails>()};
  PushScope(Scope::Kind::Module, &symbol);
  details.set_scope(&currScope());
  if (subpPart) {
    subpNamesOnly_ = SubprogramKind::Module;
    parser::Walk(*subpPart, *static_cast<ResolveNamesVisitor *>(this));
    subpNamesOnly_ = std::nullopt;
  }
  return symbol;
}

// Find a module or submodule by name and return its scope.
// If ancestor is present, look for a submodule of that ancestor module.
// May have to read a .mod file to find it.
// If an error occurs, report it and return nullptr.
Scope *ModuleVisitor::FindModule(const SourceName &name, Scope *ancestor) {
  ModFileReader reader{searchDirectories_};
  auto *scope{reader.Read(name, ancestor)};
  if (!scope) {
    Annex(std::move(reader.errors()));
    return nullptr;
  }
  if (scope->kind() != Scope::Kind::Module) {
    Say(name, "'%s' is not a module"_err_en_US);
    return nullptr;
  }
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
  if (genericSymbol_) {
    if (const auto *proc{
            genericSymbol_->get<GenericDetails>().CheckSpecific()}) {
      SayAlreadyDeclared(genericSymbol_->name(), *proc);
    }
    genericSymbol_ = nullptr;
  }
  inInterfaceBlock_ = false;
  isAbstract_ = false;
}

// Create a symbol for the generic in genericSymbol_
bool InterfaceVisitor::Pre(const parser::GenericSpec &x) {
  const SourceName *genericName{nullptr};
  GenericSpec genericSpec{MapGenericSpec(x)};
  switch (genericSpec.kind()) {
  case GenericSpec::Kind::GENERIC_NAME:
    genericName = &genericSpec.genericName();
    break;
  case GenericSpec::Kind::OP_DEFINED:
    genericName = &genericSpec.definedOp();
    break;
  default: CHECK(!"TODO: intrinsic ops");
  }
  genericSymbol_ = FindSymbol(*genericName);
  if (genericSymbol_) {
    if (genericSymbol_->has<DerivedTypeDetails>()) {
      // A generic and derived type with same name: create a generic symbol
      // and save derived type in it.
      CHECK(genericSymbol_->scope()->symbol() == genericSymbol_);
      GenericDetails details;
      details.set_derivedType(*genericSymbol_);
      EraseSymbol(*genericName);
      genericSymbol_ = &MakeSymbol(*genericName);
      genericSymbol_->set_details(details);
    } else if (!genericSymbol_->isSubprogram()) {
      SayAlreadyDeclared(*genericName, *genericSymbol_);
      EraseSymbol(*genericName);
      genericSymbol_ = nullptr;
    } else if (genericSymbol_->has<UseDetails>()) {
      // copy the USEd symbol into this scope so we can modify it
      const Symbol &ultimate{genericSymbol_->GetUltimate()};
      EraseSymbol(*genericName);
      genericSymbol_ = &MakeSymbol(ultimate.name(), ultimate.attrs());
      if (const auto *details{ultimate.detailsIf<GenericDetails>()}) {
        genericSymbol_->set_details(GenericDetails{details->specificProcs()});
      } else if (const auto *details{ultimate.detailsIf<SubprogramDetails>()}) {
        genericSymbol_->set_details(SubprogramDetails{*details});
      } else {
        CHECK(!"can't happen");
      }
    }
  }
  if (!genericSymbol_) {
    genericSymbol_ = &MakeSymbol(*genericName);
    genericSymbol_->set_details(GenericDetails{});
  }
  if (genericSymbol_->has<GenericDetails>()) {
    // okay
  } else if (genericSymbol_->has<SubprogramDetails>() ||
      genericSymbol_->has<SubprogramNameDetails>()) {
    Details details;
    if (auto *d{genericSymbol_->detailsIf<SubprogramNameDetails>()}) {
      details = *d;
    } else if (auto *d{genericSymbol_->detailsIf<SubprogramDetails>()}) {
      details = *d;
    } else {
      CHECK(!"can't happen");
    }
    GenericDetails genericDetails;
    genericDetails.set_specific(*genericSymbol_);
    EraseSymbol(*genericName);
    genericSymbol_ = &MakeSymbol(*genericName, genericDetails);
  }
  CHECK(genericSymbol_->has<GenericDetails>());
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
    AddToGeneric(name, expectModuleProc);
  }
  return false;
}

void InterfaceVisitor::Post(const parser::GenericStmt &x) {
  if (auto &accessSpec{std::get<std::optional<parser::AccessSpec>>(x.t)}) {
    genericSymbol_->attrs().set(AccessSpecToAttr(*accessSpec));
  }
  for (const auto &name : std::get<std::list<parser::Name>>(x.t)) {
    AddToGeneric(name);
  }
}

void InterfaceVisitor::AddToGeneric(
    const parser::Name &name, bool expectModuleProc) {
  genericSymbol_->get<GenericDetails>().add_specificProcName(
      name.source, expectModuleProc);
}
void InterfaceVisitor::AddToGeneric(const Symbol &symbol) {
  genericSymbol_->get<GenericDetails>().add_specificProc(&symbol);
}
void InterfaceVisitor::SetSpecificInGeneric(Symbol &symbol) {
  genericSymbol_->get<GenericDetails>().set_specific(symbol);
}

// By now we should have seen all specific procedures referenced by name in
// this generic interface. Resolve those names to symbols.
void InterfaceVisitor::ResolveSpecificsInGeneric(Symbol &generic) {
  auto &details{generic.get<GenericDetails>()};
  std::set<SourceName> namesSeen;  // to check for duplicate names
  for (const auto *symbol : details.specificProcs()) {
    namesSeen.insert(symbol->name());
  }
  for (auto &pair : details.specificProcNames()) {
    const auto &name{*pair.first};
    auto expectModuleProc{pair.second};
    const auto *symbol{FindSymbol(name)};
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
    Say2(generic.name(),
        "Generic interface '%s' may only contain functions due to derived type"
        " with same name"_err_en_US,
        details.derivedType()->name(), "Derived type '%s'"_en_US);
  }
  generic.set(isFunction ? Symbol::Flag::Function : Symbol::Flag::Subroutine);
}

// SubprogramVisitor implementation

bool SubprogramVisitor::Pre(const parser::StmtFunctionStmt &x) {
  const auto &name{std::get<parser::Name>(x.t)};
  std::optional<SourceName> occurrence;
  std::optional<DeclTypeSpec> resultType;
  // Look up name: provides return type or tells us if it's an array
  if (auto *symbol{FindSymbol(name.source)}) {
    if (auto *details{symbol->detailsIf<EntityDetails>()}) {
      // TODO: check that attrs are compatible with stmt func
      resultType = details->type();
      occurrence = symbol->name();
      EraseSymbol(symbol->name());
    } else if (symbol->has<ObjectEntityDetails>()) {
      // not a stmt-func at all but an array; do nothing
      symbol->add_occurrence(name.source);
      badStmtFuncFound_ = true;
      return true;
    }
  }
  if (badStmtFuncFound_) {
    Say(name, "'%s' has not been declared as an array"_err_en_US);
    return true;
  }
  auto &symbol{PushSubprogramScope(name, Symbol::Flag::Function)};
  if (occurrence) {
    symbol.add_occurrence(*occurrence);
  }
  auto &details{symbol.get<SubprogramDetails>()};
  for (const auto &dummyName : std::get<std::list<parser::Name>>(x.t)) {
    EntityDetails dummyDetails{true};
    auto it{currScope().parent().find(dummyName.source)};
    if (it != currScope().parent().end()) {
      if (auto *d{it->second->detailsIf<EntityDetails>()}) {
        if (d->type()) {
          dummyDetails.set_type(*d->type());
        }
      }
    }
    details.add_dummyArg(MakeSymbol(dummyName, dummyDetails));
  }
  EraseSymbol(name.source);  // added by PushSubprogramScope
  EntityDetails resultDetails;
  if (resultType) {
    resultDetails.set_type(*resultType);
  }
  details.set_result(MakeSymbol(name, resultDetails));
  return true;
}

void SubprogramVisitor::Post(const parser::StmtFunctionStmt &x) {
  if (badStmtFuncFound_) {
    return;  // This wasn't really a stmt function so no scope was created
  }
  PopScope();
}

bool SubprogramVisitor::Pre(const parser::Suffix &suffix) {
  if (suffix.resultName) {
    funcResultName_ = &suffix.resultName.value();
  }
  return true;
}

bool SubprogramVisitor::Pre(const parser::SubroutineSubprogram &x) {
  const auto &name{std::get<parser::Name>(
      std::get<parser::Statement<parser::SubroutineStmt>>(x.t).statement.t)};
  const auto &subpPart{
      std::get<std::optional<parser::InternalSubprogramPart>>(x.t)};
  return BeginSubprogram(name, Symbol::Flag::Subroutine, subpPart);
}
void SubprogramVisitor::Post(const parser::SubroutineSubprogram &) {
  EndSubprogram();
}

bool SubprogramVisitor::Pre(const parser::FunctionSubprogram &x) {
  const auto &name{std::get<parser::Name>(
      std::get<parser::Statement<parser::FunctionStmt>>(x.t).statement.t)};
  const auto &subpPart{
      std::get<std::optional<parser::InternalSubprogramPart>>(x.t)};
  return BeginSubprogram(name, Symbol::Flag::Function, subpPart);
}
void SubprogramVisitor::Post(const parser::FunctionSubprogram &) {
  EndSubprogram();
}

bool SubprogramVisitor::Pre(const parser::InterfaceBody::Subroutine &x) {
  const auto &name{std::get<parser::Name>(
      std::get<parser::Statement<parser::SubroutineStmt>>(x.t).statement.t)};
  return BeginSubprogram(name, Symbol::Flag::Subroutine, std::nullopt);
}
void SubprogramVisitor::Post(const parser::InterfaceBody::Subroutine &) {
  EndSubprogram();
}
bool SubprogramVisitor::Pre(const parser::InterfaceBody::Function &x) {
  const auto &name{std::get<parser::Name>(
      std::get<parser::Statement<parser::FunctionStmt>>(x.t).statement.t)};
  return BeginSubprogram(name, Symbol::Flag::Function, std::nullopt);
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
  Symbol &symbol{*currScope().symbol()};
  CHECK(name.source == symbol.name());
  symbol.attrs() |= EndAttrs();
  auto &details{symbol.get<SubprogramDetails>()};
  for (const auto &dummyArg : std::get<std::list<parser::DummyArg>>(stmt.t)) {
    const parser::Name *dummyName = std::get_if<parser::Name>(&dummyArg.u);
    CHECK(dummyName != nullptr && "TODO: alternate return indicator");
    Symbol &dummy{MakeSymbol(*dummyName, EntityDetails(true))};
    details.add_dummyArg(dummy);
  }
}

void SubprogramVisitor::Post(const parser::FunctionStmt &stmt) {
  const auto &name{std::get<parser::Name>(stmt.t)};
  Symbol &symbol{*currScope().symbol()};
  CHECK(name.source == symbol.name());
  symbol.attrs() |= EndAttrs();
  auto &details{symbol.get<SubprogramDetails>()};
  for (const auto &dummyName : std::get<std::list<parser::Name>>(stmt.t)) {
    Symbol &dummy{MakeSymbol(dummyName, EntityDetails(true))};
    details.add_dummyArg(dummy);
  }
  // add function result to function scope
  EntityDetails funcResultDetails;
  if (auto &type{GetDeclTypeSpec()}) {
    funcResultDetails.set_type(*type);
  }
  EndDeclTypeSpec();

  const parser::Name *funcResultName;
  if (funcResultName_ && funcResultName_->source != name.source) {
    funcResultName = funcResultName_;
  } else {
    EraseSymbol(name.source);  // was added by PushSubprogramScope
    funcResultName = &name;
  }
  details.set_result(MakeSymbol(*funcResultName, funcResultDetails));
  funcResultName_ = nullptr;
}

bool SubprogramVisitor::BeginSubprogram(const parser::Name &name,
    Symbol::Flag subpFlag,
    const std::optional<parser::InternalSubprogramPart> &subpPart) {
  if (subpNamesOnly_) {
    auto &symbol{MakeSymbol(name, SubprogramNameDetails{*subpNamesOnly_})};
    symbol.set(subpFlag);
    return false;
  }
  PushSubprogramScope(name, subpFlag);
  if (subpPart) {
    subpNamesOnly_ = SubprogramKind::Internal;
    parser::Walk(*subpPart, *static_cast<ResolveNamesVisitor *>(this));
    subpNamesOnly_ = std::nullopt;
  }
  return true;
}
void SubprogramVisitor::EndSubprogram() {
  if (!subpNamesOnly_) {
    PopScope();
  }
}

Symbol &SubprogramVisitor::PushSubprogramScope(
    const parser::Name &name, Symbol::Flag subpFlag) {
  Symbol *symbol = GetSpecificFromGeneric(name.source);
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
      AddToGeneric(*symbol);
    }
    implicitRules().set_inheritFromParent(false);
  }
  // can't reuse this name inside subprogram:
  MakeSymbol(name, details).set(subpFlag);
  return *symbol;
}

// If name is a generic, return specific subprogram with the same name.
Symbol *SubprogramVisitor::GetSpecificFromGeneric(const SourceName &name) {
  if (auto *symbol{FindSymbol(name)}) {
    if (auto *details{symbol->detailsIf<GenericDetails>()}) {
      // found generic, want subprogram
      auto *specific{details->specific()};
      if (isGeneric()) {
        if (specific) {
          SayAlreadyDeclared(name, *specific);
        } else {
          symbol->remove_occurrence(name);
          specific =
              &currScope().MakeSymbol(name, Attrs{}, SubprogramDetails{});
          SetSpecificInGeneric(*specific);
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

bool DeclarationVisitor::CheckUseError(
    const SourceName &name, const Symbol &symbol) {
  const auto *details{symbol.detailsIf<UseErrorDetails>()};
  if (!details) {
    return false;
  }
  Message &msg{Say(name, "Reference to '%s' is ambiguous"_err_en_US)};
  for (const auto &pair : details->occurrences()) {
    const SourceName &location{*pair.first};
    const SourceName &moduleName{pair.second->name()};
    msg.Attach(location, "'%s' was use-associated from module '%s'"_en_US,
        name.ToString().data(), moduleName.ToString().data());
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
  if (!arraySpec().empty()) {
    DeclareObjectEntity(name, attrs);
  } else {
    Symbol &symbol{DeclareEntity<EntityDetails>(name, attrs)};
    if (auto &type{GetDeclTypeSpec()}) {
      SetType(name.source, symbol, *type);
    }
    if (attrs.test(Attr::EXTERNAL)) {
      ConvertToProcEntity(symbol);
    }
  }
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
    auto *symbol{FindSymbol(name.source)};
    if (!ConvertToProcEntity(*symbol)) {
      Say2(name.source, "EXTERNAL attribute not allowed on '%s'"_err_en_US,
          symbol->name(), "Declaration of '%s'"_en_US);
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
bool DeclarationVisitor::HandleAttributeStmt(
    Attr attr, const std::list<parser::Name> &names) {
  for (const auto &name : names) {
    const auto pair{currScope().try_emplace(name.source, Attrs{attr})};
    if (!pair.second) {
      // symbol was already there: set attribute on it
      Symbol &symbol{*pair.first->second};
      if (attr == Attr::ASYNCHRONOUS || attr == Attr::VOLATILE) {
        // TODO: if in a BLOCK, attribute should only be set while in the block
      } else if (symbol.has<UseDetails>()) {
        Say(*currStmtSource(),
            "Cannot change %s attribute on use-associated '%s'"_err_en_US,
            EnumToString(attr), name.source);
      }
      symbol.attrs().set(attr);
      symbol.add_occurrence(name.source);
    }
  }
  return false;
}
// Convert symbol to be a ProcEntity or return false if it can't be.
bool DeclarationVisitor::ConvertToProcEntity(Symbol &symbol) {
  if (symbol.has<ProcEntityDetails>()) {
    // nothing to do
  } else if (symbol.has<UnknownDetails>()) {
    symbol.set_details(ProcEntityDetails{});
  } else if (auto *details{symbol.detailsIf<EntityDetails>()}) {
    symbol.set_details(ProcEntityDetails(*details));
    symbol.set(Symbol::Flag::Function);
  } else {
    return false;
  }
  return true;
}

void DeclarationVisitor::Post(const parser::ObjectDecl &x) {
  CHECK(objectDeclAttr_.has_value());
  const auto &name{std::get<parser::ObjectName>(x.t)};
  DeclareObjectEntity(name, Attrs{*objectDeclAttr_});
}

void DeclarationVisitor::DeclareProcEntity(
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
}

void DeclarationVisitor::DeclareObjectEntity(
    const parser::Name &name, Attrs attrs) {
  Symbol &symbol{DeclareEntity<ObjectEntityDetails>(name, attrs)};
  if (auto *details{symbol.detailsIf<ObjectEntityDetails>()}) {
    if (auto &type{GetDeclTypeSpec()}) {
      if (details->type()) {
        Say(name, "The type of '%s' has already been declared"_err_en_US);
      } else {
        details->set_type(*type);
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
  }
}

void DeclarationVisitor::Post(const parser::DeclarationTypeSpec::Type &x) {
  SetDerivedDeclTypeSpec(DeclTypeSpec::TypeDerived);
  DerivedTypeSpec &type{GetDeclTypeSpec()->derivedTypeSpec()};
  if (const auto *symbol{ResolveDerivedType(type.name())}) {
    type.set_scope(*symbol->scope());
  }
}
void DeclarationVisitor::Post(const parser::DeclarationTypeSpec::Class &) {
  SetDerivedDeclTypeSpec(DeclTypeSpec::ClassDerived);
  DerivedTypeSpec &type{GetDeclTypeSpec()->derivedTypeSpec()};
  if (const auto *symbol{ResolveDerivedType(type.name())}) {
    type.set_scope(*symbol->scope());
  }
}

bool DeclarationVisitor::Pre(const parser::DerivedTypeSpec &x) {
  auto &name{std::get<parser::Name>(x.t).source};
  auto &derivedTypeSpec{currScope().MakeDerivedTypeSpec(name)};
  BeginDerivedTypeSpec(derivedTypeSpec);
  return true;
}
void DeclarationVisitor::Post(const parser::DerivedTypeDef &x) {
  std::set<SourceName> paramNames;
  auto &scope{currScope()};
  auto &stmt{std::get<parser::Statement<parser::DerivedTypeStmt>>(x.t)};
  for (auto &name : std::get<std::list<parser::Name>>(stmt.statement.t)) {
    auto &paramName{name.source};
    if (auto it{scope.find(paramName)}; it == scope.end()) {
      Say(paramName,
          "No definition found for type parameter '%s'"_err_en_US);  // C742
    } else {
      auto *symbol{it->second};
      if (!symbol->has<TypeParamDetails>()) {
        Say2(paramName, "'%s' is not defined as a type parameter"_err_en_US,
            symbol->name(),
            "Definition of '%s'"_en_US);  // C741
      } else {
        symbol->add_occurrence(paramName);
      }
    }
    if (!paramNames.insert(paramName).second) {
      Say(paramName,
          "Duplicate type parameter name: '%s'"_err_en_US);  // C731
    }
  }
  auto &details{scope.symbol()->get<DerivedTypeDetails>()};
  details.set_hasTypeParams(!paramNames.empty());
  for (const auto &pair : currScope()) {
    const auto *symbol{pair.second};
    if (symbol->has<TypeParamDetails>() && !paramNames.count(symbol->name())) {
      Say2(symbol->name(),
          "'%s' is not a type parameter of this derived type"_err_en_US,
          stmt.source, "Derived type statement"_en_US);  // C742
    }
  }
  if (derivedTypeInfo_.sequence) {
    details.set_sequence(true);
    if (derivedTypeInfo_.extends) {
      Say(stmt.source,
          "A sequence type may not have the EXTENDS attribute"_err_en_US);  // C735
    }
    if (details.hasTypeParams()) {
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
  auto &name{std::get<parser::Name>(x.t).source};
  auto &symbol{MakeSymbol(name, GetAttrs(), DerivedTypeDetails{})};
  auto &details{symbol.get<DerivedTypeDetails>()};
  PushScope(Scope::Kind::DerivedType, &symbol);
  if (derivedTypeInfo_.extends) {
    if (auto *extends{ResolveDerivedType(*derivedTypeInfo_.extends)}) {
      details.set_extends(extends);
    }
  }
  EndAttrs();
}
void DeclarationVisitor::Post(const parser::TypeParamDefStmt &x) {
  auto &type{GetDeclTypeSpec()};
  auto attr{std::get<common::TypeParamAttr>(x.t)};
  for (auto &decl : std::get<std::list<parser::TypeParamDecl>>(x.t)) {
    auto &name{std::get<parser::Name>(decl.t).source};
    // TODO: initialization
    // auto &init{
    //    std::get<std::optional<parser::ScalarIntConstantExpr>>(decl.t)};
    auto &symbol{MakeTypeSymbol(name, TypeParamDetails{attr})};
    SetType(name, symbol, *type);
  }
  EndDecl();
}
bool DeclarationVisitor::Pre(const parser::TypeAttrSpec::Extends &x) {
  derivedTypeInfo_.extends = &x.v.source;
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
  DeclareObjectEntity(name, attrs);
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
    interfaceName_ = &name->source;
  }
}

void DeclarationVisitor::Post(const parser::ProcDecl &x) {
  const auto &name{std::get<parser::Name>(x.t)};
  ProcInterface interface;
  if (interfaceName_) {
    if (auto *symbol{FindExplicitInterface(*interfaceName_)}) {
      interface.set_symbol(*symbol);
    }
  } else if (auto &type{GetDeclTypeSpec()}) {
    interface.set_type(*type);
  }
  auto attrs{GetAttrs()};
  if (currScope().kind() != Scope::Kind::DerivedType) {
    attrs.set(Attr::EXTERNAL);
  }
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
    auto &bindingName{std::get<parser::Name>(declaration.t).source};
    auto &optName{std::get<std::optional<parser::Name>>(declaration.t)};
    auto &procedureName{optName ? optName->source : bindingName};
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
          procedure->name(), "Declaration of '%s'"_en_US);
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
  Symbol *interface{FindExplicitInterface(x.interfaceName.source)};
  if (!interface) {
    return;
  }
  for (auto &bindingName : x.bindingNames) {
    MakeTypeSymbol(bindingName.source, ProcBindingDetails{*interface});
  }
}

void DeclarationVisitor::Post(const parser::FinalProcedureStmt &x) {
  for (auto &name : x.v) {
    MakeTypeSymbol(name.source, FinalProcDetails{});
  }
}

void DeclarationVisitor::SetType(
    const SourceName &name, Symbol &symbol, const DeclTypeSpec &type) {
  if (symbol.GetType()) {
    Say(name, "The type of '%s' has already been declared"_err_en_US);
    return;
  }
  symbol.SetType(type);
}

// Find the Symbol for this derived type.
const Symbol *DeclarationVisitor::ResolveDerivedType(const SourceName &name) {
  const auto *symbol{FindSymbol(name)};
  if (!symbol) {
    Say(name, "Derived type '%s' not found"_err_en_US);
    return nullptr;
  }
  if (CheckUseError(name, *symbol)) {
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

Symbol *DeclarationVisitor::FindExplicitInterface(const SourceName &name) {
  auto *symbol{FindSymbol(name)};
  if (!symbol) {
    Say(name, "Explicit interface '%s' not found"_err_en_US);
  } else if (!symbol->HasExplicitInterface()) {
    Say2(name,
        "'%s' is not an abstract interface or a procedure with an"
        " explicit interface"_err_en_US,
        symbol->name(), "Declaration of '%s'"_en_US);
    symbol = nullptr;
  }
  return symbol;
}

// Create a symbol for a type parameter, component, or procedure binding in
// the current derived type scope.
Symbol &DeclarationVisitor::MakeTypeSymbol(
    const SourceName &name, const Details &details) {
  Scope &derivedType{currScope()};
  CHECK(derivedType.kind() == Scope::Kind::DerivedType);
  if (auto it{derivedType.find(name)}; it != derivedType.end()) {
    Say2(name,
        "Type parameter, component, or procedure binding '%s'"
        " already defined in this type"_err_en_US,
        it->second->name(), "Previous definition of '%s'"_en_US);
    return *it->second;
  } else {
    auto attrs{GetAttrs()};
    // Apply binding-private-stmt if present and this is a procedure binding
    if (derivedTypeInfo_.privateBindings &&
        !attrs.HasAny({Attr::PUBLIC, Attr::PRIVATE}) &&
        std::holds_alternative<ProcBindingDetails>(details)) {
      attrs.set(Attr::PRIVATE);
    }
    return MakeSymbol(name, attrs, details);
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
    if (!scope.add_importName(name.source)) {
      Say(name, "'%s' not found in host scope"_err_en_US);
    }
  }
  prevImportStmt_ = currStmtSource();
  return false;
}

bool ResolveNamesVisitor::Pre(const parser::StructureComponent &x) {
  ResolveStructureComponent(x);
  return false;
}

const Symbol *ResolveNamesVisitor::ResolveStructureComponent(
    const parser::StructureComponent &x) {
  const Symbol *dataRef = ResolveDataRef(x.base);
  return dataRef ? FindComponent(*dataRef, x.component.source) : nullptr;
}
const Symbol *ResolveNamesVisitor::ResolveArrayElement(
    const parser::ArrayElement &x) {
  return ResolveDataRef(x.base);
}
const Symbol *ResolveNamesVisitor::ResolveCoindexedNamedObject(
    const parser::CoindexedNamedObject &x) {
  return nullptr;  // TODO
}

const Symbol *ResolveNamesVisitor::ResolveDataRef(const parser::DataRef &x) {
  return std::visit(
      common::visitors{
          [=](const parser::Name &y) {
            auto *symbol{FindSymbol(y.source)};
            if (!symbol) {
              if (isImplicitNoneType()) {
                Say(y.source, "No explicit type declared for '%s'"_err_en_US);
              } else {
                auto pair{InclusiveScope().try_emplace(y.source)};
                CHECK(pair.second);
                symbol = pair.first->second;
                ApplyImplicitRules(y.source, *symbol);
              }
            }
            return const_cast<const Symbol *>(symbol);
          },
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

// base is a part-ref of a derived type; find the named component in its type.
const Symbol *ResolveNamesVisitor::FindComponent(
    const Symbol &base, const SourceName &component) {
  std::optional<DeclTypeSpec> type;
  if (auto *details{base.detailsIf<EntityDetails>()}) {
    type = details->type();
  } else if (auto *details{base.detailsIf<ObjectEntityDetails>()}) {
    type = details->type();
  } else {
    Say2(base.occurrences().back(),
        "'%s' is not an object of derived type"_err_en_US, base.name(),
        "Declaration of '%s'"_en_US);
    return nullptr;
  }
  if (!type) {
    return nullptr;  // should have already reported error
  }
  if (type->category() != DeclTypeSpec::TypeDerived) {
    if (base.test(Symbol::Flag::Implicit)) {
      Say(base.occurrences().back(),
          "'%s' is not an object of derived type; it is implicitly typed"_err_en_US);
    } else {
      Say2(base.occurrences().back(),
          "'%s' is not an object of derived type"_err_en_US, base.name(),
          "Declaration of '%s'"_en_US);
    }
    return nullptr;
  }
  const DerivedTypeSpec &derivedTypeSpec{type->derivedTypeSpec()};
  const Scope *scope{derivedTypeSpec.scope()};
  if (!scope) {
    return nullptr;  // previously failed to resolve type
  }
  auto it{scope->find(component)};
  if (it == scope->end()) {
    auto &typeName{scope->symbol()->name()};
    Say(component, "Component '%s' not found in derived type '%s'"_err_en_US,
        component, typeName)
        .Attach(
            typeName, "Declaration of '%s'"_en_US, typeName.ToString().data());
    return nullptr;
  }
  auto *symbol{it->second};
  symbol->add_occurrence(component);
  return symbol;
}

void ResolveNamesVisitor::Post(const parser::ProcedureDesignator &x) {
  if (const auto *name{std::get_if<parser::Name>(&x.u)}) {
    auto *symbol{FindSymbol(name->source)};
    if (symbol == nullptr) {
      symbol = &MakeSymbol(name->source);
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
    } else if (CheckUseError(name->source, *symbol)) {
      // error was reported
    } else {
      symbol = &symbol->GetUltimate();
      if (auto *details{symbol->detailsIf<EntityDetails>()}) {
        symbol->set_details(ProcEntityDetails(*details));
        symbol->set(Symbol::Flag::Function);
      }
      if (symbol->test(Symbol::Flag::Function) &&
          expectedProcFlag_ == Symbol::Flag::Subroutine) {
        Say2(name->source,
            "Cannot call function '%s' like a subroutine"_err_en_US,
            symbol->name(), "Declaration of '%s'"_en_US);
      } else if (symbol->test(Symbol::Flag::Subroutine) &&
          expectedProcFlag_ == Symbol::Flag::Function) {
        Say2(name->source,
            "Cannot call subroutine '%s' like a function"_err_en_US,
            symbol->name(), "Declaration of '%s'"_en_US);
      } else if (symbol->has<ProcEntityDetails>()) {
        symbol->set(*expectedProcFlag_);  // in case it hasn't been set yet
      } else if (symbol->has<SubprogramDetails>()) {
        // OK
      } else if (symbol->has<SubprogramNameDetails>()) {
        // OK
      } else if (symbol->has<GenericDetails>()) {
        // OK
      } else if (symbol->has<DerivedTypeDetails>()) {
        // OK: type constructor
      } else {
        Say2(name->source,
            "Use of '%s' as a procedure conflicts with its declaration"_err_en_US,
            symbol->name(), "Declaration of '%s'"_en_US);
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
  for (auto &pair : currScope()) {
    auto &name{pair.first};
    auto &symbol{*pair.second};
    if (NeedsExplicitType(symbol)) {
      if (isImplicitNoneType()) {
        Say(name, "No explicit type declared for '%s'"_err_en_US);
      } else {
        ApplyImplicitRules(name, symbol);
      }
    }
    if (symbol.has<GenericDetails>()) {
      CheckGenericProcedures(symbol);
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
  auto &scope{currScope()};
  auto it{scope.find(name)};
  if (it != scope.end()) {
    Say(location, "'%s' from host is not accessible"_err_en_US,
        name.ToString().c_str())
        .Attach(it->second->name(), "'%s' is hidden by this entity"_en_US,
            it->second->name().ToString().c_str());
  }
}

bool ResolveNamesVisitor::Pre(const parser::MainProgram &x) {
  using stmtType = std::optional<parser::Statement<parser::ProgramStmt>>;
  if (auto &stmt{std::get<stmtType>(x.t)}) {
    const parser::Name &name{stmt->statement.v};
    Symbol &symbol{MakeSymbol(name, MainProgramDetails{})};
    PushScope(Scope::Kind::MainProgram, &symbol);
    MakeSymbol(name, MainProgramDetails{});
  } else {
    PushScope(Scope::Kind::MainProgram, nullptr);
  }
  if (auto &subpPart{
          std::get<std::optional<parser::InternalSubprogramPart>>(x.t)}) {
    subpNamesOnly_ = SubprogramKind::Internal;
    parser::Walk(*subpPart, *static_cast<ResolveNamesVisitor *>(this));
    subpNamesOnly_ = std::nullopt;
  }
  return true;
}

void ResolveNamesVisitor::Post(const parser::EndProgramStmt &) { PopScope(); }

bool ResolveNamesVisitor::Pre(const parser::BlockStmt &) {
  PushScope(Scope::Kind::Block, nullptr);
  return false;
}
bool ResolveNamesVisitor::Pre(const parser::EndBlockStmt &) {
  PopScope();
  return false;
}
bool ResolveNamesVisitor::Pre(const parser::ImplicitStmt &x) {
  if (currScope().kind() == Scope::Kind::Block) {
    Say("IMPLICIT statement is not allowed in BLOCK construct"_err_en_US);
    return false;
  }
  return ImplicitRulesVisitor::Pre(x);
}

const parser::Name *ResolveNamesVisitor::GetVariableName(
    const parser::DataRef &x) {
  return std::visit(
      common::visitors{
          [](const parser::Name &x) { return &x; },
          [&](const common::Indirection<parser::ArrayElement> &x) {
            return GetVariableName(x->base);
          },
          [](const auto &) {
            return static_cast<const parser::Name *>(nullptr);
          },
      },
      x.u);
}

const parser::Name *ResolveNamesVisitor::GetVariableName(
    const parser::Designator &x) {
  return std::visit(
      common::visitors{
          [](const parser::ObjectName &x) { return &x; },
          [&](const parser::DataRef &x) { return GetVariableName(x); },
          [](const auto &) {
            return static_cast<const parser::Name *>(nullptr);
          },
      },
      x.u);
}
const parser::Name *ResolveNamesVisitor::GetVariableName(
    const parser::Expr &x) {
  if (const auto *designator{
          std::get_if<common::Indirection<parser::Designator>>(&x.u)}) {
    return GetVariableName(**designator);
  } else {
    return nullptr;
  }
}
const parser::Name *ResolveNamesVisitor::GetVariableName(
    const parser::Variable &x) {
  if (const auto *designator{
          std::get_if<common::Indirection<parser::Designator>>(&x.u)}) {
    return GetVariableName(**designator);
  } else {
    return nullptr;
  }
}

// If implicit types are allowed, ensure name is in the symbol table.
// Otherwise, report an error if it hasn't been declared.
const Symbol *ResolveNamesVisitor::CheckImplicitSymbol(
    const parser::Name *name) {
  if (!name) {
    return nullptr;
  }
  if (const auto *symbol{FindSymbol(name->source)}) {
    if (CheckUseError(name->source, *symbol) ||
        !symbol->has<UnknownDetails>()) {
      return nullptr;  // reported an error or symbol is declared
    }
    return symbol;
  }
  if (isImplicitNoneType()) {
    Say(*name, "No explicit type declared for '%s'"_err_en_US);
    return nullptr;
  }
  // Create the symbol then ensure it is accessible
  InclusiveScope().try_emplace(name->source);
  auto *symbol{FindSymbol(name->source)};
  if (!symbol) {
    Say(name->source,
        "'%s' from host scoping unit is not accessible due to IMPORT"_err_en_US);
    return nullptr;
  }
  ApplyImplicitRules(name->source, *symbol);
  return symbol;
}

void ResolveNamesVisitor::Post(const parser::Program &) {
  // ensure that all temps were deallocated
  CHECK(!attrs_);
  CHECK(!GetDeclTypeSpec());
}

void ResolveNames(Scope &rootScope, parser::Program &program,
    const parser::CookedSource &cookedSource,
    const std::vector<std::string> &searchDirectories) {
  ResolveNamesVisitor visitor;
  visitor.set_rootScope(rootScope);
  for (auto &dir : searchDirectories) {
    visitor.add_searchDirectory(dir);
  }
  parser::Walk(const_cast<const parser::Program &>(program), visitor);
  if (!visitor.messages().empty()) {
    visitor.messages().Emit(std::cerr, cookedSource);
    return;
  }
  RewriteParseTree(program, cookedSource);
}

// Map the enum in the parser to the one in GenericSpec
static GenericSpec::Kind MapIntrinsicOperator(
    parser::DefinedOperator::IntrinsicOperator x) {
  switch (x) {
  case parser::DefinedOperator::IntrinsicOperator::Add:
    return GenericSpec::OP_ADD;
  case parser::DefinedOperator::IntrinsicOperator::AND:
    return GenericSpec::OP_AND;
  case parser::DefinedOperator::IntrinsicOperator::Concat:
    return GenericSpec::OP_CONCAT;
  case parser::DefinedOperator::IntrinsicOperator::Divide:
    return GenericSpec::OP_DIVIDE;
  case parser::DefinedOperator::IntrinsicOperator::EQ:
    return GenericSpec::OP_EQ;
  case parser::DefinedOperator::IntrinsicOperator::EQV:
    return GenericSpec::OP_EQV;
  case parser::DefinedOperator::IntrinsicOperator::GE:
    return GenericSpec::OP_GE;
  case parser::DefinedOperator::IntrinsicOperator::GT:
    return GenericSpec::OP_GT;
  case parser::DefinedOperator::IntrinsicOperator::LE:
    return GenericSpec::OP_LE;
  case parser::DefinedOperator::IntrinsicOperator::LT:
    return GenericSpec::OP_LT;
  case parser::DefinedOperator::IntrinsicOperator::Multiply:
    return GenericSpec::OP_MULTIPLY;
  case parser::DefinedOperator::IntrinsicOperator::NE:
    return GenericSpec::OP_NE;
  case parser::DefinedOperator::IntrinsicOperator::NEQV:
    return GenericSpec::OP_NEQV;
  case parser::DefinedOperator::IntrinsicOperator::NOT:
    return GenericSpec::OP_NOT;
  case parser::DefinedOperator::IntrinsicOperator::OR:
    return GenericSpec::OP_OR;
  case parser::DefinedOperator::IntrinsicOperator::Power:
    return GenericSpec::OP_POWER;
  case parser::DefinedOperator::IntrinsicOperator::Subtract:
    return GenericSpec::OP_SUBTRACT;
  case parser::DefinedOperator::IntrinsicOperator::XOR:
    return GenericSpec::OP_XOR;
  default: CRASH_NO_CASE;
  }
}

// Map a parser::GenericSpec to a semantics::GenericSpec
static GenericSpec MapGenericSpec(const parser::GenericSpec &genericSpec) {
  return std::visit(
      common::visitors{
          [](const parser::Name &x) {
            return GenericSpec::GenericName(x.source);
          },
          [](const parser::DefinedOperator &x) {
            return std::visit(
                common::visitors{
                    [](const parser::DefinedOpName &name) {
                      return GenericSpec::DefinedOp(name.v.source);
                    },
                    [](const parser::DefinedOperator::IntrinsicOperator &x) {
                      return GenericSpec::IntrinsicOp(MapIntrinsicOperator(x));
                    },
                },
                x.u);
          },
          [](const parser::GenericSpec::Assignment &) {
            return GenericSpec::IntrinsicOp(GenericSpec::ASSIGNMENT);
          },
          [](const parser::GenericSpec::ReadFormatted &) {
            return GenericSpec::IntrinsicOp(GenericSpec::READ_FORMATTED);
          },
          [](const parser::GenericSpec::ReadUnformatted &) {
            return GenericSpec::IntrinsicOp(GenericSpec::READ_UNFORMATTED);
          },
          [](const parser::GenericSpec::WriteFormatted &) {
            return GenericSpec::IntrinsicOp(GenericSpec::WRITE_FORMATTED);
          },
          [](const parser::GenericSpec::WriteUnformatted &) {
            return GenericSpec::IntrinsicOp(GenericSpec::WRITE_UNFORMATTED);
          },
      },
      genericSpec.u);
}

static void PutIndent(std::ostream &os, int indent) {
  for (int i = 0; i < indent; ++i) {
    os << "  ";
  }
}

static void DumpSymbols(std::ostream &os, const Scope &scope, int indent = 0) {
  PutIndent(os, indent);
  os << Scope::EnumToString(scope.kind()) << " scope:";
  if (const auto *symbol{scope.symbol()}) {
    os << ' ' << symbol->name().ToString();
  }
  os << '\n';
  ++indent;
  for (const auto &pair : scope) {
    const auto &symbol{*pair.second};
    PutIndent(os, indent);
    os << symbol << '\n';
    if (const auto *details{symbol.detailsIf<GenericDetails>()}) {
      if (const auto &type{details->derivedType()}) {
        PutIndent(os, indent);
        os << *type << '\n';
      }
    }
  }
  for (const auto &child : scope.children()) {
    DumpSymbols(os, child, indent);
  }
  --indent;
}

void DumpSymbols(std::ostream &os) { DumpSymbols(os, Scope::globalScope); }

}  // namespace Fortran::semantics
