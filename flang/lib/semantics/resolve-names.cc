#include "../parser/indirection.h"
#include "../parser/parse-tree-visitor.h"
#include "../parser/parse-tree.h"
#include "attr.h"
#include "scope.h"
#include "symbol.h"
#include "type.h"
#include <iostream>
#include <list>
#include <memory>
#include <stack>

namespace Fortran::semantics {

// Provide Post methods to collect attributes into a member variable.
class AttrsVisitor {
public:
  void beginAttrs();
  Attrs endAttrs();
  void Post(const parser::LanguageBindingSpec &x);
  bool Pre(const parser::AccessSpec &x);
  bool Pre(const parser::IntentSpec &x);

// Simple case: encountering CLASSNAME causes ATTRNAME to be set.
#define HANDLE_ATTR_CLASS(CLASSNAME, ATTRNAME) \
  bool Pre(const parser::CLASSNAME &) { \
    attrs_->Set(Attr::ATTRNAME); \
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
  std::unique_ptr<Attrs> attrs_;
  std::string langBindingName_{""};
};

// Find and create types from declaration-type-spec nodes.
class DeclTypeSpecVisitor : public AttrsVisitor {
public:
  using AttrsVisitor::Post;
  using AttrsVisitor::Pre;
  void beginDeclTypeSpec();
  void endDeclTypeSpec();
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

// Walk the parse tree and resolve names to symbols.
class ResolveNamesVisitor : public DeclTypeSpecVisitor {
public:
  using DeclTypeSpecVisitor::Post;
  using DeclTypeSpecVisitor::Pre;

  ResolveNamesVisitor() { PushScope(Scope::globalScope); }

  Scope &CurrScope() { return *scopes_.top(); }
  void PushScope(Scope &scope) { scopes_.push(&scope); }
  void PopScope() { scopes_.pop(); }

  // Default action for a parse tree node is to visit children.
  template<typename T> bool Pre(const T &) { return true; }
  template<typename T> void Post(const T &) {}

  bool Pre(const parser::TypeDeclarationStmt &);
  void Post(const parser::TypeDeclarationStmt &);
  void Post(const parser::EntityDecl &);
  bool Pre(const parser::PrefixSpec &);
  void Post(const parser::EndSubroutineStmt &);
  void Post(const parser::EndFunctionStmt &);
  bool Pre(const parser::Suffix &);
  bool Pre(const parser::SubroutineStmt &);
  void Post(const parser::SubroutineStmt &);
  bool Pre(const parser::FunctionStmt &);
  void Post(const parser::FunctionStmt &);
  void Post(const parser::Program &);

private:
  // Stack of containing scopes; memory referenced is owned by parent scopes
  std::stack<Scope *, std::list<Scope *>> scopes_;
  std::optional<Name> funcResultName_;

  // Common Post() for functions and subroutines.
  // Create a symbol in the current scope, push a new scope, add the dummies.
  void PostSubprogram(const Name &name, const std::list<Name> &dummyNames);

  // Helpers to make a Symbol in the current scope
  template<typename D> Symbol &MakeSymbol(const Name &name, D &&details) {
    return CurrScope().MakeSymbol(name, details);
  }
  template<typename D>
  Symbol &MakeSymbol(const Name &name, const Attrs &attrs, D &&details) {
    return CurrScope().MakeSymbol(name, attrs, details);
  }
};

// AttrsVisitor implementation
void AttrsVisitor::beginAttrs() {
  CHECK(!attrs_);
  attrs_ = std::make_unique<Attrs>();
}
Attrs AttrsVisitor::endAttrs() {
  const auto result = attrs_ ? *attrs_ : Attrs::EMPTY;
  attrs_.reset();
  return result;
}
void AttrsVisitor::Post(const parser::LanguageBindingSpec &x) {
  attrs_->Set(Attr::BIND_C);
  if (x.v) {
    // TODO: set langBindingName_ from ScalarDefaultCharConstantExpr
  }
}
bool AttrsVisitor::Pre(const parser::AccessSpec &x) {
  switch (x.v) {
  case parser::AccessSpec::Kind::Public: attrs_->Set(Attr::PUBLIC); break;
  case parser::AccessSpec::Kind::Private: attrs_->Set(Attr::PRIVATE); break;
  default: CRASH_NO_CASE;
  }
  return false;
}
bool AttrsVisitor::Pre(const parser::IntentSpec &x) {
  switch (x.v) {
  case parser::IntentSpec::Intent::In:
    attrs_->Set(Attr::INTENT_IN);
    break;
  case parser::IntentSpec::Intent::Out:
    attrs_->Set(Attr::INTENT_OUT);
    break;
  case parser::IntentSpec::Intent::InOut:
    attrs_->Set(Attr::INTENT_IN);
    attrs_->Set(Attr::INTENT_OUT);
    break;
  default: CRASH_NO_CASE;
  }
  return false;
}

// DeclTypeSpecVisitor implementation
void DeclTypeSpecVisitor::beginDeclTypeSpec() {
  CHECK(!expectDeclTypeSpec_);
  expectDeclTypeSpec_ = true;
}
void DeclTypeSpecVisitor::endDeclTypeSpec() {
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
  typeParamValue_ = std::make_unique<ParamValue>(
    std::visit(parser::visitors{
      [&](const parser::ScalarIntExpr &x) { return Bound{IntExpr{x}}; },
      [&](const parser::Star &x) { return Bound::ASSUMED; },
      [&](const parser::TypeParamValue::Deferred &x) { return Bound::DEFERRED; },
    }, x.u));
  return false;
}

void DeclTypeSpecVisitor::Post(const parser::DeclarationTypeSpec::Type &x) {
  SetDeclTypeSpec(
      DeclTypeSpec::MakeTypeDerivedType(*derivedTypeSpec_.release()));
}
void DeclTypeSpecVisitor::Post(const parser::DeclarationTypeSpec::Class &x) {
  SetDeclTypeSpec(
      DeclTypeSpec::MakeClassDerivedType(*derivedTypeSpec_.release()));
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
  CHECK(expectDeclTypeSpec_ && !declTypeSpec_);
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
  }
}


// ResolveNamesVisitor implementation

void ResolveNamesVisitor::Post(const parser::EntityDecl &x) {
  // TODO: may be under StructureStmt
  const auto &name = std::get<parser::ObjectName>(x.t);
  // TODO: optional ArraySpec, CoarraySpec, CharLength, Initialization
  Symbol &symbol = CurrScope().GetOrMakeSymbol(name.ToString());
  symbol.attrs().Add(*attrs_);  //TODO: check attribute consistency
  if (symbol.has<UnknownDetails>()) {
    symbol.set_details(EntityDetails());
  }
  if (EntityDetails *details = symbol.detailsIf<EntityDetails>()) {
    if (details->type().has_value()) {
      std::cerr << "ERROR: symbol already has a type declared: "
          << name.ToString() << "\n";
    } else {
      details->set_type(*declTypeSpec_);
    }
  } else {
    std::cerr
        << "ERROR: symbol already declared, can't appear in entity-decl: "
        << name.ToString() << "\n";
  }
}

bool ResolveNamesVisitor::Pre(const parser::TypeDeclarationStmt &x) {
  beginDeclTypeSpec();
  beginAttrs();
  return true;
}

void ResolveNamesVisitor::Post(const parser::TypeDeclarationStmt &x) {
  endDeclTypeSpec();
  endAttrs();
}

bool ResolveNamesVisitor::Pre(const parser::PrefixSpec &stmt) {
  return true;  // TODO
}

void ResolveNamesVisitor::Post(const parser::EndSubroutineStmt &subp) {
  std::cout << "End of subroutine scope\n";
  std::cout << CurrScope();
  PopScope();
}

void ResolveNamesVisitor::Post(const parser::EndFunctionStmt &subp) {
  std::cout << "End of function scope\n";
  std::cout << CurrScope();
  PopScope();
}

bool ResolveNamesVisitor::Pre(const parser::Suffix &suffix) {
  if (suffix.resultName.has_value()) {
    funcResultName_ = std::make_optional(suffix.resultName->ToString());
  }
  return true;
}

bool ResolveNamesVisitor::Pre(const parser::SubroutineStmt &stmt) {
  beginAttrs();
  return true;
}

void ResolveNamesVisitor::Post(const parser::SubroutineStmt &stmt) {
  Name subrName = std::get<parser::Name>(stmt.t).ToString();
  std::list<Name> dummyNames;
  const auto &dummyArgs = std::get<std::list<parser::DummyArg>>(stmt.t);
  for (const parser::DummyArg &dummyArg : dummyArgs) {
    const parser::Name *dummyName = std::get_if<parser::Name>(&dummyArg.u);
    CHECK(dummyName != nullptr && "TODO: alternate return indicator");
    dummyNames.push_back(dummyName->ToString());
  }
  PostSubprogram(subrName, dummyNames);
  MakeSymbol(subrName, SubprogramDetails(dummyNames));
}

bool ResolveNamesVisitor::Pre(const parser::FunctionStmt &stmt) {
  beginAttrs();
  beginDeclTypeSpec();
  CHECK(!funcResultName_);
  return true;
}

void ResolveNamesVisitor::Post(const parser::FunctionStmt &stmt) {
  Name funcName = std::get<parser::Name>(stmt.t).ToString();
  std::list<Name> dummyNames;
  for (const auto &dummy : std::get<std::list<parser::Name>>(stmt.t)) {
    dummyNames.push_back(dummy.ToString());
  }
  PostSubprogram(funcName, dummyNames);
  // add function result to function scope
  EntityDetails funcResultDetails;
  if (declTypeSpec_) {
    funcResultDetails.set_type(*declTypeSpec_);
  }
  const auto &resultName = funcResultName_ ? *funcResultName_ : funcName;
  MakeSymbol(resultName, funcResultDetails);
  if (resultName != funcName) {
    // add symbol for function to its scope; name can't be reused
    MakeSymbol(funcName, SubprogramDetails(dummyNames, funcResultName_));
  }
  endDeclTypeSpec();
  funcResultName_ = std::nullopt;
}

void ResolveNamesVisitor::PostSubprogram(const Name &name, const std::list<Name> &dummyNames) {
  const auto attrs = endAttrs();
  MakeSymbol(name, attrs, SubprogramDetails(dummyNames));
  Scope &subpScope = CurrScope().MakeScope(Scope::Kind::Subprogram);
  PushScope(subpScope);
  for (const auto &dummyName : dummyNames) {
    MakeSymbol(dummyName, EntityDetails(true));
  }
}

void ResolveNamesVisitor::Post(const parser::Program &) {
  // ensure that all temps were deallocated
  CHECK(!attrs_);
  CHECK(!declTypeSpec_);
}


void ResolveNames(const parser::Program &program) {
  ResolveNamesVisitor visitor;
  parser::Walk(program, visitor);
}

}  // namespace Fortran::semantics
