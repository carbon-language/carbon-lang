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

using namespace parser::literals;

class MessageHandler;

// ImplicitRules maps initial character of identifier to the DeclTypeSpec*
// representing the implicit type; nullptr if none.
class ImplicitRules {
public:
  ImplicitRules(MessageHandler &messages);
  // Get the implicit type for identifiers starting with ch. May be null.
  const DeclTypeSpec *GetType(char ch) const;
  // Record the implicit type for this range of characters.
  void SetType(const DeclTypeSpec &type, parser::Location lo, parser::Location,
      bool isDefault = false);
  // Apply the default implicit rules (if no IMPLICIT NONE).
  void ApplyDefaultRules();

private:
  static char Incr(char ch);

  MessageHandler &messages_;
  // map initial character of identifier to nullptr or its default type
  std::map<char, const DeclTypeSpec> map_;
  friend std::ostream &operator<<(std::ostream &, const ImplicitRules &);
  friend void ShowImplicitRule(std::ostream &, const ImplicitRules &, char);
};

// Provide Post methods to collect attributes into a member variable.
class AttrsVisitor {
public:
  void beginAttrs();
  Attrs endAttrs();
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

// Track statement source locations and save messages.
class MessageHandler {
public:
  using Message = parser::Message;

  MessageHandler(parser::Messages &messages) : messages_{messages} {}

  template<typename T> bool Pre(const parser::Statement<T> &x) {
    currStmtSource_ = &x.source;
    return true;
  }
  template<typename T> void Post(const parser::Statement<T> &) {
    currStmtSource_ = nullptr;
  }

  const parser::CharBlock *currStmtSource() { return currStmtSource_; }

  // Emit a message associated with the current statement source.
  void Say(Message &&);
  void Say(parser::MessageFixedText &&);
  void Say(parser::MessageFormattedText &&);

private:
  // Where messages are emitted:
  parser::Messages &messages_;
  // Source location of current statement; null if not in a statement
  const parser::CharBlock *currStmtSource_{nullptr};
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
  void Post(const parser::ImplicitPart &);

protected:
  void PushScope();
  void PopScope();

private:
  // implicit rules in effect for current scope
  std::stack<ImplicitRules, std::list<ImplicitRules>> implicitRules_;
  // previous occurence of these kinds of statements:
  const parser::CharBlock *prevImplicit_{nullptr};
  const parser::CharBlock *prevImplicitNone_{nullptr};
  const parser::CharBlock *prevImplicitNoneType_{nullptr};
  const parser::CharBlock *prevParameterStmt_{nullptr};

  bool HandleImplicitNone(const std::list<ImplicitNoneNameSpec> &nameSpecs);
};

// Walk the parse tree and resolve names to symbols.
class ResolveNamesVisitor : public ImplicitRulesVisitor {
public:
  using ImplicitRulesVisitor::Post;
  using ImplicitRulesVisitor::Pre;

  ResolveNamesVisitor(parser::Messages &messages)
    : ImplicitRulesVisitor(messages) {
    PushScope(Scope::globalScope);
  }

  Scope &CurrScope() { return *scopes_.top(); }
  void PushScope(Scope &scope) {
    scopes_.push(&scope);
    ImplicitRulesVisitor::PushScope();
  }
  void PopScope() {
    scopes_.pop();
    ImplicitRulesVisitor::PopScope();
  }

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
      messages_.Say(parser::Message{lo,
          parser::MessageFormattedText{
              "More than one implicit type specified for '%c'"_err_en_US, ch}});
    }
    if (ch == *hi) {
      break;
    }
  }
}

void ImplicitRules::ApplyDefaultRules() {
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

void AttrsVisitor::beginAttrs() {
  CHECK(!attrs_);
  attrs_ = std::make_optional<Attrs>();
}
Attrs AttrsVisitor::endAttrs() {
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
  switch (x.v) {
  case parser::AccessSpec::Kind::Public: attrs_->set(Attr::PUBLIC); break;
  case parser::AccessSpec::Kind::Private: attrs_->set(Attr::PRIVATE); break;
  default: CRASH_NO_CASE;
  }
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
  }
}

// MessageHandler implementation

void MessageHandler::Say(Message &&x) { messages_.Put(std::move(x)); }

void MessageHandler::Say(parser::MessageFixedText &&x) {
  CHECK(currStmtSource_);
  messages_.Put(Message{currStmtSource_->begin(), std::move(x)});
}

void MessageHandler::Say(parser::MessageFormattedText &&x) {
  CHECK(currStmtSource_);
  messages_.Put(Message{currStmtSource_->begin(), std::move(x)});
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
      Say(Message{hiLoc,
          parser::MessageFormattedText{
              "'%c' does not follow '%c' alphabetically"_err_en_US, *hiLoc,
              *loLoc}});
      return false;
    }
  }
  implicitRules_.top().SetType(*declTypeSpec_.get(), loLoc, hiLoc);
  return false;
}

void ImplicitRulesVisitor::Post(const parser::ImplicitPart &) {
  if (!prevImplicitNoneType_) {
    implicitRules_.top().ApplyDefaultRules();
  }
}

bool ImplicitRulesVisitor::Pre(const parser::ImplicitSpec &) {
  beginDeclTypeSpec();
  return true;
}

void ImplicitRulesVisitor::Post(const parser::ImplicitSpec &) {
  endDeclTypeSpec();
}

void ImplicitRulesVisitor::PushScope() {
  implicitRules_.push(ImplicitRules(*this));
  prevImplicit_ = nullptr;
  prevImplicitNone_ = nullptr;
  prevImplicitNoneType_ = nullptr;
  prevParameterStmt_ = nullptr;
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
        ++sawExternal;
        // TODO:
        // C894 If IMPLICIT NONE with an implicit-none-spec of EXTERNAL
        // appears within a scoping unit, the  name of an external or dummy
        // procedure in that scoping unit or in a contained subprogram or
        // BLOCK  construct shall have an explicit interface or be explicitly
        // declared to have the EXTERNAL attribute.
        break;
      case ImplicitNoneNameSpec::Type:
        prevImplicitNoneType_ = currStmtSource();
        if (prevImplicit_) {
          Say("IMPLICIT NONE(TYPE) after IMPLICIT statement"_err_en_US);
          return false;
        }
        ++sawType;
        break;
      default: CRASH_NO_CASE;
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

// ResolveNamesVisitor implementation

void ResolveNamesVisitor::Post(const parser::EntityDecl &x) {
  // TODO: may be under StructureStmt
  const auto &name{std::get<parser::ObjectName>(x.t)};
  // TODO: optional ArraySpec, CoarraySpec, CharLength, Initialization
  Symbol &symbol{CurrScope().GetOrMakeSymbol(name.ToString())};
  symbol.attrs() |= *attrs_;  // TODO: check attribute consistency
  if (symbol.has<UnknownDetails>()) {
    symbol.set_details(EntityDetails());
  }
  if (EntityDetails *details = symbol.detailsIf<EntityDetails>()) {
    if (details->type().has_value()) {
      Say(parser::Message{name.source.begin(),
          parser::MessageFormattedText{
              "'%s' already has a type declared"_err_en_US,
              name.ToString().c_str()}});
    } else {
      details->set_type(*declTypeSpec_);
    }
  } else {
    Say(parser::Message{name.source.begin(),
        parser::MessageFormattedText{
            "'%s' is already declared"_err_en_US, name.ToString().c_str()}});
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

void ResolveNamesVisitor::PostSubprogram(
    const Name &name, const std::list<Name> &dummyNames) {
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

void ResolveNames(
    const parser::Program &program, const parser::CookedSource &cookedSource) {
  parser::Messages messages{cookedSource};
  ResolveNamesVisitor visitor{messages};
  parser::Walk(program, visitor);
  messages.Emit(std::cerr);
}

}  // namespace Fortran::semantics
