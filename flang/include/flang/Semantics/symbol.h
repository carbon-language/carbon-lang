//===-- include/flang/Semantics/symbol.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_SYMBOL_H_
#define FORTRAN_SEMANTICS_SYMBOL_H_

#include "type.h"
#include "flang/Common/Fortran.h"
#include "flang/Common/enum-set.h"
#include "flang/Common/reference.h"
#include "llvm/ADT/DenseMapInfo.h"
#include <array>
#include <functional>
#include <list>
#include <optional>
#include <set>
#include <vector>

namespace llvm {
class raw_ostream;
}

namespace Fortran::semantics {

/// A Symbol consists of common information (name, owner, and attributes)
/// and details information specific to the kind of symbol, represented by the
/// *Details classes.

class Scope;
class Symbol;
class ProgramTree;

using SymbolRef = common::Reference<const Symbol>;
using SymbolVector = std::vector<SymbolRef>;
using MutableSymbolRef = common::Reference<Symbol>;
using MutableSymbolVector = std::vector<MutableSymbolRef>;

// A module or submodule.
class ModuleDetails {
public:
  ModuleDetails(bool isSubmodule = false) : isSubmodule_{isSubmodule} {}
  bool isSubmodule() const { return isSubmodule_; }
  const Scope *scope() const { return scope_; }
  const Scope *ancestor() const; // for submodule; nullptr for module
  const Scope *parent() const; // for submodule; nullptr for module
  void set_scope(const Scope *);

private:
  bool isSubmodule_;
  const Scope *scope_{nullptr};
};

class MainProgramDetails {
public:
private:
};

class WithBindName {
public:
  const std::string *bindName() const {
    return bindName_ ? &*bindName_ : nullptr;
  }
  void set_bindName(std::string &&name) { bindName_ = std::move(name); }

private:
  std::optional<std::string> bindName_;
};

class SubprogramDetails : public WithBindName {
public:
  bool isFunction() const { return result_ != nullptr; }
  bool isInterface() const { return isInterface_; }
  void set_isInterface(bool value = true) { isInterface_ = value; }
  bool isDummy() const { return isDummy_; }
  void set_isDummy(bool value = true) { isDummy_ = value; }
  Scope *entryScope() { return entryScope_; }
  const Scope *entryScope() const { return entryScope_; }
  void set_entryScope(Scope &scope) { entryScope_ = &scope; }
  const Symbol &result() const {
    CHECK(isFunction());
    return *result_;
  }
  void set_result(Symbol &result) {
    CHECK(!result_);
    result_ = &result;
  }
  const std::vector<Symbol *> &dummyArgs() const { return dummyArgs_; }
  void add_dummyArg(Symbol &symbol) { dummyArgs_.push_back(&symbol); }
  void add_alternateReturn() { dummyArgs_.push_back(nullptr); }
  const MaybeExpr &stmtFunction() const { return stmtFunction_; }
  void set_stmtFunction(SomeExpr &&expr) { stmtFunction_ = std::move(expr); }

private:
  bool isInterface_{false}; // true if this represents an interface-body
  bool isDummy_{false}; // true when interface of dummy procedure
  std::vector<Symbol *> dummyArgs_; // nullptr -> alternate return indicator
  Symbol *result_{nullptr};
  Scope *entryScope_{nullptr}; // if ENTRY, points to subprogram's scope
  MaybeExpr stmtFunction_;
  friend llvm::raw_ostream &operator<<(
      llvm::raw_ostream &, const SubprogramDetails &);
};

// For SubprogramNameDetails, the kind indicates whether it is the name
// of a module subprogram or an internal subprogram or ENTRY.
ENUM_CLASS(SubprogramKind, Module, Internal)

// Symbol with SubprogramNameDetails is created when we scan for module and
// internal procedure names, to record that there is a subprogram with this
// name. Later they are replaced by SubprogramDetails with dummy and result
// type information.
class SubprogramNameDetails {
public:
  SubprogramNameDetails(SubprogramKind kind, ProgramTree &node)
      : kind_{kind}, node_{node} {}
  SubprogramNameDetails() = delete;
  SubprogramKind kind() const { return kind_; }
  ProgramTree &node() const { return *node_; }
  bool isEntryStmt() const { return isEntryStmt_; }
  SubprogramNameDetails &set_isEntryStmt(bool yes = true) {
    isEntryStmt_ = yes;
    return *this;
  }

private:
  SubprogramKind kind_;
  common::Reference<ProgramTree> node_;
  bool isEntryStmt_{false};
};

// A name from an entity-decl -- could be object or function.
class EntityDetails : public WithBindName {
public:
  explicit EntityDetails(bool isDummy = false) : isDummy_{isDummy} {}
  const DeclTypeSpec *type() const { return type_; }
  void set_type(const DeclTypeSpec &);
  void ReplaceType(const DeclTypeSpec &);
  bool isDummy() const { return isDummy_; }
  void set_isDummy(bool value = true) { isDummy_ = value; }
  bool isFuncResult() const { return isFuncResult_; }
  void set_funcResult(bool x) { isFuncResult_ = x; }

private:
  bool isDummy_{false};
  bool isFuncResult_{false};
  const DeclTypeSpec *type_{nullptr};
  friend llvm::raw_ostream &operator<<(
      llvm::raw_ostream &, const EntityDetails &);
};

// Symbol is associated with a name or expression in a SELECT TYPE or ASSOCIATE.
class AssocEntityDetails : public EntityDetails {
public:
  AssocEntityDetails() {}
  explicit AssocEntityDetails(SomeExpr &&expr) : expr_{std::move(expr)} {}
  AssocEntityDetails(const AssocEntityDetails &) = default;
  AssocEntityDetails(AssocEntityDetails &&) = default;
  AssocEntityDetails &operator=(const AssocEntityDetails &) = default;
  AssocEntityDetails &operator=(AssocEntityDetails &&) = default;
  const MaybeExpr &expr() const { return expr_; }
  void set_rank(int rank);
  std::optional<int> rank() const { return rank_; }

private:
  MaybeExpr expr_;
  std::optional<int> rank_;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const AssocEntityDetails &);

// An entity known to be an object.
class ObjectEntityDetails : public EntityDetails {
public:
  explicit ObjectEntityDetails(EntityDetails &&);
  ObjectEntityDetails(const ObjectEntityDetails &) = default;
  ObjectEntityDetails &operator=(const ObjectEntityDetails &) = default;
  ObjectEntityDetails(bool isDummy = false) : EntityDetails(isDummy) {}
  MaybeExpr &init() { return init_; }
  const MaybeExpr &init() const { return init_; }
  void set_init(MaybeExpr &&expr) { init_ = std::move(expr); }
  ArraySpec &shape() { return shape_; }
  const ArraySpec &shape() const { return shape_; }
  ArraySpec &coshape() { return coshape_; }
  const ArraySpec &coshape() const { return coshape_; }
  void set_shape(const ArraySpec &);
  void set_coshape(const ArraySpec &);
  const Symbol *commonBlock() const { return commonBlock_; }
  void set_commonBlock(const Symbol &commonBlock) {
    commonBlock_ = &commonBlock;
  }
  bool IsArray() const { return !shape_.empty(); }
  bool IsCoarray() const { return !coshape_.empty(); }
  bool CanBeAssumedShape() const {
    return isDummy() && shape_.CanBeAssumedShape();
  }
  bool CanBeDeferredShape() const { return shape_.CanBeDeferredShape(); }
  bool IsAssumedSize() const { return isDummy() && shape_.CanBeAssumedSize(); }
  bool IsAssumedRank() const { return isDummy() && shape_.IsAssumedRank(); }

private:
  MaybeExpr init_;
  ArraySpec shape_;
  ArraySpec coshape_;
  const Symbol *commonBlock_{nullptr}; // common block this object is in
  friend llvm::raw_ostream &operator<<(
      llvm::raw_ostream &, const ObjectEntityDetails &);
};

// Mixin for details with passed-object dummy argument.
// If a procedure pointer component or type-bound procedure does not have
// the NOPASS attribute on its symbol, then PASS is assumed; the name
// is optional; if it is missing, the first dummy argument of the procedure's
// interface is the passed-object dummy argument.
class WithPassArg {
public:
  std::optional<SourceName> passName() const { return passName_; }
  void set_passName(const SourceName &passName) { passName_ = passName; }

private:
  std::optional<SourceName> passName_;
};

// A procedure pointer, dummy procedure, or external procedure
class ProcEntityDetails : public EntityDetails, public WithPassArg {
public:
  ProcEntityDetails() = default;
  explicit ProcEntityDetails(EntityDetails &&d);

  const ProcInterface &interface() const { return interface_; }
  ProcInterface &interface() { return interface_; }
  void set_interface(const ProcInterface &interface) { interface_ = interface; }
  bool IsInterfaceSet() {
    return interface_.symbol() != nullptr || interface_.type() != nullptr;
  }
  inline bool HasExplicitInterface() const;

  // Be advised: !init().has_value() => uninitialized pointer,
  // while *init() == nullptr => explicit NULL() initialization.
  std::optional<const Symbol *> init() const { return init_; }
  void set_init(const Symbol &symbol) { init_ = &symbol; }
  void set_init(std::nullptr_t) { init_ = nullptr; }

private:
  ProcInterface interface_;
  std::optional<const Symbol *> init_;
  friend llvm::raw_ostream &operator<<(
      llvm::raw_ostream &, const ProcEntityDetails &);
};

// These derived type details represent the characteristics of a derived
// type definition that are shared by all instantiations of that type.
// The DerivedTypeSpec instances whose type symbols share these details
// each own a scope into which the components' symbols have been cloned
// and specialized for each distinct set of type parameter values.
class DerivedTypeDetails {
public:
  const std::list<SourceName> &paramNames() const { return paramNames_; }
  const SymbolVector &paramDecls() const { return paramDecls_; }
  bool sequence() const { return sequence_; }
  bool isDECStructure() const { return isDECStructure_; }
  std::map<SourceName, SymbolRef> &finals() { return finals_; }
  const std::map<SourceName, SymbolRef> &finals() const { return finals_; }
  bool isForwardReferenced() const { return isForwardReferenced_; }
  void add_paramName(const SourceName &name) { paramNames_.push_back(name); }
  void add_paramDecl(const Symbol &symbol) { paramDecls_.push_back(symbol); }
  void add_component(const Symbol &);
  void set_sequence(bool x = true) { sequence_ = x; }
  void set_isDECStructure(bool x = true) { isDECStructure_ = x; }
  void set_isForwardReferenced(bool value) { isForwardReferenced_ = value; }
  const std::list<SourceName> &componentNames() const {
    return componentNames_;
  }

  // If this derived type extends another, locate the parent component's symbol.
  const Symbol *GetParentComponent(const Scope &) const;

  std::optional<SourceName> GetParentComponentName() const {
    if (componentNames_.empty()) {
      return std::nullopt;
    } else {
      return componentNames_.front();
    }
  }

  const Symbol *GetFinalForRank(int) const;

private:
  // These are (1) the names of the derived type parameters in the order
  // in which they appear on the type definition statement(s), and (2) the
  // symbols that correspond to those names in the order in which their
  // declarations appear in the derived type definition(s).
  std::list<SourceName> paramNames_;
  SymbolVector paramDecls_;
  // These are the names of the derived type's components in component
  // order.  A parent component, if any, appears first in this list.
  std::list<SourceName> componentNames_;
  std::map<SourceName, SymbolRef> finals_; // FINAL :: subr
  bool sequence_{false};
  bool isDECStructure_{false};
  bool isForwardReferenced_{false};
  friend llvm::raw_ostream &operator<<(
      llvm::raw_ostream &, const DerivedTypeDetails &);
};

class ProcBindingDetails : public WithPassArg {
public:
  explicit ProcBindingDetails(const Symbol &symbol) : symbol_{symbol} {}
  const Symbol &symbol() const { return symbol_; }

private:
  SymbolRef symbol_; // procedure bound to; may be forward
};

class NamelistDetails {
public:
  const SymbolVector &objects() const { return objects_; }
  void add_object(const Symbol &object) { objects_.push_back(object); }
  void add_objects(const SymbolVector &objects) {
    objects_.insert(objects_.end(), objects.begin(), objects.end());
  }

private:
  SymbolVector objects_;
};

class CommonBlockDetails : public WithBindName {
public:
  MutableSymbolVector &objects() { return objects_; }
  const MutableSymbolVector &objects() const { return objects_; }
  void add_object(Symbol &object) { objects_.emplace_back(object); }
  std::size_t alignment() const { return alignment_; }
  void set_alignment(std::size_t alignment) { alignment_ = alignment; }

private:
  MutableSymbolVector objects_;
  std::size_t alignment_{0}; // required alignment in bytes
};

class MiscDetails {
public:
  ENUM_CLASS(Kind, None, ConstructName, ScopeName, PassName, ComplexPartRe,
      ComplexPartIm, KindParamInquiry, LenParamInquiry, SelectRankAssociateName,
      SelectTypeAssociateName, TypeBoundDefinedOp);
  MiscDetails(Kind kind) : kind_{kind} {}
  Kind kind() const { return kind_; }

private:
  Kind kind_;
};

class TypeParamDetails {
public:
  explicit TypeParamDetails(common::TypeParamAttr attr) : attr_{attr} {}
  TypeParamDetails(const TypeParamDetails &) = default;
  common::TypeParamAttr attr() const { return attr_; }
  MaybeIntExpr &init() { return init_; }
  const MaybeIntExpr &init() const { return init_; }
  void set_init(MaybeIntExpr &&expr) { init_ = std::move(expr); }
  const DeclTypeSpec *type() const { return type_; }
  void set_type(const DeclTypeSpec &);
  void ReplaceType(const DeclTypeSpec &);

private:
  common::TypeParamAttr attr_;
  MaybeIntExpr init_;
  const DeclTypeSpec *type_{nullptr};
};

// Record the USE of a symbol: location is where (USE statement or renaming);
// symbol is in the USEd module.
class UseDetails {
public:
  UseDetails(const SourceName &location, const Symbol &symbol)
      : location_{location}, symbol_{symbol} {}
  const SourceName &location() const { return location_; }
  const Symbol &symbol() const { return symbol_; }

private:
  SourceName location_;
  SymbolRef symbol_;
};

// A symbol with ambiguous use-associations. Record where they were so
// we can report the error if it is used.
class UseErrorDetails {
public:
  UseErrorDetails(const UseDetails &);
  UseErrorDetails &add_occurrence(const SourceName &, const Scope &);
  using listType = std::list<std::pair<SourceName, const Scope *>>;
  const listType occurrences() const { return occurrences_; };

private:
  listType occurrences_;
};

// A symbol host-associated from an enclosing scope.
class HostAssocDetails {
public:
  HostAssocDetails(const Symbol &symbol) : symbol_{symbol} {}
  const Symbol &symbol() const { return symbol_; }
  bool implicitOrSpecExprError{false};
  bool implicitOrExplicitTypeError{false};

private:
  SymbolRef symbol_;
};

// A GenericKind is one of: generic name, defined operator,
// defined assignment, intrinsic operator, or defined I/O.
struct GenericKind {
  ENUM_CLASS(OtherKind, Name, DefinedOp, Assignment, Concat)
  ENUM_CLASS(DefinedIo, // defined io
      ReadFormatted, ReadUnformatted, WriteFormatted, WriteUnformatted)
  GenericKind() : u{OtherKind::Name} {}
  template <typename T> GenericKind(const T &x) { u = x; }
  bool IsName() const { return Is(OtherKind::Name); }
  bool IsAssignment() const { return Is(OtherKind::Assignment); }
  bool IsDefinedOperator() const { return Is(OtherKind::DefinedOp); }
  bool IsIntrinsicOperator() const;
  bool IsOperator() const;
  std::string ToString() const;
  std::variant<OtherKind, common::NumericOperator, common::LogicalOperator,
      common::RelationalOperator, DefinedIo>
      u;

private:
  template <typename T> bool Has() const {
    return std::holds_alternative<T>(u);
  }
  bool Is(OtherKind) const;
};

// A generic interface or type-bound generic.
class GenericDetails {
public:
  GenericDetails() {}

  GenericKind kind() const { return kind_; }
  void set_kind(GenericKind kind) { kind_ = kind; }

  const SymbolVector &specificProcs() const { return specificProcs_; }
  const std::vector<SourceName> &bindingNames() const { return bindingNames_; }
  void AddSpecificProc(const Symbol &, SourceName bindingName);
  const SymbolVector &uses() const { return uses_; }

  // specific and derivedType indicate a specific procedure or derived type
  // with the same name as this generic. Only one of them may be set.
  Symbol *specific() { return specific_; }
  const Symbol *specific() const { return specific_; }
  void set_specific(Symbol &specific);
  Symbol *derivedType() { return derivedType_; }
  const Symbol *derivedType() const { return derivedType_; }
  void set_derivedType(Symbol &derivedType);
  void AddUse(const Symbol &);

  // Copy in specificProcs, specific, and derivedType from another generic
  void CopyFrom(const GenericDetails &);

  // Check that specific is one of the specificProcs. If not, return the
  // specific as a raw pointer.
  const Symbol *CheckSpecific() const;
  Symbol *CheckSpecific();

private:
  GenericKind kind_;
  // all of the specific procedures for this generic
  SymbolVector specificProcs_;
  std::vector<SourceName> bindingNames_;
  // Symbols used from other modules merged into this one
  SymbolVector uses_;
  // a specific procedure with the same name as this generic, if any
  Symbol *specific_{nullptr};
  // a derived type with the same name as this generic, if any
  Symbol *derivedType_{nullptr};
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const GenericDetails &);

class UnknownDetails {};

using Details = std::variant<UnknownDetails, MainProgramDetails, ModuleDetails,
    SubprogramDetails, SubprogramNameDetails, EntityDetails,
    ObjectEntityDetails, ProcEntityDetails, AssocEntityDetails,
    DerivedTypeDetails, UseDetails, UseErrorDetails, HostAssocDetails,
    GenericDetails, ProcBindingDetails, NamelistDetails, CommonBlockDetails,
    TypeParamDetails, MiscDetails>;
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const Details &);
std::string DetailsToString(const Details &);

class Symbol {
public:
  ENUM_CLASS(Flag,
      Function, // symbol is a function
      Subroutine, // symbol is a subroutine
      StmtFunction, // symbol is a statement function (Function is set too)
      Implicit, // symbol is implicitly typed
      ImplicitOrError, // symbol must be implicitly typed or it's an error
      ModFile, // symbol came from .mod file
      ParentComp, // symbol is the "parent component" of an extended type
      CrayPointer, CrayPointee,
      LocalityLocal, // named in LOCAL locality-spec
      LocalityLocalInit, // named in LOCAL_INIT locality-spec
      LocalityShared, // named in SHARED locality-spec
      InDataStmt, // initialized in a DATA statement, =>object, or /init/
      InNamelist, // in a Namelist group
      CompilerCreated,
      // OpenACC data-sharing attribute
      AccPrivate, AccFirstPrivate, AccShared,
      // OpenACC data-mapping attribute
      AccCopyIn, AccCopyOut, AccCreate, AccDelete, AccPresent,
      // OpenACC miscellaneous flags
      AccCommonBlock, AccThreadPrivate, AccReduction, AccNone, AccPreDetermined,
      // OpenMP data-sharing attribute
      OmpShared, OmpPrivate, OmpLinear, OmpFirstPrivate, OmpLastPrivate,
      // OpenMP data-mapping attribute
      OmpMapTo, OmpMapFrom, OmpMapAlloc, OmpMapRelease, OmpMapDelete,
      // OpenMP data-copying attribute
      OmpCopyIn, OmpCopyPrivate,
      // OpenMP miscellaneous flags
      OmpCommonBlock, OmpReduction, OmpAligned, OmpNontemporal, OmpAllocate,
      OmpDeclarativeAllocateDirective, OmpExecutableAllocateDirective,
      OmpDeclareSimd, OmpDeclareTarget, OmpThreadprivate, OmpDeclareReduction,
      OmpFlushed, OmpCriticalLock, OmpIfSpecified, OmpNone, OmpPreDetermined);
  using Flags = common::EnumSet<Flag, Flag_enumSize>;

  const Scope &owner() const { return *owner_; }
  const SourceName &name() const { return name_; }
  Attrs &attrs() { return attrs_; }
  const Attrs &attrs() const { return attrs_; }
  Flags &flags() { return flags_; }
  const Flags &flags() const { return flags_; }
  bool test(Flag flag) const { return flags_.test(flag); }
  void set(Flag flag, bool value = true) { flags_.set(flag, value); }
  // The Scope introduced by this symbol, if any.
  Scope *scope() { return scope_; }
  const Scope *scope() const { return scope_; }
  void set_scope(Scope *scope) { scope_ = scope; }
  std::size_t size() const { return size_; }
  void set_size(std::size_t size) { size_ = size; }
  std::size_t offset() const { return offset_; }
  void set_offset(std::size_t offset) { offset_ = offset; }
  // Give the symbol a name with a different source location but same chars.
  void ReplaceName(const SourceName &);

  // Does symbol have this type of details?
  template <typename D> bool has() const {
    return std::holds_alternative<D>(details_);
  }

  // Return a non-owning pointer to details if it is type D, else nullptr.
  template <typename D> D *detailsIf() { return std::get_if<D>(&details_); }
  template <typename D> const D *detailsIf() const {
    return std::get_if<D>(&details_);
  }

  // Return a reference to the details which must be of type D.
  template <typename D> D &get() {
    return const_cast<D &>(const_cast<const Symbol *>(this)->get<D>());
  }
  template <typename D> const D &get() const {
    const auto *p{detailsIf<D>()};
    CHECK(p);
    return *p;
  }

  Details &details() { return details_; }
  const Details &details() const { return details_; }
  // Assign the details of the symbol from one of the variants.
  // Only allowed in certain cases.
  void set_details(Details &&);

  // Can the details of this symbol be replaced with the given details?
  bool CanReplaceDetails(const Details &details) const;

  // Follow use-associations and host-associations to get the ultimate entity.
  inline Symbol &GetUltimate();
  inline const Symbol &GetUltimate() const;

  inline DeclTypeSpec *GetType();
  inline const DeclTypeSpec *GetType() const;
  void SetType(const DeclTypeSpec &);

  const std::string *GetBindName() const;
  void SetBindName(std::string &&);
  bool IsFuncResult() const;
  bool IsObjectArray() const;
  bool IsSubprogram() const;
  bool IsFromModFile() const;
  bool HasExplicitInterface() const {
    return std::visit(common::visitors{
                          [](const SubprogramDetails &) { return true; },
                          [](const SubprogramNameDetails &) { return true; },
                          [&](const ProcEntityDetails &x) {
                            return attrs_.test(Attr::INTRINSIC) ||
                                x.HasExplicitInterface();
                          },
                          [](const ProcBindingDetails &x) {
                            return x.symbol().HasExplicitInterface();
                          },
                          [](const UseDetails &x) {
                            return x.symbol().HasExplicitInterface();
                          },
                          [](const HostAssocDetails &x) {
                            return x.symbol().HasExplicitInterface();
                          },
                          [](const auto &) { return false; },
                      },
        details_);
  }

  bool operator==(const Symbol &that) const { return this == &that; }
  bool operator!=(const Symbol &that) const { return !(*this == that); }

  int Rank() const {
    return std::visit(
        common::visitors{
            [](const SubprogramDetails &sd) {
              return sd.isFunction() ? sd.result().Rank() : 0;
            },
            [](const GenericDetails &) {
              return 0; /*TODO*/
            },
            [](const ProcBindingDetails &x) { return x.symbol().Rank(); },
            [](const UseDetails &x) { return x.symbol().Rank(); },
            [](const HostAssocDetails &x) { return x.symbol().Rank(); },
            [](const ObjectEntityDetails &oed) { return oed.shape().Rank(); },
            [](const AssocEntityDetails &aed) {
              if (const auto &expr{aed.expr()}) {
                if (auto assocRank{aed.rank()}) {
                  return *assocRank;
                } else {
                  return expr->Rank();
                }
              } else {
                return 0;
              }
            },
            [](const auto &) { return 0; },
        },
        details_);
  }

  int Corank() const {
    return std::visit(
        common::visitors{
            [](const SubprogramDetails &sd) {
              return sd.isFunction() ? sd.result().Corank() : 0;
            },
            [](const GenericDetails &) {
              return 0; /*TODO*/
            },
            [](const UseDetails &x) { return x.symbol().Corank(); },
            [](const HostAssocDetails &x) { return x.symbol().Corank(); },
            [](const ObjectEntityDetails &oed) { return oed.coshape().Rank(); },
            [](const auto &) { return 0; },
        },
        details_);
  }

  // If there is a parent component, return a pointer to its derived type spec.
  // The Scope * argument defaults to this->scope_ but should be overridden
  // for a parameterized derived type instantiation with the instance's scope.
  const DerivedTypeSpec *GetParentTypeSpec(const Scope * = nullptr) const;

  SemanticsContext &GetSemanticsContext() const;
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif

private:
  const Scope *owner_;
  SourceName name_;
  Attrs attrs_;
  Flags flags_;
  Scope *scope_{nullptr};
  std::size_t size_{0}; // size in bytes
  std::size_t offset_{0}; // byte offset in scope or common block
  Details details_;

  Symbol() {} // only created in class Symbols
  const std::string GetDetailsName() const;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &, const Symbol &);
  friend llvm::raw_ostream &DumpForUnparse(
      llvm::raw_ostream &, const Symbol &, bool);

  // If a derived type's symbol refers to an extended derived type,
  // return the parent component's symbol.  The scope of the derived type
  // can be overridden.
  const Symbol *GetParentComponent(const Scope * = nullptr) const;

  template <std::size_t> friend class Symbols;
  template <class, std::size_t> friend class std::array;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &, Symbol::Flag);

// Manage memory for all symbols. BLOCK_SIZE symbols at a time are allocated.
// Make() returns a reference to the next available one. They are never
// deleted.
template <std::size_t BLOCK_SIZE> class Symbols {
public:
  Symbol &Make(const Scope &owner, const SourceName &name, const Attrs &attrs,
      Details &&details) {
    Symbol &symbol = Get();
    symbol.owner_ = &owner;
    symbol.name_ = name;
    symbol.attrs_ = attrs;
    symbol.details_ = std::move(details);
    return symbol;
  }

private:
  using blockType = std::array<Symbol, BLOCK_SIZE>;
  std::list<blockType *> blocks_;
  std::size_t nextIndex_{0};
  blockType *currBlock_{nullptr};

  Symbol &Get() {
    if (nextIndex_ == 0) {
      blocks_.push_back(new blockType());
      currBlock_ = blocks_.back();
    }
    Symbol &result = (*currBlock_)[nextIndex_];
    if (++nextIndex_ >= BLOCK_SIZE) {
      nextIndex_ = 0; // allocate a new block next time
    }
    return result;
  }
};

// Define a few member functions here in the header so that they
// can be used by lib/Evaluate without inducing a dependence cycle
// between the two shared libraries.

inline bool ProcEntityDetails::HasExplicitInterface() const {
  if (auto *symbol{interface_.symbol()}) {
    return symbol->HasExplicitInterface();
  }
  return false;
}

inline Symbol &Symbol::GetUltimate() {
  return const_cast<Symbol &>(const_cast<const Symbol *>(this)->GetUltimate());
}
inline const Symbol &Symbol::GetUltimate() const {
  if (const auto *details{detailsIf<UseDetails>()}) {
    return details->symbol().GetUltimate();
  } else if (const auto *details{detailsIf<HostAssocDetails>()}) {
    return details->symbol().GetUltimate();
  } else {
    return *this;
  }
}

inline DeclTypeSpec *Symbol::GetType() {
  return const_cast<DeclTypeSpec *>(
      const_cast<const Symbol *>(this)->GetType());
}
inline const DeclTypeSpec *Symbol::GetType() const {
  return std::visit(
      common::visitors{
          [](const EntityDetails &x) { return x.type(); },
          [](const ObjectEntityDetails &x) { return x.type(); },
          [](const AssocEntityDetails &x) { return x.type(); },
          [](const SubprogramDetails &x) {
            return x.isFunction() ? x.result().GetType() : nullptr;
          },
          [](const ProcEntityDetails &x) {
            const Symbol *symbol{x.interface().symbol()};
            return symbol ? symbol->GetType() : x.interface().type();
          },
          [](const ProcBindingDetails &x) { return x.symbol().GetType(); },
          [](const TypeParamDetails &x) { return x.type(); },
          [](const UseDetails &x) { return x.symbol().GetType(); },
          [](const HostAssocDetails &x) { return x.symbol().GetType(); },
          [](const auto &) -> const DeclTypeSpec * { return nullptr; },
      },
      details_);
}

// Sets and maps keyed by Symbols

struct SymbolAddressCompare {
  bool operator()(const SymbolRef &x, const SymbolRef &y) const {
    return &*x < &*y;
  }
  bool operator()(const MutableSymbolRef &x, const MutableSymbolRef &y) const {
    return &*x < &*y;
  }
};

// Symbol comparison is usually based on the order of cooked source
// stream creation and, when both are from the same cooked source,
// their positions in that cooked source stream.
// Don't use this comparator or OrderedSymbolSet to hold
// Symbols that might be subject to ReplaceName().
struct SymbolSourcePositionCompare {
  // These functions are implemented in Evaluate/tools.cpp to
  // satisfy complicated shared library interdependency.
  bool operator()(const SymbolRef &, const SymbolRef &) const;
  bool operator()(const MutableSymbolRef &, const MutableSymbolRef &) const;
};

struct SymbolOffsetCompare {
  bool operator()(const SymbolRef &, const SymbolRef &) const;
  bool operator()(const MutableSymbolRef &, const MutableSymbolRef &) const;
};

using UnorderedSymbolSet = std::set<SymbolRef, SymbolAddressCompare>;
using SourceOrderedSymbolSet = std::set<SymbolRef, SymbolSourcePositionCompare>;

template <typename A>
SourceOrderedSymbolSet OrderBySourcePosition(const A &container) {
  SourceOrderedSymbolSet result;
  for (SymbolRef x : container) {
    result.emplace(x);
  }
  return result;
}

} // namespace Fortran::semantics

// Define required  info so that SymbolRef can be used inside llvm::DenseMap.
namespace llvm {
template <> struct DenseMapInfo<Fortran::semantics::SymbolRef> {
  static inline Fortran::semantics::SymbolRef getEmptyKey() {
    auto ptr = DenseMapInfo<const Fortran::semantics::Symbol *>::getEmptyKey();
    return *reinterpret_cast<Fortran::semantics::SymbolRef *>(&ptr);
  }

  static inline Fortran::semantics::SymbolRef getTombstoneKey() {
    auto ptr =
        DenseMapInfo<const Fortran::semantics::Symbol *>::getTombstoneKey();
    return *reinterpret_cast<Fortran::semantics::SymbolRef *>(&ptr);
  }

  static unsigned getHashValue(const Fortran::semantics::SymbolRef &sym) {
    return DenseMapInfo<const Fortran::semantics::Symbol *>::getHashValue(
        &sym.get());
  }

  static bool isEqual(const Fortran::semantics::SymbolRef &LHS,
      const Fortran::semantics::SymbolRef &RHS) {
    return LHS == RHS;
  }
};
} // namespace llvm
#endif // FORTRAN_SEMANTICS_SYMBOL_H_
