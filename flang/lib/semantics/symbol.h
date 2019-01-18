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

#ifndef FORTRAN_SEMANTICS_SYMBOL_H_
#define FORTRAN_SEMANTICS_SYMBOL_H_

#include "type.h"
#include "../common/enum-set.h"
#include "../common/fortran.h"
#include <functional>
#include <list>
#include <optional>

namespace Fortran::evaluate {
struct FoldingContext;
}

namespace Fortran::semantics {

/// A Symbol consists of common information (name, owner, and attributes)
/// and details information specific to the kind of symbol, represented by the
/// *Details classes.

class Scope;
class Symbol;

using SymbolList = std::list<const Symbol *>;

// A module or submodule.
class ModuleDetails {
public:
  ModuleDetails(bool isSubmodule = false) : isSubmodule_{isSubmodule} {}
  bool isSubmodule() const { return isSubmodule_; }
  const Scope *scope() const { return scope_; }
  const Scope *ancestor() const;  // for submodule; nullptr for module
  const Scope *parent() const;  // for submodule; nullptr for module
  void set_scope(const Scope *);

private:
  bool isSubmodule_;
  const Scope *scope_{nullptr};
};

class MainProgramDetails {
public:
private:
};

class SubprogramDetails {
public:
  SubprogramDetails() {}
  SubprogramDetails(const SubprogramDetails &that)
    : dummyArgs_{that.dummyArgs_}, result_{that.result_} {}

  bool isFunction() const { return result_.has_value(); }
  bool isInterface() const { return isInterface_; }
  void set_isInterface(bool value = true) { isInterface_ = value; }
  MaybeExpr bindName() const { return bindName_; }
  void set_bindName(MaybeExpr &&expr) { bindName_ = std::move(expr); }
  const Symbol &result() const {
    CHECK(isFunction());
    return **result_;
  }
  void set_result(Symbol &result) {
    CHECK(!result_.has_value());
    result_ = &result;
  }
  const std::list<Symbol *> &dummyArgs() const { return dummyArgs_; }
  void add_dummyArg(Symbol &symbol) { dummyArgs_.push_back(&symbol); }

private:
  bool isInterface_{false};  // true if this represents an interface-body
  MaybeExpr bindName_;
  std::list<Symbol *> dummyArgs_;
  std::optional<Symbol *> result_;
  friend std::ostream &operator<<(std::ostream &, const SubprogramDetails &);
};

// For SubprogramNameDetails, the kind indicates whether it is the name
// of a module subprogram or internal subprogram.
ENUM_CLASS(SubprogramKind, Module, Internal)

// Symbol with SubprogramNameDetails is created when we scan for module and
// internal procedure names, to record that there is a subprogram with this
// name. Later they are replaced by SubprogramDetails with dummy and result
// type information.
class SubprogramNameDetails {
public:
  SubprogramNameDetails(SubprogramKind kind) : kind_{kind} {}
  SubprogramNameDetails() = delete;
  SubprogramKind kind() const { return kind_; }

private:
  SubprogramKind kind_;
};

// A name from an entity-decl -- could be object or function.
class EntityDetails {
public:
  explicit EntityDetails(bool isDummy = false) : isDummy_{isDummy} {}
  const DeclTypeSpec *type() const { return type_; }
  void set_type(const DeclTypeSpec &);
  void ReplaceType(const DeclTypeSpec &);
  bool isDummy() const { return isDummy_; }
  MaybeExpr bindName() const { return bindName_; }
  void set_bindName(MaybeExpr &&expr) { bindName_ = std::move(expr); }

private:
  bool isDummy_;
  const DeclTypeSpec *type_{nullptr};
  MaybeExpr bindName_;
  friend std::ostream &operator<<(std::ostream &, const EntityDetails &);
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

private:
  MaybeExpr expr_;
};

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
  void set_shape(const ArraySpec &shape);
  bool IsArray() const { return !shape_.empty(); }
  bool IsAssumedShape() const {
    return isDummy() && IsArray() && shape_.back().ubound().isDeferred() &&
        !shape_.back().lbound().isDeferred();
  }
  bool IsDeferredShape() const {
    return !isDummy() && IsArray() && shape_.back().ubound().isDeferred() &&
        shape_.back().lbound().isDeferred();
  }
  bool IsAssumedSize() const {
    return isDummy() && IsArray() && shape_.back().ubound().isAssumed() &&
        !shape_.back().lbound().isAssumed();
  }
  bool IsAssumedRank() const {
    return isDummy() && IsArray() && shape_.back().ubound().isAssumed() &&
        shape_.back().lbound().isAssumed();
  }
  bool IsDescriptor() const;

private:
  MaybeExpr init_;
  ArraySpec shape_;
  friend std::ostream &operator<<(std::ostream &, const ObjectEntityDetails &);
};

// A procedure pointer, dummy procedure, or external procedure
class ProcEntityDetails : public EntityDetails {
public:
  ProcEntityDetails() = default;
  explicit ProcEntityDetails(EntityDetails &&d);

  const ProcInterface &interface() const { return interface_; }
  ProcInterface &interface() { return interface_; }
  void set_interface(const ProcInterface &interface) { interface_ = interface; }
  bool HasExplicitInterface() const;
  bool IsDescriptor() const;
  const std::optional<SourceName> &passName() const { return passName_; }
  void set_passName(const SourceName &passName) { passName_ = passName; }

private:
  ProcInterface interface_;
  std::optional<SourceName> passName_;
  friend std::ostream &operator<<(std::ostream &, const ProcEntityDetails &);
};

class DerivedTypeDetails {
public:
  const std::list<SourceName> &paramNames() const { return paramNames_; }
  const std::list<Symbol *> &paramDecls() const { return paramDecls_; }
  SourceName extends() const { return extends_; }
  bool sequence() const { return sequence_; }
  void add_paramName(const SourceName &name) { paramNames_.emplace_back(name); }
  void add_paramDecl(Symbol &symbol) { paramDecls_.emplace_back(&symbol); }
  void set_extends(const SourceName &name) { extends_ = name; }
  void set_sequence(bool x = true) { sequence_ = x; }

  // Returns the complete list of derived type parameter names in the
  // order defined by 7.5.3.2.
  std::list<SourceName> OrderParameterNames(const Symbol &) const;

  // Returns the complete list of derived type parameter symbols in
  // the order in which their declarations appear in the derived type
  // definitions (parents first).
  std::list<Symbol *> OrderParameterDeclarations(const Symbol &) const;

private:
  // These are the names of the derived type parameters in (1) the order
  // in which they appear on the type definition statement, and (2) the
  // order in which their declarations appear in the derived type definition.
  std::list<SourceName> paramNames_;
  std::list<Symbol *> paramDecls_;
  SourceName extends_;
  bool sequence_{false};
};

class ProcBindingDetails {
public:
  explicit ProcBindingDetails(const Symbol &symbol) : symbol_{&symbol} {}
  const Symbol &symbol() const { return *symbol_; }
  std::optional<SourceName> passName() const { return passName_; }
  void set_passName(const SourceName &passName) { passName_ = passName; }

private:
  const Symbol *symbol_;  // procedure bound to
  std::optional<SourceName> passName_;  // name in PASS attribute
};

class GenericBindingDetails {
public:
  GenericBindingDetails() {}
  const SymbolList &specificProcs() const { return specificProcs_; }
  void add_specificProc(const Symbol &proc) { specificProcs_.push_back(&proc); }
  void add_specificProcs(const SymbolList &procs) {
    specificProcs_.insert(specificProcs_.end(), procs.begin(), procs.end());
  }

private:
  SymbolList specificProcs_;
};

class FinalProcDetails {};

class MiscDetails {
public:
  ENUM_CLASS(Kind, None, ConstructName, ScopeName, PassName, ComplexPartRe,
      ComplexPartIm, KindParamInquiry, LenParamInquiry,
      SelectTypeAssociateName);
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
// symbol is the USEd module.
class UseDetails {
public:
  UseDetails(const SourceName &location, const Symbol &symbol)
    : location_{location}, symbol_{&symbol} {}
  const SourceName &location() const { return location_; }
  const Symbol &symbol() const { return *symbol_; }
  const Symbol &module() const;

private:
  SourceName location_;
  const Symbol *symbol_;
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
  HostAssocDetails(const Symbol &symbol) : symbol_{&symbol} {}
  const Symbol &symbol() const { return *symbol_; }

private:
  const Symbol *symbol_;
};

class GenericDetails {
public:
  GenericDetails() {}
  GenericDetails(const SymbolList &specificProcs);
  GenericDetails(Symbol *specific) : specific_{specific} {}

  const SymbolList specificProcs() const { return specificProcs_; }

  void add_specificProc(const Symbol &proc) { specificProcs_.push_back(&proc); }

  Symbol *specific() { return specific_; }
  void set_specific(Symbol &specific);

  // Derived type with same name as generic, if any.
  Symbol *derivedType() { return derivedType_; }
  const Symbol *derivedType() const { return derivedType_; }
  void set_derivedType(Symbol &derivedType);

  // Check that specific is one of the specificProcs. If not, return the
  // specific as a raw pointer.
  const Symbol *CheckSpecific() const;

private:
  // all of the specific procedures for this generic
  SymbolList specificProcs_;
  // a specific procedure with the same name as this generic, if any
  Symbol *specific_{nullptr};
  // a derived type with the same name as this generic, if any
  Symbol *derivedType_{nullptr};
};

class UnknownDetails {};

using Details = std::variant<UnknownDetails, MainProgramDetails, ModuleDetails,
    SubprogramDetails, SubprogramNameDetails, EntityDetails,
    ObjectEntityDetails, ProcEntityDetails, AssocEntityDetails,
    DerivedTypeDetails, UseDetails, UseErrorDetails, HostAssocDetails,
    GenericDetails, ProcBindingDetails, GenericBindingDetails, FinalProcDetails,
    TypeParamDetails, MiscDetails>;
std::ostream &operator<<(std::ostream &, const Details &);
std::string DetailsToString(const Details &);

class Symbol {
public:
  ENUM_CLASS(Flag,
      Function,  // symbol is a function
      Subroutine,  // symbol is a subroutine
      Implicit,  // symbol is implicitly typed
      ModFile,  // symbol came from .mod file
      ParentComp,  // symbol is the "parent component" of an extended type
      LocalityLocal,  // named in LOCAL locality-spec
      LocalityLocalInit,  // named in LOCAL_INIT locality-spec
      LocalityShared  // named in SHARED locality-spec
  );
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

  // Does symbol have this type of details?
  template<typename D> bool has() const {
    return std::holds_alternative<D>(details_);
  }

  // Return a non-owning pointer to details if it is type D, else nullptr.
  template<typename D> D *detailsIf() { return std::get_if<D>(&details_); }
  template<typename D> const D *detailsIf() const {
    return std::get_if<D>(&details_);
  }

  // Return a reference to the details which must be of type D.
  template<typename D> D &get() {
    return const_cast<D &>(static_cast<const Symbol *>(this)->get<D>());
  }
  template<typename D> const D &get() const {
    if (const auto p{detailsIf<D>()}) {
      return *p;
    } else {
      common::die("unexpected %s details at %s(%d)", GetDetailsName().c_str(),
          __FILE__, __LINE__);
    }
  }

  Details &details() { return details_; }
  const Details &details() const { return details_; }
  // Assign the details of the symbol from one of the variants.
  // Only allowed in certain cases.
  void set_details(Details &&);

  // Can the details of this symbol be replaced with the given details?
  bool CanReplaceDetails(const Details &details) const;

  // Follow use-associations to get the ultimate entity.
  Symbol &GetUltimate();
  const Symbol &GetUltimate() const;

  DeclTypeSpec *GetType();
  const DeclTypeSpec *GetType() const;
  void SetType(const DeclTypeSpec &);

  bool IsSubprogram() const;
  bool HasExplicitInterface() const;
  bool IsSeparateModuleProc() const;
  bool IsDescriptor() const;

  bool operator==(const Symbol &that) const { return this == &that; }
  bool operator!=(const Symbol &that) const { return this != &that; }

  int Rank() const;

  // Clones the Symbol in the context of a parameterized derived type instance
  Symbol &Instantiate(Scope &, evaluate::FoldingContext &) const;

  // If there is a parent component, return a pointer to its
  // derived type spec.
  // The Scope * argument defaults to this->scope_ but should be overridden
  // for a parameterized derived type instantiation with the instance's scope.
  const DerivedTypeSpec *GetParentTypeSpec(const Scope * = nullptr) const;

private:
  const Scope *owner_;
  SourceName name_;
  Attrs attrs_;
  Flags flags_;
  Scope *scope_{nullptr};
  Details details_;

  Symbol() {}  // only created in class Symbols
  const std::string GetDetailsName() const;
  friend std::ostream &operator<<(std::ostream &, const Symbol &);
  friend std::ostream &DumpForUnparse(std::ostream &, const Symbol &, bool);

  // If the symbol refers to a derived type with a parent component,
  // return that parent component's symbol.
  const Symbol *GetParentComponent(const Scope * = nullptr) const;

  template<std::size_t> friend class Symbols;
  template<class, std::size_t> friend struct std::array;
};

std::ostream &operator<<(std::ostream &, Symbol::Flag);

// Manage memory for all symbols. BLOCK_SIZE symbols at a time are allocated.
// Make() returns a reference to the next available one. They are never
// deleted.
template<std::size_t BLOCK_SIZE> class Symbols {
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
      nextIndex_ = 0;  // allocate a new block next time
    }
    return result;
  }
};
}
#endif  // FORTRAN_SEMANTICS_SYMBOL_H_
