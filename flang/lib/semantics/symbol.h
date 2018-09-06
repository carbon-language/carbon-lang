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

#ifndef FORTRAN_SEMANTICS_SYMBOL_H_
#define FORTRAN_SEMANTICS_SYMBOL_H_

#include "type.h"
#include "../common/enum-set.h"
#include "../common/fortran.h"
#include <functional>
#include <memory>

namespace Fortran::semantics {

/// A Symbol consists of common information (name, owner, and attributes)
/// and details information specific to the kind of symbol, represented by the
/// *Details classes.

class Scope;
class Symbol;

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
  std::list<Symbol *> dummyArgs_;
  std::optional<Symbol *> result_;
  bool isInterface_{false};  // true if this represents an interface-body
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
  EntityDetails(bool isDummy = false) : isDummy_{isDummy} {}
  const std::optional<DeclTypeSpec> &type() const { return type_; }
  void set_type(const DeclTypeSpec &type);
  bool isDummy() const { return isDummy_; }

private:
  bool isDummy_;
  std::optional<DeclTypeSpec> type_;
  friend std::ostream &operator<<(std::ostream &, const EntityDetails &);
};

// An entity known to be an object.
class ObjectEntityDetails {
public:
  ObjectEntityDetails(const EntityDetails &);
  ObjectEntityDetails(bool isDummy = false) : isDummy_{isDummy} {}
  const std::optional<DeclTypeSpec> &type() const { return type_; }
  void set_type(const DeclTypeSpec &type);
  const ArraySpec &shape() const { return shape_; }
  void set_shape(const ArraySpec &shape);
  bool isDummy() const { return isDummy_; }
  bool isArray() const { return !shape_.empty(); }

private:
  bool isDummy_;
  std::optional<DeclTypeSpec> type_;
  ArraySpec shape_;
  friend std::ostream &operator<<(std::ostream &, const ObjectEntityDetails &);
};

// A procedure pointer, dummy procedure, or external procedure
class ProcEntityDetails {
public:
  ProcEntityDetails() = default;
  ProcEntityDetails(const EntityDetails &d);

  const ProcInterface &interface() const { return interface_; }
  ProcInterface &interface() { return interface_; }
  void set_interface(const ProcInterface &interface) { interface_ = interface; }
  bool HasExplicitInterface() const;

private:
  ProcInterface interface_;
  friend std::ostream &operator<<(std::ostream &, const ProcEntityDetails &);
};

class DerivedTypeDetails {
public:
  bool hasTypeParams() const { return hasTypeParams_; }
  const Symbol *extends() const { return extends_; }
  bool sequence() const { return sequence_; }
  void set_hasTypeParams(bool x = true) { hasTypeParams_ = x; }
  void set_extends(const Symbol *extends) { extends_ = extends; }
  void set_sequence(bool x = true) { sequence_ = x; }

private:
  bool hasTypeParams_{false};
  const Symbol *extends_{nullptr};
  bool sequence_{false};
};

class ProcBindingDetails {
public:
  ProcBindingDetails(const Symbol &symbol) : symbol_{&symbol} {}
  const Symbol &symbol() const { return *symbol_; }

private:
  const Symbol *symbol_;  // procedure bound to
};

class GenericBindingDetails {};

class FinalProcDetails {};

class TypeParamDetails {
public:
  TypeParamDetails(common::TypeParamAttr attr) : attr_{attr} {}
  common::TypeParamAttr attr() const { return attr_; }
  const std::optional<DeclTypeSpec> &type() const { return type_; }
  void set_type(const DeclTypeSpec &type) {
    CHECK(!type_);
    type_ = type;
  }

private:
  common::TypeParamAttr attr_;
  std::optional<DeclTypeSpec> type_;
};

// Record the USE of a symbol: location is where (USE statement or renaming);
// symbol is the USEd module.
class UseDetails {
public:
  UseDetails(const SourceName &location, const Symbol &symbol)
    : location_{&location}, symbol_{&symbol} {}
  const SourceName &location() const { return *location_; }
  const Symbol &symbol() const { return *symbol_; }
  const Symbol &module() const;

private:
  const SourceName *location_;
  const Symbol *symbol_;
};

// A symbol with ambiguous use-associations. Record where they were so
// we can report the error if it is used.
class UseErrorDetails {
public:
  UseErrorDetails(const UseDetails &);
  UseErrorDetails &add_occurrence(const SourceName &, const Scope &);

  using listType = std::list<std::pair<const SourceName *, const Scope *>>;
  const listType occurrences() const { return occurrences_; };

private:
  listType occurrences_;
};

class GenericDetails {
public:
  using listType = std::list<const Symbol *>;
  using procNamesType = std::list<std::pair<const SourceName *, bool>>;

  GenericDetails() {}
  GenericDetails(const listType &specificProcs);
  GenericDetails(Symbol *specific) : specific_{specific} {}

  const listType specificProcs() const { return specificProcs_; }
  const procNamesType specificProcNames() const { return specificProcNames_; }

  void add_specificProc(const Symbol *proc) { specificProcs_.push_back(proc); }
  void add_specificProcName(const SourceName &name, bool isModuleProc) {
    specificProcNames_.emplace_back(&name, isModuleProc);
  }
  void ClearSpecificProcNames() { specificProcNames_.clear(); }

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
  listType specificProcs_;
  // specific procs referenced by name and whether it's a module proc
  procNamesType specificProcNames_;
  // a specific procedure with the same name as this generic, if any
  Symbol *specific_{nullptr};
  // a derived type with the same name as this generic, if any
  Symbol *derivedType_{nullptr};
};

class UnknownDetails {};

using Details = std::variant<UnknownDetails, MainProgramDetails, ModuleDetails,
    SubprogramDetails, SubprogramNameDetails, EntityDetails,
    ObjectEntityDetails, ProcEntityDetails, DerivedTypeDetails, UseDetails,
    UseErrorDetails, GenericDetails, ProcBindingDetails, GenericBindingDetails,
    FinalProcDetails, TypeParamDetails>;
std::ostream &operator<<(std::ostream &, const Details &);
std::string DetailsToString(const Details &);

class Symbol {
public:
  ENUM_CLASS(Flag, Function, Subroutine, Implicit, ModFile);
  using Flags = common::EnumSet<Flag, Flag_enumSize>;

  const Scope &owner() const { return *owner_; }
  const SourceName &name() const { return occurrences_.front(); }
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

  const Details &details() const { return details_; }
  // Assign the details of the symbol from one of the variants.
  // Only allowed in certain cases.
  void set_details(const Details &);

  // Can the details of this symbol be replaced with the given details?
  bool CanReplaceDetails(const Details &details) const;

  const std::list<SourceName> &occurrences() const { return occurrences_; }
  void add_occurrence(const SourceName &);
  void remove_occurrence(const SourceName &);

  // Follow use-associations to get the ultimate entity.
  Symbol &GetUltimate();
  const Symbol &GetUltimate() const;

  const DeclTypeSpec *GetType() const;
  void SetType(const DeclTypeSpec &);

  bool isSubprogram() const;
  bool HasExplicitInterface() const;

  bool operator==(const Symbol &that) const { return this == &that; }
  bool operator!=(const Symbol &that) const { return this != &that; }

private:
  const Scope *owner_;
  std::list<SourceName> occurrences_;
  Attrs attrs_;
  Flags flags_;
  Scope *scope_{nullptr};
  Details details_;

  Symbol() {}  // only created in class Symbols
  const std::string GetDetailsName() const;
  friend std::ostream &operator<<(std::ostream &, const Symbol &);
  friend std::ostream &DumpForUnparse(std::ostream &, const Symbol &, bool);
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
      const Details &details) {
    Symbol &symbol = Get();
    symbol.owner_ = &owner;
    symbol.occurrences_.push_back(name);
    symbol.attrs_ = attrs;
    symbol.details_ = details;
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

}  // namespace Fortran::semantics
#endif  // FORTRAN_SEMANTICS_SYMBOL_H_
