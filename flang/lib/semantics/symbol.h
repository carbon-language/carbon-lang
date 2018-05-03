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
#include <functional>
#include <memory>

namespace Fortran::semantics {

/// A SourceName is a name in the cooked character stream,
/// i.e. a range of lower-case characters with provenance.
using SourceName = parser::CharBlock;

/// A Symbol consists of common information (name, owner, and attributes)
/// and details information specific to the kind of symbol, represented by the
/// *Details classes.

class Scope;
class Symbol;

class ModuleDetails {
public:
  const Scope *scope() const { return scope_; }
  void set_scope(const Scope *scope) {
    CHECK(!scope_);
    scope_ = scope;
  }

private:
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
  friend std::ostream &operator<<(std::ostream &, const SubprogramDetails &);
};

class EntityDetails {
public:
  EntityDetails(bool isDummy = false) : isDummy_{isDummy} {}
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
  friend std::ostream &operator<<(std::ostream &, const EntityDetails &);
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
  UseErrorDetails(const SourceName &location, const Scope &module) {
    add_occurrence(location, module);
  }

  UseErrorDetails &add_occurrence(
      const SourceName &location, const Scope &module) {
    occurrences_.push_back(std::make_pair(&location, &module));
    return *this;
  }

  using listType = std::list<std::pair<const SourceName *, const Scope *>>;
  const listType occurrences() const { return occurrences_; };

private:
  listType occurrences_;
};

class UnknownDetails {};

using Details = std::variant<UnknownDetails, MainProgramDetails, ModuleDetails,
    SubprogramDetails, EntityDetails, UseDetails, UseErrorDetails>;
std::ostream &operator<<(std::ostream &, const Details &);

class Symbol {
public:
  Symbol(const Scope &owner, const SourceName &name, const Attrs &attrs,
      Details &&details)
    : owner_{owner}, attrs_{attrs}, details_{std::move(details)} {
    add_occurrence(name);
  }
  const Scope &owner() const { return owner_; }
  const SourceName &name() const { return occurrences_.front(); }
  Attrs &attrs() { return attrs_; }
  const Attrs &attrs() const { return attrs_; }

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
  template<typename D> D &details() {
    return const_cast<D &>(static_cast<const Symbol *>(this)->details<D>());
  }
  template<typename D> const D &details() const {
    if (const auto p = detailsIf<D>()) {
      return *p;
    } else {
      Fortran::parser::die("unexpected %s details at %s(%d)",
          GetDetailsName().c_str(), __FILE__, __LINE__);
    }
  }

  // Assign the details of the symbol from one of the variants.
  // Only allowed in certain cases.
  void set_details(Details &&details) {
    if (has<UnknownDetails>()) {
      // can always replace UnknownDetails
    } else if (has<UseDetails>() &&
        std::holds_alternative<UseErrorDetails>(details)) {
      // can replace UseDetails with UseErrorDetails
    } else {
      CHECK(!"can't replace details");
    }
    details_.swap(details);
  }

  const std::list<SourceName> &occurrences() const { return occurrences_; }
  void add_occurrence(const SourceName &name) { occurrences_.push_back(name); }

  // Follow use-associations to get the ultimate entity.
  const Symbol &GetUltimate() const;

  bool operator==(const Symbol &that) const { return this == &that; }
  bool operator!=(const Symbol &that) const { return this != &that; }

private:
  const Scope &owner_;
  std::list<SourceName> occurrences_;
  Attrs attrs_;
  Details details_;

  const std::string GetDetailsName() const;
  friend std::ostream &operator<<(std::ostream &, const Symbol &);
};

}  // namespace Fortran::semantics
#endif  // FORTRAN_SEMANTICS_SYMBOL_H_
