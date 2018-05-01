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
private:
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
  const Symbol &result() const { CHECK(isFunction()); return **result_; }
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

class UnknownDetails {};

class Symbol {
public:
  // TODO: more kinds of details
  using Details = std::variant<UnknownDetails, MainProgramDetails,
      ModuleDetails, SubprogramDetails, EntityDetails>;

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
  // Only allowed if unknown.
  void set_details(Details &&details) {
    CHECK(has<UnknownDetails>());
    details_.swap(details);
  }

  const std::list<SourceName> &occurrences() const {
    return occurrences_;
  }
  void add_occurrence(const SourceName &name) {
    occurrences_.push_back(name);
  }

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
