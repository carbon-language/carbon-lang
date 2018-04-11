#ifndef FORTRAN_SEMANTICS_SYMBOL_H_
#define FORTRAN_SEMANTICS_SYMBOL_H_

#include "type.h"
#include <functional>
#include <memory>

namespace Fortran::semantics {

/// A SourceName is a name in the cooked character stream,
/// i.e. a range of characters with provenance.
using SourceName = parser::CharBlock;

/// A Symbol consists of common information (name, owner, and attributes)
/// and details information specific to the kind of symbol, represented by the
/// *Details classes.

class Scope;

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
  // Subroutine:
  SubprogramDetails() {}
  // Function:
  SubprogramDetails(const SourceName &resultName) : resultName_{resultName} {}

  bool isFunction() const { return resultName_.has_value(); }
  const std::list<SourceName> &dummyNames() const { return dummyNames_; }
  const std::optional<SourceName> &resultName() const { return resultName_; }
  void AddDummyName(const SourceName &name) { dummyNames_.push_back(name); }

private:
  std::list<SourceName> dummyNames_;
  std::optional<SourceName> resultName_;
  friend std::ostream &operator<<(std::ostream &, const SubprogramDetails &);
};

class EntityDetails {
public:
  EntityDetails(bool isDummy = false) : isDummy_{isDummy} {}
  const std::optional<DeclTypeSpec> &type() const { return type_; }
  void set_type(const DeclTypeSpec &type) { type_ = type; };
  bool isDummy() const { return isDummy_; }

private:
  bool isDummy_;
  std::optional<DeclTypeSpec> type_;
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
    : owner_{owner}, name_{name}, attrs_{attrs},
      details_{std::move(details)} {}
  const Scope &owner() const { return owner_; }
  const SourceName &name() { return name_; }
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
    auto p = detailsIf<D>();
    CHECK(p && "unexpected type");
    return *p;
  }
  template<typename D> const D &details() const {
    const auto p = detailsIf<D>();
    CHECK(p && "unexpected type");
    return *p;
  }

  // Assign the details of the symbol from one of the variants.
  // Only allowed if unknown.
  void set_details(Details &&details) {
    CHECK(has<UnknownDetails>());
    details_.swap(details);
  }

private:
  const Scope &owner_;
  const SourceName name_;
  Attrs attrs_;
  Details details_;
  friend std::ostream &operator<<(std::ostream &, const Symbol &);
};

}  // namespace Fortran::semantics
#endif  // FORTRAN_SEMANTICS_SYMBOL_H_
