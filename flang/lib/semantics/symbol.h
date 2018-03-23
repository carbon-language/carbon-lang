#ifndef FORTRAN_SEMANTICS_SYMBOL_H_
#define FORTRAN_SEMANTICS_SYMBOL_H_

#include "type.h"
#include <functional>
#include <memory>

namespace Fortran::semantics {

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
  SubprogramDetails(const std::list<Name> &dummyNames)
    : isFunction_{false}, dummyNames_{dummyNames} {}
  SubprogramDetails(
      const std::list<Name> &dummyNames, const std::optional<Name> &resultName)
    : isFunction_{true}, dummyNames_{dummyNames}, resultName_{resultName} {}

  bool isFunction() const { return isFunction_; }
  const std::list<Name> &dummyNames() const { return dummyNames_; }
  const std::optional<Name> &resultName() const { return resultName_; }

private:
  bool isFunction_;
  std::list<Name> dummyNames_;
  std::optional<Name> resultName_;
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

  Symbol(const Scope &owner, const Name &name, const Attrs &attrs,
      Details &&details)
    : owner_{owner}, name_{name}, attrs_{attrs}, details_{std::move(details)} {}
  const Scope &owner() const { return owner_; }
  const Name &name() const { return name_; }
  const Attrs &attrs() const { return attrs_; }

  // Does symbol have this type of details?
  template<typename D> bool has() const {
    return std::holds_alternative<D>(details_);
  }

  // Return a non-owning pointer to details if it is type D, else nullptr.
  template<typename D> D *detailsIf() { return std::get_if<D>(&details_); }

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
  void set_details(const Details &details) {
    CHECK(has<UnknownDetails>());
    details_ = details;
  };

private:
  const Scope &owner_;
  const Name name_;
  const Attrs attrs_;
  Details details_;
  friend std::ostream &operator<<(std::ostream &, const Symbol &);
};

}  // namespace Fortran::semantics
#endif  // FORTRAN_SEMANTICS_SYMBOL_H_
