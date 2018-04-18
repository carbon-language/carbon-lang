#ifndef FORTRAN_SEMANTICS_SCOPE_H_
#define FORTRAN_SEMANTICS_SCOPE_H_

#include "attr.h"
#include "symbol.h"
#include "../parser/idioms.h"
#include "../parser/parse-tree.h"
#include <list>
#include <map>
#include <string>

namespace Fortran::semantics {

class Scope {
  using mapType = std::map<SourceName, Symbol>;

public:
  // root of the scope tree; contains intrinsics:
  static const Scope systemScope;
  static Scope globalScope;  // contains program-units

  ENUM_CLASS(Kind, System, Global, Module, MainProgram, Subprogram)

  Scope(const Scope &parent, Kind kind, const Symbol *symbol)
    : parent_{parent}, kind_{kind}, symbol_{symbol} {}

  const Scope &parent() const {
    CHECK(kind_ != Kind::System);
    return parent_;
  }
  Kind kind() const { return kind_; }
  const Symbol *symbol() const { return symbol_; }

  /// Make a scope nested in this one
  Scope &MakeScope(Kind kind, const Symbol *symbol = nullptr);

  using size_type = mapType::size_type;
  using iterator = mapType::iterator;
  using const_iterator = mapType::const_iterator;

  iterator begin() { return symbols_.begin(); }
  iterator end() { return symbols_.end(); }
  const_iterator begin() const { return symbols_.begin(); }
  const_iterator end() const { return symbols_.end(); }
  const_iterator cbegin() const { return symbols_.cbegin(); }
  const_iterator cend() const { return symbols_.cend(); }

  iterator find(const SourceName &name) { return symbols_.find(name); }
  const_iterator find(const SourceName &name) const {
    return symbols_.find(name);
  }

  size_type erase(const SourceName &name) { return symbols_.erase(name); }

  /// Make a Symbol with unknown details.
  std::pair<iterator, bool> try_emplace(
      const SourceName &name, Attrs attrs = Attrs()) {
    return try_emplace(name, attrs, UnknownDetails());
  }
  /// Make a Symbol with provided details.
  template<typename D>
  std::pair<iterator, bool> try_emplace(const SourceName &name, D &&details) {
    return try_emplace(name, Attrs(), details);
  }
  /// Make a Symbol with attrs and details
  template<typename D>
  std::pair<iterator, bool> try_emplace(
      const SourceName &name, Attrs attrs, D &&details) {
    return symbols_.try_emplace(name, *this, name, attrs, details);
  }

  std::list<Scope> &children() { return children_; }
  const std::list<Scope> &children() const { return children_; }

private:
  const Scope &parent_;
  const Kind kind_;
  const Symbol *const symbol_;
  std::list<Scope> children_;
  mapType symbols_;

  friend std::ostream &operator<<(std::ostream &, const Scope &);
};

}  // namespace Fortran::semantics
#endif  // FORTRAN_SEMANTICS_SCOPE_H_
