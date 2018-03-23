#ifndef FORTRAN_SEMANTICS_SCOPE_H_
#define FORTRAN_SEMANTICS_SCOPE_H_

#include "../parser/idioms.h"
#include "attr.h"
#include "symbol.h"
#include <list>
#include <map>
#include <string>

namespace Fortran {
namespace semantics {

using Name = std::string;

class Scope {
public:
  // root of the scope tree; contains intrinsics:
  static const Scope systemScope;
  static Scope globalScope;  // contains program-units

  enum class Kind {
    System,
    Global,
    Module,
    MainProgram,
    Subprogram,
  };

  Scope(const Scope &parent, Kind kind) : parent_{parent}, kind_{kind} {}

  const Scope &parent() const {
    CHECK(kind_ != Kind::System);
    return parent_;
  }
  Kind kind() const { return kind_; }

  /// Make a scope nested in this one
  Scope &MakeScope(Kind kind);

  /// If there is a symbol with this name already in the scope, return it.
  /// Otherwise make a new one and return that.
  Symbol &GetOrMakeSymbol(const Name &name);

  /// Make a Symbol with unknown details.
  Symbol &MakeSymbol(const Name &name, const Attrs &attrs = Attrs::EMPTY);

  /// Make a Symbol with provided details.
  template<typename D> Symbol &MakeSymbol(const Name &name, D &&details) {
    const auto &result =
        symbols_.try_emplace(name, *this, name, Attrs::EMPTY, details);
    return result.first->second;
  }
  template<typename D>
  Symbol &MakeSymbol(const Name &name, const Attrs &attrs, D &&details) {
    const auto &result =
        symbols_.try_emplace(name, *this, name, attrs, details);
    return result.first->second;
  }

private:
  const Scope &parent_;
  const Kind kind_;
  std::list<Scope> children_;
  std::map<Name, Symbol> symbols_;

  friend std::ostream &operator<<(std::ostream &, const Scope &);
};

}  // namespace semantics
}  // namespace Fortran
#endif  // FORTRAN_SEMANTICS_SCOPE_H_
