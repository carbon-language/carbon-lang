#include "scope.h"
#include "symbol.h"
#include <memory>

namespace Fortran::semantics {

const Scope Scope::systemScope{Scope::systemScope, Scope::Kind::System};
Scope Scope::globalScope{Scope::systemScope, Scope::Kind::Global};

Scope &Scope::MakeScope(Kind kind) {
  children_.emplace_back(*this, kind);
  return children_.back();
}

Symbol &Scope::GetOrMakeSymbol(const Name &name) {
  const auto it = symbols_.find(name);
  if (it != symbols_.end()) {
    return it->second;
  } else {
    return MakeSymbol(name);
  }
}

Symbol &Scope::MakeSymbol(const Name &name, Attrs attrs) {
  return MakeSymbol(name, attrs, UnknownDetails());
}

static const char *ToString(Scope::Kind kind) {
  switch (kind) {
  case Scope::Kind::System: return "System";
  case Scope::Kind::Global: return "Global";
  case Scope::Kind::Module: return "Module";
  case Scope::Kind::MainProgram: return "MainProgram";
  case Scope::Kind::Subprogram: return "Subprogram";
  default: CRASH_NO_CASE;
  }
}

std::ostream &operator<<(std::ostream &os, const Scope &scope) {
  os << ToString(scope.kind()) << " scope: " << scope.children_.size()
     << " children\n";
  for (const auto &sym : scope.symbols_) {
    os << "  " << sym.second << "\n";
  }
  return os;
}

}  // namespace Fortran::semantics
