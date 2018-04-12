#include "symbol.h"
#include "scope.h"
#include "../parser/idioms.h"
#include <memory>

namespace Fortran::semantics {

void EntityDetails::set_type(const DeclTypeSpec &type) {
  CHECK(!type_);
  type_ = type;
}

void EntityDetails::set_shape(const ArraySpec &shape) {
  CHECK(shape_.empty());
  for (const auto &shapeSpec : shape) {
    shape_.push_back(shapeSpec);
  }
}

std::ostream &operator<<(std::ostream &os, const EntityDetails &x) {
  os << "Entity";
  if (x.type()) {
    os << " type: " << *x.type();
  }
  if (!x.shape().empty()) {
    os << " shape:";
    for (const auto &s : x.shape()) {
      os << ' ' << s;
    }
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const Symbol &sym) {
  os << sym.name_.ToString();
  if (!sym.attrs().empty()) {
    os << ", " << sym.attrs();
  }
  os << ": ";
  std::visit(
      parser::visitors{
          [&](const UnknownDetails &x) { os << " Unknown"; },
          [&](const MainProgramDetails &x) { os << " MainProgram"; },
          [&](const ModuleDetails &x) { os << " Module"; },
          [&](const SubprogramDetails &x) {
            os << " Subprogram (";
            int n = 0;
            for (const auto &dummy : x.dummyNames()) {
              if (n++ > 0) os << ", ";
              os << dummy.ToString();
            }
            os << ')';
            if (x.resultName()) {
              os << " result(" << x.resultName()->ToString() << ')';
            }
          },
          [&](const EntityDetails &x) { os << ' ' << x; },
      },
      sym.details_);
  return os;
}

}  // namespace Fortran::semantics
