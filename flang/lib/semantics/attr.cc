#include "../parser/idioms.h"
#include "attr.h"
#include <stddef.h>

namespace Fortran {
namespace semantics {

void Attrs::CheckValid(const Attrs &allowed) const {
  if (!allowed.HasAll(*this)) {
    parser::die("invalid attribute");
  }
}

std::ostream &operator<<(std::ostream &o, Attr attr) {
  return o << EnumToString(attr);
}

std::ostream &operator<<(std::ostream &o, const Attrs &attrs) {
  std::size_t n{attrs.count()};
  std::size_t seen{0};
  for (std::size_t j{0}; seen < n; ++j) {
    Attr attr{static_cast<Attr>(j)};
    if (attrs.test(attr)) {
      if (seen > 0) {
        o << ", ";
      }
      o << attr;
      ++seen;
    }
  }
  return o;
}
}  // namespace semantics
}  // namespace Fortran
