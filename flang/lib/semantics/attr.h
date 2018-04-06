#ifndef FORTRAN_ATTR_H_
#define FORTRAN_ATTR_H_

#include "../parser/idioms.h"
#include "enum-set.h"
#include <cinttypes>
#include <iostream>
#include <string>

namespace Fortran {
namespace semantics {

// All available attributes.
ENUM_CLASS(Attr, ABSTRACT, ALLOCATABLE, ASYNCHRONOUS, BIND_C, CONTIGUOUS,
    DEFERRED, ELEMENTAL, EXTERNAL, IMPURE, INTENT_IN, INTENT_OUT, INTRINSIC,
    MODULE, NON_OVERRIDABLE, NON_RECURSIVE, NOPASS, OPTIONAL, PARAMETER, PASS,
    POINTER, PRIVATE, PROTECTED, PUBLIC, PURE, RECURSIVE, SAVE, TARGET, VALUE,
    VOLATILE)

// Set of attributes
class Attrs : public EnumSet<Attr, Attr_enumSize> {
private:
  using enumSetType = EnumSet<Attr, Attr_enumSize>;
public:
  using enumSetType::enumSetType;
  constexpr bool HasAny(const Attrs &x) const {
    return !(*this & x).none();
  }
  constexpr bool HasAll(const Attrs &x) const {
    return (~*this & x).none();
  }
  // Internal error if any of these attributes are not in allowed.
  void CheckValid(const Attrs &allowed) const;

private:
  friend std::ostream &operator<<(std::ostream &, const Attrs &);
};

std::ostream &operator<<(std::ostream &o, Attr attr);
std::ostream &operator<<(std::ostream &o, const Attrs &attrs);
}  // namespace semantics
}  // namespace Fortran
#endif
