#include "../parser/idioms.h"
#include "attr.h"
#include <stddef.h>

namespace Fortran {
namespace semantics {

constexpr static size_t toInt(Attr attr) { return static_cast<size_t>(attr); }

const Attrs Attrs::EMPTY;

Attrs::Attrs(std::initializer_list<Attr> attrs) {
  bits_ = 0;
  for (auto attr : attrs) {
    Set(attr);
  }
}

Attrs &Attrs::Set(Attr attr) {
  bits_ |= 1u << toInt(attr);
  return *this;
}
Attrs &Attrs::Add(const Attrs &attrs) {
  bits_ |= attrs.bits_;
  return *this;
}

bool Attrs::Has(Attr attr) const { return (bits_ & (1u << toInt(attr))) != 0; }

bool Attrs::HasAny(const Attrs &attrs) const {
  return (bits_ & attrs.bits_) != 0;
}

bool Attrs::HasAll(const Attrs &attrs) const {
  return (bits_ & attrs.bits_) == attrs.bits_;
}

void Attrs::CheckValid(const Attrs &allowed) const {
  if (!allowed.HasAll(*this)) {
    parser::die("invalid attribute");
  }
}

std::ostream &operator<<(std::ostream &o, Attr attr) {
  return o << EnumToString(attr);
}

std::ostream &operator<<(std::ostream &o, const Attrs &attrs) {
  int i = 0, n = 0;
  for (std::uint64_t bits = attrs.bits_; bits != 0; bits >>= 1, ++i) {
    if (bits & 1) {
      if (n++) {
        o << ", ";
      }
      o << EnumToString(static_cast<Attr>(i));
    }
  }
  return o;
}

}  // namespace semantics
}  // namespace Fortran
