#include "../parser/idioms.h"
#include "attr.h"
#include <stddef.h>

namespace Fortran {
namespace semantics {

constexpr static size_t toInt(Attr attr) { return static_cast<size_t>(attr); }

static const char *attrToString[] = {
    [toInt(Attr::ABSTRACT)] = "ABSTRACT",
    [toInt(Attr::ALLOCATABLE)] = "ALLOCATABLE",
    [toInt(Attr::ASYNCHRONOUS)] = "ASYNCHRONOUS",
    [toInt(Attr::BIND_C)] = "BIND(C)",
    [toInt(Attr::CONTIGUOUS)] = "CONTIGUOUS",
    [toInt(Attr::EXTERNAL)] = "EXTERNAL",
    [toInt(Attr::INTENT_IN)] = "INTENT_IN",
    [toInt(Attr::INTENT_OUT)] = "INTENT_OUT",
    [toInt(Attr::INTRINSIC)] = "INTRINSIC",
    [toInt(Attr::NOPASS)] = "NOPASS",
    [toInt(Attr::OPTIONAL)] = "OPTIONAL",
    [toInt(Attr::PARAMETER)] = "PARAMETER",
    [toInt(Attr::PASS)] = "PASS",
    [toInt(Attr::POINTER)] = "POINTER",
    [toInt(Attr::PRIVATE)] = "PRIVATE",
    [toInt(Attr::PROTECTED)] = "PROTECTED",
    [toInt(Attr::PUBLIC)] = "PUBLIC",
    [toInt(Attr::SAVE)] = "SAVE",
    [toInt(Attr::TARGET)] = "TARGET",
    [toInt(Attr::VALUE)] = "VALUE",
    [toInt(Attr::VOLATILE)] = "VOLATILE",
};

Attrs::Attrs(std::initializer_list<Attr> attrs) {
  bits_ = 0;
  for (auto attr : attrs) {
    set(attr);
  }
}

Attrs &Attrs::set(Attr attr) {
  bits_ |= 1u << toInt(attr);
  return *this;
}
Attrs &Attrs::add(const Attrs &attrs) {
  bits_ |= attrs.bits_;
  return *this;
}

bool Attrs::has(Attr attr) const { return (bits_ & (1u << toInt(attr))) != 0; }

bool Attrs::hasAny(const Attrs &attrs) const {
  return (bits_ & attrs.bits_) != 0;
}

bool Attrs::hasAll(const Attrs &attrs) const {
  return (bits_ & attrs.bits_) == attrs.bits_;
}

void Attrs::checkValid(const Attrs &allowed) const {
  if (!allowed.hasAll(*this)) {
    parser::die("invalid attribute");
  }
}

std::ostream &operator<<(std::ostream &o, Attr attr) {
  return o << attrToString[toInt(attr)];
}

std::ostream &operator<<(std::ostream &o, const Attrs &attrs) {
  int i = 0, n = 0;
  for (std::uint64_t bits = attrs.bits_; bits != 0; bits >>= 1, ++i) {
    if (bits & 1) {
      if (n++) {
        o << ", ";
      }
      o << attrToString[i];
    }
  }
  return o;
}

}  // namespace semantics
}  // namespace Fortran
