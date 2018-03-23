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
    [toInt(Attr::BIND_C)] = "BIND_C",
    [toInt(Attr::CONTIGUOUS)] = "CONTIGUOUS",
    [toInt(Attr::DEFERRED)] = "DEFERRED",
    [toInt(Attr::ELEMENTAL)] = "ELEMENTAL",
    [toInt(Attr::EXTERNAL)] = "EXTERNAL",
    [toInt(Attr::IMPURE)] = "IMPURE",
    [toInt(Attr::INTENT_IN)] = "INTENT_IN",
    [toInt(Attr::INTENT_OUT)] = "INTENT_OUT",
    [toInt(Attr::INTRINSIC)] = "INTRINSIC",
    [toInt(Attr::MODULE)] = "MODULE",
    [toInt(Attr::NON_OVERRIDABLE)] = "NON_OVERRIDABLE",
    [toInt(Attr::NON_RECURSIVE)] = "NON_RECURSIVE",
    [toInt(Attr::NOPASS)] = "NOPASS",
    [toInt(Attr::OPTIONAL)] = "OPTIONAL",
    [toInt(Attr::PARAMETER)] = "PARAMETER",
    [toInt(Attr::PASS)] = "PASS",
    [toInt(Attr::POINTER)] = "POINTER",
    [toInt(Attr::PRIVATE)] = "PRIVATE",
    [toInt(Attr::PROTECTED)] = "PROTECTED",
    [toInt(Attr::PUBLIC)] = "PUBLIC",
    [toInt(Attr::PURE)] = "PURE",
    [toInt(Attr::RECURSIVE)] = "RECURSIVE",
    [toInt(Attr::SAVE)] = "SAVE",
    [toInt(Attr::TARGET)] = "TARGET",
    [toInt(Attr::VALUE)] = "VALUE",
    [toInt(Attr::VOLATILE)] = "VOLATILE",
};

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
