#ifndef FORTRAN_ATTR_H_
#define FORTRAN_ATTR_H_

#include <cinttypes>
#include <iostream>
#include <string>

namespace Fortran {
namespace semantics {

// All available attributes.
enum class Attr {
  ABSTRACT,
  ALLOCATABLE,
  ASYNCHRONOUS,
  BIND_C,
  CONTIGUOUS,
  EXTERNAL,
  INTENT_IN,
  INTENT_OUT,
  INTRINSIC,
  NOPASS,
  OPTIONAL,
  PARAMETER,
  PASS,
  POINTER,
  PRIVATE,
  PROTECTED,
  PUBLIC,
  SAVE,
  TARGET,
  VALUE,
  VOLATILE,
};

// Set of attributes
class Attrs {
public:
  Attrs() : bits_{0} {}
  Attrs(std::initializer_list<Attr> attrs);
  bool empty() const { return bits_ == 0; }
  Attrs &set(Attr attr);
  Attrs &add(const Attrs &attrs);
  bool has(Attr attr) const;
  bool hasAny(const Attrs &attrs) const;
  bool hasAll(const Attrs &attrs) const;
  // Internal error if any of these attributes are not in allowed.
  void checkValid(const Attrs &allowed) const;

private:
  std::uint64_t bits_;
  friend std::ostream &operator<<(std::ostream &, const Attrs &);
};

std::ostream &operator<<(std::ostream &o, Attr attr);
std::ostream &operator<<(std::ostream &o, const Attrs &attrs);

}  // namespace semantics
}  // namespace Fortran

#endif
