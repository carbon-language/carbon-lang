#ifndef FORTRAN_ATTR_H_
#define FORTRAN_ATTR_H_

#include <iostream>
#include <set>
#include <string>

#include "idioms.h"

namespace Fortran {

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
  OPTIONAL,
  PARAMETER,
  POINTER,
  PRIVATE,
  PROTECTED,
  PUBLIC,
  SAVE,
  TARGET,
  VALUE,
  VOLATILE,
};

using Attrs = std::set<Attr>;

std::ostream &operator<<(std::ostream &o, Attr attr);
std::ostream &operator<<(std::ostream &o, const Attrs &attrs);

// Report internal error if attrs is not a subset of allowed.
void checkAttrs(std::string className, Attrs attrs, Attrs allowed);

}  // namespace Fortran

#endif
