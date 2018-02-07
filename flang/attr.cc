#include "attr.h"

#include <sstream>
#include <string>

namespace Fortran {

std::ostream &operator<<(std::ostream &o, Attr attr) {
  switch (attr) {
  case Attr::ABSTRACT: return o << "ABSTRACT";
  case Attr::ALLOCATABLE: return o << "ALLOCATABLE";
  case Attr::ASYNCHRONOUS: return o << "ASYNCHRONOUS";
  case Attr::BIND_C: return o << "BIND(C)";
  case Attr::CONTIGUOUS: return o << "CONTIGUOUS";
  case Attr::EXTERNAL: return o << "EXTERNAL";
  case Attr::INTENT_IN: return o << "INTENT_IN";
  case Attr::INTENT_OUT: return o << "INTENT_OUT";
  case Attr::INTRINSIC: return o << "INTRINSIC";
  case Attr::OPTIONAL: return o << "OPTIONAL";
  case Attr::PARAMETER: return o << "PARAMETER";
  case Attr::POINTER: return o << "POINTER";
  case Attr::PRIVATE: return o << "PRIVATE";
  case Attr::PROTECTED: return o << "PROTECTED";
  case Attr::PUBLIC: return o << "PUBLIC";
  case Attr::SAVE: return o << "SAVE";
  case Attr::TARGET: return o << "TARGET";
  case Attr::VALUE: return o << "VALUE";
  case Attr::VOLATILE: return o << "VOLATILE";
  default: CRASH_NO_CASE;
  }
}

std::ostream &operator<<(std::ostream &o, const Attrs &attrs) {
  int n = 0;
  for (auto attr : attrs) {
    if (n++) { o << ", "; }
    o << attr;
  }
  return o;
}

void checkAttrs(std::string className, Attrs attrs, Attrs allowed) {
  for (auto attr : attrs) {
    if (allowed.find(attr) == allowed.end()) {
      std::stringstream temp;
      temp << attr;
      parser::die("invalid attribute '%s' for class %s", temp.str().c_str(),
          className.c_str());
    }
  }
}

}  // namespace Fortran
