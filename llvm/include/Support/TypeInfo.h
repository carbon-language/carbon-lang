//===- Support/TypeInfo.h - Support class for type_info objects -*- C++ -*-===//
//
// This class makes std::type_info objects behave like first class objects that
// can be put in maps and hashtables.  This code is based off of code in the
// Loki C++ library from the Modern C++ Design book.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TYPEINFO_H
#define SUPPORT_TYPEINFO_H

#include <typeinfo>

struct TypeInfo {
  TypeInfo() {                     // needed for containers
    struct Nil {};  // Anonymous class distinct from all others...
    Info = &typeid(Nil);
  }

  TypeInfo(const std::type_info &ti) : Info(&ti) { // non-explicit
  }

  // Access for the wrapped std::type_info
  const std::type_info &get() const {
    return *Info;
  }

  // Compatibility functions
  bool before(const TypeInfo &rhs) const {
    return Info->before(*rhs.Info);
  }
  const char *getClassName() const {
    return Info->name();
  }

private:
  const std::type_info *Info;
};
    
// Comparison operators
inline bool operator==(const TypeInfo &lhs, const TypeInfo &rhs) {
  return lhs.get() == rhs.get();
}

inline bool operator<(const TypeInfo &lhs, const TypeInfo &rhs) {
  return lhs.before(rhs);
}

inline bool operator!=(const TypeInfo &lhs, const TypeInfo &rhs) {
  return !(lhs == rhs);
}

inline bool operator>(const TypeInfo &lhs, const TypeInfo &rhs) {
  return rhs < lhs;
}
    
inline bool operator<=(const TypeInfo &lhs, const TypeInfo &rhs) {
  return !(lhs > rhs);
}
     
inline bool operator>=(const TypeInfo &lhs, const TypeInfo &rhs) {
  return !(lhs < rhs);
}

#endif
