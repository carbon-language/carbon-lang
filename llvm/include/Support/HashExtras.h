//===-- HashExtras.h - Useful functions for STL hash containers --*- C++ -*--=//
//
// This file contains some templates that are useful if you are working with the
// STL Hashed containers.
//
// No library is required when using these functinons.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_HASHEXTRAS_H
#define SUPPORT_HASHEXTRAS_H

#include <string>
#include "Support/hash_map"

// Cannot specialize hash template from outside of the std namespace.
namespace HASH_NAMESPACE {

template <> struct hash<std::string> {
  size_t operator()(std::string const &str) const {
    return hash<char const *>()(str.c_str());
  }
};

// Provide a hash function for arbitrary pointers...
template <class T> struct hash<T *> {
  inline size_t operator()(const T *Val) const { return (size_t)Val; }
};

}  // End namespace std

#endif
