//===-- HashExtras.h - Useful functions for STL hash containers --*- C++ -*--=//
//
// This file contains some templates that are useful if you are working with the
// STL Hashed containers.
//
// No library is required when using these functinons.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_HASHEXTRAS_H
#define LLVM_SUPPORT_HASHEXTRAS_H

#include <string>
#include <hash_map>

template <> struct hash<string> {
  size_t operator()(string const &str) const {
    return hash<char const *>()(str.c_str());
  }
};

#endif
