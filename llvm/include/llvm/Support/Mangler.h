//===-- Mangler.h - Self-contained c/asm llvm name mangler ------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Unified name mangler for CWriter and assembly backends.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_MANGLER_H
#define LLVM_SUPPORT_MANGLER_H

namespace llvm {

class Value;
class Module;

} // End llvm namespace

#include <map>
#include <set>
#include <string>

namespace llvm {

class Mangler {
  /// This keeps track of which global values have had their names
  /// mangled in the current module.
  ///
  std::set<const Value *> MangledGlobals;

  Module &M;
  bool AddUnderscorePrefix;

  typedef std::map<const Value *, std::string> ValueMap;
  ValueMap Memo;

  unsigned Count;
public:

  // Mangler ctor - if AddUnderscorePrefix is true, then all public global
  // symbols will be prefixed with an underscore.
  Mangler(Module &M, bool AddUnderscorePrefix = false);

  /// getValueName - Returns the mangled name of V, an LLVM Value,
  /// in the current module.
  ///
  std::string getValueName(const Value *V);

  /// makeNameProper - We don't want identifier names with ., space, or
  /// - in them, so we mangle these characters into the strings "d_",
  /// "s_", and "D_", respectively. This is a very simple mangling that
  /// doesn't guarantee unique names for values. getValueName already
  /// does this for you, so there's no point calling it on the result
  /// from getValueName.
  /// 
  static std::string makeNameProper(const std::string &x);
};

} // End llvm namespace

#endif // LLVM_SUPPORT_MANGLER_H
