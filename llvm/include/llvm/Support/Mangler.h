//===-- Mangler.h - Self-contained c/asm llvm name mangler -*- C++ -*- ----===//
//
// Unified name mangler for CWriter and assembly backends.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_MANGLER_H
#define LLVM_SUPPORT_MANGLER_H

class Value;
class Module;
#include <map>
#include <set>

class Mangler {
public:
  /// getValueName - Returns the mangled name of V, an LLVM Value,
  /// in the current module.
  ///
  std::string getValueName(const Value *V);

  Mangler(Module &M_);

  /// makeNameProper - We don't want identifier names with ., space, or
  /// - in them, so we mangle these characters into the strings "d_",
  /// "s_", and "D_", respectively. This is a very simple mangling that
  /// doesn't guarantee unique names for values. getValueName already
  /// does this for you, so there's no point calling it on the result
  /// from getValueName.
  /// 
  static std::string makeNameProper(const std::string &x);

private:
  /// This keeps track of which global values have had their names
  /// mangled in the current module.
  ///
  std::set<const Value *> MangledGlobals;

  Module &M;

  typedef std::map<const Value *, std::string> ValueMap;
  ValueMap Memo;

  unsigned int Count;
};

#endif // LLVM_SUPPORT_MANGLER_H
