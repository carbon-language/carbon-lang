//===-- SpecialCaseList.h - blacklist for sanitizers ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===//
//
// This is a utility class for instrumentation passes (like AddressSanitizer
// or ThreadSanitizer) to avoid instrumenting some functions or global
// variables based on a user-supplied blacklist.
//
// The blacklist disables instrumentation of various functions and global
// variables.  Each line contains a prefix, followed by a wild card expression.
// Empty lines and lines starting with "#" are ignored.
// ---
// # Blacklisted items:
// fun:*_ZN4base6subtle*
// global:*global_with_bad_access_or_initialization*
// global-init:*global_with_initialization_issues*
// global-init-type:*Namespace::ClassName*
// src:file_with_tricky_code.cc
// global-init-src:ignore-global-initializers-issues.cc
// ---
// Note that the wild card is in fact an llvm::Regex, but * is automatically
// replaced with .*
// This is similar to the "ignore" feature of ThreadSanitizer.
// http://code.google.com/p/data-race-test/wiki/ThreadSanitizerIgnores
//
//===----------------------------------------------------------------------===//
//

#include "llvm/ADT/StringMap.h"

namespace llvm {
class Function;
class GlobalVariable;
class MemoryBuffer;
class Module;
class Regex;
class StringRef;

class SpecialCaseList {
 public:
  SpecialCaseList(const StringRef Path);
  SpecialCaseList(const MemoryBuffer *MB);

  // Returns whether either this function or it's source file are blacklisted.
  bool isIn(const Function &F) const;
  // Returns whether either this global or it's source file are blacklisted.
  bool isIn(const GlobalVariable &G) const;
  // Returns whether this module is blacklisted by filename.
  bool isIn(const Module &M) const;
  // Returns whether a global should be excluded from initialization checking.
  bool isInInit(const GlobalVariable &G) const;
 private:
  StringMap<Regex*> Entries;

  void init(const MemoryBuffer *MB);
  bool inSection(const StringRef Section, const StringRef Query) const;
};

}  // namespace llvm
