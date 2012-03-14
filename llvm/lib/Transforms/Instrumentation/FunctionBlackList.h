//===-- FunctionBlackList.cpp - blacklist of functions ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===//
//
// This is a utility class for instrumentation passes (like AddressSanitizer
// or ThreadSanitizer) to avoid instrumenting some functions based on
// user-supplied blacklist.
//
//===----------------------------------------------------------------------===//
//

#include <string>

namespace llvm {
class Function;
class Regex;

// Blacklisted functions are not instrumented.
// The blacklist file contains one or more lines like this:
// ---
// fun:FunctionWildCard
// ---
// This is similar to the "ignore" feature of ThreadSanitizer.
// http://code.google.com/p/data-race-test/wiki/ThreadSanitizerIgnores
class FunctionBlackList {
 public:
  FunctionBlackList(const std::string &Path);
  bool isIn(const Function &F);
 private:
  Regex *Functions;
};

}  // namespace llvm
