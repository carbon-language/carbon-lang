// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MIGRATE_CPP_CPP_REFACTORING_FN_INSERTER_H_
#define MIGRATE_CPP_CPP_REFACTORING_FN_INSERTER_H_

#include "migrate_cpp/cpp_refactoring/matcher.h"

namespace Carbon {

// Inserts `fn` for functions and methods.
class FnInserter : public Matcher {
 public:
  explicit FnInserter(std::map<std::string, Replacements>& in_replacements,
                      MatchFinder* finder);

  void run(const MatchFinder::MatchResult& result) override;

 private:
  static constexpr char Label[] = "FnInserter";
};

}  // namespace Carbon

#endif  // MIGRATE_CPP_CPP_REFACTORING_FN_INSERTER_H_
