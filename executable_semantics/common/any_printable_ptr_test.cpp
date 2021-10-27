// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/common/any_printable_ptr.h"

#include "gtest/gtest.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {
namespace {

struct DummyPrintable {
  void Print(llvm::raw_ostream& out) const { out << "dummy"; }
};

TEST(AnyPrintablePtrTest, PrintDummy) {
  std::string output;
  llvm::raw_string_ostream stream(output);

  DummyPrintable dummy;
  AnyPrintablePtr ptr(&dummy);
  stream << ptr;

  EXPECT_EQ(output, "dummy");
}

TEST(AnyPrintablePtrTest, PrintInt) {
  std::string output;
  llvm::raw_string_ostream stream(output);

  int i = 0;
  AnyPrintablePtr ptr(&i);
  stream << ptr;

  EXPECT_EQ(output, "0");
}

}  // namespace
}  // namespace Carbon
