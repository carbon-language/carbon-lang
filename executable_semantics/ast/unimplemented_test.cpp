// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/unimplemented.h"

#include "executable_semantics/ast/expression.h"
#include "gtest/gtest.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {
namespace {

struct DummyPrintable {
  void Print(llvm::raw_ostream& out) const { out << output; }

  std::string output;
};

TEST(UnimplementedTest, UnimplementedTest) {
  DummyPrintable dummy = {"dummy1"};
  Unimplemented<Expression> unimplemented("Foo", SourceLocation("", 0), &dummy,
                                          DummyPrintable{"dummy2"}, 42);

  EXPECT_EQ(unimplemented.kind(), Expression::Kind::Unimplemented);

  std::string output;
  llvm::raw_string_ostream stream(output);
  stream << unimplemented;

  EXPECT_EQ(output, "UnimplementedFoo(dummy1, dummy2, 42)");
}

}  // namespace
}  // namespace Carbon
