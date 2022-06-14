//===- TestMatchers.cpp - Pass to test matchers ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// This is a test pass for verifying matchers.
struct TestMatchers
    : public PassWrapper<TestMatchers, InterfacePass<FunctionOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestMatchers)

  void runOnOperation() override;
  StringRef getArgument() const final { return "test-matchers"; }
  StringRef getDescription() const final {
    return "Test C++ pattern matchers.";
  }
};
} // namespace

// This could be done better but is not worth the variadic template trouble.
template <typename Matcher>
static unsigned countMatches(FunctionOpInterface f, Matcher &matcher) {
  unsigned count = 0;
  f.walk([&count, &matcher](Operation *op) {
    if (matcher.match(op))
      ++count;
  });
  return count;
}

using mlir::matchers::m_Any;
using mlir::matchers::m_Val;
static void test1(FunctionOpInterface f) {
  assert(f.getNumArguments() == 3 && "matcher test funcs must have 3 args");

  auto a = m_Val(f.getArgument(0));
  auto b = m_Val(f.getArgument(1));
  auto c = m_Val(f.getArgument(2));

  auto p0 = m_Op<arith::AddFOp>(); // using 0-arity matcher
  llvm::outs() << "Pattern add(*) matched " << countMatches(f, p0)
               << " times\n";

  auto p1 = m_Op<arith::MulFOp>(); // using 0-arity matcher
  llvm::outs() << "Pattern mul(*) matched " << countMatches(f, p1)
               << " times\n";

  auto p2 = m_Op<arith::AddFOp>(m_Op<arith::AddFOp>(), m_Any());
  llvm::outs() << "Pattern add(add(*), *) matched " << countMatches(f, p2)
               << " times\n";

  auto p3 = m_Op<arith::AddFOp>(m_Any(), m_Op<arith::AddFOp>());
  llvm::outs() << "Pattern add(*, add(*)) matched " << countMatches(f, p3)
               << " times\n";

  auto p4 = m_Op<arith::MulFOp>(m_Op<arith::AddFOp>(), m_Any());
  llvm::outs() << "Pattern mul(add(*), *) matched " << countMatches(f, p4)
               << " times\n";

  auto p5 = m_Op<arith::MulFOp>(m_Any(), m_Op<arith::AddFOp>());
  llvm::outs() << "Pattern mul(*, add(*)) matched " << countMatches(f, p5)
               << " times\n";

  auto p6 = m_Op<arith::MulFOp>(m_Op<arith::MulFOp>(), m_Any());
  llvm::outs() << "Pattern mul(mul(*), *) matched " << countMatches(f, p6)
               << " times\n";

  auto p7 = m_Op<arith::MulFOp>(m_Op<arith::MulFOp>(), m_Op<arith::MulFOp>());
  llvm::outs() << "Pattern mul(mul(*), mul(*)) matched " << countMatches(f, p7)
               << " times\n";

  auto mulOfMulmul =
      m_Op<arith::MulFOp>(m_Op<arith::MulFOp>(), m_Op<arith::MulFOp>());
  auto p8 = m_Op<arith::MulFOp>(mulOfMulmul, mulOfMulmul);
  llvm::outs()
      << "Pattern mul(mul(mul(*), mul(*)), mul(mul(*), mul(*))) matched "
      << countMatches(f, p8) << " times\n";

  // clang-format off
  auto mulOfMuladd = m_Op<arith::MulFOp>(m_Op<arith::MulFOp>(), m_Op<arith::AddFOp>());
  auto mulOfAnyadd = m_Op<arith::MulFOp>(m_Any(), m_Op<arith::AddFOp>());
  auto p9 = m_Op<arith::MulFOp>(m_Op<arith::MulFOp>(
                     mulOfMuladd, m_Op<arith::MulFOp>()),
                   m_Op<arith::MulFOp>(mulOfAnyadd, mulOfAnyadd));
  // clang-format on
  llvm::outs() << "Pattern mul(mul(mul(mul(*), add(*)), mul(*)), mul(mul(*, "
                  "add(*)), mul(*, add(*)))) matched "
               << countMatches(f, p9) << " times\n";

  auto p10 = m_Op<arith::AddFOp>(a, b);
  llvm::outs() << "Pattern add(a, b) matched " << countMatches(f, p10)
               << " times\n";

  auto p11 = m_Op<arith::AddFOp>(a, c);
  llvm::outs() << "Pattern add(a, c) matched " << countMatches(f, p11)
               << " times\n";

  auto p12 = m_Op<arith::AddFOp>(b, a);
  llvm::outs() << "Pattern add(b, a) matched " << countMatches(f, p12)
               << " times\n";

  auto p13 = m_Op<arith::AddFOp>(c, a);
  llvm::outs() << "Pattern add(c, a) matched " << countMatches(f, p13)
               << " times\n";

  auto p14 = m_Op<arith::MulFOp>(a, m_Op<arith::AddFOp>(c, b));
  llvm::outs() << "Pattern mul(a, add(c, b)) matched " << countMatches(f, p14)
               << " times\n";

  auto p15 = m_Op<arith::MulFOp>(a, m_Op<arith::AddFOp>(b, c));
  llvm::outs() << "Pattern mul(a, add(b, c)) matched " << countMatches(f, p15)
               << " times\n";

  auto mulOfAany = m_Op<arith::MulFOp>(a, m_Any());
  auto p16 = m_Op<arith::MulFOp>(mulOfAany, m_Op<arith::AddFOp>(a, c));
  llvm::outs() << "Pattern mul(mul(a, *), add(a, c)) matched "
               << countMatches(f, p16) << " times\n";

  auto p17 = m_Op<arith::MulFOp>(mulOfAany, m_Op<arith::AddFOp>(c, b));
  llvm::outs() << "Pattern mul(mul(a, *), add(c, b)) matched "
               << countMatches(f, p17) << " times\n";
}

void test2(FunctionOpInterface f) {
  auto a = m_Val(f.getArgument(0));
  FloatAttr floatAttr;
  auto p =
      m_Op<arith::MulFOp>(a, m_Op<arith::AddFOp>(a, m_Constant(&floatAttr)));
  auto p1 = m_Op<arith::MulFOp>(a, m_Op<arith::AddFOp>(a, m_Constant()));
  // Last operation that is not the terminator.
  Operation *lastOp = f.getBody().front().back().getPrevNode();
  if (p.match(lastOp))
    llvm::outs()
        << "Pattern add(add(a, constant), a) matched and bound constant to: "
        << floatAttr.getValueAsDouble() << "\n";
  if (p1.match(lastOp))
    llvm::outs() << "Pattern add(add(a, constant), a) matched\n";
}

void TestMatchers::runOnOperation() {
  auto f = getOperation();
  llvm::outs() << f.getName() << "\n";
  if (f.getName() == "test1")
    test1(f);
  if (f.getName() == "test2")
    test2(f);
}

namespace mlir {
void registerTestMatchers() { PassRegistration<TestMatchers>(); }
} // namespace mlir
