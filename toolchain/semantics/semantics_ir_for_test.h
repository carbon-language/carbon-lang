// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_FOR_TEST_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_FOR_TEST_H_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <sstream>

#include "common/check.h"
#include "llvm/ADT/StringExtras.h"
//#include "toolchain/semantics/nodes/infix_operator.h"
#include "toolchain/semantics/semantics_ir.h"

namespace Carbon::Testing {

// A singleton SemanticsIR instance, used by the test helpers.
//
// This provides a singleton so that calls like PrintTo(Semantics::Declaration)
// have a SemanticsIR to refer back to; PrintTo must be static.
class SemanticsIRForTest {
 public:
  template <typename NodeT>
  static auto GetNode(Semantics::NodeRef node_ref) -> llvm::Optional<NodeT> {
    if (node_ref.kind() != NodeT::Kind) {
      return llvm::None;
    }
    return semantics().nodes_.Get<NodeT>(node_ref);
  }

  static auto semantics() -> const SemanticsIR& {
    CARBON_CHECK(g_semantics != llvm::None);
    return *g_semantics;
  }

  static void set_semantics(SemanticsIR semantics) {
    CARBON_CHECK(g_semantics == llvm::None)
        << "Call clear() before setting again.";
    g_semantics = std::move(semantics);
  }

  static void clear() { g_semantics = llvm::None; }

 private:
  static llvm::Optional<SemanticsIR> g_semantics;
};

}  // namespace Carbon::Testing

namespace Carbon::Semantics {

inline void PrintTo(const NodeRef& node_ref, std::ostream* out) {
  llvm::raw_os_ostream wrapped_out(*out);
  Carbon::Testing::SemanticsIRForTest::semantics().Print(wrapped_out, node_ref);
}

}  // namespace Carbon::Semantics

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_FOR_TEST_H_
