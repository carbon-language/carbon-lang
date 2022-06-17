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
#include "toolchain/semantics/nodes/infix_operator.h"
#include "toolchain/semantics/semantics_ir.h"

namespace Carbon::Testing {

// A singleton SemanticsIR instance, used by the test helpers.
//
// This provides a singleton so that calls like PrintTo(Semantics::Declaration)
// have a SemanticsIR to refer back to; PrintTo must be static.
class SemanticsIRForTest {
 public:
  template <typename NodeT>
  static auto GetDeclaration(Semantics::Declaration decl)
      -> llvm::Optional<NodeT> {
    if (decl.kind() != NodeT::MetaNodeKind) {
      return llvm::None;
    }
    return semantics().declarations_.Get<NodeT>(decl);
  }

  template <typename NodeT>
  static auto GetExpression(Semantics::Expression expr)
      -> llvm::Optional<NodeT> {
    if (expr.kind() != NodeT::MetaNodeKind) {
      return llvm::None;
    }
    return semantics().expressions_.Get<NodeT>(expr);
  }

  template <typename NodeT>
  static auto GetStatement(Semantics::Statement expr) -> llvm::Optional<NodeT> {
    if (expr.kind() != NodeT::MetaNodeKind) {
      return llvm::None;
    }
    return semantics().statements_.Get<NodeT>(expr);
  }

  static auto GetNodeText(ParseTree::Node node) -> llvm::StringRef {
    return semantics().parse_tree_->GetNodeText(node);
  }

  template <typename PrintableT>
  static void PrintTo(const PrintableT& printable, std::ostream* out) {
    llvm::raw_os_ostream wrapped_out(*out);
    semantics().Print(wrapped_out, printable);
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

// Meta node printers.
inline void PrintTo(const Declaration& node, std::ostream* out) {
  Carbon::Testing::SemanticsIRForTest::PrintTo(node, out);
}
inline void PrintTo(const Expression& node, std::ostream* out) {
  Carbon::Testing::SemanticsIRForTest::PrintTo(node, out);
}
inline void PrintTo(const Statement& node, std::ostream* out) {
  Carbon::Testing::SemanticsIRForTest::PrintTo(node, out);
}

// Other node printers.
inline void PrintTo(const DeclaredName& node, std::ostream* out) {
  Carbon::Testing::SemanticsIRForTest::PrintTo(node, out);
}
inline void PrintTo(const ExpressionStatement& node, std::ostream* out) {
  Carbon::Testing::SemanticsIRForTest::PrintTo(node, out);
}
inline void PrintTo(const Function& node, std::ostream* out) {
  Carbon::Testing::SemanticsIRForTest::PrintTo(node, out);
}
inline void PrintTo(const InfixOperator& node, std::ostream* out) {
  Carbon::Testing::SemanticsIRForTest::PrintTo(node, out);
}
inline void PrintTo(const Literal& node, std::ostream* out) {
  Carbon::Testing::SemanticsIRForTest::PrintTo(node, out);
}
inline void PrintTo(const PatternBinding& node, std::ostream* out) {
  Carbon::Testing::SemanticsIRForTest::PrintTo(node, out);
}
inline void PrintTo(const Return& node, std::ostream* out) {
  Carbon::Testing::SemanticsIRForTest::PrintTo(node, out);
}
inline void PrintTo(const StatementBlock& node, std::ostream* out) {
  Carbon::Testing::SemanticsIRForTest::PrintTo(node, out);
}

}  // namespace Carbon::Semantics

namespace llvm {

// Prints a StringMapEntry for gmock.
inline void PrintTo(
    const llvm::StringMapEntry<Carbon::Semantics::Declaration>& entry,
    std::ostream* out) {
  *out << "StringMapEntry(" << entry.getKey() << ", ";
  Carbon::Testing::SemanticsIRForTest::PrintTo(entry.getValue(), out);
  *out << ")";
}

// Prints a StringMapEntry for gmock.
inline void PrintTo(
    const llvm::StringMapEntry<Carbon::Semantics::Statement>& entry,
    std::ostream* out) {
  *out << "StringMapEntry(" << entry.getKey() << ", ";
  Carbon::Testing::SemanticsIRForTest::PrintTo(entry.getValue(), out);
  *out << ")";
}

}  // namespace llvm

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_FOR_TEST_H_
