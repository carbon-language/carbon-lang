// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_FUNCTION_DEFINITION_H_
#define EXECUTABLE_SEMANTICS_AST_FUNCTION_DEFINITION_H_

#include "common/ostream.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/pattern.h"
#include "executable_semantics/ast/source_location.h"
#include "executable_semantics/ast/statement.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

// TODO: expand the kinds of things that can be deduced parameters.
//   For now, only generic parameters are supported.
struct GenericBinding {
  std::string name;
  Nonnull<const Expression*> type;
};

class FunctionDefinition {
 public:
  FunctionDefinition(SourceLocation source_loc, std::string name,
                     std::vector<GenericBinding> deduced_params,
                     Nonnull<TuplePattern*> param_pattern,
                     Nonnull<Pattern*> return_type, bool is_omitted_return_type,
                     std::optional<Nonnull<Statement*>> body)
      : source_loc_(source_loc),
        name_(std::move(name)),
        deduced_parameters_(std::move(deduced_params)),
        param_pattern_(param_pattern),
        return_type_(return_type),
        is_omitted_return_type_(is_omitted_return_type),
        body_(body) {}

  void Print(llvm::raw_ostream& out) const { PrintDepth(-1, out); }
  void PrintDepth(int depth, llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

  auto source_loc() const -> SourceLocation { return source_loc_; }
  auto name() const -> const std::string& { return name_; }
  auto deduced_parameters() const -> llvm::ArrayRef<GenericBinding> {
    return deduced_parameters_;
  }
  auto param_pattern() const -> const TuplePattern& { return *param_pattern_; }
  auto param_pattern() -> TuplePattern& { return *param_pattern_; }
  auto return_type() const -> const Pattern& { return *return_type_; }
  auto return_type() -> Pattern& { return *return_type_; }
  auto is_omitted_return_type() const -> bool {
    return is_omitted_return_type_;
  }
  auto body() const -> std::optional<Nonnull<const Statement*>> {
    return body_;
  }
  auto body() -> std::optional<Nonnull<Statement*>> { return body_; }

 private:
  SourceLocation source_loc_;
  std::string name_;
  std::vector<GenericBinding> deduced_parameters_;
  Nonnull<TuplePattern*> param_pattern_;
  Nonnull<Pattern*> return_type_;
  bool is_omitted_return_type_;
  std::optional<Nonnull<Statement*>> body_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_FUNCTION_DEFINITION_H_
