// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/syntax/antlr/parser_visitor.h"

#include "explorer/ast/ast.h"

namespace Carbon::Antlr {

auto ParserVisitor::visitInput(CarbonParser::InputContext* /*ctx*/)
    -> antlrcpp::Any {
  AST ast;
  return ast;
}

}  // namespace Carbon::Antlr
