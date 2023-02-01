// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_SYNTAX_ANTLR_PARSER_VISITOR_H_
#define CARBON_EXPLORER_SYNTAX_ANTLR_PARSER_VISITOR_H_

#include "antlr4-runtime.h"
#include "explorer/syntax/antlr/CarbonBaseVisitor.h"

namespace Carbon::Antlr {

class ParserVisitor : public CarbonBaseVisitor {
  auto visitInput(CarbonParser::InputContext* ctx) -> antlrcpp::Any override;
};

}  // namespace Carbon::Antlr

#endif  // CARBON_EXPLORER_SYNTAX_ANTLR_PARSER_VISITOR_H_
