// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/syntax/antlr/parse.h"

#include "antlr4-runtime.h"
#include "explorer/syntax/antlr/CarbonLexer.h"
#include "explorer/syntax/antlr/CarbonParser.h"
#include "explorer/syntax/antlr/parser_visitor.h"
#include "llvm/Support/FormatVariadic.h"

namespace Carbon::Antlr {

class ErrorListener : public antlr4::BaseErrorListener {
 public:
  explicit ErrorListener(std::string_view input_file_name)
      : input_file_name_(input_file_name) {}

  void syntaxError(antlr4::Recognizer* /*recognizer*/,
                   antlr4::Token* /*offending_symbol*/, size_t line,
                   size_t /*position_in_line*/, const std::string& message,
                   std::exception_ptr /*e*/) override {
    Error e = ProgramError(SourceLocation(input_file_name_, line)) << message;
    llvm::errs() << e << "\n";
    if (error_.has_value()) {
      error_ = std::move(e);
    }
  }

  auto error() -> std::optional<Error>& { return error_; }

 private:
  std::optional<Error> error_;
  std::string_view input_file_name_;
};

auto Parse(Nonnull<Arena*> /*arena*/, std::string_view input_file_name,
           bool parser_debug) -> ErrorOr<AST> {
  antlr4::ANTLRFileStream input_file;
  input_file.loadFromFile(std::string(input_file_name));

  CarbonLexer lexer(&input_file);
  antlr4::CommonTokenStream tokens(&lexer);
  CarbonParser parser(&tokens);

  ParserVisitor visitor;

  ErrorListener error_listener(input_file_name);
  lexer.removeErrorListeners();
  lexer.addErrorListener(&error_listener);
  parser.removeErrorListeners();
  parser.addErrorListener(&error_listener);
  parser.setTrace(parser_debug);
  parser.setProfile(true);
  /*
  antlrcpp::Any result = visitor.visit(parser.input());
  for (const auto& decision : parser.getParseInfo().getDecisionInfo()) {
    if (decision.timeInPrediction > 1e6) {
      llvm::errs()
          << "Time: " << static_cast<int>(decision.timeInPrediction / 1e6) << "
  in "
          << decision.invocations << " calls; rule: "
          << parser.getRuleNames().at(
                 parser.getATN().getDecisionState(decision.decision)->ruleIndex)
          << "\n";
    }
  }
  */
  parser.getInterpreter<antlr4::atn::ParserATNSimulator>()->setPredictionMode(
      antlr4::atn::PredictionMode::SLL);
  /*
  try {
    parser.stat();  // STAGE 1
  } catch (Exception ex) {
    tokens.reset();  // rewind input stream
    parser.reset();
    parser.getInterpreter().setPredictionMode(PredictionMode.LL);
    parser.stat();  // STAGE 2
                    // if we parse ok, it's LL not SLL
  }
  */
  visitor.visit(parser.input());
  return AST();
  /*
  if (result.has_value()) {
    return std::any_cast<AST>(result);
  } else {
    return std::move(*error_listener.error());
  }
  */
}

}  // namespace Carbon::Antlr
