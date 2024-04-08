// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/tokenized_buffer.h"

#include <cmath>

#include "common/check.h"
#include "common/string_helpers.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/base/value_store.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/lex/character_set.h"
#include "toolchain/lex/numeric_literal.h"
#include "toolchain/lex/string_literal.h"

namespace Carbon::Lex {

auto TokenizedBuffer::GetKind(TokenIndex token) const -> TokenKind {
  return GetTokenInfo(token).kind;
}

auto TokenizedBuffer::GetLine(TokenIndex token) const -> LineIndex {
  return GetTokenInfo(token).token_line;
}

auto TokenizedBuffer::GetLineNumber(TokenIndex token) const -> int {
  return GetLineNumber(GetLine(token));
}

auto TokenizedBuffer::GetColumnNumber(TokenIndex token) const -> int {
  return GetTokenInfo(token).column + 1;
}

auto TokenizedBuffer::GetEndLoc(TokenIndex token) const
    -> std::pair<LineIndex, int> {
  auto line = GetLine(token);
  int column = GetColumnNumber(token);
  auto token_text = GetTokenText(token);

  if (auto [before_newline, after_newline] = token_text.rsplit('\n');
      before_newline.size() == token_text.size()) {
    // Token fits on one line, advance the column number.
    column += before_newline.size();
  } else {
    // Token contains newlines.
    line.index += before_newline.count('\n') + 1;
    column = 1 + after_newline.size();
  }

  return {line, column};
}

auto TokenizedBuffer::GetTokenText(TokenIndex token) const -> llvm::StringRef {
  const auto& token_info = GetTokenInfo(token);
  llvm::StringRef fixed_spelling = token_info.kind.fixed_spelling();
  if (!fixed_spelling.empty()) {
    return fixed_spelling;
  }

  if (token_info.kind == TokenKind::Error) {
    const auto& line_info = GetLineInfo(token_info.token_line);
    int64_t token_start = line_info.start + token_info.column;
    return source_->text().substr(token_start, token_info.error_length);
  }

  // Refer back to the source text to preserve oddities like radix or digit
  // separators the author included.
  if (token_info.kind == TokenKind::IntLiteral ||
      token_info.kind == TokenKind::RealLiteral) {
    const auto& line_info = GetLineInfo(token_info.token_line);
    int64_t token_start = line_info.start + token_info.column;
    std::optional<NumericLiteral> relexed_token =
        NumericLiteral::Lex(source_->text().substr(token_start));
    CARBON_CHECK(relexed_token) << "Could not reform numeric literal token.";
    return relexed_token->text();
  }

  // Refer back to the source text to find the original spelling, including
  // escape sequences etc.
  if (token_info.kind == TokenKind::StringLiteral) {
    const auto& line_info = GetLineInfo(token_info.token_line);
    int64_t token_start = line_info.start + token_info.column;
    std::optional<StringLiteral> relexed_token =
        StringLiteral::Lex(source_->text().substr(token_start));
    CARBON_CHECK(relexed_token) << "Could not reform string literal token.";
    return relexed_token->text();
  }

  // Refer back to the source text to avoid needing to reconstruct the
  // spelling from the size.
  if (token_info.kind.is_sized_type_literal()) {
    const auto& line_info = GetLineInfo(token_info.token_line);
    int64_t token_start = line_info.start + token_info.column;
    llvm::StringRef suffix =
        source_->text().substr(token_start + 1).take_while(IsDecimalDigit);
    return llvm::StringRef(suffix.data() - 1, suffix.size() + 1);
  }

  if (token_info.kind == TokenKind::FileStart ||
      token_info.kind == TokenKind::FileEnd) {
    return llvm::StringRef();
  }

  CARBON_CHECK(token_info.kind == TokenKind::Identifier) << token_info.kind;
  return value_stores_->identifiers().Get(token_info.ident_id);
}

auto TokenizedBuffer::GetIdentifier(TokenIndex token) const -> IdentifierId {
  const auto& token_info = GetTokenInfo(token);
  CARBON_CHECK(token_info.kind == TokenKind::Identifier) << token_info.kind;
  return token_info.ident_id;
}

auto TokenizedBuffer::GetIntLiteral(TokenIndex token) const -> IntId {
  const auto& token_info = GetTokenInfo(token);
  CARBON_CHECK(token_info.kind == TokenKind::IntLiteral) << token_info.kind;
  return token_info.int_id;
}

auto TokenizedBuffer::GetRealLiteral(TokenIndex token) const -> RealId {
  const auto& token_info = GetTokenInfo(token);
  CARBON_CHECK(token_info.kind == TokenKind::RealLiteral) << token_info.kind;
  return token_info.real_id;
}

auto TokenizedBuffer::GetStringLiteralValue(TokenIndex token) const
    -> StringLiteralValueId {
  const auto& token_info = GetTokenInfo(token);
  CARBON_CHECK(token_info.kind == TokenKind::StringLiteral) << token_info.kind;
  return token_info.string_literal_id;
}

auto TokenizedBuffer::GetTypeLiteralSize(TokenIndex token) const -> IntId {
  const auto& token_info = GetTokenInfo(token);
  CARBON_CHECK(token_info.kind.is_sized_type_literal()) << token_info.kind;
  return token_info.int_id;
}

auto TokenizedBuffer::GetMatchedClosingToken(TokenIndex opening_token) const
    -> TokenIndex {
  const auto& opening_token_info = GetTokenInfo(opening_token);
  CARBON_CHECK(opening_token_info.kind.is_opening_symbol())
      << opening_token_info.kind;
  return opening_token_info.closing_token;
}

auto TokenizedBuffer::GetMatchedOpeningToken(TokenIndex closing_token) const
    -> TokenIndex {
  const auto& closing_token_info = GetTokenInfo(closing_token);
  CARBON_CHECK(closing_token_info.kind.is_closing_symbol())
      << closing_token_info.kind;
  return closing_token_info.opening_token;
}

auto TokenizedBuffer::HasLeadingWhitespace(TokenIndex token) const -> bool {
  auto it = TokenIterator(token);
  return it == tokens().begin() || GetTokenInfo(*(it - 1)).has_trailing_space;
}

auto TokenizedBuffer::HasTrailingWhitespace(TokenIndex token) const -> bool {
  return GetTokenInfo(token).has_trailing_space;
}

auto TokenizedBuffer::IsRecoveryToken(TokenIndex token) const -> bool {
  return GetTokenInfo(token).is_recovery;
}

auto TokenizedBuffer::GetLineNumber(LineIndex line) const -> int {
  return line.index + 1;
}

auto TokenizedBuffer::GetNextLine(LineIndex line) const -> LineIndex {
  LineIndex next(line.index + 1);
  CARBON_DCHECK(static_cast<size_t>(next.index) < line_infos_.size());
  return next;
}

auto TokenizedBuffer::GetPrevLine(LineIndex line) const -> LineIndex {
  CARBON_CHECK(line.index > 0);
  return LineIndex(line.index - 1);
}

auto TokenizedBuffer::GetIndentColumnNumber(LineIndex line) const -> int {
  return GetLineInfo(line).indent + 1;
}

auto TokenizedBuffer::PrintWidths::Widen(const PrintWidths& widths) -> void {
  index = std::max(widths.index, index);
  kind = std::max(widths.kind, kind);
  column = std::max(widths.column, column);
  line = std::max(widths.line, line);
  indent = std::max(widths.indent, indent);
}

// Compute the printed width of a number. When numbers are printed in decimal,
// the number of digits needed is one more than the log-base-10 of the
// value. We handle a value of `zero` explicitly.
//
// This routine requires its argument to be *non-negative*.
static auto ComputeDecimalPrintedWidth(int number) -> int {
  CARBON_CHECK(number >= 0) << "Negative numbers are not supported.";
  if (number == 0) {
    return 1;
  }

  return static_cast<int>(std::log10(number)) + 1;
}

auto TokenizedBuffer::GetTokenPrintWidths(TokenIndex token) const
    -> PrintWidths {
  PrintWidths widths = {};
  widths.index = ComputeDecimalPrintedWidth(token_infos_.size());
  widths.kind = GetKind(token).name().size();
  widths.line = ComputeDecimalPrintedWidth(GetLineNumber(token));
  widths.column = ComputeDecimalPrintedWidth(GetColumnNumber(token));
  widths.indent =
      ComputeDecimalPrintedWidth(GetIndentColumnNumber(GetLine(token)));
  return widths;
}

auto TokenizedBuffer::Print(llvm::raw_ostream& output_stream) const -> void {
  if (tokens().begin() == tokens().end()) {
    return;
  }

  output_stream << "- filename: " << source_->filename() << "\n"
                << "  tokens: [\n";

  PrintWidths widths = {};
  widths.index = ComputeDecimalPrintedWidth((token_infos_.size()));
  for (TokenIndex token : tokens()) {
    widths.Widen(GetTokenPrintWidths(token));
  }

  for (TokenIndex token : tokens()) {
    PrintToken(output_stream, token, widths);
    output_stream << "\n";
  }
  output_stream << "  ]\n";
}

auto TokenizedBuffer::PrintToken(llvm::raw_ostream& output_stream,
                                 TokenIndex token) const -> void {
  PrintToken(output_stream, token, {});
}

auto TokenizedBuffer::PrintToken(llvm::raw_ostream& output_stream,
                                 TokenIndex token, PrintWidths widths) const
    -> void {
  widths.Widen(GetTokenPrintWidths(token));
  int token_index = token.index;
  const auto& token_info = GetTokenInfo(token);
  llvm::StringRef token_text = GetTokenText(token);

  // Output the main chunk using one format string. We have to do the
  // justification manually in order to use the dynamically computed widths
  // and get the quotes included.
  output_stream << llvm::formatv(
      "    { index: {0}, kind: {1}, line: {2}, column: {3}, indent: {4}, "
      "spelling: '{5}'",
      llvm::format_decimal(token_index, widths.index),
      llvm::right_justify(llvm::formatv("'{0}'", token_info.kind.name()).str(),
                          widths.kind + 2),
      llvm::format_decimal(GetLineNumber(token_info.token_line), widths.line),
      llvm::format_decimal(GetColumnNumber(token), widths.column),
      llvm::format_decimal(GetIndentColumnNumber(token_info.token_line),
                           widths.indent),
      token_text);

  switch (token_info.kind) {
    case TokenKind::Identifier:
      output_stream << ", identifier: " << GetIdentifier(token).index;
      break;
    case TokenKind::IntLiteral:
      output_stream << ", value: `";
      value_stores_->ints()
          .Get(GetIntLiteral(token))
          .print(output_stream, /*isSigned=*/false);
      output_stream << "`";
      break;
    case TokenKind::RealLiteral:
      output_stream << ", value: `"
                    << value_stores_->reals().Get(GetRealLiteral(token)) << "`";
      break;
    case TokenKind::StringLiteral:
      output_stream << ", value: `"
                    << value_stores_->string_literal_values().Get(
                           GetStringLiteralValue(token))
                    << "`";
      break;
    default:
      if (token_info.kind.is_opening_symbol()) {
        output_stream << ", closing_token: "
                      << GetMatchedClosingToken(token).index;
      } else if (token_info.kind.is_closing_symbol()) {
        output_stream << ", opening_token: "
                      << GetMatchedOpeningToken(token).index;
      }
      break;
  }

  if (token_info.has_trailing_space) {
    output_stream << ", has_trailing_space: true";
  }
  if (token_info.is_recovery) {
    output_stream << ", recovery: true";
  }

  output_stream << " },";
}

auto TokenizedBuffer::GetLineInfo(LineIndex line) -> LineInfo& {
  return line_infos_[line.index];
}

auto TokenizedBuffer::GetLineInfo(LineIndex line) const -> const LineInfo& {
  return line_infos_[line.index];
}

auto TokenizedBuffer::AddLine(LineInfo info) -> LineIndex {
  line_infos_.push_back(info);
  return LineIndex(static_cast<int>(line_infos_.size()) - 1);
}

auto TokenizedBuffer::GetTokenInfo(TokenIndex token) -> TokenInfo& {
  return token_infos_[token.index];
}

auto TokenizedBuffer::GetTokenInfo(TokenIndex token) const -> const TokenInfo& {
  return token_infos_[token.index];
}

auto TokenizedBuffer::AddToken(TokenInfo info) -> TokenIndex {
  token_infos_.push_back(info);
  expected_parse_tree_size_ += info.kind.expected_parse_tree_size();
  return TokenIndex(static_cast<int>(token_infos_.size()) - 1);
}

auto TokenIterator::Print(llvm::raw_ostream& output) const -> void {
  output << token_.index;
}

auto TokenizedBuffer::SourceBufferDiagnosticConverter::ConvertLoc(
    const char* loc, ContextFnT /*context_fn*/) const -> DiagnosticLoc {
  CARBON_CHECK(StringRefContainsPointer(buffer_->source_->text(), loc))
      << "location not within buffer";
  int64_t offset = loc - buffer_->source_->text().begin();

  // Find the first line starting after the given location. Note that we can't
  // inspect `line.length` here because it is not necessarily correct for the
  // final line during lexing (but will be correct later for the parse tree).
  const auto* line_it = std::partition_point(
      buffer_->line_infos_.begin(), buffer_->line_infos_.end(),
      [offset](const LineInfo& line) { return line.start <= offset; });

  // Step back one line to find the line containing the given position.
  CARBON_CHECK(line_it != buffer_->line_infos_.begin())
      << "location precedes the start of the first line";
  --line_it;
  int line_number = line_it - buffer_->line_infos_.begin();
  int column_number = offset - line_it->start;

  // Start by grabbing the line from the buffer. If the line isn't fully lexed,
  // the length will be npos and the line will be grabbed from the known start
  // to the end of the buffer; we'll then adjust the length.
  llvm::StringRef line =
      buffer_->source_->text().substr(line_it->start, line_it->length);
  if (line_it->length == static_cast<int32_t>(llvm::StringRef::npos)) {
    CARBON_CHECK(line.take_front(column_number).count('\n') == 0)
        << "Currently we assume no unlexed newlines prior to the error column, "
           "but there was one when erroring at "
        << buffer_->source_->filename() << ":" << line_number << ":"
        << column_number;
    // Look for the next newline since we don't know the length. We can start at
    // the column because prior newlines will have been lexed.
    auto end_newline_pos = line.find('\n', column_number);
    if (end_newline_pos != llvm::StringRef::npos) {
      line = line.take_front(end_newline_pos);
    }
  }

  return {.filename = buffer_->source_->filename(),
          .line = line,
          .line_number = line_number + 1,
          .column_number = column_number + 1};
}

auto TokenDiagnosticConverter::ConvertLoc(TokenIndex token,
                                          ContextFnT context_fn) const
    -> DiagnosticLoc {
  // Map the token location into a position within the source buffer.
  const auto& token_info = buffer_->GetTokenInfo(token);
  const auto& line_info = buffer_->GetLineInfo(token_info.token_line);
  const char* token_start =
      buffer_->source_->text().begin() + line_info.start + token_info.column;

  // Find the corresponding file location.
  // TODO: Should we somehow indicate in the diagnostic location if this token
  // is a recovery token that doesn't correspond to the original source?
  DiagnosticLoc loc =
      TokenizedBuffer::SourceBufferDiagnosticConverter(buffer_).ConvertLoc(
          token_start, context_fn);
  loc.length = buffer_->GetTokenText(token).size();
  return loc;
}

}  // namespace Carbon::Lex
