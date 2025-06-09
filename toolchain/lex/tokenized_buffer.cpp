// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/tokenized_buffer.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <optional>
#include <utility>

#include "common/check.h"
#include "common/string_helpers.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/lex/character_set.h"
#include "toolchain/lex/numeric_literal.h"
#include "toolchain/lex/string_literal.h"

namespace Carbon::Lex {

auto TokenizedBuffer::GetLine(TokenIndex token) const -> LineIndex {
  return FindLineIndex(token_infos_.Get(token).byte_offset());
}

auto TokenizedBuffer::GetLineNumber(TokenIndex token) const -> int {
  return GetLine(token).index + 1;
}

auto TokenizedBuffer::GetColumnNumber(TokenIndex token) const -> int {
  const auto& token_info = token_infos_.Get(token);
  const auto& line_info =
      line_infos_.Get(FindLineIndex(token_info.byte_offset()));
  return token_info.byte_offset() - line_info.start + 1;
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
  const auto& token_info = token_infos_.Get(token);
  llvm::StringRef fixed_spelling = token_info.kind().fixed_spelling();
  if (!fixed_spelling.empty()) {
    return fixed_spelling;
  }

  if (token_info.kind() == TokenKind::Error) {
    return source_->text().substr(token_info.byte_offset(),
                                  token_info.error_length());
  }

  // Refer back to the source text to preserve oddities like radix or digit
  // separators the author included.
  if (token_info.kind() == TokenKind::IntLiteral ||
      token_info.kind() == TokenKind::RealLiteral) {
    std::optional<NumericLiteral> relexed_token =
        NumericLiteral::Lex(source_->text().substr(token_info.byte_offset()),
                            token_info.kind() == TokenKind::RealLiteral);
    CARBON_CHECK(relexed_token, "Could not reform numeric literal token.");
    return relexed_token->text();
  }

  // Refer back to the source text to find the original spelling, including
  // escape sequences etc.
  if (token_info.kind() == TokenKind::StringLiteral) {
    std::optional<StringLiteral> relexed_token =
        StringLiteral::Lex(source_->text().substr(token_info.byte_offset()));
    CARBON_CHECK(relexed_token, "Could not reform string literal token.");
    return relexed_token->text();
  }

  // Refer back to the source text to avoid needing to reconstruct the
  // spelling from the size.
  if (token_info.kind().is_sized_type_literal()) {
    llvm::StringRef suffix = source_->text()
                                 .substr(token_info.byte_offset() + 1)
                                 .take_while(IsDecimalDigit);
    return llvm::StringRef(suffix.data() - 1, suffix.size() + 1);
  }

  if (token_info.kind() == TokenKind::FileStart ||
      token_info.kind() == TokenKind::FileEnd) {
    return llvm::StringRef();
  }

  CARBON_CHECK(token_info.kind() == TokenKind::Identifier, "{0}",
               token_info.kind());
  return value_stores_->identifiers().Get(token_info.ident_id());
}

auto TokenizedBuffer::GetIdentifier(TokenIndex token) const -> IdentifierId {
  const auto& token_info = token_infos_.Get(token);
  CARBON_CHECK(token_info.kind() == TokenKind::Identifier, "{0}",
               token_info.kind());
  return token_info.ident_id();
}

auto TokenizedBuffer::GetIntLiteral(TokenIndex token) const -> IntId {
  const auto& token_info = token_infos_.Get(token);
  CARBON_CHECK(token_info.kind() == TokenKind::IntLiteral, "{0}",
               token_info.kind());
  return token_info.int_id();
}

auto TokenizedBuffer::GetRealLiteral(TokenIndex token) const -> RealId {
  const auto& token_info = token_infos_.Get(token);
  CARBON_CHECK(token_info.kind() == TokenKind::RealLiteral, "{0}",
               token_info.kind());
  return token_info.real_id();
}

auto TokenizedBuffer::GetStringLiteralValue(TokenIndex token) const
    -> StringLiteralValueId {
  const auto& token_info = token_infos_.Get(token);
  CARBON_CHECK(token_info.kind() == TokenKind::StringLiteral, "{0}",
               token_info.kind());
  return token_info.string_literal_id();
}

auto TokenizedBuffer::GetTypeLiteralSize(TokenIndex token) const -> IntId {
  const auto& token_info = token_infos_.Get(token);
  CARBON_CHECK(token_info.kind().is_sized_type_literal(), "{0}",
               token_info.kind());
  return token_info.int_id();
}

auto TokenizedBuffer::GetMatchedClosingToken(TokenIndex opening_token) const
    -> TokenIndex {
  const auto& opening_token_info = token_infos_.Get(opening_token);
  CARBON_CHECK(opening_token_info.kind().is_opening_symbol(), "{0}",
               opening_token_info.kind());
  return opening_token_info.closing_token_index();
}

auto TokenizedBuffer::GetMatchedOpeningToken(TokenIndex closing_token) const
    -> TokenIndex {
  const auto& closing_token_info = token_infos_.Get(closing_token);
  CARBON_CHECK(closing_token_info.kind().is_closing_symbol(), "{0}",
               closing_token_info.kind());
  return closing_token_info.opening_token_index();
}

auto TokenizedBuffer::IsRecoveryToken(TokenIndex token) const -> bool {
  if (recovery_tokens_.empty()) {
    return false;
  }
  return recovery_tokens_[token.index];
}

auto TokenizedBuffer::GetIndentColumnNumber(LineIndex line) const -> int {
  return line_infos_.Get(line).indent + 1;
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
  CARBON_CHECK(number >= 0, "Negative numbers are not supported.");
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

auto TokenizedBuffer::Print(llvm::raw_ostream& output_stream,
                            bool omit_file_boundary_tokens) const -> void {
  output_stream << "- filename: " << source_->filename() << "\n"
                << "  tokens:\n";

  PrintWidths widths = {};
  widths.index = ComputeDecimalPrintedWidth((token_infos_.size()));
  for (TokenIndex token : tokens()) {
    widths.Widen(GetTokenPrintWidths(token));
  }

  for (TokenIndex token : tokens()) {
    if (omit_file_boundary_tokens) {
      auto kind = GetKind(token);
      if (kind == TokenKind::FileStart || kind == TokenKind::FileEnd) {
        continue;
      }
    }
    PrintToken(output_stream, token, widths);
    output_stream << "\n";
  }

  if (!dump_sem_ir_ranges_.empty()) {
    output_stream << "  dump_sem_ir_ranges:\n";
    for (auto range : dump_sem_ir_ranges_) {
      output_stream << "  - {begin: " << range.begin.index
                    << ", end: " << range.end.index << "}\n";
    }
  }
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
  const auto& token_info = token_infos_.Get(token);
  LineIndex line_index = FindLineIndex(token_info.byte_offset());
  llvm::StringRef token_text = GetTokenText(token);

  // Output the main chunk using one format string. We have to do the
  // justification manually in order to use the dynamically computed widths
  // and get the quotes included.
  output_stream << llvm::formatv(
      "  - { index: {0}, kind: {1}, line: {2}, column: {3}, indent: {4}, "
      "spelling: \"{5}\"",
      llvm::format_decimal(token_index, widths.index),
      llvm::right_justify(
          llvm::formatv("\"{0}\"", token_info.kind().name()).str(),
          widths.kind + 2),
      llvm::format_decimal(GetLineNumber(token), widths.line),
      llvm::format_decimal(GetColumnNumber(token), widths.column),
      llvm::format_decimal(GetIndentColumnNumber(line_index), widths.indent),
      FormatEscaped(token_text, /*use_hex_escapes=*/true));

  switch (token_info.kind()) {
    case TokenKind::Identifier:
      output_stream << ", identifier: " << GetIdentifier(token).index;
      break;
    case TokenKind::IntLiteral:
      output_stream << ", value: \"";
      value_stores_->ints()
          .Get(GetIntLiteral(token))
          .print(output_stream, /*isSigned=*/false);
      output_stream << "\"";
      break;
    case TokenKind::RealLiteral:
      output_stream << ", value: \""
                    << value_stores_->reals().Get(GetRealLiteral(token))
                    << "\"";
      break;
    case TokenKind::StringLiteral:
      output_stream << ", value: \""
                    << FormatEscaped(value_stores_->string_literal_values().Get(
                                         GetStringLiteralValue(token)),
                                     /*use_hex_escapes=*/true)
                    << "\"";
      break;
    default:
      if (token_info.kind().is_opening_symbol()) {
        output_stream << ", closing_token: "
                      << GetMatchedClosingToken(token).index;
      } else if (token_info.kind().is_closing_symbol()) {
        output_stream << ", opening_token: "
                      << GetMatchedOpeningToken(token).index;
      }
      break;
  }

  if (token_info.has_leading_space()) {
    output_stream << ", has_leading_space: true";
  }
  if (IsRecoveryToken(token)) {
    output_stream << ", recovery: true";
  }

  output_stream << " }";
}

// Find the line index corresponding to a specific byte offset within the source
// text for this tokenized buffer.
//
// This takes advantage of the lines being sorted by their starting byte offsets
// to do a binary search for the line that contains the provided offset.
auto TokenizedBuffer::FindLineIndex(int32_t byte_offset) const -> LineIndex {
  CARBON_DCHECK(line_infos_.size() > 0);

  auto line_range = line_infos_.values();
  auto line_it =
      llvm::partition_point(line_range, [byte_offset](LineInfo line_info) {
        return line_info.start <= byte_offset;
      });
  --line_it;

  // If this isn't the first line but it starts past the end of the source, then
  // this is a synthetic line added for simplicity of lexing. Step back one
  // further to find the last non-synthetic line.
  if (line_it != line_range.begin() &&
      line_it->start == static_cast<int32_t>(source_->text().size())) {
    --line_it;
  }
  CARBON_DCHECK(line_it->start <= byte_offset);
  return LineIndex(line_it - line_range.begin());
}

auto TokenizedBuffer::IsAfterComment(TokenIndex token,
                                     CommentIndex comment_index) const -> bool {
  const auto& comment_data = comments_.Get(comment_index);
  return token_infos_.Get(token).byte_offset() > comment_data.start;
}

auto TokenizedBuffer::GetCommentText(CommentIndex comment_index) const
    -> llvm::StringRef {
  const auto& comment_data = comments_.Get(comment_index);
  return source_->text().substr(comment_data.start, comment_data.length);
}

auto TokenizedBuffer::AddComment(int32_t indent, int32_t start, int32_t end)
    -> void {
  if (comments_.size() > 0) {
    auto& comment = comments_.Get(CommentIndex(comments_.size() - 1));
    if (comment.start + comment.length + indent == start) {
      comment.length = end - comment.start;
      return;
    }
  }
  comments_.Add({.start = start, .length = end - start});
}

auto TokenizedBuffer::CollectMemUsage(MemUsage& mem_usage,
                                      llvm::StringRef label) const -> void {
  mem_usage.Collect(MemUsage::ConcatLabel(label, "allocator_"), allocator_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "token_infos_"), token_infos_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "line_infos_"), line_infos_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "comments_"), comments_);
}

auto TokenizedBuffer::SourcePointerToDiagnosticLoc(const char* loc) const
    -> Diagnostics::ConvertedLoc {
  CARBON_CHECK(StringRefContainsPointer(source_->text(), loc),
               "location not within buffer");
  int32_t offset = loc - source_->text().begin();

  auto line_range = line_infos_.values();

  // Find the first line starting after the given location.
  const auto next_line_it = llvm::partition_point(
      line_range,
      [offset](const LineInfo& line) { return line.start <= offset; });

  // Step back one line to find the line containing the given position.
  CARBON_CHECK(next_line_it != line_range.begin(),
               "location precedes the start of the first line");
  const auto line_it = std::prev(next_line_it);
  int line_number = line_it - line_range.begin();
  int column_number = offset - line_it->start;

  // Grab the line from the buffer by slicing from this line to the next
  // minus the newline. When on the last line, instead use the start to the end
  // of the buffer.
  llvm::StringRef text = source_->text();
  llvm::StringRef line = next_line_it != line_range.end()
                             ? text.slice(line_it->start, next_line_it->start)
                             : text.substr(line_it->start);

  // Remove a newline at the end of the line if present.
  // TODO: This should expand to remove all vertical whitespace bytes at the
  // tail of the line such as CR+LF, etc.
  line.consume_back("\n");

  return {.loc = {.filename = source_->filename(),
                  .line = line,
                  .line_number = line_number + 1,
                  .column_number = column_number + 1},
          .last_byte_offset = offset};
}

auto TokenizedBuffer::TokenToDiagnosticLoc(TokenIndex token) const
    -> Diagnostics::ConvertedLoc {
  // Map the token location into a position within the source buffer.
  const char* token_start =
      source_->text().begin() + token_infos_.Get(token).byte_offset();

  // Find the corresponding file location.
  // TODO: Should we somehow indicate in the diagnostic location if this token
  // is a recovery token that doesn't correspond to the original source?
  auto converted = SourcePointerToDiagnosticLoc(token_start);
  converted.loc.length = GetTokenText(token).size();
  return converted;
}

auto TokenizedBuffer::OverlapsWithDumpSemIRRange(
    Lex::InclusiveTokenRange range) const -> bool {
  CARBON_CHECK(!dump_sem_ir_ranges_.empty());

  // Ranges are ordered, so we can decide overlap as soon as we find a range
  // that ends after `begin`.
  for (auto dump_range : dump_sem_ir_ranges_) {
    if (dump_range.end >= range.begin) {
      return dump_range.begin <= range.end;
    }
  }
  return false;
}

}  // namespace Carbon::Lex
