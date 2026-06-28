// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <optional>
#include <string>

#include "common/check.h"
#include "common/raw_string_ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "testing/fuzzing/libfuzzer.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/diagnostics/null_diagnostics.h"
#include "toolchain/format/format.h"
#include "toolchain/lex/lex.h"
#include "toolchain/lex/token_index.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/parse.h"
#include "toolchain/parse/tree.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon::Testing {
namespace {

// The outcome of formatting one source text, recorded for the invariant checks
// below.
struct FormatResult {
  // The formatted text, from the whole-text `Format` path.
  std::string formatted;
  // The formatted text, reconstructed from the minimal-edit
  // `FormatReplacements` path. Equal to `formatted` for any input.
  std::string via_replacements;
  // Whether `FormatReplacements` produced any edits.
  bool has_replacements;
  // Whether the input was free of lex and parse errors.
  bool clean;
  // The kinds of the input's tokens, in order, for the token-preservation
  // invariant.
  llvm::SmallVector<Lex::TokenKind> token_kinds;
};

}  // namespace

// Lexes, parses, and formats `text` both ways, returning the results. Returns
// nullopt only when no source buffer can be made for `text` or when it has lex
// errors.
static auto FormatOnce(llvm::StringRef text) -> std::optional<FormatResult> {
  std::optional<SourceBuffer> source = SourceBuffer::MakeFromStringCopy(
      "fuzz.carbon", text, Diagnostics::NullConsumer());
  if (!source) {
    return std::nullopt;
  }

  SharedValueStores value_stores;
  Lex::LexOptions lex_options;
  lex_options.consumer = &Diagnostics::NullConsumer();
  Lex::TokenizedBuffer tokens = Lex::Lex(value_stores, *source, lex_options);
  if (tokens.has_errors()) {
    // Parsing lex-error tokens is outside the parser's tested contract, so skip
    // it -- as parse_fuzzer.cpp does -- rather than fuzz that separate path.
    return std::nullopt;
  }

  Parse::ParseOptions parse_options;
  parse_options.consumer = &Diagnostics::NullConsumer();
  Parse::Tree tree = Parse::Parse(tokens, parse_options);

  FormatResult result;
  // Lex errors are filtered above, so this reflects parse errors. The formatter
  // is best-effort on those, exercised below but without the stronger checks.
  result.clean = !tree.has_errors();
  for (Lex::TokenIndex token : tokens.tokens()) {
    result.token_kinds.push_back(tokens.GetKind(token));
  }

  RawStringOstream out;
  Format::Format(tree, out);
  result.formatted = out.TakeStr();

  llvm::SmallVector<Format::Replacement> replacements;
  Format::FormatReplacements(tree, replacements);
  result.has_replacements = !replacements.empty();
  result.via_replacements = Format::ApplyReplacements(text, replacements);
  return result;
}

// Whether `text` is free of control characters other than tab and newline. The
// stronger formatting invariants are only checked on such ordinary source: the
// toolchain's handling of stray control characters (carriage returns, NULs) is
// best-effort, and they can make formatting non-idempotent without indicating a
// real defect. Crashes are still checked on any input.
static auto IsOrdinarySource(llvm::StringRef text) -> bool {
  for (char c : text) {
    auto byte = static_cast<unsigned char>(c);
    if ((byte < ' ' || byte == 0x7F) && c != '\t' && c != '\n') {
      return false;
    }
  }
  return true;
}

// NOLINTNEXTLINE: Match the documented fuzzer entry point declaration style.
extern "C" int LLVMFuzzerTestOneInput(const unsigned char* data, size_t size) {
  // Ignore large inputs.
  // TODO: See tokenized_buffer_fuzzer.cpp.
  if (size > 100000) {
    return 0;
  }
  llvm::StringRef input(reinterpret_cast<const char*>(data), size);

  // Run the formatter both ways -- whole-text and minimal-edit -- to catch
  // crashes, assertion failures, and sanitizer errors. `FormatOnce` returns
  // nullopt for input with no source buffer or with lex errors.
  std::optional<FormatResult> first = FormatOnce(input);
  if (!first) {
    return 0;
  }

  // The formatter is best-effort on parse errors and stray control characters,
  // so the formatting invariants are only guaranteed for error-free, ordinary
  // source -- the contract `formatter_test` covers. Other input still exercises
  // the code above.
  if (!first->clean || !IsOrdinarySource(input)) {
    return 0;
  }

  // The minimal-edit path reconstructs the same text as the whole-text path.
  CARBON_CHECK(first->via_replacements == first->formatted,
               "minimal-edit output did not match whole-text output");

  std::optional<FormatResult> second = FormatOnce(first->formatted);
  CARBON_CHECK(second, "formatted output could not be re-read");

  // Idempotency: formatting already-formatted output changes nothing, and
  // produces no edits.
  CARBON_CHECK(second->formatted == first->formatted,
               "formatting is not idempotent");
  CARBON_CHECK(!second->has_replacements,
               "re-formatting already-formatted output produced edits");

  // Token preservation: formatting only changes whitespace and comments, so the
  // output lexes to the same token kinds as the input.
  CARBON_CHECK(second->token_kinds == first->token_kinds,
               "formatting changed the token sequence");

  // Validity preservation: error-free input formats to error-free output. The
  // token kinds alone cannot catch a layout decision that changes meaning
  // through whitespace, such as splitting a unary operator from its operand,
  // which the parser's fixity rules reject.
  CARBON_CHECK(second->clean, "formatting error-free input introduced errors");
  return 0;
}

}  // namespace Carbon::Testing
