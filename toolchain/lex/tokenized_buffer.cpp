// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/tokenized_buffer.h"

#include <algorithm>
#include <array>
#include <cmath>

#include "common/check.h"
#include "common/string_helpers.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "toolchain/lex/character_set.h"
#include "toolchain/lex/helpers.h"
#include "toolchain/lex/numeric_literal.h"
#include "toolchain/lex/string_literal.h"

#if __x86_64__
#include <x86intrin.h>
#endif

namespace Carbon::Lex {

// TODO: Move Overload and VariantMatch somewhere more central.

// Form an overload set from a list of functions. For example:
//
// ```
// auto overloaded = Overload{[] (int) {}, [] (float) {}};
// ```
template <typename... Fs>
struct Overload : Fs... {
  using Fs::operator()...;
};
template <typename... Fs>
Overload(Fs...) -> Overload<Fs...>;

// Pattern-match against the type of the value stored in the variant `V`. Each
// element of `fs` should be a function that takes one or more of the variant
// values in `V`.
template <typename V, typename... Fs>
auto VariantMatch(V&& v, Fs&&... fs) -> decltype(auto) {
  return std::visit(Overload{std::forward<Fs&&>(fs)...}, std::forward<V&&>(v));
}

#if __x86_64__
#define CARBON_USE_SIMD 1

// A table of masks to include 0-16 bytes of an SSE register.
// TODO: Make this constexpr to avoid dynamic initialization.
static const std::array<__m128i, sizeof(__m128i) + 1> prefix_masks = [] {
  std::array<__m128i, sizeof(__m128i) + 1> masks = {};
  for (auto [i, mask] : llvm::enumerate(masks)) {
    memset(&mask, 0xFF, i);
  }
  return masks;
}();
#else
#define CARBON_USE_SIMD 0
#endif

// Scans the provided text and returns the prefix `StringRef` of contiguous
// identifier characters.
//
// This is a performance sensitive function and so uses vectorized code
// sequences to optimize its scanning. When modifying, the identifier lexing
// benchmarks should be checked for regressions.
//
// Identifier characters here are currently the ASCII characters `[0-9A-Za-z_]`.
//
// TODO: Currently, this code does not implement Carbon's design for Unicode
// characters in identifiers. It does work on UTF-8 code unit sequences, but
// currently considers non-ASCII characters to be non-identifier characters.
// Some work has been done to ensure the hot loop, while optimized, retains
// enough information to add Unicode handling without completely destroying the
// relevant optimizations.
static auto ScanForIdentifierPrefix(llvm::StringRef text) -> llvm::StringRef {
  // A table of booleans that we can use to classify bytes as being valid
  // identifier (or keyword) characters. This is used in the generic,
  // non-vectorized fallback code to scan for length of an identifier.
  static constexpr std::array<bool, 256> IsIdByteTable = ([]() constexpr {
    std::array<bool, 256> table = {};
    for (char c = '0'; c <= '9'; ++c) {
      table[c] = true;
    }
    for (char c = 'A'; c <= 'Z'; ++c) {
      table[c] = true;
    }
    for (char c = 'a'; c <= 'z'; ++c) {
      table[c] = true;
    }
    table['_'] = true;
    return table;
  })();

#if CARBON_USE_SIMD && __x86_64__
  // This code uses a scheme derived from the techniques in Geoff Langdale and
  // Daniel Lemire's work on parsing JSON[1]. Specifically, that paper outlines
  // a technique of using two 4-bit indexed in-register look-up tables (LUTs) to
  // classify bytes in a branchless SIMD code sequence.
  //
  // [1]: https://arxiv.org/pdf/1902.08318.pdf
  //
  // The goal is to get a bit mask classifying different sets of bytes. For each
  // input byte, we first test for a high bit indicating a UTF-8 encoded Unicode
  // character. Otherwise, we want the mask bits to be set with the following
  // logic derived by inspecting the high nibble and low nibble of the input:
  // bit0 = 1 for `_`: high `0x5` and low `0xF`
  // bit1 = 1 for `0-9`: high `0x3` and low `0x0` - `0x9`
  // bit2 = 1 for `A-O` and `a-o`: high `0x4` or `0x6` and low `0x1` - `0xF`
  // bit3 = 1 for `P-Z` and 'p-z': high `0x5` or `0x7` and low `0x0` - `0xA`
  // bit4 = unused
  // bit5 = unused
  // bit6 = unused
  // bit7 = unused
  //
  // No bits set means definitively non-ID ASCII character.
  //
  // bits 4-7 remain unused if we need to classify more characters.
  const auto high_lut = _mm_setr_epi8(
      /* __b0=*/0b0000'0000,
      /* __b1=*/0b0000'0000,
      /* __b2=*/0b0000'0000,
      /* __b3=*/0b0000'0010,
      /* __b4=*/0b0000'0100,
      /* __b5=*/0b0000'1001,
      /* __b6=*/0b0000'0100,
      /* __b7=*/0b0000'1000,
      /* __b8=*/0b0000'0000,
      /* __b9=*/0b0000'0000,
      /*__b10=*/0b0000'0000,
      /*__b11=*/0b0000'0000,
      /*__b12=*/0b0000'0000,
      /*__b13=*/0b0000'0000,
      /*__b14=*/0b0000'0000,
      /*__b15=*/0b0000'0000);
  const auto low_lut = _mm_setr_epi8(
      /* __b0=*/0b0000'1010,
      /* __b1=*/0b0000'1110,
      /* __b2=*/0b0000'1110,
      /* __b3=*/0b0000'1110,
      /* __b4=*/0b0000'1110,
      /* __b5=*/0b0000'1110,
      /* __b6=*/0b0000'1110,
      /* __b7=*/0b0000'1110,
      /* __b8=*/0b0000'1110,
      /* __b9=*/0b0000'1110,
      /*__b10=*/0b0000'1100,
      /*__b11=*/0b0000'0100,
      /*__b12=*/0b0000'0100,
      /*__b13=*/0b0000'0100,
      /*__b14=*/0b0000'0100,
      /*__b15=*/0b0000'0101);

  // Use `ssize_t` for performance here as we index memory in a tight loop.
  ssize_t i = 0;
  const ssize_t size = text.size();
  while ((i + 16) <= size) {
    __m128i input =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(text.data() + i));

    // The high bits of each byte indicate a non-ASCII character encoded using
    // UTF-8. Test those and fall back to the scalar code if present. These
    // bytes will also cause spurious zeros in the LUT results, but we can
    // ignore that because we track them independently here.
#if __SSE4_1__
    if (!_mm_test_all_zeros(_mm_set1_epi8(0x80), input)) {
      break;
    }
#else
    if (_mm_movemask_epi8(input) != 0) {
      break;
    }
#endif

    // Do two LUT lookups and mask the results together to get the results for
    // both low and high nibbles. Note that we don't need to mask out the high
    // bit of input here because we track that above for UTF-8 handling.
    __m128i low_mask = _mm_shuffle_epi8(low_lut, input);
    // Note that the input needs to be masked to only include the high nibble or
    // we could end up with bit7 set forcing the result to a zero byte.
    __m128i input_high =
        _mm_and_si128(_mm_srli_epi32(input, 4), _mm_set1_epi8(0x0f));
    __m128i high_mask = _mm_shuffle_epi8(high_lut, input_high);
    __m128i mask = _mm_and_si128(low_mask, high_mask);

    // Now compare to find the completely zero bytes.
    __m128i id_byte_mask_vec = _mm_cmpeq_epi8(mask, _mm_setzero_si128());
    int tail_ascii_mask = _mm_movemask_epi8(id_byte_mask_vec);

    // Check if there are bits in the tail mask, which means zero bytes and the
    // end of the identifier. We could do this without materializing the scalar
    // mask on more recent CPUs, but we generally expect the median length we
    // encounter to be <16 characters and so we avoid the extra instruction in
    // that case and predict this branch to succeed so it is laid out in a
    // reasonable way.
    if (LLVM_LIKELY(tail_ascii_mask != 0)) {
      // Move past the definitively classified bytes that are part of the
      // identifier, and return the complete identifier text.
      i += __builtin_ctz(tail_ascii_mask);
      return text.substr(0, i);
    }
    i += 16;
  }

  // Fallback to scalar loop. We only end up here when we don't have >=16
  // bytes to scan or we find a UTF-8 unicode character.
  // TODO: This assumes all Unicode characters are non-identifiers.
  while (i < size && IsIdByteTable[static_cast<unsigned char>(text[i])]) {
    ++i;
  }

  return text.substr(0, i);
#else
  // TODO: Optimize this with SIMD for other architectures.
  return text.take_while(
      [](char c) { return IsIdByteTable[static_cast<unsigned char>(c)]; });
#endif
}

// Implementation of the lexer logic itself.
//
// The design is that lexing can loop over the source buffer, consuming it into
// tokens by calling into this API. This class handles the state and breaks down
// the different lexing steps that may be used. It directly updates the provided
// tokenized buffer with the lexed tokens.
class [[clang::internal_linkage]] TokenizedBuffer::Lexer {
 public:
  // Symbolic result of a lexing action. This indicates whether we successfully
  // lexed a token, or whether other lexing actions should be attempted.
  //
  // While it wraps a simple boolean state, its API both helps make the failures
  // more self documenting, and by consuming the actual token constructively
  // when one is produced, it helps ensure the correct result is returned.
  class LexResult {
   public:
    // Consumes (and discard) a valid token to construct a result
    // indicating a token has been produced. Relies on implicit conversions.
    // NOLINTNEXTLINE(google-explicit-constructor)
    LexResult(Token /*discarded_token*/) : LexResult(true) {}

    // Returns a result indicating no token was produced.
    static auto NoMatch() -> LexResult { return LexResult(false); }

    // Tests whether a token was produced by the lexing routine, and
    // the lexer can continue forming tokens.
    explicit operator bool() const { return formed_token_; }

   private:
    explicit LexResult(bool formed_token) : formed_token_(formed_token) {}

    bool formed_token_;
  };

  Lexer(SourceBuffer& source, DiagnosticConsumer& consumer)
      : buffer_(source),
        consumer_(consumer),
        translator_(&buffer_),
        emitter_(translator_, consumer_),
        token_translator_(&buffer_),
        token_emitter_(token_translator_, consumer_) {}

  // Find all line endings and create the line data structures. Explicitly kept
  // out-of-line because this is a significant loop that is useful to have in
  // the profile and it doesn't simplify by inlining at all. But because it can,
  // the compiler will flatten this otherwise.
  [[gnu::noinline]] auto CreateLines(llvm::StringRef source_text) -> void {
    // We currently use `memchr` here which typically is well optimized to use
    // SIMD or other significantly faster than byte-wise scanning. We also use
    // carefully selected variables and the `ssize_t` type for performance and
    // code size of this hot loop.
    //
    // TODO: Eventually, we'll likely need to roll our own SIMD-optimized
    // routine here in order to handle CR+LF line endings, as we'll want those
    // to stay on the fast path. We'll also need to detect and diagnose Unicode
    // vertical whitespace. Starting with `memchr` should give us a strong
    // baseline performance target when adding those features.
    const char* const text = source_text.data();
    const ssize_t size = source_text.size();
    ssize_t start = 0;
    while (const char* nl = reinterpret_cast<const char*>(
               memchr(&text[start], '\n', size - start))) {
      ssize_t nl_index = nl - text;
      buffer_.AddLine(LineInfo(start, nl_index - start));
      start = nl_index + 1;
    }
    // The last line ends at the end of the file.
    buffer_.AddLine(LineInfo(start, size - start));

    // If the last line wasn't empty, the file ends with an unterminated line.
    // Add an extra blank line so that we never need to handle the special case
    // of being on the last line inside the lexer and needing to not increment
    // to the next line.
    if (start != size) {
      buffer_.AddLine(LineInfo(size, 0));
    }

    // Now that all the infos are allocated, get a fresh pointer to the first
    // info for use while lexing.
    line_index_ = 0;
  }

  auto current_line() -> Line { return Line(line_index_); }

  auto current_line_info() -> LineInfo* {
    return &buffer_.line_infos_[line_index_];
  }

  auto ComputeColumn(ssize_t position) -> int {
    CARBON_DCHECK(position >= current_line_info()->start);
    return position - current_line_info()->start;
  }

  auto NoteWhitespace() -> void {
    buffer_.token_infos_.back().has_trailing_space = true;
  }

  auto SkipHorizontalWhitespace(llvm::StringRef source_text, ssize_t& position)
      -> void {
    // Handle adjacent whitespace quickly. This comes up frequently for example
    // due to indentation. We don't expect *huge* runs, so just use a scalar
    // loop. While still scalar, this avoids repeated table dispatch and marking
    // whitespace.
    while (position < static_cast<ssize_t>(source_text.size()) &&
           (source_text[position] == ' ' || source_text[position] == '\t')) {
      ++position;
    }
  }

  auto LexHorizontalWhitespace(llvm::StringRef source_text, ssize_t& position)
      -> void {
    CARBON_DCHECK(source_text[position] == ' ' ||
                  source_text[position] == '\t');
    NoteWhitespace();
    // Skip runs using an optimized code path.
    SkipHorizontalWhitespace(source_text, position);
  }

  auto LexVerticalWhitespace(llvm::StringRef source_text, ssize_t& position)
      -> void {
    NoteWhitespace();
    ++line_index_;
    auto* line_info = current_line_info();
    ssize_t line_start = line_info->start;
    position = line_start;
    SkipHorizontalWhitespace(source_text, position);
    line_info->indent = position - line_start;
  }

  auto LexCommentOrSlash(llvm::StringRef source_text, ssize_t& position)
      -> void {
    CARBON_DCHECK(source_text[position] == '/');

    // Both comments and slash symbols start with a `/`. We disambiguate with a
    // max-munch rule -- if the next character is another `/` then we lex it as
    // a comment start. If it isn't, then we lex as a slash. We also optimize
    // for the comment case as we expect that to be much more important for
    // overall lexer performance.
    if (LLVM_LIKELY(position + 1 < static_cast<ssize_t>(source_text.size()) &&
                    source_text[position + 1] == '/')) {
      LexComment(source_text, position);
      return;
    }

    // This code path should produce a token, make sure that happens.
    LexResult result = LexSymbolToken(source_text, position);
    CARBON_CHECK(result) << "Failed to form a token!";
  }

  auto LexComment(llvm::StringRef source_text, ssize_t& position) -> void {
    CARBON_DCHECK(source_text.substr(position).startswith("//"));

    // Any comment must be the only non-whitespace on the line.
    const auto* line_info = current_line_info();
    if (LLVM_UNLIKELY(position != line_info->start + line_info->indent)) {
      CARBON_DIAGNOSTIC(TrailingComment, Error,
                        "Trailing comments are not permitted.");

      emitter_.Emit(source_text.begin() + position, TrailingComment);

      // Note that we cannot fall-through here as the logic below doesn't handle
      // trailing comments. For simplicity, we just consume the trailing comment
      // itself and let the normal lexer handle the newline as if there weren't
      // a comment at all.
      position = line_info->start + line_info->length;
      return;
    }

    // The introducer '//' must be followed by whitespace or EOF.
    bool is_valid_after_slashes = true;
    if (position + 2 < static_cast<ssize_t>(source_text.size()) &&
        LLVM_UNLIKELY(!IsSpace(source_text[position + 2]))) {
      CARBON_DIAGNOSTIC(NoWhitespaceAfterCommentIntroducer, Error,
                        "Whitespace is required after '//'.");
      emitter_.Emit(source_text.begin() + position + 2,
                    NoWhitespaceAfterCommentIntroducer);

      // We use this to tweak the lexing of blocks below.
      is_valid_after_slashes = false;
    }

    // Skip over this line.
    ssize_t line_index = line_index_;
    ++line_index;
    position = buffer_.line_infos_[line_index].start;

    // A very common pattern is a long block of comment lines all with the same
    // indent and comment start. We skip these comment blocks in bulk both for
    // speed and to reduce redundant diagnostics if each line has the same
    // erroneous comment start like `//!`.
    //
    // When we have SIMD support this is even more important for speed, as short
    // indents can be scanned extremely quickly with SIMD and we expect these to
    // be the dominant cases.
    //
    // TODO: We should extend this to 32-byte SIMD on platforms with support.
    constexpr int MaxIndent = 13;
    const int indent = line_info->indent;
    const ssize_t first_line_start = line_info->start;
    ssize_t prefix_size = indent + (is_valid_after_slashes ? 3 : 2);
    auto skip_to_next_line = [this, indent, &line_index, &position] {
      // We're guaranteed to have a line here even on a comment on the last line
      // as we ensure there is an empty line structure at the end of every file.
      ++line_index;
      auto* next_line_info = &buffer_.line_infos_[line_index];
      next_line_info->indent = indent;
      position = next_line_info->start;
    };
    if (CARBON_USE_SIMD &&
        position + 16 < static_cast<ssize_t>(source_text.size()) &&
        indent <= MaxIndent) {
#if __x86_64__
      // Load a mask based on the amount of text we want to compare.
      auto mask = prefix_masks[prefix_size];
      // And use the current line's prefix as the exemplar to compare against.
      // We don't mask here as we will mask when doing the comparison.
      auto prefix = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
          source_text.data() + first_line_start));
      do {
        // Load the next line to consider's prefix.
        auto next_prefix = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(source_text.data() + position));
        // Compute the difference between the next line and our exemplar. Again,
        // we don't mask the difference because the comparison below will be
        // masked.
        auto prefix_diff = _mm_xor_si128(prefix, next_prefix);
        // If we have any differences (non-zero bits) within the mask, we can't
        // skip the next line too.
        if (!_mm_test_all_zeros(mask, prefix_diff)) {
          break;
        }

        skip_to_next_line();
      } while (position + 16 < static_cast<ssize_t>(source_text.size()));
      // TODO: If we finish the loop due to the position approaching the end of
      // the buffer we may fail to skip the last line in a comment block that
      // has an invalid initial sequence and thus emit extra diagnostics. We
      // should really fall through to the generic skipping logic, but the code
      // organization will need to change significantly to allow that.
#elif CARBON_USE_SIMD
#error Unknown target for SIMD comment skipping.
#endif
    } else {
      while (position + prefix_size <
                 static_cast<ssize_t>(source_text.size()) &&
             memcmp(source_text.data() + first_line_start,
                    source_text.data() + position, prefix_size) == 0) {
        skip_to_next_line();
      }
    }

    // Now compute the indent of this next line before we finish.
    ssize_t line_start = position;
    SkipHorizontalWhitespace(source_text, position);

    // Now that we're done scanning, update to the latest line index and indent.
    line_index_ = line_index;
    current_line_info()->indent = position - line_start;
  }

  auto LexNumericLiteral(llvm::StringRef source_text, ssize_t& position)
      -> LexResult {
    std::optional<NumericLiteral> literal =
        NumericLiteral::Lex(source_text.substr(position));
    if (!literal) {
      return LexError(source_text, position);
    }

    int int_column = ComputeColumn(position);
    int token_size = literal->text().size();
    position += token_size;

    return VariantMatch(
        literal->ComputeValue(emitter_),
        [&](NumericLiteral::IntegerValue&& value) {
          auto token = buffer_.AddToken({.kind = TokenKind::IntegerLiteral,
                                         .token_line = current_line(),
                                         .column = int_column});
          buffer_.GetTokenInfo(token).literal_index =
              buffer_.literal_int_storage_.size();
          buffer_.literal_int_storage_.push_back(std::move(value.value));
          return token;
        },
        [&](NumericLiteral::RealValue&& value) {
          auto token = buffer_.AddToken({.kind = TokenKind::RealLiteral,
                                         .token_line = current_line(),
                                         .column = int_column});
          buffer_.GetTokenInfo(token).literal_index =
              buffer_.literal_int_storage_.size();
          buffer_.literal_int_storage_.push_back(std::move(value.mantissa));
          buffer_.literal_int_storage_.push_back(std::move(value.exponent));
          CARBON_CHECK(buffer_.GetRealLiteral(token).is_decimal ==
                       (value.radix == NumericLiteral::Radix::Decimal));
          return token;
        },
        [&](NumericLiteral::UnrecoverableError) {
          auto token = buffer_.AddToken({
              .kind = TokenKind::Error,
              .token_line = current_line(),
              .column = int_column,
              .error_length = token_size,
          });
          return token;
        });
  }

  auto LexStringLiteral(llvm::StringRef source_text, ssize_t& position)
      -> LexResult {
    std::optional<StringLiteral> literal =
        StringLiteral::Lex(source_text.substr(position));
    if (!literal) {
      return LexError(source_text, position);
    }

    Line string_line = current_line();
    int string_column = ComputeColumn(position);
    ssize_t literal_size = literal->text().size();
    position += literal_size;

    // Update line and column information.
    if (literal->is_multi_line()) {
      while (current_line_info()->start + current_line_info()->length <
             position) {
        ++line_index_;
        current_line_info()->indent = string_column;
      }
      // Note that we've updated the current line at this point, but
      // `set_indent_` is already true from above. That remains correct as the
      // last line of the multi-line literal *also* has its indent set.
    }

    if (literal->is_terminated()) {
      auto token =
          buffer_.AddToken({.kind = TokenKind::StringLiteral,
                            .token_line = string_line,
                            .column = string_column,
                            .literal_index = static_cast<int32_t>(
                                buffer_.literal_string_storage_.size())});
      buffer_.literal_string_storage_.push_back(
          literal->ComputeValue(emitter_));
      return token;
    } else {
      CARBON_DIAGNOSTIC(UnterminatedString, Error,
                        "String is missing a terminator.");
      emitter_.Emit(literal->text().begin(), UnterminatedString);
      return buffer_.AddToken(
          {.kind = TokenKind::Error,
           .token_line = string_line,
           .column = string_column,
           .error_length = static_cast<int32_t>(literal_size)});
    }
  }

  auto LexOneCharSymbolToken(llvm::StringRef source_text, TokenKind kind,
                             ssize_t& position) -> Token {
    // Verify in a debug build that the incoming token kind is correct.
    CARBON_DCHECK(kind != TokenKind::Error);
    CARBON_DCHECK(kind.fixed_spelling().size() == 1);
    CARBON_DCHECK(source_text[position] == kind.fixed_spelling().front())
        << "Source text starts with '" << source_text[position]
        << "' instead of the spelling '" << kind.fixed_spelling()
        << "' of the incoming token kind '" << kind << "'";

    Token token = buffer_.AddToken({.kind = kind,
                                    .token_line = current_line(),
                                    .column = ComputeColumn(position)});
    ++position;
    return token;
  }

  auto LexOpeningSymbolToken(llvm::StringRef source_text, TokenKind kind,
                             ssize_t& position) -> LexResult {
    Token token = LexOneCharSymbolToken(source_text, kind, position);
    open_groups_.push_back(token);
    return token;
  }

  auto LexClosingSymbolToken(llvm::StringRef source_text, TokenKind kind,
                             ssize_t& position) -> LexResult {
    auto unmatched_error = [&] {
      CARBON_DIAGNOSTIC(
          UnmatchedClosing, Error,
          "Closing symbol without a corresponding opening symbol.");
      emitter_.Emit(source_text.begin() + position, UnmatchedClosing);
      Token token = buffer_.AddToken({.kind = TokenKind::Error,
                                      .token_line = current_line(),
                                      .column = ComputeColumn(position),
                                      .error_length = 1});
      ++position;
      return token;
    };

    // If we have no open groups, this is an error.
    if (LLVM_UNLIKELY(open_groups_.empty())) {
      return unmatched_error();
    }

    Token opening_token = open_groups_.back();
    // Close any invalid open groups first.
    if (LLVM_UNLIKELY(buffer_.GetTokenInfo(opening_token).kind !=
                      kind.opening_symbol())) {
      CloseInvalidOpenGroups(kind, position);
      // This may exhaust the open groups so re-check and re-error if needed.
      if (open_groups_.empty()) {
        return unmatched_error();
      }
      opening_token = open_groups_.back();
      CARBON_DCHECK(buffer_.GetTokenInfo(opening_token).kind ==
                    kind.opening_symbol());
    }
    open_groups_.pop_back();

    // Now that the groups are all matched up, lex the actual token.
    Token token = LexOneCharSymbolToken(source_text, kind, position);

    // Note that it is important to get fresh token infos here as lexing the
    // open token would invalidate any pointers.
    buffer_.GetTokenInfo(opening_token).closing_token = token;
    buffer_.GetTokenInfo(token).opening_token = opening_token;

    return token;
  }

  auto LexSymbolToken(llvm::StringRef source_text, ssize_t& position)
      -> LexResult {
    // One character symbols and grouping symbols are handled with dedicated
    // dispatch. We only lex the multi-character tokens here.
    TokenKind kind = llvm::StringSwitch<TokenKind>(source_text.substr(position))
#define CARBON_SYMBOL_TOKEN(Name, Spelling) \
  .StartsWith(Spelling, TokenKind::Name)
#define CARBON_ONE_CHAR_SYMBOL_TOKEN(TokenName, Spelling)
#define CARBON_OPENING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, ClosingName)
#define CARBON_CLOSING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, OpeningName)
#include "toolchain/lex/token_kind.def"
                         .Default(TokenKind::Error);
    if (kind == TokenKind::Error) {
      return LexError(source_text, position);
    }

    Token token = buffer_.AddToken({.kind = kind,
                                    .token_line = current_line(),
                                    .column = ComputeColumn(position)});
    position += kind.fixed_spelling().size();
    return token;
  }

  // Given a word that has already been lexed, determine whether it is a type
  // literal and if so form the corresponding token.
  auto LexWordAsTypeLiteralToken(llvm::StringRef word, int column)
      -> LexResult {
    if (word.size() < 2) {
      // Too short to form one of these tokens.
      return LexResult::NoMatch();
    }
    if (word[1] < '1' || word[1] > '9') {
      // Doesn't start with a valid initial digit.
      return LexResult::NoMatch();
    }

    std::optional<TokenKind> kind;
    switch (word.front()) {
      case 'i':
        kind = TokenKind::IntegerTypeLiteral;
        break;
      case 'u':
        kind = TokenKind::UnsignedIntegerTypeLiteral;
        break;
      case 'f':
        kind = TokenKind::FloatingPointTypeLiteral;
        break;
      default:
        return LexResult::NoMatch();
    };

    llvm::StringRef suffix = word.substr(1);
    if (!CanLexInteger(emitter_, suffix)) {
      return buffer_.AddToken(
          {.kind = TokenKind::Error,
           .token_line = current_line(),
           .column = column,
           .error_length = static_cast<int32_t>(word.size())});
    }
    llvm::APInt suffix_value;
    if (suffix.getAsInteger(10, suffix_value)) {
      return LexResult::NoMatch();
    }

    auto token = buffer_.AddToken(
        {.kind = *kind, .token_line = current_line(), .column = column});
    buffer_.GetTokenInfo(token).literal_index =
        buffer_.literal_int_storage_.size();
    buffer_.literal_int_storage_.push_back(std::move(suffix_value));
    return token;
  }

  // Closes all open groups that cannot remain open across a closing symbol.
  // Users may pass `Error` to close all open groups.
  [[gnu::noinline]] auto CloseInvalidOpenGroups(TokenKind kind,
                                                ssize_t position) -> void {
    CARBON_CHECK(kind.is_closing_symbol() || kind == TokenKind::Error);
    CARBON_CHECK(!open_groups_.empty());

    int column = ComputeColumn(position);

    do {
      Token opening_token = open_groups_.back();
      TokenKind opening_kind = buffer_.GetTokenInfo(opening_token).kind;
      if (kind == opening_kind.closing_symbol()) {
        return;
      }

      open_groups_.pop_back();
      CARBON_DIAGNOSTIC(
          MismatchedClosing, Error,
          "Closing symbol does not match most recent opening symbol.");
      token_emitter_.Emit(opening_token, MismatchedClosing);

      CARBON_CHECK(!buffer_.tokens().empty())
          << "Must have a prior opening token!";
      Token prev_token = buffer_.tokens().end()[-1];

      // TODO: do a smarter backwards scan for where to put the closing
      // token.
      Token closing_token = buffer_.AddToken(
          {.kind = opening_kind.closing_symbol(),
           .has_trailing_space = buffer_.HasTrailingWhitespace(prev_token),
           .is_recovery = true,
           .token_line = current_line(),
           .column = column});
      TokenInfo& opening_token_info = buffer_.GetTokenInfo(opening_token);
      TokenInfo& closing_token_info = buffer_.GetTokenInfo(closing_token);
      opening_token_info.closing_token = closing_token;
      closing_token_info.opening_token = opening_token;
    } while (!open_groups_.empty());
  }

  auto GetOrCreateIdentifier(llvm::StringRef text) -> Identifier {
    auto insert_result = buffer_.identifier_map_.insert(
        {text, Identifier(buffer_.identifier_infos_.size())});
    if (insert_result.second) {
      buffer_.identifier_infos_.push_back({text});
    }
    return insert_result.first->second;
  }

  auto LexKeywordOrIdentifier(llvm::StringRef source_text, ssize_t& position)
      -> LexResult {
    if (static_cast<unsigned char>(source_text[position]) > 0x7F) {
      // TODO: Need to add support for Unicode lexing.
      return LexError(source_text, position);
    }
    CARBON_CHECK(IsAlpha(source_text[position]) ||
                 source_text[position] == '_');

    int column = ComputeColumn(position);

    // Take the valid characters off the front of the source buffer.
    llvm::StringRef identifier_text =
        ScanForIdentifierPrefix(source_text.substr(position));
    CARBON_CHECK(!identifier_text.empty())
        << "Must have at least one character!";
    position += identifier_text.size();

    // Check if the text is a type literal, and if so form such a literal.
    if (LexResult result = LexWordAsTypeLiteralToken(identifier_text, column)) {
      return result;
    }

    // Check if the text matches a keyword token, and if so use that.
    TokenKind kind = llvm::StringSwitch<TokenKind>(identifier_text)
#define CARBON_KEYWORD_TOKEN(Name, Spelling) .Case(Spelling, TokenKind::Name)
#include "toolchain/lex/token_kind.def"
                         .Default(TokenKind::Error);
    if (kind != TokenKind::Error) {
      return buffer_.AddToken(
          {.kind = kind, .token_line = current_line(), .column = column});
    }

    // Otherwise we have a generic identifier.
    return buffer_.AddToken({.kind = TokenKind::Identifier,
                             .token_line = current_line(),
                             .column = column,
                             .id = GetOrCreateIdentifier(identifier_text)});
  }

  auto LexError(llvm::StringRef source_text, ssize_t& position) -> LexResult {
    llvm::StringRef error_text =
        source_text.substr(position).take_while([](char c) {
          if (IsAlnum(c)) {
            return false;
          }
          switch (c) {
            case '_':
            case '\t':
            case '\n':
              return false;
            default:
              break;
          }
          return llvm::StringSwitch<bool>(llvm::StringRef(&c, 1))
#define CARBON_SYMBOL_TOKEN(Name, Spelling) .StartsWith(Spelling, false)
#include "toolchain/lex/token_kind.def"
              .Default(true);
        });
    if (error_text.empty()) {
      // TODO: Reimplement this to use the lexer properly. In the meantime,
      // guarantee that we eat at least one byte.
      error_text = source_text.substr(position, 1);
    }

    auto token = buffer_.AddToken(
        {.kind = TokenKind::Error,
         .token_line = current_line(),
         .column = ComputeColumn(position),
         .error_length = static_cast<int32_t>(error_text.size())});
    CARBON_DIAGNOSTIC(UnrecognizedCharacters, Error,
                      "Encountered unrecognized characters while parsing.");
    emitter_.Emit(error_text.begin(), UnrecognizedCharacters);

    position += error_text.size();
    return token;
  }

  auto LexStartOfFile(llvm::StringRef source_text, ssize_t& position) -> void {
    // Before lexing any source text, add the start-of-file token so that code
    // can assume a non-empty token buffer for the rest of lexing. Note that the
    // start-of-file always has trailing space because it *is* whitespace.
    buffer_.AddToken({.kind = TokenKind::StartOfFile,
                      .has_trailing_space = true,
                      .token_line = current_line(),
                      .column = 0});

    // Also skip any horizontal whitespace and record the indentation of the
    // first line.
    SkipHorizontalWhitespace(source_text, position);
    auto* line_info = current_line_info();
    CARBON_CHECK(line_info->start == 0);
    line_info->indent = position;
  }

  auto LexEndOfFile(llvm::StringRef source_text, ssize_t position) -> void {
    CARBON_CHECK(position == static_cast<ssize_t>(source_text.size()));
    // Check if the last line is empty and not the first line (and only). If so,
    // re-pin the last line to be the prior one so that diagnostics and editors
    // can treat newlines as terminators even though we internally handle them
    // as separators in case of a missing newline on the last line. We do this
    // here instead of detecting this when we see the newline to avoid more
    // conditions along that fast path.
    if (position == current_line_info()->start && line_index_ != 0) {
      --line_index_;
      --position;
    } else {
      // Update the line length as this is also the end of a line.
      current_line_info()->length = ComputeColumn(position);
    }

    // The end-of-file token is always considered to be whitespace.
    NoteWhitespace();

    // Close any open groups. We do this after marking whitespace, it will
    // preserve that.
    if (!open_groups_.empty()) {
      CloseInvalidOpenGroups(TokenKind::Error, position);
    }

    buffer_.AddToken({.kind = TokenKind::EndOfFile,
                      .token_line = current_line(),
                      .column = ComputeColumn(position)});
  }

  // We use a collection of static member functions for table-based dispatch to
  // lexer methods. These are named static member functions so that they show up
  // helpfully in profiles and backtraces, but they tend to not contain the
  // interesting logic and simply delegate to the relevant methods. All of their
  // signatures need to be exactly the same however in order to ensure we can
  // build efficient dispatch tables out of them. All of them end by doing a
  // must-tail return call to this routine. It handles continuing the dispatch
  // chain.
  static auto DispatchNext(Lexer& lexer, llvm::StringRef source_text,
                           ssize_t position) -> void {
    if (LLVM_LIKELY(position < static_cast<ssize_t>(source_text.size()))) {
      // The common case is to tail recurse based on the next character. Note
      // that because this is a must-tail return, this cannot fail to tail-call
      // and will not grow the stack. This is in essence a loop with dynamic
      // tail dispatch to the next stage of the loop.
      [[clang::musttail]] return DispatchTable[static_cast<unsigned char>(
          source_text[position])](lexer, source_text, position);
    }

    // When we finish the source text, stop recursing. We also hint this so that
    // the tail-dispatch is optimized as that's essentially the loop back-edge
    // and this is the loop exit.
    lexer.LexEndOfFile(source_text, position);
  }

  // Define a set of dispatch functions that simply forward to a method that
  // lexes a token. This includes validating that an actual token was produced,
  // and continuing the dispatch.
#define CARBON_DISPATCH_LEX_TOKEN(LexMethod)                                 \
  static auto Dispatch##LexMethod(Lexer& lexer, llvm::StringRef source_text, \
                                  ssize_t position)                          \
      ->void {                                                               \
    LexResult result = lexer.LexMethod(source_text, position);               \
    CARBON_CHECK(result) << "Failed to form a token!";                       \
    [[clang::musttail]] return DispatchNext(lexer, source_text, position);   \
  }
  CARBON_DISPATCH_LEX_TOKEN(LexError)
  CARBON_DISPATCH_LEX_TOKEN(LexSymbolToken)
  CARBON_DISPATCH_LEX_TOKEN(LexKeywordOrIdentifier)
  CARBON_DISPATCH_LEX_TOKEN(LexNumericLiteral)
  CARBON_DISPATCH_LEX_TOKEN(LexStringLiteral)

  // A custom dispatch functions that pre-select the symbol token to lex.
#define CARBON_DISPATCH_LEX_SYMBOL_TOKEN(LexMethod)                           \
  static auto Dispatch##LexMethod##SymbolToken(                               \
      Lexer& lexer, llvm::StringRef source_text, ssize_t position)            \
      ->void {                                                                \
    LexResult result = lexer.LexMethod##SymbolToken(                          \
        source_text, OneCharTokenKindTable[source_text[position]], position); \
    CARBON_CHECK(result) << "Failed to form a token!";                        \
    [[clang::musttail]] return DispatchNext(lexer, source_text, position);    \
  }
  CARBON_DISPATCH_LEX_SYMBOL_TOKEN(LexOneChar)
  CARBON_DISPATCH_LEX_SYMBOL_TOKEN(LexOpening)
  CARBON_DISPATCH_LEX_SYMBOL_TOKEN(LexClosing)

  // Define a set of non-token dispatch functions that handle things like
  // whitespace and comments.
#define CARBON_DISPATCH_LEX_NON_TOKEN(LexMethod)                             \
  static auto Dispatch##LexMethod(Lexer& lexer, llvm::StringRef source_text, \
                                  ssize_t position)                          \
      ->void {                                                               \
    lexer.LexMethod(source_text, position);                                  \
    [[clang::musttail]] return DispatchNext(lexer, source_text, position);   \
  }
  CARBON_DISPATCH_LEX_NON_TOKEN(LexHorizontalWhitespace)
  CARBON_DISPATCH_LEX_NON_TOKEN(LexVerticalWhitespace)
  CARBON_DISPATCH_LEX_NON_TOKEN(LexCommentOrSlash)

  // The main entry point for dispatching through the lexer's table. This method
  // should always fully consume the source text.
  auto Lex() && -> TokenizedBuffer {
    llvm::StringRef source_text = buffer_.source_->text();

    // First build up our line data structures.
    CreateLines(source_text);

    ssize_t position = 0;
    LexStartOfFile(source_text, position);

    // Manually enter the dispatch loop. This call will tail-recurse through the
    // dispatch table until everything from source_text is consumed.
    DispatchNext(*this, source_text, position);

    if (consumer_.seen_error()) {
      buffer_.has_errors_ = true;
    }

    return std::move(buffer_);
  }

 private:
  using DispatchFunctionT = auto(Lexer& lexer, llvm::StringRef source_text,
                                 ssize_t position) -> void;
  using DispatchTableT = std::array<DispatchFunctionT*, 256>;

  // Build a table of function pointers that we can use to dispatch to the
  // correct lexer routine based on the first byte of source text.
  //
  // While it is tempting to simply use a `switch` on the first byte and
  // dispatch with cases into this, in practice that doesn't produce great code.
  // There seem to be two issues that are the root cause.
  //
  // First, there are lots of different values of bytes that dispatch to a
  // fairly small set of routines, and then some byte values that dispatch
  // differently for each byte. This pattern isn't one that the compiler-based
  // lowering of switches works well with -- it tries to balance all the cases,
  // and in doing so emits several compares and other control flow rather than a
  // simple jump table.
  //
  // Second, with a `case`, it isn't as obvious how to create a single, uniform
  // interface that is effective for *every* byte value, and thus makes for a
  // single consistent table-based dispatch. By forcing these to be function
  // pointers, we also coerce the code to use a strictly homogeneous structure
  // that can form a single dispatch table.
  //
  // These two actually interact -- the second issue is part of what makes the
  // non-table lowering in the first one desirable for many switches and cases.
  //
  // Ultimately, when table-based dispatch is such an important technique, we
  // get better results by taking full control and manually creating the
  // dispatch structures.
  //
  // The functions in this table also use tail-recursion to implement the loop
  // of the lexer. This is based on the technique described more fully for any
  // kind of byte-stream loop structure here:
  // https://blog.reverberate.org/2021/04/21/musttail-efficient-interpreters.html
  constexpr static auto MakeDispatchTable() -> DispatchTableT {
    DispatchTableT table = {};
    // First set the table entries to dispatch to our error token handler as the
    // base case. Everything valid comes from an override below.
    for (int i = 0; i < 256; ++i) {
      table[i] = &DispatchLexError;
    }

    // Symbols have some special dispatching. First, set the first character of
    // each symbol token spelling to dispatch to the symbol lexer. We don't
    // provide a pre-computed token here, so the symbol lexer will compute the
    // exact symbol token kind. We'll override this with more specific dispatch
    // below.
#define CARBON_SYMBOL_TOKEN(TokenName, Spelling) \
  table[(Spelling)[0]] = &DispatchLexSymbolToken;
#include "toolchain/lex/token_kind.def"

    // Now special cased single-character symbols that are guaranteed to not
    // join with another symbol. These are grouping symbols, terminators,
    // or separators in the grammar and have a good reason to be
    // orthogonal to any other punctuation. We do this separately because this
    // needs to override some of the generic handling above, and provide a
    // custom token.
#define CARBON_ONE_CHAR_SYMBOL_TOKEN(TokenName, Spelling) \
  table[(Spelling)[0]] = &DispatchLexOneCharSymbolToken;
#define CARBON_OPENING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, ClosingName) \
  table[(Spelling)[0]] = &DispatchLexOpeningSymbolToken;
#define CARBON_CLOSING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, OpeningName) \
  table[(Spelling)[0]] = &DispatchLexClosingSymbolToken;
#include "toolchain/lex/token_kind.def"

    // Override the handling for `/` to consider comments as well as a `/`
    // symbol.
    table['/'] = &DispatchLexCommentOrSlash;

    table['_'] = &DispatchLexKeywordOrIdentifier;
    // Note that we don't use `llvm::seq` because this needs to be `constexpr`
    // evaluated.
    for (unsigned char c = 'a'; c <= 'z'; ++c) {
      table[c] = &DispatchLexKeywordOrIdentifier;
    }
    for (unsigned char c = 'A'; c <= 'Z'; ++c) {
      table[c] = &DispatchLexKeywordOrIdentifier;
    }
    // We dispatch all non-ASCII UTF-8 characters to the identifier lexing
    // as whitespace characters should already have been skipped and the
    // only remaining valid Unicode characters would be part of an
    // identifier. That code can either accept or reject.
    for (int i = 0x80; i < 0x100; ++i) {
      table[i] = &DispatchLexKeywordOrIdentifier;
    }

    for (unsigned char c = '0'; c <= '9'; ++c) {
      table[c] = &DispatchLexNumericLiteral;
    }

    table['\''] = &DispatchLexStringLiteral;
    table['"'] = &DispatchLexStringLiteral;
    table['#'] = &DispatchLexStringLiteral;

    table[' '] = &DispatchLexHorizontalWhitespace;
    table['\t'] = &DispatchLexHorizontalWhitespace;
    table['\n'] = &DispatchLexVerticalWhitespace;

    return table;
  };

  static const DispatchTableT DispatchTable;

  static const std::array<TokenKind, 256> OneCharTokenKindTable;

  TokenizedBuffer buffer_;

  ssize_t line_index_;

  llvm::SmallVector<Token> open_groups_;

  ErrorTrackingDiagnosticConsumer consumer_;

  SourceBufferLocationTranslator translator_;
  LexerDiagnosticEmitter emitter_;

  TokenLocationTranslator token_translator_;
  TokenDiagnosticEmitter token_emitter_;
};

constexpr TokenizedBuffer::Lexer::DispatchTableT
    TokenizedBuffer::Lexer::DispatchTable = MakeDispatchTable();

constexpr std::array<TokenKind, 256>
    TokenizedBuffer::Lexer::OneCharTokenKindTable = [] {
      std::array<TokenKind, 256> table = {};
#define CARBON_ONE_CHAR_SYMBOL_TOKEN(TokenName, Spelling) \
  table[(Spelling)[0]] = TokenKind::TokenName;
#define CARBON_OPENING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, ClosingName) \
  table[(Spelling)[0]] = TokenKind::TokenName;
#define CARBON_CLOSING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, OpeningName) \
  table[(Spelling)[0]] = TokenKind::TokenName;
#include "toolchain/lex/token_kind.def"
      return table;
    }();

auto TokenizedBuffer::Lex(SourceBuffer& source, DiagnosticConsumer& consumer)
    -> TokenizedBuffer {
  Lexer lexer(source, consumer);
  return std::move(lexer).Lex();
}

auto TokenizedBuffer::GetKind(Token token) const -> TokenKind {
  return GetTokenInfo(token).kind;
}

auto TokenizedBuffer::GetLine(Token token) const -> Line {
  return GetTokenInfo(token).token_line;
}

auto TokenizedBuffer::GetLineNumber(Token token) const -> int {
  return GetLineNumber(GetLine(token));
}

auto TokenizedBuffer::GetColumnNumber(Token token) const -> int {
  return GetTokenInfo(token).column + 1;
}

auto TokenizedBuffer::GetTokenText(Token token) const -> llvm::StringRef {
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
  if (token_info.kind == TokenKind::IntegerLiteral ||
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

  if (token_info.kind == TokenKind::StartOfFile ||
      token_info.kind == TokenKind::EndOfFile) {
    return llvm::StringRef();
  }

  CARBON_CHECK(token_info.kind == TokenKind::Identifier) << token_info.kind;
  return GetIdentifierText(token_info.id);
}

auto TokenizedBuffer::GetIdentifier(Token token) const -> Identifier {
  const auto& token_info = GetTokenInfo(token);
  CARBON_CHECK(token_info.kind == TokenKind::Identifier) << token_info.kind;
  return token_info.id;
}

auto TokenizedBuffer::GetIntegerLiteral(Token token) const
    -> const llvm::APInt& {
  const auto& token_info = GetTokenInfo(token);
  CARBON_CHECK(token_info.kind == TokenKind::IntegerLiteral) << token_info.kind;
  return literal_int_storage_[token_info.literal_index];
}

auto TokenizedBuffer::GetRealLiteral(Token token) const -> RealLiteralValue {
  const auto& token_info = GetTokenInfo(token);
  CARBON_CHECK(token_info.kind == TokenKind::RealLiteral) << token_info.kind;

  // Note that every real literal is at least three characters long, so we can
  // safely look at the second character to determine whether we have a
  // decimal or hexadecimal literal.
  const auto& line_info = GetLineInfo(token_info.token_line);
  int64_t token_start = line_info.start + token_info.column;
  char second_char = source_->text()[token_start + 1];
  bool is_decimal = second_char != 'x' && second_char != 'b';

  return {.mantissa = literal_int_storage_[token_info.literal_index],
          .exponent = literal_int_storage_[token_info.literal_index + 1],
          .is_decimal = is_decimal};
}

auto TokenizedBuffer::GetStringLiteral(Token token) const -> llvm::StringRef {
  const auto& token_info = GetTokenInfo(token);
  CARBON_CHECK(token_info.kind == TokenKind::StringLiteral) << token_info.kind;
  return literal_string_storage_[token_info.literal_index];
}

auto TokenizedBuffer::GetTypeLiteralSize(Token token) const
    -> const llvm::APInt& {
  const auto& token_info = GetTokenInfo(token);
  CARBON_CHECK(token_info.kind.is_sized_type_literal()) << token_info.kind;
  return literal_int_storage_[token_info.literal_index];
}

auto TokenizedBuffer::GetMatchedClosingToken(Token opening_token) const
    -> Token {
  const auto& opening_token_info = GetTokenInfo(opening_token);
  CARBON_CHECK(opening_token_info.kind.is_opening_symbol())
      << opening_token_info.kind;
  return opening_token_info.closing_token;
}

auto TokenizedBuffer::GetMatchedOpeningToken(Token closing_token) const
    -> Token {
  const auto& closing_token_info = GetTokenInfo(closing_token);
  CARBON_CHECK(closing_token_info.kind.is_closing_symbol())
      << closing_token_info.kind;
  return closing_token_info.opening_token;
}

auto TokenizedBuffer::HasLeadingWhitespace(Token token) const -> bool {
  auto it = TokenIterator(token);
  return it == tokens().begin() || GetTokenInfo(*(it - 1)).has_trailing_space;
}

auto TokenizedBuffer::HasTrailingWhitespace(Token token) const -> bool {
  return GetTokenInfo(token).has_trailing_space;
}

auto TokenizedBuffer::IsRecoveryToken(Token token) const -> bool {
  return GetTokenInfo(token).is_recovery;
}

auto TokenizedBuffer::GetLineNumber(Line line) const -> int {
  return line.index + 1;
}

auto TokenizedBuffer::GetNextLine(Line line) const -> Line {
  Line next(line.index + 1);
  CARBON_DCHECK(static_cast<size_t>(next.index) < line_infos_.size());
  return next;
}

auto TokenizedBuffer::GetPrevLine(Line line) const -> Line {
  CARBON_CHECK(line.index > 0);
  return Line(line.index - 1);
}

auto TokenizedBuffer::GetIndentColumnNumber(Line line) const -> int {
  return GetLineInfo(line).indent + 1;
}

auto TokenizedBuffer::GetIdentifierText(Identifier identifier) const
    -> llvm::StringRef {
  return identifier_infos_[identifier.index].text;
}

auto TokenizedBuffer::PrintWidths::Widen(const PrintWidths& widths) -> void {
  index = std::max(widths.index, index);
  kind = std::max(widths.kind, kind);
  column = std::max(widths.column, column);
  line = std::max(widths.line, line);
  indent = std::max(widths.indent, indent);
}

// Compute the printed width of a number. When numbers are printed in decimal,
// the number of digits needed is is one more than the log-base-10 of the
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

auto TokenizedBuffer::GetTokenPrintWidths(Token token) const -> PrintWidths {
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
  for (Token token : tokens()) {
    widths.Widen(GetTokenPrintWidths(token));
  }

  for (Token token : tokens()) {
    PrintToken(output_stream, token, widths);
    output_stream << "\n";
  }
  output_stream << "  ]\n";
}

auto TokenizedBuffer::PrintToken(llvm::raw_ostream& output_stream,
                                 Token token) const -> void {
  PrintToken(output_stream, token, {});
}

auto TokenizedBuffer::PrintToken(llvm::raw_ostream& output_stream, Token token,
                                 PrintWidths widths) const -> void {
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
    case TokenKind::IntegerLiteral:
      output_stream << ", value: `";
      GetIntegerLiteral(token).print(output_stream, /*isSigned=*/false);
      output_stream << "`";
      break;
    case TokenKind::RealLiteral:
      output_stream << ", value: `" << GetRealLiteral(token) << "`";
      break;
    case TokenKind::StringLiteral:
      output_stream << ", value: `" << GetStringLiteral(token) << "`";
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

auto TokenizedBuffer::GetLineInfo(Line line) -> LineInfo& {
  return line_infos_[line.index];
}

auto TokenizedBuffer::GetLineInfo(Line line) const -> const LineInfo& {
  return line_infos_[line.index];
}

auto TokenizedBuffer::AddLine(LineInfo info) -> Line {
  line_infos_.push_back(info);
  return Line(static_cast<int>(line_infos_.size()) - 1);
}

auto TokenizedBuffer::GetTokenInfo(Token token) -> TokenInfo& {
  return token_infos_[token.index];
}

auto TokenizedBuffer::GetTokenInfo(Token token) const -> const TokenInfo& {
  return token_infos_[token.index];
}

auto TokenizedBuffer::AddToken(TokenInfo info) -> Token {
  token_infos_.push_back(info);
  expected_parse_tree_size_ += info.kind.expected_parse_tree_size();
  return Token(static_cast<int>(token_infos_.size()) - 1);
}

auto TokenIterator::Print(llvm::raw_ostream& output) const -> void {
  output << token_.index;
}

auto TokenizedBuffer::SourceBufferLocationTranslator::GetLocation(
    const char* loc) -> DiagnosticLocation {
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

  return {.file_name = buffer_->source_->filename(),
          .line = line,
          .line_number = line_number + 1,
          .column_number = column_number + 1};
}

auto TokenLocationTranslator::GetLocation(Token token) -> DiagnosticLocation {
  // Map the token location into a position within the source buffer.
  const auto& token_info = buffer_->GetTokenInfo(token);
  const auto& line_info = buffer_->GetLineInfo(token_info.token_line);
  const char* token_start =
      buffer_->source_->text().begin() + line_info.start + token_info.column;

  // Find the corresponding file location.
  // TODO: Should we somehow indicate in the diagnostic location if this token
  // is a recovery token that doesn't correspond to the original source?
  return TokenizedBuffer::SourceBufferLocationTranslator(buffer_).GetLocation(
      token_start);
}

}  // namespace Carbon::Lex
