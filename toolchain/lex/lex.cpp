// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/lex.h"

#include <array>
#include <limits>

#include "common/check.h"
#include "common/variant_helpers.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Compiler.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/lex/character_set.h"
#include "toolchain/lex/helpers.h"
#include "toolchain/lex/numeric_literal.h"
#include "toolchain/lex/string_literal.h"
#include "toolchain/lex/token_index.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/lex/tokenized_buffer.h"

#if __ARM_NEON
#include <arm_neon.h>
#define CARBON_USE_SIMD 1
#elif __x86_64__
#include <x86intrin.h>
#define CARBON_USE_SIMD 1
#else
#define CARBON_USE_SIMD 0
#endif

namespace Carbon::Lex {

// Implementation of the lexer logic itself.
//
// The design is that lexing can loop over the source buffer, consuming it into
// tokens by calling into this API. This class handles the state and breaks down
// the different lexing steps that may be used. It directly updates the provided
// tokenized buffer with the lexed tokens.
//
// We'd typically put this in an anonymous namespace, but it is `friend`-ed by
// the `TokenizedBuffer`. One of the important benefits of being in an anonymous
// namespace is having internal linkage. That allows the optimizer to much more
// aggressively inline away functions that are called in only one place. We keep
// that benefit for now by using the `internal_linkage` attribute.
//
// TODO: Investigate ways to refactor the code that allow moving this into an
// anonymous namespace without overly exposing implementation details of the
// `TokenizedBuffer` or undermining the performance constraints of the lexer.
class [[clang::internal_linkage]] Lexer {
 public:
  using TokenInfo = TokenizedBuffer::TokenInfo;
  using LineInfo = TokenizedBuffer::LineInfo;

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
    LexResult(TokenIndex /*discarded_token*/) : LexResult(true) {}

    // Returns a result indicating no token was produced.
    static auto NoMatch() -> LexResult { return LexResult(false); }

    // Tests whether a token was produced by the lexing routine, and
    // the lexer can continue forming tokens.
    explicit operator bool() const { return formed_token_; }

   private:
    explicit LexResult(bool formed_token) : formed_token_(formed_token) {}

    bool formed_token_;
  };

  Lexer(SharedValueStores& value_stores, SourceBuffer& source,
        DiagnosticConsumer& consumer)
      : buffer_(value_stores, source),
        consumer_(consumer),
        emitter_(&consumer_, &buffer_),
        token_emitter_(&consumer_, &buffer_) {}

  // Find all line endings and create the line data structures.
  //
  // Explicitly kept out-of-line because this is a significant loop that is
  // useful to have in the profile and it doesn't simplify by inlining at all.
  // But because it can, the compiler will flatten this otherwise.
  [[gnu::noinline]] auto MakeLines(llvm::StringRef source_text) -> void;

  auto current_line() -> LineIndex { return LineIndex(line_index_); }

  auto current_line_info() -> LineInfo* {
    return &buffer_.line_infos_[line_index_];
  }

  auto next_line() -> LineIndex { return LineIndex(line_index_ + 1); }

  auto next_line_info() -> LineInfo* {
    CARBON_CHECK(line_index_ + 1 <
                 static_cast<ssize_t>(buffer_.line_infos_.size()));
    return &buffer_.line_infos_[line_index_ + 1];
  }

  // Note when the lexer has encountered whitespace, and the next lexed token
  // should reflect that it was preceded by some amount of whitespace.
  auto NoteWhitespace() -> void { has_leading_space_ = true; }

  // Add a lexed token to the tokenized buffer, and reset any token-specific
  // state tracked in the lexer for the next token.
  auto AddLexedToken(TokenInfo info) -> TokenIndex {
    has_leading_space_ = false;
    return buffer_.AddToken(info);
  }

  // Lexes a token with no payload: builds the correctly encoded token info,
  // adds it to the tokenized buffer and returns the token index.
  auto LexToken(TokenKind kind, int32_t byte_offset) -> TokenIndex {
    // Check that we don't accidentally call this for one of the token kinds
    // that *always* has a payload up front.
    CARBON_DCHECK(!kind.IsOneOf(
        {TokenKind::Identifier, TokenKind::StringLiteral, TokenKind::IntLiteral,
         TokenKind::IntTypeLiteral, TokenKind::UnsignedIntTypeLiteral,
         TokenKind::FloatTypeLiteral, TokenKind::RealLiteral,
         TokenKind::Error}));
    return AddLexedToken(TokenInfo(kind, has_leading_space_, byte_offset));
  }

  // Lexes a token with a payload: builds the correctly encoded token info,
  // adds it to the tokenized buffer and returns the token index.
  auto LexTokenWithPayload(TokenKind kind, int token_payload,
                           int32_t byte_offset) -> TokenIndex {
    return AddLexedToken(
        TokenInfo(kind, has_leading_space_, token_payload, byte_offset));
  }

  auto SkipHorizontalWhitespace(llvm::StringRef source_text, ssize_t& position)
      -> void;

  auto LexHorizontalWhitespace(llvm::StringRef source_text, ssize_t& position)
      -> void;

  auto LexVerticalWhitespace(llvm::StringRef source_text, ssize_t& position)
      -> void;

  auto LexCR(llvm::StringRef source_text, ssize_t& position) -> void;

  auto LexCommentOrSlash(llvm::StringRef source_text, ssize_t& position)
      -> void;

  auto LexComment(llvm::StringRef source_text, ssize_t& position) -> void;

  // Determines whether a real literal can be formed at the current location.
  // This is the case unless the preceding token is `.` or `->` and there is no
  // intervening whitespace.
  auto CanFormRealLiteral() -> bool;

  auto LexNumericLiteral(llvm::StringRef source_text, ssize_t& position)
      -> LexResult;

  auto LexStringLiteral(llvm::StringRef source_text, ssize_t& position)
      -> LexResult;

  auto LexOneCharSymbolToken(llvm::StringRef source_text, TokenKind kind,
                             ssize_t& position) -> TokenIndex;

  auto LexOpeningSymbolToken(llvm::StringRef source_text, TokenKind kind,
                             ssize_t& position) -> LexResult;

  auto LexClosingSymbolToken(llvm::StringRef source_text, TokenKind kind,
                             ssize_t& position) -> LexResult;

  auto LexSymbolToken(llvm::StringRef source_text, ssize_t& position)
      -> LexResult;

  // Given a word that has already been lexed, determine whether it is a type
  // literal and if so form the corresponding token.
  auto LexWordAsTypeLiteralToken(llvm::StringRef word, int32_t byte_offset)
      -> LexResult;

  auto LexKeywordOrIdentifier(llvm::StringRef source_text, ssize_t& position)
      -> LexResult;

  auto LexHash(llvm::StringRef source_text, ssize_t& position) -> LexResult;

  auto LexError(llvm::StringRef source_text, ssize_t& position) -> LexResult;

  auto LexFileStart(llvm::StringRef source_text, ssize_t& position) -> void;

  auto LexFileEnd(llvm::StringRef source_text, ssize_t position) -> void;

  // Perform final checking and cleanup that should be done once we have
  // finished lexing the whole file, and before we consider the tokenized buffer
  // to be complete.
  auto Finalize() -> void;

  auto DiagnoseAndFixMismatchedBrackets() -> void;

  // The main entry point for dispatching through the lexer's table. This method
  // should always fully consume the source text.
  auto Lex() && -> TokenizedBuffer;

 private:
  class ErrorRecoveryBuffer;

  TokenizedBuffer buffer_;

  ssize_t line_index_;

  // Tracks whether the lexer has encountered whitespace that will be leading
  // whitespace for the next lexed token. Reset after each token lexed.
  bool has_leading_space_ = false;

  llvm::SmallVector<TokenIndex> open_groups_;
  bool has_mismatched_brackets_ = false;

  ErrorTrackingDiagnosticConsumer consumer_;

  TokenizedBuffer::SourcePointerDiagnosticEmitter emitter_;

  TokenizedBuffer::TokenDiagnosticEmitter token_emitter_;
};

#if CARBON_USE_SIMD
namespace {
#if __ARM_NEON
using SIMDMaskT = uint8x16_t;
#elif __x86_64__
using SIMDMaskT = __m128i;
#else
#error "Unsupported SIMD architecture!"
#endif
using SIMDMaskArrayT = std::array<SIMDMaskT, sizeof(SIMDMaskT) + 1>;
}  // namespace
// A table of masks to include 0-16 bytes of an SSE register.
static constexpr SIMDMaskArrayT PrefixMasks = []() constexpr {
  SIMDMaskArrayT masks = {};
  for (int i = 1; i < static_cast<int>(masks.size()); ++i) {
    masks[i] =
        // The SIMD types and constexpr require a C-style cast.
        // NOLINTNEXTLINE(google-readability-casting)
        (SIMDMaskT)(std::numeric_limits<unsigned __int128>::max() >>
                    ((sizeof(SIMDMaskT) - i) * 8));
  }
  return masks;
}();
#endif  // CARBON_USE_SIMD

// A table of booleans that we can use to classify bytes as being valid
// identifier start. This is used by raw identifier detection.
static constexpr std::array<bool, 256> IsIdStartByteTable = [] {
  std::array<bool, 256> table = {};
  for (char c = 'A'; c <= 'Z'; ++c) {
    table[c] = true;
  }
  for (char c = 'a'; c <= 'z'; ++c) {
    table[c] = true;
  }
  table['_'] = true;
  return table;
}();

// A table of booleans that we can use to classify bytes as being valid
// identifier (or keyword) characters. This is used in the generic,
// non-vectorized fallback code to scan for length of an identifier.
static constexpr std::array<bool, 256> IsIdByteTable = [] {
  std::array<bool, 256> table = IsIdStartByteTable;
  for (char c = '0'; c <= '9'; ++c) {
    table[c] = true;
  }
  return table;
}();

// Baseline scalar version, also available for scalar-fallback in SIMD code.
// Uses `ssize_t` for performance when indexing in the loop.
//
// TODO: This assumes all Unicode characters are non-identifiers.
static auto ScanForIdentifierPrefixScalar(llvm::StringRef text, ssize_t i)
    -> llvm::StringRef {
  const ssize_t size = text.size();
  while (i < size && IsIdByteTable[static_cast<unsigned char>(text[i])]) {
    ++i;
  }

  return text.substr(0, i);
}

#if CARBON_USE_SIMD && __x86_64__
// The SIMD code paths uses a scheme derived from the techniques in Geoff
// Langdale and Daniel Lemire's work on parsing JSON[1]. Specifically, that
// paper outlines a technique of using two 4-bit indexed in-register look-up
// tables (LUTs) to classify bytes in a branchless SIMD code sequence.
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
// Bits 4-7 remain unused if we need to classify more characters.
namespace {
// Struct used to implement the nibble LUT for SIMD implementations.
//
// Forced to 16-byte alignment to ensure we can load it easily in SIMD code.
struct alignas(16) NibbleLUT {
  auto Load() const -> __m128i {
    return _mm_load_si128(reinterpret_cast<const __m128i*>(this));
  }

  uint8_t nibble_0;
  uint8_t nibble_1;
  uint8_t nibble_2;
  uint8_t nibble_3;
  uint8_t nibble_4;
  uint8_t nibble_5;
  uint8_t nibble_6;
  uint8_t nibble_7;
  uint8_t nibble_8;
  uint8_t nibble_9;
  uint8_t nibble_a;
  uint8_t nibble_b;
  uint8_t nibble_c;
  uint8_t nibble_d;
  uint8_t nibble_e;
  uint8_t nibble_f;
};
}  // namespace

static constexpr NibbleLUT HighLUT = {
    .nibble_0 = 0b0000'0000,
    .nibble_1 = 0b0000'0000,
    .nibble_2 = 0b0000'0000,
    .nibble_3 = 0b0000'0010,
    .nibble_4 = 0b0000'0100,
    .nibble_5 = 0b0000'1001,
    .nibble_6 = 0b0000'0100,
    .nibble_7 = 0b0000'1000,
    .nibble_8 = 0b1000'0000,
    .nibble_9 = 0b1000'0000,
    .nibble_a = 0b1000'0000,
    .nibble_b = 0b1000'0000,
    .nibble_c = 0b1000'0000,
    .nibble_d = 0b1000'0000,
    .nibble_e = 0b1000'0000,
    .nibble_f = 0b1000'0000,
};
static constexpr NibbleLUT LowLUT = {
    .nibble_0 = 0b1000'1010,
    .nibble_1 = 0b1000'1110,
    .nibble_2 = 0b1000'1110,
    .nibble_3 = 0b1000'1110,
    .nibble_4 = 0b1000'1110,
    .nibble_5 = 0b1000'1110,
    .nibble_6 = 0b1000'1110,
    .nibble_7 = 0b1000'1110,
    .nibble_8 = 0b1000'1110,
    .nibble_9 = 0b1000'1110,
    .nibble_a = 0b1000'1100,
    .nibble_b = 0b1000'0100,
    .nibble_c = 0b1000'0100,
    .nibble_d = 0b1000'0100,
    .nibble_e = 0b1000'0100,
    .nibble_f = 0b1000'0101,
};

static auto ScanForIdentifierPrefixX86(llvm::StringRef text)
    -> llvm::StringRef {
  const auto high_lut = HighLUT.Load();
  const auto low_lut = LowLUT.Load();

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

  return ScanForIdentifierPrefixScalar(text, i);
}

#endif  // CARBON_USE_SIMD && __x86_64__

// Scans the provided text and returns the prefix `StringRef` of contiguous
// identifier characters.
//
// This is a performance sensitive function and where profitable uses vectorized
// code sequences to optimize its scanning. When modifying, the identifier
// lexing benchmarks should be checked for regressions.
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
  // Dispatch to an optimized architecture optimized routine.
#if CARBON_USE_SIMD && __x86_64__
  return ScanForIdentifierPrefixX86(text);
#elif CARBON_USE_SIMD && __ARM_NEON
  // Somewhat surprisingly, there is basically nothing worth doing in SIMD on
  // Arm to optimize this scan. The Neon SIMD operations end up requiring you to
  // move from the SIMD unit to the scalar unit in the critical path of finding
  // the offset of the end of an identifier. Current ARM cores make the code
  // sequences here (quite) unpleasant. For example, on Apple M1 and similar
  // cores, the latency is as much as 10 cycles just to extract from the vector.
  // SIMD might be more interesting on Neoverse cores, but it'd be nice to avoid
  // core-specific tunings at this point.
  //
  // If this proves problematic and critical to optimize, the current leading
  // theory is to have the newline searching code also create a bitmask for the
  // entire source file of identifier and non-identifier bytes, and then use the
  // bit-counting instructions here to do a fast scan of that bitmask. However,
  // crossing that bridge will add substantial complexity to the newline
  // scanner, and so currently we just use a boring scalar loop that pipelines
  // well.
#endif
  return ScanForIdentifierPrefixScalar(text, 0);
}

using DispatchFunctionT = auto(Lexer& lexer, llvm::StringRef source_text,
                               ssize_t position) -> void;
using DispatchTableT = std::array<DispatchFunctionT*, 256>;

static constexpr std::array<TokenKind, 256> OneCharTokenKindTable = [] {
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

// We use a collection of static member functions for table-based dispatch to
// lexer methods. These are named static member functions so that they show up
// helpfully in profiles and backtraces, but they tend to not contain the
// interesting logic and simply delegate to the relevant methods. All of their
// signatures need to be exactly the same however in order to ensure we can
// build efficient dispatch tables out of them. All of them end by doing a
// must-tail return call to this routine. It handles continuing the dispatch
// chain.
static auto DispatchNext(Lexer& lexer, llvm::StringRef source_text,
                         ssize_t position) -> void;

// Define a set of dispatch functions that simply forward to a method that
// lexes a token. This includes validating that an actual token was produced,
// and continuing the dispatch.
#define CARBON_DISPATCH_LEX_TOKEN(LexMethod)                                 \
  static auto Dispatch##LexMethod(Lexer& lexer, llvm::StringRef source_text, \
                                  ssize_t position) -> void {                \
    Lexer::LexResult result = lexer.LexMethod(source_text, position);        \
    CARBON_CHECK(result, "Failed to form a token!");                         \
    [[clang::musttail]] return DispatchNext(lexer, source_text, position);   \
  }
CARBON_DISPATCH_LEX_TOKEN(LexError)
CARBON_DISPATCH_LEX_TOKEN(LexSymbolToken)
CARBON_DISPATCH_LEX_TOKEN(LexKeywordOrIdentifier)
CARBON_DISPATCH_LEX_TOKEN(LexHash)
CARBON_DISPATCH_LEX_TOKEN(LexNumericLiteral)
CARBON_DISPATCH_LEX_TOKEN(LexStringLiteral)

// A set of custom dispatch functions that pre-select the symbol token to lex.
#define CARBON_DISPATCH_LEX_SYMBOL_TOKEN(LexMethod)                          \
  static auto Dispatch##LexMethod##SymbolToken(                              \
      Lexer& lexer, llvm::StringRef source_text, ssize_t position) -> void { \
    Lexer::LexResult result = lexer.LexMethod##SymbolToken(                  \
        source_text,                                                         \
        OneCharTokenKindTable[static_cast<unsigned char>(                    \
            source_text[position])],                                         \
        position);                                                           \
    CARBON_CHECK(result, "Failed to form a token!");                         \
    [[clang::musttail]] return DispatchNext(lexer, source_text, position);   \
  }
CARBON_DISPATCH_LEX_SYMBOL_TOKEN(LexOneChar)
CARBON_DISPATCH_LEX_SYMBOL_TOKEN(LexOpening)
CARBON_DISPATCH_LEX_SYMBOL_TOKEN(LexClosing)

// Define a set of non-token dispatch functions that handle things like
// whitespace and comments.
#define CARBON_DISPATCH_LEX_NON_TOKEN(LexMethod)                             \
  static auto Dispatch##LexMethod(Lexer& lexer, llvm::StringRef source_text, \
                                  ssize_t position) -> void {                \
    lexer.LexMethod(source_text, position);                                  \
    [[clang::musttail]] return DispatchNext(lexer, source_text, position);   \
  }
CARBON_DISPATCH_LEX_NON_TOKEN(LexHorizontalWhitespace)
CARBON_DISPATCH_LEX_NON_TOKEN(LexVerticalWhitespace)
CARBON_DISPATCH_LEX_NON_TOKEN(LexCR)
CARBON_DISPATCH_LEX_NON_TOKEN(LexCommentOrSlash)

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
static constexpr auto MakeDispatchTable() -> DispatchTableT {
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
  table['#'] = &DispatchLexHash;

  table[' '] = &DispatchLexHorizontalWhitespace;
  table['\t'] = &DispatchLexHorizontalWhitespace;
  table['\n'] = &DispatchLexVerticalWhitespace;
  table['\r'] = &DispatchLexCR;

  return table;
}

static constexpr DispatchTableT DispatchTable = MakeDispatchTable();

static auto DispatchNext(Lexer& lexer, llvm::StringRef source_text,
                         ssize_t position) -> void {
  if (LLVM_LIKELY(position < static_cast<ssize_t>(source_text.size()))) {
    // The common case is to tail recurse based on the next character. Note
    // that because this is a must-tail return, this cannot fail to tail-call
    // and will not grow the stack. This is in essence a loop with dynamic
    // tail dispatch to the next stage of the loop.
    // NOLINTNEXTLINE(readability-avoid-return-with-void-value): For musttail.
    [[clang::musttail]] return DispatchTable[static_cast<unsigned char>(
        source_text[position])](lexer, source_text, position);
  }

  // When we finish the source text, stop recursing. We also hint this so that
  // the tail-dispatch is optimized as that's essentially the loop back-edge
  // and this is the loop exit.
  lexer.LexFileEnd(source_text, position);
}

// Estimate an upper bound on the number of identifiers we will need to lex.
//
// When analyzing both Carbon and LLVM's C++ code, we have found a roughly
// normal distribution of unique identifiers in the file centered at 0.5 *
// lines, and in the vast majority of cases bounded below 1.0 * lines. For
// example, here is LLVM's distribution computed with `scripts/source_stats.py`
// and rendered in an ASCII-art histogram:
//
//   ## Unique IDs per 10 lines ## (median: 5, p90: 8, p95: 9, p99: 14)
//   1 ids   [  29]  ▍
//   2 ids   [ 282]  ███▊
//   3 ids   [1492]  ███████████████████▉
//   4 ids   [2674]  ███████████████████████████████████▌
//   5 ids   [3011]  ████████████████████████████████████████
//   6 ids   [2267]  ██████████████████████████████▏
//   7 ids   [1549]  ████████████████████▋
//   8 ids   [ 817]  ██████████▉
//   9 ids   [ 301]  ████
//   10 ids  [  98]  █▎
//
//   (Trimmed to only cover 1 - 10 unique IDs per 10 lines of code, 272 files
//    with more unique IDs in the tail.)
//
// We have checked this distribution with several large codebases (currently
// those at Google, happy to cross check with others) that use a similar coding
// style, and it appears to be very consistent. However, we suspect it may be
// dependent on the column width style. Currently, Carbon's toolchain style
// specifies 80-columns, but if we expect the lexer to routinely see files in
// different styles we should re-compute this estimate.
static auto EstimateUpperBoundOnNumIdentifiers(int line_count) -> int {
  return line_count;
}

auto Lexer::Lex() && -> TokenizedBuffer {
  llvm::StringRef source_text = buffer_.source_->text();

  // Enforced by the source buffer, but something we heavily rely on throughout
  // the lexer.
  CARBON_CHECK(source_text.size() < std::numeric_limits<int32_t>::max());

  // First build up our line data structures.
  MakeLines(source_text);

  // Use the line count (and any other info needed from this scan) to make rough
  // estimated reservations of memory in the hot data structures used by the
  // lexer. In practice, scanning for lines is one of the easiest parts of the
  // lexer to accelerate, and we can use its results to minimize the cost of
  // incrementally growing data structures during the hot path of the lexer.
  //
  // Note that for hashtables we want estimates near the upper bound to minimize
  // growth across the vast majority of inputs. They will also typically reserve
  // more memory than we request due to load factor and rounding to power-of-two
  // size. This overshoot is usually fine for hot parts of the lexer where
  // latency is expected to be more important than minimizing memory usage.
  buffer_.value_stores_->identifiers().Reserve(
      EstimateUpperBoundOnNumIdentifiers(buffer_.line_infos_.size()));

  ssize_t position = 0;
  LexFileStart(source_text, position);

  // Manually enter the dispatch loop. This call will tail-recurse through the
  // dispatch table until everything from source_text is consumed.
  DispatchNext(*this, source_text, position);

  Finalize();

  if (consumer_.seen_error()) {
    buffer_.has_errors_ = true;
  }

  return std::move(buffer_);
}

auto Lexer::MakeLines(llvm::StringRef source_text) -> void {
  // We currently use `memchr` here which typically is well optimized to use
  // SIMD or other significantly faster than byte-wise scanning. We also use
  // carefully selected variables and the `ssize_t` type for performance and
  // code size of this hot loop.
  //
  // Note that the `memchr` approach here works equally well for LF and CR+LF
  // line endings. Either way, it finds the end of the line and the start of the
  // next line. The lexer below will find the CR byte and peek to see the
  // following LF and jump to the next line correctly. However, this approach
  // does *not* support plain CR or LF+CR line endings. Nor does it support
  // vertical tab or other vertical whitespace.
  //
  // TODO: Eventually, we should extend this to have correct fallback support
  // for handling CR, LF+CR, vertical tab, and other esoteric vertical
  // whitespace as line endings. Notably, including *mixtures* of them. This
  // will likely be somewhat tricky as even detecting their absence without
  // performance overhead and without a custom scanner here rather than memchr
  // is likely to be difficult.
  const char* const text = source_text.data();
  const ssize_t size = source_text.size();
  ssize_t start = 0;
  while (const char* nl = reinterpret_cast<const char*>(
             memchr(&text[start], '\n', size - start))) {
    ssize_t nl_index = nl - text;
    buffer_.AddLine(TokenizedBuffer::LineInfo(start));
    start = nl_index + 1;
  }
  // The last line ends at the end of the file.
  buffer_.AddLine(TokenizedBuffer::LineInfo(start));

  // If the last line wasn't empty, the file ends with an unterminated line.
  // Add an extra blank line so that we never need to handle the special case
  // of being on the last line inside the lexer and needing to not increment
  // to the next line.
  if (start != size) {
    buffer_.AddLine(TokenizedBuffer::LineInfo(size));
  }

  // Now that all the infos are allocated, get a fresh pointer to the first
  // info for use while lexing.
  line_index_ = 0;
}

auto Lexer::SkipHorizontalWhitespace(llvm::StringRef source_text,
                                     ssize_t& position) -> void {
  // Handle adjacent whitespace quickly. This comes up frequently for example
  // due to indentation. We don't expect *huge* runs, so just use a scalar
  // loop. While still scalar, this avoids repeated table dispatch and marking
  // whitespace.
  while (position < static_cast<ssize_t>(source_text.size()) &&
         (source_text[position] == ' ' || source_text[position] == '\t')) {
    ++position;
  }
}

auto Lexer::LexHorizontalWhitespace(llvm::StringRef source_text,
                                    ssize_t& position) -> void {
  CARBON_DCHECK(source_text[position] == ' ' || source_text[position] == '\t');
  NoteWhitespace();
  // Skip runs using an optimized code path.
  SkipHorizontalWhitespace(source_text, position);
}

auto Lexer::LexVerticalWhitespace(llvm::StringRef source_text,
                                  ssize_t& position) -> void {
  NoteWhitespace();
  ++line_index_;
  auto* line_info = current_line_info();
  ssize_t line_start = line_info->start;
  position = line_start;
  SkipHorizontalWhitespace(source_text, position);
  line_info->indent = position - line_start;
}

auto Lexer::LexCR(llvm::StringRef source_text, ssize_t& position) -> void {
  if (LLVM_LIKELY((position + 1) < static_cast<ssize_t>(source_text.size())) &&
      LLVM_LIKELY(source_text[position + 1] == '\n')) {
    // Skip to the vertical whitespace path, it will skip over both CR and LF.
    LexVerticalWhitespace(source_text, position);
    return;
  }

  CARBON_DIAGNOSTIC(UnsupportedLFCRLineEnding, Error,
                    "the LF+CR line ending is not supported, only LF and CR+LF "
                    "are supported");
  CARBON_DIAGNOSTIC(UnsupportedCRLineEnding, Error,
                    "a raw CR line ending is not supported, only LF and CR+LF "
                    "are supported");
  bool is_lfcr = position > 0 && source_text[position - 1] == '\n';
  // TODO: This diagnostic has an unfortunate snippet -- we should tweak the
  // snippet rendering to gracefully handle CRs.
  emitter_.Emit(source_text.begin() + position,
                is_lfcr ? UnsupportedLFCRLineEnding : UnsupportedCRLineEnding);

  // Recover by treating the CR as a horizontal whitespace. This should make our
  // whitespace rules largely work and parse cleanly without disrupting the line
  // tracking data structures that were pre-built.
  NoteWhitespace();
  ++position;
}

auto Lexer::LexCommentOrSlash(llvm::StringRef source_text, ssize_t& position)
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
  CARBON_CHECK(result, "Failed to form a token!");
}

auto Lexer::LexComment(llvm::StringRef source_text, ssize_t& position) -> void {
  CARBON_DCHECK(source_text.substr(position).starts_with("//"));
  int32_t comment_start = position;

  // Any comment must be the only non-whitespace on the line.
  const auto* line_info = current_line_info();
  if (LLVM_UNLIKELY(position != line_info->start + line_info->indent)) {
    CARBON_DIAGNOSTIC(TrailingComment, Error,
                      "trailing comments are not permitted");

    emitter_.Emit(source_text.begin() + position, TrailingComment);

    // Note that we cannot fall-through here as the logic below doesn't handle
    // trailing comments. Instead, we treat trailing comments as vertical
    // whitespace, which already is designed to skip over any erroneous text at
    // the end of the line.
    LexVerticalWhitespace(source_text, position);
    buffer_.AddComment(line_info->indent, comment_start, position);
    return;
  }

  // The introducer '//' must be followed by whitespace or EOF.
  bool is_valid_after_slashes = true;
  if (position + 2 < static_cast<ssize_t>(source_text.size()) &&
      LLVM_UNLIKELY(!IsSpace(source_text[position + 2]))) {
    CARBON_DIAGNOSTIC(NoWhitespaceAfterCommentIntroducer, Error,
                      "whitespace is required after '//'");
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
    // Load a mask based on the amount of text we want to compare.
    auto mask = PrefixMasks[prefix_size];
#if __ARM_NEON
    // Load and mask the prefix of the current line.
    auto prefix = vld1q_u8(reinterpret_cast<const uint8_t*>(source_text.data() +
                                                            first_line_start));
    prefix = vandq_u8(mask, prefix);
    do {
      // Load and mask the next line to consider's prefix.
      auto next_prefix = vld1q_u8(
          reinterpret_cast<const uint8_t*>(source_text.data() + position));
      next_prefix = vandq_u8(mask, next_prefix);
      // Compare the two prefixes and if any lanes differ, break.
      auto compare = vceqq_u8(prefix, next_prefix);
      if (vminvq_u8(compare) == 0) {
        break;
      }

      skip_to_next_line();
    } while (position + 16 < static_cast<ssize_t>(source_text.size()));
#elif __x86_64__
    // Use the current line's prefix as the exemplar to compare against.
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
#else
#error "Unsupported SIMD architecture!"
#endif
    // TODO: If we finish the loop due to the position approaching the end of
    // the buffer we may fail to skip the last line in a comment block that
    // has an invalid initial sequence and thus emit extra diagnostics. We
    // should really fall through to the generic skipping logic, but the code
    // organization will need to change significantly to allow that.
  } else {
    while (position + prefix_size < static_cast<ssize_t>(source_text.size()) &&
           memcmp(source_text.data() + first_line_start,
                  source_text.data() + position, prefix_size) == 0) {
      skip_to_next_line();
    }
  }

  buffer_.AddComment(indent, comment_start, position);

  // Now compute the indent of this next line before we finish.
  ssize_t line_start = position;
  SkipHorizontalWhitespace(source_text, position);

  // Now that we're done scanning, update to the latest line index and indent.
  line_index_ = line_index;
  current_line_info()->indent = position - line_start;
}

auto Lexer::CanFormRealLiteral() -> bool {
  // When a numeric literal immediately follows a `.` or `->` token, with no
  // intervening whitespace, a real literal is never formed.
  if (has_leading_space_) {
    return true;
  }
  auto kind = buffer_.GetKind(buffer_.tokens().end()[-1]);
  return kind != TokenKind::Period && kind != TokenKind::MinusGreater;
}

auto Lexer::LexNumericLiteral(llvm::StringRef source_text, ssize_t& position)
    -> LexResult {
  std::optional<NumericLiteral> literal =
      NumericLiteral::Lex(source_text.substr(position), CanFormRealLiteral());
  if (!literal) {
    return LexError(source_text, position);
  }

  // Capture the position before we step past the token.
  int32_t byte_offset = position;
  int token_size = literal->text().size();
  position += token_size;

  return VariantMatch(
      literal->ComputeValue(emitter_),
      [&](NumericLiteral::IntValue&& value) {
        return LexTokenWithPayload(TokenKind::IntLiteral,
                                   buffer_.value_stores_->ints()
                                       .AddUnsigned(std::move(value.value))
                                       .AsTokenPayload(),
                                   byte_offset);
      },
      [&](NumericLiteral::RealValue&& value) {
        auto real_id = buffer_.value_stores_->reals().Add(Real{
            .mantissa = value.mantissa,
            .exponent = value.exponent,
            .is_decimal = (value.radix == NumericLiteral::Radix::Decimal)});
        return LexTokenWithPayload(TokenKind::RealLiteral, real_id.index,
                                   byte_offset);
      },
      [&](NumericLiteral::UnrecoverableError) {
        return LexTokenWithPayload(TokenKind::Error, token_size, byte_offset);
      });
}

auto Lexer::LexStringLiteral(llvm::StringRef source_text, ssize_t& position)
    -> LexResult {
  std::optional<StringLiteral> literal =
      StringLiteral::Lex(source_text.substr(position));
  if (!literal) {
    return LexError(source_text, position);
  }

  // Capture the position before we step past the token.
  int32_t byte_offset = position;
  int string_column = byte_offset - current_line_info()->start;
  ssize_t literal_size = literal->text().size();
  position += literal_size;

  // Update line and column information.
  if (literal->is_multi_line()) {
    while (next_line_info()->start < position) {
      ++line_index_;
      current_line_info()->indent = string_column;
    }
    // Note that we've updated the current line at this point, but
    // `set_indent_` is already true from above. That remains correct as the
    // last line of the multi-line literal *also* has its indent set.
  }

  if (literal->is_terminated()) {
    auto string_id = buffer_.value_stores_->string_literal_values().Add(
        literal->ComputeValue(buffer_.allocator_, emitter_));
    return LexTokenWithPayload(TokenKind::StringLiteral, string_id.index,
                               byte_offset);
  } else {
    CARBON_DIAGNOSTIC(UnterminatedString, Error,
                      "string is missing a terminator");
    emitter_.Emit(literal->text().begin(), UnterminatedString);
    return LexTokenWithPayload(TokenKind::Error, literal_size, byte_offset);
  }
}

auto Lexer::LexOneCharSymbolToken(llvm::StringRef source_text, TokenKind kind,
                                  ssize_t& position) -> TokenIndex {
  // Verify in a debug build that the incoming token kind is correct.
  CARBON_DCHECK(kind != TokenKind::Error);
  CARBON_DCHECK(kind.fixed_spelling().size() == 1);
  CARBON_DCHECK(source_text[position] == kind.fixed_spelling().front(),
                "Source text starts with '{0}' instead of the spelling '{1}' "
                "of the incoming token kind '{2}'",
                source_text[position], kind.fixed_spelling(), kind);

  TokenIndex token = LexToken(kind, position);
  ++position;
  return token;
}

auto Lexer::LexOpeningSymbolToken(llvm::StringRef source_text, TokenKind kind,
                                  ssize_t& position) -> LexResult {
  CARBON_DCHECK(kind.is_opening_symbol());
  CARBON_DCHECK(kind.fixed_spelling().size() == 1);
  CARBON_DCHECK(source_text[position] == kind.fixed_spelling().front(),
                "Source text starts with '{0}' instead of the spelling '{1}' "
                "of the incoming token kind '{2}'",
                source_text[position], kind.fixed_spelling(), kind);

  int32_t byte_offset = position;
  ++position;

  // Lex the opening symbol with a zero closing index. We'll add a payload later
  // when we match a closing symbol or in recovery.
  TokenIndex token = LexToken(kind, byte_offset);
  open_groups_.push_back(token);
  return token;
}

auto Lexer::LexClosingSymbolToken(llvm::StringRef source_text, TokenKind kind,
                                  ssize_t& position) -> LexResult {
  CARBON_DCHECK(kind.is_closing_symbol());
  CARBON_DCHECK(kind.fixed_spelling().size() == 1);
  CARBON_DCHECK(source_text[position] == kind.fixed_spelling().front(),
                "Source text starts with '{0}' instead of the spelling '{1}' "
                "of the incoming token kind '{2}'",
                source_text[position], kind.fixed_spelling(), kind);

  int32_t byte_offset = position;
  ++position;

  // If there's not a matching opening symbol, just track that we had an error.
  // We will diagnose and recover when we reach the end of the file. See
  // `DiagnoseAndFixMismatchedBrackets` for details.
  if (LLVM_UNLIKELY(open_groups_.empty())) {
    has_mismatched_brackets_ = true;
    // Lex without a matching index payload -- we'll add one during recovery.
    return LexToken(kind, byte_offset);
  }

  TokenIndex opening_token = open_groups_.pop_back_val();
  TokenIndex token =
      LexTokenWithPayload(kind, opening_token.index, byte_offset);

  auto& opening_token_info = buffer_.GetTokenInfo(opening_token);
  if (LLVM_UNLIKELY(opening_token_info.kind() != kind.opening_symbol())) {
    has_mismatched_brackets_ = true;
    buffer_.GetTokenInfo(token).set_opening_token_index(TokenIndex::None);
    return token;
  }

  opening_token_info.set_closing_token_index(token);
  return token;
}

auto Lexer::LexSymbolToken(llvm::StringRef source_text, ssize_t& position)
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

  TokenIndex token = LexToken(kind, position);
  position += kind.fixed_spelling().size();
  return token;
}

auto Lexer::LexWordAsTypeLiteralToken(llvm::StringRef word, int32_t byte_offset)
    -> LexResult {
  if (word.size() < 2) {
    // Too short to form one of these tokens.
    return LexResult::NoMatch();
  }

  TokenKind kind;
  switch (word.front()) {
    case 'i':
      kind = TokenKind::IntTypeLiteral;
      break;
    case 'u':
      kind = TokenKind::UnsignedIntTypeLiteral;
      break;
    case 'f':
      kind = TokenKind::FloatTypeLiteral;
      break;
    default:
      return LexResult::NoMatch();
  };

  // No leading zeros allowed.
  if ('1' > word[1] || word[1] > '9') {
    return LexResult::NoMatch();
  }

  llvm::StringRef suffix = word.substr(1);

  // Type bit-widths can't usefully be large integers so we restrict to small
  // ones that are especially easy to parse into a normal integer variable by
  // restricting the number of digits to round trip.
  int64_t suffix_value;
  constexpr ssize_t DigitLimit =
      std::numeric_limits<decltype(suffix_value)>::digits10;
  if (suffix.size() > DigitLimit) {
    // See if this is not actually a type literal.
    if (!llvm::all_of(suffix, IsDecimalDigit)) {
      return LexResult::NoMatch();
    }

    // Otherwise, diagnose and produce an error token.
    CARBON_DIAGNOSTIC(TooManyTypeBitWidthDigits, Error,
                      "found a type literal with a bit width using {0} digits, "
                      "which is greater than the limit of {1}",
                      size_t, size_t);
    emitter_.Emit(word.begin() + 1, TooManyTypeBitWidthDigits, suffix.size(),
                  DigitLimit);
    return LexTokenWithPayload(TokenKind::Error, word.size(), byte_offset);
  }

  // It's tempting to do something more clever because we know the length ahead
  // of time, but we expect these to be short (1-3 digits) and profiling doesn't
  // show the loop as hot in the short cases.
  suffix_value = suffix[0] - '0';
  for (char c : suffix.drop_front()) {
    if (!IsDecimalDigit(c)) {
      return LexResult::NoMatch();
    }
    suffix_value = suffix_value * 10 + (c - '0');
  }

  // Add the bit width to our integer store and get its index. We treat it as
  // unsigned as that's less expensive and it can't be negative.
  CARBON_CHECK(suffix_value >= 0);
  auto bit_width_payload =
      buffer_.value_stores_->ints().Add(suffix_value).AsTokenPayload();

  return LexTokenWithPayload(kind, bit_width_payload, byte_offset);
}

auto Lexer::LexKeywordOrIdentifier(llvm::StringRef source_text,
                                   ssize_t& position) -> LexResult {
  if (static_cast<unsigned char>(source_text[position]) > 0x7F) {
    // TODO: Need to add support for Unicode lexing.
    return LexError(source_text, position);
  }
  CARBON_CHECK(
      IsIdStartByteTable[static_cast<unsigned char>(source_text[position])]);

  // Capture the position before we step past the token.
  int32_t byte_offset = position;

  // Take the valid characters off the front of the source buffer.
  llvm::StringRef identifier_text =
      ScanForIdentifierPrefix(source_text.substr(position));
  CARBON_CHECK(!identifier_text.empty(), "Must have at least one character!");
  position += identifier_text.size();

  // Check if the text is a type literal, and if so form such a literal.
  if (LexResult result =
          LexWordAsTypeLiteralToken(identifier_text, byte_offset)) {
    return result;
  }

  // Check if the text matches a keyword token, and if so use that.
  TokenKind kind = llvm::StringSwitch<TokenKind>(identifier_text)
#define CARBON_KEYWORD_TOKEN(Name, Spelling) .Case(Spelling, TokenKind::Name)
#include "toolchain/lex/token_kind.def"
                       .Default(TokenKind::Error);
  if (kind != TokenKind::Error) {
    return LexToken(kind, byte_offset);
  }

  // Otherwise we have a generic identifier.
  return LexTokenWithPayload(
      TokenKind::Identifier,
      buffer_.value_stores_->identifiers().Add(identifier_text).index,
      byte_offset);
}

auto Lexer::LexHash(llvm::StringRef source_text, ssize_t& position)
    -> LexResult {
  // For `r#`, we already lexed an `r` identifier token. Detect that case and
  // replace that token with a raw identifier. We do this to keep identifier
  // lexing as fast as possible.

  // Look for the `r` token. Note that this is always in bounds because we
  // create a start of file token.
  auto& prev_token_info = buffer_.token_infos_.back();

  // If the previous token isn't the identifier `r`, or the character after `#`
  // isn't the start of an identifier, this is not a raw identifier.
  if (prev_token_info.kind() != TokenKind::Identifier ||
      source_text[position - 1] != 'r' ||
      position + 1 == static_cast<ssize_t>(source_text.size()) ||
      !IsIdStartByteTable[static_cast<unsigned char>(
          source_text[position + 1])] ||
      prev_token_info.byte_offset() != static_cast<int32_t>(position) - 1) {
    [[clang::musttail]] return LexStringLiteral(source_text, position);
  }
  CARBON_DCHECK(buffer_.value_stores_->identifiers().Get(
                    prev_token_info.ident_id()) == "r");

  // Take the valid characters off the front of the source buffer.
  llvm::StringRef identifier_text =
      ScanForIdentifierPrefix(source_text.substr(position + 1));
  CARBON_CHECK(!identifier_text.empty(), "Must have at least one character!");
  position += 1 + identifier_text.size();

  // Replace the `r` identifier's value with the raw identifier.
  // TODO: This token doesn't carry any indicator that it's raw, so
  // diagnostics are unclear.
  prev_token_info.set_ident_id(
      buffer_.value_stores_->identifiers().Add(identifier_text));
  return LexResult(TokenIndex(buffer_.token_infos_.size() - 1));
}

auto Lexer::LexError(llvm::StringRef source_text, ssize_t& position)
    -> LexResult {
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

  auto token =
      LexTokenWithPayload(TokenKind::Error, error_text.size(), position);
  CARBON_DIAGNOSTIC(UnrecognizedCharacters, Error,
                    "encountered unrecognized characters while parsing");
  emitter_.Emit(error_text.begin(), UnrecognizedCharacters);

  position += error_text.size();
  return token;
}

auto Lexer::LexFileStart(llvm::StringRef source_text, ssize_t& position)
    -> void {
  CARBON_CHECK(position == 0);

  // Before lexing any source text, add the start-of-file token so that code
  // can assume a non-empty token buffer for the rest of lexing.
  LexToken(TokenKind::FileStart, 0);

  // The file start also represents whitespace.
  NoteWhitespace();

  // Also skip any horizontal whitespace and record the indentation of the
  // first line.
  SkipHorizontalWhitespace(source_text, position);
  auto* line_info = current_line_info();
  CARBON_CHECK(line_info->start == 0);
  line_info->indent = position;
}

auto Lexer::LexFileEnd(llvm::StringRef source_text, ssize_t position) -> void {
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
  }

  // The end-of-file token is always considered to be whitespace.
  NoteWhitespace();

  LexToken(TokenKind::FileEnd, position);
}

auto Lexer::Finalize() -> void {
  // If we had any mismatched brackets, issue diagnostics and fix them.
  if (has_mismatched_brackets_ || !open_groups_.empty()) {
    DiagnoseAndFixMismatchedBrackets();
  }

  // Reject source files with so many tokens that we may have exceeded the
  // number of bits in `token_payload_`.
  //
  // Note that we rely on this check also catching the case where there are too
  // many identifiers to fit an `IdentifierId` into a `token_payload_`, and
  // likewise for `IntId` and so on. If we start adding any of those IDs prior
  // to lexing, we may need to also limit the number of those IDs here.
  if (buffer_.token_infos_.size() > TokenIndex::Max) {
    CARBON_DIAGNOSTIC(TooManyTokens, Error,
                      "too many tokens in source file; try splitting into "
                      "multiple source files");
    // Subtract one to leave room for the `FileEnd` token.
    token_emitter_.Emit(TokenIndex(TokenIndex::Max - 1), TooManyTokens);
    // TODO: Convert tokens after the token limit to error tokens to avoid
    // misinterpretation by consumers of the tokenized buffer.
  }
}

// A list of pending insertions to make into a tokenized buffer for error
// recovery. These are buffered so that we can perform them in linear time.
class Lexer::ErrorRecoveryBuffer {
 public:
  explicit ErrorRecoveryBuffer(TokenizedBuffer& buffer) : buffer_(buffer) {}

  auto empty() const -> bool {
    return new_tokens_.empty() && !any_error_tokens_;
  }

  // Insert a recovery token of kind `kind` before `insert_before`. Note that we
  // currently require insertions to be specified in source order, but this
  // restriction would be easy to relax.
  auto InsertBefore(TokenIndex insert_before, TokenKind kind) -> void {
    CARBON_CHECK(insert_before.index > 0,
                 "Cannot insert before the start of file token.");
    CARBON_CHECK(
        insert_before.index < static_cast<int>(buffer_.token_infos_.size()),
        "Cannot insert after the end of file token.");
    CARBON_CHECK(
        new_tokens_.empty() || new_tokens_.back().first <= insert_before,
        "Insertions performed out of order.");

    // If the `insert_before` token has leading whitespace, mark the
    // inserted token as also having leading whitespace. This avoids changing
    // whether the prior tokens had leading or trailing whitespace when
    // inserting.
    bool insert_leading_space = buffer_.HasLeadingWhitespace(insert_before);

    // Find the end of the token before the target token, and add the new token
    // there.
    TokenIndex insert_after(insert_before.index - 1);
    const auto& prev_info = buffer_.GetTokenInfo(insert_after);
    int32_t byte_offset =
        prev_info.byte_offset() + buffer_.GetTokenText(insert_after).size();
    new_tokens_.push_back(
        {insert_before, TokenInfo(kind, insert_leading_space, byte_offset)});
  }

  // Replace the given token with an error token. We do this immediately,
  // because we don't benefit from buffering it.
  auto ReplaceWithError(TokenIndex token) -> void {
    auto& token_info = buffer_.GetTokenInfo(token);
    int error_length = buffer_.GetTokenText(token).size();
    token_info.ResetAsError(error_length);
    any_error_tokens_ = true;
  }

  // Merge the recovery tokens into the token list of the tokenized buffer.
  auto Apply() -> void {
    auto old_tokens = std::move(buffer_.token_infos_);
    buffer_.token_infos_.clear();
    int new_size = old_tokens.size() + new_tokens_.size();
    buffer_.token_infos_.reserve(new_size);
    buffer_.recovery_tokens_.resize(new_size);

    int old_tokens_offset = 0;
    for (auto [next_offset, info] : new_tokens_) {
      buffer_.token_infos_.append(old_tokens.begin() + old_tokens_offset,
                                  old_tokens.begin() + next_offset.index);
      buffer_.AddToken(info);
      buffer_.recovery_tokens_.set(next_offset.index);
      old_tokens_offset = next_offset.index;
    }
    buffer_.token_infos_.append(old_tokens.begin() + old_tokens_offset,
                                old_tokens.end());
  }

  // Perform bracket matching to fix cross-references between tokens. This must
  // be done after all recovery is performed and all brackets match, because
  // recovery will change token indexes.
  auto FixTokenCrossReferences() -> void {
    llvm::SmallVector<TokenIndex> open_groups;
    for (auto token : buffer_.tokens()) {
      auto kind = buffer_.GetKind(token);
      if (kind.is_opening_symbol()) {
        open_groups.push_back(token);
      } else if (kind.is_closing_symbol()) {
        CARBON_CHECK(!open_groups.empty(), "Failed to balance brackets");
        auto opening_token = open_groups.pop_back_val();

        CARBON_CHECK(
            kind == buffer_.GetTokenInfo(opening_token).kind().closing_symbol(),
            "Failed to balance brackets");
        auto& opening_token_info = buffer_.GetTokenInfo(opening_token);
        auto& closing_token_info = buffer_.GetTokenInfo(token);
        opening_token_info.set_closing_token_index(token);
        closing_token_info.set_opening_token_index(opening_token);
      }
    }
  }

 private:
  TokenizedBuffer& buffer_;

  // A list of tokens to insert into the token stream to fix mismatched
  // brackets. The first element in each pair is the original token index to
  // insert the new token before.
  llvm::SmallVector<std::pair<TokenIndex, TokenizedBuffer::TokenInfo>>
      new_tokens_;

  // Whether we have changed any tokens into error tokens.
  bool any_error_tokens_ = false;
};

// Issue an UnmatchedOpening diagnostic.
static auto DiagnoseUnmatchedOpening(DiagnosticEmitter<TokenIndex>& emitter,
                                     TokenIndex opening_token) -> void {
  CARBON_DIAGNOSTIC(UnmatchedOpening, Error,
                    "opening symbol without a corresponding closing symbol");
  emitter.Emit(opening_token, UnmatchedOpening);
}

// If brackets didn't pair or nest properly, find a set of places to insert
// brackets to fix the nesting, issue suitable diagnostics, and update the
// token list to describe the fixes.
auto Lexer::DiagnoseAndFixMismatchedBrackets() -> void {
  ErrorRecoveryBuffer fixes(buffer_);

  // Look for mismatched brackets and decide where to add tokens to fix them.
  //
  // TODO: For now, we use a greedy algorithm for this. We could do better by
  // taking indentation into account. For example:
  //
  //     1  fn F() {
  //     2    if (thing1)
  //     3      thing2;
  //     4    }
  //     5  }
  //
  // Here, we'll match the `{` on line 1 with the `}` on line 4, and then
  // report that the `}` on line 5 is unmatched. Instead, we should notice that
  // line 1 matches better with line 5 due to indentation, and work out that
  // the missing `{` was on line 2, also based on indentation.
  open_groups_.clear();
  for (auto token : buffer_.tokens()) {
    auto kind = buffer_.GetKind(token);
    if (kind.is_opening_symbol()) {
      open_groups_.push_back(token);
      continue;
    }

    if (!kind.is_closing_symbol()) {
      continue;
    }

    // Find the innermost matching opening symbol.
    auto opening_it = llvm::find_if(
        llvm::reverse(open_groups_), [&](TokenIndex opening_token) {
          return buffer_.GetTokenInfo(opening_token).kind().closing_symbol() ==
                 kind;
        });
    if (opening_it == open_groups_.rend()) {
      CARBON_DIAGNOSTIC(
          UnmatchedClosing, Error,
          "closing symbol without a corresponding opening symbol");
      token_emitter_.Emit(token, UnmatchedClosing);
      fixes.ReplaceWithError(token);
      continue;
    }

    // All intermediate open tokens have no matching close token.
    for (auto it = open_groups_.rbegin(); it != opening_it; ++it) {
      DiagnoseUnmatchedOpening(token_emitter_, *it);

      // Add a closing bracket for the unclosed group here.
      //
      // TODO: Indicate in the diagnostic that we did this, perhaps by
      // annotating the snippet.
      auto opening_kind = buffer_.GetKind(*it);
      fixes.InsertBefore(token, opening_kind.closing_symbol());
    }

    open_groups_.erase(opening_it.base() - 1, open_groups_.end());
  }

  // Diagnose any remaining unmatched opening symbols.
  for (auto token : open_groups_) {
    // We don't have a good location to insert a close bracket. Convert the
    // opening token from a bracket to an error.
    DiagnoseUnmatchedOpening(token_emitter_, token);
    fixes.ReplaceWithError(token);
  }

  CARBON_CHECK(!fixes.empty(), "Didn't find anything to fix");
  fixes.Apply();
  fixes.FixTokenCrossReferences();
}

auto Lex(SharedValueStores& value_stores, SourceBuffer& source,
         DiagnosticConsumer& consumer) -> TokenizedBuffer {
  return Lexer(value_stores, source, consumer).Lex();
}

}  // namespace Carbon::Lex
