// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/lex.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>

#include "common/check.h"
#include "common/preserve_none.h"
#include "common/vlog.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Endian.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/lex/character_set.h"
#include "toolchain/lex/helpers.h"
#include "toolchain/lex/mismatched_brackets.h"
#include "toolchain/lex/numeric_literal.h"
#include "toolchain/lex/source_loc.h"
#include "toolchain/lex/string_literal.h"
#include "toolchain/lex/token_index.h"
#include "toolchain/lex/token_info.h"
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

// Whether `SourceBlock` classifies 64-byte blocks of source text with SIMD.
// This is implemented with SSSE3 or AVX2 on x86, and with Neon on little-endian
// Arm, as the Neon code relies on the bytes of a vector being the little-endian
// bytes of its 64-bit lanes.
#if CARBON_USE_SIMD && \
    (__SSSE3__ || (__ARM_NEON && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)))
#define CARBON_USE_BLOCK_SIMD 1
#else
#define CARBON_USE_BLOCK_SIMD 0
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
    explicit(false) LexResult(TokenIndex /*discarded_token*/)
        : LexResult(true) {}

    // Returns a result indicating no token was produced.
    static auto NoMatch() -> LexResult { return LexResult(false); }

    // Tests whether a token was produced by the lexing routine, and
    // the lexer can continue forming tokens.
    explicit operator bool() const { return formed_token_; }

   private:
    explicit LexResult(bool formed_token) : formed_token_(formed_token) {}

    bool formed_token_;
  };

  Lexer(const LexOptions& options, SharedValueStores& value_stores,
        SourceBuffer& source, Diagnostics::Consumer& consumer)
      : options_(options),
        buffer_(value_stores, source),
        consumer_(consumer),
        emitter_(&consumer_, &buffer_),
        token_emitter_(&consumer_, &buffer_) {}

  // Find all line endings, create the line data structures, and populate the
  // identifier-ending-byte bitmap.
  //
  // Explicitly kept out-of-line because this is a significant loop that is
  // useful to have in the profile and it doesn't simplify by inlining at all.
  // But because it can, the compiler will flatten this otherwise.
  [[gnu::noinline]] auto MakeLines(llvm::StringRef source_text) -> void;

  // Returns the text of the identifier starting at `position`: the byte there,
  // which must be in `IsIdStartByteTable`, and the run of bytes in
  // `IsIdByteTable` that follows it. Uses `id_end_bitmap_`, so must be called
  // after `MakeLines`.
  //
  // TODO: This doesn't implement Carbon's design for Unicode characters in
  // identifiers, and instead treats all non-ASCII characters as
  // non-identifier characters.
  [[clang::always_inline]] auto ScanIdentifier(llvm::StringRef source_text,
                                               ssize_t position) const
      -> llvm::StringRef;

  auto current_line() -> LineIndex { return line_index_; }

  auto current_line_info() -> LineInfo& {
    return buffer_.line_infos_.Get(line_index_);
  }

  auto next_line() -> LineIndex { return LineIndex(line_index_.index + 1); }

  auto next_line_info() -> LineInfo& {
    return buffer_.line_infos_.Get(next_line());
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

  // Starts a new line, skipping whitespace and setting the indent.
  [[clang::always_inline]] auto AdvanceToLine(llvm::StringRef source_text,
                                              ssize_t& position,
                                              LineIndex to_line_index) -> void;

  auto LexHorizontalWhitespace(llvm::StringRef source_text, ssize_t& position)
      -> void;

  auto LexVerticalWhitespace(llvm::StringRef source_text, ssize_t& position)
      -> void;

  auto LexCR(llvm::StringRef source_text, ssize_t& position) -> void;

  // Diagnoses a CR that doesn't begin a CR+LF line ending.
  [[gnu::noinline]] auto DiagnoseCR(llvm::StringRef source_text,
                                    ssize_t position) -> void;

  auto LexComment(llvm::StringRef source_text, ssize_t& position) -> void;

  // Determines whether a real literal can be formed at the current location.
  // This is the case unless the preceding token is `.` or `->` and there is no
  // intervening whitespace.
  auto CanFormRealLiteral() -> bool;

  // Always inlined into its dispatch function, so that the inline fast paths in
  // `NumericLiteral` for single digits don't pay for a call.
  [[clang::always_inline]] auto LexNumericLiteral(llvm::StringRef source_text,
                                                  ssize_t& position)
      -> LexResult;

  auto LexStringLiteral(llvm::StringRef source_text, ssize_t& position)
      -> LexResult;

  auto LexOneCharSymbolToken(llvm::StringRef source_text, TokenKind kind,
                             ssize_t& position) -> TokenIndex;

  auto LexOpeningSymbolToken(llvm::StringRef source_text, TokenKind kind,
                             ssize_t& position) -> LexResult;

  auto LexClosingSymbolToken(llvm::StringRef source_text, TokenKind kind,
                             ssize_t& position) -> LexResult;

  // Given a word that has already been lexed, determine whether it is a type
  // literal and if so form the corresponding token.
  auto LexWordAsTypeLiteralToken(llvm::StringRef word, int32_t byte_offset)
      -> LexResult;

  // Given a lexed word, determine whether it is a dollar int literal and if so
  // form the corresponding token,
  auto LexWordAsDollarIntLiteralToken(llvm::StringRef word, int32_t byte_offset)
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

  // Checks for an ends a `DumpSemIRRange` that's missing an explicit end
  // marker.
  auto EndDumpSemIRRangeIfIncomplete(const char* diag_loc) -> void;

  auto has_dump_sem_ir_ranges() -> bool {
    return buffer_.has_dump_sem_ir_ranges();
  }

 private:
  class ErrorRecoveryBuffer;

  // Handles `//@dump-sem-ir-begin` for a `DumpSemIRRange`.
  auto BeginDumpSemIRRange(const char* diag_loc) -> void;

  // Handles `//@dump-sem-ir-end` for a `DumpSemIRRange`.
  auto EndDumpSemIRRange(const char* diag_loc) -> void;

  LexOptions options_;

  TokenizedBuffer buffer_;

  LineIndex line_index_ = LineIndex::None;

  // Tracks whether the lexer has encountered whitespace that will be leading
  // whitespace for the next lexed token. Reset after each token lexed.
  bool has_leading_space_ = false;

  llvm::SmallVector<TokenIndex> open_groups_;
  bool has_mismatched_brackets_ = false;

  Diagnostics::ErrorTrackingConsumer consumer_;

  TokenizedBuffer::SourceLocDiagnosticEmitter emitter_;

  TokenizedBuffer::TokenDiagnosticEmitter token_emitter_;

  // A 1-bit-per-byte bitmap built during `MakeLines` tracking whether the
  // corresponding byte would potentially end an identifier.
  //
  // Each bit `i` is `1` if `source_text[i]` is not in `IsIdByteTable`. This
  // intentionally includes every non-ASCII byte, to keep scanning ASCII
  // identifiers fast. Supporting Unicode identifiers will require treating a
  // non-ASCII byte that stops a scan as a possible part of the identifier,
  // rather than its end.
  //
  // The bitmap is then padded out with `1` bits through a complete sentinel
  // `uint64_t`. This lets code scanning these bits always read a full 64 bits,
  // even when starting at the bit for the last byte of the source text, and
  // ensures any scan for a `1` bit stops by the end of the source text.
  //
  // Bit `i` is bit `i % 8` of byte `i / 8`, so 64 bits of the bitmap can be
  // read as a little-endian `uint64_t` starting at any byte.
  llvm::SmallVector<uint8_t, 2048> id_end_bitmap_;
};

#if CARBON_USE_SIMD
namespace {
#if __ARM_NEON
using SimdMaskT = uint8x16_t;
#elif __x86_64__
using SimdMaskT = __m128i;
#else
#error "Unsupported SIMD architecture!"
#endif
using SimdMaskArrayT = std::array<SimdMaskT, sizeof(SimdMaskT) + 1>;
}  // namespace

// A table of masks to include 0-16 bytes of an SSE register.
static constexpr SimdMaskArrayT PrefixMasks = []() constexpr {
  SimdMaskArrayT masks = {};
  for (int i = 1; i < static_cast<int>(masks.size()); ++i) {
    masks[i] =
        // The SIMD types and constexpr require a C-style cast.
        // NOLINTNEXTLINE(google-readability-casting)
        (SimdMaskT)(std::numeric_limits<unsigned __int128>::max() >>
                    ((sizeof(SimdMaskT) - i) * 8));
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
  table['$'] = true;
  return table;
}();

// A table of booleans that we can use to classify bytes as being valid
// identifier (or keyword) characters. This is used in the portable,
// non-vectorized `SourceBlock` code to find the bytes ending identifiers.
static constexpr std::array<bool, 256> IsIdByteTable = [] {
  std::array<bool, 256> table = IsIdStartByteTable;
  for (char c = '0'; c <= '9'; ++c) {
    table[c] = true;
  }
  // Identifiers can only have `$` in start.
  table['$'] = false;
  return table;
}();

// A table of booleans classifying the bytes that end a run of unrecognized
// characters forming an error token: alphanumeric characters, `_`, tabs and
// newlines, and any character that is by itself a symbol.
static constexpr std::array<bool, 256> IsErrorEndByteTable = [] {
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
  table['\t'] = true;
  table['\n'] = true;
#define CARBON_SYMBOL_TOKEN(TokenName, Spelling) \
  if (sizeof(Spelling) == 2) {                   \
    table[(Spelling)[0]] = true;                 \
  }
#include "toolchain/lex/token_kind.def"
  return table;
}();

#if CARBON_USE_BLOCK_SIMD
// The SIMD code paths uses a scheme derived from the techniques in Geoff
// Langdale and Daniel Lemire's work on parsing JSON[1]. Specifically, that
// paper outlines a technique of using two 4-bit indexed in-register look-up
// tables (LUTs) to classify bytes in a branchless SIMD code sequence.
//
// [1]: https://arxiv.org/pdf/1902.08318.pdf
//
// The goal is to get a bit mask classifying different sets of bytes. For each
// input byte, a high bit indicates a UTF-8 encoded Unicode character, and the
// high nibble LUT sets no bits for these. Otherwise, we want the mask bits to
// be set with the following logic derived by inspecting the high nibble and low
// nibble of the input:
// bit0 = 1 for `_`: high `0x5` and low `0xF`
// bit1 = 1 for `0-9`: high `0x3` and low `0x0` - `0x9`
// bit2 = 1 for `A-O` and `a-o`: high `0x4` or `0x6` and low `0x1` - `0xF`
// bit3 = 1 for `P-Z` and 'p-z': high `0x5` or `0x7` and low `0x0` - `0xA`
// bit4 = unused
// bit5 = unused
// bit6 = unused
// bit7 = unused
//
// No bits set means definitively non-ID character.
//
// Bits 4-7 remain unused if we need to classify more characters.
namespace {
// Struct used to implement the nibble LUT for SIMD implementations.
//
// Forced to 16-byte alignment to ensure we can load it easily in SIMD code.
struct alignas(16) NibbleLUT {
#if __AVX2__
  // Loads the LUT into both 128-bit lanes, as `_mm256_shuffle_epi8` looks up
  // the bytes in each lane separately.
  [[clang::always_inline]] auto Load() const -> __m256i {
    return _mm256_broadcastsi128_si256(
        _mm_load_si128(reinterpret_cast<const __m128i*>(this)));
  }
#elif __SSSE3__
  [[clang::always_inline]] auto Load() const -> __m128i {
    return _mm_load_si128(reinterpret_cast<const __m128i*>(this));
  }
#elif __ARM_NEON
  [[clang::always_inline]] auto Load() const -> uint8x16_t {
    return vld1q_u8(reinterpret_cast<const uint8_t*>(this));
  }
#endif

  uint8_t nibble_0 [[clang::require_explicit_initialization]];
  uint8_t nibble_1 [[clang::require_explicit_initialization]];
  uint8_t nibble_2 [[clang::require_explicit_initialization]];
  uint8_t nibble_3 [[clang::require_explicit_initialization]];
  uint8_t nibble_4 [[clang::require_explicit_initialization]];
  uint8_t nibble_5 [[clang::require_explicit_initialization]];
  uint8_t nibble_6 [[clang::require_explicit_initialization]];
  uint8_t nibble_7 [[clang::require_explicit_initialization]];
  uint8_t nibble_8 [[clang::require_explicit_initialization]];
  uint8_t nibble_9 [[clang::require_explicit_initialization]];
  uint8_t nibble_a [[clang::require_explicit_initialization]];
  uint8_t nibble_b [[clang::require_explicit_initialization]];
  uint8_t nibble_c [[clang::require_explicit_initialization]];
  uint8_t nibble_d [[clang::require_explicit_initialization]];
  uint8_t nibble_e [[clang::require_explicit_initialization]];
  uint8_t nibble_f [[clang::require_explicit_initialization]];
};
// `Load` reads the LUT as a 128-bit vector.
static_assert(sizeof(NibbleLUT) == 128 / 8);
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
    .nibble_8 = 0b0000'0000,
    .nibble_9 = 0b0000'0000,
    .nibble_a = 0b0000'0000,
    .nibble_b = 0b0000'0000,
    .nibble_c = 0b0000'0000,
    .nibble_d = 0b0000'0000,
    .nibble_e = 0b0000'0000,
    .nibble_f = 0b0000'0000,
};
static constexpr NibbleLUT LowLUT = {
    .nibble_0 = 0b0000'1010,
    .nibble_1 = 0b0000'1110,
    .nibble_2 = 0b0000'1110,
    .nibble_3 = 0b0000'1110,
    .nibble_4 = 0b0000'1110,
    .nibble_5 = 0b0000'1110,
    .nibble_6 = 0b0000'1110,
    .nibble_7 = 0b0000'1110,
    .nibble_8 = 0b0000'1110,
    .nibble_9 = 0b0000'1110,
    .nibble_a = 0b0000'1100,
    .nibble_b = 0b0000'0100,
    .nibble_c = 0b0000'0100,
    .nibble_d = 0b0000'0100,
    .nibble_e = 0b0000'0100,
    .nibble_f = 0b0000'0101,
};
#endif  // CARBON_USE_BLOCK_SIMD

namespace {

// A 64-byte source text block classified in parallel into 64-bit 1-bit-per-byte
// masks. Like `RawHashtable::MetadataGroup`, a portable classification is
// cross-checked against the platform-specific SIMD classification in debug
// builds, and is used directly without block SIMD support.
class SourceBlock {
 public:
  static constexpr ssize_t Size = 64;

#if CARBON_USE_BLOCK_SIMD
  // Whether to cross-check the SIMD classification against the portable one.
#if !defined(NDEBUG)
  static constexpr bool DebugSimd = true;
#else
  static constexpr bool DebugSimd = false;
#endif
#endif

  // The classification of the bytes of a block. Bit `i` of each mask is for
  // byte `i` of the block, so the lowest bit is for the first byte and the
  // highest bit is for the last byte.
  struct BlockMasks {
    // Bits set for the newline bytes.
    uint64_t newlines [[clang::require_explicit_initialization]];
    // Bits set for the bytes that can't continue an identifier, and so end any
    // identifier before them.
    uint64_t id_ends [[clang::require_explicit_initialization]];
    // The number of newlines, which is the number of bits set in `newlines`.
    int newline_count [[clang::require_explicit_initialization]];

    auto operator==(const BlockMasks&) const -> bool = default;
  };

  // Loads a block from the first `Size` bytes of `text`, which must have at
  // least that many bytes.
  [[clang::always_inline]] static auto Load(llvm::StringRef text)
      -> SourceBlock {
    CARBON_DCHECK(static_cast<ssize_t>(text.size()) >= Size);
    SourceBlock block;
    std::memcpy(block.bytes_, text.data(), Size);
    return block;
  }

  // Classifies the bytes of the block.
  [[clang::always_inline]] auto Classify() const -> BlockMasks {
#if CARBON_USE_BLOCK_SIMD
    BlockMasks simd_result = SimdClassify();
    if constexpr (DebugSimd) {
      CARBON_DCHECK(simd_result == PortableClassify());
    }
    return simd_result;
#else
    return PortableClassify();
#endif
  }

 private:
  // The bytes of the block, which the SIMD classification reads as vectors.
  union {
    uint8_t bytes_[Size] = {};
#if CARBON_USE_BLOCK_SIMD
#if __AVX2__
    __m256i vec_[2];
#elif __SSSE3__
    __m128i vec_[4];
#elif __ARM_NEON
    uint8x16_t vec_[4];
#endif
    static_assert(sizeof(vec_) == Size);
#endif
  };

  auto PortableClassify() const -> BlockMasks {
    uint64_t newlines = 0;
    uint64_t id_ends = 0;
    for (ssize_t i = 0; i < Size; ++i) {
      // Shift in each bit rather than branching, as identifier and
      // non-identifier bytes are interleaved unpredictably.
      newlines |= uint64_t{bytes_[i] == '\n'} << i;
      id_ends |= uint64_t{!IsIdByteTable[bytes_[i]]} << i;
    }
    return {.newlines = newlines,
            .id_ends = id_ends,
            .newline_count = std::popcount(newlines)};
  }

#if CARBON_USE_BLOCK_SIMD
#if __AVX2__
  // Returns a mask of the newline bytes in `input`.
  [[clang::always_inline]] static auto Avx2MatchNewlines(__m256i input)
      -> uint32_t {
    // `_mm256_cmpeq_epi8` compares 32 bytes in parallel (`0xFF` if equal,
    // `0x00` otherwise), and `_mm256_movemask_epi8` packs bit 7 of each byte
    // into a 32-bit `int` mask.
    return static_cast<uint32_t>(
        _mm256_movemask_epi8(_mm256_cmpeq_epi8(input, _mm256_set1_epi8('\n'))));
  }

  // Returns a mask of the bytes in `input` that aren't identifier bytes.
  [[clang::always_inline]] static auto Avx2MatchIdEnds(__m256i input)
      -> uint32_t {
    const __m256i high_lut = HighLUT.Load();
    const __m256i low_lut = LowLUT.Load();

    // Do two LUT lookups and mask the results together to get the results for
    // both low and high nibbles. `_mm256_shuffle_epi8` only uses the low nibble
    // of each input byte as an index, so the low nibble lookup doesn't need to
    // mask the input. It also produces a zero byte for any input byte with the
    // high bit set, which classifies those bytes as non-ID bytes.
    __m256i low_mask = _mm256_shuffle_epi8(low_lut, input);
    // Note that the input needs to be masked to only include the high nibble or
    // we could end up with bit7 set forcing the result to a zero byte.
    __m256i input_high =
        _mm256_and_si256(_mm256_srli_epi32(input, 4), _mm256_set1_epi8(0x0f));
    __m256i high_mask = _mm256_shuffle_epi8(high_lut, input_high);
    __m256i mask = _mm256_and_si256(low_mask, high_mask);

    // Now compare to find the completely zero bytes.
    __m256i id_end_mask_vec = _mm256_cmpeq_epi8(mask, _mm256_setzero_si256());
    return static_cast<uint32_t>(_mm256_movemask_epi8(id_end_mask_vec));
  }
#elif __SSSE3__
  // Returns a mask of the newline bytes in `input`.
  [[clang::always_inline]] static auto SseMatchNewlines(__m128i input)
      -> uint32_t {
    // `_mm_cmpeq_epi8` compares 16 bytes in parallel (`0xFF` if equal, `0x00`
    // otherwise), and `_mm_movemask_epi8` packs bit 7 of each byte into the low
    // 16 bits of an `int` mask.
    return static_cast<uint32_t>(
        _mm_movemask_epi8(_mm_cmpeq_epi8(input, _mm_set1_epi8('\n'))));
  }

  // Returns a mask of the bytes in `input` that aren't identifier bytes.
  [[clang::always_inline]] static auto SseMatchIdEnds(__m128i input)
      -> uint32_t {
    const __m128i high_lut = HighLUT.Load();
    const __m128i low_lut = LowLUT.Load();

    // Do two LUT lookups and mask the results together to get the results for
    // both low and high nibbles. `_mm_shuffle_epi8` only uses the low nibble of
    // each input byte as an index, so the low nibble lookup doesn't need to
    // mask the input. It also produces a zero byte for any input byte with the
    // high bit set, which classifies those bytes as non-ID bytes.
    __m128i low_mask = _mm_shuffle_epi8(low_lut, input);
    // Note that the input needs to be masked to only include the high nibble or
    // we could end up with bit7 set forcing the result to a zero byte.
    __m128i input_high =
        _mm_and_si128(_mm_srli_epi32(input, 4), _mm_set1_epi8(0x0f));
    __m128i high_mask = _mm_shuffle_epi8(high_lut, input_high);
    __m128i mask = _mm_and_si128(low_mask, high_mask);

    // Now compare to find the completely zero bytes.
    __m128i id_end_mask_vec = _mm_cmpeq_epi8(mask, _mm_setzero_si128());
    return static_cast<uint32_t>(_mm_movemask_epi8(id_end_mask_vec));
  }
#elif __ARM_NEON
  // Returns `1 << (i % 8)` in each byte `i`, the bit for that byte within the
  // byte of a bit mask covering its group of 8 bytes.
  [[clang::always_inline]] static auto NeonBitWeights() -> uint8x16_t {
    return vreinterpretq_u8_u64(vdupq_n_u64(0x8040201008040201ULL));
  }

  // Starts reducing the four 16-byte vectors of a 64-byte block, each byte of
  // which is either all ones or all zeros, to a 64-bit mask of its all-ones
  // bytes. Returns a vector whose 16 bytes each hold four bits of the mask, in
  // order, so that a `vpaddq_u8` of its adjacent bytes produces the mask.
  [[clang::always_inline]] static auto NeonReduceQuarter(uint8x16_t v0,
                                                         uint8x16_t v1,
                                                         uint8x16_t v2,
                                                         uint8x16_t v3)
      -> uint8x16_t {
    // Neon has no equivalent of `_mm_movemask_epi8` to gather one bit from each
    // byte into a mask. Instead, mask each byte down to its bit from
    // `NeonBitWeights`, and then use two rounds of pairwise byte additions to
    // sum each run of four adjacent bytes into one byte. The additions can't
    // carry, as the bits being added are distinct.
    const uint8x16_t weights = NeonBitWeights();
    return vpaddq_u8(vpaddq_u8(vandq_u8(v0, weights), vandq_u8(v1, weights)),
                     vpaddq_u8(vandq_u8(v2, weights), vandq_u8(v3, weights)));
  }

  // Returns a vector with all bits set in each byte of `input` that isn't an
  // identifier byte.
  [[clang::always_inline]] static auto NeonMatchIdEnds(uint8x16_t input)
      -> uint8x16_t {
    const uint8x16_t high_lut = HighLUT.Load();
    const uint8x16_t low_lut = LowLUT.Load();

    // Do two LUT lookups and mask the results together to get the results for
    // both low and high nibbles. Unlike `_mm_shuffle_epi8` on x86, `vqtbl1q_u8`
    // only produces a zero byte for indices past the end of the LUT, so the
    // input needs to be masked to only include the low nibble.
    uint8x16_t low_mask =
        vqtbl1q_u8(low_lut, vandq_u8(input, vdupq_n_u8(0x0f)));
    uint8x16_t high_mask = vqtbl1q_u8(high_lut, vshrq_n_u8(input, 4));
    uint8x16_t mask = vandq_u8(low_mask, high_mask);

    // Now compare to find the completely zero bytes.
    return vceqzq_u8(mask);
  }
#endif

  [[clang::always_inline]] auto SimdClassify() const -> BlockMasks {
#if __AVX2__
    // Each 32-byte vector provides 32 bits of each mask.
    uint64_t newlines = Avx2MatchNewlines(vec_[0]) |
                        (uint64_t{Avx2MatchNewlines(vec_[1])} << 32);
    uint64_t id_ends =
        Avx2MatchIdEnds(vec_[0]) | (uint64_t{Avx2MatchIdEnds(vec_[1])} << 32);
    return {.newlines = newlines,
            .id_ends = id_ends,
            .newline_count = std::popcount(newlines)};
#elif __SSSE3__
    // Each 16-byte vector provides 16 bits of each mask.
    uint64_t newlines = 0;
    uint64_t id_ends = 0;
    for (int i = 0; i < 4; ++i) {
      newlines |= uint64_t{SseMatchNewlines(vec_[i])} << (16 * i);
      id_ends |= uint64_t{SseMatchIdEnds(vec_[i])} << (16 * i);
    }
    return {.newlines = newlines,
            .id_ends = id_ends,
            .newline_count = std::popcount(newlines)};
#elif __ARM_NEON
    // Unlike x86, Neon has no instruction to build a mask from each vector, and
    // moving data from the vector registers to the general-purpose registers is
    // slow. Instead, this classifies each byte of the four 16-byte vectors as
    // all ones or all zeros, and only reduces those to the two 64-bit masks at
    // the end, which then leave the vector registers together. `vceqq_u8`
    // compares 16 bytes in parallel (`0xFF` if equal, `0x00` otherwise).
    const uint8x16_t newline = vdupq_n_u8('\n');
    uint8x16_t nl0 = vceqq_u8(vec_[0], newline);
    uint8x16_t nl1 = vceqq_u8(vec_[1], newline);
    uint8x16_t nl2 = vceqq_u8(vec_[2], newline);
    uint8x16_t nl3 = vceqq_u8(vec_[3], newline);
    // Count the newlines in the vector registers, in parallel with building
    // their mask, rather than with `std::popcount`, which on Arm would move the
    // mask from a general-purpose register back to a vector register. Each
    // matching byte is `0xFF` (`-1` as `int8_t`), so the sum of all bytes is
    // the negated count.
    uint8x16_t neg_sum = vaddq_u8(vaddq_u8(nl0, nl1), vaddq_u8(nl2, nl3));
    int count = -vaddvq_s8(vreinterpretq_s8_u8(neg_sum));

    // `NeonMatchIdEnds` classifies the bytes that aren't identifier bytes in
    // the same way.
    uint8x16_t end0 = NeonMatchIdEnds(vec_[0]);
    uint8x16_t end1 = NeonMatchIdEnds(vec_[1]);
    uint8x16_t end2 = NeonMatchIdEnds(vec_[2]);
    uint8x16_t end3 = NeonMatchIdEnds(vec_[3]);

    // Reduce both sets of four vectors to quarter-sums, and then share the
    // final `vpaddq_u8` that produces both masks, so lane 0 holds `newlines`
    // and lane 1 holds `id_ends`.
    uint8x16_t q_nl = NeonReduceQuarter(nl0, nl1, nl2, nl3);
    uint8x16_t q_end = NeonReduceQuarter(end0, end1, end2, end3);
    uint64x2_t masks = vreinterpretq_u64_u8(vpaddq_u8(q_nl, q_end));
    return {.newlines = vgetq_lane_u64(masks, 0),
            .id_ends = vgetq_lane_u64(masks, 1),
            .newline_count = count};
#endif
  }
#endif  // CARBON_USE_BLOCK_SIMD
};

}  // namespace

auto Lexer::ScanIdentifier(llvm::StringRef source_text, ssize_t position) const
    -> llvm::StringRef {
  // This is performance sensitive, so check the identifier lexing benchmarks
  // for regressions when modifying it.
  CARBON_DCHECK(position >= 0);
  CARBON_DCHECK(position < static_cast<ssize_t>(source_text.size()));
  CARBON_DCHECK(
      IsIdStartByteTable[static_cast<unsigned char>(source_text[position])]);

  // The identifier ends at the first byte after its start byte whose bit is set
  // in `id_end_bitmap_`. The start byte is skipped rather than scanned because
  // it may be `$`, which can start an identifier but not continue one, and so
  // has its bit set.
  const size_t scan_start = static_cast<size_t>(position) + 1;

  // Load 8 bytes of the bitmap starting at the byte holding the bit for
  // `scan_start`, and shift that bit down to bit 0. This leaves bits for at
  // least the 57 bytes starting at `scan_start`, which is enough to find the
  // end of all keywords and nearly all identifiers.
  const uint8_t* bitmap = id_end_bitmap_.data();
  size_t bitmap_offset = scan_start / 8;
  CARBON_DCHECK(bitmap_offset + sizeof(uint64_t) <= id_end_bitmap_.size());
  uint64_t bits = llvm::support::endian::read64le(bitmap + bitmap_offset) >>
                  (scan_start % 8);
  size_t length;
  if (LLVM_LIKELY(bits != 0)) {
    // The start byte, and the bytes before the first set bit.
    length = 1 + std::countr_zero(bits);
  } else {
    // Otherwise, keep loading 8 bytes at a time, starting where the previous
    // load ended. This always stops by the end of the source text, because the
    // bits from there on are set.
    length = 1;
    do {
      bitmap_offset += sizeof(uint64_t);
      length += 64;
      bits = llvm::support::endian::read64le(bitmap + bitmap_offset);
    } while (bits == 0);
    // It's tempting to refactor the subtraction here to come before the loop,
    // or another refactoring. But we don't actually want the arithmetic here to
    // look that much like what we use in the hot `if` above. LLVM will
    // sometimes de-duplicate the code, which is a non-trivial regression for
    // the hot path because of how cold this path ends up being in practice.
    length += std::countr_zero(bits) - (scan_start % 8);
  }

  CARBON_DCHECK(static_cast<size_t>(position) + length <= source_text.size());
  return llvm::StringRef(source_text.data() + position, length);
}

// We use a collection of static member functions for table-based dispatch to
// lexer methods. These are named static member functions so that they show up
// helpfully in profiles and backtraces, but they tend to not contain the
// interesting logic and simply delegate to the relevant methods. All of their
// signatures need to be exactly the same however in order to ensure we can
// build efficient dispatch tables out of them. All of them end by doing a
// must-tail return call to this routine. It handles continuing the dispatch
// chain.
CARBON_PRESERVE_NONE static auto DispatchNext(Lexer& lexer,
                                              llvm::StringRef source_text,
                                              ssize_t position) -> void;

using DispatchFunctionT = decltype(DispatchNext);
using DispatchTableT = std::array<DispatchFunctionT*, 256>;

CARBON_PRESERVE_NONE static auto DispatchLexError(Lexer& lexer,
                                                  llvm::StringRef source_text,
                                                  ssize_t position) -> void {
  Lexer::LexResult result = lexer.LexError(source_text, position);
  CARBON_CHECK(result, "Failed to form a token!");
  // NOLINTNEXTLINE(readability-avoid-return-with-void-value): For musttail.
  [[clang::musttail]] return DispatchNext(lexer, source_text, position);
}

// Define a set of dispatch functions that simply forward to a method that
// lexes a token. If no token is matched, they tail-dispatch to the error token
// handler; otherwise they continue dispatching the next token.
#define CARBON_DISPATCH_LEX_TOKEN(LexMethod)                                   \
  CARBON_PRESERVE_NONE static auto Dispatch##LexMethod(                        \
      Lexer& lexer, llvm::StringRef source_text, ssize_t position) -> void {   \
    if (LLVM_LIKELY(lexer.LexMethod(source_text, position))) {                 \
      /* NOLINTNEXTLINE(readability-avoid-return-with-void-value) */           \
      [[clang::musttail]] return DispatchNext(lexer, source_text, position);   \
    }                                                                          \
    /* NOLINTNEXTLINE(readability-avoid-return-with-void-value) */             \
    [[clang::musttail]] return DispatchLexError(lexer, source_text, position); \
  }
CARBON_DISPATCH_LEX_TOKEN(LexKeywordOrIdentifier)
CARBON_DISPATCH_LEX_TOKEN(LexNumericLiteral)
CARBON_DISPATCH_LEX_TOKEN(LexStringLiteral)

CARBON_PRESERVE_NONE static auto DispatchLexHash(Lexer& lexer,
                                                 llvm::StringRef source_text,
                                                 ssize_t position) -> void {
  if (lexer.LexHash(source_text, position)) {
    // NOLINTNEXTLINE(readability-avoid-return-with-void-value): For musttail.
    [[clang::musttail]] return DispatchNext(lexer, source_text, position);
  }
  // NOLINTNEXTLINE(readability-avoid-return-with-void-value): For musttail.
  [[clang::musttail]] return DispatchLexStringLiteral(lexer, source_text,
                                                      position);
}

// A set of custom dispatch functions that preselect the symbol token to lex.
#define CARBON_DISPATCH_LEX_SYMBOL_TOKEN(LexMethod)                           \
  template <TokenKind::RawEnumType Kind>                                      \
  CARBON_PRESERVE_NONE static auto Dispatch##LexMethod##SymbolToken(          \
      Lexer& lexer, llvm::StringRef source_text, ssize_t position) -> void {  \
    Lexer::LexResult result = lexer.LexMethod##SymbolToken(                   \
        source_text, TokenKind::Make(Kind), position);                        \
    CARBON_CHECK(result, "Failed to form a token!");                          \
    /* NOLINTNEXTLINE(readability-avoid-return-with-void-value): musttail. */ \
    [[clang::musttail]] return DispatchNext(lexer, source_text, position);    \
  }
CARBON_DISPATCH_LEX_SYMBOL_TOKEN(LexOneChar)
CARBON_DISPATCH_LEX_SYMBOL_TOKEN(LexOpening)
CARBON_DISPATCH_LEX_SYMBOL_TOKEN(LexClosing)

template <char FirstChar>
CARBON_PRESERVE_NONE static auto DispatchLexSymbolToken(
    Lexer& lexer, llvm::StringRef source_text, ssize_t position) -> void {
  CARBON_DCHECK(source_text[position] == FirstChar);
  llvm::StringRef tail = source_text.substr(position + 1);
#define CARBON_SYMBOL_TOKEN(Name, Spelling)                                    \
  if constexpr ((Spelling)[0] == FirstChar) {                                  \
    if (tail.starts_with(llvm::StringLiteral(Spelling).substr(1))) {           \
      lexer.LexToken(TokenKind::Name, position);                               \
      /* NOLINTNEXTLINE(readability-avoid-return-with-void-value): musttail */ \
      [[clang::musttail]] return DispatchNext(                                 \
          lexer, source_text, position + (sizeof(Spelling) - 1));              \
    }                                                                          \
  }
#define CARBON_ONE_CHAR_SYMBOL_TOKEN(TokenName, Spelling)
#define CARBON_OPENING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, ClosingName)
#define CARBON_CLOSING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, OpeningName)
#include "toolchain/lex/token_kind.def"
  // NOLINTNEXTLINE(readability-avoid-return-with-void-value): For musttail.
  [[clang::musttail]] return DispatchLexError(lexer, source_text, position);
}

// Define a set of non-token dispatch functions that handle things like
// whitespace and comments.
#define CARBON_DISPATCH_LEX_NON_TOKEN(LexMethod)                             \
  CARBON_PRESERVE_NONE static auto Dispatch##LexMethod(                      \
      Lexer& lexer, llvm::StringRef source_text, ssize_t position) -> void { \
    lexer.LexMethod(source_text, position);                                  \
    [[clang::musttail]] return DispatchNext(lexer, source_text, position);   \
  }
CARBON_DISPATCH_LEX_NON_TOKEN(LexHorizontalWhitespace)
CARBON_DISPATCH_LEX_NON_TOKEN(LexVerticalWhitespace)
CARBON_DISPATCH_LEX_NON_TOKEN(LexCR)

CARBON_PRESERVE_NONE static auto DispatchLexCommentOrSlash(
    Lexer& lexer, llvm::StringRef source_text, ssize_t position) -> void {
  CARBON_DCHECK(source_text[position] == '/');

  // Both comments and slash symbols start with a `/`. We disambiguate with a
  // max-munch rule -- if the next character is another `/` then we lex it as
  // a comment start. If it isn't, then we lex as a slash. We also optimize
  // for the comment case as we expect that to be much more important for
  // overall lexer performance.
  if (LLVM_LIKELY(position + 1 < static_cast<ssize_t>(source_text.size()) &&
                  source_text[position + 1] == '/')) {
    lexer.LexComment(source_text, position);
    // NOLINTNEXTLINE(readability-avoid-return-with-void-value): For musttail.
    [[clang::musttail]] return DispatchNext(lexer, source_text, position);
  }

  // NOLINTNEXTLINE(readability-avoid-return-with-void-value): For musttail.
  [[clang::musttail]] return DispatchLexSymbolToken<'/'>(lexer, source_text,
                                                         position);
}

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
  // each symbol token spelling to dispatch to the symbol lexer for that
  // starting character. We'll override this with more specific dispatch below.
#define CARBON_SYMBOL_TOKEN(TokenName, Spelling) \
  table[(Spelling)[0]] = &DispatchLexSymbolToken<(Spelling)[0]>;
#include "toolchain/lex/token_kind.def"

  // Now special cased single-character symbols that are guaranteed to not
  // join with another symbol. These are grouping symbols, terminators,
  // or separators in the grammar and have a good reason to be
  // orthogonal to any other punctuation. We do this separately because this
  // needs to override some of the generic handling above, and provide a
  // custom token.
#define CARBON_ONE_CHAR_SYMBOL_TOKEN(TokenName, Spelling) \
  table[(Spelling)[0]] = &DispatchLexOneCharSymbolToken<TokenKind::TokenName>;
#define CARBON_OPENING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, ClosingName) \
  table[(Spelling)[0]] = &DispatchLexOpeningSymbolToken<TokenKind::TokenName>;
#define CARBON_CLOSING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, OpeningName) \
  table[(Spelling)[0]] = &DispatchLexClosingSymbolToken<TokenKind::TokenName>;
#include "toolchain/lex/token_kind.def"

  // Override the handling for `/` to consider comments as well as a `/`
  // symbol.
  table['/'] = &DispatchLexCommentOrSlash;

  table['_'] = &DispatchLexKeywordOrIdentifier;
  table['$'] = &DispatchLexKeywordOrIdentifier;
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

CARBON_PRESERVE_NONE static auto DispatchFileEnd(Lexer& lexer,
                                                 llvm::StringRef source_text,
                                                 ssize_t position) -> void {
  if (lexer.has_dump_sem_ir_ranges()) {
    // Incomplete ranges will use the next token for their end; we want that to
    // be `FileEnd` in this case, so check before adding `FileEnd`. The argument
    // is just the final character for diagnostic locations.
    // TODO: This offset may not be needed if `file_test` handled diagnostics
    // pointing at `.end()`.
    lexer.EndDumpSemIRRangeIfIncomplete(source_text.end() - 1);
  }

  // When we finish the source text, stop recursing. We also hint this so that
  // the tail-dispatch is optimized as that's essentially the loop back-edge
  // and this is the loop exit.
  lexer.LexFileEnd(source_text, position);
}

CARBON_PRESERVE_NONE static auto DispatchNext(Lexer& lexer,
                                              llvm::StringRef source_text,
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

  // NOLINTNEXTLINE(readability-avoid-return-with-void-value): For musttail.
  [[clang::musttail]] return DispatchFileEnd(lexer, source_text, position);
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

// Clears the lowest set bit of `mask`.
[[clang::always_inline]] static auto ClearLowestSetBit(uint64_t& mask) -> void {
  mask &= mask - 1;
}

auto Lexer::MakeLines(llvm::StringRef source_text) -> void {
  // We scan the source a 64-byte `SourceBlock` at a time, finding both the
  // newlines and the bytes that end identifiers with SIMD where available. The
  // scanning loop also uses carefully selected variables and the `ssize_t` type
  // for performance and code size.
  //
  // Note that searching for `'\n'` here works equally well for LF and CR+LF
  // line endings. Either way, it finds the end of the line and the start of the
  // next line. The lexer below will find the CR byte and peek to see the
  // following LF and jump to the next line correctly. However, this approach
  // does *not* support plain CR or LF+CR line endings. Nor does it support
  // vertical tab or other vertical whitespace.
  //
  // TODO: Eventually, we should extend this to have correct fallback support
  // for handling CR, LF+CR, vertical tab, and other esoteric vertical
  // whitespace as line endings. Notably, including *mixtures* of them. This
  // will likely be somewhat tricky, although `SourceBlock` provides a natural
  // place to detect their absence.
  constexpr ssize_t BlockSize = SourceBlock::Size;
  constexpr int DecodedPerStep = 4;
  const ssize_t size = source_text.size();
  const ssize_t num_blocks = (size + BlockSize - 1) / BlockSize;

  // Allocate 64 bits of `id_end_bitmap_` for each block, including any final
  // partial block, and a 64-bit sentinel. `ScanIdentifier` loads 8 bytes of the
  // bitmap starting at the byte holding the bit for any position up to `size`,
  // and the sentinel keeps those loads in bounds.
  id_end_bitmap_.resize_for_overwrite((num_blocks + 1) * sizeof(uint64_t));

  // Buffer the 32-bit byte offset following each `'\n'` (`int32_t` matches
  // `LineInfo::start` under Carbon's `< 2 GiB` source limit) in a contiguous
  // array, then `Reserve` and `Add` `LineInfo`s in a second pass so the block
  // loop only performs flat 32-bit stores rather than chunked `ValueStore::Add`
  // capacity checks.
  //
  // Initial capacity model (`size / 16 + 2 * BlockSize`):
  // - `size / 16` estimates one newline every 16 bytes (~2-3x denser than
  //   typical source lines), avoiding reallocation on ordinary files while
  //   keeping temporary memory to `size / 4` bytes.
  // - Maintaining at least `BlockSize` (64) unused slots before each iteration
  //   guarantees room for the worst case where all 64 bytes are `'\n'` (and up
  //   to 16 speculative 4-slot writes), replacing per-newline bounds checks
  //   with one cold check per 64-byte block that doubles the buffer on
  //   newline-dense inputs. Adding `2 * BlockSize` up front preserves that
  //   headroom even when `size / 16` is small.
  llvm::SmallVector<int32_t, 0> line_starts;
  line_starts.resize_for_overwrite(size / 16 + 2 * BlockSize);
  int32_t* line_starts_end = line_starts.data();

  // Any final partial block is scanned from a copy padded with zero bytes.
  // These are neither newlines nor identifier bytes, so they end any identifier
  // at the end of the source text.
  const ssize_t full_blocks_size = size - size % BlockSize;
  char padded_final_block[BlockSize] = {};
  if (full_blocks_size < size) {
    auto padded_block_text = source_text.substr(full_blocks_size);
    std::memcpy(padded_final_block, padded_block_text.data(),
                padded_block_text.size());
  }

  uint8_t* id_end_out = id_end_bitmap_.data();
  for (ssize_t block_start = 0; block_start < size; block_start += BlockSize) {
    if (LLVM_UNLIKELY(line_starts_end >
                      line_starts.data() + line_starts.size() - BlockSize)) {
      ssize_t count = line_starts_end - line_starts.data();
      // Double the size to preserve geometric growth when using
      // `resize_for_overwrite`.
      line_starts.resize_for_overwrite(line_starts.size() * 2);
      line_starts_end = line_starts.data() + count;
    }

    SourceBlock::BlockMasks masks =
        SourceBlock::Load(block_start < full_blocks_size
                              ? source_text.substr(block_start)
                              : llvm::StringRef(padded_final_block, BlockSize))
            .Classify();
    llvm::support::endian::write64le(id_end_out, masks.id_ends);
    id_end_out += sizeof(uint64_t);

    // Most 64-byte blocks contain 0 to 4 newlines. Unrolling a
    // `while (newlines)` loop with `#pragma clang loop unroll_count(4)` would
    // emit a data-dependent exit branch on every bit, suffering branch
    // mispredictions as newline counts vary from block to block. Instead,
    // advance `line_starts_end` by `newline_count` up front and use
    // `#pragma clang loop unroll(full)` to unroll 4 unconditional steps of
    // `std::countr_zero` (byte index of lowest set bit) and
    // `ClearLowestSetBit`. Any extra writes when fewer than 4 bits are set land
    // harmlessly in the `BlockSize` slack past `line_starts_end` and are
    // overwritten by the next block; blocks with > 4 newlines repeat the
    // unrolled loop (at most `64 / 4 = 16` times).
    int32_t base = block_start + 1;
    int32_t* out = line_starts_end;
    line_starts_end += masks.newline_count;
    uint64_t newlines = masks.newlines;
    do {
#pragma clang loop unroll(full)
      for (int i = 0; i < DecodedPerStep; ++i) {
        out[i] = base + std::countr_zero(newlines);
        ClearLowestSetBit(newlines);
      }
      out += DecodedPerStep;
    } while (LLVM_UNLIKELY(newlines != 0));
  }

  // Set all the bits of the sentinel, which follows the bits for the last
  // block. Together with the bits for the zero padding bytes of any final
  // partial block, this sets every bit at or past the end of the source text,
  // so scanning for the end of an identifier stops at the end of the source
  // text.
  llvm::support::endian::write64le(id_end_out, ~uint64_t{0});

  // The first line starts at the start of the file, and each newline starts
  // another line.
  buffer_.line_infos_.Reserve((line_starts_end - line_starts.data()) + 2);
  int32_t start = 0;
  buffer_.line_infos_.Add(LineInfo(start));
  for (const int32_t* next_start = line_starts.data();
       next_start != line_starts_end; ++next_start) {
    start = *next_start;
    buffer_.line_infos_.Add(LineInfo(start));
  }

  // If the last line wasn't empty, the file ends with an unterminated line.
  // Add an extra blank line so that we never need to handle the special case
  // of being on the last line inside the lexer and needing to not increment
  // to the next line.
  if (start != size) {
    buffer_.line_infos_.Add(LineInfo(size));
  }

  // Now that all the infos are allocated, get a fresh pointer to the first
  // info for use while lexing.
  line_index_ = LineIndex(0);
}

static auto SkipHorizontalWhitespaceScalar(llvm::StringRef source_text,
                                           ssize_t& position) -> void {
  const ssize_t size = source_text.size();
  while (position < size &&
         (source_text[position] == ' ' || source_text[position] == '\t')) {
    ++position;
  }
}

#if CARBON_USE_SIMD && __x86_64__
// Advances `position` past contiguous `' '` and `'\t'` bytes in `source_text`
// using 16-byte SSE2 scans, falling back to `SkipHorizontalWhitespaceScalar`
// for any `< 16`-byte tail near EOF.
static auto SkipHorizontalWhitespaceX86(llvm::StringRef source_text,
                                        ssize_t& position) -> void {
  const ssize_t size = source_text.size();
  const char* const data = source_text.data();
  // Broadcast `' '` and `'\t'` across all 16 byte lanes of 128-bit registers.
  const __m128i spaces = _mm_set1_epi8(' ');
  const __m128i tabs = _mm_set1_epi8('\t');
  while (LLVM_LIKELY(position + 16 <= size)) {
    // Unaligned 16-byte load; the loop guard guarantees `[position, position +
    // 16)` lies within `source_text` without requiring trailing buffer padding.
    __m128i chunk =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + position));
    // `_mm_cmpeq_epi8` sets matching byte lanes to `0xFF` and others to `0x00`;
    // OR-ing the two comparisons marks every `' '` or `'\t'` byte with `0xFF`.
    __m128i is_ws = _mm_or_si128(_mm_cmpeq_epi8(chunk, spaces),
                                 _mm_cmpeq_epi8(chunk, tabs));
    // `_mm_movemask_epi8` packs bit 7 of each byte into the low 16 bits of an
    // `int`. Invert and mask to 16 bits so set bits mark non-whitespace bytes.
    // Most indentation runs are shorter than 16 characters, so predict a hit
    // here and advance directly to the first non-whitespace byte with
    // `tzcnt`/`bsf`
    // (`std::countr_zero`) instead of looping per byte.
    uint32_t non_ws_mask =
        ~static_cast<uint32_t>(_mm_movemask_epi8(is_ws)) & 0xFFFFU;
    if (LLVM_LIKELY(non_ws_mask != 0)) {
      position += std::countr_zero(non_ws_mask);
      return;
    }
    position += 16;
  }
  SkipHorizontalWhitespaceScalar(source_text, position);
}
#endif

auto Lexer::SkipHorizontalWhitespace(llvm::StringRef source_text,
                                     ssize_t& position) -> void {
  // Handle adjacent whitespace quickly. This comes up frequently for example
  // due to indentation.
#if CARBON_USE_SIMD && __x86_64__
  SkipHorizontalWhitespaceX86(source_text, position);
#else
  SkipHorizontalWhitespaceScalar(source_text, position);
#endif
}

auto Lexer::AdvanceToLine(llvm::StringRef source_text, ssize_t& position,
                          LineIndex to_line_index) -> void {
  CARBON_DCHECK(to_line_index >= line_index_);
  line_index_ = to_line_index;
  auto& line_info = current_line_info();
  ssize_t line_start = line_info.start;
  position = line_start;
  SkipHorizontalWhitespace(source_text, position);
  line_info.indent = position - line_start;
}

auto Lexer::LexHorizontalWhitespace(llvm::StringRef source_text,
                                    ssize_t& position) -> void {
  CARBON_DCHECK(source_text[position] == ' ' || source_text[position] == '\t');
  NoteWhitespace();
  ++position;
  // Check if we're already done before an optimized skip routine.
  if (LLVM_LIKELY(position < static_cast<ssize_t>(source_text.size()) &&
                  source_text[position] != ' ' &&
                  source_text[position] != '\t')) {
    return;
  }
  // Skip runs using an optimized code path.
  SkipHorizontalWhitespace(source_text, position);
}

auto Lexer::LexVerticalWhitespace(llvm::StringRef source_text,
                                  ssize_t& position) -> void {
  NoteWhitespace();
  AdvanceToLine(source_text, position, next_line());
}

auto Lexer::DiagnoseCR(llvm::StringRef source_text, ssize_t position) -> void {
  CARBON_DIAGNOSTIC(UnsupportedLfCrLineEnding, Error,
                    "the LF+CR line ending is not supported, only LF and CR+LF "
                    "are supported");
  CARBON_DIAGNOSTIC(UnsupportedCrLineEnding, Error,
                    "a raw CR line ending is not supported, only LF and CR+LF "
                    "are supported");
  bool is_lfcr = position > 0 && source_text[position - 1] == '\n';
  // TODO: This diagnostic has an unfortunate snippet -- we should tweak the
  // snippet rendering to gracefully handle CRs.
  emitter_.Emit(source_text.begin() + position,
                is_lfcr ? UnsupportedLfCrLineEnding : UnsupportedCrLineEnding);
}

auto Lexer::LexCR(llvm::StringRef source_text, ssize_t& position) -> void {
  if (LLVM_LIKELY((position + 1) < static_cast<ssize_t>(source_text.size())) &&
      LLVM_LIKELY(source_text[position + 1] == '\n')) {
    // Skip to the vertical whitespace path, it will skip over both CR and LF.
    LexVerticalWhitespace(source_text, position);
    return;
  }

  DiagnoseCR(source_text, position);

  // Recover by treating the CR as a horizontal whitespace. This should make our
  // whitespace rules largely work and parse cleanly without disrupting the line
  // tracking data structures that were pre-built.
  NoteWhitespace();
  ++position;
}

auto Lexer::BeginDumpSemIRRange(const char* diag_loc) -> void {
  EndDumpSemIRRangeIfIncomplete(diag_loc);

  // The begin here will be the next token, which may be dump-sem-ir-begin. The
  // end will be assigned by either AddDumpSemIREnd or, if invalid,
  // EndDumpSemIRRangeIfIncomplete.
  buffer_.dump_sem_ir_ranges_.push_back(
      {.begin = TokenIndex(buffer_.size()), .end = TokenIndex::None});
}

auto Lexer::EndDumpSemIRRange(const char* diag_loc) -> void {
  if (buffer_.dump_sem_ir_ranges_.empty() ||
      buffer_.dump_sem_ir_ranges_.back().end != TokenIndex::None) {
    CARBON_DIAGNOSTIC(
        DumpSemIRRangeMissingBegin, Error,
        "missing `//@dump-sem-ir-begin` to match `//@dump-sem-ir-end`");
    emitter_.Emit(diag_loc, DumpSemIRRangeMissingBegin);
    return;
  }

  buffer_.dump_sem_ir_ranges_.back().end = TokenIndex(buffer_.size() - 1);
}

auto Lexer::EndDumpSemIRRangeIfIncomplete(const char* diag_loc) -> void {
  if (buffer_.dump_sem_ir_ranges_.empty() ||
      buffer_.dump_sem_ir_ranges_.back().end != TokenIndex::None) {
    return;
  }

  // The location here won't be closely associated with the start location.
  // However, this is a developer feature and not worth complexity to diagnose
  // better.
  CARBON_DIAGNOSTIC(
      DumpSemIRRangeMissingEnd, Error,
      "missing `//@dump-sem-ir-end` to match `//@dump-sem-ir-begin`");
  emitter_.Emit(diag_loc, DumpSemIRRangeMissingEnd);

  EndDumpSemIRRange(diag_loc);
}

auto Lexer::LexComment(llvm::StringRef source_text, ssize_t& position) -> void {
  CARBON_DCHECK(source_text.substr(position).starts_with("//"));
  int32_t comment_start = position;
  const auto line_info = current_line_info();

  // A comment is _trailing_ when it follows other content on its line, rather
  // than being the only non-whitespace on the line. Both kinds of comment are
  // lexed identically -- they run from `//` to the end of the line -- but we
  // record the distinction so that tooling can tell a comment annotating the
  // code on its line apart from one introducing the code below it.
  //
  // `line_info.indent` is the width of the line's leading whitespace, so
  // `line_info.start + line_info.indent` is the line's first non-whitespace
  // byte.
  const bool is_trailing = position != line_info.start + line_info.indent;

  // Check whether the `//` introducer is followed by something valid.
  bool is_valid_after_slashes = true;
  if (position + 2 < static_cast<ssize_t>(source_text.size()) &&
      LLVM_UNLIKELY(!IsSpace(source_text[position + 2]))) {
    llvm::StringRef comment_text = source_text.substr(position);
    // The `//@...` directives are tooling markers that are only meaningful as
    // full-line comments, so we only recognize them when not trailing.
    if (!is_trailing) {
      // A directive is also recorded as a comment: the tokens and comments
      // together reconstruct the source, so tooling such as the formatter
      // would otherwise silently drop the directive line. Despite being on a
      // cold path, calls below force inlining to keep fast-path register
      // allocation efficient.
      auto add_directive_comment_line = [&] {
        buffer_.AddComment(line_info.indent, comment_start,
                           buffer_.line_infos_.Get(next_line()).start,
                           /*is_trailing=*/false);
        AdvanceToLine(source_text, position, next_line());
      };
      if (comment_text.starts_with("//@include-in-dumps\n")) {
        buffer_.has_include_in_dumps_ = true;
        [[clang::always_inline]] add_directive_comment_line();
        return;
      }
      if (comment_text.starts_with("//@dump-sem-ir-begin\n")) {
        BeginDumpSemIRRange(comment_text.begin());
        [[clang::always_inline]] add_directive_comment_line();
        return;
      }
      if (comment_text.starts_with("//@dump-sem-ir-end\n")) {
        EndDumpSemIRRange(comment_text.begin());
        [[clang::always_inline]] add_directive_comment_line();
        return;
      }
    }
    CARBON_DIAGNOSTIC(NoWhitespaceAfterCommentIntroducer, Error,
                      "whitespace is required after '//'");
    emitter_.Emit(SourceLoc::Before(comment_text.begin() + 2),
                  NoWhitespaceAfterCommentIntroducer);

    // We use this to tweak the lexing of blocks below.
    is_valid_after_slashes = false;
  }

  // Skip over this line.
  LineIndex line_index = next_line();
  position = buffer_.line_infos_.Get(line_index).start;

  // A trailing comment runs to the end of its line. Unlike a full-line comment,
  // it can never be part of a block of identical comment lines, so we skip the
  // block-skipping optimization below and simply advance past this one line. We
  // also don't optimize for the case of a trailing comment as we expect them to
  // be relatively rare compared to other comment structures.
  if (LLVM_UNLIKELY(is_trailing)) {
    buffer_.AddComment(line_info.indent, comment_start, position,
                       /*is_trailing=*/true);
    // Unlike a full-line comment, a trailing comment can directly follow a
    // token that cleared the leading-whitespace flag (`x;// y`), so restore it
    // here for the next line's first token.
    NoteWhitespace();
    AdvanceToLine(source_text, position, line_index);
    return;
  }

  // A very common pattern is a long block of comment lines all with the same
  // indent and comment start. We skip these comment blocks in bulk for speed,
  // and with SIMD support short indents can be scanned extremely quickly; we
  // expect these to be the dominant cases.
  //
  // An invalid comment start was already diagnosed above, so its block is
  // instead skipped line by line below: a run of invalid comment lines lumps
  // into one block regardless of which invalid byte follows each `//`, keeping
  // the diagnostic noise to one per run, while a line whose introducer is
  // valid (whitespace, or a `//@...` directive) ends the run and is lexed on
  // its own.
  //
  // TODO: We should extend this to 32-byte SIMD on platforms with support.
  constexpr int MaxIndent = 13;
  const int indent = line_info.indent;
  const ssize_t first_line_start = line_info.start;
  ssize_t prefix_size = indent + (is_valid_after_slashes ? 3 : 2);
  auto skip_to_next_line = [this, indent, &line_index, &position] {
    // We're guaranteed to have a line here even on a comment on the last line
    // as we ensure there is an empty line structure at the end of every file.
    ++line_index.index;
    auto& next_line_info = buffer_.line_infos_.Get(line_index);
    next_line_info.indent = indent;
    position = next_line_info.start;
  };
  if (CARBON_USE_SIMD && is_valid_after_slashes &&
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
  } else {
    auto continues_block = [&](ssize_t position) -> bool {
      // Make sure the source text extends far enough for us to continue the
      // block.
      if (position + prefix_size > static_cast<ssize_t>(source_text.size())) {
        return false;
      }

      // Check that the prefix matches. Otherwise, the block is done.
      if (memcmp(source_text.data() + first_line_start,
                 source_text.data() + position, prefix_size) != 0) {
        return false;
      }

      // For something valid after `//`, we're done as we've ensured it was the
      // _same_ valid suffix in the `memcmp`.
      if (LLVM_LIKELY(is_valid_after_slashes)) {
        return true;
      }

      // Past here, the block is a run of invalid comment lines and the
      // matched prefix only covers this line's `//`, so examine what follows
      // those slashes: only another invalid comment line continues the block,
      // while a valid comment or directive ends it and is lexed on its own.

      // A `//` that ends the source is a valid comment, so it doesn't
      // continue the block.
      if (position + prefix_size == static_cast<ssize_t>(source_text.size())) {
        return false;
      }

      char after_slashes = source_text[position + prefix_size];

      // Whitespace after the `//` makes this line a valid comment, so it
      // doesn't continue the block.
      if (IsSpace(after_slashes)) {
        return false;
      }

      // An `@` makes this line a `//@...` directive that must be lexed on its
      // own to recognize its side effects, so it doesn't continue the block.
      if (after_slashes == '@') {
        return false;
      }

      // Anything else is another invalid comment line continuing the block.
      return true;
    };

    // Skip lines that are combined into a comment block.
    while (continues_block(position)) {
      skip_to_next_line();
    }
  }

  buffer_.AddComment(indent, comment_start, position, /*is_trailing=*/false);
  AdvanceToLine(source_text, position, line_index);
}

auto Lexer::CanFormRealLiteral() -> bool {
  // When a numeric literal immediately follows a `.` or `->` token, with no
  // intervening whitespace, a real literal is never formed.
  if (has_leading_space_) {
    return true;
  }
  // The flat kind array isn't built until lexing finishes, so look at the
  // token info directly.
  auto kind = buffer_.token_infos_.Get(buffer_.tokens().end()[-1]).kind();
  return kind != TokenKind::Period && kind != TokenKind::MinusGreater;
}

auto Lexer::LexNumericLiteral(llvm::StringRef source_text, ssize_t& position)
    -> LexResult {
  // This is only dispatched at a decimal digit, as `NumericLiteral::Lex`
  // requires.
  NumericLiteral literal = NumericLiteral::Lex(
      source_text.substr(position), [this] { return CanFormRealLiteral(); });

  // Capture the position before we step past the token.
  int32_t byte_offset = position;
  int token_size = literal.text().size();
  position += token_size;

  CARBON_KIND_SWITCH(literal.AddValue(*buffer_.value_stores_, emitter_)) {
    case CARBON_KIND(IntId id): {
      return LexTokenWithPayload(TokenKind::IntLiteral, id.AsTokenPayload(),
                                 byte_offset);
    }
    case CARBON_KIND(RealId id): {
      return LexTokenWithPayload(TokenKind::RealLiteral, id.index, byte_offset);
    }
    case CARBON_KIND(NumericLiteral::UnrecoverableError _): {
      return LexTokenWithPayload(TokenKind::Error, token_size, byte_offset);
    }
  }
}

static auto DiagnoseUnterminatedString(Diagnostics::Emitter<SourceLoc>& emitter,
                                       const StringLiteral& literal,
                                       bool is_char) -> void {
  CARBON_DIAGNOSTIC(UnterminatedString, Error,
                    "{0:character|string} literal is missing a terminator",
                    Diagnostics::BoolAsSelect);
  // The terminator goes where the literal stops.
  emitter
      .Build(SourceLoc::Before(literal.text().end()), UnterminatedString,
             is_char)
      .Attach(SourceLoc::Range(literal.text()))
      .Emit();
}

auto Lexer::LexStringLiteral(llvm::StringRef source_text, ssize_t& position)
    -> LexResult {
  std::optional<StringLiteral> literal =
      StringLiteral::Lex(source_text.substr(position));
  if (!literal) {
    return LexResult::NoMatch();
  }

  // Capture the position before we step past the token.
  int32_t byte_offset = position;
  position += literal->text().size();

  // Helper for error paths.
  auto lex_as_error = [&]() {
    return LexTokenWithPayload(TokenKind::Error, literal->text().size(),
                               byte_offset);
  };

  if (literal->kind() == StringLiteral::Kind::Char) {
    if (!literal->is_terminated()) {
      DiagnoseUnterminatedString(emitter_, *literal, /*is_char=*/true);
      return lex_as_error();
    }
    if (auto value = literal->ComputeCharLiteralValue(emitter_)) {
      return LexTokenWithPayload(TokenKind::CharLiteral, value->value,
                                 byte_offset);
    }
    return lex_as_error();
  }

  if (literal->has_invalid_introducer()) {
    // The literal covers only the malformed introducer line, so it spans no
    // lines and needs no line updates.
    CARBON_DIAGNOSTIC(MultiLineStringInvalidIntroducer, Error,
                      "invalid multi-line string literal introducer; a file "
                      "type indicator may not contain `'`, `#`, or `\"`, and "
                      "the content must begin on a new line");
    emitter_.Build(literal->text().begin(), MultiLineStringInvalidIntroducer)
        .Attach(SourceLoc::Range(literal->text()))
        .Emit();
    return lex_as_error();
  }

  // Update line and column information.
  if (literal->kind() != StringLiteral::Kind::SingleLine) {
    // A block string literal's content is indented to match its closing
    // delimiter: leading whitespace up to the delimiter's column is
    // indentation, and anything past it is part of the content. Each line the
    // literal spans is given the closing delimiter's column as its indentation.
    // The closing line's indentation must be correct because tokens and
    // comments can follow the closing delimiter and rely on it, for example to
    // detect a trailing comment. Multi-line literals are rare, so this cold
    // path need not be fast.
    LineIndex first_spanned_line(line_index_.index + 1);
    while (next_line_info().start < position) {
      ++line_index_.index;
    }
    // The closing delimiter is the first non-whitespace on the closing line, so
    // its column is that line's leading-whitespace width.
    LineInfo& closing_line_info = current_line_info();
    ssize_t indent_end = closing_line_info.start;
    SkipHorizontalWhitespace(source_text, indent_end);
    int32_t indent = indent_end - closing_line_info.start;
    for (int32_t i = first_spanned_line.index; i <= line_index_.index; ++i) {
      buffer_.line_infos_.Get(LineIndex(i)).indent = indent;
    }
  }

  // TODO: Recover from an unterminated literal in a way that lets parsing
  // continue as if it were terminated. The error token makes the parser also
  // report the enclosing construct as incomplete, for example a missing `;`
  // after `var s: str = "abc`.
  if (!literal->is_terminated()) {
    DiagnoseUnterminatedString(emitter_, *literal, /*is_char=*/false);
    return lex_as_error();
  }
  auto string_id = buffer_.value_stores_->string_literal_values().Add(
      literal->ComputeStringValue(buffer_.allocator_, emitter_));
  return LexTokenWithPayload(TokenKind::StringLiteral, string_id.index,
                             byte_offset);
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

  auto& opening_token_info = buffer_.token_infos_.Get(opening_token);
  if (LLVM_UNLIKELY(opening_token_info.kind() != kind.opening_symbol())) {
    has_mismatched_brackets_ = true;
    buffer_.token_infos_.Get(token).set_opening_token_index(TokenIndex::None);
    return token;
  }

  opening_token_info.set_closing_token_index(token);
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
    emitter_
        .Build(word.begin() + 1, TooManyTypeBitWidthDigits, suffix.size(),
               DigitLimit)
        .Attach(SourceLoc::Range(suffix))
        .Emit();
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

auto Lexer::LexWordAsDollarIntLiteralToken(llvm::StringRef word,
                                           int32_t byte_offset) -> LexResult {
  if (!word.starts_with('$')) {
    return LexResult::NoMatch();
  }

  if (!has_leading_space_) {
    auto prev_token = buffer_.tokens().end()[-1];
    // The flat kind array isn't built yet, so look at the token info directly.
    auto kind = buffer_.token_infos_.Get(prev_token).kind();
    if (kind.is_word()) {
      CARBON_DIAGNOSTIC(
          CharacterOnlyAllowedAtStart, Error,
          "`$` is only allowed at the start of a positional parameter");
      emitter_.Emit(word.begin(), CharacterOnlyAllowedAtStart);

      auto& prev_token_info = buffer_.token_infos_.Get(prev_token);
      auto prev_token_text_size = buffer_.GetTokenText(prev_token).size();
      prev_token_info = TokenInfo(TokenKind::Error, has_leading_space_,
                                  prev_token_text_size + word.size(),
                                  prev_token_info.byte_offset());
      return LexResult(TokenIndex(buffer_.token_infos_.size() - 1));
    }
  }

  auto diagnose_invalid_char = [&]() {
    CARBON_DIAGNOSTIC(
        InvalidCharacterInDollarIntLiteral, Error,
        "positional parameters can only contain digits after `$`");
    emitter_.Emit(word.begin() + 1, InvalidCharacterInDollarIntLiteral);
    return LexTokenWithPayload(TokenKind::Error, word.size(), byte_offset);
  };

  if (word.size() < 2) {
    CARBON_DIAGNOSTIC(DollarIntLiteralMissingNumber, Error,
                      "expected digits after `$`");
    emitter_.Emit(SourceLoc::Before(word.end()), DollarIntLiteralMissingNumber);
    return LexTokenWithPayload(TokenKind::Error, word.size(), byte_offset);
  }
  if (word[1] == '0' && word.size() > 2) {
    CARBON_DIAGNOSTIC(
        DollarIntLiteralLeadingZero, Error,
        "leading zeroes are not allowed in positional parameters");
    emitter_.Emit(word.begin() + 1, DollarIntLiteralLeadingZero);
    return LexTokenWithPayload(TokenKind::Error, word.size(), byte_offset);
  }
  if ((word[1] < '0' || word[1] > '9')) {
    return diagnose_invalid_char();
  }

  auto suffix = word.substr(1);
  int64_t suffix_value;
  constexpr ssize_t DigitLimit =
      std::numeric_limits<decltype(suffix_value)>::digits10;
  if (suffix.size() > DigitLimit) {
    // See if this is not actually a dollar int literal.
    if (!llvm::all_of(suffix, IsDecimalDigit)) {
      return diagnose_invalid_char();
    }

    // Otherwise, diagnose and produce an error token.
    CARBON_DIAGNOSTIC(TooManyDollarIntDigits, Error,
                      "found a positional parameter using {0} digits, "
                      "which is greater than the limit of {1}",
                      size_t, size_t);
    emitter_.Emit(word.begin() + 1, TooManyDollarIntDigits, suffix.size(),
                  DigitLimit);
    return LexTokenWithPayload(TokenKind::Error, word.size(), byte_offset);
  }

  suffix_value = suffix[0] - '0';
  for (char c : suffix.drop_front()) {
    if (!IsDecimalDigit(c)) {
      return diagnose_invalid_char();
    }
    suffix_value = suffix_value * 10 + (c - '0');
  }
  CARBON_CHECK(suffix_value >= 0);
  return LexTokenWithPayload(
      TokenKind::DollarIntLiteral,
      buffer_.value_stores_->ints().Add(suffix_value).AsTokenPayload(),
      byte_offset);
}

auto Lexer::LexKeywordOrIdentifier(llvm::StringRef source_text,
                                   ssize_t& position) -> LexResult {
  if (static_cast<unsigned char>(source_text[position]) > 0x7F) {
    // TODO: Need to add support for Unicode lexing.
    return LexResult::NoMatch();
  }
  CARBON_CHECK(
      IsIdStartByteTable[static_cast<unsigned char>(source_text[position])]);

  // Capture the position before we step past the token.
  int32_t byte_offset = position;

  // Take the valid characters off the front of the source buffer.
  llvm::StringRef identifier_text = ScanIdentifier(source_text, position);
  CARBON_DCHECK(!identifier_text.empty(), "Must have at least one character!");
  position += identifier_text.size();

  // Check if the text is a type literal, and if so form such a literal.
  if (LexResult result =
          LexWordAsTypeLiteralToken(identifier_text, byte_offset)) {
    return result;
  }
  if (LexResult result =
          LexWordAsDollarIntLiteralToken(identifier_text, byte_offset)) {
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
  auto& prev_token_info =
      buffer_.token_infos_.Get(TokenIndex(buffer_.token_infos_.size() - 1));

  // If the previous token isn't the identifier `r`, or the character after `#`
  // isn't the start of an identifier, this is not a raw identifier.
  if (prev_token_info.kind() != TokenKind::Identifier ||
      source_text[position - 1] != 'r' ||
      position + 1 == static_cast<ssize_t>(source_text.size()) ||
      !IsIdStartByteTable[static_cast<unsigned char>(
          source_text[position + 1])] ||
      prev_token_info.byte_offset() != static_cast<int32_t>(position) - 1) {
    return LexResult::NoMatch();
  }
  CARBON_DCHECK(buffer_.value_stores_->identifiers().Get(
                    prev_token_info.ident_id()) == "r");

  // Take the valid characters off the front of the source buffer.
  llvm::StringRef identifier_text = ScanIdentifier(source_text, position + 1);
  CARBON_DCHECK(!identifier_text.empty(), "Must have at least one character!");
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
        return !IsErrorEndByteTable[static_cast<unsigned char>(c)];
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
  emitter_.Build(error_text.begin(), UnrecognizedCharacters)
      .Attach(SourceLoc::Range(error_text))
      .Emit();

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
  CARBON_CHECK(current_line_info().start == 0);
  AdvanceToLine(source_text, position, /*to_line_index=*/LineIndex(0));
}

auto Lexer::LexFileEnd(llvm::StringRef source_text, ssize_t position) -> void {
  CARBON_CHECK(position == static_cast<ssize_t>(source_text.size()));
  // Check if the last line is empty and not the first line (and only). If so,
  // re-pin the last line to be the prior one so that diagnostics and editors
  // can treat newlines as terminators even though we internally handle them
  // as separators in case of a missing newline on the last line. We do this
  // here instead of detecting this when we see the newline to avoid more
  // conditions along that fast path.
  if (position == current_line_info().start && line_index_.index != 0) {
    --line_index_.index;
    --position;
  }

  // The end-of-file token is always considered to be whitespace.
  NoteWhitespace();

  LexToken(TokenKind::FileEnd, position);
}

auto Lexer::Finalize() -> void {
  // The token sequence is complete, except for any bracket recovery below, so
  // build the flat token arrays. Bracket recovery reads them, and rebuilds them
  // if it changes the sequence.
  buffer_.BuildTokenArrays();

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
  // `buffer` must not be null.
  explicit ErrorRecoveryBuffer(TokenizedBuffer* buffer) : buffer_(buffer) {}

  auto empty() const -> bool {
    return insertions_.empty() && !any_error_tokens_;
  }

  // Insert a recovery token of kind `kind` before `insert_before`. Multiple
  // insertions before the same token are applied in reverse of the order they
  // were requested (LIFO). Returns an id that `GetInsertedTokenIndex` maps to
  // the inserted token, once `Apply` has run.
  auto InsertBefore(TokenIndex insert_before, TokenKind kind) -> int {
    return AddInsertion(insert_before, kind, /*is_after=*/false);
  }

  // Insert a recovery token of kind `kind` after `insert_after`. Multiple
  // insertions after the same token are applied in the order they were
  // requested (FIFO). Returns an id as `InsertBefore` does.
  auto InsertAfter(TokenIndex insert_after, TokenKind kind) -> int {
    return AddInsertion(insert_after, kind, /*is_after=*/true);
  }

  // Replace the given token with an error token. We do this immediately,
  // because we don't benefit from buffering it. The flat token arrays are
  // brought up to date when `Apply` rebuilds them.
  auto ReplaceWithError(TokenIndex token) -> void {
    auto& token_info = buffer_->token_infos_.Get(token);
    int error_length = buffer_->GetTokenText(token).size();
    token_info = token_info.AsError(error_length);
    any_error_tokens_ = true;
  }

  // Merge the recovery tokens into the token list of the tokenized buffer.
  auto Apply() -> void {
    llvm::sort(insertions_);

    ValueStore<TokenIndex, TokenInfo> old_tokens =
        std::exchange(buffer_->token_infos_, {});
    int new_size = old_tokens.size() + insertions_.size();
    buffer_->token_infos_.Reserve(new_size);
    buffer_->recovery_tokens_.resize(new_size);
    inserted_token_index_.assign(insertions_.size(), TokenIndex::None);
    new_token_index_.assign(old_tokens.size(), TokenIndex::None);

    size_t ins_idx = 0;
    for (TokenIndex old_idx(0);
         old_idx.index < static_cast<int>(old_tokens.size()); ++old_idx.index) {
      if (ins_idx == insertions_.size() ||
          insertions_[ins_idx].target() != old_idx) {
        // Nothing is inserted before this token: it keeps its leading
        // whitespace and simply moves across.
        new_token_index_[old_idx.index] =
            buffer_->token_infos_.Add(old_tokens.Get(old_idx));
        continue;
      }

      // At least one insertion goes before this token. `insertions_` is sorted
      // by target, so they are exactly the next run of entries. The first of
      // them takes over the token's leading whitespace, so that the whitespace
      // stays at the start of the run.
      bool orig_leading_space = old_tokens.Get(old_idx).has_leading_space();
      bool is_first = true;
      for (; ins_idx < insertions_.size() &&
             insertions_[ins_idx].target() == old_idx;
           ++ins_idx) {
        TokenInfo info = insertions_[ins_idx].info.WithLeadingSpace(
            is_first ? orig_leading_space : false);
        is_first = false;
        TokenIndex added = buffer_->AddToken(info);
        buffer_->recovery_tokens_.set(added.index);
        inserted_token_index_[insertions_[ins_idx].insertion_order] = added;
      }
      new_token_index_[old_idx.index] = buffer_->token_infos_.Add(
          old_tokens.Get(old_idx).WithLeadingSpace(false));
    }

    // The token sequence changed, so rebuild the flat token arrays from it.
    buffer_->BuildTokenArrays();
  }

  // Maps a token index from before `Apply` to the same token's index after it.
  // `Apply` must have run, except that with no insertions to apply this is the
  // identity either way.
  auto GetNewTokenIndex(TokenIndex old_index) const -> TokenIndex {
    if (insertions_.empty()) {
      return old_index;
    }
    CARBON_CHECK(!new_token_index_.empty(),
                 "Token indexes are only renumbered by `Apply`.");
    return new_token_index_[old_index.index];
  }

  // Maps an id returned by `InsertBefore` or `InsertAfter` to the token that
  // insertion added. `Apply` must have run.
  auto GetInsertedTokenIndex(int insertion_id) const -> TokenIndex {
    return inserted_token_index_[insertion_id];
  }

  // Perform bracket matching to fix cross-references between tokens. This must
  // be done after all recovery is performed and all brackets match, because
  // recovery will change token indexes.
  auto FixTokenCrossReferences() -> void {
    llvm::SmallVector<TokenIndex> open_groups;
    for (auto token : buffer_->tokens()) {
      auto kind = buffer_->GetKind(token);
      if (kind.is_opening_symbol()) {
        open_groups.push_back(token);
      } else if (kind.is_closing_symbol()) {
        CARBON_CHECK(!open_groups.empty(), "Failed to balance brackets");
        auto opening_token = open_groups.pop_back_val();

        CARBON_CHECK(kind == buffer_->token_infos_.Get(opening_token)
                                 .kind()
                                 .closing_symbol(),
                     "Failed to balance brackets");
        auto& opening_token_info = buffer_->token_infos_.Get(opening_token);
        auto& closing_token_info = buffer_->token_infos_.Get(token);
        opening_token_info.set_closing_token_index(token);
        closing_token_info.set_opening_token_index(opening_token);
      }
    }
  }

 private:
  struct Insertion {
    TokenIndex anchor;
    bool is_after;
    // Where this insertion came in the sequence of requests, which is both the
    // id handed back to the caller and the tiebreak between insertions that go
    // in the same place.
    int insertion_order;
    TokenInfo info;

    // The token this insertion goes before, indexed in the stream as it was
    // before `Apply`. `Apply` sorts insertions by this.
    auto target() const -> TokenIndex {
      return is_after ? TokenIndex(anchor.index + 1) : anchor;
    }

    // Orders insertions as `Apply` emits them, so that sorting produces the
    // final token order. Insertions are grouped by the token they go before,
    // because `Apply` walks the old tokens in order, and within a group:
    //
    // - An insertion requested as "after the previous token" comes before one
    //   requested as "before this token", so that each stays on the side of
    //   the gap it was anchored to.
    // - A closing bracket comes before an opening one, so that a group ending
    //   in the gap is closed before a new group is opened in it.
    // - Otherwise the request order breaks the tie, in the direction that puts
    //   the earliest request nearest its anchor: first-requested first for
    //   insertions after a token, last-requested first for insertions before
    //   one.
    friend auto operator<(const Insertion& lhs, const Insertion& rhs) -> bool {
      if (lhs.target() != rhs.target()) {
        return lhs.target() < rhs.target();
      }
      if (lhs.is_after != rhs.is_after) {
        return lhs.is_after;
      }
      bool lhs_is_closing = lhs.info.kind().is_closing_symbol();
      bool rhs_is_closing = rhs.info.kind().is_closing_symbol();
      if (lhs_is_closing != rhs_is_closing) {
        return lhs_is_closing;
      }
      return lhs.is_after ? lhs.insertion_order < rhs.insertion_order
                          : lhs.insertion_order > rhs.insertion_order;
    }
  };

  auto AddInsertion(TokenIndex anchor, TokenKind kind, bool is_after) -> int {
    CARBON_CHECK(anchor.index >= 0, "Invalid anchor token index.");
    CARBON_CHECK(anchor.index < static_cast<int>(buffer_->token_infos_.size()),
                 "Cannot insert past the end of file token.");
    if (!is_after) {
      CARBON_CHECK(anchor.index > 0,
                   "Cannot insert before the start of file token.");
    }

    bool insert_leading_space = false;
    int32_t byte_offset = 0;

    if (!is_after) {
      insert_leading_space = buffer_->HasLeadingWhitespace(anchor);
      TokenIndex insert_after_idx(anchor.index - 1);
      const auto& prev_info = buffer_->token_infos_.Get(insert_after_idx);
      byte_offset = prev_info.byte_offset() +
                    buffer_->GetTokenText(insert_after_idx).size();
    } else {
      const auto& anchor_info = buffer_->token_infos_.Get(anchor);
      byte_offset =
          anchor_info.byte_offset() + buffer_->GetTokenText(anchor).size();
      TokenIndex next_tok(anchor.index + 1);
      if (next_tok.index < static_cast<int>(buffer_->token_infos_.size())) {
        insert_leading_space = buffer_->HasLeadingWhitespace(next_tok);
      }
    }

    int insertion_order = static_cast<int>(insertions_.size());
    insertions_.push_back({
        .anchor = anchor,
        .is_after = is_after,
        .insertion_order = insertion_order,
        .info = TokenInfo(kind, insert_leading_space, byte_offset),
    });
    return insertion_order;
  }

  TokenizedBuffer* buffer_;
  llvm::SmallVector<Insertion> insertions_;
  bool any_error_tokens_ = false;

  // Filled in by `Apply`: which token each insertion added, indexed by the id
  // `AddInsertion` handed out.
  llvm::SmallVector<TokenIndex> inserted_token_index_;

  // Filled in by `Apply`: the new index of each token that existed before it,
  // indexed by that token's old index.
  llvm::SmallVector<TokenIndex> new_token_index_;
};

// Returns true if the token kind forms a complete primary expression on its
// own: an identifier, a literal, `self`, a type keyword, and so on.
static auto IsLeafTokenKind(TokenKind kind) -> bool {
  switch (kind) {
    case TokenKind::Identifier:
    case TokenKind::IntLiteral:
    case TokenKind::RealLiteral:
    case TokenKind::StringLiteral:
    case TokenKind::CharLiteral:
    case TokenKind::IntTypeLiteral:
    case TokenKind::UnsignedIntTypeLiteral:
    case TokenKind::FloatTypeLiteral:
    case TokenKind::True:
    case TokenKind::False:
    case TokenKind::SelfValueIdentifier:
    case TokenKind::SelfTypeIdentifier:
    case TokenKind::Underscore:
    case TokenKind::Bool:
    case TokenKind::Type:
    case TokenKind::Auto:
    case TokenKind::Array:
    case TokenKind::Str:
    case TokenKind::Char:
    case TokenKind::Core:
    case TokenKind::Cpp:
      return true;
    default:
      return false;
  }
}

static auto CollectMismatchedBracketTokens(const TokenizedBuffer& buffer)
    -> llvm::SmallVector<MismatchedBracketToken> {
  llvm::SmallVector<MismatchedBracketToken> input_tokens;
  input_tokens.reserve(buffer.size());

  // Where the previous token in the full stream ended (comments are not
  // tokens, so they don't interrupt this).
  int32_t prev_end_byte = -1;
  int32_t prev_line_index = -1;

  for (auto it = buffer.tokens().begin(); it != buffer.tokens().end(); ++it) {
    TokenIndex token = *it;
    auto kind = buffer.GetKind(token);
    int32_t byte_offset = buffer.GetByteOffset(token);
    auto token_line = buffer.GetLine(token);
    bool has_wide_leading_space = prev_end_byte >= 0 &&
                                  token_line.index == prev_line_index &&
                                  byte_offset - prev_end_byte >= 2;
    prev_end_byte =
        byte_offset + static_cast<int32_t>(buffer.GetTokenText(token).size());
    prev_line_index = token_line.index;

    bool is_paren_keyword = false;
    bool is_else_keyword = false;
    BracketTokenKind bracket_kind;
    switch (kind) {
      case TokenKind::OpenParen:
        bracket_kind = BracketTokenKind::OpenParen;
        break;
      case TokenKind::OpenCurlyBrace:
        bracket_kind = BracketTokenKind::OpenCurlyBrace;
        break;
      case TokenKind::OpenSquareBracket:
        bracket_kind = BracketTokenKind::OpenSquareBracket;
        break;
      case TokenKind::CloseParen:
        bracket_kind = BracketTokenKind::CloseParen;
        break;
      case TokenKind::CloseCurlyBrace:
        bracket_kind = BracketTokenKind::CloseCurlyBrace;
        break;
      case TokenKind::CloseSquareBracket:
        bracket_kind = BracketTokenKind::CloseSquareBracket;
        break;
      case TokenKind::Semi:
        bracket_kind = BracketTokenKind::Semi;
        break;
      case TokenKind::Comma:
        bracket_kind = BracketTokenKind::Comma;
        break;
      case TokenKind::Period:
        bracket_kind = BracketTokenKind::Period;
        break;
      case TokenKind::If:
      case TokenKind::While:
      case TokenKind::For:
      case TokenKind::Match:
        bracket_kind = BracketTokenKind::StatementIntroducer;
        is_paren_keyword = true;
        break;
      case TokenKind::Else: {
        bracket_kind = BracketTokenKind::StatementIntroducer;
        // Only a statement `else` (followed by `{` or `if`) normally follows
        // a `}`; a ternary `if..then..else` is followed by an expression.
        auto else_next = std::next(it);
        if (else_next != buffer.tokens().end()) {
          auto next_kind = buffer.GetKind(*else_next);
          is_else_keyword = next_kind == TokenKind::OpenCurlyBrace ||
                            next_kind == TokenKind::If;
        }
        break;
      }
#define CARBON_DECL_INTRODUCER_TOKEN(kind, name) case TokenKind::kind:
#include "toolchain/lex/token_kind.def"
      case TokenKind::Abstract:
      case TokenKind::Case:
      case TokenKind::Continue:
      case TokenKind::Default:
      case TokenKind::Eval:
      case TokenKind::Extend:
      case TokenKind::Final:
      case TokenKind::Friend:
      case TokenKind::Inline:
      case TokenKind::MustEval:
      case TokenKind::Observe:
      case TokenKind::Override:
      case TokenKind::Private:
      case TokenKind::Protected:
      case TokenKind::Return:
      case TokenKind::Returned:
      case TokenKind::Static:
      case TokenKind::Virtual:
        bracket_kind = BracketTokenKind::StatementIntroducer;
        break;
      case TokenKind::Forall:
        bracket_kind = BracketTokenKind::Other;
        is_paren_keyword = true;
        break;
      case TokenKind::Equal:
        bracket_kind = BracketTokenKind::Assignment;
        break;
      case TokenKind::As:
        bracket_kind = BracketTokenKind::As;
        break;
      case TokenKind::MinusGreater:
      case TokenKind::Where:
        bracket_kind = BracketTokenKind::StructuralOp;
        break;
      case TokenKind::Ref:
      case TokenKind::Unused:
      case TokenKind::Template:
      case TokenKind::Const:
        bracket_kind = BracketTokenKind::ModifierKeyword;
        break;
      case TokenKind::EqualEqual:
      case TokenKind::ExclaimEqual:
      case TokenKind::Less:
      case TokenKind::LessEqual:
      case TokenKind::Greater:
      case TokenKind::GreaterEqual:
      case TokenKind::And:
      case TokenKind::Or:
        bracket_kind = BracketTokenKind::ComparisonOp;
        break;
      case TokenKind::FileStart:
        continue;
      case TokenKind::FileEnd:
        bracket_kind = BracketTokenKind::FileEnd;
        break;
      default:
        bracket_kind = IsLeafTokenKind(kind) ? BracketTokenKind::Leaf
                                             : BracketTokenKind::Other;
        break;
    }

    auto line = token_line;
    int32_t line_indent =
        (kind == TokenKind::FileEnd) ? 0 : buffer.GetIndentColumnNumber(line);

    auto next_it = std::next(it);
    bool is_at_end_of_line =
        (next_it == buffer.tokens().end() || buffer.GetLine(*next_it) != line);

    bool is_struct_brace = false;
    if (kind == TokenKind::OpenCurlyBrace) {
      if (next_it != buffer.tokens().end()) {
        auto next_kind = buffer.GetKind(*next_it);
        if (next_kind == TokenKind::Period ||
            next_kind == TokenKind::CloseCurlyBrace) {
          is_struct_brace = true;
        } else if (next_kind == TokenKind::Identifier) {
          auto next2_it = std::next(next_it);
          if (next2_it != buffer.tokens().end() &&
              buffer.GetKind(*next2_it) == TokenKind::Colon) {
            is_struct_brace = true;
          }
        }
      }
    }

    input_tokens.push_back(MismatchedBracketToken{
        .token_index = token,
        .kind = bracket_kind,
        .line = line.index,
        .line_indent = line_indent,
        .is_at_end_of_line = is_at_end_of_line,
        .is_struct_brace = is_struct_brace,
        .is_paren_keyword = is_paren_keyword,
        .is_else_keyword = is_else_keyword,
        .has_leading_space = buffer.HasLeadingWhitespace(token),
        .has_wide_leading_space = has_wide_leading_space,
    });
  }

  return input_tokens;
}

// The source position where `token` starts. Note that this can't go through
// `GetTokenText`, which returns the kind's fixed spelling rather than a pointer
// into the source for most token kinds.
static auto TokenStartPosition(const TokenizedBuffer& buffer, TokenIndex token)
    -> const char* {
  return buffer.source().text().begin() + buffer.GetByteOffset(token);
}

// The source position just past the end of `token`.
static auto TokenEndPosition(const TokenizedBuffer& buffer, TokenIndex token)
    -> const char* {
  return TokenStartPosition(buffer, token) + buffer.GetTokenText(token).size();
}

// Where a user would insert the bracket that `correction` proposes: just past
// the end of the token it goes after, rather than at the start of the token it
// goes before. For `f(x` on one line and `;` on the next, that points the
// suggestion at the position directly after the `x`, where the `)` belongs,
// instead of down at the `;`. An opening bracket and a `}` closing a multi-line
// scope are inserted elsewhere, as described below.
static auto BracketInsertionPosition(const TokenizedBuffer& buffer,
                                     const BracketCorrection& correction)
    -> const char* {
  TokenIndex insert_after =
      correction.fix_action == BracketFixAction::InsertAfter
          ? correction.fix_token_index
          : TokenIndex(correction.fix_token_index.index - 1);
  // Nothing precedes the first token, so an insertion before it goes at the
  // very start of the file.
  if (!insert_after.has_value()) {
    return TokenStartPosition(buffer, correction.fix_token_index);
  }

  const char* pos = TokenEndPosition(buffer, insert_after);
  const char* end = buffer.source().text().end();

  // Advances over the spaces starting at `from`, up to `limit` of them.
  auto skip_spaces = [end](const char* from, int32_t limit) {
    for (; limit > 0 && from != end && *from == ' '; --limit) {
      ++from;
    }
    return from;
  };
  constexpr int32_t NoLimit = std::numeric_limits<int32_t>::max();

  if (correction.fix_token_kind == TokenKind::CloseCurlyBrace) {
    // A `}` closing a multi-line scope goes on a line of its own, indented to
    // line up with the `{` it closes. Point at that column when the next line
    // is indented far enough to have a position there, and otherwise at the
    // start of it: a diagnostic can't name the virtual space to the right of
    // the end of a line, so this is as close as the source gets.
    if (const char* line_end = skip_spaces(pos, NoLimit);
        line_end != end && *line_end == '\n') {
      auto open_line = buffer.GetLine(correction.diagnostic_token_index);
      pos = skip_spaces(line_end + 1,
                        buffer.GetIndentColumnNumber(open_line) - 1);
    }
  } else if (correction.fix_token_kind.is_opening_symbol()) {
    // An opening bracket is inserted against the token after it when that token
    // is on the same line.
    TokenIndex insert_before(insert_after.index + 1);
    if (buffer.GetEndLoc(insert_after).first == buffer.GetLine(insert_before)) {
      pos = TokenStartPosition(buffer, insert_before);
    }
  }
  return pos;
}

// If brackets didn't pair or nest properly, find a set of places to insert
// brackets to fix the nesting, issue suitable diagnostics, and update the
// token list to describe the fixes.
auto Lexer::DiagnoseAndFixMismatchedBrackets() -> void {
  ErrorRecoveryBuffer fixes(&buffer_);
  auto input_tokens = CollectMismatchedBracketTokens(buffer_);
  auto corrections = FixMismatchedBrackets(input_tokens);

  // For each correction, the insertion it requested, or -1 if it made none.
  // Applying the fixes renumbers the token stream, so this is what lets the
  // corrections be reported against the indexes the caller will see.
  llvm::SmallVector<int> insertion_ids(corrections.size(), -1);

  for (auto [correction_index, correction] : llvm::enumerate(corrections)) {
    CARBON_DIAGNOSTIC(UnmatchedOpening, Error,
                      "opening symbol without a corresponding closing symbol");
    CARBON_DIAGNOSTIC(UnmatchedClosing, Error,
                      "closing symbol without a corresponding opening symbol");
    CARBON_DIAGNOSTIC_LABEL(PossiblyMissingBracketHere, Info,
                            "possibly missing `{0}` here", Lex::TokenKind);

    // The label names a position between two tokens, which only the source
    // pointer emitter can express.
    auto builder = emitter_.Build(
        TokenStartPosition(buffer_, correction.diagnostic_token_index),
        correction.diagnostic_kind == BracketDiagnosticKind::UnmatchedOpening
            ? UnmatchedOpening
            : UnmatchedClosing);

    if (correction.fix_action == BracketFixAction::ReplaceWithError) {
      fixes.ReplaceWithError(correction.fix_token_index);
    } else if (correction.is_tied) {
      // The cheapest repairs disagree about where this bracket goes, so give up
      // on the bracket rather than suggest one of them.
      fixes.ReplaceWithError(correction.diagnostic_token_index);
    } else {
      builder.Attach(
          SourceLoc::Before(BracketInsertionPosition(buffer_, correction)),
          PossiblyMissingBracketHere, correction.fix_token_kind);
      insertion_ids[correction_index] =
          correction.fix_action == BracketFixAction::InsertBefore
              ? fixes.InsertBefore(correction.fix_token_index,
                                   correction.fix_token_kind)
              : fixes.InsertAfter(correction.fix_token_index,
                                  correction.fix_token_kind);
    }

    builder.Emit();
  }

  buffer_.has_errors_ = true;

  if (!fixes.empty()) {
    fixes.Apply();
    fixes.FixTokenCrossReferences();
  }

  if (options_.bracket_corrections) {
    // Report the corrections against the token stream the caller sees, rather
    // than the one recovery started from, which no longer exists. A correction
    // that inserted a bracket names the inserted token itself; one that only
    // proposed a bracket, or replaced one with an error, names the token it
    // still refers to.
    for (auto [correction_index, correction] : llvm::enumerate(corrections)) {
      correction.diagnostic_token_index =
          fixes.GetNewTokenIndex(correction.diagnostic_token_index);
      correction.fix_token_index =
          insertion_ids[correction_index] >= 0
              ? fixes.GetInsertedTokenIndex(insertion_ids[correction_index])
              : fixes.GetNewTokenIndex(correction.fix_token_index);
    }
    *options_.bracket_corrections = std::move(corrections);
  }
}

auto Lex(SharedValueStores& value_stores, SourceBuffer& source,
         LexOptions options) -> TokenizedBuffer {
  auto* consumer =
      options.consumer ? options.consumer : &Diagnostics::ConsoleConsumer();
  auto tokens = Lexer(options, value_stores, source, *consumer).Lex();

  if (options.vlog_stream || options.dump_stream) {
    // Flush diagnostics before printing.
    consumer->Flush();
  }
  CARBON_VLOG_TO(options.vlog_stream, "*** Lex::TokenizedBuffer ***\n{0}",
                 tokens);
  if (options.dump_stream) {
    tokens.Print(*options.dump_stream, options.omit_file_boundary_tokens);
  }
  return tokens;
}

}  // namespace Carbon::Lex
