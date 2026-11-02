// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LEX_TOKENIZED_BUFFER_H_
#define CARBON_TOOLCHAIN_LEX_TOKENIZED_BUFFER_H_

#include <cstdint>

#include "common/check.h"
#include "common/ostream.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/raw_ostream.h"
#include "toolchain/base/index_base.h"
#include "toolchain/base/mem_usage.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/diagnostics/emitter.h"
#include "toolchain/lex/source_loc.h"
#include "toolchain/lex/token_index.h"
#include "toolchain/lex/token_info.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon::Lex {

class TokenizedBuffer;

struct LineInfo {
  explicit LineInfo(int32_t start) : start(start), indent(0) {}

  // Zero-based byte offset of the start of the line within the source buffer
  // provided.
  int32_t start;

  // The byte offset from the start of the line of the first non-whitespace
  // character.
  int32_t indent;
};

// A lightweight handle to a lexed `LineInfo` in a `TokenizedBuffer`.
//
// `LineIndex` objects are designed to be passed by value, not reference or
// pointer. They are also designed to be small and efficient to store in data
// structures.
//
// Each `LineIndex` object refers to a specific line in the source code that was
// lexed. They can be compared directly to establish that they refer to the
// same line or the relative position of different lines within the source.
//
// All other APIs to query a `LineIndex` are on the `TokenizedBuffer`.
struct LineIndex : public IndexBase<LineIndex> {
  static constexpr llvm::StringLiteral Label = "line";
  static const LineIndex None;
  using IndexBase::IndexBase;
};

inline constexpr LineIndex LineIndex::None(NoneIndex);

// A comment, which can be a block of lines. These are tracked separately from
// tokens because they don't affect parse; if they were part of tokens, we'd
// need more general special-casing within token logic.
//
// Note that `CommentInfo` is used for an API to expose the comment.
struct CommentData {
  // Zero-based byte offset of the start of the comment within the source
  // buffer provided.
  int32_t start;

  // The comment's length, in the low 31 bits. The high bit is stolen for
  // `is_trailing` below to keep `CommentData` at 8 bytes. A byte length is
  // non-negative and the source size is required to be less than `INT32_MAX`
  // (see `Lex`), so the length always fits in 31 bits.
  uint32_t length : 31;

  // Whether this is a _trailing_ comment: one that follows other content on its
  // line, rather than being the only non-whitespace on the line. Trailing
  // comments are never coalesced with adjacent comments.
  uint32_t is_trailing : 1;
};
static_assert(sizeof(CommentData) == 8, "CommentData should pack to 8 bytes");

// Indices for `CommentData` within the buffer.
struct CommentIndex : public IndexBase<CommentIndex> {
  static constexpr llvm::StringLiteral Label = "comment";
  static const CommentIndex None;
  using IndexBase::IndexBase;
};

inline constexpr CommentIndex CommentIndex::None(NoneIndex);

// Random-access iterator over comments within the buffer.
using CommentIterator = IndexIterator<CommentIndex>;

// Random-access iterator over tokens within the buffer.
using TokenIterator = IndexIterator<TokenIndex>;

// A token range which is inclusive of the begin and end.
struct InclusiveTokenRange {
  TokenIndex begin;
  TokenIndex end;
};

// A buffer of tokenized Carbon source code.
//
// This is constructed by lexing the source code text into a series of tokens.
// The buffer provides lightweight handles to tokens and other lexed entities,
// as well as iterations to walk the sequence of tokens found in the buffer.
//
// Lexing errors result in a potentially incomplete sequence of tokens and
// `HasError` returning true.
class TokenizedBuffer : public Printable<TokenizedBuffer> {
 public:
  // A comment, which can be a block of lines.
  //
  // This is the API version of `CommentData`.
  struct CommentInfo {
    // The comment's full text, including `//` symbols. This may have several
    // lines for block comments.
    llvm::StringRef text;

    // The comment's indent.
    int32_t indent;

    // The first line of the comment.
    LineIndex start_line;
  };

  // The number of `TokenKind::FileEnd` entries guaranteed to follow the last
  // token in the array returned by `kind_data`.
  static constexpr int32_t KindDataPadding = 64;

  auto GetKind(TokenIndex token) const -> TokenKind;

  // Returns a flat, contiguous array of the kinds of all tokens, indexed by
  // `TokenIndex`. The first `size()` entries are the kinds of the tokens, and
  // they are followed by at least `KindDataPadding` entries of
  // `TokenKind::FileEnd` so that callers can scan past the end of the tokens
  // without bounds checks, for example with SIMD loads.
  //
  // The returned pointer is invalidated by
  // `AddPostLexingRecoveryTokenAsIdentifier`.
  auto kind_data() const -> const TokenKind* { return token_kinds_.data(); }

  auto GetLine(TokenIndex token) const -> LineIndex;

  // Returns the 1-based line number.
  auto GetLineNumber(TokenIndex token) const -> int;

  // Returns the 1-based column number.
  auto GetColumnNumber(TokenIndex token) const -> int;

  // Returns the line and 1-based column number of the first character after
  // this token.
  auto GetEndLoc(TokenIndex token) const -> std::pair<LineIndex, int>;

  // Returns the source text lexed into this token.
  auto GetTokenText(TokenIndex token) const -> llvm::StringRef;

  // Returns the identifier associated with this token. The token kind must be
  // an `Identifier`.
  auto GetIdentifier(TokenIndex token) const -> IdentifierId;

  // Returns the value of an `IntLiteral` token.
  auto GetIntLiteral(TokenIndex token) const -> IntId;

  // Returns the value of an `RealLiteral` token.
  auto GetRealLiteral(TokenIndex token) const -> RealId;

  // Returns the value of a `StringLiteral` token.
  auto GetStringLiteralValue(TokenIndex token) const -> StringLiteralValueId;

  // Returns the value of a `CharLiteral` token.
  auto GetCharLiteralValue(TokenIndex token) const -> CharLiteralValue;

  // Returns the size specified in a `*TypeLiteral` token.
  auto GetTypeLiteralSize(TokenIndex token) const -> IntId;

  // Returns the closing token matched with the given opening token.
  //
  // The given token must be an opening token kind.
  auto GetMatchedClosingToken(TokenIndex opening_token) const -> TokenIndex;

  // Returns the opening token matched with the given closing token.
  //
  // The given token must be a closing token kind.
  auto GetMatchedOpeningToken(TokenIndex closing_token) const -> TokenIndex;

  // Returns whether the given token has leading whitespace.
  auto HasLeadingWhitespace(TokenIndex token) const -> bool;
  // Returns whether the given token has trailing whitespace.
  auto HasTrailingWhitespace(TokenIndex token) const -> bool;

  // Returns whether the token was spelled as a raw identifier.
  auto IsRawIdentifier(TokenIndex token) const -> bool;

  // Returns whether the token was created as part of an error recovery effort.
  //
  // For example, a closing paren inserted to match an unmatched paren.
  auto IsRecoveryToken(TokenIndex token) const -> bool;

  // Adds and returns the index of an identifier token formed from the spelling
  // of the given word token. This should only be used for error recovery
  // purposes, when a phase after the lexer determines that a word token was
  // intended to represent an identifier rather than its special meaning.
  //
  // TODO: Recover with a raw identifier only when it names something, perhaps
  // with a note suggesting `r#` otherwise. The parser recovers with one for any
  // reserved word it takes as a name, so check cascades, for example to "name
  // `r#Cpp` not found" on `class Cpp.C {}`.
  auto AddPostLexingRecoveryTokenAsIdentifier(TokenIndex token) -> TokenIndex;

  // Returns the 1-based indentation column number.
  auto GetIndentColumnNumber(LineIndex line) const -> int;

  auto GetByteOffset(TokenIndex token) const -> int32_t {
    return token_infos_.Get(token).byte_offset();
  }

  // Returns true if the token comes after the comment.
  auto IsAfterComment(TokenIndex token, CommentIndex comment_index) const
      -> bool;

  // Returns the comment's full text range.
  auto GetCommentText(CommentIndex comment_index) const -> llvm::StringRef;

  // Returns whether the comment is a trailing comment: one that follows other
  // content on its line.
  auto IsTrailingComment(CommentIndex comment_index) const -> bool;

  // Returns the line the comment begins on.
  auto GetLine(CommentIndex comment) const -> LineIndex;

  // Returns tokens as YAML. This prints the tracked token information on a
  // single line for each token. We use the single-line format so that output is
  // compact, and so that tools like `grep` are compatible.
  //
  // An example token looks like:
  //
  // - { index: 1, kind: 'Semi', line: 1, column: 1, indent: 1, spelling: ';' }
  auto Print(llvm::raw_ostream& out,
             bool omit_file_boundary_tokens = false) const -> void;

  // Prints a description of a single token.  See `Print` for details on the
  // format.
  auto PrintToken(llvm::raw_ostream& output_stream, TokenIndex token) const
      -> void;

  // Collects memory usage of members.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void;

  // Converts a token to a diagnostic location.
  auto TokenToDiagnosticLoc(TokenIndex token) const
      -> Diagnostics::ConvertedLoc;

  // Converts a run of tokens to a diagnostic location. A run spanning lines
  // covers the rest of its first line.
  auto TokenRangeToDiagnosticLoc(InclusiveTokenRange tokens) const
      -> Diagnostics::ConvertedLoc;

  // Converts the point just past `token`, where something missing after it
  // would be written, to a diagnostic location.
  auto PointAfterTokenToDiagnosticLoc(TokenIndex token) const
      -> Diagnostics::ConvertedLoc;

  // Returns true if the given range overlaps with an entry in
  // `dump_sem_ir_ranges_`. Must not be called when there are no ranges; query
  // `has_dump_sem_ir_ranges` first.
  auto OverlapsWithDumpSemIRRange(Lex::InclusiveTokenRange range) const -> bool;

  // Returns true if the buffer has errors that were detected at lexing time.
  auto has_errors() const -> bool { return has_errors_; }

  // Returns the tokens produced by lexing the source file. This includes any
  // recovery tokens inserted inline by lexing, such as matching brackets, but
  // excludes recovery tokens added after lexing finished.
  auto tokens() const -> llvm::iterator_range<TokenIterator> {
    return llvm::make_range(TokenIterator(TokenIndex(0)),
                            TokenIterator(TokenIndex(num_lexed_tokens())));
  }

  auto size() const -> int { return token_infos_.size(); }

  auto comments() const -> llvm::iterator_range<CommentIterator> {
    return llvm::make_range(CommentIterator(CommentIndex(0)),
                            CommentIterator(CommentIndex(comments_.size())));
  }

  auto comments_size() const -> size_t { return comments_.size(); }

  auto has_include_in_dumps() const -> bool { return has_include_in_dumps_; }

  // Returns true if any `DumpSemIRRange`s were provided.
  auto has_dump_sem_ir_ranges() const -> bool {
    return !dump_sem_ir_ranges_.empty();
  }

  // This is an upper bound on the number of output parse nodes in the absence
  // of errors.
  auto expected_max_parse_tree_size() const -> int {
    return expected_max_parse_tree_size_;
  }

  auto source() const -> const SourceBuffer& { return *source_; }

 private:
  friend class Lexer;

  class SourceLocDiagnosticEmitter : public Diagnostics::Emitter<SourceLoc> {
   public:
    explicit SourceLocDiagnosticEmitter(Diagnostics::Consumer* consumer,
                                        const TokenizedBuffer* tokens)
        : Emitter(consumer), tokens_(tokens) {}

   protected:
    auto ConvertLoc(SourceLoc loc, ContextFnT /*context_fn*/) const
        -> Diagnostics::ConvertedLoc override {
      auto converted =
          tokens_->SourcePointerToDiagnosticLoc(loc.text().begin());
      converted.loc.length = loc.text().size();
      return converted;
    }

   private:
    const TokenizedBuffer* tokens_;
  };

  class TokenDiagnosticEmitter : public Diagnostics::Emitter<TokenIndex> {
   public:
    explicit TokenDiagnosticEmitter(Diagnostics::Consumer* consumer,
                                    const TokenizedBuffer* tokens)
        : Emitter(consumer), tokens_(tokens) {}

   protected:
    auto ConvertLoc(TokenIndex token, ContextFnT /*context_fn*/) const
        -> Diagnostics::ConvertedLoc override {
      return tokens_->TokenToDiagnosticLoc(token);
    }

   private:
    const TokenizedBuffer* tokens_;
  };

  // Converts a pointer into the source to a diagnostic location.
  auto SourcePointerToDiagnosticLoc(const char* loc) const
      -> Diagnostics::ConvertedLoc;

  // Specifies minimum widths to use when printing a token's fields via
  // `printToken`.
  struct PrintWidths {
    // Widens `this` to the maximum of `this` and `new_width` for each
    // dimension.
    auto Widen(const PrintWidths& widths) -> void;

    int index;
    int kind;
    int line;
    int column;
    int indent;
  };

  // The constructor is merely responsible for trivial initialization of
  // members. A working object of this type is built with `Lex::Lex` so that its
  // return can indicate if an error was encountered while lexing.
  explicit TokenizedBuffer(SharedValueStores& value_stores
                           [[clang::lifetimebound]],
                           SourceBuffer& source [[clang::lifetimebound]])
      : value_stores_(&value_stores), source_(&source) {}

  auto FindLineIndex(int32_t byte_offset) const -> LineIndex;

  // Returns the number of tokens produced by lexing, excluding recovery tokens
  // added after lexing finished. This is the end of the `tokens()` range.
  auto num_lexed_tokens() const -> int32_t {
    return token_infos_.size() - post_lexing_recovery_tokens_;
  }

  // Adds the token and adjusts the expected tree size.
  auto AddToken(TokenInfo info) -> TokenIndex;

  // Builds `token_kinds_` and `token_has_leading_space_` from `token_infos_`,
  // including the `FileEnd` padding. The lexer calls this once its token
  // sequence is complete, and again if bracket recovery rewrites the sequence.
  // Until then, `GetKind`, `HasLeadingWhitespace`, and `HasTrailingWhitespace`
  // must not be used; the lexer reads `token_infos_` directly instead.
  auto BuildTokenArrays() -> void;

  // Adds a post-lexing recovery token and adjusts the buffer state.
  auto AddPostLexingRecoveryToken(TokenInfo info) -> TokenIndex;

  auto GetTokenPrintWidths(TokenIndex token) const -> PrintWidths;
  auto PrintToken(llvm::raw_ostream& output_stream, TokenIndex token,
                  PrintWidths widths) const -> void;

  // Adds a comment. For full-line comments (`is_trailing` is false), this uses
  // the indent to potentially stitch together two adjacent comments into a
  // single block. Trailing comments are never stitched, either onto a preceding
  // comment or as the target of a following one.
  auto AddComment(int32_t indent, int32_t start, int32_t end, bool is_trailing)
      -> void;

  // Used to allocate computed string literals.
  llvm::BumpPtrAllocator allocator_;

  SharedValueStores* value_stores_;
  SourceBuffer* source_;

  ValueStore<TokenIndex, TokenInfo> token_infos_;

  // The kind of each token in `token_infos_`, as a flat array indexed by the
  // token index, followed by `KindDataPadding` entries of `TokenKind::FileEnd`.
  // The kind is also stored in each `TokenInfo`, but this flat array makes the
  // kind, which is by far the hottest token query after lexing, a single
  // contiguous load rather than a chunked `ValueStore` lookup. It also allows
  // scanning kinds in bulk; see `kind_data`.
  //
  // This is built from `token_infos_` in one pass once lexing is complete, by
  // `BuildTokenArrays`, rather than maintained while lexing: the lexer's token
  // emission is hot enough that even a few extra instructions there cost more
  // than the pass.
  llvm::SmallVector<TokenKind, 0> token_kinds_;

  // Whether each token in `token_infos_` has leading whitespace, as a flat
  // array indexed by the token index. This duplicates
  // `TokenInfo::has_leading_space` for the same reason as `token_kinds_` and is
  // built at the same time.
  //
  // This intentionally uses a byte-per-token `SmallVector<bool, 0>` rather than
  // `llvm::BitVector` to facilitate vectorized population in
  // `BuildTokenArrays`.
  llvm::SmallVector<bool, 0> token_has_leading_space_;

  ValueStore<LineIndex, LineInfo> line_infos_;

  // Comments in the file.
  ValueStore<CommentIndex, CommentData> comments_;

  // Whether SemIR dumping is explicitly enabled for this file. This is marked
  // by `//@include-in-dumps`, and overrides other file-inclusion selection
  // choices. It can be combined with ranges.
  bool has_include_in_dumps_ = false;

  // A range of tokens marked by `//@dump-sem-ir-[begin|end]`.
  //
  // The particular syntax was chosen because it can be lexed efficiently. It
  // only occurs in invalid comment strings, so shouldn't slow down lexing of
  // correct code. It's also comment-like because its presence won't affect
  // parse/check.
  llvm::SmallVector<InclusiveTokenRange> dump_sem_ir_ranges_;

  // An upper bound on the number of parse tree nodes that we expect to be
  // created for the tokens in this buffer.
  int expected_max_parse_tree_size_ = 0;

  bool has_errors_ = false;

  // The number of recovery tokens created after lexing finished. These tokens
  // are included in `token_infos_`, but are excluded from the token sequence
  // produced by `tokens()`.
  int32_t post_lexing_recovery_tokens_ = 0;

  // A vector of flags for recovery tokens. If empty, there are none. When doing
  // token recovery, this will be extended to be indexable by token indices and
  // contain true for the tokens that were synthesized for recovery.
  llvm::BitVector recovery_tokens_;
};

inline auto TokenizedBuffer::GetKind(TokenIndex token) const -> TokenKind {
  CARBON_DCHECK(token.index >= 0 && token.index < size(), "{0}", token);
  CARBON_DCHECK(static_cast<size_t>(token.index) < token_kinds_.size(),
                "Token arrays haven't been built yet");
  return token_kinds_[token.index];
}

inline auto TokenizedBuffer::HasLeadingWhitespace(TokenIndex token) const
    -> bool {
  CARBON_DCHECK(token.index >= 0 && token.index < size(), "{0}", token);
  CARBON_DCHECK(
      static_cast<size_t>(token.index) < token_has_leading_space_.size(),
      "Token arrays haven't been built yet");
  return token_has_leading_space_[token.index];
}

inline auto TokenizedBuffer::HasTrailingWhitespace(TokenIndex token) const
    -> bool {
  CARBON_DCHECK(token.index >= 0 && token.index < size(), "{0}", token);
  // Trailing whitespace is the leading whitespace of the next lexed token, if
  // there is one.
  int32_t next = token.index + 1;
  return next < num_lexed_tokens() && HasLeadingWhitespace(TokenIndex(next));
}

inline auto TokenizedBuffer::AddToken(TokenInfo info) -> TokenIndex {
  expected_max_parse_tree_size_ += info.kind().expected_max_parse_tree_size();
  return token_infos_.Add(info);
}

}  // namespace Carbon::Lex

namespace Carbon {
extern template class ValueStore<Lex::TokenIndex, Lex::TokenInfo>;
extern template class ValueStore<Lex::LineIndex, Lex::LineInfo>;
extern template class ValueStore<Lex::CommentIndex, Lex::CommentData>;
}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_LEX_TOKENIZED_BUFFER_H_
