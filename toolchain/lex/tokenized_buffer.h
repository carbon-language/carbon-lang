// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LEX_TOKENIZED_BUFFER_H_
#define CARBON_TOOLCHAIN_LEX_TOKENIZED_BUFFER_H_

#include <cstdint>

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
#include "toolchain/diagnostics/diagnostic_emitter.h"
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

constexpr LineIndex LineIndex::None(NoneIndex);

// A comment, which can be a block of lines. These are tracked separately from
// tokens because they don't affect parse; if they were part of tokens, we'd
// need more general special-casing within token logic.
//
// Note that `CommentInfo` is used for an API to expose the comment.
struct CommentData {
  // Zero-based byte offset of the start of the comment within the source
  // buffer provided.
  int32_t start;

  // The comment's length.
  int32_t length;
};

// Indices for `CommentData` within the buffer.
struct CommentIndex : public IndexBase<CommentIndex> {
  static constexpr llvm::StringLiteral Label = "comment";
  static const CommentIndex None;
  using IndexBase::IndexBase;
};

constexpr CommentIndex CommentIndex::None(NoneIndex);

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

  auto GetKind(TokenIndex token) const -> TokenKind;
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

  // Returns whether the token was created as part of an error recovery effort.
  //
  // For example, a closing paren inserted to match an unmatched paren.
  auto IsRecoveryToken(TokenIndex token) const -> bool;

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

  // Returns true if the given range overlaps with an entry in
  // `dump_sem_ir_ranges_`. Must not be called when there are no ranges; query
  // `has_dump_sem_ir_ranges` first.
  auto OverlapsWithDumpSemIRRange(Lex::InclusiveTokenRange range) const -> bool;

  // Returns true if the buffer has errors that were detected at lexing time.
  auto has_errors() const -> bool { return has_errors_; }

  auto tokens() const -> llvm::iterator_range<TokenIterator> {
    return llvm::make_range(TokenIterator(TokenIndex(0)),
                            TokenIterator(TokenIndex(token_infos_.size())));
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

  class SourcePointerDiagnosticEmitter
      : public Diagnostics::Emitter<const char*> {
   public:
    explicit SourcePointerDiagnosticEmitter(Diagnostics::Consumer* consumer,
                                            const TokenizedBuffer* tokens)
        : Emitter(consumer), tokens_(tokens) {}

   protected:
    auto ConvertLoc(const char* loc, ContextFnT /*context_fn*/) const
        -> Diagnostics::ConvertedLoc override {
      return tokens_->SourcePointerToDiagnosticLoc(loc);
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

  // Adds the token and adjusts the expected tree size.
  auto AddToken(TokenInfo info) -> TokenIndex;

  auto GetTokenPrintWidths(TokenIndex token) const -> PrintWidths;
  auto PrintToken(llvm::raw_ostream& output_stream, TokenIndex token,
                  PrintWidths widths) const -> void;

  // Adds a comment. This uses the indent to potentially stitch together two
  // adjacent comments.
  auto AddComment(int32_t indent, int32_t start, int32_t end) -> void;

  // Used to allocate computed string literals.
  llvm::BumpPtrAllocator allocator_;

  SharedValueStores* value_stores_;
  SourceBuffer* source_;

  ValueStore<TokenIndex, TokenInfo> token_infos_;

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

  // A vector of flags for recovery tokens. If empty, there are none. When doing
  // token recovery, this will be extended to be indexable by token indices and
  // contain true for the tokens that were synthesized for recovery.
  llvm::BitVector recovery_tokens_;
};

inline auto TokenizedBuffer::GetKind(TokenIndex token) const -> TokenKind {
  return token_infos_.Get(token).kind();
}

inline auto TokenizedBuffer::HasLeadingWhitespace(TokenIndex token) const
    -> bool {
  return token_infos_.Get(token).has_leading_space();
}

inline auto TokenizedBuffer::HasTrailingWhitespace(TokenIndex token) const
    -> bool {
  TokenIterator it(token);
  ++it;
  return it != tokens().end() && token_infos_.Get(*it).has_leading_space();
}

inline auto TokenizedBuffer::AddToken(TokenInfo info) -> TokenIndex {
  expected_max_parse_tree_size_ += info.kind().expected_max_parse_tree_size();
  return token_infos_.Add(info);
}

}  // namespace Carbon::Lex

#endif  // CARBON_TOOLCHAIN_LEX_TOKENIZED_BUFFER_H_
