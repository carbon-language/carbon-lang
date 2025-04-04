// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_CONTEXT_H_
#define CARBON_TOOLCHAIN_PARSE_CONTEXT_H_

#include <optional>

#include "common/check.h"
#include "common/vlog.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/parse/precedence.h"
#include "toolchain/parse/state.h"
#include "toolchain/parse/tree.h"

namespace Carbon::Parse {

// An amount by which to look ahead of the current token. Lookahead should be
// used sparingly, and unbounded lookahead should be avoided.
//
// TODO: Decide whether we want to avoid lookahead altogether.
//
// NOLINTNEXTLINE(performance-enum-size): Deliberately matches index size.
enum class Lookahead : int32_t {
  CurrentToken = 0,
  NextToken = 1,
};

// Context and shared functionality for parser handlers. See state.def for state
// documentation.
class Context {
 public:
  // A token-based emitter for use during parse.
  class DiagnosticEmitter : public Diagnostics::Emitter<Lex::TokenIndex> {
   public:
    explicit DiagnosticEmitter(Diagnostics::Consumer* consumer,
                               Context* context)
        : Emitter(consumer), context_(context) {}

   protected:
    // Applies the `position_` to the `last_byte_offset` returned by
    // `TokenToDiagnosticLoc`.
    auto ConvertLoc(Lex::TokenIndex token, ContextFnT /*context_fn*/) const
        -> Diagnostics::ConvertedLoc override {
      auto converted = context_->tokens().TokenToDiagnosticLoc(token);
      converted.last_byte_offset =
          std::max(converted.last_byte_offset,
                   context_->tokens().GetByteOffset(*context_->position()));
      return converted;
    }

   private:
    Context* context_;
  };

  // Possible operator fixities for errors.
  enum class OperatorFixity : int8_t { Prefix, Infix, Postfix };

  // Possible return values for FindListToken.
  enum class ListTokenKind : int8_t { Comma, Close, CommaClose };

  // Used for restricting ordering of `package` and `import` declarations.
  enum class PackagingState : int8_t {
    FileStart,
    InImports,
    AfterNonPackagingDecl,
    // A warning about `import` placement has been issued so we don't keep
    // issuing more (when `import` is repeated) until more non-`import`
    // declarations come up.
    InImportsAfterNonPackagingDecl,
  };

  // Used to track state on state_stack_.
  // TODO: Rename to `State`.
  struct StateStackEntry : public Printable<StateStackEntry> {
    // Prints state information for verbose output.
    auto Print(llvm::raw_ostream& output) const -> void {
      output << kind << " @" << token << " subtree_start=" << subtree_start
             << " has_error=" << has_error;
    }

    // The state.
    StateKind kind;
    // Set to true to indicate that an error was found, and that contextual
    // error recovery may be needed.
    bool has_error : 1 = false;

    // Set to true to indicate that this state is handling a pattern nested
    // inside a `var` pattern.
    // TODO: This is meaningful only for patterns, and the precedence fields
    // are meaningful only for expressions, so expressing them as a union
    // could help catch errors.
    bool in_var_pattern : 1 = false;

    // Precedence information used by expression states in order to determine
    // operator precedence. The ambient_precedence deals with how the expression
    // should interact with outside context, while the lhs_precedence is
    // specific to the lhs of an operator expression.
    PrecedenceGroup ambient_precedence = PrecedenceGroup::ForTopLevelExpr();
    PrecedenceGroup lhs_precedence = PrecedenceGroup::ForTopLevelExpr();

    // A token providing context based on the subtree. This will typically be
    // the first token in the subtree, but may sometimes be a token within. It
    // will typically be used for the subtree's root node.
    Lex::TokenIndex token;
    // The offset within the Tree of the subtree start.
    int32_t subtree_start;
  };

  // We expect StateStackEntry to fit into 12 bytes:
  //   state = 1 byte
  //   has_error and in_var_pattern = 1 byte
  //   ambient_precedence = 1 byte
  //   lhs_precedence = 1 byte
  //   token = 4 bytes
  //   subtree_start = 4 bytes
  // If it becomes bigger, it'd be worth examining better packing; it should be
  // feasible to pack the 1-byte entries more tightly.
  static_assert(sizeof(StateStackEntry) == 12,
                "StateStackEntry has unexpected size!");

  explicit Context(Tree* tree, Lex::TokenizedBuffer* tokens,
                   Diagnostics::Consumer* consumer,
                   llvm::raw_ostream* vlog_stream);

  // Adds a node to the parse tree that has no children (a leaf).
  auto AddLeafNode(NodeKind kind, Lex::TokenIndex token, bool has_error = false)
      -> void {
    AddNode(kind, token, has_error);
  }

  // Adds a node to the parse tree that has children.
  auto AddNode(NodeKind kind, Lex::TokenIndex token, bool has_error) -> void {
    CARBON_CHECK(has_error || (kind != NodeKind::InvalidParse &&
                               kind != NodeKind::InvalidParseStart &&
                               kind != NodeKind::InvalidParseSubtree),
                 "{0} nodes must always have an error", kind);
    tree_->node_impls_.push_back(Tree::NodeImpl(kind, has_error, token));
  }

  // Adds a node and returns its typed NodeId.
  template <const Parse::NodeKind& Kind>
  auto AddNode(Lex::TokenIndex token, bool has_error) -> NodeIdForKind<Kind> {
    AddNode(Kind, token, has_error);
    return NodeIdForKind<Kind>::UnsafeMake(
        NodeId(tree_->node_impls_.size() - 1));
  }

  // Adds an invalid parse node.
  auto AddInvalidParse(Lex::TokenIndex token) -> void {
    AddNode(NodeKind::InvalidParse, token, /*has_error=*/true);
  }

  // Replaces the placeholder node at the indicated position with a leaf node.
  //
  // To reserve a position in the parse tree, you may add a placeholder parse
  // node using code like:
  //   ```
  //   context.PushState(State::WillFillInPlaceholder);
  //   context.AddLeafNode(NodeKind::Placeholder, *context.position());
  //   ```
  // It may be replaced with the intended leaf parse node with code like:
  //   ```
  //   auto HandleWillFillInPlaceholder(Context& context) -> void {
  //     auto state = context.PopState();
  //     context.ReplacePlaceholderNode(state.subtree_start, /* replacement */);
  //   }
  //   ```
  auto ReplacePlaceholderNode(int32_t position, NodeKind kind,
                              Lex::TokenIndex token, bool has_error = false)
      -> void;

  // Returns the current position and moves past it.
  auto Consume() -> Lex::TokenIndex { return *(position_++); }

  // Consumes the current token. Does not return it.
  auto ConsumeAndDiscard() -> void { ++position_; }

  // Parses an open paren token, possibly diagnosing if necessary. Creates a
  // leaf parse node of the specified start kind. The default_token is used when
  // there's no open paren. Returns the open paren token if it was found.
  auto ConsumeAndAddOpenParen(Lex::TokenIndex default_token,
                              NodeKind start_kind)
      -> std::optional<Lex::TokenIndex>;

  // Parses a closing symbol corresponding to the opening symbol
  // `expected_open`, possibly skipping forward and diagnosing if necessary.
  // Creates a parse node of the specified close kind. If `expected_open` is not
  // an opening symbol, the parse node will be associated with `state.token`,
  // no input will be consumed, and no diagnostic will be emitted.
  auto ConsumeAndAddCloseSymbol(Lex::TokenIndex expected_open,
                                StateStackEntry state, NodeKind close_kind)
      -> void;

  // Composes `ConsumeIf` and `AddLeafNode`, returning false when ConsumeIf
  // fails.
  auto ConsumeAndAddLeafNodeIf(Lex::TokenKind token_kind, NodeKind node_kind)
      -> bool;

  // Returns the current position and moves past it. Requires the token is the
  // expected kind.
  auto ConsumeChecked(Lex::TokenKind kind) -> Lex::TokenIndex;

  // If the current position's token matches this `Kind`, returns it and
  // advances to the next position. Otherwise returns an empty optional.
  auto ConsumeIf(Lex::TokenKind kind) -> std::optional<Lex::TokenIndex> {
    if (!PositionIs(kind)) {
      return std::nullopt;
    }
    return Consume();
  }

  // Find the next token of any of the given kinds at the current bracketing
  // level.
  auto FindNextOf(std::initializer_list<Lex::TokenKind> desired_kinds)
      -> std::optional<Lex::TokenIndex>;

  // If the token is an opening symbol for a matched group, skips to the matched
  // closing symbol and returns true. Otherwise, returns false.
  auto SkipMatchingGroup() -> bool;

  // Skips forward to move past the likely end of a declaration or statement.
  //
  // Looks forward, skipping over any matched symbol groups, to find the next
  // position that is likely past the end of a declaration or statement. This
  // is a heuristic and should only be called when skipping past parse errors.
  //
  // The strategy for recognizing when we have likely passed the end of a
  // declaration or statement:
  // - If we get to a close curly brace, we likely ended the entire context.
  // - If we get to a semicolon, that should have ended the declaration or
  //   statement.
  // - If we get to a new line from the `SkipRoot` token, but with the same or
  //   less indentation, there is likely a missing semicolon. Continued
  //   declarations or statements across multiple lines should be indented.
  //
  // Returns the last token consumed.
  auto SkipPastLikelyEnd(Lex::TokenIndex skip_root) -> Lex::TokenIndex;

  // Skip forward to the given token. Verifies that it is actually forward.
  auto SkipTo(Lex::TokenIndex t) -> void;

  // Returns true if the current token satisfies the lexical validity rules
  // for an infix operator.
  auto IsLexicallyValidInfixOperator() -> bool;

  // Determines whether the current trailing operator should be treated as
  // infix.
  auto IsTrailingOperatorInfix() -> bool;

  // Diagnoses whether the current token is not written properly for the given
  // fixity. For example, because mandatory whitespace is missing. Regardless of
  // whether there's an error, it's expected that parsing continues.
  auto DiagnoseOperatorFixity(OperatorFixity fixity) -> void;

  // If the current position is a `,`, consumes it, adds the provided token, and
  // returns `Comma`. Returns `Close` if the current position is close_token
  // (for example, `)`). `CommaClose` indicates it found both (for example,
  // `,)`). Handles cases where invalid tokens are present by advancing the
  // position, and may emit errors. Pass already_has_error in order to suppress
  // duplicate errors.
  auto ConsumeListToken(NodeKind comma_kind, Lex::TokenKind close_kind,
                        bool already_has_error) -> ListTokenKind;

  // Gets the kind of the next token to be consumed. If `lookahead` is
  // provided, it specifies which token to inspect.
  auto PositionKind(Lookahead lookahead = Lookahead::CurrentToken) const
      -> Lex::TokenKind {
    return tokens_->GetKind(position_[static_cast<int32_t>(lookahead)]);
  }

  // Tests whether the next token to be consumed is of the specified kind. If
  // `lookahead` is provided, it specifies which token to inspect.
  auto PositionIs(Lex::TokenKind kind,
                  Lookahead lookahead = Lookahead::CurrentToken) const -> bool {
    return PositionKind(lookahead) == kind;
  }

  // Pops the state and keeps the value for inspection.
  auto PopState() -> StateStackEntry {
    auto back = state_stack_.pop_back_val();
    CARBON_VLOG("Pop {0}: {1}\n", state_stack_.size(), back);
    return back;
  }

  // Pops the state and discards it.
  auto PopAndDiscardState() -> void {
    CARBON_VLOG("PopAndDiscard {0}: {1}\n", state_stack_.size() - 1,
                state_stack_.back());
    state_stack_.pop_back();
  }

  // Pushes a new state with the current position for context.
  auto PushState(StateKind kind) -> void { PushState(kind, *position_); }

  // Pushes a new state with a specific token for context. Used when forming a
  // new subtree when the current position isn't the start of the subtree.
  auto PushState(StateKind kind, Lex::TokenIndex token) -> void {
    PushState({.kind = kind, .token = token, .subtree_start = tree_->size()});
  }

  // Pushes a new expression state with specific precedence.
  auto PushStateForExpr(PrecedenceGroup ambient_precedence) -> void {
    PushState({.kind = StateKind::Expr,
               .ambient_precedence = ambient_precedence,
               .token = *position_,
               .subtree_start = tree_->size()});
  }

  // Pushes a new state with detailed precedence for expression resume states.
  auto PushStateForExprLoop(StateKind kind, PrecedenceGroup ambient_precedence,
                            PrecedenceGroup lhs_precedence) -> void {
    PushState({.kind = kind,
               .ambient_precedence = ambient_precedence,
               .lhs_precedence = lhs_precedence,
               .token = *position_,
               .subtree_start = tree_->size()});
  }

  // Pushes a new state for handling a pattern. `in_var_pattern` indicates
  // whether that pattern is nested inside a `var` pattern.
  auto PushStateForPattern(StateKind kind, bool in_var_pattern) -> void {
    PushState({.kind = kind,
               .in_var_pattern = in_var_pattern,
               .token = *position_,
               .subtree_start = tree_->size()});
  }

  // Pushes a constructed state onto the stack.
  auto PushState(StateStackEntry state) -> void {
    CARBON_VLOG("Push {0}: {1}\n", state_stack_.size(), state);
    state_stack_.push_back(state);
    CARBON_CHECK(state_stack_.size() < (1 << 20),
                 "Excessive stack size: likely infinite loop");
  }

  // Pushes a constructed state onto the stack, with a different kind.
  auto PushState(StateStackEntry state_entry, StateKind kind) -> void {
    state_entry.kind = kind;
    PushState(state_entry);
  }

  // Propagates an error up the state stack, to the parent state.
  auto ReturnErrorOnState() -> void { state_stack_.back().has_error = true; }

  // Adds a node for a declaration's semicolon. Includes error recovery when the
  // token is not a semicolon, using `decl_kind` and `is_def_allowed` to inform
  // diagnostics.
  auto AddNodeExpectingDeclSemi(StateStackEntry state, NodeKind node_kind,
                                Lex::TokenKind decl_kind, bool is_def_allowed)
      -> void;

  // Emits a diagnostic for a declaration missing a semi.
  auto DiagnoseExpectedDeclSemi(Lex::TokenKind expected_kind) -> void;

  // Emits a diagnostic for a declaration missing a semi or definition.
  auto DiagnoseExpectedDeclSemiOrDefinition(Lex::TokenKind expected_kind)
      -> void;

  // Handles error recovery in a declaration, particularly before any possible
  // definition has started (although one could be present). Recover to a
  // semicolon when it makes sense as a possible end, otherwise use the
  // introducer token for the error.
  auto RecoverFromDeclError(StateStackEntry state, NodeKind node_kind,
                            bool skip_past_likely_end) -> void;

  // Handles parsing of the library name. Returns the name's ID on success,
  // which may be invalid for `default`.
  // TODO: Add an invalid node on error, fix callers to adapt.
  auto ParseLibraryName(bool accept_default)
      -> std::optional<StringLiteralValueId>;

  // Handles parsing `library <name>`. Requires that the position is a `library`
  // token. Returns the name's ID on success, which may be invalid for
  // `default`.
  auto ParseLibrarySpecifier(bool accept_default)
      -> std::optional<StringLiteralValueId>;

  // Sets the package declaration information. Called at most once.
  auto set_packaging_decl(Tree::PackagingNames packaging_names, bool is_impl)
      -> void {
    CARBON_CHECK(!tree_->packaging_decl_);
    tree_->packaging_decl_ = {.names = packaging_names, .is_impl = is_impl};
  }

  // Adds an import.
  auto AddImport(Tree::PackagingNames package) -> void {
    tree_->imports_.push_back(package);
  }

  // Adds a function definition start node, and begins tracking a deferred
  // definition if necessary.
  auto AddFunctionDefinitionStart(Lex::TokenIndex token, bool has_error)
      -> void;
  // Adds a function definition node, and ends tracking a deferred definition if
  // necessary.
  auto AddFunctionDefinition(Lex::TokenIndex token, bool has_error) -> void;

  // Prints information for a stack dump.
  auto PrintForStackDump(llvm::raw_ostream& output) const -> void;

  auto tree() const -> const Tree& { return *tree_; }

  auto tokens() const -> const Lex::TokenizedBuffer& { return *tokens_; }

  auto has_errors() const -> bool { return err_tracker_.seen_error(); }

  auto emitter() -> DiagnosticEmitter& { return emitter_; }

  auto position() -> Lex::TokenIterator& { return position_; }
  auto position() const -> Lex::TokenIterator { return position_; }

  auto state_stack() -> llvm::SmallVector<StateStackEntry>& {
    return state_stack_;
  }

  auto state_stack() const -> const llvm::SmallVector<StateStackEntry>& {
    return state_stack_;
  }

  auto packaging_state() const -> PackagingState { return packaging_state_; }
  auto set_packaging_state(PackagingState packaging_state) -> void {
    packaging_state_ = packaging_state;
  }
  auto first_non_packaging_token() const -> Lex::TokenIndex {
    return first_non_packaging_token_;
  }
  auto set_first_non_packaging_token(Lex::TokenIndex token) -> void {
    CARBON_CHECK(!first_non_packaging_token_.has_value());
    first_non_packaging_token_ = token;
  }

 private:
  // Prints a single token for a stack dump. Used by PrintForStackDump.
  auto PrintTokenForStackDump(llvm::raw_ostream& output,
                              Lex::TokenIndex token) const -> void;

  Tree* tree_;
  Lex::TokenizedBuffer* tokens_;

  Diagnostics::ErrorTrackingConsumer err_tracker_;
  DiagnosticEmitter emitter_;

  // Whether to print verbose output.
  llvm::raw_ostream* vlog_stream_;

  // The current position within the token buffer.
  Lex::TokenIterator position_;
  // The FileEnd token.
  Lex::TokenIterator end_;

  llvm::SmallVector<StateStackEntry> state_stack_;

  // The deferred definition indexes of functions whose definitions have begun
  // but not yet finished.
  llvm::SmallVector<DeferredDefinitionIndex> deferred_definition_stack_;

  // The current packaging state, whether `import`/`package` are allowed.
  PackagingState packaging_state_ = PackagingState::FileStart;
  // The first non-packaging token, starting as invalid. Used for packaging
  // state warnings.
  Lex::TokenIndex first_non_packaging_token_ = Lex::TokenIndex::None;
};

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_CONTEXT_H_
