// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_CONTEXT_H_
#define CARBON_TOOLCHAIN_PARSE_CONTEXT_H_

#include <optional>

#include "common/check.h"
#include "common/vlog.h"
#include "llvm/ADT/StringRef.h"
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
// The size of the enum deliberately matches the index size.
enum class Lookahead : int32_t {
  CurrentToken = 0,
  NextToken = 1,
};

// The syntactic context of a binding pattern, which determines its default
// phase when no explicit `generic`/`template`/`runtime` keyword is present.
enum class BindingContext : uint8_t {
  // An explicit `()` parameter of a function, or a local binding
  // (`let`/`var`/`for`/`match`). Runtime by default.
  ExplicitParam,
  // A deduced `[]` parameter, whether of a function, a compile-time entity, or
  // an `impl`'s `forall`. Checked generic by default.
  DeducedParam,
  // An explicit `()` parameter of a compile-time entity (such as a `class`), or
  // of a parameterized name qualifier. Checked generic by default.
  CompileTimeEntityParam,
};

// The location of a parse diagnostic: a run of tokens, or the point just before
// or just past a token, where something missing would be written.
class DiagnosticLoc {
 public:
  enum class Kind : int8_t {
    Tokens,
    PointBefore,
    PointAfter,
  };

  // NOLINTNEXTLINE(google-explicit-constructor): A token is a location.
  DiagnosticLoc(Lex::TokenIndex token)
      : DiagnosticLoc(Kind::Tokens, {.begin = token, .end = token}) {}

  // The tokens from `tokens.begin` through `tokens.end`.
  static auto Tokens(Lex::InclusiveTokenRange tokens) -> DiagnosticLoc {
    return DiagnosticLoc(Kind::Tokens, tokens);
  }

  // The point just before `token`.
  static auto Before(Lex::TokenIndex token) -> DiagnosticLoc {
    return DiagnosticLoc(Kind::PointBefore, {.begin = token, .end = token});
  }

  // The point just past `token`.
  static auto After(Lex::TokenIndex token) -> DiagnosticLoc {
    return DiagnosticLoc(Kind::PointAfter, {.begin = token, .end = token});
  }

  auto kind() const -> Kind { return kind_; }

  // The tokens this location covers, or the one token a point is before or
  // after.
  auto tokens() const -> Lex::InclusiveTokenRange { return tokens_; }

 private:
  explicit DiagnosticLoc(Kind kind, Lex::InclusiveTokenRange tokens)
      : kind_(kind), tokens_(tokens) {}

  Kind kind_;
  Lex::InclusiveTokenRange tokens_;
};

// Context and shared functionality for parser handlers. See state.def for state
// documentation.
class Context {
 public:
  // A token-based emitter for use during parse.
  class DiagnosticEmitter : public Diagnostics::Emitter<DiagnosticLoc> {
   public:
    explicit DiagnosticEmitter(Diagnostics::Consumer* consumer,
                               Context* context)
        : Emitter(consumer), context_(context) {}

   protected:
    // Applies the `position_` to the `last_byte_offset` of the converted
    // location, so that a diagnostic sorts by how far the parser had read even
    // when the location it names is earlier. Diagnostics with the same offset
    // stay in the order they were emitted.
    auto ConvertLoc(DiagnosticLoc loc, ContextFnT /*context_fn*/) const
        -> Diagnostics::ConvertedLoc override {
      auto converted =
          context_->tokens().TokenRangeToDiagnosticLoc(loc.tokens());
      switch (loc.kind()) {
        case DiagnosticLoc::Kind::Tokens:
          break;
        case DiagnosticLoc::Kind::PointBefore:
          converted.loc.length = 0;
          break;
        case DiagnosticLoc::Kind::PointAfter:
          converted = context_->tokens().PointAfterTokenToDiagnosticLoc(
              loc.tokens().begin);
          break;
      }
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
  struct State : public Printable<State> {
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

    // Set to true to indicate that this state is handling a pattern nested
    // inside an `unused` pattern.
    // TODO: This is meaningful only for patterns, and the precedence fields
    // are meaningful only for expressions, so expressing them as a union
    // could help catch errors.
    bool in_unused_pattern : 1 = false;

    // Set to true to indicate that this state is handling a shorthand pattern
    // inside a struct pattern such as `{var a: T}`
    // TODO: This is meaningful only for patterns, and the precedence fields
    // are meaningful only for expressions, so expressing them as a union
    // could help catch errors.
    bool in_field_shorthand_pattern : 1 = false;

    // The binding context governing this state, used to pick the default phase
    // of its bindings. Meaningful for pattern states, and for the
    // declaration-name states that carry an enclosing declaration's default
    // down to its parameter lists.
    BindingContext binding_context : 2 = BindingContext::ExplicitParam;

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

  // We expect State to fit into 12 bytes:
  //   state = 1 byte
  //   has_error, in_var_pattern, in_unused_pattern, in_field_shorthand_pattern,
  //   binding_context = 1 byte
  //   ambient_precedence = 1 byte
  //   lhs_precedence = 1 byte
  //   token = 4 bytes
  //   subtree_start = 4 bytes
  // If it becomes bigger, it'd be worth examining better packing; it should be
  // feasible to pack the 1-byte entries more tightly.
  static_assert(sizeof(State) == 12, "State has unexpected size!");

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
    if (vlog_stream_ != nullptr) {
      VlogAddNode();
    }
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
      -> void {
    CARBON_CHECK(
        kind != NodeKind::InvalidParse && kind != NodeKind::InvalidParseSubtree,
        "{0} shouldn't occur in Placeholder use-cases", kind);
    CARBON_CHECK(position >= 0 && position < tree_->size(),
                 "position: {0} size: {1}", position, tree_->size());
    auto* node_impl = &tree_->node_impls_[position];
    CARBON_CHECK(node_impl->kind() == NodeKind::Placeholder);
    *node_impl = Tree::NodeImpl(kind, has_error, token);
  }

  // Returns the current position and moves past it.
  auto Consume() -> Lex::TokenIndex { return *(position_++); }

  // Consumes the current token. Does not return it.
  auto ConsumeAndDiscard() -> void { ++position_; }

  // Parses an open curly brace token, possibly diagnosing if necessary. Creates
  // a leaf parse node of the specified start kind. The default_token is used
  // when there's no open curly brace. Returns the open curly brace token if it
  // was found.
  auto ConsumeAndAddOpenCurlyBrace(Lex::TokenIndex default_token,
                                   NodeKind start_kind)
      -> std::optional<Lex::TokenIndex> {
    if (!PositionIs(Lex::TokenKind::OpenCurlyBrace)) {
      DiagnoseExpectedOpenCurlyBrace(default_token, start_kind);
      return std::nullopt;
    }
    auto open_curly = Consume();
    AddLeafNode(start_kind, open_curly, /*has_error=*/false);
    return open_curly;
  }

  // Parses an open paren token, possibly diagnosing if necessary. Creates a
  // leaf parse node of the specified start kind. The default_token is used when
  // there's no open paren. Returns the open paren token if it was found.
  auto ConsumeAndAddOpenParen(Lex::TokenIndex default_token,
                              NodeKind start_kind)
      -> std::optional<Lex::TokenIndex> {
    if (!PositionIs(Lex::TokenKind::OpenParen)) {
      DiagnoseExpectedOpenParen(default_token, start_kind);
      return std::nullopt;
    }
    auto open_paren = Consume();
    AddLeafNode(start_kind, open_paren, /*has_error=*/false);
    return open_paren;
  }

  // Parses a closing symbol corresponding to the opening symbol `state.token`,
  // possibly skipping forward and diagnosing if necessary. Creates a parse node
  // of the specified close kind. If `state.token` is not an opening symbol,
  // no input will be consumed, and no diagnostic will be emitted, but the parse
  // node will still be marked as having an error.
  auto ConsumeAndAddCloseSymbol(State state, NodeKind close_kind) -> void {
    Lex::TokenKind open_token_kind = tokens().GetKind(state.token);
    if (open_token_kind.is_opening_symbol() &&
        PositionIs(open_token_kind.closing_symbol())) {
      AddNode(close_kind, Consume(), state.has_error);
    } else {
      DiagnoseAndSkipToCloseSymbol(state, close_kind);
    }
  }

  // Composes `ConsumeIf` and `AddLeafNode`, returning false when ConsumeIf
  // fails.
  auto ConsumeAndAddLeafNodeIf(Lex::TokenKind token_kind, NodeKind node_kind)
      -> bool {
    auto token = ConsumeIf(token_kind);
    if (!token) {
      return false;
    }

    AddLeafNode(node_kind, *token);
    return true;
  }

  // Returns the current position and moves past it. Requires the token is the
  // expected kind.
  auto ConsumeChecked(Lex::TokenKind kind) -> Lex::TokenIndex {
    CARBON_CHECK(PositionIs(kind), "Required {0}, found {1}", kind,
                 PositionKind());
    return Consume();
  }

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
  auto IsLexicallyValidInfixOperator() -> bool {
    CARBON_CHECK(position_ != end_, "Expected an operator token.");

    bool leading_space = tokens().HasLeadingWhitespace(*position_);
    bool trailing_space = tokens().HasTrailingWhitespace(*position_);

    // If there's whitespace on both sides, it's an infix operator.
    if (leading_space && trailing_space) {
      return true;
    }

    // If there's whitespace on exactly one side, it's not an infix operator.
    if (leading_space || trailing_space) {
      return false;
    }

    // Otherwise, for an infix operator, the preceding token must be any close
    // bracket, identifier, or literal and the next token must be an open paren,
    // identifier, or literal.
    if (position_ == tokens().tokens().begin() ||
        !IsAssumedEndOfOperand(tokens().GetKind(*(position_ - 1))) ||
        !IsAssumedStartOfOperand(tokens().GetKind(*(position_ + 1)))) {
      return false;
    }

    return true;
  }

  // Determines whether the current trailing operator should be treated as
  // infix.
  auto IsTrailingOperatorInfix() -> bool {
    if (position_ == end_) {
      return false;
    }

    // An operator that follows the infix operator rules is parsed as
    // infix, unless the next token means that it can't possibly be.
    if (IsLexicallyValidInfixOperator() &&
        IsPossibleStartOfOperand(tokens().GetKind(*(position_ + 1)))) {
      return true;
    }

    // A trailing operator with leading whitespace that's not valid as infix is
    // not valid at all. If the next token looks like the start of an operand,
    // then parse as infix, otherwise as postfix. Either way we'll produce a
    // diagnostic later on.
    if (tokens().HasLeadingWhitespace(*position_) &&
        IsAssumedStartOfOperand(tokens().GetKind(*(position_ + 1)))) {
      return true;
    }

    return false;
  }

  // Diagnoses whether the current token is not written properly for the given
  // fixity. For example, because mandatory whitespace is missing. Regardless of
  // whether there's an error, it's expected that parsing continues.
  auto DiagnoseOperatorFixity(OperatorFixity fixity) -> void {
    if (!PositionKind().is_symbol()) {
      // Whitespace-based fixity rules only apply to symbolic operators.
      return;
    }

    if (fixity == OperatorFixity::Infix) {
      // Infix operators must satisfy the infix operator rules.
      if (IsLexicallyValidInfixOperator()) {
        return;
      }
    } else {
      bool prefix = fixity == OperatorFixity::Prefix;

      // Whitespace is not permitted between a symbolic pre/postfix operator
      // and its operand, and pre/postfix operators must not satisfy the infix
      // operator rules.
      if (!(prefix ? tokens().HasTrailingWhitespace(*position_)
                   : tokens().HasLeadingWhitespace(*position_)) &&
          !IsLexicallyValidInfixOperator()) {
        return;
      }
    }
    DiagnoseInvalidOperatorFixity(fixity);
  }

  // If the current position is a `,`, consumes it, adds the provided token, and
  // returns `Comma`. Returns `Close` if the current position is close_token
  // (for example, `)`). `CommaClose` indicates it found both (for example,
  // `,)`). Handles cases where invalid tokens are present by advancing the
  // position, and may emit errors. Pass already_has_error in order to suppress
  // duplicate errors.
  auto ConsumeListToken(NodeKind comma_kind, Lex::TokenKind close_kind,
                        bool already_has_error) -> ListTokenKind {
    if (!PositionIs(Lex::TokenKind::Comma) && !PositionIs(close_kind)) {
      RecoverFromInvalidListToken(close_kind, already_has_error);
    }

    if (PositionIs(close_kind)) {
      return ListTokenKind::Close;
    } else {
      AddLeafNode(comma_kind, Consume(),
                  /*has_error=*/comma_kind == NodeKind::InvalidParse);
      return PositionIs(close_kind) ? ListTokenKind::CommaClose
                                    : ListTokenKind::Comma;
    }
  }

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
  auto PopState() -> State {
    if (vlog_stream_ != nullptr) {
      VlogPopState("Pop");
    }
    return state_stack_.pop_back_val();
  }

  // Pops the state and discards it.
  auto PopAndDiscardState() -> void {
    if (vlog_stream_ != nullptr) {
      VlogPopState("PopAndDiscard");
    }
    state_stack_.pop_back();
  }

  // Pushes a new state with the current position for context.
  auto PushState(StateKind kind) -> void { PushState(kind, *position_); }

  // Pushes a new state with a specific token for context. Used when forming a
  // new subtree when the current position isn't the start of the subtree.
  auto PushState(StateKind kind, Lex::TokenIndex token) -> void {
    PushState({.kind = kind, .token = token, .subtree_start = tree_->size()});
  }

  // Pushes a new state with a specific token and binding context. Used when
  // starting a parameter list whose bindings have a non-default phase, or when
  // carrying a declaration's binding context across a qualified name.
  auto PushState(StateKind kind, Lex::TokenIndex token,
                 BindingContext binding_context) -> void {
    PushState({.kind = kind,
               .binding_context = binding_context,
               .token = token,
               .subtree_start = tree_->size()});
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

  // Pushes a new state for handling a pattern. `in_var_pattern` and
  // `in_unused_pattern`  and `in_field_shorthand_pattern` indicate whether that
  // pattern is nested inside a `var`, `unused` or a struct shorthand field
  // pattern. `binding_context` is the binding context that determines the
  // default phase of bindings in this pattern.
  auto PushStateForPattern(StateKind kind, bool in_var_pattern,
                           bool in_unused_pattern,
                           bool in_field_shorthand_pattern,
                           BindingContext binding_context,
                           PrecedenceGroup precedence) -> void {
    PushState({.kind = kind,
               .in_var_pattern = in_var_pattern,
               .in_unused_pattern = in_unused_pattern,
               .in_field_shorthand_pattern = in_field_shorthand_pattern,
               .binding_context = binding_context,
               .ambient_precedence = precedence,
               .token = *position_,
               .subtree_start = tree_->size()});
  }

  // Pushes a constructed state onto the stack.
  auto PushState(State state) -> void {
    state_stack_.push_back(state);
    if (vlog_stream_ != nullptr) {
      VlogPushState();
    }
    CARBON_CHECK(state_stack_.size() < (1 << 20),
                 "Excessive stack size: likely infinite loop");
  }

  // Pushes a constructed state onto the stack, with a different kind.
  auto PushState(State state, StateKind kind) -> void {
    state.kind = kind;
    PushState(state);
  }

  // Propagates an error up the state stack, to the parent state.
  auto ReturnErrorOnState() -> void { state_stack_.back().has_error = true; }

  // Adds a node for a declaration's semicolon. Includes error recovery when the
  // token is not a semicolon, using `decl_kind` and `is_def_allowed` to inform
  // diagnostics.
  auto AddNodeExpectingDeclSemi(State state, NodeKind node_kind,
                                Lex::TokenKind decl_kind, bool is_def_allowed)
      -> void {
    if (!state.has_error && PositionIs(Lex::TokenKind::Semi)) {
      AddNode(node_kind, Consume(), /*has_error=*/false);
      return;
    }
    RecoverFromMissingDeclSemi(state, node_kind, decl_kind, is_def_allowed);
  }

  // Emits a diagnostic for a declaration missing a semi.
  auto DiagnoseExpectedDeclSemi(Lex::TokenKind expected_kind) -> void;

  // Emits a diagnostic for a declaration missing a semi or definition.
  auto DiagnoseExpectedDeclSemiOrDefinition(Lex::TokenKind expected_kind)
      -> void;

  // Handles error recovery in a declaration, particularly before any possible
  // definition has started (although one could be present). Recover to a
  // semicolon when it makes sense as a possible end, otherwise use the
  // introducer token for the error.
  auto RecoverFromDeclError(State state, NodeKind node_kind,
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
  // Adds a function terse definition start node, and begins tracking a deferred
  // definition if necessary.
  auto AddFunctionTerseDefinitionStart(Lex::TokenIndex token, bool has_error)
      -> void;
  // Adds a function definition node, and ends tracking a deferred definition if
  // necessary.
  auto AddFunctionDefinition(Lex::TokenIndex token, bool has_error) -> void;
  // Adds a function terse definition node, and ends tracking a deferred
  // definition if necessary.
  auto AddFunctionTerseDefinition(Lex::TokenIndex token, bool has_error)
      -> void;

  // Prints information for a stack dump.
  auto PrintForStackDump(llvm::raw_ostream& output) const -> void;

  auto tree() const -> const Tree& { return *tree_; }

  auto tokens() const -> Lex::TokenizedBuffer& { return *tokens_; }

  auto has_errors() const -> bool { return err_tracker_.seen_error(); }

  auto emitter() -> DiagnosticEmitter& { return emitter_; }

  auto position() -> Lex::TokenIterator& { return position_; }
  auto position() const -> Lex::TokenIterator { return position_; }

  // Returns the point just past the token before `position_`, which is where a
  // token missing at `position_` would be written.
  auto PointAfterPreviousToken() const -> DiagnosticLoc {
    return DiagnosticLoc::After(*(position_ - 1));
  }

  // Returns where a user would insert an opening bracket missing at
  // `position_`: against the token at `position_` when it's on the line where
  // the previous token ends, and otherwise just past the previous token.
  auto OpeningBracketInsertionPoint() const -> DiagnosticLoc;

  auto state_stack() -> llvm::SmallVector<State>& { return state_stack_; }

  auto state_stack() const -> const llvm::SmallVector<State>& {
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
  // Determines whether the given token is considered to be the start of an
  // operand according to the rules for infix operator parsing.
  static auto IsAssumedStartOfOperand(Lex::TokenKind kind) -> bool {
    return kind.IsOneOf({Lex::TokenKind::OpenParen, Lex::TokenKind::Identifier,
                         Lex::TokenKind::IntLiteral,
                         Lex::TokenKind::RealLiteral,
                         Lex::TokenKind::StringLiteral});
  }

  // Determines whether the given token is considered to be the end of an
  // operand according to the rules for infix operator parsing.
  static auto IsAssumedEndOfOperand(Lex::TokenKind kind) -> bool {
    return kind.IsOneOf(
        {Lex::TokenKind::CloseParen, Lex::TokenKind::CloseCurlyBrace,
         Lex::TokenKind::CloseSquareBracket, Lex::TokenKind::Identifier,
         Lex::TokenKind::IntLiteral, Lex::TokenKind::RealLiteral,
         Lex::TokenKind::StringLiteral});
  }

  // Determines whether the given token could possibly be the start of an
  // operand. This is conservatively correct, and will never incorrectly return
  // `false`, but can incorrectly return `true`.
  static auto IsPossibleStartOfOperand(Lex::TokenKind kind) -> bool {
    return !kind.IsOneOf(
        {Lex::TokenKind::CloseParen, Lex::TokenKind::CloseCurlyBrace,
         Lex::TokenKind::CloseSquareBracket, Lex::TokenKind::Comma,
         Lex::TokenKind::Semi, Lex::TokenKind::Colon});
  }

  // The following are the slow paths of inline methods above, which are kept
  // out of line so that the inline fast paths stay small. They handle verbose
  // logging, diagnostics, and error recovery.

  // Logs the most recently added node.
  [[gnu::cold]] auto VlogAddNode() const -> void;
  // Logs the most recently pushed state.
  [[gnu::cold]] auto VlogPushState() const -> void;
  // Logs the state that is about to be popped, prefixed by `label`.
  [[gnu::cold]] auto VlogPopState(llvm::StringLiteral label) const -> void;

  // Diagnoses a missing open curly brace in `ConsumeAndAddOpenCurlyBrace`.
  [[gnu::cold]] auto DiagnoseExpectedOpenCurlyBrace(
      Lex::TokenIndex default_token, NodeKind start_kind) -> void;
  // Diagnoses a missing open paren in `ConsumeAndAddOpenParen`.
  [[gnu::cold]] auto DiagnoseExpectedOpenParen(Lex::TokenIndex default_token,
                                               NodeKind start_kind) -> void;
  // Handles the error cases of `ConsumeAndAddCloseSymbol`, diagnosing and
  // skipping to the matching closing symbol when `state.token` is an opening
  // symbol.
  [[gnu::cold]] auto DiagnoseAndSkipToCloseSymbol(State state,
                                                  NodeKind close_kind) -> void;
  // Diagnoses the current operator token not being written properly for the
  // given fixity in `DiagnoseOperatorFixity`.
  [[gnu::cold]] auto DiagnoseInvalidOperatorFixity(OperatorFixity fixity)
      -> void;
  // Diagnoses and skips past an invalid token in `ConsumeListToken`, leaving
  // the position at either a `,` or `close_kind`.
  [[gnu::cold]] auto RecoverFromInvalidListToken(Lex::TokenKind close_kind,
                                                 bool already_has_error)
      -> void;
  // Handles the error cases of `AddNodeExpectingDeclSemi`.
  [[gnu::cold]] auto RecoverFromMissingDeclSemi(State state, NodeKind node_kind,
                                                Lex::TokenKind decl_kind,
                                                bool is_def_allowed) -> void;

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

  llvm::SmallVector<State> state_stack_;

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
