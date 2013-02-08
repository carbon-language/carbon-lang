//===--- TokenAnnotator.cpp - Format C++ code -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements a token annotator, i.e. creates
/// \c AnnotatedTokens out of \c FormatTokens with required extra information.
///
//===----------------------------------------------------------------------===//

#include "TokenAnnotator.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"

namespace clang {
namespace format {

static bool isUnaryOperator(const AnnotatedToken &Tok) {
  switch (Tok.FormatTok.Tok.getKind()) {
  case tok::plus:
  case tok::plusplus:
  case tok::minus:
  case tok::minusminus:
  case tok::exclaim:
  case tok::tilde:
  case tok::kw_sizeof:
  case tok::kw_alignof:
    return true;
  default:
    return false;
  }
}

static bool isBinaryOperator(const AnnotatedToken &Tok) {
  // Comma is a binary operator, but does not behave as such wrt. formatting.
  return getPrecedence(Tok) > prec::Comma;
}

// Returns the previous token ignoring comments.
static AnnotatedToken *getPreviousToken(AnnotatedToken &Tok) {
  AnnotatedToken *PrevToken = Tok.Parent;
  while (PrevToken != NULL && PrevToken->is(tok::comment))
    PrevToken = PrevToken->Parent;
  return PrevToken;
}
static const AnnotatedToken *getPreviousToken(const AnnotatedToken &Tok) {
  return getPreviousToken(const_cast<AnnotatedToken &>(Tok));
}

static bool isTrailingComment(AnnotatedToken *Tok) {
  return Tok != NULL && Tok->is(tok::comment) &&
         (Tok->Children.empty() ||
          Tok->Children[0].FormatTok.NewlinesBefore > 0);
}

// Returns the next token ignoring comments.
static const AnnotatedToken *getNextToken(const AnnotatedToken &Tok) {
  if (Tok.Children.empty())
    return NULL;
  const AnnotatedToken *NextToken = &Tok.Children[0];
  while (NextToken->is(tok::comment)) {
    if (NextToken->Children.empty())
      return NULL;
    NextToken = &NextToken->Children[0];
  }
  return NextToken;
}

/// \brief A parser that gathers additional information about tokens.
///
/// The \c TokenAnnotator tries to matches parenthesis and square brakets and
/// store a parenthesis levels. It also tries to resolve matching "<" and ">"
/// into template parameter lists.
class AnnotatingParser {
public:
  AnnotatingParser(SourceManager &SourceMgr, Lexer &Lex, AnnotatedLine &Line)
      : SourceMgr(SourceMgr), Lex(Lex), Line(Line), CurrentToken(&Line.First),
        KeywordVirtualFound(false) {
    Contexts.push_back(Context(1, /*IsExpression=*/ false));
    Contexts.back().LookForFunctionName = Line.MustBeDeclaration;
  }

  bool parseAngle() {
    if (CurrentToken == NULL)
      return false;
    ScopedContextCreator ContextCreator(*this, 10);
    AnnotatedToken *Left = CurrentToken->Parent;
    Contexts.back().IsExpression = false;
    while (CurrentToken != NULL) {
      if (CurrentToken->is(tok::greater)) {
        Left->MatchingParen = CurrentToken;
        CurrentToken->MatchingParen = Left;
        CurrentToken->Type = TT_TemplateCloser;
        next();
        return true;
      }
      if (CurrentToken->is(tok::r_paren) || CurrentToken->is(tok::r_square) ||
          CurrentToken->is(tok::r_brace))
        return false;
      if (CurrentToken->is(tok::pipepipe) || CurrentToken->is(tok::ampamp) ||
          CurrentToken->is(tok::question) || CurrentToken->is(tok::colon))
        return false;
      if (CurrentToken->is(tok::comma))
        ++Left->ParameterCount;
      if (!consumeToken())
        return false;
    }
    return false;
  }

  bool parseParens(bool LookForDecls = false) {
    if (CurrentToken == NULL)
      return false;
    ScopedContextCreator ContextCreator(*this, 1);

    // FIXME: This is a bit of a hack. Do better.
    Contexts.back().ColonIsForRangeExpr =
        Contexts.size() == 2 && Contexts[0].ColonIsForRangeExpr;

    bool StartsObjCMethodExpr = false;
    AnnotatedToken *Left = CurrentToken->Parent;
    if (CurrentToken->is(tok::caret)) {
      // ^( starts a block.
      Left->Type = TT_ObjCBlockLParen;
    } else if (AnnotatedToken *MaybeSel = Left->Parent) {
      // @selector( starts a selector.
      if (MaybeSel->isObjCAtKeyword(tok::objc_selector) && MaybeSel->Parent &&
          MaybeSel->Parent->is(tok::at)) {
        StartsObjCMethodExpr = true;
      }
    }

    if (StartsObjCMethodExpr) {
      Contexts.back().ColonIsObjCMethodExpr = true;
      Left->Type = TT_ObjCMethodExpr;
    }

    while (CurrentToken != NULL) {
      // LookForDecls is set when "if (" has been seen. Check for
      // 'identifier' '*' 'identifier' followed by not '=' -- this
      // '*' has to be a binary operator but determineStarAmpUsage() will
      // categorize it as an unary operator, so set the right type here.
      if (LookForDecls && !CurrentToken->Children.empty()) {
        AnnotatedToken &Prev = *CurrentToken->Parent;
        AnnotatedToken &Next = CurrentToken->Children[0];
        if (Prev.Parent->is(tok::identifier) &&
            (Prev.is(tok::star) || Prev.is(tok::amp)) &&
            CurrentToken->is(tok::identifier) && Next.isNot(tok::equal)) {
          Prev.Type = TT_BinaryOperator;
          LookForDecls = false;
        }
      }

      if (CurrentToken->is(tok::r_paren)) {
        Left->MatchingParen = CurrentToken;
        CurrentToken->MatchingParen = Left;

        if (StartsObjCMethodExpr) {
          CurrentToken->Type = TT_ObjCMethodExpr;
          if (Contexts.back().FirstObjCSelectorName != NULL) {
            Contexts.back().FirstObjCSelectorName->LongestObjCSelectorName =
                Contexts.back().LongestObjCSelectorName;
          }
        }

        next();
        return true;
      }
      if (CurrentToken->is(tok::r_square) || CurrentToken->is(tok::r_brace))
        return false;
      if (CurrentToken->is(tok::comma))
        ++Left->ParameterCount;
      if (!consumeToken())
        return false;
    }
    return false;
  }

  bool parseSquare() {
    if (!CurrentToken)
      return false;
    ScopedContextCreator ContextCreator(*this, 10);

    // A '[' could be an index subscript (after an indentifier or after
    // ')' or ']'), or it could be the start of an Objective-C method
    // expression.
    AnnotatedToken *Left = CurrentToken->Parent;
    AnnotatedToken *Parent = getPreviousToken(*Left);
    bool StartsObjCMethodExpr =
        !Parent || Parent->is(tok::colon) || Parent->is(tok::l_square) ||
        Parent->is(tok::l_paren) || Parent->is(tok::kw_return) ||
        Parent->is(tok::kw_throw) || isUnaryOperator(*Parent) ||
        getBinOpPrecedence(Parent->FormatTok.Tok.getKind(), true, true) >
        prec::Unknown;

    if (StartsObjCMethodExpr) {
      Contexts.back().ColonIsObjCMethodExpr = true;
      Left->Type = TT_ObjCMethodExpr;
    }

    while (CurrentToken != NULL) {
      if (CurrentToken->is(tok::r_square)) {
        if (!CurrentToken->Children.empty() &&
            CurrentToken->Children[0].is(tok::l_paren)) {
          // An ObjC method call is rarely followed by an open parenthesis.
          // FIXME: Do we incorrectly label ":" with this?
          StartsObjCMethodExpr = false;
          Left->Type = TT_Unknown;
        }
        if (StartsObjCMethodExpr) {
          CurrentToken->Type = TT_ObjCMethodExpr;
          // determineStarAmpUsage() thinks that '*' '[' is allocating an
          // array of pointers, but if '[' starts a selector then '*' is a
          // binary operator.
          if (Parent != NULL &&
              (Parent->is(tok::star) || Parent->is(tok::amp)) &&
              Parent->Type == TT_PointerOrReference)
            Parent->Type = TT_BinaryOperator;
        }
        Left->MatchingParen = CurrentToken;
        CurrentToken->MatchingParen = Left;
        if (Contexts.back().FirstObjCSelectorName != NULL)
          Contexts.back().FirstObjCSelectorName->LongestObjCSelectorName =
              Contexts.back().LongestObjCSelectorName;
        next();
        return true;
      }
      if (CurrentToken->is(tok::r_paren) || CurrentToken->is(tok::r_brace))
        return false;
      if (CurrentToken->is(tok::comma))
        ++Left->ParameterCount;
      if (!consumeToken())
        return false;
    }
    return false;
  }

  bool parseBrace() {
    // Lines are fine to end with '{'.
    if (CurrentToken == NULL)
      return true;
    ScopedContextCreator ContextCreator(*this, 1);
    AnnotatedToken *Left = CurrentToken->Parent;
    while (CurrentToken != NULL) {
      if (CurrentToken->is(tok::r_brace)) {
        Left->MatchingParen = CurrentToken;
        CurrentToken->MatchingParen = Left;
        next();
        return true;
      }
      if (CurrentToken->is(tok::r_paren) || CurrentToken->is(tok::r_square))
        return false;
      if (CurrentToken->is(tok::comma))
        ++Left->ParameterCount;
      if (!consumeToken())
        return false;
    }
    return true;
  }

  bool parseConditional() {
    while (CurrentToken != NULL) {
      if (CurrentToken->is(tok::colon)) {
        CurrentToken->Type = TT_ConditionalExpr;
        next();
        return true;
      }
      if (!consumeToken())
        return false;
    }
    return false;
  }

  bool parseTemplateDeclaration() {
    if (CurrentToken != NULL && CurrentToken->is(tok::less)) {
      CurrentToken->Type = TT_TemplateOpener;
      next();
      if (!parseAngle())
        return false;
      CurrentToken->Parent->ClosesTemplateDeclaration = true;
      return true;
    }
    return false;
  }

  bool consumeToken() {
    AnnotatedToken *Tok = CurrentToken;
    next();
    switch (Tok->FormatTok.Tok.getKind()) {
    case tok::plus:
    case tok::minus:
      // At the start of the line, +/- specific ObjectiveC method
      // declarations.
      if (Tok->Parent == NULL)
        Tok->Type = TT_ObjCMethodSpecifier;
      break;
    case tok::colon:
      // Colons from ?: are handled in parseConditional().
      if (Tok->Parent->is(tok::r_paren)) {
        Tok->Type = TT_CtorInitializerColon;
      } else if (Contexts.back().ColonIsObjCMethodExpr ||
                 Line.First.Type == TT_ObjCMethodSpecifier) {
        Tok->Type = TT_ObjCMethodExpr;
        Tok->Parent->Type = TT_ObjCSelectorName;
        if (Tok->Parent->FormatTok.TokenLength >
            Contexts.back().LongestObjCSelectorName)
          Contexts.back().LongestObjCSelectorName =
              Tok->Parent->FormatTok.TokenLength;
        if (Contexts.back().FirstObjCSelectorName == NULL)
          Contexts.back().FirstObjCSelectorName = Tok->Parent;
      } else if (Contexts.back().ColonIsForRangeExpr) {
        Tok->Type = TT_RangeBasedForLoopColon;
      }
      break;
    case tok::kw_if:
    case tok::kw_while:
      if (CurrentToken != NULL && CurrentToken->is(tok::l_paren)) {
        next();
        if (!parseParens(/*LookForDecls=*/ true))
          return false;
      }
      break;
    case tok::kw_for:
      Contexts.back().ColonIsForRangeExpr = true;
      next();
      if (!parseParens())
        return false;
      break;
    case tok::l_paren:
      if (!parseParens())
        return false;
      break;
    case tok::l_square:
      if (!parseSquare())
        return false;
      break;
    case tok::l_brace:
      if (!parseBrace())
        return false;
      break;
    case tok::less:
      if (parseAngle())
        Tok->Type = TT_TemplateOpener;
      else {
        Tok->Type = TT_BinaryOperator;
        CurrentToken = Tok;
        next();
      }
      break;
    case tok::r_paren:
    case tok::r_square:
      return false;
    case tok::r_brace:
      // Lines can start with '}'.
      if (Tok->Parent != NULL)
        return false;
      break;
    case tok::greater:
      Tok->Type = TT_BinaryOperator;
      break;
    case tok::kw_operator:
      if (CurrentToken != NULL && CurrentToken->is(tok::l_paren)) {
        CurrentToken->Type = TT_OverloadedOperator;
        next();
        if (CurrentToken != NULL && CurrentToken->is(tok::r_paren)) {
          CurrentToken->Type = TT_OverloadedOperator;
          next();
        }
      } else {
        while (CurrentToken != NULL && CurrentToken->isNot(tok::l_paren)) {
          CurrentToken->Type = TT_OverloadedOperator;
          next();
        }
      }
      break;
    case tok::question:
      parseConditional();
      break;
    case tok::kw_template:
      parseTemplateDeclaration();
      break;
    default:
      break;
    }
    return true;
  }

  void parseIncludeDirective() {
    next();
    if (CurrentToken != NULL && CurrentToken->is(tok::less)) {
      next();
      while (CurrentToken != NULL) {
        if (CurrentToken->isNot(tok::comment) ||
            !CurrentToken->Children.empty())
          CurrentToken->Type = TT_ImplicitStringLiteral;
        next();
      }
    } else {
      while (CurrentToken != NULL) {
        next();
      }
    }
  }

  void parseWarningOrError() {
    next();
    // We still want to format the whitespace left of the first token of the
    // warning or error.
    next();
    while (CurrentToken != NULL) {
      CurrentToken->Type = TT_ImplicitStringLiteral;
      next();
    }
  }

  void parsePreprocessorDirective() {
    next();
    if (CurrentToken == NULL)
      return;
    // Hashes in the middle of a line can lead to any strange token
    // sequence.
    if (CurrentToken->FormatTok.Tok.getIdentifierInfo() == NULL)
      return;
    switch (CurrentToken->FormatTok.Tok.getIdentifierInfo()->getPPKeywordID()) {
    case tok::pp_include:
    case tok::pp_import:
      parseIncludeDirective();
      break;
    case tok::pp_error:
    case tok::pp_warning:
      parseWarningOrError();
      break;
    default:
      break;
    }
    while (CurrentToken != NULL)
      next();
  }

  LineType parseLine() {
    int PeriodsAndArrows = 0;
    bool CanBeBuilderTypeStmt = true;
    if (CurrentToken->is(tok::hash)) {
      parsePreprocessorDirective();
      return LT_PreprocessorDirective;
    }
    while (CurrentToken != NULL) {
      if (CurrentToken->is(tok::kw_virtual))
        KeywordVirtualFound = true;
      if (CurrentToken->is(tok::period) || CurrentToken->is(tok::arrow))
        ++PeriodsAndArrows;
      if (getPrecedence(*CurrentToken) > prec::Assignment &&
          CurrentToken->isNot(tok::less) && CurrentToken->isNot(tok::greater))
        CanBeBuilderTypeStmt = false;
      if (!consumeToken())
        return LT_Invalid;
    }
    if (KeywordVirtualFound)
      return LT_VirtualFunctionDecl;

    // Assume a builder-type call if there are 2 or more "." and "->".
    if (PeriodsAndArrows >= 2 && CanBeBuilderTypeStmt)
      return LT_BuilderTypeCall;

    if (Line.First.Type == TT_ObjCMethodSpecifier) {
      if (Contexts.back().FirstObjCSelectorName != NULL)
        Contexts.back().FirstObjCSelectorName->LongestObjCSelectorName =
            Contexts.back().LongestObjCSelectorName;
      return LT_ObjCMethodDecl;
    }

    return LT_Other;
  }

  void next() {
    if (CurrentToken != NULL) {
      determineTokenType(*CurrentToken);
      CurrentToken->BindingStrength = Contexts.back().BindingStrength;
    }

    if (CurrentToken != NULL && !CurrentToken->Children.empty())
      CurrentToken = &CurrentToken->Children[0];
    else
      CurrentToken = NULL;
  }

private:
  /// \brief A struct to hold information valid in a specific context, e.g.
  /// a pair of parenthesis.
  struct Context {
    Context(unsigned BindingStrength, bool IsExpression)
        : BindingStrength(BindingStrength), LongestObjCSelectorName(0),
          ColonIsForRangeExpr(false), ColonIsObjCMethodExpr(false),
          FirstObjCSelectorName(NULL), IsExpression(IsExpression),
          LookForFunctionName(false) {
    }

    unsigned BindingStrength;
    unsigned LongestObjCSelectorName;
    bool ColonIsForRangeExpr;
    bool ColonIsObjCMethodExpr;
    AnnotatedToken *FirstObjCSelectorName;
    bool IsExpression;
    bool LookForFunctionName;
  };

  /// \brief Puts a new \c Context onto the stack \c Contexts for the lifetime
  /// of each instance.
  struct ScopedContextCreator {
    AnnotatingParser &P;

    ScopedContextCreator(AnnotatingParser &P, unsigned Increase)
        : P(P) {
      P.Contexts.push_back(Context(
          P.Contexts.back().BindingStrength + Increase,
          P.Contexts.back().IsExpression));
    }

    ~ScopedContextCreator() { P.Contexts.pop_back(); }
  };

  void determineTokenType(AnnotatedToken &Current) {
    if (getPrecedence(Current) == prec::Assignment) {
      Contexts.back().IsExpression = true;
      AnnotatedToken *Previous = Current.Parent;
      while (Previous != NULL && Previous->isNot(tok::comma)) {
        if (Previous->Type == TT_BinaryOperator &&
            (Previous->is(tok::star) || Previous->is(tok::amp))) {
          Previous->Type = TT_PointerOrReference;
        }
        Previous = Previous->Parent;
      }
    }
    if (Current.is(tok::kw_return) || Current.is(tok::kw_throw) ||
        (Current.is(tok::l_paren) && !Line.MustBeDeclaration &&
         (Current.Parent == NULL || Current.Parent->isNot(tok::kw_for))))
      Contexts.back().IsExpression = true;

    if (Current.Type == TT_Unknown) {
      if (Contexts.back().LookForFunctionName && Current.is(tok::l_paren)) {
        findFunctionName(&Current);
        Contexts.back().LookForFunctionName = false;
      } else if (Current.is(tok::star) || Current.is(tok::amp)) {
        Current.Type =
            determineStarAmpUsage(Current, Contexts.back().IsExpression);
      } else if (Current.is(tok::minus) || Current.is(tok::plus) ||
                 Current.is(tok::caret)) {
        Current.Type = determinePlusMinusCaretUsage(Current);
      } else if (Current.is(tok::minusminus) || Current.is(tok::plusplus)) {
        Current.Type = determineIncrementUsage(Current);
      } else if (Current.is(tok::exclaim)) {
        Current.Type = TT_UnaryOperator;
      } else if (isBinaryOperator(Current)) {
        Current.Type = TT_BinaryOperator;
      } else if (Current.is(tok::comment)) {
        std::string Data(Lexer::getSpelling(Current.FormatTok.Tok, SourceMgr,
                                            Lex.getLangOpts()));
        if (StringRef(Data).startswith("//"))
          Current.Type = TT_LineComment;
        else
          Current.Type = TT_BlockComment;
      } else if (Current.is(tok::r_paren) &&
                 (Current.Parent->Type == TT_PointerOrReference ||
                  Current.Parent->Type == TT_TemplateCloser) &&
                 (Current.Children.empty() ||
                  (Current.Children[0].isNot(tok::equal) &&
                   Current.Children[0].isNot(tok::semi) &&
                   Current.Children[0].isNot(tok::l_brace)))) {
        // FIXME: We need to get smarter and understand more cases of casts.
        Current.Type = TT_CastRParen;
      } else if (Current.is(tok::at) && Current.Children.size()) {
        switch (Current.Children[0].FormatTok.Tok.getObjCKeywordID()) {
        case tok::objc_interface:
        case tok::objc_implementation:
        case tok::objc_protocol:
          Current.Type = TT_ObjCDecl;
          break;
        case tok::objc_property:
          Current.Type = TT_ObjCProperty;
          break;
        default:
          break;
        }
      }
    }
  }

  /// \brief Starting from \p Current, this searches backwards for an
  /// identifier which could be the start of a function name and marks it.
  void findFunctionName(AnnotatedToken *Current) {
    AnnotatedToken *Parent = Current->Parent;
    while (Parent != NULL && Parent->Parent != NULL) {
      if (Parent->is(tok::identifier) &&
          (Parent->Parent->is(tok::identifier) ||
           Parent->Parent->Type == TT_PointerOrReference ||
           Parent->Parent->Type == TT_TemplateCloser)) {
        Parent->Type = TT_StartOfName;
        break;
      }
      Parent = Parent->Parent;
    }
  }

  /// \brief Return the type of the given token assuming it is * or &.
  TokenType
  determineStarAmpUsage(const AnnotatedToken &Tok, bool IsExpression) {
    const AnnotatedToken *PrevToken = getPreviousToken(Tok);
    if (PrevToken == NULL)
      return TT_UnaryOperator;

    const AnnotatedToken *NextToken = getNextToken(Tok);
    if (NextToken == NULL)
      return TT_Unknown;

    if (PrevToken->is(tok::l_paren) || PrevToken->is(tok::l_square) ||
        PrevToken->is(tok::l_brace) || PrevToken->is(tok::comma) ||
        PrevToken->is(tok::kw_return) || PrevToken->is(tok::colon) ||
        PrevToken->is(tok::equal) || PrevToken->Type == TT_BinaryOperator ||
        PrevToken->Type == TT_UnaryOperator || PrevToken->Type == TT_CastRParen)
      return TT_UnaryOperator;

    if (NextToken->is(tok::l_square))
      return TT_PointerOrReference;

    if (PrevToken->FormatTok.Tok.isLiteral() || PrevToken->is(tok::r_paren) ||
        PrevToken->is(tok::r_square) || NextToken->FormatTok.Tok.isLiteral() ||
        isUnaryOperator(*NextToken) || NextToken->is(tok::l_paren) ||
        NextToken->is(tok::l_square))
      return TT_BinaryOperator;

    if (NextToken->is(tok::comma) || NextToken->is(tok::r_paren) ||
        NextToken->is(tok::greater))
      return TT_PointerOrReference;

    // It is very unlikely that we are going to find a pointer or reference type
    // definition on the RHS of an assignment.
    if (IsExpression)
      return TT_BinaryOperator;

    return TT_PointerOrReference;
  }

  TokenType determinePlusMinusCaretUsage(const AnnotatedToken &Tok) {
    const AnnotatedToken *PrevToken = getPreviousToken(Tok);
    if (PrevToken == NULL)
      return TT_UnaryOperator;

    // Use heuristics to recognize unary operators.
    if (PrevToken->is(tok::equal) || PrevToken->is(tok::l_paren) ||
        PrevToken->is(tok::comma) || PrevToken->is(tok::l_square) ||
        PrevToken->is(tok::question) || PrevToken->is(tok::colon) ||
        PrevToken->is(tok::kw_return) || PrevToken->is(tok::kw_case) ||
        PrevToken->is(tok::at) || PrevToken->is(tok::l_brace))
      return TT_UnaryOperator;

    // There can't be two consecutive binary operators.
    if (PrevToken->Type == TT_BinaryOperator)
      return TT_UnaryOperator;

    // Fall back to marking the token as binary operator.
    return TT_BinaryOperator;
  }

  /// \brief Determine whether ++/-- are pre- or post-increments/-decrements.
  TokenType determineIncrementUsage(const AnnotatedToken &Tok) {
    const AnnotatedToken *PrevToken = getPreviousToken(Tok);
    if (PrevToken == NULL)
      return TT_UnaryOperator;
    if (PrevToken->is(tok::r_paren) || PrevToken->is(tok::r_square) ||
        PrevToken->is(tok::identifier))
      return TT_TrailingUnaryOperator;

    return TT_UnaryOperator;
  }

  SmallVector<Context, 8> Contexts;

  SourceManager &SourceMgr;
  Lexer &Lex;
  AnnotatedLine &Line;
  AnnotatedToken *CurrentToken;
  bool KeywordVirtualFound;
};

/// \brief Parses binary expressions by inserting fake parenthesis based on
/// operator precedence.
class ExpressionParser {
public:
  ExpressionParser(AnnotatedLine &Line) : Current(&Line.First) {}

  /// \brief Parse expressions with the given operatore precedence.
  void parse(unsigned Precedence = prec::Unknown) {
    if (Precedence > prec::PointerToMember || Current == NULL)
      return;

    // Skip over "return" until we can properly parse it.
    if (Current->is(tok::kw_return))
      next();

    // Eagerly consume trailing comments.
    while (isTrailingComment(Current)) {
      next();
    }

    AnnotatedToken *Start = Current;
    bool OperatorFound = false;

    while (Current != NULL) {
      // Consume operators with higher precedence.
      parse(Precedence + 1);

      // At the end of the line or when an operator with higher precedence is
      // found, insert fake parenthesis and return.
      if (Current == NULL || Current->is(tok::semi) || closesScope(*Current) ||
          ((Current->Type == TT_BinaryOperator || Current->is(tok::comma)) &&
           getPrecedence(*Current) < Precedence)) {
        if (OperatorFound) {
          ++Start->FakeLParens;
          if (Current != NULL)
            ++Current->Parent->FakeRParens;
        }
        return;
      }

      // Consume scopes: (), [], <> and {}
      if (opensScope(*Current)) {
        while (Current != NULL && !closesScope(*Current)) {
          next();
          parse();
        }
        next();
      } else {
        // Operator found.
        if (getPrecedence(*Current) == Precedence)
          OperatorFound = true;

        next();
      }
    }
  }

private:
  void next() {
    if (Current != NULL)
      Current = Current->Children.empty() ? NULL : &Current->Children[0];
  }

  bool closesScope(const AnnotatedToken &Tok) {
    return Current->is(tok::r_paren) || Current->Type == TT_TemplateCloser ||
           Current->is(tok::r_brace) || Current->is(tok::r_square);
  }

  bool opensScope(const AnnotatedToken &Tok) {
    return Current->is(tok::l_paren) || Current->Type == TT_TemplateOpener ||
           Current->is(tok::l_brace) || Current->is(tok::l_square);
  }

  AnnotatedToken *Current;
};

void TokenAnnotator::annotate(AnnotatedLine &Line) {
  AnnotatingParser Parser(SourceMgr, Lex, Line);
  Line.Type = Parser.parseLine();
  if (Line.Type == LT_Invalid)
    return;

  ExpressionParser ExprParser(Line);
  ExprParser.parse();

  if (Line.First.Type == TT_ObjCMethodSpecifier)
    Line.Type = LT_ObjCMethodDecl;
  else if (Line.First.Type == TT_ObjCDecl)
    Line.Type = LT_ObjCDecl;
  else if (Line.First.Type == TT_ObjCProperty)
    Line.Type = LT_ObjCProperty;

  Line.First.SpaceRequiredBefore = true;
  Line.First.MustBreakBefore = Line.First.FormatTok.MustBreakBefore;
  Line.First.CanBreakBefore = Line.First.MustBreakBefore;

  Line.First.TotalLength = Line.First.FormatTok.TokenLength;
}

void TokenAnnotator::calculateFormattingInformation(AnnotatedLine &Line) {
  if (Line.First.Children.empty())
    return;
  AnnotatedToken *Current = &Line.First.Children[0];
  while (Current != NULL) {
    Current->SpaceRequiredBefore = spaceRequiredBefore(Line, *Current);

    if (Current->FormatTok.MustBreakBefore) {
      Current->MustBreakBefore = true;
    } else if (Current->Type == TT_LineComment) {
      Current->MustBreakBefore = Current->FormatTok.NewlinesBefore > 0;
    } else if (isTrailingComment(Current->Parent) ||
               (Current->is(tok::string_literal) &&
                Current->Parent->is(tok::string_literal))) {
      Current->MustBreakBefore = true;
    } else if (Current->is(tok::lessless) && !Current->Children.empty() &&
               Current->Parent->is(tok::string_literal) &&
               Current->Children[0].is(tok::string_literal)) {
      Current->MustBreakBefore = true;
    } else {
      Current->MustBreakBefore = false;
    }
    Current->CanBreakBefore =
        Current->MustBreakBefore || canBreakBefore(Line, *Current);
    if (Current->MustBreakBefore)
      Current->TotalLength = Current->Parent->TotalLength + Style.ColumnLimit;
    else
      Current->TotalLength =
          Current->Parent->TotalLength + Current->FormatTok.TokenLength +
          (Current->SpaceRequiredBefore ? 1 : 0);
    // FIXME: Only calculate this if CanBreakBefore is true once static
    // initializers etc. are sorted out.
    // FIXME: Move magic numbers to a better place.
    Current->SplitPenalty =
        20 * Current->BindingStrength + splitPenalty(Line, *Current);

    Current = Current->Children.empty() ? NULL : &Current->Children[0];
  }
}

unsigned TokenAnnotator::splitPenalty(const AnnotatedLine &Line,
                                      const AnnotatedToken &Tok) {
  const AnnotatedToken &Left = *Tok.Parent;
  const AnnotatedToken &Right = Tok;

  if (Left.is(tok::l_brace) && Right.isNot(tok::l_brace))
    return 50;
  if (Left.is(tok::equal) && Right.is(tok::l_brace))
    return 150;
  if (Left.is(tok::coloncolon))
    return 500;

  if (Left.Type == TT_RangeBasedForLoopColon)
    return 5;

  if (Right.is(tok::arrow) || Right.is(tok::period)) {
    if (Left.is(tok::r_paren) && Line.Type == LT_BuilderTypeCall)
      return 5; // Should be smaller than breaking at a nested comma.
    return 150;
  }

  // In for-loops, prefer breaking at ',' and ';'.
  if (Line.First.is(tok::kw_for) &&
      (Left.isNot(tok::comma) && Left.isNot(tok::semi)))
    return 20;

  if (Left.is(tok::semi))
    return 0;
  if (Left.is(tok::comma))
    return 1;

  // In Objective-C method expressions, prefer breaking before "param:" over
  // breaking after it.
  if (Right.Type == TT_ObjCSelectorName)
    return 0;
  if (Left.is(tok::colon) && Left.Type == TT_ObjCMethodExpr)
    return 20;

  if (Left.is(tok::l_paren) || Left.is(tok::l_square) ||
      Left.Type == TT_TemplateOpener)
    return 20;

  if (Right.is(tok::lessless)) {
    if (Left.is(tok::string_literal)) {
      char LastChar =
          StringRef(Left.FormatTok.Tok.getLiteralData(),
                    Left.FormatTok.TokenLength).drop_back(1).rtrim().back();
      if (LastChar == ':' || LastChar == '=')
        return 100;
    }
    return prec::Shift;
  }
  if (Left.Type == TT_ConditionalExpr)
    return prec::Assignment;
  prec::Level Level = getPrecedence(Left);

  if (Level != prec::Unknown)
    return Level;

  return 3;
}

bool TokenAnnotator::spaceRequiredBetween(const AnnotatedLine &Line,
                                          const AnnotatedToken &Left,
                                          const AnnotatedToken &Right) {
  if (Right.is(tok::hashhash))
    return Left.is(tok::hash);
  if (Left.is(tok::hashhash) || Left.is(tok::hash))
    return Right.is(tok::hash);
  if (Right.is(tok::r_paren) || Right.is(tok::semi) || Right.is(tok::comma))
    return false;
  if (Right.is(tok::less) &&
      (Left.is(tok::kw_template) ||
       (Line.Type == LT_ObjCDecl && Style.ObjCSpaceBeforeProtocolList)))
    return true;
  if (Left.is(tok::arrow) || Right.is(tok::arrow))
    return false;
  if (Left.is(tok::exclaim) || Left.is(tok::tilde))
    return false;
  if (Left.is(tok::at) &&
      (Right.is(tok::identifier) || Right.is(tok::string_literal) ||
       Right.is(tok::char_constant) || Right.is(tok::numeric_constant) ||
       Right.is(tok::l_paren) || Right.is(tok::l_brace) ||
       Right.is(tok::kw_true) || Right.is(tok::kw_false)))
    return false;
  if (Left.is(tok::coloncolon))
    return false;
  if (Right.is(tok::coloncolon))
    return Left.isNot(tok::identifier) && Left.isNot(tok::greater) &&
           Left.isNot(tok::l_paren);
  if (Left.is(tok::less) || Right.is(tok::greater) || Right.is(tok::less))
    return false;
  if (Right.is(tok::amp) || Right.is(tok::star))
    return Left.FormatTok.Tok.isLiteral() ||
           (Left.isNot(tok::star) && Left.isNot(tok::amp) &&
            !Style.PointerBindsToType);
  if (Left.is(tok::amp) || Left.is(tok::star))
    return Right.FormatTok.Tok.isLiteral() || Style.PointerBindsToType;
  if (Right.is(tok::star) && Left.is(tok::l_paren))
    return false;
  if (Left.is(tok::l_square) || Right.is(tok::r_square))
    return false;
  if (Right.is(tok::l_square) && Right.Type != TT_ObjCMethodExpr)
    return false;
  if (Left.is(tok::period) || Right.is(tok::period))
    return false;
  if (Left.is(tok::colon))
    return Left.Type != TT_ObjCMethodExpr;
  if (Right.is(tok::colon))
    return Right.Type != TT_ObjCMethodExpr;
  if (Left.is(tok::l_paren))
    return false;
  if (Right.is(tok::l_paren)) {
    return Line.Type == LT_ObjCDecl || Left.is(tok::kw_if) ||
           Left.is(tok::kw_for) || Left.is(tok::kw_while) ||
           Left.is(tok::kw_switch) || Left.is(tok::kw_return) ||
           Left.is(tok::kw_catch) || Left.is(tok::kw_new) ||
           Left.is(tok::kw_delete);
  }
  if (Left.is(tok::at) &&
      Right.FormatTok.Tok.getObjCKeywordID() != tok::objc_not_keyword)
    return false;
  if (Left.is(tok::l_brace) && Right.is(tok::r_brace))
    return false;
  return true;
}

bool TokenAnnotator::spaceRequiredBefore(const AnnotatedLine &Line,
                                         const AnnotatedToken &Tok) {
  if (Line.Type == LT_ObjCMethodDecl) {
    if (Tok.is(tok::identifier) && !Tok.Children.empty() &&
        Tok.Children[0].is(tok::colon) && Tok.Parent->is(tok::identifier))
      return true;
    if (Tok.is(tok::colon))
      return false;
    if (Tok.Parent->Type == TT_ObjCMethodSpecifier)
      return true;
    if (Tok.Parent->is(tok::r_paren) && Tok.is(tok::identifier))
      // Don't space between ')' and <id>
      return false;
    if (Tok.Parent->is(tok::colon) && Tok.is(tok::l_paren))
      // Don't space between ':' and '('
      return false;
  }
  if (Line.Type == LT_ObjCProperty &&
      (Tok.is(tok::equal) || Tok.Parent->is(tok::equal)))
    return false;

  if (Tok.Parent->is(tok::comma))
    return true;
  if (Tok.Type == TT_CtorInitializerColon || Tok.Type == TT_ObjCBlockLParen)
    return true;
  if (Tok.Type == TT_OverloadedOperator)
    return Tok.is(tok::identifier) || Tok.is(tok::kw_new) ||
           Tok.is(tok::kw_delete) || Tok.is(tok::kw_bool);
  if (Tok.Parent->Type == TT_OverloadedOperator)
    return false;
  if (Tok.is(tok::colon))
    return Line.First.isNot(tok::kw_case) && !Tok.Children.empty() &&
           Tok.Type != TT_ObjCMethodExpr;
  if (Tok.Parent->Type == TT_UnaryOperator || Tok.Parent->Type == TT_CastRParen)
    return false;
  if (Tok.Type == TT_UnaryOperator)
    return Tok.Parent->isNot(tok::l_paren) &&
           Tok.Parent->isNot(tok::l_square) && Tok.Parent->isNot(tok::at) &&
           (Tok.Parent->isNot(tok::colon) ||
            Tok.Parent->Type != TT_ObjCMethodExpr);
  if (Tok.Parent->is(tok::greater) && Tok.is(tok::greater)) {
    return Tok.Type == TT_TemplateCloser &&
           Tok.Parent->Type == TT_TemplateCloser &&
           Style.Standard != FormatStyle::LS_Cpp11;
  }
  if (Tok.Type == TT_BinaryOperator || Tok.Parent->Type == TT_BinaryOperator)
    return true;
  if (Tok.Parent->Type == TT_TemplateCloser && Tok.is(tok::l_paren))
    return false;
  if (Tok.is(tok::less) && Line.First.is(tok::hash))
    return true;
  if (Tok.Type == TT_TrailingUnaryOperator)
    return false;
  return spaceRequiredBetween(Line, *Tok.Parent, Tok);
}

bool TokenAnnotator::canBreakBefore(const AnnotatedLine &Line,
                                    const AnnotatedToken &Right) {
  const AnnotatedToken &Left = *Right.Parent;
  if (Line.Type == LT_ObjCMethodDecl) {
    if (Right.is(tok::identifier) && !Right.Children.empty() &&
        Right.Children[0].is(tok::colon) && Left.is(tok::identifier))
      return true;
    if (Right.is(tok::identifier) && Left.is(tok::l_paren) &&
        Left.Parent->is(tok::colon))
      // Don't break this identifier as ':' or identifier
      // before it will break.
      return false;
    if (Right.is(tok::colon) && Left.is(tok::identifier) && Left.CanBreakBefore)
      // Don't break at ':' if identifier before it can beak.
      return false;
  }
  if (Right.Type == TT_StartOfName && Style.AllowReturnTypeOnItsOwnLine)
    return true;
  if (Right.is(tok::colon) && Right.Type == TT_ObjCMethodExpr)
    return false;
  if (Left.is(tok::colon) && Left.Type == TT_ObjCMethodExpr)
    return true;
  if (Right.Type == TT_ObjCSelectorName)
    return true;
  if (Left.ClosesTemplateDeclaration)
    return true;
  if (Right.Type == TT_ConditionalExpr || Right.is(tok::question))
    return true;
  if (Left.Type == TT_RangeBasedForLoopColon)
    return true;
  if (Left.Type == TT_PointerOrReference || Left.Type == TT_TemplateCloser ||
      Left.Type == TT_UnaryOperator || Left.Type == TT_ConditionalExpr ||
      Left.is(tok::question))
    return false;
  if (Left.is(tok::equal) && Line.Type == LT_VirtualFunctionDecl)
    return false;

  if (Right.Type == TT_LineComment)
    // We rely on MustBreakBefore being set correctly here as we should not
    // change the "binding" behavior of a comment.
    return false;

  // Allow breaking after a trailing 'const', e.g. after a method declaration,
  // unless it is follow by ';', '{' or '='.
  if (Left.is(tok::kw_const) && Left.Parent != NULL &&
      Left.Parent->is(tok::r_paren))
    return Right.isNot(tok::l_brace) && Right.isNot(tok::semi) &&
           Right.isNot(tok::equal);

  // We only break before r_brace if there was a corresponding break before
  // the l_brace, which is tracked by BreakBeforeClosingBrace.
  if (Right.is(tok::r_brace))
    return false;

  if (Right.is(tok::r_paren) || Right.is(tok::greater))
    return false;
  return (isBinaryOperator(Left) && Left.isNot(tok::lessless)) ||
         Left.is(tok::comma) || Right.is(tok::lessless) ||
         Right.is(tok::arrow) || Right.is(tok::period) ||
         Right.is(tok::colon) || Left.is(tok::coloncolon) ||
         Left.is(tok::semi) || Left.is(tok::l_brace) ||
         (Left.is(tok::r_paren) && Left.Type != TT_CastRParen &&
          Right.is(tok::identifier)) ||
         (Left.is(tok::l_paren) && !Right.is(tok::r_paren)) ||
         (Left.is(tok::l_square) && !Right.is(tok::r_square));
}

} // namespace format
} // namespace clang
