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

static bool closesScope(const AnnotatedToken &Tok) {
  return Tok.isOneOf(tok::r_paren, tok::r_brace, tok::r_square) ||
         Tok.Type == TT_TemplateCloser;
}

static bool opensScope(const AnnotatedToken &Tok) {
  return Tok.isOneOf(tok::l_paren, tok::l_brace, tok::l_square) ||
         Tok.Type == TT_TemplateOpener;
}

/// \brief A parser that gathers additional information about tokens.
///
/// The \c TokenAnnotator tries to match parenthesis and square brakets and
/// store a parenthesis levels. It also tries to resolve matching "<" and ">"
/// into template parameter lists.
class AnnotatingParser {
public:
  AnnotatingParser(SourceManager &SourceMgr, Lexer &Lex, AnnotatedLine &Line,
                   IdentifierInfo &Ident_in)
      : SourceMgr(SourceMgr), Lex(Lex), Line(Line), CurrentToken(&Line.First),
        KeywordVirtualFound(false), Ident_in(Ident_in) {
    Contexts.push_back(Context(1, /*IsExpression=*/ false));
  }

private:
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
      if (CurrentToken->isOneOf(tok::r_paren, tok::r_square, tok::r_brace,
                                tok::pipepipe, tok::ampamp, tok::question,
                                tok::colon))
        return false;
      updateParameterCount(Left, CurrentToken);
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
            Prev.isOneOf(tok::star, tok::amp, tok::ampamp) &&
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
      if (CurrentToken->isOneOf(tok::r_square, tok::r_brace))
        return false;
      updateParameterCount(Left, CurrentToken);
      if (!consumeToken())
        return false;
    }
    return false;
  }

  bool parseSquare() {
    if (!CurrentToken)
      return false;

    // A '[' could be an index subscript (after an indentifier or after
    // ')' or ']'), it could be the start of an Objective-C method
    // expression, or it could the the start of an Objective-C array literal.
    AnnotatedToken *Left = CurrentToken->Parent;
    AnnotatedToken *Parent = getPreviousToken(*Left);
    bool StartsObjCMethodExpr =
        Contexts.back().CanBeExpression &&
        (!Parent || Parent->isOneOf(tok::colon, tok::l_square, tok::l_paren,
                                    tok::kw_return, tok::kw_throw) ||
         isUnaryOperator(*Parent) || Parent->Type == TT_ObjCForIn ||
         Parent->Type == TT_CastRParen ||
         getBinOpPrecedence(Parent->FormatTok.Tok.getKind(), true, true) >
         prec::Unknown);
    ScopedContextCreator ContextCreator(*this, 10);
    Contexts.back().IsExpression = true;
    bool StartsObjCArrayLiteral = Parent && Parent->is(tok::at);

    if (StartsObjCMethodExpr) {
      Contexts.back().ColonIsObjCMethodExpr = true;
      Left->Type = TT_ObjCMethodExpr;
    } else if (StartsObjCArrayLiteral) {
      Left->Type = TT_ObjCArrayLiteral;
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
          if (Parent != NULL && Parent->Type == TT_PointerOrReference)
            Parent->Type = TT_BinaryOperator;
        } else if (StartsObjCArrayLiteral) {
          CurrentToken->Type = TT_ObjCArrayLiteral;
        }
        Left->MatchingParen = CurrentToken;
        CurrentToken->MatchingParen = Left;
        if (Contexts.back().FirstObjCSelectorName != NULL)
          Contexts.back().FirstObjCSelectorName->LongestObjCSelectorName =
              Contexts.back().LongestObjCSelectorName;
        next();
        return true;
      }
      if (CurrentToken->isOneOf(tok::r_paren, tok::r_brace))
        return false;
      updateParameterCount(Left, CurrentToken);
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
      if (CurrentToken->isOneOf(tok::r_paren, tok::r_square))
        return false;
      updateParameterCount(Left, CurrentToken);
      if (!consumeToken())
        return false;
    }
    return true;
  }

  void updateParameterCount(AnnotatedToken *Left, AnnotatedToken *Current) {
    if (Current->is(tok::comma))
      ++Left->ParameterCount;
    else if (Left->ParameterCount == 0 && Current->isNot(tok::comment))
      Left->ParameterCount = 1;
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
      if (CurrentToken != NULL)
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
      } else if (Contexts.size() == 1) {
        Tok->Type = TT_InheritanceColon;
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
      if (Line.MustBeDeclaration)
        Line.MightBeFunctionDecl = true;
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
      while (CurrentToken && CurrentToken->isNot(tok::l_paren)) {
        if (CurrentToken->isOneOf(tok::star, tok::amp))
          CurrentToken->Type = TT_PointerOrReference;
        consumeToken();
      }
      if (CurrentToken)
        CurrentToken->Type = TT_OverloadedOperatorLParen;
      break;
    case tok::question:
      parseConditional();
      break;
    case tok::kw_template:
      parseTemplateDeclaration();
      break;
    case tok::identifier:
      if (Line.First.is(tok::kw_for) &&
          Tok->FormatTok.Tok.getIdentifierInfo() == &Ident_in)
        Tok->Type = TT_ObjCForIn;
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
        if (CurrentToken->is(tok::string_literal))
          // Mark these string literals as "implicit" literals, too, so that
          // they are not split or line-wrapped.
          CurrentToken->Type = TT_ImplicitStringLiteral;
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

public:
  LineType parseLine() {
    int PeriodsAndArrows = 0;
    AnnotatedToken *LastPeriodOrArrow = NULL;
    bool CanBeBuilderTypeStmt = true;
    if (CurrentToken->is(tok::hash)) {
      parsePreprocessorDirective();
      return LT_PreprocessorDirective;
    }
    while (CurrentToken != NULL) {
      if (CurrentToken->is(tok::kw_virtual))
        KeywordVirtualFound = true;
      if (CurrentToken->isOneOf(tok::period, tok::arrow)) {
        ++PeriodsAndArrows;
        LastPeriodOrArrow = CurrentToken;
      }
      AnnotatedToken *TheToken = CurrentToken;
      if (!consumeToken())
        return LT_Invalid;
      if (getPrecedence(*TheToken) > prec::Assignment &&
          TheToken->Type == TT_BinaryOperator)
        CanBeBuilderTypeStmt = false;
    }
    if (KeywordVirtualFound)
      return LT_VirtualFunctionDecl;

    // Assume a builder-type call if there are 2 or more "." and "->".
    if (PeriodsAndArrows >= 2 && CanBeBuilderTypeStmt) {
      LastPeriodOrArrow->LastInChainOfCalls = true;
      return LT_BuilderTypeCall;
    }

    if (Line.First.Type == TT_ObjCMethodSpecifier) {
      if (Contexts.back().FirstObjCSelectorName != NULL)
        Contexts.back().FirstObjCSelectorName->LongestObjCSelectorName =
            Contexts.back().LongestObjCSelectorName;
      return LT_ObjCMethodDecl;
    }

    return LT_Other;
  }

private:
  void next() {
    if (CurrentToken != NULL) {
      determineTokenType(*CurrentToken);
      CurrentToken->BindingStrength = Contexts.back().BindingStrength;
    }

    if (CurrentToken != NULL && !CurrentToken->Children.empty())
      CurrentToken = &CurrentToken->Children[0];
    else
      CurrentToken = NULL;

    // Reset token type in case we have already looked at it and then recovered
    // from an error (e.g. failure to find the matching >).
    if (CurrentToken != NULL)
      CurrentToken->Type = TT_Unknown;
  }

  /// \brief A struct to hold information valid in a specific context, e.g.
  /// a pair of parenthesis.
  struct Context {
    Context(unsigned BindingStrength, bool IsExpression)
        : BindingStrength(BindingStrength), LongestObjCSelectorName(0),
          ColonIsForRangeExpr(false), ColonIsObjCMethodExpr(false),
          FirstObjCSelectorName(NULL), IsExpression(IsExpression),
          CanBeExpression(true) {}

    unsigned BindingStrength;
    unsigned LongestObjCSelectorName;
    bool ColonIsForRangeExpr;
    bool ColonIsObjCMethodExpr;
    AnnotatedToken *FirstObjCSelectorName;
    bool IsExpression;
    bool CanBeExpression;
  };

  /// \brief Puts a new \c Context onto the stack \c Contexts for the lifetime
  /// of each instance.
  struct ScopedContextCreator {
    AnnotatingParser &P;

    ScopedContextCreator(AnnotatingParser &P, unsigned Increase) : P(P) {
      P.Contexts.push_back(Context(P.Contexts.back().BindingStrength + Increase,
                                   P.Contexts.back().IsExpression));
    }

    ~ScopedContextCreator() { P.Contexts.pop_back(); }
  };

  void determineTokenType(AnnotatedToken &Current) {
    if (getPrecedence(Current) == prec::Assignment) {
      Contexts.back().IsExpression = true;
      for (AnnotatedToken *Previous = Current.Parent;
           Previous && Previous->isNot(tok::comma);
           Previous = Previous->Parent) {
        if (Previous->is(tok::r_square))
          Previous = Previous->MatchingParen;
        if (Previous->Type == TT_BinaryOperator &&
            Previous->isOneOf(tok::star, tok::amp)) {
          Previous->Type = TT_PointerOrReference;
        }
      }
    } else if (Current.isOneOf(tok::kw_return, tok::kw_throw) ||
               (Current.is(tok::l_paren) && !Line.MustBeDeclaration &&
                (!Current.Parent || Current.Parent->isNot(tok::kw_for)))) {
      Contexts.back().IsExpression = true;
    } else if (Current.isOneOf(tok::r_paren, tok::greater, tok::comma)) {
      for (AnnotatedToken *Previous = Current.Parent;
           Previous && Previous->isOneOf(tok::star, tok::amp);
           Previous = Previous->Parent)
        Previous->Type = TT_PointerOrReference;
    } else if (Current.Parent &&
               Current.Parent->Type == TT_CtorInitializerColon) {
      Contexts.back().IsExpression = true;
    } else if (Current.is(tok::kw_new)) {
      Contexts.back().CanBeExpression = false;
    }

    if (Current.Type == TT_Unknown) {
      if (Current.Parent && Current.is(tok::identifier) &&
          ((Current.Parent->is(tok::identifier) &&
            Current.Parent->FormatTok.Tok.getIdentifierInfo()
                ->getPPKeywordID() == tok::pp_not_keyword) ||
           Current.Parent->Type == TT_PointerOrReference ||
           Current.Parent->Type == TT_TemplateCloser)) {
        Current.Type = TT_StartOfName;
      } else if (Current.isOneOf(tok::star, tok::amp, tok::ampamp)) {
        Current.Type =
            determineStarAmpUsage(Current, Contexts.back().IsExpression);
      } else if (Current.isOneOf(tok::minus, tok::plus, tok::caret)) {
        Current.Type = determinePlusMinusCaretUsage(Current);
      } else if (Current.isOneOf(tok::minusminus, tok::plusplus)) {
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
      } else if (Current.is(tok::r_paren)) {
        bool ParensNotExpr = !Current.Parent ||
                             Current.Parent->Type == TT_PointerOrReference ||
                             Current.Parent->Type == TT_TemplateCloser;
        bool ParensCouldEndDecl =
            !Current.Children.empty() &&
            Current.Children[0].isOneOf(tok::equal, tok::semi, tok::l_brace);
        if (ParensNotExpr && !ParensCouldEndDecl &&
            Contexts.back().IsExpression)
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

  /// \brief Return the type of the given token assuming it is * or &.
  TokenType
  determineStarAmpUsage(const AnnotatedToken &Tok, bool IsExpression) {
    const AnnotatedToken *PrevToken = getPreviousToken(Tok);
    if (PrevToken == NULL)
      return TT_UnaryOperator;

    const AnnotatedToken *NextToken = getNextToken(Tok);
    if (NextToken == NULL)
      return TT_Unknown;

    if (PrevToken->is(tok::l_paren) && !IsExpression)
      return TT_PointerOrReference;

    if (PrevToken->isOneOf(tok::l_paren, tok::l_square, tok::l_brace,
                           tok::comma, tok::kw_return, tok::colon,
                           tok::equal) ||
        PrevToken->Type == TT_BinaryOperator ||
        PrevToken->Type == TT_UnaryOperator || PrevToken->Type == TT_CastRParen)
      return TT_UnaryOperator;

    if (NextToken->is(tok::l_square))
      return TT_PointerOrReference;

    if (PrevToken->FormatTok.Tok.isLiteral() ||
        PrevToken->isOneOf(tok::r_paren, tok::r_square) ||
        NextToken->FormatTok.Tok.isLiteral() || isUnaryOperator(*NextToken) ||
        NextToken->isOneOf(tok::l_paren, tok::l_square))
      return TT_BinaryOperator;

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
    if (PrevToken->isOneOf(tok::equal, tok::l_paren, tok::comma, tok::l_square,
                           tok::question, tok::colon, tok::kw_return,
                           tok::kw_case, tok::at, tok::l_brace))
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
    if (PrevToken->isOneOf(tok::r_paren, tok::r_square, tok::identifier))
      return TT_TrailingUnaryOperator;

    return TT_UnaryOperator;
  }

  SmallVector<Context, 8> Contexts;

  SourceManager &SourceMgr;
  Lexer &Lex;
  AnnotatedLine &Line;
  AnnotatedToken *CurrentToken;
  bool KeywordVirtualFound;
  IdentifierInfo &Ident_in;
};

/// \brief Parses binary expressions by inserting fake parenthesis based on
/// operator precedence.
class ExpressionParser {
public:
  ExpressionParser(AnnotatedLine &Line) : Current(&Line.First) {}

  /// \brief Parse expressions with the given operatore precedence.
  void parse(int Precedence = 0) {
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

    while (Current) {
      // Consume operators with higher precedence.
      parse(prec::Level(Precedence + 1));

      int CurrentPrecedence = 0;
      if (Current) {
        if (Current->Type == TT_ConditionalExpr)
          CurrentPrecedence = 1 + (int) prec::Conditional;
        else if (Current->is(tok::semi))
          CurrentPrecedence = 1;
        else if (Current->Type == TT_BinaryOperator || Current->is(tok::comma))
          CurrentPrecedence = 1 + (int) getPrecedence(*Current);
      }

      // At the end of the line or when an operator with higher precedence is
      // found, insert fake parenthesis and return.
      if (Current == NULL || closesScope(*Current) ||
          (CurrentPrecedence != 0 && CurrentPrecedence < Precedence)) {
        if (OperatorFound) {
          ++Start->FakeLParens;
          if (Current)
            ++Current->Parent->FakeRParens;
        }
        return;
      }

      // Consume scopes: (), [], <> and {}
      if (opensScope(*Current)) {
        AnnotatedToken *Left = Current;
        while (Current && !closesScope(*Current)) {
          next();
          parse();
        }
        // Remove fake parens that just duplicate the real parens.
        if (Current && Left->Children[0].FakeLParens > 0 &&
            Current->Parent->FakeRParens > 0) {
          --Left->Children[0].FakeLParens;
          --Current->Parent->FakeRParens;
        }
        next();
      } else {
        // Operator found.
        if (CurrentPrecedence == Precedence)
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

  AnnotatedToken *Current;
};

void TokenAnnotator::annotate(AnnotatedLine &Line) {
  AnnotatingParser Parser(SourceMgr, Lex, Line, Ident_in);
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

  Line.First.SpacesRequiredBefore = 1;
  Line.First.MustBreakBefore = Line.First.FormatTok.MustBreakBefore;
  Line.First.CanBreakBefore = Line.First.MustBreakBefore;

  Line.First.TotalLength = Line.First.FormatTok.TokenLength;
}

void TokenAnnotator::calculateFormattingInformation(AnnotatedLine &Line) {
  if (Line.First.Children.empty())
    return;
  AnnotatedToken *Current = &Line.First.Children[0];
  while (Current != NULL) {
    if (Current->Type == TT_LineComment)
      Current->SpacesRequiredBefore = Style.SpacesBeforeTrailingComments;
    else
      Current->SpacesRequiredBefore =
          spaceRequiredBefore(Line, *Current) ? 1 : 0;

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
          Current->SpacesRequiredBefore;
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

  if (Right.Type == TT_StartOfName) {
    if (Line.First.is(tok::kw_for))
      return 3;
    else if (Line.MightBeFunctionDecl && Right.BindingStrength == 1)
      // FIXME: Clean up hack of using BindingStrength to find top-level names.
      return Style.PenaltyReturnTypeOnItsOwnLine;
    else
      return 100;
  }
  if (Left.is(tok::equal) && Right.is(tok::l_brace))
    return 150;
  if (Left.is(tok::coloncolon))
    return 500;

  if (Left.Type == TT_RangeBasedForLoopColon ||
      Left.Type == TT_InheritanceColon)
    return 2;

  if (Right.isOneOf(tok::arrow, tok::period)) {
    if (Line.Type == LT_BuilderTypeCall)
      return 5;
    if (Left.isOneOf(tok::r_paren, tok::r_square) && Left.MatchingParen &&
        Left.MatchingParen->ParameterCount > 0)
      return 20; // Should be smaller than breaking at a nested comma.
    return 150;
  }

  // In for-loops, prefer breaking at ',' and ';'.
  if (Line.First.is(tok::kw_for) && Left.is(tok::equal))
    return 4;

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

  if (opensScope(Left))
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
    return prec::Conditional;
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
  if (Left.isOneOf(tok::hashhash, tok::hash))
    return Right.is(tok::hash);
  if (Right.isOneOf(tok::r_paren, tok::semi, tok::comma))
    return false;
  if (Right.is(tok::less) &&
      (Left.is(tok::kw_template) ||
       (Line.Type == LT_ObjCDecl && Style.ObjCSpaceBeforeProtocolList)))
    return true;
  if (Left.is(tok::arrow) || Right.is(tok::arrow))
    return false;
  if (Left.isOneOf(tok::exclaim, tok::tilde))
    return false;
  if (Left.is(tok::at) &&
      Right.isOneOf(tok::identifier, tok::string_literal, tok::char_constant,
                    tok::numeric_constant, tok::l_paren, tok::l_brace,
                    tok::kw_true, tok::kw_false))
    return false;
  if (Left.is(tok::coloncolon))
    return false;
  if (Right.is(tok::coloncolon))
    return !Left.isOneOf(tok::identifier, tok::greater, tok::l_paren);
  if (Left.is(tok::less) || Right.isOneOf(tok::greater, tok::less))
    return false;
  if (Right.Type == TT_PointerOrReference)
    return Left.FormatTok.Tok.isLiteral() ||
           ((Left.Type != TT_PointerOrReference) && Left.isNot(tok::l_paren) &&
            !Style.PointerBindsToType);
  if (Left.Type == TT_PointerOrReference)
    return Right.FormatTok.Tok.isLiteral() ||
           ((Right.Type != TT_PointerOrReference) && Style.PointerBindsToType);
  if (Right.is(tok::star) && Left.is(tok::l_paren))
    return false;
  if (Left.is(tok::l_square))
    return Left.Type == TT_ObjCArrayLiteral && Right.isNot(tok::r_square);
  if (Right.is(tok::r_square))
    return Right.Type == TT_ObjCArrayLiteral;
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
    return Line.Type == LT_ObjCDecl ||
           Left.isOneOf(tok::kw_if, tok::kw_for, tok::kw_while, tok::kw_switch,
                        tok::kw_return, tok::kw_catch, tok::kw_new,
                        tok::kw_delete);
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
  if (Tok.FormatTok.Tok.getIdentifierInfo() &&
      Tok.Parent->FormatTok.Tok.getIdentifierInfo())
    return true; // Never ever merge two identifiers.
  if (Line.Type == LT_ObjCMethodDecl) {
    if (Tok.Parent->Type == TT_ObjCMethodSpecifier)
      return true;
    if (Tok.Parent->is(tok::r_paren) && Tok.is(tok::identifier))
      // Don't space between ')' and <id>
      return false;
  }
  if (Line.Type == LT_ObjCProperty &&
      (Tok.is(tok::equal) || Tok.Parent->is(tok::equal)))
    return false;

  if (Tok.Parent->is(tok::comma))
    return true;
  if (Tok.is(tok::comma))
    return false;
  if (Tok.Type == TT_CtorInitializerColon || Tok.Type == TT_ObjCBlockLParen)
    return true;
  if (Tok.Parent->FormatTok.Tok.is(tok::kw_operator))
    return false;
  if (Tok.Type == TT_OverloadedOperatorLParen)
    return false;
  if (Tok.is(tok::colon))
    return !Line.First.isOneOf(tok::kw_case, tok::kw_default) &&
           !Tok.Children.empty() && Tok.Type != TT_ObjCMethodExpr;
  if (Tok.is(tok::l_paren) && !Tok.Children.empty() &&
      Tok.Children[0].Type == TT_PointerOrReference &&
      !Tok.Children[0].Children.empty() &&
      Tok.Children[0].Children[0].isNot(tok::r_paren))
    return true;
  if (Tok.Parent->Type == TT_UnaryOperator || Tok.Parent->Type == TT_CastRParen)
    return false;
  if (Tok.Type == TT_UnaryOperator)
    return !Tok.Parent->isOneOf(tok::l_paren, tok::l_square, tok::at) &&
           (Tok.Parent->isNot(tok::colon) ||
            Tok.Parent->Type != TT_ObjCMethodExpr);
  if (Tok.Parent->is(tok::greater) && Tok.is(tok::greater)) {
    return Tok.Type == TT_TemplateCloser &&
           Tok.Parent->Type == TT_TemplateCloser &&
           Style.Standard != FormatStyle::LS_Cpp11;
  }
  if (Tok.is(tok::arrowstar) || Tok.Parent->is(tok::arrowstar))
    return false;
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
  if (Right.Type == TT_StartOfName)
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
  if (Right.Type == TT_RangeBasedForLoopColon ||
      Right.Type == TT_InheritanceColon)
    return false;
  if (Left.Type == TT_RangeBasedForLoopColon ||
      Left.Type == TT_InheritanceColon)
    return true;
  if (Right.Type == TT_RangeBasedForLoopColon)
    return false;
  if (Left.Type == TT_PointerOrReference || Left.Type == TT_TemplateCloser ||
      Left.Type == TT_UnaryOperator || Left.Type == TT_ConditionalExpr ||
      Left.isOneOf(tok::question, tok::kw_operator))
    return false;
  if (Left.is(tok::equal) && Line.Type == LT_VirtualFunctionDecl)
    return false;
  if (Left.is(tok::l_paren) && Right.is(tok::l_paren) && Left.Parent &&
      Left.Parent->is(tok::kw___attribute))
    return false;

  if (Right.Type == TT_LineComment)
    // We rely on MustBreakBefore being set correctly here as we should not
    // change the "binding" behavior of a comment.
    return false;

  // Allow breaking after a trailing 'const', e.g. after a method declaration,
  // unless it is follow by ';', '{' or '='.
  if (Left.is(tok::kw_const) && Left.Parent != NULL &&
      Left.Parent->is(tok::r_paren))
    return !Right.isOneOf(tok::l_brace, tok::semi, tok::equal);

  // We only break before r_brace if there was a corresponding break before
  // the l_brace, which is tracked by BreakBeforeClosingBrace.
  if (Right.is(tok::r_brace))
    return false;

  if (Right.isOneOf(tok::r_paren, tok::greater))
    return false;
  if (Left.is(tok::identifier) && Right.is(tok::string_literal))
    return true;
  return (isBinaryOperator(Left) && Left.isNot(tok::lessless)) ||
         Left.isOneOf(tok::comma, tok::coloncolon, tok::semi, tok::l_brace) ||
         Right.isOneOf(tok::lessless, tok::arrow, tok::period, tok::colon) ||
         (Left.is(tok::r_paren) && Left.Type != TT_CastRParen &&
          Right.isOneOf(tok::identifier, tok::kw___attribute)) ||
         (Left.is(tok::l_paren) && !Right.is(tok::r_paren)) ||
         (Left.is(tok::l_square) && !Right.is(tok::r_square));
}

} // namespace format
} // namespace clang
