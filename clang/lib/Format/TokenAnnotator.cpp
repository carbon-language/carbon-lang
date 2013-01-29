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

/// \brief Returns if a token is an Objective-C selector name.
///
/// For example, "bar" is a selector name in [foo bar:(4 + 5)].
static bool isObjCSelectorName(const AnnotatedToken &Tok) {
  return Tok.is(tok::identifier) && !Tok.Children.empty() &&
         Tok.Children[0].is(tok::colon) &&
         Tok.Children[0].Type == TT_ObjCMethodExpr;
}

static bool isBinaryOperator(const AnnotatedToken &Tok) {
  // Comma is a binary operator, but does not behave as such wrt. formatting.
  return getPrecedence(Tok) > prec::Comma;
}

/// \brief A parser that gathers additional information about tokens.
///
/// The \c TokenAnnotator tries to matches parenthesis and square brakets and
/// store a parenthesis levels. It also tries to resolve matching "<" and ">"
/// into template parameter lists.
class AnnotatingParser {
public:
  AnnotatingParser(AnnotatedToken &RootToken)
      : CurrentToken(&RootToken), KeywordVirtualFound(false),
        ColonIsObjCMethodExpr(false), ColonIsForRangeExpr(false) {
  }

  /// \brief A helper class to manage AnnotatingParser::ColonIsObjCMethodExpr.
  struct ObjCSelectorRAII {
    AnnotatingParser &P;
    bool ColonWasObjCMethodExpr;

    ObjCSelectorRAII(AnnotatingParser &P)
        : P(P), ColonWasObjCMethodExpr(P.ColonIsObjCMethodExpr) {
    }

    ~ObjCSelectorRAII() { P.ColonIsObjCMethodExpr = ColonWasObjCMethodExpr; }

    void markStart(AnnotatedToken &Left) {
      P.ColonIsObjCMethodExpr = true;
      Left.Type = TT_ObjCMethodExpr;
    }

    void markEnd(AnnotatedToken &Right) { Right.Type = TT_ObjCMethodExpr; }
  };

  bool parseAngle() {
    if (CurrentToken == NULL)
      return false;
    AnnotatedToken *Left = CurrentToken->Parent;
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

    ObjCSelectorRAII objCSelector(*this);
    if (StartsObjCMethodExpr)
      objCSelector.markStart(*Left);

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

        if (StartsObjCMethodExpr)
          objCSelector.markEnd(*CurrentToken);

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

    // A '[' could be an index subscript (after an indentifier or after
    // ')' or ']'), or it could be the start of an Objective-C method
    // expression.
    AnnotatedToken *Left = CurrentToken->Parent;
    bool StartsObjCMethodExpr =
        !Left->Parent || Left->Parent->is(tok::colon) ||
        Left->Parent->is(tok::l_square) || Left->Parent->is(tok::l_paren) ||
        Left->Parent->is(tok::kw_return) || Left->Parent->is(tok::kw_throw) ||
        getBinOpPrecedence(Left->Parent->FormatTok.Tok.getKind(), true, true) >
        prec::Unknown;

    ObjCSelectorRAII objCSelector(*this);
    if (StartsObjCMethodExpr)
      objCSelector.markStart(*Left);

    while (CurrentToken != NULL) {
      if (CurrentToken->is(tok::r_square)) {
        if (!CurrentToken->Children.empty() &&
            CurrentToken->Children[0].is(tok::l_paren)) {
          // An ObjC method call can't be followed by an open parenthesis.
          // FIXME: Do we incorrectly label ":" with this?
          StartsObjCMethodExpr = false;
          Left->Type = TT_Unknown;
        }
        if (StartsObjCMethodExpr)
          objCSelector.markEnd(*CurrentToken);
        Left->MatchingParen = CurrentToken;
        CurrentToken->MatchingParen = Left;
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
      if (Tok->Parent->is(tok::r_paren))
        Tok->Type = TT_CtorInitializerColon;
      else if (ColonIsObjCMethodExpr)
        Tok->Type = TT_ObjCMethodExpr;
      else if (ColonIsForRangeExpr)
        Tok->Type = TT_RangeBasedForLoopColon;
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
      ColonIsForRangeExpr = true;
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

    return LT_Other;
  }

  void next() {
    if (CurrentToken != NULL && !CurrentToken->Children.empty())
      CurrentToken = &CurrentToken->Children[0];
    else
      CurrentToken = NULL;
  }

private:
  AnnotatedToken *CurrentToken;
  bool KeywordVirtualFound;
  bool ColonIsObjCMethodExpr;
  bool ColonIsForRangeExpr;
};

void TokenAnnotator::annotate() {
  AnnotatingParser Parser(Line.First);
  Line.Type = Parser.parseLine();
  if (Line.Type == LT_Invalid)
    return;

  bool LookForFunctionName = Line.MustBeDeclaration;
  determineTokenTypes(Line.First, /*IsExpression=*/ false, LookForFunctionName);

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
  if (!Line.First.Children.empty())
    calculateExtraInformation(Line.First.Children[0]);
}

void TokenAnnotator::calculateExtraInformation(AnnotatedToken &Current) {
  Current.SpaceRequiredBefore = spaceRequiredBefore(Current);

  if (Current.FormatTok.MustBreakBefore) {
    Current.MustBreakBefore = true;
  } else {
    if (Current.Type == TT_LineComment) {
      Current.MustBreakBefore = Current.FormatTok.NewlinesBefore > 0;
    } else if ((Current.Parent->is(tok::comment) &&
                Current.FormatTok.NewlinesBefore > 0) ||
               (Current.is(tok::string_literal) &&
                Current.Parent->is(tok::string_literal))) {
      Current.MustBreakBefore = true;
    } else {
      Current.MustBreakBefore = false;
    }
  }
  Current.CanBreakBefore = Current.MustBreakBefore || canBreakBefore(Current);
  if (Current.MustBreakBefore)
    Current.TotalLength = Current.Parent->TotalLength + Style.ColumnLimit;
  else
    Current.TotalLength =
        Current.Parent->TotalLength + Current.FormatTok.TokenLength +
        (Current.SpaceRequiredBefore ? 1 : 0);
  // FIXME: Only calculate this if CanBreakBefore is true once static
  // initializers etc. are sorted out.
  Current.SplitPenalty = splitPenalty(Current);
  if (!Current.Children.empty())
    calculateExtraInformation(Current.Children[0]);
}

unsigned TokenAnnotator::splitPenalty(const AnnotatedToken &Tok) {
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

  if (Left.is(tok::semi) || Left.is(tok::comma))
    return 0;

  // In Objective-C method expressions, prefer breaking before "param:" over
  // breaking after it.
  if (isObjCSelectorName(Right))
    return 0;
  if (Right.is(tok::colon) && Right.Type == TT_ObjCMethodExpr)
    return 20;

  if (Left.is(tok::l_paren))
    return 20;
  // FIXME: The penalty for a trailing "<" or "[" being higher than the
  // penalty for a trainling "(" is a temporary workaround until we can
  // properly avoid breaking in array subscripts or template parameters.
  if (Left.is(tok::l_square) || Left.Type == TT_TemplateOpener)
    return 50;

  if (Left.Type == TT_ConditionalExpr)
    return prec::Assignment;
  prec::Level Level = getPrecedence(Left);

  if (Level != prec::Unknown)
    return Level;

  return 3;
}

void TokenAnnotator::determineTokenTypes(
    AnnotatedToken &Current, bool IsExpression, bool LookForFunctionName) {
  if (getPrecedence(Current) == prec::Assignment) {
    IsExpression = true;
    AnnotatedToken *Previous = Current.Parent;
    while (Previous != NULL) {
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
    IsExpression = true;

  if (Current.Type == TT_Unknown) {
    if (LookForFunctionName && Current.is(tok::l_paren)) {
      findFunctionName(&Current);
      LookForFunctionName = false;
    } else if (Current.is(tok::star) || Current.is(tok::amp)) {
      Current.Type = determineStarAmpUsage(Current, IsExpression);
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

  if (!Current.Children.empty())
    determineTokenTypes(Current.Children[0], IsExpression, LookForFunctionName);
}

void TokenAnnotator::findFunctionName(AnnotatedToken *Current) {
  AnnotatedToken *Parent = Current->Parent;
  while (Parent != NULL && Parent->Parent != NULL) {
    if (Parent->is(tok::identifier) &&
        (Parent->Parent->is(tok::identifier) || Parent->Parent->Type ==
         TT_PointerOrReference || Parent->Parent->Type == TT_TemplateCloser)) {
      Parent->Type = TT_StartOfName;
      break;
    }
    Parent = Parent->Parent;
  }
}

TokenType TokenAnnotator::determineStarAmpUsage(const AnnotatedToken &Tok,
                                                bool IsExpression) {
  const AnnotatedToken *PrevToken = getPreviousToken(Tok);
  if (PrevToken == NULL)
    return TT_UnaryOperator;

  const AnnotatedToken *NextToken = getNextToken(Tok);
  if (NextToken == NULL)
    return TT_Unknown;

  if (NextToken->is(tok::l_square) && NextToken->Type != TT_ObjCMethodExpr)
    return TT_PointerOrReference;

  if (PrevToken->is(tok::l_paren) || PrevToken->is(tok::l_square) ||
      PrevToken->is(tok::l_brace) || PrevToken->is(tok::comma) ||
      PrevToken->is(tok::kw_return) || PrevToken->is(tok::colon) ||
      PrevToken->Type == TT_BinaryOperator ||
      PrevToken->Type == TT_UnaryOperator || PrevToken->Type == TT_CastRParen)
    return TT_UnaryOperator;

  if (PrevToken->FormatTok.Tok.isLiteral() || PrevToken->is(tok::r_paren) ||
      PrevToken->is(tok::r_square) || NextToken->FormatTok.Tok.isLiteral() ||
      NextToken->is(tok::plus) || NextToken->is(tok::minus) ||
      NextToken->is(tok::plusplus) || NextToken->is(tok::minusminus) ||
      NextToken->is(tok::tilde) || NextToken->is(tok::exclaim) ||
      NextToken->is(tok::l_paren) || NextToken->is(tok::l_square) ||
      NextToken->is(tok::kw_alignof) || NextToken->is(tok::kw_sizeof))
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

TokenType
TokenAnnotator::determinePlusMinusCaretUsage(const AnnotatedToken &Tok) {
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

  // There can't be to consecutive binary operators.
  if (PrevToken->Type == TT_BinaryOperator)
    return TT_UnaryOperator;

  // Fall back to marking the token as binary operator.
  return TT_BinaryOperator;
}

TokenType TokenAnnotator::determineIncrementUsage(const AnnotatedToken &Tok) {
  const AnnotatedToken *PrevToken = getPreviousToken(Tok);
  if (PrevToken == NULL)
    return TT_UnaryOperator;
  if (PrevToken->is(tok::r_paren) || PrevToken->is(tok::r_square) ||
      PrevToken->is(tok::identifier))
    return TT_TrailingUnaryOperator;

  return TT_UnaryOperator;
}

bool TokenAnnotator::spaceRequiredBetween(const AnnotatedToken &Left,
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
    return Left.isNot(tok::identifier) && Left.isNot(tok::greater);
  if (Left.is(tok::less) || Right.is(tok::greater) || Right.is(tok::less))
    return false;
  if (Right.is(tok::amp) || Right.is(tok::star))
    return Left.FormatTok.Tok.isLiteral() ||
           (Left.isNot(tok::star) && Left.isNot(tok::amp) &&
            !Style.PointerAndReferenceBindToType);
  if (Left.is(tok::amp) || Left.is(tok::star))
    return Right.FormatTok.Tok.isLiteral() ||
           Style.PointerAndReferenceBindToType;
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

bool TokenAnnotator::spaceRequiredBefore(const AnnotatedToken &Tok) {
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
    return Tok.Type == TT_TemplateCloser && Tok.Parent->Type ==
           TT_TemplateCloser && Style.SplitTemplateClosingGreater;
  }
  if (Tok.Type == TT_BinaryOperator || Tok.Parent->Type == TT_BinaryOperator)
    return true;
  if (Tok.Parent->Type == TT_TemplateCloser && Tok.is(tok::l_paren))
    return false;
  if (Tok.is(tok::less) && Line.First.is(tok::hash))
    return true;
  if (Tok.Type == TT_TrailingUnaryOperator)
    return false;
  return spaceRequiredBetween(*Tok.Parent, Tok);
}

bool TokenAnnotator::canBreakBefore(const AnnotatedToken &Right) {
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
  if (isObjCSelectorName(Right))
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
