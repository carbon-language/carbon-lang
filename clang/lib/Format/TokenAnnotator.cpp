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
#include "llvm/Support/Debug.h"

namespace clang {
namespace format {

bool AnnotatedToken::isUnaryOperator() const {
  switch (FormatTok.Tok.getKind()) {
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

bool AnnotatedToken::isBinaryOperator() const {
  // Comma is a binary operator, but does not behave as such wrt. formatting.
  return getPrecedence(*this) > prec::Comma;
}

bool AnnotatedToken::isTrailingComment() const {
  return is(tok::comment) &&
         (Children.empty() || Children[0].FormatTok.NewlinesBefore > 0);
}

AnnotatedToken *AnnotatedToken::getPreviousNoneComment() const {
  AnnotatedToken *Tok = Parent;
  while (Tok != NULL && Tok->is(tok::comment))
    Tok = Tok->Parent;
  return Tok;
}

const AnnotatedToken *AnnotatedToken::getNextNoneComment() const {
  const AnnotatedToken *Tok = Children.empty() ? NULL : &Children[0];
  while (Tok != NULL && Tok->is(tok::comment))
    Tok = Tok->Children.empty() ? NULL : &Tok->Children[0];
  return Tok;
}

bool AnnotatedToken::closesScope() const {
  return isOneOf(tok::r_paren, tok::r_brace, tok::r_square) ||
         Type == TT_TemplateCloser;
}

bool AnnotatedToken::opensScope() const {
  return isOneOf(tok::l_paren, tok::l_brace, tok::l_square) ||
         Type == TT_TemplateOpener;
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
        KeywordVirtualFound(false), NameFound(false), Ident_in(Ident_in) {
    Contexts.push_back(Context(tok::unknown, 1, /*IsExpression=*/ false));
  }

private:
  bool parseAngle() {
    if (CurrentToken == NULL)
      return false;
    ScopedContextCreator ContextCreator(*this, tok::less, 10);
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
                                tok::question, tok::colon))
        return false;
      if (CurrentToken->isOneOf(tok::pipepipe, tok::ampamp) &&
          Line.First.isNot(tok::kw_template))
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
    ScopedContextCreator ContextCreator(*this, tok::l_paren, 1);

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
        if (CurrentToken->Children.empty() ||
            !CurrentToken->Children[0].isOneOf(tok::l_paren, tok::l_square))
          Left->DefinesFunctionType = false;
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
      if (CurrentToken->Parent->Type == TT_PointerOrReference &&
          CurrentToken->Parent->Parent->isOneOf(tok::l_paren, tok::coloncolon))
        Left->DefinesFunctionType = true;
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
    AnnotatedToken *Parent = Left->getPreviousNoneComment();
    bool StartsObjCMethodExpr =
        Contexts.back().CanBeExpression &&
        (!Parent || Parent->isOneOf(tok::colon, tok::l_square, tok::l_paren,
                                    tok::kw_return, tok::kw_throw) ||
         Parent->isUnaryOperator() || Parent->Type == TT_ObjCForIn ||
         Parent->Type == TT_CastRParen ||
         getBinOpPrecedence(Parent->FormatTok.Tok.getKind(), true, true) >
             prec::Unknown);
    ScopedContextCreator ContextCreator(*this, tok::l_square, 10);
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
    if (CurrentToken != NULL) {
      ScopedContextCreator ContextCreator(*this, tok::l_brace, 1);
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
    }
    // No closing "}" found, this probably starts a definition.
    Line.StartsDefinition = true;
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
      if (Tok->Parent == NULL && Line.MustBeDeclaration)
        Tok->Type = TT_ObjCMethodSpecifier;
      break;
    case tok::colon:
      if (Tok->Parent == NULL)
        return false;
      // Colons from ?: are handled in parseConditional().
      if (Tok->Parent->is(tok::r_paren) && Contexts.size() == 1) {
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
      } else if (Contexts.back().ContextKind == tok::l_paren) {
        Tok->Type = TT_InlineASMColon;
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
      if (Line.MustBeDeclaration && NameFound && !Contexts.back().IsExpression)
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
      if (CurrentToken) {
        CurrentToken->Type = TT_OverloadedOperatorLParen;
        if (CurrentToken->Parent->Type == TT_BinaryOperator)
          CurrentToken->Parent->Type = TT_OverloadedOperator;
      }
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
    case tok::comma:
      if (Contexts.back().FirstStartOfName)
        Contexts.back().FirstStartOfName->PartOfMultiVariableDeclStmt = true;
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
    case tok::pp_if:
    case tok::pp_elif:
      parseLine();
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
    Context(tok::TokenKind ContextKind, unsigned BindingStrength,
            bool IsExpression)
        : ContextKind(ContextKind), BindingStrength(BindingStrength),
          LongestObjCSelectorName(0), ColonIsForRangeExpr(false),
          ColonIsObjCMethodExpr(false), FirstObjCSelectorName(NULL),
          FirstStartOfName(NULL), IsExpression(IsExpression),
          CanBeExpression(true) {}

    tok::TokenKind ContextKind;
    unsigned BindingStrength;
    unsigned LongestObjCSelectorName;
    bool ColonIsForRangeExpr;
    bool ColonIsObjCMethodExpr;
    AnnotatedToken *FirstObjCSelectorName;
    AnnotatedToken *FirstStartOfName;
    bool IsExpression;
    bool CanBeExpression;
  };

  /// \brief Puts a new \c Context onto the stack \c Contexts for the lifetime
  /// of each instance.
  struct ScopedContextCreator {
    AnnotatingParser &P;

    ScopedContextCreator(AnnotatingParser &P, tok::TokenKind ContextKind,
                         unsigned Increase)
        : P(P) {
      P.Contexts.push_back(
          Context(ContextKind, P.Contexts.back().BindingStrength + Increase,
                  P.Contexts.back().IsExpression));
    }

    ~ScopedContextCreator() { P.Contexts.pop_back(); }
  };

  void determineTokenType(AnnotatedToken &Current) {
    if (getPrecedence(Current) == prec::Assignment &&
        (!Current.Parent || Current.Parent->isNot(tok::kw_operator))) {
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
                !Line.InPPDirective &&
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
    } else if (Current.is(tok::semi)) {
      // This should be the condition or increment in a for-loop.
      Contexts.back().IsExpression = true;
    }

    if (Current.Type == TT_Unknown) {
      if (Current.Parent && Current.is(tok::identifier) &&
          ((Current.Parent->is(tok::identifier) &&
            Current.Parent->FormatTok.Tok.getIdentifierInfo()
                    ->getPPKeywordID() ==
                tok::pp_not_keyword) ||
           isSimpleTypeSpecifier(*Current.Parent) ||
           Current.Parent->Type == TT_PointerOrReference ||
           Current.Parent->Type == TT_TemplateCloser)) {
        Contexts.back().FirstStartOfName = &Current;
        Current.Type = TT_StartOfName;
        NameFound = true;
      } else if (Current.isOneOf(tok::star, tok::amp, tok::ampamp)) {
        Current.Type =
            determineStarAmpUsage(Current, Contexts.back().IsExpression);
      } else if (Current.isOneOf(tok::minus, tok::plus, tok::caret)) {
        Current.Type = determinePlusMinusCaretUsage(Current);
      } else if (Current.isOneOf(tok::minusminus, tok::plusplus)) {
        Current.Type = determineIncrementUsage(Current);
      } else if (Current.is(tok::exclaim)) {
        Current.Type = TT_UnaryOperator;
      } else if (Current.isBinaryOperator()) {
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
        bool IsSizeOfOrAlignOf =
            Current.MatchingParen && Current.MatchingParen->Parent &&
            Current.MatchingParen->Parent->isOneOf(tok::kw_sizeof,
                                                   tok::kw_alignof);
        if (ParensNotExpr && !ParensCouldEndDecl && !IsSizeOfOrAlignOf &&
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
  TokenType determineStarAmpUsage(const AnnotatedToken &Tok,
                                  bool IsExpression) {
    const AnnotatedToken *PrevToken = Tok.getPreviousNoneComment();
    if (PrevToken == NULL)
      return TT_UnaryOperator;

    const AnnotatedToken *NextToken = Tok.getNextNoneComment();
    if (NextToken == NULL)
      return TT_Unknown;

    if (PrevToken->is(tok::l_paren) && !IsExpression)
      return TT_PointerOrReference;

    if (PrevToken->isOneOf(tok::l_paren, tok::l_square, tok::l_brace,
                           tok::comma, tok::semi, tok::kw_return, tok::colon,
                           tok::equal, tok::kw_delete) ||
        PrevToken->Type == TT_BinaryOperator ||
        PrevToken->Type == TT_UnaryOperator || PrevToken->Type == TT_CastRParen)
      return TT_UnaryOperator;

    if (NextToken->is(tok::l_square))
      return TT_PointerOrReference;

    if (PrevToken->FormatTok.Tok.isLiteral() ||
        PrevToken->isOneOf(tok::r_paren, tok::r_square) ||
        NextToken->FormatTok.Tok.isLiteral() || NextToken->isUnaryOperator())
      return TT_BinaryOperator;

    // It is very unlikely that we are going to find a pointer or reference type
    // definition on the RHS of an assignment.
    if (IsExpression)
      return TT_BinaryOperator;

    return TT_PointerOrReference;
  }

  TokenType determinePlusMinusCaretUsage(const AnnotatedToken &Tok) {
    const AnnotatedToken *PrevToken = Tok.getPreviousNoneComment();
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
    const AnnotatedToken *PrevToken = Tok.getPreviousNoneComment();
    if (PrevToken == NULL)
      return TT_UnaryOperator;
    if (PrevToken->isOneOf(tok::r_paren, tok::r_square, tok::identifier))
      return TT_TrailingUnaryOperator;

    return TT_UnaryOperator;
  }

  // FIXME: This is copy&pasted from Sema. Put it in a common place and remove
  // duplication.
  /// \brief Determine whether the token kind starts a simple-type-specifier.
  bool isSimpleTypeSpecifier(const AnnotatedToken &Tok) const {
    switch (Tok.FormatTok.Tok.getKind()) {
    case tok::kw_short:
    case tok::kw_long:
    case tok::kw___int64:
    case tok::kw___int128:
    case tok::kw_signed:
    case tok::kw_unsigned:
    case tok::kw_void:
    case tok::kw_char:
    case tok::kw_int:
    case tok::kw_half:
    case tok::kw_float:
    case tok::kw_double:
    case tok::kw_wchar_t:
    case tok::kw_bool:
    case tok::kw___underlying_type:
      return true;
    case tok::annot_typename:
    case tok::kw_char16_t:
    case tok::kw_char32_t:
    case tok::kw_typeof:
    case tok::kw_decltype:
      return Lex.getLangOpts().CPlusPlus;
    default:
      break;
    }
    return false;
  }

  SmallVector<Context, 8> Contexts;

  SourceManager &SourceMgr;
  Lexer &Lex;
  AnnotatedLine &Line;
  AnnotatedToken *CurrentToken;
  bool KeywordVirtualFound;
  bool NameFound;
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

    // Eagerly consume trailing comments.
    while (Current && Current->isTrailingComment()) {
      next();
    }

    AnnotatedToken *Start = Current;
    bool OperatorFound = false;

    while (Current) {
      // Consume operators with higher precedence.
      parse(Precedence + 1);

      int CurrentPrecedence = 0;
      if (Current) {
        if (Current->Type == TT_ConditionalExpr)
          CurrentPrecedence = 1 + (int) prec::Conditional;
        else if (Current->is(tok::semi) || Current->Type == TT_InlineASMColon)
          CurrentPrecedence = 1;
        else if (Current->Type == TT_BinaryOperator || Current->is(tok::comma))
          CurrentPrecedence = 1 + (int) getPrecedence(*Current);
      }

      // At the end of the line or when an operator with higher precedence is
      // found, insert fake parenthesis and return.
      if (Current == NULL || Current->closesScope() ||
          (CurrentPrecedence != 0 && CurrentPrecedence < Precedence)) {
        if (OperatorFound) {
          Start->FakeLParens.push_back(prec::Level(Precedence - 1));
          if (Current)
            ++Current->Parent->FakeRParens;
        }
        return;
      }

      // Consume scopes: (), [], <> and {}
      if (Current->opensScope()) {
        while (Current && !Current->closesScope()) {
          next();
          parse();
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
    } else if (Current->Parent->isTrailingComment() ||
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

  DEBUG({
    printDebugInfo(Line);
  });
}

unsigned TokenAnnotator::splitPenalty(const AnnotatedLine &Line,
                                      const AnnotatedToken &Tok) {
  const AnnotatedToken &Left = *Tok.Parent;
  const AnnotatedToken &Right = Tok;

  if (Right.Type == TT_StartOfName) {
    if (Line.First.is(tok::kw_for) && Right.PartOfMultiVariableDeclStmt)
      return 3;
    else if (Line.MightBeFunctionDecl && Right.BindingStrength == 1)
      // FIXME: Clean up hack of using BindingStrength to find top-level names.
      return Style.PenaltyReturnTypeOnItsOwnLine;
    else
      return 200;
  }
  if (Left.is(tok::equal) && Right.is(tok::l_brace))
    return 150;
  if (Left.is(tok::coloncolon))
    return 500;
  if (Left.isOneOf(tok::kw_class, tok::kw_struct))
    return 5000;

  if (Left.Type == TT_RangeBasedForLoopColon ||
      Left.Type == TT_InheritanceColon)
    return 2;

  if (Right.isOneOf(tok::arrow, tok::period)) {
    if (Line.Type == LT_BuilderTypeCall)
      return prec::PointerToMember;
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

  if (Left.is(tok::l_paren) && Line.MightBeFunctionDecl)
    return 100;
  if (Left.opensScope())
    return Left.ParameterCount > 1 ? prec::Comma : 20;

  if (Right.is(tok::lessless)) {
    if (Left.is(tok::string_literal)) {
      StringRef Content = StringRef(Left.FormatTok.Tok.getLiteralData(),
                                    Left.FormatTok.TokenLength);
      Content = Content.drop_back(1).drop_front(1).trim();
      if (Content.size() > 1 &&
          (Content.back() == ':' || Content.back() == '='))
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
  if (Right.DefinesFunctionType &&
      (Left.Type != TT_PointerOrReference || Style.PointerBindsToType))
    return true;
  if (Left.Type == TT_PointerOrReference)
    return Right.FormatTok.Tok.isLiteral() ||
           ((Right.Type != TT_PointerOrReference) &&
            Right.isNot(tok::l_paren) && Style.PointerBindsToType &&
            Left.Parent &&
            !Left.Parent->isOneOf(tok::l_paren, tok::coloncolon));
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
                        tok::kw_delete, tok::semi);
  }
  if (Left.is(tok::at) &&
      Right.FormatTok.Tok.getObjCKeywordID() != tok::objc_not_keyword)
    return false;
  if (Left.is(tok::l_brace) && Right.is(tok::r_brace))
    return false;
  if (Right.is(tok::ellipsis))
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
           Tok.getNextNoneComment() != NULL && Tok.Type != TT_ObjCMethodExpr;
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
  if (Tok.isOneOf(tok::arrowstar, tok::periodstar) ||
      Tok.Parent->isOneOf(tok::arrowstar, tok::periodstar))
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
      Right.Type == TT_OverloadedOperatorLParen)
    return false;
  if (Left.Type == TT_RangeBasedForLoopColon)
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

  if (Right.is(tok::kw___attribute))
    return true;

  // We only break before r_brace if there was a corresponding break before
  // the l_brace, which is tracked by BreakBeforeClosingBrace.
  if (Right.isOneOf(tok::r_brace, tok::r_paren, tok::greater))
    return false;
  if (Left.is(tok::identifier) && Right.is(tok::string_literal))
    return true;
  return (Left.isBinaryOperator() && Left.isNot(tok::lessless)) ||
         Left.isOneOf(tok::comma, tok::coloncolon, tok::semi, tok::l_brace,
                      tok::kw_class, tok::kw_struct) ||
         Right.isOneOf(tok::lessless, tok::arrow, tok::period, tok::colon) ||
         (Left.is(tok::r_paren) && Left.Type != TT_CastRParen &&
          Right.isOneOf(tok::identifier, tok::kw_const, tok::kw___attribute)) ||
         (Left.is(tok::l_paren) && !Right.is(tok::r_paren)) ||
         (Left.is(tok::l_square) && !Right.is(tok::r_square));
}

void TokenAnnotator::printDebugInfo(const AnnotatedLine &Line) {
  llvm::errs() << "AnnotatedTokens:\n";
  const AnnotatedToken *Tok = &Line.First;
  while (Tok) {
    llvm::errs() << " M=" << Tok->MustBreakBefore
                 << " C=" << Tok->CanBreakBefore << " T=" << Tok->Type
                 << " S=" << Tok->SpacesRequiredBefore
                 << " P=" << Tok->SplitPenalty
                 << " Name=" << Tok->FormatTok.Tok.getName() << " FakeLParens=";
    for (unsigned i = 0, e = Tok->FakeLParens.size(); i != e; ++i)
      llvm::errs() << Tok->FakeLParens[i] << "/";
    llvm::errs() << " FakeRParens=" << Tok->FakeRParens << "\n";
    Tok = Tok->Children.empty() ? NULL : &Tok->Children[0];
  }
  llvm::errs() << "----\n";
}

} // namespace format
} // namespace clang
