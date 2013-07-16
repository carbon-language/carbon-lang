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
#include "llvm/Support/Debug.h"

namespace clang {
namespace format {

namespace {

/// \brief A parser that gathers additional information about tokens.
///
/// The \c TokenAnnotator tries to match parenthesis and square brakets and
/// store a parenthesis levels. It also tries to resolve matching "<" and ">"
/// into template parameter lists.
class AnnotatingParser {
public:
  AnnotatingParser(AnnotatedLine &Line, IdentifierInfo &Ident_in)
      : Line(Line), CurrentToken(Line.First), KeywordVirtualFound(false),
        NameFound(false), AutoFound(false), Ident_in(Ident_in) {
    Contexts.push_back(Context(tok::unknown, 1, /*IsExpression=*/false));
  }

private:
  bool parseAngle() {
    if (CurrentToken == NULL)
      return false;
    ScopedContextCreator ContextCreator(*this, tok::less, 10);
    FormatToken *Left = CurrentToken->Previous;
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
      // If a && or || is found and interpreted as a binary operator, this set
      // of angles is likely part of something like "a < b && c > d". If the
      // angles are inside an expression, the ||/&& might also be a binary
      // operator that was misinterpreted because we are parsing template
      // parameters.
      // FIXME: This is getting out of hand, write a decent parser.
      if (CurrentToken->Previous->isOneOf(tok::pipepipe, tok::ampamp) &&
          (CurrentToken->Previous->Type == TT_BinaryOperator ||
           Contexts[Contexts.size() - 2].IsExpression) &&
          Line.First->isNot(tok::kw_template))
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
    FormatToken *Left = CurrentToken->Previous;
    if (CurrentToken->is(tok::caret)) {
      // ^( starts a block.
      Left->Type = TT_ObjCBlockLParen;
    } else if (FormatToken *MaybeSel = Left->Previous) {
      // @selector( starts a selector.
      if (MaybeSel->isObjCAtKeyword(tok::objc_selector) && MaybeSel->Previous &&
          MaybeSel->Previous->is(tok::at)) {
        StartsObjCMethodExpr = true;
      }
    }

    if (StartsObjCMethodExpr) {
      Contexts.back().ColonIsObjCMethodExpr = true;
      Left->Type = TT_ObjCMethodExpr;
    }

    bool MightBeFunctionType = CurrentToken->is(tok::star);
    bool HasMultipleLines = false;
    bool HasMultipleParametersOnALine = false;
    while (CurrentToken != NULL) {
      // LookForDecls is set when "if (" has been seen. Check for
      // 'identifier' '*' 'identifier' followed by not '=' -- this
      // '*' has to be a binary operator but determineStarAmpUsage() will
      // categorize it as an unary operator, so set the right type here.
      if (LookForDecls && CurrentToken->Next) {
        FormatToken *Prev = CurrentToken->getPreviousNonComment();
        if (Prev) {
          FormatToken *PrevPrev = Prev->getPreviousNonComment();
          FormatToken *Next = CurrentToken->Next;
          if (PrevPrev && PrevPrev->is(tok::identifier) &&
              Prev->isOneOf(tok::star, tok::amp, tok::ampamp) &&
              CurrentToken->is(tok::identifier) && Next->isNot(tok::equal)) {
            Prev->Type = TT_BinaryOperator;
            LookForDecls = false;
          }
        }
      }

      if (CurrentToken->is(tok::r_paren)) {
        if (MightBeFunctionType && CurrentToken->Next &&
            (CurrentToken->Next->is(tok::l_paren) ||
             (CurrentToken->Next->is(tok::l_square) &&
              !Contexts.back().IsExpression)))
          Left->Type = TT_FunctionTypeLParen;
        Left->MatchingParen = CurrentToken;
        CurrentToken->MatchingParen = Left;

        if (StartsObjCMethodExpr) {
          CurrentToken->Type = TT_ObjCMethodExpr;
          if (Contexts.back().FirstObjCSelectorName != NULL) {
            Contexts.back().FirstObjCSelectorName->LongestObjCSelectorName =
                Contexts.back().LongestObjCSelectorName;
          }
        }

        if (!HasMultipleLines)
          Left->PackingKind = PPK_Inconclusive;
        else if (HasMultipleParametersOnALine)
          Left->PackingKind = PPK_BinPacked;
        else
          Left->PackingKind = PPK_OnePerLine;

        next();
        return true;
      }
      if (CurrentToken->isOneOf(tok::r_square, tok::r_brace))
        return false;
      if (CurrentToken->Previous->Type == TT_PointerOrReference &&
          CurrentToken->Previous->Previous->isOneOf(tok::l_paren,
                                                    tok::coloncolon))
        MightBeFunctionType = true;
      updateParameterCount(Left, CurrentToken);
      if (CurrentToken->is(tok::comma) && CurrentToken->Next &&
          !CurrentToken->Next->HasUnescapedNewline &&
          !CurrentToken->Next->isTrailingComment())
        HasMultipleParametersOnALine = true;
      if (!consumeToken())
        return false;
      if (CurrentToken && CurrentToken->HasUnescapedNewline)
        HasMultipleLines = true;
    }
    return false;
  }

  bool parseSquare() {
    if (!CurrentToken)
      return false;

    // A '[' could be an index subscript (after an identifier or after
    // ')' or ']'), it could be the start of an Objective-C method
    // expression, or it could the the start of an Objective-C array literal.
    FormatToken *Left = CurrentToken->Previous;
    FormatToken *Parent = Left->getPreviousNonComment();
    bool StartsObjCMethodExpr =
        Contexts.back().CanBeExpression &&
        (!Parent || Parent->isOneOf(tok::colon, tok::l_square, tok::l_paren,
                                    tok::kw_return, tok::kw_throw) ||
         Parent->isUnaryOperator() || Parent->Type == TT_ObjCForIn ||
         Parent->Type == TT_CastRParen ||
         getBinOpPrecedence(Parent->Tok.getKind(), true, true) > prec::Unknown);
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
        if (CurrentToken->Next && CurrentToken->Next->is(tok::l_paren)) {
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
      FormatToken *Left = CurrentToken->Previous;

      FormatToken *Parent = Left->getPreviousNonComment();
      bool StartsObjCDictLiteral = Parent && Parent->is(tok::at);
      if (StartsObjCDictLiteral) {
        Contexts.back().ColonIsObjCDictLiteral = true;
        Left->Type = TT_ObjCDictLiteral;
      }

      while (CurrentToken != NULL) {
        if (CurrentToken->is(tok::r_brace)) {
          if (StartsObjCDictLiteral)
            CurrentToken->Type = TT_ObjCDictLiteral;
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

  void updateParameterCount(FormatToken *Left, FormatToken *Current) {
    if (Current->is(tok::comma)) {
      ++Left->ParameterCount;
    } else if (Left->ParameterCount == 0 && Current->isNot(tok::comment)) {
      Left->ParameterCount = 1;
    }
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
        CurrentToken->Previous->ClosesTemplateDeclaration = true;
      return true;
    }
    return false;
  }

  bool consumeToken() {
    FormatToken *Tok = CurrentToken;
    next();
    switch (Tok->Tok.getKind()) {
    case tok::plus:
    case tok::minus:
      if (Tok->Previous == NULL && Line.MustBeDeclaration)
        Tok->Type = TT_ObjCMethodSpecifier;
      break;
    case tok::colon:
      if (Tok->Previous == NULL)
        return false;
      // Colons from ?: are handled in parseConditional().
      if (Tok->Previous->is(tok::r_paren) && Contexts.size() == 1) {
        Tok->Type = TT_CtorInitializerColon;
      } else if (Contexts.back().ColonIsObjCDictLiteral) {
        Tok->Type = TT_ObjCDictLiteral;
      } else if (Contexts.back().ColonIsObjCMethodExpr ||
                 Line.First->Type == TT_ObjCMethodSpecifier) {
        Tok->Type = TT_ObjCMethodExpr;
        Tok->Previous->Type = TT_ObjCSelectorName;
        if (Tok->Previous->CodePointCount >
            Contexts.back().LongestObjCSelectorName) {
          Contexts.back().LongestObjCSelectorName =
              Tok->Previous->CodePointCount;
        }
        if (Contexts.back().FirstObjCSelectorName == NULL)
          Contexts.back().FirstObjCSelectorName = Tok->Previous;
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
        if (!parseParens(/*LookForDecls=*/true))
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
      if (Tok->Previous != NULL)
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
        if (CurrentToken->Previous->Type == TT_BinaryOperator)
          CurrentToken->Previous->Type = TT_OverloadedOperator;
      }
      break;
    case tok::question:
      parseConditional();
      break;
    case tok::kw_template:
      parseTemplateDeclaration();
      break;
    case tok::identifier:
      if (Line.First->is(tok::kw_for) &&
          Tok->Tok.getIdentifierInfo() == &Ident_in)
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
        if (CurrentToken->isNot(tok::comment) || CurrentToken->Next)
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
    if (CurrentToken->Tok.getIdentifierInfo() == NULL)
      return;
    switch (CurrentToken->Tok.getIdentifierInfo()->getPPKeywordID()) {
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
    FormatToken *LastPeriodOrArrow = NULL;
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
      FormatToken *TheToken = CurrentToken;
      if (!consumeToken())
        return LT_Invalid;
      if (TheToken->getPrecedence() > prec::Assignment &&
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

    if (Line.First->Type == TT_ObjCMethodSpecifier) {
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

    if (CurrentToken != NULL)
      CurrentToken = CurrentToken->Next;

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
          ColonIsObjCDictLiteral(false), ColonIsObjCMethodExpr(false),
          FirstObjCSelectorName(NULL), FirstStartOfName(NULL),
          IsExpression(IsExpression), CanBeExpression(true) {}

    tok::TokenKind ContextKind;
    unsigned BindingStrength;
    unsigned LongestObjCSelectorName;
    bool ColonIsForRangeExpr;
    bool ColonIsObjCDictLiteral;
    bool ColonIsObjCMethodExpr;
    FormatToken *FirstObjCSelectorName;
    FormatToken *FirstStartOfName;
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
      P.Contexts.push_back(Context(ContextKind,
                                   P.Contexts.back().BindingStrength + Increase,
                                   P.Contexts.back().IsExpression));
    }

    ~ScopedContextCreator() { P.Contexts.pop_back(); }
  };

  void determineTokenType(FormatToken &Current) {
    if (Current.getPrecedence() == prec::Assignment &&
        (!Current.Previous || Current.Previous->isNot(tok::kw_operator))) {
      Contexts.back().IsExpression = true;
      for (FormatToken *Previous = Current.Previous;
           Previous && Previous->isNot(tok::comma);
           Previous = Previous->Previous) {
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
                (!Current.Previous || Current.Previous->isNot(tok::kw_for)))) {
      Contexts.back().IsExpression = true;
    } else if (Current.isOneOf(tok::r_paren, tok::greater, tok::comma)) {
      for (FormatToken *Previous = Current.Previous;
           Previous && Previous->isOneOf(tok::star, tok::amp);
           Previous = Previous->Previous)
        Previous->Type = TT_PointerOrReference;
    } else if (Current.Previous &&
               Current.Previous->Type == TT_CtorInitializerColon) {
      Contexts.back().IsExpression = true;
    } else if (Current.is(tok::kw_new)) {
      Contexts.back().CanBeExpression = false;
    } else if (Current.is(tok::semi)) {
      // This should be the condition or increment in a for-loop.
      Contexts.back().IsExpression = true;
    }

    if (Current.Type == TT_Unknown) {
      if (isStartOfName(Current)) {
        Contexts.back().FirstStartOfName = &Current;
        Current.Type = TT_StartOfName;
        NameFound = true;
      } else if (Current.is(tok::kw_auto)) {
        AutoFound = true;
      } else if (Current.is(tok::arrow) && AutoFound &&
                 Line.MustBeDeclaration) {
        Current.Type = TT_TrailingReturnArrow;
      } else if (Current.isOneOf(tok::star, tok::amp, tok::ampamp)) {
        Current.Type =
            determineStarAmpUsage(Current, Contexts.back().CanBeExpression &&
                                               Contexts.back().IsExpression);
      } else if (Current.isOneOf(tok::minus, tok::plus, tok::caret)) {
        Current.Type = determinePlusMinusCaretUsage(Current);
      } else if (Current.isOneOf(tok::minusminus, tok::plusplus)) {
        Current.Type = determineIncrementUsage(Current);
      } else if (Current.is(tok::exclaim)) {
        Current.Type = TT_UnaryOperator;
      } else if (Current.isBinaryOperator()) {
        Current.Type = TT_BinaryOperator;
      } else if (Current.is(tok::comment)) {
        if (Current.TokenText.startswith("//"))
          Current.Type = TT_LineComment;
        else
          Current.Type = TT_BlockComment;
      } else if (Current.is(tok::r_paren)) {
        FormatToken *LeftOfParens = NULL;
        if (Current.MatchingParen)
          LeftOfParens = Current.MatchingParen->getPreviousNonComment();
        bool IsCast = false;
        bool ParensAreEmpty = Current.Previous == Current.MatchingParen;
        bool ParensAreType = !Current.Previous ||
                             Current.Previous->Type == TT_PointerOrReference ||
                             Current.Previous->Type == TT_TemplateCloser ||
                             isSimpleTypeSpecifier(*Current.Previous);
        bool ParensCouldEndDecl =
            Current.Next &&
            Current.Next->isOneOf(tok::equal, tok::semi, tok::l_brace);
        bool IsSizeOfOrAlignOf =
            LeftOfParens &&
            LeftOfParens->isOneOf(tok::kw_sizeof, tok::kw_alignof);
        if (ParensAreType && !ParensCouldEndDecl && !IsSizeOfOrAlignOf &&
            (Contexts.back().IsExpression ||
             (Current.Next && Current.Next->isBinaryOperator())))
          IsCast = true;
        if (Current.Next && Current.Next->isNot(tok::string_literal) &&
            (Current.Next->Tok.isLiteral() ||
             Current.Next->isOneOf(tok::kw_sizeof, tok::kw_alignof)))
          IsCast = true;
        // If there is an identifier after the (), it is likely a cast, unless
        // there is also an identifier before the ().
        if (LeftOfParens && (LeftOfParens->Tok.getIdentifierInfo() == NULL ||
                             LeftOfParens->is(tok::kw_return)) &&
            LeftOfParens->Type != TT_OverloadedOperator &&
            LeftOfParens->Type != TT_TemplateCloser && Current.Next &&
            Current.Next->is(tok::identifier))
          IsCast = true;
        if (IsCast && !ParensAreEmpty)
          Current.Type = TT_CastRParen;
      } else if (Current.is(tok::at) && Current.Next) {
        switch (Current.Next->Tok.getObjCKeywordID()) {
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
      } else if (Current.is(tok::period)) {
        FormatToken *PreviousNoComment = Current.getPreviousNonComment();
        if (PreviousNoComment &&
            PreviousNoComment->isOneOf(tok::comma, tok::l_brace))
          Current.Type = TT_DesignatedInitializerPeriod;
      }
    }
  }

  /// \brief Take a guess at whether \p Tok starts a name of a function or
  /// variable declaration.
  ///
  /// This is a heuristic based on whether \p Tok is an identifier following
  /// something that is likely a type.
  bool isStartOfName(const FormatToken &Tok) {
    if (Tok.isNot(tok::identifier) || Tok.Previous == NULL)
      return false;

    // Skip "const" as it does not have an influence on whether this is a name.
    FormatToken *PreviousNotConst = Tok.Previous;
    while (PreviousNotConst != NULL && PreviousNotConst->is(tok::kw_const))
      PreviousNotConst = PreviousNotConst->Previous;

    if (PreviousNotConst == NULL)
      return false;

    bool IsPPKeyword = PreviousNotConst->is(tok::identifier) &&
                       PreviousNotConst->Previous &&
                       PreviousNotConst->Previous->is(tok::hash);

    return (!IsPPKeyword && PreviousNotConst->is(tok::identifier)) ||
           PreviousNotConst->Type == TT_PointerOrReference ||
           PreviousNotConst->Type == TT_TemplateCloser ||
           isSimpleTypeSpecifier(*PreviousNotConst);
  }

  /// \brief Return the type of the given token assuming it is * or &.
  TokenType determineStarAmpUsage(const FormatToken &Tok, bool IsExpression) {
    const FormatToken *PrevToken = Tok.getPreviousNonComment();
    if (PrevToken == NULL)
      return TT_UnaryOperator;

    const FormatToken *NextToken = Tok.getNextNonComment();
    if (NextToken == NULL)
      return TT_Unknown;

    if (PrevToken->is(tok::coloncolon) ||
        (PrevToken->is(tok::l_paren) && !IsExpression))
      return TT_PointerOrReference;

    if (PrevToken->isOneOf(tok::l_paren, tok::l_square, tok::l_brace,
                           tok::comma, tok::semi, tok::kw_return, tok::colon,
                           tok::equal, tok::kw_delete) ||
        PrevToken->Type == TT_BinaryOperator ||
        PrevToken->Type == TT_UnaryOperator || PrevToken->Type == TT_CastRParen)
      return TT_UnaryOperator;

    if (NextToken->is(tok::l_square))
      return TT_PointerOrReference;

    if (PrevToken->Tok.isLiteral() ||
        PrevToken->isOneOf(tok::r_paren, tok::r_square) ||
        NextToken->Tok.isLiteral() || NextToken->isUnaryOperator())
      return TT_BinaryOperator;

    // It is very unlikely that we are going to find a pointer or reference type
    // definition on the RHS of an assignment.
    if (IsExpression)
      return TT_BinaryOperator;

    return TT_PointerOrReference;
  }

  TokenType determinePlusMinusCaretUsage(const FormatToken &Tok) {
    const FormatToken *PrevToken = Tok.getPreviousNonComment();
    if (PrevToken == NULL || PrevToken->Type == TT_CastRParen)
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
  TokenType determineIncrementUsage(const FormatToken &Tok) {
    const FormatToken *PrevToken = Tok.getPreviousNonComment();
    if (PrevToken == NULL || PrevToken->Type == TT_CastRParen)
      return TT_UnaryOperator;
    if (PrevToken->isOneOf(tok::r_paren, tok::r_square, tok::identifier))
      return TT_TrailingUnaryOperator;

    return TT_UnaryOperator;
  }

  // FIXME: This is copy&pasted from Sema. Put it in a common place and remove
  // duplication.
  /// \brief Determine whether the token kind starts a simple-type-specifier.
  bool isSimpleTypeSpecifier(const FormatToken &Tok) const {
    switch (Tok.Tok.getKind()) {
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
    case tok::annot_typename:
    case tok::kw_char16_t:
    case tok::kw_char32_t:
    case tok::kw_typeof:
    case tok::kw_decltype:
      return true;
    default:
      return false;
    }
  }

  SmallVector<Context, 8> Contexts;

  AnnotatedLine &Line;
  FormatToken *CurrentToken;
  bool KeywordVirtualFound;
  bool NameFound;
  bool AutoFound;
  IdentifierInfo &Ident_in;
};

/// \brief Parses binary expressions by inserting fake parenthesis based on
/// operator precedence.
class ExpressionParser {
public:
  ExpressionParser(AnnotatedLine &Line) : Current(Line.First) {
    // Skip leading "}", e.g. in "} else if (...) {".
    if (Current->is(tok::r_brace))
      next();
  }

  /// \brief Parse expressions with the given operatore precedence.
  void parse(int Precedence = 0) {
    // Conditional expressions need to be parsed separately for proper nesting.
    if (Precedence == prec::Conditional + 1) {
      parseConditionalExpr();
      return;
    }
    if (Precedence > prec::PointerToMember || Current == NULL)
      return;

    FormatToken *Start = Current;
    bool OperatorFound = false;

    while (Current) {
      // Consume operators with higher precedence.
      parse(Precedence + 1);

      int CurrentPrecedence = 0;
      if (Current) {
        if (Current->Type == TT_ConditionalExpr)
          CurrentPrecedence = 1 + (int)prec::Conditional;
        else if (Current->is(tok::semi) || Current->Type == TT_InlineASMColon)
          CurrentPrecedence = 1;
        else if (Current->Type == TT_BinaryOperator || Current->is(tok::comma))
          CurrentPrecedence = 1 + (int)Current->getPrecedence();
      }

      // At the end of the line or when an operator with higher precedence is
      // found, insert fake parenthesis and return.
      if (Current == NULL || Current->closesScope() ||
          (CurrentPrecedence != 0 && CurrentPrecedence < Precedence)) {
        if (OperatorFound)
          addFakeParenthesis(Start, prec::Level(Precedence - 1));
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
  void addFakeParenthesis(FormatToken *Start, prec::Level Precedence) {
    Start->FakeLParens.push_back(Precedence);
    if (Current)
      ++Current->Previous->FakeRParens;
  }

  void parseConditionalExpr() {
    FormatToken *Start = Current;
    parse(prec::LogicalOr + 1);
    if (!Current || !Current->is(tok::question))
      return;
    next();
    parse(prec::LogicalOr + 1);
    if (!Current || Current->Type != TT_ConditionalExpr)
      return;
    next();
    parseConditionalExpr();
    addFakeParenthesis(Start, prec::Conditional);
  }

  void next() {
    if (Current)
      Current = Current->Next;
    while (Current && Current->isTrailingComment())
      Current = Current->Next;
  }

  FormatToken *Current;
};

} // end anonymous namespace

void TokenAnnotator::annotate(AnnotatedLine &Line) {
  AnnotatingParser Parser(Line, Ident_in);
  Line.Type = Parser.parseLine();
  if (Line.Type == LT_Invalid)
    return;

  ExpressionParser ExprParser(Line);
  ExprParser.parse();

  if (Line.First->Type == TT_ObjCMethodSpecifier)
    Line.Type = LT_ObjCMethodDecl;
  else if (Line.First->Type == TT_ObjCDecl)
    Line.Type = LT_ObjCDecl;
  else if (Line.First->Type == TT_ObjCProperty)
    Line.Type = LT_ObjCProperty;

  Line.First->SpacesRequiredBefore = 1;
  Line.First->CanBreakBefore = Line.First->MustBreakBefore;
}

void TokenAnnotator::calculateFormattingInformation(AnnotatedLine &Line) {
  Line.First->TotalLength = Line.First->CodePointCount;
  if (!Line.First->Next)
    return;
  FormatToken *Current = Line.First->Next;
  while (Current != NULL) {
    if (Current->Type == TT_LineComment)
      Current->SpacesRequiredBefore = Style.SpacesBeforeTrailingComments;
    else
      Current->SpacesRequiredBefore =
          spaceRequiredBefore(Line, *Current) ? 1 : 0;

    if (Current->MustBreakBefore) {
    } else if (Current->is(tok::comment)) {
      Current->MustBreakBefore = Current->NewlinesBefore > 0;
    } else if (Current->Previous->isTrailingComment() ||
               (Current->is(tok::string_literal) &&
                Current->Previous->is(tok::string_literal))) {
      Current->MustBreakBefore = true;
    } else if (Current->is(tok::lessless) && Current->Next &&
               Current->Previous->is(tok::string_literal) &&
               Current->Next->is(tok::string_literal)) {
      Current->MustBreakBefore = true;
    } else if (Current->Previous->ClosesTemplateDeclaration &&
               Style.AlwaysBreakTemplateDeclarations) {
      Current->MustBreakBefore = true;
    } else if (Style.AlwaysBreakBeforeMultilineStrings &&
               Current->is(tok::string_literal) &&
               Current->Previous->isNot(tok::lessless) &&
               Current->Previous->Type != TT_InlineASMColon &&
               Current->getNextNonComment() &&
               Current->getNextNonComment()->is(tok::string_literal)) {
      Current->MustBreakBefore = true;
    } else {
      Current->MustBreakBefore = false;
    }
    Current->CanBreakBefore =
        Current->MustBreakBefore || canBreakBefore(Line, *Current);
    if (Current->MustBreakBefore)
      Current->TotalLength = Current->Previous->TotalLength + Style.ColumnLimit;
    else
      Current->TotalLength = Current->Previous->TotalLength +
                             Current->CodePointCount +
                             Current->SpacesRequiredBefore;
    // FIXME: Only calculate this if CanBreakBefore is true once static
    // initializers etc. are sorted out.
    // FIXME: Move magic numbers to a better place.
    Current->SplitPenalty =
        20 * Current->BindingStrength + splitPenalty(Line, *Current);

    Current = Current->Next;
  }

  calculateUnbreakableTailLengths(Line);
  DEBUG({
    printDebugInfo(Line);
  });
}

void TokenAnnotator::calculateUnbreakableTailLengths(AnnotatedLine &Line) {
  unsigned UnbreakableTailLength = 0;
  FormatToken *Current = Line.Last;
  while (Current != NULL) {
    Current->UnbreakableTailLength = UnbreakableTailLength;
    if (Current->CanBreakBefore ||
        Current->isOneOf(tok::comment, tok::string_literal)) {
      UnbreakableTailLength = 0;
    } else {
      UnbreakableTailLength +=
          Current->CodePointCount + Current->SpacesRequiredBefore;
    }
    Current = Current->Previous;
  }
}

unsigned TokenAnnotator::splitPenalty(const AnnotatedLine &Line,
                                      const FormatToken &Tok) {
  const FormatToken &Left = *Tok.Previous;
  const FormatToken &Right = Tok;

  if (Left.is(tok::semi))
    return 0;
  if (Left.is(tok::comma))
    return 1;
  if (Right.is(tok::l_square))
    return 150;

  if (Right.Type == TT_StartOfName || Right.is(tok::kw_operator)) {
    if (Line.First->is(tok::kw_for) && Right.PartOfMultiVariableDeclStmt)
      return 3;
    if (Left.Type == TT_StartOfName)
      return 20;
    else if (Line.MightBeFunctionDecl && Right.BindingStrength == 1)
      // FIXME: Clean up hack of using BindingStrength to find top-level names.
      return Style.PenaltyReturnTypeOnItsOwnLine;
    else
      return 200;
  }
  if (Left.is(tok::equal) && Right.is(tok::l_brace))
    return 150;
  if (Left.Type == TT_CastRParen)
    return 100;
  if (Left.is(tok::coloncolon))
    return 500;
  if (Left.isOneOf(tok::kw_class, tok::kw_struct))
    return 5000;

  if (Left.Type == TT_RangeBasedForLoopColon ||
      Left.Type == TT_InheritanceColon)
    return 2;

  if (Right.isOneOf(tok::arrow, tok::period) &&
      Right.Type != TT_DesignatedInitializerPeriod) {
    if (Line.Type == LT_BuilderTypeCall)
      return prec::PointerToMember;
    if (Left.isOneOf(tok::r_paren, tok::r_square) && Left.MatchingParen &&
        Left.MatchingParen->ParameterCount > 0)
      return 20; // Should be smaller than breaking at a nested comma.
    return 150;
  }

  // Breaking before a trailing 'const' or not-function-like annotation is bad.
  if (Left.is(tok::r_paren) &&
      (Right.is(tok::kw_const) || (Right.is(tok::identifier) && Right.Next &&
                                   Right.Next->isNot(tok::l_paren))))
    return 150;

  // In for-loops, prefer breaking at ',' and ';'.
  if (Line.First->is(tok::kw_for) && Left.is(tok::equal))
    return 4;

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
      StringRef Content = Left.TokenText;
      Content = Content.drop_back(1).drop_front(1).trim();
      if (Content.size() > 1 &&
          (Content.back() == ':' || Content.back() == '='))
        return 25;
    }
    return 1; // Breaking at a << is really cheap.
  }
  if (Left.Type == TT_ConditionalExpr)
    return prec::Conditional;
  prec::Level Level = Left.getPrecedence();

  if (Level != prec::Unknown)
    return Level;

  return 3;
}

bool TokenAnnotator::spaceRequiredBetween(const AnnotatedLine &Line,
                                          const FormatToken &Left,
                                          const FormatToken &Right) {
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
  if (Right.is(tok::ellipsis))
    return false;
  if (Right.Type == TT_PointerOrReference)
    return Left.Tok.isLiteral() ||
           ((Left.Type != TT_PointerOrReference) && Left.isNot(tok::l_paren) &&
            !Style.PointerBindsToType);
  if (Right.Type == TT_FunctionTypeLParen && Left.isNot(tok::l_paren) &&
      (Left.Type != TT_PointerOrReference || Style.PointerBindsToType))
    return true;
  if (Left.Type == TT_PointerOrReference)
    return Right.Tok.isLiteral() || Right.Type == TT_BlockComment ||
           ((Right.Type != TT_PointerOrReference) &&
            Right.isNot(tok::l_paren) && Style.PointerBindsToType &&
            Left.Previous &&
            !Left.Previous->isOneOf(tok::l_paren, tok::coloncolon));
  if (Right.is(tok::star) && Left.is(tok::l_paren))
    return false;
  if (Left.is(tok::l_square))
    return Left.Type == TT_ObjCArrayLiteral && Right.isNot(tok::r_square);
  if (Right.is(tok::r_square))
    return Right.Type == TT_ObjCArrayLiteral;
  if (Right.is(tok::l_square) && Right.Type != TT_ObjCMethodExpr)
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
  if (Left.is(tok::at) && Right.Tok.getObjCKeywordID() != tok::objc_not_keyword)
    return false;
  if (Left.is(tok::l_brace) && Right.is(tok::r_brace))
    return false; // No spaces in "{}".
  if (Left.is(tok::l_brace) || Right.is(tok::r_brace))
    return Style.SpacesInBracedLists;
  if (Right.Type == TT_UnaryOperator)
    return !Left.isOneOf(tok::l_paren, tok::l_square, tok::at) &&
           (Left.isNot(tok::colon) || Left.Type != TT_ObjCMethodExpr);
  if (Left.isOneOf(tok::identifier, tok::greater, tok::r_square) &&
      Right.is(tok::l_brace) && Right.getNextNonComment())
    return false;
  if (Left.is(tok::period) || Right.is(tok::period))
    return false;
  if (Left.Type == TT_BlockComment && Left.TokenText.endswith("=*/"))
    return false;
  return true;
}

bool TokenAnnotator::spaceRequiredBefore(const AnnotatedLine &Line,
                                         const FormatToken &Tok) {
  if (Tok.Tok.getIdentifierInfo() && Tok.Previous->Tok.getIdentifierInfo())
    return true; // Never ever merge two identifiers.
  if (Line.Type == LT_ObjCMethodDecl) {
    if (Tok.Previous->Type == TT_ObjCMethodSpecifier)
      return true;
    if (Tok.Previous->is(tok::r_paren) && Tok.is(tok::identifier))
      // Don't space between ')' and <id>
      return false;
  }
  if (Line.Type == LT_ObjCProperty &&
      (Tok.is(tok::equal) || Tok.Previous->is(tok::equal)))
    return false;

  if (Tok.Type == TT_TrailingReturnArrow ||
      Tok.Previous->Type == TT_TrailingReturnArrow)
    return true;
  if (Tok.Previous->is(tok::comma))
    return true;
  if (Tok.is(tok::comma))
    return false;
  if (Tok.Type == TT_CtorInitializerColon || Tok.Type == TT_ObjCBlockLParen)
    return true;
  if (Tok.Previous->Tok.is(tok::kw_operator))
    return false;
  if (Tok.Type == TT_OverloadedOperatorLParen)
    return false;
  if (Tok.is(tok::colon))
    return !Line.First->isOneOf(tok::kw_case, tok::kw_default) &&
           Tok.getNextNonComment() != NULL && Tok.Type != TT_ObjCMethodExpr;
  if (Tok.Previous->Type == TT_UnaryOperator ||
      Tok.Previous->Type == TT_CastRParen)
    return false;
  if (Tok.Previous->is(tok::greater) && Tok.is(tok::greater)) {
    return Tok.Type == TT_TemplateCloser &&
           Tok.Previous->Type == TT_TemplateCloser &&
           Style.Standard != FormatStyle::LS_Cpp11;
  }
  if (Tok.isOneOf(tok::arrowstar, tok::periodstar) ||
      Tok.Previous->isOneOf(tok::arrowstar, tok::periodstar))
    return false;
  if (Tok.Type == TT_BinaryOperator || Tok.Previous->Type == TT_BinaryOperator)
    return true;
  if (Tok.Previous->Type == TT_TemplateCloser && Tok.is(tok::l_paren))
    return false;
  if (Tok.is(tok::less) && Line.First->is(tok::hash))
    return true;
  if (Tok.Type == TT_TrailingUnaryOperator)
    return false;
  return spaceRequiredBetween(Line, *Tok.Previous, Tok);
}

bool TokenAnnotator::canBreakBefore(const AnnotatedLine &Line,
                                    const FormatToken &Right) {
  const FormatToken &Left = *Right.Previous;
  if (Right.Type == TT_StartOfName || Right.is(tok::kw_operator))
    return true;
  if (Right.is(tok::colon) &&
      (Right.Type == TT_ObjCDictLiteral || Right.Type == TT_ObjCMethodExpr))
    return false;
  if (Left.is(tok::colon) &&
      (Left.Type == TT_ObjCDictLiteral || Left.Type == TT_ObjCMethodExpr))
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
  if (Left.Previous) {
    if (Left.is(tok::l_paren) && Right.is(tok::l_paren) &&
        Left.Previous->is(tok::kw___attribute))
      return false;
    if (Left.is(tok::l_paren) && (Left.Previous->Type == TT_BinaryOperator ||
                                  Left.Previous->is(tok::r_paren)))
      return false;
  }

  if (Right.isTrailingComment())
    // We rely on MustBreakBefore being set correctly here as we should not
    // change the "binding" behavior of a comment.
    return false;

  // We only break before r_brace if there was a corresponding break before
  // the l_brace, which is tracked by BreakBeforeClosingBrace.
  if (Right.isOneOf(tok::r_brace, tok::r_paren, tok::greater))
    return false;

  // Allow breaking after a trailing 'const', e.g. after a method declaration,
  // unless it is follow by ';', '{' or '='.
  if (Left.is(tok::kw_const) && Left.Previous != NULL &&
      Left.Previous->is(tok::r_paren))
    return !Right.isOneOf(tok::l_brace, tok::semi, tok::equal);

  if (Right.is(tok::kw___attribute))
    return true;

  if (Left.is(tok::identifier) && Right.is(tok::string_literal))
    return true;
  return (Left.isBinaryOperator() && Left.isNot(tok::lessless)) ||
         Left.isOneOf(tok::comma, tok::coloncolon, tok::semi, tok::l_brace,
                      tok::kw_class, tok::kw_struct) ||
         Right.isOneOf(tok::lessless, tok::arrow, tok::period, tok::colon) ||
         (Left.is(tok::r_paren) &&
          Right.isOneOf(tok::identifier, tok::kw_const, tok::kw___attribute)) ||
         (Left.is(tok::l_paren) && !Right.is(tok::r_paren)) ||
         Right.is(tok::l_square);
}

void TokenAnnotator::printDebugInfo(const AnnotatedLine &Line) {
  llvm::errs() << "AnnotatedTokens:\n";
  const FormatToken *Tok = Line.First;
  while (Tok) {
    llvm::errs() << " M=" << Tok->MustBreakBefore
                 << " C=" << Tok->CanBreakBefore << " T=" << Tok->Type
                 << " S=" << Tok->SpacesRequiredBefore
                 << " P=" << Tok->SplitPenalty << " Name=" << Tok->Tok.getName()
                 << " PPK=" << Tok->PackingKind << " FakeLParens=";
    for (unsigned i = 0, e = Tok->FakeLParens.size(); i != e; ++i)
      llvm::errs() << Tok->FakeLParens[i] << "/";
    llvm::errs() << " FakeRParens=" << Tok->FakeRParens << "\n";
    Tok = Tok->Next;
  }
  llvm::errs() << "----\n";
}

} // namespace format
} // namespace clang
