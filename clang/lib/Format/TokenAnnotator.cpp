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

#define DEBUG_TYPE "format-token-annotator"

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
  AnnotatingParser(const FormatStyle &Style, AnnotatedLine &Line,
                   IdentifierInfo &Ident_in)
      : Style(Style), Line(Line), CurrentToken(Line.First),
        KeywordVirtualFound(false), AutoFound(false), Ident_in(Ident_in) {
    Contexts.push_back(Context(tok::unknown, 1, /*IsExpression=*/false));
    resetTokenMetadata(CurrentToken);
  }

private:
  bool parseAngle() {
    if (!CurrentToken)
      return false;
    ScopedContextCreator ContextCreator(*this, tok::less, 10);
    FormatToken *Left = CurrentToken->Previous;
    Contexts.back().IsExpression = false;
    // If there's a template keyword before the opening angle bracket, this is a
    // template parameter, not an argument.
    Contexts.back().InTemplateArgument =
        Left->Previous && Left->Previous->Tok.isNot(tok::kw_template);

    while (CurrentToken) {
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
          CurrentToken->Previous->Type == TT_BinaryOperator &&
          Contexts[Contexts.size() - 2].IsExpression &&
          Line.First->isNot(tok::kw_template))
        return false;
      updateParameterCount(Left, CurrentToken);
      if (!consumeToken())
        return false;
    }
    return false;
  }

  bool parseParens(bool LookForDecls = false) {
    if (!CurrentToken)
      return false;
    ScopedContextCreator ContextCreator(*this, tok::l_paren, 1);

    // FIXME: This is a bit of a hack. Do better.
    Contexts.back().ColonIsForRangeExpr =
        Contexts.size() == 2 && Contexts[0].ColonIsForRangeExpr;

    bool StartsObjCMethodExpr = false;
    FormatToken *Left = CurrentToken->Previous;
    if (CurrentToken->is(tok::caret)) {
      // (^ can start a block type.
      Left->Type = TT_ObjCBlockLParen;
    } else if (FormatToken *MaybeSel = Left->Previous) {
      // @selector( starts a selector.
      if (MaybeSel->isObjCAtKeyword(tok::objc_selector) && MaybeSel->Previous &&
          MaybeSel->Previous->is(tok::at)) {
        StartsObjCMethodExpr = true;
      }
    }

    if (Left->Previous &&
        (Left->Previous->isOneOf(tok::kw_static_assert, tok::kw_if,
                                 tok::kw_while, tok::l_paren, tok::comma) ||
         Left->Previous->Type == TT_BinaryOperator)) {
      // static_assert, if and while usually contain expressions.
      Contexts.back().IsExpression = true;
    } else if (Line.InPPDirective &&
               (!Left->Previous ||
                (Left->Previous->isNot(tok::identifier) &&
                 Left->Previous->Type != TT_OverloadedOperator))) {
      Contexts.back().IsExpression = true;
    } else if (Left->Previous && Left->Previous->is(tok::r_square) &&
               Left->Previous->MatchingParen &&
               Left->Previous->MatchingParen->Type == TT_LambdaLSquare) {
      // This is a parameter list of a lambda expression.
      Contexts.back().IsExpression = false;
    } else if (Contexts[Contexts.size() - 2].CaretFound) {
      // This is the parameter list of an ObjC block.
      Contexts.back().IsExpression = false;
    } else if (Left->Previous && Left->Previous->is(tok::kw___attribute)) {
      Left->Type = TT_AttributeParen;
    } else if (Left->Previous && Left->Previous->IsForEachMacro) {
      // The first argument to a foreach macro is a declaration.
      Contexts.back().IsForEachMacro = true;
      Contexts.back().IsExpression = false;
    }

    if (StartsObjCMethodExpr) {
      Contexts.back().ColonIsObjCMethodExpr = true;
      Left->Type = TT_ObjCMethodExpr;
    }

    bool MightBeFunctionType = CurrentToken->is(tok::star);
    bool HasMultipleLines = false;
    bool HasMultipleParametersOnALine = false;
    while (CurrentToken) {
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

      if (CurrentToken->Previous->Type == TT_PointerOrReference &&
          CurrentToken->Previous->Previous->isOneOf(tok::l_paren,
                                                    tok::coloncolon))
        MightBeFunctionType = true;
      if (CurrentToken->Previous->Type == TT_BinaryOperator)
        Contexts.back().IsExpression = true;
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
          if (Contexts.back().FirstObjCSelectorName) {
            Contexts.back().FirstObjCSelectorName->LongestObjCSelectorName =
                Contexts.back().LongestObjCSelectorName;
          }
        }

        if (Left->Type == TT_AttributeParen)
          CurrentToken->Type = TT_AttributeParen;

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
      else if (CurrentToken->is(tok::l_brace))
        Left->Type = TT_Unknown; // Not TT_ObjCBlockLParen
      if (CurrentToken->is(tok::comma) && CurrentToken->Next &&
          !CurrentToken->Next->HasUnescapedNewline &&
          !CurrentToken->Next->isTrailingComment())
        HasMultipleParametersOnALine = true;
      if (CurrentToken->isOneOf(tok::kw_const, tok::kw_auto) ||
          CurrentToken->isSimpleTypeSpecifier())
        Contexts.back().IsExpression = false;
      FormatToken *Tok = CurrentToken;
      if (!consumeToken())
        return false;
      updateParameterCount(Left, Tok);
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
        Contexts.back().CanBeExpression && Left->Type != TT_LambdaLSquare &&
        CurrentToken->isNot(tok::l_brace) &&
        (!Parent || Parent->isOneOf(tok::colon, tok::l_square, tok::l_paren,
                                    tok::kw_return, tok::kw_throw) ||
         Parent->isUnaryOperator() || Parent->Type == TT_ObjCForIn ||
         Parent->Type == TT_CastRParen ||
         getBinOpPrecedence(Parent->Tok.getKind(), true, true) > prec::Unknown);
    ScopedContextCreator ContextCreator(*this, tok::l_square, 10);
    Contexts.back().IsExpression = true;
    bool ColonFound = false;

    if (StartsObjCMethodExpr) {
      Contexts.back().ColonIsObjCMethodExpr = true;
      Left->Type = TT_ObjCMethodExpr;
    } else if (Parent && Parent->is(tok::at)) {
      Left->Type = TT_ArrayInitializerLSquare;
    } else if (Left->Type == TT_Unknown) {
      Left->Type = TT_ArraySubscriptLSquare;
    }

    while (CurrentToken) {
      if (CurrentToken->is(tok::r_square)) {
        if (CurrentToken->Next && CurrentToken->Next->is(tok::l_paren) &&
            Left->Type == TT_ObjCMethodExpr) {
          // An ObjC method call is rarely followed by an open parenthesis.
          // FIXME: Do we incorrectly label ":" with this?
          StartsObjCMethodExpr = false;
          Left->Type = TT_Unknown;
        }
        if (StartsObjCMethodExpr && CurrentToken->Previous != Left) {
          CurrentToken->Type = TT_ObjCMethodExpr;
          // determineStarAmpUsage() thinks that '*' '[' is allocating an
          // array of pointers, but if '[' starts a selector then '*' is a
          // binary operator.
          if (Parent && Parent->Type == TT_PointerOrReference)
            Parent->Type = TT_BinaryOperator;
        }
        Left->MatchingParen = CurrentToken;
        CurrentToken->MatchingParen = Left;
        if (Contexts.back().FirstObjCSelectorName) {
          Contexts.back().FirstObjCSelectorName->LongestObjCSelectorName =
              Contexts.back().LongestObjCSelectorName;
          if (Left->BlockParameterCount > 1)
            Contexts.back().FirstObjCSelectorName->LongestObjCSelectorName = 0;
        }
        next();
        return true;
      }
      if (CurrentToken->isOneOf(tok::r_paren, tok::r_brace))
        return false;
      if (CurrentToken->is(tok::colon))
        ColonFound = true;
      if (CurrentToken->is(tok::comma) &&
          Style.Language != FormatStyle::LK_Proto &&
          (Left->Type == TT_ArraySubscriptLSquare ||
           (Left->Type == TT_ObjCMethodExpr && !ColonFound)))
        Left->Type = TT_ArrayInitializerLSquare;
      FormatToken* Tok = CurrentToken;
      if (!consumeToken())
        return false;
      updateParameterCount(Left, Tok);
    }
    return false;
  }

  bool parseBrace() {
    if (CurrentToken) {
      FormatToken *Left = CurrentToken->Previous;

      if (Contexts.back().CaretFound)
        Left->Type = TT_ObjCBlockLBrace;
      Contexts.back().CaretFound = false;

      ScopedContextCreator ContextCreator(*this, tok::l_brace, 1);
      Contexts.back().ColonIsDictLiteral = true;
      if (Left->BlockKind == BK_BracedInit)
        Contexts.back().IsExpression = true;

      while (CurrentToken) {
        if (CurrentToken->is(tok::r_brace)) {
          Left->MatchingParen = CurrentToken;
          CurrentToken->MatchingParen = Left;
          next();
          return true;
        }
        if (CurrentToken->isOneOf(tok::r_paren, tok::r_square))
          return false;
        updateParameterCount(Left, CurrentToken);
        if (CurrentToken->is(tok::colon)) {
          if (CurrentToken->getPreviousNonComment()->is(tok::identifier))
            CurrentToken->getPreviousNonComment()->Type = TT_SelectorName;
          Left->Type = TT_DictLiteral;
        }
        if (!consumeToken())
          return false;
      }
    }
    return true;
  }

  void updateParameterCount(FormatToken *Left, FormatToken *Current) {
    if (Current->Type == TT_LambdaLSquare ||
        (Current->is(tok::caret) && Current->Type == TT_UnaryOperator) ||
        (Style.Language == FormatStyle::LK_JavaScript &&
         Current->TokenText == "function")) {
      ++Left->BlockParameterCount;
    }
    if (Current->is(tok::comma)) {
      ++Left->ParameterCount;
      if (!Left->Role)
        Left->Role.reset(new CommaSeparatedList(Style));
      Left->Role->CommaFound(Current);
    } else if (Left->ParameterCount == 0 && Current->isNot(tok::comment)) {
      Left->ParameterCount = 1;
    }
  }

  bool parseConditional() {
    while (CurrentToken) {
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
    if (CurrentToken && CurrentToken->is(tok::less)) {
      CurrentToken->Type = TT_TemplateOpener;
      next();
      if (!parseAngle())
        return false;
      if (CurrentToken)
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
      if (!Tok->Previous && Line.MustBeDeclaration)
        Tok->Type = TT_ObjCMethodSpecifier;
      break;
    case tok::colon:
      if (!Tok->Previous)
        return false;
      // Colons from ?: are handled in parseConditional().
      if (Tok->Previous->is(tok::r_paren) && Contexts.size() == 1 &&
          Line.First->isNot(tok::kw_case)) {
        Tok->Type = TT_CtorInitializerColon;
      } else if (Contexts.back().ColonIsDictLiteral) {
        Tok->Type = TT_DictLiteral;
      } else if (Contexts.back().ColonIsObjCMethodExpr ||
                 Line.First->Type == TT_ObjCMethodSpecifier) {
        Tok->Type = TT_ObjCMethodExpr;
        Tok->Previous->Type = TT_SelectorName;
        if (Tok->Previous->ColumnWidth >
            Contexts.back().LongestObjCSelectorName) {
          Contexts.back().LongestObjCSelectorName = Tok->Previous->ColumnWidth;
        }
        if (!Contexts.back().FirstObjCSelectorName)
          Contexts.back().FirstObjCSelectorName = Tok->Previous;
      } else if (Contexts.back().ColonIsForRangeExpr) {
        Tok->Type = TT_RangeBasedForLoopColon;
      } else if (CurrentToken && CurrentToken->is(tok::numeric_constant)) {
        Tok->Type = TT_BitFieldColon;
      } else if (Contexts.size() == 1 &&
                 !Line.First->isOneOf(tok::kw_enum, tok::kw_case)) {
        Tok->Type = TT_InheritanceColon;
      } else if (Contexts.back().ContextKind == tok::l_paren) {
        Tok->Type = TT_InlineASMColon;
      }
      break;
    case tok::kw_if:
    case tok::kw_while:
      if (CurrentToken && CurrentToken->is(tok::l_paren)) {
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
      if (Line.MustBeDeclaration && Contexts.size() == 1 &&
          !Contexts.back().IsExpression &&
          Line.First->Type != TT_ObjCProperty &&
          (!Tok->Previous || Tok->Previous->isNot(tok::kw_decltype)))
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
      if (Tok->Previous && !Tok->Previous->Tok.isLiteral() && parseAngle())
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
      if (Tok->Previous)
        return false;
      break;
    case tok::greater:
      Tok->Type = TT_BinaryOperator;
      break;
    case tok::kw_operator:
      while (CurrentToken &&
             !CurrentToken->isOneOf(tok::l_paren, tok::semi, tok::r_paren)) {
        if (CurrentToken->isOneOf(tok::star, tok::amp))
          CurrentToken->Type = TT_PointerOrReference;
        consumeToken();
        if (CurrentToken && CurrentToken->Previous->Type == TT_BinaryOperator)
          CurrentToken->Previous->Type = TT_OverloadedOperator;
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
      if (Contexts.back().InCtorInitializer)
        Tok->Type = TT_CtorInitializerComma;
      if (Contexts.back().IsForEachMacro)
        Contexts.back().IsExpression = true;
      break;
    default:
      break;
    }
    return true;
  }

  void parseIncludeDirective() {
    next();
    if (CurrentToken && CurrentToken->is(tok::less)) {
      next();
      while (CurrentToken) {
        if (CurrentToken->isNot(tok::comment) || CurrentToken->Next)
          CurrentToken->Type = TT_ImplicitStringLiteral;
        next();
      }
    } else {
      while (CurrentToken) {
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
    while (CurrentToken) {
      CurrentToken->Type = TT_ImplicitStringLiteral;
      next();
    }
  }

  void parsePragma() {
    next(); // Consume "pragma".
    if (CurrentToken && CurrentToken->TokenText == "mark") {
      next(); // Consume "mark".
      next(); // Consume first token (so we fix leading whitespace).
      while (CurrentToken) {
        CurrentToken->Type = TT_ImplicitStringLiteral;
        next();
      }
    }
  }

  void parsePreprocessorDirective() {
    next();
    if (!CurrentToken)
      return;
    if (CurrentToken->Tok.is(tok::numeric_constant)) {
      CurrentToken->SpacesRequiredBefore = 1;
      return;
    }
    // Hashes in the middle of a line can lead to any strange token
    // sequence.
    if (!CurrentToken->Tok.getIdentifierInfo())
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
    case tok::pp_pragma:
      parsePragma();
      break;
    case tok::pp_if:
    case tok::pp_elif:
      Contexts.back().IsExpression = true;
      parseLine();
      break;
    default:
      break;
    }
    while (CurrentToken)
      next();
  }

public:
  LineType parseLine() {
    if (CurrentToken->is(tok::hash)) {
      parsePreprocessorDirective();
      return LT_PreprocessorDirective;
    }

    // Directly allow to 'import <string-literal>' to support protocol buffer
    // definitions (code.google.com/p/protobuf) or missing "#" (either way we
    // should not break the line).
    IdentifierInfo *Info = CurrentToken->Tok.getIdentifierInfo();
    if (Info && Info->getPPKeywordID() == tok::pp_import &&
        CurrentToken->Next && CurrentToken->Next->is(tok::string_literal))
      parseIncludeDirective();

    while (CurrentToken) {
      if (CurrentToken->is(tok::kw_virtual))
        KeywordVirtualFound = true;
      if (!consumeToken())
        return LT_Invalid;
    }
    if (KeywordVirtualFound)
      return LT_VirtualFunctionDecl;

    if (Line.First->Type == TT_ObjCMethodSpecifier) {
      if (Contexts.back().FirstObjCSelectorName)
        Contexts.back().FirstObjCSelectorName->LongestObjCSelectorName =
            Contexts.back().LongestObjCSelectorName;
      return LT_ObjCMethodDecl;
    }

    return LT_Other;
  }

private:
  void resetTokenMetadata(FormatToken *Token) {
    if (!Token)
      return;

    // Reset token type in case we have already looked at it and then
    // recovered from an error (e.g. failure to find the matching >).
    if (CurrentToken->Type != TT_LambdaLSquare &&
        CurrentToken->Type != TT_FunctionLBrace &&
        CurrentToken->Type != TT_ImplicitStringLiteral &&
        CurrentToken->Type != TT_RegexLiteral &&
        CurrentToken->Type != TT_TrailingReturnArrow)
      CurrentToken->Type = TT_Unknown;
    CurrentToken->Role.reset();
    CurrentToken->FakeLParens.clear();
    CurrentToken->FakeRParens = 0;
  }

  void next() {
    if (CurrentToken) {
      determineTokenType(*CurrentToken);
      CurrentToken->BindingStrength = Contexts.back().BindingStrength;
      CurrentToken->NestingLevel = Contexts.size() - 1;
      CurrentToken = CurrentToken->Next;
    }

    resetTokenMetadata(CurrentToken);
  }

  /// \brief A struct to hold information valid in a specific context, e.g.
  /// a pair of parenthesis.
  struct Context {
    Context(tok::TokenKind ContextKind, unsigned BindingStrength,
            bool IsExpression)
        : ContextKind(ContextKind), BindingStrength(BindingStrength),
          LongestObjCSelectorName(0), ColonIsForRangeExpr(false),
          ColonIsDictLiteral(false), ColonIsObjCMethodExpr(false),
          FirstObjCSelectorName(nullptr), FirstStartOfName(nullptr),
          IsExpression(IsExpression), CanBeExpression(true),
          InTemplateArgument(false), InCtorInitializer(false),
          CaretFound(false), IsForEachMacro(false) {}

    tok::TokenKind ContextKind;
    unsigned BindingStrength;
    unsigned LongestObjCSelectorName;
    bool ColonIsForRangeExpr;
    bool ColonIsDictLiteral;
    bool ColonIsObjCMethodExpr;
    FormatToken *FirstObjCSelectorName;
    FormatToken *FirstStartOfName;
    bool IsExpression;
    bool CanBeExpression;
    bool InTemplateArgument;
    bool InCtorInitializer;
    bool CaretFound;
    bool IsForEachMacro;
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
        !Line.First->isOneOf(tok::kw_template, tok::kw_using) &&
        (!Current.Previous || Current.Previous->isNot(tok::kw_operator))) {
      Contexts.back().IsExpression = true;
      for (FormatToken *Previous = Current.Previous;
           Previous && !Previous->isOneOf(tok::comma, tok::semi);
           Previous = Previous->Previous) {
        if (Previous->isOneOf(tok::r_square, tok::r_paren))
          Previous = Previous->MatchingParen;
        if ((Previous->Type == TT_BinaryOperator ||
             Previous->Type == TT_UnaryOperator) &&
            Previous->isOneOf(tok::star, tok::amp)) {
          Previous->Type = TT_PointerOrReference;
        }
      }
    } else if (Current.isOneOf(tok::kw_return, tok::kw_throw)) {
      Contexts.back().IsExpression = true;
    } else if (Current.is(tok::l_paren) && !Line.MustBeDeclaration &&
               !Line.InPPDirective &&
               (!Current.Previous ||
                Current.Previous->isNot(tok::kw_decltype))) {
      bool ParametersOfFunctionType =
          Current.Previous && Current.Previous->is(tok::r_paren) &&
          Current.Previous->MatchingParen &&
          Current.Previous->MatchingParen->Type == TT_FunctionTypeLParen;
      bool IsForOrCatch = Current.Previous &&
                          Current.Previous->isOneOf(tok::kw_for, tok::kw_catch);
      Contexts.back().IsExpression = !ParametersOfFunctionType && !IsForOrCatch;
    } else if (Current.isOneOf(tok::r_paren, tok::greater, tok::comma)) {
      for (FormatToken *Previous = Current.Previous;
           Previous && Previous->isOneOf(tok::star, tok::amp);
           Previous = Previous->Previous)
        Previous->Type = TT_PointerOrReference;
    } else if (Current.Previous &&
               Current.Previous->Type == TT_CtorInitializerColon) {
      Contexts.back().IsExpression = true;
      Contexts.back().InCtorInitializer = true;
    } else if (Current.is(tok::kw_new)) {
      Contexts.back().CanBeExpression = false;
    } else if (Current.is(tok::semi) || Current.is(tok::exclaim)) {
      // This should be the condition or increment in a for-loop.
      Contexts.back().IsExpression = true;
    }

    if (Current.Type == TT_Unknown) {
      // Line.MightBeFunctionDecl can only be true after the parentheses of a
      // function declaration have been found. In this case, 'Current' is a
      // trailing token of this declaration and thus cannot be a name.
      if (isStartOfName(Current) && !Line.MightBeFunctionDecl) {
        Contexts.back().FirstStartOfName = &Current;
        Current.Type = TT_StartOfName;
      } else if (Current.is(tok::kw_auto)) {
        AutoFound = true;
      } else if (Current.is(tok::arrow) && AutoFound &&
                 Line.MustBeDeclaration) {
        Current.Type = TT_TrailingReturnArrow;
      } else if (Current.isOneOf(tok::star, tok::amp, tok::ampamp)) {
        Current.Type =
            determineStarAmpUsage(Current, Contexts.back().CanBeExpression &&
                                               Contexts.back().IsExpression,
                                  Contexts.back().InTemplateArgument);
      } else if (Current.isOneOf(tok::minus, tok::plus, tok::caret)) {
        Current.Type = determinePlusMinusCaretUsage(Current);
        if (Current.Type == TT_UnaryOperator && Current.is(tok::caret))
          Contexts.back().CaretFound = true;
      } else if (Current.isOneOf(tok::minusminus, tok::plusplus)) {
        Current.Type = determineIncrementUsage(Current);
      } else if (Current.isOneOf(tok::exclaim, tok::tilde)) {
        Current.Type = TT_UnaryOperator;
      } else if (Current.is(tok::question)) {
        Current.Type = TT_ConditionalExpr;
      } else if (Current.isBinaryOperator() &&
                 (!Current.Previous ||
                  Current.Previous->isNot(tok::l_square))) {
        Current.Type = TT_BinaryOperator;
      } else if (Current.is(tok::comment)) {
        if (Current.TokenText.startswith("//"))
          Current.Type = TT_LineComment;
        else
          Current.Type = TT_BlockComment;
      } else if (Current.is(tok::r_paren)) {
        if (rParenEndsCast(Current))
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
      } else if (Current.isOneOf(tok::identifier, tok::kw_const) &&
                 Current.Previous && Current.Previous->isNot(tok::equal) &&
                 Line.MightBeFunctionDecl && Contexts.size() == 1) {
        // Line.MightBeFunctionDecl can only be true after the parentheses of a
        // function declaration have been found.
        Current.Type = TT_TrailingAnnotation;
      }
    }
  }

  /// \brief Take a guess at whether \p Tok starts a name of a function or
  /// variable declaration.
  ///
  /// This is a heuristic based on whether \p Tok is an identifier following
  /// something that is likely a type.
  bool isStartOfName(const FormatToken &Tok) {
    if (Tok.isNot(tok::identifier) || !Tok.Previous)
      return false;

    // Skip "const" as it does not have an influence on whether this is a name.
    FormatToken *PreviousNotConst = Tok.Previous;
    while (PreviousNotConst && PreviousNotConst->is(tok::kw_const))
      PreviousNotConst = PreviousNotConst->Previous;

    if (!PreviousNotConst)
      return false;

    bool IsPPKeyword = PreviousNotConst->is(tok::identifier) &&
                       PreviousNotConst->Previous &&
                       PreviousNotConst->Previous->is(tok::hash);

    if (PreviousNotConst->Type == TT_TemplateCloser)
      return PreviousNotConst && PreviousNotConst->MatchingParen &&
             PreviousNotConst->MatchingParen->Previous &&
             PreviousNotConst->MatchingParen->Previous->isNot(tok::kw_template);

    if (PreviousNotConst->is(tok::r_paren) && PreviousNotConst->MatchingParen &&
        PreviousNotConst->MatchingParen->Previous &&
        PreviousNotConst->MatchingParen->Previous->is(tok::kw_decltype))
      return true;

    return (!IsPPKeyword && PreviousNotConst->is(tok::identifier)) ||
           PreviousNotConst->Type == TT_PointerOrReference ||
           PreviousNotConst->isSimpleTypeSpecifier();
  }

  /// \brief Determine whether ')' is ending a cast.
  bool rParenEndsCast(const FormatToken &Tok) {
    FormatToken *LeftOfParens = nullptr;
    if (Tok.MatchingParen)
      LeftOfParens = Tok.MatchingParen->getPreviousNonComment();
    if (LeftOfParens && LeftOfParens->is(tok::r_paren))
      return false;
    if (LeftOfParens && LeftOfParens->is(tok::r_square) &&
        LeftOfParens->MatchingParen &&
        LeftOfParens->MatchingParen->Type == TT_LambdaLSquare)
      return false;
    bool IsCast = false;
    bool ParensAreEmpty = Tok.Previous == Tok.MatchingParen;
    bool ParensAreType = !Tok.Previous ||
                         Tok.Previous->Type == TT_PointerOrReference ||
                         Tok.Previous->Type == TT_TemplateCloser ||
                         Tok.Previous->isSimpleTypeSpecifier();
    bool ParensCouldEndDecl =
        Tok.Next && Tok.Next->isOneOf(tok::equal, tok::semi, tok::l_brace);
    bool IsSizeOfOrAlignOf =
        LeftOfParens && LeftOfParens->isOneOf(tok::kw_sizeof, tok::kw_alignof);
    if (ParensAreType && !ParensCouldEndDecl && !IsSizeOfOrAlignOf &&
        ((Contexts.size() > 1 && Contexts[Contexts.size() - 2].IsExpression) ||
         (Tok.Next && Tok.Next->isBinaryOperator())))
      IsCast = true;
    else if (Tok.Next && Tok.Next->isNot(tok::string_literal) &&
             (Tok.Next->Tok.isLiteral() ||
              Tok.Next->isOneOf(tok::kw_sizeof, tok::kw_alignof)))
      IsCast = true;
    // If there is an identifier after the (), it is likely a cast, unless
    // there is also an identifier before the ().
    else if (LeftOfParens &&
             (LeftOfParens->Tok.getIdentifierInfo() == nullptr ||
              LeftOfParens->is(tok::kw_return)) &&
             LeftOfParens->Type != TT_OverloadedOperator &&
             LeftOfParens->isNot(tok::at) &&
             LeftOfParens->Type != TT_TemplateCloser && Tok.Next) {
      if (Tok.Next->isOneOf(tok::identifier, tok::numeric_constant)) {
        IsCast = true;
      } else {
        // Use heuristics to recognize c style casting.
        FormatToken *Prev = Tok.Previous;
        if (Prev && Prev->isOneOf(tok::amp, tok::star))
          Prev = Prev->Previous;

        if (Prev && Tok.Next && Tok.Next->Next) {
          bool NextIsUnary = Tok.Next->isUnaryOperator() ||
                             Tok.Next->isOneOf(tok::amp, tok::star);
          IsCast = NextIsUnary && Tok.Next->Next->isOneOf(
                                      tok::identifier, tok::numeric_constant);
        }

        for (; Prev != Tok.MatchingParen; Prev = Prev->Previous) {
          if (!Prev || !Prev->isOneOf(tok::kw_const, tok::identifier)) {
            IsCast = false;
            break;
          }
        }
      }
    }
    return IsCast && !ParensAreEmpty;
  }

  /// \brief Return the type of the given token assuming it is * or &.
  TokenType determineStarAmpUsage(const FormatToken &Tok, bool IsExpression,
                                  bool InTemplateArgument) {
    const FormatToken *PrevToken = Tok.getPreviousNonComment();
    if (!PrevToken)
      return TT_UnaryOperator;

    const FormatToken *NextToken = Tok.getNextNonComment();
    if (!NextToken || NextToken->is(tok::l_brace))
      return TT_Unknown;

    if (PrevToken->is(tok::coloncolon) ||
        (PrevToken->is(tok::l_paren) && !IsExpression))
      return TT_PointerOrReference;

    if (PrevToken->isOneOf(tok::l_paren, tok::l_square, tok::l_brace,
                           tok::comma, tok::semi, tok::kw_return, tok::colon,
                           tok::equal, tok::kw_delete, tok::kw_sizeof) ||
        PrevToken->Type == TT_BinaryOperator ||
        PrevToken->Type == TT_ConditionalExpr ||
        PrevToken->Type == TT_UnaryOperator || PrevToken->Type == TT_CastRParen)
      return TT_UnaryOperator;

    if (NextToken->is(tok::l_square) && NextToken->Type != TT_LambdaLSquare)
      return TT_PointerOrReference;
    if (NextToken->isOneOf(tok::kw_operator, tok::comma))
      return TT_PointerOrReference;

    if (PrevToken->is(tok::r_paren) && PrevToken->MatchingParen &&
        PrevToken->MatchingParen->Previous &&
        PrevToken->MatchingParen->Previous->isOneOf(tok::kw_typeof,
                                                    tok::kw_decltype))
      return TT_PointerOrReference;

    if (PrevToken->Tok.isLiteral() ||
        PrevToken->isOneOf(tok::r_paren, tok::r_square, tok::kw_true,
                           tok::kw_false) ||
        NextToken->Tok.isLiteral() ||
        NextToken->isOneOf(tok::kw_true, tok::kw_false) ||
        NextToken->isUnaryOperator() ||
        // If we know we're in a template argument, there are no named
        // declarations. Thus, having an identifier on the right-hand side
        // indicates a binary operator.
        (InTemplateArgument && NextToken->Tok.isAnyIdentifier()))
      return TT_BinaryOperator;

    // This catches some cases where evaluation order is used as control flow:
    //   aaa && aaa->f();
    const FormatToken *NextNextToken = NextToken->getNextNonComment();
    if (NextNextToken && NextNextToken->is(tok::arrow))
      return TT_BinaryOperator;

    // It is very unlikely that we are going to find a pointer or reference type
    // definition on the RHS of an assignment.
    if (IsExpression)
      return TT_BinaryOperator;

    return TT_PointerOrReference;
  }

  TokenType determinePlusMinusCaretUsage(const FormatToken &Tok) {
    const FormatToken *PrevToken = Tok.getPreviousNonComment();
    if (!PrevToken || PrevToken->Type == TT_CastRParen)
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
    if (!PrevToken || PrevToken->Type == TT_CastRParen)
      return TT_UnaryOperator;
    if (PrevToken->isOneOf(tok::r_paren, tok::r_square, tok::identifier))
      return TT_TrailingUnaryOperator;

    return TT_UnaryOperator;
  }

  SmallVector<Context, 8> Contexts;

  const FormatStyle &Style;
  AnnotatedLine &Line;
  FormatToken *CurrentToken;
  bool KeywordVirtualFound;
  bool AutoFound;
  IdentifierInfo &Ident_in;
};

static int PrecedenceUnaryOperator = prec::PointerToMember + 1;
static int PrecedenceArrowAndPeriod = prec::PointerToMember + 2;

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
    // Skip 'return' and ObjC selector colons as they are not part of a binary
    // expression.
    while (Current &&
           (Current->is(tok::kw_return) ||
            (Current->is(tok::colon) && (Current->Type == TT_ObjCMethodExpr ||
                                         Current->Type == TT_DictLiteral))))
      next();

    if (!Current || Precedence > PrecedenceArrowAndPeriod)
      return;

    // Conditional expressions need to be parsed separately for proper nesting.
    if (Precedence == prec::Conditional) {
      parseConditionalExpr();
      return;
    }

    // Parse unary operators, which all have a higher precedence than binary
    // operators.
    if (Precedence == PrecedenceUnaryOperator) {
      parseUnaryOperator();
      return;
    }

    FormatToken *Start = Current;
    FormatToken *LatestOperator = nullptr;
    unsigned OperatorIndex = 0;

    while (Current) {
      // Consume operators with higher precedence.
      parse(Precedence + 1);

      int CurrentPrecedence = getCurrentPrecedence();

      if (Current && Current->Type == TT_SelectorName &&
          Precedence == CurrentPrecedence) {
        if (LatestOperator)
          addFakeParenthesis(Start, prec::Level(Precedence));
        Start = Current;
      }

      // At the end of the line or when an operator with higher precedence is
      // found, insert fake parenthesis and return.
      if (!Current || Current->closesScope() ||
          (CurrentPrecedence != -1 && CurrentPrecedence < Precedence)) {
        if (LatestOperator) {
          LatestOperator->LastOperator = true;
          if (Precedence == PrecedenceArrowAndPeriod) {
            // Call expressions don't have a binary operator precedence.
            addFakeParenthesis(Start, prec::Unknown);
          } else {
            addFakeParenthesis(Start, prec::Level(Precedence));
          }
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
        if (CurrentPrecedence == Precedence) {
          LatestOperator = Current;
          Current->OperatorIndex = OperatorIndex;
          ++OperatorIndex;
        }

        next();
      }
    }
  }

private:
  /// \brief Gets the precedence (+1) of the given token for binary operators
  /// and other tokens that we treat like binary operators.
  int getCurrentPrecedence() {
    if (Current) {
      if (Current->Type == TT_ConditionalExpr)
        return prec::Conditional;
      else if (Current->is(tok::semi) || Current->Type == TT_InlineASMColon ||
               Current->Type == TT_SelectorName)
        return 0;
      else if (Current->Type == TT_RangeBasedForLoopColon)
        return prec::Comma;
      else if (Current->Type == TT_BinaryOperator || Current->is(tok::comma))
        return Current->getPrecedence();
      else if (Current->isOneOf(tok::period, tok::arrow))
        return PrecedenceArrowAndPeriod;
    }
    return -1;
  }

  void addFakeParenthesis(FormatToken *Start, prec::Level Precedence) {
    Start->FakeLParens.push_back(Precedence);
    if (Precedence > prec::Unknown)
      Start->StartsBinaryExpression = true;
    if (Current) {
      ++Current->Previous->FakeRParens;
      if (Precedence > prec::Unknown)
        Current->Previous->EndsBinaryExpression = true;
    }
  }

  /// \brief Parse unary operator expressions and surround them with fake
  /// parentheses if appropriate.
  void parseUnaryOperator() {
    if (!Current || Current->Type != TT_UnaryOperator) {
      parse(PrecedenceArrowAndPeriod);
      return;
    }

    FormatToken *Start = Current;
    next();
    parseUnaryOperator();

    // The actual precedence doesn't matter.
    addFakeParenthesis(Start, prec::Unknown);
  }

  void parseConditionalExpr() {
    FormatToken *Start = Current;
    parse(prec::LogicalOr);
    if (!Current || !Current->is(tok::question))
      return;
    next();
    parseConditionalExpr();
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

void
TokenAnnotator::setCommentLineLevels(SmallVectorImpl<AnnotatedLine *> &Lines) {
  const AnnotatedLine *NextNonCommentLine = nullptr;
  for (SmallVectorImpl<AnnotatedLine *>::reverse_iterator I = Lines.rbegin(),
                                                          E = Lines.rend();
       I != E; ++I) {
    if (NextNonCommentLine && (*I)->First->is(tok::comment) &&
        (*I)->First->Next == nullptr)
      (*I)->Level = NextNonCommentLine->Level;
    else
      NextNonCommentLine = (*I)->First->isNot(tok::r_brace) ? (*I) : nullptr;

    setCommentLineLevels((*I)->Children);
  }
}

void TokenAnnotator::annotate(AnnotatedLine &Line) {
  for (SmallVectorImpl<AnnotatedLine *>::iterator I = Line.Children.begin(),
                                                  E = Line.Children.end();
       I != E; ++I) {
    annotate(**I);
  }
  AnnotatingParser Parser(Style, Line, Ident_in);
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

// This function heuristically determines whether 'Current' starts the name of a
// function declaration.
static bool isFunctionDeclarationName(const FormatToken &Current) {
  if (Current.Type != TT_StartOfName ||
      Current.NestingLevel != 0 ||
      Current.Previous->Type == TT_StartOfName)
    return false;
  const FormatToken *Next = Current.Next;
  for (; Next; Next = Next->Next) {
    if (Next->Type == TT_TemplateOpener) {
      Next = Next->MatchingParen;
    } else if (Next->is(tok::coloncolon)) {
      Next = Next->Next;
      if (!Next || !Next->is(tok::identifier))
        return false;
    } else if (Next->is(tok::l_paren)) {
      break;
    } else {
      return false;
    }
  }
  if (!Next)
    return false;
  assert(Next->is(tok::l_paren));
  if (Next->Next == Next->MatchingParen)
    return true;
  for (const FormatToken *Tok = Next->Next; Tok != Next->MatchingParen;
       Tok = Tok->Next) {
    if (Tok->is(tok::kw_const) || Tok->isSimpleTypeSpecifier() ||
        Tok->Type == TT_PointerOrReference || Tok->Type == TT_StartOfName)
      return true;
    if (Tok->isOneOf(tok::l_brace, tok::string_literal) || Tok->Tok.isLiteral())
      return false;
  }
  return false;
}

void TokenAnnotator::calculateFormattingInformation(AnnotatedLine &Line) {
  for (SmallVectorImpl<AnnotatedLine *>::iterator I = Line.Children.begin(),
                                                  E = Line.Children.end();
       I != E; ++I) {
    calculateFormattingInformation(**I);
  }

  Line.First->TotalLength =
      Line.First->IsMultiline ? Style.ColumnLimit : Line.First->ColumnWidth;
  if (!Line.First->Next)
    return;
  FormatToken *Current = Line.First->Next;
  bool InFunctionDecl = Line.MightBeFunctionDecl;
  while (Current) {
    if (isFunctionDeclarationName(*Current))
      Current->Type = TT_FunctionDeclarationName;
    if (Current->Type == TT_LineComment) {
      if (Current->Previous->BlockKind == BK_BracedInit &&
          Current->Previous->opensScope())
        Current->SpacesRequiredBefore = Style.Cpp11BracedListStyle ? 0 : 1;
      else
        Current->SpacesRequiredBefore = Style.SpacesBeforeTrailingComments;

      // If we find a trailing comment, iterate backwards to determine whether
      // it seems to relate to a specific parameter. If so, break before that
      // parameter to avoid changing the comment's meaning. E.g. don't move 'b'
      // to the previous line in:
      //   SomeFunction(a,
      //                b, // comment
      //                c);
      if (!Current->HasUnescapedNewline) {
        for (FormatToken *Parameter = Current->Previous; Parameter;
             Parameter = Parameter->Previous) {
          if (Parameter->isOneOf(tok::comment, tok::r_brace))
            break;
          if (Parameter->Previous && Parameter->Previous->is(tok::comma)) {
            if (Parameter->Previous->Type != TT_CtorInitializerComma &&
                Parameter->HasUnescapedNewline)
              Parameter->MustBreakBefore = true;
            break;
          }
        }
      }
    } else if (Current->SpacesRequiredBefore == 0 &&
               spaceRequiredBefore(Line, *Current)) {
      Current->SpacesRequiredBefore = 1;
    }

    Current->MustBreakBefore =
        Current->MustBreakBefore || mustBreakBefore(Line, *Current);

    if (Style.AlwaysBreakAfterDefinitionReturnType &&
        InFunctionDecl && Current->Type == TT_FunctionDeclarationName &&
        Line.Last->is(tok::l_brace))  // Only for definitions.
      Current->MustBreakBefore = true;

    Current->CanBreakBefore =
        Current->MustBreakBefore || canBreakBefore(Line, *Current);
    unsigned ChildSize = 0;
    if (Current->Previous->Children.size() == 1) {
      FormatToken &LastOfChild = *Current->Previous->Children[0]->Last;
      ChildSize = LastOfChild.isTrailingComment() ? Style.ColumnLimit
                                                  : LastOfChild.TotalLength + 1;
    }
    if (Current->MustBreakBefore || Current->Previous->Children.size() > 1 ||
        Current->IsMultiline)
      Current->TotalLength = Current->Previous->TotalLength + Style.ColumnLimit;
    else
      Current->TotalLength = Current->Previous->TotalLength +
                             Current->ColumnWidth + ChildSize +
                             Current->SpacesRequiredBefore;

    if (Current->Type == TT_CtorInitializerColon)
      InFunctionDecl = false;

    // FIXME: Only calculate this if CanBreakBefore is true once static
    // initializers etc. are sorted out.
    // FIXME: Move magic numbers to a better place.
    Current->SplitPenalty = 20 * Current->BindingStrength +
                            splitPenalty(Line, *Current, InFunctionDecl);

    Current = Current->Next;
  }

  calculateUnbreakableTailLengths(Line);
  for (Current = Line.First; Current != nullptr; Current = Current->Next) {
    if (Current->Role)
      Current->Role->precomputeFormattingInfos(Current);
  }

  DEBUG({ printDebugInfo(Line); });
}

void TokenAnnotator::calculateUnbreakableTailLengths(AnnotatedLine &Line) {
  unsigned UnbreakableTailLength = 0;
  FormatToken *Current = Line.Last;
  while (Current) {
    Current->UnbreakableTailLength = UnbreakableTailLength;
    if (Current->CanBreakBefore ||
        Current->isOneOf(tok::comment, tok::string_literal)) {
      UnbreakableTailLength = 0;
    } else {
      UnbreakableTailLength +=
          Current->ColumnWidth + Current->SpacesRequiredBefore;
    }
    Current = Current->Previous;
  }
}

unsigned TokenAnnotator::splitPenalty(const AnnotatedLine &Line,
                                      const FormatToken &Tok,
                                      bool InFunctionDecl) {
  const FormatToken &Left = *Tok.Previous;
  const FormatToken &Right = Tok;

  if (Left.is(tok::semi))
    return 0;
  if (Left.is(tok::comma) || (Right.is(tok::identifier) && Right.Next &&
                              Right.Next->Type == TT_DictLiteral))
    return 1;
  if (Right.is(tok::l_square)) {
    if (Style.Language == FormatStyle::LK_Proto)
      return 1;
    if (Right.Type != TT_ObjCMethodExpr && Right.Type != TT_LambdaLSquare)
      return 500;
  }
  if (Right.Type == TT_StartOfName ||
      Right.Type == TT_FunctionDeclarationName || Right.is(tok::kw_operator)) {
    if (Line.First->is(tok::kw_for) && Right.PartOfMultiVariableDeclStmt)
      return 3;
    if (Left.Type == TT_StartOfName)
      return 20;
    if (InFunctionDecl && Right.NestingLevel == 0)
      return Style.PenaltyReturnTypeOnItsOwnLine;
    return 200;
  }
  if (Left.is(tok::equal) && Right.is(tok::l_brace))
    return 150;
  if (Left.Type == TT_CastRParen)
    return 100;
  if (Left.is(tok::coloncolon) ||
      (Right.is(tok::period) && Style.Language == FormatStyle::LK_Proto))
    return 500;
  if (Left.isOneOf(tok::kw_class, tok::kw_struct))
    return 5000;

  if (Left.Type == TT_RangeBasedForLoopColon ||
      Left.Type == TT_InheritanceColon)
    return 2;

  if (Right.isMemberAccess()) {
    if (Left.is(tok::r_paren) && Left.MatchingParen &&
        Left.MatchingParen->ParameterCount > 0)
      return 20; // Should be smaller than breaking at a nested comma.
    return 150;
  }

  if (Right.Type == TT_TrailingAnnotation &&
      (!Right.Next || Right.Next->isNot(tok::l_paren))) {
    // Generally, breaking before a trailing annotation is bad unless it is
    // function-like. It seems to be especially preferable to keep standard
    // annotations (i.e. "const", "final" and "override") on the same line.
    // Use a slightly higher penalty after ")" so that annotations like
    // "const override" are kept together.
    bool is_short_annotation = Right.TokenText.size() < 10;
    return (Left.is(tok::r_paren) ? 100 : 120) + (is_short_annotation ? 50 : 0);
  }

  // In for-loops, prefer breaking at ',' and ';'.
  if (Line.First->is(tok::kw_for) && Left.is(tok::equal))
    return 4;

  // In Objective-C method expressions, prefer breaking before "param:" over
  // breaking after it.
  if (Right.Type == TT_SelectorName)
    return 0;
  if (Left.is(tok::colon) && Left.Type == TT_ObjCMethodExpr)
    return Line.MightBeFunctionDecl ? 50 : 500;

  if (Left.is(tok::l_paren) && InFunctionDecl)
    return 100;
  if (Left.is(tok::equal) && InFunctionDecl)
    return 110;
  if (Right.is(tok::r_brace))
    return 1;
  if (Left.opensScope())
    return Left.ParameterCount > 1 ? Style.PenaltyBreakBeforeFirstCallParameter
                                   : 19;

  if (Right.is(tok::lessless)) {
    if (Left.is(tok::string_literal)) {
      StringRef Content = Left.TokenText;
      if (Content.startswith("\""))
        Content = Content.drop_front(1);
      if (Content.endswith("\""))
        Content = Content.drop_back(1);
      Content = Content.trim();
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
  if (Style.Language == FormatStyle::LK_Proto) {
    if (Right.is(tok::period) &&
        (Left.TokenText == "optional" || Left.TokenText == "required" ||
         Left.TokenText == "repeated"))
      return true;
    if (Right.is(tok::l_paren) &&
        (Left.TokenText == "returns" || Left.TokenText == "option"))
      return true;
  } else if (Style.Language == FormatStyle::LK_JavaScript) {
    if (Left.TokenText == "var")
      return true;
  }
  if (Left.is(tok::kw_return) && Right.isNot(tok::semi))
    return true;
  if (Style.ObjCSpaceAfterProperty && Line.Type == LT_ObjCProperty &&
      Left.Tok.getObjCKeywordID() == tok::objc_property)
    return true;
  if (Right.is(tok::hashhash))
    return Left.is(tok::hash);
  if (Left.isOneOf(tok::hashhash, tok::hash))
    return Right.is(tok::hash);
  if (Left.is(tok::l_paren) && Right.is(tok::r_paren))
    return Style.SpaceInEmptyParentheses;
  if (Left.is(tok::l_paren) || Right.is(tok::r_paren))
    return (Right.Type == TT_CastRParen ||
            (Left.MatchingParen && Left.MatchingParen->Type == TT_CastRParen))
               ? Style.SpacesInCStyleCastParentheses
               : Style.SpacesInParentheses;
  if (Style.SpacesInAngles &&
      ((Left.Type == TT_TemplateOpener) != (Right.Type == TT_TemplateCloser)))
    return true;
  if (Right.isOneOf(tok::semi, tok::comma))
    return false;
  if (Right.is(tok::less) &&
      (Left.isOneOf(tok::kw_template, tok::r_paren) ||
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
  if (Right.is(tok::coloncolon) && Left.isNot(tok::l_brace))
    return (Left.is(tok::less) && Style.Standard == FormatStyle::LS_Cpp03) ||
           !Left.isOneOf(tok::identifier, tok::greater, tok::l_paren,
                         tok::r_paren, tok::less);
  if (Left.is(tok::less) || Right.isOneOf(tok::greater, tok::less))
    return false;
  if (Right.is(tok::ellipsis))
    return Left.Tok.isLiteral();
  if (Left.is(tok::l_square) && Right.is(tok::amp))
    return false;
  if (Right.Type == TT_PointerOrReference)
    return Left.Tok.isLiteral() ||
           ((Left.Type != TT_PointerOrReference) && Left.isNot(tok::l_paren) &&
            Style.PointerAlignment != FormatStyle::PAS_Left);
  if (Right.Type == TT_FunctionTypeLParen && Left.isNot(tok::l_paren) &&
      (Left.Type != TT_PointerOrReference ||
       Style.PointerAlignment != FormatStyle::PAS_Right))
    return true;
  if (Left.Type == TT_PointerOrReference)
    return Right.Tok.isLiteral() || Right.Type == TT_BlockComment ||
           ((Right.Type != TT_PointerOrReference) &&
            Right.isNot(tok::l_paren) &&
            Style.PointerAlignment != FormatStyle::PAS_Right && Left.Previous &&
            !Left.Previous->isOneOf(tok::l_paren, tok::coloncolon));
  if (Right.is(tok::star) && Left.is(tok::l_paren))
    return false;
  if (Left.is(tok::l_square))
    return Left.Type == TT_ArrayInitializerLSquare &&
           Style.SpacesInContainerLiterals && Right.isNot(tok::r_square);
  if (Right.is(tok::r_square))
    return Right.MatchingParen && Style.SpacesInContainerLiterals &&
           Right.MatchingParen->Type == TT_ArrayInitializerLSquare;
  if (Right.is(tok::l_square) && Right.Type != TT_ObjCMethodExpr &&
      Right.Type != TT_LambdaLSquare && Left.isNot(tok::numeric_constant) &&
      Left.Type != TT_DictLiteral)
    return false;
  if (Left.is(tok::colon))
    return Left.Type != TT_ObjCMethodExpr;
  if (Left.Type == TT_BlockComment)
    return !Left.TokenText.endswith("=*/");
  if (Right.is(tok::l_paren)) {
    if (Left.is(tok::r_paren) && Left.Type == TT_AttributeParen)
      return true;
    return Line.Type == LT_ObjCDecl ||
           Left.isOneOf(tok::kw_new, tok::kw_delete, tok::semi) ||
           (Style.SpaceBeforeParens != FormatStyle::SBPO_Never &&
            (Left.isOneOf(tok::kw_if, tok::kw_for, tok::kw_while,
                          tok::kw_switch, tok::kw_catch, tok::kw_case) ||
             Left.IsForEachMacro)) ||
           (Style.SpaceBeforeParens == FormatStyle::SBPO_Always &&
            (Left.is(tok::identifier) || Left.isFunctionLikeKeyword()) &&
            Line.Type != LT_PreprocessorDirective);
  }
  if (Left.is(tok::at) && Right.Tok.getObjCKeywordID() != tok::objc_not_keyword)
    return false;
  if (Left.is(tok::l_brace) && Right.is(tok::r_brace))
    return !Left.Children.empty(); // No spaces in "{}".
  if ((Left.is(tok::l_brace) && Left.BlockKind != BK_Block) ||
      (Right.is(tok::r_brace) && Right.MatchingParen &&
       Right.MatchingParen->BlockKind != BK_Block))
    return !Style.Cpp11BracedListStyle;
  if (Right.Type == TT_UnaryOperator)
    return !Left.isOneOf(tok::l_paren, tok::l_square, tok::at) &&
           (Left.isNot(tok::colon) || Left.Type != TT_ObjCMethodExpr);
  if ((Left.isOneOf(tok::identifier, tok::greater, tok::r_square,
                    tok::r_paren) ||
       Left.isSimpleTypeSpecifier()) &&
      Right.is(tok::l_brace) && Right.getNextNonComment() &&
      Right.BlockKind != BK_Block)
    return false;
  if (Left.is(tok::period) || Right.is(tok::period))
    return false;
  if (Right.is(tok::hash) && Left.is(tok::identifier) && Left.TokenText == "L")
    return false;
  return true;
}

bool TokenAnnotator::spaceRequiredBefore(const AnnotatedLine &Line,
                                         const FormatToken &Tok) {
  if (Tok.Tok.getIdentifierInfo() && Tok.Previous->Tok.getIdentifierInfo())
    return true; // Never ever merge two identifiers.
  if (Tok.Previous->Type == TT_ImplicitStringLiteral)
    return Tok.WhitespaceRange.getBegin() != Tok.WhitespaceRange.getEnd();
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
    return Tok.is(tok::coloncolon);
  if (Tok.Type == TT_OverloadedOperatorLParen)
    return false;
  if (Tok.is(tok::colon))
    return !Line.First->isOneOf(tok::kw_case, tok::kw_default) &&
           Tok.getNextNonComment() && Tok.Type != TT_ObjCMethodExpr &&
           !Tok.Previous->is(tok::question) &&
           (Tok.Type != TT_DictLiteral || Style.SpacesInContainerLiterals);
  if (Tok.Previous->Type == TT_UnaryOperator ||
      Tok.Previous->Type == TT_CastRParen)
    return Tok.Type == TT_BinaryOperator;
  if (Tok.Previous->is(tok::greater) && Tok.is(tok::greater)) {
    return Tok.Type == TT_TemplateCloser &&
           Tok.Previous->Type == TT_TemplateCloser &&
           (Style.Standard != FormatStyle::LS_Cpp11 || Style.SpacesInAngles);
  }
  if (Tok.isOneOf(tok::arrowstar, tok::periodstar) ||
      Tok.Previous->isOneOf(tok::arrowstar, tok::periodstar))
    return false;
  if (!Style.SpaceBeforeAssignmentOperators &&
      Tok.getPrecedence() == prec::Assignment)
    return false;
  if ((Tok.Type == TT_BinaryOperator && !Tok.Previous->is(tok::l_paren)) ||
      Tok.Previous->Type == TT_BinaryOperator ||
      Tok.Previous->Type == TT_ConditionalExpr)
    return true;
  if (Tok.Previous->Type == TT_TemplateCloser && Tok.is(tok::l_paren))
    return Style.SpaceBeforeParens == FormatStyle::SBPO_Always;
  if (Tok.is(tok::less) && Tok.Previous->isNot(tok::l_paren) &&
      Line.First->is(tok::hash))
    return true;
  if (Tok.Type == TT_TrailingUnaryOperator)
    return false;
  if (Tok.Previous->Type == TT_RegexLiteral)
    return false;
  return spaceRequiredBetween(Line, *Tok.Previous, Tok);
}

// Returns 'true' if 'Tok' is a brace we'd want to break before in Allman style.
static bool isAllmanBrace(const FormatToken &Tok) {
  return Tok.is(tok::l_brace) && Tok.BlockKind == BK_Block &&
         Tok.Type != TT_ObjCBlockLBrace && Tok.Type != TT_DictLiteral;
}

bool TokenAnnotator::mustBreakBefore(const AnnotatedLine &Line,
                                     const FormatToken &Right) {
  const FormatToken &Left = *Right.Previous;
  if (Right.NewlinesBefore > 1)
    return true;
  if (Right.is(tok::comment)) {
    return Right.Previous->BlockKind != BK_BracedInit &&
           Right.Previous->Type != TT_CtorInitializerColon &&
           (Right.NewlinesBefore > 0 && Right.HasUnescapedNewline);
  } else if (Right.Previous->isTrailingComment() ||
             (Right.isStringLiteral() && Right.Previous->isStringLiteral())) {
    return true;
  } else if (Right.Previous->IsUnterminatedLiteral) {
    return true;
  } else if (Right.is(tok::lessless) && Right.Next &&
             Right.Previous->is(tok::string_literal) &&
             Right.Next->is(tok::string_literal)) {
    return true;
  } else if (Right.Previous->ClosesTemplateDeclaration &&
             Right.Previous->MatchingParen &&
             Right.Previous->MatchingParen->NestingLevel == 0 &&
             Style.AlwaysBreakTemplateDeclarations) {
    return true;
  } else if ((Right.Type == TT_CtorInitializerComma ||
              Right.Type == TT_CtorInitializerColon) &&
             Style.BreakConstructorInitializersBeforeComma &&
             !Style.ConstructorInitializerAllOnOneLineOrOnePerLine) {
    return true;
  } else if (Right.is(tok::string_literal) &&
             Right.TokenText.startswith("R\"")) {
    // Raw string literals are special wrt. line breaks. The author has made a
    // deliberate choice and might have aligned the contents of the string
    // literal accordingly. Thus, we try keep existing line breaks.
    return Right.NewlinesBefore > 0;
  } else if (Right.Previous->is(tok::l_brace) && Right.NestingLevel == 1 &&
             Style.Language == FormatStyle::LK_Proto) {
    // Don't enums onto single lines in protocol buffers.
    return true;
  } else if (isAllmanBrace(Left) || isAllmanBrace(Right)) {
    return Style.BreakBeforeBraces == FormatStyle::BS_Allman ||
           Style.BreakBeforeBraces == FormatStyle::BS_GNU;
  } else if (Style.Language == FormatStyle::LK_Proto &&
             Left.isNot(tok::l_brace) && Right.Type == TT_SelectorName) {
    return true;
  }

  // If the last token before a '}' is a comma or a comment, the intention is to
  // insert a line break after it in order to make shuffling around entries
  // easier.
  const FormatToken *BeforeClosingBrace = nullptr;
  if (Left.is(tok::l_brace) && Left.MatchingParen)
    BeforeClosingBrace = Left.MatchingParen->Previous;
  else if (Right.is(tok::r_brace))
    BeforeClosingBrace = Right.Previous;
  if (BeforeClosingBrace &&
      BeforeClosingBrace->isOneOf(tok::comma, tok::comment))
    return true;

  if (Style.Language == FormatStyle::LK_JavaScript) {
    // FIXME: This might apply to other languages and token kinds.
    if (Right.is(tok::char_constant) && Left.is(tok::plus) && Left.Previous &&
        Left.Previous->is(tok::char_constant))
      return true;
  }

  return false;
}

bool TokenAnnotator::canBreakBefore(const AnnotatedLine &Line,
                                    const FormatToken &Right) {
  const FormatToken &Left = *Right.Previous;
  if (Left.is(tok::at))
    return false;
  if (Left.Tok.getObjCKeywordID() == tok::objc_interface)
    return false;
  if (Right.Type == TT_StartOfName ||
      Right.Type == TT_FunctionDeclarationName || Right.is(tok::kw_operator))
    return true;
  if (Right.isTrailingComment())
    // We rely on MustBreakBefore being set correctly here as we should not
    // change the "binding" behavior of a comment.
    // The first comment in a braced lists is always interpreted as belonging to
    // the first list element. Otherwise, it should be placed outside of the
    // list.
    return Left.BlockKind == BK_BracedInit;
  if (Left.is(tok::question) && Right.is(tok::colon))
    return false;
  if (Right.Type == TT_ConditionalExpr || Right.is(tok::question))
    return Style.BreakBeforeTernaryOperators;
  if (Left.Type == TT_ConditionalExpr || Left.is(tok::question))
    return !Style.BreakBeforeTernaryOperators;
  if (Right.Type == TT_InheritanceColon)
    return true;
  if (Right.is(tok::colon) && (Right.Type != TT_CtorInitializerColon &&
                               Right.Type != TT_InlineASMColon))
    return false;
  if (Left.is(tok::colon) &&
      (Left.Type == TT_DictLiteral || Left.Type == TT_ObjCMethodExpr))
    return true;
  if (Right.Type == TT_SelectorName)
    return true;
  if (Left.is(tok::r_paren) && Line.Type == LT_ObjCProperty)
    return true;
  if (Left.ClosesTemplateDeclaration)
    return true;
  if (Right.Type == TT_RangeBasedForLoopColon ||
      Right.Type == TT_OverloadedOperatorLParen ||
      Right.Type == TT_OverloadedOperator)
    return false;
  if (Left.Type == TT_RangeBasedForLoopColon)
    return true;
  if (Right.Type == TT_RangeBasedForLoopColon)
    return false;
  if (Left.Type == TT_PointerOrReference || Left.Type == TT_TemplateCloser ||
      Left.Type == TT_UnaryOperator || Left.is(tok::kw_operator))
    return false;
  if (Left.is(tok::equal) && Line.Type == LT_VirtualFunctionDecl)
    return false;
  if (Left.is(tok::l_paren) && Left.Type == TT_AttributeParen)
    return false;
  if (Left.is(tok::l_paren) && Left.Previous &&
      (Left.Previous->Type == TT_BinaryOperator ||
       Left.Previous->Type == TT_CastRParen || Left.Previous->is(tok::kw_if)))
    return false;
  if (Right.Type == TT_ImplicitStringLiteral)
    return false;

  if (Right.is(tok::r_paren) || Right.Type == TT_TemplateCloser)
    return false;

  // We only break before r_brace if there was a corresponding break before
  // the l_brace, which is tracked by BreakBeforeClosingBrace.
  if (Right.is(tok::r_brace))
    return Right.MatchingParen && Right.MatchingParen->BlockKind == BK_Block;

  // Allow breaking after a trailing annotation, e.g. after a method
  // declaration.
  if (Left.Type == TT_TrailingAnnotation)
    return !Right.isOneOf(tok::l_brace, tok::semi, tok::equal, tok::l_paren,
                          tok::less, tok::coloncolon);

  if (Right.is(tok::kw___attribute))
    return true;

  if (Left.is(tok::identifier) && Right.is(tok::string_literal))
    return true;

  if (Right.is(tok::identifier) && Right.Next &&
      Right.Next->Type == TT_DictLiteral)
    return true;

  if (Left.Type == TT_CtorInitializerComma &&
      Style.BreakConstructorInitializersBeforeComma)
    return false;
  if (Right.Type == TT_CtorInitializerComma &&
      Style.BreakConstructorInitializersBeforeComma)
    return true;
  if (Left.is(tok::greater) && Right.is(tok::greater) &&
      Left.Type != TT_TemplateCloser)
    return false;
  if (Right.Type == TT_BinaryOperator && Style.BreakBeforeBinaryOperators)
    return true;
  if (Left.Type == TT_ArrayInitializerLSquare)
    return true;
  return (Left.isBinaryOperator() &&
          !Left.isOneOf(tok::arrowstar, tok::lessless) &&
          !Style.BreakBeforeBinaryOperators) ||
         Left.isOneOf(tok::comma, tok::coloncolon, tok::semi, tok::l_brace,
                      tok::kw_class, tok::kw_struct) ||
         Right.isMemberAccess() ||
         Right.isOneOf(tok::lessless, tok::colon, tok::l_square, tok::at,
                       tok::kw_typename) ||
         (Left.is(tok::r_paren) &&
          Right.isOneOf(tok::identifier, tok::kw_const)) ||
         (Left.is(tok::l_paren) && !Right.is(tok::r_paren));
}

void TokenAnnotator::printDebugInfo(const AnnotatedLine &Line) {
  llvm::errs() << "AnnotatedTokens:\n";
  const FormatToken *Tok = Line.First;
  while (Tok) {
    llvm::errs() << " M=" << Tok->MustBreakBefore
                 << " C=" << Tok->CanBreakBefore << " T=" << Tok->Type
                 << " S=" << Tok->SpacesRequiredBefore
                 << " B=" << Tok->BlockParameterCount
                 << " P=" << Tok->SplitPenalty << " Name=" << Tok->Tok.getName()
                 << " L=" << Tok->TotalLength << " PPK=" << Tok->PackingKind
                 << " FakeLParens=";
    for (unsigned i = 0, e = Tok->FakeLParens.size(); i != e; ++i)
      llvm::errs() << Tok->FakeLParens[i] << "/";
    llvm::errs() << " FakeRParens=" << Tok->FakeRParens << "\n";
    if (!Tok->Next)
      assert(Tok == Line.Last);
    Tok = Tok->Next;
  }
  llvm::errs() << "----\n";
}

} // namespace format
} // namespace clang
