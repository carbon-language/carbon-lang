//===--- Format.cpp - Format C++ code -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements functions declared in Format.h. This will be
/// split into separate files as we go.
///
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "format-formatter"

#include "BreakableToken.h"
#include "TokenAnnotator.h"
#include "UnwrappedLineParser.h"
#include "WhitespaceManager.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/OperatorPrecedence.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Format/Format.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/YAMLTraits.h"
#include <queue>
#include <string>

namespace llvm {
namespace yaml {
template <>
struct ScalarEnumerationTraits<clang::format::FormatStyle::LanguageStandard> {
  static void enumeration(IO &IO,
                          clang::format::FormatStyle::LanguageStandard &Value) {
    IO.enumCase(Value, "C++03", clang::format::FormatStyle::LS_Cpp03);
    IO.enumCase(Value, "C++11", clang::format::FormatStyle::LS_Cpp11);
    IO.enumCase(Value, "Auto", clang::format::FormatStyle::LS_Auto);
  }
};

template <>
struct ScalarEnumerationTraits<clang::format::FormatStyle::BraceBreakingStyle> {
  static void
  enumeration(IO &IO, clang::format::FormatStyle::BraceBreakingStyle &Value) {
    IO.enumCase(Value, "Attach", clang::format::FormatStyle::BS_Attach);
    IO.enumCase(Value, "Linux", clang::format::FormatStyle::BS_Linux);
    IO.enumCase(Value, "Stroustrup", clang::format::FormatStyle::BS_Stroustrup);
  }
};

template <> struct MappingTraits<clang::format::FormatStyle> {
  static void mapping(llvm::yaml::IO &IO, clang::format::FormatStyle &Style) {
    if (IO.outputting()) {
      StringRef StylesArray[] = { "LLVM", "Google", "Chromium", "Mozilla" };
      ArrayRef<StringRef> Styles(StylesArray);
      for (size_t i = 0, e = Styles.size(); i < e; ++i) {
        StringRef StyleName(Styles[i]);
        clang::format::FormatStyle PredefinedStyle;
        if (clang::format::getPredefinedStyle(StyleName, &PredefinedStyle) &&
            Style == PredefinedStyle) {
          IO.mapOptional("# BasedOnStyle", StyleName);
          break;
        }
      }
    } else {
      StringRef BasedOnStyle;
      IO.mapOptional("BasedOnStyle", BasedOnStyle);
      if (!BasedOnStyle.empty())
        if (!clang::format::getPredefinedStyle(BasedOnStyle, &Style)) {
          IO.setError(Twine("Unknown value for BasedOnStyle: ", BasedOnStyle));
          return;
        }
    }

    IO.mapOptional("AccessModifierOffset", Style.AccessModifierOffset);
    IO.mapOptional("AlignEscapedNewlinesLeft", Style.AlignEscapedNewlinesLeft);
    IO.mapOptional("AllowAllParametersOfDeclarationOnNextLine",
                   Style.AllowAllParametersOfDeclarationOnNextLine);
    IO.mapOptional("AllowShortIfStatementsOnASingleLine",
                   Style.AllowShortIfStatementsOnASingleLine);
    IO.mapOptional("AllowShortLoopsOnASingleLine",
                   Style.AllowShortLoopsOnASingleLine);
    IO.mapOptional("AlwaysBreakTemplateDeclarations",
                   Style.AlwaysBreakTemplateDeclarations);
    IO.mapOptional("AlwaysBreakBeforeMultilineStrings",
                   Style.AlwaysBreakBeforeMultilineStrings);
    IO.mapOptional("BinPackParameters", Style.BinPackParameters);
    IO.mapOptional("ColumnLimit", Style.ColumnLimit);
    IO.mapOptional("ConstructorInitializerAllOnOneLineOrOnePerLine",
                   Style.ConstructorInitializerAllOnOneLineOrOnePerLine);
    IO.mapOptional("DerivePointerBinding", Style.DerivePointerBinding);
    IO.mapOptional("IndentCaseLabels", Style.IndentCaseLabels);
    IO.mapOptional("MaxEmptyLinesToKeep", Style.MaxEmptyLinesToKeep);
    IO.mapOptional("ObjCSpaceBeforeProtocolList",
                   Style.ObjCSpaceBeforeProtocolList);
    IO.mapOptional("PenaltyBreakComment", Style.PenaltyBreakComment);
    IO.mapOptional("PenaltyBreakString", Style.PenaltyBreakString);
    IO.mapOptional("PenaltyExcessCharacter", Style.PenaltyExcessCharacter);
    IO.mapOptional("PenaltyReturnTypeOnItsOwnLine",
                   Style.PenaltyReturnTypeOnItsOwnLine);
    IO.mapOptional("PointerBindsToType", Style.PointerBindsToType);
    IO.mapOptional("SpacesBeforeTrailingComments",
                   Style.SpacesBeforeTrailingComments);
    IO.mapOptional("SpacesInBracedLists", Style.SpacesInBracedLists);
    IO.mapOptional("Standard", Style.Standard);
    IO.mapOptional("IndentWidth", Style.IndentWidth);
    IO.mapOptional("UseTab", Style.UseTab);
    IO.mapOptional("BreakBeforeBraces", Style.BreakBeforeBraces);
    IO.mapOptional("IndentFunctionDeclarationAfterType",
                   Style.IndentFunctionDeclarationAfterType);
  }
};
}
}

namespace clang {
namespace format {

FormatStyle getLLVMStyle() {
  FormatStyle LLVMStyle;
  LLVMStyle.AccessModifierOffset = -2;
  LLVMStyle.AlignEscapedNewlinesLeft = false;
  LLVMStyle.AllowAllParametersOfDeclarationOnNextLine = true;
  LLVMStyle.AllowShortIfStatementsOnASingleLine = false;
  LLVMStyle.AllowShortLoopsOnASingleLine = false;
  LLVMStyle.AlwaysBreakTemplateDeclarations = false;
  LLVMStyle.AlwaysBreakBeforeMultilineStrings = false;
  LLVMStyle.BinPackParameters = true;
  LLVMStyle.ColumnLimit = 80;
  LLVMStyle.ConstructorInitializerAllOnOneLineOrOnePerLine = false;
  LLVMStyle.DerivePointerBinding = false;
  LLVMStyle.IndentCaseLabels = false;
  LLVMStyle.MaxEmptyLinesToKeep = 1;
  LLVMStyle.ObjCSpaceBeforeProtocolList = true;
  LLVMStyle.PenaltyBreakComment = 45;
  LLVMStyle.PenaltyBreakString = 1000;
  LLVMStyle.PenaltyExcessCharacter = 1000000;
  LLVMStyle.PenaltyReturnTypeOnItsOwnLine = 75;
  LLVMStyle.PointerBindsToType = false;
  LLVMStyle.SpacesBeforeTrailingComments = 1;
  LLVMStyle.SpacesInBracedLists = true;
  LLVMStyle.Standard = FormatStyle::LS_Cpp03;
  LLVMStyle.IndentWidth = 2;
  LLVMStyle.UseTab = false;
  LLVMStyle.BreakBeforeBraces = FormatStyle::BS_Attach;
  LLVMStyle.IndentFunctionDeclarationAfterType = false;
  return LLVMStyle;
}

FormatStyle getGoogleStyle() {
  FormatStyle GoogleStyle;
  GoogleStyle.AccessModifierOffset = -1;
  GoogleStyle.AlignEscapedNewlinesLeft = true;
  GoogleStyle.AllowAllParametersOfDeclarationOnNextLine = true;
  GoogleStyle.AllowShortIfStatementsOnASingleLine = true;
  GoogleStyle.AllowShortLoopsOnASingleLine = true;
  GoogleStyle.AlwaysBreakTemplateDeclarations = true;
  GoogleStyle.AlwaysBreakBeforeMultilineStrings = true;
  GoogleStyle.BinPackParameters = true;
  GoogleStyle.ColumnLimit = 80;
  GoogleStyle.ConstructorInitializerAllOnOneLineOrOnePerLine = true;
  GoogleStyle.DerivePointerBinding = true;
  GoogleStyle.IndentCaseLabels = true;
  GoogleStyle.MaxEmptyLinesToKeep = 1;
  GoogleStyle.ObjCSpaceBeforeProtocolList = false;
  GoogleStyle.PenaltyBreakComment = 45;
  GoogleStyle.PenaltyBreakString = 1000;
  GoogleStyle.PenaltyExcessCharacter = 1000000;
  GoogleStyle.PenaltyReturnTypeOnItsOwnLine = 200;
  GoogleStyle.PointerBindsToType = true;
  GoogleStyle.SpacesBeforeTrailingComments = 2;
  GoogleStyle.SpacesInBracedLists = false;
  GoogleStyle.Standard = FormatStyle::LS_Auto;
  GoogleStyle.IndentWidth = 2;
  GoogleStyle.UseTab = false;
  GoogleStyle.BreakBeforeBraces = FormatStyle::BS_Attach;
  GoogleStyle.IndentFunctionDeclarationAfterType = true;
  return GoogleStyle;
}

FormatStyle getChromiumStyle() {
  FormatStyle ChromiumStyle = getGoogleStyle();
  ChromiumStyle.AllowAllParametersOfDeclarationOnNextLine = false;
  ChromiumStyle.AllowShortIfStatementsOnASingleLine = false;
  ChromiumStyle.AllowShortLoopsOnASingleLine = false;
  ChromiumStyle.BinPackParameters = false;
  ChromiumStyle.Standard = FormatStyle::LS_Cpp03;
  ChromiumStyle.DerivePointerBinding = false;
  return ChromiumStyle;
}

FormatStyle getMozillaStyle() {
  FormatStyle MozillaStyle = getLLVMStyle();
  MozillaStyle.AllowAllParametersOfDeclarationOnNextLine = false;
  MozillaStyle.ConstructorInitializerAllOnOneLineOrOnePerLine = true;
  MozillaStyle.DerivePointerBinding = true;
  MozillaStyle.IndentCaseLabels = true;
  MozillaStyle.ObjCSpaceBeforeProtocolList = false;
  MozillaStyle.PenaltyReturnTypeOnItsOwnLine = 200;
  MozillaStyle.PointerBindsToType = true;
  return MozillaStyle;
}

bool getPredefinedStyle(StringRef Name, FormatStyle *Style) {
  if (Name.equals_lower("llvm"))
    *Style = getLLVMStyle();
  else if (Name.equals_lower("chromium"))
    *Style = getChromiumStyle();
  else if (Name.equals_lower("mozilla"))
    *Style = getMozillaStyle();
  else if (Name.equals_lower("google"))
    *Style = getGoogleStyle();
  else
    return false;

  return true;
}

llvm::error_code parseConfiguration(StringRef Text, FormatStyle *Style) {
  if (Text.trim().empty())
    return llvm::make_error_code(llvm::errc::invalid_argument);
  llvm::yaml::Input Input(Text);
  Input >> *Style;
  return Input.error();
}

std::string configurationAsText(const FormatStyle &Style) {
  std::string Text;
  llvm::raw_string_ostream Stream(Text);
  llvm::yaml::Output Output(Stream);
  // We use the same mapping method for input and output, so we need a non-const
  // reference here.
  FormatStyle NonConstStyle = Style;
  Output << NonConstStyle;
  return Stream.str();
}

// Returns the length of everything up to the first possible line break after
// the ), ], } or > matching \c Tok.
static unsigned getLengthToMatchingParen(const FormatToken &Tok) {
  if (Tok.MatchingParen == NULL)
    return 0;
  FormatToken *End = Tok.MatchingParen;
  while (End->Next && !End->Next->CanBreakBefore) {
    End = End->Next;
  }
  return End->TotalLength - Tok.TotalLength + 1;
}

namespace {

class UnwrappedLineFormatter {
public:
  UnwrappedLineFormatter(const FormatStyle &Style, SourceManager &SourceMgr,
                         const AnnotatedLine &Line, unsigned FirstIndent,
                         const FormatToken *RootToken,
                         WhitespaceManager &Whitespaces,
                         encoding::Encoding Encoding)
      : Style(Style), SourceMgr(SourceMgr), Line(Line),
        FirstIndent(FirstIndent), RootToken(RootToken),
        Whitespaces(Whitespaces), Count(0), Encoding(Encoding) {}

  /// \brief Formats an \c UnwrappedLine.
  void format(const AnnotatedLine *NextLine) {
    // Initialize state dependent on indent.
    LineState State;
    State.Column = FirstIndent;
    State.NextToken = RootToken;
    State.Stack.push_back(
        ParenState(FirstIndent, FirstIndent, /*AvoidBinPacking=*/false,
                   /*NoLineBreak=*/false));
    State.LineContainsContinuedForLoopSection = false;
    State.ParenLevel = 0;
    State.StartOfStringLiteral = 0;
    State.StartOfLineLevel = State.ParenLevel;
    State.LowestCallLevel = State.ParenLevel;
    State.IgnoreStackForComparison = false;

    // The first token has already been indented and thus consumed.
    moveStateToNextToken(State, /*DryRun=*/false);

    // If everything fits on a single line, just put it there.
    unsigned ColumnLimit = Style.ColumnLimit;
    if (NextLine && NextLine->InPPDirective &&
        !NextLine->First->HasUnescapedNewline)
      ColumnLimit = getColumnLimit();
    if (Line.Last->TotalLength <= ColumnLimit - FirstIndent) {
      while (State.NextToken != NULL) {
        addTokenToState(false, false, State);
      }
    }

    // If the ObjC method declaration does not fit on a line, we should format
    // it with one arg per line.
    if (Line.Type == LT_ObjCMethodDecl)
      State.Stack.back().BreakBeforeParameter = true;

    // Find best solution in solution space.
    analyzeSolutionSpace(State);
  }

private:
  void DebugTokenState(const FormatToken &FormatTok) {
    const Token &Tok = FormatTok.Tok;
    llvm::dbgs() << StringRef(SourceMgr.getCharacterData(Tok.getLocation()),
                              Tok.getLength());
    llvm::dbgs();
  }

  struct ParenState {
    ParenState(unsigned Indent, unsigned LastSpace, bool AvoidBinPacking,
               bool NoLineBreak)
        : Indent(Indent), LastSpace(LastSpace), FirstLessLess(0),
          BreakBeforeClosingBrace(false), QuestionColumn(0),
          AvoidBinPacking(AvoidBinPacking), BreakBeforeParameter(false),
          NoLineBreak(NoLineBreak), ColonPos(0), StartOfFunctionCall(0),
          NestedNameSpecifierContinuation(0), CallContinuation(0),
          VariablePos(0), ForFakeParenthesis(false) {}

    /// \brief The position to which a specific parenthesis level needs to be
    /// indented.
    unsigned Indent;

    /// \brief The position of the last space on each level.
    ///
    /// Used e.g. to break like:
    /// functionCall(Parameter, otherCall(
    ///                             OtherParameter));
    unsigned LastSpace;

    /// \brief The position the first "<<" operator encountered on each level.
    ///
    /// Used to align "<<" operators. 0 if no such operator has been encountered
    /// on a level.
    unsigned FirstLessLess;

    /// \brief Whether a newline needs to be inserted before the block's closing
    /// brace.
    ///
    /// We only want to insert a newline before the closing brace if there also
    /// was a newline after the beginning left brace.
    bool BreakBeforeClosingBrace;

    /// \brief The column of a \c ? in a conditional expression;
    unsigned QuestionColumn;

    /// \brief Avoid bin packing, i.e. multiple parameters/elements on multiple
    /// lines, in this context.
    bool AvoidBinPacking;

    /// \brief Break after the next comma (or all the commas in this context if
    /// \c AvoidBinPacking is \c true).
    bool BreakBeforeParameter;

    /// \brief Line breaking in this context would break a formatting rule.
    bool NoLineBreak;

    /// \brief The position of the colon in an ObjC method declaration/call.
    unsigned ColonPos;

    /// \brief The start of the most recent function in a builder-type call.
    unsigned StartOfFunctionCall;

    /// \brief If a nested name specifier was broken over multiple lines, this
    /// contains the start column of the second line. Otherwise 0.
    unsigned NestedNameSpecifierContinuation;

    /// \brief If a call expression was broken over multiple lines, this
    /// contains the start column of the second line. Otherwise 0.
    unsigned CallContinuation;

    /// \brief The column of the first variable name in a variable declaration.
    ///
    /// Used to align further variables if necessary.
    unsigned VariablePos;

    /// \brief \c true if this \c ParenState was created for a fake parenthesis.
    ///
    /// Does not need to be considered for memoization / the comparison function
    /// as otherwise identical states will have the same fake/non-fake
    /// \c ParenStates.
    bool ForFakeParenthesis;

    bool operator<(const ParenState &Other) const {
      if (Indent != Other.Indent)
        return Indent < Other.Indent;
      if (LastSpace != Other.LastSpace)
        return LastSpace < Other.LastSpace;
      if (FirstLessLess != Other.FirstLessLess)
        return FirstLessLess < Other.FirstLessLess;
      if (BreakBeforeClosingBrace != Other.BreakBeforeClosingBrace)
        return BreakBeforeClosingBrace;
      if (QuestionColumn != Other.QuestionColumn)
        return QuestionColumn < Other.QuestionColumn;
      if (AvoidBinPacking != Other.AvoidBinPacking)
        return AvoidBinPacking;
      if (BreakBeforeParameter != Other.BreakBeforeParameter)
        return BreakBeforeParameter;
      if (NoLineBreak != Other.NoLineBreak)
        return NoLineBreak;
      if (ColonPos != Other.ColonPos)
        return ColonPos < Other.ColonPos;
      if (StartOfFunctionCall != Other.StartOfFunctionCall)
        return StartOfFunctionCall < Other.StartOfFunctionCall;
      if (CallContinuation != Other.CallContinuation)
        return CallContinuation < Other.CallContinuation;
      if (VariablePos != Other.VariablePos)
        return VariablePos < Other.VariablePos;
      return false;
    }
  };

  /// \brief The current state when indenting a unwrapped line.
  ///
  /// As the indenting tries different combinations this is copied by value.
  struct LineState {
    /// \brief The number of used columns in the current line.
    unsigned Column;

    /// \brief The token that needs to be next formatted.
    const FormatToken *NextToken;

    /// \brief \c true if this line contains a continued for-loop section.
    bool LineContainsContinuedForLoopSection;

    /// \brief The level of nesting inside (), [], <> and {}.
    unsigned ParenLevel;

    /// \brief The \c ParenLevel at the start of this line.
    unsigned StartOfLineLevel;

    /// \brief The lowest \c ParenLevel of "." or "->" on the current line.
    unsigned LowestCallLevel;

    /// \brief The start column of the string literal, if we're in a string
    /// literal sequence, 0 otherwise.
    unsigned StartOfStringLiteral;

    /// \brief A stack keeping track of properties applying to parenthesis
    /// levels.
    std::vector<ParenState> Stack;

    /// \brief Ignore the stack of \c ParenStates for state comparison.
    ///
    /// In long and deeply nested unwrapped lines, the current algorithm can
    /// be insufficient for finding the best formatting with a reasonable amount
    /// of time and memory. Setting this flag will effectively lead to the
    /// algorithm not analyzing some combinations. However, these combinations
    /// rarely contain the optimal solution: In short, accepting a higher
    /// penalty early would need to lead to different values in the \c
    /// ParenState stack (in an otherwise identical state) and these different
    /// values would need to lead to a significant amount of avoided penalty
    /// later.
    ///
    /// FIXME: Come up with a better algorithm instead.
    bool IgnoreStackForComparison;

    /// \brief Comparison operator to be able to used \c LineState in \c map.
    bool operator<(const LineState &Other) const {
      if (NextToken != Other.NextToken)
        return NextToken < Other.NextToken;
      if (Column != Other.Column)
        return Column < Other.Column;
      if (LineContainsContinuedForLoopSection !=
          Other.LineContainsContinuedForLoopSection)
        return LineContainsContinuedForLoopSection;
      if (ParenLevel != Other.ParenLevel)
        return ParenLevel < Other.ParenLevel;
      if (StartOfLineLevel != Other.StartOfLineLevel)
        return StartOfLineLevel < Other.StartOfLineLevel;
      if (LowestCallLevel != Other.LowestCallLevel)
        return LowestCallLevel < Other.LowestCallLevel;
      if (StartOfStringLiteral != Other.StartOfStringLiteral)
        return StartOfStringLiteral < Other.StartOfStringLiteral;
      if (IgnoreStackForComparison || Other.IgnoreStackForComparison)
        return false;
      return Stack < Other.Stack;
    }
  };

  /// \brief Appends the next token to \p State and updates information
  /// necessary for indentation.
  ///
  /// Puts the token on the current line if \p Newline is \c false and adds a
  /// line break and necessary indentation otherwise.
  ///
  /// If \p DryRun is \c false, also creates and stores the required
  /// \c Replacement.
  unsigned addTokenToState(bool Newline, bool DryRun, LineState &State) {
    const FormatToken &Current = *State.NextToken;
    const FormatToken &Previous = *State.NextToken->Previous;

    if (State.Stack.size() == 0 || Current.Type == TT_ImplicitStringLiteral) {
      // FIXME: Is this correct?
      int WhitespaceLength = SourceMgr.getSpellingColumnNumber(
                                 State.NextToken->WhitespaceRange.getEnd()) -
                             SourceMgr.getSpellingColumnNumber(
                                 State.NextToken->WhitespaceRange.getBegin());
      State.Column += WhitespaceLength + State.NextToken->CodePointCount;
      State.NextToken = State.NextToken->Next;
      return 0;
    }

    // If we are continuing an expression, we want to indent an extra 4 spaces.
    unsigned ContinuationIndent =
        std::max(State.Stack.back().LastSpace, State.Stack.back().Indent) + 4;
    if (Newline) {
      if (Current.is(tok::r_brace)) {
        State.Column = Line.Level * Style.IndentWidth;
      } else if (Current.is(tok::string_literal) &&
                 State.StartOfStringLiteral != 0) {
        State.Column = State.StartOfStringLiteral;
        State.Stack.back().BreakBeforeParameter = true;
      } else if (Current.is(tok::lessless) &&
                 State.Stack.back().FirstLessLess != 0) {
        State.Column = State.Stack.back().FirstLessLess;
      } else if (Current.isOneOf(tok::period, tok::arrow) &&
                 Current.Type != TT_DesignatedInitializerPeriod) {
        if (State.Stack.back().CallContinuation == 0) {
          State.Column = ContinuationIndent;
          State.Stack.back().CallContinuation = State.Column;
        } else {
          State.Column = State.Stack.back().CallContinuation;
        }
      } else if (Current.Type == TT_ConditionalExpr) {
        State.Column = State.Stack.back().QuestionColumn;
      } else if (Previous.is(tok::comma) &&
                 State.Stack.back().VariablePos != 0) {
        State.Column = State.Stack.back().VariablePos;
      } else if (Previous.ClosesTemplateDeclaration ||
                 (Current.Type == TT_StartOfName && State.ParenLevel == 0 &&
                  (!Style.IndentFunctionDeclarationAfterType ||
                   Line.StartsDefinition))) {
        State.Column = State.Stack.back().Indent;
      } else if (Current.Type == TT_ObjCSelectorName) {
        if (State.Stack.back().ColonPos > Current.CodePointCount) {
          State.Column = State.Stack.back().ColonPos - Current.CodePointCount;
        } else {
          State.Column = State.Stack.back().Indent;
          State.Stack.back().ColonPos = State.Column + Current.CodePointCount;
        }
      } else if (Current.Type == TT_StartOfName ||
                 Previous.isOneOf(tok::coloncolon, tok::equal) ||
                 Previous.Type == TT_ObjCMethodExpr) {
        State.Column = ContinuationIndent;
      } else {
        State.Column = State.Stack.back().Indent;
        // Ensure that we fall back to indenting 4 spaces instead of just
        // flushing continuations left.
        if (State.Column == FirstIndent)
          State.Column += 4;
      }

      if (Current.is(tok::question))
        State.Stack.back().BreakBeforeParameter = true;
      if ((Previous.isOneOf(tok::comma, tok::semi) &&
           !State.Stack.back().AvoidBinPacking) ||
          Previous.Type == TT_BinaryOperator)
        State.Stack.back().BreakBeforeParameter = false;
      if (Previous.Type == TT_TemplateCloser && State.ParenLevel == 0)
        State.Stack.back().BreakBeforeParameter = false;

      if (!DryRun) {
        unsigned NewLines = 1;
        if (Current.is(tok::comment))
          NewLines = std::max(
              NewLines,
              std::min(Current.NewlinesBefore, Style.MaxEmptyLinesToKeep + 1));
        Whitespaces.replaceWhitespace(Current, NewLines, State.Column,
                                      State.Column, Line.InPPDirective);
      }

      State.Stack.back().LastSpace = State.Column;
      if (Current.isOneOf(tok::arrow, tok::period) &&
          Current.Type != TT_DesignatedInitializerPeriod)
        State.Stack.back().LastSpace += Current.CodePointCount;
      State.StartOfLineLevel = State.ParenLevel;
      State.LowestCallLevel = State.ParenLevel;

      // Any break on this level means that the parent level has been broken
      // and we need to avoid bin packing there.
      for (unsigned i = 0, e = State.Stack.size() - 1; i != e; ++i) {
        State.Stack[i].BreakBeforeParameter = true;
      }
      const FormatToken *TokenBefore = Current.getPreviousNoneComment();
      if (TokenBefore && !TokenBefore->isOneOf(tok::comma, tok::semi) &&
          TokenBefore->Type != TT_TemplateCloser &&
          TokenBefore->Type != TT_BinaryOperator && !TokenBefore->opensScope())
        State.Stack.back().BreakBeforeParameter = true;

      // If we break after {, we should also break before the corresponding }.
      if (Previous.is(tok::l_brace))
        State.Stack.back().BreakBeforeClosingBrace = true;

      if (State.Stack.back().AvoidBinPacking) {
        // If we are breaking after '(', '{', '<', this is not bin packing
        // unless AllowAllParametersOfDeclarationOnNextLine is false.
        if (!(Previous.isOneOf(tok::l_paren, tok::l_brace) ||
              Previous.Type == TT_BinaryOperator) ||
            (!Style.AllowAllParametersOfDeclarationOnNextLine &&
             Line.MustBeDeclaration))
          State.Stack.back().BreakBeforeParameter = true;
      }
    } else {
      if (Current.is(tok::equal) &&
          (RootToken->is(tok::kw_for) || State.ParenLevel == 0) &&
          State.Stack.back().VariablePos == 0) {
        State.Stack.back().VariablePos = State.Column;
        // Move over * and & if they are bound to the variable name.
        const FormatToken *Tok = &Previous;
        while (Tok && State.Stack.back().VariablePos >= Tok->CodePointCount) {
          State.Stack.back().VariablePos -= Tok->CodePointCount;
          if (Tok->SpacesRequiredBefore != 0)
            break;
          Tok = Tok->Previous;
        }
        if (Previous.PartOfMultiVariableDeclStmt)
          State.Stack.back().LastSpace = State.Stack.back().VariablePos;
      }

      unsigned Spaces = State.NextToken->SpacesRequiredBefore;

      if (!DryRun)
        Whitespaces.replaceWhitespace(Current, 0, Spaces,
                                      State.Column + Spaces);

      if (Current.Type == TT_ObjCSelectorName &&
          State.Stack.back().ColonPos == 0) {
        if (State.Stack.back().Indent + Current.LongestObjCSelectorName >
            State.Column + Spaces + Current.CodePointCount)
          State.Stack.back().ColonPos =
              State.Stack.back().Indent + Current.LongestObjCSelectorName;
        else
          State.Stack.back().ColonPos =
              State.Column + Spaces + Current.CodePointCount;
      }

      if (Previous.opensScope() && Previous.Type != TT_ObjCMethodExpr &&
          Current.Type != TT_LineComment)
        State.Stack.back().Indent = State.Column + Spaces;
      if (Previous.is(tok::comma) && !Current.isTrailingComment() &&
          State.Stack.back().AvoidBinPacking)
        State.Stack.back().NoLineBreak = true;

      State.Column += Spaces;
      if (Current.is(tok::l_paren) && Previous.isOneOf(tok::kw_if, tok::kw_for))
        // Treat the condition inside an if as if it was a second function
        // parameter, i.e. let nested calls have an indent of 4.
        State.Stack.back().LastSpace = State.Column + 1; // 1 is length of "(".
      else if (Previous.is(tok::comma))
        State.Stack.back().LastSpace = State.Column;
      else if ((Previous.Type == TT_BinaryOperator ||
                Previous.Type == TT_ConditionalExpr ||
                Previous.Type == TT_CtorInitializerColon) &&
               !(Previous.getPrecedence() == prec::Assignment &&
                 Current.FakeLParens.empty()))
        // Always indent relative to the RHS of the expression unless this is a
        // simple assignment without binary expression on the RHS.
        State.Stack.back().LastSpace = State.Column;
      else if (Previous.Type == TT_InheritanceColon)
        State.Stack.back().Indent = State.Column;
      else if (Previous.opensScope() && !Current.FakeLParens.empty())
        // If this function has multiple parameters or a binary expression
        // parameter, indent nested calls from the start of the first parameter.
        State.Stack.back().LastSpace = State.Column;
    }

    return moveStateToNextToken(State, DryRun);
  }

  /// \brief Mark the next token as consumed in \p State and modify its stacks
  /// accordingly.
  unsigned moveStateToNextToken(LineState &State, bool DryRun) {
    const FormatToken &Current = *State.NextToken;
    assert(State.Stack.size());

    if (Current.Type == TT_InheritanceColon)
      State.Stack.back().AvoidBinPacking = true;
    if (Current.is(tok::lessless) && State.Stack.back().FirstLessLess == 0)
      State.Stack.back().FirstLessLess = State.Column;
    if (Current.is(tok::question))
      State.Stack.back().QuestionColumn = State.Column;
    if (Current.isOneOf(tok::period, tok::arrow)) {
      State.LowestCallLevel = std::min(State.LowestCallLevel, State.ParenLevel);
      if (Line.Type == LT_BuilderTypeCall && State.ParenLevel == 0)
        State.Stack.back().StartOfFunctionCall =
            Current.LastInChainOfCalls ? 0
                                       : State.Column + Current.CodePointCount;
    }
    if (Current.Type == TT_CtorInitializerColon) {
      // Indent 2 from the column, so:
      // SomeClass::SomeClass()
      //     : First(...), ...
      //       Next(...)
      //       ^ line up here.
      State.Stack.back().Indent = State.Column + 2;
      if (Style.ConstructorInitializerAllOnOneLineOrOnePerLine)
        State.Stack.back().AvoidBinPacking = true;
      State.Stack.back().BreakBeforeParameter = false;
    }

    // If return returns a binary expression, align after it.
    if (Current.is(tok::kw_return) && !Current.FakeLParens.empty())
      State.Stack.back().LastSpace = State.Column + 7;

    // In ObjC method declaration we align on the ":" of parameters, but we need
    // to ensure that we indent parameters on subsequent lines by at least 4.
    if (Current.Type == TT_ObjCMethodSpecifier)
      State.Stack.back().Indent += 4;

    // Insert scopes created by fake parenthesis.
    const FormatToken *Previous = Current.getPreviousNoneComment();
    // Don't add extra indentation for the first fake parenthesis after
    // 'return', assignements or opening <({[. The indentation for these cases
    // is special cased.
    bool SkipFirstExtraIndent =
        Current.is(tok::kw_return) ||
        (Previous && (Previous->opensScope() ||
                      Previous->getPrecedence() == prec::Assignment));
    for (SmallVector<prec::Level, 4>::const_reverse_iterator
             I = Current.FakeLParens.rbegin(),
             E = Current.FakeLParens.rend();
         I != E; ++I) {
      ParenState NewParenState = State.Stack.back();
      NewParenState.ForFakeParenthesis = true;
      NewParenState.Indent =
          std::max(std::max(State.Column, NewParenState.Indent),
                   State.Stack.back().LastSpace);

      // Always indent conditional expressions. Never indent expression where
      // the 'operator' is ',', ';' or an assignment (i.e. *I <=
      // prec::Assignment) as those have different indentation rules. Indent
      // other expression, unless the indentation needs to be skipped.
      if (*I == prec::Conditional ||
          (!SkipFirstExtraIndent && *I > prec::Assignment))
        NewParenState.Indent += 4;
      if (Previous && !Previous->opensScope())
        NewParenState.BreakBeforeParameter = false;
      State.Stack.push_back(NewParenState);
      SkipFirstExtraIndent = false;
    }

    // If we encounter an opening (, [, { or <, we add a level to our stacks to
    // prepare for the following tokens.
    if (Current.opensScope()) {
      unsigned NewIndent;
      unsigned LastSpace = State.Stack.back().LastSpace;
      bool AvoidBinPacking;
      if (Current.is(tok::l_brace)) {
        NewIndent = Style.IndentWidth + LastSpace;
        const FormatToken *NextNoComment = Current.getNextNoneComment();
        AvoidBinPacking = NextNoComment &&
                          NextNoComment->Type == TT_DesignatedInitializerPeriod;
      } else {
        NewIndent =
            4 + std::max(LastSpace, State.Stack.back().StartOfFunctionCall);
        AvoidBinPacking = !Style.BinPackParameters;
      }

      State.Stack.push_back(ParenState(NewIndent, LastSpace, AvoidBinPacking,
                                       State.Stack.back().NoLineBreak));
      ++State.ParenLevel;
    }

    // If this '[' opens an ObjC call, determine whether all parameters fit into
    // one line and put one per line if they don't.
    if (Current.is(tok::l_square) && Current.Type == TT_ObjCMethodExpr &&
        Current.MatchingParen != NULL) {
      if (getLengthToMatchingParen(Current) + State.Column > getColumnLimit())
        State.Stack.back().BreakBeforeParameter = true;
    }

    // If we encounter a closing ), ], } or >, we can remove a level from our
    // stacks.
    if (Current.isOneOf(tok::r_paren, tok::r_square) ||
        (Current.is(tok::r_brace) && State.NextToken != RootToken) ||
        State.NextToken->Type == TT_TemplateCloser) {
      State.Stack.pop_back();
      --State.ParenLevel;
    }

    // Remove scopes created by fake parenthesis.
    for (unsigned i = 0, e = Current.FakeRParens; i != e; ++i) {
      unsigned VariablePos = State.Stack.back().VariablePos;
      State.Stack.pop_back();
      State.Stack.back().VariablePos = VariablePos;
    }

    if (Current.is(tok::string_literal) && State.StartOfStringLiteral == 0) {
      State.StartOfStringLiteral = State.Column;
    } else if (!Current.isOneOf(tok::comment, tok::identifier, tok::hash,
                                tok::string_literal)) {
      State.StartOfStringLiteral = 0;
    }

    State.Column += Current.CodePointCount;

    State.NextToken = State.NextToken->Next;

    return breakProtrudingToken(Current, State, DryRun);
  }

  /// \brief If the current token sticks out over the end of the line, break
  /// it if possible.
  ///
  /// \returns An extra penalty if a token was broken, otherwise 0.
  ///
  /// The returned penalty will cover the cost of the additional line breaks and
  /// column limit violation in all lines except for the last one. The penalty
  /// for the column limit violation in the last line (and in single line
  /// tokens) is handled in \c addNextStateToQueue.
  unsigned breakProtrudingToken(const FormatToken &Current, LineState &State,
                                bool DryRun) {
    llvm::OwningPtr<BreakableToken> Token;
    unsigned StartColumn = State.Column - Current.CodePointCount;
    unsigned OriginalStartColumn =
        SourceMgr.getSpellingColumnNumber(Current.getStartOfNonWhitespace()) -
        1;

    if (Current.is(tok::string_literal) &&
        Current.Type != TT_ImplicitStringLiteral) {
      // Only break up default narrow strings.
      if (!Current.TokenText.startswith("\""))
        return 0;

      Token.reset(new BreakableStringLiteral(Current, StartColumn,
                                             Line.InPPDirective, Encoding));
    } else if (Current.Type == TT_BlockComment) {
      Token.reset(new BreakableBlockComment(
          Style, Current, StartColumn, OriginalStartColumn, !Current.Previous,
          Line.InPPDirective, Encoding));
    } else if (Current.Type == TT_LineComment &&
               (Current.Previous == NULL ||
                Current.Previous->Type != TT_ImplicitStringLiteral)) {
      Token.reset(new BreakableLineComment(Current, StartColumn,
                                           Line.InPPDirective, Encoding));
    } else {
      return 0;
    }
    if (Current.UnbreakableTailLength >= getColumnLimit())
      return 0;

    unsigned RemainingSpace = getColumnLimit() - Current.UnbreakableTailLength;
    bool BreakInserted = false;
    unsigned Penalty = 0;
    unsigned RemainingTokenColumns = 0;
    for (unsigned LineIndex = 0, EndIndex = Token->getLineCount();
         LineIndex != EndIndex; ++LineIndex) {
      if (!DryRun)
        Token->replaceWhitespaceBefore(LineIndex, Whitespaces);
      unsigned TailOffset = 0;
      RemainingTokenColumns = Token->getLineLengthAfterSplit(
          LineIndex, TailOffset, StringRef::npos);
      while (RemainingTokenColumns > RemainingSpace) {
        BreakableToken::Split Split =
            Token->getSplit(LineIndex, TailOffset, getColumnLimit());
        if (Split.first == StringRef::npos) {
          // The last line's penalty is handled in addNextStateToQueue().
          if (LineIndex < EndIndex - 1)
            Penalty += Style.PenaltyExcessCharacter *
                       (RemainingTokenColumns - RemainingSpace);
          break;
        }
        assert(Split.first != 0);
        unsigned NewRemainingTokenColumns = Token->getLineLengthAfterSplit(
            LineIndex, TailOffset + Split.first + Split.second,
            StringRef::npos);
        assert(NewRemainingTokenColumns < RemainingTokenColumns);
        if (!DryRun)
          Token->insertBreak(LineIndex, TailOffset, Split, Whitespaces);
        Penalty += Current.is(tok::string_literal) ? Style.PenaltyBreakString
                                                   : Style.PenaltyBreakComment;
        unsigned ColumnsUsed =
            Token->getLineLengthAfterSplit(LineIndex, TailOffset, Split.first);
        if (ColumnsUsed > getColumnLimit()) {
          Penalty +=
              Style.PenaltyExcessCharacter * (ColumnsUsed - getColumnLimit());
        }
        TailOffset += Split.first + Split.second;
        RemainingTokenColumns = NewRemainingTokenColumns;
        BreakInserted = true;
      }
    }

    State.Column = RemainingTokenColumns;

    if (BreakInserted) {
      // If we break the token inside a parameter list, we need to break before
      // the next parameter on all levels, so that the next parameter is clearly
      // visible. Line comments already introduce a break.
      if (Current.Type != TT_LineComment) {
        for (unsigned i = 0, e = State.Stack.size(); i != e; ++i)
          State.Stack[i].BreakBeforeParameter = true;
      }

      State.Stack.back().LastSpace = StartColumn;
    }
    return Penalty;
  }

  unsigned getColumnLimit() {
    // In preprocessor directives reserve two chars for trailing " \"
    return Style.ColumnLimit - (Line.InPPDirective ? 2 : 0);
  }

  /// \brief An edge in the solution space from \c Previous->State to \c State,
  /// inserting a newline dependent on the \c NewLine.
  struct StateNode {
    StateNode(const LineState &State, bool NewLine, StateNode *Previous)
        : State(State), NewLine(NewLine), Previous(Previous) {}
    LineState State;
    bool NewLine;
    StateNode *Previous;
  };

  /// \brief A pair of <penalty, count> that is used to prioritize the BFS on.
  ///
  /// In case of equal penalties, we want to prefer states that were inserted
  /// first. During state generation we make sure that we insert states first
  /// that break the line as late as possible.
  typedef std::pair<unsigned, unsigned> OrderedPenalty;

  /// \brief An item in the prioritized BFS search queue. The \c StateNode's
  /// \c State has the given \c OrderedPenalty.
  typedef std::pair<OrderedPenalty, StateNode *> QueueItem;

  /// \brief The BFS queue type.
  typedef std::priority_queue<QueueItem, std::vector<QueueItem>,
                              std::greater<QueueItem> > QueueType;

  /// \brief Analyze the entire solution space starting from \p InitialState.
  ///
  /// This implements a variant of Dijkstra's algorithm on the graph that spans
  /// the solution space (\c LineStates are the nodes). The algorithm tries to
  /// find the shortest path (the one with lowest penalty) from \p InitialState
  /// to a state where all tokens are placed.
  void analyzeSolutionSpace(LineState &InitialState) {
    std::set<LineState> Seen;

    // Insert start element into queue.
    StateNode *Node =
        new (Allocator.Allocate()) StateNode(InitialState, false, NULL);
    Queue.push(QueueItem(OrderedPenalty(0, Count), Node));
    ++Count;

    // While not empty, take first element and follow edges.
    while (!Queue.empty()) {
      unsigned Penalty = Queue.top().first.first;
      StateNode *Node = Queue.top().second;
      if (Node->State.NextToken == NULL) {
        DEBUG(llvm::dbgs() << "\n---\nPenalty for line: " << Penalty << "\n");
        break;
      }
      Queue.pop();

      // Cut off the analysis of certain solutions if the analysis gets too
      // complex. See description of IgnoreStackForComparison.
      if (Count > 10000)
        Node->State.IgnoreStackForComparison = true;

      if (!Seen.insert(Node->State).second)
        // State already examined with lower penalty.
        continue;

      addNextStateToQueue(Penalty, Node, /*NewLine=*/false);
      addNextStateToQueue(Penalty, Node, /*NewLine=*/true);
    }

    if (Queue.empty())
      // We were unable to find a solution, do nothing.
      // FIXME: Add diagnostic?
      return;

    // Reconstruct the solution.
    reconstructPath(InitialState, Queue.top().second);
    DEBUG(llvm::dbgs() << "Total number of analyzed states: " << Count << "\n");
    DEBUG(llvm::dbgs() << "---\n");
  }

  void reconstructPath(LineState &State, StateNode *Current) {
    std::deque<StateNode *> Path;
    // We do not need a break before the initial token.
    while (Current->Previous) {
      Path.push_front(Current);
      Current = Current->Previous;
    }
    for (std::deque<StateNode *>::iterator I = Path.begin(), E = Path.end();
         I != E; ++I) {
      DEBUG({
        if ((*I)->NewLine) {
          llvm::dbgs() << "Penalty for splitting before "
                       << (*I)->Previous->State.NextToken->Tok.getName() << ": "
                       << (*I)->Previous->State.NextToken->SplitPenalty << "\n";
        }
      });
      addTokenToState((*I)->NewLine, false, State);
    }
  }

  /// \brief Add the following state to the analysis queue \c Queue.
  ///
  /// Assume the current state is \p PreviousNode and has been reached with a
  /// penalty of \p Penalty. Insert a line break if \p NewLine is \c true.
  void addNextStateToQueue(unsigned Penalty, StateNode *PreviousNode,
                           bool NewLine) {
    if (NewLine && !canBreak(PreviousNode->State))
      return;
    if (!NewLine && mustBreak(PreviousNode->State))
      return;
    if (NewLine)
      Penalty += PreviousNode->State.NextToken->SplitPenalty;

    StateNode *Node = new (Allocator.Allocate())
        StateNode(PreviousNode->State, NewLine, PreviousNode);
    Penalty += addTokenToState(NewLine, true, Node->State);
    if (Node->State.Column > getColumnLimit()) {
      unsigned ExcessCharacters = Node->State.Column - getColumnLimit();
      Penalty += Style.PenaltyExcessCharacter * ExcessCharacters;
    }

    Queue.push(QueueItem(OrderedPenalty(Penalty, Count), Node));
    ++Count;
  }

  /// \brief Returns \c true, if a line break after \p State is allowed.
  bool canBreak(const LineState &State) {
    const FormatToken &Current = *State.NextToken;
    const FormatToken &Previous = *Current.Previous;
    assert(&Previous == Current.Previous);
    if (!Current.CanBreakBefore &&
        !(Current.is(tok::r_brace) &&
          State.Stack.back().BreakBeforeClosingBrace))
      return false;
    // The opening "{" of a braced list has to be on the same line as the first
    // element if it is nested in another braced init list or function call.
    if (!Current.MustBreakBefore && Previous.is(tok::l_brace) &&
        Previous.Previous &&
        Previous.Previous->isOneOf(tok::l_brace, tok::l_paren, tok::comma))
      return false;
    // This prevents breaks like:
    //   ...
    //   SomeParameter, OtherParameter).DoSomething(
    //   ...
    // As they hide "DoSomething" and are generally bad for readability.
    if (Previous.opensScope() && State.LowestCallLevel < State.StartOfLineLevel)
      return false;
    return !State.Stack.back().NoLineBreak;
  }

  /// \brief Returns \c true, if a line break after \p State is mandatory.
  bool mustBreak(const LineState &State) {
    const FormatToken &Current = *State.NextToken;
    const FormatToken &Previous = *Current.Previous;
    if (Current.MustBreakBefore || Current.Type == TT_InlineASMColon)
      return true;
    if (Current.is(tok::r_brace) && State.Stack.back().BreakBeforeClosingBrace)
      return true;
    if (Previous.is(tok::semi) && State.LineContainsContinuedForLoopSection)
      return true;
    if ((Previous.isOneOf(tok::comma, tok::semi) || Current.is(tok::question) ||
         Current.Type == TT_ConditionalExpr) &&
        State.Stack.back().BreakBeforeParameter &&
        !Current.isTrailingComment() &&
        !Current.isOneOf(tok::r_paren, tok::r_brace))
      return true;

    // If we need to break somewhere inside the LHS of a binary expression, we
    // should also break after the operator. Otherwise, the formatting would
    // hide the operator precedence, e.g. in:
    //   if (aaaaaaaaaaaaaa ==
    //           bbbbbbbbbbbbbb && c) {..
    // For comparisons, we only apply this rule, if the LHS is a binary
    // expression itself as otherwise, the line breaks seem superfluous.
    // We need special cases for ">>" which we have split into two ">" while
    // lexing in order to make template parsing easier.
    bool IsComparison = (Previous.getPrecedence() == prec::Relational ||
                         Previous.getPrecedence() == prec::Equality) &&
                        Previous.Previous &&
                        Previous.Previous->Type != TT_BinaryOperator; // For >>.
    bool LHSIsBinaryExpr =
        Previous.Previous && Previous.Previous->FakeRParens > 0;
    if (Previous.Type == TT_BinaryOperator &&
        (!IsComparison || LHSIsBinaryExpr) &&
        Current.Type != TT_BinaryOperator && // For >>.
        !Current.isTrailingComment() &&
        !Previous.isOneOf(tok::lessless, tok::question) &&
        Previous.getPrecedence() != prec::Assignment &&
        State.Stack.back().BreakBeforeParameter)
      return true;

    // FIXME: Comparing LongestObjCSelectorName to 0 is a hacky way of finding
    // out whether it is the first parameter. Clean this up.
    if (Current.Type == TT_ObjCSelectorName &&
        Current.LongestObjCSelectorName == 0 &&
        State.Stack.back().BreakBeforeParameter)
      return true;
    if ((Current.Type == TT_CtorInitializerColon ||
         (Previous.ClosesTemplateDeclaration && State.ParenLevel == 0)))
      return true;

    if (Current.Type == TT_StartOfName && Line.MightBeFunctionDecl &&
        State.Stack.back().BreakBeforeParameter && State.ParenLevel == 0)
      return true;
    return false;
  }

  // Returns the total number of columns required for the remaining tokens.
  unsigned getRemainingLength(const LineState &State) {
    if (State.NextToken && State.NextToken->Previous)
      return Line.Last->TotalLength - State.NextToken->Previous->TotalLength;
    return 0;
  }

  FormatStyle Style;
  SourceManager &SourceMgr;
  const AnnotatedLine &Line;
  const unsigned FirstIndent;
  const FormatToken *RootToken;
  WhitespaceManager &Whitespaces;

  llvm::SpecificBumpPtrAllocator<StateNode> Allocator;
  QueueType Queue;
  // Increasing count of \c StateNode items we have created. This is used
  // to create a deterministic order independent of the container.
  unsigned Count;
  encoding::Encoding Encoding;
};

class FormatTokenLexer {
public:
  FormatTokenLexer(Lexer &Lex, SourceManager &SourceMgr,
                   encoding::Encoding Encoding)
      : FormatTok(NULL), GreaterStashed(false), TrailingWhitespace(0), Lex(Lex),
        SourceMgr(SourceMgr), IdentTable(Lex.getLangOpts()),
        Encoding(Encoding) {
    Lex.SetKeepWhitespaceMode(true);
  }

  ArrayRef<FormatToken *> lex() {
    assert(Tokens.empty());
    do {
      Tokens.push_back(getNextToken());
    } while (Tokens.back()->Tok.isNot(tok::eof));
    return Tokens;
  }

  IdentifierTable &getIdentTable() { return IdentTable; }

private:
  FormatToken *getNextToken() {
    if (GreaterStashed) {
      // Create a synthesized second '>' token.
      Token Greater = FormatTok->Tok;
      FormatTok = new (Allocator.Allocate()) FormatToken;
      FormatTok->Tok = Greater;
      SourceLocation GreaterLocation =
          FormatTok->Tok.getLocation().getLocWithOffset(1);
      FormatTok->WhitespaceRange =
          SourceRange(GreaterLocation, GreaterLocation);
      FormatTok->TokenText = ">";
      FormatTok->CodePointCount = 1;
      GreaterStashed = false;
      return FormatTok;
    }

    FormatTok = new (Allocator.Allocate()) FormatToken;
    Lex.LexFromRawLexer(FormatTok->Tok);
    StringRef Text = rawTokenText(FormatTok->Tok);
    SourceLocation WhitespaceStart =
        FormatTok->Tok.getLocation().getLocWithOffset(-TrailingWhitespace);
    if (SourceMgr.getFileOffset(WhitespaceStart) == 0)
      FormatTok->IsFirst = true;

    // Consume and record whitespace until we find a significant token.
    unsigned WhitespaceLength = TrailingWhitespace;
    while (FormatTok->Tok.is(tok::unknown)) {
      unsigned Newlines = Text.count('\n');
      if (Newlines > 0)
        FormatTok->LastNewlineOffset = WhitespaceLength + Text.rfind('\n') + 1;
      FormatTok->NewlinesBefore += Newlines;
      unsigned EscapedNewlines = Text.count("\\\n");
      FormatTok->HasUnescapedNewline |= EscapedNewlines != Newlines;
      WhitespaceLength += FormatTok->Tok.getLength();

      Lex.LexFromRawLexer(FormatTok->Tok);
      Text = rawTokenText(FormatTok->Tok);
    }

    // In case the token starts with escaped newlines, we want to
    // take them into account as whitespace - this pattern is quite frequent
    // in macro definitions.
    // FIXME: What do we want to do with other escaped spaces, and escaped
    // spaces or newlines in the middle of tokens?
    // FIXME: Add a more explicit test.
    while (Text.size() > 1 && Text[0] == '\\' && Text[1] == '\n') {
      // FIXME: ++FormatTok->NewlinesBefore is missing...
      WhitespaceLength += 2;
      Text = Text.substr(2);
    }

    TrailingWhitespace = 0;
    if (FormatTok->Tok.is(tok::comment)) {
      StringRef UntrimmedText = Text;
      Text = Text.rtrim();
      TrailingWhitespace = UntrimmedText.size() - Text.size();
    } else if (FormatTok->Tok.is(tok::raw_identifier)) {
      IdentifierInfo &Info = IdentTable.get(Text);
      FormatTok->Tok.setIdentifierInfo(&Info);
      FormatTok->Tok.setKind(Info.getTokenID());
    } else if (FormatTok->Tok.is(tok::greatergreater)) {
      FormatTok->Tok.setKind(tok::greater);
      Text = Text.substr(0, 1);
      GreaterStashed = true;
    }

    // Now FormatTok is the next non-whitespace token.
    FormatTok->TokenText = Text;
    FormatTok->CodePointCount = encoding::getCodePointCount(Text, Encoding);

    FormatTok->WhitespaceRange = SourceRange(
        WhitespaceStart, WhitespaceStart.getLocWithOffset(WhitespaceLength));
    return FormatTok;
  }

  FormatToken *FormatTok;
  bool GreaterStashed;
  unsigned TrailingWhitespace;
  Lexer &Lex;
  SourceManager &SourceMgr;
  IdentifierTable IdentTable;
  encoding::Encoding Encoding;
  llvm::SpecificBumpPtrAllocator<FormatToken> Allocator;
  SmallVector<FormatToken *, 16> Tokens;

  /// Returns the text of \c FormatTok.
  StringRef rawTokenText(Token &Tok) {
    return StringRef(SourceMgr.getCharacterData(Tok.getLocation()),
                     Tok.getLength());
  }
};

class Formatter : public UnwrappedLineConsumer {
public:
  Formatter(const FormatStyle &Style, Lexer &Lex, SourceManager &SourceMgr,
            const std::vector<CharSourceRange> &Ranges)
      : Style(Style), Lex(Lex), SourceMgr(SourceMgr),
        Whitespaces(SourceMgr, Style), Ranges(Ranges),
        Encoding(encoding::detectEncoding(Lex.getBuffer())) {
    DEBUG(llvm::dbgs()
          << "File encoding: "
          << (Encoding == encoding::Encoding_UTF8 ? "UTF8" : "unknown")
          << "\n");
  }

  virtual ~Formatter() {}

  tooling::Replacements format() {
    FormatTokenLexer Tokens(Lex, SourceMgr, Encoding);

    UnwrappedLineParser Parser(Style, Tokens.lex(), *this);
    bool StructuralError = Parser.parse();
    TokenAnnotator Annotator(Style, Tokens.getIdentTable().get("in"));
    for (unsigned i = 0, e = AnnotatedLines.size(); i != e; ++i) {
      Annotator.annotate(AnnotatedLines[i]);
    }
    deriveLocalStyle();
    for (unsigned i = 0, e = AnnotatedLines.size(); i != e; ++i) {
      Annotator.calculateFormattingInformation(AnnotatedLines[i]);
    }

    // Adapt level to the next line if this is a comment.
    // FIXME: Can/should this be done in the UnwrappedLineParser?
    const AnnotatedLine *NextNoneCommentLine = NULL;
    for (unsigned i = AnnotatedLines.size() - 1; i > 0; --i) {
      if (NextNoneCommentLine && AnnotatedLines[i].First->is(tok::comment) &&
          !AnnotatedLines[i].First->Next)
        AnnotatedLines[i].Level = NextNoneCommentLine->Level;
      else
        NextNoneCommentLine =
            AnnotatedLines[i].First->isNot(tok::r_brace) ? &AnnotatedLines[i]
                                                         : NULL;
    }

    std::vector<int> IndentForLevel;
    bool PreviousLineWasTouched = false;
    const FormatToken *PreviousLineLastToken = 0;
    bool FormatPPDirective = false;
    for (std::vector<AnnotatedLine>::iterator I = AnnotatedLines.begin(),
                                              E = AnnotatedLines.end();
         I != E; ++I) {
      const AnnotatedLine &TheLine = *I;
      const FormatToken *FirstTok = TheLine.First;
      int Offset = getIndentOffset(*TheLine.First);

      // Check whether this line is part of a formatted preprocessor directive.
      if (FirstTok->HasUnescapedNewline)
        FormatPPDirective = false;
      if (!FormatPPDirective && TheLine.InPPDirective &&
          (touchesLine(TheLine) || touchesPPDirective(I + 1, E)))
        FormatPPDirective = true;

      // Determine indent and try to merge multiple unwrapped lines.
      while (IndentForLevel.size() <= TheLine.Level)
        IndentForLevel.push_back(-1);
      IndentForLevel.resize(TheLine.Level + 1);
      unsigned Indent = getIndent(IndentForLevel, TheLine.Level);
      if (static_cast<int>(Indent) + Offset >= 0)
        Indent += Offset;
      tryFitMultipleLinesInOne(Indent, I, E);

      bool WasMoved = PreviousLineWasTouched && FirstTok->NewlinesBefore == 0;
      if (TheLine.First->is(tok::eof)) {
        if (PreviousLineWasTouched) {
          unsigned NewLines = std::min(FirstTok->NewlinesBefore, 1u);
          Whitespaces.replaceWhitespace(*TheLine.First, NewLines, /*Indent*/ 0,
                                        /*TargetColumn*/ 0);
        }
      } else if (TheLine.Type != LT_Invalid &&
                 (WasMoved || FormatPPDirective || touchesLine(TheLine))) {
        unsigned LevelIndent = getIndent(IndentForLevel, TheLine.Level);
        if (FirstTok->WhitespaceRange.isValid() &&
            // Insert a break even if there is a structural error in case where
            // we break apart a line consisting of multiple unwrapped lines.
            (FirstTok->NewlinesBefore == 0 || !StructuralError)) {
          formatFirstToken(*TheLine.First, PreviousLineLastToken, Indent,
                           TheLine.InPPDirective);
        } else {
          Indent = LevelIndent =
              SourceMgr.getSpellingColumnNumber(FirstTok->Tok.getLocation()) -
              1;
        }
        UnwrappedLineFormatter Formatter(Style, SourceMgr, TheLine, Indent,
                                         TheLine.First, Whitespaces, Encoding);
        Formatter.format(I + 1 != E ? &*(I + 1) : NULL);
        IndentForLevel[TheLine.Level] = LevelIndent;
        PreviousLineWasTouched = true;
      } else {
        // Format the first token if necessary, and notify the WhitespaceManager
        // about the unchanged whitespace.
        for (const FormatToken *Tok = TheLine.First; Tok != NULL;
             Tok = Tok->Next) {
          if (Tok == TheLine.First &&
              (Tok->NewlinesBefore > 0 || Tok->IsFirst)) {
            unsigned LevelIndent =
                SourceMgr.getSpellingColumnNumber(Tok->Tok.getLocation()) - 1;
            // Remove trailing whitespace of the previous line if it was
            // touched.
            if (PreviousLineWasTouched || touchesEmptyLineBefore(TheLine)) {
              formatFirstToken(*Tok, PreviousLineLastToken, LevelIndent,
                               TheLine.InPPDirective);
            } else {
              Whitespaces.addUntouchableToken(*Tok, TheLine.InPPDirective);
            }

            if (static_cast<int>(LevelIndent) - Offset >= 0)
              LevelIndent -= Offset;
            if (Tok->isNot(tok::comment))
              IndentForLevel[TheLine.Level] = LevelIndent;
          } else {
            Whitespaces.addUntouchableToken(*Tok, TheLine.InPPDirective);
          }
        }
        // If we did not reformat this unwrapped line, the column at the end of
        // the last token is unchanged - thus, we can calculate the end of the
        // last token.
        PreviousLineWasTouched = false;
      }
      PreviousLineLastToken = I->Last;
    }
    return Whitespaces.generateReplacements();
  }

private:
  void deriveLocalStyle() {
    unsigned CountBoundToVariable = 0;
    unsigned CountBoundToType = 0;
    bool HasCpp03IncompatibleFormat = false;
    for (unsigned i = 0, e = AnnotatedLines.size(); i != e; ++i) {
      if (!AnnotatedLines[i].First->Next)
        continue;
      FormatToken *Tok = AnnotatedLines[i].First->Next;
      while (Tok->Next) {
        if (Tok->Type == TT_PointerOrReference) {
          bool SpacesBefore =
              Tok->WhitespaceRange.getBegin() != Tok->WhitespaceRange.getEnd();
          bool SpacesAfter = Tok->Next->WhitespaceRange.getBegin() !=
                             Tok->Next->WhitespaceRange.getEnd();
          if (SpacesBefore && !SpacesAfter)
            ++CountBoundToVariable;
          else if (!SpacesBefore && SpacesAfter)
            ++CountBoundToType;
        }

        if (Tok->Type == TT_TemplateCloser &&
            Tok->Previous->Type == TT_TemplateCloser &&
            Tok->WhitespaceRange.getBegin() == Tok->WhitespaceRange.getEnd())
          HasCpp03IncompatibleFormat = true;
        Tok = Tok->Next;
      }
    }
    if (Style.DerivePointerBinding) {
      if (CountBoundToType > CountBoundToVariable)
        Style.PointerBindsToType = true;
      else if (CountBoundToType < CountBoundToVariable)
        Style.PointerBindsToType = false;
    }
    if (Style.Standard == FormatStyle::LS_Auto) {
      Style.Standard = HasCpp03IncompatibleFormat ? FormatStyle::LS_Cpp11
                                                  : FormatStyle::LS_Cpp03;
    }
  }

  /// \brief Get the indent of \p Level from \p IndentForLevel.
  ///
  /// \p IndentForLevel must contain the indent for the level \c l
  /// at \p IndentForLevel[l], or a value < 0 if the indent for
  /// that level is unknown.
  unsigned getIndent(const std::vector<int> IndentForLevel, unsigned Level) {
    if (IndentForLevel[Level] != -1)
      return IndentForLevel[Level];
    if (Level == 0)
      return 0;
    return getIndent(IndentForLevel, Level - 1) + Style.IndentWidth;
  }

  /// \brief Get the offset of the line relatively to the level.
  ///
  /// For example, 'public:' labels in classes are offset by 1 or 2
  /// characters to the left from their level.
  int getIndentOffset(const FormatToken &RootToken) {
    if (RootToken.isAccessSpecifier(false) || RootToken.isObjCAccessSpecifier())
      return Style.AccessModifierOffset;
    return 0;
  }

  /// \brief Tries to merge lines into one.
  ///
  /// This will change \c Line and \c AnnotatedLine to contain the merged line,
  /// if possible; note that \c I will be incremented when lines are merged.
  void tryFitMultipleLinesInOne(unsigned Indent,
                                std::vector<AnnotatedLine>::iterator &I,
                                std::vector<AnnotatedLine>::iterator E) {
    // We can never merge stuff if there are trailing line comments.
    if (I->Last->Type == TT_LineComment)
      return;

    unsigned Limit = Style.ColumnLimit - Indent;
    // If we already exceed the column limit, we set 'Limit' to 0. The different
    // tryMerge..() functions can then decide whether to still do merging.
    Limit = I->Last->TotalLength > Limit ? 0 : Limit - I->Last->TotalLength;

    if (I + 1 == E || (I + 1)->Type == LT_Invalid)
      return;

    if (I->Last->is(tok::l_brace)) {
      tryMergeSimpleBlock(I, E, Limit);
    } else if (Style.AllowShortIfStatementsOnASingleLine &&
               I->First->is(tok::kw_if)) {
      tryMergeSimpleControlStatement(I, E, Limit);
    } else if (Style.AllowShortLoopsOnASingleLine &&
               I->First->isOneOf(tok::kw_for, tok::kw_while)) {
      tryMergeSimpleControlStatement(I, E, Limit);
    } else if (I->InPPDirective &&
               (I->First->HasUnescapedNewline || I->First->IsFirst)) {
      tryMergeSimplePPDirective(I, E, Limit);
    }
  }

  void tryMergeSimplePPDirective(std::vector<AnnotatedLine>::iterator &I,
                                 std::vector<AnnotatedLine>::iterator E,
                                 unsigned Limit) {
    if (Limit == 0)
      return;
    AnnotatedLine &Line = *I;
    if (!(I + 1)->InPPDirective || (I + 1)->First->HasUnescapedNewline)
      return;
    if (I + 2 != E && (I + 2)->InPPDirective &&
        !(I + 2)->First->HasUnescapedNewline)
      return;
    if (1 + (I + 1)->Last->TotalLength > Limit)
      return;
    join(Line, *(++I));
  }

  void tryMergeSimpleControlStatement(std::vector<AnnotatedLine>::iterator &I,
                                      std::vector<AnnotatedLine>::iterator E,
                                      unsigned Limit) {
    if (Limit == 0)
      return;
    if ((I + 1)->InPPDirective != I->InPPDirective ||
        ((I + 1)->InPPDirective && (I + 1)->First->HasUnescapedNewline))
      return;
    AnnotatedLine &Line = *I;
    if (Line.Last->isNot(tok::r_paren))
      return;
    if (1 + (I + 1)->Last->TotalLength > Limit)
      return;
    if ((I + 1)->First->isOneOf(tok::semi, tok::kw_if, tok::kw_for,
                                tok::kw_while) ||
        (I + 1)->First->Type == TT_LineComment)
      return;
    // Only inline simple if's (no nested if or else).
    if (I + 2 != E && Line.First->is(tok::kw_if) &&
        (I + 2)->First->is(tok::kw_else))
      return;
    join(Line, *(++I));
  }

  void tryMergeSimpleBlock(std::vector<AnnotatedLine>::iterator &I,
                           std::vector<AnnotatedLine>::iterator E,
                           unsigned Limit) {
    // No merging if the brace already is on the next line.
    if (Style.BreakBeforeBraces != FormatStyle::BS_Attach)
      return;

    // First, check that the current line allows merging. This is the case if
    // we're not in a control flow statement and the last token is an opening
    // brace.
    AnnotatedLine &Line = *I;
    if (Line.First->isOneOf(tok::kw_if, tok::kw_while, tok::kw_do, tok::r_brace,
                            tok::kw_else, tok::kw_try, tok::kw_catch,
                            tok::kw_for,
                            // This gets rid of all ObjC @ keywords and methods.
                            tok::at, tok::minus, tok::plus))
      return;

    FormatToken *Tok = (I + 1)->First;
    if (Tok->is(tok::r_brace) && !Tok->MustBreakBefore &&
        (Tok->getNextNoneComment() == NULL ||
         Tok->getNextNoneComment()->is(tok::semi))) {
      // We merge empty blocks even if the line exceeds the column limit.
      Tok->SpacesRequiredBefore = 0;
      Tok->CanBreakBefore = true;
      join(Line, *(I + 1));
      I += 1;
    } else if (Limit != 0 && Line.First->isNot(tok::kw_namespace)) {
      // Check that we still have three lines and they fit into the limit.
      if (I + 2 == E || (I + 2)->Type == LT_Invalid ||
          !nextTwoLinesFitInto(I, Limit))
        return;

      // Second, check that the next line does not contain any braces - if it
      // does, readability declines when putting it into a single line.
      if ((I + 1)->Last->Type == TT_LineComment || Tok->MustBreakBefore)
        return;
      do {
        if (Tok->isOneOf(tok::l_brace, tok::r_brace))
          return;
        Tok = Tok->Next;
      } while (Tok != NULL);

      // Last, check that the third line contains a single closing brace.
      Tok = (I + 2)->First;
      if (Tok->getNextNoneComment() != NULL || Tok->isNot(tok::r_brace) ||
          Tok->MustBreakBefore)
        return;

      join(Line, *(I + 1));
      join(Line, *(I + 2));
      I += 2;
    }
  }

  bool nextTwoLinesFitInto(std::vector<AnnotatedLine>::iterator I,
                           unsigned Limit) {
    return 1 + (I + 1)->Last->TotalLength + 1 + (I + 2)->Last->TotalLength <=
           Limit;
  }

  void join(AnnotatedLine &A, const AnnotatedLine &B) {
    assert(!A.Last->Next);
    assert(!B.First->Previous);
    A.Last->Next = B.First;
    B.First->Previous = A.Last;
    unsigned LengthA = A.Last->TotalLength + B.First->SpacesRequiredBefore;
    for (FormatToken *Tok = B.First; Tok; Tok = Tok->Next) {
      Tok->TotalLength += LengthA;
      A.Last = Tok;
    }
  }

  bool touchesRanges(const CharSourceRange &Range) {
    for (unsigned i = 0, e = Ranges.size(); i != e; ++i) {
      if (!SourceMgr.isBeforeInTranslationUnit(Range.getEnd(),
                                               Ranges[i].getBegin()) &&
          !SourceMgr.isBeforeInTranslationUnit(Ranges[i].getEnd(),
                                               Range.getBegin()))
        return true;
    }
    return false;
  }

  bool touchesLine(const AnnotatedLine &TheLine) {
    const FormatToken *First = TheLine.First;
    const FormatToken *Last = TheLine.Last;
    CharSourceRange LineRange = CharSourceRange::getCharRange(
        First->WhitespaceRange.getBegin().getLocWithOffset(
            First->LastNewlineOffset),
        Last->Tok.getLocation().getLocWithOffset(Last->TokenText.size() - 1));
    return touchesRanges(LineRange);
  }

  bool touchesPPDirective(std::vector<AnnotatedLine>::iterator I,
                          std::vector<AnnotatedLine>::iterator E) {
    for (; I != E; ++I) {
      if (I->First->HasUnescapedNewline)
        return false;
      if (touchesLine(*I))
        return true;
    }
    return false;
  }

  bool touchesEmptyLineBefore(const AnnotatedLine &TheLine) {
    const FormatToken *First = TheLine.First;
    CharSourceRange LineRange = CharSourceRange::getCharRange(
        First->WhitespaceRange.getBegin(),
        First->WhitespaceRange.getBegin().getLocWithOffset(
            First->LastNewlineOffset));
    return touchesRanges(LineRange);
  }

  virtual void consumeUnwrappedLine(const UnwrappedLine &TheLine) {
    AnnotatedLines.push_back(AnnotatedLine(TheLine));
  }

  /// \brief Add a new line and the required indent before the first Token
  /// of the \c UnwrappedLine if there was no structural parsing error.
  /// Returns the indent level of the \c UnwrappedLine.
  void formatFirstToken(const FormatToken &RootToken,
                        const FormatToken *PreviousToken, unsigned Indent,
                        bool InPPDirective) {
    unsigned Newlines =
        std::min(RootToken.NewlinesBefore, Style.MaxEmptyLinesToKeep + 1);
    // Remove empty lines before "}" where applicable.
    if (RootToken.is(tok::r_brace) &&
        (!RootToken.Next ||
         (RootToken.Next->is(tok::semi) && !RootToken.Next->Next)))
      Newlines = std::min(Newlines, 1u);
    if (Newlines == 0 && !RootToken.IsFirst)
      Newlines = 1;

    // Insert extra new line before access specifiers.
    if (PreviousToken && PreviousToken->isOneOf(tok::semi, tok::r_brace) &&
        RootToken.isAccessSpecifier() && RootToken.NewlinesBefore == 1)
      ++Newlines;

    Whitespaces.replaceWhitespace(
        RootToken, Newlines, Indent, Indent,
        InPPDirective && !RootToken.HasUnescapedNewline);
  }

  FormatStyle Style;
  Lexer &Lex;
  SourceManager &SourceMgr;
  WhitespaceManager Whitespaces;
  std::vector<CharSourceRange> Ranges;
  std::vector<AnnotatedLine> AnnotatedLines;

  encoding::Encoding Encoding;
};

} // end anonymous namespace

tooling::Replacements reformat(const FormatStyle &Style, Lexer &Lex,
                               SourceManager &SourceMgr,
                               std::vector<CharSourceRange> Ranges) {
  Formatter formatter(Style, Lex, SourceMgr, Ranges);
  return formatter.format();
}

tooling::Replacements reformat(const FormatStyle &Style, StringRef Code,
                               std::vector<tooling::Range> Ranges,
                               StringRef FileName) {
  FileManager Files((FileSystemOptions()));
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs),
      new DiagnosticOptions);
  SourceManager SourceMgr(Diagnostics, Files);
  llvm::MemoryBuffer *Buf = llvm::MemoryBuffer::getMemBuffer(Code, FileName);
  const clang::FileEntry *Entry =
      Files.getVirtualFile(FileName, Buf->getBufferSize(), 0);
  SourceMgr.overrideFileContents(Entry, Buf);
  FileID ID =
      SourceMgr.createFileID(Entry, SourceLocation(), clang::SrcMgr::C_User);
  Lexer Lex(ID, SourceMgr.getBuffer(ID), SourceMgr,
            getFormattingLangOpts(Style.Standard));
  SourceLocation StartOfFile = SourceMgr.getLocForStartOfFile(ID);
  std::vector<CharSourceRange> CharRanges;
  for (unsigned i = 0, e = Ranges.size(); i != e; ++i) {
    SourceLocation Start = StartOfFile.getLocWithOffset(Ranges[i].getOffset());
    SourceLocation End = Start.getLocWithOffset(Ranges[i].getLength());
    CharRanges.push_back(CharSourceRange::getCharRange(Start, End));
  }
  return reformat(Style, Lex, SourceMgr, CharRanges);
}

LangOptions getFormattingLangOpts(FormatStyle::LanguageStandard Standard) {
  LangOptions LangOpts;
  LangOpts.CPlusPlus = 1;
  LangOpts.CPlusPlus11 = Standard == FormatStyle::LS_Cpp03 ? 0 : 1;
  LangOpts.LineComment = 1;
  LangOpts.Bool = 1;
  LangOpts.ObjC1 = 1;
  LangOpts.ObjC2 = 1;
  return LangOpts;
}

} // namespace format
} // namespace clang
