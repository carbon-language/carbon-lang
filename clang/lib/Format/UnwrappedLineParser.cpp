//===--- UnwrappedLineParser.cpp - Format C++ code ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains the implementation of the UnwrappedLineParser,
/// which turns a stream of tokens into UnwrappedLines.
///
/// This is EXPERIMENTAL code under heavy development. It is not in a state yet,
/// where it can be used to format real code.
///
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "format-parser"

#include "UnwrappedLineParser.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/Support/Debug.h"

// Uncomment to get debug output from tests:
// #define DEBUG_WITH_TYPE(T, X) do { X; } while(0)

namespace clang {
namespace format {

class ScopedMacroState : public FormatTokenSource {
public:
  ScopedMacroState(UnwrappedLine &Line, FormatTokenSource *&TokenSource,
                   FormatToken &ResetToken)
      : Line(Line), TokenSource(TokenSource), ResetToken(ResetToken),
        PreviousLineLevel(Line.Level), PreviousTokenSource(TokenSource) {
    TokenSource = this;
    Line.Level = 0;
    Line.InPPDirective = true;
  }

  ~ScopedMacroState() {
    TokenSource = PreviousTokenSource;
    ResetToken = Token;
    Line.InPPDirective = false;
    Line.Level = PreviousLineLevel;
  }

  virtual FormatToken getNextToken() {
    // The \c UnwrappedLineParser guards against this by never calling
    // \c getNextToken() after it has encountered the first eof token.
    assert(!eof());
    Token = PreviousTokenSource->getNextToken();
    if (eof())
      return createEOF();
    return Token;
  }

private:
  bool eof() {
    return Token.NewlinesBefore > 0 && Token.HasUnescapedNewline;
  }

  FormatToken createEOF() {
    FormatToken FormatTok;
    FormatTok.Tok.startToken();
    FormatTok.Tok.setKind(tok::eof);
    return FormatTok;
  }

  UnwrappedLine &Line;
  FormatTokenSource *&TokenSource;
  FormatToken &ResetToken;
  unsigned PreviousLineLevel;
  FormatTokenSource *PreviousTokenSource;

  FormatToken Token;
};

class ScopedLineState {
public:
  ScopedLineState(UnwrappedLineParser &Parser,
                  bool SwitchToPreprocessorLines = false)
      : Parser(Parser), SwitchToPreprocessorLines(SwitchToPreprocessorLines) {
    if (SwitchToPreprocessorLines)
      Parser.CurrentLines = &Parser.PreprocessorDirectives;
    PreBlockLine = Parser.Line.take();
    Parser.Line.reset(new UnwrappedLine());
    Parser.Line->Level = PreBlockLine->Level;
    Parser.Line->InPPDirective = PreBlockLine->InPPDirective;
  }

  ~ScopedLineState() {
    if (!Parser.Line->Tokens.empty()) {
      Parser.addUnwrappedLine();
    }
    assert(Parser.Line->Tokens.empty());
    Parser.Line.reset(PreBlockLine);
    Parser.MustBreakBeforeNextToken = true;
    if (SwitchToPreprocessorLines)
      Parser.CurrentLines = &Parser.Lines;
  }

private:
  UnwrappedLineParser &Parser;
  const bool SwitchToPreprocessorLines;

  UnwrappedLine *PreBlockLine;
};

UnwrappedLineParser::UnwrappedLineParser(
    clang::DiagnosticsEngine &Diag, const FormatStyle &Style,
    FormatTokenSource &Tokens, UnwrappedLineConsumer &Callback)
    : Line(new UnwrappedLine), MustBreakBeforeNextToken(false),
      CurrentLines(&Lines), Diag(Diag), Style(Style), Tokens(&Tokens),
      Callback(Callback) {}

bool UnwrappedLineParser::parse() {
  DEBUG(llvm::dbgs() << "----\n");
  readToken();
  bool Error = parseFile();
  for (std::vector<UnwrappedLine>::iterator I = Lines.begin(),
                                            E = Lines.end();
       I != E; ++I) {
    Callback.consumeUnwrappedLine(*I);
  }
  return Error;
}

bool UnwrappedLineParser::parseFile() {
  bool Error = parseLevel(/*HasOpeningBrace=*/false);
  // Make sure to format the remaining tokens.
  addUnwrappedLine();
  return Error;
}

bool UnwrappedLineParser::parseLevel(bool HasOpeningBrace) {
  bool Error = false;
  do {
    switch (FormatTok.Tok.getKind()) {
    case tok::comment:
      nextToken();
      addUnwrappedLine();
      break;
    case tok::l_brace:
      Error |= parseBlock();
      addUnwrappedLine();
      break;
    case tok::r_brace:
      if (HasOpeningBrace) {
        return false;
      } else {
        Diag.Report(FormatTok.Tok.getLocation(),
                    Diag.getCustomDiagID(clang::DiagnosticsEngine::Error,
                                         "unexpected '}'"));
        Error = true;
        nextToken();
        addUnwrappedLine();
      }
      break;
    default:
      parseStructuralElement();
      break;
    }
  } while (!eof());
  return Error;
}

bool UnwrappedLineParser::parseBlock(unsigned AddLevels) {
  assert(FormatTok.Tok.is(tok::l_brace) && "'{' expected");
  nextToken();

  if (!FormatTok.Tok.is(tok::r_brace)) {
    addUnwrappedLine();

    Line->Level += AddLevels;
    parseLevel(/*HasOpeningBrace=*/true);
    Line->Level -= AddLevels;

    if (!FormatTok.Tok.is(tok::r_brace))
      return true;

  }
  nextToken();  // Munch the closing brace.
  return false;
}

void UnwrappedLineParser::parsePPDirective() {
  assert(FormatTok.Tok.is(tok::hash) && "'#' expected");
  ScopedMacroState MacroState(*Line, Tokens, FormatTok);
  nextToken();

  if (FormatTok.Tok.getIdentifierInfo() == NULL) {
    addUnwrappedLine();
    return;
  }

  switch (FormatTok.Tok.getIdentifierInfo()->getPPKeywordID()) {
  case tok::pp_define:
    parsePPDefine();
    break;
  default:
    parsePPUnknown();
    break;
  }
}

void UnwrappedLineParser::parsePPDefine() {
  nextToken();

  if (FormatTok.Tok.getKind() != tok::identifier) {
    parsePPUnknown();
    return;
  }
  nextToken();
  if (FormatTok.Tok.getKind() == tok::l_paren) {
    parseParens();
  }
  addUnwrappedLine();
  Line->Level = 1;

  // Errors during a preprocessor directive can only affect the layout of the
  // preprocessor directive, and thus we ignore them. An alternative approach
  // would be to use the same approach we use on the file level (no
  // re-indentation if there was a structural error) within the macro
  // definition.
  parseFile();
}

void UnwrappedLineParser::parsePPUnknown() {
  do {
    nextToken();
  } while (!eof());
  addUnwrappedLine();
}

void UnwrappedLineParser::parseComments() {
  // Consume leading line comments, e.g. for branches without compounds.
  while (FormatTok.Tok.is(tok::comment)) {
    nextToken();
    addUnwrappedLine();
  }
}

void UnwrappedLineParser::parseStructuralElement() {
  assert(!FormatTok.Tok.is(tok::l_brace));
  parseComments();

  int TokenNumber = 0;
  switch (FormatTok.Tok.getKind()) {
  case tok::at:
    nextToken();
    switch (FormatTok.Tok.getObjCKeywordID()) {
    case tok::objc_public:
    case tok::objc_protected:
    case tok::objc_package:
    case tok::objc_private:
      return parseAccessSpecifier();
    case tok::objc_interface:
    case tok::objc_implementation:
      return parseObjCInterfaceOrImplementation();
    case tok::objc_protocol:
      return parseObjCProtocol();
    case tok::objc_end:
      return; // Handled by the caller.
    case tok::objc_optional:
    case tok::objc_required:
      nextToken();
      addUnwrappedLine();
      return;
    default:
      break;
    }
    break;
  case tok::kw_namespace:
    parseNamespace();
    return;
  case tok::kw_inline:
    nextToken();
    TokenNumber++;
    if (FormatTok.Tok.is(tok::kw_namespace)) {
      parseNamespace();
      return;
    }
    break;
  case tok::kw_public:
  case tok::kw_protected:
  case tok::kw_private:
    parseAccessSpecifier();
    return;
  case tok::kw_if:
    parseIfThenElse();
    return;
  case tok::kw_for:
  case tok::kw_while:
    parseForOrWhileLoop();
    return;
  case tok::kw_do:
    parseDoWhile();
    return;
  case tok::kw_switch:
    parseSwitch();
    return;
  case tok::kw_default:
    nextToken();
    parseLabel();
    return;
  case tok::kw_case:
    parseCaseLabel();
    return;
  case tok::kw_return:
    parseReturn();
    return;
  default:
    break;
  }
  do {
    ++TokenNumber;
    switch (FormatTok.Tok.getKind()) {
    case tok::kw_enum:
      parseEnum();
      return;
    case tok::kw_struct:
    case tok::kw_union:
    case tok::kw_class:
      parseRecord();
      // A record declaration or definition is always the start of a structural
      // element.
      break;
    case tok::semi:
      nextToken();
      addUnwrappedLine();
      return;
    case tok::r_brace:
      addUnwrappedLine();
      return;
    case tok::l_paren:
      parseParens();
      break;
    case tok::l_brace:
      // A block outside of parentheses must be the last part of a
      // structural element.
      // FIXME: Figure out cases where this is not true, and add projections for
      // them (the one we know is missing are lambdas).
      parseBlock();
      addUnwrappedLine();
      return;
    case tok::identifier:
      nextToken();
      if (TokenNumber == 1 && FormatTok.Tok.is(tok::colon)) {
        parseLabel();
        return;
      }
      break;
    case tok::equal:
      nextToken();
      if (FormatTok.Tok.is(tok::l_brace)) {
        parseBracedList();
      }
      break;
    default:
      nextToken();
      break;
    }
  } while (!eof());
}

void UnwrappedLineParser::parseBracedList() {
  nextToken();

  do {
    switch (FormatTok.Tok.getKind()) {
    case tok::l_brace:
      parseBracedList();
      break;
    case tok::r_brace:
      nextToken();
      return;
    default:
      nextToken();
      break;
    }
  } while (!eof());
}

void UnwrappedLineParser::parseReturn() {
  nextToken();

  do {
    switch (FormatTok.Tok.getKind()) {
    case tok::l_brace:
      parseBracedList();
      break;
    case tok::l_paren:
      parseParens();
      break;
    case tok::r_brace:
      // Assume missing ';'.
      addUnwrappedLine();
      return;
    case tok::semi:
      nextToken();
      addUnwrappedLine();
      return;
    default:
      nextToken();
      break;
    }
  } while (!eof());
}

void UnwrappedLineParser::parseParens() {
  assert(FormatTok.Tok.is(tok::l_paren) && "'(' expected.");
  nextToken();
  do {
    switch (FormatTok.Tok.getKind()) {
    case tok::l_paren:
      parseParens();
      break;
    case tok::r_paren:
      nextToken();
      return;
    case tok::l_brace:
      {
        nextToken();
        ScopedLineState LineState(*this);
        Line->Level += 1;
        parseLevel(/*HasOpeningBrace=*/true);
        Line->Level -= 1;
      }
      break;
    default:
      nextToken();
      break;
    }
  } while (!eof());
}

void UnwrappedLineParser::parseIfThenElse() {
  assert(FormatTok.Tok.is(tok::kw_if) && "'if' expected");
  nextToken();
  if (FormatTok.Tok.is(tok::l_paren))
    parseParens();
  bool NeedsUnwrappedLine = false;
  if (FormatTok.Tok.is(tok::l_brace)) {
    parseBlock();
    NeedsUnwrappedLine = true;
  } else {
    addUnwrappedLine();
    ++Line->Level;
    parseStructuralElement();
    --Line->Level;
  }
  if (FormatTok.Tok.is(tok::kw_else)) {
    nextToken();
    if (FormatTok.Tok.is(tok::l_brace)) {
      parseBlock();
      addUnwrappedLine();
    } else if (FormatTok.Tok.is(tok::kw_if)) {
      parseIfThenElse();
    } else {
      addUnwrappedLine();
      ++Line->Level;
      parseStructuralElement();
      --Line->Level;
    }
  } else if (NeedsUnwrappedLine) {
    addUnwrappedLine();
  }
}

void UnwrappedLineParser::parseNamespace() {
  assert(FormatTok.Tok.is(tok::kw_namespace) && "'namespace' expected");
  nextToken();
  if (FormatTok.Tok.is(tok::identifier))
    nextToken();
  if (FormatTok.Tok.is(tok::l_brace)) {
    parseBlock(0);
    addUnwrappedLine();
  }
  // FIXME: Add error handling.
}

void UnwrappedLineParser::parseForOrWhileLoop() {
  assert((FormatTok.Tok.is(tok::kw_for) || FormatTok.Tok.is(tok::kw_while)) &&
         "'for' or 'while' expected");
  nextToken();
  if (FormatTok.Tok.is(tok::l_paren))
    parseParens();
  if (FormatTok.Tok.is(tok::l_brace)) {
    parseBlock();
    addUnwrappedLine();
  } else {
    addUnwrappedLine();
    ++Line->Level;
    parseStructuralElement();
    --Line->Level;
  }
}

void UnwrappedLineParser::parseDoWhile() {
  assert(FormatTok.Tok.is(tok::kw_do) && "'do' expected");
  nextToken();
  if (FormatTok.Tok.is(tok::l_brace)) {
    parseBlock();
  } else {
    addUnwrappedLine();
    ++Line->Level;
    parseStructuralElement();
    --Line->Level;
  }

  // FIXME: Add error handling.
  if (!FormatTok.Tok.is(tok::kw_while)) {
    addUnwrappedLine();
    return;
  }

  nextToken();
  parseStructuralElement();
}

void UnwrappedLineParser::parseLabel() {
  // FIXME: remove all asserts.
  assert(FormatTok.Tok.is(tok::colon) && "':' expected");
  nextToken();
  unsigned OldLineLevel = Line->Level;
  if (Line->Level > 0)
    --Line->Level;
  if (FormatTok.Tok.is(tok::l_brace)) {
    parseBlock();
    if (FormatTok.Tok.is(tok::kw_break))
      parseStructuralElement(); // "break;" after "}" goes on the same line.
  }
  addUnwrappedLine();
  Line->Level = OldLineLevel;
}

void UnwrappedLineParser::parseCaseLabel() {
  assert(FormatTok.Tok.is(tok::kw_case) && "'case' expected");
  // FIXME: fix handling of complex expressions here.
  do {
    nextToken();
  } while (!eof() && !FormatTok.Tok.is(tok::colon));
  parseLabel();
}

void UnwrappedLineParser::parseSwitch() {
  assert(FormatTok.Tok.is(tok::kw_switch) && "'switch' expected");
  nextToken();
  if (FormatTok.Tok.is(tok::l_paren))
    parseParens();
  if (FormatTok.Tok.is(tok::l_brace)) {
    parseBlock(Style.IndentCaseLabels ? 2 : 1);
    addUnwrappedLine();
  } else {
    addUnwrappedLine();
    Line->Level += (Style.IndentCaseLabels ? 2 : 1);
    parseStructuralElement();
    Line->Level -= (Style.IndentCaseLabels ? 2 : 1);
  }
}

void UnwrappedLineParser::parseAccessSpecifier() {
  nextToken();
  // Otherwise, we don't know what it is, and we'd better keep the next token.
  if (FormatTok.Tok.is(tok::colon))
    nextToken();
  addUnwrappedLine();
}

void UnwrappedLineParser::parseEnum() {
  bool HasContents = false;
  do {
    switch (FormatTok.Tok.getKind()) {
    case tok::l_brace:
      nextToken();
      addUnwrappedLine();
      ++Line->Level;
      parseComments();
      break;
    case tok::l_paren:
      parseParens();
      break;
    case tok::comma:
      nextToken();
      addUnwrappedLine();
      parseComments();
      break;
    case tok::r_brace:
      if (HasContents)
        addUnwrappedLine();
      --Line->Level;
      nextToken();
      break;
    case tok::semi:
      nextToken();
      addUnwrappedLine();
      return;
    default:
      HasContents = true;
      nextToken();
      break;
    }
  } while (!eof());
}

void UnwrappedLineParser::parseRecord() {
  nextToken();
  if (FormatTok.Tok.is(tok::identifier) ||
      FormatTok.Tok.is(tok::kw___attribute) ||
      FormatTok.Tok.is(tok::kw___declspec)) {
    nextToken();
    // We can have macros or attributes in between 'class' and the class name.
    if (FormatTok.Tok.is(tok::l_paren)) {
      parseParens();
    }
    if (FormatTok.Tok.is(tok::identifier))
      nextToken();

    if (FormatTok.Tok.is(tok::colon)) {
      while (FormatTok.Tok.isNot(tok::l_brace)) {
        if (FormatTok.Tok.is(tok::semi))
          return;
        nextToken();
      }
    }
  }
  if (FormatTok.Tok.is(tok::l_brace))
    parseBlock();
}

void UnwrappedLineParser::parseObjCProtocolList() {
  assert(FormatTok.Tok.is(tok::less) && "'<' expected.");
  do
    nextToken();
  while (!eof() && FormatTok.Tok.isNot(tok::greater));
  nextToken(); // Skip '>'.
}

void UnwrappedLineParser::parseObjCUntilAtEnd() {
  do {
    if (FormatTok.Tok.isObjCAtKeyword(tok::objc_end)) {
      nextToken();
      addUnwrappedLine();
      break;
    }
    parseStructuralElement();
  } while (!eof());
}

void UnwrappedLineParser::parseObjCInterfaceOrImplementation() {
  nextToken();
  nextToken();  // interface name

  // @interface can be followed by either a base class, or a category.
  if (FormatTok.Tok.is(tok::colon)) {
    nextToken();
    nextToken();  // base class name
  } else if (FormatTok.Tok.is(tok::l_paren))
    // Skip category, if present.
    parseParens();

  if (FormatTok.Tok.is(tok::less))
    parseObjCProtocolList();

  // If instance variables are present, keep the '{' on the first line too.
  if (FormatTok.Tok.is(tok::l_brace))
    parseBlock();

  // With instance variables, this puts '}' on its own line.  Without instance
  // variables, this ends the @interface line.
  addUnwrappedLine();

  parseObjCUntilAtEnd();
}

void UnwrappedLineParser::parseObjCProtocol() {
  nextToken();
  nextToken();  // protocol name

  if (FormatTok.Tok.is(tok::less))
    parseObjCProtocolList();

  // Check for protocol declaration.
  if (FormatTok.Tok.is(tok::semi)) {
    nextToken();
    return addUnwrappedLine();
  }

  addUnwrappedLine();
  parseObjCUntilAtEnd();
}

void UnwrappedLineParser::addUnwrappedLine() {
  if (Line->Tokens.empty())
    return;
  // Consume trailing comments.
  while (!eof() && FormatTok.NewlinesBefore == 0 &&
         FormatTok.Tok.is(tok::comment)) {
    nextToken();
  }
  DEBUG({
    llvm::dbgs() << "Line: ";
    for (std::list<FormatToken>::iterator I = Line->Tokens.begin(),
                                          E = Line->Tokens.end();
         I != E; ++I) {
      llvm::dbgs() << I->Tok.getName() << " ";

    }
    llvm::dbgs() << "\n";
  });
  CurrentLines->push_back(*Line);
  Line->Tokens.clear();
  if (CurrentLines == &Lines && !PreprocessorDirectives.empty()) {
    for (std::vector<UnwrappedLine>::iterator I = PreprocessorDirectives
             .begin(), E = PreprocessorDirectives.end();
         I != E; ++I) {
      CurrentLines->push_back(*I);
    }
    PreprocessorDirectives.clear();
  }

}

bool UnwrappedLineParser::eof() const {
  return FormatTok.Tok.is(tok::eof);
}

void UnwrappedLineParser::nextToken() {
  if (eof())
    return;
  Line->Tokens.push_back(FormatTok);
  if (MustBreakBeforeNextToken) {
    Line->Tokens.back().MustBreakBefore = true;
    MustBreakBeforeNextToken = false;
  }
  readToken();
}

void UnwrappedLineParser::readToken() {
  FormatTok = Tokens->getNextToken();
  while (!Line->InPPDirective && FormatTok.Tok.is(tok::hash) &&
         ((FormatTok.NewlinesBefore > 0 && FormatTok.HasUnescapedNewline) ||
          FormatTok.IsFirst)) {
    // If there is an unfinished unwrapped line, we flush the preprocessor
    // directives only after that unwrapped line was finished later.
    bool SwitchToPreprocessorLines = !Line->Tokens.empty() &&
                                     CurrentLines == &Lines;
    ScopedLineState BlockState(*this, SwitchToPreprocessorLines);
    parsePPDirective();
  }
}

} // end namespace format
} // end namespace clang
