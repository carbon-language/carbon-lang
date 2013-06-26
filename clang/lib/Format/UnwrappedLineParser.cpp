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
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "format-parser"

#include "UnwrappedLineParser.h"
#include "llvm/Support/Debug.h"

namespace clang {
namespace format {

class FormatTokenSource {
public:
  virtual ~FormatTokenSource() {}
  virtual FormatToken *getNextToken() = 0;

  virtual unsigned getPosition() = 0;
  virtual FormatToken *setPosition(unsigned Position) = 0;
};

class ScopedDeclarationState {
public:
  ScopedDeclarationState(UnwrappedLine &Line, std::vector<bool> &Stack,
                         bool MustBeDeclaration)
      : Line(Line), Stack(Stack) {
    Line.MustBeDeclaration = MustBeDeclaration;
    Stack.push_back(MustBeDeclaration);
  }
  ~ScopedDeclarationState() {
    Stack.pop_back();
    if (!Stack.empty())
      Line.MustBeDeclaration = Stack.back();
    else
      Line.MustBeDeclaration = true;
  }

private:
  UnwrappedLine &Line;
  std::vector<bool> &Stack;
};

class ScopedMacroState : public FormatTokenSource {
public:
  ScopedMacroState(UnwrappedLine &Line, FormatTokenSource *&TokenSource,
                   FormatToken *&ResetToken, bool &StructuralError)
      : Line(Line), TokenSource(TokenSource), ResetToken(ResetToken),
        PreviousLineLevel(Line.Level), PreviousTokenSource(TokenSource),
        StructuralError(StructuralError),
        PreviousStructuralError(StructuralError), Token(NULL) {
    TokenSource = this;
    Line.Level = 0;
    Line.InPPDirective = true;
  }

  ~ScopedMacroState() {
    TokenSource = PreviousTokenSource;
    ResetToken = Token;
    Line.InPPDirective = false;
    Line.Level = PreviousLineLevel;
    StructuralError = PreviousStructuralError;
  }

  virtual FormatToken *getNextToken() {
    // The \c UnwrappedLineParser guards against this by never calling
    // \c getNextToken() after it has encountered the first eof token.
    assert(!eof());
    Token = PreviousTokenSource->getNextToken();
    if (eof())
      return getFakeEOF();
    return Token;
  }

  virtual unsigned getPosition() { return PreviousTokenSource->getPosition(); }

  virtual FormatToken *setPosition(unsigned Position) {
    Token = PreviousTokenSource->setPosition(Position);
    return Token;
  }

private:
  bool eof() { return Token && Token->HasUnescapedNewline; }

  FormatToken *getFakeEOF() {
    static bool EOFInitialized = false;
    static FormatToken FormatTok;
    if (!EOFInitialized) {
      FormatTok.Tok.startToken();
      FormatTok.Tok.setKind(tok::eof);
      EOFInitialized = true;
    }
    return &FormatTok;
  }

  UnwrappedLine &Line;
  FormatTokenSource *&TokenSource;
  FormatToken *&ResetToken;
  unsigned PreviousLineLevel;
  FormatTokenSource *PreviousTokenSource;
  bool &StructuralError;
  bool PreviousStructuralError;

  FormatToken *Token;
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

class IndexedTokenSource : public FormatTokenSource {
public:
  IndexedTokenSource(ArrayRef<FormatToken *> Tokens)
      : Tokens(Tokens), Position(-1) {}

  virtual FormatToken *getNextToken() {
    ++Position;
    return Tokens[Position];
  }

  virtual unsigned getPosition() {
    assert(Position >= 0);
    return Position;
  }

  virtual FormatToken *setPosition(unsigned P) {
    Position = P;
    return Tokens[Position];
  }

private:
  ArrayRef<FormatToken *> Tokens;
  int Position;
};

UnwrappedLineParser::UnwrappedLineParser(const FormatStyle &Style,
                                         ArrayRef<FormatToken *> Tokens,
                                         UnwrappedLineConsumer &Callback)
    : Line(new UnwrappedLine), MustBreakBeforeNextToken(false),
      CurrentLines(&Lines), StructuralError(false), Style(Style), Tokens(NULL),
      Callback(Callback), AllTokens(Tokens) {
  LBraces.resize(Tokens.size(), BS_Unknown);
}

bool UnwrappedLineParser::parse() {
  DEBUG(llvm::dbgs() << "----\n");
  IndexedTokenSource TokenSource(AllTokens);
  Tokens = &TokenSource;
  readToken();
  parseFile();
  for (std::vector<UnwrappedLine>::iterator I = Lines.begin(), E = Lines.end();
       I != E; ++I) {
    Callback.consumeUnwrappedLine(*I);
  }

  // Create line with eof token.
  pushToken(FormatTok);
  Callback.consumeUnwrappedLine(*Line);
  return StructuralError;
}

void UnwrappedLineParser::parseFile() {
  ScopedDeclarationState DeclarationState(
      *Line, DeclarationScopeStack,
      /*MustBeDeclaration=*/ !Line->InPPDirective);
  parseLevel(/*HasOpeningBrace=*/false);
  // Make sure to format the remaining tokens.
  flushComments(true);
  addUnwrappedLine();
}

void UnwrappedLineParser::parseLevel(bool HasOpeningBrace) {
  do {
    switch (FormatTok->Tok.getKind()) {
    case tok::comment:
      nextToken();
      addUnwrappedLine();
      break;
    case tok::l_brace:
      // FIXME: Add parameter whether this can happen - if this happens, we must
      // be in a non-declaration context.
      parseBlock(/*MustBeDeclaration=*/false);
      addUnwrappedLine();
      break;
    case tok::r_brace:
      if (HasOpeningBrace)
        return;
      StructuralError = true;
      nextToken();
      addUnwrappedLine();
      break;
    default:
      parseStructuralElement();
      break;
    }
  } while (!eof());
}

void UnwrappedLineParser::calculateBraceTypes() {
  // We'll parse forward through the tokens until we hit
  // a closing brace or eof - note that getNextToken() will
  // parse macros, so this will magically work inside macro
  // definitions, too.
  unsigned StoredPosition = Tokens->getPosition();
  unsigned Position = StoredPosition;
  FormatToken *Tok = FormatTok;
  // Keep a stack of positions of lbrace tokens. We will
  // update information about whether an lbrace starts a
  // braced init list or a different block during the loop.
  SmallVector<unsigned, 8> LBraceStack;
  assert(Tok->Tok.is(tok::l_brace));
  do {
    FormatToken *NextTok = Tokens->getNextToken();
    switch (Tok->Tok.getKind()) {
    case tok::l_brace:
      LBraceStack.push_back(Position);
      break;
    case tok::r_brace:
      if (!LBraceStack.empty()) {
        if (LBraces[LBraceStack.back()] == BS_Unknown) {
          // If there is a comma, semicolon or right paren after the closing
          // brace, we assume this is a braced initializer list.

          // FIXME: Note that this currently works only because we do not
          // use the brace information while inside a braced init list.
          // Thus, if the parent is a braced init list, we consider all
          // brace blocks inside it braced init list. That works good enough
          // for now, but we will need to fix it to correctly handle lambdas.
          if (NextTok->isOneOf(tok::comma, tok::semi, tok::r_paren,
                               tok::l_brace, tok::colon))
            LBraces[LBraceStack.back()] = BS_BracedInit;
          else
            LBraces[LBraceStack.back()] = BS_Block;
        }
        LBraceStack.pop_back();
      }
      break;
    case tok::semi:
    case tok::kw_if:
    case tok::kw_while:
    case tok::kw_for:
    case tok::kw_switch:
    case tok::kw_try:
      if (!LBraceStack.empty())
        LBraces[LBraceStack.back()] = BS_Block;
      break;
    default:
      break;
    }
    Tok = NextTok;
    ++Position;
  } while (Tok->Tok.isNot(tok::eof));
  // Assume other blocks for all unclosed opening braces.
  for (unsigned i = 0, e = LBraceStack.size(); i != e; ++i) {
    if (LBraces[LBraceStack[i]] == BS_Unknown)
      LBraces[LBraceStack[i]] = BS_Block;
  }
  FormatTok = Tokens->setPosition(StoredPosition);
}

void UnwrappedLineParser::parseBlock(bool MustBeDeclaration,
                                     unsigned AddLevels) {
  assert(FormatTok->Tok.is(tok::l_brace) && "'{' expected");
  nextToken();

  addUnwrappedLine();

  ScopedDeclarationState DeclarationState(*Line, DeclarationScopeStack,
                                          MustBeDeclaration);
  Line->Level += AddLevels;
  parseLevel(/*HasOpeningBrace=*/true);

  if (!FormatTok->Tok.is(tok::r_brace)) {
    Line->Level -= AddLevels;
    StructuralError = true;
    return;
  }

  nextToken(); // Munch the closing brace.
  Line->Level -= AddLevels;
}

void UnwrappedLineParser::parsePPDirective() {
  assert(FormatTok->Tok.is(tok::hash) && "'#' expected");
  ScopedMacroState MacroState(*Line, Tokens, FormatTok, StructuralError);
  nextToken();

  if (FormatTok->Tok.getIdentifierInfo() == NULL) {
    parsePPUnknown();
    return;
  }

  switch (FormatTok->Tok.getIdentifierInfo()->getPPKeywordID()) {
  case tok::pp_define:
    parsePPDefine();
    return;
  case tok::pp_if:
    parsePPIf();
    break;
  case tok::pp_ifdef:
  case tok::pp_ifndef:
    parsePPIfdef();
    break;
  case tok::pp_else:
    parsePPElse();
    break;
  case tok::pp_elif:
    parsePPElIf();
    break;
  case tok::pp_endif:
    parsePPEndIf();
    break;
  default:
    parsePPUnknown();
    break;
  }
}

void UnwrappedLineParser::pushPPConditional() {
  if (!PPStack.empty() && PPStack.back() == PP_Unreachable)
    PPStack.push_back(PP_Unreachable);
  else
    PPStack.push_back(PP_Conditional);
}

void UnwrappedLineParser::parsePPIf() {
  nextToken();
  if ((FormatTok->Tok.isLiteral() &&
       StringRef(FormatTok->Tok.getLiteralData(), FormatTok->Tok.getLength()) ==
           "0") ||
      FormatTok->Tok.is(tok::kw_false)) {
    PPStack.push_back(PP_Unreachable);
  } else {
    pushPPConditional();
  }
  parsePPUnknown();
}

void UnwrappedLineParser::parsePPIfdef() {
  pushPPConditional();
  parsePPUnknown();
}

void UnwrappedLineParser::parsePPElse() {
  if (!PPStack.empty())
    PPStack.pop_back();
  pushPPConditional();
  parsePPUnknown();
}

void UnwrappedLineParser::parsePPElIf() { parsePPElse(); }

void UnwrappedLineParser::parsePPEndIf() {
  if (!PPStack.empty())
    PPStack.pop_back();
  parsePPUnknown();
}

void UnwrappedLineParser::parsePPDefine() {
  nextToken();

  if (FormatTok->Tok.getKind() != tok::identifier) {
    parsePPUnknown();
    return;
  }
  nextToken();
  if (FormatTok->Tok.getKind() == tok::l_paren &&
      FormatTok->WhitespaceRange.getBegin() ==
          FormatTok->WhitespaceRange.getEnd()) {
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

// Here we blacklist certain tokens that are not usually the first token in an
// unwrapped line. This is used in attempt to distinguish macro calls without
// trailing semicolons from other constructs split to several lines.
bool tokenCanStartNewLine(clang::Token Tok) {
  // Semicolon can be a null-statement, l_square can be a start of a macro or
  // a C++11 attribute, but this doesn't seem to be common.
  return Tok.isNot(tok::semi) && Tok.isNot(tok::l_brace) &&
         Tok.isNot(tok::l_square) &&
         // Tokens that can only be used as binary operators and a part of
         // overloaded operator names.
         Tok.isNot(tok::period) && Tok.isNot(tok::periodstar) &&
         Tok.isNot(tok::arrow) && Tok.isNot(tok::arrowstar) &&
         Tok.isNot(tok::less) && Tok.isNot(tok::greater) &&
         Tok.isNot(tok::slash) && Tok.isNot(tok::percent) &&
         Tok.isNot(tok::lessless) && Tok.isNot(tok::greatergreater) &&
         Tok.isNot(tok::equal) && Tok.isNot(tok::plusequal) &&
         Tok.isNot(tok::minusequal) && Tok.isNot(tok::starequal) &&
         Tok.isNot(tok::slashequal) && Tok.isNot(tok::percentequal) &&
         Tok.isNot(tok::ampequal) && Tok.isNot(tok::pipeequal) &&
         Tok.isNot(tok::caretequal) && Tok.isNot(tok::greatergreaterequal) &&
         Tok.isNot(tok::lesslessequal) &&
         // Colon is used in labels, base class lists, initializer lists,
         // range-based for loops, ternary operator, but should never be the
         // first token in an unwrapped line.
         Tok.isNot(tok::colon);
}

void UnwrappedLineParser::parseStructuralElement() {
  assert(!FormatTok->Tok.is(tok::l_brace));
  switch (FormatTok->Tok.getKind()) {
  case tok::at:
    nextToken();
    if (FormatTok->Tok.is(tok::l_brace)) {
      parseBracedList();
      break;
    }
    switch (FormatTok->Tok.getObjCKeywordID()) {
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
    if (FormatTok->Tok.is(tok::kw_namespace)) {
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
  case tok::kw_extern:
    nextToken();
    if (FormatTok->Tok.is(tok::string_literal)) {
      nextToken();
      if (FormatTok->Tok.is(tok::l_brace)) {
        parseBlock(/*MustBeDeclaration=*/true, 0);
        addUnwrappedLine();
        return;
      }
    }
    // In all other cases, parse the declaration.
    break;
  default:
    break;
  }
  do {
    switch (FormatTok->Tok.getKind()) {
    case tok::at:
      nextToken();
      if (FormatTok->Tok.is(tok::l_brace))
        parseBracedList();
      break;
    case tok::kw_enum:
      parseEnum();
      break;
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
      if (!tryToParseBracedList()) {
        // A block outside of parentheses must be the last part of a
        // structural element.
        // FIXME: Figure out cases where this is not true, and add projections
        // for them (the one we know is missing are lambdas).
        if (Style.BreakBeforeBraces == FormatStyle::BS_Linux ||
            Style.BreakBeforeBraces == FormatStyle::BS_Stroustrup)
          addUnwrappedLine();
        parseBlock(/*MustBeDeclaration=*/false);
        addUnwrappedLine();
        return;
      }
      // Otherwise this was a braced init list, and the structural
      // element continues.
      break;
    case tok::identifier: {
      StringRef Text = FormatTok->TokenText;
      nextToken();
      if (Line->Tokens.size() == 1) {
        if (FormatTok->Tok.is(tok::colon)) {
          parseLabel();
          return;
        }
        // Recognize function-like macro usages without trailing semicolon.
        if (FormatTok->Tok.is(tok::l_paren)) {
          parseParens();
          if (FormatTok->HasUnescapedNewline &&
              tokenCanStartNewLine(FormatTok->Tok)) {
            addUnwrappedLine();
            return;
          }
        } else if (FormatTok->HasUnescapedNewline && Text.size() >= 5 &&
                   Text == Text.upper()) {
          // Recognize free-standing macros like Q_OBJECT.
          addUnwrappedLine();
          return;
        }
      }
      break;
    }
    case tok::equal:
      nextToken();
      if (FormatTok->Tok.is(tok::l_brace)) {
        parseBracedList();
      }
      break;
    default:
      nextToken();
      break;
    }
  } while (!eof());
}

bool UnwrappedLineParser::tryToParseBracedList() {
  if (LBraces[Tokens->getPosition()] == BS_Unknown)
    calculateBraceTypes();
  assert(LBraces[Tokens->getPosition()] != BS_Unknown);
  if (LBraces[Tokens->getPosition()] == BS_Block)
    return false;
  parseBracedList();
  return true;
}

void UnwrappedLineParser::parseBracedList() {
  nextToken();

  // FIXME: Once we have an expression parser in the UnwrappedLineParser,
  // replace this by using parseAssigmentExpression() inside.
  do {
    // FIXME: When we start to support lambdas, we'll want to parse them away
    // here, otherwise our bail-out scenarios below break. The better solution
    // might be to just implement a more or less complete expression parser.
    switch (FormatTok->Tok.getKind()) {
    case tok::l_brace:
      parseBracedList();
      break;
    case tok::r_brace:
      nextToken();
      return;
    case tok::semi:
      // Probably a missing closing brace. Bail out.
      return;
    case tok::comma:
      nextToken();
      break;
    default:
      nextToken();
      break;
    }
  } while (!eof());
}

void UnwrappedLineParser::parseReturn() {
  nextToken();

  do {
    switch (FormatTok->Tok.getKind()) {
    case tok::l_brace:
      parseBracedList();
      if (FormatTok->Tok.isNot(tok::semi)) {
        // Assume missing ';'.
        addUnwrappedLine();
        return;
      }
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
  assert(FormatTok->Tok.is(tok::l_paren) && "'(' expected.");
  nextToken();
  do {
    switch (FormatTok->Tok.getKind()) {
    case tok::l_paren:
      parseParens();
      break;
    case tok::r_paren:
      nextToken();
      return;
    case tok::r_brace:
      // A "}" inside parenthesis is an error if there wasn't a matching "{".
      return;
    case tok::l_brace: {
      if (!tryToParseBracedList()) {
        nextToken();
        {
          ScopedLineState LineState(*this);
          ScopedDeclarationState DeclarationState(*Line, DeclarationScopeStack,
                                                  /*MustBeDeclaration=*/false);
          Line->Level += 1;
          parseLevel(/*HasOpeningBrace=*/true);
          Line->Level -= 1;
        }
        nextToken();
      }
      break;
    }
    case tok::at:
      nextToken();
      if (FormatTok->Tok.is(tok::l_brace))
        parseBracedList();
      break;
    default:
      nextToken();
      break;
    }
  } while (!eof());
}

void UnwrappedLineParser::parseIfThenElse() {
  assert(FormatTok->Tok.is(tok::kw_if) && "'if' expected");
  nextToken();
  if (FormatTok->Tok.is(tok::l_paren))
    parseParens();
  bool NeedsUnwrappedLine = false;
  if (FormatTok->Tok.is(tok::l_brace)) {
    parseBlock(/*MustBeDeclaration=*/false);
    NeedsUnwrappedLine = true;
  } else {
    addUnwrappedLine();
    ++Line->Level;
    parseStructuralElement();
    --Line->Level;
  }
  if (FormatTok->Tok.is(tok::kw_else)) {
    nextToken();
    if (FormatTok->Tok.is(tok::l_brace)) {
      parseBlock(/*MustBeDeclaration=*/false);
      addUnwrappedLine();
    } else if (FormatTok->Tok.is(tok::kw_if)) {
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
  assert(FormatTok->Tok.is(tok::kw_namespace) && "'namespace' expected");
  nextToken();
  if (FormatTok->Tok.is(tok::identifier))
    nextToken();
  if (FormatTok->Tok.is(tok::l_brace)) {
    if (Style.BreakBeforeBraces == FormatStyle::BS_Linux)
      addUnwrappedLine();

    parseBlock(/*MustBeDeclaration=*/true, 0);
    // Munch the semicolon after a namespace. This is more common than one would
    // think. Puttin the semicolon into its own line is very ugly.
    if (FormatTok->Tok.is(tok::semi))
      nextToken();
    addUnwrappedLine();
  }
  // FIXME: Add error handling.
}

void UnwrappedLineParser::parseForOrWhileLoop() {
  assert((FormatTok->Tok.is(tok::kw_for) || FormatTok->Tok.is(tok::kw_while)) &&
         "'for' or 'while' expected");
  nextToken();
  if (FormatTok->Tok.is(tok::l_paren))
    parseParens();
  if (FormatTok->Tok.is(tok::l_brace)) {
    parseBlock(/*MustBeDeclaration=*/false);
    addUnwrappedLine();
  } else {
    addUnwrappedLine();
    ++Line->Level;
    parseStructuralElement();
    --Line->Level;
  }
}

void UnwrappedLineParser::parseDoWhile() {
  assert(FormatTok->Tok.is(tok::kw_do) && "'do' expected");
  nextToken();
  if (FormatTok->Tok.is(tok::l_brace)) {
    parseBlock(/*MustBeDeclaration=*/false);
  } else {
    addUnwrappedLine();
    ++Line->Level;
    parseStructuralElement();
    --Line->Level;
  }

  // FIXME: Add error handling.
  if (!FormatTok->Tok.is(tok::kw_while)) {
    addUnwrappedLine();
    return;
  }

  nextToken();
  parseStructuralElement();
}

void UnwrappedLineParser::parseLabel() {
  if (FormatTok->Tok.isNot(tok::colon))
    return;
  nextToken();
  unsigned OldLineLevel = Line->Level;
  if (Line->Level > 1 || (!Line->InPPDirective && Line->Level > 0))
    --Line->Level;
  if (CommentsBeforeNextToken.empty() && FormatTok->Tok.is(tok::l_brace)) {
    parseBlock(/*MustBeDeclaration=*/false);
    if (FormatTok->Tok.is(tok::kw_break))
      parseStructuralElement(); // "break;" after "}" goes on the same line.
  }
  addUnwrappedLine();
  Line->Level = OldLineLevel;
}

void UnwrappedLineParser::parseCaseLabel() {
  assert(FormatTok->Tok.is(tok::kw_case) && "'case' expected");
  // FIXME: fix handling of complex expressions here.
  do {
    nextToken();
  } while (!eof() && !FormatTok->Tok.is(tok::colon));
  parseLabel();
}

void UnwrappedLineParser::parseSwitch() {
  assert(FormatTok->Tok.is(tok::kw_switch) && "'switch' expected");
  nextToken();
  if (FormatTok->Tok.is(tok::l_paren))
    parseParens();
  if (FormatTok->Tok.is(tok::l_brace)) {
    parseBlock(/*MustBeDeclaration=*/false, Style.IndentCaseLabels ? 2 : 1);
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
  if (FormatTok->Tok.is(tok::colon))
    nextToken();
  addUnwrappedLine();
}

void UnwrappedLineParser::parseEnum() {
  nextToken();
  if (FormatTok->Tok.is(tok::identifier) ||
      FormatTok->Tok.is(tok::kw___attribute) ||
      FormatTok->Tok.is(tok::kw___declspec)) {
    nextToken();
    // We can have macros or attributes in between 'enum' and the enum name.
    if (FormatTok->Tok.is(tok::l_paren)) {
      parseParens();
    }
    if (FormatTok->Tok.is(tok::identifier))
      nextToken();
  }
  if (FormatTok->Tok.is(tok::l_brace)) {
    nextToken();
    addUnwrappedLine();
    ++Line->Level;
    do {
      switch (FormatTok->Tok.getKind()) {
      case tok::l_paren:
        parseParens();
        break;
      case tok::r_brace:
        addUnwrappedLine();
        nextToken();
        --Line->Level;
        return;
      case tok::comma:
        nextToken();
        addUnwrappedLine();
        break;
      default:
        nextToken();
        break;
      }
    } while (!eof());
  }
  // We fall through to parsing a structural element afterwards, so that in
  // enum A {} n, m;
  // "} n, m;" will end up in one unwrapped line.
}

void UnwrappedLineParser::parseRecord() {
  nextToken();
  if (FormatTok->Tok.is(tok::identifier) ||
      FormatTok->Tok.is(tok::kw___attribute) ||
      FormatTok->Tok.is(tok::kw___declspec)) {
    nextToken();
    // We can have macros or attributes in between 'class' and the class name.
    if (FormatTok->Tok.is(tok::l_paren)) {
      parseParens();
    }
    // The actual identifier can be a nested name specifier, and in macros
    // it is often token-pasted.
    while (FormatTok->Tok.is(tok::identifier) ||
           FormatTok->Tok.is(tok::coloncolon) ||
           FormatTok->Tok.is(tok::hashhash))
      nextToken();

    // Note that parsing away template declarations here leads to incorrectly
    // accepting function declarations as record declarations.
    // In general, we cannot solve this problem. Consider:
    // class A<int> B() {}
    // which can be a function definition or a class definition when B() is a
    // macro. If we find enough real-world cases where this is a problem, we
    // can parse for the 'template' keyword in the beginning of the statement,
    // and thus rule out the record production in case there is no template
    // (this would still leave us with an ambiguity between template function
    // and class declarations).
    if (FormatTok->Tok.is(tok::colon) || FormatTok->Tok.is(tok::less)) {
      while (!eof() && FormatTok->Tok.isNot(tok::l_brace)) {
        if (FormatTok->Tok.is(tok::semi))
          return;
        nextToken();
      }
    }
  }
  if (FormatTok->Tok.is(tok::l_brace)) {
    if (Style.BreakBeforeBraces == FormatStyle::BS_Linux)
      addUnwrappedLine();

    parseBlock(/*MustBeDeclaration=*/true);
  }
  // We fall through to parsing a structural element afterwards, so
  // class A {} n, m;
  // will end up in one unwrapped line.
}

void UnwrappedLineParser::parseObjCProtocolList() {
  assert(FormatTok->Tok.is(tok::less) && "'<' expected.");
  do
    nextToken();
  while (!eof() && FormatTok->Tok.isNot(tok::greater));
  nextToken(); // Skip '>'.
}

void UnwrappedLineParser::parseObjCUntilAtEnd() {
  do {
    if (FormatTok->Tok.isObjCAtKeyword(tok::objc_end)) {
      nextToken();
      addUnwrappedLine();
      break;
    }
    parseStructuralElement();
  } while (!eof());
}

void UnwrappedLineParser::parseObjCInterfaceOrImplementation() {
  nextToken();
  nextToken(); // interface name

  // @interface can be followed by either a base class, or a category.
  if (FormatTok->Tok.is(tok::colon)) {
    nextToken();
    nextToken(); // base class name
  } else if (FormatTok->Tok.is(tok::l_paren))
    // Skip category, if present.
    parseParens();

  if (FormatTok->Tok.is(tok::less))
    parseObjCProtocolList();

  // If instance variables are present, keep the '{' on the first line too.
  if (FormatTok->Tok.is(tok::l_brace))
    parseBlock(/*MustBeDeclaration=*/true);

  // With instance variables, this puts '}' on its own line.  Without instance
  // variables, this ends the @interface line.
  addUnwrappedLine();

  parseObjCUntilAtEnd();
}

void UnwrappedLineParser::parseObjCProtocol() {
  nextToken();
  nextToken(); // protocol name

  if (FormatTok->Tok.is(tok::less))
    parseObjCProtocolList();

  // Check for protocol declaration.
  if (FormatTok->Tok.is(tok::semi)) {
    nextToken();
    return addUnwrappedLine();
  }

  addUnwrappedLine();
  parseObjCUntilAtEnd();
}

void UnwrappedLineParser::addUnwrappedLine() {
  if (Line->Tokens.empty())
    return;
  DEBUG({
    llvm::dbgs() << "Line(" << Line->Level << ")"
                 << (Line->InPPDirective ? " MACRO" : "") << ": ";
    for (std::list<FormatToken *>::iterator I = Line->Tokens.begin(),
                                            E = Line->Tokens.end();
         I != E; ++I) {
      llvm::dbgs() << (*I)->Tok.getName() << " ";
    }
    llvm::dbgs() << "\n";
  });
  CurrentLines->push_back(*Line);
  Line->Tokens.clear();
  if (CurrentLines == &Lines && !PreprocessorDirectives.empty()) {
    for (std::vector<UnwrappedLine>::iterator
             I = PreprocessorDirectives.begin(),
             E = PreprocessorDirectives.end();
         I != E; ++I) {
      CurrentLines->push_back(*I);
    }
    PreprocessorDirectives.clear();
  }
}

bool UnwrappedLineParser::eof() const { return FormatTok->Tok.is(tok::eof); }

void UnwrappedLineParser::flushComments(bool NewlineBeforeNext) {
  bool JustComments = Line->Tokens.empty();
  for (SmallVectorImpl<FormatToken *>::const_iterator
           I = CommentsBeforeNextToken.begin(),
           E = CommentsBeforeNextToken.end();
       I != E; ++I) {
    if ((*I)->NewlinesBefore && JustComments) {
      addUnwrappedLine();
    }
    pushToken(*I);
  }
  if (NewlineBeforeNext && JustComments) {
    addUnwrappedLine();
  }
  CommentsBeforeNextToken.clear();
}

void UnwrappedLineParser::nextToken() {
  if (eof())
    return;
  flushComments(FormatTok->NewlinesBefore > 0);
  pushToken(FormatTok);
  readToken();
}

void UnwrappedLineParser::readToken() {
  bool CommentsInCurrentLine = true;
  do {
    FormatTok = Tokens->getNextToken();
    while (!Line->InPPDirective && FormatTok->Tok.is(tok::hash) &&
           (FormatTok->HasUnescapedNewline || FormatTok->IsFirst)) {
      // If there is an unfinished unwrapped line, we flush the preprocessor
      // directives only after that unwrapped line was finished later.
      bool SwitchToPreprocessorLines =
          !Line->Tokens.empty() && CurrentLines == &Lines;
      ScopedLineState BlockState(*this, SwitchToPreprocessorLines);
      // Comments stored before the preprocessor directive need to be output
      // before the preprocessor directive, at the same level as the
      // preprocessor directive, as we consider them to apply to the directive.
      flushComments(FormatTok->NewlinesBefore > 0);
      parsePPDirective();
    }

    if (!PPStack.empty() && (PPStack.back() == PP_Unreachable) &&
        !Line->InPPDirective) {
      continue;
    }

    if (!FormatTok->Tok.is(tok::comment))
      return;
    if (FormatTok->NewlinesBefore > 0 || FormatTok->IsFirst) {
      CommentsInCurrentLine = false;
    }
    if (CommentsInCurrentLine) {
      pushToken(FormatTok);
    } else {
      CommentsBeforeNextToken.push_back(FormatTok);
    }
  } while (!eof());
}

void UnwrappedLineParser::pushToken(FormatToken *Tok) {
  Line->Tokens.push_back(Tok);
  if (MustBreakBeforeNextToken) {
    Line->Tokens.back()->MustBreakBefore = true;
    MustBreakBeforeNextToken = false;
  }
}

} // end namespace format
} // end namespace clang
