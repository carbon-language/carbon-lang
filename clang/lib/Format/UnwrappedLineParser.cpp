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

#include "UnwrappedLineParser.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "format-parser"

namespace clang {
namespace format {

class FormatTokenSource {
public:
  virtual ~FormatTokenSource() {}
  virtual FormatToken *getNextToken() = 0;

  virtual unsigned getPosition() = 0;
  virtual FormatToken *setPosition(unsigned Position) = 0;
};

namespace {

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
        PreviousStructuralError(StructuralError), Token(nullptr) {
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

  FormatToken *getNextToken() override {
    // The \c UnwrappedLineParser guards against this by never calling
    // \c getNextToken() after it has encountered the first eof token.
    assert(!eof());
    Token = PreviousTokenSource->getNextToken();
    if (eof())
      return getFakeEOF();
    return Token;
  }

  unsigned getPosition() override { return PreviousTokenSource->getPosition(); }

  FormatToken *setPosition(unsigned Position) override {
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

} // end anonymous namespace

class ScopedLineState {
public:
  ScopedLineState(UnwrappedLineParser &Parser,
                  bool SwitchToPreprocessorLines = false)
      : Parser(Parser), OriginalLines(Parser.CurrentLines) {
    if (SwitchToPreprocessorLines)
      Parser.CurrentLines = &Parser.PreprocessorDirectives;
    else if (!Parser.Line->Tokens.empty())
      Parser.CurrentLines = &Parser.Line->Tokens.back().Children;
    PreBlockLine = std::move(Parser.Line);
    Parser.Line = llvm::make_unique<UnwrappedLine>();
    Parser.Line->Level = PreBlockLine->Level;
    Parser.Line->InPPDirective = PreBlockLine->InPPDirective;
  }

  ~ScopedLineState() {
    if (!Parser.Line->Tokens.empty()) {
      Parser.addUnwrappedLine();
    }
    assert(Parser.Line->Tokens.empty());
    Parser.Line = std::move(PreBlockLine);
    if (Parser.CurrentLines == &Parser.PreprocessorDirectives)
      Parser.MustBreakBeforeNextToken = true;
    Parser.CurrentLines = OriginalLines;
  }

private:
  UnwrappedLineParser &Parser;

  std::unique_ptr<UnwrappedLine> PreBlockLine;
  SmallVectorImpl<UnwrappedLine> *OriginalLines;
};

class CompoundStatementIndenter {
public:
  CompoundStatementIndenter(UnwrappedLineParser *Parser,
                            const FormatStyle &Style, unsigned &LineLevel)
      : LineLevel(LineLevel), OldLineLevel(LineLevel) {
    if (Style.BreakBeforeBraces == FormatStyle::BS_Allman) {
      Parser->addUnwrappedLine();
    } else if (Style.BreakBeforeBraces == FormatStyle::BS_GNU) {
      Parser->addUnwrappedLine();
      ++LineLevel;
    }
  }
  ~CompoundStatementIndenter() { LineLevel = OldLineLevel; }

private:
  unsigned &LineLevel;
  unsigned OldLineLevel;
};

namespace {

class IndexedTokenSource : public FormatTokenSource {
public:
  IndexedTokenSource(ArrayRef<FormatToken *> Tokens)
      : Tokens(Tokens), Position(-1) {}

  FormatToken *getNextToken() override {
    ++Position;
    return Tokens[Position];
  }

  unsigned getPosition() override {
    assert(Position >= 0);
    return Position;
  }

  FormatToken *setPosition(unsigned P) override {
    Position = P;
    return Tokens[Position];
  }

  void reset() { Position = -1; }

private:
  ArrayRef<FormatToken *> Tokens;
  int Position;
};

} // end anonymous namespace

UnwrappedLineParser::UnwrappedLineParser(const FormatStyle &Style,
                                         const AdditionalKeywords &Keywords,
                                         ArrayRef<FormatToken *> Tokens,
                                         UnwrappedLineConsumer &Callback)
    : Line(new UnwrappedLine), MustBreakBeforeNextToken(false),
      CurrentLines(&Lines), StructuralError(false), Style(Style),
      Keywords(Keywords), Tokens(nullptr), Callback(Callback),
      AllTokens(Tokens), PPBranchLevel(-1) {}

void UnwrappedLineParser::reset() {
  PPBranchLevel = -1;
  Line.reset(new UnwrappedLine);
  CommentsBeforeNextToken.clear();
  FormatTok = nullptr;
  MustBreakBeforeNextToken = false;
  PreprocessorDirectives.clear();
  CurrentLines = &Lines;
  DeclarationScopeStack.clear();
  StructuralError = false;
  PPStack.clear();
}

bool UnwrappedLineParser::parse() {
  IndexedTokenSource TokenSource(AllTokens);
  do {
    DEBUG(llvm::dbgs() << "----\n");
    reset();
    Tokens = &TokenSource;
    TokenSource.reset();

    readToken();
    parseFile();
    // Create line with eof token.
    pushToken(FormatTok);
    addUnwrappedLine();

    for (SmallVectorImpl<UnwrappedLine>::iterator I = Lines.begin(),
                                                  E = Lines.end();
         I != E; ++I) {
      Callback.consumeUnwrappedLine(*I);
    }
    Callback.finishRun();
    Lines.clear();
    while (!PPLevelBranchIndex.empty() &&
           PPLevelBranchIndex.back() + 1 >= PPLevelBranchCount.back()) {
      PPLevelBranchIndex.resize(PPLevelBranchIndex.size() - 1);
      PPLevelBranchCount.resize(PPLevelBranchCount.size() - 1);
    }
    if (!PPLevelBranchIndex.empty()) {
      ++PPLevelBranchIndex.back();
      assert(PPLevelBranchIndex.size() == PPLevelBranchCount.size());
      assert(PPLevelBranchIndex.back() <= PPLevelBranchCount.back());
    }
  } while (!PPLevelBranchIndex.empty());

  return StructuralError;
}

void UnwrappedLineParser::parseFile() {
  ScopedDeclarationState DeclarationState(
      *Line, DeclarationScopeStack,
      /*MustBeDeclaration=*/!Line->InPPDirective);
  parseLevel(/*HasOpeningBrace=*/false);
  // Make sure to format the remaining tokens.
  flushComments(true);
  addUnwrappedLine();
}

void UnwrappedLineParser::parseLevel(bool HasOpeningBrace) {
  bool SwitchLabelEncountered = false;
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
    case tok::kw_default:
    case tok::kw_case:
      if (!SwitchLabelEncountered &&
          (Style.IndentCaseLabels || (Line->InPPDirective && Line->Level == 1)))
        ++Line->Level;
      SwitchLabelEncountered = true;
      parseStructuralElement();
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
  FormatToken *Tok = FormatTok;
  // Keep a stack of positions of lbrace tokens. We will
  // update information about whether an lbrace starts a
  // braced init list or a different block during the loop.
  SmallVector<FormatToken *, 8> LBraceStack;
  assert(Tok->Tok.is(tok::l_brace));
  do {
    // Get next none-comment token.
    FormatToken *NextTok;
    unsigned ReadTokens = 0;
    do {
      NextTok = Tokens->getNextToken();
      ++ReadTokens;
    } while (NextTok->is(tok::comment));

    switch (Tok->Tok.getKind()) {
    case tok::l_brace:
      LBraceStack.push_back(Tok);
      break;
    case tok::r_brace:
      if (!LBraceStack.empty()) {
        if (LBraceStack.back()->BlockKind == BK_Unknown) {
          bool ProbablyBracedList = false;
          if (Style.Language == FormatStyle::LK_Proto) {
            ProbablyBracedList = NextTok->isOneOf(tok::comma, tok::r_square);
          } else {
            // Using OriginalColumn to distinguish between ObjC methods and
            // binary operators is a bit hacky.
            bool NextIsObjCMethod = NextTok->isOneOf(tok::plus, tok::minus) &&
                                    NextTok->OriginalColumn == 0;

            // If there is a comma, semicolon or right paren after the closing
            // brace, we assume this is a braced initializer list.  Note that
            // regardless how we mark inner braces here, we will overwrite the
            // BlockKind later if we parse a braced list (where all blocks
            // inside are by default braced lists), or when we explicitly detect
            // blocks (for example while parsing lambdas).
            //
            // We exclude + and - as they can be ObjC visibility modifiers.
            ProbablyBracedList =
                NextTok->isOneOf(tok::comma, tok::semi, tok::period, tok::colon,
                                 tok::r_paren, tok::r_square, tok::l_brace,
                                 tok::l_paren, tok::ellipsis) ||
                (NextTok->isBinaryOperator() && !NextIsObjCMethod);
          }
          if (ProbablyBracedList) {
            Tok->BlockKind = BK_BracedInit;
            LBraceStack.back()->BlockKind = BK_BracedInit;
          } else {
            Tok->BlockKind = BK_Block;
            LBraceStack.back()->BlockKind = BK_Block;
          }
        }
        LBraceStack.pop_back();
      }
      break;
    case tok::at:
    case tok::semi:
    case tok::kw_if:
    case tok::kw_while:
    case tok::kw_for:
    case tok::kw_switch:
    case tok::kw_try:
    case tok::kw___try:
      if (!LBraceStack.empty())
        LBraceStack.back()->BlockKind = BK_Block;
      break;
    default:
      break;
    }
    Tok = NextTok;
  } while (Tok->Tok.isNot(tok::eof) && !LBraceStack.empty());
  // Assume other blocks for all unclosed opening braces.
  for (unsigned i = 0, e = LBraceStack.size(); i != e; ++i) {
    if (LBraceStack[i]->BlockKind == BK_Unknown)
      LBraceStack[i]->BlockKind = BK_Block;
  }

  FormatTok = Tokens->setPosition(StoredPosition);
}

void UnwrappedLineParser::parseBlock(bool MustBeDeclaration, bool AddLevel,
                                     bool MunchSemi) {
  assert(FormatTok->Tok.is(tok::l_brace) && "'{' expected");
  unsigned InitialLevel = Line->Level;
  nextToken();

  addUnwrappedLine();

  ScopedDeclarationState DeclarationState(*Line, DeclarationScopeStack,
                                          MustBeDeclaration);
  if (AddLevel)
    ++Line->Level;
  parseLevel(/*HasOpeningBrace=*/true);

  if (!FormatTok->Tok.is(tok::r_brace)) {
    Line->Level = InitialLevel;
    StructuralError = true;
    return;
  }

  nextToken(); // Munch the closing brace.
  if (MunchSemi && FormatTok->Tok.is(tok::semi))
    nextToken();
  Line->Level = InitialLevel;
}

static bool IsGoogScope(const UnwrappedLine &Line) {
  // FIXME: Closure-library specific stuff should not be hard-coded but be
  // configurable.
  if (Line.Tokens.size() < 4)
    return false;
  auto I = Line.Tokens.begin();
  if (I->Tok->TokenText != "goog")
    return false;
  ++I;
  if (I->Tok->isNot(tok::period))
    return false;
  ++I;
  if (I->Tok->TokenText != "scope")
    return false;
  ++I;
  return I->Tok->is(tok::l_paren);
}

static bool ShouldBreakBeforeBrace(const FormatStyle &Style,
                                   const FormatToken &InitialToken) {
  switch (Style.BreakBeforeBraces) {
  case FormatStyle::BS_Linux:
    return InitialToken.isOneOf(tok::kw_namespace, tok::kw_class);
  case FormatStyle::BS_Allman:
  case FormatStyle::BS_GNU:
    return true;
  default:
    return false;
  }
}

void UnwrappedLineParser::parseChildBlock() {
  FormatTok->BlockKind = BK_Block;
  nextToken();
  {
    bool GoogScope =
        Style.Language == FormatStyle::LK_JavaScript && IsGoogScope(*Line);
    ScopedLineState LineState(*this);
    ScopedDeclarationState DeclarationState(*Line, DeclarationScopeStack,
                                            /*MustBeDeclaration=*/false);
    Line->Level += GoogScope ? 0 : 1;
    parseLevel(/*HasOpeningBrace=*/true);
    Line->Level -= GoogScope ? 0 : 1;
  }
  nextToken();
}

void UnwrappedLineParser::parsePPDirective() {
  assert(FormatTok->Tok.is(tok::hash) && "'#' expected");
  ScopedMacroState MacroState(*Line, Tokens, FormatTok, StructuralError);
  nextToken();

  if (!FormatTok->Tok.getIdentifierInfo()) {
    parsePPUnknown();
    return;
  }

  switch (FormatTok->Tok.getIdentifierInfo()->getPPKeywordID()) {
  case tok::pp_define:
    parsePPDefine();
    return;
  case tok::pp_if:
    parsePPIf(/*IfDef=*/false);
    break;
  case tok::pp_ifdef:
  case tok::pp_ifndef:
    parsePPIf(/*IfDef=*/true);
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

void UnwrappedLineParser::conditionalCompilationCondition(bool Unreachable) {
  if (Unreachable || (!PPStack.empty() && PPStack.back() == PP_Unreachable))
    PPStack.push_back(PP_Unreachable);
  else
    PPStack.push_back(PP_Conditional);
}

void UnwrappedLineParser::conditionalCompilationStart(bool Unreachable) {
  ++PPBranchLevel;
  assert(PPBranchLevel >= 0 && PPBranchLevel <= (int)PPLevelBranchIndex.size());
  if (PPBranchLevel == (int)PPLevelBranchIndex.size()) {
    PPLevelBranchIndex.push_back(0);
    PPLevelBranchCount.push_back(0);
  }
  PPChainBranchIndex.push(0);
  bool Skip = PPLevelBranchIndex[PPBranchLevel] > 0;
  conditionalCompilationCondition(Unreachable || Skip);
}

void UnwrappedLineParser::conditionalCompilationAlternative() {
  if (!PPStack.empty())
    PPStack.pop_back();
  assert(PPBranchLevel < (int)PPLevelBranchIndex.size());
  if (!PPChainBranchIndex.empty())
    ++PPChainBranchIndex.top();
  conditionalCompilationCondition(
      PPBranchLevel >= 0 && !PPChainBranchIndex.empty() &&
      PPLevelBranchIndex[PPBranchLevel] != PPChainBranchIndex.top());
}

void UnwrappedLineParser::conditionalCompilationEnd() {
  assert(PPBranchLevel < (int)PPLevelBranchIndex.size());
  if (PPBranchLevel >= 0 && !PPChainBranchIndex.empty()) {
    if (PPChainBranchIndex.top() + 1 > PPLevelBranchCount[PPBranchLevel]) {
      PPLevelBranchCount[PPBranchLevel] = PPChainBranchIndex.top() + 1;
    }
  }
  // Guard against #endif's without #if.
  if (PPBranchLevel > 0)
    --PPBranchLevel;
  if (!PPChainBranchIndex.empty())
    PPChainBranchIndex.pop();
  if (!PPStack.empty())
    PPStack.pop_back();
}

void UnwrappedLineParser::parsePPIf(bool IfDef) {
  nextToken();
  bool IsLiteralFalse = (FormatTok->Tok.isLiteral() &&
                         FormatTok->Tok.getLiteralData() != nullptr &&
                         StringRef(FormatTok->Tok.getLiteralData(),
                                   FormatTok->Tok.getLength()) == "0") ||
                        FormatTok->Tok.is(tok::kw_false);
  conditionalCompilationStart(!IfDef && IsLiteralFalse);
  parsePPUnknown();
}

void UnwrappedLineParser::parsePPElse() {
  conditionalCompilationAlternative();
  parsePPUnknown();
}

void UnwrappedLineParser::parsePPElIf() { parsePPElse(); }

void UnwrappedLineParser::parsePPEndIf() {
  conditionalCompilationEnd();
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
         Tok.isNot(tok::colon) &&
         // 'noexcept' is a trailing annotation.
         Tok.isNot(tok::kw_noexcept);
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
    case tok::objc_try:
      // This branch isn't strictly necessary (the kw_try case below would
      // do this too after the tok::at is parsed above).  But be explicit.
      parseTryCatch();
      return;
    default:
      break;
    }
    break;
  case tok::kw_asm:
    nextToken();
    if (FormatTok->is(tok::l_brace)) {
      nextToken();
      while (FormatTok && FormatTok->isNot(tok::eof)) {
        if (FormatTok->is(tok::r_brace)) {
          nextToken();
          break;
        }
        FormatTok->Finalized = true;
        nextToken();
      }
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
    if (Style.Language == FormatStyle::LK_Java ||
        Style.Language == FormatStyle::LK_JavaScript)
      nextToken();
    else
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
  case tok::kw_try:
  case tok::kw___try:
    parseTryCatch();
    return;
  case tok::kw_extern:
    nextToken();
    if (FormatTok->Tok.is(tok::string_literal)) {
      nextToken();
      if (FormatTok->Tok.is(tok::l_brace)) {
        parseBlock(/*MustBeDeclaration=*/true, /*AddLevel=*/false);
        addUnwrappedLine();
        return;
      }
    }
    break;
  case tok::kw_export:
    if (Style.Language == FormatStyle::LK_JavaScript) {
      parseJavaScriptEs6ImportExport();
      return;
    }
    break;
  case tok::identifier:
    if (FormatTok->IsForEachMacro) {
      parseForOrWhileLoop();
      return;
    }
    if (Style.Language == FormatStyle::LK_JavaScript &&
        FormatTok->is(Keywords.kw_import)) {
      parseJavaScriptEs6ImportExport();
      return;
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
    case tok::kw_typedef:
      nextToken();
      if (FormatTok->isOneOf(Keywords.kw_NS_ENUM, Keywords.kw_NS_OPTIONS,
                             Keywords.kw_CF_ENUM, Keywords.kw_CF_OPTIONS))
        parseEnum();
      break;
    case tok::kw_struct:
    case tok::kw_union:
    case tok::kw_class:
      parseRecord();
      // A record declaration or definition is always the start of a structural
      // element.
      break;
    case tok::period:
      nextToken();
      // In Java, classes have an implicit static member "class".
      if (Style.Language == FormatStyle::LK_Java && FormatTok &&
          FormatTok->is(tok::kw_class))
        nextToken();
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
    case tok::caret:
      nextToken();
      if (FormatTok->Tok.isAnyIdentifier() ||
          FormatTok->isSimpleTypeSpecifier())
        nextToken();
      if (FormatTok->is(tok::l_paren))
        parseParens();
      if (FormatTok->is(tok::l_brace))
        parseChildBlock();
      break;
    case tok::l_brace:
      if (!tryToParseBracedList()) {
        // A block outside of parentheses must be the last part of a
        // structural element.
        // FIXME: Figure out cases where this is not true, and add projections
        // for them (the one we know is missing are lambdas).
        if (Style.BreakBeforeBraces != FormatStyle::BS_Attach)
          addUnwrappedLine();
        FormatTok->Type = TT_FunctionLBrace;
        parseBlock(/*MustBeDeclaration=*/false);
        addUnwrappedLine();
        return;
      }
      // Otherwise this was a braced init list, and the structural
      // element continues.
      break;
    case tok::kw_try:
      // We arrive here when parsing function-try blocks.
      parseTryCatch();
      return;
    case tok::identifier: {
      StringRef Text = FormatTok->TokenText;
      // Parse function literal unless 'function' is the first token in a line
      // in which case this should be treated as a free-standing function.
      if (Style.Language == FormatStyle::LK_JavaScript && Text == "function" &&
          Line->Tokens.size() > 0) {
        tryToParseJSFunction();
        break;
      }
      nextToken();
      if (Line->Tokens.size() == 1 &&
          // JS doesn't have macros, and within classes colons indicate fields,
          // not labels.
          (Style.Language != FormatStyle::LK_JavaScript ||
           !Line->MustBeDeclaration)) {
        if (FormatTok->Tok.is(tok::colon)) {
          parseLabel();
          return;
        }
        // Recognize function-like macro usages without trailing semicolon as
        // well as free-standing macros like Q_OBJECT.
        bool FunctionLike = FormatTok->is(tok::l_paren);
        if (FunctionLike)
          parseParens();
        if (FormatTok->NewlinesBefore > 0 &&
            (Text.size() >= 5 || FunctionLike) &&
            tokenCanStartNewLine(FormatTok->Tok) && Text == Text.upper()) {
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
    case tok::l_square:
      parseSquare();
      break;
    default:
      nextToken();
      break;
    }
  } while (!eof());
}

bool UnwrappedLineParser::tryToParseLambda() {
  // FIXME: This is a dirty way to access the previous token. Find a better
  // solution.
  if (!Line->Tokens.empty() &&
      (Line->Tokens.back().Tok->isOneOf(tok::identifier, tok::kw_operator,
                                        tok::kw_new, tok::kw_delete) ||
       Line->Tokens.back().Tok->closesScope() ||
       Line->Tokens.back().Tok->isSimpleTypeSpecifier())) {
    nextToken();
    return false;
  }
  assert(FormatTok->is(tok::l_square));
  FormatToken &LSquare = *FormatTok;
  if (!tryToParseLambdaIntroducer())
    return false;

  while (FormatTok->isNot(tok::l_brace)) {
    if (FormatTok->isSimpleTypeSpecifier()) {
      nextToken();
      continue;
    }
    switch (FormatTok->Tok.getKind()) {
    case tok::l_brace:
      break;
    case tok::l_paren:
      parseParens();
      break;
    case tok::amp:
    case tok::star:
    case tok::kw_const:
    case tok::comma:
    case tok::less:
    case tok::greater:
    case tok::identifier:
    case tok::coloncolon:
    case tok::kw_mutable:
      nextToken();
      break;
    case tok::arrow:
      FormatTok->Type = TT_TrailingReturnArrow;
      nextToken();
      break;
    default:
      return true;
    }
  }
  LSquare.Type = TT_LambdaLSquare;
  parseChildBlock();
  return true;
}

bool UnwrappedLineParser::tryToParseLambdaIntroducer() {
  nextToken();
  if (FormatTok->is(tok::equal)) {
    nextToken();
    if (FormatTok->is(tok::r_square)) {
      nextToken();
      return true;
    }
    if (FormatTok->isNot(tok::comma))
      return false;
    nextToken();
  } else if (FormatTok->is(tok::amp)) {
    nextToken();
    if (FormatTok->is(tok::r_square)) {
      nextToken();
      return true;
    }
    if (!FormatTok->isOneOf(tok::comma, tok::identifier)) {
      return false;
    }
    if (FormatTok->is(tok::comma))
      nextToken();
  } else if (FormatTok->is(tok::r_square)) {
    nextToken();
    return true;
  }
  do {
    if (FormatTok->is(tok::amp))
      nextToken();
    if (!FormatTok->isOneOf(tok::identifier, tok::kw_this))
      return false;
    nextToken();
    if (FormatTok->is(tok::ellipsis))
      nextToken();
    if (FormatTok->is(tok::comma)) {
      nextToken();
    } else if (FormatTok->is(tok::r_square)) {
      nextToken();
      return true;
    } else {
      return false;
    }
  } while (!eof());
  return false;
}

void UnwrappedLineParser::tryToParseJSFunction() {
  nextToken();

  // Consume function name.
  if (FormatTok->is(tok::identifier))
    nextToken();

  if (FormatTok->isNot(tok::l_paren))
    return;
  nextToken();
  while (FormatTok->isNot(tok::l_brace)) {
    // Err on the side of caution in order to avoid consuming the full file in
    // case of incomplete code.
    if (!FormatTok->isOneOf(tok::identifier, tok::comma, tok::r_paren,
                            tok::comment))
      return;
    nextToken();
  }
  parseChildBlock();
}

bool UnwrappedLineParser::tryToParseBracedList() {
  if (FormatTok->BlockKind == BK_Unknown)
    calculateBraceTypes();
  assert(FormatTok->BlockKind != BK_Unknown);
  if (FormatTok->BlockKind == BK_Block)
    return false;
  parseBracedList();
  return true;
}

bool UnwrappedLineParser::parseBracedList(bool ContinueOnSemicolons) {
  bool HasError = false;
  nextToken();

  // FIXME: Once we have an expression parser in the UnwrappedLineParser,
  // replace this by using parseAssigmentExpression() inside.
  do {
    if (Style.Language == FormatStyle::LK_JavaScript &&
        FormatTok->is(Keywords.kw_function)) {
      tryToParseJSFunction();
      continue;
    }
    switch (FormatTok->Tok.getKind()) {
    case tok::caret:
      nextToken();
      if (FormatTok->is(tok::l_brace)) {
        parseChildBlock();
      }
      break;
    case tok::l_square:
      tryToParseLambda();
      break;
    case tok::l_brace:
      // Assume there are no blocks inside a braced init list apart
      // from the ones we explicitly parse out (like lambdas).
      FormatTok->BlockKind = BK_BracedInit;
      parseBracedList();
      break;
    case tok::r_brace:
      nextToken();
      return !HasError;
    case tok::semi:
      HasError = true;
      if (!ContinueOnSemicolons)
        return !HasError;
      nextToken();
      break;
    case tok::comma:
      nextToken();
      break;
    default:
      nextToken();
      break;
    }
  } while (!eof());
  return false;
}

void UnwrappedLineParser::parseParens() {
  assert(FormatTok->Tok.is(tok::l_paren) && "'(' expected.");
  nextToken();
  do {
    switch (FormatTok->Tok.getKind()) {
    case tok::l_paren:
      parseParens();
      if (Style.Language == FormatStyle::LK_Java && FormatTok->is(tok::l_brace))
        parseChildBlock();
      break;
    case tok::r_paren:
      nextToken();
      return;
    case tok::r_brace:
      // A "}" inside parenthesis is an error if there wasn't a matching "{".
      return;
    case tok::l_square:
      tryToParseLambda();
      break;
    case tok::l_brace:
      if (!tryToParseBracedList()) {
        parseChildBlock();
      }
      break;
    case tok::at:
      nextToken();
      if (FormatTok->Tok.is(tok::l_brace))
        parseBracedList();
      break;
    case tok::identifier:
      if (Style.Language == FormatStyle::LK_JavaScript &&
          FormatTok->is(Keywords.kw_function))
        tryToParseJSFunction();
      else
        nextToken();
      break;
    default:
      nextToken();
      break;
    }
  } while (!eof());
}

void UnwrappedLineParser::parseSquare() {
  assert(FormatTok->Tok.is(tok::l_square) && "'[' expected.");
  if (tryToParseLambda())
    return;
  do {
    switch (FormatTok->Tok.getKind()) {
    case tok::l_paren:
      parseParens();
      break;
    case tok::r_square:
      nextToken();
      return;
    case tok::r_brace:
      // A "}" inside parenthesis is an error if there wasn't a matching "{".
      return;
    case tok::l_square:
      parseSquare();
      break;
    case tok::l_brace: {
      if (!tryToParseBracedList()) {
        parseChildBlock();
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
    CompoundStatementIndenter Indenter(this, Style, Line->Level);
    parseBlock(/*MustBeDeclaration=*/false);
    if (Style.BreakBeforeBraces == FormatStyle::BS_Allman ||
        Style.BreakBeforeBraces == FormatStyle::BS_GNU) {
      addUnwrappedLine();
    } else {
      NeedsUnwrappedLine = true;
    }
  } else {
    addUnwrappedLine();
    ++Line->Level;
    parseStructuralElement();
    --Line->Level;
  }
  if (FormatTok->Tok.is(tok::kw_else)) {
    if (Style.BreakBeforeBraces == FormatStyle::BS_Stroustrup)
      addUnwrappedLine();
    nextToken();
    if (FormatTok->Tok.is(tok::l_brace)) {
      CompoundStatementIndenter Indenter(this, Style, Line->Level);
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

void UnwrappedLineParser::parseTryCatch() {
  assert(FormatTok->isOneOf(tok::kw_try, tok::kw___try) && "'try' expected");
  nextToken();
  bool NeedsUnwrappedLine = false;
  if (FormatTok->is(tok::colon)) {
    // We are in a function try block, what comes is an initializer list.
    nextToken();
    while (FormatTok->is(tok::identifier)) {
      nextToken();
      if (FormatTok->is(tok::l_paren))
        parseParens();
      else
        StructuralError = true;
      if (FormatTok->is(tok::comma))
        nextToken();
    }
  }
  // Parse try with resource.
  if (Style.Language == FormatStyle::LK_Java && FormatTok->is(tok::l_paren)) {
    parseParens();
  }
  if (FormatTok->is(tok::l_brace)) {
    CompoundStatementIndenter Indenter(this, Style, Line->Level);
    parseBlock(/*MustBeDeclaration=*/false);
    if (Style.BreakBeforeBraces == FormatStyle::BS_Allman ||
        Style.BreakBeforeBraces == FormatStyle::BS_GNU ||
        Style.BreakBeforeBraces == FormatStyle::BS_Stroustrup) {
      addUnwrappedLine();
    } else {
      NeedsUnwrappedLine = true;
    }
  } else if (!FormatTok->is(tok::kw_catch)) {
    // The C++ standard requires a compound-statement after a try.
    // If there's none, we try to assume there's a structuralElement
    // and try to continue.
    StructuralError = true;
    addUnwrappedLine();
    ++Line->Level;
    parseStructuralElement();
    --Line->Level;
  }
  while (1) {
    if (FormatTok->is(tok::at))
      nextToken();
    if (!(FormatTok->isOneOf(tok::kw_catch, Keywords.kw___except,
                             tok::kw___finally) ||
          ((Style.Language == FormatStyle::LK_Java ||
            Style.Language == FormatStyle::LK_JavaScript) &&
           FormatTok->is(Keywords.kw_finally)) ||
          (FormatTok->Tok.isObjCAtKeyword(tok::objc_catch) ||
           FormatTok->Tok.isObjCAtKeyword(tok::objc_finally))))
      break;
    nextToken();
    while (FormatTok->isNot(tok::l_brace)) {
      if (FormatTok->is(tok::l_paren)) {
        parseParens();
        continue;
      }
      if (FormatTok->isOneOf(tok::semi, tok::r_brace, tok::eof))
        return;
      nextToken();
    }
    NeedsUnwrappedLine = false;
    CompoundStatementIndenter Indenter(this, Style, Line->Level);
    parseBlock(/*MustBeDeclaration=*/false);
    if (Style.BreakBeforeBraces == FormatStyle::BS_Allman ||
        Style.BreakBeforeBraces == FormatStyle::BS_GNU ||
        Style.BreakBeforeBraces == FormatStyle::BS_Stroustrup) {
      addUnwrappedLine();
    } else {
      NeedsUnwrappedLine = true;
    }
  }
  if (NeedsUnwrappedLine) {
    addUnwrappedLine();
  }
}

void UnwrappedLineParser::parseNamespace() {
  assert(FormatTok->Tok.is(tok::kw_namespace) && "'namespace' expected");

  const FormatToken &InitialToken = *FormatTok;
  nextToken();
  if (FormatTok->Tok.is(tok::identifier))
    nextToken();
  if (FormatTok->Tok.is(tok::l_brace)) {
    if (ShouldBreakBeforeBrace(Style, InitialToken))
      addUnwrappedLine();

    bool AddLevel = Style.NamespaceIndentation == FormatStyle::NI_All ||
                    (Style.NamespaceIndentation == FormatStyle::NI_Inner &&
                     DeclarationScopeStack.size() > 1);
    parseBlock(/*MustBeDeclaration=*/true, AddLevel);
    // Munch the semicolon after a namespace. This is more common than one would
    // think. Puttin the semicolon into its own line is very ugly.
    if (FormatTok->Tok.is(tok::semi))
      nextToken();
    addUnwrappedLine();
  }
  // FIXME: Add error handling.
}

void UnwrappedLineParser::parseForOrWhileLoop() {
  assert((FormatTok->Tok.is(tok::kw_for) || FormatTok->Tok.is(tok::kw_while) ||
          FormatTok->IsForEachMacro) &&
         "'for', 'while' or foreach macro expected");
  nextToken();
  if (FormatTok->Tok.is(tok::l_paren))
    parseParens();
  if (FormatTok->Tok.is(tok::l_brace)) {
    CompoundStatementIndenter Indenter(this, Style, Line->Level);
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
    CompoundStatementIndenter Indenter(this, Style, Line->Level);
    parseBlock(/*MustBeDeclaration=*/false);
    if (Style.BreakBeforeBraces == FormatStyle::BS_GNU)
      addUnwrappedLine();
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
  nextToken();
  unsigned OldLineLevel = Line->Level;
  if (Line->Level > 1 || (!Line->InPPDirective && Line->Level > 0))
    --Line->Level;
  if (CommentsBeforeNextToken.empty() && FormatTok->Tok.is(tok::l_brace)) {
    CompoundStatementIndenter Indenter(this, Style, Line->Level);
    parseBlock(/*MustBeDeclaration=*/false);
    if (FormatTok->Tok.is(tok::kw_break)) {
      // "break;" after "}" on its own line only for BS_Allman and BS_GNU
      if (Style.BreakBeforeBraces == FormatStyle::BS_Allman ||
          Style.BreakBeforeBraces == FormatStyle::BS_GNU) {
        addUnwrappedLine();
      }
      parseStructuralElement();
    }
    addUnwrappedLine();
  } else {
    addUnwrappedLine();
  }
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
    CompoundStatementIndenter Indenter(this, Style, Line->Level);
    parseBlock(/*MustBeDeclaration=*/false);
    addUnwrappedLine();
  } else {
    addUnwrappedLine();
    ++Line->Level;
    parseStructuralElement();
    --Line->Level;
  }
}

void UnwrappedLineParser::parseAccessSpecifier() {
  nextToken();
  // Understand Qt's slots.
  if (FormatTok->is(tok::identifier) &&
      (FormatTok->TokenText == "slots" || FormatTok->TokenText == "Q_SLOTS"))
    nextToken();
  // Otherwise, we don't know what it is, and we'd better keep the next token.
  if (FormatTok->Tok.is(tok::colon))
    nextToken();
  addUnwrappedLine();
}

void UnwrappedLineParser::parseEnum() {
  // Won't be 'enum' for NS_ENUMs.
  if (FormatTok->Tok.is(tok::kw_enum))
    nextToken();

  // Eat up enum class ...
  if (FormatTok->Tok.is(tok::kw_class) || FormatTok->Tok.is(tok::kw_struct))
    nextToken();
  while (FormatTok->Tok.getIdentifierInfo() ||
         FormatTok->isOneOf(tok::colon, tok::coloncolon, tok::less,
                            tok::greater, tok::comma, tok::question)) {
    nextToken();
    // We can have macros or attributes in between 'enum' and the enum name.
    if (FormatTok->is(tok::l_paren))
      parseParens();
    if (FormatTok->is(tok::identifier))
      nextToken();
  }

  // Just a declaration or something is wrong.
  if (FormatTok->isNot(tok::l_brace))
    return;
  FormatTok->BlockKind = BK_Block;

  if (Style.Language == FormatStyle::LK_Java) {
    // Java enums are different.
    parseJavaEnumBody();
    return;
  }

  // Parse enum body.
  bool HasError = !parseBracedList(/*ContinueOnSemicolons=*/true);
  if (HasError) {
    if (FormatTok->is(tok::semi))
      nextToken();
    addUnwrappedLine();
  }

  // We fall through to parsing a structural element afterwards, so that in
  // enum A {} n, m;
  // "} n, m;" will end up in one unwrapped line.
}

void UnwrappedLineParser::parseJavaEnumBody() {
  // Determine whether the enum is simple, i.e. does not have a semicolon or
  // constants with class bodies. Simple enums can be formatted like braced
  // lists, contracted to a single line, etc.
  unsigned StoredPosition = Tokens->getPosition();
  bool IsSimple = true;
  FormatToken *Tok = Tokens->getNextToken();
  while (Tok) {
    if (Tok->is(tok::r_brace))
      break;
    if (Tok->isOneOf(tok::l_brace, tok::semi)) {
      IsSimple = false;
      break;
    }
    // FIXME: This will also mark enums with braces in the arguments to enum
    // constants as "not simple". This is probably fine in practice, though.
    Tok = Tokens->getNextToken();
  }
  FormatTok = Tokens->setPosition(StoredPosition);

  if (IsSimple) {
    parseBracedList();
    addUnwrappedLine();
    return;
  }

  // Parse the body of a more complex enum.
  // First add a line for everything up to the "{".
  nextToken();
  addUnwrappedLine();
  ++Line->Level;

  // Parse the enum constants.
  while (FormatTok) {
    if (FormatTok->is(tok::l_brace)) {
      // Parse the constant's class body.
      parseBlock(/*MustBeDeclaration=*/true, /*AddLevel=*/true,
                 /*MunchSemi=*/false);
    } else if (FormatTok->is(tok::l_paren)) {
      parseParens();
    } else if (FormatTok->is(tok::comma)) {
      nextToken();
      addUnwrappedLine();
    } else if (FormatTok->is(tok::semi)) {
      nextToken();
      addUnwrappedLine();
      break;
    } else if (FormatTok->is(tok::r_brace)) {
      addUnwrappedLine();
      break;
    } else {
      nextToken();
    }
  }

  // Parse the class body after the enum's ";" if any.
  parseLevel(/*HasOpeningBrace=*/true);
  nextToken();
  --Line->Level;
  addUnwrappedLine();
}

void UnwrappedLineParser::parseRecord() {
  const FormatToken &InitialToken = *FormatTok;
  nextToken();
  if (FormatTok->isOneOf(tok::identifier, tok::coloncolon, tok::kw___attribute,
                         tok::kw___declspec, tok::kw_alignas)) {
    nextToken();
    // We can have macros or attributes in between 'class' and the class name.
    if (FormatTok->Tok.is(tok::l_paren)) {
      parseParens();
    }
    // The actual identifier can be a nested name specifier, and in macros
    // it is often token-pasted.
    while (FormatTok->is(tok::identifier) || FormatTok->is(tok::coloncolon) ||
           FormatTok->is(tok::hashhash) ||
           (Style.Language == FormatStyle::LK_Java &&
            FormatTok->isOneOf(tok::period, tok::comma)))
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
    if (ShouldBreakBeforeBrace(Style, InitialToken))
      addUnwrappedLine();

    parseBlock(/*MustBeDeclaration=*/true, /*AddLevel=*/true,
               /*MunchSemi=*/false);
  }
  // We fall through to parsing a structural element afterwards, so
  // class A {} n, m;
  // will end up in one unwrapped line.
  // This does not apply for Java.
  if (Style.Language == FormatStyle::LK_Java ||
      Style.Language == FormatStyle::LK_JavaScript)
    addUnwrappedLine();
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
    if (FormatTok->is(tok::l_brace)) {
      parseBlock(/*MustBeDeclaration=*/false);
      // In ObjC interfaces, nothing should be following the "}".
      addUnwrappedLine();
    } else if (FormatTok->is(tok::r_brace)) {
      // Ignore stray "}". parseStructuralElement doesn't consume them.
      nextToken();
      addUnwrappedLine();
    } else {
      parseStructuralElement();
    }
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

  if (FormatTok->Tok.is(tok::l_brace)) {
    if (Style.BreakBeforeBraces == FormatStyle::BS_Allman ||
        Style.BreakBeforeBraces == FormatStyle::BS_GNU)
      addUnwrappedLine();
    parseBlock(/*MustBeDeclaration=*/true);
  }

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

void UnwrappedLineParser::parseJavaScriptEs6ImportExport() {
  assert(FormatTok->isOneOf(Keywords.kw_import, tok::kw_export));
  nextToken();

  if (FormatTok->isOneOf(tok::kw_const, tok::kw_class, Keywords.kw_function,
                         Keywords.kw_var))
    return; // Fall through to parsing the corresponding structure.

  if (FormatTok->is(tok::kw_default)) {
    nextToken(); // export default ..., fall through after eating 'default'.
    return;
  }

  if (FormatTok->is(tok::l_brace)) {
    FormatTok->BlockKind = BK_Block;
    parseBracedList();
  }

  while (!eof() && FormatTok->isNot(tok::semi) &&
         FormatTok->isNot(tok::l_brace)) {
    nextToken();
  }
}

LLVM_ATTRIBUTE_UNUSED static void printDebugInfo(const UnwrappedLine &Line,
                                                 StringRef Prefix = "") {
  llvm::dbgs() << Prefix << "Line(" << Line.Level << ")"
               << (Line.InPPDirective ? " MACRO" : "") << ": ";
  for (std::list<UnwrappedLineNode>::const_iterator I = Line.Tokens.begin(),
                                                    E = Line.Tokens.end();
       I != E; ++I) {
    llvm::dbgs() << I->Tok->Tok.getName() << "[" << I->Tok->Type << "] ";
  }
  for (std::list<UnwrappedLineNode>::const_iterator I = Line.Tokens.begin(),
                                                    E = Line.Tokens.end();
       I != E; ++I) {
    const UnwrappedLineNode &Node = *I;
    for (SmallVectorImpl<UnwrappedLine>::const_iterator
             I = Node.Children.begin(),
             E = Node.Children.end();
         I != E; ++I) {
      printDebugInfo(*I, "\nChild: ");
    }
  }
  llvm::dbgs() << "\n";
}

void UnwrappedLineParser::addUnwrappedLine() {
  if (Line->Tokens.empty())
    return;
  DEBUG({
    if (CurrentLines == &Lines)
      printDebugInfo(*Line);
  });
  CurrentLines->push_back(*Line);
  Line->Tokens.clear();
  if (CurrentLines == &Lines && !PreprocessorDirectives.empty()) {
    for (SmallVectorImpl<UnwrappedLine>::iterator
             I = PreprocessorDirectives.begin(),
             E = PreprocessorDirectives.end();
         I != E; ++I) {
      CurrentLines->push_back(*I);
    }
    PreprocessorDirectives.clear();
  }
}

bool UnwrappedLineParser::eof() const { return FormatTok->Tok.is(tok::eof); }

bool UnwrappedLineParser::isOnNewLine(const FormatToken &FormatTok) {
  return (Line->InPPDirective || FormatTok.HasUnescapedNewline) &&
         FormatTok.NewlinesBefore > 0;
}

void UnwrappedLineParser::flushComments(bool NewlineBeforeNext) {
  bool JustComments = Line->Tokens.empty();
  for (SmallVectorImpl<FormatToken *>::const_iterator
           I = CommentsBeforeNextToken.begin(),
           E = CommentsBeforeNextToken.end();
       I != E; ++I) {
    if (isOnNewLine(**I) && JustComments) {
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
  flushComments(isOnNewLine(*FormatTok));
  pushToken(FormatTok);
  readToken();
}

void UnwrappedLineParser::readToken() {
  bool CommentsInCurrentLine = true;
  do {
    FormatTok = Tokens->getNextToken();
    assert(FormatTok);
    while (!Line->InPPDirective && FormatTok->Tok.is(tok::hash) &&
           (FormatTok->HasUnescapedNewline || FormatTok->IsFirst)) {
      // If there is an unfinished unwrapped line, we flush the preprocessor
      // directives only after that unwrapped line was finished later.
      bool SwitchToPreprocessorLines = !Line->Tokens.empty();
      ScopedLineState BlockState(*this, SwitchToPreprocessorLines);
      // Comments stored before the preprocessor directive need to be output
      // before the preprocessor directive, at the same level as the
      // preprocessor directive, as we consider them to apply to the directive.
      flushComments(isOnNewLine(*FormatTok));
      parsePPDirective();
    }
    while (FormatTok->Type == TT_ConflictStart ||
           FormatTok->Type == TT_ConflictEnd ||
           FormatTok->Type == TT_ConflictAlternative) {
      if (FormatTok->Type == TT_ConflictStart) {
        conditionalCompilationStart(/*Unreachable=*/false);
      } else if (FormatTok->Type == TT_ConflictAlternative) {
        conditionalCompilationAlternative();
      } else if (FormatTok->Type == TT_ConflictEnd) {
        conditionalCompilationEnd();
      }
      FormatTok = Tokens->getNextToken();
      FormatTok->MustBreakBefore = true;
    }

    if (!PPStack.empty() && (PPStack.back() == PP_Unreachable) &&
        !Line->InPPDirective) {
      continue;
    }

    if (!FormatTok->Tok.is(tok::comment))
      return;
    if (isOnNewLine(*FormatTok) || FormatTok->IsFirst) {
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
  Line->Tokens.push_back(UnwrappedLineNode(Tok));
  if (MustBreakBeforeNextToken) {
    Line->Tokens.back().Tok->MustBreakBefore = true;
    MustBreakBeforeNextToken = false;
  }
}

} // end namespace format
} // end namespace clang
