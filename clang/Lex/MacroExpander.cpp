//===--- MacroExpander.cpp - Lex from a macro expansion -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MacroExpander interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/MacroExpander.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/SourceManager.h"
using namespace llvm;
using namespace clang;

//===----------------------------------------------------------------------===//
// MacroFormalArgs Implementation
//===----------------------------------------------------------------------===//

MacroFormalArgs::MacroFormalArgs(const MacroInfo *MI) {
  assert(MI->isFunctionLike() &&
         "Can't have formal args for an object-like macro!");
  // Reserve space for arguments to avoid reallocation.
  unsigned NumArgs = MI->getNumArgs();
  if (MI->isC99Varargs() || MI->isGNUVarargs())
    NumArgs += 3;    // Varargs can have more than this, just some guess.
  
  ArgTokens.reserve(NumArgs);
}

/// StringifyArgument - Implement C99 6.10.3.2p2.
static LexerToken StringifyArgument(const std::vector<LexerToken> &Toks,
                                    Preprocessor &PP) {
  LexerToken Tok;
  Tok.StartToken();
  Tok.SetKind(tok::string_literal);
  
  std::string Val = "\"XXYZLAKSDFJAS\"";
  Tok.SetLength(Val.size());
  Tok.SetLocation(PP.CreateString(&Val[0], Val.size()));
  return Tok;
}

/// getStringifiedArgument - Compute, cache, and return the specified argument
/// that has been 'stringified' as required by the # operator.
const LexerToken &MacroFormalArgs::getStringifiedArgument(unsigned ArgNo,
                                                          Preprocessor &PP) {
  assert(ArgNo < ArgTokens.size() && "Invalid argument number!");
  if (StringifiedArgs.empty()) {
    StringifiedArgs.resize(ArgTokens.size());
    memset(&StringifiedArgs[0], 0, sizeof(StringifiedArgs[0])*ArgTokens.size());
  }
  if (StringifiedArgs[ArgNo].getKind() != tok::string_literal)
    StringifiedArgs[ArgNo] = StringifyArgument(ArgTokens[ArgNo], PP);
  return StringifiedArgs[ArgNo];
}

//===----------------------------------------------------------------------===//
// MacroExpander Implementation
//===----------------------------------------------------------------------===//

MacroExpander::MacroExpander(LexerToken &Tok, MacroFormalArgs *Formals,
                             Preprocessor &pp)
  : Macro(*Tok.getIdentifierInfo()->getMacroInfo()),
    FormalArgs(Formals), PP(pp), CurToken(0),
    InstantiateLoc(Tok.getLocation()),
    AtStartOfLine(Tok.isAtStartOfLine()),
    HasLeadingSpace(Tok.hasLeadingSpace()) {
  MacroTokens = &Macro.getReplacementTokens();

  // If this is a function-like macro, expand the arguments and change
  // MacroTokens to point to the expanded tokens.
  if (Macro.isFunctionLike() && Macro.getNumArgs())
    ExpandFunctionArguments();
}

MacroExpander::~MacroExpander() {
  // If this was a function-like macro that actually uses its arguments, delete
  // the expanded tokens.
  if (MacroTokens != &Macro.getReplacementTokens())
    delete MacroTokens;
  
  // MacroExpander owns its formal arguments.
  delete FormalArgs;
}

/// Expand the arguments of a function-like macro so that we can quickly
/// return preexpanded tokens from MacroTokens.
void MacroExpander::ExpandFunctionArguments() {
  std::vector<LexerToken> ResultToks;
  
  // Loop through the MacroTokens tokens, expanding them into ResultToks.  Keep
  // track of whether we change anything.  If not, no need to keep them.  If so,
  // we install the newly expanded sequence as MacroTokens.
  bool MadeChange = false;
  for (unsigned i = 0, e = MacroTokens->size(); i != e; ++i) {
    // If we found the stringify operator, get the argument stringified.  The
    // preprocessor already verified that the following token is a macro name
    // when the #define was parsed.
    const LexerToken &CurTok = (*MacroTokens)[i];
    if (CurTok.getKind() == tok::hash) {
      int ArgNo = Macro.getArgumentNum((*MacroTokens)[i+1].getIdentifierInfo());
      assert(ArgNo != -1 && "Token following # is not an argument?");
      
      ResultToks.push_back(FormalArgs->getStringifiedArgument(ArgNo, PP));
      
      // FIXME: Should the stringified string leading space flag get set to
      // match the # or the identifier?
      
      MadeChange = true;
      ++i;  // Skip arg name.
    } else {
      // FIXME: handle microsoft charize extension.
      
      ResultToks.push_back(CurTok);
    }
  }
  
  // If anything changed, install this as the new MacroTokens list.
  if (MadeChange) {
    // This is deleted in the dtor.
    std::vector<LexerToken> *Res = new std::vector<LexerToken>();
    Res->swap(ResultToks);
    MacroTokens = Res;
  }
}

/// Lex - Lex and return a token from this macro stream.
///
void MacroExpander::Lex(LexerToken &Tok) {
  // Lexing off the end of the macro, pop this macro off the expansion stack.
  if (isAtEnd())
    return PP.HandleEndOfMacro(Tok);
  
  // Get the next token to return.
  Tok = (*MacroTokens)[CurToken++];

  // The token's current location indicate where the token was lexed from.  We
  // need this information to compute the spelling of the token, but any
  // diagnostics for the expanded token should appear as if they came from
  // InstantiationLoc.  Pull this information together into a new SourceLocation
  // that captures all of this.
  Tok.SetLocation(PP.getSourceManager().getInstantiationLoc(Tok.getLocation(),
                                                            InstantiateLoc));

  // If this is the first token, set the lexical properties of the token to
  // match the lexical properties of the macro identifier.
  if (CurToken == 1) {
    Tok.SetFlagValue(LexerToken::StartOfLine , AtStartOfLine);
    Tok.SetFlagValue(LexerToken::LeadingSpace, HasLeadingSpace);
  }
  
  // Handle recursive expansion!
  if (Tok.getIdentifierInfo())
    return PP.HandleIdentifier(Tok);

  // Otherwise, return a normal token.
}

/// isNextTokenLParen - If the next token lexed will pop this macro off the
/// expansion stack, return 2.  If the next unexpanded token is a '(', return
/// 1, otherwise return 0.
unsigned MacroExpander::isNextTokenLParen() const {
  // Out of tokens?
  if (isAtEnd())
    return 2;
  return (*MacroTokens)[CurToken].getKind() == tok::l_paren;
}
