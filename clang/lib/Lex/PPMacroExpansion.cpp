//===--- MacroExpansion.cpp - Top level Macro Expansion -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the top level handling of macro expasion for the
// preprocessor.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/Preprocessor.h"
#include "MacroArgs.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/Diagnostic.h"
#include <ctime>
using namespace clang;

/// setMacroInfo - Specify a macro for this identifier.
///
void Preprocessor::setMacroInfo(IdentifierInfo *II, MacroInfo *MI) {
  if (MI == 0) {
    if (II->hasMacroDefinition()) {
      Macros.erase(II);
      II->setHasMacroDefinition(false);
    }
  } else {
    Macros[II] = MI;
    II->setHasMacroDefinition(true);
  }
}

/// RegisterBuiltinMacro - Register the specified identifier in the identifier
/// table and mark it as a builtin macro to be expanded.
IdentifierInfo *Preprocessor::RegisterBuiltinMacro(const char *Name) {
  // Get the identifier.
  IdentifierInfo *Id = getIdentifierInfo(Name);
  
  // Mark it as being a macro that is builtin.
  MacroInfo *MI = new MacroInfo(SourceLocation());
  MI->setIsBuiltinMacro();
  setMacroInfo(Id, MI);
  return Id;
}


/// RegisterBuiltinMacros - Register builtin macros, such as __LINE__ with the
/// identifier table.
void Preprocessor::RegisterBuiltinMacros() {
  Ident__LINE__ = RegisterBuiltinMacro("__LINE__");
  Ident__FILE__ = RegisterBuiltinMacro("__FILE__");
  Ident__DATE__ = RegisterBuiltinMacro("__DATE__");
  Ident__TIME__ = RegisterBuiltinMacro("__TIME__");
  Ident_Pragma  = RegisterBuiltinMacro("_Pragma");
  
  // GCC Extensions.
  Ident__BASE_FILE__     = RegisterBuiltinMacro("__BASE_FILE__");
  Ident__INCLUDE_LEVEL__ = RegisterBuiltinMacro("__INCLUDE_LEVEL__");
  Ident__TIMESTAMP__     = RegisterBuiltinMacro("__TIMESTAMP__");
}

/// isTrivialSingleTokenExpansion - Return true if MI, which has a single token
/// in its expansion, currently expands to that token literally.
static bool isTrivialSingleTokenExpansion(const MacroInfo *MI,
                                          const IdentifierInfo *MacroIdent,
                                          Preprocessor &PP) {
  IdentifierInfo *II = MI->getReplacementToken(0).getIdentifierInfo();

  // If the token isn't an identifier, it's always literally expanded.
  if (II == 0) return true;
  
  // If the identifier is a macro, and if that macro is enabled, it may be
  // expanded so it's not a trivial expansion.
  if (II->hasMacroDefinition() && PP.getMacroInfo(II)->isEnabled() &&
      // Fast expanding "#define X X" is ok, because X would be disabled.
      II != MacroIdent)
    return false;
  
  // If this is an object-like macro invocation, it is safe to trivially expand
  // it.
  if (MI->isObjectLike()) return true;

  // If this is a function-like macro invocation, it's safe to trivially expand
  // as long as the identifier is not a macro argument.
  for (MacroInfo::arg_iterator I = MI->arg_begin(), E = MI->arg_end();
       I != E; ++I)
    if (*I == II)
      return false;   // Identifier is a macro argument.
  
  return true;
}


/// isNextPPTokenLParen - Determine whether the next preprocessor token to be
/// lexed is a '('.  If so, consume the token and return true, if not, this
/// method should have no observable side-effect on the lexed tokens.
bool Preprocessor::isNextPPTokenLParen() {
  // Do some quick tests for rejection cases.
  unsigned Val;
  if (CurLexer)
    Val = CurLexer->isNextPPTokenLParen();
  else
    Val = CurTokenLexer->isNextTokenLParen();
  
  if (Val == 2) {
    // We have run off the end.  If it's a source file we don't
    // examine enclosing ones (C99 5.1.1.2p4).  Otherwise walk up the
    // macro stack.
    if (CurLexer)
      return false;
    for (unsigned i = IncludeMacroStack.size(); i != 0; --i) {
      IncludeStackInfo &Entry = IncludeMacroStack[i-1];
      if (Entry.TheLexer)
        Val = Entry.TheLexer->isNextPPTokenLParen();
      else
        Val = Entry.TheTokenLexer->isNextTokenLParen();
      
      if (Val != 2)
        break;
      
      // Ran off the end of a source file?
      if (Entry.TheLexer)
        return false;
    }
  }

  // Okay, if we know that the token is a '(', lex it and return.  Otherwise we
  // have found something that isn't a '(' or we found the end of the
  // translation unit.  In either case, return false.
  if (Val != 1)
    return false;
  
  Token Tok;
  LexUnexpandedToken(Tok);
  assert(Tok.is(tok::l_paren) && "Error computing l-paren-ness?");
  return true;
}

/// HandleMacroExpandedIdentifier - If an identifier token is read that is to be
/// expanded as a macro, handle it and return the next token as 'Identifier'.
bool Preprocessor::HandleMacroExpandedIdentifier(Token &Identifier, 
                                                 MacroInfo *MI) {
  // If this is a macro exapnsion in the "#if !defined(x)" line for the file,
  // then the macro could expand to different things in other contexts, we need
  // to disable the optimization in this case.
  if (CurLexer) CurLexer->MIOpt.ExpandedMacro();
  
  // If this is a builtin macro, like __LINE__ or _Pragma, handle it specially.
  if (MI->isBuiltinMacro()) {
    ExpandBuiltinMacro(Identifier);
    return false;
  }
  
  /// Args - If this is a function-like macro expansion, this contains,
  /// for each macro argument, the list of tokens that were provided to the
  /// invocation.
  MacroArgs *Args = 0;
  
  // If this is a function-like macro, read the arguments.
  if (MI->isFunctionLike()) {
    // C99 6.10.3p10: If the preprocessing token immediately after the the macro
    // name isn't a '(', this macro should not be expanded.  Otherwise, consume
    // it.
    if (!isNextPPTokenLParen())
      return true;
    
    // Remember that we are now parsing the arguments to a macro invocation.
    // Preprocessor directives used inside macro arguments are not portable, and
    // this enables the warning.
    InMacroArgs = true;
    Args = ReadFunctionLikeMacroArgs(Identifier, MI);
    
    // Finished parsing args.
    InMacroArgs = false;
    
    // If there was an error parsing the arguments, bail out.
    if (Args == 0) return false;
    
    ++NumFnMacroExpanded;
  } else {
    ++NumMacroExpanded;
  }
  
  // Notice that this macro has been used.
  MI->setIsUsed(true);
  
  // If we started lexing a macro, enter the macro expansion body.
  
  // If this macro expands to no tokens, don't bother to push it onto the
  // expansion stack, only to take it right back off.
  if (MI->getNumTokens() == 0) {
    // No need for arg info.
    if (Args) Args->destroy();
    
    // Ignore this macro use, just return the next token in the current
    // buffer.
    bool HadLeadingSpace = Identifier.hasLeadingSpace();
    bool IsAtStartOfLine = Identifier.isAtStartOfLine();
    
    Lex(Identifier);
    
    // If the identifier isn't on some OTHER line, inherit the leading
    // whitespace/first-on-a-line property of this token.  This handles
    // stuff like "! XX," -> "! ," and "   XX," -> "    ,", when XX is
    // empty.
    if (!Identifier.isAtStartOfLine()) {
      if (IsAtStartOfLine) Identifier.setFlag(Token::StartOfLine);
      if (HadLeadingSpace) Identifier.setFlag(Token::LeadingSpace);
    }
    ++NumFastMacroExpanded;
    return false;
    
  } else if (MI->getNumTokens() == 1 &&
             isTrivialSingleTokenExpansion(MI, Identifier.getIdentifierInfo(),
                                           *this)){
    // Otherwise, if this macro expands into a single trivially-expanded
    // token: expand it now.  This handles common cases like 
    // "#define VAL 42".

    // No need for arg info.
    if (Args) Args->destroy();

    // Propagate the isAtStartOfLine/hasLeadingSpace markers of the macro
    // identifier to the expanded token.
    bool isAtStartOfLine = Identifier.isAtStartOfLine();
    bool hasLeadingSpace = Identifier.hasLeadingSpace();
    
    // Remember where the token is instantiated.
    SourceLocation InstantiateLoc = Identifier.getLocation();
    
    // Replace the result token.
    Identifier = MI->getReplacementToken(0);
    
    // Restore the StartOfLine/LeadingSpace markers.
    Identifier.setFlagValue(Token::StartOfLine , isAtStartOfLine);
    Identifier.setFlagValue(Token::LeadingSpace, hasLeadingSpace);
    
    // Update the tokens location to include both its logical and physical
    // locations.
    SourceLocation Loc =
      SourceMgr.getInstantiationLoc(Identifier.getLocation(), InstantiateLoc);
    Identifier.setLocation(Loc);
    
    // If this is #define X X, we must mark the result as unexpandible.
    if (IdentifierInfo *NewII = Identifier.getIdentifierInfo())
      if (getMacroInfo(NewII) == MI)
        Identifier.setFlag(Token::DisableExpand);
    
    // Since this is not an identifier token, it can't be macro expanded, so
    // we're done.
    ++NumFastMacroExpanded;
    return false;
  }
  
  // Start expanding the macro.
  EnterMacro(Identifier, Args);
  
  // Now that the macro is at the top of the include stack, ask the
  // preprocessor to read the next token from it.
  Lex(Identifier);
  return false;
}

/// ReadFunctionLikeMacroArgs - After reading "MACRO(", this method is
/// invoked to read all of the actual arguments specified for the macro
/// invocation.  This returns null on error.
MacroArgs *Preprocessor::ReadFunctionLikeMacroArgs(Token &MacroName,
                                                   MacroInfo *MI) {
  // The number of fixed arguments to parse.
  unsigned NumFixedArgsLeft = MI->getNumArgs();
  bool isVariadic = MI->isVariadic();
  
  // Outer loop, while there are more arguments, keep reading them.
  Token Tok;
  Tok.setKind(tok::comma);
  --NumFixedArgsLeft;  // Start reading the first arg.

  // ArgTokens - Build up a list of tokens that make up each argument.  Each
  // argument is separated by an EOF token.  Use a SmallVector so we can avoid
  // heap allocations in the common case.
  llvm::SmallVector<Token, 64> ArgTokens;

  unsigned NumActuals = 0;
  while (Tok.is(tok::comma)) {
    // C99 6.10.3p11: Keep track of the number of l_parens we have seen.  Note
    // that we already consumed the first one.
    unsigned NumParens = 0;
    
    while (1) {
      // Read arguments as unexpanded tokens.  This avoids issues, e.g., where
      // an argument value in a macro could expand to ',' or '(' or ')'.
      LexUnexpandedToken(Tok);
      
      if (Tok.is(tok::eof) || Tok.is(tok::eom)) { // "#if f(<eof>" & "#if f(\n"
        Diag(MacroName, diag::err_unterm_macro_invoc);
        // Do not lose the EOF/EOM.  Return it to the client.
        MacroName = Tok;
        return 0;
      } else if (Tok.is(tok::r_paren)) {
        // If we found the ) token, the macro arg list is done.
        if (NumParens-- == 0)
          break;
      } else if (Tok.is(tok::l_paren)) {
        ++NumParens;
      } else if (Tok.is(tok::comma) && NumParens == 0) {
        // Comma ends this argument if there are more fixed arguments expected.
        if (NumFixedArgsLeft)
          break;
        
        // If this is not a variadic macro, too many args were specified.
        if (!isVariadic) {
          // Emit the diagnostic at the macro name in case there is a missing ).
          // Emitting it at the , could be far away from the macro name.
          Diag(MacroName, diag::err_too_many_args_in_macro_invoc);
          return 0;
        }
        // Otherwise, continue to add the tokens to this variable argument.
      } else if (Tok.is(tok::comment) && !KeepMacroComments) {
        // If this is a comment token in the argument list and we're just in
        // -C mode (not -CC mode), discard the comment.
        continue;
      } else if (Tok.is(tok::identifier)) {
        // Reading macro arguments can cause macros that we are currently
        // expanding from to be popped off the expansion stack.  Doing so causes
        // them to be reenabled for expansion.  Here we record whether any
        // identifiers we lex as macro arguments correspond to disabled macros.
        // If so, we mark the token as noexpand.  This is a subtle aspect of 
        // C99 6.10.3.4p2.
        if (MacroInfo *MI = getMacroInfo(Tok.getIdentifierInfo()))
          if (!MI->isEnabled())
            Tok.setFlag(Token::DisableExpand);
      }
  
      ArgTokens.push_back(Tok);
    }

    // Empty arguments are standard in C99 and supported as an extension in
    // other modes.
    if (ArgTokens.empty() && !Features.C99)
      Diag(Tok, diag::ext_empty_fnmacro_arg);
    
    // Add a marker EOF token to the end of the token list for this argument.
    Token EOFTok;
    EOFTok.startToken();
    EOFTok.setKind(tok::eof);
    EOFTok.setLocation(Tok.getLocation());
    EOFTok.setLength(0);
    ArgTokens.push_back(EOFTok);
    ++NumActuals;
    --NumFixedArgsLeft;
  };
  
  // Okay, we either found the r_paren.  Check to see if we parsed too few
  // arguments.
  unsigned MinArgsExpected = MI->getNumArgs();
  
  // See MacroArgs instance var for description of this.
  bool isVarargsElided = false;
  
  if (NumActuals < MinArgsExpected) {
    // There are several cases where too few arguments is ok, handle them now.
    if (NumActuals+1 == MinArgsExpected && MI->isVariadic()) {
      // Varargs where the named vararg parameter is missing: ok as extension.
      // #define A(x, ...)
      // A("blah")
      Diag(Tok, diag::ext_missing_varargs_arg);

      // Remember this occurred if this is a macro invocation with at least
      // one actual argument.  This allows us to elide the comma when used for
      // cases like:
      //   #define A(x, foo...) blah(a, ## foo) 
      //   #define A(x, ...) blah(a, ## __VA_ARGS__) 
      isVarargsElided = MI->getNumArgs() > 1;
    } else if (MI->getNumArgs() == 1) {
      // #define A(x)
      //   A()
      // is ok because it is an empty argument.
      
      // Empty arguments are standard in C99 and supported as an extension in
      // other modes.
      if (ArgTokens.empty() && !Features.C99)
        Diag(Tok, diag::ext_empty_fnmacro_arg);
    } else {
      // Otherwise, emit the error.
      Diag(Tok, diag::err_too_few_args_in_macro_invoc);
      return 0;
    }
    
    // Add a marker EOF token to the end of the token list for this argument.
    SourceLocation EndLoc = Tok.getLocation();
    Tok.startToken();
    Tok.setKind(tok::eof);
    Tok.setLocation(EndLoc);
    Tok.setLength(0);
    ArgTokens.push_back(Tok);
  }
  
  return MacroArgs::create(MI, &ArgTokens[0], ArgTokens.size(),isVarargsElided);
}

/// ComputeDATE_TIME - Compute the current time, enter it into the specified
/// scratch buffer, then return DATELoc/TIMELoc locations with the position of
/// the identifier tokens inserted.
static void ComputeDATE_TIME(SourceLocation &DATELoc, SourceLocation &TIMELoc,
                             Preprocessor &PP) {
  time_t TT = time(0);
  struct tm *TM = localtime(&TT);
  
  static const char * const Months[] = {
    "Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"
  };
  
  char TmpBuffer[100];
  sprintf(TmpBuffer, "\"%s %2d %4d\"", Months[TM->tm_mon], TM->tm_mday, 
          TM->tm_year+1900);
  DATELoc = PP.CreateString(TmpBuffer, strlen(TmpBuffer));

  sprintf(TmpBuffer, "\"%02d:%02d:%02d\"", TM->tm_hour, TM->tm_min, TM->tm_sec);
  TIMELoc = PP.CreateString(TmpBuffer, strlen(TmpBuffer));
}

/// ExpandBuiltinMacro - If an identifier token is read that is to be expanded
/// as a builtin macro, handle it and return the next token as 'Tok'.
void Preprocessor::ExpandBuiltinMacro(Token &Tok) {
  // Figure out which token this is.
  IdentifierInfo *II = Tok.getIdentifierInfo();
  assert(II && "Can't be a macro without id info!");
  
  // If this is an _Pragma directive, expand it, invoke the pragma handler, then
  // lex the token after it.
  if (II == Ident_Pragma)
    return Handle_Pragma(Tok);
  
  ++NumBuiltinMacroExpanded;

  char TmpBuffer[100];

  // Set up the return result.
  Tok.setIdentifierInfo(0);
  Tok.clearFlag(Token::NeedsCleaning);
  
  if (II == Ident__LINE__) {
    // __LINE__ expands to a simple numeric value.  Add a space after it so that
    // it will tokenize as a number (and not run into stuff after it in the temp
    // buffer).
    sprintf(TmpBuffer, "%u ",
            SourceMgr.getLogicalLineNumber(Tok.getLocation()));
    unsigned Length = strlen(TmpBuffer)-1;
    Tok.setKind(tok::numeric_constant);
    Tok.setLength(Length);
    Tok.setLocation(CreateString(TmpBuffer, Length+1, Tok.getLocation()));
  } else if (II == Ident__FILE__ || II == Ident__BASE_FILE__) {
    SourceLocation Loc = Tok.getLocation();
    if (II == Ident__BASE_FILE__) {
      Diag(Tok, diag::ext_pp_base_file);
      SourceLocation NextLoc = SourceMgr.getIncludeLoc(Loc);
      while (NextLoc.isValid()) {
        Loc = NextLoc;
        NextLoc = SourceMgr.getIncludeLoc(Loc);
      }
    }
    
    // Escape this filename.  Turn '\' -> '\\' '"' -> '\"'
    std::string FN = SourceMgr.getSourceName(SourceMgr.getLogicalLoc(Loc));
    FN = '"' + Lexer::Stringify(FN) + '"';
    Tok.setKind(tok::string_literal);
    Tok.setLength(FN.size());
    Tok.setLocation(CreateString(&FN[0], FN.size(), Tok.getLocation()));
  } else if (II == Ident__DATE__) {
    if (!DATELoc.isValid())
      ComputeDATE_TIME(DATELoc, TIMELoc, *this);
    Tok.setKind(tok::string_literal);
    Tok.setLength(strlen("\"Mmm dd yyyy\""));
    Tok.setLocation(SourceMgr.getInstantiationLoc(DATELoc, Tok.getLocation()));
  } else if (II == Ident__TIME__) {
    if (!TIMELoc.isValid())
      ComputeDATE_TIME(DATELoc, TIMELoc, *this);
    Tok.setKind(tok::string_literal);
    Tok.setLength(strlen("\"hh:mm:ss\""));
    Tok.setLocation(SourceMgr.getInstantiationLoc(TIMELoc, Tok.getLocation()));
  } else if (II == Ident__INCLUDE_LEVEL__) {
    Diag(Tok, diag::ext_pp_include_level);

    // Compute the include depth of this token.
    unsigned Depth = 0;
    SourceLocation Loc = SourceMgr.getIncludeLoc(Tok.getLocation());
    for (; Loc.isValid(); ++Depth)
      Loc = SourceMgr.getIncludeLoc(Loc);
    
    // __INCLUDE_LEVEL__ expands to a simple numeric value.  Add a space after
    // it so that it will tokenize as a number (and not run into stuff after it
    // in the temp buffer).
    sprintf(TmpBuffer, "%u ", Depth);
    unsigned Length = strlen(TmpBuffer)-1;
    Tok.setKind(tok::numeric_constant);
    Tok.setLength(Length);
    Tok.setLocation(CreateString(TmpBuffer, Length, Tok.getLocation()));
  } else if (II == Ident__TIMESTAMP__) {
    // MSVC, ICC, GCC, VisualAge C++ extension.  The generated string should be
    // of the form "Ddd Mmm dd hh::mm::ss yyyy", which is returned by asctime.
    Diag(Tok, diag::ext_pp_timestamp);

    // Get the file that we are lexing out of.  If we're currently lexing from
    // a macro, dig into the include stack.
    const FileEntry *CurFile = 0;
    Lexer *TheLexer = getCurrentFileLexer();
    
    if (TheLexer)
      CurFile = SourceMgr.getFileEntryForLoc(TheLexer->getFileLoc());
    
    // If this file is older than the file it depends on, emit a diagnostic.
    const char *Result;
    if (CurFile) {
      time_t TT = CurFile->getModificationTime();
      struct tm *TM = localtime(&TT);
      Result = asctime(TM);
    } else {
      Result = "??? ??? ?? ??:??:?? ????\n";
    }
    TmpBuffer[0] = '"';
    strcpy(TmpBuffer+1, Result);
    unsigned Len = strlen(TmpBuffer);
    TmpBuffer[Len-1] = '"';  // Replace the newline with a quote.
    Tok.setKind(tok::string_literal);
    Tok.setLength(Len);
    Tok.setLocation(CreateString(TmpBuffer, Len+1, Tok.getLocation()));
  } else {
    assert(0 && "Unknown identifier!");
  }
}
