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
#include "clang/Lex/LexDiagnostic.h"
#include "llvm/ADT/StringSwitch.h"
#include <cstdio>
#include <ctime>
using namespace clang;

/// setMacroInfo - Specify a macro for this identifier.
///
void Preprocessor::setMacroInfo(IdentifierInfo *II, MacroInfo *MI) {
  if (MI) {
    Macros[II] = MI;
    II->setHasMacroDefinition(true);
  } else if (II->hasMacroDefinition()) {
    Macros.erase(II);
    II->setHasMacroDefinition(false);
  }
}

/// RegisterBuiltinMacro - Register the specified identifier in the identifier
/// table and mark it as a builtin macro to be expanded.
static IdentifierInfo *RegisterBuiltinMacro(Preprocessor &PP, const char *Name){
  // Get the identifier.
  IdentifierInfo *Id = PP.getIdentifierInfo(Name);

  // Mark it as being a macro that is builtin.
  MacroInfo *MI = PP.AllocateMacroInfo(SourceLocation());
  MI->setIsBuiltinMacro();
  PP.setMacroInfo(Id, MI);
  return Id;
}


/// RegisterBuiltinMacros - Register builtin macros, such as __LINE__ with the
/// identifier table.
void Preprocessor::RegisterBuiltinMacros() {
  Ident__LINE__ = RegisterBuiltinMacro(*this, "__LINE__");
  Ident__FILE__ = RegisterBuiltinMacro(*this, "__FILE__");
  Ident__DATE__ = RegisterBuiltinMacro(*this, "__DATE__");
  Ident__TIME__ = RegisterBuiltinMacro(*this, "__TIME__");
  Ident__COUNTER__ = RegisterBuiltinMacro(*this, "__COUNTER__");
  Ident_Pragma  = RegisterBuiltinMacro(*this, "_Pragma");

  // GCC Extensions.
  Ident__BASE_FILE__     = RegisterBuiltinMacro(*this, "__BASE_FILE__");
  Ident__INCLUDE_LEVEL__ = RegisterBuiltinMacro(*this, "__INCLUDE_LEVEL__");
  Ident__TIMESTAMP__     = RegisterBuiltinMacro(*this, "__TIMESTAMP__");

  // Clang Extensions.
  Ident__has_feature      = RegisterBuiltinMacro(*this, "__has_feature");
  Ident__has_builtin      = RegisterBuiltinMacro(*this, "__has_builtin");
  Ident__has_include      = RegisterBuiltinMacro(*this, "__has_include");
  Ident__has_include_next = RegisterBuiltinMacro(*this, "__has_include_next");
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
  else if (CurPTHLexer)
    Val = CurPTHLexer->isNextPPTokenLParen();
  else
    Val = CurTokenLexer->isNextTokenLParen();

  if (Val == 2) {
    // We have run off the end.  If it's a source file we don't
    // examine enclosing ones (C99 5.1.1.2p4).  Otherwise walk up the
    // macro stack.
    if (CurPPLexer)
      return false;
    for (unsigned i = IncludeMacroStack.size(); i != 0; --i) {
      IncludeStackInfo &Entry = IncludeMacroStack[i-1];
      if (Entry.TheLexer)
        Val = Entry.TheLexer->isNextPPTokenLParen();
      else if (Entry.ThePTHLexer)
        Val = Entry.ThePTHLexer->isNextPPTokenLParen();
      else
        Val = Entry.TheTokenLexer->isNextTokenLParen();

      if (Val != 2)
        break;

      // Ran off the end of a source file?
      if (Entry.ThePPLexer)
        return false;
    }
  }

  // Okay, if we know that the token is a '(', lex it and return.  Otherwise we
  // have found something that isn't a '(' or we found the end of the
  // translation unit.  In either case, return false.
  return Val == 1;
}

/// HandleMacroExpandedIdentifier - If an identifier token is read that is to be
/// expanded as a macro, handle it and return the next token as 'Identifier'.
bool Preprocessor::HandleMacroExpandedIdentifier(Token &Identifier,
                                                 MacroInfo *MI) {
  if (Callbacks) Callbacks->MacroExpands(Identifier, MI);

  // If this is a macro exapnsion in the "#if !defined(x)" line for the file,
  // then the macro could expand to different things in other contexts, we need
  // to disable the optimization in this case.
  if (CurPPLexer) CurPPLexer->MIOpt.ExpandedMacro();

  // If this is a builtin macro, like __LINE__ or _Pragma, handle it specially.
  if (MI->isBuiltinMacro()) {
    ExpandBuiltinMacro(Identifier);
    return false;
  }

  /// Args - If this is a function-like macro expansion, this contains,
  /// for each macro argument, the list of tokens that were provided to the
  /// invocation.
  MacroArgs *Args = 0;

  // Remember where the end of the instantiation occurred.  For an object-like
  // macro, this is the identifier.  For a function-like macro, this is the ')'.
  SourceLocation InstantiationEnd = Identifier.getLocation();

  // If this is a function-like macro, read the arguments.
  if (MI->isFunctionLike()) {
    // C99 6.10.3p10: If the preprocessing token immediately after the the macro
    // name isn't a '(', this macro should not be expanded.
    if (!isNextPPTokenLParen())
      return true;

    // Remember that we are now parsing the arguments to a macro invocation.
    // Preprocessor directives used inside macro arguments are not portable, and
    // this enables the warning.
    InMacroArgs = true;
    Args = ReadFunctionLikeMacroArgs(Identifier, MI, InstantiationEnd);

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
    if (Args) Args->destroy(*this);

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
                                           *this)) {
    // Otherwise, if this macro expands into a single trivially-expanded
    // token: expand it now.  This handles common cases like
    // "#define VAL 42".

    // No need for arg info.
    if (Args) Args->destroy(*this);

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

    // Update the tokens location to include both its instantiation and physical
    // locations.
    SourceLocation Loc =
      SourceMgr.createInstantiationLoc(Identifier.getLocation(), InstantiateLoc,
                                       InstantiationEnd,Identifier.getLength());
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
  EnterMacro(Identifier, InstantiationEnd, Args);

  // Now that the macro is at the top of the include stack, ask the
  // preprocessor to read the next token from it.
  Lex(Identifier);
  return false;
}

/// ReadFunctionLikeMacroArgs - After reading "MACRO" and knowing that the next
/// token is the '(' of the macro, this method is invoked to read all of the
/// actual arguments specified for the macro invocation.  This returns null on
/// error.
MacroArgs *Preprocessor::ReadFunctionLikeMacroArgs(Token &MacroName,
                                                   MacroInfo *MI,
                                                   SourceLocation &MacroEnd) {
  // The number of fixed arguments to parse.
  unsigned NumFixedArgsLeft = MI->getNumArgs();
  bool isVariadic = MI->isVariadic();

  // Outer loop, while there are more arguments, keep reading them.
  Token Tok;

  // Read arguments as unexpanded tokens.  This avoids issues, e.g., where
  // an argument value in a macro could expand to ',' or '(' or ')'.
  LexUnexpandedToken(Tok);
  assert(Tok.is(tok::l_paren) && "Error computing l-paren-ness?");

  // ArgTokens - Build up a list of tokens that make up each argument.  Each
  // argument is separated by an EOF token.  Use a SmallVector so we can avoid
  // heap allocations in the common case.
  llvm::SmallVector<Token, 64> ArgTokens;

  unsigned NumActuals = 0;
  while (Tok.isNot(tok::r_paren)) {
    assert((Tok.is(tok::l_paren) || Tok.is(tok::comma)) &&
           "only expect argument separators here");

    unsigned ArgTokenStart = ArgTokens.size();
    SourceLocation ArgStartLoc = Tok.getLocation();

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
        if (NumParens-- == 0) {
          MacroEnd = Tok.getLocation();
          break;
        }
      } else if (Tok.is(tok::l_paren)) {
        ++NumParens;
      } else if (Tok.is(tok::comma) && NumParens == 0) {
        // Comma ends this argument if there are more fixed arguments expected.
        // However, if this is a variadic macro, and this is part of the
        // variadic part, then the comma is just an argument token.
        if (!isVariadic) break;
        if (NumFixedArgsLeft > 1)
          break;
      } else if (Tok.is(tok::comment) && !KeepMacroComments) {
        // If this is a comment token in the argument list and we're just in
        // -C mode (not -CC mode), discard the comment.
        continue;
      } else if (Tok.getIdentifierInfo() != 0) {
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

    // If this was an empty argument list foo(), don't add this as an empty
    // argument.
    if (ArgTokens.empty() && Tok.getKind() == tok::r_paren)
      break;

    // If this is not a variadic macro, and too many args were specified, emit
    // an error.
    if (!isVariadic && NumFixedArgsLeft == 0) {
      if (ArgTokens.size() != ArgTokenStart)
        ArgStartLoc = ArgTokens[ArgTokenStart].getLocation();

      // Emit the diagnostic at the macro name in case there is a missing ).
      // Emitting it at the , could be far away from the macro name.
      Diag(ArgStartLoc, diag::err_too_many_args_in_macro_invoc);
      return 0;
    }

    // Empty arguments are standard in C99 and supported as an extension in
    // other modes.
    if (ArgTokens.size() == ArgTokenStart && !Features.C99)
      Diag(Tok, diag::ext_empty_fnmacro_arg);

    // Add a marker EOF token to the end of the token list for this argument.
    Token EOFTok;
    EOFTok.startToken();
    EOFTok.setKind(tok::eof);
    EOFTok.setLocation(Tok.getLocation());
    EOFTok.setLength(0);
    ArgTokens.push_back(EOFTok);
    ++NumActuals;
    assert(NumFixedArgsLeft != 0 && "Too many arguments parsed");
    --NumFixedArgsLeft;
  }

  // Okay, we either found the r_paren.  Check to see if we parsed too few
  // arguments.
  unsigned MinArgsExpected = MI->getNumArgs();

  // See MacroArgs instance var for description of this.
  bool isVarargsElided = false;

  if (NumActuals < MinArgsExpected) {
    // There are several cases where too few arguments is ok, handle them now.
    if (NumActuals == 0 && MinArgsExpected == 1) {
      // #define A(X)  or  #define A(...)   ---> A()

      // If there is exactly one argument, and that argument is missing,
      // then we have an empty "()" argument empty list.  This is fine, even if
      // the macro expects one argument (the argument is just empty).
      isVarargsElided = MI->isVariadic();
    } else if (MI->isVariadic() &&
               (NumActuals+1 == MinArgsExpected ||  // A(x, ...) -> A(X)
                (NumActuals == 0 && MinArgsExpected == 2))) {// A(x,...) -> A()
      // Varargs where the named vararg parameter is missing: ok as extension.
      // #define A(x, ...)
      // A("blah")
      Diag(Tok, diag::ext_missing_varargs_arg);

      // Remember this occurred, allowing us to elide the comma when used for
      // cases like:
      //   #define A(x, foo...) blah(a, ## foo)
      //   #define B(x, ...) blah(a, ## __VA_ARGS__)
      //   #define C(...) blah(a, ## __VA_ARGS__)
      //  A(x) B(x) C()
      isVarargsElided = true;
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

    // If we expect two arguments, add both as empty.
    if (NumActuals == 0 && MinArgsExpected == 2)
      ArgTokens.push_back(Tok);

  } else if (NumActuals > MinArgsExpected && !MI->isVariadic()) {
    // Emit the diagnostic at the macro name in case there is a missing ).
    // Emitting it at the , could be far away from the macro name.
    Diag(MacroName, diag::err_too_many_args_in_macro_invoc);
    return 0;
  }

  return MacroArgs::create(MI, ArgTokens.data(), ArgTokens.size(),
                           isVarargsElided, *this);
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

  Token TmpTok;
  TmpTok.startToken();
  PP.CreateString(TmpBuffer, strlen(TmpBuffer), TmpTok);
  DATELoc = TmpTok.getLocation();

  sprintf(TmpBuffer, "\"%02d:%02d:%02d\"", TM->tm_hour, TM->tm_min, TM->tm_sec);
  PP.CreateString(TmpBuffer, strlen(TmpBuffer), TmpTok);
  TIMELoc = TmpTok.getLocation();
}


/// HasFeature - Return true if we recognize and implement the specified feature
/// specified by the identifier.
static bool HasFeature(const Preprocessor &PP, const IdentifierInfo *II) {
  const LangOptions &LangOpts = PP.getLangOptions();

  return llvm::StringSwitch<bool>(II->getName())
           .Case("blocks", LangOpts.Blocks)
           .Case("cxx_rtti", LangOpts.RTTI)
           .Case("cxx_exceptions", LangOpts.Exceptions)
           .Case("objc_nonfragile_abi", LangOpts.ObjCNonFragileABI)
           .Case("attribute_overloadable", true)
           .Case("attribute_ext_vector_type", true)
           .Case("attribute_analyzer_noreturn", true)
           .Case("attribute_ns_returns_retained", true)
           .Case("attribute_cf_returns_retained", true)
           .Default(false);
}

/// EvaluateHasIncludeCommon - Process a '__has_include("path")'
/// or '__has_include_next("path")' expression.
/// Returns true if successful.
static bool EvaluateHasIncludeCommon(bool &Result, Token &Tok,
        IdentifierInfo *II, Preprocessor &PP,
        const DirectoryLookup *LookupFrom) {
  SourceLocation LParenLoc;

  // Get '('.
  PP.LexNonComment(Tok);

  // Ensure we have a '('.
  if (Tok.isNot(tok::l_paren)) {
    PP.Diag(Tok.getLocation(), diag::err_pp_missing_lparen) << II->getName();
    return false;
  }

  // Save '(' location for possible missing ')' message.
  LParenLoc = Tok.getLocation();

  // Get the file name.
  PP.getCurrentLexer()->LexIncludeFilename(Tok);

  // Reserve a buffer to get the spelling.
  llvm::SmallVector<char, 128> FilenameBuffer;
  const char *FilenameStart, *FilenameEnd;

  switch (Tok.getKind()) {
  case tok::eom:
    // If the token kind is EOM, the error has already been diagnosed.
    return false;

  case tok::angle_string_literal:
  case tok::string_literal: {
    FilenameBuffer.resize(Tok.getLength());
    FilenameStart = &FilenameBuffer[0];
    unsigned Len = PP.getSpelling(Tok, FilenameStart);
    FilenameEnd = FilenameStart+Len;
    break;
  }

  case tok::less:
    // This could be a <foo/bar.h> file coming from a macro expansion.  In this
    // case, glue the tokens together into FilenameBuffer and interpret those.
    FilenameBuffer.push_back('<');
    if (PP.ConcatenateIncludeName(FilenameBuffer))
      return false;   // Found <eom> but no ">"?  Diagnostic already emitted.
    FilenameStart = FilenameBuffer.data();
    FilenameEnd = FilenameStart + FilenameBuffer.size();
    break;
  default:
    PP.Diag(Tok.getLocation(), diag::err_pp_expects_filename);
    return false;
  }

  bool isAngled = PP.GetIncludeFilenameSpelling(Tok.getLocation(),
                                             FilenameStart, FilenameEnd);
  // If GetIncludeFilenameSpelling set the start ptr to null, there was an
  // error.
  if (FilenameStart == 0) {
    return false;
  }

  // Search include directories.
  const DirectoryLookup *CurDir;
  const FileEntry *File = PP.LookupFile(FilenameStart, FilenameEnd,
                                     isAngled, LookupFrom, CurDir);

  // Get the result value.  Result = true means the file exists.
  Result = File != 0;

  // Get ')'.
  PP.LexNonComment(Tok);

  // Ensure we have a trailing ).
  if (Tok.isNot(tok::r_paren)) {
    PP.Diag(Tok.getLocation(), diag::err_pp_missing_rparen) << II->getName();
    PP.Diag(LParenLoc, diag::note_matching) << "(";
    return false;
  }

  return true;
}

/// EvaluateHasInclude - Process a '__has_include("path")' expression.
/// Returns true if successful.
static bool EvaluateHasInclude(bool &Result, Token &Tok, IdentifierInfo *II,
                               Preprocessor &PP) {
  return(EvaluateHasIncludeCommon(Result, Tok, II, PP, NULL));
}

/// EvaluateHasIncludeNext - Process '__has_include_next("path")' expression.
/// Returns true if successful.
static bool EvaluateHasIncludeNext(bool &Result, Token &Tok,
                                   IdentifierInfo *II, Preprocessor &PP) {
  // __has_include_next is like __has_include, except that we start
  // searching after the current found directory.  If we can't do this,
  // issue a diagnostic.
  const DirectoryLookup *Lookup = PP.GetCurDirLookup();
  if (PP.isInPrimaryFile()) {
    Lookup = 0;
    PP.Diag(Tok, diag::pp_include_next_in_primary);
  } else if (Lookup == 0) {
    PP.Diag(Tok, diag::pp_include_next_absolute_path);
  } else {
    // Start looking up in the next directory.
    ++Lookup;
  }

  return(EvaluateHasIncludeCommon(Result, Tok, II, PP, Lookup));
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
    // C99 6.10.8: "__LINE__: The presumed line number (within the current
    // source file) of the current source line (an integer constant)".  This can
    // be affected by #line.
    SourceLocation Loc = Tok.getLocation();

    // Advance to the location of the first _, this might not be the first byte
    // of the token if it starts with an escaped newline.
    Loc = AdvanceToTokenCharacter(Loc, 0);

    // One wrinkle here is that GCC expands __LINE__ to location of the *end* of
    // a macro instantiation.  This doesn't matter for object-like macros, but
    // can matter for a function-like macro that expands to contain __LINE__.
    // Skip down through instantiation points until we find a file loc for the
    // end of the instantiation history.
    Loc = SourceMgr.getInstantiationRange(Loc).second;
    PresumedLoc PLoc = SourceMgr.getPresumedLoc(Loc);

    // __LINE__ expands to a simple numeric value.
    sprintf(TmpBuffer, "%u", PLoc.getLine());
    Tok.setKind(tok::numeric_constant);
    CreateString(TmpBuffer, strlen(TmpBuffer), Tok, Tok.getLocation());
  } else if (II == Ident__FILE__ || II == Ident__BASE_FILE__) {
    // C99 6.10.8: "__FILE__: The presumed name of the current source file (a
    // character string literal)". This can be affected by #line.
    PresumedLoc PLoc = SourceMgr.getPresumedLoc(Tok.getLocation());

    // __BASE_FILE__ is a GNU extension that returns the top of the presumed
    // #include stack instead of the current file.
    if (II == Ident__BASE_FILE__) {
      SourceLocation NextLoc = PLoc.getIncludeLoc();
      while (NextLoc.isValid()) {
        PLoc = SourceMgr.getPresumedLoc(NextLoc);
        NextLoc = PLoc.getIncludeLoc();
      }
    }

    // Escape this filename.  Turn '\' -> '\\' '"' -> '\"'
    std::string FN = PLoc.getFilename();
    FN = '"' + Lexer::Stringify(FN) + '"';
    Tok.setKind(tok::string_literal);
    CreateString(&FN[0], FN.size(), Tok, Tok.getLocation());
  } else if (II == Ident__DATE__) {
    if (!DATELoc.isValid())
      ComputeDATE_TIME(DATELoc, TIMELoc, *this);
    Tok.setKind(tok::string_literal);
    Tok.setLength(strlen("\"Mmm dd yyyy\""));
    Tok.setLocation(SourceMgr.createInstantiationLoc(DATELoc, Tok.getLocation(),
                                                     Tok.getLocation(),
                                                     Tok.getLength()));
  } else if (II == Ident__TIME__) {
    if (!TIMELoc.isValid())
      ComputeDATE_TIME(DATELoc, TIMELoc, *this);
    Tok.setKind(tok::string_literal);
    Tok.setLength(strlen("\"hh:mm:ss\""));
    Tok.setLocation(SourceMgr.createInstantiationLoc(TIMELoc, Tok.getLocation(),
                                                     Tok.getLocation(),
                                                     Tok.getLength()));
  } else if (II == Ident__INCLUDE_LEVEL__) {
    // Compute the presumed include depth of this token.  This can be affected
    // by GNU line markers.
    unsigned Depth = 0;

    PresumedLoc PLoc = SourceMgr.getPresumedLoc(Tok.getLocation());
    PLoc = SourceMgr.getPresumedLoc(PLoc.getIncludeLoc());
    for (; PLoc.isValid(); ++Depth)
      PLoc = SourceMgr.getPresumedLoc(PLoc.getIncludeLoc());

    // __INCLUDE_LEVEL__ expands to a simple numeric value.
    sprintf(TmpBuffer, "%u", Depth);
    Tok.setKind(tok::numeric_constant);
    CreateString(TmpBuffer, strlen(TmpBuffer), Tok, Tok.getLocation());
  } else if (II == Ident__TIMESTAMP__) {
    // MSVC, ICC, GCC, VisualAge C++ extension.  The generated string should be
    // of the form "Ddd Mmm dd hh::mm::ss yyyy", which is returned by asctime.

    // Get the file that we are lexing out of.  If we're currently lexing from
    // a macro, dig into the include stack.
    const FileEntry *CurFile = 0;
    PreprocessorLexer *TheLexer = getCurrentFileLexer();

    if (TheLexer)
      CurFile = SourceMgr.getFileEntryForID(TheLexer->getFileID());

    const char *Result;
    if (CurFile) {
      time_t TT = CurFile->getModificationTime();
      struct tm *TM = localtime(&TT);
      Result = asctime(TM);
    } else {
      Result = "??? ??? ?? ??:??:?? ????\n";
    }
    TmpBuffer[0] = '"';
    unsigned Len = strlen(Result);
    memcpy(TmpBuffer+1, Result, Len-1); // Copy string without the newline.
    TmpBuffer[Len] = '"';
    Tok.setKind(tok::string_literal);
    CreateString(TmpBuffer, Len+1, Tok, Tok.getLocation());
  } else if (II == Ident__COUNTER__) {
    // __COUNTER__ expands to a simple numeric value.
    sprintf(TmpBuffer, "%u", CounterValue++);
    Tok.setKind(tok::numeric_constant);
    CreateString(TmpBuffer, strlen(TmpBuffer), Tok, Tok.getLocation());
  } else if (II == Ident__has_feature ||
             II == Ident__has_builtin) {
    // The argument to these two builtins should be a parenthesized identifier.
    SourceLocation StartLoc = Tok.getLocation();

    bool IsValid = false;
    IdentifierInfo *FeatureII = 0;

    // Read the '('.
    Lex(Tok);
    if (Tok.is(tok::l_paren)) {
      // Read the identifier
      Lex(Tok);
      if (Tok.is(tok::identifier)) {
        FeatureII = Tok.getIdentifierInfo();

        // Read the ')'.
        Lex(Tok);
        if (Tok.is(tok::r_paren))
          IsValid = true;
      }
    }

    bool Value = false;
    if (!IsValid)
      Diag(StartLoc, diag::err_feature_check_malformed);
    else if (II == Ident__has_builtin) {
      // Check for a builtin is trivial.
      Value = FeatureII->getBuiltinID() != 0;
    } else {
      assert(II == Ident__has_feature && "Must be feature check");
      Value = HasFeature(*this, FeatureII);
    }

    sprintf(TmpBuffer, "%d", (int)Value);
    Tok.setKind(tok::numeric_constant);
    CreateString(TmpBuffer, strlen(TmpBuffer), Tok, Tok.getLocation());
  } else if (II == Ident__has_include ||
             II == Ident__has_include_next) {
    // The argument to these two builtins should be a parenthesized
    // file name string literal using angle brackets (<>) or
    // double-quotes ("").
    bool Value = false;
    bool IsValid;
    if (II == Ident__has_include)
      IsValid = EvaluateHasInclude(Value, Tok, II, *this);
    else
      IsValid = EvaluateHasIncludeNext(Value, Tok, II, *this);
    sprintf(TmpBuffer, "%d", (int)Value);
    Tok.setKind(tok::numeric_constant);
    CreateString(TmpBuffer, strlen(TmpBuffer), Tok, Tok.getLocation());
  } else {
    assert(0 && "Unknown identifier!");
  }
}
