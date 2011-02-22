//===--- Pragma.cpp - Pragma registration and handling --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PragmaHandler/PragmaTable interfaces and implements
// pragma related methods of the Preprocessor class.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/Pragma.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/LiteralSupport.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
using namespace clang;

// Out-of-line destructor to provide a home for the class.
PragmaHandler::~PragmaHandler() {
}

//===----------------------------------------------------------------------===//
// EmptyPragmaHandler Implementation.
//===----------------------------------------------------------------------===//

EmptyPragmaHandler::EmptyPragmaHandler() {}

void EmptyPragmaHandler::HandlePragma(Preprocessor &PP, 
                                      PragmaIntroducerKind Introducer,
                                      Token &FirstToken) {}

//===----------------------------------------------------------------------===//
// PragmaNamespace Implementation.
//===----------------------------------------------------------------------===//


PragmaNamespace::~PragmaNamespace() {
  for (llvm::StringMap<PragmaHandler*>::iterator
         I = Handlers.begin(), E = Handlers.end(); I != E; ++I)
    delete I->second;
}

/// FindHandler - Check to see if there is already a handler for the
/// specified name.  If not, return the handler for the null identifier if it
/// exists, otherwise return null.  If IgnoreNull is true (the default) then
/// the null handler isn't returned on failure to match.
PragmaHandler *PragmaNamespace::FindHandler(llvm::StringRef Name,
                                            bool IgnoreNull) const {
  if (PragmaHandler *Handler = Handlers.lookup(Name))
    return Handler;
  return IgnoreNull ? 0 : Handlers.lookup(llvm::StringRef());
}

void PragmaNamespace::AddPragma(PragmaHandler *Handler) {
  assert(!Handlers.lookup(Handler->getName()) &&
         "A handler with this name is already registered in this namespace");
  llvm::StringMapEntry<PragmaHandler *> &Entry =
    Handlers.GetOrCreateValue(Handler->getName());
  Entry.setValue(Handler);
}

void PragmaNamespace::RemovePragmaHandler(PragmaHandler *Handler) {
  assert(Handlers.lookup(Handler->getName()) &&
         "Handler not registered in this namespace");
  Handlers.erase(Handler->getName());
}

void PragmaNamespace::HandlePragma(Preprocessor &PP, 
                                   PragmaIntroducerKind Introducer,
                                   Token &Tok) {
  // Read the 'namespace' that the directive is in, e.g. STDC.  Do not macro
  // expand it, the user can have a STDC #define, that should not affect this.
  PP.LexUnexpandedToken(Tok);

  // Get the handler for this token.  If there is no handler, ignore the pragma.
  PragmaHandler *Handler
    = FindHandler(Tok.getIdentifierInfo() ? Tok.getIdentifierInfo()->getName()
                                          : llvm::StringRef(),
                  /*IgnoreNull=*/false);
  if (Handler == 0) {
    PP.Diag(Tok, diag::warn_pragma_ignored);
    return;
  }

  // Otherwise, pass it down.
  Handler->HandlePragma(PP, Introducer, Tok);
}

//===----------------------------------------------------------------------===//
// Preprocessor Pragma Directive Handling.
//===----------------------------------------------------------------------===//

/// HandlePragmaDirective - The "#pragma" directive has been parsed.  Lex the
/// rest of the pragma, passing it to the registered pragma handlers.
void Preprocessor::HandlePragmaDirective(unsigned Introducer) {
  ++NumPragma;

  // Invoke the first level of pragma handlers which reads the namespace id.
  Token Tok;
  PragmaHandlers->HandlePragma(*this, PragmaIntroducerKind(Introducer), Tok);

  // If the pragma handler didn't read the rest of the line, consume it now.
  if ((CurTokenLexer && CurTokenLexer->isParsingPreprocessorDirective()) 
   || (CurPPLexer && CurPPLexer->ParsingPreprocessorDirective))
    DiscardUntilEndOfDirective();
}

/// Handle_Pragma - Read a _Pragma directive, slice it up, process it, then
/// return the first token after the directive.  The _Pragma token has just
/// been read into 'Tok'.
void Preprocessor::Handle_Pragma(Token &Tok) {
  // Remember the pragma token location.
  SourceLocation PragmaLoc = Tok.getLocation();

  // Read the '('.
  Lex(Tok);
  if (Tok.isNot(tok::l_paren)) {
    Diag(PragmaLoc, diag::err__Pragma_malformed);
    return;
  }

  // Read the '"..."'.
  Lex(Tok);
  if (Tok.isNot(tok::string_literal) && Tok.isNot(tok::wide_string_literal)) {
    Diag(PragmaLoc, diag::err__Pragma_malformed);
    return;
  }

  // Remember the string.
  std::string StrVal = getSpelling(Tok);

  // Read the ')'.
  Lex(Tok);
  if (Tok.isNot(tok::r_paren)) {
    Diag(PragmaLoc, diag::err__Pragma_malformed);
    return;
  }

  SourceLocation RParenLoc = Tok.getLocation();

  // The _Pragma is lexically sound.  Destringize according to C99 6.10.9.1:
  // "The string literal is destringized by deleting the L prefix, if present,
  // deleting the leading and trailing double-quotes, replacing each escape
  // sequence \" by a double-quote, and replacing each escape sequence \\ by a
  // single backslash."
  if (StrVal[0] == 'L')  // Remove L prefix.
    StrVal.erase(StrVal.begin());
  assert(StrVal[0] == '"' && StrVal[StrVal.size()-1] == '"' &&
         "Invalid string token!");

  // Remove the front quote, replacing it with a space, so that the pragma
  // contents appear to have a space before them.
  StrVal[0] = ' ';

  // Replace the terminating quote with a \n.
  StrVal[StrVal.size()-1] = '\n';

  // Remove escaped quotes and escapes.
  for (unsigned i = 0, e = StrVal.size(); i != e-1; ++i) {
    if (StrVal[i] == '\\' &&
        (StrVal[i+1] == '\\' || StrVal[i+1] == '"')) {
      // \\ -> '\' and \" -> '"'.
      StrVal.erase(StrVal.begin()+i);
      --e;
    }
  }
  
  // Plop the string (including the newline and trailing null) into a buffer
  // where we can lex it.
  Token TmpTok;
  TmpTok.startToken();
  CreateString(&StrVal[0], StrVal.size(), TmpTok);
  SourceLocation TokLoc = TmpTok.getLocation();

  // Make and enter a lexer object so that we lex and expand the tokens just
  // like any others.
  Lexer *TL = Lexer::Create_PragmaLexer(TokLoc, PragmaLoc, RParenLoc,
                                        StrVal.size(), *this);

  EnterSourceFileWithLexer(TL, 0);

  // With everything set up, lex this as a #pragma directive.
  HandlePragmaDirective(PIK__Pragma);

  // Finally, return whatever came after the pragma directive.
  return Lex(Tok);
}

/// HandleMicrosoft__pragma - Like Handle_Pragma except the pragma text
/// is not enclosed within a string literal.
void Preprocessor::HandleMicrosoft__pragma(Token &Tok) {
  // Remember the pragma token location.
  SourceLocation PragmaLoc = Tok.getLocation();

  // Read the '('.
  Lex(Tok);
  if (Tok.isNot(tok::l_paren)) {
    Diag(PragmaLoc, diag::err__Pragma_malformed);
    return;
  }

  // Get the tokens enclosed within the __pragma(), as well as the final ')'.
  llvm::SmallVector<Token, 32> PragmaToks;
  int NumParens = 0;
  Lex(Tok);
  while (Tok.isNot(tok::eof)) {
    PragmaToks.push_back(Tok);
    if (Tok.is(tok::l_paren))
      NumParens++;
    else if (Tok.is(tok::r_paren) && NumParens-- == 0)
      break;
    Lex(Tok);
  }

  if (Tok.is(tok::eof)) {
    Diag(PragmaLoc, diag::err_unterminated___pragma);
    return;
  }

  PragmaToks.front().setFlag(Token::LeadingSpace);

  // Replace the ')' with an EOM to mark the end of the pragma.
  PragmaToks.back().setKind(tok::eom);

  Token *TokArray = new Token[PragmaToks.size()];
  std::copy(PragmaToks.begin(), PragmaToks.end(), TokArray);

  // Push the tokens onto the stack.
  EnterTokenStream(TokArray, PragmaToks.size(), true, true);

  // With everything set up, lex this as a #pragma directive.
  HandlePragmaDirective(PIK___pragma);

  // Finally, return whatever came after the pragma directive.
  return Lex(Tok);
}

/// HandlePragmaOnce - Handle #pragma once.  OnceTok is the 'once'.
///
void Preprocessor::HandlePragmaOnce(Token &OnceTok) {
  if (isInPrimaryFile()) {
    Diag(OnceTok, diag::pp_pragma_once_in_main_file);
    return;
  }

  // Get the current file lexer we're looking at.  Ignore _Pragma 'files' etc.
  // Mark the file as a once-only file now.
  HeaderInfo.MarkFileIncludeOnce(getCurrentFileLexer()->getFileEntry());
}

void Preprocessor::HandlePragmaMark() {
  assert(CurPPLexer && "No current lexer?");
  if (CurLexer)
    CurLexer->ReadToEndOfLine();
  else
    CurPTHLexer->DiscardToEndOfLine();
}


/// HandlePragmaPoison - Handle #pragma GCC poison.  PoisonTok is the 'poison'.
///
void Preprocessor::HandlePragmaPoison(Token &PoisonTok) {
  Token Tok;

  while (1) {
    // Read the next token to poison.  While doing this, pretend that we are
    // skipping while reading the identifier to poison.
    // This avoids errors on code like:
    //   #pragma GCC poison X
    //   #pragma GCC poison X
    if (CurPPLexer) CurPPLexer->LexingRawMode = true;
    LexUnexpandedToken(Tok);
    if (CurPPLexer) CurPPLexer->LexingRawMode = false;

    // If we reached the end of line, we're done.
    if (Tok.is(tok::eom)) return;

    // Can only poison identifiers.
    if (Tok.isNot(tok::raw_identifier)) {
      Diag(Tok, diag::err_pp_invalid_poison);
      return;
    }

    // Look up the identifier info for the token.  We disabled identifier lookup
    // by saying we're skipping contents, so we need to do this manually.
    IdentifierInfo *II = LookUpIdentifierInfo(Tok);

    // Already poisoned.
    if (II->isPoisoned()) continue;

    // If this is a macro identifier, emit a warning.
    if (II->hasMacroDefinition())
      Diag(Tok, diag::pp_poisoning_existing_macro);

    // Finally, poison it!
    II->setIsPoisoned();
  }
}

/// HandlePragmaSystemHeader - Implement #pragma GCC system_header.  We know
/// that the whole directive has been parsed.
void Preprocessor::HandlePragmaSystemHeader(Token &SysHeaderTok) {
  if (isInPrimaryFile()) {
    Diag(SysHeaderTok, diag::pp_pragma_sysheader_in_main_file);
    return;
  }

  // Get the current file lexer we're looking at.  Ignore _Pragma 'files' etc.
  PreprocessorLexer *TheLexer = getCurrentFileLexer();

  // Mark the file as a system header.
  HeaderInfo.MarkFileSystemHeader(TheLexer->getFileEntry());


  PresumedLoc PLoc = SourceMgr.getPresumedLoc(SysHeaderTok.getLocation());
  if (PLoc.isInvalid())
    return;
  
  unsigned FilenameLen = strlen(PLoc.getFilename());
  unsigned FilenameID = SourceMgr.getLineTableFilenameID(PLoc.getFilename(),
                                                         FilenameLen);

  // Emit a line marker.  This will change any source locations from this point
  // forward to realize they are in a system header.
  // Create a line note with this information.
  SourceMgr.AddLineNote(SysHeaderTok.getLocation(), PLoc.getLine(), FilenameID,
                        false, false, true, false);

  // Notify the client, if desired, that we are in a new source file.
  if (Callbacks)
    Callbacks->FileChanged(SysHeaderTok.getLocation(),
                           PPCallbacks::SystemHeaderPragma, SrcMgr::C_System);
}

/// HandlePragmaDependency - Handle #pragma GCC dependency "foo" blah.
///
void Preprocessor::HandlePragmaDependency(Token &DependencyTok) {
  Token FilenameTok;
  CurPPLexer->LexIncludeFilename(FilenameTok);

  // If the token kind is EOM, the error has already been diagnosed.
  if (FilenameTok.is(tok::eom))
    return;

  // Reserve a buffer to get the spelling.
  llvm::SmallString<128> FilenameBuffer;
  bool Invalid = false;
  llvm::StringRef Filename = getSpelling(FilenameTok, FilenameBuffer, &Invalid);
  if (Invalid)
    return;

  bool isAngled =
    GetIncludeFilenameSpelling(FilenameTok.getLocation(), Filename);
  // If GetIncludeFilenameSpelling set the start ptr to null, there was an
  // error.
  if (Filename.empty())
    return;

  // Search include directories for this file.
  const DirectoryLookup *CurDir;
  const FileEntry *File = LookupFile(Filename, isAngled, 0, CurDir);
  if (File == 0) {
    Diag(FilenameTok, diag::err_pp_file_not_found) << Filename;
    return;
  }

  const FileEntry *CurFile = getCurrentFileLexer()->getFileEntry();

  // If this file is older than the file it depends on, emit a diagnostic.
  if (CurFile && CurFile->getModificationTime() < File->getModificationTime()) {
    // Lex tokens at the end of the message and include them in the message.
    std::string Message;
    Lex(DependencyTok);
    while (DependencyTok.isNot(tok::eom)) {
      Message += getSpelling(DependencyTok) + " ";
      Lex(DependencyTok);
    }

    // Remove the trailing ' ' if present.
    if (!Message.empty())
      Message.erase(Message.end()-1);
    Diag(FilenameTok, diag::pp_out_of_date_dependency) << Message;
  }
}

/// HandlePragmaComment - Handle the microsoft #pragma comment extension.  The
/// syntax is:
///   #pragma comment(linker, "foo")
/// 'linker' is one of five identifiers: compiler, exestr, lib, linker, user.
/// "foo" is a string, which is fully macro expanded, and permits string
/// concatenation, embedded escape characters etc.  See MSDN for more details.
void Preprocessor::HandlePragmaComment(Token &Tok) {
  SourceLocation CommentLoc = Tok.getLocation();
  Lex(Tok);
  if (Tok.isNot(tok::l_paren)) {
    Diag(CommentLoc, diag::err_pragma_comment_malformed);
    return;
  }

  // Read the identifier.
  Lex(Tok);
  if (Tok.isNot(tok::identifier)) {
    Diag(CommentLoc, diag::err_pragma_comment_malformed);
    return;
  }

  // Verify that this is one of the 5 whitelisted options.
  // FIXME: warn that 'exestr' is deprecated.
  const IdentifierInfo *II = Tok.getIdentifierInfo();
  if (!II->isStr("compiler") && !II->isStr("exestr") && !II->isStr("lib") &&
      !II->isStr("linker") && !II->isStr("user")) {
    Diag(Tok.getLocation(), diag::err_pragma_comment_unknown_kind);
    return;
  }

  // Read the optional string if present.
  Lex(Tok);
  std::string ArgumentString;
  if (Tok.is(tok::comma)) {
    Lex(Tok); // eat the comma.

    // We need at least one string.
    if (Tok.isNot(tok::string_literal)) {
      Diag(Tok.getLocation(), diag::err_pragma_comment_malformed);
      return;
    }

    // String concatenation allows multiple strings, which can even come from
    // macro expansion.
    // "foo " "bar" "Baz"
    llvm::SmallVector<Token, 4> StrToks;
    while (Tok.is(tok::string_literal)) {
      StrToks.push_back(Tok);
      Lex(Tok);
    }

    // Concatenate and parse the strings.
    StringLiteralParser Literal(&StrToks[0], StrToks.size(), *this);
    assert(!Literal.AnyWide && "Didn't allow wide strings in");
    if (Literal.hadError)
      return;
    if (Literal.Pascal) {
      Diag(StrToks[0].getLocation(), diag::err_pragma_comment_malformed);
      return;
    }

    ArgumentString = std::string(Literal.GetString(),
                                 Literal.GetString()+Literal.GetStringLength());
  }

  // FIXME: If the kind is "compiler" warn if the string is present (it is
  // ignored).
  // FIXME: 'lib' requires a comment string.
  // FIXME: 'linker' requires a comment string, and has a specific list of
  // things that are allowable.

  if (Tok.isNot(tok::r_paren)) {
    Diag(Tok.getLocation(), diag::err_pragma_comment_malformed);
    return;
  }
  Lex(Tok);  // eat the r_paren.

  if (Tok.isNot(tok::eom)) {
    Diag(Tok.getLocation(), diag::err_pragma_comment_malformed);
    return;
  }

  // If the pragma is lexically sound, notify any interested PPCallbacks.
  if (Callbacks)
    Callbacks->PragmaComment(CommentLoc, II, ArgumentString);
}

/// HandlePragmaMessage - Handle the microsoft and gcc #pragma message
/// extension.  The syntax is:
///   #pragma message(string)
/// OR, in GCC mode:
///   #pragma message string
/// string is a string, which is fully macro expanded, and permits string
/// concatenation, embedded escape characters, etc... See MSDN for more details.
void Preprocessor::HandlePragmaMessage(Token &Tok) {
  SourceLocation MessageLoc = Tok.getLocation();
  Lex(Tok);
  bool ExpectClosingParen = false;
  switch (Tok.getKind()) {
  case tok::l_paren:
    // We have a MSVC style pragma message.
    ExpectClosingParen = true;
    // Read the string.
    Lex(Tok);
    break;
  case tok::string_literal:
    // We have a GCC style pragma message, and we just read the string.
    break;
  default:
    Diag(MessageLoc, diag::err_pragma_message_malformed);
    return;
  }

  // We need at least one string.
  if (Tok.isNot(tok::string_literal)) {
    Diag(Tok.getLocation(), diag::err_pragma_message_malformed);
    return;
  }

  // String concatenation allows multiple strings, which can even come from
  // macro expansion.
  // "foo " "bar" "Baz"
  llvm::SmallVector<Token, 4> StrToks;
  while (Tok.is(tok::string_literal)) {
    StrToks.push_back(Tok);
    Lex(Tok);
  }

  // Concatenate and parse the strings.
  StringLiteralParser Literal(&StrToks[0], StrToks.size(), *this);
  assert(!Literal.AnyWide && "Didn't allow wide strings in");
  if (Literal.hadError)
    return;
  if (Literal.Pascal) {
    Diag(StrToks[0].getLocation(), diag::err_pragma_message_malformed);
    return;
  }

  llvm::StringRef MessageString(Literal.GetString(), Literal.GetStringLength());

  if (ExpectClosingParen) {
    if (Tok.isNot(tok::r_paren)) {
      Diag(Tok.getLocation(), diag::err_pragma_message_malformed);
      return;
    }
    Lex(Tok);  // eat the r_paren.
  }

  if (Tok.isNot(tok::eom)) {
    Diag(Tok.getLocation(), diag::err_pragma_message_malformed);
    return;
  }

  // Output the message.
  Diag(MessageLoc, diag::warn_pragma_message) << MessageString;

  // If the pragma is lexically sound, notify any interested PPCallbacks.
  if (Callbacks)
    Callbacks->PragmaMessage(MessageLoc, MessageString);
}

/// ParsePragmaPushOrPopMacro - Handle parsing of pragma push_macro/pop_macro.  
/// Return the IdentifierInfo* associated with the macro to push or pop.
IdentifierInfo *Preprocessor::ParsePragmaPushOrPopMacro(Token &Tok) {
  // Remember the pragma token location.
  Token PragmaTok = Tok;

  // Read the '('.
  Lex(Tok);
  if (Tok.isNot(tok::l_paren)) {
    Diag(PragmaTok.getLocation(), diag::err_pragma_push_pop_macro_malformed)
      << getSpelling(PragmaTok);
    return 0;
  }

  // Read the macro name string.
  Lex(Tok);
  if (Tok.isNot(tok::string_literal)) {
    Diag(PragmaTok.getLocation(), diag::err_pragma_push_pop_macro_malformed)
      << getSpelling(PragmaTok);
    return 0;
  }

  // Remember the macro string.
  std::string StrVal = getSpelling(Tok);

  // Read the ')'.
  Lex(Tok);
  if (Tok.isNot(tok::r_paren)) {
    Diag(PragmaTok.getLocation(), diag::err_pragma_push_pop_macro_malformed)
      << getSpelling(PragmaTok);
    return 0;
  }

  assert(StrVal[0] == '"' && StrVal[StrVal.size()-1] == '"' &&
         "Invalid string token!");

  // Create a Token from the string.
  Token MacroTok;
  MacroTok.startToken();
  MacroTok.setKind(tok::raw_identifier);
  CreateString(&StrVal[1], StrVal.size() - 2, MacroTok);

  // Get the IdentifierInfo of MacroToPushTok.
  return LookUpIdentifierInfo(MacroTok);
}

/// HandlePragmaPushMacro - Handle #pragma push_macro.  
/// The syntax is:
///   #pragma push_macro("macro")
void Preprocessor::HandlePragmaPushMacro(Token &PushMacroTok) {
  // Parse the pragma directive and get the macro IdentifierInfo*.
  IdentifierInfo *IdentInfo = ParsePragmaPushOrPopMacro(PushMacroTok);
  if (!IdentInfo) return;

  // Get the MacroInfo associated with IdentInfo.
  MacroInfo *MI = getMacroInfo(IdentInfo);
 
  MacroInfo *MacroCopyToPush = 0;
  if (MI) {
    // Make a clone of MI.
    MacroCopyToPush = CloneMacroInfo(*MI);
    
    // Allow the original MacroInfo to be redefined later.
    MI->setIsAllowRedefinitionsWithoutWarning(true);
  }

  // Push the cloned MacroInfo so we can retrieve it later.
  PragmaPushMacroInfo[IdentInfo].push_back(MacroCopyToPush);
}

/// HandlePragmaPopMacro - Handle #pragma pop_macro.  
/// The syntax is:
///   #pragma pop_macro("macro")
void Preprocessor::HandlePragmaPopMacro(Token &PopMacroTok) {
  SourceLocation MessageLoc = PopMacroTok.getLocation();

  // Parse the pragma directive and get the macro IdentifierInfo*.
  IdentifierInfo *IdentInfo = ParsePragmaPushOrPopMacro(PopMacroTok);
  if (!IdentInfo) return;

  // Find the vector<MacroInfo*> associated with the macro.
  llvm::DenseMap<IdentifierInfo*, std::vector<MacroInfo*> >::iterator iter =
    PragmaPushMacroInfo.find(IdentInfo);
  if (iter != PragmaPushMacroInfo.end()) {
    // Release the MacroInfo currently associated with IdentInfo.
    MacroInfo *CurrentMI = getMacroInfo(IdentInfo);
    if (CurrentMI) {
      if (CurrentMI->isWarnIfUnused())
        WarnUnusedMacroLocs.erase(CurrentMI->getDefinitionLoc());
      ReleaseMacroInfo(CurrentMI);
    }

    // Get the MacroInfo we want to reinstall.
    MacroInfo *MacroToReInstall = iter->second.back();

    // Reinstall the previously pushed macro.
    setMacroInfo(IdentInfo, MacroToReInstall);

    // Pop PragmaPushMacroInfo stack.
    iter->second.pop_back();
    if (iter->second.size() == 0)
      PragmaPushMacroInfo.erase(iter);
  } else {
    Diag(MessageLoc, diag::warn_pragma_pop_macro_no_push)
      << IdentInfo->getName();
  }
}

/// AddPragmaHandler - Add the specified pragma handler to the preprocessor.
/// If 'Namespace' is non-null, then it is a token required to exist on the
/// pragma line before the pragma string starts, e.g. "STDC" or "GCC".
void Preprocessor::AddPragmaHandler(llvm::StringRef Namespace,
                                    PragmaHandler *Handler) {
  PragmaNamespace *InsertNS = PragmaHandlers;

  // If this is specified to be in a namespace, step down into it.
  if (!Namespace.empty()) {
    // If there is already a pragma handler with the name of this namespace,
    // we either have an error (directive with the same name as a namespace) or
    // we already have the namespace to insert into.
    if (PragmaHandler *Existing = PragmaHandlers->FindHandler(Namespace)) {
      InsertNS = Existing->getIfNamespace();
      assert(InsertNS != 0 && "Cannot have a pragma namespace and pragma"
             " handler with the same name!");
    } else {
      // Otherwise, this namespace doesn't exist yet, create and insert the
      // handler for it.
      InsertNS = new PragmaNamespace(Namespace);
      PragmaHandlers->AddPragma(InsertNS);
    }
  }

  // Check to make sure we don't already have a pragma for this identifier.
  assert(!InsertNS->FindHandler(Handler->getName()) &&
         "Pragma handler already exists for this identifier!");
  InsertNS->AddPragma(Handler);
}

/// RemovePragmaHandler - Remove the specific pragma handler from the
/// preprocessor. If \arg Namespace is non-null, then it should be the
/// namespace that \arg Handler was added to. It is an error to remove
/// a handler that has not been registered.
void Preprocessor::RemovePragmaHandler(llvm::StringRef Namespace,
                                       PragmaHandler *Handler) {
  PragmaNamespace *NS = PragmaHandlers;

  // If this is specified to be in a namespace, step down into it.
  if (!Namespace.empty()) {
    PragmaHandler *Existing = PragmaHandlers->FindHandler(Namespace);
    assert(Existing && "Namespace containing handler does not exist!");

    NS = Existing->getIfNamespace();
    assert(NS && "Invalid namespace, registered as a regular pragma handler!");
  }

  NS->RemovePragmaHandler(Handler);

  // If this is a non-default namespace and it is now empty, remove
  // it.
  if (NS != PragmaHandlers && NS->IsEmpty())
    PragmaHandlers->RemovePragmaHandler(NS);
}

bool Preprocessor::LexOnOffSwitch(tok::OnOffSwitch &Result) {
  Token Tok;
  LexUnexpandedToken(Tok);

  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::ext_on_off_switch_syntax);
    return true;
  }
  IdentifierInfo *II = Tok.getIdentifierInfo();
  if (II->isStr("ON"))
    Result = tok::OOS_ON;
  else if (II->isStr("OFF"))
    Result = tok::OOS_OFF;
  else if (II->isStr("DEFAULT"))
    Result = tok::OOS_DEFAULT;
  else {
    Diag(Tok, diag::ext_on_off_switch_syntax);
    return true;
  }

  // Verify that this is followed by EOM.
  LexUnexpandedToken(Tok);
  if (Tok.isNot(tok::eom))
    Diag(Tok, diag::ext_pragma_syntax_eom);
  return false;
}

namespace {
/// PragmaOnceHandler - "#pragma once" marks the file as atomically included.
struct PragmaOnceHandler : public PragmaHandler {
  PragmaOnceHandler() : PragmaHandler("once") {}
  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &OnceTok) {
    PP.CheckEndOfDirective("pragma once");
    PP.HandlePragmaOnce(OnceTok);
  }
};

/// PragmaMarkHandler - "#pragma mark ..." is ignored by the compiler, and the
/// rest of the line is not lexed.
struct PragmaMarkHandler : public PragmaHandler {
  PragmaMarkHandler() : PragmaHandler("mark") {}
  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &MarkTok) {
    PP.HandlePragmaMark();
  }
};

/// PragmaPoisonHandler - "#pragma poison x" marks x as not usable.
struct PragmaPoisonHandler : public PragmaHandler {
  PragmaPoisonHandler() : PragmaHandler("poison") {}
  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &PoisonTok) {
    PP.HandlePragmaPoison(PoisonTok);
  }
};

/// PragmaSystemHeaderHandler - "#pragma system_header" marks the current file
/// as a system header, which silences warnings in it.
struct PragmaSystemHeaderHandler : public PragmaHandler {
  PragmaSystemHeaderHandler() : PragmaHandler("system_header") {}
  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &SHToken) {
    PP.HandlePragmaSystemHeader(SHToken);
    PP.CheckEndOfDirective("pragma");
  }
};
struct PragmaDependencyHandler : public PragmaHandler {
  PragmaDependencyHandler() : PragmaHandler("dependency") {}
  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &DepToken) {
    PP.HandlePragmaDependency(DepToken);
  }
};

struct PragmaDebugHandler : public PragmaHandler {
  PragmaDebugHandler() : PragmaHandler("__debug") {}
  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &DepToken) {
    Token Tok;
    PP.LexUnexpandedToken(Tok);
    if (Tok.isNot(tok::identifier)) {
      PP.Diag(Tok, diag::warn_pragma_diagnostic_invalid);
      return;
    }
    IdentifierInfo *II = Tok.getIdentifierInfo();

    if (II->isStr("assert")) {
      assert(0 && "This is an assertion!");
    } else if (II->isStr("crash")) {
      *(volatile int*) 0x11 = 0;
    } else if (II->isStr("llvm_fatal_error")) {
      llvm::report_fatal_error("#pragma clang __debug llvm_fatal_error");
    } else if (II->isStr("llvm_unreachable")) {
      llvm_unreachable("#pragma clang __debug llvm_unreachable");
    } else if (II->isStr("overflow_stack")) {
      DebugOverflowStack();
    } else if (II->isStr("handle_crash")) {
      llvm::CrashRecoveryContext *CRC =llvm::CrashRecoveryContext::GetCurrent();
      if (CRC)
        CRC->HandleCrash();
    } else {
      PP.Diag(Tok, diag::warn_pragma_debug_unexpected_command)
        << II->getName();
    }
  }

  void DebugOverflowStack() {
    DebugOverflowStack();
  }
};

/// PragmaDiagnosticHandler - e.g. '#pragma GCC diagnostic ignored "-Wformat"'
struct PragmaDiagnosticHandler : public PragmaHandler {
public:
  explicit PragmaDiagnosticHandler() : PragmaHandler("diagnostic") {}
  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &DiagToken) {
    SourceLocation DiagLoc = DiagToken.getLocation();
    Token Tok;
    PP.LexUnexpandedToken(Tok);
    if (Tok.isNot(tok::identifier)) {
      PP.Diag(Tok, diag::warn_pragma_diagnostic_invalid);
      return;
    }
    IdentifierInfo *II = Tok.getIdentifierInfo();

    diag::Mapping Map;
    if (II->isStr("warning"))
      Map = diag::MAP_WARNING;
    else if (II->isStr("error"))
      Map = diag::MAP_ERROR;
    else if (II->isStr("ignored"))
      Map = diag::MAP_IGNORE;
    else if (II->isStr("fatal"))
      Map = diag::MAP_FATAL;
    else if (II->isStr("pop")) {
      if (!PP.getDiagnostics().popMappings(DiagLoc))
        PP.Diag(Tok, diag::warn_pragma_diagnostic_cannot_pop);

      return;
    } else if (II->isStr("push")) {
      PP.getDiagnostics().pushMappings(DiagLoc);
      return;
    } else {
      PP.Diag(Tok, diag::warn_pragma_diagnostic_invalid);
      return;
    }

    PP.LexUnexpandedToken(Tok);

    // We need at least one string.
    if (Tok.isNot(tok::string_literal)) {
      PP.Diag(Tok.getLocation(), diag::warn_pragma_diagnostic_invalid_token);
      return;
    }

    // String concatenation allows multiple strings, which can even come from
    // macro expansion.
    // "foo " "bar" "Baz"
    llvm::SmallVector<Token, 4> StrToks;
    while (Tok.is(tok::string_literal)) {
      StrToks.push_back(Tok);
      PP.LexUnexpandedToken(Tok);
    }

    if (Tok.isNot(tok::eom)) {
      PP.Diag(Tok.getLocation(), diag::warn_pragma_diagnostic_invalid_token);
      return;
    }

    // Concatenate and parse the strings.
    StringLiteralParser Literal(&StrToks[0], StrToks.size(), PP);
    assert(!Literal.AnyWide && "Didn't allow wide strings in");
    if (Literal.hadError)
      return;
    if (Literal.Pascal) {
      PP.Diag(Tok, diag::warn_pragma_diagnostic_invalid);
      return;
    }

    std::string WarningName(Literal.GetString(),
                            Literal.GetString()+Literal.GetStringLength());

    if (WarningName.size() < 3 || WarningName[0] != '-' ||
        WarningName[1] != 'W') {
      PP.Diag(StrToks[0].getLocation(),
              diag::warn_pragma_diagnostic_invalid_option);
      return;
    }

    if (PP.getDiagnostics().setDiagnosticGroupMapping(WarningName.c_str()+2,
                                                      Map, DiagLoc))
      PP.Diag(StrToks[0].getLocation(),
              diag::warn_pragma_diagnostic_unknown_warning) << WarningName;
  }
};

/// PragmaCommentHandler - "#pragma comment ...".
struct PragmaCommentHandler : public PragmaHandler {
  PragmaCommentHandler() : PragmaHandler("comment") {}
  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &CommentTok) {
    PP.HandlePragmaComment(CommentTok);
  }
};

/// PragmaMessageHandler - "#pragma message("...")".
struct PragmaMessageHandler : public PragmaHandler {
  PragmaMessageHandler() : PragmaHandler("message") {}
  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &CommentTok) {
    PP.HandlePragmaMessage(CommentTok);
  }
};

/// PragmaPushMacroHandler - "#pragma push_macro" saves the value of the
/// macro on the top of the stack.
struct PragmaPushMacroHandler : public PragmaHandler {
  PragmaPushMacroHandler() : PragmaHandler("push_macro") {}
  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &PushMacroTok) {
    PP.HandlePragmaPushMacro(PushMacroTok);
  }
};


/// PragmaPopMacroHandler - "#pragma pop_macro" sets the value of the
/// macro to the value on the top of the stack.
struct PragmaPopMacroHandler : public PragmaHandler {
  PragmaPopMacroHandler() : PragmaHandler("pop_macro") {}
  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &PopMacroTok) {
    PP.HandlePragmaPopMacro(PopMacroTok);
  }
};

// Pragma STDC implementations.

/// PragmaSTDC_FENV_ACCESSHandler - "#pragma STDC FENV_ACCESS ...".
struct PragmaSTDC_FENV_ACCESSHandler : public PragmaHandler {
  PragmaSTDC_FENV_ACCESSHandler() : PragmaHandler("FENV_ACCESS") {}
  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &Tok) {
    tok::OnOffSwitch OOS;
    if (PP.LexOnOffSwitch(OOS))
     return;
    if (OOS == tok::OOS_ON)
      PP.Diag(Tok, diag::warn_stdc_fenv_access_not_supported);
  }
};

/// PragmaSTDC_CX_LIMITED_RANGEHandler - "#pragma STDC CX_LIMITED_RANGE ...".
struct PragmaSTDC_CX_LIMITED_RANGEHandler : public PragmaHandler {
  PragmaSTDC_CX_LIMITED_RANGEHandler()
    : PragmaHandler("CX_LIMITED_RANGE") {}
  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &Tok) {
    tok::OnOffSwitch OOS;
    PP.LexOnOffSwitch(OOS);
  }
};

/// PragmaSTDC_UnknownHandler - "#pragma STDC ...".
struct PragmaSTDC_UnknownHandler : public PragmaHandler {
  PragmaSTDC_UnknownHandler() {}
  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &UnknownTok) {
    // C99 6.10.6p2, unknown forms are not allowed.
    PP.Diag(UnknownTok, diag::ext_stdc_pragma_ignored);
  }
};

}  // end anonymous namespace


/// RegisterBuiltinPragmas - Install the standard preprocessor pragmas:
/// #pragma GCC poison/system_header/dependency and #pragma once.
void Preprocessor::RegisterBuiltinPragmas() {
  AddPragmaHandler(new PragmaOnceHandler());
  AddPragmaHandler(new PragmaMarkHandler());
  AddPragmaHandler(new PragmaPushMacroHandler());
  AddPragmaHandler(new PragmaPopMacroHandler());
  AddPragmaHandler(new PragmaMessageHandler());

  // #pragma GCC ...
  AddPragmaHandler("GCC", new PragmaPoisonHandler());
  AddPragmaHandler("GCC", new PragmaSystemHeaderHandler());
  AddPragmaHandler("GCC", new PragmaDependencyHandler());
  AddPragmaHandler("GCC", new PragmaDiagnosticHandler());
  // #pragma clang ...
  AddPragmaHandler("clang", new PragmaPoisonHandler());
  AddPragmaHandler("clang", new PragmaSystemHeaderHandler());
  AddPragmaHandler("clang", new PragmaDebugHandler());
  AddPragmaHandler("clang", new PragmaDependencyHandler());
  AddPragmaHandler("clang", new PragmaDiagnosticHandler());

  AddPragmaHandler("STDC", new PragmaSTDC_FENV_ACCESSHandler());
  AddPragmaHandler("STDC", new PragmaSTDC_CX_LIMITED_RANGEHandler());
  AddPragmaHandler("STDC", new PragmaSTDC_UnknownHandler());

  // MS extensions.
  if (Features.Microsoft) {
    AddPragmaHandler(new PragmaCommentHandler());
  }
}
