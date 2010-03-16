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
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include <algorithm>
using namespace clang;

// Out-of-line destructor to provide a home for the class.
PragmaHandler::~PragmaHandler() {
}

//===----------------------------------------------------------------------===//
// PragmaNamespace Implementation.
//===----------------------------------------------------------------------===//


PragmaNamespace::~PragmaNamespace() {
  for (unsigned i = 0, e = Handlers.size(); i != e; ++i)
    delete Handlers[i];
}

/// FindHandler - Check to see if there is already a handler for the
/// specified name.  If not, return the handler for the null identifier if it
/// exists, otherwise return null.  If IgnoreNull is true (the default) then
/// the null handler isn't returned on failure to match.
PragmaHandler *PragmaNamespace::FindHandler(const IdentifierInfo *Name,
                                            bool IgnoreNull) const {
  PragmaHandler *NullHandler = 0;
  for (unsigned i = 0, e = Handlers.size(); i != e; ++i) {
    if (Handlers[i]->getName() == Name)
      return Handlers[i];

    if (Handlers[i]->getName() == 0)
      NullHandler = Handlers[i];
  }
  return IgnoreNull ? 0 : NullHandler;
}

void PragmaNamespace::RemovePragmaHandler(PragmaHandler *Handler) {
  for (unsigned i = 0, e = Handlers.size(); i != e; ++i) {
    if (Handlers[i] == Handler) {
      Handlers[i] = Handlers.back();
      Handlers.pop_back();
      return;
    }
  }
  assert(0 && "Handler not registered in this namespace");
}

void PragmaNamespace::HandlePragma(Preprocessor &PP, Token &Tok) {
  // Read the 'namespace' that the directive is in, e.g. STDC.  Do not macro
  // expand it, the user can have a STDC #define, that should not affect this.
  PP.LexUnexpandedToken(Tok);

  // Get the handler for this token.  If there is no handler, ignore the pragma.
  PragmaHandler *Handler = FindHandler(Tok.getIdentifierInfo(), false);
  if (Handler == 0) {
    PP.Diag(Tok, diag::warn_pragma_ignored);
    return;
  }

  // Otherwise, pass it down.
  Handler->HandlePragma(PP, Tok);
}

//===----------------------------------------------------------------------===//
// Preprocessor Pragma Directive Handling.
//===----------------------------------------------------------------------===//

/// HandlePragmaDirective - The "#pragma" directive has been parsed.  Lex the
/// rest of the pragma, passing it to the registered pragma handlers.
void Preprocessor::HandlePragmaDirective() {
  ++NumPragma;

  // Invoke the first level of pragma handlers which reads the namespace id.
  Token Tok;
  PragmaHandlers->HandlePragma(*this, Tok);

  // If the pragma handler didn't read the rest of the line, consume it now.
  if (CurPPLexer && CurPPLexer->ParsingPreprocessorDirective)
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
  HandlePragmaDirective();

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
    if (Tok.isNot(tok::identifier)) {
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




/// AddPragmaHandler - Add the specified pragma handler to the preprocessor.
/// If 'Namespace' is non-null, then it is a token required to exist on the
/// pragma line before the pragma string starts, e.g. "STDC" or "GCC".
void Preprocessor::AddPragmaHandler(const char *Namespace,
                                    PragmaHandler *Handler) {
  PragmaNamespace *InsertNS = PragmaHandlers;

  // If this is specified to be in a namespace, step down into it.
  if (Namespace) {
    IdentifierInfo *NSID = getIdentifierInfo(Namespace);

    // If there is already a pragma handler with the name of this namespace,
    // we either have an error (directive with the same name as a namespace) or
    // we already have the namespace to insert into.
    if (PragmaHandler *Existing = PragmaHandlers->FindHandler(NSID)) {
      InsertNS = Existing->getIfNamespace();
      assert(InsertNS != 0 && "Cannot have a pragma namespace and pragma"
             " handler with the same name!");
    } else {
      // Otherwise, this namespace doesn't exist yet, create and insert the
      // handler for it.
      InsertNS = new PragmaNamespace(NSID);
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
void Preprocessor::RemovePragmaHandler(const char *Namespace,
                                       PragmaHandler *Handler) {
  PragmaNamespace *NS = PragmaHandlers;

  // If this is specified to be in a namespace, step down into it.
  if (Namespace) {
    IdentifierInfo *NSID = getIdentifierInfo(Namespace);
    PragmaHandler *Existing = PragmaHandlers->FindHandler(NSID);
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

namespace {
/// PragmaOnceHandler - "#pragma once" marks the file as atomically included.
struct PragmaOnceHandler : public PragmaHandler {
  PragmaOnceHandler(const IdentifierInfo *OnceID) : PragmaHandler(OnceID) {}
  virtual void HandlePragma(Preprocessor &PP, Token &OnceTok) {
    PP.CheckEndOfDirective("pragma once");
    PP.HandlePragmaOnce(OnceTok);
  }
};

/// PragmaMarkHandler - "#pragma mark ..." is ignored by the compiler, and the
/// rest of the line is not lexed.
struct PragmaMarkHandler : public PragmaHandler {
  PragmaMarkHandler(const IdentifierInfo *MarkID) : PragmaHandler(MarkID) {}
  virtual void HandlePragma(Preprocessor &PP, Token &MarkTok) {
    PP.HandlePragmaMark();
  }
};

/// PragmaPoisonHandler - "#pragma poison x" marks x as not usable.
struct PragmaPoisonHandler : public PragmaHandler {
  PragmaPoisonHandler(const IdentifierInfo *ID) : PragmaHandler(ID) {}
  virtual void HandlePragma(Preprocessor &PP, Token &PoisonTok) {
    PP.HandlePragmaPoison(PoisonTok);
  }
};

/// PragmaSystemHeaderHandler - "#pragma system_header" marks the current file
/// as a system header, which silences warnings in it.
struct PragmaSystemHeaderHandler : public PragmaHandler {
  PragmaSystemHeaderHandler(const IdentifierInfo *ID) : PragmaHandler(ID) {}
  virtual void HandlePragma(Preprocessor &PP, Token &SHToken) {
    PP.HandlePragmaSystemHeader(SHToken);
    PP.CheckEndOfDirective("pragma");
  }
};
struct PragmaDependencyHandler : public PragmaHandler {
  PragmaDependencyHandler(const IdentifierInfo *ID) : PragmaHandler(ID) {}
  virtual void HandlePragma(Preprocessor &PP, Token &DepToken) {
    PP.HandlePragmaDependency(DepToken);
  }
};

/// PragmaDiagnosticHandler - e.g. '#pragma GCC diagnostic ignored "-Wformat"'
/// Since clang's diagnostic supports extended functionality beyond GCC's
/// the constructor takes a clangMode flag to tell it whether or not to allow
/// clang's extended functionality, or whether to reject it.
struct PragmaDiagnosticHandler : public PragmaHandler {
private:
  const bool ClangMode;
public:
  PragmaDiagnosticHandler(const IdentifierInfo *ID,
                          const bool clangMode) : PragmaHandler(ID),
                                                  ClangMode(clangMode) {}
  virtual void HandlePragma(Preprocessor &PP, Token &DiagToken) {
    Token Tok;
    PP.LexUnexpandedToken(Tok);
    if (Tok.isNot(tok::identifier)) {
      unsigned Diag = ClangMode ? diag::warn_pragma_diagnostic_clang_invalid
                                 : diag::warn_pragma_diagnostic_gcc_invalid;
      PP.Diag(Tok, Diag);
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
    else if (ClangMode) {
      if (II->isStr("pop")) {
        if (!PP.getDiagnostics().popMappings())
          PP.Diag(Tok, diag::warn_pragma_diagnostic_clang_cannot_ppp);
        return;
      }

      if (II->isStr("push")) {
        PP.getDiagnostics().pushMappings();
        return;
      }

      PP.Diag(Tok, diag::warn_pragma_diagnostic_clang_invalid);
      return;
    } else {
      PP.Diag(Tok, diag::warn_pragma_diagnostic_gcc_invalid);
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
      unsigned Diag = ClangMode ? diag::warn_pragma_diagnostic_clang_invalid
                                 : diag::warn_pragma_diagnostic_gcc_invalid;
      PP.Diag(Tok, Diag);
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
                                                      Map))
      PP.Diag(StrToks[0].getLocation(),
              diag::warn_pragma_diagnostic_unknown_warning) << WarningName;
  }
};

/// PragmaCommentHandler - "#pragma comment ...".
struct PragmaCommentHandler : public PragmaHandler {
  PragmaCommentHandler(const IdentifierInfo *ID) : PragmaHandler(ID) {}
  virtual void HandlePragma(Preprocessor &PP, Token &CommentTok) {
    PP.HandlePragmaComment(CommentTok);
  }
};

// Pragma STDC implementations.

enum STDCSetting {
  STDC_ON, STDC_OFF, STDC_DEFAULT, STDC_INVALID
};

static STDCSetting LexOnOffSwitch(Preprocessor &PP) {
  Token Tok;
  PP.LexUnexpandedToken(Tok);

  if (Tok.isNot(tok::identifier)) {
    PP.Diag(Tok, diag::ext_stdc_pragma_syntax);
    return STDC_INVALID;
  }
  IdentifierInfo *II = Tok.getIdentifierInfo();
  STDCSetting Result;
  if (II->isStr("ON"))
    Result = STDC_ON;
  else if (II->isStr("OFF"))
    Result = STDC_OFF;
  else if (II->isStr("DEFAULT"))
    Result = STDC_DEFAULT;
  else {
    PP.Diag(Tok, diag::ext_stdc_pragma_syntax);
    return STDC_INVALID;
  }

  // Verify that this is followed by EOM.
  PP.LexUnexpandedToken(Tok);
  if (Tok.isNot(tok::eom))
    PP.Diag(Tok, diag::ext_stdc_pragma_syntax_eom);
  return Result;
}

/// PragmaSTDC_FP_CONTRACTHandler - "#pragma STDC FP_CONTRACT ...".
struct PragmaSTDC_FP_CONTRACTHandler : public PragmaHandler {
  PragmaSTDC_FP_CONTRACTHandler(const IdentifierInfo *ID) : PragmaHandler(ID) {}
  virtual void HandlePragma(Preprocessor &PP, Token &Tok) {
    // We just ignore the setting of FP_CONTRACT. Since we don't do contractions
    // at all, our default is OFF and setting it to ON is an optimization hint
    // we can safely ignore.  When we support -ffma or something, we would need
    // to diagnose that we are ignoring FMA.
    LexOnOffSwitch(PP);
  }
};

/// PragmaSTDC_FENV_ACCESSHandler - "#pragma STDC FENV_ACCESS ...".
struct PragmaSTDC_FENV_ACCESSHandler : public PragmaHandler {
  PragmaSTDC_FENV_ACCESSHandler(const IdentifierInfo *ID) : PragmaHandler(ID) {}
  virtual void HandlePragma(Preprocessor &PP, Token &Tok) {
    if (LexOnOffSwitch(PP) == STDC_ON)
      PP.Diag(Tok, diag::warn_stdc_fenv_access_not_supported);
  }
};

/// PragmaSTDC_CX_LIMITED_RANGEHandler - "#pragma STDC CX_LIMITED_RANGE ...".
struct PragmaSTDC_CX_LIMITED_RANGEHandler : public PragmaHandler {
  PragmaSTDC_CX_LIMITED_RANGEHandler(const IdentifierInfo *ID)
    : PragmaHandler(ID) {}
  virtual void HandlePragma(Preprocessor &PP, Token &Tok) {
    LexOnOffSwitch(PP);
  }
};

/// PragmaSTDC_UnknownHandler - "#pragma STDC ...".
struct PragmaSTDC_UnknownHandler : public PragmaHandler {
  PragmaSTDC_UnknownHandler() : PragmaHandler(0) {}
  virtual void HandlePragma(Preprocessor &PP, Token &UnknownTok) {
    // C99 6.10.6p2, unknown forms are not allowed.
    PP.Diag(UnknownTok, diag::ext_stdc_pragma_ignored);
  }
};

}  // end anonymous namespace


/// RegisterBuiltinPragmas - Install the standard preprocessor pragmas:
/// #pragma GCC poison/system_header/dependency and #pragma once.
void Preprocessor::RegisterBuiltinPragmas() {
  AddPragmaHandler(0, new PragmaOnceHandler(getIdentifierInfo("once")));
  AddPragmaHandler(0, new PragmaMarkHandler(getIdentifierInfo("mark")));

  // #pragma GCC ...
  AddPragmaHandler("GCC", new PragmaPoisonHandler(getIdentifierInfo("poison")));
  AddPragmaHandler("GCC", new PragmaSystemHeaderHandler(
                                          getIdentifierInfo("system_header")));
  AddPragmaHandler("GCC", new PragmaDependencyHandler(
                                          getIdentifierInfo("dependency")));
  AddPragmaHandler("GCC", new PragmaDiagnosticHandler(
                                              getIdentifierInfo("diagnostic"),
                                              false));
  // #pragma clang ...
  AddPragmaHandler("clang", new PragmaPoisonHandler(
                                          getIdentifierInfo("poison")));
  AddPragmaHandler("clang", new PragmaSystemHeaderHandler(
                                          getIdentifierInfo("system_header")));
  AddPragmaHandler("clang", new PragmaDependencyHandler(
                                          getIdentifierInfo("dependency")));
  AddPragmaHandler("clang", new PragmaDiagnosticHandler(
                                          getIdentifierInfo("diagnostic"),
                                          true));

  AddPragmaHandler("STDC", new PragmaSTDC_FP_CONTRACTHandler(
                                             getIdentifierInfo("FP_CONTRACT")));
  AddPragmaHandler("STDC", new PragmaSTDC_FENV_ACCESSHandler(
                                             getIdentifierInfo("FENV_ACCESS")));
  AddPragmaHandler("STDC", new PragmaSTDC_CX_LIMITED_RANGEHandler(
                                        getIdentifierInfo("CX_LIMITED_RANGE")));
  AddPragmaHandler("STDC", new PragmaSTDC_UnknownHandler());

  // MS extensions.
  if (Features.Microsoft)
    AddPragmaHandler(0, new PragmaCommentHandler(getIdentifierInfo("comment")));
}
