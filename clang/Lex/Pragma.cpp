//===--- Pragma.cpp - Pragma registration and handling --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PragmaHandler/PragmaTable interfaces and implements
// pragma related methods of the Preprocessor class.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/Pragma.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
using namespace llvm;
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

void PragmaNamespace::HandlePragma(Preprocessor &PP, LexerToken &Tok) {
  // Read the 'namespace' that the directive is in, e.g. STDC.  Do not macro
  // expand it, the user can have a STDC #define, that should not affect this.
  PP.LexUnexpandedToken(Tok);
  
  // Get the handler for this token.  If there is no handler, ignore the pragma.
  PragmaHandler *Handler = FindHandler(Tok.getIdentifierInfo(), false);
  if (Handler == 0) return;
  
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
  LexerToken Tok;
  PragmaHandlers->HandlePragma(*this, Tok);
  
  // If the pragma handler didn't read the rest of the line, consume it now.
  if (CurLexer->ParsingPreprocessorDirective)
    DiscardUntilEndOfDirective();
}

/// Handle_Pragma - Read a _Pragma directive, slice it up, process it, then
/// return the first token after the directive.  The _Pragma token has just
/// been read into 'Tok'.
void Preprocessor::Handle_Pragma(LexerToken &Tok) {
  // Remember the pragma token location.
  SourceLocation PragmaLoc = Tok.getLocation();
  
  // Read the '('.
  Lex(Tok);
  if (Tok.getKind() != tok::l_paren)
    return Diag(PragmaLoc, diag::err__Pragma_malformed);

  // Read the '"..."'.
  Lex(Tok);
  if (Tok.getKind() != tok::string_literal)
    return Diag(PragmaLoc, diag::err__Pragma_malformed);
  
  // Remember the string.
  std::string StrVal = getSpelling(Tok);
  SourceLocation StrLoc = Tok.getLocation();

  // Read the ')'.
  Lex(Tok);
  if (Tok.getKind() != tok::r_paren)
    return Diag(PragmaLoc, diag::err__Pragma_malformed);
  
  // The _Pragma is lexically sound.  Destringize according to C99 6.10.9.1.
  if (StrVal[0] == 'L')  // Remove L prefix.
    StrVal.erase(StrVal.begin());
  assert(StrVal[0] == '"' && StrVal[StrVal.size()-1] == '"' &&
         "Invalid string token!");
  
  // Remove the front quote, replacing it with a space, so that the pragma
  // contents appear to have a space before them.
  StrVal[0] = ' ';
  
  // Replace the terminating quote with a \n\0.
  StrVal[StrVal.size()-1] = '\n';
  StrVal += '\0';
  
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
  SourceLocation TokLoc = CreateString(&StrVal[0], StrVal.size(), StrLoc);
  const char *StrData = SourceMgr.getCharacterData(TokLoc);

  unsigned FileID = TokLoc.getFileID();
  assert(FileID && "Could not get FileID for _Pragma?");

  // Make and enter a lexer object so that we lex and expand the tokens just
  // like any others.
  Lexer *TL = new Lexer(SourceMgr.getBuffer(FileID), FileID, *this,
                        StrData, StrData+StrVal.size()-1 /* no null */);
  
  // Ensure that the lexer thinks it is inside a directive, so that end \n will
  // return an EOM token.
  TL->ParsingPreprocessorDirective = true;
  
  // This lexer really is for _Pragma.
  TL->Is_PragmaLexer = true;

  EnterSourceFileWithLexer(TL, 0);

  // With everything set up, lex this as a #pragma directive.
  HandlePragmaDirective();
  
  // Finally, return whatever came after the pragma directive.
  return Lex(Tok);
}



/// HandlePragmaOnce - Handle #pragma once.  OnceTok is the 'once'.
///
void Preprocessor::HandlePragmaOnce(LexerToken &OnceTok) {
  if (isInPrimaryFile()) {
    Diag(OnceTok, diag::pp_pragma_once_in_main_file);
    return;
  }
  
  // Get the current file lexer we're looking at.  Ignore _Pragma 'files' etc.
  unsigned FileID = getCurrentFileLexer()->getCurFileID();
  
  // Mark the file as a once-only file now.
  getFileInfo(SourceMgr.getFileEntryForFileID(FileID)).isImport = true;
}

/// HandlePragmaPoison - Handle #pragma GCC poison.  PoisonTok is the 'poison'.
///
void Preprocessor::HandlePragmaPoison(LexerToken &PoisonTok) {
  LexerToken Tok;
  assert(!isSkipping() && "Why are we handling pragmas while skipping?");
  while (1) {
    // Read the next token to poison.  While doing this, pretend that we are
    // skipping while reading the identifier to poison.
    // This avoids errors on code like:
    //   #pragma GCC poison X
    //   #pragma GCC poison X
    if (CurLexer) CurLexer->LexingRawMode = true;
    LexUnexpandedToken(Tok);
    if (CurLexer) CurLexer->LexingRawMode = false;
    
    // If we reached the end of line, we're done.
    if (Tok.getKind() == tok::eom) return;
    
    // Can only poison identifiers.
    if (Tok.getKind() != tok::identifier) {
      Diag(Tok, diag::err_pp_invalid_poison);
      return;
    }
    
    // Look up the identifier info for the token.  We disabled identifier lookup
    // by saying we're skipping contents, so we need to do this manually.
    IdentifierInfo *II = LookUpIdentifierInfo(Tok);
    
    // Already poisoned.
    if (II->isPoisoned()) continue;
    
    // If this is a macro identifier, emit a warning.
    if (II->getMacroInfo())
      Diag(Tok, diag::pp_poisoning_existing_macro);
    
    // Finally, poison it!
    II->setIsPoisoned();
  }
}

/// HandlePragmaSystemHeader - Implement #pragma GCC system_header.  We know
/// that the whole directive has been parsed.
void Preprocessor::HandlePragmaSystemHeader(LexerToken &SysHeaderTok) {
  if (isInPrimaryFile()) {
    Diag(SysHeaderTok, diag::pp_pragma_sysheader_in_main_file);
    return;
  }
  
  // Get the current file lexer we're looking at.  Ignore _Pragma 'files' etc.
  Lexer *TheLexer = getCurrentFileLexer();
  
  // Mark the file as a system header.
  const FileEntry *File = 
    SourceMgr.getFileEntryForFileID(TheLexer->getCurFileID());
  getFileInfo(File).DirInfo = DirectoryLookup::SystemHeaderDir;
  
  // Notify the client, if desired, that we are in a new source file.
  if (FileChangeHandler)
    FileChangeHandler(TheLexer->getSourceLocation(TheLexer->BufferPtr),
                      SystemHeaderPragma, DirectoryLookup::SystemHeaderDir);
}

/// HandlePragmaDependency - Handle #pragma GCC dependency "foo" blah.
///
void Preprocessor::HandlePragmaDependency(LexerToken &DependencyTok) {
  LexerToken FilenameTok;
  std::string Filename = CurLexer->LexIncludeFilename(FilenameTok);

  // If the token kind is EOM, the error has already been diagnosed.
  if (FilenameTok.getKind() == tok::eom)
    return;
  
  // Find out whether the filename is <x> or "x".
  bool isAngled = Filename[0] == '<';
  
  // Remove the quotes.
  Filename = std::string(Filename.begin()+1, Filename.end()-1);
  
  // Search include directories.
  const DirectoryLookup *CurDir;
  const FileEntry *File = LookupFile(Filename, isAngled, 0, CurDir);
  if (File == 0)
    return Diag(FilenameTok, diag::err_pp_file_not_found);
  
  unsigned FileID = getCurrentFileLexer()->getCurFileID();
  const FileEntry *CurFile = SourceMgr.getFileEntryForFileID(FileID);

  // If this file is older than the file it depends on, emit a diagnostic.
  if (CurFile && CurFile->getModificationTime() < File->getModificationTime()) {
    // Lex tokens at the end of the message and include them in the message.
    std::string Message;
    Lex(DependencyTok);
    while (DependencyTok.getKind() != tok::eom) {
      Message += getSpelling(DependencyTok) + " ";
      Lex(DependencyTok);
    }
    
    Message.erase(Message.end()-1);
    Diag(FilenameTok, diag::pp_out_of_date_dependency, Message);
  }
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

namespace {
struct PragmaOnceHandler : public PragmaHandler {
  PragmaOnceHandler(const IdentifierInfo *OnceID) : PragmaHandler(OnceID) {}
  virtual void HandlePragma(Preprocessor &PP, LexerToken &OnceTok) {
    PP.CheckEndOfDirective("#pragma once");
    PP.HandlePragmaOnce(OnceTok);
  }
};

struct PragmaPoisonHandler : public PragmaHandler {
  PragmaPoisonHandler(const IdentifierInfo *ID) : PragmaHandler(ID) {}
  virtual void HandlePragma(Preprocessor &PP, LexerToken &PoisonTok) {
    PP.HandlePragmaPoison(PoisonTok);
  }
};

struct PragmaSystemHeaderHandler : public PragmaHandler {
  PragmaSystemHeaderHandler(const IdentifierInfo *ID) : PragmaHandler(ID) {}
  virtual void HandlePragma(Preprocessor &PP, LexerToken &SHToken) {
    PP.HandlePragmaSystemHeader(SHToken);
    PP.CheckEndOfDirective("#pragma");
  }
};
struct PragmaDependencyHandler : public PragmaHandler {
  PragmaDependencyHandler(const IdentifierInfo *ID) : PragmaHandler(ID) {}
  virtual void HandlePragma(Preprocessor &PP, LexerToken &DepToken) {
    PP.HandlePragmaDependency(DepToken);
  }
};
}  // end anonymous namespace


/// RegisterBuiltinPragmas - Install the standard preprocessor pragmas:
/// #pragma GCC poison/system_header/dependency and #pragma once.
void Preprocessor::RegisterBuiltinPragmas() {
  AddPragmaHandler(0, new PragmaOnceHandler(getIdentifierInfo("once")));
  AddPragmaHandler("GCC", new PragmaPoisonHandler(getIdentifierInfo("poison")));
  AddPragmaHandler("GCC", new PragmaSystemHeaderHandler(
                                          getIdentifierInfo("system_header")));
  AddPragmaHandler("GCC", new PragmaDependencyHandler(
                                          getIdentifierInfo("dependency")));
}
