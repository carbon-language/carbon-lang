//===--- Lexer.cpp - C Language Family Lexer ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Lexer and Token interfaces.
//
//===----------------------------------------------------------------------===//
//
// TODO: GCC Diagnostics emitted by the lexer:
// PEDWARN: (form feed|vertical tab) in preprocessing directive
//
// Universal characters, unicode, char mapping:
// WARNING: `%.*s' is not in NFKC
// WARNING: `%.*s' is not in NFC
//
// Other:
// TODO: Options to support:
//    -fexec-charset,-fwide-exec-charset
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cctype>
using namespace clang;

static void InitCharacterInfo();

//===----------------------------------------------------------------------===//
// Token Class Implementation
//===----------------------------------------------------------------------===//

/// isObjCAtKeyword - Return true if we have an ObjC keyword identifier. 
bool Token::isObjCAtKeyword(tok::ObjCKeywordKind objcKey) const {
  return is(tok::identifier) && 
         getIdentifierInfo()->getObjCKeywordID() == objcKey;
}

/// getObjCKeywordID - Return the ObjC keyword kind.
tok::ObjCKeywordKind Token::getObjCKeywordID() const {
  IdentifierInfo *specId = getIdentifierInfo();
  return specId ? specId->getObjCKeywordID() : tok::objc_not_keyword;
}


//===----------------------------------------------------------------------===//
// Lexer Class Implementation
//===----------------------------------------------------------------------===//


/// Lexer constructor - Create a new lexer object for the specified buffer
/// with the specified preprocessor managing the lexing process.  This lexer
/// assumes that the associated file buffer and Preprocessor objects will
/// outlive it, so it doesn't take ownership of either of them.
Lexer::Lexer(SourceLocation fileloc, Preprocessor &pp,
             const char *BufStart, const char *BufEnd)
  : FileLoc(fileloc), PP(&pp), Features(pp.getLangOptions()) {
      
  SourceManager &SourceMgr = PP->getSourceManager();
  unsigned InputFileID = SourceMgr.getPhysicalLoc(FileLoc).getFileID();
  const llvm::MemoryBuffer *InputFile = SourceMgr.getBuffer(InputFileID);
      
  Is_PragmaLexer = false;
  InitCharacterInfo();
  
  // BufferStart must always be InputFile->getBufferStart().
  BufferStart = InputFile->getBufferStart();
  
  // BufferPtr and BufferEnd can start out somewhere inside the current buffer.
  // If unspecified, they starts at the start/end of the buffer.
  BufferPtr = BufStart ? BufStart : BufferStart;
  BufferEnd = BufEnd ? BufEnd : InputFile->getBufferEnd();

  assert(BufferEnd[0] == 0 &&
         "We assume that the input buffer has a null character at the end"
         " to simplify lexing!");
  
  // Start of the file is a start of line.
  IsAtStartOfLine = true;

  // We are not after parsing a #.
  ParsingPreprocessorDirective = false;

  // We are not after parsing #include.
  ParsingFilename = false;

  // We are not in raw mode.  Raw mode disables diagnostics and interpretation
  // of tokens (e.g. identifiers, thus disabling macro expansion).  It is used
  // to quickly lex the tokens of the buffer, e.g. when handling a "#if 0" block
  // or otherwise skipping over tokens.
  LexingRawMode = false;
  
  // Default to keeping comments if requested.
  KeepCommentMode = PP->getCommentRetentionState();
}

/// Lexer constructor - Create a new raw lexer object.  This object is only
/// suitable for calls to 'LexRawToken'.  This lexer assumes that the text
/// range will outlive it, so it doesn't take ownership of it.
Lexer::Lexer(SourceLocation fileloc, const LangOptions &features,
             const char *BufStart, const char *BufEnd,
             const llvm::MemoryBuffer *FromFile)
  : FileLoc(fileloc), PP(0), Features(features) {
  Is_PragmaLexer = false;
  InitCharacterInfo();
  
  // If a MemoryBuffer was specified, use its start as BufferStart. This affects
  // the source location objects produced by this lexer.
  BufferStart = FromFile ? FromFile->getBufferStart() : BufStart;
  BufferPtr = BufStart;
  BufferEnd = BufEnd;
  
  assert(BufferEnd[0] == 0 &&
         "We assume that the input buffer has a null character at the end"
         " to simplify lexing!");
  
  // Start of the file is a start of line.
  IsAtStartOfLine = true;
  
  // We are not after parsing a #.
  ParsingPreprocessorDirective = false;
  
  // We are not after parsing #include.
  ParsingFilename = false;
  
  // We *are* in raw mode.
  LexingRawMode = true;
  
  // Default to not keeping comments in raw mode.
  KeepCommentMode = false;
}


/// Stringify - Convert the specified string into a C string, with surrounding
/// ""'s, and with escaped \ and " characters.
std::string Lexer::Stringify(const std::string &Str, bool Charify) {
  std::string Result = Str;
  char Quote = Charify ? '\'' : '"';
  for (unsigned i = 0, e = Result.size(); i != e; ++i) {
    if (Result[i] == '\\' || Result[i] == Quote) {
      Result.insert(Result.begin()+i, '\\');
      ++i; ++e;
    }
  }
  return Result;
}

/// Stringify - Convert the specified string into a C string by escaping '\'
/// and " characters.  This does not add surrounding ""'s to the string.
void Lexer::Stringify(llvm::SmallVectorImpl<char> &Str) {
  for (unsigned i = 0, e = Str.size(); i != e; ++i) {
    if (Str[i] == '\\' || Str[i] == '"') {
      Str.insert(Str.begin()+i, '\\');
      ++i; ++e;
    }
  }
}


/// MeasureTokenLength - Relex the token at the specified location and return
/// its length in bytes in the input file.  If the token needs cleaning (e.g.
/// includes a trigraph or an escaped newline) then this count includes bytes
/// that are part of that.
unsigned Lexer::MeasureTokenLength(SourceLocation Loc,
                                   const SourceManager &SM) {
  // If this comes from a macro expansion, we really do want the macro name, not
  // the token this macro expanded to.
  Loc = SM.getLogicalLoc(Loc);
  
  const char *StrData = SM.getCharacterData(Loc);
  
  // TODO: this could be special cased for common tokens like identifiers, ')',
  // etc to make this faster, if it mattered.  Just look at StrData[0] to handle
  // all obviously single-char tokens.  This could use 
  // Lexer::isObviouslySimpleCharacter for example to handle identifiers or
  // something.
  
  
  const char *BufEnd = SM.getBufferData(Loc.getFileID()).second;
  
  // Create a langops struct and enable trigraphs.  This is sufficient for
  // measuring tokens.
  LangOptions LangOpts;
  LangOpts.Trigraphs = true;
  
  // Create a lexer starting at the beginning of this token.
  Lexer TheLexer(Loc, LangOpts, StrData, BufEnd);
  Token TheTok;
  TheLexer.LexFromRawLexer(TheTok);
  return TheTok.getLength();
}

//===----------------------------------------------------------------------===//
// Character information.
//===----------------------------------------------------------------------===//

static unsigned char CharInfo[256];

enum {
  CHAR_HORZ_WS  = 0x01,  // ' ', '\t', '\f', '\v'.  Note, no '\0'
  CHAR_VERT_WS  = 0x02,  // '\r', '\n'
  CHAR_LETTER   = 0x04,  // a-z,A-Z
  CHAR_NUMBER   = 0x08,  // 0-9
  CHAR_UNDER    = 0x10,  // _
  CHAR_PERIOD   = 0x20   // .
};

static void InitCharacterInfo() {
  static bool isInited = false;
  if (isInited) return;
  isInited = true;
  
  // Intiialize the CharInfo table.
  // TODO: statically initialize this.
  CharInfo[(int)' '] = CharInfo[(int)'\t'] = 
  CharInfo[(int)'\f'] = CharInfo[(int)'\v'] = CHAR_HORZ_WS;
  CharInfo[(int)'\n'] = CharInfo[(int)'\r'] = CHAR_VERT_WS;
  
  CharInfo[(int)'_'] = CHAR_UNDER;
  CharInfo[(int)'.'] = CHAR_PERIOD;
  for (unsigned i = 'a'; i <= 'z'; ++i)
    CharInfo[i] = CharInfo[i+'A'-'a'] = CHAR_LETTER;
  for (unsigned i = '0'; i <= '9'; ++i)
    CharInfo[i] = CHAR_NUMBER;
}

/// isIdentifierBody - Return true if this is the body character of an
/// identifier, which is [a-zA-Z0-9_].
static inline bool isIdentifierBody(unsigned char c) {
  return (CharInfo[c] & (CHAR_LETTER|CHAR_NUMBER|CHAR_UNDER)) ? true : false;
}

/// isHorizontalWhitespace - Return true if this character is horizontal
/// whitespace: ' ', '\t', '\f', '\v'.  Note that this returns false for '\0'.
static inline bool isHorizontalWhitespace(unsigned char c) {
  return (CharInfo[c] & CHAR_HORZ_WS) ? true : false;
}

/// isWhitespace - Return true if this character is horizontal or vertical
/// whitespace: ' ', '\t', '\f', '\v', '\n', '\r'.  Note that this returns false
/// for '\0'.
static inline bool isWhitespace(unsigned char c) {
  return (CharInfo[c] & (CHAR_HORZ_WS|CHAR_VERT_WS)) ? true : false;
}

/// isNumberBody - Return true if this is the body character of an
/// preprocessing number, which is [a-zA-Z0-9_.].
static inline bool isNumberBody(unsigned char c) {
  return (CharInfo[c] & (CHAR_LETTER|CHAR_NUMBER|CHAR_UNDER|CHAR_PERIOD)) ? 
    true : false;
}


//===----------------------------------------------------------------------===//
// Diagnostics forwarding code.
//===----------------------------------------------------------------------===//

/// GetMappedTokenLoc - If lexing out of a 'mapped buffer', where we pretend the
/// lexer buffer was all instantiated at a single point, perform the mapping.
/// This is currently only used for _Pragma implementation, so it is the slow
/// path of the hot getSourceLocation method.  Do not allow it to be inlined.
static SourceLocation GetMappedTokenLoc(Preprocessor &PP,
                                        SourceLocation FileLoc,
                                        unsigned CharNo) DISABLE_INLINE;
static SourceLocation GetMappedTokenLoc(Preprocessor &PP,
                                        SourceLocation FileLoc,
                                        unsigned CharNo) {
  // Otherwise, we're lexing "mapped tokens".  This is used for things like
  // _Pragma handling.  Combine the instantiation location of FileLoc with the
  // physical location.
  SourceManager &SourceMgr = PP.getSourceManager();
  
  // Create a new SLoc which is expanded from logical(FileLoc) but whose
  // characters come from phys(FileLoc)+Offset.
  SourceLocation VirtLoc = SourceMgr.getLogicalLoc(FileLoc);
  SourceLocation PhysLoc = SourceMgr.getPhysicalLoc(FileLoc);
  PhysLoc = SourceLocation::getFileLoc(PhysLoc.getFileID(), CharNo);
  return SourceMgr.getInstantiationLoc(PhysLoc, VirtLoc);
}

/// getSourceLocation - Return a source location identifier for the specified
/// offset in the current file.
SourceLocation Lexer::getSourceLocation(const char *Loc) const {
  assert(Loc >= BufferStart && Loc <= BufferEnd &&
         "Location out of range for this buffer!");

  // In the normal case, we're just lexing from a simple file buffer, return
  // the file id from FileLoc with the offset specified.
  unsigned CharNo = Loc-BufferStart;
  if (FileLoc.isFileID())
    return SourceLocation::getFileLoc(FileLoc.getFileID(), CharNo);
  
  assert(PP && "This doesn't work on raw lexers");
  return GetMappedTokenLoc(*PP, FileLoc, CharNo);
}

/// Diag - Forwarding function for diagnostics.  This translate a source
/// position in the current buffer into a SourceLocation object for rendering.
void Lexer::Diag(const char *Loc, unsigned DiagID,
                 const std::string &Msg) const {
  if (LexingRawMode && Diagnostic::isBuiltinNoteWarningOrExtension(DiagID))
    return;
  PP->Diag(getSourceLocation(Loc), DiagID, Msg);
}
void Lexer::Diag(SourceLocation Loc, unsigned DiagID,
                 const std::string &Msg) const {
  if (LexingRawMode && Diagnostic::isBuiltinNoteWarningOrExtension(DiagID))
    return;
  PP->Diag(Loc, DiagID, Msg);
}


//===----------------------------------------------------------------------===//
// Trigraph and Escaped Newline Handling Code.
//===----------------------------------------------------------------------===//

/// GetTrigraphCharForLetter - Given a character that occurs after a ?? pair,
/// return the decoded trigraph letter it corresponds to, or '\0' if nothing.
static char GetTrigraphCharForLetter(char Letter) {
  switch (Letter) {
  default:   return 0;
  case '=':  return '#';
  case ')':  return ']';
  case '(':  return '[';
  case '!':  return '|';
  case '\'': return '^';
  case '>':  return '}';
  case '/':  return '\\';
  case '<':  return '{';
  case '-':  return '~';
  }
}

/// DecodeTrigraphChar - If the specified character is a legal trigraph when
/// prefixed with ??, emit a trigraph warning.  If trigraphs are enabled,
/// return the result character.  Finally, emit a warning about trigraph use
/// whether trigraphs are enabled or not.
static char DecodeTrigraphChar(const char *CP, Lexer *L) {
  char Res = GetTrigraphCharForLetter(*CP);
  if (Res && L) {
    if (!L->getFeatures().Trigraphs) {
      L->Diag(CP-2, diag::trigraph_ignored);
      return 0;
    } else {
      L->Diag(CP-2, diag::trigraph_converted, std::string()+Res);
    }
  }
  return Res;
}

/// getCharAndSizeSlow - Peek a single 'character' from the specified buffer,
/// get its size, and return it.  This is tricky in several cases:
///   1. If currently at the start of a trigraph, we warn about the trigraph,
///      then either return the trigraph (skipping 3 chars) or the '?',
///      depending on whether trigraphs are enabled or not.
///   2. If this is an escaped newline (potentially with whitespace between
///      the backslash and newline), implicitly skip the newline and return
///      the char after it.
///   3. If this is a UCN, return it.  FIXME: C++ UCN's?
///
/// This handles the slow/uncommon case of the getCharAndSize method.  Here we
/// know that we can accumulate into Size, and that we have already incremented
/// Ptr by Size bytes.
///
/// NOTE: When this method is updated, getCharAndSizeSlowNoWarn (below) should
/// be updated to match.
///
char Lexer::getCharAndSizeSlow(const char *Ptr, unsigned &Size,
                               Token *Tok) {
  // If we have a slash, look for an escaped newline.
  if (Ptr[0] == '\\') {
    ++Size;
    ++Ptr;
Slash:
    // Common case, backslash-char where the char is not whitespace.
    if (!isWhitespace(Ptr[0])) return '\\';
    
    // See if we have optional whitespace characters followed by a newline.
    {
      unsigned SizeTmp = 0;
      do {
        ++SizeTmp;
        if (Ptr[SizeTmp-1] == '\n' || Ptr[SizeTmp-1] == '\r') {
          // Remember that this token needs to be cleaned.
          if (Tok) Tok->setFlag(Token::NeedsCleaning);

          // Warn if there was whitespace between the backslash and newline.
          if (SizeTmp != 1 && Tok)
            Diag(Ptr, diag::backslash_newline_space);
          
          // If this is a \r\n or \n\r, skip the newlines.
          if ((Ptr[SizeTmp] == '\r' || Ptr[SizeTmp] == '\n') &&
              Ptr[SizeTmp-1] != Ptr[SizeTmp])
            ++SizeTmp;
          
          // Found backslash<whitespace><newline>.  Parse the char after it.
          Size += SizeTmp;
          Ptr  += SizeTmp;
          // Use slow version to accumulate a correct size field.
          return getCharAndSizeSlow(Ptr, Size, Tok);
        }
      } while (isWhitespace(Ptr[SizeTmp]));
    }
      
    // Otherwise, this is not an escaped newline, just return the slash.
    return '\\';
  }
  
  // If this is a trigraph, process it.
  if (Ptr[0] == '?' && Ptr[1] == '?') {
    // If this is actually a legal trigraph (not something like "??x"), emit
    // a trigraph warning.  If so, and if trigraphs are enabled, return it.
    if (char C = DecodeTrigraphChar(Ptr+2, Tok ? this : 0)) {
      // Remember that this token needs to be cleaned.
      if (Tok) Tok->setFlag(Token::NeedsCleaning);

      Ptr += 3;
      Size += 3;
      if (C == '\\') goto Slash;
      return C;
    }
  }
  
  // If this is neither, return a single character.
  ++Size;
  return *Ptr;
}


/// getCharAndSizeSlowNoWarn - Handle the slow/uncommon case of the
/// getCharAndSizeNoWarn method.  Here we know that we can accumulate into Size,
/// and that we have already incremented Ptr by Size bytes.
///
/// NOTE: When this method is updated, getCharAndSizeSlow (above) should
/// be updated to match.
char Lexer::getCharAndSizeSlowNoWarn(const char *Ptr, unsigned &Size,
                                     const LangOptions &Features) {
  // If we have a slash, look for an escaped newline.
  if (Ptr[0] == '\\') {
    ++Size;
    ++Ptr;
Slash:
    // Common case, backslash-char where the char is not whitespace.
    if (!isWhitespace(Ptr[0])) return '\\';
    
    // See if we have optional whitespace characters followed by a newline.
    {
      unsigned SizeTmp = 0;
      do {
        ++SizeTmp;
        if (Ptr[SizeTmp-1] == '\n' || Ptr[SizeTmp-1] == '\r') {
          
          // If this is a \r\n or \n\r, skip the newlines.
          if ((Ptr[SizeTmp] == '\r' || Ptr[SizeTmp] == '\n') &&
              Ptr[SizeTmp-1] != Ptr[SizeTmp])
            ++SizeTmp;
          
          // Found backslash<whitespace><newline>.  Parse the char after it.
          Size += SizeTmp;
          Ptr  += SizeTmp;
          
          // Use slow version to accumulate a correct size field.
          return getCharAndSizeSlowNoWarn(Ptr, Size, Features);
        }
      } while (isWhitespace(Ptr[SizeTmp]));
    }
    
    // Otherwise, this is not an escaped newline, just return the slash.
    return '\\';
  }
  
  // If this is a trigraph, process it.
  if (Features.Trigraphs && Ptr[0] == '?' && Ptr[1] == '?') {
    // If this is actually a legal trigraph (not something like "??x"), return
    // it.
    if (char C = GetTrigraphCharForLetter(Ptr[2])) {
      Ptr += 3;
      Size += 3;
      if (C == '\\') goto Slash;
      return C;
    }
  }
  
  // If this is neither, return a single character.
  ++Size;
  return *Ptr;
}

//===----------------------------------------------------------------------===//
// Helper methods for lexing.
//===----------------------------------------------------------------------===//

void Lexer::LexIdentifier(Token &Result, const char *CurPtr) {
  // Match [_A-Za-z0-9]*, we have already matched [_A-Za-z$]
  unsigned Size;
  unsigned char C = *CurPtr++;
  while (isIdentifierBody(C)) {
    C = *CurPtr++;
  }
  --CurPtr;   // Back up over the skipped character.

  // Fast path, no $,\,? in identifier found.  '\' might be an escaped newline
  // or UCN, and ? might be a trigraph for '\', an escaped newline or UCN.
  // FIXME: UCNs.
  if (C != '\\' && C != '?' && (C != '$' || !Features.DollarIdents)) {
FinishIdentifier:
    const char *IdStart = BufferPtr;
    FormTokenWithChars(Result, CurPtr);
    Result.setKind(tok::identifier);
    
    // If we are in raw mode, return this identifier raw.  There is no need to
    // look up identifier information or attempt to macro expand it.
    if (LexingRawMode) return;
    
    // Fill in Result.IdentifierInfo, looking up the identifier in the
    // identifier table.
    PP->LookUpIdentifierInfo(Result, IdStart);
    
    // Finally, now that we know we have an identifier, pass this off to the
    // preprocessor, which may macro expand it or something.
    return PP->HandleIdentifier(Result);
  }
  
  // Otherwise, $,\,? in identifier found.  Enter slower path.
  
  C = getCharAndSize(CurPtr, Size);
  while (1) {
    if (C == '$') {
      // If we hit a $ and they are not supported in identifiers, we are done.
      if (!Features.DollarIdents) goto FinishIdentifier;
      
      // Otherwise, emit a diagnostic and continue.
      Diag(CurPtr, diag::ext_dollar_in_identifier);
      CurPtr = ConsumeChar(CurPtr, Size, Result);
      C = getCharAndSize(CurPtr, Size);
      continue;
    } else if (!isIdentifierBody(C)) { // FIXME: UCNs.
      // Found end of identifier.
      goto FinishIdentifier;
    }

    // Otherwise, this character is good, consume it.
    CurPtr = ConsumeChar(CurPtr, Size, Result);

    C = getCharAndSize(CurPtr, Size);
    while (isIdentifierBody(C)) { // FIXME: UCNs.
      CurPtr = ConsumeChar(CurPtr, Size, Result);
      C = getCharAndSize(CurPtr, Size);
    }
  }
}


/// LexNumericConstant - Lex the remainder of a integer or floating point
/// constant. From[-1] is the first character lexed.  Return the end of the
/// constant.
void Lexer::LexNumericConstant(Token &Result, const char *CurPtr) {
  unsigned Size;
  char C = getCharAndSize(CurPtr, Size);
  char PrevCh = 0;
  while (isNumberBody(C)) { // FIXME: UCNs?
    CurPtr = ConsumeChar(CurPtr, Size, Result);
    PrevCh = C;
    C = getCharAndSize(CurPtr, Size);
  }
  
  // If we fell out, check for a sign, due to 1e+12.  If we have one, continue.
  if ((C == '-' || C == '+') && (PrevCh == 'E' || PrevCh == 'e'))
    return LexNumericConstant(Result, ConsumeChar(CurPtr, Size, Result));

  // If we have a hex FP constant, continue.
  if (Features.HexFloats &&
      (C == '-' || C == '+') && (PrevCh == 'P' || PrevCh == 'p'))
    return LexNumericConstant(Result, ConsumeChar(CurPtr, Size, Result));
  
  Result.setKind(tok::numeric_constant);

  // Update the location of token as well as BufferPtr.
  FormTokenWithChars(Result, CurPtr);
}

/// LexStringLiteral - Lex the remainder of a string literal, after having lexed
/// either " or L".
void Lexer::LexStringLiteral(Token &Result, const char *CurPtr, bool Wide){
  const char *NulCharacter = 0; // Does this string contain the \0 character?
  
  char C = getAndAdvanceChar(CurPtr, Result);
  while (C != '"') {
    // Skip escaped characters.
    if (C == '\\') {
      // Skip the escaped character.
      C = getAndAdvanceChar(CurPtr, Result);
    } else if (C == '\n' || C == '\r' ||             // Newline.
               (C == 0 && CurPtr-1 == BufferEnd)) {  // End of file.
      if (!LexingRawMode) Diag(BufferPtr, diag::err_unterminated_string);
      Result.setKind(tok::unknown);
      FormTokenWithChars(Result, CurPtr-1);
      return;
    } else if (C == 0) {
      NulCharacter = CurPtr-1;
    }
    C = getAndAdvanceChar(CurPtr, Result);
  }
  
  // If a nul character existed in the string, warn about it.
  if (NulCharacter) Diag(NulCharacter, diag::null_in_string);

  Result.setKind(Wide ? tok::wide_string_literal : tok::string_literal);

  // Update the location of the token as well as the BufferPtr instance var.
  FormTokenWithChars(Result, CurPtr);
}

/// LexAngledStringLiteral - Lex the remainder of an angled string literal,
/// after having lexed the '<' character.  This is used for #include filenames.
void Lexer::LexAngledStringLiteral(Token &Result, const char *CurPtr) {
  const char *NulCharacter = 0; // Does this string contain the \0 character?
  
  char C = getAndAdvanceChar(CurPtr, Result);
  while (C != '>') {
    // Skip escaped characters.
    if (C == '\\') {
      // Skip the escaped character.
      C = getAndAdvanceChar(CurPtr, Result);
    } else if (C == '\n' || C == '\r' ||             // Newline.
               (C == 0 && CurPtr-1 == BufferEnd)) {  // End of file.
      if (!LexingRawMode) Diag(BufferPtr, diag::err_unterminated_string);
      Result.setKind(tok::unknown);
      FormTokenWithChars(Result, CurPtr-1);
      return;
    } else if (C == 0) {
      NulCharacter = CurPtr-1;
    }
    C = getAndAdvanceChar(CurPtr, Result);
  }
  
  // If a nul character existed in the string, warn about it.
  if (NulCharacter) Diag(NulCharacter, diag::null_in_string);
  
  Result.setKind(tok::angle_string_literal);
  
  // Update the location of token as well as BufferPtr.
  FormTokenWithChars(Result, CurPtr);
}


/// LexCharConstant - Lex the remainder of a character constant, after having
/// lexed either ' or L'.
void Lexer::LexCharConstant(Token &Result, const char *CurPtr) {
  const char *NulCharacter = 0; // Does this character contain the \0 character?

  // Handle the common case of 'x' and '\y' efficiently.
  char C = getAndAdvanceChar(CurPtr, Result);
  if (C == '\'') {
    if (!LexingRawMode) Diag(BufferPtr, diag::err_empty_character);
    Result.setKind(tok::unknown);
    FormTokenWithChars(Result, CurPtr);
    return;
  } else if (C == '\\') {
    // Skip the escaped character.
    // FIXME: UCN's.
    C = getAndAdvanceChar(CurPtr, Result);
  }
  
  if (C && C != '\n' && C != '\r' && CurPtr[0] == '\'') {
    ++CurPtr;
  } else {
    // Fall back on generic code for embedded nulls, newlines, wide chars.
    do {
      // Skip escaped characters.
      if (C == '\\') {
        // Skip the escaped character.
        C = getAndAdvanceChar(CurPtr, Result);
      } else if (C == '\n' || C == '\r' ||               // Newline.
                 (C == 0 && CurPtr-1 == BufferEnd)) {    // End of file.
        if (!LexingRawMode) Diag(BufferPtr, diag::err_unterminated_char);
        Result.setKind(tok::unknown);
        FormTokenWithChars(Result, CurPtr-1);
        return;
      } else if (C == 0) {
        NulCharacter = CurPtr-1;
      }
      C = getAndAdvanceChar(CurPtr, Result);
    } while (C != '\'');
  }
  
  if (NulCharacter) Diag(NulCharacter, diag::null_in_char);

  Result.setKind(tok::char_constant);
  
  // Update the location of token as well as BufferPtr.
  FormTokenWithChars(Result, CurPtr);
}

/// SkipWhitespace - Efficiently skip over a series of whitespace characters.
/// Update BufferPtr to point to the next non-whitespace character and return.
void Lexer::SkipWhitespace(Token &Result, const char *CurPtr) {
  // Whitespace - Skip it, then return the token after the whitespace.
  unsigned char Char = *CurPtr;  // Skip consequtive spaces efficiently.
  while (1) {
    // Skip horizontal whitespace very aggressively.
    while (isHorizontalWhitespace(Char))
      Char = *++CurPtr;
    
    // Otherwise if we something other than whitespace, we're done.
    if (Char != '\n' && Char != '\r')
      break;
    
    if (ParsingPreprocessorDirective) {
      // End of preprocessor directive line, let LexTokenInternal handle this.
      BufferPtr = CurPtr;
      return;
    }
    
    // ok, but handle newline.
    // The returned token is at the start of the line.
    Result.setFlag(Token::StartOfLine);
    // No leading whitespace seen so far.
    Result.clearFlag(Token::LeadingSpace);
    Char = *++CurPtr;
  }

  // If this isn't immediately after a newline, there is leading space.
  char PrevChar = CurPtr[-1];
  if (PrevChar != '\n' && PrevChar != '\r')
    Result.setFlag(Token::LeadingSpace);

  BufferPtr = CurPtr;
}

// SkipBCPLComment - We have just read the // characters from input.  Skip until
// we find the newline character thats terminate the comment.  Then update
/// BufferPtr and return.
bool Lexer::SkipBCPLComment(Token &Result, const char *CurPtr) {
  // If BCPL comments aren't explicitly enabled for this language, emit an
  // extension warning.
  if (!Features.BCPLComment) {
    Diag(BufferPtr, diag::ext_bcpl_comment);
    
    // Mark them enabled so we only emit one warning for this translation
    // unit.
    Features.BCPLComment = true;
  }
  
  // Scan over the body of the comment.  The common case, when scanning, is that
  // the comment contains normal ascii characters with nothing interesting in
  // them.  As such, optimize for this case with the inner loop.
  char C;
  do {
    C = *CurPtr;
    // FIXME: Speedup BCPL comment lexing.  Just scan for a \n or \r character.
    // If we find a \n character, scan backwards, checking to see if it's an
    // escaped newline, like we do for block comments.
    
    // Skip over characters in the fast loop.
    while (C != 0 &&                // Potentially EOF.
           C != '\\' &&             // Potentially escaped newline.
           C != '?' &&              // Potentially trigraph.
           C != '\n' && C != '\r')  // Newline or DOS-style newline.
      C = *++CurPtr;

    // If this is a newline, we're done.
    if (C == '\n' || C == '\r')
      break;  // Found the newline? Break out!
    
    // Otherwise, this is a hard case.  Fall back on getAndAdvanceChar to
    // properly decode the character.
    const char *OldPtr = CurPtr;
    C = getAndAdvanceChar(CurPtr, Result);
    
    // If we read multiple characters, and one of those characters was a \r or
    // \n, then we had an escaped newline within the comment.  Emit diagnostic
    // unless the next line is also a // comment.
    if (CurPtr != OldPtr+1 && C != '/' && CurPtr[0] != '/') {
      for (; OldPtr != CurPtr; ++OldPtr)
        if (OldPtr[0] == '\n' || OldPtr[0] == '\r') {
          // Okay, we found a // comment that ends in a newline, if the next
          // line is also a // comment, but has spaces, don't emit a diagnostic.
          if (isspace(C)) {
            const char *ForwardPtr = CurPtr;
            while (isspace(*ForwardPtr))  // Skip whitespace.
              ++ForwardPtr;
            if (ForwardPtr[0] == '/' && ForwardPtr[1] == '/')
              break;
          }
          
          Diag(OldPtr-1, diag::ext_multi_line_bcpl_comment);
          break;
        }
    }
    
    if (CurPtr == BufferEnd+1) { --CurPtr; break; }
  } while (C != '\n' && C != '\r');

  // Found but did not consume the newline.
    
  // If we are returning comments as tokens, return this comment as a token.
  if (inKeepCommentMode())
    return SaveBCPLComment(Result, CurPtr);

  // If we are inside a preprocessor directive and we see the end of line,
  // return immediately, so that the lexer can return this as an EOM token.
  if (ParsingPreprocessorDirective || CurPtr == BufferEnd) {
    BufferPtr = CurPtr;
    return true;
  }
  
  // Otherwise, eat the \n character.  We don't care if this is a \n\r or
  // \r\n sequence.  This is an efficiency hack (because we know the \n can't
  // contribute to another token), it isn't needed for correctness.
  ++CurPtr;
    
  // The next returned token is at the start of the line.
  Result.setFlag(Token::StartOfLine);
  // No leading whitespace seen so far.
  Result.clearFlag(Token::LeadingSpace);
  BufferPtr = CurPtr;
  return true;
}

/// SaveBCPLComment - If in save-comment mode, package up this BCPL comment in
/// an appropriate way and return it.
bool Lexer::SaveBCPLComment(Token &Result, const char *CurPtr) {
  Result.setKind(tok::comment);
  FormTokenWithChars(Result, CurPtr);
  
  // If this BCPL-style comment is in a macro definition, transmogrify it into
  // a C-style block comment.
  if (ParsingPreprocessorDirective) {
    std::string Spelling = PP->getSpelling(Result);
    assert(Spelling[0] == '/' && Spelling[1] == '/' && "Not bcpl comment?");
    Spelling[1] = '*';   // Change prefix to "/*".
    Spelling += "*/";    // add suffix.
    
    Result.setLocation(PP->CreateString(&Spelling[0], Spelling.size(),
                                        Result.getLocation()));
    Result.setLength(Spelling.size());
  }
  return false;
}

/// isBlockCommentEndOfEscapedNewLine - Return true if the specified newline
/// character (either \n or \r) is part of an escaped newline sequence.  Issue a
/// diagnostic if so.  We know that the is inside of a block comment.
static bool isEndOfBlockCommentWithEscapedNewLine(const char *CurPtr, 
                                                  Lexer *L) {
  assert(CurPtr[0] == '\n' || CurPtr[0] == '\r');
  
  // Back up off the newline.
  --CurPtr;
  
  // If this is a two-character newline sequence, skip the other character.
  if (CurPtr[0] == '\n' || CurPtr[0] == '\r') {
    // \n\n or \r\r -> not escaped newline.
    if (CurPtr[0] == CurPtr[1])
      return false;
    // \n\r or \r\n -> skip the newline.
    --CurPtr;
  }
  
  // If we have horizontal whitespace, skip over it.  We allow whitespace
  // between the slash and newline.
  bool HasSpace = false;
  while (isHorizontalWhitespace(*CurPtr) || *CurPtr == 0) {
    --CurPtr;
    HasSpace = true;
  }
  
  // If we have a slash, we know this is an escaped newline.
  if (*CurPtr == '\\') {
    if (CurPtr[-1] != '*') return false;
  } else {
    // It isn't a slash, is it the ?? / trigraph?
    if (CurPtr[0] != '/' || CurPtr[-1] != '?' || CurPtr[-2] != '?' ||
        CurPtr[-3] != '*')
      return false;
    
    // This is the trigraph ending the comment.  Emit a stern warning!
    CurPtr -= 2;

    // If no trigraphs are enabled, warn that we ignored this trigraph and
    // ignore this * character.
    if (!L->getFeatures().Trigraphs) {
      L->Diag(CurPtr, diag::trigraph_ignored_block_comment);
      return false;
    }
    L->Diag(CurPtr, diag::trigraph_ends_block_comment);
  }
  
  // Warn about having an escaped newline between the */ characters.
  L->Diag(CurPtr, diag::escaped_newline_block_comment_end);
  
  // If there was space between the backslash and newline, warn about it.
  if (HasSpace) L->Diag(CurPtr, diag::backslash_newline_space);
  
  return true;
}

#ifdef __SSE2__
#include <emmintrin.h>
#elif __ALTIVEC__
#include <altivec.h>
#undef bool
#endif

/// SkipBlockComment - We have just read the /* characters from input.  Read
/// until we find the */ characters that terminate the comment.  Note that we
/// don't bother decoding trigraphs or escaped newlines in block comments,
/// because they cannot cause the comment to end.  The only thing that can
/// happen is the comment could end with an escaped newline between the */ end
/// of comment.
bool Lexer::SkipBlockComment(Token &Result, const char *CurPtr) {
  // Scan one character past where we should, looking for a '/' character.  Once
  // we find it, check to see if it was preceeded by a *.  This common
  // optimization helps people who like to put a lot of * characters in their
  // comments.

  // The first character we get with newlines and trigraphs skipped to handle
  // the degenerate /*/ case below correctly if the * has an escaped newline
  // after it.
  unsigned CharSize;
  unsigned char C = getCharAndSize(CurPtr, CharSize);
  CurPtr += CharSize;
  if (C == 0 && CurPtr == BufferEnd+1) {
    if (!LexingRawMode)
      Diag(BufferPtr, diag::err_unterminated_block_comment);
    BufferPtr = CurPtr-1;
    return true;
  }
  
  // Check to see if the first character after the '/*' is another /.  If so,
  // then this slash does not end the block comment, it is part of it.
  if (C == '/')
    C = *CurPtr++;
  
  while (1) {
    // Skip over all non-interesting characters until we find end of buffer or a
    // (probably ending) '/' character.
    if (CurPtr + 24 < BufferEnd) {
      // While not aligned to a 16-byte boundary.
      while (C != '/' && ((intptr_t)CurPtr & 0x0F) != 0)
        C = *CurPtr++;
      
      if (C == '/') goto FoundSlash;

#ifdef __SSE2__
      __m128i Slashes = _mm_set_epi8('/', '/', '/', '/', '/', '/', '/', '/',
                                     '/', '/', '/', '/', '/', '/', '/', '/');
      while (CurPtr+16 <= BufferEnd &&
             _mm_movemask_epi8(_mm_cmpeq_epi8(*(__m128i*)CurPtr, Slashes)) == 0)
        CurPtr += 16;
#elif __ALTIVEC__
      __vector unsigned char Slashes = {
        '/', '/', '/', '/',  '/', '/', '/', '/', 
        '/', '/', '/', '/',  '/', '/', '/', '/'
      };
      while (CurPtr+16 <= BufferEnd &&
             !vec_any_eq(*(vector unsigned char*)CurPtr, Slashes))
        CurPtr += 16;
#else    
      // Scan for '/' quickly.  Many block comments are very large.
      while (CurPtr[0] != '/' &&
             CurPtr[1] != '/' &&
             CurPtr[2] != '/' &&
             CurPtr[3] != '/' &&
             CurPtr+4 < BufferEnd) {
        CurPtr += 4;
      }
#endif
      
      // It has to be one of the bytes scanned, increment to it and read one.
      C = *CurPtr++;
    }
    
    // Loop to scan the remainder.
    while (C != '/' && C != '\0')
      C = *CurPtr++;
    
  FoundSlash:
    if (C == '/') {
      if (CurPtr[-2] == '*')  // We found the final */.  We're done!
        break;
      
      if ((CurPtr[-2] == '\n' || CurPtr[-2] == '\r')) {
        if (isEndOfBlockCommentWithEscapedNewLine(CurPtr-2, this)) {
          // We found the final */, though it had an escaped newline between the
          // * and /.  We're done!
          break;
        }
      }
      if (CurPtr[0] == '*' && CurPtr[1] != '/') {
        // If this is a /* inside of the comment, emit a warning.  Don't do this
        // if this is a /*/, which will end the comment.  This misses cases with
        // embedded escaped newlines, but oh well.
        Diag(CurPtr-1, diag::warn_nested_block_comment);
      }
    } else if (C == 0 && CurPtr == BufferEnd+1) {
      if (!LexingRawMode) Diag(BufferPtr, diag::err_unterminated_block_comment);
      // Note: the user probably forgot a */.  We could continue immediately
      // after the /*, but this would involve lexing a lot of what really is the
      // comment, which surely would confuse the parser.
      BufferPtr = CurPtr-1;
      return true;
    }
    C = *CurPtr++;
  }
  
  // If we are returning comments as tokens, return this comment as a token.
  if (inKeepCommentMode()) {
    Result.setKind(tok::comment);
    FormTokenWithChars(Result, CurPtr);
    return false;
  }

  // It is common for the tokens immediately after a /**/ comment to be
  // whitespace.  Instead of going through the big switch, handle it
  // efficiently now.
  if (isHorizontalWhitespace(*CurPtr)) {
    Result.setFlag(Token::LeadingSpace);
    SkipWhitespace(Result, CurPtr+1);
    return true;
  }

  // Otherwise, just return so that the next character will be lexed as a token.
  BufferPtr = CurPtr;
  Result.setFlag(Token::LeadingSpace);
  return true;
}

//===----------------------------------------------------------------------===//
// Primary Lexing Entry Points
//===----------------------------------------------------------------------===//

/// LexIncludeFilename - After the preprocessor has parsed a #include, lex and
/// (potentially) macro expand the filename.
void Lexer::LexIncludeFilename(Token &FilenameTok) {
  assert(ParsingPreprocessorDirective &&
         ParsingFilename == false &&
         "Must be in a preprocessing directive!");

  // We are now parsing a filename!
  ParsingFilename = true;
  
  // Lex the filename.
  Lex(FilenameTok);

  // We should have obtained the filename now.
  ParsingFilename = false;
  
  // No filename?
  if (FilenameTok.is(tok::eom))
    Diag(FilenameTok.getLocation(), diag::err_pp_expects_filename);
}

/// ReadToEndOfLine - Read the rest of the current preprocessor line as an
/// uninterpreted string.  This switches the lexer out of directive mode.
std::string Lexer::ReadToEndOfLine() {
  assert(ParsingPreprocessorDirective && ParsingFilename == false &&
         "Must be in a preprocessing directive!");
  std::string Result;
  Token Tmp;

  // CurPtr - Cache BufferPtr in an automatic variable.
  const char *CurPtr = BufferPtr;
  while (1) {
    char Char = getAndAdvanceChar(CurPtr, Tmp);
    switch (Char) {
    default:
      Result += Char;
      break;
    case 0:  // Null.
      // Found end of file?
      if (CurPtr-1 != BufferEnd) {
        // Nope, normal character, continue.
        Result += Char;
        break;
      }
      // FALL THROUGH.
    case '\r':
    case '\n':
      // Okay, we found the end of the line. First, back up past the \0, \r, \n.
      assert(CurPtr[-1] == Char && "Trigraphs for newline?");
      BufferPtr = CurPtr-1;
      
      // Next, lex the character, which should handle the EOM transition.
      Lex(Tmp);
      assert(Tmp.is(tok::eom) && "Unexpected token!");
      
      // Finally, we're done, return the string we found.
      return Result;
    }
  }
}

/// LexEndOfFile - CurPtr points to the end of this file.  Handle this
/// condition, reporting diagnostics and handling other edge cases as required.
/// This returns true if Result contains a token, false if PP.Lex should be
/// called again.
bool Lexer::LexEndOfFile(Token &Result, const char *CurPtr) {
  // If we hit the end of the file while parsing a preprocessor directive,
  // end the preprocessor directive first.  The next token returned will
  // then be the end of file.
  if (ParsingPreprocessorDirective) {
    // Done parsing the "line".
    ParsingPreprocessorDirective = false;
    Result.setKind(tok::eom);
    // Update the location of token as well as BufferPtr.
    FormTokenWithChars(Result, CurPtr);
    
    // Restore comment saving mode, in case it was disabled for directive.
    KeepCommentMode = PP->getCommentRetentionState();
    return true;  // Have a token.
  }        

  // If we are in raw mode, return this event as an EOF token.  Let the caller
  // that put us in raw mode handle the event.
  if (LexingRawMode) {
    Result.startToken();
    BufferPtr = BufferEnd;
    FormTokenWithChars(Result, BufferEnd);
    Result.setKind(tok::eof);
    return true;
  }
  
  // Otherwise, issue diagnostics for unterminated #if and missing newline.

  // If we are in a #if directive, emit an error.
  while (!ConditionalStack.empty()) {
    Diag(ConditionalStack.back().IfLoc, diag::err_pp_unterminated_conditional);
    ConditionalStack.pop_back();
  }
  
  // C99 5.1.1.2p2: If the file is non-empty and didn't end in a newline, issue
  // a pedwarn.
  if (CurPtr != BufferStart && (CurPtr[-1] != '\n' && CurPtr[-1] != '\r'))
    Diag(BufferEnd, diag::ext_no_newline_eof);
  
  BufferPtr = CurPtr;

  // Finally, let the preprocessor handle this.
  return PP->HandleEndOfFile(Result);
}

/// isNextPPTokenLParen - Return 1 if the next unexpanded token lexed from
/// the specified lexer will return a tok::l_paren token, 0 if it is something
/// else and 2 if there are no more tokens in the buffer controlled by the
/// lexer.
unsigned Lexer::isNextPPTokenLParen() {
  assert(!LexingRawMode && "How can we expand a macro from a skipping buffer?");
  
  // Switch to 'skipping' mode.  This will ensure that we can lex a token
  // without emitting diagnostics, disables macro expansion, and will cause EOF
  // to return an EOF token instead of popping the include stack.
  LexingRawMode = true;
  
  // Save state that can be changed while lexing so that we can restore it.
  const char *TmpBufferPtr = BufferPtr;
  
  Token Tok;
  Tok.startToken();
  LexTokenInternal(Tok);
  
  // Restore state that may have changed.
  BufferPtr = TmpBufferPtr;
  
  // Restore the lexer back to non-skipping mode.
  LexingRawMode = false;
  
  if (Tok.is(tok::eof))
    return 2;
  return Tok.is(tok::l_paren);
}


/// LexTokenInternal - This implements a simple C family lexer.  It is an
/// extremely performance critical piece of code.  This assumes that the buffer
/// has a null character at the end of the file.  Return true if an error
/// occurred and compilation should terminate, false if normal.  This returns a
/// preprocessing token, not a normal token, as such, it is an internal
/// interface.  It assumes that the Flags of result have been cleared before
/// calling this.
void Lexer::LexTokenInternal(Token &Result) {
LexNextToken:
  // New token, can't need cleaning yet.
  Result.clearFlag(Token::NeedsCleaning);
  Result.setIdentifierInfo(0);
  
  // CurPtr - Cache BufferPtr in an automatic variable.
  const char *CurPtr = BufferPtr;

  // Small amounts of horizontal whitespace is very common between tokens.
  if ((*CurPtr == ' ') || (*CurPtr == '\t')) {
    ++CurPtr;
    while ((*CurPtr == ' ') || (*CurPtr == '\t'))
      ++CurPtr;
    BufferPtr = CurPtr;
    Result.setFlag(Token::LeadingSpace);
  }
  
  unsigned SizeTmp, SizeTmp2;   // Temporaries for use in cases below.
  
  // Read a character, advancing over it.
  char Char = getAndAdvanceChar(CurPtr, Result);
  switch (Char) {
  case 0:  // Null.
    // Found end of file?
    if (CurPtr-1 == BufferEnd) {
      // Read the PP instance variable into an automatic variable, because
      // LexEndOfFile will often delete 'this'.
      Preprocessor *PPCache = PP;
      if (LexEndOfFile(Result, CurPtr-1))  // Retreat back into the file.
        return;   // Got a token to return.
      assert(PPCache && "Raw buffer::LexEndOfFile should return a token");
      return PPCache->Lex(Result);
    }
    
    Diag(CurPtr-1, diag::null_in_file);
    Result.setFlag(Token::LeadingSpace);
    SkipWhitespace(Result, CurPtr);
    goto LexNextToken;   // GCC isn't tail call eliminating.
  case '\n':
  case '\r':
    // If we are inside a preprocessor directive and we see the end of line,
    // we know we are done with the directive, so return an EOM token.
    if (ParsingPreprocessorDirective) {
      // Done parsing the "line".
      ParsingPreprocessorDirective = false;
      
      // Restore comment saving mode, in case it was disabled for directive.
      KeepCommentMode = PP->getCommentRetentionState();
      
      // Since we consumed a newline, we are back at the start of a line.
      IsAtStartOfLine = true;
      
      Result.setKind(tok::eom);
      break;
    }
    // The returned token is at the start of the line.
    Result.setFlag(Token::StartOfLine);
    // No leading whitespace seen so far.
    Result.clearFlag(Token::LeadingSpace);
    SkipWhitespace(Result, CurPtr);
    goto LexNextToken;   // GCC isn't tail call eliminating.
  case ' ':
  case '\t':
  case '\f':
  case '\v':
  SkipHorizontalWhitespace:
    Result.setFlag(Token::LeadingSpace);
    SkipWhitespace(Result, CurPtr);

  SkipIgnoredUnits:
    CurPtr = BufferPtr;
    
    // If the next token is obviously a // or /* */ comment, skip it efficiently
    // too (without going through the big switch stmt).
    if (CurPtr[0] == '/' && CurPtr[1] == '/' && !inKeepCommentMode()) {
      SkipBCPLComment(Result, CurPtr+2);
      goto SkipIgnoredUnits;
    } else if (CurPtr[0] == '/' && CurPtr[1] == '*' && !inKeepCommentMode()) {
      SkipBlockComment(Result, CurPtr+2);
      goto SkipIgnoredUnits;
    } else if (isHorizontalWhitespace(*CurPtr)) {
      goto SkipHorizontalWhitespace;
    }
    goto LexNextToken;   // GCC isn't tail call eliminating.

  // C99 6.4.4.1: Integer Constants.
  // C99 6.4.4.2: Floating Constants.
  case '0': case '1': case '2': case '3': case '4':
  case '5': case '6': case '7': case '8': case '9':
    // Notify MIOpt that we read a non-whitespace/non-comment token.
    MIOpt.ReadToken();
    return LexNumericConstant(Result, CurPtr);
    
  case 'L':   // Identifier (Loony) or wide literal (L'x' or L"xyz").
    // Notify MIOpt that we read a non-whitespace/non-comment token.
    MIOpt.ReadToken();
    Char = getCharAndSize(CurPtr, SizeTmp);

    // Wide string literal.
    if (Char == '"')
      return LexStringLiteral(Result, ConsumeChar(CurPtr, SizeTmp, Result),
                              true);

    // Wide character constant.
    if (Char == '\'')
      return LexCharConstant(Result, ConsumeChar(CurPtr, SizeTmp, Result));
    // FALL THROUGH, treating L like the start of an identifier.
    
  // C99 6.4.2: Identifiers.
  case 'A': case 'B': case 'C': case 'D': case 'E': case 'F': case 'G':
  case 'H': case 'I': case 'J': case 'K':    /*'L'*/case 'M': case 'N':
  case 'O': case 'P': case 'Q': case 'R': case 'S': case 'T': case 'U':
  case 'V': case 'W': case 'X': case 'Y': case 'Z':
  case 'a': case 'b': case 'c': case 'd': case 'e': case 'f': case 'g':
  case 'h': case 'i': case 'j': case 'k': case 'l': case 'm': case 'n':
  case 'o': case 'p': case 'q': case 'r': case 's': case 't': case 'u':
  case 'v': case 'w': case 'x': case 'y': case 'z':
  case '_':
    // Notify MIOpt that we read a non-whitespace/non-comment token.
    MIOpt.ReadToken();
    return LexIdentifier(Result, CurPtr);

  case '$':   // $ in identifiers.
    if (Features.DollarIdents) {
      Diag(CurPtr-1, diag::ext_dollar_in_identifier);
      // Notify MIOpt that we read a non-whitespace/non-comment token.
      MIOpt.ReadToken();
      return LexIdentifier(Result, CurPtr);
    }
    
    Result.setKind(tok::unknown);
    break;
    
  // C99 6.4.4: Character Constants.
  case '\'':
    // Notify MIOpt that we read a non-whitespace/non-comment token.
    MIOpt.ReadToken();
    return LexCharConstant(Result, CurPtr);

  // C99 6.4.5: String Literals.
  case '"':
    // Notify MIOpt that we read a non-whitespace/non-comment token.
    MIOpt.ReadToken();
    return LexStringLiteral(Result, CurPtr, false);

  // C99 6.4.6: Punctuators.
  case '?':
    Result.setKind(tok::question);
    break;
  case '[':
    Result.setKind(tok::l_square);
    break;
  case ']':
    Result.setKind(tok::r_square);
    break;
  case '(':
    Result.setKind(tok::l_paren);
    break;
  case ')':
    Result.setKind(tok::r_paren);
    break;
  case '{':
    Result.setKind(tok::l_brace);
    break;
  case '}':
    Result.setKind(tok::r_brace);
    break;
  case '.':
    Char = getCharAndSize(CurPtr, SizeTmp);
    if (Char >= '0' && Char <= '9') {
      // Notify MIOpt that we read a non-whitespace/non-comment token.
      MIOpt.ReadToken();

      return LexNumericConstant(Result, ConsumeChar(CurPtr, SizeTmp, Result));
    } else if (Features.CPlusPlus && Char == '*') {
      Result.setKind(tok::periodstar);
      CurPtr += SizeTmp;
    } else if (Char == '.' &&
               getCharAndSize(CurPtr+SizeTmp, SizeTmp2) == '.') {
      Result.setKind(tok::ellipsis);
      CurPtr = ConsumeChar(ConsumeChar(CurPtr, SizeTmp, Result),
                           SizeTmp2, Result);
    } else {
      Result.setKind(tok::period);
    }
    break;
  case '&':
    Char = getCharAndSize(CurPtr, SizeTmp);
    if (Char == '&') {
      Result.setKind(tok::ampamp);
      CurPtr = ConsumeChar(CurPtr, SizeTmp, Result);
    } else if (Char == '=') {
      Result.setKind(tok::ampequal);
      CurPtr = ConsumeChar(CurPtr, SizeTmp, Result);
    } else {
      Result.setKind(tok::amp);
    }
    break;
  case '*': 
    if (getCharAndSize(CurPtr, SizeTmp) == '=') {
      Result.setKind(tok::starequal);
      CurPtr = ConsumeChar(CurPtr, SizeTmp, Result);
    } else {
      Result.setKind(tok::star);
    }
    break;
  case '+':
    Char = getCharAndSize(CurPtr, SizeTmp);
    if (Char == '+') {
      Result.setKind(tok::plusplus);
      CurPtr = ConsumeChar(CurPtr, SizeTmp, Result);
    } else if (Char == '=') {
      Result.setKind(tok::plusequal);
      CurPtr = ConsumeChar(CurPtr, SizeTmp, Result);
    } else {
      Result.setKind(tok::plus);
    }
    break;
  case '-':
    Char = getCharAndSize(CurPtr, SizeTmp);
    if (Char == '-') {
      Result.setKind(tok::minusminus);
      CurPtr = ConsumeChar(CurPtr, SizeTmp, Result);
    } else if (Char == '>' && Features.CPlusPlus && 
               getCharAndSize(CurPtr+SizeTmp, SizeTmp2) == '*') {
      Result.setKind(tok::arrowstar);  // C++ ->*
      CurPtr = ConsumeChar(ConsumeChar(CurPtr, SizeTmp, Result),
                           SizeTmp2, Result);
    } else if (Char == '>') {
      Result.setKind(tok::arrow);
      CurPtr = ConsumeChar(CurPtr, SizeTmp, Result);
    } else if (Char == '=') {
      Result.setKind(tok::minusequal);
      CurPtr = ConsumeChar(CurPtr, SizeTmp, Result);
    } else {
      Result.setKind(tok::minus);
    }
    break;
  case '~':
    Result.setKind(tok::tilde);
    break;
  case '!':
    if (getCharAndSize(CurPtr, SizeTmp) == '=') {
      Result.setKind(tok::exclaimequal);
      CurPtr = ConsumeChar(CurPtr, SizeTmp, Result);
    } else {
      Result.setKind(tok::exclaim);
    }
    break;
  case '/':
    // 6.4.9: Comments
    Char = getCharAndSize(CurPtr, SizeTmp);
    if (Char == '/') {         // BCPL comment.
      if (SkipBCPLComment(Result, ConsumeChar(CurPtr, SizeTmp, Result))) {
        // It is common for the tokens immediately after a // comment to be
        // whitespace (indentation for the next line).  Instead of going through
        // the big switch, handle it efficiently now.
        goto SkipIgnoredUnits;
      }        
      return; // KeepCommentMode
    } else if (Char == '*') {  // /**/ comment.
      if (SkipBlockComment(Result, ConsumeChar(CurPtr, SizeTmp, Result)))
        goto LexNextToken;   // GCC isn't tail call eliminating.
      return; // KeepCommentMode
    } else if (Char == '=') {
      Result.setKind(tok::slashequal);
      CurPtr = ConsumeChar(CurPtr, SizeTmp, Result);
    } else {
      Result.setKind(tok::slash);
    }
    break;
  case '%':
    Char = getCharAndSize(CurPtr, SizeTmp);
    if (Char == '=') {
      Result.setKind(tok::percentequal);
      CurPtr = ConsumeChar(CurPtr, SizeTmp, Result);
    } else if (Features.Digraphs && Char == '>') {
      Result.setKind(tok::r_brace);    // '%>' -> '}'
      CurPtr = ConsumeChar(CurPtr, SizeTmp, Result);
    } else if (Features.Digraphs && Char == ':') {
      CurPtr = ConsumeChar(CurPtr, SizeTmp, Result);
      Char = getCharAndSize(CurPtr, SizeTmp);
      if (Char == '%' && getCharAndSize(CurPtr+SizeTmp, SizeTmp2) == ':') {
        Result.setKind(tok::hashhash);   // '%:%:' -> '##'
        CurPtr = ConsumeChar(ConsumeChar(CurPtr, SizeTmp, Result),
                             SizeTmp2, Result);
      } else if (Char == '@' && Features.Microsoft) {  // %:@ -> #@ -> Charize
        Result.setKind(tok::hashat);
        CurPtr = ConsumeChar(CurPtr, SizeTmp, Result);
        Diag(BufferPtr, diag::charize_microsoft_ext);
      } else {
        Result.setKind(tok::hash);       // '%:' -> '#'
        
        // We parsed a # character.  If this occurs at the start of the line,
        // it's actually the start of a preprocessing directive.  Callback to
        // the preprocessor to handle it.
        // FIXME: -fpreprocessed mode??
        if (Result.isAtStartOfLine() && !LexingRawMode) {
          BufferPtr = CurPtr;
          PP->HandleDirective(Result);
          
          // As an optimization, if the preprocessor didn't switch lexers, tail
          // recurse.
          if (PP->isCurrentLexer(this)) {
            // Start a new token. If this is a #include or something, the PP may
            // want us starting at the beginning of the line again.  If so, set
            // the StartOfLine flag.
            if (IsAtStartOfLine) {
              Result.setFlag(Token::StartOfLine);
              IsAtStartOfLine = false;
            }
            goto LexNextToken;   // GCC isn't tail call eliminating.
          }
          
          return PP->Lex(Result);
        }
      }
    } else {
      Result.setKind(tok::percent);
    }
    break;
  case '<':
    Char = getCharAndSize(CurPtr, SizeTmp);
    if (ParsingFilename) {
      return LexAngledStringLiteral(Result, CurPtr+SizeTmp);
    } else if (Char == '<' &&
               getCharAndSize(CurPtr+SizeTmp, SizeTmp2) == '=') {
      Result.setKind(tok::lesslessequal);
      CurPtr = ConsumeChar(ConsumeChar(CurPtr, SizeTmp, Result),
                           SizeTmp2, Result);
    } else if (Char == '<') {
      Result.setKind(tok::lessless);
      CurPtr = ConsumeChar(CurPtr, SizeTmp, Result);
    } else if (Char == '=') {
      Result.setKind(tok::lessequal);
      CurPtr = ConsumeChar(CurPtr, SizeTmp, Result);
    } else if (Features.Digraphs && Char == ':') {
      Result.setKind(tok::l_square); // '<:' -> '['
      CurPtr = ConsumeChar(CurPtr, SizeTmp, Result);
    } else if (Features.Digraphs && Char == '%') {
      Result.setKind(tok::l_brace); // '<%' -> '{'
      CurPtr = ConsumeChar(CurPtr, SizeTmp, Result);
    } else {
      Result.setKind(tok::less);
    }
    break;
  case '>':
    Char = getCharAndSize(CurPtr, SizeTmp);
    if (Char == '=') {
      Result.setKind(tok::greaterequal);
      CurPtr = ConsumeChar(CurPtr, SizeTmp, Result);
    } else if (Char == '>' && 
               getCharAndSize(CurPtr+SizeTmp, SizeTmp2) == '=') {
      Result.setKind(tok::greatergreaterequal);
      CurPtr = ConsumeChar(ConsumeChar(CurPtr, SizeTmp, Result),
                           SizeTmp2, Result);
    } else if (Char == '>') {
      Result.setKind(tok::greatergreater);
      CurPtr = ConsumeChar(CurPtr, SizeTmp, Result);
    } else {
      Result.setKind(tok::greater);
    }
    break;
  case '^':
    Char = getCharAndSize(CurPtr, SizeTmp);
    if (Char == '=') {
      Result.setKind(tok::caretequal);
      CurPtr = ConsumeChar(CurPtr, SizeTmp, Result);
    } else {
      Result.setKind(tok::caret);
    }
    break;
  case '|':
    Char = getCharAndSize(CurPtr, SizeTmp);
    if (Char == '=') {
      Result.setKind(tok::pipeequal);
      CurPtr = ConsumeChar(CurPtr, SizeTmp, Result);
    } else if (Char == '|') {
      Result.setKind(tok::pipepipe);
      CurPtr = ConsumeChar(CurPtr, SizeTmp, Result);
    } else {
      Result.setKind(tok::pipe);
    }
    break;
  case ':':
    Char = getCharAndSize(CurPtr, SizeTmp);
    if (Features.Digraphs && Char == '>') {
      Result.setKind(tok::r_square); // ':>' -> ']'
      CurPtr = ConsumeChar(CurPtr, SizeTmp, Result);
    } else if (Features.CPlusPlus && Char == ':') {
      Result.setKind(tok::coloncolon);
      CurPtr = ConsumeChar(CurPtr, SizeTmp, Result);
    } else {    
      Result.setKind(tok::colon);
    }
    break;
  case ';':
    Result.setKind(tok::semi);
    break;
  case '=':
    Char = getCharAndSize(CurPtr, SizeTmp);
    if (Char == '=') {
      Result.setKind(tok::equalequal);
      CurPtr = ConsumeChar(CurPtr, SizeTmp, Result);
    } else {      
      Result.setKind(tok::equal);
    }
    break;
  case ',':
    Result.setKind(tok::comma);
    break;
  case '#':
    Char = getCharAndSize(CurPtr, SizeTmp);
    if (Char == '#') {
      Result.setKind(tok::hashhash);
      CurPtr = ConsumeChar(CurPtr, SizeTmp, Result);
    } else if (Char == '@' && Features.Microsoft) {  // #@ -> Charize
      Result.setKind(tok::hashat);
      Diag(BufferPtr, diag::charize_microsoft_ext);
      CurPtr = ConsumeChar(CurPtr, SizeTmp, Result);
    } else {
      Result.setKind(tok::hash);
      // We parsed a # character.  If this occurs at the start of the line,
      // it's actually the start of a preprocessing directive.  Callback to
      // the preprocessor to handle it.
      // FIXME: -fpreprocessed mode??
      if (Result.isAtStartOfLine() && !LexingRawMode) {
        BufferPtr = CurPtr;
        PP->HandleDirective(Result);
        
        // As an optimization, if the preprocessor didn't switch lexers, tail
        // recurse.
        if (PP->isCurrentLexer(this)) {
          // Start a new token.  If this is a #include or something, the PP may
          // want us starting at the beginning of the line again.  If so, set
          // the StartOfLine flag.
          if (IsAtStartOfLine) {
            Result.setFlag(Token::StartOfLine);
            IsAtStartOfLine = false;
          }
          goto LexNextToken;   // GCC isn't tail call eliminating.
        }
        return PP->Lex(Result);
      }
    }
    break;

  case '@':
    // Objective C support.
    if (CurPtr[-1] == '@' && Features.ObjC1)
      Result.setKind(tok::at);
    else
      Result.setKind(tok::unknown);
    break;
    
  case '\\':
    // FIXME: UCN's.
    // FALL THROUGH.
  default:
    Result.setKind(tok::unknown);
    break;
  }
  
  // Notify MIOpt that we read a non-whitespace/non-comment token.
  MIOpt.ReadToken();

  // Update the location of token as well as BufferPtr.
  FormTokenWithChars(Result, CurPtr);
}
