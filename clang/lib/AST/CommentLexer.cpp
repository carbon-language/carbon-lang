#include "clang/AST/CommentLexer.h"
#include "clang/AST/CommentCommandTraits.h"
#include "clang/Basic/ConvertUTF.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"

namespace clang {
namespace comments {

void Token::dump(const Lexer &L, const SourceManager &SM) const {
  llvm::errs() << "comments::Token Kind=" << Kind << " ";
  Loc.dump(SM);
  llvm::errs() << " " << Length << " \"" << L.getSpelling(*this, SM) << "\"\n";
}

namespace {
bool isHTMLNamedCharacterReferenceCharacter(char C) {
  return (C >= 'a' && C <= 'z') ||
         (C >= 'A' && C <= 'Z');
}

bool isHTMLDecimalCharacterReferenceCharacter(char C) {
  return C >= '0' && C <= '9';
}

bool isHTMLHexCharacterReferenceCharacter(char C) {
  return (C >= '0' && C <= '9') ||
         (C >= 'a' && C <= 'f') ||
         (C >= 'A' && C <= 'F');
}

#include "clang/AST/CommentHTMLTags.inc"

} // unnamed namespace

static unsigned getCodePoint(StringRef Name) {
  unsigned CodePoint = 0;
  for (unsigned i = 0, e = Name.size(); i != e; ++i) {
    CodePoint *= 16;
    const char C = Name[i];
    assert(isHTMLHexCharacterReferenceCharacter(C));
    CodePoint += llvm::hexDigitValue(C);
  }
  return CodePoint;
}

StringRef Lexer::helperResolveHTMLHexCharacterReference(unsigned CodePoint) const {
  char *Resolved = Allocator.Allocate<char>(UNI_MAX_UTF8_BYTES_PER_CODE_POINT);
  char *ResolvedPtr = Resolved;
  if (ConvertCodePointToUTF8(CodePoint, ResolvedPtr))
    return StringRef(Resolved, ResolvedPtr - Resolved);
  else
    return StringRef();
}
  
StringRef Lexer::resolveHTMLHexCharacterReference(StringRef Name) const {
  unsigned CodePoint = getCodePoint(Name);
  return helperResolveHTMLHexCharacterReference(CodePoint);
}

StringRef Lexer::resolveHTMLNamedCharacterReference(StringRef Name) const {
  return llvm::StringSwitch<StringRef>(Name)
      .Case("amp", "&")
      .Case("lt", "<")
      .Case("gt", ">")
      .Case("quot", "\"")
      .Case("apos", "\'")
      .Default("");
}
  
StringRef Lexer::HTMLDoxygenCharacterReference(StringRef Name) const {
  return llvm::StringSwitch<StringRef>(Name)
  .Case("copy", helperResolveHTMLHexCharacterReference(0x000A9))
  .Case("trade",        helperResolveHTMLHexCharacterReference(0x02122))
  .Case("reg",  helperResolveHTMLHexCharacterReference(0x000AE))
  .Case("lt",   helperResolveHTMLHexCharacterReference(0x0003C))
  .Case("gt",   helperResolveHTMLHexCharacterReference(0x0003C))
  .Case("amp",  helperResolveHTMLHexCharacterReference(0x00026))
  .Case("apos", helperResolveHTMLHexCharacterReference(0x00027))
  .Case("quot", helperResolveHTMLHexCharacterReference(0x00022))
  .Case("lsquo",        helperResolveHTMLHexCharacterReference(0x02018))
  .Case("rsquo",        helperResolveHTMLHexCharacterReference(0x02019))
  .Case("ldquo",        helperResolveHTMLHexCharacterReference(0x0201C))
  .Case("rdquo",        helperResolveHTMLHexCharacterReference(0x0201D))
  .Case("ndash",        helperResolveHTMLHexCharacterReference(0x02013))
  .Case("mdash",        helperResolveHTMLHexCharacterReference(0x02014))
  .Case("Auml", helperResolveHTMLHexCharacterReference(0x000C4))
  .Case("Euml", helperResolveHTMLHexCharacterReference(0x000CB))
  .Case("Iuml", helperResolveHTMLHexCharacterReference(0x000CF))
  .Case("Ouml", helperResolveHTMLHexCharacterReference(0x000D6))
  .Case("Uuml", helperResolveHTMLHexCharacterReference(0x000DC))
  .Case("Yuml", helperResolveHTMLHexCharacterReference(0x00178))
  .Case("auml", helperResolveHTMLHexCharacterReference(0x000E4))
  .Case("euml", helperResolveHTMLHexCharacterReference(0x000EB))
  .Case("iuml", helperResolveHTMLHexCharacterReference(0x000EF))
  .Case("ouml", helperResolveHTMLHexCharacterReference(0x000F6))
  .Case("uuml", helperResolveHTMLHexCharacterReference(0x000FC))
  .Case("yuml", helperResolveHTMLHexCharacterReference(0x000FF))
  .Case("Aacute",       helperResolveHTMLHexCharacterReference(0x000C1))
  .Case("Eacute",       helperResolveHTMLHexCharacterReference(0x000C9))
  .Case("Iacute",       helperResolveHTMLHexCharacterReference(0x000CD))
  .Case("Oacute",       helperResolveHTMLHexCharacterReference(0x000D3))
  .Case("Uacute",       helperResolveHTMLHexCharacterReference(0x000DA))
  .Case("Yacute",       helperResolveHTMLHexCharacterReference(0x000DD))
  .Case("aacute",       helperResolveHTMLHexCharacterReference(0x000E1))
  .Case("eacute",       helperResolveHTMLHexCharacterReference(0x000E9))
  .Case("iacute",       helperResolveHTMLHexCharacterReference(0x000ED))
  .Case("oacute",       helperResolveHTMLHexCharacterReference(0x000F3))
  .Case("uacute",       helperResolveHTMLHexCharacterReference(0x000FA))
  .Case("yacute",       helperResolveHTMLHexCharacterReference(0x000FD))
  .Case("Agrave",       helperResolveHTMLHexCharacterReference(0x000C0))
  .Case("Egrave",       helperResolveHTMLHexCharacterReference(0x000C8))
  .Case("Igrave",       helperResolveHTMLHexCharacterReference(0x000CC))
  .Case("Ograve",       helperResolveHTMLHexCharacterReference(0x000D2))
  .Case("Ugrave",       helperResolveHTMLHexCharacterReference(0x000D9))
  .Case("agrave",       helperResolveHTMLHexCharacterReference(0x000E0))
  .Case("egrave",       helperResolveHTMLHexCharacterReference(0x000E8))
  .Case("igrave",       helperResolveHTMLHexCharacterReference(0x000EC))
  .Case("ograve",       helperResolveHTMLHexCharacterReference(0x000F2))
  .Case("ugrave",       helperResolveHTMLHexCharacterReference(0x000F9))
  .Case("ygrave",       helperResolveHTMLHexCharacterReference(0x01EF3))
  .Case("Acirc",        helperResolveHTMLHexCharacterReference(0x000C2))
  .Case("Ecirc",        helperResolveHTMLHexCharacterReference(0x000CA))
  .Case("Icirc",        helperResolveHTMLHexCharacterReference(0x000CE))
  .Case("Ocirc",        helperResolveHTMLHexCharacterReference(0x000D4))
  .Case("Ucirc",        helperResolveHTMLHexCharacterReference(0x000DB))
  .Case("acirc",        helperResolveHTMLHexCharacterReference(0x000E2))
  .Case("ecirc",        helperResolveHTMLHexCharacterReference(0x000EA))
  .Case("icirc",        helperResolveHTMLHexCharacterReference(0x000EE))
  .Case("ocirc",        helperResolveHTMLHexCharacterReference(0x000F4))
  .Case("ucirc",        helperResolveHTMLHexCharacterReference(0x000FB))
  .Case("ycirc",        helperResolveHTMLHexCharacterReference(0x00177))
  .Case("Atilde",       helperResolveHTMLHexCharacterReference(0x000C3))
  .Case("Ntilde",       helperResolveHTMLHexCharacterReference(0x000D1))
  .Case("Otilde",       helperResolveHTMLHexCharacterReference(0x000D5))
  .Case("atilde",       helperResolveHTMLHexCharacterReference(0x000E3))
  .Case("ntilde",       helperResolveHTMLHexCharacterReference(0x000F1))
  .Case("otilde",       helperResolveHTMLHexCharacterReference(0x000F5))
  .Case("szlig",        helperResolveHTMLHexCharacterReference(0x000DF))
  .Case("ccedil",       helperResolveHTMLHexCharacterReference(0x000E7))
  .Case("Ccedil",       helperResolveHTMLHexCharacterReference(0x000C7))
  .Case("aring",        helperResolveHTMLHexCharacterReference(0x000E5))
  .Case("Aring",        helperResolveHTMLHexCharacterReference(0x000C5))
  .Case("nbsp", helperResolveHTMLHexCharacterReference(0x000A0))
  .Case("Gamma",        helperResolveHTMLHexCharacterReference(0x00393))
  .Case("Delta",        helperResolveHTMLHexCharacterReference(0x00394))
  .Case("Theta",        helperResolveHTMLHexCharacterReference(0x00398))
  .Case("Lambda",       helperResolveHTMLHexCharacterReference(0x0039B))
  .Case("Xi",   helperResolveHTMLHexCharacterReference(0x0039E))
  .Case("Pi",   helperResolveHTMLHexCharacterReference(0x003A0))
  .Case("Sigma",        helperResolveHTMLHexCharacterReference(0x003A3))
  .Case("Upsilon",      helperResolveHTMLHexCharacterReference(0x003A5))
  .Case("Phi",  helperResolveHTMLHexCharacterReference(0x003A6))
  .Case("Psi",  helperResolveHTMLHexCharacterReference(0x003A8))
  .Case("Omega",        helperResolveHTMLHexCharacterReference(0x003A9))
  .Case("alpha",        helperResolveHTMLHexCharacterReference(0x003B1))
  .Case("beta", helperResolveHTMLHexCharacterReference(0x003B2))
  .Case("gamma",        helperResolveHTMLHexCharacterReference(0x003B3))
  .Case("delta",        helperResolveHTMLHexCharacterReference(0x003B4))
  .Case("epsilon",      helperResolveHTMLHexCharacterReference(0x003B5))
  .Case("zeta", helperResolveHTMLHexCharacterReference(0x003B6))
  .Case("eta",  helperResolveHTMLHexCharacterReference(0x003B7))
  .Case("theta",        helperResolveHTMLHexCharacterReference(0x003B8))
  .Case("iota", helperResolveHTMLHexCharacterReference(0x003B9))
  .Case("kappa",        helperResolveHTMLHexCharacterReference(0x003BA))
  .Case("lambda",       helperResolveHTMLHexCharacterReference(0x003BB))
  .Case("mu",   helperResolveHTMLHexCharacterReference(0x003BC))
  .Case("nu",   helperResolveHTMLHexCharacterReference(0x003BD))
  .Case("xi",   helperResolveHTMLHexCharacterReference(0x003BE))
  .Case("pi",   helperResolveHTMLHexCharacterReference(0x003C0))
  .Case("rho",  helperResolveHTMLHexCharacterReference(0x003C1))
  .Case("sigma",        helperResolveHTMLHexCharacterReference(0x003C3))
  .Case("tau",  helperResolveHTMLHexCharacterReference(0x003C4))
  .Case("upsilon",      helperResolveHTMLHexCharacterReference(0x003C5))
  .Case("phi",  helperResolveHTMLHexCharacterReference(0x003C6))
  .Case("chi",  helperResolveHTMLHexCharacterReference(0x003C7))
  .Case("psi",  helperResolveHTMLHexCharacterReference(0x003C8))
  .Case("omega",        helperResolveHTMLHexCharacterReference(0x003C9))
  .Case("sigmaf",       helperResolveHTMLHexCharacterReference(0x003C2))
  .Case("sect", helperResolveHTMLHexCharacterReference(0x000A7))
  .Case("deg",  helperResolveHTMLHexCharacterReference(0x000B0))
  .Case("prime",        helperResolveHTMLHexCharacterReference(0x02032))
  .Case("Prime",        helperResolveHTMLHexCharacterReference(0x02033))
  .Case("infin",        helperResolveHTMLHexCharacterReference(0x0221E))
  .Case("empty",        helperResolveHTMLHexCharacterReference(0x02205))
  .Case("plusmn",       helperResolveHTMLHexCharacterReference(0x000B1))
  .Case("times",        helperResolveHTMLHexCharacterReference(0x000D7))
  .Case("minus",        helperResolveHTMLHexCharacterReference(0x02212))
  .Case("sdot", helperResolveHTMLHexCharacterReference(0x022C5))
  .Case("part", helperResolveHTMLHexCharacterReference(0x02202))
  .Case("nabla",        helperResolveHTMLHexCharacterReference(0x02207))
  .Case("radic",        helperResolveHTMLHexCharacterReference(0x0221A))
  .Case("perp", helperResolveHTMLHexCharacterReference(0x022A5))
  .Case("sum",  helperResolveHTMLHexCharacterReference(0x02211))
  .Case("int",  helperResolveHTMLHexCharacterReference(0x0222B))
  .Case("prod", helperResolveHTMLHexCharacterReference(0x0220F))
  .Case("sim",  helperResolveHTMLHexCharacterReference(0x0223C))
  .Case("asymp",        helperResolveHTMLHexCharacterReference(0x02248))
  .Case("ne",   helperResolveHTMLHexCharacterReference(0x02260))
  .Case("equiv",        helperResolveHTMLHexCharacterReference(0x02261))
  .Case("prop", helperResolveHTMLHexCharacterReference(0x0221D))
  .Case("le",   helperResolveHTMLHexCharacterReference(0x02264))
  .Case("ge",   helperResolveHTMLHexCharacterReference(0x02265))
  .Case("larr", helperResolveHTMLHexCharacterReference(0x02190))
  .Case("rarr", helperResolveHTMLHexCharacterReference(0x02192))
  .Case("isin", helperResolveHTMLHexCharacterReference(0x02208))
  .Case("notin",        helperResolveHTMLHexCharacterReference(0x02209))
  .Case("lceil",        helperResolveHTMLHexCharacterReference(0x02308))
  .Case("rceil",        helperResolveHTMLHexCharacterReference(0x02309))
  .Case("lfloor",       helperResolveHTMLHexCharacterReference(0x0230A))
  .Case("rfloor",       helperResolveHTMLHexCharacterReference(0x0230B))
  .Default("");
}

StringRef Lexer::resolveHTMLDecimalCharacterReference(StringRef Name) const {
  unsigned CodePoint = 0;
  for (unsigned i = 0, e = Name.size(); i != e; ++i) {
    assert(isHTMLDecimalCharacterReferenceCharacter(Name[i]));
    CodePoint *= 10;
    CodePoint += Name[i] - '0';
  }

  char *Resolved = Allocator.Allocate<char>(UNI_MAX_UTF8_BYTES_PER_CODE_POINT);
  char *ResolvedPtr = Resolved;
  if (ConvertCodePointToUTF8(CodePoint, ResolvedPtr))
    return StringRef(Resolved, ResolvedPtr - Resolved);
  else
    return StringRef();
}

void Lexer::skipLineStartingDecorations() {
  // This function should be called only for C comments
  assert(CommentState == LCS_InsideCComment);

  if (BufferPtr == CommentEnd)
    return;

  switch (*BufferPtr) {
  case ' ':
  case '\t':
  case '\f':
  case '\v': {
    const char *NewBufferPtr = BufferPtr;
    NewBufferPtr++;
    if (NewBufferPtr == CommentEnd)
      return;

    char C = *NewBufferPtr;
    while (C == ' ' || C == '\t' || C == '\f' || C == '\v') {
      NewBufferPtr++;
      if (NewBufferPtr == CommentEnd)
        return;
      C = *NewBufferPtr;
    }
    if (C == '*')
      BufferPtr = NewBufferPtr + 1;
    break;
  }
  case '*':
    BufferPtr++;
    break;
  }
}

namespace {
/// Returns pointer to the first newline character in the string.
const char *findNewline(const char *BufferPtr, const char *BufferEnd) {
  for ( ; BufferPtr != BufferEnd; ++BufferPtr) {
    const char C = *BufferPtr;
    if (C == '\n' || C == '\r')
      return BufferPtr;
  }
  return BufferEnd;
}

const char *skipNewline(const char *BufferPtr, const char *BufferEnd) {
  if (BufferPtr == BufferEnd)
    return BufferPtr;

  if (*BufferPtr == '\n')
    BufferPtr++;
  else {
    assert(*BufferPtr == '\r');
    BufferPtr++;
    if (BufferPtr != BufferEnd && *BufferPtr == '\n')
      BufferPtr++;
  }
  return BufferPtr;
}

const char *skipNamedCharacterReference(const char *BufferPtr,
                                        const char *BufferEnd) {
  for ( ; BufferPtr != BufferEnd; ++BufferPtr) {
    if (!isHTMLNamedCharacterReferenceCharacter(*BufferPtr))
      return BufferPtr;
  }
  return BufferEnd;
}

const char *skipDecimalCharacterReference(const char *BufferPtr,
                                          const char *BufferEnd) {
  for ( ; BufferPtr != BufferEnd; ++BufferPtr) {
    if (!isHTMLDecimalCharacterReferenceCharacter(*BufferPtr))
      return BufferPtr;
  }
  return BufferEnd;
}

const char *skipHexCharacterReference(const char *BufferPtr,
                                          const char *BufferEnd) {
  for ( ; BufferPtr != BufferEnd; ++BufferPtr) {
    if (!isHTMLHexCharacterReferenceCharacter(*BufferPtr))
      return BufferPtr;
  }
  return BufferEnd;
}

bool isHTMLIdentifierStartingCharacter(char C) {
  return (C >= 'a' && C <= 'z') ||
         (C >= 'A' && C <= 'Z');
}

bool isHTMLIdentifierCharacter(char C) {
  return (C >= 'a' && C <= 'z') ||
         (C >= 'A' && C <= 'Z') ||
         (C >= '0' && C <= '9');
}

const char *skipHTMLIdentifier(const char *BufferPtr, const char *BufferEnd) {
  for ( ; BufferPtr != BufferEnd; ++BufferPtr) {
    if (!isHTMLIdentifierCharacter(*BufferPtr))
      return BufferPtr;
  }
  return BufferEnd;
}

/// Skip HTML string quoted in single or double quotes.  Escaping quotes inside
/// string allowed.
///
/// Returns pointer to closing quote.
const char *skipHTMLQuotedString(const char *BufferPtr, const char *BufferEnd)
{
  const char Quote = *BufferPtr;
  assert(Quote == '\"' || Quote == '\'');

  BufferPtr++;
  for ( ; BufferPtr != BufferEnd; ++BufferPtr) {
    const char C = *BufferPtr;
    if (C == Quote && BufferPtr[-1] != '\\')
      return BufferPtr;
  }
  return BufferEnd;
}

bool isHorizontalWhitespace(char C) {
  return C == ' ' || C == '\t' || C == '\f' || C == '\v';
}

bool isWhitespace(char C) {
  return C == ' ' || C == '\n' || C == '\r' ||
         C == '\t' || C == '\f' || C == '\v';
}

const char *skipWhitespace(const char *BufferPtr, const char *BufferEnd) {
  for ( ; BufferPtr != BufferEnd; ++BufferPtr) {
    if (!isWhitespace(*BufferPtr))
      return BufferPtr;
  }
  return BufferEnd;
}

bool isWhitespace(const char *BufferPtr, const char *BufferEnd) {
  return skipWhitespace(BufferPtr, BufferEnd) == BufferEnd;
}

bool isCommandNameStartCharacter(char C) {
  return (C >= 'a' && C <= 'z') ||
         (C >= 'A' && C <= 'Z');
}

bool isCommandNameCharacter(char C) {
  return (C >= 'a' && C <= 'z') ||
         (C >= 'A' && C <= 'Z') ||
         (C >= '0' && C <= '9');
}

const char *skipCommandName(const char *BufferPtr, const char *BufferEnd) {
  for ( ; BufferPtr != BufferEnd; ++BufferPtr) {
    if (!isCommandNameCharacter(*BufferPtr))
      return BufferPtr;
  }
  return BufferEnd;
}

/// Return the one past end pointer for BCPL comments.
/// Handles newlines escaped with backslash or trigraph for backslahs.
const char *findBCPLCommentEnd(const char *BufferPtr, const char *BufferEnd) {
  const char *CurPtr = BufferPtr;
  while (CurPtr != BufferEnd) {
    char C = *CurPtr;
    while (C != '\n' && C != '\r') {
      CurPtr++;
      if (CurPtr == BufferEnd)
        return BufferEnd;
      C = *CurPtr;
    }
    // We found a newline, check if it is escaped.
    const char *EscapePtr = CurPtr - 1;
    while(isHorizontalWhitespace(*EscapePtr))
      EscapePtr--;

    if (*EscapePtr == '\\' ||
        (EscapePtr - 2 >= BufferPtr && EscapePtr[0] == '/' &&
         EscapePtr[-1] == '?' && EscapePtr[-2] == '?')) {
      // We found an escaped newline.
      CurPtr = skipNewline(CurPtr, BufferEnd);
    } else
      return CurPtr; // Not an escaped newline.
  }
  return BufferEnd;
}

/// Return the one past end pointer for C comments.
/// Very dumb, does not handle escaped newlines or trigraphs.
const char *findCCommentEnd(const char *BufferPtr, const char *BufferEnd) {
  for ( ; BufferPtr != BufferEnd; ++BufferPtr) {
    if (*BufferPtr == '*') {
      assert(BufferPtr + 1 != BufferEnd);
      if (*(BufferPtr + 1) == '/')
        return BufferPtr;
    }
  }
  llvm_unreachable("buffer end hit before '*/' was seen");
}
} // unnamed namespace

void Lexer::lexCommentText(Token &T) {
  assert(CommentState == LCS_InsideBCPLComment ||
         CommentState == LCS_InsideCComment);

  switch (State) {
  case LS_Normal:
    break;
  case LS_VerbatimBlockFirstLine:
    lexVerbatimBlockFirstLine(T);
    return;
  case LS_VerbatimBlockBody:
    lexVerbatimBlockBody(T);
    return;
  case LS_VerbatimLineText:
    lexVerbatimLineText(T);
    return;
  case LS_HTMLStartTag:
    lexHTMLStartTag(T);
    return;
  case LS_HTMLEndTag:
    lexHTMLEndTag(T);
    return;
  }

  assert(State == LS_Normal);

  const char *TokenPtr = BufferPtr;
  assert(TokenPtr < CommentEnd);
  while (TokenPtr != CommentEnd) {
    switch(*TokenPtr) {
      case '\\':
      case '@': {
        TokenPtr++;
        if (TokenPtr == CommentEnd) {
          formTextToken(T, TokenPtr);
          return;
        }
        char C = *TokenPtr;
        switch (C) {
        default:
          break;

        case '\\': case '@': case '&': case '$':
        case '#':  case '<': case '>': case '%':
        case '\"': case '.': case ':':
          // This is one of \\ \@ \& \$ etc escape sequences.
          TokenPtr++;
          if (C == ':' && TokenPtr != CommentEnd && *TokenPtr == ':') {
            // This is the \:: escape sequence.
            TokenPtr++;
          }
          StringRef UnescapedText(BufferPtr + 1, TokenPtr - (BufferPtr + 1));
          formTokenWithChars(T, TokenPtr, tok::text);
          T.setText(UnescapedText);
          return;
        }

        // Don't make zero-length commands.
        if (!isCommandNameStartCharacter(*TokenPtr)) {
          formTextToken(T, TokenPtr);
          return;
        }

        TokenPtr = skipCommandName(TokenPtr, CommentEnd);
        unsigned Length = TokenPtr - (BufferPtr + 1);

        // Hardcoded support for lexing LaTeX formula commands
        // \f$ \f[ \f] \f{ \f} as a single command.
        if (Length == 1 && TokenPtr[-1] == 'f' && TokenPtr != CommentEnd) {
          C = *TokenPtr;
          if (C == '$' || C == '[' || C == ']' || C == '{' || C == '}') {
            TokenPtr++;
            Length++;
          }
        }

        const StringRef CommandName(BufferPtr + 1, Length);

        const CommandInfo *Info = Traits.getCommandInfoOrNULL(CommandName);
        if (!Info) {
          formTokenWithChars(T, TokenPtr, tok::unknown_command);
          T.setUnknownCommandName(CommandName);
          return;
        }
        if (Info->IsVerbatimBlockCommand) {
          setupAndLexVerbatimBlock(T, TokenPtr, *BufferPtr, Info);
          return;
        }
        if (Info->IsVerbatimLineCommand) {
          setupAndLexVerbatimLine(T, TokenPtr, Info);
          return;
        }
        formTokenWithChars(T, TokenPtr, tok::command);
        T.setCommandID(Info->getID());
        return;
      }

      case '&':
        lexHTMLCharacterReference(T);
        return;

      case '<': {
        TokenPtr++;
        if (TokenPtr == CommentEnd) {
          formTextToken(T, TokenPtr);
          return;
        }
        const char C = *TokenPtr;
        if (isHTMLIdentifierStartingCharacter(C))
          setupAndLexHTMLStartTag(T);
        else if (C == '/')
          setupAndLexHTMLEndTag(T);
        else
          formTextToken(T, TokenPtr);

        return;
      }

      case '\n':
      case '\r':
        TokenPtr = skipNewline(TokenPtr, CommentEnd);
        formTokenWithChars(T, TokenPtr, tok::newline);

        if (CommentState == LCS_InsideCComment)
          skipLineStartingDecorations();
        return;

      default: {
        size_t End = StringRef(TokenPtr, CommentEnd - TokenPtr).
                         find_first_of("\n\r\\@&<");
        if (End != StringRef::npos)
          TokenPtr += End;
        else
          TokenPtr = CommentEnd;
        formTextToken(T, TokenPtr);
        return;
      }
    }
  }
}

void Lexer::setupAndLexVerbatimBlock(Token &T,
                                     const char *TextBegin,
                                     char Marker, const CommandInfo *Info) {
  assert(Info->IsVerbatimBlockCommand);

  VerbatimBlockEndCommandName.clear();
  VerbatimBlockEndCommandName.append(Marker == '\\' ? "\\" : "@");
  VerbatimBlockEndCommandName.append(Info->EndCommandName);

  formTokenWithChars(T, TextBegin, tok::verbatim_block_begin);
  T.setVerbatimBlockID(Info->getID());

  // If there is a newline following the verbatim opening command, skip the
  // newline so that we don't create an tok::verbatim_block_line with empty
  // text content.
  if (BufferPtr != CommentEnd) {
    const char C = *BufferPtr;
    if (C == '\n' || C == '\r') {
      BufferPtr = skipNewline(BufferPtr, CommentEnd);
      State = LS_VerbatimBlockBody;
      return;
    }
  }

  State = LS_VerbatimBlockFirstLine;
}

void Lexer::lexVerbatimBlockFirstLine(Token &T) {
again:
  assert(BufferPtr < CommentEnd);

  // FIXME: It would be better to scan the text once, finding either the block
  // end command or newline.
  //
  // Extract current line.
  const char *Newline = findNewline(BufferPtr, CommentEnd);
  StringRef Line(BufferPtr, Newline - BufferPtr);

  // Look for end command in current line.
  size_t Pos = Line.find(VerbatimBlockEndCommandName);
  const char *TextEnd;
  const char *NextLine;
  if (Pos == StringRef::npos) {
    // Current line is completely verbatim.
    TextEnd = Newline;
    NextLine = skipNewline(Newline, CommentEnd);
  } else if (Pos == 0) {
    // Current line contains just an end command.
    const char *End = BufferPtr + VerbatimBlockEndCommandName.size();
    StringRef Name(BufferPtr + 1, End - (BufferPtr + 1));
    formTokenWithChars(T, End, tok::verbatim_block_end);
    T.setVerbatimBlockID(Traits.getCommandInfo(Name)->getID());
    State = LS_Normal;
    return;
  } else {
    // There is some text, followed by end command.  Extract text first.
    TextEnd = BufferPtr + Pos;
    NextLine = TextEnd;
    // If there is only whitespace before end command, skip whitespace.
    if (isWhitespace(BufferPtr, TextEnd)) {
      BufferPtr = TextEnd;
      goto again;
    }
  }

  StringRef Text(BufferPtr, TextEnd - BufferPtr);
  formTokenWithChars(T, NextLine, tok::verbatim_block_line);
  T.setVerbatimBlockText(Text);

  State = LS_VerbatimBlockBody;
}

void Lexer::lexVerbatimBlockBody(Token &T) {
  assert(State == LS_VerbatimBlockBody);

  if (CommentState == LCS_InsideCComment)
    skipLineStartingDecorations();

  lexVerbatimBlockFirstLine(T);
}

void Lexer::setupAndLexVerbatimLine(Token &T, const char *TextBegin,
                                    const CommandInfo *Info) {
  assert(Info->IsVerbatimLineCommand);
  formTokenWithChars(T, TextBegin, tok::verbatim_line_name);
  T.setVerbatimLineID(Info->getID());

  State = LS_VerbatimLineText;
}

void Lexer::lexVerbatimLineText(Token &T) {
  assert(State == LS_VerbatimLineText);

  // Extract current line.
  const char *Newline = findNewline(BufferPtr, CommentEnd);
  const StringRef Text(BufferPtr, Newline - BufferPtr);
  formTokenWithChars(T, Newline, tok::verbatim_line_text);
  T.setVerbatimLineText(Text);

  State = LS_Normal;
}

void Lexer::lexHTMLCharacterReference(Token &T) {
  const char *TokenPtr = BufferPtr;
  assert(*TokenPtr == '&');
  TokenPtr++;
  if (TokenPtr == CommentEnd) {
    formTextToken(T, TokenPtr);
    return;
  }
  const char *NamePtr;
  bool isNamed = false;
  bool isDecimal = false;
  char C = *TokenPtr;
  if (isHTMLNamedCharacterReferenceCharacter(C)) {
    NamePtr = TokenPtr;
    TokenPtr = skipNamedCharacterReference(TokenPtr, CommentEnd);
    isNamed = true;
  } else if (C == '#') {
    TokenPtr++;
    if (TokenPtr == CommentEnd) {
      formTextToken(T, TokenPtr);
      return;
    }
    C = *TokenPtr;
    if (isHTMLDecimalCharacterReferenceCharacter(C)) {
      NamePtr = TokenPtr;
      TokenPtr = skipDecimalCharacterReference(TokenPtr, CommentEnd);
      isDecimal = true;
    } else if (C == 'x' || C == 'X') {
      TokenPtr++;
      NamePtr = TokenPtr;
      TokenPtr = skipHexCharacterReference(TokenPtr, CommentEnd);
    } else {
      formTextToken(T, TokenPtr);
      return;
    }
  } else {
    formTextToken(T, TokenPtr);
    return;
  }
  if (NamePtr == TokenPtr || TokenPtr == CommentEnd ||
      *TokenPtr != ';') {
    formTextToken(T, TokenPtr);
    return;
  }
  StringRef Name(NamePtr, TokenPtr - NamePtr);
  TokenPtr++; // Skip semicolon.
  StringRef Resolved;
  if (isNamed) {
    Resolved = resolveHTMLNamedCharacterReference(Name);
    if (Resolved.empty()) {
      Resolved = HTMLDoxygenCharacterReference(Name);
      if (!Resolved.empty()) {
        formTokenWithChars(T, TokenPtr, tok::text);
        T.setText(Resolved);
        return;
      }
    }
  }
  else if (isDecimal)
    Resolved = resolveHTMLDecimalCharacterReference(Name);
  else
    Resolved = resolveHTMLHexCharacterReference(Name);

  if (Resolved.empty()) {
    formTextToken(T, TokenPtr);
    return;
  }
  formTokenWithChars(T, TokenPtr, tok::text);
  T.setText(Resolved);
  return;
}

void Lexer::setupAndLexHTMLStartTag(Token &T) {
  assert(BufferPtr[0] == '<' &&
         isHTMLIdentifierStartingCharacter(BufferPtr[1]));
  const char *TagNameEnd = skipHTMLIdentifier(BufferPtr + 2, CommentEnd);
  StringRef Name(BufferPtr + 1, TagNameEnd - (BufferPtr + 1));
  if (!isHTMLTagName(Name)) {
    formTextToken(T, TagNameEnd);
    return;
  }

  formTokenWithChars(T, TagNameEnd, tok::html_start_tag);
  T.setHTMLTagStartName(Name);

  BufferPtr = skipWhitespace(BufferPtr, CommentEnd);

  const char C = *BufferPtr;
  if (BufferPtr != CommentEnd &&
      (C == '>' || C == '/' || isHTMLIdentifierStartingCharacter(C)))
    State = LS_HTMLStartTag;
}

void Lexer::lexHTMLStartTag(Token &T) {
  assert(State == LS_HTMLStartTag);

  const char *TokenPtr = BufferPtr;
  char C = *TokenPtr;
  if (isHTMLIdentifierCharacter(C)) {
    TokenPtr = skipHTMLIdentifier(TokenPtr, CommentEnd);
    StringRef Ident(BufferPtr, TokenPtr - BufferPtr);
    formTokenWithChars(T, TokenPtr, tok::html_ident);
    T.setHTMLIdent(Ident);
  } else {
    switch (C) {
    case '=':
      TokenPtr++;
      formTokenWithChars(T, TokenPtr, tok::html_equals);
      break;
    case '\"':
    case '\'': {
      const char *OpenQuote = TokenPtr;
      TokenPtr = skipHTMLQuotedString(TokenPtr, CommentEnd);
      const char *ClosingQuote = TokenPtr;
      if (TokenPtr != CommentEnd) // Skip closing quote.
        TokenPtr++;
      formTokenWithChars(T, TokenPtr, tok::html_quoted_string);
      T.setHTMLQuotedString(StringRef(OpenQuote + 1,
                                      ClosingQuote - (OpenQuote + 1)));
      break;
    }
    case '>':
      TokenPtr++;
      formTokenWithChars(T, TokenPtr, tok::html_greater);
      State = LS_Normal;
      return;
    case '/':
      TokenPtr++;
      if (TokenPtr != CommentEnd && *TokenPtr == '>') {
        TokenPtr++;
        formTokenWithChars(T, TokenPtr, tok::html_slash_greater);
      } else
        formTextToken(T, TokenPtr);

      State = LS_Normal;
      return;
    }
  }

  // Now look ahead and return to normal state if we don't see any HTML tokens
  // ahead.
  BufferPtr = skipWhitespace(BufferPtr, CommentEnd);
  if (BufferPtr == CommentEnd) {
    State = LS_Normal;
    return;
  }

  C = *BufferPtr;
  if (!isHTMLIdentifierStartingCharacter(C) &&
      C != '=' && C != '\"' && C != '\'' && C != '>') {
    State = LS_Normal;
    return;
  }
}

void Lexer::setupAndLexHTMLEndTag(Token &T) {
  assert(BufferPtr[0] == '<' && BufferPtr[1] == '/');

  const char *TagNameBegin = skipWhitespace(BufferPtr + 2, CommentEnd);
  const char *TagNameEnd = skipHTMLIdentifier(TagNameBegin, CommentEnd);
  StringRef Name(TagNameBegin, TagNameEnd - TagNameBegin);
  if (!isHTMLTagName(Name)) {
    formTextToken(T, TagNameEnd);
    return;
  }

  const char *End = skipWhitespace(TagNameEnd, CommentEnd);

  formTokenWithChars(T, End, tok::html_end_tag);
  T.setHTMLTagEndName(Name);

  if (BufferPtr != CommentEnd && *BufferPtr == '>')
    State = LS_HTMLEndTag;
}

void Lexer::lexHTMLEndTag(Token &T) {
  assert(BufferPtr != CommentEnd && *BufferPtr == '>');

  formTokenWithChars(T, BufferPtr + 1, tok::html_greater);
  State = LS_Normal;
}

Lexer::Lexer(llvm::BumpPtrAllocator &Allocator, const CommandTraits &Traits,
             SourceLocation FileLoc,
             const char *BufferStart, const char *BufferEnd):
    Allocator(Allocator), Traits(Traits),
    BufferStart(BufferStart), BufferEnd(BufferEnd),
    FileLoc(FileLoc), BufferPtr(BufferStart),
    CommentState(LCS_BeforeComment), State(LS_Normal) {
}

void Lexer::lex(Token &T) {
again:
  switch (CommentState) {
  case LCS_BeforeComment:
    if (BufferPtr == BufferEnd) {
      formTokenWithChars(T, BufferPtr, tok::eof);
      return;
    }

    assert(*BufferPtr == '/');
    BufferPtr++; // Skip first slash.
    switch(*BufferPtr) {
    case '/': { // BCPL comment.
      BufferPtr++; // Skip second slash.

      if (BufferPtr != BufferEnd) {
        // Skip Doxygen magic marker, if it is present.
        // It might be missing because of a typo //< or /*<, or because we
        // merged this non-Doxygen comment into a bunch of Doxygen comments
        // around it: /** ... */ /* ... */ /** ... */
        const char C = *BufferPtr;
        if (C == '/' || C == '!')
          BufferPtr++;
      }

      // Skip less-than symbol that marks trailing comments.
      // Skip it even if the comment is not a Doxygen one, because //< and /*<
      // are frequent typos.
      if (BufferPtr != BufferEnd && *BufferPtr == '<')
        BufferPtr++;

      CommentState = LCS_InsideBCPLComment;
      if (State != LS_VerbatimBlockBody && State != LS_VerbatimBlockFirstLine)
        State = LS_Normal;
      CommentEnd = findBCPLCommentEnd(BufferPtr, BufferEnd);
      goto again;
    }
    case '*': { // C comment.
      BufferPtr++; // Skip star.

      // Skip Doxygen magic marker.
      const char C = *BufferPtr;
      if ((C == '*' && *(BufferPtr + 1) != '/') || C == '!')
        BufferPtr++;

      // Skip less-than symbol that marks trailing comments.
      if (BufferPtr != BufferEnd && *BufferPtr == '<')
        BufferPtr++;

      CommentState = LCS_InsideCComment;
      State = LS_Normal;
      CommentEnd = findCCommentEnd(BufferPtr, BufferEnd);
      goto again;
    }
    default:
      llvm_unreachable("second character of comment should be '/' or '*'");
    }

  case LCS_BetweenComments: {
    // Consecutive comments are extracted only if there is only whitespace
    // between them.  So we can search for the start of the next comment.
    const char *EndWhitespace = BufferPtr;
    while(EndWhitespace != BufferEnd && *EndWhitespace != '/')
      EndWhitespace++;

    // Turn any whitespace between comments (and there is only whitespace
    // between them -- guaranteed by comment extraction) into a newline.  We
    // have two newlines between C comments in total (first one was synthesized
    // after a comment).
    formTokenWithChars(T, EndWhitespace, tok::newline);

    CommentState = LCS_BeforeComment;
    break;
  }

  case LCS_InsideBCPLComment:
  case LCS_InsideCComment:
    if (BufferPtr != CommentEnd) {
      lexCommentText(T);
      break;
    } else {
      // Skip C comment closing sequence.
      if (CommentState == LCS_InsideCComment) {
        assert(BufferPtr[0] == '*' && BufferPtr[1] == '/');
        BufferPtr += 2;
        assert(BufferPtr <= BufferEnd);

        // Synthenize newline just after the C comment, regardless if there is
        // actually a newline.
        formTokenWithChars(T, BufferPtr, tok::newline);

        CommentState = LCS_BetweenComments;
        break;
      } else {
        // Don't synthesized a newline after BCPL comment.
        CommentState = LCS_BetweenComments;
        goto again;
      }
    }
  }
}

StringRef Lexer::getSpelling(const Token &Tok,
                             const SourceManager &SourceMgr,
                             bool *Invalid) const {
  SourceLocation Loc = Tok.getLocation();
  std::pair<FileID, unsigned> LocInfo = SourceMgr.getDecomposedLoc(Loc);

  bool InvalidTemp = false;
  StringRef File = SourceMgr.getBufferData(LocInfo.first, &InvalidTemp);
  if (InvalidTemp) {
    *Invalid = true;
    return StringRef();
  }

  const char *Begin = File.data() + LocInfo.second;
  return StringRef(Begin, Tok.getLength());
}

} // end namespace comments
} // end namespace clang

