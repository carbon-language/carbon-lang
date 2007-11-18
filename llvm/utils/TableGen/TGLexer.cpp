//===- TGLexer.cpp - Lexer for TableGen -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implement the Lexer for TableGen.
//
//===----------------------------------------------------------------------===//

#include "TGLexer.h"
#include "Record.h"
#include "llvm/Support/Streams.h"
#include "Record.h"
#include "llvm/Support/MemoryBuffer.h"
typedef std::pair<llvm::Record*, std::vector<llvm::Init*>*> SubClassRefTy;
#include "FileParser.h"
#include <cctype>
using namespace llvm;

// FIXME: REMOVE THIS.
#define YYEOF 0
#define YYERROR -2

TGLexer::TGLexer(MemoryBuffer *StartBuf) : CurLineNo(1), CurBuf(StartBuf) {
  CurPtr = CurBuf->getBufferStart();
}

TGLexer::~TGLexer() {
  while (!IncludeStack.empty()) {
    delete IncludeStack.back().Buffer;
    IncludeStack.pop_back();
  }
  delete CurBuf;
}


std::ostream &TGLexer::err() {
  PrintIncludeStack(*cerr.stream());
  return *cerr.stream();
}


void TGLexer::PrintIncludeStack(std::ostream &OS) {
  for (unsigned i = 0, e = IncludeStack.size(); i != e; ++i)
    OS << "Included from " << IncludeStack[i].Buffer->getBufferIdentifier()
       << ":" << IncludeStack[i].LineNo << ":\n";
  OS << "Parsing " << CurBuf->getBufferIdentifier() << ":"
     << CurLineNo << ": ";
}

int TGLexer::getNextChar() {
  char CurChar = *CurPtr++;
  switch (CurChar) {
  default:
    return (unsigned char)CurChar;
  case 0:
    // A nul character in the stream is either the end of the current buffer or
    // a random nul in the file.  Disambiguate that here.
    if (CurPtr-1 != CurBuf->getBufferEnd())
      return 0;  // Just whitespace.
    
    // If this is the end of an included file, pop the parent file off the
    // include stack.
    if (!IncludeStack.empty()) {
      delete CurBuf;
      CurBuf = IncludeStack.back().Buffer;
      CurLineNo = IncludeStack.back().LineNo;
      CurPtr = IncludeStack.back().CurPtr;
      IncludeStack.pop_back();
      return getNextChar();
    }
    
    // Otherwise, return end of file.
    --CurPtr;  // Another call to lex will return EOF again.  
    return EOF;
  case '\n':
  case '\r':
    // Handle the newline character by ignoring it and incrementing the line
    // count.  However, be careful about 'dos style' files with \n\r in them.
    // Only treat a \n\r or \r\n as a single line.
    if ((*CurPtr == '\n' || (*CurPtr == '\r')) &&
        *CurPtr != CurChar)
      ++CurPtr;  // Eat the two char newline sequence.
      
    ++CurLineNo;
    return '\n';
  }  
}

int TGLexer::LexToken() {
  // This always consumes at least one character.
  int CurChar = getNextChar();

  switch (CurChar) {
  default:
    // Handle letters: [a-zA-Z_]
    if (isalpha(CurChar) || CurChar == '_')
      return LexIdentifier();
      
    // Unknown character, return the char itself.
    return (unsigned char)CurChar;
  case EOF: return YYEOF;
  case 0:
  case ' ':
  case '\t':
  case '\n':
  case '\r':
    // Ignore whitespace.
    return LexToken();
  case '/':
    // If this is the start of a // comment, skip until the end of the line or
    // the end of the buffer.
    if (*CurPtr == '/')
      SkipBCPLComment();
    else if (*CurPtr == '*') {
      if (SkipCComment())
        return YYERROR;
    } else // Otherwise, return this / as a token.
      return CurChar;
    return LexToken();
  case '-': case '+':
  case '0': case '1': case '2': case '3': case '4': case '5': case '6':
  case '7': case '8': case '9':  
    return LexNumber();
  case '"': return LexString();
  case '$': return LexVarName();
  case '[': return LexBracket();
  case '!': return LexExclaim();
  }
}

/// LexString - Lex "[^"]*"
int TGLexer::LexString() {
  const char *StrStart = CurPtr;
  
  while (*CurPtr != '"') {
    // If we hit the end of the buffer, report an error.
    if (*CurPtr == 0 && CurPtr == CurBuf->getBufferEnd()) {
      TheError = "End of file in string literal";
      return YYERROR;
    } else if (*CurPtr == '\n' || *CurPtr == '\r') {
      TheError = "End of line in string literal";
      return YYERROR;
    }
    
    ++CurPtr;
  }
  
  Filelval.StrVal = new std::string(StrStart, CurPtr);
  ++CurPtr;
  return STRVAL;
}

int TGLexer::LexVarName() {
  if (!isalpha(CurPtr[0]) && CurPtr[0] != '_')
    return '$'; // Invalid varname.
  
  // Otherwise, we're ok, consume the rest of the characters.
  const char *VarNameStart = CurPtr++;
  
  while (isalpha(*CurPtr) || isdigit(*CurPtr) || *CurPtr == '_')
    ++CurPtr;

  Filelval.StrVal = new std::string(VarNameStart, CurPtr);
  return VARNAME;
}


int TGLexer::LexIdentifier() {
  // The first letter is [a-zA-Z_].
  const char *IdentStart = CurPtr-1;
  
  // Match the rest of the identifier regex: [0-9a-zA-Z_]*
  while (isalpha(*CurPtr) || isdigit(*CurPtr) || *CurPtr == '_')
    ++CurPtr;
  
  // Check to see if this identifier is a keyword.
  unsigned Len = CurPtr-IdentStart;
  
  if (Len == 3 && !memcmp(IdentStart, "int", 3)) return INT;
  if (Len == 3 && !memcmp(IdentStart, "bit", 3)) return BIT;
  if (Len == 4 && !memcmp(IdentStart, "bits", 4)) return BITS;
  if (Len == 6 && !memcmp(IdentStart, "string", 6)) return STRING;
  if (Len == 4 && !memcmp(IdentStart, "list", 4)) return LIST;
  if (Len == 4 && !memcmp(IdentStart, "code", 4)) return CODE;
  if (Len == 3 && !memcmp(IdentStart, "dag", 3)) return DAG;
  
  if (Len == 5 && !memcmp(IdentStart, "class", 5)) return CLASS;
  if (Len == 3 && !memcmp(IdentStart, "def", 3)) return DEF;
  if (Len == 4 && !memcmp(IdentStart, "defm", 4)) return DEFM;
  if (Len == 10 && !memcmp(IdentStart, "multiclass", 10)) return MULTICLASS;
  if (Len == 5 && !memcmp(IdentStart, "field", 5)) return FIELD;
  if (Len == 3 && !memcmp(IdentStart, "let", 3)) return LET;
  if (Len == 2 && !memcmp(IdentStart, "in", 2)) return IN;
  
  if (Len == 7 && !memcmp(IdentStart, "include", 7)) {
    if (LexInclude()) return YYERROR;
    return LexToken();
  }
    
  Filelval.StrVal = new std::string(IdentStart, CurPtr);
  return ID;
}

/// LexInclude - We just read the "include" token.  Get the string token that
/// comes next and enter the include.
bool TGLexer::LexInclude() {
  // The token after the include must be a string.
  int Tok = LexToken();
  if (Tok == YYERROR) return true;
  if (Tok != STRVAL) {
    TheError = "Expected filename after include";
    return true;
  }

  // Get the string.
  std::string Filename = *Filelval.StrVal;
  delete Filelval.StrVal;

  // Try to find the file.
  MemoryBuffer *NewBuf = MemoryBuffer::getFile(&Filename[0], Filename.size());

  // If the file didn't exist directly, see if it's in an include path.
  for (unsigned i = 0, e = IncludeDirectories.size(); i != e && !NewBuf; ++i) {
    std::string IncFile = IncludeDirectories[i] + "/" + Filename;
    NewBuf = MemoryBuffer::getFile(&IncFile[0], IncFile.size());
  }
    
  if (NewBuf == 0) {
    TheError = "Could not find include file '" + Filename + "'";
    return true;
  }
  
  // Save the line number and lex buffer of the includer.
  IncludeStack.push_back(IncludeRec(CurBuf, CurPtr, CurLineNo));
  
  CurLineNo = 1;  // Reset line numbering.
  CurBuf = NewBuf;
  CurPtr = CurBuf->getBufferStart();
  return false;
}

void TGLexer::SkipBCPLComment() {
  ++CurPtr;  // skip the second slash.
  while (1) {
    switch (*CurPtr) {
    case '\n':
    case '\r':
      return;  // Newline is end of comment.
    case 0:
      // If this is the end of the buffer, end the comment.
      if (CurPtr == CurBuf->getBufferEnd())
        return;
      break;
    }
    // Otherwise, skip the character.
    ++CurPtr;
  }
}

/// SkipCComment - This skips C-style /**/ comments.  The only difference from C
/// is that we allow nesting.
bool TGLexer::SkipCComment() {
  ++CurPtr;  // skip the star.
  unsigned CommentDepth = 1;
  
  while (1) {
    int CurChar = getNextChar();
    switch (CurChar) {
    case EOF:
      TheError = "Unterminated comment!";
      return true;
    case '*':
      // End of the comment?
      if (CurPtr[0] != '/') break;
      
      ++CurPtr;   // End the */.
      if (--CommentDepth == 0)
        return false;
      break;
    case '/':
      // Start of a nested comment?
      if (CurPtr[0] != '*') break;
      ++CurPtr;
      ++CommentDepth;
      break;
    }
  }
}

/// LexNumber - Lex:
///    [-+]?[0-9]+
///    0x[0-9a-fA-F]+
///    0b[01]+
int TGLexer::LexNumber() {
  const char *NumStart = CurPtr-1;
  
  if (CurPtr[-1] == '0') {
    if (CurPtr[0] == 'x') {
      ++CurPtr;
      NumStart = CurPtr;
      while (isxdigit(CurPtr[0]))
        ++CurPtr;
      
      if (CurPtr == NumStart) {
        TheError = "Invalid hexadecimal number";
        return YYERROR;
      }
      Filelval.IntVal = strtoll(NumStart, 0, 16);
      return INTVAL;
    } else if (CurPtr[0] == 'b') {
      ++CurPtr;
      NumStart = CurPtr;
      while (CurPtr[0] == '0' || CurPtr[0] == '1')
        ++CurPtr;
      
      if (CurPtr == NumStart) {
        TheError = "Invalid binary number";
        return YYERROR;
      }
      Filelval.IntVal = strtoll(NumStart, 0, 2);
      return INTVAL;
    }
  }

  // Check for a sign without a digit.
  if (CurPtr[-1] == '-' || CurPtr[-1] == '+') {
    if (!isdigit(CurPtr[0]))
      return CurPtr[-1];
  }
  
  while (isdigit(CurPtr[0]))
    ++CurPtr;
  Filelval.IntVal = strtoll(NumStart, 0, 10);
  return INTVAL;
}

/// LexBracket - We just read '['.  If this is a code block, return it,
/// otherwise return the bracket.  Match: '[' and '[{ ( [^}]+ | }[^]] )* }]'
int TGLexer::LexBracket() {
  if (CurPtr[0] != '{')
    return '[';
  ++CurPtr;
  const char *CodeStart = CurPtr;
  while (1) {
    int Char = getNextChar();
    if (Char == EOF) break;
    
    if (Char != '}') continue;
    
    Char = getNextChar();
    if (Char == EOF) break;
    if (Char == ']') {
      Filelval.StrVal = new std::string(CodeStart, CurPtr-2);
      return CODEFRAGMENT;
    }
  }
  
  TheError = "Invalid Code Block";
  return YYERROR;
}

/// LexExclaim - Lex '!' and '![a-zA-Z]+'.
int TGLexer::LexExclaim() {
  if (!isalpha(*CurPtr))
    return '!';
  
  const char *Start = CurPtr++;
  while (isalpha(*CurPtr))
    ++CurPtr;
  
  // Check to see which operator this is.
  unsigned Len = CurPtr-Start;
  
  if (Len == 3 && !memcmp(Start, "con", 3)) return CONCATTOK;
  if (Len == 3 && !memcmp(Start, "sra", 3)) return SRATOK;
  if (Len == 3 && !memcmp(Start, "srl", 3)) return SRLTOK;
  if (Len == 3 && !memcmp(Start, "shl", 3)) return SHLTOK;
  if (Len == 9 && !memcmp(Start, "strconcat", 9)) return STRCONCATTOK;
  
  TheError = "Unknown operator";
  return YYERROR;
}

//===----------------------------------------------------------------------===//
//  Interfaces used by the Bison parser.
//===----------------------------------------------------------------------===//

int Fileparse();
static TGLexer *TheLexer;

namespace llvm {
  
std::ostream &err() {
  return TheLexer->err();
}

/// ParseFile - this function begins the parsing of the specified tablegen
/// file.
///
void ParseFile(const std::string &Filename, 
               const std::vector<std::string> &IncludeDirs) {
  std::string ErrorStr;
  MemoryBuffer *F = MemoryBuffer::getFileOrSTDIN(&Filename[0], Filename.size(),
                                                 &ErrorStr);
  if (F == 0) {
    cerr << "Could not open input file '" + Filename + "': " << ErrorStr <<"\n";
    exit(1);
  }
  
  assert(!TheLexer && "Lexer isn't reentrant yet!");
  TheLexer = new TGLexer(F);
  
  // Record the location of the include directory so that the lexer can find
  // it later.
  TheLexer->setIncludeDirs(IncludeDirs);
  
  Fileparse();
  
  // Cleanup
  delete TheLexer;
  TheLexer = 0;
}
} // End llvm namespace


int Filelex() {
  assert(TheLexer && "No lexer setup yet!");
  int Tok = TheLexer->LexToken();
  if (Tok == YYERROR) {
    err() << TheLexer->getError() << "\n";
    exit(1);
  }
  return Tok;
}
