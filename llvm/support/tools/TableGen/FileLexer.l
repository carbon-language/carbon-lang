/*===-- FileLexer.l - Scanner for TableGen Files ----------------*- C++ -*-===//
//
// This file defines a simple flex scanner for TableGen files.  This is pretty
// straight-forward, except for the magic to handle file inclusion.
//
//===----------------------------------------------------------------------===*/

%option prefix="File"
%option yylineno
%option nostdinit
%option never-interactive
%option batch
%option nodefault
%option 8bit
%option outfile="Lexer.cpp"
%option ecs
%option noreject
%option noyymore

%x comment

%{
#include "Record.h"
#include "Support/CommandLine.h"
typedef std::pair<Record*, std::vector<Init*>*> SubClassRefTy;
#include "FileParser.h"

// Global variable recording the location of the include directory
std::string IncludeDirectory;

// ParseInt - This has to handle the special case of binary numbers 0b0101
static int ParseInt(const char *Str) {
  if (Str[0] == '0' && Str[1] == 'b')
    return strtol(Str+2, 0, 2);
  return strtol(Str, 0, 0); 
}

static int CommentDepth = 0;

struct IncludeRec {
  std::string Filename;
  FILE *File;
  unsigned LineNo;
  YY_BUFFER_STATE Buffer;

  IncludeRec(const std::string &FN, FILE *F)
    : Filename(FN), File(F), LineNo(0){
  }
};

static std::vector<IncludeRec> IncludeStack;


std::ostream &err() {
  if (IncludeStack.empty())
    return std::cerr << "At end of input: ";

  for (unsigned i = 0, e = IncludeStack.size()-1; i != e; ++i)
    std::cerr << "Included from " << IncludeStack[i].Filename << ":"
              << IncludeStack[i].LineNo << ":\n";
  return std::cerr << "Parsing " << IncludeStack.back().Filename << ":"
                   << Filelineno << ": ";
}


int Fileparse();

//
// Function: ParseFile()
//
// Description:
//	This function begins the parsing of the specified tablegen file.
//
// Inputs:
//	Filename - A string containing the name of the file to parse.
//	IncludeDir - A string containing the directory from which include
//	             files can be found.
//
void ParseFile(const std::string &Filename, const std::string & IncludeDir) {
  FILE *F = stdin;
  if (Filename != "-") {
    F = fopen(Filename.c_str(), "r");

    if (F == 0) {
      std::cerr << "Could not open input file '" + Filename + "'!\n";
      abort();
    }
    IncludeStack.push_back(IncludeRec(Filename, F));
  } else {
    IncludeStack.push_back(IncludeRec("<stdin>", stdin));
  }

  //
  // Record the location of the include directory so that the lexer can find
  // it later.
  //
  IncludeDirectory = IncludeDir;
 
  Filein = F;
  Filelineno = 1;
  Fileparse();
  Filein = stdin;
}

// HandleInclude - This function is called when an include directive is
// encountered in the input stream...
static void HandleInclude(const char *Buffer) {
  unsigned Length = yyleng;
  assert(Buffer[Length-1] == '"');
  Buffer += strlen("include ");
  Length -= strlen("include ");
  while (*Buffer != '"') {
    ++Buffer;
    --Length;
  }
  assert(Length >= 2 && "Double quotes not found?");
  std::string Filename(Buffer+1, Buffer+Length-1);
  //std::cerr << "Filename = '" << Filename << "'\n";

  // Save the line number and lex buffer of the includer...
  IncludeStack.back().LineNo = Filelineno;
  IncludeStack.back().Buffer = YY_CURRENT_BUFFER;

  // Open the new input file...
  yyin = fopen(Filename.c_str(), "r");
  if (yyin == 0) {
    //
    // If we couldn't find the file in the current directory, look for it in
    // the include directories.
    //
    // NOTE:
    //  Right now, there is only one directory.  We need to eventually add
    //  support for more.
    //
    Filename = IncludeDirectory + "/" + Filename;
    yyin = fopen(Filename.c_str(), "r");
    if (yyin == 0) {
      err() << "Could not find include file '" << Filename << "'!\n";
      abort();
    }
  }

  // Add the file to our include stack...
  IncludeStack.push_back(IncludeRec(Filename, yyin));
  Filelineno = 1;  // Reset line numbering...
  //yyrestart(yyin);    // Start lexing the new file...

  yy_switch_to_buffer(yy_create_buffer(yyin, YY_BUF_SIZE));
}


// yywrap - This is called when the lexer runs out of input in one of the files.
// Switch back to an includer if an includee has run out of input.
//
extern "C"
int yywrap() {
  if (IncludeStack.back().File != stdin)
    fclose(IncludeStack.back().File);
  IncludeStack.pop_back();
  if (IncludeStack.empty()) return 1;  // Top-level file is done.

  // Otherwise, we need to switch back to a file which included the current one.
  Filelineno = IncludeStack.back().LineNo;  // Restore current line number
  yy_switch_to_buffer(IncludeStack.back().Buffer);
  return 0;
}

%}

Comment      \/\/.*

Identifier   [a-zA-Z_][0-9a-zA-Z_]*
Integer      [-+]?[0-9]+|0x[0-9a-fA-F]+|0b[01]+
CodeFragment \[\{([^}]+|\}[^\]])*\}\]
StringVal    \"[^"]*\"
IncludeStr   include[ \t\n]+\"[^"]*\"

%%

{Comment}      { /* Ignore comments */ }

{IncludeStr}   { HandleInclude(yytext); }
{CodeFragment} { Filelval.StrVal = new std::string(yytext+2, yytext+yyleng-2);
                 return CODEFRAGMENT; }

int            { return INT; }
bit            { return BIT; }
bits           { return BITS; }
string         { return STRING; }
list           { return LIST; }
code           { return CODE; }
dag            { return DAG; }

class          { return CLASS; }
def            { return DEF; }
field          { return FIELD; }
let            { return LET; }
in             { return IN; }

{Identifier}   { Filelval.StrVal = new std::string(yytext, yytext+yyleng);
                 return ID; }
${Identifier}  { Filelval.StrVal = new std::string(yytext+1, yytext+yyleng);
                 return VARNAME; } 

{StringVal}    { Filelval.StrVal = new std::string(yytext+1, yytext+yyleng-1);
                 return STRVAL; }

{Integer}      { Filelval.IntVal = ParseInt(Filetext); return INTVAL; }

[ \t\n]+       { /* Ignore whitespace */ }


"/*"                    { BEGIN(comment); CommentDepth++; }
<comment>[^*/]*         /* eat anything that's not a '*' or '/' */
<comment>"*"+[^*/]*     /* eat up '*'s not followed by '/'s */
<comment>"/*"           { ++CommentDepth; }
<comment>"/"+[^*]*      /* eat up /'s not followed by *'s */
<comment>"*"+"/"        { if (!--CommentDepth) { BEGIN(INITIAL); } }
<comment><<EOF>>        { err() << "Unterminated comment!\n"; abort(); }

.              { return Filetext[0]; }

%%
