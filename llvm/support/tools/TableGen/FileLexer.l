/*===-- FileLexer.l - Scanner for TableGen Files ----------------*- C++ -*-===//
//
//
//===------------------------------------------------------------------------=*/

%option prefix="File"
%option yylineno
%option nostdinit
%option never-interactive
%option batch
%option noyywrap
%option nodefault
%option 8bit
%option outfile="Lexer.cpp"
%option ecs
%option noreject
%option noyymore

%x comment

%{
#include "Record.h"
typedef std::pair<Record*, std::vector<Init*>*> SubClassRefTy;
#include "FileParser.h"

// ParseInt - This has to handle the special case of binary numbers 0b0101
static int ParseInt(const char *Str) {
  if (Str[0] == '0' && Str[1] == 'b')
    return strtol(Str+2, 0, 2);
  return strtol(Str, 0, 0); 
}

static int CommentDepth = 0;

%}

Comment     \/\/.*

Identifier  [a-zA-Z_][0-9a-zA-Z_]*
Integer     [-+]?[0-9]+|0x[0-9a-fA-F]+|0b[01]+
StringVal   \"[^"]*\"

%%

{Comment}      { /* Ignore comments */ }

int            { return INT; }
bit            { return BIT; }
bits           { return BITS; }
string         { return STRING; }
list           { return LIST; }

class          { return CLASS; }
def            { return DEF; }
field          { return FIELD; }
set            { return SET; }
in             { return IN; }

{Identifier}   { Filelval.StrVal = new std::string(yytext, yytext+yyleng);
                 return ID; }

{StringVal}    { Filelval.StrVal = new std::string(yytext+1, yytext+yyleng-1);
                 return STRVAL; }

{Integer}      { Filelval.IntVal = ParseInt(Filetext); return INTVAL; }

[ \t\n]+       { /* Ignore whitespace */ }
.              { return Filetext[0]; }


"/*"                    { BEGIN(comment); CommentDepth++; }
<comment>[^*/]*         /* eat anything that's not a '*' or '/' */
<comment>"*"+[^*/]*     /* eat up '*'s not followed by '/'s */
<comment>"/*"           { ++CommentDepth; }
<comment>"/"+[^*]*      /* eat up /'s not followed by *'s */
<comment>"*"+"/"        { if (!--CommentDepth) { BEGIN(INITIAL); } }
<comment><<EOF>>        { fprintf(stderr, "Unterminated comment!\n"); abort(); }

%%
