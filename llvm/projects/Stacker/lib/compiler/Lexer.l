/*===-- Lexer.l - Scanner for Stacker language -----------------*- C++ -*--===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and donated to the LLVM research 
// group and is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//  This file implements the flex scanner for Stacker languages files.
//
//===----------------------------------------------------------------------===*/

%option prefix="Stacker"
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

%{

#include "StackerCompiler.h"
#include "StackerParser.h"

/* Conversion of text ints to binary */
static uint64_t IntToVal(const char *Buffer) {
  uint64_t Result = 0;
  for (; *Buffer; Buffer++) {
    uint64_t OldRes = Result;
    Result *= 10;
    Result += *Buffer-'0';
    if (Result < OldRes)   // Uh, oh, overflow detected!!!
      StackerCompiler::ThrowException("constant bigger than 64 bits detected!");
  }
  return Result;
}

/* Conversion of text hexadecimal ints to binary */
static uint64_t HexIntToVal(const char *Buffer) {
  uint64_t Result = 0;
  for (; *Buffer; ++Buffer) {
    uint64_t OldRes = Result;
    Result *= 16;
    char C = *Buffer;
    if (C >= '0' && C <= '9')
      Result += C-'0';
    else if (C >= 'A' && C <= 'F')
      Result += C-'A'+10;
    else if (C >= 'a' && C <= 'f')
      Result += C-'a'+10;

    if (Result < OldRes)   // Uh, oh, overflow detected!!!
      StackerCompiler::ThrowException("constant bigger than 64 bits detected!");
  }
  return Result;
}

#define YY_NEVER_INTERACTIVE 1
%}

/* Comments start with a ; and go till end of line */
Comment1	[#].*$
/* You can also embed them in ( ... ) */
Comment2	\(.*\)
/* We ignore white space */
White		[ \t\n]

/* jdentifiers start with a % sign */
Identifier  	[A-Za-z][-A-Za-z0-9_]*

/* Strings can contain any character except " and \ */
String		\"[^\"]*\"

/* Positive and negative integer constants*/
PInteger	[+]?[0-9]+
NInteger	-[0-9]+
HexInteger 	0x[0-9A-Fa-f]+

/* Special Characters - name them to avoid flex confusion */
Semi 		[;]
Colon 		[:]
Less		\<
More		\>
LessEq		\<\=
MoreEq		\>\=
NotEq		\<\>
Equal		\=
Plus		\+
Minus		\-
Incr		\+\+
Decr		\-\-
Mult		\*
Div		\/
StarSlash	\*\/
LShift		\<\<
RShift		\>\>
InStr		\<s
InNum		\<d
InChar		\<c
OutStr		\>s
OutNum		\>d
OutChar		\>c

%%

{Comment1}	{ /* Ignore comments */ }
{Comment2}	{ /* Ignore comments */ }

{Colon}		{ return COLON; }
{Semi}		{ return SEMI; }

TRUE		{ return TRUETOK; }
FALSE		{ return FALSETOK; }
ON		{ return TRUETOK; }
OFF		{ return FALSETOK; }
{Less}		{ return LESS; }
LT		{ return LESS; }
{More}		{ return MORE; }
GT		{ return MORE; }
{LessEq}	{ return LESS_EQUAL; }
LE		{ return LESS_EQUAL; }
{MoreEq}	{ return MORE_EQUAL; }
GE		{ return MORE_EQUAL; }
{NotEq}		{ return NOT_EQUAL; }
NE		{ return NOT_EQUAL; }
{Equal}		{ return EQUAL; }
EQ		{ return EQUAL; }

{Plus}		{ return PLUS; }
{Minus}		{ return MINUS; }
{Incr}		{ return INCR; }
{Decr}		{ return DECR; }
{Mult}		{ return MULT; }
{Div}		{ return DIV; }
MOD		{ return MODULUS; }
NEG		{ return NEGATE; }
ABS		{ return ABS; }
MIN		{ return MIN; }
MAX		{ return MAX; }
{StarSlash}	{ return STAR_SLASH; }

AND		{ return AND; }
OR		{ return OR; }
XOR		{ return XOR; }
{LShift}	{ return LSHIFT; }
{RShift}	{ return RSHIFT; }

DROP		{ return DROP; }
NIP		{ return NIP; }
DUP		{ return DUP; }
SWAP		{ return SWAP; }
OVER		{ return OVER; }
PICK		{ return PICK; }
SELECT		{ return SELECT; }
ROT		{ return ROT; }
RROT		{ return RROT; }
ROLL		{ return ROLL; }
TUCK		{ return TUCK; }
DROP2		{ return DROP2; }
NIP2		{ return NIP2; }
DUP2		{ return DUP2; }
SWAP2		{ return SWAP2; }
OVER2		{ return OVER2; }
TUCK2		{ return TUCK2; }
ROT2		{ return ROT2; }
RROT2		{ return RROT2; }

MALLOC		{ return MALLOC; }
FREE		{ return FREE; }
GET		{ return GET; }
PUT		{ return PUT; }

IF		{ return IF; }
ELSE		{ return ELSE; }
ENDIF		{ return ENDIF; }
WHILE		{ return WHILE; }
END		{ return END; }

RECURSE		{ return RECURSE; }
RETURN		{ return RETURN; }
EXIT		{ return EXIT; }
FORWARD		{ return FORWARD; }
TAB		{ return TAB; }
SPACE		{ return SPACE; }
CR		{ return CR; }

{InStr}		{ return IN_STR; }
{InNum}		{ return IN_NUM; }
{InChar} 	{ return IN_CHAR; }

{OutStr}	{ return OUT_STR; }
{OutNum}	{ return OUT_NUM; }
{OutChar} 	{ return OUT_CHAR; }

MAIN		{ return MAIN; }

DUMP		{ return DUMP; }

!=		{ StackerCompiler::ThrowException(
		  "You probably meant to use a <> instead of !=" ); }

==		{ StackerCompiler::ThrowException(
		  "You probably meant to use a single = .. this isn't C"); }

{PInteger}      { Stackerlval.IntegerVal = IntToVal(yytext); return INTEGER; }
{NInteger}      { uint64_t Val = IntToVal(yytext+1);
		  // +1:  we have bigger negative range
		  if (Val > (uint64_t)INT64_MAX+1)
		    StackerCompiler::ThrowException(
		    "Constant too large for signed 64 bits!");
                  Stackerlval.IntegerVal = -Val; 
		  return INTEGER; 
                }
{HexInteger} 	{ Stackerlval.IntegerVal = HexIntToVal(yytext+3); 
                   return INTEGER;
                }

{String} 	{ yytext[strlen(yytext)-1] = 0;           // nuke end quote
		  Stackerlval.StringVal = strdup(yytext+1);  // Nuke start quote
		  return STRING;
                }

{Identifier}    { Stackerlval.StringVal = strdup(yytext); return IDENTIFIER; }

{White}         { /* Ignore whitespace */ }
%%
