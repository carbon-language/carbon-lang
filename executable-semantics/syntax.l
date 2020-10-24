%{
#include <stdlib.h>
#include "ast.h"
#include "syntax.tab.h"
%}
%option yylineno

integer_literal   [0-9]+
identifier    [A-Za-z_][A-Za-z0-9_]*
AND   "and"
OR    "or"
NOT   "not"
EQUAL "=="
ARROW "->"
COMMENT \/\/[^\n]*\n
RETURN "return"
IF     "if"
ELSE   "else"
WHILE   "while"
BREAK   "break"
CONTINUE "continue"
INT  "Int"
BOOL "Bool"
TYPE "Type"
FN    "fn"
FNTY   "fnty"
TUPLETY "Tuple"
TRUE   "true"
FALSE  "false"
VAR    "var"
STRUCT "struct"
CHOICE "choice"
MATCH "match"
CASE "case"
DBLARROW "=>"
DEFAULT "default"
AUTO "auto"
%%
{STRUCT} { return STRUCT; }
{CHOICE} { return CHOICE; }
{MATCH}  { return MATCH; }
{CASE}   { return CASE; }
{DBLARROW} { return DBLARROW; }
{DEFAULT} { return DEFAULT; }
{AUTO} { return AUTO; }
{AND}    { return AND; }
{OR}     { return OR; }
{NOT}    { return NOT; }
{IF}     { return IF; }
{ELSE}   { return ELSE; }
{WHILE}  { return WHILE; }
{BREAK}  { return BREAK; }
{CONTINUE} { return CONTINUE; }
{INT}  { return INT; }
{BOOL} { return BOOL; }
{TYPE} { return TYPE; }
{FN}    { return FN; }
{FNTY}    { return FNTY; }
{ARROW}  { return ARROW; }
{VAR}    { return VAR; }
{EQUAL}  { return EQUAL; }
{RETURN} { return RETURN; }
{TRUE}   { return TRUE; }
{FALSE}  { return FALSE; }
{identifier}     {
           int n = strlen(yytext);
           yylval.str = (char*)malloc((n + 1) * sizeof(char));
           strncpy(yylval.str, yytext, n + 1);
           return identifier;
         }
{integer_literal}    {yylval.num = atof(yytext);  return integer_literal;}
[ \t\n]+   ;
{COMMENT} ;
.     {return yytext[0];}
%%
int yywrap()  {return 1;}

