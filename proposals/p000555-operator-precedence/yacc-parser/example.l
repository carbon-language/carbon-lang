/*
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

%option noyywrap
%{
#include <stdio.h>
#define YY_DECL int yylex()
#include "example.tab.h"
%}

%%

[0-9]+ { yylval = atoi(yytext); return INT; }
"*" { return '*'; }
"+" { return '+'; }
"<<" { return LSH; }
"==" { return EQEQ; }
"(" { return '('; }
")" { return ')'; }
";" { return ';'; }

%%
