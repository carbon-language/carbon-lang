/*
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

%{
#include <stdio.h>

extern int yylex();
extern int yyparse();
extern FILE* yyin;

void yyerror(const char *s) { fprintf(stderr, "%s\n", s); }
int main() { while (yyparse()) {} }
%}

%define api.value.type {int}
%token INT LSH EQEQ

%%

interpreter:
  %empty
| interpreter expression ';' { printf("%d\n", $2); }

expression: compare_expression | compare_operand;

compare_expression: compare_lhs EQEQ compare_operand { $$ = ($1 == $3); };
compare_lhs: compare_expression | compare_operand;
compare_operand: add_expression | multiply_expression | shift_expression | primary_expression;

add_expression: add_lhs '+' add_operand { $$ = ($1 + $3); };
add_lhs: add_expression | add_operand;
add_operand: multiply_expression | multiply_operand;

multiply_expression: multiply_lhs '*' multiply_operand { $$ = ($1 * $3); };
multiply_lhs: multiply_expression | multiply_operand;
multiply_operand: primary_expression;

shift_expression: shift_lhs LSH shift_operand { $$ = ($1 << $3); };
shift_lhs: shift_expression | shift_operand;
shift_operand: primary_expression;

primary_expression: INT | '(' expression ')' { $$ = $2; };

%%
