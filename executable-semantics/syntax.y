%{
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <stdarg.h>
#include <algorithm>
#include "ast.h"
#include "typecheck.h"
#include "interp.h"

extern FILE* yyin;
extern int yylineno;
char* input_filename;

void yyerror(char*s)  {
  fprintf(stderr, "%s:%d: %s\n", input_filename, yylineno, s);
  exit(-1);
}
// void yyerror(char *s, ...);

extern int yylex();
extern int yywrap();

//#include "typecheck.h"
//#include "eval.h"
#include <list>
#include "ast.h"
using std::list;
using std::pair;
using std::make_pair;
using std::cout;
using std::endl;
  
static list<FunctionDefinition*> program;
%}
%union
 {
   char* str;
   int  num;
   Expression* expression;
   VarTypes* field_types;
   Statement* statement;
   Statement* statement_list;
   struct FunctionDefinition* function_definition;
   Declaration* declaration;
   list<Declaration*>* declaration_list;
   Member* member;
   list<Member*>* member_list;
   ExpOrFieldList* field_list;
   pair<string, Expression*>* alternative;
   list<pair<string, Expression* > >* alternative_list;
   pair<Expression*,Statement*>* clause;
   list< pair<Expression*,Statement*> >* clause_list;
   Expression* fun_type;
};

%token <num> integer_literal
%token <str> identifier
%type <declaration> declaration
%type <function_definition> function_declaration
%type <function_definition> function_definition
%type <declaration_list> declaration_list
%type <statement> statement
%type <statement_list> statement_list
%type <expression> expression
%type <expression> pattern
%type <expression> return_type
%type <expression> tuple
%type <member> member
%type <member_list> member_list
%type <field_list> field
%type <field_list> field_list
%type <alternative> alternative
%type <alternative_list> alternative_list
%type <clause> clause
%type <clause_list> clause_list
%token AND
%token OR
%token NOT
%token INT
%token BOOL
%token TYPE
%token FN
%token FNTY
%token ARROW
%token PTR
%token VAR
%token DIV
%token EQUAL
%token IF
%token ELSE
%token WHILE
%token BREAK
%token CONTINUE
%token DELETE
%token RETURN
%token TRUE
%token FALSE
%token NEW
%token STRUCT
%token CHOICE
%token MATCH
%token CASE
%token DBLARROW
%token DEFAULT
%token AUTO
%nonassoc '{' '}' 
%nonassoc ':' ',' DBLARROW
%left OR AND
%nonassoc EQUAL NOT
%left '+' '-'
%left '.' ARROW 
%nonassoc '(' ')' '[' ']' 
%start input
%locations
%%
input: declaration_list
    {
      printf("********** source program **********\n");
      print_list($1, print_decl, "");
      printf("********** type checking **********\n");
      state = new State(); // compile-time state
      pair<TypeEnv*,Env*> p = top_level($1);
      TypeEnv* top = p.first;
      Env* ct_top = p.second;
      list<Declaration*> new_decls;
      for (auto i = $1->begin(); i != $1->end(); ++i) {
        new_decls.push_back(typecheck_decl(*i, top, ct_top));
      }
      printf("\n");
      printf("********** type checking complete **********\n");
      print_list(&new_decls, print_decl, "");
      printf("********** starting execution **********\n");
      int result = interp_program(&new_decls);
      cout << "result: " << result << endl;
    }
;
pattern:
  expression
    { $$ = $1 }
;
expression:
  identifier
    { $$ = make_var(yylineno, $1); }
| expression '.' identifier
    { $$ = make_get_field(yylineno, $1, $3); }
| expression '[' expression ']'
    { $$ = make_index(yylineno, $1, $3); }
| expression ':' identifier
    { $$ = make_var_pat(yylineno, $3, $1); }
| integer_literal
    { $$ = make_int(yylineno, $1); }
| TRUE
    { $$ = make_bool(yylineno, true); }
| FALSE
    { $$ = make_bool(yylineno, false); }
| INT
    { $$ = make_int_type(yylineno); }
| BOOL
    { $$ = make_bool_type(yylineno); }
| TYPE
    { $$ = make_type_type(yylineno); }
| AUTO
    { $$ = make_auto_type(yylineno); }
| tuple { $$ = $1; }
| expression EQUAL expression
    { $$ = make_binop(yylineno, Eq, $1, $3); }
| expression '+' expression
    { $$ = make_binop(yylineno, Add, $1, $3); }
| expression '-' expression
    { $$ = make_binop(yylineno, Sub, $1, $3); }
| expression AND expression
    { $$ = make_binop(yylineno, And, $1, $3); }
| expression OR expression
    { $$ = make_binop(yylineno, Or, $1, $3); }
| NOT expression
    { $$ = make_unop(yylineno, Not, $2); }
| '-' expression
    { $$ = make_unop(yylineno, Neg, $2); }
| expression tuple { $$ = make_call(yylineno, $1, $2); }
| FNTY tuple return_type
    { $$ = make_fun_type(yylineno, $2, $3); }
;
tuple: '(' field_list ')'
    {
      switch ($2->tag) {
      case Exp:
        $$ = $2->u.exp;
        break;
      case FieldList:
        auto vec = new vector<pair<string,Expression*> >($2->u.fields->begin(),
                                                         $2->u.fields->end());
        $$ = make_tuple(yylineno, vec);
        break;
      }
    }
;
field:
  pattern
    { $$ = make_exp($1); }
| '.' identifier '=' pattern
    { auto fields = new list<pair<string,Expression*> >();
      fields->push_back(make_pair($2, $4));
      $$ = make_field_list(fields); }
;
field_list:
  /* empty */
    { $$ = make_field_list(new list<pair<string,Expression*> >()); }
| field
    { $$ = $1; }
| field ',' field_list
    { $$ = cons_field($1, $3); }
;
clause:
  CASE pattern DBLARROW statement
    { $$ = new pair<Expression*,Statement*>($2, $4); }
| DEFAULT DBLARROW statement 
    { 
      auto vp = make_var_pat(yylineno, "_", make_auto_type(yylineno));
      $$ = new pair<Expression*,Statement*>(vp, $3);
    }
;
clause_list:
  /* empty */
    { $$ = new list< pair<Expression*,Statement*> >(); }
| clause clause_list
    { $$ = $2; $$->push_front(*$1); }
;
statement:
  expression '=' expression ';'
    { $$ = make_assign(yylineno, $1, $3); }
| VAR pattern '=' expression ';'
    { $$ = make_var_def(yylineno, $2, $4); }
| expression ';'
    { $$ = make_exp_stmt(yylineno, $1); }
| IF '(' expression ')' statement ELSE statement
    { $$ = make_if(yylineno, $3, $5, $7); }
| WHILE '(' expression ')' statement
    { $$ = make_while(yylineno, $3, $5); }
| BREAK ';'
    { $$ = make_break(yylineno); }
| CONTINUE ';'
    { $$ = make_continue(yylineno); }
| RETURN expression ';'
    { $$ = make_return(yylineno, $2); }
| '{' statement_list '}'
    { $$ = make_block(yylineno, $2); }
| MATCH '(' expression ')' '{' clause_list '}'
    { $$ = make_match(yylineno, $3, $6); }
;
statement_list:
  /* empty */
    { $$ = 0; }
| statement statement_list
    { $$ = make_seq(yylineno, $1, $2); }
;
return_type:
  /* empty */
    { $$ = make_tuple(yylineno, new vector<pair<string,Expression*> >()); }
| ARROW expression
    { $$ = $2; }
;
function_definition:
  FN identifier tuple return_type '{' statement_list '}'
    { $$ = make_fun_def(yylineno, $2, $4, $3, $6); }
| FN identifier tuple DBLARROW expression ';'
    { $$ = make_fun_def(yylineno, $2, make_auto_type(yylineno), $3,
                        make_return(yylineno, $5)); }
;
function_declaration:
  FN identifier tuple return_type ';'
    { $$ = make_fun_def(yylineno, $2, $4, $3, 0); }
;
member:
  VAR expression ':' identifier ';'
    { $$ = make_field(yylineno, $4, $2); }
;
member_list:
  /* empty */
    { $$ = new list<Member*>(); }
| member member_list
    { $$ = $2; $$->push_front($1); }
;
alternative:
  identifier tuple ';'
    { $$ = new pair<string,Expression*>($1, $2); }
;
alternative_list:
  /* empty */
    { $$ = new list<pair<string, Expression*> >(); }
| alternative alternative_list
    { $$ = $2; $$->push_front(*$1); }
;
declaration:
  function_definition
    { $$ = make_fun_decl($1); }
| function_declaration
    { $$ = make_fun_decl($1); }
| STRUCT identifier '{' member_list '}'
    { $$ = make_struct_decl(yylineno, $2, $4); }
| CHOICE identifier '{' alternative_list '}'
    { $$ = make_choice_decl(yylineno, $2, $4); }
;
declaration_list:
  /* empty */
    { $$ = new list<Declaration*>(); }
| declaration declaration_list
    { $$ = $2; $$->push_front($1); }
;
%%
int main(int argc, char* argv[])  { 
  /*yydebug = 1;*/

  if (argc > 1) {
    input_filename = argv[1];
    yyin = fopen(argv[1], "r");
  }
  if (argc > 2) {
    FILE* program = fopen(argv[2], "r");
    input = read_file(program);
  }
  yyparse(); 
  return 0;
}
