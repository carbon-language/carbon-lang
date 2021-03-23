import SwiParse

enum NonTerminal: SwiParsable {
  case
    start,
    pattern,
    expression,
    designator,
    paren_expression,
    tuple,
    field,
    field_list,
    clause,
    clause_list,
    statement,
    optional_else,
    statement_list,
    return_type,
    function_definition,
    function_declaration,
    variable_declaration,
    member,
    member_list,
    alternative,
    alternative_list,
    declaration,
    declaration_list
}

//
func makeParser() {
  let parser = try SwiParse<Terminal, NonTerminal>(
    rules: [
      .start => [Word(.declaration_list)],
      .pattern => [Word(.expression)],
      .expression => [Word(.identifier)],
      .expression => [Word(.designator)],
      .expression
        => [Word(.expression),
            Word(.LEFT_SQUARE_BRACKET)], Word(.expression),
      Word(.RIGHT_SQUARE_BRACKET)])
}
/*  
expression:
  identifier
    { $$ = Carbon::MakeVar(yylineno, $1); }
| expression designator
    { $$ = Carbon::MakeGetField(yylineno, $1, $2); }
| expression "[" expression "]"
    { $$ = Carbon::MakeIndex(yylineno, $1, $3); }
| expression ":" identifier
    { $$ = Carbon::MakeVarPat(yylineno, $3, $1); }
| integer_literal
    { $$ = Carbon::MakeInt(yylineno, $1); }
| TRUE
    { $$ = Carbon::MakeBool(yylineno, true); }
| FALSE
    { $$ = Carbon::MakeBool(yylineno, false); }
| INT
    { $$ = Carbon::MakeIntType(yylineno); }
| BOOL
    { $$ = Carbon::MakeBoolType(yylineno); }
| TYPE
    { $$ = Carbon::MakeTypeType(yylineno); }
| AUTO
    { $$ = Carbon::MakeAutoType(yylineno); }
| paren_expression { $$ = $1; }
| expression EQUAL_EQUAL expression
    { $$ = Carbon::MakeBinOp(yylineno, Carbon::Operator::Eq, $1, $3); }
| expression "+" expression
    { $$ = Carbon::MakeBinOp(yylineno, Carbon::Operator::Add, $1, $3); }
| expression "-" expression
    { $$ = Carbon::MakeBinOp(yylineno, Carbon::Operator::Sub, $1, $3); }
| expression AND expression
    { $$ = Carbon::MakeBinOp(yylineno, Carbon::Operator::And, $1, $3); }
| expression OR expression
    { $$ = Carbon::MakeBinOp(yylineno, Carbon::Operator::Or, $1, $3); }
| NOT expression
    { $$ = Carbon::MakeUnOp(yylineno, Carbon::Operator::Not, $2); }
| "-" expression
    { $$ = Carbon::MakeUnOp(yylineno, Carbon::Operator::Neg, $2); }
| expression tuple
    { $$ = Carbon::MakeCall(yylineno, $1, $2); }
| FNTY tuple return_type
    { $$ = Carbon::MakeFunType(yylineno, $2, $3); }
;
designator: "." identifier { $$ = $2; }
;
paren_expression: "(" field_list ")"
    {
     if ($2->fields->size() == 1 &&
         $2->fields->front().first == "" &&
	 !$2->has_explicit_comma) {
	$$ = $2->fields->front().second;
     } else {
        auto vec = new std::vector<std::pair<std::string,Carbon::Expression*>>(
            $2->fields->begin(), $2->fields->end());
        $$ = Carbon::MakeTuple(yylineno, vec);
      }
    }
;
tuple: "(" field_list ")"
    {
     auto vec = new std::vector<std::pair<std::string,Carbon::Expression*>>(
         $2->fields->begin(), $2->fields->end());
     $$ = Carbon::MakeTuple(yylineno, vec);
    }
field:
  pattern
    {
      auto fields =
          new std::list<std::pair<std::string, Carbon::Expression*>>();
      fields->push_back(std::make_pair("", $1));
      $$ = Carbon::MakeFieldList(fields);
    }
| designator "=" pattern
    {
      auto fields =
          new std::list<std::pair<std::string, Carbon::Expression*>>();
      fields->push_back(std::make_pair($1, $3));
      $$ = Carbon::MakeFieldList(fields);
    }
;
field_list:
  // Empty
    {
      $$ = Carbon::MakeFieldList(
          new std::list<std::pair<std::string, Carbon::Expression*>>());
    }
| field
    { $$ = $1; }
| field "," field_list
    { $$ = Carbon::MakeConsField($1, $3); }
;
clause:
  CASE pattern DBLARROW statement
    { $$ = new std::pair<Carbon::Expression*, Carbon::Statement*>($2, $4); }
| DEFAULT DBLARROW statement
    {
      auto vp = Carbon::MakeVarPat(yylineno, "_",
                                   Carbon::MakeAutoType(yylineno));
      $$ = new std::pair<Carbon::Expression*, Carbon::Statement*>(vp, $3);
    }
;
clause_list:
  // Empty
    {
      $$ = new std::list<std::pair<Carbon::Expression*, Carbon::Statement*>>();
    }
| clause clause_list
    { $$ = $2; $$->push_front(*$1); }
;
statement:
  expression "=" expression ";"
    { $$ = Carbon::MakeAssign(yylineno, $1, $3); }
| VAR pattern "=" expression ";"
    { $$ = Carbon::MakeVarDef(yylineno, $2, $4); }
| expression ";"
    { $$ = Carbon::MakeExpStmt(yylineno, $1); }
| IF "(" expression ")" statement optional_else
    { $$ = Carbon::MakeIf(yylineno, $3, $5, $6); }
| WHILE "(" expression ")" statement
    { $$ = Carbon::MakeWhile(yylineno, $3, $5); }
| BREAK ";"
    { $$ = Carbon::MakeBreak(yylineno); }
| CONTINUE ";"
    { $$ = Carbon::MakeContinue(yylineno); }
| RETURN expression ";"
    { $$ = Carbon::MakeReturn(yylineno, $2); }
| "{" statement_list "}"
    { $$ = Carbon::MakeBlock(yylineno, $2); }
| MATCH "(" expression ")" "{" clause_list "}"
    { $$ = Carbon::MakeMatch(yylineno, $3, $6); }
;
optional_else:
  // Empty
    { $$ = 0; }
| ELSE statement { $$ = $2; }
;
statement_list:
  // Empty
    { $$ = 0; }
| statement statement_list
    { $$ = Carbon::MakeSeq(yylineno, $1, $2); }
;
return_type:
  // Empty
    {
      $$ = Carbon::MakeTuple(
          yylineno,
          new std::vector<std::pair<std::string, Carbon::Expression*>>());
    }
| ARROW expression
    { $$ = $2; }
;
function_definition:
  FN identifier tuple return_type "{" statement_list "}"
    { $$ = MakeFunDef(yylineno, $2, $4, $3, $6); }
| FN identifier tuple DBLARROW expression ";"
    {
      $$ = Carbon::MakeFunDef(yylineno, $2, Carbon::MakeAutoType(yylineno), $3,
                              Carbon::MakeReturn(yylineno, $5));
    }
;
function_declaration:
  FN identifier tuple return_type ";"
    { $$ = MakeFunDef(yylineno, $2, $4, $3, 0); }
;
variable_declaration: expression ":" identifier
    { $$ = MakeField(yylineno, $3, $1); }
;
member: VAR variable_declaration ";"
    { $$ = $2; }
;
member_list:
  // Empty
    { $$ = new std::list<Carbon::Member*>(); }
| member member_list
    { $$ = $2; $$->push_front($1); }
;
alternative:
  identifier tuple
    { $$ = new std::pair<std::string, Carbon::Expression*>($1, $2); }
| identifier
    {
      $$ = new std::pair<std::string, Carbon::Expression*>(
          $1, Carbon::MakeTuple(
            yylineno,
            new std::vector<std::pair<std::string, Carbon::Expression*>>()));
    }
;
alternative_list:
  // Empty
    { $$ = new std::list<std::pair<std::string, Carbon::Expression*>>(); }
| alternative
    {
      $$ = new std::list<std::pair<std::string, Carbon::Expression*>>();
      $$->push_front(*$1);
    }
| alternative "," alternative_list
    { $$ = $3; $$->push_front(*$1); }
;
declaration:
  function_definition
    { $$ = new Carbon::Declaration(Carbon::FunctionDeclaration{$1}); }
| function_declaration
    { $$ = new Carbon::Declaration(Carbon::FunctionDeclaration{$1}); }
| STRUCT identifier "{" member_list "}"
    {
      $$ = new Carbon::Declaration(
        Carbon::StructDeclaration{yylineno, $2, $4});
    }
| CHOICE identifier "{" alternative_list "}"
    {
      $$ = new Carbon::Declaration(
        Carbon::ChoiceDeclaration{yylineno, $2, std::list(*$4)});
    }
| VAR variable_declaration "=" expression ";"
    {
      $$ = new Carbon::Declaration(
        Carbon::VariableDeclaration(yylineno, *$2->u.field.name, $2->u.field.type, $4));
    }
;
declaration_list:
  // Empty
    { $$ = new std::list<Carbon::Declaration>(); }
| declaration declaration_list
    {
      $$ = $2;
      $$->push_front(*$1);
      }*/
  

