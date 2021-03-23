import SwiParse
import SwiLex

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

prefix operator ^

extension Terminal {
  static prefix func ^(x:Self) -> Word<Terminal, NonTerminal> { Word(x) }
}

extension NonTerminal {
  static prefix func ^(x:Self) -> Word<Terminal, NonTerminal> { Word(x) }
}

//
func makeParser() {
  print("RUNNING")
  let parser = try! SwiParse<Terminal, NonTerminal>(
    rules: [
      .start => [^.declaration_list],
      .pattern => [^.expression],
      .expression => [^.identifier],
      .expression => [^.designator],
      .expression
        => [^.expression,
            ^.LEFT_SQUARE_BRACKET, ^.expression,
            ^.RIGHT_SQUARE_BRACKET],
      .expression => [^.expression, ^.COLON, ^.identifier],
      .expression => [^.integer_literal],
      .expression => [^.TRUE],
      .expression => [^.FALSE],
      .expression => [^.INT],
      .expression => [^.BOOL],
      .expression => [^.TYPE],
      .expression => [^.AUTO],
      .expression => [^.paren_expression],
      .expression => [^.expression, ^.EQUAL_EQUAL, ^.expression],
      .expression => [^.expression, ^.PLUS, ^.expression],
      .expression => [^.expression, ^.MINUS, ^.expression],
      .expression => [^.expression, ^.AND, ^.expression],
      .expression => [^.expression, ^.OR, ^.expression],
      .expression => [^.NOT, ^.expression],
      .expression => [^.MINUS, ^.expression],
      .expression => [^.expression, ^.tuple],
      .expression => [^.FNTY, ^.tuple, ^.return_type],
      .designator => [^.PERIOD, ^.identifier],
      .paren_expression => [
        ^.LEFT_PARENTHESIS, ^.field_list, ^.RIGHT_PARENTHESIS],
      .tuple => [
        ^.LEFT_PARENTHESIS, ^.field_list, ^.RIGHT_PARENTHESIS],
      .field => [^.pattern],
      .field => [^.designator, ^.EQUAL, ^.pattern],
      .field_list => [],
      .field_list => [^.field],
      .field_list => [^.field, ^.COMMA, ^.field_list],
      .clause => [^.CASE, ^.pattern, ^.DBLARROW, ^.statement],
      .clause => [^.DEFAULT, ^.DBLARROW, ^.statement],
      .clause_list => [],
      .clause_list => [^.clause, ^.clause_list],
      .statement => [^.expression, ^.EQUAL, ^.expression, ^.SEMICOLON],
      .statement => [^.VAR, ^.pattern, ^.EQUAL, ^.expression, ^.SEMICOLON],
      .statement => [^.expression],
      .statement => [^.IF, ^.expression, ^.statement, ^.optional_else],
      .statement => [^.WHILE, ^.LEFT_PARENTHESIS, ^.expression, ^.RIGHT_PARENTHESIS, ^.statement],
      .statement => [^.BREAK, ^.SEMICOLON],
      .statement => [^.CONTINUE, ^.SEMICOLON],
      .statement => [^.RETURN, ^.expression, ^.SEMICOLON],
      .statement => [^.LEFT_CURLY_BRACE, ^.statement_list, ^.RIGHT_CURLY_BRACE],
      .statement => [^.MATCH, ^.LEFT_PARENTHESIS, ^.expression, ^.RIGHT_PARENTHESIS, ^.LEFT_CURLY_BRACE, ^.clause_list, ^.RIGHT_CURLY_BRACE],
      .optional_else => [^.ELSE, ^.statement],
      .statement_list => [],
      .statement_list => [^.statement, ^.statement_list],
      .return_type => [],
      .return_type => [^.ARROW, ^.expression],
      .function_definition => [^.FN, ^.identifier, ^.tuple, ^.return_type, ^.LEFT_CURLY_BRACE, ^.statement_list, ^.RIGHT_CURLY_BRACE],
      .function_definition => [^.FN, ^.identifier, ^.tuple, ^.DBLARROW, ^.expression],
      .function_declaration => [^.FN, ^.identifier, ^.tuple, ^.return_type, ^.SEMICOLON],
      .variable_declaration => [^.expression, ^.COLON, ^.identifier],
      .member => [^.VAR, ^.variable_declaration],
      .member_list => [],
      .member_list => [^.member, ^.member_list],
      .alternative => [^.identifier, ^.tuple],
    ])
}
/*
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
  
