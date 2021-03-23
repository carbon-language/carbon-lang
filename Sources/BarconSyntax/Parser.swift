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

prefix operator ^
typealias RHS = [Word<Terminal, NonTerminal>]
protocol RHSElement {
  static prefix func ^(_: Self) -> RHS
}

extension RHSElement {
  static func ^(words: RHS, x: Self) -> RHS {
    words + ^x
  }
}

extension Terminal : RHSElement {
  static prefix func ^(x: Self) -> RHS { [Word(x)] }
}

extension NonTerminal : RHSElement {
  static prefix func ^(x: Self) -> RHS { [Word(x)] }
}

//
func makeParser() {
  print("RUNNING")
  typealias Z = Word<Terminal, NonTerminal>
  let parser = try! SwiParse<Terminal, NonTerminal>(
    rules: [
      .start => [Z(.declaration_list)],
      .pattern => [Z(.expression)],
      .expression => [Z(.identifier)],
      .expression => [Z(.designator)],
      .expression
        => [Z(.expression),
            Z(.LEFT_SQUARE_BRACKET), Z(.expression),
            Z(.RIGHT_SQUARE_BRACKET)],
      .expression => [Z(.expression), Z(.COLON), Z(.identifier)],
      .expression => [Z(.integer_literal)],
      .expression => [Z(.TRUE)],
      .expression => [Z(.FALSE)],
      .expression => [Z(.INT)],
      .expression => [Z(.BOOL)],
      .expression => [Z(.TYPE)],
      .expression => [Z(.AUTO)],
      .expression => [Z(.paren_expression)],
      .expression => [Z(.expression), Z(.EQUAL_EQUAL), Z(.expression)],
      .expression => [Z(.expression), Z(.PLUS), Z(.expression)],
      .expression => [Z(.expression), Z(.MINUS), Z(.expression)],
      .expression => [Z(.expression), Z(.AND), Z(.expression)],
      .expression => [Z(.expression), Z(.OR), Z(.expression)],
      .expression => [Z(.NOT), Z(.expression)],
      .expression => [Z(.MINUS), Z(.expression)],
      .expression => [Z(.expression), Z(.tuple)],
      .expression => [Z(.FNTY), Z(.tuple), Z(.return_type)],
      .designator => [Z(.PERIOD), Z(.identifier)],
      .paren_expression => [
        Z(.LEFT_PARENTHESIS), Z(.field_list), Z(.RIGHT_PARENTHESIS)],
      .tuple => [
        Z(.LEFT_PARENTHESIS), Z(.field_list), Z(.RIGHT_PARENTHESIS)],
      .field => [Z(.pattern)],
      .field => [Z(.designator), Z(.EQUAL), Z(.pattern)],
      .field_list => [],
      .field_list => [Z(.field)],
      .field_list => [Z(.field), Z(.COMMA), Z(.field_list)],
      .clause => [Z(.CASE), Z(.pattern), Z(.DBLARROW), Z(.statement)],
      .clause => [Z(.DEFAULT), Z(.DBLARROW), Z(.statement)],
      .clause_list => [],
      .clause_list => [Z(.clause), Z(.clause_list)],
      .statement => [Z(.expression), Z(.EQUAL), Z(.expression), Z(.SEMICOLON)],
      .statement => [Z(.VAR), Z(.pattern), Z(.EQUAL), Z(.expression), Z(.SEMICOLON)],
      .statement => [Z(.expression)],
      .statement => [Z(.IF), Z(.expression), Z(.statement), Z(.optional_else)],
      .statement => [Z(.WHILE), Z(.LEFT_PARENTHESIS), Z(.expression), Z(.RIGHT_PARENTHESIS), Z(.statement)],
      .statement => [Z(.BREAK), Z(.SEMICOLON)],
      .statement => [Z(.CONTINUE), Z(.SEMICOLON)],
      .statement => [Z(.RETURN), Z(.expression), Z(.SEMICOLON)],
      .statement => [Z(.LEFT_CURLY_BRACE), Z(.statement_list), Z(.RIGHT_CURLY_BRACE)],
      .statement => [Z(.MATCH), Z(.LEFT_PARENTHESIS), Z(.expression), Z(.RIGHT_PARENTHESIS), Z(.LEFT_CURLY_BRACE), Z(.clause_list), Z(.RIGHT_CURLY_BRACE)],
      .optional_else => [Z(.ELSE), Z(.statement)],
      .statement_list => [],
      .statement_list => [Z(.statement), Z(.statement_list)],
      .return_type => [],
      .return_type => [Z(.ARROW), Z(.expression)],
      .function_definition => [Z(.FN), Z(.identifier), Z(.tuple), Z(.return_type), Z(.LEFT_CURLY_BRACE), Z(.statement_list), Z(.RIGHT_CURLY_BRACE)],
      .function_definition => [Z(.FN), Z(.identifier), Z(.tuple), Z(.DBLARROW), Z(.expression)],
      .function_declaration => [Z(.FN), Z(.identifier), Z(.tuple), Z(.return_type), Z(.SEMICOLON)],
      .variable_declaration => [Z(.expression), Z(.COLON), Z(.identifier)],
      .member => [Z(.VAR), Z(.variable_declaration)],
      .member_list => [],
      .member_list => [Z(.member), Z(.member_list)],
      .alternative => [Z(.identifier), Z(.tuple)],
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
  
