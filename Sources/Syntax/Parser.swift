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
      .statement => [
        ^.WHILE, ^.LEFT_PARENTHESIS, ^.expression, ^.RIGHT_PARENTHESIS,
        ^.statement],
      .statement => [^.BREAK, ^.SEMICOLON],
      .statement => [^.CONTINUE, ^.SEMICOLON],
      .statement => [^.RETURN, ^.expression, ^.SEMICOLON],
      .statement => [^.LEFT_CURLY_BRACE, ^.statement_list, ^.RIGHT_CURLY_BRACE],
      .statement => [
        ^.MATCH, ^.LEFT_PARENTHESIS, ^.expression, ^.RIGHT_PARENTHESIS,
        ^.LEFT_CURLY_BRACE, ^.clause_list, ^.RIGHT_CURLY_BRACE],

      .optional_else => [^.ELSE, ^.statement],
      
      .statement_list => [],
      .statement_list => [^.statement, ^.statement_list],
      
      .return_type => [],
      .return_type => [^.ARROW, ^.expression],
      
      .function_definition => [
        ^.FN, ^.identifier, ^.tuple, ^.return_type,
        ^.LEFT_CURLY_BRACE, ^.statement_list, ^.RIGHT_CURLY_BRACE],
      .function_definition => [
        ^.FN, ^.identifier, ^.tuple, ^.DBLARROW, ^.expression],
      
      .function_declaration => [
        ^.FN, ^.identifier, ^.tuple, ^.return_type, ^.SEMICOLON],
      .variable_declaration => [^.expression, ^.COLON, ^.identifier],
      
      .member => [^.VAR, ^.variable_declaration],
      
      .member_list => [],
      .member_list => [^.member, ^.member_list],
      
      .alternative => [^.identifier, ^.tuple],
      .alternative => [^.identifier],

      .alternative_list => [],
      .alternative_list => [^.alternative],
      .alternative_list => [^.alternative, ^.COMMA, ^.alternative_list],

      .declaration => [^.function_definition],
      .declaration => [^.function_declaration],
      .declaration => [
        ^.STRUCT, ^.identifier,
        ^.LEFT_CURLY_BRACE, ^.member_list, ^.RIGHT_CURLY_BRACE],
      .declaration => [
        ^.CHOICE, ^.identifier,
        ^.LEFT_CURLY_BRACE, ^.alternative_list, ^.RIGHT_CURLY_BRACE],
      .declaration => [
        ^.VAR, ^.variable_declaration, ^.EQUAL, ^.expression, ^.SEMICOLON],
      
      .declaration_list => [],
      .declaration_list => [^.declaration, ^.declaration_list],
    ])
}
