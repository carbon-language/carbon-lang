enum TokenKind {
  Semicolon,
  Comma,
  PtrMemberAccess,
  MemberAccess,
  Comment,
  // identifiers
  Identifier,
  // basic types tokens
  String,
  Int8,
  Int32,
  Int64,
  Float16,
  Float32,
  Float64,
  Bool,
  Array,
  // variable scope tokens
  As,
  Alias,
  Let,
  Variable,
  Const,
  Dynamic,
  // function tokens
  Function,
  Returned,
  Return,
  // opject related tokens
  Namespace,
  Choice,
  Implementation,
  Extend,
  Interface,
  Base,
  Class,
  Self,
  // Modular tokens
  Package,
  Import,
  Library,
  // conditional and loops tokens
  While,
  For,
  In,
  Break,
  Continue,
  If,
  Else,
  ElseIf,
  Match,
  Case,
  Default,
  // Symbol tokens
  Urinary,
  Binary,
  Logical,
  // punctuation tokens
  OpenParen,
  CloseParen,
  OpenBrace,
  CloseBrace,
  OpenBracket,
  CloseBracket,
}

interface Token {
  type: TokenKind;
  name: string;
}
