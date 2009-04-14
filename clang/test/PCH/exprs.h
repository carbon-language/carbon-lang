// Header for PCH test exprs.c

// DeclRefExpr
int i = 17;
enum Enum { Enumerator = 18 };
typedef typeof(i) int_decl_ref;
typedef typeof(Enumerator) enum_decl_ref;

// IntegerLiteralExpr
typedef typeof(17) integer_literal;
typedef typeof(17l) long_literal;

// CharacterLiteralExpr
typedef typeof('a') char_literal;
