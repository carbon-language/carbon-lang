// Header for PCH test exprs.c

// DeclRefExpr
int i = 17;
enum Enum { Enumerator = 18 };
typedef typeof(i) int_decl_ref;
typedef typeof(Enumerator) enum_decl_ref;

// IntegerLiteral
typedef typeof(17) integer_literal;
typedef typeof(17l) long_literal;

// FloatingLiteral and ParenExpr
typedef typeof((42.5)) floating_literal;

// StringLiteral
const char *hello = "Hello" "PCH" "World";

// CharacterLiteral
typedef typeof('a') char_literal;

// UnaryOperator
typedef typeof(-Enumerator) negate_enum;

// SizeOfAlignOfExpr
typedef typeof(sizeof(int)) typeof_sizeof;
typedef typeof(sizeof(Enumerator)) typeof_sizeof2;

// BinaryOperator
typedef typeof(i + Enumerator) add_result;

// CStyleCastExpr
typedef typeof((void *)0) void_ptr;

