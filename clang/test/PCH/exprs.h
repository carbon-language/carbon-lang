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

// ImaginaryLiteral
typedef typeof(17.0i) imaginary_literal;

// StringLiteral
const char *hello = "Hello" "PCH" "World";

// CharacterLiteral
typedef typeof('a') char_literal;

// UnaryOperator
typedef typeof(-Enumerator) negate_enum;

// SizeOfAlignOfExpr
typedef typeof(sizeof(int)) typeof_sizeof;
typedef typeof(sizeof(Enumerator)) typeof_sizeof2;

// ArraySubscriptExpr
extern double values[];
typedef typeof(values[2]) array_subscript;

// CallExpr
double dplus(double x, double y);
double d0, d1;
typedef typeof((&dplus)(d0, d1)) call_returning_double;

// MemberExpr
struct S {
  double x;
};
typedef typeof(((struct S*)0)->x) member_ref_double;

// BinaryOperator
typedef typeof(i + Enumerator) add_result;

// CompoundAssignOperator
typedef typeof(i += Enumerator) addeq_result;

// ConditionalOperator
typedef typeof(i? : d0) conditional_operator;

// CStyleCastExpr
typedef typeof((void *)0) void_ptr;

// ExtVectorElementExpr
typedef __attribute__(( ext_vector_type(2) )) double double2;
double2 vec2;
typedef typeof(vec2.x) ext_vector_element;
