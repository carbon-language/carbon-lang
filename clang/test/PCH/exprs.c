// Test this without pch.
// RUN: clang-cc -fblocks -include %S/exprs.h -fsyntax-only -verify %s

// Test with pch.
// RUN: clang-cc -emit-pch -fblocks -o %t %S/exprs.h
// RUN: clang-cc -fblocks -include-pch %t -fsyntax-only -verify %s 

int integer;
long long_integer;
double floating;
_Complex double floating_complex;

// DeclRefExpr
int_decl_ref *int_ptr1 = &integer;
enum_decl_ref *enum_ptr1 = &integer;

// IntegerLiteral
integer_literal *int_ptr2 = &integer;
long_literal *long_ptr1 = &long_integer;

// FloatingLiteral + ParenExpr
floating_literal *double_ptr = &floating;

// ImaginaryLiteral
imaginary_literal *cdouble_ptr = &floating_complex;

// StringLiteral
const char* printHello() {
  return hello;
}

// CharacterLiteral
char_literal *int_ptr3 = &integer;

// UnaryOperator
negate_enum *int_ptr4 = &integer;

// SizeOfAlignOfExpr
typeof(sizeof(float)) size_t_value;
typeof_sizeof *size_t_ptr = &size_t_value;
typeof_sizeof2 *size_t_ptr2 = &size_t_value;

// ArraySubscriptExpr
array_subscript *double_ptr1_5 = &floating;

// CallExpr
call_returning_double *double_ptr2 = &floating;

// MemberExpr
member_ref_double *double_ptr3 = &floating;

// BinaryOperator
add_result *int_ptr5 = &integer;

// CompoundAssignOperator
addeq_result *int_ptr6 = &integer;

// ConditionalOperator
conditional_operator *double_ptr4 = &floating;

// CStyleCastExpr
void_ptr vp1 = &integer;

// CompoundLiteral
struct S s;
compound_literal *sptr = &s;

// ExtVectorElementExpr
ext_vector_element *double_ptr5 = &floating;

// InitListExpr
double get_from_double_array(unsigned Idx) { return double_array[Idx]; }

/// DesignatedInitExpr
float get_from_designated(unsigned Idx) {
  return designated_inits[2].y;
}

// TypesCompatibleExpr
types_compatible *int_ptr7 = &integer;

// ChooseExpr
choose_expr *int_ptr8 = &integer;

// GNUNullExpr FIXME: needs C++
//null_type null = __null;

// ShuffleVectorExpr
shuffle_expr *vec_ptr = &vec2;
