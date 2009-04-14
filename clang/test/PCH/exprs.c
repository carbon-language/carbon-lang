// Test this without pch.
// RUN: clang-cc -fblocks -include %S/exprs.h -fsyntax-only -verify %s

// Test with pch.
// RUN: clang-cc -emit-pch -fblocks -o %t %S/exprs.h &&
// RUN: clang-cc -fblocks -include-pch %t -fsyntax-only -verify %s 

int integer;
long long_integer;
double floating;

// DeclRefExpr
int_decl_ref *int_ptr1 = &integer;
enum_decl_ref *enum_ptr1 = &integer;

// IntegerLiteral
integer_literal *int_ptr2 = &integer;
long_literal *long_ptr1 = &long_integer;

// FloatingLiteral
floating_literal *double_ptr = &floating;

// CharacterLiteral
char_literal *int_ptr3 = &integer;
