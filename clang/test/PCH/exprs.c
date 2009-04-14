// Test this without pch.
// RUN: clang-cc -fblocks -include %S/exprs.h -fsyntax-only -verify %s

// Test with pch.
// RUN: clang-cc -emit-pch -fblocks -o %t %S/exprs.h &&
// RUN: clang-cc -fblocks -include-pch %t -fsyntax-only -verify %s 

int integer;
long long_integer;

// DeclRefExpr
int_decl_ref *int_ptr1 = &integer;
enum_decl_ref *enum_ptr1 = &integer;
// IntegerLiteralExpr
integer_literal *int_ptr2 = &integer;
long_literal *long_ptr1 = &long_integer;

// CharacterLiteralExpr
char_literal *int_ptr3 = &integer;
