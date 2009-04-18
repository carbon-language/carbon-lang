// Test this without pch.
// RUN: clang-cc -include %S/enum.h -fsyntax-only -verify %s &&

// Test with pch.
// RUN: clang-cc -emit-pch -o %t %S/enum.h &&
// RUN: clang-cc -include-pch %t -fsyntax-only -verify %s 

int i = Red;

int return_enum_constant() {
  int result = aRoundShape;
  return result;
}

enum Shape s = Triangle;
