// Test this without pch.
// RUN: %clang_cc1 -x c++ -std=c++0x -include %S/cxx-for-range.h -fsyntax-only -emit-llvm -o - %s

// Test with pch.
// RUN: %clang_cc1 -x c++ -std=c++0x -emit-pch -o %t %S/cxx-for-range.h
// RUN: %clang_cc1 -x c++ -std=c++0x -include-pch %t -fsyntax-only -emit-llvm -o - %s 

void h() {
  f();

  g<int>();

  char a[3] = { 0, 1, 2 };
  for (auto w : a)
    for (auto x : S())
      for (auto y : T())
        for (auto z : U())
          ;
}
