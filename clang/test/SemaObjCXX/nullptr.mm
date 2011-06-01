// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

@interface A
@end

void comparisons(A *a) {
  (void)(a == nullptr);
  (void)(nullptr == a);
}

void assignment(A *a) {
  a = nullptr;
}
