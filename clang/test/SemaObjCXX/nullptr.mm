// RUN: %clang_cc1 -std=c++0x -fblocks -fsyntax-only -verify %s

@interface A
@end

void comparisons(A *a) {
  (void)(a == nullptr);
  (void)(nullptr == a);
}

void assignment(A *a) {
  a = nullptr;
}

int PR10145a = (void(^)())0 == nullptr;
int PR10145b = nullptr == (void(^)())0;
