// RUN: %clang_cc1  -fsyntax-only -verify %s
// rdar:// 6812436

@interface A @end

@interface A () { // expected-error {{ivars may not be placed in class extension}}
  int _p0;
}
@property int p0;
@end

@interface A(CAT) { // expected-error {{ivars may not be placed in categories}}
  int _p1;
}
@end
