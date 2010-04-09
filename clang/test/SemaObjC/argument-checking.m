// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s

struct S { int a; };

extern int charStarFunc(char *);
extern int charFunc(char);

@interface Test
+alloc;
-(int)charStarMeth:(char *)s;
-structMeth:(struct S)s;
-structMeth:(struct S)s :(struct S)s2;
@end

void test() {
  id obj = [Test alloc];
  struct S sInst;

  charStarFunc(1); // expected-warning {{incompatible integer to pointer conversion passing 'int' to parameter of type 'char *'}}
  charFunc("abc"); // expected-warning {{incompatible pointer to integer conversion passing 'char [4]' to parameter of type 'char'}}

  [obj charStarMeth:1]; // expected-warning {{incompatible integer to pointer conversion sending 'int'}}
  [obj structMeth:1]; // expected-error {{sending 'int'}}
  [obj structMeth:sInst :1]; // expected-error {{sending 'int'}}
}
