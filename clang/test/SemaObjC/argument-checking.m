// RUN: clang-cc -fsyntax-only -verify -pedantic %s

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

  charStarFunc(1); // expected-warning {{incompatible integer to pointer conversion passing 'int', expected 'char *'}}
  charFunc("abc"); // expected-warning {{incompatible pointer to integer conversion passing 'char [4]', expected 'char'}}

  [obj charStarMeth:1]; // expected-warning {{incompatible integer to pointer conversion sending 'int'}}
  [obj structMeth:1]; // expected-error {{incompatible type sending 'int'}}
  [obj structMeth:sInst :1]; // expected-error {{incompatible type sending 'int'}}
}
