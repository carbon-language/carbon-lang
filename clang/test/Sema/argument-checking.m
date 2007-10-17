// RUN: clang -fsyntax-only -verify %s

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

  charStarFunc(1); // expected-warning {{incompatible types passing 'int' to function expecting 'char *'}}
  charFunc("abc"); // expected-warning {{incompatible types passing 'char *' to function expecting 'char'}}

  [obj charStarMeth:1]; // expected-warning {{incompatible types passing 'int' to method expecting 'char *'}}
  [obj structMeth:1]; // expected-error {{incompatible types passing 'int' to method expecting 'struct S'}}
  [obj structMeth:sInst :1]; // expected-error {{incompatible types passing 'int' to method expecting 'struct S'}}
}
