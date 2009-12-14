// RUN: clang -cc1 -fsyntax-only -verify %s

@interface NSObject @end

@interface NSObject (NSScriptClassDescription)
@end

void f() {
  NSScriptClassDescription *f; // expected-error {{use of undeclared identifier 'NSScriptClassDescription'}}
}
