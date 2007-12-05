// RUN: clang -fsyntax-only -verify %s

@interface Subclass 
+ (int)magicNumber;
@end

int main (void) {
  return Subclass.magicNumber; // expected-error {{unexpected interface name 'Subclass': expected expression}}
}

