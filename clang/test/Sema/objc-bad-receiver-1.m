// RUN: clang -fsyntax-only -verify %s

@interface I
- (id) retain;
@end

void __raiseExc1() {
 [objc_lookUpClass("NSString") retain]; // expected-error {{ "bad receiver type 'int'" }}
}
