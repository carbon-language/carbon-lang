// RUN: clang %s -fsyntax-only -verify
// rdar://5957506

@interface NSWhatever :
NSObject     // expected-error {{cannot find interface declaration for 'NSObject'}}
<NSCopying>  // expected-error {{cannot find protocol declaration for 'NSCopying'}}
@end


// rdar://6095245
@interface A
{
  int x
}  // expected-error {{expected ';' at end of declaration list}}
@end


// rdar://4304469
@interface INT1
@end

void test2() {
    INT1 b[3];          // expected-warning {{array of interface 'INT1' should probably be an array of pointers}}
    INT1 *c = &b[0];
    ++c;
}

