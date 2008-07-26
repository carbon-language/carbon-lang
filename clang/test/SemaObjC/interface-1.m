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


