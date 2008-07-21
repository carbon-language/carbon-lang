// RUN: clang %s -fsyntax-only -verify
// rdar://5957506

@interface NSWhatever :
NSObject     // expected-error {{cannot find interface declaration for 'NSObject'}}
<NSCopying>  // expected-error {{cannot find protocol definition for 'NSCopying'}}
@end

