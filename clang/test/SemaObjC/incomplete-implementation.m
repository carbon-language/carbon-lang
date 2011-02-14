// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface I
- Meth; // expected-note{{method definition for 'Meth' not found}}
@end

@implementation  I  // expected-warning{{incomplete implementation}}
@end

@implementation I(CAT)
- Meth {return 0;}
@end

#pragma GCC diagnostic ignored "-Wincomplete-implementation"
@interface I2
- Meth;
@end

@implementation  I2
@end

@implementation I2(CAT)
- Meth {return 0;}
@end


