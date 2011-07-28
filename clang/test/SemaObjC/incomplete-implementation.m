// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface I
- Meth; // expected-note{{method definition for 'Meth' not found}} \
        // expected-note{{method declared here}}
@end

@implementation  I  // expected-warning{{incomplete implementation}}
@end

@implementation I(CAT)
- Meth {return 0;} // expected-warning {{category is implementing a method which will also be implemented by its primary class}}
@end

#pragma GCC diagnostic ignored "-Wincomplete-implementation"
@interface I2
- Meth; // expected-note{{method declared here}}
@end

@implementation  I2
@end

@implementation I2(CAT)
- Meth {return 0;} // expected-warning {{category is implementing a method which will also be implemented by its primary class}}
@end


