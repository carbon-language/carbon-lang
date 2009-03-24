// RUN: clang-cc -fsyntax-only -verify %s

@interface XX
int x;  // expected-error {{cannot declare variable inside a class, protocol or category}}
int one=1;  // expected-error {{cannot declare variable inside a class, protocol or category}}
@end

@protocol PPP
int ddd; // expected-error {{cannot declare variable inside a class, protocol or category}}
@end

@interface XX(CAT)
  char * III; // expected-error {{cannot declare variable inside a class, protocol or category}}
  extern int OK;
@end


int main( int argc, const char *argv[] ) {
    return x+one;
}

