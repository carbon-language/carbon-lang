// RUN: clang -fsyntax-only -verify %s

class C {
public:
  C(int);
};

@interface D 
- (void)g:(int)a, ...;
@end

void t1(D *d)
{
  C c(10);

  [d g:10, c]; // expected-warning{{cannot pass object of non-POD type 'class C' through variadic method; call will abort at runtime}}
}

