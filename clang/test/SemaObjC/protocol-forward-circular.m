// RUN: clang -fsyntax-only -verify %s

@protocol B;
@protocol C < B > // expected-warning{{cannot find protocol definition for 'B'}} // expected-note{{previous definition is here}}
@end
@protocol A < C > 
@end
@protocol B < A > // expected-error{{protocol has circular dependency}}
@end

