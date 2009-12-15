// RUN: %clang_cc1 -fsyntax-only -Wreadonly-setter-attrs -verify %s

@protocol P0
@property(readonly,assign) id X; // expected-warning {{property attributes 'readonly' and 'assign' are mutually exclusive}}
@end

@protocol P1
@property(readonly,retain) id X; // expected-warning {{property attributes 'readonly' and 'retain' are mutually exclusive}}
@end

@protocol P2
@property(readonly,copy) id X; // expected-warning {{property attributes 'readonly' and 'copy' are mutually exclusive}}
@end

@protocol P3
@property(readonly,readwrite) id X; // expected-error {{property attributes 'readonly' and 'readwrite' are mutually exclusive}}
@end

@protocol P4
@property(assign,copy) id X; // expected-error {{property attributes 'assign' and 'copy' are mutually exclusive}}
@end

@protocol P5
@property(assign,retain) id X; // expected-error {{property attributes 'assign' and 'retain' are mutually exclusive}}
@end

@protocol P6
@property(copy,retain) id X; // expected-error {{property attributes 'copy' and 'retain' are mutually exclusive}}
@end



