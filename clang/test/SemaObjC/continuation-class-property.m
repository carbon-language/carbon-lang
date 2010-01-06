// RUN: %clang_cc1  -fsyntax-only -verify %s
// radar 7509234

@protocol Foo
@property (readonly, copy) id foos;
@end

@interface Bar <Foo> {
}

@end

@interface Baz  <Foo> {
}
@end

@interface Bar ()
@property (readwrite, copy) id foos;
@end

@interface Baz ()
@property (readwrite, copy) id foos;
@end

