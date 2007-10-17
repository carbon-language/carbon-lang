// RUN: clang %s -fsyntax-only -verify

// id is now builtin. There should be no errors. Should probably remove this file.
id obj; 

@interface Foo

- defaultToId; 

@end
