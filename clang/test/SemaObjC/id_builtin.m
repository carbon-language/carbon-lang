// RUN: clang -cc1 %s -fsyntax-only -verify

// id is now builtin. There should be no errors. 
id obj; 

@interface Foo

- defaultToId; 

@end
