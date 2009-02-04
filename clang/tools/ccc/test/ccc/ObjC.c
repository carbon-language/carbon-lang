// RUN: xcc -fsyntax-only %s -ObjC &&
// RUN: ! xcc -fsyntax-only -x c %s -ObjC &&
// RUN: xcc -fsyntax-only %s -ObjC++ &&
// RUN: ! xcc -fsyntax-only -x c %s -ObjC++

@interface A
@end
