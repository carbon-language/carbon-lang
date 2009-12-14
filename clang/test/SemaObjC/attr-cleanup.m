// RUN: clang -cc1 %s -verify -fsyntax-only

@class NSString;

void c1(id *a);

void t1()
{
  NSString *s __attribute((cleanup(c1)));
}
