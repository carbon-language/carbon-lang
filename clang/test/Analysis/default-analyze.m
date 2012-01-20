// RUN: %clang --analyze %s -o %t

// Tests that some specific checkers are enabled by default.

id foo(int x) {
  id title;
  switch (x) {
  case 1:
    title = @"foo"; // expected-warning {{never read}}
  case 2:
    title = @"bar";
    break;
  default:
    title = "@baz";
    break;
  }
  return title;
}


