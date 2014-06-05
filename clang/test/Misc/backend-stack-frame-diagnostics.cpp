// REQUIRES: x86-registered-target
// RUN: %clang -target i386-apple-darwin -std=c++11 -fblocks -Wframe-larger-than=70 -Xclang -verify -o /dev/null -c %s

// Test that:
//  * The driver passes the option through to the backend.
//  * The frontend diagnostic handler 'demangles' and resolves the correct function definition.

// TODO: Support rich backend diagnostics for Objective-C methods.

extern void doIt(char *);

void frameSizeWarning(int, int) {}

void frameSizeWarning();

void frameSizeWarning() { // expected-warning-re {{stack frame size of {{[0-9]+}} bytes in function 'frameSizeWarning'}}
  char buffer[80];
  doIt(buffer);
}

void frameSizeWarning();

void frameSizeWarning(int) {}

void frameSizeLocalClassWarning() {
  struct S {
    S() { // expected-warning-re {{stack frame size of {{[0-9]+}} bytes in function 'frameSizeLocalClassWarning()::S::S'}}
      char buffer[80];
      doIt(buffer);
    }
  };
  S();
}

void frameSizeLambdaWarning() {
  auto fn =
      []() { // expected-warning-re {{stack frame size of {{[0-9]+}} bytes in lambda expression}}
    char buffer[80];
    doIt(buffer);
  };
  fn();
}

void frameSizeBlocksWarning() {
  auto fn =
      ^() { // expected-warning-re {{stack frame size of {{[0-9]+}} bytes in block literal}}
    char buffer[80];
    doIt(buffer);
  };
  fn();
}
