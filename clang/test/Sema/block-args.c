// RUN: clang-cc %s -fsyntax-only -verify -fblocks

void take(void*);

void test() {
  take(^(int x){});
  take(^(int x, int y){});
  take(^(int x, int y){});
  take(^(int x, int x){});  // expected-error {{redefinition of parameter 'x'}}


  take(^(int x) { return x+1; });

  int (^CP)(int) = ^(int x) { return x*x; };
  take(CP);

  int arg;
  ^{return 1;}();
  ^{return 2;}(arg); // expected-error {{too many arguments to block call}}
  ^(void){return 3;}(1); // expected-error {{too many arguments to block call}}
  ^(){return 4;}(arg); // expected-error {{too many arguments to block call}}
  ^(int x, ...){return 5;}(arg, arg);   // Explicit varargs, ok.
}

int main(int argc, char** argv) {
  ^(int argCount) {
    argCount = 3;
  }(argc);
}
