// RUN: %clang_cc1 %s -fsyntax-only -verify -fblocks -Wblock-missing-explicit-return-type
// rdar://10735698

int f;
int main() {
  int (^bar)() = ^{  if (f) return 'a'; // expected-warning {{block literal is missing explicit return type and returns non-void values}}
                      else return 10; 
                   };

  void (^bar1)() = ^{ f = 100; };

  void (^bar2)() = ^(void){ f = 100; };

  int (^bar3)() = ^ int {  if (f) return 'a';
                    	   else return 10;
                        };

}
