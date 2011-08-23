// RUN: %clang_cc1 %s -emit-llvm -o - -fblocks -triple x86_64-apple-darwin10 
// rdar://10001085

int main() {
  ^{
                __attribute__((__blocks__(byref))) int index = ({ int __a; int __b; __a < __b ? __b : __a; });
   };
}
