// RUN: clang-cc -fnext-runtime -fobjc-gc -emit-llvm -o %t %s &&
// RUN: grep -F '@objc_assign_global' %t  | count 2 &&
// RUN: true
id a;
int main() {
        a = 0;
}

