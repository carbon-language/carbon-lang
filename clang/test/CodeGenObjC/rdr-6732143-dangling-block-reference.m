// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -emit-llvm -fobjc-exceptions %s -o -

void f0(id x) {
  @synchronized (x) {      
    do { ; } while(0);
    @try {
    } @finally {
    }
  }
}
