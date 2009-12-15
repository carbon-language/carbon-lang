// RUN: %clang -O3 -emit-llvm -S -o %t %s
// RUN: grep 'ret i64 4294967292' %t | count 2
// RUN: grep 'ret i64 -4' %t | count 1

long long f0(void) {
 struct { unsigned f0 : 32; } x = { 18 };
 return (long long) (x.f0 - (int) 22);
}

long long f1(void) {
 struct { unsigned f0 : 31; } x = { 18 };
 return (long long) (x.f0 - (int) 22);
}

long long f2(void) {
 struct { unsigned f0     ; } x = { 18 };
 return (long long) (x.f0 - (int) 22);
}
