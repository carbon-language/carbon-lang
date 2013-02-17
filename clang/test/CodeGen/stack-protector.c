// RUN: %clang_cc1 -emit-llvm -o - %s -stack-protector 0 | FileCheck -check-prefix=NOSSP %s
// NOSSP: define void @test1(i8* %msg) nounwind {{.*}}{
// RUN: %clang_cc1 -emit-llvm -o - %s -stack-protector 1 | FileCheck -check-prefix=WITHSSP %s
// WITHSSP: define void @test1(i8* %msg) nounwind ssp {{.*}}{
// RUN: %clang_cc1 -emit-llvm -o - %s -stack-protector 2 | FileCheck -check-prefix=SSPREQ %s
// SSPREQ: define void @test1(i8* %msg) nounwind sspreq {{.*}}{

typedef __SIZE_TYPE__ size_t;

int printf(const char * _Format, ...);
size_t strlen(const char *s);
char *strcpy(char *s1, const char *s2);

void test1(const char *msg) {
  char a[strlen(msg) + 1];
  strcpy(a, msg);
  printf("%s\n", a);
}
