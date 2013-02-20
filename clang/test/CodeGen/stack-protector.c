// RUN: %clang_cc1 -emit-llvm -o - %s -stack-protector 0 | FileCheck -check-prefix=NOSSP %s
// NOSSP: define void @test1(i8* %msg) #0 {
// RUN: %clang_cc1 -emit-llvm -o - %s -stack-protector 1 | FileCheck -check-prefix=WITHSSP %s
// WITHSSP: define void @test1(i8* %msg) #0 {
// RUN: %clang_cc1 -emit-llvm -o - %s -stack-protector 2 | FileCheck -check-prefix=SSPREQ %s
// SSPREQ: define void @test1(i8* %msg) #0 {

typedef __SIZE_TYPE__ size_t;

int printf(const char * _Format, ...);
size_t strlen(const char *s);
char *strcpy(char *s1, const char *s2);

void test1(const char *msg) {
  char a[strlen(msg) + 1];
  strcpy(a, msg);
  printf("%s\n", a);
}

// NOSSP: attributes #0 = { nounwind "target-features"={{.*}} }
// NOSSP: attributes #1 = { "target-features"={{.*}} }
// NOSSP: attributes #2 = { nounwind }

// WITHSSP: attributes #0 = { nounwind ssp "target-features"={{.*}} }
// WITHSSP: attributes #1 = { "target-features"={{.*}} }
// WITHSSP: attributes #2 = { nounwind }

// SSPREQ: attributes #0 = { nounwind sspreq "target-features"={{.*}} }
// SSPREQ: attributes #1 = { "target-features"={{.*}} }
// SSPREQ: attributes #2 = { nounwind }
