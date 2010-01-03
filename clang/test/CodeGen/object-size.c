// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm %s -o - | FileCheck %s

#define strcpy(dest, src) \
  ((__builtin_object_size(dest, 0) != -1ULL) \
   ? __builtin___strcpy_chk (dest, src, __builtin_object_size(dest, 1)) \
   : __inline_strcpy_chk(dest, src))

static char *__inline_strcpy_chk (char *dest, const char *src) {
  return __builtin___strcpy_chk(dest, src, __builtin_object_size(dest, 1));
}

char gbuf[63];
char *gp;
int gi, gj;

void test1() {
  // CHECK:     %call = call i8* @__strcpy_chk(i8* getelementptr inbounds ([63 x i8]* @gbuf, i32 0, i64 4), i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0), i64 59)
  strcpy(&gbuf[4], "Hi there");
}

void test2() {
  // CHECK:     %call = call i8* @__strcpy_chk(i8* getelementptr inbounds ([63 x i8]* @gbuf, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0), i64 63)
  strcpy(gbuf, "Hi there");
}

void test3() {
  // CHECK:     %call = call i8* @__strcpy_chk(i8* getelementptr inbounds ([63 x i8]* @gbuf, i64 1, i64 37), i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0), i64 0)
  strcpy(&gbuf[100], "Hi there");
}

void test4() {
  // CHECK:     %call = call i8* @__strcpy_chk(i8* getelementptr inbounds ([63 x i8]* @gbuf, i32 0, i64 -1), i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0), i64 0)
  strcpy((char*)(void*)&gbuf[-1], "Hi there");
}

void test5() {
  // CHECK:     %tmp = load i8** @gp
  // CHECK-NEXT:%0 = call i64 @llvm.objectsize.i64(i8* %tmp, i1 false)
  // CHECK-NEXT:%cmp = icmp ne i64 %0, -1
  strcpy(gp, "Hi there");
}

void test6() {
  char buf[57];

  // CHECK:       %call = call i8* @__strcpy_chk(i8* %arrayidx, i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0), i64 53)
  strcpy(&buf[4], "Hi there");
}

void test7() {
  int i;
  // CHECK-NOT:   __strcpy_chk
  // CHECK:       %call = call i8* @__inline_strcpy_chk(i8* getelementptr inbounds ([63 x i8]* @gbuf, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0))
  strcpy((++i, gbuf), "Hi there");
}

void test8() {
  char *buf[50];
  // CHECK-NOT:   __strcpy_chk
  // CHECK:       %call = call i8* @__inline_strcpy_chk(i8* %tmp1, i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0))
  strcpy(buf[++gi], "Hi there");
}

void test9() {
  // CHECK-NOT:   __strcpy_chk
  // CHECK:       %call = call i8* @__inline_strcpy_chk(i8* %0, i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0))
  strcpy((char *)((++gi) + gj), "Hi there");
}

char **p;
void test10() {
  // CHECK-NOT:   __strcpy_chk
  // CHECK:       %call = call i8* @__inline_strcpy_chk(i8* %tmp1, i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0))
  strcpy(*(++p), "Hi there");
}

void test11() {
  // CHECK-NOT:   __strcpy_chk
  // CHECK:       %call = call i8* @__inline_strcpy_chk(i8* %tmp, i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0))
  strcpy(gp = gbuf, "Hi there");
}

void test12() {
  // CHECK-NOT:   __strcpy_chk
  // CHECK:       %call = call i8* @__inline_strcpy_chk(i8* %ptrincdec, i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0))
  strcpy(++gp, "Hi there");
}

void test13() {
  // CHECK-NOT:   __strcpy_chk
  // CHECK:       %call = call i8* @__inline_strcpy_chk(i8* %tmp, i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0))
  strcpy(gp++, "Hi there");
}

void test14() {
  // CHECK-NOT:   __strcpy_chk
  // CHECK:       %call = call i8* @__inline_strcpy_chk(i8* %ptrincdec, i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0))
  strcpy(--gp, "Hi there");
}

void test15() {
  // CHECK-NOT:   __strcpy_chk
  // CHECK:       %call = call i8* @__inline_strcpy_chk(i8* %tmp, i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0))
  strcpy(gp--, "Hi there");
}

void test16() {
  // CHECK-NOT:   __strcpy_chk
  // CHECK:       %call = call i8* @__inline_strcpy_chk(i8* %tmp1, i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0))
  strcpy(gp += 1, "Hi there");
}

void test17() {
  // CHECK: store i32 -1
  gi = __builtin_object_size(gp++, 0);
  // CHECK: store i32 -1
  gi = __builtin_object_size(gp++, 1);
  // CHECK: store i32 0
  gi = __builtin_object_size(gp++, 2);
  // CHECK: store i32 0
  gi = __builtin_object_size(gp++, 3);
}
