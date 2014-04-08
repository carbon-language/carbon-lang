// RUN: %clang_cc1 -emit-llvm -fms-extensions -xc++ -o - < %s | FileCheck %s

#ifdef __cplusplus
extern "C" {
#endif
#pragma const_seg(".my_const")
#pragma bss_seg(".my_bss")
int D = 1;
#pragma data_seg(".data")
int a = 1;
#pragma data_seg(push, label, ".data2")
extern const int b;
const int b = 1;
const char* s = "my string!";
#pragma data_seg(push, ".my_seg")
int c = 1;
#pragma data_seg(pop, label)
int d = 1;
int e;
#pragma bss_seg(".c")
int f;
void g(void){}
#pragma code_seg(".my_code")
void h(void){}
#pragma bss_seg()
int i;
#pragma bss_seg(".bss1")
#pragma bss_seg(push, test, ".bss2")
#pragma bss_seg()
#pragma bss_seg()
int TEST1;
#pragma bss_seg(pop)
int TEST2;
#ifdef __cplusplus
}
#endif

//CHECK: @D = global i32 1
//CHECK: @a = global i32 1, section ".data"
//CHECK: @b = constant i32 1, section ".my_const"
//CHECK: @[[MYSTR:.*]] = {{.*}} unnamed_addr constant [11 x i8] c"my string!\00"
//CHECK: @s = global i8* getelementptr inbounds ([11 x i8]* @[[MYSTR]], i32 0, i32 0), section ".data2"
//CHECK: @c = global i32 1, section ".my_seg"
//CHECK: @d = global i32 1, section ".data"
//CHECK: @e = global i32 0, section ".my_bss"
//CHECK: @f = global i32 0, section ".c"
//CHECK: @i = global i32 0
//CHECK: @TEST1 = global i32 0
//CHECK: @TEST2 = global i32 0, section ".bss1"
//CHECK: define void @g()
//CHECK: define void @h() {{.*}} section ".my_code"
