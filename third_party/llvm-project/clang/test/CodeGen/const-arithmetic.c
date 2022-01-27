// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

// CHECK: @g1 ={{.*}} global [2 x i8*] [i8* getelementptr (i8, i8* getelementptr inbounds ([0 x %struct.anon], [0 x %struct.anon]* @g0, i32 0, i32 0, i32 0), i64 -2), i8* getelementptr (i8, i8* getelementptr inbounds ([0 x %struct.anon], [0 x %struct.anon]* @g0, i32 0, i32 0, i32 0), i64 -46)], align 16
// CHECK: @g2 ={{.*}} global [2 x i8*] [i8* getelementptr (i8, i8* getelementptr inbounds ([0 x %struct.anon], [0 x %struct.anon]* @g0, i32 0, i32 0, i32 0), i64 -2), i8* getelementptr (i8, i8* getelementptr inbounds ([0 x %struct.anon], [0 x %struct.anon]* @g0, i32 0, i32 0, i32 0), i64 -46)], align 16

extern struct { unsigned char a, b; } g0[];
void *g1[] = {g0 + -1, g0 + -23 };
void *g2[] = {g0 - 1, g0 - 23 };
