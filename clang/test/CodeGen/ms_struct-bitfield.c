// RUN: %clang_cc1 -emit-llvm -o - -triple x86_64-apple-darwin9 %s | FileCheck %s

// rdar://8823265

#define ATTR __attribute__((__ms_struct__))

struct
{
   char foo;
   long : 0;
   char bar;
} ATTR t1;
int s1 = sizeof(t1);
// CHECK: @s1 = global i32 2

struct
{
   char foo;
   long : 0;
   char : 0;
   int : 0;
   char bar;
} ATTR t2;
int s2 = sizeof(t2);
// CHECK: @s2 = global i32 2

struct
{
   char foo;
   long : 0;
   char : 0;
   int : 0;
   char bar;
   long : 0;
   char : 0;
} ATTR t3;
int s3 = sizeof(t3);
// CHECK: @s3 = global i32 2

struct
{
   long : 0;
   char bar;
} ATTR t4;
int s4 = sizeof(t4);
// CHECK: @s4 = global i32 1

struct
{
   long : 0;
   long : 0;
   char : 0;
   char bar;
} ATTR t5;
int s5 = sizeof(t5);
// CHECK: @s5 = global i32 1

struct
{
   long : 0;
   long : 0;
   char : 0;
   char bar;
} ATTR t6;
int s6 = sizeof(t6);
// CHECK: @s6 = global i32 1

struct
{
   char foo;
   long : 0;
   int : 0;
   char bar;
   char bar1;
   long : 0;
   char bar2;
   char bar3;
   char : 0;
   char bar4;
   char bar5;
   char : 0;
   char bar6;
   char bar7;
} ATTR t7;
int s7 = sizeof(t7);
// CHECK: @s7 = global i32 9

struct
{
   long : 0;
   long : 0;
   char : 0;
} ATTR t8;
int s8 = sizeof(t8);
// CHECK: @s8 = global i32 0

struct
{
   char foo;
   long : 0;
   int : 0;
   char bar;
   char bar1;
   long : 0;
   char bar2;
   char bar3;
   char : 0;
   char bar4;
   char bar5;
   char : 0;
   char bar6;
   char bar7;
   int  i1;
   char : 0;
   long : 0;
   char :4;
   char bar8;
   char : 0;
   char bar9;
   char bar10;
   int  i2;
   char : 0;
   long : 0;
   char :4;
} ATTR t9;
int s9 = sizeof(t9);
// CHECK: @s9 = global i32 28

struct
{
   char foo: 8;
   long : 0;
   char bar;
} ATTR t10;
int s10 = sizeof(t10);
// CHECK: @s10 = global i32 16
