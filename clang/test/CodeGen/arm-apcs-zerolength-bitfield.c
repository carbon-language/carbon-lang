// RUN: %clang_cc1 -target-abi apcs-gnu -triple armv7-apple-darwin10 %s -verify
//
// Note: gcc forces the alignment to 4 bytes, regardless of the type of the
// zero length bitfield.
// rdar://9859156

#include <stddef.h>

struct t1
{
  int foo : 1;
  char : 0;
  char bar;
};
static int arr1_offset[(offsetof(struct t1, bar) == 4) ? 0 : -1];
static int arr1_sizeof[(sizeof(struct t1) == 8) ? 0 : -1];

struct t2
{
  int foo : 1;
  short : 0;
  char bar;
};
static int arr2_offset[(offsetof(struct t2, bar) == 4) ? 0 : -1];
static int arr2_sizeof[(sizeof(struct t2) == 8) ? 0 : -1];

struct t3
{
  int foo : 1;
  int : 0;
  char bar;
};
static int arr3_offset[(offsetof(struct t3, bar) == 4) ? 0 : -1];
static int arr3_sizeof[(sizeof(struct t3) == 8) ? 0 : -1];

struct t4
{
  int foo : 1;
  long : 0;
  char bar;
};
static int arr4_offset[(offsetof(struct t4, bar) == 4) ? 0 : -1];
static int arr4_sizeof[(sizeof(struct t4) == 8) ? 0 : -1];

struct t5
{
  int foo : 1;
  long long : 0;
  char bar;
};
static int arr5_offset[(offsetof(struct t5, bar) == 4) ? 0 : -1];
static int arr5_sizeof[(sizeof(struct t5) == 8) ? 0 : -1];

struct t6
{
  int foo : 1;
  char : 0;
  char bar : 1;
  char bar2;
};
static int arr6_offset[(offsetof(struct t6, bar2) == 1) ? 0 : -1];
static int arr6_sizeof[(sizeof(struct t6) == 2) ? 0 : -1];

struct t7
{
  int foo : 1;
  short : 0;
  char bar1 : 1;
  char bar2;
};
static int arr7_offset[(offsetof(struct t7, bar2) == 1) ? 0 : -1];
static int arr7_sizeof[(sizeof(struct t7) == 2) ? 0 : -1];

struct t8
{
  int foo : 1;
  int : 0;
  char bar1 : 1;
  char bar2;
};
static int arr8_offset[(offsetof(struct t8, bar2) == 1) ? 0 : -1];
static int arr8_sizeof[(sizeof(struct t8) == 2) ? 0 : -1];

struct t9
{
  int foo : 1;
  long : 0;
  char bar1 : 1;
  char bar2;
};
static int arr9_offset[(offsetof(struct t9, bar2) == 1) ? 0 : -1];
static int arr9_sizeof[(sizeof(struct t9) == 2) ? 0 : -1];

struct t10
{
  int foo : 1;
  long long : 0;
  char bar1 : 1;
  char bar2;
};
static int arr10_offset[(offsetof(struct t10, bar2) == 1) ? 0 : -1];
static int arr10_sizeof[(sizeof(struct t10) == 2) ? 0 : -1];


struct t11
{
  int foo : 1;
  long long : 0;
  char : 0;
  char bar1 : 1;
  char bar2;
};
static int arr11_offset[(offsetof(struct t11, bar2) == 1) ? 0 : -1];
static int arr11_sizeof[(sizeof(struct t11) == 2) ? 0 : -1];

struct t12
{
  int foo : 1;
  char : 0;
  long long : 0;
  char : 0;
  char bar;
};
static int arr12_offset[(offsetof(struct t12, bar) == 4) ? 0 : -1];
static int arr12_sizeof[(sizeof(struct t12) == 8) ? 0 : -1];

int main() {
  return 0;
}

