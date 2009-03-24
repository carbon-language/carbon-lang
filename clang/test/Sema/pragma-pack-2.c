// RUN: clang-cc -triple i686-apple-darwin9 %s -fsyntax-only -verify

#include <stddef.h>

#pragma pack(4)

// Baseline
struct s0 {
  char f0;
  int  f1;
};
extern int a0[offsetof(struct s0, f1) == 4 ? 1 : -1];

#pragma pack(push, 2)
struct s1 {
  char f0;
  int  f1;
};
extern int a1[offsetof(struct s1, f1) == 2 ? 1 : -1];
#pragma pack(pop)

// Test scope of definition

#pragma pack(push, 2)
struct s2_0 {
#pragma pack(pop)
  char f0;
  int  f1;
};
extern int a2_0[offsetof(struct s2_0, f1) == 2 ? 1 : -1];

struct s2_1 {
  char f0;
#pragma pack(push, 2)
  int  f1;
#pragma pack(pop)
};
extern int a2_1[offsetof(struct s2_1, f1) == 4 ? 1 : -1];

struct s2_2 {
  char f0;
  int  f1;
#pragma pack(push, 2)
};
#pragma pack(pop)
extern int a2_2[offsetof(struct s2_2, f1) == 4 ? 1 : -1];

struct s2_3 {
  char f0;
#pragma pack(push, 2)
  struct s2_3_0 { 
#pragma pack(pop)
    int f0; 
  } f1;
};
extern int a2_3[offsetof(struct s2_3, f1) == 2 ? 1 : -1];

struct s2_4 {
  char f0;
  struct s2_4_0 { 
    int f0; 
#pragma pack(push, 2)
  } f1;
#pragma pack(pop)
};
extern int a2_4[offsetof(struct s2_4, f1) == 4 ? 1 : -1];

#pragma pack(1)
struct s3_0 {
  char f0;
  int f1;
};
#pragma pack()
struct s3_1 {
  char f0;
  int f1;
};
extern int a3_0[offsetof(struct s3_0, f1) == 1 ? 1 : -1];
extern int a3_1[offsetof(struct s3_1, f1) == 4 ? 1 : -1];

// pack(0) is like pack()
#pragma pack(1)
struct s4_0 {
  char f0;
  int f1;
};
#pragma pack(0)
struct s4_1 {
  char f0;
  int f1;
};
extern int a4_0[offsetof(struct s4_0, f1) == 1 ? 1 : -1];
extern int a4_1[offsetof(struct s4_1, f1) == 4 ? 1 : -1];
