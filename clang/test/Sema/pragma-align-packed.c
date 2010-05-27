// RUN: %clang-cc1 -triple i386-apple-darwin9 -fsyntax-only -verify %s

#pragma pack(push, 1)
struct s0 {
  char f0;
  int  f1 __attribute__((aligned(4)));
};
extern int a[sizeof(struct s0) == 5 ? 1 : -1];
#pragma pack(pop)

struct __attribute__((packed)) s1 {
  char f0;
  int  f1 __attribute__((aligned(4)));
};
extern int a[sizeof(struct s1) == 8 ? 1 : -1];

#pragma options align=packed
struct s2 {
  char f0;
  int  f1 __attribute__((aligned(4)));
};
extern int a[sizeof(struct s2) == 5 ? 1 : -1];
#pragma options align=reset
