// RUN: clang %s -fsyntax-only -verify

// Packed structs.
struct s {
    char a;
    int b  __attribute__((packed));
    char c;
    int d;
};

extern int a1[sizeof(struct s) == 12 ? 1 : -1];
extern int a2[__alignof(struct s) == 4 ? 1 : -1];

struct __attribute__((packed)) packed_s {
    char a;
    int b  __attribute__((packed));
    char c;
    int d;
};

extern int b1[sizeof(struct packed_s) == 10 ? 1 : -1];
extern int b2[__alignof(struct packed_s) == 1 ? 1 : -1];

struct fas {
    char a;
    int b[];
};

extern int c1[sizeof(struct fas) == 4 ? 1 : -1];
extern int c2[__alignof(struct fas) == 4 ? 1 : -1];

struct __attribute__((packed)) packed_fas {
    char a;
    int b[];
};

extern int d1[sizeof(struct packed_fas) == 1 ? 1 : -1];
extern int d2[__alignof(struct packed_fas) == 1 ? 1 : -1];

// Alignment

struct __attribute__((aligned(8))) as1 {
    char c;
};

extern int e1[sizeof(struct as1) == 8 ? 1 : -1];
extern int e2[__alignof(struct as1) == 8 ? 1 : -1];

struct as2 {
    char c;
    int __attribute__((aligned(8))) a;
};

extern int f1[sizeof(struct as2) == 16 ? 1 : -1];
extern int f2[__alignof(struct as2) == 8 ? 1 : -1];

struct __attribute__((packed)) as3 {
    char c;
    int a;
    int __attribute__((aligned(8))) b;
};

extern int g1[sizeof(struct as3) == 16 ? 1 : -1];
extern int g2[__alignof(struct as3) == 8 ? 1 : -1];


// rdar://5921025
struct packedtest {
  int ted_likes_cheese;
  void *args[] __attribute__((packed));
};
