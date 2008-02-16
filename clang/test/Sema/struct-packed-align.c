// RUN: clang %s -fsyntax-only -verify

struct s {
    char a;
    int b  __attribute__((packed));
    char c;
    int d;
};

struct __attribute__((packed)) packed_s {
    char a;
    int b  __attribute__((packed));
    char c;
    int d;
};

struct fas {
    char a;
    int b[];
};

struct __attribute__((packed)) packed_fas {
    char a;
    int b[];
};

extern int a1[sizeof(struct s) == 12 ? 1 : -1];
extern int a2[__alignof(struct s) == 4 ? 1 : -1];

extern int b1[sizeof(struct packed_s) == 10 ? 1 : -1];
extern int b2[__alignof(struct packed_s) == 1 ? 1 : -1];

extern int c1[sizeof(struct fas) == 4 ? 1 : -1];
extern int c2[__alignof(struct fas) == 4 ? 1 : -1];

extern int d1[sizeof(struct packed_fas) == 1 ? 1 : -1];
extern int d2[__alignof(struct packed_fas) == 1 ? 1 : -1];
