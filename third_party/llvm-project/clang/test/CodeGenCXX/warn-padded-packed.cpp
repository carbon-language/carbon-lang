// RUN: %clang_cc1 -triple=x86_64-none-none -Wpadded -Wpacked -verify %s -emit-llvm-only

struct S1 {
  char c;
  short s; // expected-warning {{padding struct 'S1' with 1 byte to align 's'}}
  long l; // expected-warning {{padding struct 'S1' with 4 bytes to align 'l'}}
};

struct S2 { // expected-warning {{padding size of 'S2' with 3 bytes to alignment boundary}}
  int i;
  char c;
};

struct S3 {
  char c;
  int i;
} __attribute__((packed));

struct S4 {
  int i;
  char c;
} __attribute__((packed));

struct S5 {
  char c;
  union {
    char c;
    int i;
  } u; // expected-warning {{padding struct 'S5' with 3 bytes to align 'u'}}
};

struct S6 { // expected-warning {{padding size of 'S6' with 30 bits to alignment boundary}}
  int i : 2;
};

struct S7 { // expected-warning {{padding size of 'S7' with 7 bytes to alignment boundary}}
  char c;
  virtual void m();
};

struct B {
  char c;
};

struct S8 : B {
  int i; // expected-warning {{padding struct 'S8' with 3 bytes to align 'i'}}
};

struct S9 {
  int x;
  int y;
} __attribute__((packed));

struct S10 {
  int x;
  char a,b,c,d;
} __attribute__((packed));


struct S11 { // expected-warning {{packed attribute is unnecessary for 'S11'}}
  bool x;
  char a,b,c,d;
} __attribute__((packed));

struct S12 {
  bool b : 1;
  char c; // expected-warning {{padding struct 'S12' with 7 bits to align 'c'}}
};

struct S13 { // expected-warning {{padding size of 'S13' with 6 bits to alignment boundary}}
  char c;
  bool b : 10;
};

struct S14 { // expected-warning {{packed attribute is unnecessary for 'S14'}}
  char a,b,c,d;
} __attribute__((packed));

struct S15 { // expected-warning {{packed attribute is unnecessary for 'S15'}}
  struct S14 s;
  char a;
} __attribute__((packed));

struct S16 { // expected-warning {{padding size of 'S16' with 2 bytes to alignment boundary}} expected-warning {{packed attribute is unnecessary for 'S16'}}
  char a,b;
} __attribute__((packed, aligned(4)));

struct S17 {
  struct S16 s;
  char a,b;
} __attribute__((packed, aligned(2)));

struct S18 { // expected-warning {{padding size of 'S18' with 2 bytes to alignment boundary}} expected-warning {{packed attribute is unnecessary for 'S18'}}
  struct S16 s;
  char a,b;
} __attribute__((packed, aligned(4)));

struct S19 { // expected-warning {{packed attribute is unnecessary for 'S19'}}
  bool b;
  char a;
} __attribute__((packed, aligned(1)));

struct S20 {
  int i;
  char a;
} __attribute__((packed, aligned(1)));

struct S21 { // expected-warning {{padding size of 'S21' with 4 bits to alignment boundary}}
  unsigned char a : 6;
  unsigned char b : 6;
} __attribute__((packed, aligned(1)));

struct S22 { // expected-warning {{packed attribute is unnecessary for 'S22'}}
  unsigned char a : 4;
  unsigned char b : 4;
} __attribute__((packed));

struct S23 { // expected-warning {{padding size of 'S23' with 4 bits to alignment boundary}} expected-warning {{packed attribute is unnecessary for 'S23'}}
  unsigned char a : 2;
  unsigned char b : 2;
} __attribute__((packed));

struct S24 {
  unsigned char a : 6;
  unsigned char b : 6;
  unsigned char c : 6;
  unsigned char d : 6;
  unsigned char e : 6;
  unsigned char f : 6;
  unsigned char g : 6;
  unsigned char h : 6;
} __attribute__((packed));

struct S25 { // expected-warning {{padding size of 'S25' with 7 bits to alignment boundary}} expected-warning {{packed attribute is unnecessary for 'S25'}}
  unsigned char a;
  unsigned char b : 1;
} __attribute__((packed));

struct S26 { // expected-warning {{packed attribute is unnecessary for 'S26'}}
  unsigned char a : 1;
  unsigned char b; //expected-warning {{padding struct 'S26' with 7 bits to align 'b'}}
} __attribute__((packed));

struct S27 { // expected-warning {{padding size of 'S27' with 7 bits to alignment boundary}}
  unsigned char a : 1;
  unsigned char b : 8;
} __attribute__((packed));


// The warnings are emitted when the layout of the structs is computed, so we have to use them.
void f(S1*, S2*, S3*, S4*, S5*, S6*, S7*, S8*, S9*, S10*, S11*, S12*, S13*,
       S14*, S15*, S16*, S17*, S18*, S19*, S20*, S21*, S22*, S23*, S24*, S25*,
       S26*, S27*){}
