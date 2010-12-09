// RUN: %clang_cc1 -triple i686-apple-darwin9 -fsyntax-only -verify %s

namespace rdar8745206 {

struct Base {
  int i;
};

#pragma pack(1)
struct Sub : public Base {
  char c;
};

int check[sizeof(Sub) == 5 ? 1 : -1];

}
