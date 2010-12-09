// RUN: %clang_cc1 -triple i686-apple-darwin9 -fsyntax-only -verify %s

namespace rdar8745206 {

struct Base {
  int i;
};

#pragma pack(push, 1)
struct Sub : public Base {
  char c;
};
#pragma pack(pop)

int check[sizeof(Sub) == 5 ? 1 : -1];

}

namespace check2 {

struct Base {
  virtual ~Base();
  int x;
};

#pragma pack(push, 1)
struct Sub : virtual Base {
  char c;
};
#pragma pack(pop)

int check[sizeof(Sub) == 13 ? 1 : -1];

}
