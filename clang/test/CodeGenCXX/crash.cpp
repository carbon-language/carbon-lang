// RUN: %clang_cc1 %s -std=c++11 -emit-llvm-only
// RUN: %clang_cc1 -emit-obj -o %t -gline-tables-only -O2 -std=c++11 %s
// CHECK that we don't crash.

// PR11676's example is ill-formed:
/*
union _XEvent {
};
void ProcessEvent() {
  _XEvent pluginEvent = _XEvent();
}
*/

// Example from PR11665:
void f() {
  union U { int field; } u = U();
  (void)U().field;
}

namespace PR17476 {
struct string {
  string(const char *__s);
  string &operator+=(const string &__str);
};

template <class ELFT> void finalizeDefaultAtomValues() {
  auto startEnd = [&](const char * sym)->void {
    string start("__");
    start += sym;
  }
  ;
  startEnd("preinit_array");
}

void f() { finalizeDefaultAtomValues<int>(); }
}

namespace PR22096 {
class _String_val {
  union _Bxty { int i; } _Bx;
};
struct string : public _String_val {
  string(const char *_Ptr) : _String_val() {}
};


int ConvertIPv4NumberToIPv6Number(int);
struct IPEndPoint {
  IPEndPoint();
  IPEndPoint(const int &address, int port);
  const int &address() const {}
};

struct SourceAddressTokenTest {
  SourceAddressTokenTest()
      : ip4_dual_(ConvertIPv4NumberToIPv6Number(ip4_.address()), 1) {}
  const string kPrimary = "<primary>";
  IPEndPoint ip4_;
  IPEndPoint ip4_dual_;
} s;
}
