// RUN: %clang_cc1 -verify -fsyntax-only -Wargument-larger-than=100 %s

// rdar://8548050
namespace rdar8548050 {

struct S100 {
    char x[100];
};

struct S101 {
    char x[101];
};

S100 f100(S100 s) { return s; }

S101 f101(S101 s) { return s; } // expected-warning {{return value of 'f101' is 101 bytes}} \
                                // expected-warning {{size of 's' is 101 bytes}}

typedef int Arr[200];
void farr(Arr a) { }

struct NonPOD {
  char x[200];
  virtual void m();
};

NonPOD fNonPOD(NonPOD s) { return s; }

template <unsigned size>
struct TS {
    char x[size];
};

template <unsigned size>
void tf(TS<size> ts) {} // expected-warning {{size of 'ts' is 300 bytes}}

void g() {
    TS<300> ts;
    tf<300>(ts); // expected-note {{instantiation}}
}

}
