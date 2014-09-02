// RUN: clang-tidy -checks=-*,google-runtime-int %s -- -x c++ 2>&1 | FileCheck %s -implicit-check-not='{{warning:|error:}}'

long a();
// CHECK: [[@LINE-1]]:1: warning: consider replacing 'long' with 'int{{..}}'

typedef unsigned long long uint64; // NOLINT

long b(long = 1);
// CHECK: [[@LINE-1]]:1: warning: consider replacing 'long' with 'int{{..}}'
// CHECK: [[@LINE-2]]:8: warning: consider replacing 'long' with 'int{{..}}'

template <typename T>
void tmpl() {
  T i;
}

short bar(const short, unsigned short) {
// CHECK: [[@LINE-1]]:1: warning: consider replacing 'short' with 'int16'
// CHECK: [[@LINE-2]]:17: warning: consider replacing 'short' with 'int16'
// CHECK: [[@LINE-3]]:24: warning: consider replacing 'unsigned short' with 'uint16'
  long double foo = 42;
  uint64 qux = 42;
  unsigned short port;

  const unsigned short bar = 0;
// CHECK: [[@LINE-1]]:9: warning: consider replacing 'unsigned short' with 'uint16'
  long long *baar;
// CHECK: [[@LINE-1]]:3: warning: consider replacing 'long long' with 'int64'
  const unsigned short &bara = bar;
// CHECK: [[@LINE-1]]:9: warning: consider replacing 'unsigned short' with 'uint16'
  long const long moo = 1;
// CHECK: [[@LINE-1]]:3: warning: consider replacing 'long long' with 'int64'
  long volatile long wat = 42;
// CHECK: [[@LINE-1]]:3: warning: consider replacing 'long long' with 'int64'
  unsigned long y;
// CHECK: [[@LINE-1]]:3: warning: consider replacing 'unsigned long' with 'uint{{..}}'
  unsigned long long **const *tmp;
// CHECK: [[@LINE-1]]:3: warning: consider replacing 'unsigned long long' with 'uint64'
  unsigned long long **const *&z = tmp;
// CHECK: [[@LINE-1]]:3: warning: consider replacing 'unsigned long long' with 'uint64'
  unsigned short porthole;
// CHECK: [[@LINE-1]]:3: warning: consider replacing 'unsigned short' with 'uint16'

  uint64 cast = (short)42;
// CHECK: [[@LINE-1]]:18: warning: consider replacing 'short' with 'int16'

#define l long
  l x;

  tmpl<short>();
// CHECK: [[@LINE-1]]:8: warning: consider replacing 'short' with 'int16'
}

void p(unsigned short port);

void qux() {
  short port;
// CHECK: [[@LINE-1]]:3: warning: consider replacing 'short' with 'int16'
}
