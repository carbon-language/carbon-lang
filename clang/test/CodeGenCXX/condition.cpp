// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s
void *f();

template <typename T> T* g() {
 if (T* t = f())
   return t;

 return 0;
}

void h() {
 void *a = g<void>();
}

struct X {
  X();
  X(const X&);
  ~X();
  operator bool();
};

struct Y {
  Y();
  ~Y();
};

X getX();

void if_destruct(int z) {
  // Verify that the condition variable is destroyed at the end of the
  // "if" statement.
  // CHECK: call void @_ZN1XC1Ev
  // CHECK: call zeroext i1 @_ZN1XcvbEv
  if (X x = X()) {
    // CHECK: store i32 18
    z = 18;
  }
  // CHECK: call void @_ZN1XD1Ev
  // CHECK: store i32 17
  z = 17;

  // CHECK: call void @_ZN1XC1Ev
  if (X x = X())
    Y y;
  // CHECK: br
  // CHECK: call  void @_ZN1YC1Ev
  // CHECK: call  void @_ZN1YD1Ev
  // CHECK: br
  // CHECK: call  void @_ZN1XD1Ev

  // CHECK: call void @_Z4getXv
  // CHECK: call zeroext i1 @_ZN1XcvbEv
  // CHECK: call void @_ZN1XD1Ev
  // CHECK: br
  if (getX()) { }

  // CHECK: ret
}

struct ConvertibleToInt {
  ConvertibleToInt();
  ~ConvertibleToInt();
  operator int();
};

ConvertibleToInt getConvToInt();

void switch_destruct(int z) {
  // CHECK: call void @_ZN16ConvertibleToIntC1Ev
  switch (ConvertibleToInt conv = ConvertibleToInt()) {
  case 0:
    break;

  default:
    // CHECK: {{sw.default:|:5}}
    // CHECK: store i32 19
    z = 19;
    break;
  }
  // CHECK: {{sw.epilog:|:6}}
  // CHECK: call void @_ZN16ConvertibleToIntD1Ev
  // CHECK: store i32 20
  z = 20;

  // CHECK: call void @_Z12getConvToIntv
  // CHECK: call i32 @_ZN16ConvertibleToIntcviEv
  // CHECK: call void @_ZN16ConvertibleToIntD1Ev
  switch(getConvToInt()) {
  case 0:
    break;
  }
  // CHECK: store i32 27
  z = 27;
  // CHECK: ret
}

int foo();

void while_destruct(int z) {
  // CHECK: define void @_Z14while_destructi
  // CHECK: {{while.cond:|:3}}
  while (X x = X()) {
    // CHECK: call void @_ZN1XC1Ev

    // CHECK: {{while.body:|:5}}
    // CHECK: store i32 21
    z = 21;

    // CHECK: {{while.cleanup:|:6}}
    // CHECK: call void @_ZN1XD1Ev
  }
  // CHECK: {{while.end|:8}}
  // CHECK: store i32 22
  z = 22;

  // CHECK: call void @_Z4getXv
  // CHECK: call zeroext i1 @_ZN1XcvbEv
  // CHECK: call void @_ZN1XD1Ev
  // CHECK: br
  while(getX()) { }

  // CHECK: store i32 25
  z = 25;

  // CHECK: ret
}

void for_destruct(int z) {
  // CHECK: define void @_Z12for_destruct
  // CHECK: call void @_ZN1YC1Ev
  for(Y y = Y(); X x = X(); ++z)
    // CHECK: {{for.cond:|:4}}
    // CHECK: call void @_ZN1XC1Ev
    // CHECK: {{for.body:|:6}}
    // CHECK: store i32 23
    z = 23;
    // CHECK: {{for.inc:|:7}}
    // CHECK: br label %{{for.cond.cleanup|10}}
    // CHECK: {{for.cond.cleanup:|:10}}
    // CHECK: call void @_ZN1XD1Ev
  // CHECK: {{for.end:|:12}}
  // CHECK: call void @_ZN1YD1Ev
  // CHECK: store i32 24
  z = 24;

  // CHECK: call void @_Z4getXv
  // CHECK: call zeroext i1 @_ZN1XcvbEv
  // CHECK: call void @_ZN1XD1Ev
  // CHECK: br
  // CHECK: call void @_Z4getXv
  // CHECK: load
  // CHECK: add
  // CHECK: call void @_ZN1XD1Ev
  int i = 0;
  for(; getX(); getX(), ++i) { }
  z = 26;
  // CHECK: store i32 26
  // CHECK: ret
}

void do_destruct(int z) {
  // CHECK: define void @_Z11do_destruct
  do {
    // CHECK: store i32 77
    z = 77;
    // CHECK: call void @_Z4getXv
    // CHECK: call zeroext i1 @_ZN1XcvbEv
    // CHECK: call void @_ZN1XD1Ev
    // CHECK: br
  } while (getX());
  // CHECK: store i32 99
  z = 99;
  // CHECK: ret
}

int f(X); 

template<typename T>
int instantiated(T x) { 
  int result;

  // CHECK: call void @_ZN1XC1ERKS_
  // CHECK: call i32 @_Z1f1X
  // CHECK: call void @_ZN1XD1Ev
  // CHECK: br
  // CHECK: store i32 2
  // CHECK: br
  // CHECK: store i32 3
  if (f(x)) { result = 2; } else { result = 3; }

  // CHECK: call void @_ZN1XC1ERKS_
  // CHECK: call i32 @_Z1f1X
  // CHECK: call void @_ZN1XD1Ev
  // CHECK: br
  // CHECK: store i32 4
  // CHECK: br
  while (f(x)) { result = 4; }

  // CHECK: call void @_ZN1XC1ERKS_
  // CHECK: call i32 @_Z1f1X
  // CHECK: call void @_ZN1XD1Ev
  // CHECK: br
  // CHECK: store i32 6
  // CHECK: br
  // CHECK: call void @_ZN1XC1ERKS_
  // CHECK: call i32 @_Z1f1X
  // CHECK: store i32 5
  // CHECK: call void @_ZN1XD1Ev
  // CHECK: br
  for (; f(x); f(x), result = 5) {
    result = 6;
  }

  // CHECK: call void @_ZN1XC1ERKS_
  // CHECK: call i32 @_Z1f1X
  // CHECK: call void @_ZN1XD1Ev
  // CHECK: switch i32
  // CHECK: store i32 7
  // CHECK: store i32 8
  switch (f(x)) {
  case 0: 
    result = 7;
    break;

  case 1:
    result = 8;
  }

  // CHECK: store i32 9
  // CHECK: br
  // CHECK: call void @_ZN1XC1ERKS_
  // CHECK: call i32 @_Z1f1X
  // CHECK: call void @_ZN1XD1Ev
  // CHECK: br
  do {
    result = 9;
  } while (f(x));

  // CHECK: store i32 10
  // CHECK: call void @_ZN1XC1ERKS_
  // CHECK: call zeroext i1 @_ZN1XcvbEv
  // CHECK: call void @_ZN1XD1Ev
  // CHECK: br
  do {
    result = 10;
  } while (X(x));

  // CHECK: ret i32
  return result;
}

template int instantiated(X);
