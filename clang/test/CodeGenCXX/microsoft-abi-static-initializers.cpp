// RUN: %clang_cc1 -fms-extensions -emit-llvm %s -o - -mconstructor-aliases -triple=i386-pc-win32 | FileCheck %s

// CHECK: @llvm.global_ctors = appending global [2 x { i32, void ()*, i8* }]
// CHECK: [{ i32, void ()*, i8* } { i32 65535, void ()* @"\01??__Efoo@?$B@H@@2VA@@A@YAXXZ",
// CHECK:       i8* bitcast (%class.A* @"\01?foo@?$B@H@@2VA@@A" to i8*) },
// CHECK:  { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_microsoft_abi_static_initializers.cpp, i8* null }]

struct S {
  S();
  ~S();
};

S s;

// CHECK: define internal void @"\01??__Es@@YAXXZ"()
// CHECK: call x86_thiscallcc %struct.S* @"\01??0S@@QAE@XZ"
// CHECK: call i32 @atexit(void ()* @"\01??__Fs@@YAXXZ")
// CHECK: ret void

// CHECK: define internal void @"\01??__Fs@@YAXXZ"()
// CHECK: call x86_thiscallcc void @"\01??1S@@QAE@XZ"
// CHECK: ret void

// These globals should use distinct guard variables, and not different bits of
// the same global.
__declspec(selectany) S selectany1;
__declspec(selectany) S selectany2;
// CHECK: define linkonce_odr void @"\01??__Eselectany1@@YAXXZ"()
// CHECK-NOT: @"\01??_Bselectany1
// CHECK: call x86_thiscallcc %struct.S* @"\01??0S@@QAE@XZ"
// CHECK: ret void
// CHECK: define linkonce_odr void @"\01??__Eselectany2@@YAXXZ"()
// CHECK-NOT: @"\01??_Bselectany2
// CHECK: call x86_thiscallcc %struct.S* @"\01??0S@@QAE@XZ"
// CHECK: ret void

void StaticLocal() {
  static S TheS;
}
// CHECK-LABEL: define void @"\01?StaticLocal@@YAXXZ"()
// CHECK: load i32* @"\01?$S1@?0??StaticLocal@@YAXXZ@4IA"
// CHECK: store i32 {{.*}}, i32* @"\01?$S1@?0??StaticLocal@@YAXXZ@4IA"
// CHECK: ret

void MultipleStatics() {
  static S S1;
  static S S2;
  static S S3;
  static S S4;
  static S S5;
  static S S6;
  static S S7;
  static S S8;
  static S S9;
  static S S10;
  static S S11;
  static S S12;
  static S S13;
  static S S14;
  static S S15;
  static S S16;
  static S S17;
  static S S18;
  static S S19;
  static S S20;
  static S S21;
  static S S22;
  static S S23;
  static S S24;
  static S S25;
  static S S26;
  static S S27;
  static S S28;
  static S S29;
  static S S30;
  static S S31;
  static S S32;
  static S S33;
  static S S34;
  static S S35;
}
// CHECK-LABEL: define void @"\01?MultipleStatics@@YAXXZ"()
// CHECK: load i32* @"\01?$S1@?0??MultipleStatics@@YAXXZ@4IA"
// CHECK: and i32 {{.*}}, 1
// CHECK: and i32 {{.*}}, 2
// CHECK: and i32 {{.*}}, 4
// CHECK: and i32 {{.*}}, 8
// CHECK: and i32 {{.*}}, 16
//   ...
// CHECK: and i32 {{.*}}, -2147483648
// CHECK: load i32* @"\01?$S1@?0??MultipleStatics@@YAXXZ@4IA1"
// CHECK: and i32 {{.*}}, 1
// CHECK: and i32 {{.*}}, 2
// CHECK: and i32 {{.*}}, 4
// CHECK: ret

// Force WeakODRLinkage by using templates
class A {
 public:
  A() {}
  ~A() {}
  int a;
};

template<typename T>
class B {
 public:
  static A foo;
};

template<typename T> A B<T>::foo;

inline S &UnreachableStatic() {
  if (0) {
    static S s; // bit 1
    return s;
  }
  static S s; // bit 2
  return s;
}

// CHECK-LABEL: define linkonce_odr %struct.S* @"\01?UnreachableStatic@@YAAAUS@@XZ"()
// CHECK: and i32 {{.*}}, 2
// CHECK: or i32 {{.*}}, 2
// CHECK: ret

inline S &getS() {
  static S TheS;
  return TheS;
}

// CHECK-LABEL: define linkonce_odr %struct.S* @"\01?getS@@YAAAUS@@XZ"
// CHECK: load i32* @"\01??_B?1??getS@@YAAAUS@@XZ@51"
// CHECK: and i32 {{.*}}, 1
// CHECK: icmp ne i32 {{.*}}, 0
// CHECK: br i1
//   init:
// CHECK: or i32 {{.*}}, 1
// CHECK: store i32 {{.*}}, i32* @"\01??_B?1??getS@@YAAAUS@@XZ@51"
// CHECK: call x86_thiscallcc %struct.S* @"\01??0S@@QAE@XZ"(%struct.S* @"\01?TheS@?1??getS@@YAAAUS@@XZ@4U2@A")
// CHECK: call i32 @atexit(void ()* @"\01??__FTheS@?1??getS@@YAAAUS@@XZ@YAXXZ")
// CHECK: br label
//   init.end:
// CHECK: ret %struct.S* @"\01?TheS@?1??getS@@YAAAUS@@XZ@4U2@A"

void force_usage() {
  UnreachableStatic();
  getS();
  (void)B<int>::foo;  // (void) - force usage
}

// CHECK: define linkonce_odr void @"\01??__Efoo@?$B@H@@2VA@@A@YAXXZ"()
// CHECK-NOT: and
// CHECK-NOT: ?_Bfoo@
// CHECK: call x86_thiscallcc %class.A* @"\01??0A@@QAE@XZ"
// CHECK: call i32 @atexit(void ()* @"\01??__Ffoo@?$B@H@@2VA@@A@YAXXZ")
// CHECK: ret void

// CHECK: define linkonce_odr x86_thiscallcc %class.A* @"\01??0A@@QAE@XZ"

// CHECK: define linkonce_odr x86_thiscallcc void @"\01??1A@@QAE@XZ"

// CHECK: define internal void @"\01??__Ffoo@?$B@H@@2VA@@A@YAXXZ"
// CHECK: call x86_thiscallcc void @"\01??1A@@QAE@XZ"{{.*}}foo
// CHECK: ret void

// CHECK: define internal void @_GLOBAL__sub_I_microsoft_abi_static_initializers.cpp()
// CHECK: call void @"\01??__Es@@YAXXZ"()
// CHECK: ret void
