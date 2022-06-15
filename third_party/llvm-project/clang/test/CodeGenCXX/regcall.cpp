// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-linux-gnu -emit-llvm -std=c++11     %s -o - | FileCheck -allow-deprecated-dag-overlap -check-prefix=CHECK-LIN -check-prefix=CHECK-LIN64 %s
// RUN: %clang_cc1 -no-opaque-pointers -triple i386-linux-gnu -emit-llvm -std=c++11     %s -o -   | FileCheck -allow-deprecated-dag-overlap -check-prefix=CHECK-LIN -check-prefix=CHECK-LIN32 %s
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-windows-msvc -emit-llvm -std=c++11  %s -o - -DWIN_TEST | FileCheck -allow-deprecated-dag-overlap -check-prefix=CHECK-WIN64 %s
// RUN: %clang_cc1 -no-opaque-pointers -triple i386-windows-msvc -emit-llvm -std=c++11  %s -o - -DWIN_TEST   | FileCheck -allow-deprecated-dag-overlap -check-prefix=CHECK-WIN32 %s

int __regcall foo(int i);

int main()
{
  int p = 0, _data;
  auto lambda = [&](int parameter) -> int {
    _data = foo(parameter);
    return _data;
  };
  return lambda(p);
}
// CHECK-LIN: call x86_regcallcc {{.+}} @_Z15__regcall3__foo
// CHECK-WIN64: call x86_regcallcc {{.+}} @"?foo@@YwHH@Z"
// CHECK-WIN32: call x86_regcallcc {{.+}} @"?foo@@YwHH@Z"

int __regcall foo (int i){
  return i;
}
// CHECK-LIN: define{{.*}} x86_regcallcc noundef {{.+}}@_Z15__regcall3__foo
// CHECK-WIN64: define dso_local x86_regcallcc noundef {{.+}}@"?foo@@YwHH@Z"
// CHECK-WIN32: define dso_local x86_regcallcc noundef {{.+}}@"?foo@@YwHH@Z"

// used to give a body to test_class functions
static int x = 0;
class test_class {
  int a;
public:
#ifndef WIN_TEST
  __regcall
#endif
    test_class(){++x;}
  // CHECK-LIN-DAG: define linkonce_odr x86_regcallcc void @_ZN10test_classC1Ev
  // CHECK-LIN-DAG: define linkonce_odr x86_regcallcc void @_ZN10test_classC2Ev
  // Windows ignores calling convention on constructor/destructors.
  // CHECK-WIN64-DAG: define linkonce_odr dso_local noundef %class.test_class* @"??0test_class@@QEAA@XZ"
  // CHECK-WIN32-DAG: define linkonce_odr dso_local x86_thiscallcc noundef %class.test_class* @"??0test_class@@QAE@XZ"

#ifndef WIN_TEST
  __regcall
#endif
  ~test_class(){--x;}
  // CHECK-LIN-DAG: define linkonce_odr x86_regcallcc void @_ZN10test_classD2Ev
  // CHECK-LIN-DAG: define linkonce_odr x86_regcallcc void @_ZN10test_classD1Ev
  // Windows ignores calling convention on constructor/destructors.
  // CHECK-WIN64-DAG: define linkonce_odr dso_local void @"??1test_class@@QEAA@XZ"
  // CHECK-WIN32-DAG: define linkonce_odr dso_local x86_thiscallcc void @"??1test_class@@QAE@XZ"

  test_class& __regcall operator+=(const test_class&){
    return *this;
  }
  // CHECK-LIN-DAG: define linkonce_odr x86_regcallcc noundef nonnull align 4 dereferenceable(4) %class.test_class* @_ZN10test_classpLERKS_
  // CHECK-WIN64-DAG: define linkonce_odr dso_local x86_regcallcc noundef nonnull align 4 dereferenceable(4) %class.test_class* @"??Ytest_class@@QEAwAEAV0@AEBV0@@Z"
  // CHECK-WIN32-DAG: define linkonce_odr dso_local x86_regcallcc noundef nonnull align 4 dereferenceable(4) %class.test_class* @"??Ytest_class@@QAwAAV0@ABV0@@Z"
  void __regcall do_thing(){}
  // CHECK-LIN-DAG: define linkonce_odr x86_regcallcc void @_ZN10test_class20__regcall3__do_thingEv
  // CHECK-WIN64-DAG: define linkonce_odr dso_local x86_regcallcc void @"?do_thing@test_class@@QEAwXXZ"
  // CHECK-WIN32-DAG: define linkonce_odr dso_local x86_regcallcc void @"?do_thing@test_class@@QAwXXZ"

  template<typename T>
  void __regcall tempFunc(T i){}
  // CHECK-LIN-DAG: define linkonce_odr x86_regcallcc void @_ZN10test_class20__regcall3__tempFuncIiEEvT_
  // CHECK-WIN64-DAG: define linkonce_odr dso_local x86_regcallcc void @"??$freeTempFunc@H@@YwXH@Z"
  // CHECK-WIN32-DAG: define linkonce_odr dso_local x86_regcallcc void @"??$freeTempFunc@H@@YwXH@Z"
};

bool __regcall operator ==(const test_class&, const test_class&){ --x; return false;}
// CHECK-LIN-DAG: define{{.*}} x86_regcallcc noundef zeroext i1 @_ZeqRK10test_classS1_
// CHECK-WIN64-DAG: define dso_local x86_regcallcc noundef zeroext i1 @"??8@Yw_NAEBVtest_class@@0@Z"
// CHECK-WIN32-DAG: define dso_local x86_regcallcc noundef zeroext i1 @"??8@Yw_NABVtest_class@@0@Z"

test_class __regcall operator""_test_class (unsigned long long) { ++x; return test_class{};}
// CHECK-LIN64-DAG: define{{.*}} x86_regcallcc void @_Zli11_test_classy(%class.test_class* noalias sret(%class.test_class) align 4 %agg.result, i64 noundef %0)
// CHECK-LIN32-DAG: define{{.*}} x86_regcallcc void @_Zli11_test_classy(%class.test_class* inreg noalias sret(%class.test_class) align 4 %agg.result, i64 noundef %0)
// CHECK-WIN64-DAG: ??__K_test_class@@Yw?AVtest_class@@_K@Z"
// CHECK-WIN32-DAG: ??__K_test_class@@Yw?AVtest_class@@_K@Z"

template<typename T>
void __regcall freeTempFunc(T i){}
// CHECK-LIN-DAG: define linkonce_odr x86_regcallcc void @_Z24__regcall3__freeTempFuncIiEvT_
// CHECK-WIN64-DAG: define linkonce_odr dso_local x86_regcallcc void @"??$freeTempFunc@H@@YwXH@Z"
// CHECK-WIN32-DAG: define linkonce_odr dso_local x86_regcallcc void @"??$freeTempFunc@H@@YwXH@Z"

// class to force generation of functions
void force_gen() {
  test_class t;
  test_class t2 = 12_test_class;
  t += t2;
  auto t3 = 100_test_class;
  t3.tempFunc(1);
  freeTempFunc(1);
  t3.do_thing();
}

long double _Complex __regcall foo(long double _Complex f) {
  return f;
}
// CHECK-LIN64-DAG: define{{.*}} x86_regcallcc void @_Z15__regcall3__fooCe({ x86_fp80, x86_fp80 }* noalias sret({ x86_fp80, x86_fp80 }) align 16 %agg.result, { x86_fp80, x86_fp80 }* noundef byval({ x86_fp80, x86_fp80 }) align 16 %f)
// CHECK-LIN32-DAG: define{{.*}} x86_regcallcc void @_Z15__regcall3__fooCe({ x86_fp80, x86_fp80 }* inreg noalias sret({ x86_fp80, x86_fp80 }) align 4 %agg.result, { x86_fp80, x86_fp80 }* noundef byval({ x86_fp80, x86_fp80 }) align 4 %f)
// CHECK-WIN64-DAG: define dso_local x86_regcallcc noundef { double, double } @"?foo@@YwU?$_Complex@O@__clang@@U12@@Z"(double noundef %f.0, double noundef %f.1)
// CHECK-WIN32-DAG: define dso_local x86_regcallcc noundef { double, double } @"?foo@@YwU?$_Complex@O@__clang@@U12@@Z"(double noundef %f.0, double noundef %f.1)

// The following caused us to dereference uninitialized memory. The long name
// seems necessary, as does the return types.
float _Complex __regcall callee(float _Complex f);
// CHECK-LIN64-DAG: declare x86_regcallcc noundef <2 x float> @_Z18__regcall3__calleeCf(<2 x float> noundef)
// CHECK-LIN32-DAG: declare x86_regcallcc noundef { float, float } @_Z18__regcall3__calleeCf(float noundef, float noundef)
// CHECK-WIN64-DAG: declare dso_local x86_regcallcc noundef { float, float } @"?callee@@YwU?$_Complex@M@__clang@@U12@@Z"(float noundef, float noundef)
// CHECK-WIN32-DAG: declare dso_local x86_regcallcc noundef { float, float } @"?callee@@YwU?$_Complex@M@__clang@@U12@@Z"(float noundef, float noundef)

__regcall int
some_really_long_name_that_manages_to_hit_the_right_spot_of_mem(int a) {
  float _Complex x[2];
  x[0] = callee(x[0]);
  return a;
}
