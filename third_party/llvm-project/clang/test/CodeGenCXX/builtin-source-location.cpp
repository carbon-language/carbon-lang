// RUN: %clang_cc1 -std=c++2a -fblocks %s -triple x86_64-unknown-unknown -emit-llvm -o %t.ll

// This needs to be performed before #line directives which alter filename
// RUN: %clang_cc1 -fmacro-prefix-map=%p=/UNLIKELY/PATH -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-PREFIX-MAP
//
// CHECK-PREFIX-MAP: /UNLIKELY/PATH{{/|\\\\}}builtin-source-location.cpp
void testRemap() {
  const char *file = __builtin_FILE();
}

#line 8 "builtin-source-location.cpp"

struct source_location {
private:
  unsigned int __m_line = 0;
  unsigned int __m_col = 0;
  const char *__m_file = nullptr;
  const char *__m_func = nullptr;

public:
  constexpr void set(unsigned l, unsigned c, const char *f, const char *func) {
    __m_line = l;
    __m_col = c;
    __m_file = f;
    __m_func = func;
  }
  static constexpr source_location current(
      unsigned int __line = __builtin_LINE(),
      unsigned int __col = __builtin_COLUMN(),
      const char *__file = __builtin_FILE(),
      const char *__func = __builtin_FUNCTION()) noexcept {
    source_location __loc;
    __loc.set(__line, __col, __file, __func);
    return __loc;
  }
  static source_location bad_current(
      unsigned int __line = __builtin_LINE(),
      unsigned int __col = __builtin_COLUMN(),
      const char *__file = __builtin_FILE(),
      const char *__func = __builtin_FUNCTION()) noexcept {
    source_location __loc;
    __loc.set(__line, __col, __file, __func);
    return __loc;
  }
  constexpr source_location() = default;
  constexpr source_location(source_location const &) = default;
  constexpr unsigned int line() const noexcept { return __m_line; }
  constexpr unsigned int column() const noexcept { return __m_col; }
  constexpr const char *file() const noexcept { return __m_file; }
  constexpr const char *function() const noexcept { return __m_func; }
};

using SL = source_location;

extern "C" int sink(...);


// RUN: FileCheck --input-file %t.ll %s --check-prefix=CHECK-GLOBAL-ONE
//
// CHECK-GLOBAL-ONE-DAG: @[[FILE:.*]] = {{.*}}c"test_const_init.cpp\00"
// CHECK-GLOBAL-ONE-DAG: @[[FUNC:.*]] = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
//
// CHECK-GLOBAL-ONE: @const_init_global ={{.*}} global %struct.source_location { i32 1000, i32 {{[0-9]+}}, {{[^@]*}}@[[FILE]], {{[^@]*}}@[[FUNC]]
#line 1000 "test_const_init.cpp"
SL const_init_global = SL::current();

// RUN: FileCheck --input-file %t.ll %s --check-prefix=CHECK-GLOBAL-TWO
//
// CHECK-GLOBAL-TWO-DAG: @runtime_init_global ={{.*}} global %struct.source_location zeroinitializer, align 8
//
// CHECK-GLOBAL-TWO-DAG: @[[FILE:.*]] = {{.*}}c"test_runtime_init.cpp\00"
// CHECK-GLOBAL-TWO-DAG: @[[FUNC:.*]] = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
//
// CHECK-GLOBAL-TWO: define internal void @__cxx_global_var_init()
// CHECK-GLOBAL-TWO-NOT: ret
// CHECK-GLOBAL-TWO: call void @_ZN15source_location11bad_currentEjjPKcS1_(%struct.source_location* sret(%struct.source_location) align 8 @runtime_init_global,
// CHECK-GLOBAL-TWO-SAME: i32 1100, i32 {{[0-9]+}}, {{[^@]*}}@[[FILE]], {{[^@]*}}@[[FUNC]],
#line 1100 "test_runtime_init.cpp"
SL runtime_init_global = SL::bad_current();

#line 2000 "test_function.cpp"
extern "C" void test_function() {
// RUN: FileCheck --input-file %t.ll %s --check-prefix=CHECK-LOCAL-ONE
//
// CHECK-LOCAL-ONE-DAG: @[[FILE:.*]] = {{.*}}c"test_current.cpp\00"
// CHECK-LOCAL-ONE-DAG: @[[FUNC:.*]] = {{.*}}c"test_function\00"
//
// CHECK-LOCAL-ONE:  call void @_ZN15source_location7currentEjjPKcS1_(%struct.source_location* sret(%struct.source_location) align 8 %local,
// CHECK-LOCAL-ONE-SAME: i32 2100, i32 {{[0-9]+}},
// CHECK-LOCAL-ONE-SAME: {{[^@]*}}@[[FILE]], {{[^@]*}}@[[FUNC]],
#line 2100 "test_current.cpp"
  SL local = SL::current();
}

#line 3000 "TestInitClass.cpp"
struct TestInit {
  SL info = SL::current();
  SL arg_info;

#line 3100 "TestInitCtor.cpp"
  TestInit(SL arg_info = SL::current()) : arg_info(arg_info) {}
};

// RUN: FileCheck --input-file %t.ll %s --check-prefix=CHECK-CTOR-GLOBAL
//
// CHECK-CTOR-GLOBAL-DAG: @GlobalInitVal ={{.*}} global %struct.TestInit zeroinitializer, align 8
// CHECK-CTOR-GLOBAL-DAG: @[[FILE:.*]] = {{.*}}c"GlobalInitVal.cpp\00"
// CHECK-CTOR-GLOBAL-DAG: @[[FUNC:.*]] = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
//
// CHECK-CTOR-GLOBAL: define internal void @__cxx_global_var_init.{{[0-9]+}}()
// CHECK-CTOR-GLOBAL-NOT: ret
//
// CHECK-CTOR-GLOBAL: call void @_ZN15source_location7currentEjjPKcS1_(%struct.source_location* sret(%struct.source_location) align 8 %[[TMP_ONE:[^,]*]],
// CHECK-CTOR-GLOBAL-SAME: i32 3400, i32 {{[0-9]+}}, {{[^@]*}}@[[FILE]], {{[^@]*}}@[[FUNC]],
// CHECK-CTOR-GLOBAL-NEXT: call void @_ZN8TestInitC1E15source_location(%struct.TestInit* {{[^,]*}} @GlobalInitVal, %struct.source_location* {{.*}}%[[TMP_ONE]])
#line 3400 "GlobalInitVal.cpp"
TestInit GlobalInitVal;

extern "C" void test_init_function() {
// RUN: FileCheck --input-file %t.ll %s --check-prefix=CHECK-CTOR-LOCAL
//
// CHECK-CTOR-LOCAL-DAG: @[[FILE:.*]] = {{.*}}c"LocalInitVal.cpp\00"
// CHECK-CTOR-LOCAL-DAG: @[[FUNC:.*]] = {{.*}}c"test_init_function\00"
//
// CHECK-CTOR-LOCAL: define{{.*}} void @test_init_function()
// CHECK-CTOR-LOCAL-NOT: ret
//
// CHECK-CTOR-LOCAL: call void @_ZN15source_location7currentEjjPKcS1_(%struct.source_location* sret(%struct.source_location) align 8 %[[TMP:[^,]*]],
// CHECK-CTOR-LOCAL-SAME: i32 3500, i32 {{[0-9]+}}, {{[^@]*}}@[[FILE]], {{[^@]*}}@[[FUNC]],
// CHECK-CTOR-LOCAL-NEXT: call void @_ZN8TestInitC1E15source_location(%struct.TestInit* {{[^,]*}} %init_local, %struct.source_location* {{.*}}%[[TMP]])
#line 3500 "LocalInitVal.cpp"
  TestInit init_local;
  sink(init_local);
}

#line 4000 "ConstexprClass.cpp"
struct TestInitConstexpr {
  SL info = SL::current();
  SL arg_info;
#line 4200 "ConstexprCtor.cpp"
  constexpr TestInitConstexpr(SL arg_info = SL::current()) : arg_info(arg_info) {}
};

// RUN: FileCheck --input-file %t.ll %s --check-prefix=CHECK-CONSTEXPR-T2
//
// CHECK-CONSTEXPR-T2-DAG: @[[FILE_INIT:.*]] = {{.*}}c"ConstexprCtor.cpp\00"
// CHECK-CONSTEXPR-T2-DAG: @[[FUNC_INIT:.*]] = {{.*}}c"TestInitConstexpr\00"
// CHECK-CONSTEXPR-T2-DAG: @[[FILE_ARG:.*]] = {{.*}}c"ConstexprGlobal.cpp\00"
// CHECK-CONSTEXPR-T2-DAG: @[[EMPTY:.*]] = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
//
// CHECK-CONSTEXPR-T2: @ConstexprGlobal ={{.*}} global %struct.TestInitConstexpr {
// CHECK-CONSTEXPR-T2-SAME: %struct.source_location { i32 4200, i32 {{[0-9]+}}, {{[^@]*}}@[[FILE_INIT]], {{[^@]*}}@[[FUNC_INIT]],
// CHECK-CONSTEXPR-T2-SAME: {{[^%]*}}%struct.source_location { i32 4400, i32 {{[0-9]+}},  {{[^@]*}}@[[FILE_ARG]], {{[^@]*}}@[[EMPTY]]
#line 4400 "ConstexprGlobal.cpp"
TestInitConstexpr ConstexprGlobal;

extern "C" void test_init_function_constexpr() {
// RUN: FileCheck --input-file %t.ll %s --check-prefix=CHECK-CONSTEXPR-LOCAL
//
// CHECK-CONSTEXPR-LOCAL-DAG: @[[FUNC:.*]] = {{.*}}c"test_init_function_constexpr\00"
// CHECK-CONSTEXPR-LOCAL-DAG: @[[FILE:.*]] = {{.*}}c"ConstexprLocal.cpp\00"
//
// CHECK-CONSTEXPR-LOCAL: define{{.*}} void @test_init_function_constexpr()
// CHECK-CONSTEXPR-LOCAL: call void @_ZN15source_location7currentEjjPKcS1_(%struct.source_location* sret(%struct.source_location) align 8 %[[TMP:[^,]*]],
// CHECK-CONSTEXPR-LOCAL-SAME: i32 4600, i32 {{[0-9]+}}, {{[^@]*}}@[[FILE]], {{[^@]*}}@[[FUNC]]
// CHECK-CONSTEXPR-LOCAL: call void @_ZN17TestInitConstexprC1E15source_location(%struct.TestInitConstexpr* {{[^,]*}} %local_val, {{.*}}%[[TMP]])
#line 4600 "ConstexprLocal.cpp"
  TestInitConstexpr local_val;
}

#line 5000 "TestInitAgg.cpp"
struct TestInitAgg {
#line 5100 "i1.cpp"
  SL i1;
#line 5200 "i2.cpp"
  SL i2 = SL::current();
#line 5300 "TestInitAggEnd.cpp"
};

// RUN: FileCheck --input-file %t.ll %s --check-prefix=CHECK-AGG-DEFAULT
//
// CHECK-AGG-DEFAULT-DAG: @[[FILE:.*]] = {{.*}}c"TestInitAgg.cpp\00"
// CHECK-AGG-DEFAULT-DAG: @[[FUNC:.*]] = {{.*}}c"TestInitAgg\00"
//
// CHECK-AGG-DEFAULT: @GlobalAggDefault ={{.*}} global %struct.TestInitAgg {
// CHECK-AGG-DEFAULT-SAME: %struct.source_location zeroinitializer,
// CHECK-AGG-DEFAULT-SAME: %struct.source_location { i32 5000, i32 {{[0-9]+}}, {{[^@]*}}@[[FILE]], {{[^@]*}}@[[FUNC]]
#line 5400 "GlobalAggDefault.cpp"
TestInitAgg GlobalAggDefault;

#line 5500 "test_agg_init_test.cpp"
extern "C" void test_agg_init() {
// RUN: FileCheck --input-file %t.ll %s --check-prefix=CHECK-AGG-BRACE
//
// CHECK-AGG-BRACE-DAG: @[[FILE:.*]] = {{.*}}c"BraceInitEnd.cpp\00"
// CHECK-AGG-BRACE-DAG: @[[FUNC:.*]] = {{.*}}c"test_agg_init\00"
//
// CHECK-AGG-BRACE: define{{.*}} void @test_agg_init()
// CHECK-AGG-BRACE: %[[I2:.*]] = getelementptr inbounds %struct.TestInitAgg, %struct.TestInitAgg* %local_brace_init, i32 0, i32 1
// CHECK-AGG-BRACE-NEXT: call void @_ZN15source_location7currentEjjPKcS1_(%struct.source_location* sret(%struct.source_location) align 8 %[[I2]],
// CHECK-AGG-BRACE-SAME: i32 5700, i32 {{[0-9]+}}, {{[^@]*}}@[[FILE]], {{[^@]*}}@[[FUNC]]
#line 5600 "BraceInitStart.cpp"
  TestInitAgg local_brace_init{
#line 5700 "BraceInitEnd.cpp"
  };

// RUN: FileCheck --input-file %t.ll %s --check-prefix=CHECK-AGG-EQUAL
//
// CHECK-AGG-EQUAL-DAG: @[[FILE:.*]] = {{.*}}c"EqualInitEnd.cpp\00"
// CHECK-AGG-EQUAL-DAG: @[[FUNC:.*]] = {{.*}}c"test_agg_init\00"
//
// CHECK-AGG-EQUAL: define{{.*}} void @test_agg_init()
// CHECK-AGG-EQUAL: %[[I2:.*]] = getelementptr inbounds %struct.TestInitAgg, %struct.TestInitAgg* %local_equal_init, i32 0, i32 1
// CHECK-AGG-EQUAL-NEXT: call void @_ZN15source_location7currentEjjPKcS1_(%struct.source_location* sret(%struct.source_location) align 8 %[[I2]],
// CHECK-AGG-EQUAL-SAME: i32 5900, i32 {{[0-9]+}}, {{[^@]*}}@[[FILE]], {{[^@]*}}@[[FUNC]]
#line 5800 "EqualInitStart.cpp"
  TestInitAgg local_equal_init =
      {
#line 5900 "EqualInitEnd.cpp"
      };

// RUN: FileCheck --input-file %t.ll %s --check-prefix=CHECK-AGG-LIST
//
// CHECK-AGG-LIST-DAG: @[[FILE_DEFAULT:.*]] = {{.*}}c"InitListEnd.cpp\00"
// CHECK-AGG-LIST-DAG: @[[FILE_ELEM:.*]] = {{.*}}c"ListElem.cpp\00"
// CHECK-AGG-LIST-DAG: @[[FUNC:.*]] = {{.*}}c"test_agg_init\00"
//
// CHECK-AGG-LIST: define{{.*}} void @test_agg_init()
//
// CHECK-AGG-LIST: %[[I1:.*]] =  getelementptr inbounds %struct.TestInitAgg, %struct.TestInitAgg* %local_list_init, i32 0, i32 0
// CHECK-AGG-LIST-NEXT: call void @_ZN15source_location7currentEjjPKcS1_(%struct.source_location* sret(%struct.source_location) align 8 %[[I1]],
// CHECK-AGG-LIST-SAME: i32 6100, i32 {{[0-9]+}}, {{[^@]*}}@[[FILE_ELEM]], {{[^@]*}}@[[FUNC]]
//
// CHECK-AGG-LIST: %[[I2:.*]] = getelementptr inbounds %struct.TestInitAgg, %struct.TestInitAgg* %local_list_init, i32 0, i32 1
// CHECK-AGG-LIST-NEXT: call void @_ZN15source_location7currentEjjPKcS1_(%struct.source_location* sret(%struct.source_location) align 8 %[[I2]],
// CHECK-AGG-LIST-SAME: i32 6200, i32 {{[0-9]+}}, {{[^@]*}}@[[FILE_DEFAULT]], {{[^@]*}}@[[FUNC]]
#line 6000 "InitListStart.cpp"
  TestInitAgg local_list_init =
      {
#line 6100 "ListElem.cpp"
          {SL::current()}
#line 6200 "InitListEnd.cpp"
      };
}

#line 7000 "TestTemplate.cpp"
template <class Tp, int>
struct TestTemplate {
  Tp info = Tp::current();
  Tp arg_info;
#line 7100 "TestTemplateCtor.cpp"
  constexpr TestTemplate(Tp arg_info = Tp::current()) : arg_info(arg_info) {}
};

#line 7200 "test_template.cpp"
template <class T, int V>
void test_template() {

// RUN: FileCheck --input-file %t.ll %s --check-prefix=CHECK-TEMPL -DINT_ID=0
// RUN: FileCheck --input-file %t.ll %s --check-prefix=CHECK-TEMPL -DINT_ID=1
//
// CHECK-TEMPL-DAG: @[[FILE:.*]] = {{.*}}c"local_templ.cpp\00"
// CHECK-TEMPL-DAG: @[[FUNC:.*]] = {{.*}}c"test_template\00"
//
// CHECK-TEMPL: define weak_odr void @_Z13test_templateI15source_locationLi[[INT_ID]]EEvv()
// CHECK-TEMPL-NEXT: entry:
// CHECK-TEMPL-NOT: ret
//
// CHECK-TEMPL:  call void @_ZN15source_location7currentEjjPKcS1_(%struct.source_location* sret(%struct.source_location) align 8 %[[TMP:[^,]*]],
// CHECK-TEMPL-SAME: i32 7300, i32 {{[0-9]+}}, {{[^@]*}}@[[FILE]], {{[^@]*}}@[[FUNC]]
#line 7300 "local_templ.cpp"
  TestTemplate<T, V> local_templ;
}
#line 7400 "EndTestTemplate.cpp"
template void test_template<SL, 0>();
template void test_template<SL, 1>();
