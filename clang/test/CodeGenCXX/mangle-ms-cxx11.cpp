// RUN: %clang_cc1 -std=c++11 -fms-extensions -emit-llvm %s -o - -triple=i386-pc-win32 | FileCheck %s

// CHECK: "\01?DeducedType@@3HA"
auto DeducedType = 30;

// CHECK: "\01?LRef@@YAXAAH@Z"
void LRef(int& a) { }

// CHECK: "\01?RRef@@YAH$$QAH@Z"
int RRef(int&& a) { return a; }

// CHECK: "\01?Null@@YAX$$T@Z"
namespace std { typedef decltype(__nullptr) nullptr_t; }
void Null(std::nullptr_t) {}

namespace EnumMangling {
  extern enum Enum01 { } Enum;
  extern enum Enum02 : bool { } BoolEnum;
  extern enum Enum03 : char { } CharEnum;
  extern enum Enum04 : signed char { } SCharEnum;
  extern enum Enum05 : unsigned char { } UCharEnum;
  extern enum Enum06 : short { } SShortEnum;
  extern enum Enum07 : unsigned short { } UShortEnum;
  extern enum Enum08 : int { } SIntEnum;
  extern enum Enum09 : unsigned int { } UIntEnum;
  extern enum Enum10 : long { } SLongEnum;
  extern enum Enum11 : unsigned long { } ULongEnum;
  extern enum Enum12 : long long { } SLongLongEnum;
  extern enum Enum13 : unsigned long long { } ULongLongEnum;
// CHECK-DAG: @"\01?Enum@EnumMangling@@3W4Enum01@1@A"
// CHECK-DAG: @"\01?BoolEnum@EnumMangling@@3W4Enum02@1@A
// CHECK-DAG: @"\01?CharEnum@EnumMangling@@3W4Enum03@1@A
// CHECK-DAG: @"\01?SCharEnum@EnumMangling@@3W4Enum04@1@A
// CHECK-DAG: @"\01?UCharEnum@EnumMangling@@3W4Enum05@1@A
// CHECK-DAG: @"\01?SShortEnum@EnumMangling@@3W4Enum06@1@A"
// CHECK-DAG: @"\01?UShortEnum@EnumMangling@@3W4Enum07@1@A"
// CHECK-DAG: @"\01?SIntEnum@EnumMangling@@3W4Enum08@1@A"
// CHECK-DAG: @"\01?UIntEnum@EnumMangling@@3W4Enum09@1@A"
// CHECK-DAG: @"\01?SLongEnum@EnumMangling@@3W4Enum10@1@A"
// CHECK-DAG: @"\01?ULongEnum@EnumMangling@@3W4Enum11@1@A"
// CHECK-DAG: @"\01?SLongLongEnum@EnumMangling@@3W4Enum12@1@A"
// CHECK-DAG: @"\01?ULongLongEnum@EnumMangling@@3W4Enum13@1@A"
  decltype(Enum) *UseEnum() { return &Enum; }
  decltype(BoolEnum) *UseBoolEnum() { return &BoolEnum; }
  decltype(CharEnum) *UseCharEnum() { return &CharEnum; }
  decltype(SCharEnum) *UseSCharEnum() { return &SCharEnum; }
  decltype(UCharEnum) *UseUCharEnum() { return &UCharEnum; }
  decltype(SShortEnum) *UseSShortEnum() { return &SShortEnum; }
  decltype(UShortEnum) *UseUShortEnum() { return &UShortEnum; }
  decltype(SIntEnum) *UseSIntEnum() { return &SIntEnum; }
  decltype(UIntEnum) *UseUIntEnum() { return &UIntEnum; }
  decltype(SLongEnum) *UseSLongEnum() { return &SLongEnum; }
  decltype(ULongEnum) *UseULongEnum() { return &ULongEnum; }
  decltype(SLongLongEnum) *UseSLongLongEnum() { return &SLongLongEnum; }
  decltype(ULongLongEnum) *UseULongLongEnum() { return &ULongLongEnum; }
  extern enum class EnumClass01 { } EnumClass;
  extern enum class EnumClass02 : bool { } BoolEnumClass;
  extern enum class EnumClass03 : char { } CharEnumClass;
  extern enum class EnumClass04 : signed char { } SCharEnumClass;
  extern enum class EnumClass05 : unsigned char { } UCharEnumClass;
  extern enum class EnumClass06 : short { } SShortEnumClass;
  extern enum class EnumClass07 : unsigned short { } UShortEnumClass;
  extern enum class EnumClass08 : int { } SIntEnumClass;
  extern enum class EnumClass09 : unsigned int { } UIntEnumClass;
  extern enum class EnumClass10 : long { } SLongEnumClass;
  extern enum class EnumClass11 : unsigned long { } ULongEnumClass;
  extern enum class EnumClass12 : long long { } SLongLongEnumClass;
  extern enum class EnumClass13 : unsigned long long { } ULongLongEnumClass;
// CHECK-DAG: @"\01?EnumClass@EnumMangling@@3W4EnumClass01@1@A"
// CHECK-DAG: @"\01?BoolEnumClass@EnumMangling@@3W4EnumClass02@1@A
// CHECK-DAG: @"\01?CharEnumClass@EnumMangling@@3W4EnumClass03@1@A
// CHECK-DAG: @"\01?SCharEnumClass@EnumMangling@@3W4EnumClass04@1@A
// CHECK-DAG: @"\01?UCharEnumClass@EnumMangling@@3W4EnumClass05@1@A
// CHECK-DAG: @"\01?SShortEnumClass@EnumMangling@@3W4EnumClass06@1@A"
// CHECK-DAG: @"\01?UShortEnumClass@EnumMangling@@3W4EnumClass07@1@A"
// CHECK-DAG: @"\01?SIntEnumClass@EnumMangling@@3W4EnumClass08@1@A"
// CHECK-DAG: @"\01?UIntEnumClass@EnumMangling@@3W4EnumClass09@1@A"
// CHECK-DAG: @"\01?SLongEnumClass@EnumMangling@@3W4EnumClass10@1@A"
// CHECK-DAG: @"\01?ULongEnumClass@EnumMangling@@3W4EnumClass11@1@A"
// CHECK-DAG: @"\01?SLongLongEnumClass@EnumMangling@@3W4EnumClass12@1@A"
// CHECK-DAG: @"\01?ULongLongEnumClass@EnumMangling@@3W4EnumClass13@1@A"
  decltype(EnumClass) *UseEnumClass() { return &EnumClass; }
  decltype(BoolEnumClass) *UseBoolEnumClass() { return &BoolEnumClass; }
  decltype(CharEnumClass) *UseCharEnumClass() { return &CharEnumClass; }
  decltype(SCharEnumClass) *UseSCharEnumClass() { return &SCharEnumClass; }
  decltype(UCharEnumClass) *UseUCharEnumClass() { return &UCharEnumClass; }
  decltype(SShortEnumClass) *UseSShortEnumClass() { return &SShortEnumClass; }
  decltype(UShortEnumClass) *UseUShortEnumClass() { return &UShortEnumClass; }
  decltype(SIntEnumClass) *UseSIntEnumClass() { return &SIntEnumClass; }
  decltype(UIntEnumClass) *UseUIntEnumClass() { return &UIntEnumClass; }
  decltype(SLongEnumClass) *UseSLongEnumClass() { return &SLongEnumClass; }
  decltype(ULongEnumClass) *UseULongEnumClass() { return &ULongEnumClass; }
  decltype(SLongLongEnumClass) *UseSLongLongEnumClass() { return &SLongLongEnumClass; }
  decltype(ULongLongEnumClass) *UseULongLongEnumClass() { return &ULongLongEnumClass; }
}

namespace PR18022 {

struct { } a;
decltype(a) fun(decltype(a) x, decltype(a)) { return x; }
// CHECK-DAG: ?fun@PR18022@@YA?AU<unnamed-type-a>@1@U21@0@Z

}

inline int define_lambda() {
  static auto lambda = [] { static int local; ++local; return local; };
// First, we have the static local variable of type "<lambda_1>" inside of
// "define_lambda".
// CHECK-DAG: ?lambda@?1??define_lambda@@YAHXZ@4V<lambda_1>@@A
// Next, we have the "operator()" for "<lambda_1>" which is inside of
// "define_lambda".
// CHECK-DAG: ??R<lambda_1>@?define_lambda@@YAHXZ@QBEHXZ
// Finally, we have the local which is inside of "<lambda_1>" which is inside of
// "define_lambda". Hooray.
// CHECK-DAG: ?local@?2???R<lambda_1>@?define_lambda@@YAHXZ@QBEHXZ@4HA
  return lambda();
}

int call_lambda() {
  return define_lambda();
}
