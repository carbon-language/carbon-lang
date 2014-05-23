// RUN: %clang_cc1 -emit-llvm -o - -triple=i386-pc-win32 2>/dev/null %s | FileCheck %s

struct N {};
struct M : private N {};
struct X { virtual void f() {} };
class Z { virtual void f() {} };
class V : public X { virtual void f() {} };
class W : M, virtual V { public: virtual void f() {} };
class Y : Z, W, virtual V { public: virtual void g() {} } y;

struct A {};
struct B : A {};
struct C : B { virtual void f() {} } c;

struct X1 { virtual void f() {} };
struct V1 : X1 {};
struct W1 : virtual V1 {};
struct Y1 : W1, virtual V1 {} y1;

struct A1 { virtual void f() {} };
struct B1 : virtual A1 { virtual void f() {} B1() {} } b1;

struct Z2 { virtual void f() {} };
struct Y2 { virtual void f() {} };
struct A2 : Z2, Y2 {};
struct B2 : virtual A2 { B2() {} virtual void f() {} } b2;

// CHECK: @"\01??_R4B2@@6BZ2@@@" = linkonce_odr constant %MSRTTICompleteObjectLocator { i32 0, i32 8, i32 4, i8* bitcast (%"MSRTTITypeDescriptor\08"* @"\01??_R0?AUB2@@@8" to i8*), %MSRTTIClassHierarchyDescriptor* @"\01??_R3B2@@8" }
// CHECK: @"\01??_R0?AUB2@@@8" = linkonce_odr global %"MSRTTITypeDescriptor\08" { i8** @"\01??_7type_info@@6B@", i8* null, [9 x i8] c".?AUB2@@\00" }
// CHECK: @"\01??_R3B2@@8" = linkonce_odr constant %MSRTTIClassHierarchyDescriptor { i32 0, i32 3, i32 4, %MSRTTIBaseClassDescriptor** getelementptr inbounds ([5 x %MSRTTIBaseClassDescriptor*]* @"\01??_R2B2@@8", i32 0, i32 0) }
// CHECK: @"\01??_R2B2@@8" = linkonce_odr constant [5 x %MSRTTIBaseClassDescriptor*] [%MSRTTIBaseClassDescriptor* @"\01??_R1A@?0A@EA@B2@@8", %MSRTTIBaseClassDescriptor* @"\01??_R1A@A@3FA@A2@@8", %MSRTTIBaseClassDescriptor* @"\01??_R1A@A@3EA@Z2@@8", %MSRTTIBaseClassDescriptor* @"\01??_R13A@3EA@Y2@@8", %MSRTTIBaseClassDescriptor* null]
// CHECK: @"\01??_R1A@?0A@EA@B2@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\08"* @"\01??_R0?AUB2@@@8" to i8*), i32 3, i32 0, i32 -1, i32 0, i32 64, %MSRTTIClassHierarchyDescriptor* @"\01??_R3B2@@8" }
// CHECK: @"\01??_R1A@A@3FA@A2@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\08"* @"\01??_R0?AUA2@@@8" to i8*), i32 2, i32 0, i32 0, i32 4, i32 80, %MSRTTIClassHierarchyDescriptor* @"\01??_R3A2@@8" }
// CHECK: @"\01??_R0?AUA2@@@8" = linkonce_odr global %"MSRTTITypeDescriptor\08" { i8** @"\01??_7type_info@@6B@", i8* null, [9 x i8] c".?AUA2@@\00" }
// CHECK: @"\01??_R3A2@@8" = linkonce_odr constant %MSRTTIClassHierarchyDescriptor { i32 0, i32 1, i32 3, %MSRTTIBaseClassDescriptor** getelementptr inbounds ([4 x %MSRTTIBaseClassDescriptor*]* @"\01??_R2A2@@8", i32 0, i32 0) }
// CHECK: @"\01??_R2A2@@8" = linkonce_odr constant [4 x %MSRTTIBaseClassDescriptor*] [%MSRTTIBaseClassDescriptor* @"\01??_R1A@?0A@EA@A2@@8", %MSRTTIBaseClassDescriptor* @"\01??_R1A@?0A@EA@Z2@@8", %MSRTTIBaseClassDescriptor* @"\01??_R13?0A@EA@Y2@@8", %MSRTTIBaseClassDescriptor* null]
// CHECK: @"\01??_R1A@?0A@EA@A2@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\08"* @"\01??_R0?AUA2@@@8" to i8*), i32 2, i32 0, i32 -1, i32 0, i32 64, %MSRTTIClassHierarchyDescriptor* @"\01??_R3A2@@8" }
// CHECK: @"\01??_R1A@?0A@EA@Z2@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\08"* @"\01??_R0?AUZ2@@@8" to i8*), i32 0, i32 0, i32 -1, i32 0, i32 64, %MSRTTIClassHierarchyDescriptor* @"\01??_R3Z2@@8" }
// CHECK: @"\01??_R0?AUZ2@@@8" = linkonce_odr global %"MSRTTITypeDescriptor\08" { i8** @"\01??_7type_info@@6B@", i8* null, [9 x i8] c".?AUZ2@@\00" }
// CHECK: @"\01??_R3Z2@@8" = linkonce_odr constant %MSRTTIClassHierarchyDescriptor { i32 0, i32 0, i32 1, %MSRTTIBaseClassDescriptor** getelementptr inbounds ([2 x %MSRTTIBaseClassDescriptor*]* @"\01??_R2Z2@@8", i32 0, i32 0) }
// CHECK: @"\01??_R2Z2@@8" = linkonce_odr constant [2 x %MSRTTIBaseClassDescriptor*] [%MSRTTIBaseClassDescriptor* @"\01??_R1A@?0A@EA@Z2@@8", %MSRTTIBaseClassDescriptor* null]
// CHECK: @"\01??_R13?0A@EA@Y2@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\08"* @"\01??_R0?AUY2@@@8" to i8*), i32 0, i32 4, i32 -1, i32 0, i32 64, %MSRTTIClassHierarchyDescriptor* @"\01??_R3Y2@@8" }
// CHECK: @"\01??_R0?AUY2@@@8" = linkonce_odr global %"MSRTTITypeDescriptor\08" { i8** @"\01??_7type_info@@6B@", i8* null, [9 x i8] c".?AUY2@@\00" }
// CHECK: @"\01??_R3Y2@@8" = linkonce_odr constant %MSRTTIClassHierarchyDescriptor { i32 0, i32 0, i32 1, %MSRTTIBaseClassDescriptor** getelementptr inbounds ([2 x %MSRTTIBaseClassDescriptor*]* @"\01??_R2Y2@@8", i32 0, i32 0) }
// CHECK: @"\01??_R2Y2@@8" = linkonce_odr constant [2 x %MSRTTIBaseClassDescriptor*] [%MSRTTIBaseClassDescriptor* @"\01??_R1A@?0A@EA@Y2@@8", %MSRTTIBaseClassDescriptor* null]
// CHECK: @"\01??_R1A@?0A@EA@Y2@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\08"* @"\01??_R0?AUY2@@@8" to i8*), i32 0, i32 0, i32 -1, i32 0, i32 64, %MSRTTIClassHierarchyDescriptor* @"\01??_R3Y2@@8" }
// CHECK: @"\01??_R1A@A@3EA@Z2@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\08"* @"\01??_R0?AUZ2@@@8" to i8*), i32 0, i32 0, i32 0, i32 4, i32 64, %MSRTTIClassHierarchyDescriptor* @"\01??_R3Z2@@8" }
// CHECK: @"\01??_R13A@3EA@Y2@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\08"* @"\01??_R0?AUY2@@@8" to i8*), i32 0, i32 4, i32 0, i32 4, i32 64, %MSRTTIClassHierarchyDescriptor* @"\01??_R3Y2@@8" }
// CHECK: @"\01??_R4B2@@6BY2@@@" = linkonce_odr constant %MSRTTICompleteObjectLocator { i32 0, i32 12, i32 8, i8* bitcast (%"MSRTTITypeDescriptor\08"* @"\01??_R0?AUB2@@@8" to i8*), %MSRTTIClassHierarchyDescriptor* @"\01??_R3B2@@8" }
// CHECK: @"\01??_R4A2@@6BZ2@@@" = linkonce_odr constant %MSRTTICompleteObjectLocator { i32 0, i32 0, i32 0, i8* bitcast (%"MSRTTITypeDescriptor\08"* @"\01??_R0?AUA2@@@8" to i8*), %MSRTTIClassHierarchyDescriptor* @"\01??_R3A2@@8" }
// CHECK: @"\01??_R4A2@@6BY2@@@" = linkonce_odr constant %MSRTTICompleteObjectLocator { i32 0, i32 4, i32 0, i8* bitcast (%"MSRTTITypeDescriptor\08"* @"\01??_R0?AUA2@@@8" to i8*), %MSRTTIClassHierarchyDescriptor* @"\01??_R3A2@@8" }
// CHECK: @"\01??_R4Y2@@6B@" = linkonce_odr constant %MSRTTICompleteObjectLocator { i32 0, i32 0, i32 0, i8* bitcast (%"MSRTTITypeDescriptor\08"* @"\01??_R0?AUY2@@@8" to i8*), %MSRTTIClassHierarchyDescriptor* @"\01??_R3Y2@@8" }
// CHECK: @"\01??_R4Z2@@6B@" = linkonce_odr constant %MSRTTICompleteObjectLocator { i32 0, i32 0, i32 0, i8* bitcast (%"MSRTTITypeDescriptor\08"* @"\01??_R0?AUZ2@@@8" to i8*), %MSRTTIClassHierarchyDescriptor* @"\01??_R3Z2@@8" }
// CHECK: @"\01??_R4B1@@6B@" = linkonce_odr constant %MSRTTICompleteObjectLocator { i32 0, i32 8, i32 4, i8* bitcast (%"MSRTTITypeDescriptor\08"* @"\01??_R0?AUB1@@@8" to i8*), %MSRTTIClassHierarchyDescriptor* @"\01??_R3B1@@8" }
// CHECK: @"\01??_R0?AUB1@@@8" = linkonce_odr global %"MSRTTITypeDescriptor\08" { i8** @"\01??_7type_info@@6B@", i8* null, [9 x i8] c".?AUB1@@\00" }
// CHECK: @"\01??_R3B1@@8" = linkonce_odr constant %MSRTTIClassHierarchyDescriptor { i32 0, i32 0, i32 2, %MSRTTIBaseClassDescriptor** getelementptr inbounds ([3 x %MSRTTIBaseClassDescriptor*]* @"\01??_R2B1@@8", i32 0, i32 0) }
// CHECK: @"\01??_R2B1@@8" = linkonce_odr constant [3 x %MSRTTIBaseClassDescriptor*] [%MSRTTIBaseClassDescriptor* @"\01??_R1A@?0A@EA@B1@@8", %MSRTTIBaseClassDescriptor* @"\01??_R1A@A@3FA@A1@@8", %MSRTTIBaseClassDescriptor* null]
// CHECK: @"\01??_R1A@?0A@EA@B1@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\08"* @"\01??_R0?AUB1@@@8" to i8*), i32 1, i32 0, i32 -1, i32 0, i32 64, %MSRTTIClassHierarchyDescriptor* @"\01??_R3B1@@8" }
// CHECK: @"\01??_R1A@A@3FA@A1@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\08"* @"\01??_R0?AUA1@@@8" to i8*), i32 0, i32 0, i32 0, i32 4, i32 80, %MSRTTIClassHierarchyDescriptor* @"\01??_R3A1@@8" }
// CHECK: @"\01??_R0?AUA1@@@8" = linkonce_odr global %"MSRTTITypeDescriptor\08" { i8** @"\01??_7type_info@@6B@", i8* null, [9 x i8] c".?AUA1@@\00" }
// CHECK: @"\01??_R3A1@@8" = linkonce_odr constant %MSRTTIClassHierarchyDescriptor { i32 0, i32 0, i32 1, %MSRTTIBaseClassDescriptor** getelementptr inbounds ([2 x %MSRTTIBaseClassDescriptor*]* @"\01??_R2A1@@8", i32 0, i32 0) }
// CHECK: @"\01??_R2A1@@8" = linkonce_odr constant [2 x %MSRTTIBaseClassDescriptor*] [%MSRTTIBaseClassDescriptor* @"\01??_R1A@?0A@EA@A1@@8", %MSRTTIBaseClassDescriptor* null]
// CHECK: @"\01??_R1A@?0A@EA@A1@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\08"* @"\01??_R0?AUA1@@@8" to i8*), i32 0, i32 0, i32 -1, i32 0, i32 64, %MSRTTIClassHierarchyDescriptor* @"\01??_R3A1@@8" }
// CHECK: @"\01??_R4A1@@6B@" = linkonce_odr constant %MSRTTICompleteObjectLocator { i32 0, i32 0, i32 0, i8* bitcast (%"MSRTTITypeDescriptor\08"* @"\01??_R0?AUA1@@@8" to i8*), %MSRTTIClassHierarchyDescriptor* @"\01??_R3A1@@8" }
// CHECK: @"\01??_R4Y1@@6B@" = linkonce_odr constant %MSRTTICompleteObjectLocator { i32 0, i32 4, i32 0, i8* bitcast (%"MSRTTITypeDescriptor\08"* @"\01??_R0?AUY1@@@8" to i8*), %MSRTTIClassHierarchyDescriptor* @"\01??_R3Y1@@8" }
// CHECK: @"\01??_R0?AUY1@@@8" = linkonce_odr global %"MSRTTITypeDescriptor\08" { i8** @"\01??_7type_info@@6B@", i8* null, [9 x i8] c".?AUY1@@\00" }
// CHECK: @"\01??_R3Y1@@8" = linkonce_odr constant %MSRTTIClassHierarchyDescriptor { i32 0, i32 3, i32 6, %MSRTTIBaseClassDescriptor** getelementptr inbounds ([7 x %MSRTTIBaseClassDescriptor*]* @"\01??_R2Y1@@8", i32 0, i32 0) }
// CHECK: @"\01??_R2Y1@@8" = linkonce_odr constant [7 x %MSRTTIBaseClassDescriptor*] [%MSRTTIBaseClassDescriptor* @"\01??_R1A@?0A@EA@Y1@@8", %MSRTTIBaseClassDescriptor* @"\01??_R1A@?0A@EA@W1@@8", %MSRTTIBaseClassDescriptor* @"\01??_R1A@A@3FA@V1@@8", %MSRTTIBaseClassDescriptor* @"\01??_R1A@A@3EA@X1@@8", %MSRTTIBaseClassDescriptor* @"\01??_R1A@A@3FA@V1@@8", %MSRTTIBaseClassDescriptor* @"\01??_R1A@A@3EA@X1@@8", %MSRTTIBaseClassDescriptor* null]
// CHECK: @"\01??_R1A@?0A@EA@Y1@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\08"* @"\01??_R0?AUY1@@@8" to i8*), i32 5, i32 0, i32 -1, i32 0, i32 64, %MSRTTIClassHierarchyDescriptor* @"\01??_R3Y1@@8" }
// CHECK: @"\01??_R1A@?0A@EA@W1@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\08"* @"\01??_R0?AUW1@@@8" to i8*), i32 2, i32 0, i32 -1, i32 0, i32 64, %MSRTTIClassHierarchyDescriptor* @"\01??_R3W1@@8" }
// CHECK: @"\01??_R0?AUW1@@@8" = linkonce_odr global %"MSRTTITypeDescriptor\08" { i8** @"\01??_7type_info@@6B@", i8* null, [9 x i8] c".?AUW1@@\00" }
// CHECK: @"\01??_R3W1@@8" = linkonce_odr constant %MSRTTIClassHierarchyDescriptor { i32 0, i32 0, i32 3, %MSRTTIBaseClassDescriptor** getelementptr inbounds ([4 x %MSRTTIBaseClassDescriptor*]* @"\01??_R2W1@@8", i32 0, i32 0) }
// CHECK: @"\01??_R2W1@@8" = linkonce_odr constant [4 x %MSRTTIBaseClassDescriptor*] [%MSRTTIBaseClassDescriptor* @"\01??_R1A@?0A@EA@W1@@8", %MSRTTIBaseClassDescriptor* @"\01??_R1A@A@3FA@V1@@8", %MSRTTIBaseClassDescriptor* @"\01??_R1A@A@3EA@X1@@8", %MSRTTIBaseClassDescriptor* null]
// CHECK: @"\01??_R1A@A@3FA@V1@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\08"* @"\01??_R0?AUV1@@@8" to i8*), i32 1, i32 0, i32 0, i32 4, i32 80, %MSRTTIClassHierarchyDescriptor* @"\01??_R3V1@@8" }
// CHECK: @"\01??_R0?AUV1@@@8" = linkonce_odr global %"MSRTTITypeDescriptor\08" { i8** @"\01??_7type_info@@6B@", i8* null, [9 x i8] c".?AUV1@@\00" }
// CHECK: @"\01??_R3V1@@8" = linkonce_odr constant %MSRTTIClassHierarchyDescriptor { i32 0, i32 0, i32 2, %MSRTTIBaseClassDescriptor** getelementptr inbounds ([3 x %MSRTTIBaseClassDescriptor*]* @"\01??_R2V1@@8", i32 0, i32 0) }
// CHECK: @"\01??_R2V1@@8" = linkonce_odr constant [3 x %MSRTTIBaseClassDescriptor*] [%MSRTTIBaseClassDescriptor* @"\01??_R1A@?0A@EA@V1@@8", %MSRTTIBaseClassDescriptor* @"\01??_R1A@?0A@EA@X1@@8", %MSRTTIBaseClassDescriptor* null]
// CHECK: @"\01??_R1A@?0A@EA@V1@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\08"* @"\01??_R0?AUV1@@@8" to i8*), i32 1, i32 0, i32 -1, i32 0, i32 64, %MSRTTIClassHierarchyDescriptor* @"\01??_R3V1@@8" }
// CHECK: @"\01??_R1A@?0A@EA@X1@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\08"* @"\01??_R0?AUX1@@@8" to i8*), i32 0, i32 0, i32 -1, i32 0, i32 64, %MSRTTIClassHierarchyDescriptor* @"\01??_R3X1@@8" }
// CHECK: @"\01??_R0?AUX1@@@8" = linkonce_odr global %"MSRTTITypeDescriptor\08" { i8** @"\01??_7type_info@@6B@", i8* null, [9 x i8] c".?AUX1@@\00" }
// CHECK: @"\01??_R3X1@@8" = linkonce_odr constant %MSRTTIClassHierarchyDescriptor { i32 0, i32 0, i32 1, %MSRTTIBaseClassDescriptor** getelementptr inbounds ([2 x %MSRTTIBaseClassDescriptor*]* @"\01??_R2X1@@8", i32 0, i32 0) }
// CHECK: @"\01??_R2X1@@8" = linkonce_odr constant [2 x %MSRTTIBaseClassDescriptor*] [%MSRTTIBaseClassDescriptor* @"\01??_R1A@?0A@EA@X1@@8", %MSRTTIBaseClassDescriptor* null]
// CHECK: @"\01??_R1A@A@3EA@X1@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\08"* @"\01??_R0?AUX1@@@8" to i8*), i32 0, i32 0, i32 0, i32 4, i32 64, %MSRTTIClassHierarchyDescriptor* @"\01??_R3X1@@8" }
// CHECK: @"\01??_R4W1@@6B@" = linkonce_odr constant %MSRTTICompleteObjectLocator { i32 0, i32 4, i32 0, i8* bitcast (%"MSRTTITypeDescriptor\08"* @"\01??_R0?AUW1@@@8" to i8*), %MSRTTIClassHierarchyDescriptor* @"\01??_R3W1@@8" }
// CHECK: @"\01??_R4V1@@6B@" = linkonce_odr constant %MSRTTICompleteObjectLocator { i32 0, i32 0, i32 0, i8* bitcast (%"MSRTTITypeDescriptor\08"* @"\01??_R0?AUV1@@@8" to i8*), %MSRTTIClassHierarchyDescriptor* @"\01??_R3V1@@8" }
// CHECK: @"\01??_R4X1@@6B@" = linkonce_odr constant %MSRTTICompleteObjectLocator { i32 0, i32 0, i32 0, i8* bitcast (%"MSRTTITypeDescriptor\08"* @"\01??_R0?AUX1@@@8" to i8*), %MSRTTIClassHierarchyDescriptor* @"\01??_R3X1@@8" }
// CHECK: @"\01??_R4C@@6B@" = linkonce_odr constant %MSRTTICompleteObjectLocator { i32 0, i32 0, i32 0, i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUC@@@8" to i8*), %MSRTTIClassHierarchyDescriptor* @"\01??_R3C@@8" }
// CHECK: @"\01??_R0?AUC@@@8" = linkonce_odr global %"MSRTTITypeDescriptor\07" { i8** @"\01??_7type_info@@6B@", i8* null, [8 x i8] c".?AUC@@\00" }
// CHECK: @"\01??_R3C@@8" = linkonce_odr constant %MSRTTIClassHierarchyDescriptor { i32 0, i32 0, i32 3, %MSRTTIBaseClassDescriptor** getelementptr inbounds ([4 x %MSRTTIBaseClassDescriptor*]* @"\01??_R2C@@8", i32 0, i32 0) }
// CHECK: @"\01??_R2C@@8" = linkonce_odr constant [4 x %MSRTTIBaseClassDescriptor*] [%MSRTTIBaseClassDescriptor* @"\01??_R1A@?0A@EA@C@@8", %MSRTTIBaseClassDescriptor* @"\01??_R13?0A@EA@B@@8", %MSRTTIBaseClassDescriptor* @"\01??_R13?0A@EA@A@@8", %MSRTTIBaseClassDescriptor* null]
// CHECK: @"\01??_R1A@?0A@EA@C@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUC@@@8" to i8*), i32 2, i32 0, i32 -1, i32 0, i32 64, %MSRTTIClassHierarchyDescriptor* @"\01??_R3C@@8" }
// CHECK: @"\01??_R13?0A@EA@B@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUB@@@8" to i8*), i32 1, i32 4, i32 -1, i32 0, i32 64, %MSRTTIClassHierarchyDescriptor* @"\01??_R3B@@8" }
// CHECK: @"\01??_R0?AUB@@@8" = linkonce_odr global %"MSRTTITypeDescriptor\07" { i8** @"\01??_7type_info@@6B@", i8* null, [8 x i8] c".?AUB@@\00" }
// CHECK: @"\01??_R3B@@8" = linkonce_odr constant %MSRTTIClassHierarchyDescriptor { i32 0, i32 0, i32 2, %MSRTTIBaseClassDescriptor** getelementptr inbounds ([3 x %MSRTTIBaseClassDescriptor*]* @"\01??_R2B@@8", i32 0, i32 0) }
// CHECK: @"\01??_R2B@@8" = linkonce_odr constant [3 x %MSRTTIBaseClassDescriptor*] [%MSRTTIBaseClassDescriptor* @"\01??_R1A@?0A@EA@B@@8", %MSRTTIBaseClassDescriptor* @"\01??_R1A@?0A@EA@A@@8", %MSRTTIBaseClassDescriptor* null]
// CHECK: @"\01??_R1A@?0A@EA@B@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUB@@@8" to i8*), i32 1, i32 0, i32 -1, i32 0, i32 64, %MSRTTIClassHierarchyDescriptor* @"\01??_R3B@@8" }
// CHECK: @"\01??_R1A@?0A@EA@A@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUA@@@8" to i8*), i32 0, i32 0, i32 -1, i32 0, i32 64, %MSRTTIClassHierarchyDescriptor* @"\01??_R3A@@8" }
// CHECK: @"\01??_R0?AUA@@@8" = linkonce_odr global %"MSRTTITypeDescriptor\07" { i8** @"\01??_7type_info@@6B@", i8* null, [8 x i8] c".?AUA@@\00" }
// CHECK: @"\01??_R3A@@8" = linkonce_odr constant %MSRTTIClassHierarchyDescriptor { i32 0, i32 0, i32 1, %MSRTTIBaseClassDescriptor** getelementptr inbounds ([2 x %MSRTTIBaseClassDescriptor*]* @"\01??_R2A@@8", i32 0, i32 0) }
// CHECK: @"\01??_R2A@@8" = linkonce_odr constant [2 x %MSRTTIBaseClassDescriptor*] [%MSRTTIBaseClassDescriptor* @"\01??_R1A@?0A@EA@A@@8", %MSRTTIBaseClassDescriptor* null]
// CHECK: @"\01??_R13?0A@EA@A@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUA@@@8" to i8*), i32 0, i32 4, i32 -1, i32 0, i32 64, %MSRTTIClassHierarchyDescriptor* @"\01??_R3A@@8" }
// CHECK: @"\01??_R4Y@@6BZ@@@" = linkonce_odr constant %MSRTTICompleteObjectLocator { i32 0, i32 0, i32 0, i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AVY@@@8" to i8*), %MSRTTIClassHierarchyDescriptor* @"\01??_R3Y@@8" }
// CHECK: @"\01??_R0?AVY@@@8" = linkonce_odr global %"MSRTTITypeDescriptor\07" { i8** @"\01??_7type_info@@6B@", i8* null, [8 x i8] c".?AVY@@\00" }
// CHECK: @"\01??_R3Y@@8" = linkonce_odr constant %MSRTTIClassHierarchyDescriptor { i32 0, i32 3, i32 9, %MSRTTIBaseClassDescriptor** getelementptr inbounds ([10 x %MSRTTIBaseClassDescriptor*]* @"\01??_R2Y@@8", i32 0, i32 0) }
// CHECK: @"\01??_R2Y@@8" = linkonce_odr constant [10 x %MSRTTIBaseClassDescriptor*] [%MSRTTIBaseClassDescriptor* @"\01??_R1A@?0A@EA@Y@@8", %MSRTTIBaseClassDescriptor* @"\01??_R1A@?0A@EN@Z@@8", %MSRTTIBaseClassDescriptor* @"\01??_R13?0A@EN@W@@8", %MSRTTIBaseClassDescriptor* @"\01??_R17?0A@EN@M@@8", %MSRTTIBaseClassDescriptor* @"\01??_R17?0A@EN@N@@8", %MSRTTIBaseClassDescriptor* @"\01??_R1A@33FN@V@@8", %MSRTTIBaseClassDescriptor* @"\01??_R1A@33EJ@X@@8", %MSRTTIBaseClassDescriptor* @"\01??_R1A@33FN@V@@8", %MSRTTIBaseClassDescriptor* @"\01??_R1A@33EJ@X@@8", %MSRTTIBaseClassDescriptor* null]
// CHECK: @"\01??_R1A@?0A@EA@Y@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AVY@@@8" to i8*), i32 8, i32 0, i32 -1, i32 0, i32 64, %MSRTTIClassHierarchyDescriptor* @"\01??_R3Y@@8" }
// CHECK: @"\01??_R1A@?0A@EN@Z@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AVZ@@@8" to i8*), i32 0, i32 0, i32 -1, i32 0, i32 77, %MSRTTIClassHierarchyDescriptor* @"\01??_R3Z@@8" }
// CHECK: @"\01??_R0?AVZ@@@8" = linkonce_odr global %"MSRTTITypeDescriptor\07" { i8** @"\01??_7type_info@@6B@", i8* null, [8 x i8] c".?AVZ@@\00" }
// CHECK: @"\01??_R3Z@@8" = linkonce_odr constant %MSRTTIClassHierarchyDescriptor { i32 0, i32 0, i32 1, %MSRTTIBaseClassDescriptor** getelementptr inbounds ([2 x %MSRTTIBaseClassDescriptor*]* @"\01??_R2Z@@8", i32 0, i32 0) }
// CHECK: @"\01??_R2Z@@8" = linkonce_odr constant [2 x %MSRTTIBaseClassDescriptor*] [%MSRTTIBaseClassDescriptor* @"\01??_R1A@?0A@EA@Z@@8", %MSRTTIBaseClassDescriptor* null]
// CHECK: @"\01??_R1A@?0A@EA@Z@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AVZ@@@8" to i8*), i32 0, i32 0, i32 -1, i32 0, i32 64, %MSRTTIClassHierarchyDescriptor* @"\01??_R3Z@@8" }
// CHECK: @"\01??_R13?0A@EN@W@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AVW@@@8" to i8*), i32 4, i32 4, i32 -1, i32 0, i32 77, %MSRTTIClassHierarchyDescriptor* @"\01??_R3W@@8" }
// CHECK: @"\01??_R0?AVW@@@8" = linkonce_odr global %"MSRTTITypeDescriptor\07" { i8** @"\01??_7type_info@@6B@", i8* null, [8 x i8] c".?AVW@@\00" }
// CHECK: @"\01??_R3W@@8" = linkonce_odr constant %MSRTTIClassHierarchyDescriptor { i32 0, i32 3, i32 5, %MSRTTIBaseClassDescriptor** getelementptr inbounds ([6 x %MSRTTIBaseClassDescriptor*]* @"\01??_R2W@@8", i32 0, i32 0) }
// CHECK: @"\01??_R2W@@8" = linkonce_odr constant [6 x %MSRTTIBaseClassDescriptor*] [%MSRTTIBaseClassDescriptor* @"\01??_R1A@?0A@EA@W@@8", %MSRTTIBaseClassDescriptor* @"\01??_R13?0A@EN@M@@8", %MSRTTIBaseClassDescriptor* @"\01??_R13?0A@EN@N@@8", %MSRTTIBaseClassDescriptor* @"\01??_R1A@A@3FN@V@@8", %MSRTTIBaseClassDescriptor* @"\01??_R1A@A@3EJ@X@@8", %MSRTTIBaseClassDescriptor* null]
// CHECK: @"\01??_R1A@?0A@EA@W@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AVW@@@8" to i8*), i32 4, i32 0, i32 -1, i32 0, i32 64, %MSRTTIClassHierarchyDescriptor* @"\01??_R3W@@8" }
// CHECK: @"\01??_R13?0A@EN@M@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUM@@@8" to i8*), i32 1, i32 4, i32 -1, i32 0, i32 77, %MSRTTIClassHierarchyDescriptor* @"\01??_R3M@@8" }
// CHECK: @"\01??_R0?AUM@@@8" = linkonce_odr global %"MSRTTITypeDescriptor\07" { i8** @"\01??_7type_info@@6B@", i8* null, [8 x i8] c".?AUM@@\00" }
// CHECK: @"\01??_R3M@@8" = linkonce_odr constant %MSRTTIClassHierarchyDescriptor { i32 0, i32 0, i32 2, %MSRTTIBaseClassDescriptor** getelementptr inbounds ([3 x %MSRTTIBaseClassDescriptor*]* @"\01??_R2M@@8", i32 0, i32 0) }
// CHECK: @"\01??_R2M@@8" = linkonce_odr constant [3 x %MSRTTIBaseClassDescriptor*] [%MSRTTIBaseClassDescriptor* @"\01??_R1A@?0A@EA@M@@8", %MSRTTIBaseClassDescriptor* @"\01??_R1A@?0A@EN@N@@8", %MSRTTIBaseClassDescriptor* null]
// CHECK: @"\01??_R1A@?0A@EA@M@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUM@@@8" to i8*), i32 1, i32 0, i32 -1, i32 0, i32 64, %MSRTTIClassHierarchyDescriptor* @"\01??_R3M@@8" }
// CHECK: @"\01??_R1A@?0A@EN@N@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUN@@@8" to i8*), i32 0, i32 0, i32 -1, i32 0, i32 77, %MSRTTIClassHierarchyDescriptor* @"\01??_R3N@@8" }
// CHECK: @"\01??_R0?AUN@@@8" = linkonce_odr global %"MSRTTITypeDescriptor\07" { i8** @"\01??_7type_info@@6B@", i8* null, [8 x i8] c".?AUN@@\00" }
// CHECK: @"\01??_R3N@@8" = linkonce_odr constant %MSRTTIClassHierarchyDescriptor { i32 0, i32 0, i32 1, %MSRTTIBaseClassDescriptor** getelementptr inbounds ([2 x %MSRTTIBaseClassDescriptor*]* @"\01??_R2N@@8", i32 0, i32 0) }
// CHECK: @"\01??_R2N@@8" = linkonce_odr constant [2 x %MSRTTIBaseClassDescriptor*] [%MSRTTIBaseClassDescriptor* @"\01??_R1A@?0A@EA@N@@8", %MSRTTIBaseClassDescriptor* null]
// CHECK: @"\01??_R1A@?0A@EA@N@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUN@@@8" to i8*), i32 0, i32 0, i32 -1, i32 0, i32 64, %MSRTTIClassHierarchyDescriptor* @"\01??_R3N@@8" }
// CHECK: @"\01??_R13?0A@EN@N@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUN@@@8" to i8*), i32 0, i32 4, i32 -1, i32 0, i32 77, %MSRTTIClassHierarchyDescriptor* @"\01??_R3N@@8" }
// CHECK: @"\01??_R1A@A@3FN@V@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AVV@@@8" to i8*), i32 1, i32 0, i32 0, i32 4, i32 93, %MSRTTIClassHierarchyDescriptor* @"\01??_R3V@@8" }
// CHECK: @"\01??_R0?AVV@@@8" = linkonce_odr global %"MSRTTITypeDescriptor\07" { i8** @"\01??_7type_info@@6B@", i8* null, [8 x i8] c".?AVV@@\00" }
// CHECK: @"\01??_R3V@@8" = linkonce_odr constant %MSRTTIClassHierarchyDescriptor { i32 0, i32 0, i32 2, %MSRTTIBaseClassDescriptor** getelementptr inbounds ([3 x %MSRTTIBaseClassDescriptor*]* @"\01??_R2V@@8", i32 0, i32 0) }
// CHECK: @"\01??_R2V@@8" = linkonce_odr constant [3 x %MSRTTIBaseClassDescriptor*] [%MSRTTIBaseClassDescriptor* @"\01??_R1A@?0A@EA@V@@8", %MSRTTIBaseClassDescriptor* @"\01??_R1A@?0A@EA@X@@8", %MSRTTIBaseClassDescriptor* null]
// CHECK: @"\01??_R1A@?0A@EA@V@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AVV@@@8" to i8*), i32 1, i32 0, i32 -1, i32 0, i32 64, %MSRTTIClassHierarchyDescriptor* @"\01??_R3V@@8" }
// CHECK: @"\01??_R1A@?0A@EA@X@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUX@@@8" to i8*), i32 0, i32 0, i32 -1, i32 0, i32 64, %MSRTTIClassHierarchyDescriptor* @"\01??_R3X@@8" }
// CHECK: @"\01??_R0?AUX@@@8" = linkonce_odr global %"MSRTTITypeDescriptor\07" { i8** @"\01??_7type_info@@6B@", i8* null, [8 x i8] c".?AUX@@\00" }
// CHECK: @"\01??_R3X@@8" = linkonce_odr constant %MSRTTIClassHierarchyDescriptor { i32 0, i32 0, i32 1, %MSRTTIBaseClassDescriptor** getelementptr inbounds ([2 x %MSRTTIBaseClassDescriptor*]* @"\01??_R2X@@8", i32 0, i32 0) }
// CHECK: @"\01??_R2X@@8" = linkonce_odr constant [2 x %MSRTTIBaseClassDescriptor*] [%MSRTTIBaseClassDescriptor* @"\01??_R1A@?0A@EA@X@@8", %MSRTTIBaseClassDescriptor* null]
// CHECK: @"\01??_R1A@A@3EJ@X@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUX@@@8" to i8*), i32 0, i32 0, i32 0, i32 4, i32 73, %MSRTTIClassHierarchyDescriptor* @"\01??_R3X@@8" }
// CHECK: @"\01??_R17?0A@EN@M@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUM@@@8" to i8*), i32 1, i32 8, i32 -1, i32 0, i32 77, %MSRTTIClassHierarchyDescriptor* @"\01??_R3M@@8" }
// CHECK: @"\01??_R17?0A@EN@N@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUN@@@8" to i8*), i32 0, i32 8, i32 -1, i32 0, i32 77, %MSRTTIClassHierarchyDescriptor* @"\01??_R3N@@8" }
// CHECK: @"\01??_R1A@33FN@V@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AVV@@@8" to i8*), i32 1, i32 0, i32 4, i32 4, i32 93, %MSRTTIClassHierarchyDescriptor* @"\01??_R3V@@8" }
// CHECK: @"\01??_R1A@33EJ@X@@8" = linkonce_odr constant %MSRTTIBaseClassDescriptor { i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUX@@@8" to i8*), i32 0, i32 0, i32 4, i32 4, i32 73, %MSRTTIClassHierarchyDescriptor* @"\01??_R3X@@8" }
// CHECK: @"\01??_R4Y@@6BW@@@" = linkonce_odr constant %MSRTTICompleteObjectLocator { i32 0, i32 8, i32 0, i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AVY@@@8" to i8*), %MSRTTIClassHierarchyDescriptor* @"\01??_R3Y@@8" }
// CHECK: @"\01??_R4W@@6B@" = linkonce_odr constant %MSRTTICompleteObjectLocator { i32 0, i32 4, i32 0, i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AVW@@@8" to i8*), %MSRTTIClassHierarchyDescriptor* @"\01??_R3W@@8" }
// CHECK: @"\01??_R4Z@@6B@" = linkonce_odr constant %MSRTTICompleteObjectLocator { i32 0, i32 0, i32 0, i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AVZ@@@8" to i8*), %MSRTTIClassHierarchyDescriptor* @"\01??_R3Z@@8" }
// CHECK: @"\01??_R4V@@6B@" = linkonce_odr constant %MSRTTICompleteObjectLocator { i32 0, i32 0, i32 0, i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AVV@@@8" to i8*), %MSRTTIClassHierarchyDescriptor* @"\01??_R3V@@8" }
// CHECK: @"\01??_R4X@@6B@" = linkonce_odr constant %MSRTTICompleteObjectLocator { i32 0, i32 0, i32 0, i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUX@@@8" to i8*), %MSRTTIClassHierarchyDescriptor* @"\01??_R3X@@8" }
