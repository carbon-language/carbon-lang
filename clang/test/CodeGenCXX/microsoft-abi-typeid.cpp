// RUN: %clang_cc1 -emit-llvm -O2 -optzns -o - -triple=i386-pc-win32 2>/dev/null %s | FileCheck %s
// REQUIRES: asserts

struct type_info { const char* raw_name() const; };
namespace std { using ::type_info; }

struct V { virtual void f() {}; };
struct A : virtual V {};

A a;
int b;
A* fn();

const std::type_info* test0_typeid() { return &typeid(int); }
// CHECK: define %struct.type_info* @"\01?test0_typeid@@YAPBUtype_info@@XZ"() #0 {
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret %struct.type_info* bitcast (%"MSRTTITypeDescriptor\02"* @"\01??_R0H@8" to %struct.type_info*)
// CHECK-NEXT: }

const std::type_info* test1_typeid() { return &typeid(A); }
// CHECK: define %struct.type_info* @"\01?test1_typeid@@YAPBUtype_info@@XZ"() #0 {
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret %struct.type_info* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUA@@@8" to %struct.type_info*)
// CHECK-NEXT: }

const std::type_info* test2_typeid() { return &typeid(&a); }
// CHECK: define %struct.type_info* @"\01?test2_typeid@@YAPBUtype_info@@XZ"() #0 {
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret %struct.type_info* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0PAUA@@@8" to %struct.type_info*)
// CHECK-NEXT: }

const std::type_info* test3_typeid() { return &typeid(*fn()); }
// CHECK: define %struct.type_info* @"\01?test3_typeid@@YAPBUtype_info@@XZ"() #1 {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %call = tail call %struct.A* @"\01?fn@@YAPAUA@@XZ"() #3
// CHECK-NEXT:   %0 = icmp eq %struct.A* %call, null
// CHECK-NEXT:   br i1 %0, label %type_id.call, label %type_id.valid
// CHECK: type_id.valid:                                    ; preds = %entry
// CHECK-NEXT:   %1 = bitcast %struct.A* %call to i8*
// CHECK-NEXT:   %2 = bitcast %struct.A* %call to i8**
// CHECK-NEXT:   %vbtable = load i8** %2, align 4
// CHECK-NEXT:   %3 = getelementptr inbounds i8* %vbtable, i32 4
// CHECK-NEXT:   %4 = bitcast i8* %3 to i32*
// CHECK-NEXT:   %vbase_offs = load i32* %4, align 4
// CHECK-NEXT:   %5 = getelementptr inbounds i8* %1, i32 %vbase_offs
// CHECK-NEXT:   br label %type_id.call
// CHECK: type_id.call:                                     ; preds = %type_id.valid, %entry
// CHECK-NEXT:   %6 = phi i8* [ %5, %type_id.valid ], [ null, %entry ]
// CHECK-NEXT:   %7 = tail call i8* @__RTtypeid(i8* %6) #3
// CHECK-NEXT:   %8 = bitcast i8* %7 to %struct.type_info*
// CHECK-NEXT:   ret %struct.type_info* %8
// CHECK-NEXT: }

const std::type_info* test4_typeid() { return &typeid(b); }
// CHECK: define %struct.type_info* @"\01?test4_typeid@@YAPBUtype_info@@XZ"() #0 {
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret %struct.type_info* bitcast (%"MSRTTITypeDescriptor\02"* @"\01??_R0H@8" to %struct.type_info*)
// CHECK-NEXT: }
