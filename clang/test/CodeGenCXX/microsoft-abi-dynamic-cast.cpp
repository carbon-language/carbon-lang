// RUN: %clang_cc1 -emit-llvm -O2 -optzns -o - -triple=i386-pc-win32 2>/dev/null %s | FileCheck %s
// REQUIRES: asserts

struct S { char a; };
struct V { virtual void f(){} };
struct A : virtual V {};
struct B : S, virtual V {};
struct T {};

T* test0() { return dynamic_cast<T*>((B*)0); }
// CHECK: define noalias %struct.T* @"\01?test0@@YAPAUT@@XZ"() #0 {
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret %struct.T* null
// CHECK-NEXT: }

T* test1(V* x) { return &dynamic_cast<T&>(*x); }
// CHECK: define %struct.T* @"\01?test1@@YAPAUT@@PAUV@@@Z"(%struct.V* %x) #1 {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %0 = bitcast %struct.V* %x to i8*
// CHECK-NEXT:   %1 = tail call i8* @__RTDynamicCast(i8* %0, i32 0, i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUV@@@8" to i8*), i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUT@@@8" to i8*), i32 1) #2
// CHECK-NEXT:   %2 = bitcast i8* %1 to %struct.T*
// CHECK-NEXT:   ret %struct.T* %2
// CHECK-NEXT: }

T* test2(A* x) { return &dynamic_cast<T&>(*x); }
// CHECK: define %struct.T* @"\01?test2@@YAPAUT@@PAUA@@@Z"(%struct.A* %x) #1 {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %0 = bitcast %struct.A* %x to i8*
// CHECK-NEXT:   %1 = bitcast %struct.A* %x to i8**
// CHECK-NEXT:   %vbtable = load i8** %1, align 4
// CHECK-NEXT:   %2 = getelementptr inbounds i8* %vbtable, i32 4
// CHECK-NEXT:   %3 = bitcast i8* %2 to i32*
// CHECK-NEXT:   %vbase_offs = load i32* %3, align 4
// CHECK-NEXT:   %4 = getelementptr inbounds i8* %0, i32 %vbase_offs
// CHECK-NEXT:   %5 = tail call i8* @__RTDynamicCast(i8* %4, i32 %vbase_offs, i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUA@@@8" to i8*), i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUT@@@8" to i8*), i32 1) #2
// CHECK-NEXT:   %6 = bitcast i8* %5 to %struct.T*
// CHECK-NEXT:   ret %struct.T* %6
// CHECK-NEXT: }

T* test3(B* x) { return &dynamic_cast<T&>(*x); }
// CHECK: define %struct.T* @"\01?test3@@YAPAUT@@PAUB@@@Z"(%struct.B* %x) #1 {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %0 = getelementptr inbounds %struct.B* %x, i32 0, i32 0, i32 0
// CHECK-NEXT:   %vbptr = getelementptr inbounds i8* %0, i32 4
// CHECK-NEXT:   %1 = bitcast i8* %vbptr to i8**
// CHECK-NEXT:   %vbtable = load i8** %1, align 4
// CHECK-NEXT:   %2 = getelementptr inbounds i8* %vbtable, i32 4
// CHECK-NEXT:   %3 = bitcast i8* %2 to i32*
// CHECK-NEXT:   %vbase_offs = load i32* %3, align 4
// CHECK-NEXT:   %4 = add nsw i32 %vbase_offs, 4
// CHECK-NEXT:   %5 = getelementptr inbounds i8* %0, i32 %4
// CHECK-NEXT:   %6 = tail call i8* @__RTDynamicCast(i8* %5, i32 %4, i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUB@@@8" to i8*), i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUT@@@8" to i8*), i32 1) #2
// CHECK-NEXT:   %7 = bitcast i8* %6 to %struct.T*
// CHECK-NEXT:   ret %struct.T* %7
// CHECK-NEXT: }

T* test4(V* x) { return dynamic_cast<T*>(x); }
// CHECK: define %struct.T* @"\01?test4@@YAPAUT@@PAUV@@@Z"(%struct.V* %x) #1 {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %0 = bitcast %struct.V* %x to i8*
// CHECK-NEXT:   %1 = tail call i8* @__RTDynamicCast(i8* %0, i32 0, i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUV@@@8" to i8*), i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUT@@@8" to i8*), i32 0) #2
// CHECK-NEXT:   %2 = bitcast i8* %1 to %struct.T*
// CHECK-NEXT:   ret %struct.T* %2
// CHECK-NEXT: }

T* test5(A* x) { return dynamic_cast<T*>(x); }
// CHECK: define %struct.T* @"\01?test5@@YAPAUT@@PAUA@@@Z"(%struct.A* %x) #1 {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %0 = icmp eq %struct.A* %x, null
// CHECK-NEXT:   br i1 %0, label %dynamic_cast.call, label %dynamic_cast.valid
// CHECK: dynamic_cast.valid:                               ; preds = %entry
// CHECK-NEXT:   %1 = bitcast %struct.A* %x to i8*
// CHECK-NEXT:   %2 = bitcast %struct.A* %x to i8**
// CHECK-NEXT:   %vbtable = load i8** %2, align 4
// CHECK-NEXT:   %3 = getelementptr inbounds i8* %vbtable, i32 4
// CHECK-NEXT:   %4 = bitcast i8* %3 to i32*
// CHECK-NEXT:   %vbase_offs = load i32* %4, align 4
// CHECK-NEXT:   %5 = getelementptr inbounds i8* %1, i32 %vbase_offs
// CHECK-NEXT:   %6 = tail call i8* @__RTDynamicCast(i8* %5, i32 %vbase_offs, i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUA@@@8" to i8*), i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUT@@@8" to i8*), i32 0) #2
// CHECK-NEXT:   %phitmp = bitcast i8* %6 to %struct.T*
// CHECK-NEXT:   br label %dynamic_cast.call
// CHECK: dynamic_cast.call:                                ; preds = %dynamic_cast.valid, %entry
// CHECK-NEXT:   %7 = phi %struct.T* [ %phitmp, %dynamic_cast.valid ], [ null, %entry ]
// CHECK-NEXT:   ret %struct.T* %7
// CHECK-NEXT: }

T* test6(B* x) { return dynamic_cast<T*>(x); }
// CHECK: define %struct.T* @"\01?test6@@YAPAUT@@PAUB@@@Z"(%struct.B* %x) #1 {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %0 = icmp eq %struct.B* %x, null
// CHECK-NEXT:   br i1 %0, label %dynamic_cast.call, label %dynamic_cast.valid
// CHECK: dynamic_cast.valid:                               ; preds = %entry
// CHECK-NEXT:   %1 = getelementptr inbounds %struct.B* %x, i32 0, i32 0, i32 0
// CHECK-NEXT:   %vbptr = getelementptr inbounds i8* %1, i32 4
// CHECK-NEXT:   %2 = bitcast i8* %vbptr to i8**
// CHECK-NEXT:   %vbtable = load i8** %2, align 4
// CHECK-NEXT:   %3 = getelementptr inbounds i8* %vbtable, i32 4
// CHECK-NEXT:   %4 = bitcast i8* %3 to i32*
// CHECK-NEXT:   %vbase_offs = load i32* %4, align 4
// CHECK-NEXT:   %5 = add nsw i32 %vbase_offs, 4
// CHECK-NEXT:   %6 = getelementptr inbounds i8* %1, i32 %5
// CHECK-NEXT:   %7 = tail call i8* @__RTDynamicCast(i8* %6, i32 %5, i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUB@@@8" to i8*), i8* bitcast (%"MSRTTITypeDescriptor\07"* @"\01??_R0?AUT@@@8" to i8*), i32 0) #2
// CHECK-NEXT:   %phitmp = bitcast i8* %7 to %struct.T*
// CHECK-NEXT:   br label %dynamic_cast.call
// CHECK: dynamic_cast.call:                                ; preds = %dynamic_cast.valid, %entry
// CHECK-NEXT:   %8 = phi %struct.T* [ %phitmp, %dynamic_cast.valid ], [ null, %entry ]
// CHECK-NEXT:   ret %struct.T* %8
// CHECK-NEXT: }

void* test7(V* x) { return dynamic_cast<void*>(x); }
// CHECK: define i8* @"\01?test7@@YAPAXPAUV@@@Z"(%struct.V* %x) #1 {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %0 = bitcast %struct.V* %x to i8*
// CHECK-NEXT:   %1 = tail call i8* @__RTCastToVoid(i8* %0) #2
// CHECK-NEXT:   ret i8* %1
// CHECK-NEXT: }

void* test8(A* x) { return dynamic_cast<void*>(x); }
// CHECK: define i8* @"\01?test8@@YAPAXPAUA@@@Z"(%struct.A* %x) #1 {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %0 = icmp eq %struct.A* %x, null
// CHECK-NEXT:   br i1 %0, label %dynamic_cast.call, label %dynamic_cast.valid
// CHECK: dynamic_cast.valid:                               ; preds = %entry
// CHECK-NEXT:   %1 = bitcast %struct.A* %x to i8*
// CHECK-NEXT:   %2 = bitcast %struct.A* %x to i8**
// CHECK-NEXT:   %vbtable = load i8** %2, align 4
// CHECK-NEXT:   %3 = getelementptr inbounds i8* %vbtable, i32 4
// CHECK-NEXT:   %4 = bitcast i8* %3 to i32*
// CHECK-NEXT:   %vbase_offs = load i32* %4, align 4
// CHECK-NEXT:   %5 = getelementptr inbounds i8* %1, i32 %vbase_offs
// CHECK-NEXT:   %6 = tail call i8* @__RTCastToVoid(i8* %5) #2
// CHECK-NEXT:   br label %dynamic_cast.call
// CHECK: dynamic_cast.call:                                ; preds = %dynamic_cast.valid, %entry
// CHECK-NEXT:   %7 = phi i8* [ %6, %dynamic_cast.valid ], [ null, %entry ]
// CHECK-NEXT:   ret i8* %7
// CHECK-NEXT: }

void* test9(B* x) { return dynamic_cast<void*>(x); }
// CHECK: define i8* @"\01?test9@@YAPAXPAUB@@@Z"(%struct.B* %x) #1 {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %0 = icmp eq %struct.B* %x, null
// CHECK-NEXT:   br i1 %0, label %dynamic_cast.call, label %dynamic_cast.valid
// CHECK: dynamic_cast.valid:                               ; preds = %entry
// CHECK-NEXT:   %1 = getelementptr inbounds %struct.B* %x, i32 0, i32 0, i32 0
// CHECK-NEXT:   %vbptr = getelementptr inbounds i8* %1, i32 4
// CHECK-NEXT:   %2 = bitcast i8* %vbptr to i8**
// CHECK-NEXT:   %vbtable = load i8** %2, align 4
// CHECK-NEXT:   %3 = getelementptr inbounds i8* %vbtable, i32 4
// CHECK-NEXT:   %4 = bitcast i8* %3 to i32*
// CHECK-NEXT:   %vbase_offs = load i32* %4, align 4
// CHECK-NEXT:   %5 = add nsw i32 %vbase_offs, 4
// CHECK-NEXT:   %6 = getelementptr inbounds i8* %1, i32 %5
// CHECK-NEXT:   %7 = tail call i8* @__RTCastToVoid(i8* %6) #2
// CHECK-NEXT:   br label %dynamic_cast.call
// CHECK: dynamic_cast.call:                                ; preds = %dynamic_cast.valid, %entry
// CHECK-NEXT:   %8 = phi i8* [ %7, %dynamic_cast.valid ], [ null, %entry ]
// CHECK-NEXT:   ret i8* %8
// CHECK-NEXT: }
