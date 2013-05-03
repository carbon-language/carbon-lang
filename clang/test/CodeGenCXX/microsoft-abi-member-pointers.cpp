// RUN: %clang_cc1 -fno-rtti -emit-llvm %s -o - -cxx-abi microsoft -triple=i386-pc-win32 | FileCheck %s

struct B1 {
  void foo();
  int b;
};
struct B2 {
  void foo();
};
struct Single : B1 {
  void foo();
};
struct Multiple : B1, B2 {
  void foo();
};
struct Virtual : virtual B1 {
  int v;
  void foo();
};

struct POD {
  int a;
  int b;
};

struct Polymorphic {
  virtual void myVirtual();
  int a;
  int b;
};

// This class uses the virtual inheritance model, yet its vbptr offset is not 0.
// We still use zero for the null field offset, despite it being a valid field
// offset.
struct NonZeroVBPtr : POD, Virtual {
  int n;
};

struct Unspecified;

// Check that we can lower the LLVM types and get the null initializers right.
int Single     ::*s_d_memptr;
int Polymorphic::*p_d_memptr;
int Multiple   ::*m_d_memptr;
int Virtual    ::*v_d_memptr;
int NonZeroVBPtr::*n_d_memptr;
int Unspecified::*u_d_memptr;
// CHECK: @"\01?s_d_memptr@@3PQSingle@@HA" = global i32 -1, align 4
// CHECK: @"\01?p_d_memptr@@3PQPolymorphic@@HA" = global i32 0, align 4
// CHECK: @"\01?m_d_memptr@@3PQMultiple@@HA" = global i32 -1, align 4
// CHECK: @"\01?v_d_memptr@@3PQVirtual@@HA" = global { i32, i32 }
// CHECK:   { i32 0, i32 -1 }, align 4
// CHECK: @"\01?n_d_memptr@@3PQNonZeroVBPtr@@HA" = global { i32, i32 }
// CHECK:   { i32 0, i32 -1 }, align 4
// CHECK: @"\01?u_d_memptr@@3PQUnspecified@@HA" = global { i32, i32, i32 }
// CHECK:   { i32 0, i32 0, i32 -1 }, align 4

void (Single  ::*s_f_memptr)();
void (Multiple::*m_f_memptr)();
void (Virtual ::*v_f_memptr)();
// CHECK: @"\01?s_f_memptr@@3P8Single@@AEXXZA" = global i8* null, align 4
// CHECK: @"\01?m_f_memptr@@3P8Multiple@@AEXXZA" = global { i8*, i32 } zeroinitializer, align 4
// CHECK: @"\01?v_f_memptr@@3P8Virtual@@AEXXZA" = global { i8*, i32, i32 } zeroinitializer, align 4

// We can define Unspecified after locking in the inheritance model.
struct Unspecified : Virtual {
  void foo();
  int u;
};

struct UnspecWithVBPtr;
int UnspecWithVBPtr::*forceUnspecWithVBPtr;
struct UnspecWithVBPtr : B1, virtual B2 {
  int u;
  void foo();
};

// Test emitting non-virtual member pointers in a non-constexpr setting.
void EmitNonVirtualMemberPointers() {
  void (Single     ::*s_f_memptr)() = &Single::foo;
  void (Multiple   ::*m_f_memptr)() = &Multiple::foo;
  void (Virtual    ::*v_f_memptr)() = &Virtual::foo;
  void (Unspecified::*u_f_memptr)() = &Unspecified::foo;
  void (UnspecWithVBPtr::*u2_f_memptr)() = &UnspecWithVBPtr::foo;
// CHECK: define void @"\01?EmitNonVirtualMemberPointers@@YAXXZ"() #0 {
// CHECK:   alloca i8*, align 4
// CHECK:   alloca { i8*, i32 }, align 4
// CHECK:   alloca { i8*, i32, i32 }, align 4
// CHECK:   alloca { i8*, i32, i32, i32 }, align 4
// CHECK:   store i8* bitcast (void (%{{.*}}*)* @"\01?foo@Single@@QAEXXZ" to i8*), i8** %{{.*}}, align 4
// CHECK:   store { i8*, i32 }
// CHECK:     { i8* bitcast (void (%{{.*}}*)* @"\01?foo@Multiple@@QAEXXZ" to i8*), i32 0 },
// CHECK:     { i8*, i32 }* %{{.*}}, align 4
// CHECK:   store { i8*, i32, i32 }
// CHECK:     { i8* bitcast (void (%{{.*}}*)* @"\01?foo@Virtual@@QAEXXZ" to i8*), i32 0, i32 0 },
// CHECK:     { i8*, i32, i32 }* %{{.*}}, align 4
// CHECK:   store { i8*, i32, i32, i32 }
// CHECK:     { i8* bitcast (void (%{{.*}}*)* @"\01?foo@Unspecified@@QAEXXZ" to i8*), i32 0, i32 0, i32 0 },
// CHECK:     { i8*, i32, i32, i32 }* %{{.*}}, align 4
// CHECK:   store { i8*, i32, i32, i32 }
// CHECK:     { i8* bitcast (void (%{{.*}}*)* @"\01?foo@UnspecWithVBPtr@@QAEXXZ" to i8*),
// CHECK:       i32 0, i32 4, i32 0 },
// CHECK:     { i8*, i32, i32, i32 }* %{{.*}}, align 4
// CHECK:   ret void
// CHECK: }
}

void podMemPtrs() {
  int POD::*memptr;
  memptr = &POD::a;
  memptr = &POD::b;
  if (memptr)
    memptr = 0;
// Check that member pointers use the right offsets and that null is -1.
// CHECK:      define void @"\01?podMemPtrs@@YAXXZ"() #0 {
// CHECK:        %[[memptr:.*]] = alloca i32, align 4
// CHECK-NEXT:   store i32 0, i32* %[[memptr]], align 4
// CHECK-NEXT:   store i32 4, i32* %[[memptr]], align 4
// CHECK-NEXT:   %[[memptr_val:.*]] = load i32* %[[memptr]], align 4
// CHECK-NEXT:   %{{.*}} = icmp ne i32 %[[memptr_val]], -1
// CHECK-NEXT:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
// CHECK:        store i32 -1, i32* %[[memptr]], align 4
// CHECK:        ret void
// CHECK:      }
}

void polymorphicMemPtrs() {
  int Polymorphic::*memptr;
  memptr = &Polymorphic::a;
  memptr = &Polymorphic::b;
  if (memptr)
    memptr = 0;
// Member pointers for polymorphic classes include the vtable slot in their
// offset and use 0 to represent null.
// CHECK:      define void @"\01?polymorphicMemPtrs@@YAXXZ"() #0 {
// CHECK:        %[[memptr:.*]] = alloca i32, align 4
// CHECK-NEXT:   store i32 4, i32* %[[memptr]], align 4
// CHECK-NEXT:   store i32 8, i32* %[[memptr]], align 4
// CHECK-NEXT:   %[[memptr_val:.*]] = load i32* %[[memptr]], align 4
// CHECK-NEXT:   %{{.*}} = icmp ne i32 %[[memptr_val]], 0
// CHECK-NEXT:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
// CHECK:        store i32 0, i32* %[[memptr]], align 4
// CHECK:        ret void
// CHECK:      }
}

bool nullTestDataUnspecified(int Unspecified::*mp) {
  return mp;
// CHECK: define zeroext i1 @"\01?nullTestDataUnspecified@@YA_NPQUnspecified@@H@Z"{{.*}} {
// CHECK:   %{{.*}} = load { i32, i32, i32 }* %{{.*}}, align 4
// CHECK:   store { i32, i32, i32 } {{.*}} align 4
// CHECK:   %[[mp:.*]] = load { i32, i32, i32 }* %{{.*}}, align 4
// CHECK:   %[[mp0:.*]] = extractvalue { i32, i32, i32 } %[[mp]], 0
// CHECK:   %[[cmp0:.*]] = icmp ne i32 %[[mp0]], 0
// CHECK:   %[[mp1:.*]] = extractvalue { i32, i32, i32 } %[[mp]], 1
// CHECK:   %[[cmp1:.*]] = icmp ne i32 %[[mp1]], 0
// CHECK:   %[[and0:.*]] = and i1 %[[cmp0]], %[[cmp1]]
// CHECK:   %[[mp2:.*]] = extractvalue { i32, i32, i32 } %[[mp]], 2
// CHECK:   %[[cmp2:.*]] = icmp ne i32 %[[mp2]], -1
// CHECK:   %[[and1:.*]] = and i1 %[[and0]], %[[cmp2]]
// CHECK:   ret i1 %[[and1]]
// CHECK: }
}

bool nullTestFunctionUnspecified(void (Unspecified::*mp)()) {
  return mp;
// CHECK: define zeroext i1 @"\01?nullTestFunctionUnspecified@@YA_NP8Unspecified@@AEXXZ@Z"{{.*}} {
// CHECK:   %{{.*}} = load { i8*, i32, i32, i32 }* %{{.*}}, align 4
// CHECK:   store { i8*, i32, i32, i32 } {{.*}} align 4
// CHECK:   %[[mp:.*]] = load { i8*, i32, i32, i32 }* %{{.*}}, align 4
// CHECK:   %[[mp0:.*]] = extractvalue { i8*, i32, i32, i32 } %[[mp]], 0
// CHECK:   %[[cmp0:.*]] = icmp ne i8* %[[mp0]], null
// CHECK:   ret i1 %[[cmp0]]
// CHECK: }
}

int loadDataMemberPointerVirtual(Virtual *o, int Virtual::*memptr) {
  return o->*memptr;
// Test that we can unpack this aggregate member pointer and load the member
// data pointer.
// CHECK: define i32 @"\01?loadDataMemberPointerVirtual@@YAHPAUVirtual@@PQ1@H@Z"{{.*}} {
// CHECK:   %[[o:.*]] = load %{{.*}}** %{{.*}}, align 4
// CHECK:   %[[memptr:.*]] = load { i32, i32 }* %{{.*}}, align 4
// CHECK:   %[[memptr0:.*]] = extractvalue { i32, i32 } %[[memptr:.*]], 0
// CHECK:   %[[memptr1:.*]] = extractvalue { i32, i32 } %[[memptr:.*]], 1
// CHECK:   %[[v6:.*]] = bitcast %{{.*}}* %[[o]] to i8*
// CHECK:   %[[vbptr:.*]] = getelementptr inbounds i8* %[[v6]], i32 0
// CHECK:   %[[vbptr_a:.*]] = bitcast i8* %[[vbptr]] to i8**
// CHECK:   %[[vbtable:.*]] = load i8** %[[vbptr_a:.*]]
// CHECK:   %[[v7:.*]] = getelementptr inbounds i8* %[[vbtable]], i32 %[[memptr1]]
// CHECK:   %[[v8:.*]] = bitcast i8* %[[v7]] to i32*
// CHECK:   %[[vbase_offs:.*]] = load i32* %[[v8]]
// CHECK:   %[[v10:.*]] = getelementptr inbounds i8* %[[vbptr]], i32 %[[vbase_offs]]
// CHECK:   %[[offset:.*]] = getelementptr inbounds i8* %[[v10]], i32 %[[memptr0]]
// CHECK:   %[[v11:.*]] = bitcast i8* %[[offset]] to i32*
// CHECK:   %[[v12:.*]] = load i32* %[[v11]]
// CHECK:   ret i32 %[[v12]]
// CHECK: }
}

int loadDataMemberPointerUnspecified(Unspecified *o, int Unspecified::*memptr) {
  return o->*memptr;
// Test that we can unpack this aggregate member pointer and load the member
// data pointer.
// CHECK: define i32 @"\01?loadDataMemberPointerUnspecified@@YAHPAUUnspecified@@PQ1@H@Z"{{.*}} {
// CHECK:   %[[o:.*]] = load %{{.*}}** %{{.*}}, align 4
// CHECK:   %[[memptr:.*]] = load { i32, i32, i32 }* %{{.*}}, align 4
// CHECK:   %[[memptr0:.*]] = extractvalue { i32, i32, i32 } %[[memptr:.*]], 0
// CHECK:   %[[memptr1:.*]] = extractvalue { i32, i32, i32 } %[[memptr:.*]], 1
// CHECK:   %[[memptr2:.*]] = extractvalue { i32, i32, i32 } %[[memptr:.*]], 2
// CHECK:   %[[base:.*]] = bitcast %{{.*}}* %[[o]] to i8*
// CHECK:   %[[is_vbase:.*]] = icmp ne i32 %[[memptr2]], 0
// CHECK:   br i1 %[[is_vbase]], label %[[vadjust:.*]], label %[[skip:.*]]
//
// CHECK: [[vadjust]]
// CHECK:   %[[vbptr:.*]] = getelementptr inbounds i8* %[[base]], i32 %[[memptr1]]
// CHECK:   %[[vbptr_a:.*]] = bitcast i8* %[[vbptr]] to i8**
// CHECK:   %[[vbtable:.*]] = load i8** %[[vbptr_a:.*]]
// CHECK:   %[[v7:.*]] = getelementptr inbounds i8* %[[vbtable]], i32 %[[memptr2]]
// CHECK:   %[[v8:.*]] = bitcast i8* %[[v7]] to i32*
// CHECK:   %[[vbase_offs:.*]] = load i32* %[[v8]]
// CHECK:   %[[base_adj:.*]] = getelementptr inbounds i8* %[[vbptr]], i32 %[[vbase_offs]]
//
// CHECK: [[skip]]
// CHECK:   %[[new_base:.*]] = phi i8* [ %[[base]], %{{.*}} ], [ %[[base_adj]], %[[vadjust]] ]
// CHECK:   %[[offset:.*]] = getelementptr inbounds i8* %[[new_base]], i32 %[[memptr0]]
// CHECK:   %[[v11:.*]] = bitcast i8* %[[offset]] to i32*
// CHECK:   %[[v12:.*]] = load i32* %[[v11]]
// CHECK:   ret i32 %[[v12]]
// CHECK: }
}

void callMemberPointerSingle(Single *o, void (Single::*memptr)()) {
  (o->*memptr)();
// Just look for an indirect thiscall.
// CHECK: define void @"\01?callMemberPointerSingle@@{{.*}} #0 {
// CHECK:   call x86_thiscallcc void %{{.*}}(%{{.*}} %{{.*}})
// CHECK:   ret void
// CHECK: }
}

void callMemberPointerMultiple(Multiple *o, void (Multiple::*memptr)()) {
  (o->*memptr)();
// CHECK: define void @"\01?callMemberPointerMultiple@@{{.*}} #0 {
// CHECK:   %[[memptr0:.*]] = extractvalue { i8*, i32 } %{{.*}}, 0
// CHECK:   %[[memptr1:.*]] = extractvalue { i8*, i32 } %{{.*}}, 1
// CHECK:   %[[this_adjusted:.*]] = getelementptr inbounds i8* %{{.*}}, i32 %[[memptr1]]
// CHECK:   %[[this:.*]] = bitcast i8* %[[this_adjusted]] to {{.*}}
// CHECK:   %[[fptr:.*]] = bitcast i8* %[[memptr0]] to {{.*}}
// CHECK:   call x86_thiscallcc void %[[fptr]](%{{.*}} %[[this]])
// CHECK:   ret void
// CHECK: }
}

void callMemberPointerVirtualBase(Virtual *o, void (Virtual::*memptr)()) {
  (o->*memptr)();
// This shares a lot with virtual data member pointers.
// CHECK: define void @"\01?callMemberPointerVirtualBase@@{{.*}} #0 {
// CHECK:   %[[memptr0:.*]] = extractvalue { i8*, i32, i32 } %{{.*}}, 0
// CHECK:   %[[memptr1:.*]] = extractvalue { i8*, i32, i32 } %{{.*}}, 1
// CHECK:   %[[memptr2:.*]] = extractvalue { i8*, i32, i32 } %{{.*}}, 2
// CHECK:   %[[vbptr:.*]] = getelementptr inbounds i8* %{{.*}}, i32 0
// CHECK:   %[[vbptr_a:.*]] = bitcast i8* %[[vbptr]] to i8**
// CHECK:   %[[vbtable:.*]] = load i8** %[[vbptr_a:.*]]
// CHECK:   %[[v7:.*]] = getelementptr inbounds i8* %[[vbtable]], i32 %[[memptr2]]
// CHECK:   %[[v8:.*]] = bitcast i8* %[[v7]] to i32*
// CHECK:   %[[vbase_offs:.*]] = load i32* %[[v8]]
// CHECK:   %[[v10:.*]] = getelementptr inbounds i8* %[[vbptr]], i32 %[[vbase_offs]]
// CHECK:   %[[this_adjusted:.*]] = getelementptr inbounds i8* %[[v10]], i32 %[[memptr1]]
// CHECK:   %[[fptr:.*]] = bitcast i8* %[[memptr0]] to void ({{.*}})
// CHECK:   %[[this:.*]] = bitcast i8* %[[this_adjusted]] to {{.*}}
// CHECK:   call x86_thiscallcc void %[[fptr]](%{{.*}} %[[this]])
// CHECK:   ret void
// CHECK: }
}

bool compareSingleFunctionMemptr(void (Single::*l)(), void (Single::*r)()) {
  return l == r;
// Should only be one comparison here.
// CHECK: define zeroext i1 @"\01?compareSingleFunctionMemptr@@YA_NP8Single@@AEXXZ0@Z"{{.*}} {
// CHECK-NOT: icmp
// CHECK:   %[[r:.*]] = icmp eq
// CHECK-NOT: icmp
// CHECK:   ret i1 %[[r]]
// CHECK: }
}

bool compareNeqSingleFunctionMemptr(void (Single::*l)(), void (Single::*r)()) {
  return l != r;
// Should only be one comparison here.
// CHECK: define zeroext i1 @"\01?compareNeqSingleFunctionMemptr@@YA_NP8Single@@AEXXZ0@Z"{{.*}} {
// CHECK-NOT: icmp
// CHECK:   %[[r:.*]] = icmp ne
// CHECK-NOT: icmp
// CHECK:   ret i1 %[[r]]
// CHECK: }
}

bool unspecFuncMemptrEq(void (Unspecified::*l)(), void (Unspecified::*r)()) {
  return l == r;
// CHECK: define zeroext i1 @"\01?unspecFuncMemptrEq@@YA_NP8Unspecified@@AEXXZ0@Z"{{.*}} {
// CHECK:   %[[lhs0:.*]] = extractvalue { i8*, i32, i32, i32 } %[[l:.*]], 0
// CHECK:   %{{.*}} = extractvalue { i8*, i32, i32, i32 } %[[r:.*]], 0
// CHECK:   %[[cmp0:.*]] = icmp eq i8* %[[lhs0]], %{{.*}}
// CHECK:   %{{.*}} = extractvalue { i8*, i32, i32, i32 } %[[l]], 1
// CHECK:   %{{.*}} = extractvalue { i8*, i32, i32, i32 } %[[r]], 1
// CHECK:   %[[cmp1:.*]] = icmp eq i32
// CHECK:   %{{.*}} = extractvalue { i8*, i32, i32, i32 } %[[l]], 2
// CHECK:   %{{.*}} = extractvalue { i8*, i32, i32, i32 } %[[r]], 2
// CHECK:   %[[cmp2:.*]] = icmp eq i32
// CHECK:   %[[res12:.*]] = and i1 %[[cmp1]], %[[cmp2]]
// CHECK:   %{{.*}} = extractvalue { i8*, i32, i32, i32 } %[[l]], 3
// CHECK:   %{{.*}} = extractvalue { i8*, i32, i32, i32 } %[[r]], 3
// CHECK:   %[[cmp3:.*]] = icmp eq i32
// CHECK:   %[[res123:.*]] = and i1 %[[res12]], %[[cmp3]]
// CHECK:   %[[iszero:.*]] = icmp eq i8* %[[lhs0]], null
// CHECK:   %[[bits_or_null:.*]] = or i1 %[[res123]], %[[iszero]]
// CHECK:   %{{.*}} = and i1 %[[bits_or_null]], %[[cmp0]]
// CHECK:   ret i1 %{{.*}}
// CHECK: }
}

bool unspecFuncMemptrNeq(void (Unspecified::*l)(), void (Unspecified::*r)()) {
  return l != r;
// CHECK: define zeroext i1 @"\01?unspecFuncMemptrNeq@@YA_NP8Unspecified@@AEXXZ0@Z"{{.*}} {
// CHECK:   %[[lhs0:.*]] = extractvalue { i8*, i32, i32, i32 } %[[l:.*]], 0
// CHECK:   %{{.*}} = extractvalue { i8*, i32, i32, i32 } %[[r:.*]], 0
// CHECK:   %[[cmp0:.*]] = icmp ne i8* %[[lhs0]], %{{.*}}
// CHECK:   %{{.*}} = extractvalue { i8*, i32, i32, i32 } %[[l]], 1
// CHECK:   %{{.*}} = extractvalue { i8*, i32, i32, i32 } %[[r]], 1
// CHECK:   %[[cmp1:.*]] = icmp ne i32
// CHECK:   %{{.*}} = extractvalue { i8*, i32, i32, i32 } %[[l]], 2
// CHECK:   %{{.*}} = extractvalue { i8*, i32, i32, i32 } %[[r]], 2
// CHECK:   %[[cmp2:.*]] = icmp ne i32
// CHECK:   %[[res12:.*]] = or i1 %[[cmp1]], %[[cmp2]]
// CHECK:   %{{.*}} = extractvalue { i8*, i32, i32, i32 } %[[l]], 3
// CHECK:   %{{.*}} = extractvalue { i8*, i32, i32, i32 } %[[r]], 3
// CHECK:   %[[cmp3:.*]] = icmp ne i32
// CHECK:   %[[res123:.*]] = or i1 %[[res12]], %[[cmp3]]
// CHECK:   %[[iszero:.*]] = icmp ne i8* %[[lhs0]], null
// CHECK:   %[[bits_or_null:.*]] = and i1 %[[res123]], %[[iszero]]
// CHECK:   %{{.*}} = or i1 %[[bits_or_null]], %[[cmp0]]
// CHECK:   ret i1 %{{.*}}
// CHECK: }
}

bool unspecDataMemptrEq(int Unspecified::*l, int Unspecified::*r) {
  return l == r;
// CHECK: define zeroext i1 @"\01?unspecDataMemptrEq@@YA_NPQUnspecified@@H0@Z"{{.*}} {
// CHECK:   extractvalue { i32, i32, i32 } %{{.*}}, 0
// CHECK:   extractvalue { i32, i32, i32 } %{{.*}}, 0
// CHECK:   icmp eq i32
// CHECK:   extractvalue { i32, i32, i32 } %{{.*}}, 1
// CHECK:   extractvalue { i32, i32, i32 } %{{.*}}, 1
// CHECK:   icmp eq i32
// CHECK:   extractvalue { i32, i32, i32 } %{{.*}}, 2
// CHECK:   extractvalue { i32, i32, i32 } %{{.*}}, 2
// CHECK:   icmp eq i32
// CHECK:   and i1
// CHECK:   and i1
// CHECK:   ret i1
// CHECK: }
}
