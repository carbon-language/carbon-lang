// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple=x86_64-pc-windows-msvc -mconstructor-aliases -fexceptions -fcxx-exceptions | FileCheck -check-prefix WIN64 %s

extern "C" void might_throw();

// Simplify the generated IR with noexcept.
extern "C" void recover() noexcept(true);
extern "C" void handle_exception(void *e) noexcept(true);

extern "C" void catch_all() {
  try {
    might_throw();
  } catch (...) {
    recover();
  }
}

// WIN64-LABEL: define void @catch_all()
// WIN64: invoke void @might_throw()
// WIN64-NEXT: to label %[[cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// WIN64: [[cont]]
// WIN64: br label %[[ret:[^ ]*]]
//
// WIN64: [[lpad]]
// WIN64: landingpad { i8*, i32 }
// WIN64-NEXT: catch i8* null
// WIN64: call void @llvm.eh.begincatch(i8* %{{[^,]*}}, i8* null)
// WIN64: call void @recover()
// WIN64: call void @llvm.eh.endcatch()
// WIN64: br label %[[ret]]
//
// WIN64: [[ret]]
// WIN64: ret void

extern "C" void catch_int() {
  try {
    might_throw();
  } catch (int e) {
    handle_exception(&e);
  }
}

// WIN64-LABEL: define void @catch_int()
// WIN64: landingpad { i8*, i32 }
// WIN64: %[[e_i8:[^ ]*]] = bitcast i32* %[[e_addr:[^ ]*]] to i8*
// WIN64: call void @llvm.eh.begincatch(i8* %{{.*}}, i8* %[[e_i8]])
// WIN64: %[[e_i8:[^ ]*]] = bitcast i32* %[[e_addr]] to i8*
// WIN64: call void @handle_exception(i8* %[[e_i8]])
// WIN64: call void @llvm.eh.endcatch()

extern "C" void catch_int_unnamed() {
  try {
    might_throw();
  } catch (int) {
  }
}

// WIN64-LABEL: define void @catch_int_unnamed()
// WIN64: landingpad { i8*, i32 }
// WIN64: call void @llvm.eh.begincatch(i8* %{{.*}}, i8* null)
// WIN64: call void @llvm.eh.endcatch()

struct A {
  A();
  A(const A &o);
  ~A();
  int a;
};

struct B : A {
  B();
  B(const B &o);
  ~B();
  int b;
};

extern "C" void catch_a_byval() {
  try {
    might_throw();
  } catch (A e) {
    handle_exception(&e);
  }
}

// WIN64-LABEL: define void @catch_a_byval()
// WIN64: %[[e_addr:[^ ]*]] = alloca %struct.A
// WIN64: landingpad { i8*, i32 }
// WIN64: %[[e_i8:[^ ]*]] = bitcast %struct.A* %[[e_addr]] to i8*
// WIN64: call void @llvm.eh.begincatch(i8* %{{.*}}, i8* %[[e_i8]])
// WIN64: %[[e_i8:[^ ]*]] = bitcast %struct.A* %[[e_addr]] to i8*
// WIN64: call void @handle_exception(i8* %[[e_i8]])
// WIN64: call void @llvm.eh.endcatch()

extern "C" void catch_a_ref() {
  try {
    might_throw();
  } catch (A &e) {
    handle_exception(&e);
  }
}

// WIN64-LABEL: define void @catch_a_ref()
// WIN64: %[[e_addr:[^ ]*]] = alloca %struct.A*
// WIN64: landingpad { i8*, i32 }
// WIN64: %[[e_i8:[^ ]*]] = bitcast %struct.A** %[[e_addr]] to i8*
// WIN64: call void @llvm.eh.begincatch(i8* %{{.*}}, i8* %[[e_i8]])
// WIN64: %[[eptr:[^ ]*]] = load %struct.A*, %struct.A** %[[e_addr]]
// WIN64: %[[eptr_i8:[^ ]*]] = bitcast %struct.A* %[[eptr]] to i8*
// WIN64: call void @handle_exception(i8* %[[eptr_i8]])
// WIN64: call void @llvm.eh.endcatch()

extern "C" void fn_with_exc_spec() throw(int) {
  might_throw();
}

// WIN64-LABEL: define void @fn_with_exc_spec()
// WIN64: call void @might_throw()
// WIN64-NEXT: ret void

extern "C" void catch_nested() {
  try {
    might_throw();
  } catch (int) {
    try {
      might_throw();
    } catch (int) {
      might_throw();
    }
  }
}

// WIN64-LABEL: define void @catch_nested()
// WIN64: invoke void @might_throw()
// WIN64-NEXT: to label %[[cont1:[^ ]*]] unwind label %[[lp1:[^ ]*]]
// WIN64: [[cont1]]
//
// WIN64: [[lp1]]
// WIN64: landingpad { i8*, i32 }
// WIN64: call void @llvm.eh.begincatch(i8* %{{.*}}, i8* null)
// WIN64: invoke void @might_throw()
// WIN64-NEXT: to label %[[cont2:[^ ]*]] unwind label %[[lp2:[^ ]*]]
//
// WIN64: [[cont2]]
// WIN64-NEXT: br label %[[trycont:[^ ]*]]
//
// WIN64: [[lp2]]
// WIN64: landingpad { i8*, i32 }
// WIN64: call void @llvm.eh.begincatch(i8* %{{.*}}, i8* null)
// WIN64-NEXT: call void @might_throw()
// WIN64-NEXT: call void @llvm.eh.endcatch()
// WIN64-NEXT: br label %[[trycont]]
//
// WIN64: [[trycont]]
// WIN64: call void @llvm.eh.endcatch()
