// RUN: %clang_cc1 %s -triple x86_64-pc-win32 -fms-extensions -emit-llvm -o - | FileCheck %s

// FIXME: Perform this outlining automatically CodeGen.
void try_body(int numerator, int denominator, int *myres) {
  *myres = numerator / denominator;
}
// CHECK-LABEL: define void @try_body(i32 %numerator, i32 %denominator, i32* %myres)
// CHECK: sdiv i32
// CHECK: store i32 %{{.*}}, i32*
// CHECK: ret void

int safe_div(int numerator, int denominator, int *res) {
  int myres = 0;
  int success = 1;
  __try {
    try_body(numerator, denominator, &myres);
  } __except (1) {
    success = -42;
  }
  *res = myres;
  return success;
}
// CHECK-LABEL: define i32 @safe_div(i32 %numerator, i32 %denominator, i32* %res)
// CHECK: invoke void @try_body(i32 %{{.*}}, i32 %{{.*}}, i32* %{{.*}}) #[[NOINLINE:[0-9]+]]
// CHECK:       to label %{{.*}} unwind label %[[lpad:[^ ]*]]
//
// CHECK: [[lpad]]
// CHECK: landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*)
// CHECK-NEXT: catch i8* null
// CHECK-NOT: br i1
// CHECK: br label %[[except:[^ ]*]]
// CHECK: [[except]]
// CHECK-NEXT: store i32 -42, i32* %[[success:[^ ]*]]
//
// CHECK: %[[res:[^ ]*]] = load i32* %[[success]]
// CHECK: ret i32 %[[res]]

void j(void);

// FIXME: Implement local variable captures in filter expressions.
int filter_expr_capture(void) {
  int r = 42;
  __try {
    j();
  } __except(/*r =*/ -1) {
    r = 13;
  }
  return r;
}

// CHECK-LABEL: define i32 @filter_expr_capture()
// FIXMECHECK: %[[captures]] = call i8* @llvm.frameallocate(i32 4)
// CHECK: store i32 42, i32* %[[r:[^ ,]*]]
// CHECK: invoke void @j() #[[NOINLINE]]
//
// CHECK: landingpad
// CHECK-NEXT: catch i8* bitcast (i32 (i8*, i8*)* @"\01?filt$0@0@filter_expr_capture@@" to i8*)
// CHECK: store i32 13, i32* %[[r]]
//
// CHECK: %[[rv:[^ ]*]] = load i32* %[[r]]
// CHECK: ret i32 %[[rv]]

// CHECK-LABEL: define internal i32 @"\01?filt$0@0@filter_expr_capture@@"(i8* %exception_pointers, i8* %frame_pointer)
// FIXMECHECK: %[[captures]] = call i8* @llvm.framerecover(i8* bitcast (i32 ()* @filter_expr_capture, i8* %frame_pointer)
// FIXMECHECK: store i32 -1, i32* %{{.*}}
// CHECK: ret i32 -1

int nested_try(void) {
  int r = 42;
  __try {
    __try {
      j();
      r = 0;
    } __except(_exception_code() == 123) {
      r = 123;
    }
  } __except(_exception_code() == 456) {
    r = 456;
  }
  return r;
}
// CHECK-LABEL: define i32 @nested_try()
// CHECK: store i32 42, i32* %[[r:[^ ,]*]]
// CHECK: invoke void @j() #[[NOINLINE]]
// CHECK:       to label %[[cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CHECK: [[cont]]
// CHECK: store i32 0, i32* %[[r]]
// CHECK: br label %[[inner_try_cont:[^ ]*]]
//
// CHECK: [[lpad]]
// CHECK: landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*)
// CHECK: catch i8* bitcast (i32 (i8*, i8*)* @"\01?filt$1@0@nested_try@@" to i8*)
// CHECK: catch i8* bitcast (i32 (i8*, i8*)* @"\01?filt$0@0@nested_try@@" to i8*)
// CHECK: store i8* %{{.*}}, i8** %[[ehptr_slot:[^ ]*]]
// CHECK: store i32 %{{.*}}, i32* %[[sel_slot:[^ ]*]]
//
// CHECK: load i32* %[[sel_slot]]
// CHECK: call i32 @llvm.eh.typeid.for(i8* bitcast (i32 (i8*, i8*)* @"\01?filt$1@0@nested_try@@" to i8*))
// CHECK: icmp eq i32
// CHECK: br i1
//
// CHECK: load i32* %[[sel_slot]]
// CHECK: call i32 @llvm.eh.typeid.for(i8* bitcast (i32 (i8*, i8*)* @"\01?filt$0@0@nested_try@@" to i8*))
// CHECK: icmp eq i32
// CHECK: br i1
//
// CHECK: store i32 456, i32* %[[r]]
// CHECK: br label %[[outer_try_cont:[^ ]*]]
//
// CHECK: [[outer_try_cont]]
// CHECK: %[[r_load:[^ ]*]] = load i32* %[[r]]
// CHECK: ret i32 %[[r_load]]
//
// CHECK: store i32 123, i32* %[[r]]
// CHECK: br label %[[inner_try_cont]]
//
// CHECK: [[inner_try_cont]]
// CHECK: br label %[[outer_try_cont]]

// FIXME: This lowering of __finally can't actually work, it will have to
// change.
static unsigned g = 0;
void basic_finally(void) {
  ++g;
  __try {
    j();
  } __finally {
    --g;
  }
}
// CHECK-LABEL: define void @basic_finally()
// CHECK: load i32* @g
// CHECK: add i32 %{{.*}}, 1
// CHECK: store i32 %{{.*}}, i32* @g
//
// CHECK: invoke void @j()
// CHECK:       to label %[[cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CHECK: [[cont]]
// CHECK: br label %[[finally:[^ ]*]]
//
// CHECK: [[finally]]
// CHECK: load i32* @g
// CHECK: add i32 %{{.*}}, -1
// CHECK: store i32 %{{.*}}, i32* @g
// CHECK: icmp eq
// CHECK: br i1 %{{.*}}, label
//
// CHECK: ret void
//
// CHECK: [[lpad]]
// CHECK: landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*)
// CHECK-NEXT: cleanup
// CHECK: br label %[[finally]]
//
// CHECK: resume

int returns_int(void);
int except_return(void) {
  __try {
    return returns_int();
  } __except(1) {
    return 42;
  }
}
// CHECK-LABEL: define i32 @except_return()
// CHECK: %[[tmp:[^ ]*]] = invoke i32 @returns_int()
// CHECK:       to label %[[cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CHECK: [[cont]]
// CHECK: store i32 %[[tmp]], i32* %[[rv:[^ ]*]]
// CHECK: br label %[[retbb:[^ ]*]]
//
// CHECK: [[lpad]]
// CHECK: store i32 42, i32* %[[rv]]
// CHECK: br label %[[retbb]]
//
// CHECK: [[retbb]]
// CHECK: %[[r:[^ ]*]] = load i32* %[[rv]]
// CHECK: ret i32 %[[r]]

// CHECK: attributes #[[NOINLINE]] = { {{.*noinline.*}} }
