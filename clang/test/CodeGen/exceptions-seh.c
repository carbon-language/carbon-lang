// RUN: %clang_cc1 %s -triple x86_64-pc-win32 -fms-extensions -emit-llvm -o - \
// RUN:         | FileCheck %s --check-prefix=CHECK --check-prefix=X64
// RUN: %clang_cc1 %s -triple i686-pc-win32 -fms-extensions -emit-llvm -o - \
// RUN:         | FileCheck %s --check-prefix=CHECK --check-prefix=X86

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
// X64: landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*)
// X86: landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @_except_handler3 to i8*)
// CHECK-NEXT: catch i8* null
// CHECK-NOT: br i1
// CHECK: br label %[[except:[^ ]*]]
// CHECK: [[except]]
// CHECK-NEXT: store i32 -42, i32* %[[success:[^ ]*]]
//
// CHECK: %[[res:[^ ]*]] = load i32, i32* %[[success]]
// CHECK: ret i32 %[[res]]

void j(void);

int filter_expr_capture(void) {
  int r = 42;
  __try {
    j();
  } __except(r = -1) {
    r = 13;
  }
  return r;
}

// CHECK-LABEL: define i32 @filter_expr_capture()
// CHECK: call void (...) @llvm.frameescape(i32* %[[r:[^ ,]*]])
// CHECK: store i32 42, i32* %[[r]]
// CHECK: invoke void @j() #[[NOINLINE]]
//
// CHECK: landingpad
// CHECK-NEXT: catch i8* bitcast (i32 ({{.*}})* @"\01?filt$0@0@filter_expr_capture@@" to i8*)
// CHECK: store i32 13, i32* %[[r]]
//
// CHECK: %[[rv:[^ ]*]] = load i32, i32* %[[r]]
// CHECK: ret i32 %[[rv]]

// X64-LABEL: define internal i32 @"\01?filt$0@0@filter_expr_capture@@"(i8* %exception_pointers, i8* %frame_pointer)
// X64: call i8* @llvm.framerecover(i8* bitcast (i32 ()* @filter_expr_capture to i8*), i8* %frame_pointer, i32 0)
//
// X86-LABEL: define internal i32 @"\01?filt$0@0@filter_expr_capture@@"()
// X86: %[[fp:[^ ]*]] = call i8* @llvm.frameaddress(i32 1)
// X86: call i8* @llvm.framerecover(i8* bitcast (i32 ()* @filter_expr_capture to i8*), i8* %[[fp]], i32 0)
//
// CHECK: store i32 -1, i32* %{{.*}}
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
// X64: landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*)
// X86: landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @_except_handler3 to i8*)
// CHECK: catch i8* bitcast (i32 ({{.*}})* @"\01?filt$1@0@nested_try@@" to i8*)
// CHECK: catch i8* bitcast (i32 ({{.*}})* @"\01?filt$0@0@nested_try@@" to i8*)
// CHECK: store i8* %{{.*}}, i8** %[[ehptr_slot:[^ ]*]]
// CHECK: store i32 %{{.*}}, i32* %[[sel_slot:[^ ]*]]
//
// CHECK: load i32, i32* %[[sel_slot]]
// CHECK: call i32 @llvm.eh.typeid.for(i8* bitcast (i32 ({{.*}})* @"\01?filt$1@0@nested_try@@" to i8*))
// CHECK: icmp eq i32
// CHECK: br i1
//
// CHECK: load i32, i32* %[[sel_slot]]
// CHECK: call i32 @llvm.eh.typeid.for(i8* bitcast (i32 ({{.*}})* @"\01?filt$0@0@nested_try@@" to i8*))
// CHECK: icmp eq i32
// CHECK: br i1
//
// CHECK: store i32 456, i32* %[[r]]
// CHECK: br label %[[outer_try_cont:[^ ]*]]
//
// CHECK: [[outer_try_cont]]
// CHECK: %[[r_load:[^ ]*]] = load i32, i32* %[[r]]
// CHECK: ret i32 %[[r_load]]
//
// CHECK: store i32 123, i32* %[[r]]
// CHECK: br label %[[inner_try_cont]]
//
// CHECK: [[inner_try_cont]]
// CHECK: br label %[[outer_try_cont]]
//
// CHECK-LABEL: define internal i32 @"\01?filt$0@0@nested_try@@"({{.*}})
// X86: call i8* @llvm.eh.exceptioninfo()
// CHECK: load i32*, i32**
// CHECK: load i32, i32*
// CHECK: ptrtoint
// CHECK: icmp eq i32 %{{.*}}, 456
//
// CHECK-LABEL: define internal i32 @"\01?filt$1@0@nested_try@@"({{.*}})
// X86: call i8* @llvm.eh.exceptioninfo()
// CHECK: load i32*, i32**
// CHECK: load i32, i32*
// CHECK: ptrtoint
// CHECK: icmp eq i32 %{{.*}}, 123

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
// CHECK: load i32, i32* @g
// CHECK: add i32 %{{.*}}, 1
// CHECK: store i32 %{{.*}}, i32* @g
//
// CHECK: invoke void @j()
// CHECK:       to label %[[cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CHECK: [[cont]]
// X64: %[[fp:[^ ]*]] = call i8* @llvm.frameaddress(i32 0)
// X64: call void @"\01?fin$0@0@basic_finally@@"(i8 0, i8* %[[fp]])
// X86: call void @"\01?fin$0@0@basic_finally@@"()
// CHECK: ret void
//
// CHECK: [[lpad]]
// X64: landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*)
// X86: landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @_except_handler3 to i8*)
// CHECK-NEXT: cleanup
// X64: %[[fp:[^ ]*]] = call i8* @llvm.frameaddress(i32 0)
// X64: call void @"\01?fin$0@0@basic_finally@@"(i8 1, i8* %[[fp]])
// X86: call void @"\01?fin$0@0@basic_finally@@"()
// CHECK: resume

// CHECK: define internal void @"\01?fin$0@0@basic_finally@@"({{.*}})
// CHECK:   load i32, i32* @g, align 4
// CHECK:   add i32 %{{.*}}, -1
// CHECK:   store i32 %{{.*}}, i32* @g, align 4
// CHECK:   ret void

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
// CHECK: %[[r:[^ ]*]] = load i32, i32* %[[rv]]
// CHECK: ret i32 %[[r]]

// CHECK: attributes #[[NOINLINE]] = { {{.*noinline.*}} }
