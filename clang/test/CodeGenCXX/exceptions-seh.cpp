// RUN: %clang_cc1 -std=c++11 -fblocks -fms-extensions %s -triple=x86_64-windows-msvc -emit-llvm -o - -fcxx-exceptions -fexceptions -mconstructor-aliases | FileCheck %s

extern "C" void might_throw();

struct HasCleanup {
  HasCleanup();
  ~HasCleanup();
  int padding;
};

extern "C" void use_cxx() {
  HasCleanup x;
  might_throw();
}

// Make sure we use __CxxFrameHandler3 for C++ EH.

// CHECK-LABEL: define void @use_cxx()
// CHECK: call %struct.HasCleanup* @"\01??0HasCleanup@@QEAA@XZ"(%struct.HasCleanup* %{{.*}})
// CHECK: invoke void @might_throw()
// CHECK:       to label %[[cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CHECK: [[cont]]
// CHECK: call void @"\01??1HasCleanup@@QEAA@XZ"(%struct.HasCleanup* %{{.*}})
// CHECK: ret void
//
// CHECK: [[lpad]]
// CHECK: landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
// CHECK-NEXT: cleanup
// CHECK: call void @"\01??1HasCleanup@@QEAA@XZ"(%struct.HasCleanup* %{{.*}})
// CHECK: br label %[[resume:[^ ]*]]
//
// CHECK: [[resume]]
// CHECK: resume

extern "C" void use_seh() {
  __try {
    might_throw();
  } __except(1) {
  }
}

// Make sure we use __C_specific_handler for SEH.

// CHECK-LABEL: define void @use_seh()
// CHECK: invoke void @might_throw()
// CHECK:       to label %[[cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CHECK: [[cont]]
// CHECK: br label %[[ret:[^ ]*]]
//
// CHECK: [[lpad]]
// CHECK: landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*)
// CHECK-NEXT: catch i8*
//
// CHECK: br label %[[ret]]
//
// CHECK: [[ret]]
// CHECK: ret void

void use_seh_in_lambda() {
  ([]() {
    __try {
      might_throw();
    } __except(1) {
    }
  })();
  HasCleanup x;
  might_throw();
}

// CHECK-LABEL: define void @"\01?use_seh_in_lambda@@YAXXZ"()
// CHECK: landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)

// CHECK-LABEL: define internal void @"\01??R<lambda_0>@?use_seh_in_lambda@@YAXXZ@QEBAXXZ"(%class.anon* %this)
// CHECK: landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*)
