// RUN: %clang %s -O0 -emit-llvm -S -o - | FileCheck %s

// This should call rb_define_global_function, not rb_f_chop.
void rb_define_global_function (const char*,void(*)(),int);
static void rb_f_chop();
void Init_String() {
  rb_define_global_function("chop", rb_f_chop, 0);
}
static void rb_f_chop() {
}

// CHECK: call{{.*}}rb_define_global_function

// PR10335
typedef   void (* JSErrorCallback)(void);
void js_GetErrorMessage(void);
void JS_ReportErrorNumber(JSErrorCallback errorCallback, ...);
void Interpret() {
  JS_ReportErrorNumber(js_GetErrorMessage, 0);
  
  // CHECK: call void ({{.*}}, ...)* @JS_ReportErrorNumber({{.*}}@js_GetErrorMessage
}

