// RUN: %clang_cc1 %s -std=c++11 -emit-llvm-only
// CHECK that we don't crash.

union _XEvent {
};
void ProcessEvent() {
  _XEvent pluginEvent = _XEvent();
}
