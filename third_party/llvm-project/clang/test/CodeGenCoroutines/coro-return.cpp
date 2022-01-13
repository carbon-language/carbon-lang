// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm %s -o - -disable-llvm-passes | FileCheck %s

namespace std {
template <typename... T> struct coroutine_traits;

template <class Promise = void> struct coroutine_handle {
  coroutine_handle() = default;
  static coroutine_handle from_address(void *) noexcept { return {}; }
};
template <> struct coroutine_handle<void> {
  static coroutine_handle from_address(void *) { return {}; }
  coroutine_handle() = default;
  template <class PromiseType>
  coroutine_handle(coroutine_handle<PromiseType>) noexcept {}
};
} // namespace std

struct suspend_always {
  bool await_ready() noexcept;
  void await_suspend(std::coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

template <> struct std::coroutine_traits<void> {
  struct promise_type {
    void get_return_object();
    suspend_always initial_suspend();
    suspend_always final_suspend() noexcept;
    void return_void();
  };
};

// CHECK-LABEL: f0(
extern "C" void f0() {
  // CHECK: %__promise = alloca %"struct.std::coroutine_traits<void>::promise_type"
  // CHECK: %call = call noalias nonnull i8* @_Znwm(
  // CHECK: call void @_ZNSt16coroutine_traitsIJvEE12promise_type11return_voidEv(%"struct.std::coroutine_traits<void>::promise_type"* {{[^,]*}} %__promise)
  // CHECK: call void @_ZdlPv
  co_return;
}

template <>
struct std::coroutine_traits<int> {
  struct promise_type {
    int get_return_object();
    suspend_always initial_suspend();
    suspend_always final_suspend() noexcept;
    void return_value(int);
  };
};

// CHECK-LABEL: f1(
extern "C" int f1() {
  // CHECK: %__promise = alloca %"struct.std::coroutine_traits<int>::promise_type"
  // CHECK: %call = call noalias nonnull i8* @_Znwm(
  // CHECK: call void @_ZNSt16coroutine_traitsIJiEE12promise_type12return_valueEi(%"struct.std::coroutine_traits<int>::promise_type"* {{[^,]*}} %__promise, i32 42)
  // CHECK: call void @_ZdlPv
  co_return 42;
}
