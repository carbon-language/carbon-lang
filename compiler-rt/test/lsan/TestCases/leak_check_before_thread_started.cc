// Regression test for http://llvm.org/bugs/show_bug.cgi?id=21621
// This test relies on timing between threads, so any failures will be flaky.
// RUN: LSAN_BASE="use_stacks=0:use_registers=0"
// RUN: %clangxx_lsan %s -std=c++11 -o %t
// RUN: %run %t
#include <thread>
#include <chrono>

void func() {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
}

int main() {
      std::thread(func).detach();
}
