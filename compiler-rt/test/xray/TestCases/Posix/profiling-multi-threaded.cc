// Check that we can get a profile from a single-threaded application, on
// demand through the XRay logging implementation API.
//
// FIXME: Make -fxray-modes=xray-profiling part of the default?
// RUN: %clangxx_xray -std=c++11 %s -o %t -fxray-modes=xray-profiling
// RUN: %run %t
//
// UNSUPPORTED: target-is-mips64,target-is-mips64el

#include "xray/xray_interface.h"
#include "xray/xray_log_interface.h"
#include <cassert>
#include <cstdio>
#include <string>
#include <thread>

#define XRAY_ALWAYS_INSTRUMENT [[clang::xray_always_instrument]]
#define XRAY_NEVER_INSTRUMENT [[clang::xray_never_instrument]]

XRAY_ALWAYS_INSTRUMENT void f2() { return; }
XRAY_ALWAYS_INSTRUMENT void f1() { f2(); }
XRAY_ALWAYS_INSTRUMENT void f0() { f1(); }

using namespace std;

volatile int buffer_counter = 0;

XRAY_NEVER_INSTRUMENT void process_buffer(const char *, XRayBuffer) {
  // FIXME: Actually assert the contents of the buffer.
  ++buffer_counter;
}

XRAY_ALWAYS_INSTRUMENT int main(int, char **) {
  assert(__xray_log_select_mode("xray-profiling") ==
         XRayLogRegisterStatus::XRAY_REGISTRATION_OK);
  assert(__xray_log_get_current_mode() != nullptr);
  std::string current_mode = __xray_log_get_current_mode();
  assert(current_mode == "xray-profiling");
  assert(__xray_patch() == XRayPatchingStatus::SUCCESS);
  assert(__xray_log_init(0, 0, nullptr, 0) ==
         XRayLogInitStatus::XRAY_LOG_INITIALIZED);
  std::thread t0([] { f0(); });
  std::thread t1([] { f0(); });
  f0();
  t0.join();
  t1.join();
  assert(__xray_log_finalize() == XRayLogInitStatus::XRAY_LOG_FINALIZED);
  assert(__xray_log_process_buffers(process_buffer) ==
         XRayLogFlushStatus::XRAY_LOG_FLUSHED);
  // We're running three threds, so we expect three buffers.
  assert(buffer_counter == 3);
  assert(__xray_log_flushLog() == XRayLogFlushStatus::XRAY_LOG_FLUSHED);
}
