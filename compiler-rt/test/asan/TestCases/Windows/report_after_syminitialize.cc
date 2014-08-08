// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <windows.h>
#include <dbghelp.h>

int main() {
  // Make sure the RTL recovers from "no options enabled" dbghelp setup.
  SymSetOptions(0);

  // Make sure the RTL recovers from "fInvadeProcess=FALSE".
  if (!SymInitialize(GetCurrentProcess(), 0, FALSE))
    return 42;

  *(volatile int*)0 = 42;
  // CHECK: ERROR: AddressSanitizer: access-violation on unknown address
  // CHECK-NEXT: {{WARNING: .*DbgHelp}}
  // CHECK: {{#0 0x.* in main.*report_after_syminitialize.cc:}}[[@LINE-3]]
  // CHECK: AddressSanitizer can not provide additional info.
}
