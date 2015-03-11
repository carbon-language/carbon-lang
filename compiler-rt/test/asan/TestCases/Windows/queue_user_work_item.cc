// Make sure we can throw exceptions from work items executed via
// QueueUserWorkItem.
//
// Clang doesn't support exceptions on Windows yet, so for the time being we
// build this program in two parts: the code with exceptions is built with CL,
// the rest is built with Clang.  This represents the typical scenario when we
// build a large project using "clang-cl -fallback -fsanitize=address".
//
// RUN: cl -c %s -Fo%t.obj
// RUN: %clangxx_asan -o %t.exe %s %t.obj
// RUN: %run %t.exe 2>&1 | FileCheck %s

#include <windows.h>
#include <stdio.h>

void ThrowAndCatch();

#if !defined(__clang__)
__declspec(noinline)
void Throw() {
  fprintf(stderr, "Throw\n");
// CHECK: Throw
  throw 1;
}

void ThrowAndCatch() {
  int local;
  try {
    Throw();
  } catch(...) {
    fprintf(stderr, "Catch\n");
// CHECK: Catch
  }
}
#else

HANDLE done;

DWORD CALLBACK work_item(LPVOID) {
  ThrowAndCatch();
  SetEvent(done);
  return 0;
}

int main(int argc, char **argv) {
  done = CreateEvent(0, false, false, "job is done");
  if (!done)
    return 1;
  QueueUserWorkItem(&work_item, nullptr, 0);
  if (WAIT_OBJECT_0 != WaitForSingleObject(done, INFINITE))
    return 2;
  fprintf(stderr, "Done!\n");
// CHECK: Done!
}
#endif
