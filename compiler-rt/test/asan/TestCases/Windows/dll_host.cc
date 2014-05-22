// This is a host program for DLL tests.
//
// Just make sure we can compile this.
// The actual compile&run sequence is to be done by the DLL tests.
// RUN: %clangxx_asan -O0 %s -Fe%t
//
// Get the list of ASan wrappers exported by the main module RTL:
// RUN: dumpbin /EXPORTS %t | grep -o "__asan_wrap[^ ]*" | grep -v @ | sort | uniq > %t.exported_wrappers
//
// Get the list of ASan wrappers imported by the DLL RTL:
// RUN: grep INTERCEPT_LIBRARY_FUNCTION %p/../../../../lib/asan/asan_dll_thunk.cc | grep -v define | sed "s/.*(\(.*\)).*/__asan_wrap_\1/" | sort | uniq > %t.dll_imports
//
// Now make sure the DLL thunk imports everything:
// RUN: echo
// RUN: echo "=== NOTE === If you see a mismatch below, please update asan_dll_thunk.cc"
// RUN: diff %t.dll_imports %t.exported_wrappers

#include <stdio.h>
#include <windows.h>

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s [client].dll\n", argv[0]);
    return 101;
  }

  const char *dll_name = argv[1];

  HMODULE h = LoadLibrary(dll_name);
  if (!h) {
    printf("Could not load DLL: %s (code: %lu)!\n",
           dll_name, GetLastError());
    return 102;
  }

  typedef int (*test_function)();
  test_function gf = (test_function)GetProcAddress(h, "test_function");
  if (!gf) {
    printf("Could not locate test_function in the DLL!\n");
    FreeLibrary(h);
    return 103;
  }

  int ret = gf();

  FreeLibrary(h);
  return ret;
}
