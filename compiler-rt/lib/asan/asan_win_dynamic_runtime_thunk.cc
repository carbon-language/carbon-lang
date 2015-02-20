//===-- asan_win_uar_thunk.cc ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// This file defines things that need to be present in the application modules
// to interact with the ASan DLL runtime correctly and can't be implemented
// using the default "import library" generated when linking the DLL RTL.
//
// This includes:
//  - forwarding the detect_stack_use_after_return runtime option
//  - installing a custom SEH handler
//
//===----------------------------------------------------------------------===//

// Only compile this code when buidling asan_dynamic_runtime_thunk.lib
// Using #ifdef rather than relying on Makefiles etc.
// simplifies the build procedure.
#ifdef ASAN_DYNAMIC_RUNTIME_THUNK
#include <windows.h>
#include <psapi.h>

extern "C" {
////////////////////////////////////////////////////////////////////////////////
// Define a copy of __asan_option_detect_stack_use_after_return that should be
// used when linking an MD runtime with a set of object files on Windows.
//
// The ASan MD runtime dllexports '__asan_option_detect_stack_use_after_return',
// so normally we would just dllimport it.  Unfortunately, the dllimport
// attribute adds __imp_ prefix to the symbol name of a variable.
// Since in general we don't know if a given TU is going to be used
// with a MT or MD runtime and we don't want to use ugly __imp_ names on Windows
// just to work around this issue, let's clone the a variable that is
// constant after initialization anyways.
__declspec(dllimport) int __asan_should_detect_stack_use_after_return();
int __asan_option_detect_stack_use_after_return =
    __asan_should_detect_stack_use_after_return();
}

////////////////////////////////////////////////////////////////////////////////
// For some reason, the MD CRT doesn't call the C/C++ terminators as MT does.
// To work around this, for each DLL we schedule a call to
// UnregisterGlobalsInRange atexit() specifying the address range of the DLL
// image to unregister globals in that range.   We don't do the same
// for the main module (.exe) as the asan_globals.cc allocator is destroyed
// by the time UnregisterGlobalsInRange is executed.
// See PR22545 for the details.
namespace __asan {
__declspec(dllimport)
void UnregisterGlobalsInRange(void *beg, void *end);
}

namespace {
void *this_module_base, *this_module_end;

void UnregisterGlobals() {
  __asan::UnregisterGlobalsInRange(this_module_base, this_module_end);
}

int ScheduleUnregisterGlobals() {
  HMODULE this_module = 0;
  // Increments the reference counter of the DLL module, so need to call
  // FreeLibrary later.
  if (!GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                         (LPCTSTR)&UnregisterGlobals, &this_module))
    return 1;

  // Skip the main module.
  if (this_module == GetModuleHandle(0))
    return 0;

  MODULEINFO mi;
  bool success =
      GetModuleInformation(GetCurrentProcess(), this_module, &mi, sizeof(mi));
  if (!FreeLibrary(this_module))
    return 2;
  if (!success)
    return 3;

  this_module_base = mi.lpBaseOfDll;
  this_module_end = (char*)mi.lpBaseOfDll + mi.SizeOfImage;

  return atexit(UnregisterGlobals);
}
}  // namespace

///////////////////////////////////////////////////////////////////////////////
// ASan SEH handling.
extern "C" __declspec(dllimport) int __asan_set_seh_filter();
static int SetSEHFilter() { return __asan_set_seh_filter(); }

///////////////////////////////////////////////////////////////////////////////
// We schedule some work at start-up by placing callbacks to our code to the
// list of CRT C initializers.
//
// First, declare sections we'll be using:
#pragma section(".CRT$XID", long, read)  // NOLINT
#pragma section(".CRT$XIZ", long, read)  // NOLINT

// We need to call 'atexit(UnregisterGlobals);' after atexit() is initialized
// (.CRT$XIC) but before the C++ constructors (.CRT$XCA).
__declspec(allocate(".CRT$XID"))
static int (*__asan_schedule_unregister_globals)() = ScheduleUnregisterGlobals;

// We need to set the ASan-specific SEH handler at the end of CRT initialization
// of each module (see also asan_win.cc).
//
// Unfortunately, putting a pointer to __asan_set_seh_filter into
// __asan_intercept_seh gets optimized out, so we have to use an extra function.
extern "C" __declspec(allocate(".CRT$XIZ"))
int (*__asan_seh_interceptor)() = SetSEHFilter;

#endif // ASAN_DYNAMIC_RUNTIME_THUNK
