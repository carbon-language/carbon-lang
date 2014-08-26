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
// This file defines a copy of __asan_option_detect_stack_use_after_return that
// should be used when linking an MD runtime with a set of object files on
// Windows.
//
// The ASan MD runtime dllexports this variable, so normally we would dllimport
// it in each TU.  Unfortunately, in general we don't know
// if a given TU is going to be used with a MT or MD runtime.
//===----------------------------------------------------------------------===//

// Only compile this code when buidling asan_uar_thunk.lib
// Using #ifdef rather than relying on Makefiles etc.
// simplifies the build procedure.
#ifdef ASAN_UAR_THUNK
extern "C" {
__declspec(dllimport) int __asan_should_detect_stack_use_after_return();

int __asan_option_detect_stack_use_after_return =
    __asan_should_detect_stack_use_after_return();
}
#endif // ASAN_UAR_THUNK
