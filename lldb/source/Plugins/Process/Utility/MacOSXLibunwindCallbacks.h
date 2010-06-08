//===-- MacOSXLibunwindCallbacks.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_MacOSXLibunwindCallbacks_h_
#define liblldb_MacOSXLibunwindCallbacks_h_
#if defined(__cplusplus)

namespace lldb_private {

unw_accessors_t get_macosx_libunwind_callbacks ();

} // namespace lldb_utility

#endif  // #if defined(__cplusplus)
#endif // #ifndef liblldb_MacOSXLibunwindCallbacks_h_

