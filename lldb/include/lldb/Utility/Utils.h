//===-- Utils.h -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef utility_Utils_h_
#define utility_Utils_h_

namespace lldb_private {

// Return the number of elements of a static array.
template <typename T, unsigned size>
inline unsigned arraysize(T (&v)[size]) { return size; }

} // namespace lldb_private
#endif // utility_Utils
