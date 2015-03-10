//===- lld/Core/STDExtra.h - Helpers for the stdlib -----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_STD_EXTRA_H
#define LLD_CORE_STD_EXTRA_H

namespace lld {
/// \brief Deleter for smart pointers that only calls the destructor. Memory is
/// managed elsewhere. A common use of this is for things allocated with a
/// BumpPtrAllocator.
template <class T>
struct destruct_delete {
  void operator ()(T *ptr) {
    ptr->~T();
  }
};

template <class T>
using unique_bump_ptr = std::unique_ptr<T, destruct_delete<T>>;

} // end namespace lld

#endif
