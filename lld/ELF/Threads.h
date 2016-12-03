//===- Threads.h ------------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_THREADS_H
#define LLD_ELF_THREADS_H

#include "Config.h"

#include "lld/Core/Parallel.h"
#include <algorithm>
#include <functional>

namespace lld {
namespace elf {

template <class IterTy, class FuncTy>
void forEach(IterTy Begin, IterTy End, FuncTy Fn) {
  if (Config->Threads)
    parallel_for_each(Begin, End, Fn);
  else
    std::for_each(Begin, End, Fn);
}

inline void forLoop(size_t Begin, size_t End, std::function<void(size_t)> Fn) {
  if (Config->Threads) {
    parallel_for(Begin, End, Fn);
  } else {
    for (size_t I = Begin; I < End; ++I)
      Fn(I);
  }
}
}
}

#endif
