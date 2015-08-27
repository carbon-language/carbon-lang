//===-- llvm/Support/thread.h - Wrapper for <thread> ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header is a wrapper for <thread> that works around problems with the
// MSVC headers when exceptions are disabled.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_THREAD_H
#define LLVM_SUPPORT_THREAD_H

#ifdef _MSC_VER
// concrt.h depends on eh.h for __uncaught_exception declaration
// even if we disable exceptions.
#include <eh.h>

// Suppress 'C++ exception handler used, but unwind semantics are not enabled.'
#pragma warning(push)
#pragma warning(disable:4530)
#endif

#include <thread>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif
