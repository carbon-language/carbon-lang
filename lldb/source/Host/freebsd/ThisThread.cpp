//===-- ThisThread.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/HostNativeThread.h"
#include "lldb/Host/ThisThread.h"

#include "llvm/ADT/SmallVector.h"

#include <pthread.h>
#include <pthread_np.h>

using namespace lldb_private;

void
ThisThread::SetName(llvm::StringRef name)
{
    ::pthread_set_name_np(::pthread_self(), name.data());
}

void
ThisThread::GetName(llvm::SmallVectorImpl<char> &name)
{
    HostNativeThread::GetName(::pthread_getthreadid_np(), name);
}
