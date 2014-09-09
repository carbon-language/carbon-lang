//===-- ThisThread.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/linux/HostThreadLinux.h"
#include "lldb/Host/ThisThread.h"

#include "llvm/ADT/SmallVector.h"

#include <pthread.h>

using namespace lldb_private;

void
ThisThread::SetName(llvm::StringRef name)
{
    HostThread::SetName(::pthread_self(), name);
}

void
ThisThread::GetName(llvm::SmallVectorImpl<char> &name)
{
    HostThread::GetName(::pthread_self(), name);
}
