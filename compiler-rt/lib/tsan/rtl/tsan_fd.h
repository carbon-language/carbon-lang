//===-- tsan_fd.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#ifndef TSAN_FD_H
#define TSAN_FD_H

#include "tsan_rtl.h"

namespace __tsan {

void FdInit();
void FdAcquire(ThreadState *thr, uptr pc, int fd);
void FdRelease(ThreadState *thr, uptr pc, int fd);
void FdClose(ThreadState *thr, uptr pc, int fd);
void FdFileCreate(ThreadState *thr, uptr pc, int fd);
void FdDup(ThreadState *thr, uptr pc, int oldfd, int newfd);
void FdPipeCreate(ThreadState *thr, uptr pc, int rfd, int wfd);
void FdEventCreate(ThreadState *thr, uptr pc, int fd);
void FdPollCreate(ThreadState *thr, uptr pc, int fd);
void FdSocketCreate(ThreadState *thr, uptr pc, int fd);
void FdSocketAccept(ThreadState *thr, uptr pc, int fd, int newfd);
void FdSocketConnect(ThreadState *thr, uptr pc, int fd);

uptr File2addr(char *path);
uptr Dir2addr(char *path);

}  // namespace __tsan

#endif  // TSAN_INTERFACE_H
