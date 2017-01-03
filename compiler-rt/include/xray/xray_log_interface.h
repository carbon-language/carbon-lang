//===-- xray_log_interface.h ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a function call tracing system.
//
// APIs for installing a new logging implementation.
//===----------------------------------------------------------------------===//
#ifndef XRAY_XRAY_LOG_INTERFACE_H
#define XRAY_XRAY_LOG_INTERFACE_H

#include "xray/xray_interface.h"
#include <stddef.h>

extern "C" {

enum XRayLogInitStatus {
  XRAY_LOG_UNINITIALIZED,
  XRAY_LOG_INITIALIZING,
  XRAY_LOG_INITIALIZED,
  XRAY_LOG_FINALIZING,
  XRAY_LOG_FINALIZED,
};

enum XRayLogFlushStatus {
  XRAY_LOG_NOT_FLUSHING,
  XRAY_LOG_FLUSHING,
  XRAY_LOG_FLUSHED
};

struct XRayLogImpl {
  XRayLogInitStatus (*log_init)(size_t, size_t, void *, size_t);
  XRayLogInitStatus (*log_finalize)();
  void (*handle_arg0)(int32_t, XRayEntryType);
  XRayLogFlushStatus (*flush_log)();
};

void __xray_set_log_impl(XRayLogImpl Impl);
XRayLogInitStatus __xray_log_init(size_t BufferSize, size_t MaxBuffers,
                                  void *Args, size_t ArgsSize);
XRayLogInitStatus __xray_log_finalize();
XRayLogFlushStatus __xray_log_flushLog();

} // extern "C"

#endif // XRAY_XRAY_LOG_INTERFACE_H
