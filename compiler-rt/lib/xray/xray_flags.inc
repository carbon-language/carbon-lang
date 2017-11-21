//===-- xray_flags.inc ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// XRay runtime flags.
//
//===----------------------------------------------------------------------===//
#ifndef XRAY_FLAG
#error "Define XRAY_FLAG prior to including this file!"
#endif

XRAY_FLAG(bool, patch_premain, false,
          "Whether to patch instrumentation points before main.")
XRAY_FLAG(bool, xray_naive_log, true,
          "Whether to install the naive log implementation.")
XRAY_FLAG(const char *, xray_logfile_base, "xray-log.",
          "Filename base for the xray logfile.")
XRAY_FLAG(bool, xray_fdr_log, false,
          "Whether to install the flight data recorder logging implementation.")
XRAY_FLAG(int, xray_fdr_log_func_duration_threshold_us, 5,
          "FDR logging will try to skip functions that execute for fewer "
          "microseconds than this threshold.")
XRAY_FLAG(int, xray_fdr_log_grace_period_us, 100000,
          "FDR logging will wait this much time in microseconds before "
          "actually flushing the log; this gives a chance for threads to "
          "notice that the log has been finalized and clean up.")
