//===-- Definition of struct tm -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_TM_H__
#define __LLVM_LIBC_TYPES_TM_H__

struct tm {
  int tm_sec;   // seconds after the minute
  int tm_min;   // minutes after the hour
  int tm_hour;  // hours since midnight
  int tm_mday;  // day of the month
  int tm_mon;   // months since January
  int tm_year;  // years since 1900
  int tm_wday;  // days since Sunday
  int tm_yday;  // days since January
  int tm_isdst; // Daylight Saving Time flag
};

#endif // __LLVM_LIBC_TYPES_TM_H__
