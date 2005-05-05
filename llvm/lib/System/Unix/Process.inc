//===- Unix/Process.cpp - Unix Process Implementation --------- -*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file provides the generic Unix implementation of the Process class.
//
//===----------------------------------------------------------------------===//

#include "Unix.h"
#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif
#ifdef HAVE_SYS_RESOURCE_H
#include <sys/resource.h>
#endif
#ifdef HAVE_MALLOC_H
#include <malloc.h>
#endif

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic UNIX code that
//===          is guaranteed to work on *all* UNIX variants.
//===----------------------------------------------------------------------===//

namespace llvm {
using namespace sys;

unsigned 
Process::GetPageSize() 
{
#if defined(HAVE_GETPAGESIZE)
  static const int page_size = ::getpagesize();
#elif defined(HAVE_SYSCONF)
  static long page_size = ::sysconf(_SC_PAGE_SIZE);
#else
#warning Cannot get the page size on this machine
#endif
  return static_cast<unsigned>(page_size);
}

#if defined(HAVE_SBRK)
static char* som = reinterpret_cast<char*>(::sbrk(0));
#endif

size_t 
Process::GetMallocUsage()
{
#if defined(HAVE_MALLINFO)
  struct mallinfo mi;
  mi = ::mallinfo();
  return mi.uordblks;
#elif defined(HAVE_SBRK)
  // Note this is only an approximation and more closely resembles
  // the value returned by mallinfo in the arena field.
  char * eom = (char*) sbrk(0);
  if (eom != ((char*)-1) && som != ((char*)-1))
    return eom - som;
  else
    return 0;
#else
#warning Cannot get malloc info on this platform
  return 0;
#endif
}

size_t
Process::GetTotalMemoryUsage()
{
#if defined(HAVE_MALLINFO)
  struct mallinfo mi = ::mallinfo();
  return mi.uordblks + mi.hblkhd;
#elif defined(HAVE_GETRUSAGE)
  struct rusage usage;
  ::getrusage(RUSAGE_SELF, &usage);
  return usage.ru_maxrss;
#else
#warning Cannot get total memory size on this platform
  return 0;
#endif
}

void
Process::GetTimeUsage(TimeValue& elapsed, TimeValue& user_time, 
                      TimeValue& sys_time)
{
  elapsed = TimeValue::now();
#if defined(HAVE_GETRUSAGE)
  struct rusage usage;
  ::getrusage(RUSAGE_SELF, &usage);
  user_time = TimeValue( 
    static_cast<TimeValue::SecondsType>( usage.ru_utime.tv_sec ), 
    static_cast<TimeValue::NanoSecondsType>( usage.ru_utime.tv_usec * 
      TimeValue::NANOSECONDS_PER_MICROSECOND ) );
  sys_time = TimeValue( 
    static_cast<TimeValue::SecondsType>( usage.ru_stime.tv_sec ), 
    static_cast<TimeValue::NanoSecondsType>( usage.ru_stime.tv_usec * 
      TimeValue::NANOSECONDS_PER_MICROSECOND ) );
#else
#warning Cannot get usage times on this platform
  user_time.seconds(0);
  user_time.microseconds(0);
  sys_time.seconds(0);
  sys_time.microseconds(0);
#endif
}

int Process::GetCurrentUserId()
{
  return getuid();
}

int Process::GetCurrentGroupId()
{
  return getgid();
}

// Some LLVM programs such as bugpoint produce core files as a normal part of
// their operation. To prevent the disk from filling up, this function
// does what's necessary to prevent their generation.
void Process::PreventCoreFiles() {
#if HAVE_SETRLIMIT
  struct rlimit rlim;
  rlim.rlim_cur = rlim.rlim_max = 0;
  int res = setrlimit(RLIMIT_CORE, &rlim);
  if (res != 0)
    ThrowErrno("Can't prevent core file generation");
#endif
}

bool Process::StandardInIsUserInput() {
#if HAVE_ISATTY
  return isatty(0);
#endif
  // If we don't have isatty, just return false.
  return false;
}

bool Process::StandardOutIsDisplayed() {
#if HAVE_ISATTY
  return isatty(1);
#endif
  // If we don't have isatty, just return false.
  return false;
}

bool Process::StandardErrIsDisplayed() {
#if HAVE_ISATTY
  return isatty(2);
#endif
  // If we don't have isatty, just return false.
  return false;
}

}
