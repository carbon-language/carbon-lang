//===- LinuxCommon.h - Common Declarations For Linux ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Copyright (C) 2004 eXtensible Systems, Inc. All Rights Reserved.
//
// This program is open source software; you can redistribute it and/or modify
// it under the terms of the University of Illinois Open Source License. See
// LICENSE.TXT (distributed with this software) for details.
//
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
// or FITNESS FOR A PARTICULAR PURPOSE.
//
//===----------------------------------------------------------------------===//
/// @file lib/System/linux/Common.h
/// @author Reid Spencer <rspencer@x10sys.com> (original author)
/// @version \verbatim $Id$ \endverbatim
/// @date 2004/08/14
/// @since 1.4
/// @brief Provides common linux specific declarations and includes.
//===----------------------------------------------------------------------===//

#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/param.h>
#include <pthread.h>

#define FATAL(arg) \
  { llvm::sys::panic(LLVM_CONTEXT, arg); }

#define RETURN_ERRNO \
  return ErrorCode(LLVM_ERROR_CODE(OSDomain,errno))

#define RETURN_OSERROR(code) \
  return ErrorCode(LLVM_ERROR_CODE(OSDomain,code))

#ifdef assert
#define ASSERT_ARG(expr) \
  if ( ! (expr) ) {\
    return ErrorCode(ERR_SYS_INVALID_ARG);\
  }
#else
#define ASSERT_ARG(expr)
#endif

#define CHECK(var,call,args,msg) \
  { \
    if ( -1 == ( var = ::call args )) { \
      panic(LLVM_CONTEXT, msg); \
    } \
  }

#define RETURN_ON_ERROR(call,args) \
  { \
    errno = 0; \
    if ( 0 > ( call args ) ) { \
      RETURN_ERRNO; \
    } \
  }

#define RETURN_ON_ERRORCODE(call,args) \
  { \
    ErrorCode code = call args ; \
    if ( ! code ) \
      return code; \
  }

#define PTHREAD_CALL( call, args, cleanup ) \
  { \
    int errcode = ::call args ; \
    if ( errcode != 0 ) { \
      cleanup ; \
      RETURN_ERRNO ; \
    } \
  }

#define CLEANUP_CALL( call, args, cleanup ) \
  { \
    int result = call args ; \
    if ( result != 0 ) { \
      cleanup ; \
      RETURN_ERRNO ; \
    } \
  }

// Define our realtime signals
#define SIG_LOOP_TERMINATE 	( SIGRTMIN + 3 )
#define SIG_USER_EVENT_1	( SIGRTMIN + 4 )
#define SIG_USER_EVENT_2	( SIGRTMIN + 5 )
#define SIG_USER_EVENT_3	( SIGRTMIN + 6 )
#define SIG_USER_EVENT_4	( SIGRTMIN + 7 )
#define SIG_USER_EVENT_5	( SIGRTMIN + 8 )
#define SIG_USER_EVENT_6	( SIGRTMIN + 9 )
#define SIG_USER_EVENT_7	( SIGRTMIN + 10 )
#define SIG_USER_EVENT_8	( SIGRTMIN + 11 )
#define SIG_AIO_1		( SIGRTMIN + 12 )
#define SIG_AIO_2		( SIGRTMIN + 13 )
#define SIG_AIO_3		( SIGRTMIN + 14 )
#define SIG_AIO_4		( SIGRTMIN + 15 )
#define SIG_AIO_5		( SIGRTMIN + 16 )
#define SIG_AIO_6		( SIGRTMIN + 17 )
#define SIG_AIO_7		( SIGRTMIN + 18 )
#define SIG_AIO_8		( SIGRTMIN + 19 )

namespace llvm {
namespace sys {

#if 0
inline void 
time_t2TimeValue( time_t t, TimeValue &tv )
{
  TimeValue::Seconds_Type seconds = t;
  seconds -= TimeValue::posix_zero_time.seconds();
  tv.set( seconds, 0 );
}

inline void
TimeValue2timespec( const TimeValue& tv, struct timespec & ts )
{
  uint64_t seconds;
  uint64_t nanos;
  tv.timespec_time( seconds, nanos );
  ts.tv_sec = static_cast<time_t>(seconds);
  ts.tv_nsec = static_cast<long int>(nanos);
}

inline void
timeval2TimeValue( struct timeval& tval, TimeValue& tv )
{
  TimeValue::Seconds_Type seconds = tval.tv_sec;
  seconds -= TimeValue::posix_zero_time.seconds();
  tv.set( seconds, tval.tv_usec * XPS_NANOSECONDS_PER_MICROSECOND );
}

extern pid_t		pid_;		///< This processes' process identification number
extern int 		pagesize_;	///< The virtual memory page size of this machine
extern long		arg_max_;	///< The maximum size in bytes of arguments to exec()
extern long		child_max_;	///< The maximum processes per user.
extern long		clk_tck_;	///< The number of clock ticks per second on this machine
extern long		stream_max_;	///< The maximum number of streams that can be opened
extern long		tzname_max_;	///< The maximum length of a timezone name
extern long		file_max_;	///< The maximum number of files that can be opened
extern long		line_max_;	///< The maximum length of an I/O line
extern long		job_control_;	///< Non-zero if POSIX job control enabled
extern long		saved_ids_;	///< ?
extern long		phys_pages_;	///< Total number of physical pages
extern long		avphys_pages_;	///< Average number of free physical pages

#endif

}
}
// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
