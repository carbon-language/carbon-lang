/*
 * Header file: resource.h
 *
 * Description:
 *	This header file is the autoconf replacement for sys/resource.h (if it
 *	lives on the system).
 */

#ifndef _CONFIG_SYS_RESOURCE_H
#define _CONFIG_SYS_RESOURCE_H

#include "Config/config.h"

#ifdef HAVE_SYS_RESOURCE_H

/*
 * In LLVM, we use sys/resource.h to use getrusage() and maybe some other
 * stuff.  Some man pages say that you also need sys/time.h and unistd.h.
 * So, to be paranoid, we will try to include all three if possible.
 */
#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif

#include <sys/resource.h>

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#endif

#endif
