/*
 * Header file: mman.h
 *
 * Description:
 *	This header file includes the headers needed for the mmap() system/
 *	function call.  It also defines some macros so that all of our calls
 *	to mmap() can act (more or less) the same, regardless of platform.
 */

#ifndef _CONFIG_MMAN_H
#define _CONFIG_MMAN_H

#include "Config/config.h"

#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif

#ifndef HAVE_MMAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

#endif

