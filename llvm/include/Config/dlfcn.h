/*
 * Header file: dlfcn.h
 *
 * Description:
 *	This header file is the autoconf replacement for dlfcn.h (if it lives
 *	on the system).
 */

#ifndef _CONFIG_DLFCN_H
#define _CONFIG_DLFCN_H

#include "Config/config.h"

/*
 * According to the man pages on dlopen(), we sometimes need link.h.  So,
 * go grab it just in case.
 */
#ifdef HAVE_DLFCN_H
#include <dlfcn.h>

#ifdef HAVE_LINK_H
#include <link.h>
#endif

#endif

#endif
