/*
 * Header file: types.h
 *
 * Description:
 *	This header file is the autoconf substitute for sys/types.h.  It
 *	includes it for us if it exists on this system.
 */

#ifndef _CONFIG_SYS_TYPES_H
#define _CONFIG_SYS_TYPES_H

#include "Config/config.h"

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#endif

