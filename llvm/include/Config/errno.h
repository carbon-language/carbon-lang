/*
 * Header file: errno.h
 *
 * Description:
 *	This header file is the autoconf replacement for errno.h (if it lives
 *	on the system).
 */

#ifndef _CONFIG_ERRNO_H
#define _CONFIG_ERRNO_H

#include "Config/config.h"

#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif

#endif
