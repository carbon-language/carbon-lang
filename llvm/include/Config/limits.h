/*
 * Header file: limits.h
 *
 * Description:
 *	This header file is the autoconf replacement for limits.h (if it lives
 *	on the system).
 */

#ifndef _CONFIG_LIMITS_H
#define _CONFIG_LIMITS_H

#include "Config/config.h"

#ifdef HAVE_LIMITS_H
#include <limits.h>
#endif

#endif
