/*
 * Header file: assert.h
 *
 * Description:
 *	This header file includes the assert.h header file if the
 *	autoconf system has found it.
 */

#ifndef _CONFIG_ASSERT_H
#define _CONFIG_ASSERT_H

#include "Config/config.h"

/*
 * This is the suggested use by the Autoconf manual.
 *	1) The #pragma is indented so that pre-ANSI C compilers ignore it.
 */
#ifdef HAVE_ASSERT_H
#include <assert.h>
#endif

#endif

