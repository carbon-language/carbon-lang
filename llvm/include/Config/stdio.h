/*
 * Header file: stdio.h
 *
 * Description:
 *	This header file is the autoconf replacement for stdio.h (if it lives
 *	on the system).
 */

#ifndef _CONFIG_STDIO_H
#define _CONFIG_STDIO_H

#include "Config/config.h"

/*
 * Assume that stdio.h exists if autoconf find the ANSI C header files.
 * I'd think stdlib.h would be here to, but I guess not.
 */
#ifdef STDC_HEADERS
#include <stdio.h>
#endif

#endif
