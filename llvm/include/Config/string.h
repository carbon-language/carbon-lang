/*
 * Header file: string.h
 *
 * Description:
 *	This header file is the autoconf replacement for string.h (if it lives
 *	on the system).
 */

#ifndef _CONFIG_STRING_H
#define _CONFIG_STRING_H

#include "Config/config.h"

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#endif
