/*
 * Header file: strings.h
 *
 * Description:
 *	This header file is the autoconf replacement for strings.h (if it lives
 *	on the system).
 */

#ifndef _CONFIG_STRINGS_H
#define _CONFIG_STRINGS_H

#include "Config/config.h"

#ifdef HAVE_STRINGS_H
#include <strings.h>
#endif

#endif
