/*
 * Header file: memory.h
 *
 * Description:
 *	This header file is the autoconf replacement for memory.h (if it lives
 *	on the system).
 */

#ifndef _CONFIG_MEMORY_H
#define _CONFIG_MEMORY_H

#include "Config/config.h"

#ifdef HAVE_MEMORY_H
#include <memory.h>
#endif

#endif
