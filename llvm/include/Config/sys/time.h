/*
 * Header file: time.h
 *
 * Description:
 *	This header file is the autoconf replacement for sys/time.h (if it
 *	lives on the system).
 */

#ifndef _CONFIG_SYS_TIME_H
#define _CONFIG_SYS_TIME_H

#include "Config/config.h"

#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif

#endif
