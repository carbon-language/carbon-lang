/*
 * Header file: fcntl.h
 *
 * Description:
 *	This header file is the autoconf replacement for fcntl.h (if it lives
 *	on the system).
 */

#ifndef _CONFIG_FCNTL_H
#define _CONFIG_FCNTL_H

#include "Config/config.h"

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#endif
