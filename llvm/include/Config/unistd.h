/*
 * Header file: unistd.h
 *
 * Description:
 *	This header file is the autoconf replacement for unistd.h (if it lives
 *	on the system).
 */

#ifndef _CONFIG_UNISTD_H
#define _CONFIG_UNISTD_H

#include "Config/config.h"

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#endif
