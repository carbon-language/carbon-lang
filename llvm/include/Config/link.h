/*
 * Header file: link.h
 *
 * Description:
 *	This header file is the autoconf replacement for link.h (if it lives
 *	on the system).
 */

#ifndef _CONFIG_LINK_H
#define _CONFIG_LINK_H

#include "Config/config.h"

#ifdef HAVE_LINK_H
#include <link.h>
#endif

#endif
