/*
 * Header file: alloc.h
 *
 * Description:
 *	This header file includes the infamous alloc.h header file if the
 *	autoconf system has found it.  It hides all of the autoconf details
 *	from the rest of the application source code.
 */

#ifndef _CONFIG_ALLOC_H
#define _CONFIG_ALLOC_H

#include "Config/config.h"

/*
 * This is a modified version of that suggested by the Autoconf manual.
 *	1) The #pragma is indented so that pre-ANSI C compilers ignore it.
 *	2) If alloca.h cannot be found, then try stdlib.h.  Some platforms
 *	   (notably FreeBSD) defined alloca() there.
 */
#ifndef __GNUC__
#	ifdef HAVE_ALLOCA_H
#		include <alloca.h>
#	else
#		ifdef _AIX
 #			pragma alloca
#		else
#			ifndef alloca
				char * alloca ();
#			endif
#		endif
#	endif
#else
#	ifdef HAVE_ALLOCA_H
#		include <alloca.h>
#	else
#		ifdef HAVE_STDLIB_H
#			include <stdlib.h>
#		else
#			error "The function alloca() is required but not found!"
#		endif
#	endif
#endif

#endif

