/* This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 */

#ifdef __x86_64__

#include "../int_lib.h"

float __floatdisf(int64_t a)
{
	return (float)a;
}

#endif /* __x86_64__ */
