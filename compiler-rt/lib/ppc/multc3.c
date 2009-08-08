/* This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 */

#include "DD.h"
#include <math.h>

#if !defined(INFINITY) && defined(HUGE_VAL)
#define INFINITY HUGE_VAL
#endif /* INFINITY */

#define makeFinite(x)	{ \
							(x).hi = __builtin_copysign(isinf((x).hi) ? 1.0 : 0.0, (x).hi); \
							(x).lo = 0.0; \
						}

#define zeroNaN(x)		{ \
							if (isnan((x).hi)) { \
								(x).hi = __builtin_copysign(0.0, (x).hi); \
								(x).lo = 0.0; \
							} \
						}

long double __gcc_qadd(long double, long double);
long double __gcc_qsub(long double, long double);
long double __gcc_qmul(long double, long double);

long double _Complex
__multc3(long double a, long double b, long double c, long double d)
{
	long double ac = __gcc_qmul(a,c);
	long double bd = __gcc_qmul(b,d);
	long double ad = __gcc_qmul(a,d);
	long double bc = __gcc_qmul(b,c);
	
	DD real = { .ld = __gcc_qsub(ac,bd) };
	DD imag = { .ld = __gcc_qadd(ad,bc) };
	
	if (isnan(real.hi) && isnan(imag.hi))
	{
		int recalc = 0;
		
		DD aDD = { .ld = a };
		DD bDD = { .ld = b };
		DD cDD = { .ld = c };
		DD dDD = { .ld = d };
		
		if (isinf(aDD.hi) || isinf(bDD.hi))
		{
			makeFinite(aDD);
			makeFinite(bDD);
			zeroNaN(cDD);
			zeroNaN(dDD);
			recalc = 1;
		}
		
		if (isinf(cDD.hi) || isinf(dDD.hi))
		{
			makeFinite(cDD);
			makeFinite(dDD);
			zeroNaN(aDD);
			zeroNaN(bDD);
			recalc = 1;
		}
		
		if (!recalc)
		{
			DD acDD = { .ld = ac };
			DD bdDD = { .ld = bd };
			DD adDD = { .ld = ad };
			DD bcDD = { .ld = bc };
			
			if (isinf(acDD.hi) || isinf(bdDD.hi) || isinf(adDD.hi) || isinf(bcDD.hi))
			{
				zeroNaN(aDD);
				zeroNaN(bDD);
				zeroNaN(cDD);
				zeroNaN(dDD);
				recalc = 1;
			}
		}
		
		if (recalc)
		{
			real.hi = INFINITY * (aDD.hi*cDD.hi - bDD.hi*dDD.hi);
			real.lo = 0.0;
			imag.hi = INFINITY * (aDD.hi*dDD.hi + bDD.hi*cDD.hi);
			imag.lo = 0.0;
		}
	}
	
	long double _Complex z;
	__real__ z = real.ld;
	__imag__ z = imag.ld;
	
	return z;
}
