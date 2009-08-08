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

long double __gcc_qadd(long double, long double);
long double __gcc_qsub(long double, long double);
long double __gcc_qmul(long double, long double);
long double __gcc_qdiv(long double, long double);

long double _Complex
__divtc3(long double a, long double b, long double c, long double d)
{
	DD cDD = { .ld = c };
	DD dDD = { .ld = d };
	
	int ilogbw = 0;
	const double logbw = logb(__builtin_fmax( __builtin_fabs(cDD.hi), __builtin_fabs(dDD.hi) ));
	
	if (isfinite(logbw))
	{
		ilogbw = (int)logbw;
		
		cDD.hi = scalbn(cDD.hi, -ilogbw);
		cDD.lo = scalbn(cDD.lo, -ilogbw);
		dDD.hi = scalbn(dDD.hi, -ilogbw);
		dDD.lo = scalbn(dDD.lo, -ilogbw);
	}
	
	const long double denom = __gcc_qadd(__gcc_qmul(cDD.ld, cDD.ld), __gcc_qmul(dDD.ld, dDD.ld));
	const long double realNumerator = __gcc_qadd(__gcc_qmul(a,cDD.ld), __gcc_qmul(b,dDD.ld));
	const long double imagNumerator = __gcc_qsub(__gcc_qmul(b,cDD.ld), __gcc_qmul(a,dDD.ld));
	
	DD real = { .ld = __gcc_qdiv(realNumerator, denom) };
	DD imag = { .ld = __gcc_qdiv(imagNumerator, denom) };
	
	real.hi = scalbn(real.hi, -ilogbw);
	real.lo = scalbn(real.lo, -ilogbw);
	imag.hi = scalbn(imag.hi, -ilogbw);
	imag.lo = scalbn(imag.lo, -ilogbw);
	
	if (isnan(real.hi) && isnan(imag.hi))
	{
		DD aDD = { .ld = a };
		DD bDD = { .ld = b };
		DD rDD = { .ld = denom };
		
		if ((rDD.hi == 0.0) && (!isnan(aDD.hi) || !isnan(bDD.hi)))
		{
			real.hi = __builtin_copysign(INFINITY,cDD.hi) * aDD.hi;
			real.lo = 0.0;
			imag.hi = __builtin_copysign(INFINITY,cDD.hi) * bDD.hi;
			imag.lo = 0.0;
		}
		
		else if ((isinf(aDD.hi) || isinf(bDD.hi)) && isfinite(cDD.hi) && isfinite(dDD.hi))
		{
			makeFinite(aDD);
			makeFinite(bDD);
			real.hi = INFINITY * (aDD.hi*cDD.hi + bDD.hi*dDD.hi);
			real.lo = 0.0;
			imag.hi = INFINITY * (bDD.hi*cDD.hi - aDD.hi*dDD.hi);
			imag.lo = 0.0;
		}
		
		else if ((isinf(cDD.hi) || isinf(dDD.hi)) && isfinite(aDD.hi) && isfinite(bDD.hi))
		{
			makeFinite(cDD);
			makeFinite(dDD);
			real.hi = __builtin_copysign(0.0,(aDD.hi*cDD.hi + bDD.hi*dDD.hi));
			real.lo = 0.0;
			imag.hi = __builtin_copysign(0.0,(bDD.hi*cDD.hi - aDD.hi*dDD.hi));
			imag.lo = 0.0;
		}
	}
	
	long double _Complex z;
	__real__ z = real.ld;
	__imag__ z = imag.ld;
	
	return z;
}
