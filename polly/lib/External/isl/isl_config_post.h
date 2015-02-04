#ifndef HAVE___ATTRIBUTE__
#define __attribute__(x)
#endif

#if (HAVE_DECL_FFS==0) && (HAVE_DECL___BUILTIN_FFS==1)
#define ffs __builtin_ffs
#endif

#ifdef GCC_WARN_UNUSED_RESULT
#define WARN_UNUSED	GCC_WARN_UNUSED_RESULT
#else
#define WARN_UNUSED
#endif
