#ifndef STDINT_H
#define STDINT_H

#if defined(__arm__) || defined(__i386__) || defined(__mips__)
typedef unsigned int uint32_t;
typedef unsigned int uintptr_t;
#elif defined(__x86_64__)
typedef unsigned int uint32_t;
typedef unsigned long uintptr_t;
#else
#error "Unknown target architecture"
#endif

#endif /* STDINT_H */
