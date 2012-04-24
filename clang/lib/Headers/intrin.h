/*===---- intrin.h - Microsoft VS compatible X86 intrinsics -----------------===
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *===-----------------------------------------------------------------------===
 */

/* Unless we're compiling targeting MSVC platform, this header shouldn't even
 * *exist*. If there is a system header with the same name, defer to that,
 * etherwise produce an error for the user.
 */
#ifndef _MSC_VER
# if defined(__has_include_next) && __has_include_next(<intrin.h>)
#  include_next <intrin.h>
# else
#  error The <intrin.h> builtin header is for use when targeting Windows and \
         provides MSVC compatible intrinsic declarations. It shouldn't be used \
         on non-Windows targets. Instead, see <x86intrin.h> which is supported \
         by Clang, GCC, and ICC on all platforms.
# endif
#else /* _MSC_VER */

#ifndef __INTRIN_H
#define __INTRIN_H

/* These headers need to be provided by intrin.h in case users depend on any of
 * their contents. However, some of them are unavailable in freestanding
 * builds, so guard them appropriately.
 */
#if __STDC_HOSTED__
# include <crtdefs.h>
# include <setjmp.h>
#endif
#include <stddef.h>

/* Microsoft includes all of the intrinsics, and then restricts their
 * availability based on the particular target CPU; with Clang we rely on the
 * guarded includes used in our generic x86intrin header to pull in the
 * intrinsic declarations / definitions which should be available for the
 * target CPU variant.
 */
#include <x86intrin.h>

/* FIXME: We need to provide declarations for Microsoft-specific intrinsics in
 * addition to the chip-vendor intrinsics provided by x86intrin.h.
 *
 * The following is a list of the Microsoft-specific intrinsics that need
 * to be handled, separated by what header file they could be covered by.
 * However, some of these will require implementations not provided by other
 * header files.  Please keep this list up to date as you implement any of the
 * intrinsics.
 *
 * malloc.h
 * void * __cdecl _alloca(size_t);
 *
 *
 * math.h
 * int __cdecl abs(int);
 * double ceil(double);
 * long __cdecl labs(long);
 *
 *
 * conio.h
 * int __cdecl _inp(unsigned short);
 * int __cdecl inp(unsigned short);
 * unsigned long __cdecl _inpd(unsigned short);
 * unsigned long __cdecl inpd(unsigned short);
 * unsigned short __cdecl _inpw(unsigned short);
 * unsigned short __cdecl inpw(unsigned short);
 * int __cdecl _outp(unsigned short,int);
 * int __cdecl outp(unsigned short,int);
 * unsigned long __cdecl _outpd(unsigned short,unsigned long);
 * unsigned long __cdecl outpd(unsigned short,unsigned long);
 * unsigned short __cdecl _outpw(unsigned short,unsigned short);
 * unsigned short __cdecl outpw(unsigned short,unsigned short);
 *
 *
 * setjmp.h
 * void __cdecl longjmp(jmp_buf, int);
 * int __cdecl _setjmp(jmp_buf);
 * int __cdecl _setjmpex(jmp_buf);
 *
 *
 * stdlib.h
 * unsigned long __cdecl _lrotl( unsigned long, int);
 * unsigned long __cdecl _lrotr( unsigned long, int);
 * unsigned int __cdecl _rotl( unsigned int, int);
 * unsigned int __cdecl _rotr( unsigned int, int);
 * unsigned __int64 __cdecl _rotl64( unsigned __int64, int);
 * unsigned __int64 __cdecl _rotr64( unsigned __int64, int);
 * __int64 __cdecl _abs64(__int64);
 *
 *
 * memory.h
 * int __cdecl memcmp(const void *,const void *, size_t);
 * void * __cdecl memcpy(void *,const void *, size_t);
 * void * __cdecl memset(void *, int, size_t);
 *
 *
 * string.h
 * int __cdecl strcmp(const char *, const char *);
 * size_t __cdecl strlen(const char *);
 * char * __cdecl strset(char *, int);
 * wchar_t * __cdecl wcscat(wchar_t *, * const wchar_t *);
 * int __cdecl wcscmp(const wchar_t *, * const wchar_t *);
 * wchar_t * __cdecl wcscpy(wchar_t *, * const wchar_t *);
 * size_t __cdecl wcslen(const wchar_t *);
 * wchar_t * __cdecl _wcsset(wchar_t *, wchar_t);
 *
 *
 * intrin.h
 * All Architectures:
 * unsigned short __cdecl _byteswap_ushort(unsigned short);
 * unsigned long __cdecl _byteswap_ulong(unsigned long);
 * unsigned __int64 __cdecl _byteswap_uint64(unsigned __int64);
 * void __cdecl __debugbreak(void);
 *
 *
 * All Intel (x86, x64):
 * void __cdecl _disable(void);
 * __int64 __emul(int,int);
 * unsigned __int64 __emulu(unsigned int,unsigned int);
 * void __cdecl _enable(void);
 * long __cdecl _InterlockedDecrement(long volatile *);
 * long _InterlockedExchange(long volatile *, long);
 * short _InterlockedExchange16(short volatile *, short);
 * char _InterlockedExchange8(char volatile *, char);
 * long _InterlockedExchangeAdd(long volatile *, long);
 * short _InterlockedExchangeAdd16(short volatile *, short);
 * char _InterlockedExchangeAdd8(char volatile *, char);
 * long _InterlockedCompareExchange (long volatile *, long, long);
 * long __cdecl _InterlockedIncrement(long volatile *);
 * long _InterlockedOr(long volatile *, long);
 * char _InterlockedOr8(char volatile *, char);
 * short _InterlockedOr16(short volatile *, short);
 * long _InterlockedXor(long volatile *, long);
 * char _InterlockedXor8(char volatile *, char);
 * short _InterlockedXor16(short volatile *, short);
 * long _InterlockedAnd(long volatile *, long);
 * char _InterlockedAnd8(char volatile *, char);
 * short _InterlockedAnd16(short volatile *, short);
 * unsigned __int64 * __ll_lshift(unsigned __int64,int);
 * __int64 * __ll_rshift(__int64,int);
 * void * _ReturnAddress(void);
 * unsigned __int64 __ull_rshift(unsigned __int64,int);
 * void * _AddressOfReturnAddress(void);
 * void _WriteBarrier(void);
 * void _ReadWriteBarrier(void);
 * unsigned __int64 __rdtsc(void);
 * void __movsb(unsigned char *, unsigned char const *, size_t);
 * void __movsw(unsigned short *, unsigned short const *, size_t);
 * void __movsd(unsigned long *, unsigned long const *, size_t);
 * unsigned char __inbyte(unsigned short);
 * unsigned short __inword(unsigned short);
 * unsigned long __indword(unsigned short);
 * void __outbyte(unsigned short, unsigned char);
 * void __outword(unsigned short, unsigned short);
 * void __outdword(unsigned short, unsigned long);
 * void __inbytestring(unsigned short, unsigned char *, unsigned long);
 * void __inwordstring(unsigned short, unsigned short *, unsigned long);
 * void __indwordstring(unsigned short, unsigned long *, unsigned long);
 * void __outbytestring(unsigned short, unsigned char *, unsigned long);
 * void __outwordstring(unsigned short, unsigned short *, unsigned long);
 * void __outdwordstring(unsigned short, unsigned long *, unsigned long);
 * unsigned int __getcallerseflags();
 * void __vmx_vmptrst(unsigned __int64 *);
 * void __vmx_off(void);
 * void __svm_clgi(void);
 * void __svm_invlpga(void*, int);
 * void __svm_skinit(int);
 * void __svm_stgi(void);
 * void __svm_vmload(size_t);
 * void __svm_vmrun(size_t);
 * void __svm_vmsave(size_t);
 * void __halt(void);
 * void __sidt(void*);
 * void __lidt(void*);
 * void __ud2(void);
 * void __nop(void);
 * void __stosb(unsigned char *, unsigned char, size_t);
 * void __stosw(unsigned short *, unsigned short, size_t);
 * void __stosd(unsigned long *, unsigned long, size_t);
 * unsigned char _interlockedbittestandset(long volatile *, long);
 * unsigned char _interlockedbittestandreset(long volatile *, long);
 * void __cpuid(int[4], int);
 * void __cpuidex(int[4], int, int);
 * unsigned long __segmentlimit(unsigned long);
 * void __int2c(void);
 * char _InterlockedCompareExchange8(char volatile *, char, char);
 * unsigned short __lzcnt16(unsigned short);
 * unsigned int __lzcnt(unsigned int);
 * unsigned short __popcnt16(unsigned short);
 * unsigned int __popcnt(unsigned int);
 * __m128i _mm_extract_si64(__m128i,__m128i);
 * __m128i _mm_extracti_si64(__m128i, int, int);
 * __m128i _mm_insert_si64(__m128i,__m128i);
 * __m128i _mm_inserti_si64(__m128i, __m128i, int, int);
 * void _mm_stream_sd(double*,__m128d);
 * void _mm_stream_ss(float*,__m128);
 * unsigned __int64 __rdtscp(unsigned int*);
 *
 *
 * Intel x64 Only:
 * __int64 _InterlockedDecrement64(__int64 volatile *);
 * __int64 _InterlockedExchange64(__int64 volatile *, __int64);
 * void * _InterlockedExchangePointer(void * volatile *, void *);
 * __int64 _InterlockedExchangeAdd64(__int64 volatile *, __int64);
 * __int64 _InterlockedCompareExchange64(__int64 volatile *, __int64, __int64);
 * void *_InterlockedCompareExchangePointer (void * volatile *, void *, void *);
 * __int64 _InterlockedIncrement64(__int64 volatile *);
 * __int64 _InterlockedOr64(__int64 volatile *, __int64);
 * __int64 _InterlockedXor64(__int64 volatile *, __int64);
 * __int64 _InterlockedAnd64(__int64 volatile *, __int64);
 * void __faststorefence(void);
 * __int64 __mulh(__int64,__int64);
 * unsigned __int64 __umulh(unsigned __int64,unsigned __int64);
 * unsigned __int64 __readeflags(void);
 * void __writeeflags(unsigned __int64);
 * void __movsq(unsigned long long *, unsigned long long const *, size_t);
 * unsigned char __vmx_vmclear(unsigned __int64*);
 * unsigned char __vmx_vmlaunch(void);
 * unsigned char __vmx_vmptrld(unsigned __int64*);
 * unsigned char __vmx_vmread(size_t, size_t*);
 * unsigned char __vmx_vmresume(void);
 * unsigned char __vmx_vmwrite(size_t, size_t);
 * unsigned char __vmx_on(unsigned __int64*);
 * void __stosq(unsigned __int64 *, * unsigned __int64, size_t);
 * unsigned char _interlockedbittestandset64(__int64 volatile *, __int64);
 * unsigned char _interlockedbittestandreset64(__int64 volatile *, __int64);
 * short _InterlockedCompareExchange16_np(short volatile *, short, short);
 * long _InterlockedCompareExchange_np (long volatile *, long, long);
 * __int64 _InterlockedCompareExchange64_np(__int64 volatile *, __int64, __int64);
 * void *_InterlockedCompareExchangePointer_np (void * volatile *, void *, void *);
 * unsigned char _InterlockedCompareExchange128(__int64 volatile *, __int64, __int64, __int64 *);
 * unsigned char _InterlockedCompareExchange128_np(__int64 volatile *, __int64, __int64, __int64 *);
 * long _InterlockedAnd_np(long volatile *, long);
 * char _InterlockedAnd8_np(char volatile *, char);
 * short _InterlockedAnd16_np(short volatile *, short);
 * __int64 _InterlockedAnd64_np(__int64 volatile *, __int64);
 * long _InterlockedOr_np(long volatile *, long);
 * char _InterlockedOr8_np(char volatile *, char);
 * short _InterlockedOr16_np(short volatile *, short);
 * __int64 _InterlockedOr64_np(__int64 volatile *, __int64);
 * long _InterlockedXor_np(long volatile *, long);
 * char _InterlockedXor8_np(char volatile *, char);
 * short _InterlockedXor16_np(short volatile *, short);
 * __int64 _InterlockedXor64_np(__int64 volatile *, __int64);
 * unsigned __int64 __lzcnt64(unsigned __int64);
 * unsigned __int64 __popcnt64(unsigned __int64);
 *
 *
 * Intel x86 Only:
 * long _InterlockedAddLargeStatistic(__int64 volatile *, long);
 * unsigned __readeflags(void);
 * void __writeeflags(unsigned);
 * void __addfsbyte(unsigned long, unsigned char);
 * void __addfsword(unsigned long, unsigned short);
 * void __addfsdword(unsigned long, unsigned long);
 * unsigned char __readfsbyte(unsigned long);
 * unsigned short __readfsword(unsigned long);
 * unsigned long __readfsdword(unsigned long);
 * unsigned __int64 __readfsqword(unsigned long);
 * void __writefsbyte(unsigned long, unsigned char);
 * void __writefsword(unsigned long, unsigned short);
 * void __writefsdword(unsigned long, unsigned long);
 * void __writefsqword(unsigned long, unsigned __int64);
 *
 *
 * Win64, 64-bit compilers only:
 * unsigned char _bittest(long const *, long);
 * unsigned char _bittestandset(long *, long);
 * unsigned char _bittestandreset(long *, long);
 * unsigned char _bittestandcomplement(long *, long);
 * unsigned char _bittest64(__int64 const *, __int64);
 * unsigned char _bittestandset64(__int64 *, __int64);
 * unsigned char _bittestandreset64(__int64 *, __int64);
 * unsigned char _bittestandcomplement64(__int64 *, __int64);
 * unsigned char _BitScanForward(unsigned long*, unsigned long);
 * unsigned char _BitScanReverse(unsigned long*, unsigned long);
 * unsigned char _BitScanForward64(unsigned long*, unsigned __int64);
 * unsigned char _BitScanReverse64(unsigned long*, unsigned __int64);
 * unsigned __int64 __shiftleft128(unsigned __int64, unsigned __int64, unsigned char);
 * unsigned __int64 __shiftright128(unsigned __int64, unsigned __int64, unsigned char);
 * unsigned __int64 _umul128(unsigned __int64, unsigned __int64, unsigned __int64 *);
 * __int64 _mul128(__int64, __int64, __int64 *);
 * void _ReadBarrier(void);
 * unsigned char _rotr8(unsigned char, unsigned char);
 * unsigned short _rotr16(unsigned short, unsigned char);
 * unsigned char _rotl8(unsigned char, unsigned char);
 * unsigned short _rotl16(unsigned short, unsigned char);
 * short _InterlockedIncrement16(short volatile *);
 * short _InterlockedDecrement16(short volatile *);
 * short _InterlockedCompareExchange16(short volatile *, short, short);
 *
 *
 * Kernel-Only:
 * unsigned __int64 __readcr0(void);
 * unsigned __int64 __readcr2(void);
 * unsigned __int64 __readcr3(void);
 * unsigned __int64 __readcr4(void);
 * unsigned __int64 __readcr8(void);
 * unsigned long __readcr0(void);
 * unsigned long __readcr2(void);
 * unsigned long __readcr3(void);
 * unsigned long __readcr4(void);
 * unsigned long __readcr8(void);
 * void __writecr0(unsigned __int64);
 * void __writecr3(unsigned __int64);
 * void __writecr4(unsigned __int64);
 * void __writecr8(unsigned __int64);
 * void __writecr0(unsigned);
 * void __writecr3(unsigned);
 * void __writecr4(unsigned);
 * void __writecr8(unsigned);
 * unsigned __int64 __readdr(unsigned int);
 * unsigned __readdr(unsigned int);
 * void __writedr(unsigned int, unsigned __int64);
 * void __writedr(unsigned int, unsigned);
 * void __wbinvd(void);
 * void __invlpg(void*);
 * unsigned __int64 __readmsr(unsigned long);
 * void __writemsr(unsigned long, unsigned __int64);
 * unsigned char __readgsbyte(unsigned long);
 * unsigned short __readgsword(unsigned long);
 * unsigned long __readgsdword(unsigned long);
 * unsigned __int64 __readgsqword(unsigned long);
 * void __writegsbyte(unsigned long, unsigned char);
 * void __writegsword(unsigned long, unsigned short);
 * void __writegsdword(unsigned long, unsigned long);
 * void __writegsqword(unsigned long, unsigned __int64);
 * void __incfsbyte(unsigned long);
 * void __incfsword(unsigned long);
 * void __incfsdword(unsigned long);
 * void __addgsbyte(unsigned long, unsigned char);
 * void __addgsword(unsigned long, unsigned short);
 * void __addgsdword(unsigned long, unsigned long);
 * void __addgsqword(unsigned long, unsigned __int64);
 * void __incgsbyte(unsigned long);
 * void __incgsword(unsigned long);
 * void __incgsdword(unsigned long);
 * void __incgsqword(unsigned long);
 * unsigned __int64 __readpmc(unsigned long);
 *
 *
 * Entirely Undocumented on MSDN:
 * void __nvreg_save_fence(void);
 * void __nvreg_restore_fence(void);
 */

#endif /* __INTRIN_H */

#endif /* _MSC_VER */
