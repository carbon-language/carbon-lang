// RUN: %clang_cc1 -triple i386-mingw32 -fsyntax-only -verify -fms-extensions  -Wno-missing-declarations -x objective-c++ %s
__stdcall int func0();
int __stdcall func();
typedef int (__cdecl *tptr)();
void (*__fastcall fastpfunc)();
struct __declspec(uuid("00000000-0000-0000-C000-000000000046")) __declspec(novtable) IUnknown {}; /* expected-warning{{__declspec attribute 'novtable' is not supported}} */
extern __declspec(dllimport) void __stdcall VarR4FromDec();
__declspec(deprecated) __declspec(deprecated) char * __cdecl ltoa( long _Val, char * _DstBuf, int _Radix);
__declspec(safebuffers) __declspec(noalias) __declspec(restrict) void * __cdecl xxx( void * _Memory ); /* expected-warning{{__declspec attribute 'safebuffers' is not supported}} expected-warning{{__declspec attribute 'noalias' is not supported}} expected-warning{{__declspec attribute 'restrict' is not supported}} */
typedef __w64 unsigned long ULONG_PTR, *PULONG_PTR;

void * __ptr64 PtrToPtr64(const void *p)
{
  return((void * __ptr64) (unsigned __int64) (ULONG_PTR)p );
}
void * __ptr32 PtrToPtr32(const void *p)
{
  return((void * __ptr32) (unsigned __int32) (ULONG_PTR)p );
}

void __forceinline InterlockedBitTestAndSet (long *Base, long Bit)
{
  // FIXME: Re-enable this once MS inline asm stabilizes.
#if 0
  __asm {
    mov eax, Bit
    mov ecx, Base
    lock bts [ecx], eax
    setc al
  };
#endif
}

// Both inline and __forceinline is OK.
inline void __forceinline pr8264() {
}
__forceinline void inline pr8264_1() {
}
void inline __forceinline pr8264_2() {
}
void __forceinline inline pr8264_3() {
}
// But duplicate __forceinline causes warning.
void __forceinline __forceinline pr8264_4() {  // expected-warning{{duplicate '__forceinline' declaration specifier}}
}

_inline int foo99() { return 99; }

void test_ms_alignof_alias() {
  unsigned int s = _alignof(int);
  s = __builtin_alignof(int);
}

void *_alloca(int);

void foo() {
  __declspec(align(16)) int *buffer = (int *)_alloca(9);
}

typedef bool (__stdcall __stdcall *blarg)(int);

void local_callconv()
{
  bool (__stdcall *p)(int);
}

// Charify extension.
#define FOO(x) #@x
char x = FOO(a);

typedef enum E { e1 };


enum __declspec(deprecated) E2 { i, j, k }; // expected-note {{declared here}}
__declspec(deprecated) enum E3 { a, b, c } e; // expected-note {{declared here}}

void deprecated_enum_test(void)
{
  // Test to make sure the deprecated warning follows the right thing
  enum E2 e1;  // expected-warning {{'E2' is deprecated}}
  enum E3 e2; // No warning expected, the deprecation follows the variable
  enum E3 e3 = e;  // expected-warning {{'e' is deprecated}}
}

/* Microsoft attribute tests */
[repeatable][source_annotation_attribute( Parameter|ReturnValue )]
struct SA_Post{ SA_Post(); int attr; };

[returnvalue:SA_Post( attr=1)]
int foo1([SA_Post(attr=1)] void *param);



void ms_intrinsics(int a)
{
  __noop();
  __assume(a);
  __debugbreak();
}

struct __declspec(frobble) S1 {};	/* expected-warning {{unknown __declspec attribute 'frobble' ignored}} */
struct __declspec(12) S2 {};	/* expected-error {{__declspec attributes must be an identifier or string literal}} */
struct __declspec("testing") S3 {}; /* expected-warning {{__declspec attribute '"testing"' is not supported}} */

/* Ensure multiple declspec attributes are supported */
struct __declspec(align(8) deprecated) S4 {};

/* But multiple declspecs must still be legal */
struct __declspec(deprecated frobble "testing") S5 {};  /* expected-warning {{unknown __declspec attribute 'frobble' ignored}} expected-warning {{__declspec attribute '"testing"' is not supported}} */
struct __declspec(unknown(12) deprecated) S6 {};	/* expected-warning {{unknown __declspec attribute 'unknown' ignored}}*/

struct S7 {
	int foo() { return 12; }
	__declspec(property(get=foo) deprecated) int t; // expected-note {{declared here}}
};

/* Technically, this is legal (though it does nothing) */
__declspec() void quux( void ) {
  struct S7 s;
  int i = s.t;	/* expected-warning {{'t' is deprecated}} */
}

int * __sptr psp;
int * __uptr pup;
/* Either ordering is acceptable */
int * __ptr32 __sptr psp32;
int * __ptr32 __uptr pup32;
int * __sptr __ptr64 psp64;
int * __uptr __ptr64 pup64;

/* Legal to have nested pointer attributes */
int * __sptr * __ptr32 ppsp32;
