// RUN: %clang_cc1 -triple i386-mingw32 -fsyntax-only -verify -fms-extensions  -Wno-missing-declarations -x objective-c++ %s
__stdcall int func0();
int __stdcall func();
typedef int (__cdecl *tptr)();
void (*__fastcall fastpfunc)();
struct __declspec(uuid("00000000-0000-0000-C000-000000000046")) __declspec(novtable) IUnknown {};
extern __declspec(dllimport) void __stdcall VarR4FromDec();
__declspec(deprecated) __declspec(deprecated) char * __cdecl ltoa( long _Val, char * _DstBuf, int _Radix);
__declspec(noalias) __declspec(restrict) void * __cdecl xxx( void * _Memory );
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
  __asm {
    mov eax, Bit
    mov ecx, Base
    lock bts [ecx], eax
    setc al
  };
}
_inline int foo99() { return 99; }

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
