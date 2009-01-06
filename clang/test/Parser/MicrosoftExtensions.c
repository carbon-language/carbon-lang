// RUN: clang -fsyntax-only -verify -fms-extensions -x=objective-c++ %s
__stdcall int func0();
int __stdcall func();
typedef int (__cdecl *tptr)();
void (*__fastcall fastpfunc)();
extern __declspec(dllimport) void __stdcall VarR4FromDec();
__declspec(deprecated) __declspec(deprecated) char * __cdecl ltoa( long _Val, char * _DstBuf, int _Radix);
__declspec(noalias) __declspec(restrict) void * __cdecl xxx( void * _Memory );
typedef __w64 unsigned long ULONG_PTR, *PULONG_PTR;
void * __ptr64 PtrToPtr64(const void *p)
{
    return((void * __ptr64) (unsigned __int64) (ULONG_PTR)p );
}
__forceinline InterlockedBitTestAndSet (long *Base, long Bit)
{
    __asm {
           mov eax, Bit
           mov ecx, Base
           lock bts [ecx], eax
           setc al
    };
}

