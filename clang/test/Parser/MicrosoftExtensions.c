// RUN: clang -fsyntax-only -verify -fms-extensions %s
__stdcall int func0();
int __stdcall func();
typedef int (__cdecl *tptr)();
void (*__fastcall fastpfunc)();
extern __declspec(dllimport) void __stdcall VarR4FromDec();
__declspec(deprecated) __declspec(deprecated) char * __cdecl ltoa( long _Val, char * _DstBuf, int _Radix);
__declspec(noalias) __declspec(restrict) void * __cdecl xxx( void * _Memory );
