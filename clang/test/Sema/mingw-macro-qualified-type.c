// Ensure that builtin attributes do not get treated as user defined macros to
// be weapped in macro qualified types. This addresses P41852.
//
// RUN: %clang_cc1 %s -triple i686-w64-mingw32 -fsyntax-only -verify
// expected-no-diagnostics

typedef int WINBOOL;
typedef unsigned int UINT_PTR, *PUINT_PTR;
typedef unsigned long long ULONG64, *PULONG64;
#define WINAPI __stdcall
#define CALLBACK __stdcall

typedef WINBOOL(CALLBACK WINAPI *PSYMBOLSERVERCALLBACKPROC)(UINT_PTR action, ULONG64 data, ULONG64 context);
