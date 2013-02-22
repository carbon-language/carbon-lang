// RUN: %clang_cc1 -triple i686-pc-win32 -cxx-abi microsoft -fms-extensions -verify %s

// Pointers to free functions
void            free_func_default();
void __cdecl    free_func_cdecl();
void __stdcall  free_func_stdcall(); // expected-note {{previous declaration is here}}
void __fastcall free_func_fastcall(); // expected-note 2 {{previous declaration is here}}

void __cdecl    free_func_default(); // expected-note 2 {{previous declaration is here}}
void __stdcall  free_func_default(); // expected-error {{function declared 'stdcall' here was previously declared without calling convention}}
void __fastcall free_func_default(); // expected-error {{function declared 'fastcall' here was previously declared without calling convention}}

void            free_func_cdecl(); // expected-note 2 {{previous declaration is here}}
void __stdcall  free_func_cdecl(); // expected-error {{function declared 'stdcall' here was previously declared 'cdecl'}}
void __fastcall free_func_cdecl(); // expected-error {{function declared 'fastcall' here was previously declared 'cdecl'}}

void __cdecl    free_func_stdcall(); // expected-error {{function declared 'cdecl' here was previously declared 'stdcall'}}
void            free_func_stdcall(); // expected-note {{previous declaration is here}}
void __fastcall free_func_stdcall(); // expected-error {{function declared 'fastcall' here was previously declared 'stdcall'}}

void __cdecl    free_func_fastcall(); // expected-error {{function declared 'cdecl' here was previously declared 'fastcall'}}
void __stdcall  free_func_fastcall(); // expected-error {{function declared 'stdcall' here was previously declared 'fastcall'}}
void            free_func_fastcall();

// Overloaded functions may have different calling conventions
void __fastcall free_func_default(int);
void __cdecl    free_func_default(int *);

void __thiscall free_func_cdecl(char *);
void __cdecl    free_func_cdecl(double);


// Pointers to member functions
struct S {
  void            member_default1(); // expected-note {{previous declaration is here}}
  void            member_default2();
  void __cdecl    member_cdecl1();
  void __cdecl    member_cdecl2(); // expected-note {{previous declaration is here}}
  void __thiscall member_thiscall1();
  void __thiscall member_thiscall2(); // expected-note {{previous declaration is here}}
  
  // Static member functions can't be __thiscall
  static void            static_member_default1();
  static void            static_member_default2(); // expected-note {{previous declaration is here}}
  static void __cdecl    static_member_cdecl1();
  static void __cdecl    static_member_cdecl2(); // expected-note {{previous declaration is here}}
  static void __stdcall  static_member_stdcall1();
  static void __stdcall  static_member_stdcall2();

  // Variadic functions can't be other than default or __cdecl
  void            member_variadic_default(int x, ...);
  void __cdecl    member_variadic_cdecl(int x, ...);

  static void            static_member_variadic_default(int x, ...);
  static void __cdecl    static_member_variadic_cdecl(int x, ...);
};

void __cdecl    S::member_default1() {} // expected-error {{function declared 'cdecl' here was previously declared without calling convention}}
void __thiscall S::member_default2() {}

void            S::member_cdecl1() {}
void __thiscall S::member_cdecl2() {} // expected-error {{function declared 'thiscall' here was previously declared 'cdecl'}}

void            S::member_thiscall1() {}
void __cdecl    S::member_thiscall2() {} // expected-error {{function declared 'cdecl' here was previously declared 'thiscall'}}

void __cdecl    S::static_member_default1() {}
void __stdcall  S::static_member_default2() {} // expected-error {{function declared 'stdcall' here was previously declared without calling convention}}

void            S::static_member_cdecl1() {}
void __stdcall  S::static_member_cdecl2() {} // expected-error {{function declared 'stdcall' here was previously declared 'cdecl'}}

void __cdecl    S::member_variadic_default(int x, ...) {
  (void)x;
}
void            S::member_variadic_cdecl(int x, ...) {
  (void)x;
}

void __cdecl    S::static_member_variadic_default(int x, ...) {
  (void)x;
}
void            S::static_member_variadic_cdecl(int x, ...) {
  (void)x;
}

