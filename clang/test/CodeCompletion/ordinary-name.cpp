struct X { int x; };
void z(int);
typedef struct t TYPEDEF;

void foo() {
  int y = 17;
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:6:14 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
  // CHECK-CC1: COMPLETION: bool
  // CHECK-CC1-NEXT: COMPLETION: char
  // CHECK-CC1-NEXT: COMPLETION: class
  // CHECK-CC1-NEXT: COMPLETION: const
  // CHECK-CC1-NEXT: COMPLETION: Pattern : const_cast<<#type-id#>>(<#expression#>)
  // CHECK-CC1: COMPLETION: Pattern : delete <#expression#>
  // CHECK-CC1-NEXT: COMPLETION: Pattern : delete[] <#expression#>
  // CHECK-CC1-NEXT: COMPLETION: Pattern : do{<#statements#>
  // CHECK-CC1: COMPLETION: double
  // CHECK-CC1-NEXT: COMPLETION: Pattern : dynamic_cast<<#type-id#>>(<#expression#>)
  // CHECK-CC1-NEXT: COMPLETION: enum
  // CHECK-CC1-NEXT: COMPLETION: extern
  // CHECK-CC1-NEXT: COMPLETION: false
  // CHECK-CC1-NEXT: COMPLETION: float
  // CHECK-CC1-NEXT: COMPLETION: foo : [#void#]foo()
  // CHECK-CC1-NEXT: COMPLETION: Pattern : for(<#init-statement#>;<#condition#>;<#inc-expression#>){<#statements#>
  // CHECK-CC1: COMPLETION: Pattern : goto <#identifier#>;
  // CHECK-CC1-NEXT: COMPLETION: Pattern : if(<#condition#>){<#statements#>
  // CHECK-CC1: COMPLETION: int
  // CHECK-CC1-NEXT: COMPLETION: long
  // CHECK-CC1-NEXT: COMPLETION: Pattern : new <#type-id#>(<#expressions#>)
  // CHECK-CC1-NEXT: COMPLETION: Pattern : new <#type-id#>[<#size#>](<#expressions#>)
  // CHECK-CC1-NEXT: COMPLETION: operator
  // CHECK-CC1-NEXT: COMPLETION: Pattern : reinterpret_cast<<#type-id#>>(<#expression#>)
  // CHECK-CC1-NEXT: COMPLETION: Pattern : return;
  // CHECK-CC1-NEXT: COMPLETION: short
  // CHECK-CC1-NEXT: COMPLETION: signed
  // CHECK-CC1-NEXT: COMPLETION: Pattern : sizeof(<#expression-or-type#>)
  // CHECK-CC1-NEXT: COMPLETION: static
  // CHECK-CC1-NEXT: COMPLETION: Pattern : static_cast<<#type-id#>>(<#expression#>)
  // CHECK-CC1-NEXT: COMPLETION: struct
  // CHECK-CC1-NEXT: COMPLETION: Pattern : switch(<#condition#>){
  // CHECK-CC1: COMPLETION: t : t
  // CHECK-CC1-NEXT: COMPLETION: Pattern : throw <#expression#>
  // CHECK-CC1-NEXT: COMPLETION: true
  // CHECK-CC1-NEXT: COMPLETION: Pattern : try{<#statements#>
  // CHECK-CC1: COMPLETION: TYPEDEF : TYPEDEF
  // CHECK-CC1-NEXT: COMPLETION: typedef
  // CHECK-CC1-NEXT: COMPLETION: Pattern : typeid(<#expression-or-type#>)
  // CHECK-CC1-NEXT: COMPLETION: Pattern : typename <#qualified-id#>
  // CHECK-CC1-NEXT: COMPLETION: Pattern : typeof(<#expression-or-type#>)
  // CHECK-CC1-NEXT: COMPLETION: union
  // CHECK-CC1-NEXT: COMPLETION: unsigned
  // CHECK-CC1-NEXT: COMPLETION: Pattern : using namespace <#identifier#>;
  // CHECK-CC1-NEXT: COMPLETION: void
  // CHECK-CC1-NEXT: COMPLETION: volatile
  // CHECK-CC1-NEXT: COMPLETION: wchar_t
  // CHECK-CC1-NEXT: COMPLETION: Pattern : while(<#condition#>){<#statements#>
  // CHECK-CC1: COMPLETION: X : X
  // CHECK-CC1-NEXT: COMPLETION: y : [#int#]y
  // CHECK-CC1-NEXT: COMPLETION: z : [#void#]z(<#int#>)

  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:4:1 %s -o - | FileCheck -check-prefix=CHECK-CC2 %s
  // CHECK-CC2: COMPLETION: Pattern : asm(<#string-literal#>);
  // CHECK-CC2-NEXT: COMPLETION: bool
  // CHECK-CC2-NEXT: COMPLETION: char
  // CHECK-CC2-NEXT: COMPLETION: class
  // CHECK-CC2-NEXT: COMPLETION: const
  // CHECK-CC2-NEXT: COMPLETION: double
  // CHECK-CC2-NEXT: COMPLETION: enum
  // CHECK-CC2-NEXT: COMPLETION: extern
  // CHECK-CC2-NEXT: COMPLETION: float
  // CHECK-CC2-NEXT: COMPLETION: inline
  // CHECK-CC2-NEXT: COMPLETION: int
  // CHECK-CC2-NEXT: COMPLETION: long
  // CHECK-CC2-NEXT: COMPLETION: Pattern : namespace <#identifier#>{<#declarations#>
  // CHECK-CC2: COMPLETION: Pattern : namespace <#identifier#> = <#identifier#>;
  // CHECK-CC2-NEXT: COMPLETION: operator
  // CHECK-CC2-NEXT: COMPLETION: short
  // CHECK-CC2-NEXT: COMPLETION: signed
  // CHECK-CC2-NEXT: COMPLETION: static
  // CHECK-CC2-NEXT: COMPLETION: struct
  // CHECK-CC2-NEXT: COMPLETION: t : t
  // CHECK-CC2-NEXT: COMPLETION: Pattern : template <#declaration#>;
  // CHECK-CC2-NEXT: COMPLETION: Pattern : template<<#parameters#>>
  // CHECK-CC2-NEXT: COMPLETION: TYPEDEF : TYPEDEF
  // CHECK-CC2-NEXT: COMPLETION: typedef
  // CHECK-CC2-NEXT: COMPLETION: Pattern : typename <#qualified-id#>
  // CHECK-CC2-NEXT: COMPLETION: Pattern : typeof(<#expression-or-type#>)
  // CHECK-CC2-NEXT: COMPLETION: union
  // CHECK-CC2-NEXT: COMPLETION: unsigned
  // CHECK-CC2-NEXT: COMPLETION: Pattern : using namespace <#identifier#>;
  // CHECK-CC2-NEXT: COMPLETION: Pattern : using <#qualified-id#>;
  // CHECK-CC2-NEXT: COMPLETION: void
  // CHECK-CC2-NEXT: COMPLETION: volatile
  // CHECK-CC2-NEXT: COMPLETION: wchar_t
  // CHECK-CC2-NEXT: COMPLETION: X : X

  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:1:19 %s -o - | FileCheck -check-prefix=CHECK-CC3 %s
  // CHECK-CC3: COMPLETION: bool
  // CHECK-CC3-NEXT: COMPLETION: char
  // CHECK-CC3-NEXT: COMPLETION: class
  // CHECK-CC3-NEXT: COMPLETION: const
  // CHECK-CC3-NEXT: COMPLETION: double
  // CHECK-CC3-NEXT: COMPLETION: enum
  // CHECK-CC3-NEXT: COMPLETION: explicit
  // CHECK-CC3-NEXT: COMPLETION: extern
  // CHECK-CC3-NEXT: COMPLETION: float
  // CHECK-CC3-NEXT: COMPLETION: friend
  // CHECK-CC3-NEXT: COMPLETION: inline
  // CHECK-CC3-NEXT: COMPLETION: int
  // CHECK-CC3-NEXT: COMPLETION: long
  // CHECK-CC3-NEXT: COMPLETION: mutable
  // CHECK-CC3-NEXT: COMPLETION: operator
  // CHECK-CC3-NEXT: COMPLETION: Pattern : private: 
  // CHECK-CC3-NEXT: COMPLETION: Pattern : protected: 
  // CHECK-CC3-NEXT: COMPLETION: Pattern : public: 
  // CHECK-CC3-NEXT: COMPLETION: short
  // CHECK-CC3-NEXT: COMPLETION: signed
  // CHECK-CC3-NEXT: COMPLETION: static
  // CHECK-CC3-NEXT: COMPLETION: struct
  // CHECK-CC3-NEXT: COMPLETION: Pattern : template<<#parameters#>>
  // CHECK-CC3-NEXT: COMPLETION: typedef
  // CHECK-CC3-NEXT: COMPLETION: Pattern : typename <#qualified-id#>
  // CHECK-CC3-NEXT: COMPLETION: Pattern : typeof(<#expression-or-type#>)
  // CHECK-CC3-NEXT: COMPLETION: union
  // CHECK-CC3-NEXT: COMPLETION: unsigned
  // CHECK-CC3-NEXT: COMPLETION: Pattern : using <#qualified-id#>;
  // CHECK-CC3-NEXT: COMPLETION: virtual
  // CHECK-CC3-NEXT: COMPLETION: void
  // CHECK-CC3-NEXT: COMPLETION: volatile
  // CHECK-CC3-NEXT: COMPLETION: wchar_t
  // CHECK-CC3-NEXT: COMPLETION: X : X

  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:6:11 %s -o - | FileCheck -check-prefix=CHECK-CC4 %s
  // CHECK-CC4: COMPLETION: bool
  // CHECK-CC4-NEXT: COMPLETION: char
  // CHECK-CC4-NEXT: COMPLETION: class
  // CHECK-CC4-NEXT: COMPLETION: const
  // CHECK-CC4-NEXT: COMPLETION: Pattern : const_cast<<#type-id#>>(<#expression#>)
  // CHECK-CC4-NEXT: COMPLETION: Pattern : delete <#expression#>
  // CHECK-CC4-NEXT: COMPLETION: Pattern : delete[] <#expression#>
  // CHECK-CC4-NEXT: COMPLETION: double
  // CHECK-CC4-NEXT: COMPLETION: Pattern : dynamic_cast<<#type-id#>>(<#expression#>)
  // CHECK-CC4-NEXT: COMPLETION: enum
  // CHECK-CC4-NEXT: COMPLETION: false
  // CHECK-CC4-NEXT: COMPLETION: float
  // CHECK-CC4-NEXT: COMPLETION: foo : [#void#]foo()
  // CHECK-CC4-NEXT: COMPLETION: int
  // CHECK-CC4-NEXT: COMPLETION: long
  // CHECK-CC4-NEXT: COMPLETION: Pattern : new <#type-id#>(<#expressions#>)
  // CHECK-CC4-NEXT: COMPLETION: Pattern : new <#type-id#>[<#size#>](<#expressions#>)
  // CHECK-CC4-NEXT: COMPLETION: operator
  // CHECK-CC4-NEXT: COMPLETION: Pattern : reinterpret_cast<<#type-id#>>(<#expression#>)
  // CHECK-CC4-NEXT: COMPLETION: short
  // CHECK-CC4-NEXT: COMPLETION: signed
  // CHECK-CC4-NEXT: COMPLETION: Pattern : sizeof(<#expression-or-type#>)
  // CHECK-CC4-NEXT: COMPLETION: Pattern : static_cast<<#type-id#>>(<#expression#>)
  // CHECK-CC4-NEXT: COMPLETION: struct
  // CHECK-CC4-NEXT: COMPLETION: t : t
  // CHECK-CC4-NEXT: COMPLETION: Pattern : throw <#expression#>
  // CHECK-CC4-NEXT: COMPLETION: true
  // CHECK-CC4-NEXT: COMPLETION: TYPEDEF : TYPEDEF
  // CHECK-CC4-NEXT: COMPLETION: Pattern : typeid(<#expression-or-type#>)
  // CHECK-CC4-NEXT: COMPLETION: Pattern : typename <#qualified-id#>
  // CHECK-CC4-NEXT: COMPLETION: Pattern : typeof(<#expression-or-type#>)
  // CHECK-CC4-NEXT: COMPLETION: union
  // CHECK-CC4-NEXT: COMPLETION: unsigned
  // CHECK-CC4-NEXT: COMPLETION: void
  // CHECK-CC4-NEXT: COMPLETION: volatile
  // CHECK-CC4-NEXT: COMPLETION: wchar_t
  // CHECK-CC4-NEXT: COMPLETION: X : X
  // CHECK-CC4-NEXT: COMPLETION: y : [#int#]y
  // CHECK-CC4-NEXT: COMPLETION: z : [#void#]z(<#int#>)
