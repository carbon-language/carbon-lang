struct X { int x; };
void z(int);
typedef struct t TYPEDEF;

void foo() {
  int y = 17;
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:6:14 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
  // CHECK-CC1: COMPLETION: y : 0 : [#int#]y
  // CHECK-CC1-NEXT: COMPLETION: foo : 2 : [#void#]foo()
  // CHECK-CC1-NEXT: COMPLETION: t : 2 : t
  // CHECK-CC1-NEXT: COMPLETION: TYPEDEF : 2 : TYPEDEF
  // CHECK-CC1-NEXT: COMPLETION: X : 2 : X
  // CHECK-CC1-NOT: x
  // CHECK-CC1-NEXT: COMPLETION: z : 2 : [#void#]z(<#int#>)
  // CHECK-CC1-NEXT: COMPLETION: bool : 3
  // CHECK-CC1-NEXT: COMPLETION: char : 3
  // CHECK-CC1-NEXT: COMPLETION: class : 3
  // CHECK-CC1-NEXT: COMPLETION: const : 3
  // CHECK-CC1-NEXT: COMPLETION: Pattern : 3 : const_cast<<#type-id#>>(<#expression#>)
  // CHECK-CC1-NEXT: COMPLETION: Pattern : 3 : delete <#expression#>
  // CHECK-CC1-NEXT: COMPLETION: Pattern : 3 : delete[] <#expression#>
  // CHECK-CC1-NEXT: COMPLETION: Pattern : 3 : do{<#statements#>
  // CHECK-CC1: COMPLETION: double : 3
  // CHECK-CC1-NEXT: COMPLETION: Pattern : 3 : dynamic_cast<<#type-id#>>(<#expression#>)
  // CHECK-CC1-NEXT: COMPLETION: enum : 3
  // CHECK-CC1-NEXT: COMPLETION: extern : 3
  // CHECK-CC1-NEXT: COMPLETION: false : 3
  // CHECK-CC1-NEXT: COMPLETION: float : 3
  // CHECK-CC1-NEXT: COMPLETION: Pattern : 3 : for(<#init-statement#>;<#condition#>;<#inc-expression#>){<#statements#>
  // CHECK-CC1: COMPLETION: Pattern : 3 : goto <#identifier#>;
  // CHECK-CC1-NEXT: COMPLETION: Pattern : 3 : if(<#condition#>){<#statements#>
  // CHECK-CC1: COMPLETION: int : 3
  // CHECK-CC1-NEXT: COMPLETION: long : 3
  // CHECK-CC1-NEXT: COMPLETION: Pattern : 3 : new <#type-id#>(<#expressions#>)
  // CHECK-CC1-NEXT: COMPLETION: Pattern : 3 : new <#type-id#>[<#size#>](<#expressions#>)
  // CHECK-CC1-NEXT: COMPLETION: operator : 3
  // CHECK-CC1-NEXT: COMPLETION: Pattern : 3 : reinterpret_cast<<#type-id#>>(<#expression#>)
  // CHECK-CC1-NEXT: COMPLETION: Pattern : 3 : return;
  // CHECK-CC1-NEXT: COMPLETION: short : 3
  // CHECK-CC1-NEXT: COMPLETION: signed : 3
  // CHECK-CC1-NEXT: COMPLETION: Pattern : 3 : sizeof(<#expression-or-type#>)
  // CHECK-CC1-NEXT: COMPLETION: static : 3
  // CHECK-CC1-NEXT: COMPLETION: Pattern : 3 : static_cast<<#type-id#>>(<#expression#>)
  // CHECK-CC1-NEXT: COMPLETION: struct : 3
  // CHECK-CC1-NEXT: COMPLETION: Pattern : 3 : switch(<#condition#>){
  // CHECK-CC1: COMPLETION: Pattern : 3 : throw <#expression#>
  // CHECK-CC1-NEXT: COMPLETION: true : 3
  // CHECK-CC1-NEXT: COMPLETION: Pattern : 3 : try{<#statements#>
  // CHECK-CC1: COMPLETION: typedef : 3
  // CHECK-CC1-NEXT: COMPLETION: Pattern : 3 : typeid(<#expression-or-type#>)
  // CHECK-CC1-NEXT: COMPLETION: Pattern : 3 : typename <#qualified-id#>
  // CHECK-CC1-NEXT: COMPLETION: Pattern : 3 : typeof(<#expression-or-type#>)
  // CHECK-CC1-NEXT: COMPLETION: union : 3
  // CHECK-CC1-NEXT: COMPLETION: unsigned : 3
  // CHECK-CC1-NEXT: COMPLETION: Pattern : 3 : using namespace <#identifier#>;
  // CHECK-CC1-NEXT: COMPLETION: void : 3
  // CHECK-CC1-NEXT: COMPLETION: volatile : 3
  // CHECK-CC1-NEXT: COMPLETION: wchar_t : 3
  // CHECK-CC1-NEXT: COMPLETION: Pattern : 3 : while(<#condition#>){<#statements#>
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:4:1 %s -o - | FileCheck -check-prefix=CHECK-CC2 %s
  // CHECK-CC2: COMPLETION: t : 1 : t
  // CHECK-CC2-NEXT: COMPLETION: TYPEDEF : 1 : TYPEDEF
  // CHECK-CC2-NEXT: COMPLETION: X : 1 : X
  // CHECK-CC2-NOT: COMPLETION: z
  // CHECK-CC2-NEXT: COMPLETION: Pattern : 2 : asm(<#string-literal#>);
  // CHECK-CC2-NEXT: COMPLETION: bool : 2
  // CHECK-CC2-NEXT: COMPLETION: char : 2
  // CHECK-CC2-NEXT: COMPLETION: class : 2
  // CHECK-CC2-NEXT: COMPLETION: const : 2
  // CHECK-CC2-NEXT: COMPLETION: double : 2
  // CHECK-CC2-NEXT: COMPLETION: enum : 2
  // CHECK-CC2-NEXT: COMPLETION: extern : 2
  // CHECK-CC2-NEXT: COMPLETION: float : 2
  // CHECK-CC2-NEXT: COMPLETION: inline : 2
  // CHECK-CC2-NEXT: COMPLETION: int : 2
  // CHECK-CC2-NEXT: COMPLETION: long : 2
  // CHECK-CC2-NEXT: COMPLETION: Pattern : 2 : namespace <#identifier#>{<#declarations#>
  // CHECK-CC2: COMPLETION: Pattern : 2 : namespace <#identifier#> = <#identifier#>;
  // CHECK-CC2-NEXT: COMPLETION: operator : 2
  // CHECK-CC2-NEXT: COMPLETION: short : 2
  // CHECK-CC2-NEXT: COMPLETION: signed : 2
  // CHECK-CC2-NEXT: COMPLETION: static : 2
  // CHECK-CC2-NEXT: COMPLETION: struct : 2
  // CHECK-CC2-NEXT: COMPLETION: Pattern : 2 : template <#declaration#>;
  // CHECK-CC2-NEXT: COMPLETION: Pattern : 2 : template<<#parameters#>>
  // CHECK-CC2-NEXT: COMPLETION: typedef : 2
  // CHECK-CC2-NEXT: COMPLETION: Pattern : 2 : typename <#qualified-id#>
  // CHECK-CC2-NEXT: COMPLETION: Pattern : 2 : typeof(<#expression-or-type#>)
  // CHECK-CC2-NEXT: COMPLETION: union : 2
  // CHECK-CC2-NEXT: COMPLETION: unsigned : 2
  // CHECK-CC2-NEXT: COMPLETION: Pattern : 2 : using namespace <#identifier#>;
  // CHECK-CC2-NEXT: COMPLETION: Pattern : 2 : using <#qualified-id#>;
  // CHECK-CC2-NEXT: COMPLETION: void : 2
  // CHECK-CC2-NEXT: COMPLETION: volatile : 2
  // CHECK-CC2-NEXT: COMPLETION: wchar_t : 2
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:1:19 %s -o - | FileCheck -check-prefix=CHECK-CC3 %s
  // CHECK-CC3: COMPLETION: X : 1 : X
  // CHECK-CC3-NEXT: COMPLETION: bool : 4
  // CHECK-CC3-NEXT: COMPLETION: char : 4
  // CHECK-CC3-NEXT: COMPLETION: class : 4
  // CHECK-CC3-NEXT: COMPLETION: const : 4
  // CHECK-CC3-NEXT: COMPLETION: double : 4
  // CHECK-CC3-NEXT: COMPLETION: enum : 4
  // CHECK-CC3-NEXT: COMPLETION: explicit : 4
  // CHECK-CC3-NEXT: COMPLETION: extern : 4
  // CHECK-CC3-NEXT: COMPLETION: float : 4
  // CHECK-CC3-NEXT: COMPLETION: friend : 4
  // CHECK-CC3-NEXT: COMPLETION: inline : 4
  // CHECK-CC3-NEXT: COMPLETION: int : 4
  // CHECK-CC3-NEXT: COMPLETION: long : 4
  // CHECK-CC3-NEXT: COMPLETION: mutable : 4
  // CHECK-CC3-NEXT: COMPLETION: operator : 4
  // CHECK-CC3-NEXT: COMPLETION: Pattern : 4 : private: 
  // CHECK-CC3-NEXT: COMPLETION: Pattern : 4 : protected: 
  // CHECK-CC3-NEXT: COMPLETION: Pattern : 4 : public: 
  // CHECK-CC3-NEXT: COMPLETION: short : 4
  // CHECK-CC3-NEXT: COMPLETION: signed : 4
  // CHECK-CC3-NEXT: COMPLETION: static : 4
  // CHECK-CC3-NEXT: COMPLETION: struct : 4
  // CHECK-CC3-NEXT: COMPLETION: Pattern : 4 : template<<#parameters#>>
  // CHECK-CC3-NEXT: COMPLETION: typedef : 4
  // CHECK-CC3-NEXT: COMPLETION: Pattern : 4 : typename <#qualified-id#>
  // CHECK-CC3-NEXT: COMPLETION: Pattern : 4 : typeof(<#expression-or-type#>)
  // CHECK-CC3-NEXT: COMPLETION: union : 4
  // CHECK-CC3-NEXT: COMPLETION: unsigned : 4
  // CHECK-CC3-NEXT: COMPLETION: Pattern : 4 : using <#qualified-id#>;
  // CHECK-CC3-NEXT: COMPLETION: virtual : 4
  // CHECK-CC3-NEXT: COMPLETION: void : 4
  // CHECK-CC3-NEXT: COMPLETION: volatile : 4
  // CHECK-CC3-NEXT: COMPLETION: wchar_t : 4
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:6:11 %s -o - | FileCheck -check-prefix=CHECK-CC4 %s
  // CHECK-CC4: COMPLETION: y : 0 : [#int#]y
  // CHECK-CC4-NEXT: COMPLETION: foo : 2 : [#void#]foo()
  // CHECK-CC4-NEXT: COMPLETION: t : 2 : t
  // CHECK-CC4-NEXT: COMPLETION: TYPEDEF : 2 : TYPEDEF
  // CHECK-CC4-NEXT: COMPLETION: X : 2 : X
  // CHECK-CC4-NEXT: COMPLETION: z : 2 : [#void#]z(<#int#>)
  // CHECK-CC4-NEXT: COMPLETION: bool : 3
  // CHECK-CC4-NEXT: COMPLETION: char : 3
  // CHECK-CC4-NEXT: COMPLETION: class : 3
  // CHECK-CC4-NEXT: COMPLETION: const : 3
  // CHECK-CC4-NEXT: COMPLETION: Pattern : 3 : const_cast<<#type-id#>>(<#expression#>)
  // CHECK-CC4-NEXT: COMPLETION: Pattern : 3 : delete <#expression#>
  // CHECK-CC4-NEXT: COMPLETION: Pattern : 3 : delete[] <#expression#>
  // CHECK-CC4-NEXT: COMPLETION: double : 3
  // CHECK-CC4-NEXT: COMPLETION: Pattern : 3 : dynamic_cast<<#type-id#>>(<#expression#>)
  // CHECK-CC4-NEXT: COMPLETION: enum : 3
  // CHECK-CC4-NEXT: COMPLETION: false : 3
  // CHECK-CC4-NEXT: COMPLETION: float : 3
  // CHECK-CC4-NEXT: COMPLETION: int : 3
  // CHECK-CC4-NEXT: COMPLETION: long : 3
  // CHECK-CC4-NEXT: COMPLETION: Pattern : 3 : new <#type-id#>(<#expressions#>)
  // CHECK-CC4-NEXT: COMPLETION: Pattern : 3 : new <#type-id#>[<#size#>](<#expressions#>)
  // CHECK-CC4-NEXT: COMPLETION: operator : 3
  // CHECK-CC4-NEXT: COMPLETION: Pattern : 3 : reinterpret_cast<<#type-id#>>(<#expression#>)
  // CHECK-CC4-NEXT: COMPLETION: short : 3
  // CHECK-CC4-NEXT: COMPLETION: signed : 3
  // CHECK-CC4-NEXT: COMPLETION: Pattern : 3 : sizeof(<#expression-or-type#>)
  // CHECK-CC4-NEXT: COMPLETION: Pattern : 3 : static_cast<<#type-id#>>(<#expression#>)
  // CHECK-CC4-NEXT: COMPLETION: struct : 3
  // CHECK-CC4-NEXT: COMPLETION: Pattern : 3 : throw <#expression#>
  // CHECK-CC4-NEXT: COMPLETION: true : 3
  // CHECK-CC4-NEXT: COMPLETION: Pattern : 3 : typeid(<#expression-or-type#>)
  // CHECK-CC4-NEXT: COMPLETION: Pattern : 3 : typename <#qualified-id#>
  // CHECK-CC4-NEXT: COMPLETION: Pattern : 3 : typeof(<#expression-or-type#>)
  // CHECK-CC4-NEXT: COMPLETION: union : 3
  // CHECK-CC4-NEXT: COMPLETION: unsigned : 3
  // CHECK-CC4-NEXT: COMPLETION: void : 3
  // CHECK-CC4-NEXT: COMPLETION: volatile : 3
  // CHECK-CC4-NEXT: COMPLETION: wchar_t : 3
