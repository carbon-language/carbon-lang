// RUN: %clang_cc1 -triple mipsel-linux-gnu -fsyntax-only -verify %s

void foo32();
void foo16();
void __attribute__((nomips16)) foo32(); 
void __attribute__((mips16)) foo16(); 

void __attribute__((nomips16)) foo32_(); 
void __attribute__((mips16)) foo16_(); 
void foo32_();
void foo16_();

void foo32__() __attribute__((nomips16)); 
void foo32__() __attribute__((mips16)); 

void foo32a() __attribute__((nomips16(xyz))) ; // expected-error {{attribute takes no arguments}}
void __attribute__((mips16(xyz))) foo16a(); // expected-error {{attribute takes no arguments}}

void __attribute__((nomips16(1, 2))) foo32b(); // expected-error {{attribute takes no arguments}}
void __attribute__((mips16(1, 2))) foo16b(); // expected-error {{attribute takes no arguments}}


__attribute((nomips16)) int a; // expected-error {{attribute only applies to functions}}

__attribute((mips16)) int b; // expected-error {{attribute only applies to functions}}


