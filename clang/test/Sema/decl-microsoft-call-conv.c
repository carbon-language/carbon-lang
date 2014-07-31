// RUN: %clang_cc1 -triple i686-pc-win32 -verify %s

// It's important that this is a .c file.

// This is fine, as CrcGenerateTable() has a prototype.
void __fastcall CrcGenerateTable(void);
void __fastcall CrcGenerateTable() {}

void __fastcall CrcGenerateTableNoProto() {} // expected-error{{function with no prototype cannot use fastcall calling convention}}
