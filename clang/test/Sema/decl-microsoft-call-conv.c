// RUN: %clang_cc1 -triple i686-pc-win32 -verify %s

// It's important that this is a .c file.

// This is fine, as CrcGenerateTable*() has a prototype.
void __fastcall CrcGenerateTableFastcall(void);
void __fastcall CrcGenerateTableFastcall() {}
void __stdcall CrcGenerateTableStdcall(void);
void __stdcall CrcGenerateTableStdcall() {}
void __thiscall CrcGenerateTableThiscall(void);
void __thiscall CrcGenerateTableThiscall() {}
void __pascal CrcGenerateTablePascal(void);
void __pascal CrcGenerateTablePascal() {}

void __fastcall CrcGenerateTableNoProtoFastcall() {} // expected-error{{function with no prototype cannot use fastcall calling convention}}
void __stdcall CrcGenerateTableNoProtoStdcall() {} // expected-error{{function with no prototype cannot use stdcall calling convention}}
void __thiscall CrcGenerateTableNoProtoThiscall() {} // expected-error{{function with no prototype cannot use thiscall calling convention}}
void __pascal CrcGenerateTableNoProtoPascal() {} // expected-error{{function with no prototype cannot use pascal calling convention}}

// Regular calling convention is fine.
void CrcGenerateTableNoProto() {}


// In system headers, the stdcall version should be a warning.
# 1 "fake_system_header.h" 1 3 4
void __fastcall SystemHeaderFastcall() {} // expected-error{{function with no prototype cannot use fastcall calling convention}}
void __stdcall SystemHeaderStdcall() {}
void __thiscall SystemHeaderThiscall() {} // expected-error{{function with no prototype cannot use thiscall calling convention}}
void __pascal SystemHeaderPascal() {} // expected-error{{function with no prototype cannot use pascal calling convention}}
