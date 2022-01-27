// REQUIRES: x86-registered-target
// RUN: %clang_cl --target=i686-windows-msvc /c /Z7 /Fo%t.obj -- %s
// RUN: llvm-pdbutil dump --types %t.obj | FileCheck %s
// RUN: %clang_cl --target=i686-windows-msvc /c /Z7 /Fo%t.obj -fdebug-compilation-dir=. -- %s
// RUN: llvm-pdbutil dump --types %t.obj | FileCheck %s --check-prefix RELATIVE

int main() { return 42; }

// CHECK:                       Types (.debug$T)
// CHECK: ============================================================
// CHECK: 0x[[PWD:.+]] | LF_STRING_ID [size = {{.+}}] ID: <no type>, String: [[PWDVAL:.+]]
// CHECK: 0x[[FILEPATH:.+]] | LF_STRING_ID [size = {{.+}}] ID: <no type>, String: [[FILEPATHVAL:.+[\\/]debug-info-codeview-buildinfo.c]]
// CHECK: 0x[[ZIPDB:.+]] | LF_STRING_ID [size = {{.+}}] ID: <no type>, String:
// CHECK: 0x[[TOOL:.+]] | LF_STRING_ID [size = {{.+}}] ID: <no type>, String: [[TOOLVAL:.+[\\/]clang.*]]
// CHECK: 0x[[CMDLINE:.+]] | LF_STRING_ID [size = {{.+}}] ID: <no type>, String: "-cc1
// CHECK: 0x{{.+}} | LF_BUILDINFO [size = {{.+}}]
// CHECK:          0x[[PWD]]: `[[PWDVAL]]`
// CHECK:          0x[[TOOL]]: `[[TOOLVAL]]`
// CHECK:          0x[[FILEPATH]]: `[[FILEPATHVAL]]`
// CHECK:          0x[[ZIPDB]]: ``
// CHECK:          0x[[CMDLINE]]: `"-cc1

// RELATIVE:                       Types (.debug$T)
// RELATIVE: ============================================================
// RELATIVE: 0x{{.+}} | LF_BUILDINFO [size = {{.+}}]
// RELATIVE:          0x{{.+}}: `.`
