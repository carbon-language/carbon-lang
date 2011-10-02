// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -fobjc-fragile-abi -emit-llvm -o %t %s
// RUN: grep -F 'declare i8* @objc_msgSend(i8*, i8*, ...) nonlazybind' %t

void f0(id x) {
  [x foo];
}
