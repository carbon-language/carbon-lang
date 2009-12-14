// RUN: clang -cc1 -triple x86_64-apple-darwin9 -emit-llvm -o %t %s
// RUN: grep -F 'declare i8* @objc_msgSend(...)' %t

typedef struct objc_selector *SEL;
id f0(id x, SEL s) {
  return objc_msgSend(x, s);
}
