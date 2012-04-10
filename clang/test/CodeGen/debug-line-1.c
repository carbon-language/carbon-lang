// RUN: %clang_cc1 -triple x86_64-unk-unk -o - -emit-llvm -g %s | FileCheck %s
// PR9796

// Check to make sure that we emit the block for the break so that we can count the line.
// CHECK: sw.bb:                                            ; preds = %entry
// CHECK: br label %sw.epilog, !dbg !21
  
extern int atoi(const char *);

int f(char* arg) {
  int x = atoi(arg);
  
  switch(x) {
    case 1:
      break;
  }

  return 0;
}
