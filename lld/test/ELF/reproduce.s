# REQUIRES: x86

# RUN: rm -rf %t.dir
# RUN: mkdir -p %t.dir/build1
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.dir/build1/foo.o
# RUN: cd %t.dir
# RUN: ld.lld --hash-style=gnu build1/foo.o -o bar -shared --as-needed --reproduce repro1.tar
# RUN: tar xOf repro1.tar repro1/%:t.dir/build1/foo.o > build1-foo.o
# RUN: cmp build1/foo.o build1-foo.o

# RUN: tar xf repro1.tar repro1/response.txt repro1/version.txt
# RUN: FileCheck %s --check-prefix=RSP1 < repro1/response.txt
# RSP1:      {{^}}--hash-style gnu{{$}}
# RSP1-NOT:  {{^}}repro1{{[/\\]}}
# RSP1-NEXT: {{[/\\]}}foo.o
# RSP1-NEXT: -o bar
# RSP1-NEXT: -shared
# RSP1-NEXT: --as-needed

# RUN: FileCheck %s --check-prefix=VERSION < repro1/version.txt
# VERSION: LLD

# RUN: mkdir -p %t.dir/build2/a/b/c
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.dir/build2/foo.o
# RUN: cd %t.dir/build2/a/b/c
# RUN: env LLD_REPRODUCE=repro2.tar ld.lld ./../../../foo.o -o /dev/null -shared --as-needed
# RUN: tar xOf repro2.tar repro2/%:t.dir/build2/foo.o > build2-foo.o
# RUN: cmp %t.dir/build2/foo.o build2-foo.o

# RUN: mkdir -p %t.dir/build3
# RUN: cd %t.dir/build3
# RUN: echo "{ local: *; };" >  ver
# RUN: echo "{};" > dyn
# RUN: echo > file
# RUN: echo > file2
# RUN: echo "_start" > order
# RUN: mkdir "sysroot with spaces"
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o 'foo bar'
# RUN: ld.lld --reproduce repro3.tar 'foo bar' -L"foo bar" -Lfile -Tfile2 \
# RUN:   --dynamic-list dyn -rpath file --script=file --symbol-ordering-file order \
# RUN:   --sysroot "sysroot with spaces" --sysroot="sysroot with spaces" \
# RUN:   --version-script ver --dynamic-linker "some unusual/path" -soname 'foo bar' \
# RUN:   -soname='foo bar'
# RUN: tar xOf repro3.tar repro3/response.txt | FileCheck %s --check-prefix=RSP3
# RSP3:      --chroot .
# RSP3:      "{{.*}}foo bar"
# RSP3-NEXT: --library-path "[[BASEDIR:.+]]/foo bar"
# RSP3-NEXT: --library-path [[BASEDIR]]/file
# RSP3-NEXT: --script [[BASEDIR]]/file2
# RSP3-NEXT: --dynamic-list [[BASEDIR]]/dyn
# RSP3-NEXT: -rpath [[BASEDIR]]/file
# RSP3-NEXT: --script [[BASEDIR]]/file
# RSP3-NEXT: --symbol-ordering-file [[BASEDIR]]/order
# RSP3-NEXT: --sysroot "[[BASEDIR]]/sysroot with spaces"
# RSP3-NEXT: --sysroot "[[BASEDIR]]/sysroot with spaces"
# RSP3-NEXT: --version-script [[BASEDIR]]/ver
# RSP3-NEXT: --dynamic-linker "some unusual/path"
# RSP3-NEXT: -soname "foo bar"
# RSP3-NEXT: -soname "foo bar"

# RUN: tar tf repro3.tar | FileCheck %s
# CHECK:      repro3/response.txt
# CHECK-NEXT: repro3/version.txt
# CHECK-NEXT: repro3/{{.*}}/order
# CHECK-NEXT: repro3/{{.*}}/dyn
# CHECK-NEXT: repro3/{{.*}}/ver
# CHECK-NEXT: repro3/{{.*}}/foo bar
# CHECK-NEXT: repro3/{{.*}}/file2
# CHECK-NEXT: repro3/{{.*}}/file

## Check that directory path is stripped from -o <file-path>
# RUN: mkdir -p %t.dir/build4/a/b/c
# RUN: cd %t.dir
# RUN: ld.lld build1/foo.o -o build4/a/b/c/bar -shared --as-needed --reproduce=repro4.tar
# RUN: tar xOf repro4.tar repro4/response.txt | FileCheck %s --check-prefix=RSP4
# RSP4: -o bar

.globl _start
_start:
  mov $60, %rax
  mov $42, %rdi
  syscall
