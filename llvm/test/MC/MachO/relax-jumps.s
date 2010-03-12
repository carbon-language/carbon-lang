// RUN: llvm-mc -triple i386-apple-darwin9 %s -filetype=obj -o - | macho-dump --dump-section-data | FileCheck %s

// FIXME: This is a horrible way of checking the output, we need an llvm-mc
// based 'otool'. Use:
//   (f=relax-jumps;
//    llvm-mc -filetype=obj -o $f.mc.o $f.s &&
//    as -arch i386 -o $f.as.o $f.s &&
//    otool -tvr $f.mc.o | tail +2 > $f.mc.dump &&
//    otool -tvr $f.as.o | tail +2 > $f.as.dump &&
//    diff $f.{as,mc}.dump)
// to examine the results in a more sensible fashion.

// CHECK: ('_section_data', '\x90
// CHECK: \x0f\x842\xff\xff\xff\x0f\x82\xe6\x00\x00\x00\x0f\x87&\xff\xff\xff\x0f\x8f\xda\x00\x00\x00\x0f\x88\x1a\xff\xff\xff\x0f\x83\xce\x00\x00\x00\x0f\x89\x0e\xff\xff\xff\x90
// CHECK: \x901\xc0')

L1:
        .space 200, 0x90

        je L1
        jb L2
        ja L1
        jg L2
        js L1
        jae L2
        jns L1

        .space 200, 0x90
L2:

        xorl %eax, %eax
