// RUN: rm -f %t.hmap
// RUN: %hmaptool write %S/Inputs/headermap-rel2/project-headers.hmap.json %t.hmap

// RUN: %clang -fsyntax-only %s -iquote %t.hmap -isystem %S/Inputs/headermap-rel2/system/usr/include -I %S/Inputs/headermap-rel2 -H 2> %t.out
// RUN: FileCheck %s -input-file %t.out

// RUN: env CC_PRINT_HEADERS=1 %clang -fsyntax-only %s -iquote %t.hmap -isystem %S/Inputs/headermap-rel2/system/usr/include -I %S/Inputs/headermap-rel2 2> %t.out
// RUN: FileCheck %s -input-file %t.out

// CHECK: Product/someheader.h
// CHECK: system/usr/include{{[/\\]+}}someheader.h
// CHECK: system/usr/include{{[/\\]+}}someheader.h

#include "someheader.h"
#include <someheader.h>
#include <someheader.h>
