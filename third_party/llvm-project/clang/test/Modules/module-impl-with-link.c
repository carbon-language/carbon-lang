// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -fmodule-name=Clib %s -I %S/Inputs/module-impl-with-link -emit-llvm -o - | FileCheck %s
#include "foo.h"
// Make sure we don't generate linker option for module Clib since this TU is
// an implementation of Clib.
// CHECK: !llvm.linker.options = !{}
