//===-- llvm/CodeGen/GlobalISel/EmptyFile.cpp ------ EmptyFile ---*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// The purpose of this file is to please cmake by not creating an
/// empty library when we do not build GlobalISel.
/// \todo This file should be removed when GlobalISel is not optional anymore.
//===----------------------------------------------------------------------===//

// Anonymous namespace so that we do not step on anyone's toes.
namespace {
__attribute__ ((unused)) void foo(void) {
}
}
