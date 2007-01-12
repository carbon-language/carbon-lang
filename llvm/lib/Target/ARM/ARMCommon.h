//===-- ARMCommon.h - Define support functions for ARM ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the "Instituto Nokia de Tecnologia" and
// is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//
//===----------------------------------------------------------------------===//

#ifndef ARM_COMMON_H
#define ARM_COMMON_H

#include <vector>

std::vector<unsigned> splitImmediate(unsigned immediate);

#endif
