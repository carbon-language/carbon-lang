//===-- include/Support/DataTypesFix.h - Fix datatype defs ------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file overrides default system-defined types and limits which cannot be
// done in DataTypes.h.in because it is processed by autoheader first, which
// comments out any #undef statement
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DATATYPESFIX_H
#define SUPPORT_DATATYPESFIX_H

#include "llvm/Config/config.h"

#if defined(_POWER) && defined(_AIX)
// GCC is strict about defining large constants: they must have LL modifier.
#undef INT64_MAX
#define INT64_MAX 9223372036854775807LL
#undef INT64_MIN
#define INT64_MIN (-INT64_MAX-1) 
#endif

#endif  /* SUPPORT_DATATYPESFIX_H */
