//===-- sanitizer_test_config.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of *Sanitizer runtime.
//
//===----------------------------------------------------------------------===//
#if !defined(INCLUDED_FROM_SANITIZER_TEST_UTILS_H)
# error "This file should be included into sanitizer_test_utils.h only"
#endif

#ifndef SANITIZER_TEST_CONFIG_H
#define SANITIZER_TEST_CONFIG_H

#include <vector>
#include <string>
#include <map>

#if SANITIZER_USE_DEJAGNU_GTEST
# include "dejagnu-gtest.h"
#else
# include "gtest/gtest.h"
#endif

#endif  // SANITIZER_TEST_CONFIG_H
