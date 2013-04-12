//===- lld/unittest/ParallelTest.cpp --------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Parallel.h unit tests.
///
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lld/Core/Parallel.h"

#include <array>
#include <random>

uint32_t array[1024 * 1024];

TEST(Parallel, sort) {
  std::mt19937 randEngine;
  std::uniform_int_distribution<uint32_t> dist;

  for (auto &i : array)
    i = dist(randEngine);

  lld::parallel_sort(std::begin(array), std::end(array));
  ASSERT_TRUE(std::is_sorted(std::begin(array), std::end(array)));
}
