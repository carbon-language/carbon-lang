//===-- adt_test.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of the ORC runtime.
//
//===----------------------------------------------------------------------===//

#include "adt.h"
#include "gtest/gtest.h"

#include <sstream>
#include <string>

using namespace __orc_rt;

TEST(ADTTest, SpanDefaultConstruction) {
  span<int> S;
  EXPECT_TRUE(S.empty()) << "Default constructed span not empty";
  EXPECT_EQ(S.size(), 0U) << "Default constructed span size not zero";
  EXPECT_EQ(S.begin(), S.end()) << "Default constructed span begin != end";
}

TEST(ADTTest, SpanConstructFromFixedArray) {
  int A[] = {1, 2, 3, 4, 5};
  span<int> S(A);
  EXPECT_FALSE(S.empty()) << "Span should be non-empty";
  EXPECT_EQ(S.size(), 5U) << "Span has unexpected size";
  EXPECT_EQ(std::distance(S.begin(), S.end()), 5U)
      << "Unexpected iterator range size";
  EXPECT_EQ(S.data(), &A[0]) << "Span data has unexpected value";
  for (unsigned I = 0; I != S.size(); ++I)
    EXPECT_EQ(S[I], A[I]) << "Unexpected span element value";
}

TEST(ADTTest, SpanConstructFromIteratorAndSize) {
  int A[] = {1, 2, 3, 4, 5};
  span<int> S(&A[0], 5);
  EXPECT_FALSE(S.empty()) << "Span should be non-empty";
  EXPECT_EQ(S.size(), 5U) << "Span has unexpected size";
  EXPECT_EQ(std::distance(S.begin(), S.end()), 5U)
      << "Unexpected iterator range size";
  EXPECT_EQ(S.data(), &A[0]) << "Span data has unexpected value";
  for (unsigned I = 0; I != S.size(); ++I)
    EXPECT_EQ(S[I], A[I]) << "Unexpected span element value";
}

TEST(ADTTest, StringViewDefaultConstruction) {
  string_view S;
  EXPECT_TRUE(S.empty()) << "Default constructed span not empty";
  EXPECT_EQ(S.size(), 0U) << "Default constructed span size not zero";
  EXPECT_EQ(S.begin(), S.end()) << "Default constructed span begin != end";
}

TEST(ADTTest, StringViewConstructFromCharPtrAndSize) {
  const char *Str = "abcdefg";
  string_view S(Str, 5);
  EXPECT_FALSE(S.empty()) << "string_view should be non-empty";
  EXPECT_EQ(S.size(), 5U) << "string_view has unexpected size";
  EXPECT_EQ(std::distance(S.begin(), S.end()), 5U)
      << "Unexpected iterator range size";
  EXPECT_EQ(S.data(), &Str[0]) << "string_view data has unexpected value";
  for (unsigned I = 0; I != S.size(); ++I)
    EXPECT_EQ(S[I], Str[I]) << "Unexpected string_view element value";
}

TEST(ADTTest, StringViewConstructFromCharPtr) {
  const char *Str = "abcdefg";
  size_t StrLen = strlen(Str);
  string_view S(Str);

  EXPECT_FALSE(S.empty()) << "string_view should be non-empty";
  EXPECT_EQ(S.size(), StrLen) << "string_view has unexpected size";
  EXPECT_EQ(static_cast<size_t>(std::distance(S.begin(), S.end())), StrLen)
      << "Unexpected iterator range size";
  EXPECT_EQ(S.data(), &Str[0]) << "string_view data has unexpected value";
  for (unsigned I = 0; I != S.size(); ++I)
    EXPECT_EQ(S[I], Str[I]) << "Unexpected string_view element value";
}

TEST(ADTTest, StringViewConstructFromStdString) {
  std::string Str("abcdefg");
  string_view S(Str);

  EXPECT_FALSE(S.empty()) << "string_view should be non-empty";
  EXPECT_EQ(S.size(), Str.size()) << "string_view has unexpected size";
  EXPECT_EQ(static_cast<size_t>(std::distance(S.begin(), S.end())), Str.size())
      << "Unexpected iterator range size";
  EXPECT_EQ(S.data(), &Str[0]) << "string_view data has unexpected value";
  for (unsigned I = 0; I != S.size(); ++I)
    EXPECT_EQ(S[I], Str[I]) << "Unexpected string_view element value";
}

TEST(ADTTest, StringViewCopyConstructionAndAssignment) {
  // Check that string_views are copy-constructible and copy-assignable.
  std::string Str("abcdefg");
  string_view Orig(Str);
  string_view CopyConstructed(Orig);
  string_view CopyAssigned = Orig;

  EXPECT_EQ(Orig, CopyConstructed);
  EXPECT_EQ(Orig, CopyAssigned);
}

TEST(ADTTest, StringViewEquality) {
  EXPECT_EQ("", string_view());
  EXPECT_FALSE(string_view("aab") == string_view("aac"));
  EXPECT_FALSE(string_view("aab") != string_view("aab"));
  EXPECT_NE(string_view("aab"), string_view("aac"));
}

TEST(ADTTest, StringViewOStreamOperator) {
  std::string Str("abcdefg");
  string_view S(Str);
  std::ostringstream OSS;
  OSS << S;

  EXPECT_EQ(OSS.str(), Str);
}

TEST(ADTTest, StringViewHashable) {
  std::string Str("abcdefg");
  string_view S(Str);

  EXPECT_EQ(std::hash<std::string>()(Str), std::hash<string_view>()(S));
}
