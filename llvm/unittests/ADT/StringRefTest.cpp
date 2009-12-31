//===- llvm/unittest/ADT/StringRefTest.cpp - StringRef unit tests ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace llvm {

std::ostream &operator<<(std::ostream &OS, const StringRef &S) {
  OS << S;
  return OS;
}

std::ostream &operator<<(std::ostream &OS,
                         const std::pair<StringRef, StringRef> &P) {
  OS << "(" << P.first << ", " << P.second << ")";
  return OS;
}

}

namespace {
TEST(StringRefTest, Construction) {
  EXPECT_EQ("", StringRef());
  EXPECT_EQ("hello", StringRef("hello"));
  EXPECT_EQ("hello", StringRef("hello world", 5));
  EXPECT_EQ("hello", StringRef(std::string("hello")));
}

TEST(StringRefTest, Iteration) {
  StringRef S("hello");
  const char *p = "hello";
  for (const char *it = S.begin(), *ie = S.end(); it != ie; ++it, ++p)
    EXPECT_EQ(*it, *p);
}

TEST(StringRefTest, StringOps) {
  const char *p = "hello";
  EXPECT_EQ(p, StringRef(p, 0).data());
  EXPECT_TRUE(StringRef().empty());
  EXPECT_EQ((size_t) 5, StringRef("hello").size());
  EXPECT_EQ(-1, StringRef("aab").compare("aad"));
  EXPECT_EQ( 0, StringRef("aab").compare("aab"));
  EXPECT_EQ( 1, StringRef("aab").compare("aaa"));
  EXPECT_EQ(-1, StringRef("aab").compare("aabb"));
  EXPECT_EQ( 1, StringRef("aab").compare("aa"));
}

TEST(StringRefTest, Operators) {
  EXPECT_EQ("", StringRef());
  EXPECT_TRUE(StringRef("aab") < StringRef("aad"));
  EXPECT_FALSE(StringRef("aab") < StringRef("aab"));
  EXPECT_TRUE(StringRef("aab") <= StringRef("aab"));
  EXPECT_FALSE(StringRef("aab") <= StringRef("aaa"));
  EXPECT_TRUE(StringRef("aad") > StringRef("aab"));
  EXPECT_FALSE(StringRef("aab") > StringRef("aab"));
  EXPECT_TRUE(StringRef("aab") >= StringRef("aab"));
  EXPECT_FALSE(StringRef("aaa") >= StringRef("aab"));
  EXPECT_EQ(StringRef("aab"), StringRef("aab"));
  EXPECT_FALSE(StringRef("aab") == StringRef("aac"));
  EXPECT_FALSE(StringRef("aab") != StringRef("aab"));
  EXPECT_TRUE(StringRef("aab") != StringRef("aac"));
  EXPECT_EQ('a', StringRef("aab")[1]);
}

TEST(StringRefTest, Substr) {
  StringRef Str("hello");
  EXPECT_EQ("lo", Str.substr(3));
  EXPECT_EQ("", Str.substr(100));
  EXPECT_EQ("hello", Str.substr(0, 100));
  EXPECT_EQ("o", Str.substr(4, 10));
}

TEST(StringRefTest, Slice) {
  StringRef Str("hello");
  EXPECT_EQ("l", Str.slice(2, 3));
  EXPECT_EQ("ell", Str.slice(1, 4));
  EXPECT_EQ("llo", Str.slice(2, 100));
  EXPECT_EQ("", Str.slice(2, 1));
  EXPECT_EQ("", Str.slice(10, 20));
}

TEST(StringRefTest, Split) {
  StringRef Str("hello");
  EXPECT_EQ(std::make_pair(StringRef("hello"), StringRef("")),
            Str.split('X'));
  EXPECT_EQ(std::make_pair(StringRef("h"), StringRef("llo")),
            Str.split('e'));
  EXPECT_EQ(std::make_pair(StringRef(""), StringRef("ello")),
            Str.split('h'));
  EXPECT_EQ(std::make_pair(StringRef("he"), StringRef("lo")),
            Str.split('l'));
  EXPECT_EQ(std::make_pair(StringRef("hell"), StringRef("")),
            Str.split('o'));

  EXPECT_EQ(std::make_pair(StringRef("hello"), StringRef("")),
            Str.rsplit('X'));
  EXPECT_EQ(std::make_pair(StringRef("h"), StringRef("llo")),
            Str.rsplit('e'));
  EXPECT_EQ(std::make_pair(StringRef(""), StringRef("ello")),
            Str.rsplit('h'));
  EXPECT_EQ(std::make_pair(StringRef("hel"), StringRef("o")),
            Str.rsplit('l'));
  EXPECT_EQ(std::make_pair(StringRef("hell"), StringRef("")),
            Str.rsplit('o'));
}

TEST(StringRefTest, Split2) {
  SmallVector<StringRef, 5> parts;
  SmallVector<StringRef, 5> expected;

  expected.push_back("ab"); expected.push_back("c");
  StringRef(",ab,,c,").split(parts, ",", -1, false);
  EXPECT_TRUE(parts == expected);

  expected.clear(); parts.clear();
  expected.push_back(""); expected.push_back("ab"); expected.push_back("");
  expected.push_back("c"); expected.push_back("");
  StringRef(",ab,,c,").split(parts, ",", -1, true);
  EXPECT_TRUE(parts == expected);

  expected.clear(); parts.clear();
  expected.push_back("");
  StringRef("").split(parts, ",", -1, true);
  EXPECT_TRUE(parts == expected);

  expected.clear(); parts.clear();
  StringRef("").split(parts, ",", -1, false);
  EXPECT_TRUE(parts == expected);

  expected.clear(); parts.clear();
  StringRef(",").split(parts, ",", -1, false);
  EXPECT_TRUE(parts == expected);

  expected.clear(); parts.clear();
  expected.push_back(""); expected.push_back("");
  StringRef(",").split(parts, ",", -1, true);
  EXPECT_TRUE(parts == expected);

  expected.clear(); parts.clear();
  expected.push_back("a"); expected.push_back("b");
  StringRef("a,b").split(parts, ",", -1, true);
  EXPECT_TRUE(parts == expected);

  // Test MaxSplit
  expected.clear(); parts.clear();
  expected.push_back("a,,b,c");
  StringRef("a,,b,c").split(parts, ",", 0, true);
  EXPECT_TRUE(parts == expected);

  expected.clear(); parts.clear();
  expected.push_back("a,,b,c");
  StringRef("a,,b,c").split(parts, ",", 0, false);
  EXPECT_TRUE(parts == expected);

  expected.clear(); parts.clear();
  expected.push_back("a"); expected.push_back(",b,c");
  StringRef("a,,b,c").split(parts, ",", 1, true);
  EXPECT_TRUE(parts == expected);

  expected.clear(); parts.clear();
  expected.push_back("a"); expected.push_back(",b,c");
  StringRef("a,,b,c").split(parts, ",", 1, false);
  EXPECT_TRUE(parts == expected);

  expected.clear(); parts.clear();
  expected.push_back("a"); expected.push_back(""); expected.push_back("b,c");
  StringRef("a,,b,c").split(parts, ",", 2, true);
  EXPECT_TRUE(parts == expected);

  expected.clear(); parts.clear();
  expected.push_back("a"); expected.push_back("b,c");
  StringRef("a,,b,c").split(parts, ",", 2, false);
  EXPECT_TRUE(parts == expected);

  expected.clear(); parts.clear();
  expected.push_back("a"); expected.push_back(""); expected.push_back("b");
  expected.push_back("c");
  StringRef("a,,b,c").split(parts, ",", 3, true);
  EXPECT_TRUE(parts == expected);

  expected.clear(); parts.clear();
  expected.push_back("a"); expected.push_back("b"); expected.push_back("c");
  StringRef("a,,b,c").split(parts, ",", 3, false);
  EXPECT_TRUE(parts == expected);
}

TEST(StringRefTest, StartsWith) {
  StringRef Str("hello");
  EXPECT_TRUE(Str.startswith("he"));
  EXPECT_FALSE(Str.startswith("helloworld"));
  EXPECT_FALSE(Str.startswith("hi"));
}

TEST(StringRefTest, EndsWith) {
  StringRef Str("hello");
  EXPECT_TRUE(Str.endswith("lo"));
  EXPECT_FALSE(Str.endswith("helloworld"));
  EXPECT_FALSE(Str.endswith("worldhello"));
  EXPECT_FALSE(Str.endswith("so"));
}

TEST(StringRefTest, Find) {
  StringRef Str("hello");
  EXPECT_EQ(2U, Str.find('l'));
  EXPECT_EQ(StringRef::npos, Str.find('z'));
  EXPECT_EQ(StringRef::npos, Str.find("helloworld"));
  EXPECT_EQ(0U, Str.find("hello"));
  EXPECT_EQ(1U, Str.find("ello"));
  EXPECT_EQ(StringRef::npos, Str.find("zz"));
  EXPECT_EQ(2U, Str.find("ll", 2));
  EXPECT_EQ(StringRef::npos, Str.find("ll", 3));

  EXPECT_EQ(3U, Str.rfind('l'));
  EXPECT_EQ(StringRef::npos, Str.rfind('z'));
  EXPECT_EQ(StringRef::npos, Str.rfind("helloworld"));
  EXPECT_EQ(0U, Str.rfind("hello"));
  EXPECT_EQ(1U, Str.rfind("ello"));
  EXPECT_EQ(StringRef::npos, Str.rfind("zz"));

  EXPECT_EQ(2U, Str.find_first_of('l'));
  EXPECT_EQ(1U, Str.find_first_of("el"));
  EXPECT_EQ(StringRef::npos, Str.find_first_of("xyz"));

  EXPECT_EQ(1U, Str.find_first_not_of('h'));
  EXPECT_EQ(4U, Str.find_first_not_of("hel"));
  EXPECT_EQ(StringRef::npos, Str.find_first_not_of("hello"));
}

TEST(StringRefTest, Count) {
  StringRef Str("hello");
  EXPECT_EQ(2U, Str.count('l'));
  EXPECT_EQ(1U, Str.count('o'));
  EXPECT_EQ(0U, Str.count('z'));
  EXPECT_EQ(0U, Str.count("helloworld"));
  EXPECT_EQ(1U, Str.count("hello"));
  EXPECT_EQ(1U, Str.count("ello"));
  EXPECT_EQ(0U, Str.count("zz"));
}

TEST(StringRefTest, EditDistance) {
  StringRef Str("hello");
  EXPECT_EQ(2U, Str.edit_distance("hill"));
}

TEST(StringRefTest, Misc) {
  std::string Storage;
  raw_string_ostream OS(Storage);
  OS << StringRef("hello");
  EXPECT_EQ("hello", OS.str());
}

} // end anonymous namespace
