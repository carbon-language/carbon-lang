//===- llvm/unittest/VMCore/Metadata.cpp - Metadata unit tests ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/LLVMContext.h"
#include "llvm/Metadata.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ValueHandle.h"
using namespace llvm;

namespace {

LLVMContext &Context = getGlobalContext();

// Test that construction of MDString with different value produces different
// MDString objects, even with the same string pointer and nulls in the string.
TEST(MDStringTest, CreateDifferent) {
  char x[3] = { 'f', 0, 'A' };
  MDString *s1 = MDString::get(Context, StringRef(&x[0], 3));
  x[2] = 'B';
  MDString *s2 = MDString::get(Context, StringRef(&x[0], 3));
  EXPECT_NE(s1, s2);
}

// Test that creation of MDStrings with the same string contents produces the
// same MDString object, even with different pointers.
TEST(MDStringTest, CreateSame) {
  char x[4] = { 'a', 'b', 'c', 'X' };
  char y[4] = { 'a', 'b', 'c', 'Y' };

  MDString *s1 = MDString::get(Context, StringRef(&x[0], 3));
  MDString *s2 = MDString::get(Context, StringRef(&y[0], 3));
  EXPECT_EQ(s1, s2);
}

// Test that MDString prints out the string we fed it.
TEST(MDStringTest, PrintingSimple) {
  char *str = new char[13];
  strncpy(str, "testing 1 2 3", 13);
  MDString *s = MDString::get(Context, StringRef(str, 13));
  strncpy(str, "aaaaaaaaaaaaa", 13);
  delete[] str;

  std::string Str;
  raw_string_ostream oss(Str);
  s->print(oss);
  EXPECT_STREQ("metadata !\"testing 1 2 3\"", oss.str().c_str());
}

// Test printing of MDString with non-printable characters.
TEST(MDStringTest, PrintingComplex) {
  char str[5] = {0, '\n', '"', '\\', -1};
  MDString *s = MDString::get(Context, StringRef(str+0, 5));
  std::string Str;
  raw_string_ostream oss(Str);
  s->print(oss);
  EXPECT_STREQ("metadata !\"\\00\\0A\\22\\5C\\FF\"", oss.str().c_str());
}

// Test the two constructors, and containing other Constants.
TEST(MDNodeTest, Simple) {
  char x[3] = { 'a', 'b', 'c' };
  char y[3] = { '1', '2', '3' };

  MDString *s1 = MDString::get(Context, StringRef(&x[0], 3));
  MDString *s2 = MDString::get(Context, StringRef(&y[0], 3));
  ConstantInt *CI = ConstantInt::get(getGlobalContext(), APInt(8, 0));

  std::vector<Value *> V;
  V.push_back(s1);
  V.push_back(CI);
  V.push_back(s2);

  MDNode *n1 = MDNode::get(Context, &V[0], 3);
  Value *const c1 = n1;
  MDNode *n2 = MDNode::get(Context, &c1, 1);
  MDNode *n3 = MDNode::get(Context, &V[0], 3);
  EXPECT_NE(n1, n2);
#ifdef ENABLE_MDNODE_UNIQUING
  EXPECT_EQ(n1, n3);
#else
  (void) n3;
#endif

  EXPECT_EQ(3u, n1->getNumElements());
  EXPECT_EQ(s1, n1->getElement(0));
  EXPECT_EQ(CI, n1->getElement(1));
  EXPECT_EQ(s2, n1->getElement(2));

  EXPECT_EQ(1u, n2->getNumElements());
  EXPECT_EQ(n1, n2->getElement(0));

  std::string Str;
  raw_string_ostream oss(Str);
  n1->print(oss);
  EXPECT_STREQ("!0 = metadata !{metadata !\"abc\", i8 0, metadata !\"123\"}\n",
               oss.str().c_str());
  Str.clear();
  n2->print(oss);
  EXPECT_STREQ("!0 = metadata !{metadata !1}\n"
               "!1 = metadata !{metadata !\"abc\", i8 0, metadata !\"123\"}\n",
               oss.str().c_str());
}

TEST(MDNodeTest, Delete) {
  Constant *C = ConstantInt::get(Type::getInt32Ty(getGlobalContext()), 1);
  Instruction *I = new BitCastInst(C, Type::getInt32Ty(getGlobalContext()));

  Value *const V = I;
  MDNode *n = MDNode::get(Context, &V, 1);
  WeakVH wvh = n;

  EXPECT_EQ(n, wvh);

  delete I;

  std::string Str;
  raw_string_ostream oss(Str);
  wvh->print(oss);
  EXPECT_STREQ("!0 = metadata !{null}\n", oss.str().c_str());
}

TEST(NamedMDNodeTest, Search) {
  Constant *C = ConstantInt::get(Type::getInt32Ty(getGlobalContext()), 1);
  Constant *C2 = ConstantInt::get(Type::getInt32Ty(getGlobalContext()), 2);

  Value *const V = C;
  Value *const V2 = C2;
  MDNode *n = MDNode::get(Context, &V, 1);
  MDNode *n2 = MDNode::get(Context, &V2, 1);

  MetadataBase *Nodes[2] = { n, n2 };

  Module *M = new Module("MyModule", getGlobalContext());
  const char *Name = "llvm.NMD1";
  NamedMDNode *NMD = NamedMDNode::Create(getGlobalContext(), Name, &Nodes[0], 2, M);
  std::string Str;
  raw_string_ostream oss(Str);
  NMD->print(oss);
  EXPECT_STREQ("!llvm.NMD1 = !{!0, !1}\n!0 = metadata !{i32 1}\n"
               "!1 = metadata !{i32 2}\n",
               oss.str().c_str());
}
}
