//===- llvm/unittest/VMCore/DerivedTypesTest.cpp - Types unit tests -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "../lib/VMCore/LLVMContextImpl.h"
#include "llvm/Type.h"
#include "llvm/DerivedTypes.h"
#include "llvm/LLVMContext.h"
using namespace llvm;

namespace {

TEST(OpaqueTypeTest, RegisterWithContext) {
  LLVMContext C;
  LLVMContextImpl *pImpl = C.pImpl;  

  EXPECT_EQ(0u, pImpl->OpaqueTypes.size());
  {
    PATypeHolder Type = OpaqueType::get(C);
    EXPECT_EQ(1u, pImpl->OpaqueTypes.size());
  }
  EXPECT_EQ(0u, pImpl->OpaqueTypes.size());
}

}  // namespace
