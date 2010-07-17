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
#include "llvm/DerivedTypes.h"
#include "llvm/LLVMContext.h"
#include "llvm/Constants.h"
#include "llvm/Support/ValueHandle.h"
using namespace llvm;

namespace {

static void PR7658() {
  LLVMContext ctx;
  
  WeakVH NullPtr;
  PATypeHolder h1;
  {
    OpaqueType *o1 = OpaqueType::get(ctx);
    PointerType *p1 = PointerType::get(o1, 0);
    
    std::vector<const Type *> t1;
    t1.push_back(IntegerType::get(ctx, 32));
    t1.push_back(p1);
    NullPtr = ConstantPointerNull::get(p1);
    OpaqueType *o2 = OpaqueType::get (ctx);
    PointerType *p2 = PointerType::get (o2, 0);
    t1.push_back(p2);
    
    
    StructType *s1 = StructType::get(ctx, t1);
    h1 = s1;
    o1->refineAbstractTypeTo(s1);
    o2->refineAbstractTypeTo(h1.get());  // h1 = { i32, \2*, \2* }
  }
  
  
  OpaqueType *o3 = OpaqueType::get(ctx);
  PointerType *p3 = PointerType::get(o3, 0);  // p3 = opaque*
  
  std::vector<const Type *> t2;
  t2.push_back(IntegerType::get(ctx, 32));
  t2.push_back(p3);
  
  std::vector<Constant *> v2;
  v2.push_back(ConstantInt::get(IntegerType::get(ctx, 32), 14));
  v2.push_back(ConstantPointerNull::get(p3));
  
  OpaqueType *o4 = OpaqueType::get(ctx);
  {
    PointerType *p4 = PointerType::get(o4, 0);
    t2.push_back(p4);
    v2.push_back(ConstantPointerNull::get(p4));
  }
  
  WeakVH CS = ConstantStruct::get(ctx, v2, false); // { i32 14, opaque* null, opaque* null}
  
  StructType *s2 = StructType::get(ctx, t2);
  PATypeHolder h2(s2);
  o3->refineAbstractTypeTo(s2);
  o4->refineAbstractTypeTo(h2.get());
}
  

TEST(OpaqueTypeTest, RegisterWithContext) {
  LLVMContext C;
  LLVMContextImpl *pImpl = C.pImpl;

  // 1 refers to the AlwaysOpaqueTy allocated in the Context's constructor and
  // destroyed in the destructor.
  EXPECT_EQ(1u, pImpl->OpaqueTypes.size());
  {
    PATypeHolder Type = OpaqueType::get(C);
    EXPECT_EQ(2u, pImpl->OpaqueTypes.size());
  }
  EXPECT_EQ(1u, pImpl->OpaqueTypes.size());
  
  PR7658();
}

}  // namespace
