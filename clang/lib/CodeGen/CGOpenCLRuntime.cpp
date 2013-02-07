//===----- CGOpenCLRuntime.cpp - Interface to OpenCL Runtimes -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for OpenCL code generation.  Concrete
// subclasses of this implement code generation for specific OpenCL
// runtime libraries.
//
//===----------------------------------------------------------------------===//

#include "CGOpenCLRuntime.h"
#include "CodeGenFunction.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include <assert.h>

using namespace clang;
using namespace CodeGen;

CGOpenCLRuntime::~CGOpenCLRuntime() {}

void CGOpenCLRuntime::EmitWorkGroupLocalVarDecl(CodeGenFunction &CGF,
                                                const VarDecl &D) {
  return CGF.EmitStaticVarDecl(D, llvm::GlobalValue::InternalLinkage);
}

llvm::Type *CGOpenCLRuntime::convertOpenCLSpecificType(const Type *T) {
  assert(T->isOpenCLSpecificType() &&
         "Not an OpenCL specific type!");

  switch (cast<BuiltinType>(T)->getKind()) {
  default: 
    llvm_unreachable("Unexpected opencl builtin type!");
    return 0;
  case BuiltinType::OCLImage1d:
    return llvm::PointerType::get(llvm::StructType::create(
                           CGM.getLLVMContext(), "opencl.image1d_t"), 0);
  case BuiltinType::OCLImage1dArray:
    return llvm::PointerType::get(llvm::StructType::create(
                           CGM.getLLVMContext(), "opencl.image1d_array_t"), 0);
  case BuiltinType::OCLImage1dBuffer:
    return llvm::PointerType::get(llvm::StructType::create(
                           CGM.getLLVMContext(), "opencl.image1d_buffer_t"), 0);
  case BuiltinType::OCLImage2d:
    return llvm::PointerType::get(llvm::StructType::create(
                           CGM.getLLVMContext(), "opencl.image2d_t"), 0);
  case BuiltinType::OCLImage2dArray:
    return llvm::PointerType::get(llvm::StructType::create(
                           CGM.getLLVMContext(), "opencl.image2d_array_t"), 0);
  case BuiltinType::OCLImage3d:
    return llvm::PointerType::get(llvm::StructType::create(
                           CGM.getLLVMContext(), "opencl.image3d_t"), 0);
  case BuiltinType::OCLSampler:
    return llvm::IntegerType::get(CGM.getLLVMContext(),32);
  case BuiltinType::OCLEvent:
    return llvm::PointerType::get(llvm::StructType::create(
                           CGM.getLLVMContext(), "opencl.event_t"), 0);
  }
}
