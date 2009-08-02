//===--------------- LLVMContextImpl.cpp - Implementation ------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements LLVMContextImpl, the opaque implementation 
//  of LLVMContext.
//
//===----------------------------------------------------------------------===//

#include "LLVMContextImpl.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/LLVMContext.h"
#include "llvm/Metadata.h"
using namespace llvm;

LLVMContextImpl::LLVMContextImpl(LLVMContext &C) :
    Context(C), TheTrueVal(0), TheFalseVal(0) { }

