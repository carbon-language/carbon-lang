//===-- TargetMachine.cpp - General Target Information ---------------------==//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the general parts of a Target machine.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetMachine.h"
#include "llvm/Type.h"
#include "llvm/IntrinsicLowering.h"
using namespace llvm;

//---------------------------------------------------------------------------
// TargetMachine Class
//
TargetMachine::TargetMachine(const std::string &name, IntrinsicLowering *il,
                             bool LittleEndian,
                             unsigned char PtrSize, unsigned char PtrAl,
                             unsigned char DoubleAl, unsigned char FloatAl,
                             unsigned char LongAl, unsigned char IntAl,
                             unsigned char ShortAl, unsigned char ByteAl)
  : Name(name), DataLayout(name, LittleEndian,
                           PtrSize, PtrAl, DoubleAl, FloatAl, LongAl,
                           IntAl, ShortAl, ByteAl) {
  IL = il ? il : new DefaultIntrinsicLowering();
}

TargetMachine::~TargetMachine() {
  delete IL;
}

unsigned TargetMachine::findOptimalStorageSize(const Type *Ty) const {
  // All integer types smaller than ints promote to 4 byte integers.
  if (Ty->isIntegral() && Ty->getPrimitiveSize() < 4)
    return 4;

  return DataLayout.getTypeSize(Ty);
}
