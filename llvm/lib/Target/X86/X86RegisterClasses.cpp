//===- Target/X86/X86RegisterClasses.cpp - Register Classes -------*-C++-*-===//
//
// This file describes the X86 Register Classes which describe registers.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/MRegisterInfo.h"
#include "X86RegisterInfo.h"
#include "llvm/Type.h"
#include "X86.h"

namespace {
  const unsigned ByteRegClassRegs[] = {
#define R8(ENUM, NAME, FLAGS, TSFLAGS, ALIAS_SET) X86::ENUM,
#include "X86RegisterInfo.def"
  };

  TargetRegisterClass X86ByteRegisterClassInstance(1, ByteRegClassRegs,
 ByteRegClassRegs+sizeof(ByteRegClassRegs)/sizeof(ByteRegClassRegs[0]));

//
//
//
  const unsigned ShortRegClassRegs[] = {
#define R16(ENUM, NAME, FLAGS, TSFLAGS, ALIAS_SET) X86::ENUM,
#include "X86RegisterInfo.def"
  };

  TargetRegisterClass X86ShortRegisterClassInstance(2, ShortRegClassRegs,
      ShortRegClassRegs+sizeof(ShortRegClassRegs)/sizeof(ShortRegClassRegs[0]));

//
//
//

  const unsigned IntRegClassRegs[] = {
#define R32(ENUM, NAME, FLAGS, TSFLAGS, ALIAS_SET) X86::ENUM,
#include "X86RegisterInfo.def"
  };

  TargetRegisterClass X86IntRegisterClassInstance(4, IntRegClassRegs,
      IntRegClassRegs+sizeof(IntRegClassRegs)/sizeof(IntRegClassRegs[0]));

  const TargetRegisterClass * const X86RegClasses[] = {
    &X86ByteRegisterClassInstance,
    &X86ShortRegisterClassInstance,
    &X86IntRegisterClassInstance
  };
}

const TargetRegisterClass* X86RegisterInfo::getRegClassForType(const Type* Ty)
  const {
  switch (Ty->getPrimitiveID()) {
  case Type::BoolTyID:
  case Type::SByteTyID:
  case Type::UByteTyID:   return &X86ByteRegisterClassInstance;
  case Type::ShortTyID:
  case Type::UShortTyID:  return &X86ShortRegisterClassInstance;
  case Type::LongTyID:    // None of these are handled yet!
  case Type::ULongTyID:  // FIXME: Treat these like ints, this is bogus!
    
  case Type::IntTyID:
  case Type::UIntTyID:
  case Type::PointerTyID: return &X86IntRegisterClassInstance;
    
  case Type::FloatTyID:
  case Type::DoubleTyID:
    
  default:
    assert(0 && "Invalid type to getClass!");
    return 0;  // not reached
  }
}


MRegisterInfo::const_iterator X86RegisterInfo::regclass_begin() const {
  return X86RegClasses;
}

unsigned X86RegisterInfo::getNumRegClasses() const {
  return sizeof(X86RegClasses)/sizeof(X86RegClasses[0]);
}

MRegisterInfo::const_iterator X86RegisterInfo::regclass_end() const {
  return X86RegClasses+getNumRegClasses();
}

