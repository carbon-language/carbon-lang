//===- Target/X86/X86RegisterClasses.cpp - Register Classes -------*-C++-*-===//
//
// This file describes the X86 Register Classes which describe registers.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/MRegisterInfo.h"
#include "X86RegisterInfo.h"
#include "llvm/Type.h"

enum {
#define R(ENUM, NAME, FLAGS, TSFLAGS) ENUM,
#include "X86RegisterInfo.def"
};

namespace {
  static const unsigned X86ByteRegisterClassRegs[] = {
#define R(ENUM, NAME, FLAGS, TSFLAGS)
#define R8(ENUM, NAME, FLAGS, TSFLAGS) ENUM,
#include "X86RegisterInfo.def"
  };

  struct X86ByteRegisterClass : public TargetRegisterClass {
    unsigned getNumRegs() const { 
      return sizeof(X86ByteRegisterClassRegs)/
        sizeof(X86ByteRegisterClassRegs[0]);
    }
    unsigned getRegister(unsigned idx) const { 
      assert(idx < getNumRegs() && "Index out of bounds!");
      return X86ByteRegisterClassRegs[idx];
    }

    unsigned getDataSize() const { return 1; }
  } X86ByteRegisterClassInstance;


//
//
//
  static const unsigned X86ShortRegisterClassRegs[] = {
#define R(ENUM, NAME, FLAGS, TSFLAGS)
#define R16(ENUM, NAME, FLAGS, TSFLAGS) ENUM,
#include "X86RegisterInfo.def"
  };

  struct X86ShortRegisterClass : public TargetRegisterClass {
    unsigned getNumRegs() const { 
      return sizeof(X86ShortRegisterClassRegs)/
        sizeof(X86ShortRegisterClassRegs[0]); }
    unsigned getRegister(unsigned idx) const { 
      assert(idx < getNumRegs() && "Index out of bounds!");
      return X86ShortRegisterClassRegs[idx];
    }
    unsigned getDataSize() const { return 2; }
  } X86ShortRegisterClassInstance;

//
//
//

  static const unsigned X86IntRegisterClassRegs[] = {
#define R(ENUM, NAME, FLAGS, TSFLAGS)
#define R32(ENUM, NAME, FLAGS, TSFLAGS) ENUM,
#include "X86RegisterInfo.def"
  };

  struct X86IntRegisterClass : public TargetRegisterClass {
    unsigned getNumRegs() const {
      return sizeof(X86IntRegisterClassRegs)/
        sizeof(X86IntRegisterClassRegs[0]); }
    unsigned getRegister(unsigned idx) const { 
      assert(idx < getNumRegs() && "Index out of bounds!");
      return X86IntRegisterClassRegs[idx];
    }
    unsigned getDataSize() const { return 4; }
  } X86IntRegisterClassInstance;


  static const TargetRegisterClass *X86RegClasses[] = {
    &X86ByteRegisterClassInstance,
    &X86ShortRegisterClassInstance,
    &X86IntRegisterClassInstance
  };

  const TargetRegisterClass* X86RegisterInfo::getRegClassForType(const Type* Ty)
    const
  {
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

