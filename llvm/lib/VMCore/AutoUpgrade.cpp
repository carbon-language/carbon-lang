//===-- AutoUpgrade.cpp - Implement auto-upgrade helper functions ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the auto-upgrade helper functions 
//
//===----------------------------------------------------------------------===//

#include "llvm/Assembly/AutoUpgrade.h"
#include "llvm/Function.h"
#include "llvm/Type.h"
#include <iostream>

using namespace llvm;

// UpgradeIntrinsicFunction - Convert overloaded intrinsic function names to
// their non-overloaded variants by appending the appropriate suffix based on
// the argument types.
bool llvm::UpgradeIntrinsicFunction(Function* F) {

  // Get the Function's name.
  const std::string& Name = F->getName();

  // Quickly eliminate it, if it's not a candidate.
  if (Name.length() <= 5 || Name[0] != 'l' || Name[1] != 'l' || Name[2] !=
    'v' || Name[3] != 'm' || Name[4] != '.')
    return false;

  // See if its one of the name's we're interested in.
  switch (Name[5]) {
    case 'b':
      if (Name == "llvm.bswap") {
        const Type* Ty = F->getReturnType();
        std::string new_name = Name;
        if (Ty == Type::UShortTy || Ty == Type::ShortTy)
          new_name += ".i16";
        else if (Ty == Type::UIntTy || Ty == Type::IntTy)
          new_name += ".i32";
        else if (Ty == Type::ULongTy || Ty == Type::LongTy)
          new_name += ".i64";
        std::cerr << "WARNING: change " << Name << " to " 
          << new_name << "\n";
        F->setName(new_name);
        return true;
      }
      break;
    case 'c':
      if (Name == "llvm.ctpop" || Name == "llvm.ctlz" || 
          Name == "llvm.cttz") {
        const Type* Ty = F->getReturnType();
        std::string new_name = Name;
        if (Ty == Type::UByteTy || Ty == Type::SByteTy)
          new_name += ".i8";
        else if (Ty == Type::UShortTy || Ty == Type::ShortTy)
          new_name += ".i16";
        else if (Ty == Type::UIntTy || Ty == Type::IntTy)
          new_name += ".i32";
        else if (Ty == Type::ULongTy || Ty == Type::LongTy)
          new_name += ".i64";
        std::cerr << "WARNING: change " << Name << " to " 
          << new_name << "\n";
        F->setName(new_name);
        return true;
      }
      break;
    case 'i':
      if (Name == "llvm.isunordered") {
        Function::const_arg_iterator ArgIt = F->arg_begin();
        const Type* Ty = ArgIt->getType();
        std::string new_name = Name;
        if (Ty == Type::FloatTy)
          new_name += ".f32";
        else if (Ty == Type::DoubleTy)
          new_name += ".f64";
        std::cerr << "WARNING: change " << Name << " to " 
          << new_name << "\n";
        F->setName(new_name);
        return true;
      }
      break;
    case 's':
      if (Name == "llvm.sqrt") {
        const Type* Ty = F->getReturnType();
        std::string new_name = Name;
        if (Ty == Type::FloatTy)
          new_name += ".f32";
        else if (Ty == Type::DoubleTy) {
          new_name += ".f64";
        }
        std::cerr << "WARNING: change " << Name << " to " 
          << new_name << "\n";
        F->setName(new_name);
        return true;
      }
      break;
    default:
      break;
  }
  return false;
}
