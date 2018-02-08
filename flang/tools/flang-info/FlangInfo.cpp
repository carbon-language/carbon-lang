//===--- tools/clang-check/ClangCheck.cpp - Clang check tool --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file is a playground to test the build infrastucture and the new APIs.
//  It will eventually disapear.
//
//
//  PLEASE DO NOT REPORT CODING STYLE VIOLATIONS ON THAT FILE! 
//  THIS IS A TEMPORARY PLAYGROUND!
//
//
//===----------------------------------------------------------------------===//

#include "flang/Basic/Version.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"

using namespace llvm;
using namespace flang;

#include <iostream>

int main(int argc, const char **argv) {
  
  std::cout << "Flang Repository = '" << getFlangRepositoryPath() << "'\n";
  std::cout << "Flang Version    = '" << getFlangFullVersion() << "'\n";

  std::cout << "LLVM  Repository = '" << getLLVMRepositoryPath() << "'\n";

  return 1 ;
}
