//===- version.go - LLVM version info -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines LLVM version information.
//
//===----------------------------------------------------------------------===//

package llvm

/*
#include "llvm/Config/llvm-config.h"
*/
import "C"

const Version = C.LLVM_VERSION_STRING
