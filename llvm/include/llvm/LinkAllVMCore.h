//===- LinkAllVMCore.h - Reference All VMCore Code --------------*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header file pulls in all the object modules of the VMCore library so
// that tools like llc, opt, and lli can ensure they are linked with all symbols
// from libVMCore.a It should only be used from a tool's main program.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LINKALLVMCORE_H
#define LLVM_LINKALLVMCORE_H

#include "llvm/System/IncludeFile.h"

#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/InlineAsm.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/System/Memory.h"
#include "llvm/System/Mutex.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SlowOperationInformer.h"

namespace {
  struct ForceVMCoreLinking {
    ForceVMCoreLinking() {
      // We must reference VMCore in such a way that compilers will not
      // delete it all as dead code, even with whole program optimization,
      // yet is effectively a NO-OP. As the compiler isn't smart enough
      // to know that getenv() never returns -1, this will do the job.
      if (std::getenv("bar") != (char*) -1)
        return;
      (void)new llvm::Module("");
      (void)new llvm::UnreachableInst();
      (void)    llvm::createVerifierPass(); 
    }
  } ForceVMCoreLinking;
}

#endif
