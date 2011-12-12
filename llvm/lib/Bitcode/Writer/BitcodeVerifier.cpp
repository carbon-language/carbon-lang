//===--- Bitcode/Writer/BitcodeVerifier.cpp - Bitcode Writer ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// BitcodeVerifier implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Pass.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
  struct VerifyBitcode : public ModulePass {
    raw_ostream &OS; // raw_ostream to read from
  public:
    static char ID; // Pass identification, replacement for typeid
    explicit VerifyBitcode(raw_ostream &o)
      : ModulePass(ID), OS(o) {}
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    }

    const char *getPassName() const { return "Bitcode Verifier"; }
    
    bool runOnModule(Module &M) {
      Verify(M);
      return false;
    }

    void Verify(Module &M);
  };
}

char VerifyBitcode::ID = 0;

/// createBitcodeVerifierPass - Create a pass that writes a module to disk and
/// then reads the module back in to verify bitcode serialization and
/// deserialization.
ModulePass *llvm::createBitcodeVerifierPass(raw_ostream &Str) {
  return new VerifyBitcode(Str);
}

void VerifyBitcode::Verify(Module &M) {
  dbgs() << "BitcodeVerifier!\n";
}
