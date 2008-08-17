//===-- OcamlCollector.cpp - Ocaml frametable emitter ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering for the llvm.gc* intrinsics compatible with
// Objective Caml 3.10.0, which uses a liveness-accurate static stack map.
//
//===----------------------------------------------------------------------===//
                        
#include "llvm/CodeGen/GCs.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/GCStrategy.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

namespace {

  class VISIBILITY_HIDDEN OcamlCollector : public Collector {
  public:
    OcamlCollector();
  };
  
}

static CollectorRegistry::Add<OcamlCollector>
X("ocaml", "ocaml 3.10-compatible collector");

// -----------------------------------------------------------------------------

Collector *llvm::createOcamlCollector() {
  return new OcamlCollector();
}

OcamlCollector::OcamlCollector() {
  NeededSafePoints = 1 << GC::PostCall;
  UsesMetadata = true;
}
