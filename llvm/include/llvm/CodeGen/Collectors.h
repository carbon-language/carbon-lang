//===-- Collectors.h - Garbage collector registry -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Gordon Henriksen and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the CollectorRegistry class, which is used to discover
// pluggable garbage collectors.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_COLLECTORS_H
#define LLVM_CODEGEN_COLLECTORS_H

#include "llvm/Support/Registry.h"

namespace llvm {

  class Collector;
  
  /// The collector registry uses all the defaults from Registry.
  /// 
  typedef Registry<Collector> CollectorRegistry;
  
  /// Creates an ocaml-compatible garbage collector.
  Collector *createOcamlCollector();
  
  /// Creates a shadow stack garbage collector. This collector requires no code
  /// generator support.
  Collector *createShadowStackCollector();
}

#endif
