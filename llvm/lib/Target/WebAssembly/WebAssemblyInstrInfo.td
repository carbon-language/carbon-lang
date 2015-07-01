// WebAssemblyInstrInfo.td-Describe the WebAssembly Instructions-*- tablegen -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// WebAssembly Instruction definitions.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// WebAssembly Instruction Predicate Definitions.
//===----------------------------------------------------------------------===//

def HasAddr32 : Predicate<"!Subtarget->hasAddr64()">;
def HasAddr64 : Predicate<"Subtarget->hasAddr64()">;
def HasSIMD128 : Predicate<"Subtarget->hasSIMD128()">,
                           AssemblerPredicate<"FeatureSIMD128", "simd128">;

//===----------------------------------------------------------------------===//
// WebAssembly-specific DAG Node Types.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// WebAssembly-specific DAG Nodes.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// WebAssembly-specific Operands.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// WebAssembly Instruction Format Definitions.
//===----------------------------------------------------------------------===//

include "WebAssemblyInstrFormats.td"

//===----------------------------------------------------------------------===//
// Additional sets of instructions.
//===----------------------------------------------------------------------===//

include "WebAssemblyInstrAtomics.td"
include "WebAssemblyInstrSIMD.td"
