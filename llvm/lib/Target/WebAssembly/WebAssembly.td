//- WebAssembly.td - Describe the WebAssembly Target Machine --*- tablegen -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a target description file for the WebAssembly architecture, which is
// also known as "wasm".
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces which we are implementing
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// WebAssembly Subtarget features.
//===----------------------------------------------------------------------===//

def FeatureSIMD : SubtargetFeature<"simd", "HasSIMD", "true",
                                   "Enable SIMD">;

//===----------------------------------------------------------------------===//
// Architectures.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "WebAssemblyRegisterInfo.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "WebAssemblyInstrInfo.td"

def WebAssemblyInstrInfo : InstrInfo;

//===----------------------------------------------------------------------===//
// WebAssembly Processors supported.
//===----------------------------------------------------------------------===//

def : ProcessorModel<"generic", NoSchedModel, [FeatureSIMD]>;

//===----------------------------------------------------------------------===//
// Target Declaration
//===----------------------------------------------------------------------===//

def WebAssembly : Target {
  let InstructionSet = WebAssemblyInstrInfo;
}
