// WebAssemblyInstrFormats.td - WebAssembly Instruction Formats -*- tblgen -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// WebAssembly instruction format definitions.
//
//===----------------------------------------------------------------------===//

// WebAssembly Instruction Format
class WebAssemblyInst<string cstr> : Instruction {
  field bits<0> Inst; // Instruction encoding.
  let Namespace   = "WebAssembly";
  let Pattern     = [];
  let Constraints = cstr;
}

// Normal instructions
class I<dag oops, dag iops, list<dag> pattern, string cstr = "">
    : WebAssemblyInst<cstr> {
  dag OutOperandList = oops;
  dag InOperandList  = iops;
  let Pattern        = pattern;
}
