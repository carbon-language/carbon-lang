//=- MicroMips32r6InstrFormats.td - Mips32r6 Instruction Formats -*- tablegen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes microMIPS32r6 instruction formats.
//
//===----------------------------------------------------------------------===//

class MMR6Arch<string opstr> {
  string Arch = "micromipsr6";
  string BaseOpcode = opstr;
}
