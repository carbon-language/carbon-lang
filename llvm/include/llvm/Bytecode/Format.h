//===-- llvm/Bytecode/Format.h - VM bytecode file format info ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This header defines intrinsic constants that are useful to libraries that 
// need to hack on bytecode files directly, like the reader and writer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BYTECODE_FORMAT_H
#define LLVM_BYTECODE_FORMAT_H

namespace llvm {

class BytecodeFormat {   // Throw the constants into a poorman's namespace...
  BytecodeFormat();      // do not implement
public:

  // ID Numbers that are used in bytecode files...
  enum FileBlockIDs {
    // File level identifiers...
    Module = 0x01,

    // Module subtypes:
    Function = 0x11,
    ConstantPool,
    SymbolTable,
    ModuleGlobalInfo,
    GlobalTypePlane,

    // Function subtypes:
    // Can also have ConstantPool block
    // Can also have SymbolTable block
    BasicBlock = 0x31,// May contain many basic blocks (obsolete since LLVM 1.1)

    // InstructionList - The instructions in the body of a function.  This
    // superceeds the old BasicBlock node used in LLVM 1.0.
    InstructionList = 0x32,

    // CompactionTable - blocks with this id are used to define local remapping
    // tables for a function, allowing the indices used within the function to
    // be as small as possible.  This often allows the instructions to be
    // encoded more efficiently.
    CompactionTable = 0x33,
  };
};

} // End llvm namespace

#endif
