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
    MethodInfo = 0x21,
    // Can also have ConstantPool block
    // Can also have SymbolTable block
    BasicBlock = 0x31,        // May contain many basic blocks
  };
};
#endif
