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
    DependentLibs,

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

  /// In LLVM 1.3 format, the identifier and the size of the block are
  /// encoded into a single vbr_uint32 with 5 bits for the block identifier
  /// and 27-bits for block length. This limits blocks to a maximum of
  /// 128MBytes of data, and block types to 31 which should be sufficient
  /// for the foreseeable usage. Because the values of block identifiers MUST
  /// fit within 5 bits (values 1-31), this enumeration is used to ensure
  /// smaller values are used for 1.3 and subsequent bytecode versions.
  /// @brief The block number identifiers used in LLVM 1.3 bytecode
  /// format.
  enum CompressedBytecodeBlockIdentifiers {

    // Zero value ist verbotten!
    Reserved_DoNotUse = 0x00,      ///< Don't use this!

    // This is the uber block that contains the rest of the blocks.
    ModuleBlockID = 0x01,          ///< 1.3 identifier for modules

    // Module subtypes:

    // This is the identifier for a function
    FunctionBlockID = 0x02,        ///< 1.3 identifier for Functions
    ConstantPoolBlockID = 0x03,    ///< 1.3 identifier for constant pool
    SymbolTableBlockID = 0x04,     ///< 1.3 identifier for symbol table
    ModuleGlobalInfoBlockID = 0x05,///< 1.3 identifier for module globals
    GlobalTypePlaneBlockID = 0x06, ///< 1.3 identifier for global types

    // Function subtypes:

    // InstructionList - The instructions in the body of a function.  This
    // superceeds the old BasicBlock node used in LLVM 1.0.
    InstructionListBlockID = 0x07, ///< 1.3 identifier for insruction list

    // CompactionTable - blocks with this id are used to define local remapping
    // tables for a function, allowing the indices used within the function to
    // be as small as possible.  This often allows the instructions to be
    // encoded more efficiently.
    CompactionTableBlockID = 0x08, ///< 1.3 identifier for compaction tables

    // Not a block id, just used to count them
    NumberOfBlockIDs
  };

};

} // End llvm namespace

#endif
