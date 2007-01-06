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
  /// The the identifier and the size of the block are encoded into a single 
  /// vbr_uint32 with 5 bits for the block identifier and 27-bits for block 
  /// length. This limits blocks to a maximum of
  /// 128MBytes of data, and block types to 31 which should be sufficient
  /// for the foreseeable usage. Because the values of block identifiers MUST
  /// fit within 5 bits (values 1-31), this enumeration is used to ensure
  /// smaller values are used for 1.3 and subsequent bytecode versions.
  /// @brief The block number identifiers used in LLVM 1.3 bytecode
  /// format.
  enum BytecodeBlockIdentifiers {

    Reserved_DoNotUse      = 0,  ///< Zero value is forbidden, do not use.
    ModuleBlockID          = 1,  ///< Module block that contains other blocks.
    FunctionBlockID        = 2,  ///< Function block identifier
    ConstantPoolBlockID    = 3,  ///< Constant pool identifier
    ValueSymbolTableBlockID= 4,  ///< Value Symbol table identifier
    ModuleGlobalInfoBlockID= 5,  ///< Module global info identifier
    GlobalTypePlaneBlockID = 6,  ///< Global type plan identifier
    InstructionListBlockID = 7,  ///< All instructions in a function

    /// Blocks with this id are used to define a function local remapping
    /// table for the function's values. This allows the indices used within 
    /// the function to be as small as possible.  This often allows the 
    /// instructions to be encoded more efficiently because VBR takes fewer
    /// bytes with smaller values.
    /// @brief Value Compaction Table Block
    CompactionTableBlockID = 8,

    TypeSymbolTableBlockID = 9,  ///< Value Symbol table identifier
    // Not a block id, just used to count them
    NumberOfBlockIDs
  };
};

} // End llvm namespace

#endif
