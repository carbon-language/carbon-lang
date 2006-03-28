//===- CodeGenIntrinsic.h - Intrinsic Class Wrapper ------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a wrapper class for the 'Intrinsic' TableGen class.
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_INTRINSIC_H
#define CODEGEN_INTRINSIC_H

#include <string>
#include <vector>
#include "llvm/CodeGen/ValueTypes.h"

namespace llvm {
  class Record;
  class RecordKeeper;
  class CodeGenTarget;

  struct CodeGenIntrinsic {
    Record *TheDef;            // The actual record defining this instruction.
    std::string Name;          // The name of the LLVM function "llvm.bswap.i32"
    std::string EnumName;      // The name of the enum "bswap_i32"
    std::string GCCBuiltinName;// Name of the corresponding GCC builtin, or "".
    std::string TargetPrefix;  // Target prefix, e.g. "ppc" for t-s intrinsics.
    
    /// ArgTypes - The type primitive enum value for the return value and all
    /// of the arguments.  These are things like Type::UIntTyID.
    std::vector<std::string> ArgTypes;
    
    /// ArgVTs - The MVT::ValueType for each argument type.  Note that this list
    /// is only populated when in the context of a target .td file.  When
    /// building Intrinsics.td, this isn't available, because we don't know the
    /// target pointer size.
    std::vector<MVT::ValueType> ArgVTs;
    
    /// ArgTypeDefs - The records for each argument type.
    ///
    std::vector<Record*> ArgTypeDefs;
    
    // Memory mod/ref behavior of this intrinsic.
    enum {
      NoMem, ReadArgMem, ReadMem, WriteArgMem, WriteMem
    } ModRef;

    CodeGenIntrinsic(Record *R, CodeGenTarget *CGT);
  };

  /// LoadIntrinsics - Read all of the intrinsics defined in the specified
  /// .td file.
  std::vector<CodeGenIntrinsic> LoadIntrinsics(const RecordKeeper &RC);
}

#endif
