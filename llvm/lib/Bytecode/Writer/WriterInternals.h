//===- WriterInternals.h - Data structures shared by the Writer -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header defines the interface used between components of the bytecode
// writer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_BYTECODE_WRITER_WRITERINTERNALS_H
#define LLVM_LIB_BYTECODE_WRITER_WRITERINTERNALS_H

#include "SlotCalculator.h"
#include "llvm/Bytecode/Writer.h"
#include "llvm/Bytecode/Format.h"
#include "llvm/Instruction.h"
#include "llvm/Support/DataTypes.h"
#include <string>
#include <vector>

namespace llvm {

class BytecodeWriter {
  std::vector<unsigned char> &Out;
  SlotCalculator Table;
public:
  BytecodeWriter(std::vector<unsigned char> &o, const Module *M);

private:
  void outputConstants(bool isFunction);
  void outputConstantStrings();
  void outputFunction(const Function *F);
  void outputCompactionTable();
  void outputCompactionTypes(unsigned StartNo);
  void outputCompactionTablePlane(unsigned PlaneNo,
                                  const std::vector<const Value*> &TypePlane,
                                  unsigned StartNo);
  void outputInstructions(const Function *F);
  void outputInstruction(const Instruction &I);
  void outputInstructionFormat0(const Instruction *I, unsigned Opcode,
                                const SlotCalculator &Table,
                                unsigned Type);
  void outputInstrVarArgsCall(const Instruction *I,
                              unsigned Opcode,
                              const SlotCalculator &Table,
                              unsigned Type) ;
  inline void outputInstructionFormat1(const Instruction *I,
                                       unsigned Opcode,
                                       unsigned *Slots,
                                       unsigned Type) ;
  inline void outputInstructionFormat2(const Instruction *I,
                                       unsigned Opcode,
                                       unsigned *Slots,
                                       unsigned Type) ;
  inline void outputInstructionFormat3(const Instruction *I,
                                       unsigned Opcode,
                                       unsigned *Slots,
                                       unsigned Type) ;

  void outputModuleInfoBlock(const Module *C);
  void outputSymbolTable(const SymbolTable &ST);
  void outputTypes(unsigned StartNo);
  void outputConstantsInPlane(const std::vector<const Value*> &Plane,
                              unsigned StartNo);
  void outputConstant(const Constant *CPV);
  void outputType(const Type *T);

  /// @brief Unsigned integer output primitive
  inline void output(unsigned i, int pos = -1);

  /// @brief Signed integer output primitive
  inline void output(int i);

  /// @brief 64-bit variable bit rate output primitive.
  inline void output_vbr(uint64_t i);

  /// @brief 32-bit variable bit rate output primitive.
  inline void output_vbr(unsigned i);

  /// @brief Signed 64-bit variable bit rate output primitive.
  inline void output_vbr(int64_t i);

  /// @brief Signed 32-bit variable bit rate output primitive.
  inline void output_vbr(int i);

  inline void output(const std::string &s );

  inline void output_data(const void *Ptr, const void *End);

  inline void output_float(float& FloatVal);
  inline void output_double(double& DoubleVal);

  inline void output_typeid(unsigned i);

  inline size_t size() const { return Out.size(); }
  inline void resize(size_t S) { Out.resize(S); }
  friend class BytecodeBlock;
};

/// BytecodeBlock - Little helper class is used by the bytecode writer to help
/// do backpatching of bytecode block sizes really easily.  It backpatches when
/// it goes out of scope.
///
class BytecodeBlock {
  unsigned Id;
  unsigned Loc;
  BytecodeWriter& Writer;

  /// ElideIfEmpty - If this is true and the bytecode block ends up being empty,
  /// the block can remove itself from the output stream entirely.
  bool ElideIfEmpty;

  /// If this is true then the block is written with a long format header using
  /// a uint (32-bits) for both the block id and size. Otherwise, it uses the
  /// short format which is a single uint with 27 bits for size and 5 bits for
  /// the block id. Both formats are used in a bc file with version 1.3.
  /// Previously only the long format was used.
  bool HasLongFormat;

  BytecodeBlock(const BytecodeBlock &);   // do not implement
  void operator=(const BytecodeBlock &);  // do not implement
public:
  inline BytecodeBlock(unsigned ID, BytecodeWriter& w,
                       bool elideIfEmpty = false, bool hasLongFormat = false);

  inline ~BytecodeBlock();
};

} // End llvm namespace

#endif
