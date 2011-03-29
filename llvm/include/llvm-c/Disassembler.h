/*===-- llvm-c/Disassembler.h - Disassembler Public C Interface ---*- C -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header provides public interface to a disassembler library.           *|
|* LLVM provides an implementation of this interface.                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_C_DISASSEMBLER_H
#define LLVM_C_DISASSEMBLER_H  1

#include <stddef.h>
#include "llvm/Support/DataTypes.h"

/**
 * An opaque reference to a disassembler context.
 */
typedef void *LLVMDisasmContextRef;

/**
 * The type for the operand information call back function.  This is called to
 * get the symbolic information for an operand of an instruction.  Typically
 * this is from the relocation information, symbol table, etc.  That block of
 * information is saved when the disassembler context is created and passed to
 * the call back in the DisInfo parameter.  The instruction containing operand
 * is at the PC parameter.  For some instruction sets, there can be more than
 * one operand with symbolic information.  To determine the symbolic operand
 * infomation for each operand, the bytes for the specific operand in the
 * instruction are specified by the Offset parameter and its byte widith is the
 * size parameter.  For instructions sets with fixed widths and one symbolic
 * operand per instruction, the Offset parameter will be zero and Size parameter
 * will be the instruction width.  The information is returned in TagBuf and is 
 * Triple specific with its specific information defined by the value of
 * TagType for that Triple.  If symbolic information is returned the function
 * returns 1 else it returns 0.
 */
typedef int (*LLVMOpInfoCallback)(void *DisInfo,
                                  uint64_t PC,
                                  uint64_t Offset,
                                  uint64_t Size,
                                  int TagType,
                                  void *TagBuf);

/**
 * The type for the symbol lookup function.  This may be called by the
 * disassembler for such things like adding a comment for a PC plus a constant
 * offset load instruction to use a symbol name instead of a load address value.
 * It is passed the block information is saved when the disassembler context is
 * created and a value of a symbol to look up.  If no symbol is found NULL is
 * to be returned.
 */
typedef const char *(*LLVMSymbolLookupCallback)(void *DisInfo,
                                                uint64_t SymbolValue);

#ifdef __cplusplus
extern "C" {
#endif /* !defined(__cplusplus) */

/**
 * Create a disassembler for the TripleName.  Symbolic disassembly is supported
 * by passing a block of information in the DisInfo parameter and specifing the
 * TagType and call back functions as described above.  These can all be passed
 * as NULL.  If successfull this returns a disassembler context if not it
 * returns NULL.
 */
extern LLVMDisasmContextRef
LLVMCreateDisasm(const char *TripleName,
                 void *DisInfo,
                 int TagType,
                 LLVMOpInfoCallback GetOpInfo,
                 LLVMSymbolLookupCallback SymbolLookUp);

/**
 * Dispose of a disassembler context.
 */
extern void
LLVMDisasmDispose(LLVMDisasmContextRef DC);

/**
 * Disassmble a single instruction using the disassembler context specified in
 * the parameter DC.  The bytes of the instuction are specified in the parameter
 * Bytes, and contains at least BytesSize number of bytes.  The instruction is
 * at the address specified by the PC parameter.  If a valid instruction can be
 * disassembled its string is returned indirectly in OutString which whos size
 * is specified in the parameter OutStringSize.  This function returns the
 * number of bytes in the instruction or zero if there was no valid instruction.
 */
extern size_t
LLVMDisasmInstruction(LLVMDisasmContextRef DC,
                      uint8_t *Bytes,
                      uint64_t BytesSize,
                      uint64_t PC,
                      char *OutString,
                      size_t OutStringSize);

#ifdef __cplusplus
}
#endif /* !defined(__cplusplus) */

#endif /* !defined(LLVM_C_DISASSEMBLER_H) */
