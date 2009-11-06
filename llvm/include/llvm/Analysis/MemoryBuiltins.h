//===- llvm/Analysis/MemoryBuiltins.h- Calls to memory builtins -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of functions identifies calls to builtin functions that allocate
// or free memory.  
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MEMORYBUILTINS_H
#define LLVM_ANALYSIS_MEMORYBUILTINS_H

namespace llvm {
class CallInst;
class PointerType;
class TargetData;
class Type;
class Value;

//===----------------------------------------------------------------------===//
//  malloc Call Utility Functions.
//

/// isMalloc - Returns true if the value is either a malloc call or a bitcast of 
/// the result of a malloc call
bool isMalloc(const Value *I);

/// extractMallocCall - Returns the corresponding CallInst if the instruction
/// is a malloc call.  Since CallInst::CreateMalloc() only creates calls, we
/// ignore InvokeInst here.
const CallInst* extractMallocCall(const Value *I);
CallInst* extractMallocCall(Value *I);

/// extractMallocCallFromBitCast - Returns the corresponding CallInst if the
/// instruction is a bitcast of the result of a malloc call.
const CallInst* extractMallocCallFromBitCast(const Value *I);
CallInst* extractMallocCallFromBitCast(Value *I);

/// isArrayMalloc - Returns the corresponding CallInst if the instruction 
/// is a call to malloc whose array size can be determined and the array size
/// is not constant 1.  Otherwise, return NULL.
CallInst* isArrayMalloc(Value *I, const TargetData *TD);
const CallInst* isArrayMalloc(const Value *I,
                              const TargetData *TD);

/// getMallocType - Returns the PointerType resulting from the malloc call.
/// This PointerType is the result type of the call's only bitcast use.
/// If there is no unique bitcast use, then return NULL.
const PointerType* getMallocType(const CallInst *CI);

/// getMallocAllocatedType - Returns the Type allocated by malloc call. This
/// Type is the result type of the call's only bitcast use. If there is no
/// unique bitcast use, then return NULL.
const Type* getMallocAllocatedType(const CallInst *CI);

/// getMallocArraySize - Returns the array size of a malloc call.  If the 
/// argument passed to malloc is a multiple of the size of the malloced type,
/// then return that multiple.  For non-array mallocs, the multiple is
/// constant 1.  Otherwise, return NULL for mallocs whose array size cannot be
/// determined.
Value* getMallocArraySize(CallInst *CI, const TargetData *TD);
                          
//===----------------------------------------------------------------------===//
//  free Call Utility Functions.
//

/// isFreeCall - Returns true if the the value is a call to the builtin free()
bool isFreeCall(const Value* I);

} // End llvm namespace

#endif
