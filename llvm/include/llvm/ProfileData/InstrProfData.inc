//===-- InstrProfData.inc - instr profiling runtime structures-----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines templates for core runtime data structures used by  
// instrumentation based profiling and coverage mapping. The instrumentation  
// runtime library, the compiler IR lowering, and profile reader/writer need
// to use the same template to make sure the same data structure is defined
// consistently.
//
// Examples of how the template is used:
// 1. To declare a structure:
// 
// struct ProfData {
// #define INSTR_PROF_DATA(Type, LLVMType, Name, Initializer) \
//    Type Name;
// #include "llvm/ProfileData/InstrProfData.inc"
// };
//
// 2. To construct LLVM type arrays for the struct type:
//
// Type *DataTypes[] = {
// #define INSTR_PROF_DATA(Type, LLVMType, Name, Initializer) \
//   LLVMType,
// #include "llvm/ProfileData/InstrProfData.inc"
// };
//
// 4. To construct constant array for the initializers:
// #define INSTR_PROF_DATA(Type, LLVMType, Name, Initializer) \
//   Initializer,
// Constant *ConstantVals[] = {
// #include "llvm/ProfileData/InstrProfData.inc"
// };
//===----------------------------------------------------------------------===//

/* INSTR_PROF_DATA start. */
/* Definition of member fields of the per-function control structure. */
#ifndef INSTR_PROF_DATA
#define INSTR_PROF_DATA(Type, LLVMType, Name, Initializer)
#else
#define INSTR_PROF_DATA_DEFINED
#endif

INSTR_PROF_DATA(const uint32_t, llvm::Type::getInt32Ty(Ctx), NameSize, \
                ConstantInt::get(llvm::Type::getInt32Ty(Ctx), \
                NamePtr->getType()->getPointerElementType()->getArrayNumElements()))
INSTR_PROF_DATA(const uint32_t, llvm::Type::getInt32Ty(Ctx), NumCounters, \
                ConstantInt::get(llvm::Type::getInt32Ty(Ctx), NumCounters))
INSTR_PROF_DATA(const uint64_t, llvm::Type::getInt64Ty(Ctx), FuncHash, \
                ConstantInt::get(llvm::Type::getInt64Ty(Ctx), \
                Inc->getHash()->getZExtValue()))
INSTR_PROF_DATA(const IntPtrT, llvm::Type::getInt8PtrTy(Ctx), NamePtr, \
                ConstantExpr::getBitCast(NamePtr, llvm::Type::getInt8PtrTy(Ctx)))
INSTR_PROF_DATA(const IntPtrT, llvm::Type::getInt64PtrTy(Ctx), CounterPtr, \
                ConstantExpr::getBitCast(CounterPtr, \
                llvm::Type::getInt64PtrTy(Ctx)))
INSTR_PROF_DATA(const IntPtrT, llvm::Type::getInt8PtrTy(Ctx), FunctionPointer, \
                FunctionAddr)
INSTR_PROF_DATA(IntPtrT, llvm::Type::getInt8PtrTy(Ctx), Values, \
                ConstantPointerNull::get(Int8PtrTy))
INSTR_PROF_DATA(const uint16_t, Int16ArrayTy, NumValueSites[IPVK_Last+1], \
                ConstantArray::get(Int16ArrayTy, Int16ArrayVals))
#undef INSTR_PROF_DATA

/* INSTR_PROF_DATA end. */

/* INSTR_PROF_RAW_HEADER  start */
/* Definition of member fields of the raw profile header data structure. */
#ifndef INSTR_PROF_RAW_HEADER
#define INSTR_PROF_RAW_HEADER(Type, Name, Initializer)
#else
#define INSTR_PROF_DATA_DEFINED
#endif
INSTR_PROF_RAW_HEADER(const uint64_t, Magic, __llvm_profile_get_magic())
INSTR_PROF_RAW_HEADER(const uint64_t, Version, __llvm_profile_get_version())
INSTR_PROF_RAW_HEADER(const uint64_t, DataSize, DataSize)
INSTR_PROF_RAW_HEADER(const uint64_t, CountersSize, CountersSize)
INSTR_PROF_RAW_HEADER(const uint64_t, NamesSize,  NameSize)
INSTR_PROF_RAW_HEADER(const uint64_t, CountersDelta, (uintptr_t)CountersBegin)
INSTR_PROF_RAW_HEADER(const uint64_t, NamesDelta, (uintptr_t)NamesBegin)
INSTR_PROF_RAW_HEADER(const uint64_t, ValueKindLast, IPVK_Last)
INSTR_PROF_RAW_HEADER(const uint64_t, ValueDataSize, ValueDataSize)
INSTR_PROF_RAW_HEADER(const uint64_t, ValueDataDelta, (uintptr_t)ValueDataBegin)
#undef INSTR_PROF_RAW_HEADER
/* INSTR_PROF_RAW_HEADER  end */


/* COVMAP_FUNC_RECORD start */
/* Definition of member fields of the function record structure in coverage
 * map.
 */
#ifndef COVMAP_FUNC_RECORD
#define COVMAP_FUNC_RECORD(Type, LLVMType, Name, Initializer)
#else
#define INSTR_PROF_DATA_DEFINED
#endif
COVMAP_FUNC_RECORD(const IntPtrT, llvm::Type::getInt8PtrTy(Ctx), \
                   NamePtr, llvm::ConstantExpr::getBitCast(NamePtr, \
                   llvm::Type::getInt8PtrTy(Ctx))) 
COVMAP_FUNC_RECORD(const uint32_t, llvm::Type::getInt32Ty(Ctx), NameSize, \
                   llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx),\
                   NameValue.size()))
COVMAP_FUNC_RECORD(const uint32_t, llvm::Type::getInt32Ty(Ctx), DataSize, \
                   llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx),\
                   CoverageMapping.size()))
COVMAP_FUNC_RECORD(const uint64_t, llvm::Type::getInt64Ty(Ctx), FuncHash, \
                   llvm::ConstantInt::get(llvm::Type::getInt64Ty(Ctx), FuncHash))
#undef COVMAP_FUNC_RECORD
/* COVMAP_FUNC_RECORD end.  */


