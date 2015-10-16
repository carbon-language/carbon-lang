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
// #define INSTR_PROF_DATA(Type, LLVMTypeVar, LLVMType, Name, Initializer) \
//    Type Name;
// #include "ProfileData/InstrProfData.inc"
// };
//
// 2. To define local variables for struct member's LLVM types"
//
// #define INSTR_PROF_DATA(Type, LLVMTypeVar, LLVMType, Name, Initializer) \
//   LLVMTypeVar = LLVMType;
// #include "ProfileData/InstrProfData.inc"
//
// 3. To construct LLVM type arrays for the struct type:
//
// Type *DataTypes[] = {
// #define INSTR_PROF_DATA(Type, LLVMTypeVar, LLVMType, Name, Initializer) \
//   LLVMTypeVar,
// #include "ProfileData/InstrProfData.inc"
// };
//
// 4. To construct constant array for the initializers:
// #define INSTR_PROF_DATA(Type, LLVMTypeVar, LLVMType, Name, Initializer) \
//   Initializer,
// Constant *ConstantVals[] = {
// #include "ProfileData/InstrProfData.inc"
// };
//===----------------------------------------------------------------------===//

#ifndef INSTR_PROF_DATA
#define INSTR_PROF_DATA(Type, LLVMTypeVar, LLVMType, Name, Initializer)
#endif

// INSTR_PROF_DATA_START
INSTR_PROF_DATA(const uint32_t, Int32Ty, llvm::Type::getInt32Ty(Ctx), NameSize, \
                ConstantInt::get(Int32Ty, NameSize))
INSTR_PROF_DATA(const uint32_t, Int32Ty, llvm::Type::getInt32Ty(Ctx), NumCounters, \
                ConstantInt::get(Int32Ty, NumCounters))
INSTR_PROF_DATA(const uint64_t, Int64Ty, llvm::Type::getInt64Ty(Ctx), FuncHash, \
                ConstantInt::get(Int64Ty, FuncHash))
INSTR_PROF_DATA(const IntPtrT, Int8PtrTy,llvm::Type::getInt8PtrTy(Ctx), NamePtr, \
                ConstantExpr::getBitCast(Name, Int8PtrTy))
INSTR_PROF_DATA(const IntPtrT, Int64PtrTy, llvm::Type::getInt64PtrTy(Ctx), CounterPtr, \
                ConstantExpr::getBitCast(CounterPtr, Int8PtrTy))
// INSTR_PROF_DATA_END

#ifdef INSTR_PROF_DATA
#undef INSTR_PROF_DATA
#endif


#ifndef COVMAP_FUNC_RECORD
#define COVMAP_FUNC_RECORD(Type, LLVMTypeVar, LLVMType, Name, Initializer)
#endif

// COVMAP_FUNC_RECORD_START
COVMAP_FUNC_RECORD(const IntPtrT, Int8PtrTy, llvm::Type::getInt8PtrTy(Ctx), \
                   NamePtr, llvm::ConstantExpr::getBitCast(NamePtr, Int8PtrTy))
COVMAP_FUNC_RECORD(const uint32_t, Int32Ty, llvm::Type::getInt32Ty(Ctx), \
                   NameSize, llvm::ConstantInt::get(Int32Ty, NameSize))
COVMAP_FUNC_RECORD(const uint32_t, Int32Ty, llvm::Type::getInt32Ty(Ctx), \
                   DataSize, llvm::ConstantInt::get(Int32Ty, DataSize))
COVMAP_FUNC_RECORD(const uint64_t, Int64Ty, llvm::Type::getInt64Ty(Ctx), \
                   FuncHash, llvm::ConstantInt::get(Int64Ty, FuncSize))
// COVMAP_FUNC_RECORD_END

#ifdef COVMAP_FUNC_RECORD
#undef COVMAP_FUNC_RECORD
#endif


