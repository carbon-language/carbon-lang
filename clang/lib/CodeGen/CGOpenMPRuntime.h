//===----- CGOpenMPRuntime.h - Interface to OpenMP Runtimes -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides a class for OpenMP runtime code generation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIME_H
#define LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIME_H

#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

namespace llvm {
class ArrayType;
class Constant;
class Function;
class FunctionType;
class StructType;
class Type;
class Value;
} // namespace llvm

namespace clang {

class OMPExecutableDirective;
class VarDecl;

namespace CodeGen {

class CodeGenFunction;
class CodeGenModule;

class CGOpenMPRuntime {
public:
  /// \brief Values for bit flags used in the ident_t to describe the fields.
  /// All enumeric elements are named and described in accordance with the code
  /// from http://llvm.org/svn/llvm-project/openmp/trunk/runtime/src/kmp.h
  enum OpenMPLocationFlags {
    /// \brief Use trampoline for internal microtask.
    OMP_IDENT_IMD = 0x01,
    /// \brief Use c-style ident structure.
    OMP_IDENT_KMPC = 0x02,
    /// \brief Atomic reduction option for kmpc_reduce.
    OMP_ATOMIC_REDUCE = 0x10,
    /// \brief Explicit 'barrier' directive.
    OMP_IDENT_BARRIER_EXPL = 0x20,
    /// \brief Implicit barrier in code.
    OMP_IDENT_BARRIER_IMPL = 0x40,
    /// \brief Implicit barrier in 'for' directive.
    OMP_IDENT_BARRIER_IMPL_FOR = 0x40,
    /// \brief Implicit barrier in 'sections' directive.
    OMP_IDENT_BARRIER_IMPL_SECTIONS = 0xC0,
    /// \brief Implicit barrier in 'single' directive.
    OMP_IDENT_BARRIER_IMPL_SINGLE = 0x140
  };
  enum OpenMPRTLFunction {
    // Call to void __kmpc_fork_call(ident_t *loc, kmp_int32 argc, kmpc_micro
    // microtask, ...);
    OMPRTL__kmpc_fork_call,
    // Call to __kmpc_int32 kmpc_global_thread_num(ident_t *loc);
    OMPRTL__kmpc_global_thread_num,
    // Call to void __kmpc_critical(ident_t *loc, kmp_int32 global_tid,
    // kmp_critical_name *crit);
    OMPRTL__kmpc_critical,
    // Call to void __kmpc_end_critical(ident_t *loc, kmp_int32 global_tid,
    // kmp_critical_name *crit);
    OMPRTL__kmpc_end_critical,
    // Call to void __kmpc_barrier(ident_t *loc, kmp_int32 global_tid);
    OMPRTL__kmpc_barrier,
    // Call to void __kmpc_serialized_parallel(ident_t *loc, kmp_int32
    // global_tid);
    OMPRTL__kmpc_serialized_parallel,
    // Call to void __kmpc_end_serialized_parallel(ident_t *loc, kmp_int32
    // global_tid);
    OMPRTL__kmpc_end_serialized_parallel,
    // Call to void __kmpc_push_num_threads(ident_t *loc, kmp_int32 global_tid,
    // kmp_int32 num_threads);
    OMPRTL__kmpc_push_num_threads
  };

private:
  CodeGenModule &CGM;
  /// \brief Default const ident_t object used for initialization of all other
  /// ident_t objects.
  llvm::Constant *DefaultOpenMPPSource;
  /// \brief Map of flags and corresponding default locations.
  typedef llvm::DenseMap<unsigned, llvm::Value *> OpenMPDefaultLocMapTy;
  OpenMPDefaultLocMapTy OpenMPDefaultLocMap;
  llvm::Value *GetOrCreateDefaultOpenMPLocation(OpenMPLocationFlags Flags);
  /// \brief Describes ident structure that describes a source location.
  /// All descriptions are taken from
  /// http://llvm.org/svn/llvm-project/openmp/trunk/runtime/src/kmp.h
  /// Original structure:
  /// typedef struct ident {
  ///    kmp_int32 reserved_1;   /**<  might be used in Fortran;
  ///                                  see above  */
  ///    kmp_int32 flags;        /**<  also f.flags; KMP_IDENT_xxx flags;
  ///                                  KMP_IDENT_KMPC identifies this union
  ///                                  member  */
  ///    kmp_int32 reserved_2;   /**<  not really used in Fortran any more;
  ///                                  see above */
  ///#if USE_ITT_BUILD
  ///                            /*  but currently used for storing
  ///                                region-specific ITT */
  ///                            /*  contextual information. */
  ///#endif /* USE_ITT_BUILD */
  ///    kmp_int32 reserved_3;   /**< source[4] in Fortran, do not use for
  ///                                 C++  */
  ///    char const *psource;    /**< String describing the source location.
  ///                            The string is composed of semi-colon separated
  //                             fields which describe the source file,
  ///                            the function and a pair of line numbers that
  ///                            delimit the construct.
  ///                             */
  /// } ident_t;
  enum IdentFieldIndex {
    /// \brief might be used in Fortran
    IdentField_Reserved_1,
    /// \brief OMP_IDENT_xxx flags; OMP_IDENT_KMPC identifies this union member.
    IdentField_Flags,
    /// \brief Not really used in Fortran any more
    IdentField_Reserved_2,
    /// \brief Source[4] in Fortran, do not use for C++
    IdentField_Reserved_3,
    /// \brief String describing the source location. The string is composed of
    /// semi-colon separated fields which describe the source file, the function
    /// and a pair of line numbers that delimit the construct.
    IdentField_PSource
  };
  llvm::StructType *IdentTy;
  /// \brief Map for SourceLocation and OpenMP runtime library debug locations.
  typedef llvm::DenseMap<unsigned, llvm::Value *> OpenMPDebugLocMapTy;
  OpenMPDebugLocMapTy OpenMPDebugLocMap;
  /// \brief The type for a microtask which gets passed to __kmpc_fork_call().
  /// Original representation is:
  /// typedef void (kmpc_micro)(kmp_int32 global_tid, kmp_int32 bound_tid,...);
  llvm::FunctionType *Kmpc_MicroTy;
  /// \brief Stores debug location and ThreadID for the function.
  struct DebugLocThreadIdTy {
    llvm::Value *DebugLoc;
    llvm::Value *ThreadID;
  };
  /// \brief Map of local debug location, ThreadId and functions.
  typedef llvm::DenseMap<llvm::Function *, DebugLocThreadIdTy>
      OpenMPLocThreadIDMapTy;
  OpenMPLocThreadIDMapTy OpenMPLocThreadIDMap;
  /// \brief Type kmp_critical_name, originally defined as typedef kmp_int32
  /// kmp_critical_name[8];
  llvm::ArrayType *KmpCriticalNameTy;
  /// \brief Map of critical regions names and the corresponding lock objects.
  llvm::StringMap<llvm::Value *, llvm::BumpPtrAllocator> CriticalRegionVarNames;

  /// \brief Emits object of ident_t type with info for source location.
  /// \param Flags Flags for OpenMP location.
  ///
  llvm::Value *
  EmitOpenMPUpdateLocation(CodeGenFunction &CGF, SourceLocation Loc,
                           OpenMPLocationFlags Flags = OMP_IDENT_KMPC);

  /// \brief Returns pointer to ident_t type.
  llvm::Type *getIdentTyPointerTy();

  /// \brief Returns pointer to kmpc_micro type.
  llvm::Type *getKmpc_MicroPointerTy();

  /// \brief Returns specified OpenMP runtime function.
  /// \param Function OpenMP runtime function.
  /// \return Specified function.
  llvm::Constant *CreateRuntimeFunction(OpenMPRTLFunction Function);

  /// \brief Emits address of the word in a memory where current thread id is
  /// stored.
  virtual llvm::Value *EmitThreadIDAddress(CodeGenFunction &CGF,
                                           SourceLocation Loc);

  /// \brief Gets thread id value for the current thread.
  ///
  llvm::Value *GetOpenMPThreadID(CodeGenFunction &CGF, SourceLocation Loc);

public:
  explicit CGOpenMPRuntime(CodeGenModule &CGM);
  virtual ~CGOpenMPRuntime() {}

  /// \brief Emits outlined function for the specified OpenMP directive \a D
  /// (required for parallel and task directives). This outlined function has
  /// type void(*)(kmp_int32 /*ThreadID*/, kmp_int32 /*BoundID*/, struct
  /// context_vars*).
  /// \param D OpenMP directive.
  /// \param ThreadIDVar Variable for thread id in the current OpenMP region.
  ///
  virtual llvm::Value *
  EmitOpenMPOutlinedFunction(const OMPExecutableDirective &D,
                             const VarDecl *ThreadIDVar);

  /// \brief Cleans up references to the objects in finished function.
  ///
  void FunctionFinished(CodeGenFunction &CGF);

  /// \brief Emits code for parallel call of the \a OutlinedFn with variables
  /// captured in a record which address is stored in \a CapturedStruct.
  /// \param OutlinedFn Outlined function to be run in parallel threads. Type of
  /// this function is void(*)(kmp_int32, kmp_int32, struct context_vars*).
  /// \param CapturedStruct A pointer to the record with the references to
  /// variables used in \a OutlinedFn function.
  ///
  virtual void EmitOMPParallelCall(CodeGenFunction &CGF, SourceLocation Loc,
                                   llvm::Value *OutlinedFn,
                                   llvm::Value *CapturedStruct);

  /// \brief Emits code for serial call of the \a OutlinedFn with variables
  /// captured in a record which address is stored in \a CapturedStruct.
  /// \param OutlinedFn Outlined function to be run in serial mode.
  /// \param CapturedStruct A pointer to the record with the references to
  /// variables used in \a OutlinedFn function.
  ///
  virtual void EmitOMPSerialCall(CodeGenFunction &CGF, SourceLocation Loc,
                                 llvm::Value *OutlinedFn,
                                 llvm::Value *CapturedStruct);

  /// \brief Returns corresponding lock object for the specified critical region
  /// name. If the lock object does not exist it is created, otherwise the
  /// reference to the existing copy is returned.
  /// \param CriticalName Name of the critical region.
  ///
  llvm::Value *GetCriticalRegionLock(StringRef CriticalName);

  /// \brief Emits start of the critical region by calling void
  /// __kmpc_critical(ident_t *loc, kmp_int32 global_tid, kmp_critical_name
  /// * \a RegionLock)
  /// \param RegionLock The lock object for critical region.
  virtual void EmitOMPCriticalRegionStart(CodeGenFunction &CGF,
                                          llvm::Value *RegionLock,
                                          SourceLocation Loc);

  /// \brief Emits end of the critical region by calling void
  /// __kmpc_end_critical(ident_t *loc, kmp_int32 global_tid, kmp_critical_name
  /// * \a RegionLock)
  /// \param RegionLock The lock object for critical region.
  virtual void EmitOMPCriticalRegionEnd(CodeGenFunction &CGF,
                                        llvm::Value *RegionLock,
                                        SourceLocation Loc);

  /// \brief Emits a barrier for OpenMP threads.
  /// \param Flags Flags for the barrier.
  ///
  virtual void EmitOMPBarrierCall(CodeGenFunction &CGF, SourceLocation Loc,
                                  OpenMPLocationFlags Flags);

  /// \brief Emits call to void __kmpc_push_num_threads(ident_t *loc, kmp_int32
  /// global_tid, kmp_int32 num_threads) to generate code for 'num_threads'
  /// clause.
  /// \param NumThreads An integer value of threads.
  virtual void EmitOMPNumThreadsClause(CodeGenFunction &CGF,
                                       llvm::Value *NumThreads,
                                       SourceLocation Loc);
};
} // namespace CodeGen
} // namespace clang

#endif
