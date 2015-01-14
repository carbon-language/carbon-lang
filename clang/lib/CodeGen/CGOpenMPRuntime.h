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

#include "clang/Basic/OpenMPKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/ValueHandle.h"

namespace llvm {
class ArrayType;
class Constant;
class Function;
class FunctionType;
class GlobalVariable;
class StructType;
class Type;
class Value;
} // namespace llvm

namespace clang {
class Expr;
class OMPExecutableDirective;
class VarDecl;

namespace CodeGen {

class CodeGenFunction;
class CodeGenModule;

class CGOpenMPRuntime {
public:

private:
  enum OpenMPRTLFunction {
    /// \brief Call to void __kmpc_fork_call(ident_t *loc, kmp_int32 argc,
    /// kmpc_micro microtask, ...);
    OMPRTL__kmpc_fork_call,
    /// \brief Call to void *__kmpc_threadprivate_cached(ident_t *loc,
    /// kmp_int32 global_tid, void *data, size_t size, void ***cache);
    OMPRTL__kmpc_threadprivate_cached,
    /// \brief Call to void __kmpc_threadprivate_register( ident_t *,
    /// void *data, kmpc_ctor ctor, kmpc_cctor cctor, kmpc_dtor dtor);
    OMPRTL__kmpc_threadprivate_register,
    // Call to __kmpc_int32 kmpc_global_thread_num(ident_t *loc);
    OMPRTL__kmpc_global_thread_num,
    // Call to void __kmpc_critical(ident_t *loc, kmp_int32 global_tid,
    // kmp_critical_name *crit);
    OMPRTL__kmpc_critical,
    // Call to void __kmpc_end_critical(ident_t *loc, kmp_int32 global_tid,
    // kmp_critical_name *crit);
    OMPRTL__kmpc_end_critical,
    // Call to kmp_int32 __kmpc_cancel_barrier(ident_t *loc, kmp_int32
    // global_tid);
    OMPRTL__kmpc_cancel_barrier,
    // Calls for static scheduling 'omp for' loops.
    OMPRTL__kmpc_for_static_init_4,
    OMPRTL__kmpc_for_static_init_4u,
    OMPRTL__kmpc_for_static_init_8,
    OMPRTL__kmpc_for_static_init_8u,
    OMPRTL__kmpc_for_static_fini,
    // Call to void __kmpc_serialized_parallel(ident_t *loc, kmp_int32
    // global_tid);
    OMPRTL__kmpc_serialized_parallel,
    // Call to void __kmpc_end_serialized_parallel(ident_t *loc, kmp_int32
    // global_tid);
    OMPRTL__kmpc_end_serialized_parallel,
    // Call to void __kmpc_push_num_threads(ident_t *loc, kmp_int32 global_tid,
    // kmp_int32 num_threads);
    OMPRTL__kmpc_push_num_threads,
    // Call to void __kmpc_flush(ident_t *loc, ...);
    OMPRTL__kmpc_flush,
    // Call to kmp_int32 __kmpc_master(ident_t *, kmp_int32 global_tid);
    OMPRTL__kmpc_master,
    // Call to void __kmpc_end_master(ident_t *, kmp_int32 global_tid);
    OMPRTL__kmpc_end_master,
  };

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
  /// \brief An ordered map of auto-generated variables to their unique names.
  /// It stores variables with the following names: 1) ".gomp_critical_user_" +
  /// <critical_section_name> + ".var" for "omp critical" directives; 2)
  /// <mangled_name_for_global_var> + ".cache." for cache for threadprivate
  /// variables.
  llvm::StringMap<llvm::AssertingVH<llvm::Constant>, llvm::BumpPtrAllocator>
      InternalVars;

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

  /// \brief If the specified mangled name is not in the module, create and
  /// return threadprivate cache object. This object is a pointer's worth of
  /// storage that's reserved for use by the OpenMP runtime.
  /// \param VD Threadprivate variable.
  /// \return Cache variable for the specified threadprivate.
  llvm::Constant *getOrCreateThreadPrivateCache(const VarDecl *VD);

  /// \brief Emits address of the word in a memory where current thread id is
  /// stored.
  virtual llvm::Value *EmitThreadIDAddress(CodeGenFunction &CGF,
                                           SourceLocation Loc);

  /// \brief Gets thread id value for the current thread.
  ///
  llvm::Value *GetOpenMPThreadID(CodeGenFunction &CGF, SourceLocation Loc);

  /// \brief Gets (if variable with the given name already exist) or creates
  /// internal global variable with the specified Name. The created variable has
  /// linkage CommonLinkage by default and is initialized by null value.
  /// \param Ty Type of the global variable. If it is exist already the type
  /// must be the same.
  /// \param Name Name of the variable.
  llvm::Constant *GetOrCreateInternalVariable(llvm::Type *Ty,
                                              const llvm::Twine &Name);

  /// \brief Set of threadprivate variables with the generated initializer.
  llvm::DenseSet<const VarDecl *> ThreadPrivateWithDefinition;

  /// \brief Emits initialization code for the threadprivate variables.
  /// \param VDAddr Address of the global variable \a VD.
  /// \param Ctor Pointer to a global init function for \a VD.
  /// \param CopyCtor Pointer to a global copy function for \a VD.
  /// \param Dtor Pointer to a global destructor function for \a VD.
  /// \param Loc Location of threadprivate declaration.
  void EmitOMPThreadPrivateVarInit(CodeGenFunction &CGF, llvm::Value *VDAddr,
                                   llvm::Value *Ctor, llvm::Value *CopyCtor,
                                   llvm::Value *Dtor, SourceLocation Loc);

  /// \brief Returns corresponding lock object for the specified critical region
  /// name. If the lock object does not exist it is created, otherwise the
  /// reference to the existing copy is returned.
  /// \param CriticalName Name of the critical region.
  ///
  llvm::Value *GetCriticalRegionLock(StringRef CriticalName);

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

  /// \brief Emits a critical region.
  /// \param CriticalName Name of the critical region.
  /// \param CriticalOpGen Generator for the statement associated with the given
  /// critical region.
  virtual void EmitOMPCriticalRegion(CodeGenFunction &CGF,
                                     StringRef CriticalName,
                                     const std::function<void()> &CriticalOpGen,
                                     SourceLocation Loc);

  /// \brief Emits a master region.
  /// \param MasterOpGen Generator for the statement associated with the given
  /// master region.
  virtual void EmitOMPMasterRegion(CodeGenFunction &CGF,
                                   const std::function<void()> &MasterOpGen,
                                   SourceLocation Loc);

  /// \brief Emits explicit barrier for OpenMP threads.
  /// \param IsExplicit true, if it is explicitly specified barrier.
  ///
  virtual void EmitOMPBarrierCall(CodeGenFunction &CGF, SourceLocation Loc,
                                  bool IsExplicit = true);

  /// \brief Check if the specified \a ScheduleKind is static non-chunked.
  /// This kind of worksharing directive is emitted without outer loop.
  /// \param ScheduleKind Schedule kind specified in the 'schedule' clause.
  /// \param Chunked True if chunk is specified in the clause.
  ///
  virtual bool isStaticNonchunked(OpenMPScheduleClauseKind ScheduleKind,
                                  bool Chunked) const;

  /// \brief Call the appropriate runtime routine to initialize it before start
  /// of loop.
  ///
  /// Depending on the loop schedule, it is nesessary to call some runtime
  /// routine before start of the OpenMP loop to get the loop upper / lower
  /// bounds \a LB and \a UB and stride \a ST.
  ///
  /// \param CGF Reference to current CodeGenFunction.
  /// \param Loc Clang source location.
  /// \param SchedKind Schedule kind, specified by the 'schedule' clause.
  /// \param IVSize Size of the iteration variable in bits.
  /// \param IVSigned Sign of the interation variable.
  /// \param IL Address of the output variable in which the flag of the
  /// last iteration is returned.
  /// \param LB Address of the output variable in which the lower iteration
  /// number is returned.
  /// \param UB Address of the output variable in which the upper iteration
  /// number is returned.
  /// \param ST Address of the output variable in which the stride value is
  /// returned nesessary to generated the static_chunked scheduled loop.
  /// \param Chunk Value of the chunk for the static_chunked scheduled loop.
  /// For the default (nullptr) value, the chunk 1 will be used.
  ///
  virtual void EmitOMPForInit(CodeGenFunction &CGF, SourceLocation Loc,
                              OpenMPScheduleClauseKind SchedKind,
                              unsigned IVSize, bool IVSigned, llvm::Value *IL,
                              llvm::Value *LB, llvm::Value *UB, llvm::Value *ST,
                              llvm::Value *Chunk = nullptr);

  /// \brief Call the appropriate runtime routine to notify that we finished
  /// all the work with current loop.
  ///
  /// \param CGF Reference to current CodeGenFunction.
  /// \param Loc Clang source location.
  /// \param ScheduleKind Schedule kind, specified by the 'schedule' clause.
  ///
  virtual void EmitOMPForFinish(CodeGenFunction &CGF, SourceLocation Loc,
                                OpenMPScheduleClauseKind ScheduleKind);

  /// \brief Emits call to void __kmpc_push_num_threads(ident_t *loc, kmp_int32
  /// global_tid, kmp_int32 num_threads) to generate code for 'num_threads'
  /// clause.
  /// \param NumThreads An integer value of threads.
  virtual void EmitOMPNumThreadsClause(CodeGenFunction &CGF,
                                       llvm::Value *NumThreads,
                                       SourceLocation Loc);

  /// \brief Returns address of the threadprivate variable for the current
  /// thread.
  /// \param VD Threadprivate variable.
  /// \param VDAddr Address of the global variable \a VD.
  /// \param Loc Location of the reference to threadprivate var.
  /// \return Address of the threadprivate variable for the current thread.
  virtual llvm::Value *getOMPAddrOfThreadPrivate(CodeGenFunction &CGF,
                                                 const VarDecl *VD,
                                                 llvm::Value *VDAddr,
                                                 SourceLocation Loc);

  /// \brief Emit a code for initialization of threadprivate variable. It emits
  /// a call to runtime library which adds initial value to the newly created
  /// threadprivate variable (if it is not constant) and registers destructor
  /// for the variable (if any).
  /// \param VD Threadprivate variable.
  /// \param VDAddr Address of the global variable \a VD.
  /// \param Loc Location of threadprivate declaration.
  /// \param PerformInit true if initialization expression is not constant.
  virtual llvm::Function *
  EmitOMPThreadPrivateVarDefinition(const VarDecl *VD, llvm::Value *VDAddr,
                                    SourceLocation Loc, bool PerformInit,
                                    CodeGenFunction *CGF = nullptr);

  /// \brief Emit flush of the variables specified in 'omp flush' directive.
  /// \param Vars List of variables to flush.
  virtual void EmitOMPFlush(CodeGenFunction &CGF, ArrayRef<const Expr *> Vars,
                            SourceLocation Loc);
};
} // namespace CodeGen
} // namespace clang

#endif
