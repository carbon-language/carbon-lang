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

#ifndef CLANG_CODEGEN_OPENMPRUNTIME_H
#define CLANG_CODEGEN_OPENMPRUNTIME_H

#include "clang/AST/Type.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

namespace llvm {
class AllocaInst;
class CallInst;
class GlobalVariable;
class Constant;
class Function;
class Module;
class StructLayout;
class FunctionType;
class StructType;
class Type;
class Value;
} // namespace llvm

namespace clang {

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
    // Call to kmp_int32 kmpc_global_thread_num(ident_t *loc);
    OMPRTL__kmpc_global_thread_num
  };

private:
  CodeGenModule &CGM;
  /// \brief Default const ident_t object used for initialization of all other
  /// ident_t objects.
  llvm::Constant *DefaultOpenMPPSource;
  /// \brief Map of flags and corrsponding default locations.
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
  /// \brief Map for Sourcelocation and OpenMP runtime library debug locations.
  typedef llvm::DenseMap<unsigned, llvm::Value *> OpenMPDebugLocMapTy;
  OpenMPDebugLocMapTy OpenMPDebugLocMap;
  /// \brief The type for a microtask which gets passed to __kmpc_fork_call().
  /// Original representation is:
  /// typedef void (kmpc_micro)(kmp_int32 global_tid, kmp_int32 bound_tid,...);
  llvm::FunctionType *Kmpc_MicroTy;
  /// \brief Map of local debug location and functions.
  typedef llvm::DenseMap<llvm::Function *, llvm::Value *> OpenMPLocMapTy;
  OpenMPLocMapTy OpenMPLocMap;
  /// \brief Map of local gtid and functions.
  typedef llvm::DenseMap<llvm::Function *, llvm::Value *> OpenMPGtidMapTy;
  OpenMPGtidMapTy OpenMPGtidMap;

public:
  explicit CGOpenMPRuntime(CodeGenModule &CGM);
  ~CGOpenMPRuntime() {}

  /// \brief Cleans up references to the objects in finished function.
  /// \param CGF Reference to finished CodeGenFunction.
  ///
  void FunctionFinished(CodeGenFunction &CGF);

  /// \brief Emits object of ident_t type with info for source location.
  /// \param CGF Reference to current CodeGenFunction.
  /// \param Loc Clang source location.
  /// \param Flags Flags for OpenMP location.
  ///
  llvm::Value *
  EmitOpenMPUpdateLocation(CodeGenFunction &CGF, SourceLocation Loc,
                           OpenMPLocationFlags Flags = OMP_IDENT_KMPC);

  /// \brief Generates global thread number value.
  /// \param CGF Reference to current CodeGenFunction.
  /// \param Loc Clang source location.
  ///
  llvm::Value *GetOpenMPGlobalThreadNum(CodeGenFunction &CGF,
                                        SourceLocation Loc);

  /// \brief Returns pointer to ident_t type;
  llvm::Type *getIdentTyPointerTy();

  /// \brief Returns pointer to kmpc_micro type;
  llvm::Type *getKmpc_MicroPointerTy();

  /// \brief Returns specified OpenMP runtime function.
  /// \param Function OpenMP runtime function.
  /// \return Specified function.
  llvm::Constant *CreateRuntimeFunction(OpenMPRTLFunction Function);
};
} // namespace CodeGen
} // namespace clang

#endif
