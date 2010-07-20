//===---- TargetInfo.h - Encapsulate target details -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These classes wrap the information about a call or function
// definition used to handle ABI compliancy.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_TARGETINFO_H
#define CLANG_CODEGEN_TARGETINFO_H

namespace llvm {
  class GlobalValue;
  class Value;
}

namespace clang {
  class ABIInfo;
  class Decl;

  namespace CodeGen {
    class CodeGenModule;
    class CodeGenFunction;
  }

  /// TargetCodeGenInfo - This class organizes various target-specific
  /// codegeneration issues, like target-specific attributes, builtins and so
  /// on.
  class TargetCodeGenInfo {
    ABIInfo *Info;
  public:
    // WARNING: Acquires the ownership of ABIInfo.
    TargetCodeGenInfo(ABIInfo *info = 0):Info(info) { }
    virtual ~TargetCodeGenInfo();

    /// getABIInfo() - Returns ABI info helper for the target.
    const ABIInfo& getABIInfo() const { return *Info; }

    /// SetTargetAttributes - Provides a convenient hook to handle extra
    /// target-specific attributes for the given global.
    virtual void SetTargetAttributes(const Decl *D, llvm::GlobalValue *GV,
                                     CodeGen::CodeGenModule &M) const { }

    /// Determines the size of struct _Unwind_Exception on this platform,
    /// in 8-bit units.  The Itanium ABI defines this as:
    ///   struct _Unwind_Exception {
    ///     uint64 exception_class;
    ///     _Unwind_Exception_Cleanup_Fn exception_cleanup;
    ///     uint64 private_1;
    ///     uint64 private_2;
    ///   };
    unsigned getSizeOfUnwindException() const { return 32; }

    /// Controls whether __builtin_extend_pointer should sign-extend
    /// pointers to uint64_t or zero-extend them (the default).  Has
    /// no effect for targets:
    ///   - that have 64-bit pointers, or
    ///   - that cannot address through registers larger than pointers, or
    ///   - that implicitly ignore/truncate the top bits when addressing
    ///     through such registers.
    virtual bool extendPointerWithSExt() const { return false; }

    /// Determines the DWARF register number for the stack pointer, for
    /// exception-handling purposes.  Implements __builtin_dwarf_sp_column.
    ///
    /// Returns -1 if the operation is unsupported by this target.
    virtual int getDwarfEHStackPointer(CodeGen::CodeGenModule &M) const {
      return -1;
    }

    /// Initializes the given DWARF EH register-size table, a char*.
    /// Implements __builtin_init_dwarf_reg_size_table.
    ///
    /// Returns true if the operation is unsupported by this target.
    virtual bool initDwarfEHRegSizeTable(CodeGen::CodeGenFunction &CGF,
                                         llvm::Value *Address) const {
      return true;
    }

    /// Performs the code-generation required to convert a return
    /// address as stored by the system into the actual address of the
    /// next instruction that will be executed.
    ///
    /// Used by __builtin_extract_return_addr().
    virtual llvm::Value *decodeReturnAddress(CodeGen::CodeGenFunction &CGF,
                                             llvm::Value *Address) const {
      return Address;
    }

    /// Performs the code-generation required to convert the address
    /// of an instruction into a return address suitable for storage
    /// by the system in a return slot.
    ///
    /// Used by __builtin_frob_return_addr().
    virtual llvm::Value *encodeReturnAddress(CodeGen::CodeGenFunction &CGF,
                                             llvm::Value *Address) const {
      return Address;
    }
  };
}

#endif // CLANG_CODEGEN_TARGETINFO_H
