//===- llvm/Support/DiagnosticInfo.h - Diagnostic Declaration ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the different classes involved in low level diagnostics.
//
// Diagnostics reporting is still done as part of the LLVMContext.
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_DIAGNOSTICINFO_H
#define LLVM_SUPPORT_DIAGNOSTICINFO_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"

namespace llvm {

// Forward declarations.
class DiagnosticPrinter;
class Function;
class Instruction;
class Twine;
class Value;

/// \brief Defines the different supported severity of a diagnostic.
enum DiagnosticSeverity {
  DS_Error,
  DS_Warning,
  DS_Remark,
  // A note attaches additional information to one of the previous diagnostic
  // types.
  DS_Note
};

/// \brief Defines the different supported kind of a diagnostic.
/// This enum should be extended with a new ID for each added concrete subclass.
enum DiagnosticKind {
  DK_InlineAsm,
  DK_StackSize,
  DK_DebugMetadataVersion,
  DK_FirstPluginKind
};

/// \brief Get the next available kind ID for a plugin diagnostic.
/// Each time this function is called, it returns a different number.
/// Therefore, a plugin that wants to "identify" its own classes
/// with a dynamic identifier, just have to use this method to get a new ID
/// and assign it to each of its classes.
/// The returned ID will be greater than or equal to DK_FirstPluginKind.
/// Thus, the plugin identifiers will not conflict with the
/// DiagnosticKind values.
int getNextAvailablePluginDiagnosticKind();

/// \brief This is the base abstract class for diagnostic reporting in
/// the backend.
/// The print method must be overloaded by the subclasses to print a
/// user-friendly message in the client of the backend (let us call it a
/// frontend).
class DiagnosticInfo {
private:
  /// Kind defines the kind of report this is about.
  const /* DiagnosticKind */ int Kind;
  /// Severity gives the severity of the diagnostic.
  const DiagnosticSeverity Severity;

public:
  DiagnosticInfo(/* DiagnosticKind */ int Kind, DiagnosticSeverity Severity)
      : Kind(Kind), Severity(Severity) {}

  virtual ~DiagnosticInfo() {}

  /* DiagnosticKind */ int getKind() const { return Kind; }
  DiagnosticSeverity getSeverity() const { return Severity; }

  /// Print using the given \p DP a user-friendly message.
  /// This is the default message that will be printed to the user.
  /// It is used when the frontend does not directly take advantage
  /// of the information contained in fields of the subclasses.
  /// The printed message must not end with '.' nor start with a severity
  /// keyword.
  virtual void print(DiagnosticPrinter &DP) const = 0;
};

/// Diagnostic information for inline asm reporting.
/// This is basically a message and an optional location.
class DiagnosticInfoInlineAsm : public DiagnosticInfo {
private:
  /// Optional line information. 0 if not set.
  unsigned LocCookie;
  /// Message to be reported.
  const Twine &MsgStr;
  /// Optional origin of the problem.
  const Instruction *Instr;

public:
  /// \p MsgStr is the message to be reported to the frontend.
  /// This class does not copy \p MsgStr, therefore the reference must be valid
  /// for the whole life time of the Diagnostic.
  DiagnosticInfoInlineAsm(const Twine &MsgStr,
                          DiagnosticSeverity Severity = DS_Error)
      : DiagnosticInfo(DK_InlineAsm, Severity), LocCookie(0), MsgStr(MsgStr),
        Instr(NULL) {}

  /// \p LocCookie if non-zero gives the line number for this report.
  /// \p MsgStr gives the message.
  /// This class does not copy \p MsgStr, therefore the reference must be valid
  /// for the whole life time of the Diagnostic.
  DiagnosticInfoInlineAsm(unsigned LocCookie, const Twine &MsgStr,
                          DiagnosticSeverity Severity = DS_Error)
      : DiagnosticInfo(DK_InlineAsm, Severity), LocCookie(LocCookie),
        MsgStr(MsgStr), Instr(NULL) {}

  /// \p Instr gives the original instruction that triggered the diagnostic.
  /// \p MsgStr gives the message.
  /// This class does not copy \p MsgStr, therefore the reference must be valid
  /// for the whole life time of the Diagnostic.
  /// Same for \p I.
  DiagnosticInfoInlineAsm(const Instruction &I, const Twine &MsgStr,
                          DiagnosticSeverity Severity = DS_Error);

  unsigned getLocCookie() const { return LocCookie; }
  const Twine &getMsgStr() const { return MsgStr; }
  const Instruction *getInstruction() const { return Instr; }

  /// \see DiagnosticInfo::print.
  virtual void print(DiagnosticPrinter &DP) const;

  /// Hand rolled RTTI.
  static bool classof(const DiagnosticInfo *DI) {
    return DI->getKind() == DK_InlineAsm;
  }
};

/// Diagnostic information for stack size reporting.
/// This is basically a function and a size.
class DiagnosticInfoStackSize : public DiagnosticInfo {
private:
  /// The function that is concerned by this stack size diagnostic.
  const Function &Fn;
  /// The computed stack size.
  unsigned StackSize;

public:
  /// \p The function that is concerned by this stack size diagnostic.
  /// \p The computed stack size.
  DiagnosticInfoStackSize(const Function &Fn, unsigned StackSize,
                          DiagnosticSeverity Severity = DS_Warning)
      : DiagnosticInfo(DK_StackSize, Severity), Fn(Fn), StackSize(StackSize) {}

  const Function &getFunction() const { return Fn; }
  unsigned getStackSize() const { return StackSize; }

  /// \see DiagnosticInfo::print.
  virtual void print(DiagnosticPrinter &DP) const;

  /// Hand rolled RTTI.
  static bool classof(const DiagnosticInfo *DI) {
    return DI->getKind() == DK_StackSize;
  }
};

/// Diagnostic information for debug metadata version reporting.
/// This is basically a module and a version.
class DiagnosticInfoDebugMetadataVersion : public DiagnosticInfo {
private:
  /// The module that is concerned by this debug metadata version diagnostic.
  const Module &M;
  /// The actual metadata version.
  unsigned MetadataVersion;

public:
  /// \p The module that is concerned by this debug metadata version diagnostic.
  /// \p The actual metadata version.
  DiagnosticInfoDebugMetadataVersion(const Module &M, unsigned MetadataVersion,
                          DiagnosticSeverity Severity = DS_Warning)
      : DiagnosticInfo(DK_DebugMetadataVersion, Severity), M(M),
        MetadataVersion(MetadataVersion) {}

  const Module &getModule() const { return M; }
  unsigned getMetadataVersion() const { return MetadataVersion; }

  /// \see DiagnosticInfo::print.
  virtual void print(DiagnosticPrinter &DP) const;

  /// Hand rolled RTTI.
  static bool classof(const DiagnosticInfo *DI) {
    return DI->getKind() == DK_DebugMetadataVersion;
  }
};

} // End namespace llvm

#endif
