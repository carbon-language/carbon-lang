//===- ConsumedWarningsHandler.h -------------------------------*- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A handler class for warnings issued by the consumed analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CONSUMED_WARNING_HANDLER_H
#define LLVM_CLANG_CONSUMED_WARNING_HANDLER_H

#include <list>
#include <utility>

#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace consumed {

  typedef SmallVector<PartialDiagnosticAt, 1> OptionalNotes;
  typedef std::pair<PartialDiagnosticAt, OptionalNotes> DelayedDiag;
  typedef std::list<DelayedDiag> DiagList;

  class ConsumedWarningsHandlerBase {
    
  public:
    
    virtual ~ConsumedWarningsHandlerBase();
    
    /// \brief Emit the warnings and notes left by the analysis.
    virtual void emitDiagnostics() {}
    
    /// Warn about unnecessary-test errors.
    /// \param VariableName -- The name of the variable that holds the unique
    /// value.
    ///
    /// \param Loc -- The SourceLocation of the unnecessary test.
    virtual void warnUnnecessaryTest(StringRef VariableName,
                                     StringRef VariableState,
                                     SourceLocation Loc) {}
    
    /// Warn about use-while-consumed errors.
    /// \param MethodName -- The name of the method that was incorrectly
    /// invoked.
    /// 
    /// \param VariableName -- The name of the variable that holds the unique
    /// value.
    ///
    /// \param Loc -- The SourceLocation of the method invocation.
    virtual void warnUseOfTempWhileConsumed(StringRef MethodName,
                                            SourceLocation Loc) {}
    
    /// Warn about use-in-unknown-state errors.
    /// \param MethodName -- The name of the method that was incorrectly
    /// invoked.
    ///
    /// \param VariableName -- The name of the variable that holds the unique
    /// value.
    ///
    /// \param Loc -- The SourceLocation of the method invocation.
    virtual void warnUseOfTempInUnknownState(StringRef MethodName,
                                             SourceLocation Loc) {}
    
    /// Warn about use-while-consumed errors.
    /// \param MethodName -- The name of the method that was incorrectly
    /// invoked.
    ///
    /// \param VariableName -- The name of the variable that holds the unique
    /// value.
    ///
    /// \param Loc -- The SourceLocation of the method invocation.
    virtual void warnUseWhileConsumed(StringRef MethodName,
                                      StringRef VariableName,
                                      SourceLocation Loc) {}
    
    /// Warn about use-in-unknown-state errors.
    /// \param MethodName -- The name of the method that was incorrectly
    /// invoked.
    ///
    /// \param VariableName -- The name of the variable that holds the unique
    /// value.
    ///
    /// \param Loc -- The SourceLocation of the method invocation.
    virtual void warnUseInUnknownState(StringRef MethodName,
                                       StringRef VariableName,
                                       SourceLocation Loc) {}
  };
}} // end clang::consumed

#endif
