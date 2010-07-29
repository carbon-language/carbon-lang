//===-- DifferenceEngine.h - Module comparator ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header defines the interface to the LLVM difference engine,
// which structurally compares functions within a module.
//
//===----------------------------------------------------------------------===//

#ifndef _LLVM_DIFFERENCE_ENGINE_H_
#define _LLVM_DIFFERENCE_ENGINE_H_

#include <utility>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>

namespace llvm {
  class LLVMContext;
  class Module;
  class Function;
  class Twine;
  class Value;
  class GlobalValue;

  /// A class for performing structural comparisons of LLVM assembly.
  class DifferenceEngine {
  public:
    /// A temporary-object class for building up log messages.
    class LogBuilder {
      DifferenceEngine &Engine;

      /// The use of a stored StringRef here is okay because
      /// LogBuilder should be used only as a temporary, and as a
      /// temporary it will be destructed before whatever temporary
      /// might be initializing this format.
      StringRef Format;

      SmallVector<Value*, 4> Arguments;

    public:
      LogBuilder(DifferenceEngine &Engine, StringRef Format)
        : Engine(Engine), Format(Format) {}

      LogBuilder &operator<<(Value *V) {
        Arguments.push_back(V);
        return *this;
      }

      ~LogBuilder() {
        Engine.consumer.logf(*this);
      }

      StringRef getFormat() const { return Format; }

      unsigned getNumArguments() const { return Arguments.size(); }
      Value *getArgument(unsigned I) const { return Arguments[I]; }
    };

    enum DiffChange { DC_match, DC_left, DC_right };

    /// A temporary-object class for building up diff messages.
    class DiffLogBuilder {
      typedef std::pair<Instruction*,Instruction*> DiffRecord;
      SmallVector<DiffRecord, 20> Diff;

      DifferenceEngine &Engine;

    public:
      DiffLogBuilder(DifferenceEngine &Engine) : Engine(Engine) {}
      ~DiffLogBuilder() { Engine.consumer.logd(*this); }

      void addMatch(Instruction *L, Instruction *R) {
        Diff.push_back(DiffRecord(L, R));
      }
      void addLeft(Instruction *L) { Diff.push_back(DiffRecord(L, 0)); }
      void addRight(Instruction *R) { Diff.push_back(DiffRecord(0, R)); }

      unsigned getNumLines() const { return Diff.size(); }
      DiffChange getLineKind(unsigned I) const {
        return (Diff[I].first ? (Diff[I].second ? DC_match : DC_left)
                              : DC_right);
      }
      Instruction *getLeft(unsigned I) const { return Diff[I].first; }
      Instruction *getRight(unsigned I) const { return Diff[I].second; }
    };

    /// The interface for consumers of difference data.
    struct Consumer {
      /// Record that a local context has been entered.  Left and
      /// Right are IR "containers" of some sort which are being
      /// considered for structural equivalence: global variables,
      /// functions, blocks, instructions, etc.
      virtual void enterContext(Value *Left, Value *Right) = 0;

      /// Record that a local context has been exited.
      virtual void exitContext() = 0;

      /// Record a difference within the current context.
      virtual void log(StringRef Text) = 0;

      /// Record a formatted difference within the current context.
      virtual void logf(const LogBuilder &Log) = 0;

      /// Record a line-by-line instruction diff.
      virtual void logd(const DiffLogBuilder &Log) = 0;

    protected:
      ~Consumer() {}
    };

    /// A RAII object for recording the current context.
    struct Context {
      Context(DifferenceEngine &Engine, Value *L, Value *R) : Engine(Engine) {
        Engine.consumer.enterContext(L, R);
      }

      ~Context() {
        Engine.consumer.exitContext();
      }

    private:
      DifferenceEngine &Engine;
    };

    /// An oracle for answering whether two values are equivalent as
    /// operands.
    struct Oracle {
      virtual bool operator()(Value *L, Value *R) = 0;

    protected:
      ~Oracle() {}
    };

    DifferenceEngine(LLVMContext &context, Consumer &consumer)
      : context(context), consumer(consumer), globalValueOracle(0) {}

    void diff(Module *L, Module *R);
    void diff(Function *L, Function *R);

    void log(StringRef text) {
      consumer.log(text);
    }

    LogBuilder logf(StringRef text) {
      return LogBuilder(*this, text);
    }

    /// Installs an oracle to decide whether two global values are
    /// equivalent as operands.  Without an oracle, global values are
    /// considered equivalent as operands precisely when they have the
    /// same name.
    void setGlobalValueOracle(Oracle *oracle) {
      globalValueOracle = oracle;
    }

    /// Determines whether two global values are equivalent.
    bool equivalentAsOperands(GlobalValue *L, GlobalValue *R);

  private:
    LLVMContext &context;
    Consumer &consumer;
    Oracle *globalValueOracle;
  };
}

#endif
