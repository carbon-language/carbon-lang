//===-- llvm/CodeGen/Collector.h - Garbage collection -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Gordon Henriksen and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// GCInfo records sufficient information about a machine function to enable
// accurate garbage collectors. Specifically:
// 
// - Safe points
//   Garbage collection is only possible at certain points in code. Code
//   generators should record points:
//
//     - At and after any call to a subroutine
//     - Before returning from the current function
//     - Before backwards branches (loops)
// 
// - Roots
//   When a reference to a GC-allocated object exists on the stack, it must be
//   stored in an alloca registered with llvm.gcoot.
//
// This generic information should used by ABI-specific passes to emit support
// tables for the runtime garbage collector.
//
// GCSafePointPass identifies the GC safe points in the machine code. (Roots are
// identified in SelectionDAGISel.)
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_COLLECTOR_H
#define LLVM_CODEGEN_COLLECTOR_H

#include "llvm/CodeGen/CollectorMetadata.h"
#include <iosfwd>

namespace llvm {
  
  class AsmPrinter;
  class FunctionPassManager;
  class PassManager;
  class TargetAsmInfo;
  
  
  /// Collector describes a garbage collector's code generation requirements,
  /// and provides overridable hooks for those needs which cannot be abstractly
  /// described.
  class Collector {
  protected:
    unsigned NeededSafePoints; //< Bitmask of required safe points.
    bool CustomReadBarriers;   //< Default is to insert loads.
    bool CustomWriteBarriers;  //< Default is to insert stores.
    bool CustomRoots;          //< Default is to pass through to backend.
    bool InitRoots;            //< If set, roots are nulled during lowering.
    
    /// If any of the actions are set to Custom, this is expected to be
    /// overriden to create a transform to lower those actions to LLVM IR.
    virtual Pass *createCustomLoweringPass() const;
    
  public:
    Collector();
    
    virtual ~Collector();
    
    
    /// True if this collector requires safe points of any kind. By default,
    /// none are recorded.
    bool needsSafePoints() const { return NeededSafePoints != 0; }
    
    /// True if the collector requires the given kind of safe point. By default,
    /// none are recorded.
    bool needsSafePoint(GC::PointKind Kind) const {
      return (NeededSafePoints & 1 << Kind) != 0;
    }
    
    /// By default, write barriers are replaced with simple store instructions.
    /// If true, then addPassesToCustomLowerIntrinsics must instead process
    /// them.
    bool customWriteBarrier() const { return CustomWriteBarriers; }
    
    /// By default, read barriers are replaced with simple load instructions.
    /// If true, then addPassesToCustomLowerIntrinsics must instead process
    /// them.
    bool customReadBarrier() const { return CustomReadBarriers; }
    
    /// By default, roots are left for the code generator. If Custom, then 
    /// addPassesToCustomLowerIntrinsics must add passes to delete them.
    bool customRoots() const { return CustomRoots; }
    
    /// If set, gcroot intrinsics should initialize their allocas to null. This
    /// is necessary for most collectors.
    bool initializeRoots() const { return InitRoots; }
    
    
    /// Adds LLVM IR transforms to handle collection intrinsics. By default,
    /// read- and write barriers are replaced with direct memory accesses, and
    /// roots are passed on to the code generator.
    void addLoweringPasses(FunctionPassManager &PM) const;
    
    /// Same as addLoweringPasses(FunctionPassManager &), except uses a
    /// PassManager for compatibility with unusual backends (such as MSIL or
    /// CBackend).
    void addLoweringPasses(PassManager &PM) const;
    
    /// Adds target-independent MachineFunction pass to mark safe points. This 
    /// is added very late during code generation, just prior to output, and
    /// importantly after all CFG transformations (like branch folding).
    void addGenericMachineCodePass(FunctionPassManager &PM,
                                   const TargetMachine &TM, bool Fast) const;
    
    /// beginAssembly/finishAssembly - Emit module metadata as assembly code.
    virtual void beginAssembly(Module &M, std::ostream &OS, AsmPrinter &AP,
                               const TargetAsmInfo &TAI) const;
    virtual void finishAssembly(Module &M, CollectorModuleMetadata &CMM,
                                std::ostream &OS, AsmPrinter &AP,
                                const TargetAsmInfo &TAI) const;
    
  private:
    bool NeedsDefaultLoweringPass() const;
    bool NeedsCustomLoweringPass() const;
    
  };
  
  
  /// If set, the code generator should generate garbage collection as specified
  /// by the collector properties.
  extern const Collector *TheCollector;  // FIXME: Find a better home!
  
}

#endif
