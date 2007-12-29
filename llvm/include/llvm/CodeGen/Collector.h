//===-- llvm/CodeGen/Collector.h - Garbage collection -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Collector records sufficient information about a machine function to enable
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
// MachineCodeAnalysis identifies the GC safe points in the machine code. (Roots
// are identified in SelectionDAGISel.)
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_COLLECTOR_H
#define LLVM_CODEGEN_COLLECTOR_H

#include "llvm/CodeGen/CollectorMetadata.h"
#include <iosfwd>
#include <string>

namespace llvm {
  
  /// Collector describes a garbage collector's code generation requirements,
  /// and provides overridable hooks for those needs which cannot be abstractly
  /// described.
  class Collector {
  public:
    typedef std::vector<CollectorMetadata*> list_type;
    typedef list_type::iterator iterator;
    
  private:
    friend class CollectorModuleMetadata;
    const Module *M;
    std::string Name;
    
    list_type Functions;
    
  protected:
    unsigned NeededSafePoints; //< Bitmask of required safe points.
    bool CustomReadBarriers;   //< Default is to insert loads.
    bool CustomWriteBarriers;  //< Default is to insert stores.
    bool CustomRoots;          //< Default is to pass through to backend.
    bool InitRoots;            //< If set, roots are nulled during lowering.
    
  public:
    Collector();
    
    virtual ~Collector();
    
    
    /// getName - The name of the collector, for debugging.
    /// 
    const std::string &getName() const { return Name; }

    /// getModule - The module upon which the collector is operating.
    /// 
    const Module &getModule() const { return *M; }

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
    
    
    /// beginAssembly/finishAssembly - Emit module metadata as assembly code.
    virtual void beginAssembly(std::ostream &OS, AsmPrinter &AP,
                               const TargetAsmInfo &TAI);
    virtual void finishAssembly(std::ostream &OS, AsmPrinter &AP,
                                const TargetAsmInfo &TAI);
    
    /// begin/end - Iterators for function metadata.
    /// 
    iterator begin() { return Functions.begin(); }
    iterator end()   { return Functions.end();   }

    /// insertFunctionMetadata - Creates metadata for a function.
    /// 
    CollectorMetadata *insertFunctionMetadata(const Function &F);

    /// initializeCustomLowering/performCustomLowering - If any of the actions
    /// are set to custom, performCustomLowering must be overriden to create a
    /// transform to lower those actions to LLVM IR. initializeCustomLowering
    /// is optional to override. These are the only Collector methods through
    /// which the LLVM IR can be modified.
    virtual bool initializeCustomLowering(Module &F);
    virtual bool performCustomLowering(Function &F);
  };
  
}

#endif
