//===-- LevelChange.h - Functions for raising/lowering methods ---*- C++ -*--=//
//
// This family of functions is useful for changing the 'level' of a method.  
// This can either be raising (converting direct addressing to use getelementptr
// for structs and arrays), or lowering (for instruction selection).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPT_LEVELCHANGE_H
#define LLVM_OPT_LEVELCHANGE_H

#include "llvm/Transforms/Pass.h"
#include "llvm/Module.h"
#include "llvm/Method.h"

namespace opt {
  struct Level {      // Define a namespace to contain the enum
    enum ID {    // Define an enum of levels to change into
      Lowest,           // The lowest level: ...
      //...

      Normal,           // The level LLVM is assumed to be in

      Simplified,       // Elminate silly things like unnecesary casts

      StructureAccess,  // Convert pointer addressing to structure getelementptr
                        // instructions.

      ArrayAccess,      // Simple direct access through pointers is converted to
                        // array accessors

      InductionVars,    // Auxillary induction variables are eliminated by
                        // introducing a cannonical indvar, and making all
                        // others use it.  This creates more opportunites to
                        // apply the previous optimizations.

      Highest = InductionVars,
    };
  };

  // DoRaiseRepresentation - Raise a method representation to a higher level.
  // The specific features to add are specified with the ToLevel argument.
  //
  bool DoRaiseRepresentation(Method *M, Level::ID ToLevel);
  bool DoRaiseRepresentation(Module *M, Level::ID ToLevel);
  static inline bool DoRaiseRepresentation(Method *M) {
    return DoRaiseRepresentation(M, Level::Highest);
  }
  bool DoRaiseRepresentation(Module *M, Level::ID ToLevel);
  static inline bool DoRaiseRepresentation(Module *M) {
    return DoRaiseRepresentation(M, Level::Highest);
  }

  struct RaiseRepresentation : public Pass {
    virtual bool doPerMethodWork(Method *M) {
      return DoRaiseRepresentation(M);
    }
  };


  // DoEliminateAuxillaryInductionVariables - Eliminate all aux indvars.  This
  // is one of the transformations performed by DoRaiseRepresentation, that
  // converts all induction variables to reference a cannonical induction
  // variable (which starts at 0 and counts by 1).
  //
  bool DoEliminateAuxillaryInductionVariables(Method *M);
  static inline bool DoEliminateAuxillaryInductionVariables(Module *M) {
    return M->reduceApply(DoEliminateAuxillaryInductionVariables);
  }


  // DoLowerRepresentation - Lower a method representation to a lower level.
  // The specific features to eliminate are specified with the ToLevel argument.
  //
  bool DoLowerRepresentation(Method *M, Level::ID ToLevel);
  bool DoLowerRepresentation(Module *M, Level::ID ToLevel);
  static inline bool DoLowerRepresentation(Module *M) {
    return DoLowerRepresentation(M, Level::Lowest);
  }
  static inline bool DoLowerRepresentation(Method *M) {
    return DoLowerRepresentation(M, Level::Lowest);
  }
}  // End namespace opt

#endif
