; RUN: llvm-as < %s -o /dev/null -f


%MidFnTy = type void (void (%MidFnTy* )*)

implementation

