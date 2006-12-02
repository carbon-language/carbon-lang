; RUN: llvm-upgrade < %s | llvm-as -o /dev/null -f


%MidFnTy = type void (void (%MidFnTy* )*)

implementation

