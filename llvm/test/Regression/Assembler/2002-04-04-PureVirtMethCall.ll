; RUN: llvm-upgrade | llvm-as -o /dev/null -f

  type { { \2 *, \4 ** },
         { \2 *, \4 ** }
       }

implementation
