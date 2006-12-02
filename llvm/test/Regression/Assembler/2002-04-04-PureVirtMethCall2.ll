; RUN: llvm-upgrade < %s | llvm-as -o /dev/null -f

%t = type { { \2*, \2 },
            { \2*, \2 }
          }

implementation
