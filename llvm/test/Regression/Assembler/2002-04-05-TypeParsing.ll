; RUN: llvm-upgrade < %s | llvm-as -o /dev/null -f

  %Hosp = type {
                 { \2 *, { \2, \4 } * },
                 { \2 *, { \2, \4 } * }
               }

implementation
