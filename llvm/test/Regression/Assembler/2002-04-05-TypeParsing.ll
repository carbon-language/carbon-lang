; RUN: llvm-as < %s -o /dev/null -f

  %Hosp = type {
                 { \2 *, { \2, \4 } * },
                 { \2 *, { \2, \4 } * }
               }

implementation
