; RUN: llvm-as %s -o /dev/null

%t = type { { \2*, \2 },
            { \2*, \2 }
          }
