; RUN: llvm-upgrade < %s | llvm-as | llc -march=c

%fptr = global void() * %f       ;; Forward ref method defn
declare void "f"()               ;; External method

