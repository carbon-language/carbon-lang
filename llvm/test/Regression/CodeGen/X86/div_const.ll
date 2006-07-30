; RUN: llvm-as < %s | llc -march=x86 | grep 365384439

uint %f9188_mul365384439_shift27(uint %A) {
        %tmp1 = div uint %A, 1577682821         ; <uint> [#uses=1]
        ret uint %tmp1
}

