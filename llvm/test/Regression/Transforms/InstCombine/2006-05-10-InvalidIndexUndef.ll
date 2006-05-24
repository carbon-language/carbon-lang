; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep undef

%str = constant [4 x ubyte] c"str\00"

ubyte %main() {
        %A = load ubyte* getelementptr ([4 x ubyte]* %str, long 0, long 5)
        ret ubyte %A
}
