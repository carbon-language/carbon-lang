; RUN: not llvm-as %s -disable-output 2>&1 | grep "forward reference and definition of global have different types"

@a2 = alias void (), void ()* @g2
@g2 = internal global i8 42
